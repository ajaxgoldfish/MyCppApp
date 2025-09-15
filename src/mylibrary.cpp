#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "mylibrary.h"          // 声明：bs_yzx_init / bs_yzx_object_detection_lanxin
#include "FusionGeometry.h"
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <array>
#include <iomanip>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <algorithm>
using nlohmann::json;
namespace fs = std::filesystem;

#ifndef YZX_MAX_BOX
#define YZX_MAX_BOX 100
#endif

namespace {
    struct LocalOptions {
        std::string model_path;
        std::string calib_path;
        float score_thr = 0.8f;
        float mask_thr = 0.6f;
        bool paint_masks_on_vis = true;
    };

    struct LocalBoxPoseResult {
        int id = -1;
        cv::Point3f xyz_m{};
        cv::Vec3f wpr_deg{};
        float width_m = 0.f, height_m = 0.f;
        cv::RotatedRect obb;
        cv::Point2f bottomMidPx{};
        cv::Point3f p1_w_m{}, p2_w_m{}, p3_w_m{};
        Eigen::Matrix3d Rw;
    };

    std::unique_ptr<LanxinCamera> g_camera;

    struct RunnerState {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mrcnn"};
        std::unique_ptr<Ort::Session> session;
        std::string in_name;
        std::vector<std::string> out_names_s;
        std::vector<const char *> out_names;
    };

    std::unique_ptr<RunnerState> g_runner;
    cv::Mat g_K, g_Kinv, g_Twc; // K: CV_64F, Twc: CV_32F
    LocalOptions g_opt;
    bool g_ready = false;
    std::string g_root_dir = "res"; // 仅用于输出可视化


    struct Proj {
        int u, v, pid;
    };


} // namespace


int bs_yzx_init(const bool isDebug) {
    // spdlog 基础配置
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(isDebug ? spdlog::level::debug : spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);

    // Pipeline 参数（与原 main 保持一致；如需改为相机内参可在此读取 g_camera->get_param()）
    g_opt.model_path = "models/end2end.onnx";
    g_opt.calib_path = "config/params.xml";
    g_opt.score_thr = 0.7f;
    g_opt.mask_thr = 0.5f;
    g_opt.paint_masks_on_vis = true;

    // 初始化 pipeline（本地实现，内联）
    if (!g_ready) {
        try {
            FusionGeometry::initIntrinsic(g_opt.calib_path);
            FusionGeometry::initExtrinsic(g_opt.calib_path);
            g_K = FusionGeometry::getIntrinsic().clone(); // CV_64F
            g_Kinv = g_K.inv();
            g_Twc = FusionGeometry::getExtrinsic().clone(); // 4x4
            if (g_Twc.type() != CV_32F) g_Twc.convertTo(g_Twc, CV_32F);

            // 创建本地 ORT Session
            g_runner = std::make_unique<RunnerState>();
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            // 展开 to_wstring_local(g_opt.model_path)
            std::wstring model_path_wide(g_opt.model_path.begin(), g_opt.model_path.end());
            g_runner->session = std::make_unique<Ort::Session>(g_runner->env,
                                                               model_path_wide.c_str(), so);
            Ort::AllocatorWithDefaultOptions alloc;
            g_runner->in_name = g_runner->session->GetInputNameAllocated(0, alloc).get();
            size_t out_count = g_runner->session->GetOutputCount();
            g_runner->out_names_s.reserve(out_count);
            for (size_t i = 0; i < out_count; ++i) {
                g_runner->out_names_s.emplace_back(g_runner->session->GetOutputNameAllocated(i, alloc).get());
            }
            for (auto &s: g_runner->out_names_s) g_runner->out_names.push_back(s.c_str());
            g_ready = true;
        } catch (const std::exception &e) {
            spdlog::critical("Pipeline 初始化失败: {}", e.what());
            g_ready = false;
            return -1;
        }
    }

    // 初始化 camera
    if (!g_camera || !g_camera->isOpened()) {
        g_camera = std::make_unique<LanxinCamera>(); // 构造里会 connect()
        if (!g_camera->isOpened()) {
            spdlog::critical("LanxinCamera 连接失败");
            g_camera.reset();
            return -2;
        }
        spdlog::info("LanxinCamera 已连接");
    }

    spdlog::info("bs_yzx_init 完成（debug={}）", isDebug);
    return 0;
}

int bs_yzx_object_detection_lanxin(int taskId, zzb::Box boxArr[]) {
    if (!g_ready) return -10; // 未初始化 pipeline
    if (!g_camera || !g_camera->isOpened()) return -11; // 未初始化 camera

    auto t0 = std::chrono::steady_clock::now();

    // 0) 结果目录：res/<taskId>/
    const fs::path caseDir = fs::path(g_root_dir) / std::to_string(taskId);
    std::error_code ec_mkdir;
    fs::create_directories(caseDir, ec_mkdir); // 目录不存在则创建

    // 1) 从相机抓图像
    cv::Mat rgb;
    if (g_camera->CapFrame(rgb) != 0 || rgb.empty()) {
        spdlog::error("相机获取 RGB 帧失败");
        return -22;
    }

    // 1.1 保存原始 RGB
    const fs::path rgbPath = caseDir / "rgb_orig.jpg";
    if (!cv::imwrite(rgbPath.string(), rgb)) {
        spdlog::warn("保存原始 RGB 失败: {}", rgbPath.string()); // 非致命
    } else {
        spdlog::info("原始 RGB 已保存: {}", rgbPath.string());
    }

    // 2) 从相机抓点云
    open3d::geometry::PointCloud pc;
    if (g_camera->CapFrame(pc) != 0 || pc.points_.empty()) {
        spdlog::error("相机获取点云帧失败或为空");
        return -23;
    }

    const fs::path pcdPath = caseDir / "cloud_orig.pcd";

    open3d::io::WritePointCloudOption opt;
    opt.write_ascii = open3d::io::WritePointCloudOption::IsAscii::Ascii; // ASCII
    opt.compressed = open3d::io::WritePointCloudOption::Compressed::Uncompressed; // 不压缩
    opt.print_progress = false;

    bool ok_pcd = open3d::io::WritePointCloud(pcdPath.string(), pc, opt);
    if (!ok_pcd) {
        spdlog::warn("保存原始点云失败: {}", pcdPath.string());
    } else {
        spdlog::info("原始点云已保存: {}", pcdPath.string());
    }

    // 3) 执行 Pipeline（展开 pipeline_run）
    std::vector<LocalBoxPoseResult> results;
    cv::Mat vis;
    
    // 检查状态
    if (!g_ready) {
        spdlog::error("Pipeline 尚未 initialize()");
        return -24;
    }
    if (rgb.empty() || pc.points_.empty()) {
        spdlog::warn("输入为空: rgb.empty()={} pc.size()={}", rgb.empty(), pc.points_.size());
        return -24;
    }

    // 推理掩膜（展开 inferMasks_）
    std::vector<cv::Mat1b> masks;
    auto t0_infer = std::chrono::steady_clock::now();
    
    // 展开 infer_raw_local
    std::vector<Ort::Value> outs;
    if (rgb.empty()) {
        spdlog::error("输入图像为空");
        return -25;
    }
    cv::Mat rgb_converted;
    cv::cvtColor(rgb, rgb_converted, cv::COLOR_BGR2RGB);
    rgb_converted.convertTo(rgb_converted, CV_32F);
    
    // 展开 pixel_normalize_mmdet_rgb_local
    static const cv::Scalar mean(123.675, 116.28, 103.53);
    static const cv::Scalar stdv(58.395, 57.12, 57.375);
    cv::subtract(rgb_converted, mean, rgb_converted);
    cv::divide(rgb_converted, stdv, rgb_converted);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_converted, blob, 1.0, cv::Size(), {}, false, false, CV_32F);
    std::vector<int64_t> ishape = {1, 3, blob.size[2], blob.size[3]};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mi, reinterpret_cast<float *>(blob.data), static_cast<size_t>(blob.total()), ishape.data(), ishape.size());
    const char *in_names[] = {g_runner->in_name.c_str()};
    outs = g_runner->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                       g_runner->out_names.data(), g_runner->out_names.size());
    for (size_t idx = 0; idx < outs.size(); ++idx) {
        auto shape = outs[idx].GetTensorTypeAndShapeInfo().GetShape();
        std::ostringstream oss;
        oss << "outs[" << idx << "] shape = [";
        for (size_t j = 0; j < shape.size(); ++j) {
            oss << shape[j];
            if (j + 1 < shape.size()) oss << ", ";
        }
        oss << "]";
        spdlog::info("{}", oss.str());
    }
    
    auto t1_infer = std::chrono::steady_clock::now();
    double elapsed_ms_infer = std::chrono::duration<double, std::milli>(t1_infer - t0_infer).count();
    spdlog::info("paint:{}", elapsed_ms_infer);
    
    // 展开 paint_local
    if (rgb.empty()) {
        spdlog::error("paint_local: 输入图像为空");
        return -26;
    }
    if (outs.size() < 3) {
        spdlog::error("paint_local: 输出不足，期望3个");
        return -27;
    }
    const cv::Scalar color(0, 255, 0);
    const double alpha = 0.5;
    auto sh0 = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto sh2 = outs[2].GetTensorTypeAndShapeInfo().GetShape();
    int64_t N_paint = sh0[1];
    int mH_paint = (int) sh2[2];
    int mW_paint = (int) sh2[3];
    const float *dets = outs[0].GetTensorData<float>();
    const float *masks_data = outs[2].GetTensorData<float>();
    vis = rgb.clone();
    int kept = 0;
    for (int64_t i = 0; i < N_paint; ++i) {
        const float *r = dets + i * 5;
        float sc = r[4];
        if (sc < g_opt.score_thr) continue;
        int x1 = std::lround(r[0]), y1 = std::lround(r[1]);
        int x2 = std::lround(r[2]), y2 = std::lround(r[3]);
        x1 = std::clamp(x1, 0, vis.cols - 1);
        y1 = std::clamp(y1, 0, vis.rows - 1);
        x2 = std::clamp(x2, 0, vis.cols - 1);
        y2 = std::clamp(y2, 0, vis.rows - 1);
        cv::rectangle(vis, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", sc);
        cv::putText(vis, buf, {x1, std::max(0, y1 - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        int ow = std::max(1, x2 - x1), oh = std::max(1, y2 - y1);
        cv::Rect roi(x1, y1, ow, oh);
        roi &= cv::Rect(0, 0, vis.cols, vis.rows);
        if (roi.area() <= 0) continue;
        const float *mptr = masks_data + i * (mH_paint * mW_paint);
        cv::Mat mask_f32(mH_paint, mW_paint, CV_32F, const_cast<float *>(mptr));
        cv::Mat mask_up;
        cv::resize(mask_f32, mask_up, roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat1b mask8;
        cv::compare(mask_up, g_opt.mask_thr, mask8, cv::CMP_GT);
        cv::Mat roi_img = vis(roi);
        cv::Mat overlay = roi_img.clone();
        overlay.setTo(color, mask8);
        cv::addWeighted(roi_img, 1.0, overlay, 0.5, 0, roi_img);
        ++kept;
    }
    spdlog::info("绘制 {} 个实例", kept);
    
    if (vis.empty()) vis = rgb.clone();
    
    // 展开 infer_masks_local
    if (rgb.empty()) {
        spdlog::error("infer_masks_local: 输入图像为空");
        return -28;
    }
    if (outs.size() < 3) {
        spdlog::error("infer_masks_local: 输出不足，期望3个");
        return -29;
    }
    auto sh0_masks = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto sh2_masks = outs[2].GetTensorTypeAndShapeInfo().GetShape();
    int64_t N_masks = sh0_masks[1];
    int mH_masks = (int) sh2_masks[2];
    int mW_masks = (int) sh2_masks[3];
    const float *dets_masks = outs[0].GetTensorData<float>();
    const float *masks_data_masks = outs[2].GetTensorData<float>();
    masks.clear();
    masks.reserve((size_t) N_masks);
    for (int64_t i = 0; i < N_masks; ++i) {
        const float *r = dets_masks + i * 5;
        float sc = r[4];
        if (sc < g_opt.score_thr) continue;
        int x1 = std::lround(r[0]), y1 = std::lround(r[1]);
        int x2 = std::lround(r[2]), y2 = std::lround(r[3]);
        x1 = std::clamp(x1, 0, rgb.cols - 1);
        y1 = std::clamp(y1, 0, rgb.rows - 1);
        x2 = std::clamp(x2, 0, rgb.cols - 1);
        y2 = std::clamp(y2, 0, rgb.rows - 1);
        int ow = std::max(1, x2 - x1), oh = std::max(1, y2 - y1);
        cv::Rect roi(x1, y1, ow, oh);
        roi &= cv::Rect(0, 0, rgb.cols, rgb.rows);
        if (roi.area() <= 0) continue;
        const float *mptr = masks_data_masks + i * (mH_masks * mW_masks);
        cv::Mat mask_f32(mH_masks, mW_masks, CV_32F, const_cast<float *>(mptr));
        cv::Mat mask_up;
        cv::resize(mask_f32, mask_up, roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat1b mask8;
        cv::compare(mask_up, g_opt.mask_thr, mask8, cv::CMP_GT);
        cv::Mat1b fullMask(rgb.rows, rgb.cols, (uchar) 0);
        fullMask(roi).setTo(255, mask8);
        masks.emplace_back(std::move(fullMask));
    }
    spdlog::info("共导出 {} 个实例掩膜", masks.size());
    if (!g_opt.paint_masks_on_vis) vis = rgb.clone();

    // 收集矩形和底部中点（展开 collectRectsAndBottomMids_）
    std::vector<std::pair<cv::RotatedRect, cv::Point2f> > rect_and_mid;
    rect_and_mid.clear();
    rect_and_mid.reserve(masks.size());
    for (const auto &m: masks) {
        if (m.empty()) continue;
        cv::RotatedRect obb;
        if (FusionGeometry::maskToObb(m, obb)) {
            cv::Point2f midCenter;
            int midRadius;
            if (FusionGeometry::bottomMidpointCircle(obb, midCenter, midRadius)) {
                rect_and_mid.emplace_back(obb, midCenter);
            }
        }
    }

    // 投影点云（展开 projectPointCloud_）
    std::vector<Proj> proj;
    const double fx = g_K.at<double>(0, 0), fy = g_K.at<double>(1, 1);
    const double cx = g_K.at<double>(0, 2), cy = g_K.at<double>(1, 2);
    proj.clear();
    proj.reserve(pc.points_.size());
    for (int i = 0; i < (int) pc.points_.size(); ++i) {
        const auto &p = pc.points_[i];
        if (p.z() <= 0) continue;
        int u = (int) std::round(fx * p.x() / p.z() + cx);
        int v = (int) std::round(fy * p.y() / p.z() + cy);
        if ((unsigned) u >= (unsigned) rgb.cols || (unsigned) v >= (unsigned) rgb.rows) continue;
        proj.push_back({u, v, i});
    }

    results.clear();
    results.reserve(rect_and_mid.size());

    // 直接处理每个候选框，成功的立即加入结果并绘制
    for (size_t i = 0; i < rect_and_mid.size(); ++i) {
        LocalBoxPoseResult res;
        
        // 展开 solveOneBox_
        bool solved = false;
        const cv::RotatedRect &rrect = rect_and_mid[i].first;
        const cv::Point2f &midPx = rect_and_mid[i].second;

        std::vector<Eigen::Vector3d> rect_points;
        rect_points.reserve(4096);
        for (const auto &pr: proj) {
            // 展开 inRotRectFast - 判断点是否在旋转矩形内
            const float cx = rrect.center.x, cy = rrect.center.y;
            const float hw = rrect.size.width * 0.5f, hh = rrect.size.height * 0.5f;
            const float ang = rrect.angle * (float) CV_PI / 180.f;
            const float ca = std::cos(ang), sa = std::sin(ang);
            const float dx = (float) pr.u - cx, dy = (float) pr.v - cy;
            const float x = dx * ca + dy * sa;
            const float y = -dx * sa + dy * ca;
            bool in_rect = std::fabs(x) <= hw && std::fabs(y) <= hh;
            
            if (in_rect) {
                rect_points.push_back(pc.points_[pr.pid]);
            }
        }
        if (rect_points.size() < 30) {
            spdlog::info("[#{}] 框内点过少，跳过 ({} 点)", i, rect_points.size());
        } else {
            cv::Point2f p0, p1, p3;
            if (!FusionGeometry::bottomEdgeWithThirdPoint(rrect, p0, p1, p3)) {
                spdlog::info("[#{}] 无法获得底边两点/第三点", i);
            } else {
                // 展开 pix2dir
                auto pix2dir_impl = [](const cv::Point2f &px) -> Eigen::Vector3d {
                    cv::Vec3d hv(
                        g_Kinv.at<double>(0, 0) * px.x + g_Kinv.at<double>(0, 1) * px.y + g_Kinv.at<double>(0, 2),
                        g_Kinv.at<double>(1, 0) * px.x + g_Kinv.at<double>(1, 1) * px.y + g_Kinv.at<double>(1, 2),
                        g_Kinv.at<double>(2, 0) * px.x + g_Kinv.at<double>(2, 1) * px.y + g_Kinv.at<double>(2, 2)
                    );
                    Eigen::Vector3d v(hv[0], hv[1], hv[2]);
                    return v.normalized();
                };

                Eigen::Vector3d ray1_cam = pix2dir_impl(p0);
                Eigen::Vector3d ray2_cam = pix2dir_impl(p1);
                Eigen::Vector3d ray3_cam = pix2dir_impl(p3);

                cv::Point3f xyz_cam;
                cv::Vec3f n_cam, line_cam;
                cv::Point3f xyz1_cam, xyz2_cam, xyz3_cam;
                if (!FusionGeometry::computeBottomLineMidInfo3(
                    rect_points,
                    ray1_cam, ray2_cam, ray3_cam,
                    xyz_cam, n_cam, line_cam,
                    xyz1_cam, xyz2_cam, xyz3_cam)) {
                    spdlog::info("[#{}] 平面/交点求解失败", i);
                } else {
                    // 展开 reorder_vec3f
                    cv::Vec3f n_cam_re{n_cam[2], -n_cam[0], -n_cam[1]};
                    cv::Vec3f line_cam_re{line_cam[2], -line_cam[0], -line_cam[1]};

                    cv::Vec4f p_cam_re_mm(
                        xyz_cam.z * 1000.0f,
                        -xyz_cam.x * 1000.0f,
                        -xyz_cam.y * 1000.0f,
                        1.0f
                    );
                    cv::Mat T_wc = g_Twc.clone();
                    cv::Mat R_wc_33 = T_wc(cv::Rect(0, 0, 3, 3));

                    cv::Mat n_w_cv = R_wc_33 * cv::Mat(n_cam_re);
                    cv::Mat line_w_cv = R_wc_33 * cv::Mat(line_cam_re);
                    cv::Point3f n_w(n_w_cv.at<float>(0), n_w_cv.at<float>(1), n_w_cv.at<float>(2));
                    cv::Point3f y_w(line_w_cv.at<float>(0), line_w_cv.at<float>(1), line_w_cv.at<float>(2));

                    cv::Mat p_w_h = T_wc * cv::Mat(p_cam_re_mm);
                    cv::Point3f p_w_m(
                        p_w_h.at<float>(0) / 1000.0f,
                        p_w_h.at<float>(1) / 1000.0f,
                        p_w_h.at<float>(2) / 1000.0f
                    );

                    auto reorder_point = [](const cv::Point3f &p)-> cv::Vec4f {
                        return cv::Vec4f(p.z * 1000.0f, -p.x * 1000.0f, -p.y * 1000.0f, 1.0f);
                    };
                    cv::Mat p1_w_h = T_wc * cv::Mat(reorder_point(xyz1_cam));
                    cv::Mat p2_w_h = T_wc * cv::Mat(reorder_point(xyz2_cam));
                    cv::Mat p3_w_h = T_wc * cv::Mat(reorder_point(xyz3_cam));
                    cv::Point3f p1_w_m(p1_w_h.at<float>(0) / 1000.0f,
                                       p1_w_h.at<float>(1) / 1000.0f,
                                       p1_w_h.at<float>(2) / 1000.0f);
                    cv::Point3f p2_w_m(p2_w_h.at<float>(0) / 1000.0f,
                                       p2_w_h.at<float>(1) / 1000.0f,
                                       p2_w_h.at<float>(2) / 1000.0f);
                    cv::Point3f p3_w_m(p3_w_h.at<float>(0) / 1000.0f,
                                       p3_w_h.at<float>(1) / 1000.0f,
                                       p3_w_h.at<float>(2) / 1000.0f);

                    float width = 0.f, height = 0.f;
                    if (!FusionGeometry::calcWidthHeightFrom3Points(p1_w_m, p2_w_m, p3_w_m, width, height)) {
                        spdlog::warn("[#{}] 计算宽高失败", i);
                    }

                    auto norm_local = [](cv::Point3f v)-> cv::Point3f {
                        float L = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                        if (L < 1e-9f) return cv::Point3f(0, 0, 0);
                        return {v.x / L, v.y / L, v.z / L};
                    };
                    auto dot_local = [](const cv::Point3f &a, const cv::Point3f &b)-> float {
                        return a.x * b.x + a.y * b.y + a.z * b.z;
                    };
                    auto cross_local = [](const cv::Point3f &a, const cv::Point3f &b)-> cv::Point3f {
                        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
                    };

                    if (n_w.x < 0) { n_w = {-n_w.x, -n_w.y, -n_w.z}; }
                    if (y_w.y < 0) y_w = {-y_w.x, -y_w.y, -y_w.z};

                    cv::Point3f Xw = norm_local(n_w);
                    cv::Point3f Yw = norm_local(cv::Point3f(
                        y_w.x - dot_local(y_w, Xw) * Xw.x,
                        y_w.y - dot_local(y_w, Xw) * Xw.y,
                        y_w.z - dot_local(y_w, Xw) * Xw.z
                    ));
                    if (Yw.x == 0 && Yw.y == 0 && Yw.z == 0) {
                        cv::Point3f ref(0, 1, 0);
                        if (std::fabs(dot_local(ref, Xw)) > 0.95f) ref = {1, 0, 0};
                        Yw = norm_local(cv::Point3f(
                            ref.x - dot_local(ref, Xw) * Xw.x,
                            ref.y - dot_local(ref, Xw) * Xw.y,
                            ref.z - dot_local(ref, Xw) * Xw.z
                        ));
                    }
                    cv::Point3f Zw = norm_local(cross_local(Xw, Yw));
                    Yw = norm_local(cross_local(Zw, Xw));

                    Eigen::Matrix3d Rw;
                    Rw << Xw.x, Yw.x, Zw.x,
                            Xw.y, Yw.y, Zw.y,
                            Xw.z, Yw.z, Zw.z;

                    double pitch = std::asin(-Rw(2, 0));
                    double roll = std::atan2(Rw(2, 1), Rw(2, 2));
                    double yaw = std::atan2(Rw(1, 0), Rw(0, 0));
                    auto rad2deg = [](double v) { return v * 180.0 / 3.14159265358979323846; };

                    double W = rad2deg(roll);
                    double P = rad2deg(pitch);
                    double R = rad2deg(yaw);

                    res.id = static_cast<int>(i);
                    res.xyz_m = p_w_m;
                    res.wpr_deg = cv::Vec3f(static_cast<float>(W), static_cast<float>(P), static_cast<float>(R));
                    res.width_m = width;
                    res.height_m = height;
                    res.obb = rrect;
                    res.bottomMidPx = midPx;
                    res.p1_w_m = p1_w_m;
                    res.p2_w_m = p2_w_m;
                    res.p3_w_m = p3_w_m;
                    res.Rw = Rw;
                    solved = true;
                }
            }
        }
        
        if (solved) {
            results.emplace_back(std::move(res));
            const auto &r = results.back();
            // 展开 drawRotRect
            cv::Point2f pts[4];
            r.obb.points(pts);
            for (int j = 0; j < 4; ++j) cv::line(vis, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            
            cv::circle(vis, r.bottomMidPx, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            
            // 展开 drawEightLinesCentered_
            std::array<std::ostringstream, 8> oss;
            oss[0] << "#" << r.id;
            oss[1] << "x=" << std::fixed << std::setprecision(3) << r.xyz_m.x;
            oss[2] << "y=" << std::fixed << std::setprecision(3) << r.xyz_m.y;
            oss[3] << "z=" << std::fixed << std::setprecision(3) << r.xyz_m.z;
            oss[4] << "W=" << std::fixed << std::setprecision(1) << r.wpr_deg[0];
            oss[5] << "P=" << std::fixed << std::setprecision(1) << r.wpr_deg[1];
            oss[6] << "R=" << std::fixed << std::setprecision(1) << r.wpr_deg[2];
            oss[7] << std::fixed << std::setprecision(1) << r.width_m * 1000 << "," << r.height_m * 1000;

            const double fontScale = 0.45;
            const int thickness = 1;
            const cv::Scalar txtColor(0, 0, 0);
            const int lineGap = 4;

            std::array<cv::Size, 8> sizes;
            int base = 0;
            int totalH = 0;
            for (int j = 0; j < 8; ++j) {
                sizes[j] = cv::getTextSize(oss[j].str(),
                                           cv::FONT_HERSHEY_SIMPLEX,
                                           fontScale, thickness, &base);
                totalH += sizes[j].height;
            }
            totalH += lineGap * 7;

            cv::Point center = r.obb.center;
            int curY = (int) std::round(center.y - totalH * 0.5);

            for (int j = 0; j < 8; ++j) {
                const std::string txt = oss[j].str();
                const cv::Size &tsz = sizes[j];
                int orgX = (int) std::round(center.x - tsz.width * 0.5);
                int orgY = curY + tsz.height;
                cv::putText(vis, txt, cv::Point(orgX, orgY),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness, cv::LINE_AA);
                curY += tsz.height + lineGap;
            }
        }
    }

    // 4) 输出可视化到 res/<taskId>/vis_on_orig.jpg
    const fs::path outPath = caseDir / "vis_on_orig.jpg";
    if (!cv::imwrite(outPath.string(), vis)) {
        spdlog::error("写可视化文件失败: {}", outPath.string()); // 非致命
    }

    auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5) 写回 boxArr（最多 YZX_MAX_BOX 个）
    const int total = static_cast<int>(results.size());
    const int n_write = (total < YZX_MAX_BOX) ? total : YZX_MAX_BOX;
    for (int i = 0; i < n_write; ++i) {
        // 展开 write_one_box
        const LocalBoxPoseResult &src = results[i];
        ::zzb::Box &dst = boxArr[i];
        
        // 坐标（米）
        dst.x = static_cast<double>(src.xyz_m.x);
        dst.y = static_cast<double>(src.xyz_m.y);
        dst.z = static_cast<double>(src.xyz_m.z);

        // 尺寸（米）
        dst.width = static_cast<double>(src.width_m);
        dst.height = static_cast<double>(src.height_m);

        // 角度（度）：W、P、R → angle_a、angle_b、angle_c
        dst.angle_a = static_cast<double>(src.wpr_deg[0]);
        dst.angle_b = static_cast<double>(src.wpr_deg[1]);
        dst.angle_c = static_cast<double>(src.wpr_deg[2]);

        // 旋转矩阵（行优先展开为 r1..r9）
        const Eigen::Matrix3d &R = src.Rw;
        dst.rw1 = static_cast<double>(R(0, 0));
        dst.rw2 = static_cast<double>(R(0, 1));
        dst.rw3 = static_cast<double>(R(0, 2));
        dst.rw4 = static_cast<double>(R(1, 0));
        dst.rw5 = static_cast<double>(R(1, 1));
        dst.rw6 = static_cast<double>(R(1, 2));
        dst.rw7 = static_cast<double>(R(2, 0));
        dst.rw8 = static_cast<double>(R(2, 1));
        dst.rw9 = static_cast<double>(R(2, 2));
    }

    spdlog::info("[ OK ] taskId={} -> {}，目标数={}（写入 {} 个），耗时={:.3f} ms",
                 taskId, outPath.string(), total, n_write, elapsed_ms);

    // 6) 写入 JSON（增加原始数据文件路径）
    json j;
    j["taskId"] = taskId;
    j["elapsed_ms"] = elapsed_ms;
    j["total"] = total;
    j["n_write"] = n_write;
    j["out_image"] = outPath.string();
    j["raw_rgb"] = rgbPath.string();
    j["raw_pcd"] = pcdPath.string();

    auto &arr = j["boxes"] = json::array();
    for (int i = 0; i < n_write; ++i) {
        const auto &b = boxArr[i];
        arr.push_back({
            {"x", b.x}, {"y", b.y}, {"z", b.z},
            {"width", b.width}, {"height", b.height},
            {"angle_a", b.angle_a}, {"angle_b", b.angle_b}, {"angle_c", b.angle_c},
            {"rw1", b.rw1}, {"rw2", b.rw2}, {"rw3", b.rw3},
            {"rw4", b.rw4}, {"rw5", b.rw5}, {"rw6", b.rw6},
            {"rw7", b.rw7}, {"rw8", b.rw8}, {"rw9", b.rw9}
        });
    }
    const fs::path jsonPath = caseDir / "boxes.json";
    std::ofstream ofs(jsonPath);
    if (!ofs) {
        spdlog::error("写 JSON 失败（无法打开文件）: {}", jsonPath.string());
    } else {
        ofs << std::setw(2) << j;
        ofs.close();
        spdlog::info("JSON 已写入: {}", jsonPath.string());
    }
    return n_write; // 返回写入个数
}
