#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cpu_library.h"
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
    std::string g_model_path;
    std::string g_calib_path;
    float g_score_thr = 0.8f;
    float g_mask_thr = 0.6f;
    bool g_paint_masks_on_vis = true;

    // FusionGeometry 静态变量（原来的类成员变量）
    cv::Mat g_intrinsic;
    cv::Mat g_T_world_cam;


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


    struct RunnerState {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mrcnn"};
        std::unique_ptr<Ort::Session> session;
        std::string in_name;
        std::vector<std::string> out_names_s;
        std::vector<const char *> out_names;
    };

    std::unique_ptr<RunnerState> g_runner;
    cv::Mat g_K, g_Kinv, g_Twc; // K: CV_64F, Twc: CV_32F
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
    g_model_path = "models/end2end.onnx";
    g_calib_path = "config/params.xml";
    g_score_thr = 0.7f;
    g_mask_thr = 0.5f;
    g_paint_masks_on_vis = true;

    // 初始化 pipeline（本地实现，内联）
    if (!g_ready) {
        try {
            // 展开 initIntrinsic_local
            cv::FileStorage fs_intrinsic(g_calib_path, cv::FileStorage::READ);
            if (!fs_intrinsic.isOpened()) return -25;
            fs_intrinsic["intrinsicRGB"] >> g_intrinsic;
            if (g_intrinsic.empty() || g_intrinsic.rows!=3 || g_intrinsic.cols!=3) return -26;
            if (g_intrinsic.type() != CV_64F) g_intrinsic.convertTo(g_intrinsic, CV_64F);
            
            // 展开 initExtrinsic_local  
            cv::FileStorage fs_extrinsic(g_calib_path, cv::FileStorage::READ);
            if (!fs_extrinsic.isOpened()) {
                spdlog::error("[initExtrinsic] 打不开外参文件: {}", g_calib_path);
                return -27;
            }
            
            fs_extrinsic["extrinsicRGB"] >> g_T_world_cam;
            fs_extrinsic.release();
            
            if (g_T_world_cam.empty()) {
                spdlog::error("[initExtrinsic] extrinsicRGB 节点不存在或为空");
                return -28;
            }
            
            if (g_T_world_cam.rows != 4 || g_T_world_cam.cols != 4) {
                spdlog::error("[initExtrinsic] extrinsicRGB 必须是 4x4 矩阵");
                return -29;
            }
            
            if (g_T_world_cam.type() != CV_64F) {
                g_T_world_cam.convertTo(g_T_world_cam, CV_64F);
            }
            
            g_K = g_intrinsic.clone(); // CV_64F
            g_Kinv = g_K.inv();
            g_Twc = g_T_world_cam.clone(); // 4x4
            if (g_Twc.type() != CV_32F) g_Twc.convertTo(g_Twc, CV_32F);

            // 创建本地 ORT Session (CPU版本)
            g_runner = std::make_unique<RunnerState>();
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // CPU版本：不设置CUDA执行提供程序，使用默认CPU提供程序
            auto to_wstring = [](const std::string& s) -> std::wstring {
                return { s.begin(), s.end() };
            };
            
            g_runner->session = std::make_unique<Ort::Session>(g_runner->env,
                                                               to_wstring(g_model_path).c_str(), so);
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


    spdlog::info("bs_yzx_init 完成（debug={}）", isDebug);
    return 0;
}

// CPU版本的目标检测函数，从文件读取RGB图像和点云数据
// taskId: 用于指定读取数据的子目录 res/<taskId>/
// 需要的文件：rgb.jpg（RGB图像）、pcAll.pcd（点云数据）
int bs_yzx_object_detection_lanxin(int taskId, zzb::Box boxArr[]) {
    if (!g_ready) return -10; // 未初始化 pipeline

    auto t0 = std::chrono::steady_clock::now();

    // 0) 数据目录：res/<taskId>/ (从此目录读取输入数据)
    const fs::path caseDir = fs::path(g_root_dir) / std::to_string(taskId);
    if (!fs::exists(caseDir)) {
        spdlog::error("数据目录不存在: {}", caseDir.string());
        return -21;
    }

    // 从文件夹读取RGB和点云数据
    cv::Mat rgb;
    open3d::geometry::PointCloud pc;
    
    // 读取RGB图像：rgb.jpg
    const fs::path rgbPath = caseDir / "rgb.jpg";
    if (!fs::exists(rgbPath)) {
        spdlog::error("RGB文件不存在: {}", rgbPath.string());
        return -22;
    }
    
    rgb = cv::imread(rgbPath.string(), cv::IMREAD_COLOR);
    if (rgb.empty()) {
        spdlog::error("无法读取RGB图像: {}", rgbPath.string());
        return -22;
    }
    spdlog::info("从文件读取RGB: {}", rgbPath.string());
    
    // 读取点云数据：pcAll.pcd
    const fs::path pcdPath = caseDir / "pcAll.pcd";
    if (!fs::exists(pcdPath)) {
        spdlog::error("点云文件不存在: {}", pcdPath.string());
        return -23;
    }
    
    if (!open3d::io::ReadPointCloud(pcdPath.string(), pc) || pc.points_.empty()) {
        spdlog::error("无法读取点云数据: {}", pcdPath.string());
        return -23;
    }
    spdlog::info("从文件读取点云: {}", pcdPath.string());


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


    std::vector<cv::Mat1b> masks;
    auto t0_infer = std::chrono::steady_clock::now();
    
    std::vector<Ort::Value> outs;
    if (rgb.empty()) {
        spdlog::error("输入图像为空");
        return -25;
    }
    cv::Mat rgb_converted;
    cv::cvtColor(rgb, rgb_converted, cv::COLOR_BGR2RGB);
    rgb_converted.convertTo(rgb_converted, CV_32F);
    
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
        if (sc < g_score_thr) continue;
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
        cv::compare(mask_up, g_mask_thr, mask8, cv::CMP_GT);
        cv::Mat roi_img = vis(roi);
        cv::Mat overlay = roi_img.clone();
        overlay.setTo(color, mask8);
        cv::addWeighted(roi_img, 1.0, overlay, 0.5, 0, roi_img);
        ++kept;
    }
    spdlog::info("绘制 {} 个实例", kept);
    
    if (vis.empty()) vis = rgb.clone();
    
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
        if (sc < g_score_thr) continue;
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
        cv::compare(mask_up, g_mask_thr, mask8, cv::CMP_GT);
        cv::Mat1b fullMask(rgb.rows, rgb.cols, (uchar) 0);
        fullMask(roi).setTo(255, mask8);
        masks.emplace_back(std::move(fullMask));
    }
    spdlog::info("共导出 {} 个实例掩膜", masks.size());
    if (!g_paint_masks_on_vis) vis = rgb.clone();

    std::vector<std::pair<cv::RotatedRect, cv::Point2f> > rect_and_mid;
    rect_and_mid.clear();
    rect_and_mid.reserve(masks.size());
    for (const auto &m: masks) {
        if (m.empty()) continue;
        cv::RotatedRect obb;
        // 展开 maskToObb_local
        obb = cv::RotatedRect();
        if (!m.empty()) {
            // 1) 做一个"非零即前景"的二值图（不做阈值筛选、不做形态学）
            cv::Mat1b bin;
            if (m.type() == CV_8U) {
                bin = m != 0;                      // 只看非零
            } else {
                cv::Mat tmp;
                cv::compare(m, 0, tmp, cv::CMP_GT); // >0 视为前景；结果是 8U 0/255
                bin = tmp;
            }

            // 2) 收集所有前景像素坐标（把所有连通块一起算）
            std::vector<cv::Point> pts;
            cv::findNonZero(bin, pts);

            // 3) 没有前景像素就跳过；否则直接对所有点做最小外接矩形
            if (!pts.empty()) {
                obb = cv::minAreaRect(pts);           // angle ∈ (-90, 0]
                
                // 展开 bottomMidpointCircle_local
                if (obb.size.width > 0 && obb.size.height > 0) {
                    cv::Point2f pts_obb[4]; 
                    obb.points(pts_obb);
                    int ei = 0, ej = 1; 
                    float bestAvgY = -1e30f;

                    auto consider_edge = [&](int a, int b) {
                        float avgY = (pts_obb[a].y + pts_obb[b].y) * 0.5f;
                        if (avgY > bestAvgY) { bestAvgY = avgY; ei = a; ej = b; }
                    };
                    consider_edge(0,1); consider_edge(1,2); consider_edge(2,3); consider_edge(3,0);

                    cv::Point2f midCenter = (pts_obb[ei] + pts_obb[ej]) * 0.5f;
                    float major = (obb.size.width > obb.size.height) ? obb.size.width : obb.size.height;
                    int midRadius = cvRound(major * 0.02f);
                    if (midRadius < 2) midRadius = 2;
                    
                    rect_and_mid.emplace_back(obb, midCenter);
                }
            }
        }
    }

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

    for (size_t i = 0; i < rect_and_mid.size(); ++i) {
        LocalBoxPoseResult res;
        
        bool solved = false;
        const cv::RotatedRect &rrect = rect_and_mid[i].first;
        const cv::Point2f &midPx = rect_and_mid[i].second;

        std::vector<Eigen::Vector3d> rect_points;
        rect_points.reserve(4096);
        for (const auto &pr: proj) {
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
            // 展开 bottomEdgeWithThirdPoint_local
            bool bottomEdgeFound = false;
            if (rrect.size.width > 0 && rrect.size.height > 0) {
                cv::Point2f pts_edge[4];
                rrect.points(pts_edge);  // OpenCV 返回顺时针 4 顶点

                // 1) 选"平均 y 最大"的那条边为底边（图像坐标 y 向下）
                int ei = 0, ej = 1;        // 底边的两个顶点下标
                float bestAvgY = -1e30f;

                auto consider_edge = [&](int a, int b) {
                    float avgY = 0.5f * (pts_edge[a].y + pts_edge[b].y);
                    if (avgY > bestAvgY) { bestAvgY = avgY; ei = a; ej = b; }
                };
                consider_edge(0,1);
                consider_edge(1,2);
                consider_edge(2,3);
                consider_edge(3,0);

                // 2) 取出底边两个点
                cv::Point2f b0 = pts_edge[ei];
                cv::Point2f b1 = pts_edge[ej];

                // 3) 可选：保证从左到右输出（按 x 升序）
                if (b0.x > b1.x) std::swap(b0, b1);

                p0 = b0;
                p1 = b1;

                // 4) 从剩余两个点中选一个作为 p3 —— 取与 p0 距离更近的那个
                //    先找出剩余两个顶点下标
                bool used[4] = {false,false,false,false};
                used[ei] = true; used[ej] = true;

                int rIdx[2]; int r = 0;
                for (int k = 0; k < 4; ++k) if (!used[k]) rIdx[r++] = k;

                const cv::Point2f& q0 = pts_edge[rIdx[0]];
                const cv::Point2f& q1 = pts_edge[rIdx[1]];

                float d0 = cv::norm(q0 - p0);
                float d1 = cv::norm(q1 - p0);
                p3 = (d0 <= d1) ? q0 : q1;
                
                bottomEdgeFound = true;
            }
            
            if (!bottomEdgeFound) {
                spdlog::info("[#{}] 无法获得底边两点/第三点", i);
            } else {
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
                // 展开 computeBottomLineMidInfo3_local
                bool planeComputed = false;
                if (rect_points.size() >= 30) {
                    // 1) 用 RANSAC 拟合平面（相机系）
                    auto rect_pc = std::make_shared<open3d::geometry::PointCloud>();
                    rect_pc->points_.assign(rect_points.begin(), rect_points.end());

                    const double distance_threshold = 0.004; // 4 mm
                    const int    ransac_n           = 3;
                    const int    num_iterations     = 300;

                    Eigen::Vector4d plane; std::vector<size_t> inliers;
                    std::tie(plane, inliers) = rect_pc->SegmentPlane(
                        distance_threshold, ransac_n, num_iterations
                    );
                    if (inliers.size() >= 20) {
                        // 2) 规范化平面参数 [n | d]，并做方向规整（让法向朝向相机 +Z）
                        Eigen::Vector3d n(plane[0], plane[1], plane[2]);
                        double d = plane[3];

                        const double L = n.norm();
                        if (L > 0.0) {
                            n /= L; d /= L;

                            if (n.z() < 0) { n = -n; d = -d; } // 翻转时同步翻 d，保持 n·X + d = 0

                            // 3) 三条射线与平面求交（原逻辑对 1、2 射线保持不变，仅新增第 3 条）
                            if (ray1_cam.norm() > 0.0 && ray2_cam.norm() > 0.0 && ray3_cam.norm() > 0.0) {
                                const Eigen::Vector3d dir1 = ray1_cam.normalized();
                                const Eigen::Vector3d dir2 = ray2_cam.normalized();
                                const Eigen::Vector3d dir3 = ray3_cam.normalized();   // 新增

                                const double nd1 = n.dot(dir1);
                                const double nd2 = n.dot(dir2);
                                const double nd3 = n.dot(dir3);                           // 新增
                                if (std::abs(nd1) >= 1e-8 && std::abs(nd2) >= 1e-8 && std::abs(nd3) >= 1e-8) {

                                    const double t1 = -d / nd1;
                                    const double t2 = -d / nd2;
                                    const double t3 = -d / nd3;                               // 新增
                                    if (t1 > 0 && t2 > 0 && t3 > 0) {
                                        const Eigen::Vector3d P1 = t1 * dir1;
                                        const Eigen::Vector3d P2 = t2 * dir2;
                                        const Eigen::Vector3d P3 = t3 * dir3;                     // 新增

                                        // 4) 中点与底边方向（保持原逻辑：仅由 P1、P2 计算）
                                        const Eigen::Vector3d M  = 0.5 * (P1 + P2);
                                        Eigen::Vector3d edge_dir = P2 - P1;
                                        const double edge_len = edge_dir.norm();
                                        if (edge_len >= 1e-8) {
                                            edge_dir /= edge_len;

                                            // （可选）方向一致性：让 (edge_dir × n).z >= 0（保持原版一致）
                                            Eigen::Vector3d x_tmp = edge_dir.cross(n);
                                            if (x_tmp.z() < 0) edge_dir = -edge_dir;

                                            // 5) 输出（含新增三交点）
                                            xyz_cam      = cv::Point3f(static_cast<float>(M.x()),
                                                                       static_cast<float>(M.y()),
                                                                       static_cast<float>(M.z()));
                                            n_cam        = cv::Vec3f(static_cast<float>(n.x()),
                                                                     static_cast<float>(n.y()),
                                                                     static_cast<float>(n.z()));
                                            line_cam = cv::Vec3f(static_cast<float>(edge_dir.x()),
                                                                         static_cast<float>(edge_dir.y()),
                                                                         static_cast<float>(edge_dir.z()));

                                            xyz1_cam     = cv::Point3f(static_cast<float>(P1.x()),
                                                                       static_cast<float>(P1.y()),
                                                                       static_cast<float>(P1.z()));
                                            xyz2_cam     = cv::Point3f(static_cast<float>(P2.x()),
                                                                       static_cast<float>(P2.y()),
                                                                       static_cast<float>(P2.z()));
                                            xyz3_cam     = cv::Point3f(static_cast<float>(P3.x()),
                                                                       static_cast<float>(P3.y()),
                                                                       static_cast<float>(P3.z()));
                                            
                                            planeComputed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (!planeComputed) {
                    spdlog::info("[#{}] 平面/交点求解失败", i);
                } else {
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
                    // 展开 calcWidthHeightFrom3Points_local
                    // 以 p1 为公共顶点
                    const cv::Point3f v12 = p2_w_m - p1_w_m; // 宽方向向量
                    const cv::Point3f v13 = p3_w_m - p1_w_m; // 高方向向量

                    const float w = std::sqrt(v12.x * v12.x + v12.y * v12.y + v12.z * v12.z);
                    const float h = std::sqrt(v13.x * v13.x + v13.y * v13.y + v13.z * v13.z);

                    // 基本健壮性检查
                    if (std::isfinite(w) && std::isfinite(h)) {
                        width = w;
                        height = h;
                    } else {
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
            cv::Point2f pts[4];
            r.obb.points(pts);
            for (int j = 0; j < 4; ++j) cv::line(vis, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            
            cv::circle(vis, r.bottomMidPx, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            
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

    spdlog::info("[ OK ] taskId={} -> {}，目标数={}（写入 {} 个），耗时={:.3f} ms，输入：rgb.jpg, pcAll.pcd",
                 taskId, outPath.string(), total, n_write, elapsed_ms);

    // 6) 写入 JSON（增加原始数据文件路径）
    json j;
    j["taskId"] = taskId;
    j["elapsed_ms"] = elapsed_ms;
    j["total"] = total;
    j["n_write"] = n_write;
    j["out_image"] = outPath.string();
    j["input_rgb"] = rgbPath.string();
    j["input_pcd"] = pcdPath.string();

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
