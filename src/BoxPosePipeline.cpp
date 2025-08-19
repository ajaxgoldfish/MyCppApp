#include "BoxPosePipeline.h"
#include "MaskRCNNRunner.h"
#include "FusionGeometry.h"

#include <spdlog/spdlog.h>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <iomanip>
#include <array>
#include <cmath>

// ===================== 构造 & 初始化 =====================
BoxPosePipeline::BoxPosePipeline(const Options& opt)
    : options_(opt) {}
BoxPosePipeline::~BoxPosePipeline() = default;

bool BoxPosePipeline::initialize() {
    return loadCalibrationAndModel_();
}

bool BoxPosePipeline::loadCalibrationAndModel_() {
    try {
        // 读取标定
        FusionGeometry::initIntrinsic(options_.calib_path);
        FusionGeometry::initExtrinsic(options_.calib_path);

        K_    = FusionGeometry::getIntrinsic().clone(); // CV_64F
        Kinv_ = K_.inv();

        T_wc_ = FusionGeometry::getExtrinsic().clone(); // 4x4
        if (T_wc_.type() != CV_32F) T_wc_.convertTo(T_wc_, CV_32F);

        // 模型
        runner_ = std::make_unique<MaskRCNNRunner>(options_.model_path);

        ready_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::error("初始化失败: {}", e.what());
        ready_ = false;
        return false;
    }
}

// ===================== 主入口 =====================
bool BoxPosePipeline::run(const cv::Mat& rgb,
                          const open3d::geometry::PointCloud& pc_cam,
                          std::vector<BoxPoseResult>& results,
                          cv::Mat* vis_out) {
    if (!ready_) {
        spdlog::error("Pipeline 尚未 initialize()");
        return false;
    }
    if (rgb.empty() || pc_cam.points_.empty()) {
        spdlog::warn("输入为空: rgb.empty()={} pc.size()={}", rgb.empty(), pc_cam.points_.size());
        return false;
    }

    // 1) 推理（得到可视化底图与 masks）
    cv::Mat vis;
    std::vector<cv::Mat1b> masks;
    if (!inferMasks_(rgb, vis, masks)) return false;
    if (!options_.paint_masks_on_vis) vis = rgb.clone();

    // 2) 集合 OBB 与底边中点
    std::vector<std::pair<cv::RotatedRect, cv::Point2f>> rect_and_mid;
    collectRectsAndBottomMids_(masks, rect_and_mid);

    // 3) 点云投影
    std::vector<Proj> proj;
    projectPointCloud_(pc_cam, rgb.cols, rgb.rows, proj);

    // 4) 逐个实例求解
    results.clear();
    results.reserve(rect_and_mid.size());
    for (size_t i = 0; i < rect_and_mid.size(); ++i) {
        BoxPoseResult res;
        if (solveOneBox_(i, rect_and_mid[i], proj, pc_cam, &vis, res)) {
            results.push_back(res);
        }
    }

    if (vis_out) *vis_out = vis;
    return true;
}

// ===================== 步骤实现 =====================
bool BoxPosePipeline::inferMasks_(const cv::Mat& rgb,
                                  cv::Mat& vis,
                                  std::vector<cv::Mat1b>& masks) {
    std::vector<Ort::Value> outs = runner_->inferRaw(rgb);
    vis = runner_->paint(rgb, outs, options_.score_thr, options_.mask_thr);
    if (vis.empty()) vis = rgb.clone();
    masks = runner_->inferMasks(rgb, outs, options_.score_thr, options_.mask_thr);
    return true;
}

void BoxPosePipeline::collectRectsAndBottomMids_(
        const std::vector<cv::Mat1b>& masks,
        std::vector<std::pair<cv::RotatedRect, cv::Point2f>>& rect_and_mid) const {
    rect_and_mid.clear();
    rect_and_mid.reserve(masks.size());
    for (const auto& m : masks) {
        if (m.empty()) continue;

        cv::RotatedRect obb;
        if (FusionGeometry::maskToObb(m, obb)) {
            cv::Point2f midCenter; int midRadius;
            if (FusionGeometry::bottomMidpointCircle(obb, midCenter, midRadius)) {
                rect_and_mid.emplace_back(obb, midCenter);
            }
        }
    }
}

void BoxPosePipeline::projectPointCloud_(const open3d::geometry::PointCloud& pc_cam,
                                         int W, int H,
                                         std::vector<Proj>& proj) const {
    const double fx = K_.at<double>(0,0), fy = K_.at<double>(1,1);
    const double cx = K_.at<double>(0,2), cy = K_.at<double>(1,2);

    proj.clear();
    proj.reserve(pc_cam.points_.size());

    for (int i = 0; i < (int)pc_cam.points_.size(); ++i) {
        const auto& p = pc_cam.points_[i];
        if (p.z() <= 0) continue;
        int u = (int)std::round(fx * p.x()/p.z() + cx);
        int v = (int)std::round(fy * p.y()/p.z() + cy);
        if ((unsigned)u >= (unsigned)W || (unsigned)v >= (unsigned)H) continue;
        proj.push_back({u, v, i});
    }
}

// ===================== 单个实例完整求解 =====================
bool BoxPosePipeline::solveOneBox_(size_t idx,
                                   const std::pair<cv::RotatedRect, cv::Point2f>& rect_mid,
                                   const std::vector<Proj>& proj,
                                   const open3d::geometry::PointCloud& pc_cam,
                                   cv::Mat* vis_io,
                                   BoxPoseResult& out) {
    const cv::RotatedRect& rrect = rect_mid.first;
    const cv::Point2f&     midPx = rect_mid.second;

    // 0) 收集该 OBB 内的点云（相机系，单位 m）
    std::vector<Eigen::Vector3d> rect_points;
    rect_points.reserve(4096);
    for (const auto& pr : proj) {
        if (inRotRectFast(rrect, pr.u, pr.v)) {
            rect_points.push_back(pc_cam.points_[pr.pid]);
        }
    }
    if (rect_points.size() < 30) {
        spdlog::info("[#{}] 框内点过少，跳过 ({} 点)", idx, rect_points.size());
        return false;
    }

    // 1) OBB 底边两个像素点 + 第三个点
    cv::Point2f p0, p1, p3;
    if (!FusionGeometry::bottomEdgeWithThirdPoint(rrect, p0, p1, p3)) {
        spdlog::info("[#{}] 无法获得底边两点/第三点", idx);
        return false;
    }

    // 2) 像素(u,v) -> 相机系单位视线
    Eigen::Vector3d ray1_cam = pix2dir(p0);
    Eigen::Vector3d ray2_cam = pix2dir(p1);
    Eigen::Vector3d ray3_cam = pix2dir(p3);

    // 3) 拟合平面 + 三射线与平面求交（相机系：中点/法向/线方向）
    cv::Point3f xyz_cam;
    cv::Vec3f   n_cam, line_cam;
    cv::Point3f xyz1_cam, xyz2_cam, xyz3_cam; // 三个交点（相机系，m）

    if (!FusionGeometry::computeBottomLineMidInfo3(
            rect_points,
            ray1_cam, ray2_cam, ray3_cam,
            xyz_cam, n_cam, line_cam,
            xyz1_cam, xyz2_cam, xyz3_cam)) {
        spdlog::info("[#{}] 平面/交点求解失败", idx);
        return false;
    }

    // 4) 轴重排（相机 -> 厂商相机轴）：(x,y,z)->(z,-x,-y)
    cv::Vec3f n_cam_re    = reorder_vec3f(n_cam);
    cv::Vec3f line_cam_re = reorder_vec3f(line_cam);

    // 点（相机系 m）先重排再转 mm（与外参匹配），齐次
    cv::Vec4f p_cam_re_mm(
        xyz_cam.z * 1000.0f,
       -xyz_cam.x * 1000.0f,
       -xyz_cam.y * 1000.0f,
        1.0f
    );

    // 5) 外参（World <- 厂商相机，mm）
    cv::Mat T_wc = T_wc_.clone();               // 4x4, CV_32F
    cv::Mat R_wc_33 = T_wc(cv::Rect(0,0,3,3));  // 3x3
    cv::Mat t_wc_31 = T_wc(cv::Rect(3,0,1,3));  // 3x1

    // —— 法向/线方向：只旋转
    cv::Mat n_w_cv    = R_wc_33 * cv::Mat(n_cam_re);
    cv::Mat line_w_cv = R_wc_33 * cv::Mat(line_cam_re);
    cv::Point3f n_w(n_w_cv.at<float>(0), n_w_cv.at<float>(1), n_w_cv.at<float>(2));
    cv::Point3f y_w(line_w_cv.at<float>(0), line_w_cv.at<float>(1), line_w_cv.at<float>(2));

    // —— 中点：p_world = T_wc * p_cam_re_mm（得到 mm）→ m
    cv::Mat p_w_h = T_wc * cv::Mat(p_cam_re_mm);
    cv::Point3f p_w_m(
        p_w_h.at<float>(0)/1000.0f,
        p_w_h.at<float>(1)/1000.0f,
        p_w_h.at<float>(2)/1000.0f
    );

    // ========== 三个交点的重排 + 转换 ==========
    auto reorder_point = [](const cv::Point3f& p)->cv::Vec4f {
        return cv::Vec4f(p.z * 1000.0f, -p.x * 1000.0f, -p.y * 1000.0f, 1.0f);
    };

    cv::Vec4f p1_cam_re_mm = reorder_point(xyz1_cam);
    cv::Vec4f p2_cam_re_mm = reorder_point(xyz2_cam);
    cv::Vec4f p3_cam_re_mm = reorder_point(xyz3_cam);

    cv::Mat p1_w_h = T_wc * cv::Mat(p1_cam_re_mm);
    cv::Mat p2_w_h = T_wc * cv::Mat(p2_cam_re_mm);
    cv::Mat p3_w_h = T_wc * cv::Mat(p3_cam_re_mm);

    cv::Point3f p1_w_m(
        p1_w_h.at<float>(0)/1000.0f,
        p1_w_h.at<float>(1)/1000.0f,
        p1_w_h.at<float>(2)/1000.0f
    );
    cv::Point3f p2_w_m(
        p2_w_h.at<float>(0)/1000.0f,
        p2_w_h.at<float>(1)/1000.0f,
        p2_w_h.at<float>(2)/1000.0f
    );
    cv::Point3f p3_w_m(
        p3_w_h.at<float>(0)/1000.0f,
        p3_w_h.at<float>(1)/1000.0f,
        p3_w_h.at<float>(2)/1000.0f
    );

    float width = 0.f, height = 0.f;
    if (!FusionGeometry::calcWidthHeightFrom3Points(p1_w_m, p2_w_m, p3_w_m, width, height)) {
        spdlog::warn("[#{}] 计算宽高失败", idx);
    }

    // 6) 在世界系构姿态：X=法向；Y=线方向的平面投影；Z=X×Y；再正交
    auto norm_local = [](cv::Point3f v)->cv::Point3f{
        float L = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (L < 1e-9f) return cv::Point3f(0,0,0);
        return { v.x/L, v.y/L, v.z/L };
    };
    auto dot_local = [](const cv::Point3f& a, const cv::Point3f& b)->float{
        return a.x*b.x + a.y*b.y + a.z*b.z;
    };
    auto cross_local = [](const cv::Point3f& a, const cv::Point3f& b)->cv::Point3f{
        return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
    };

    // 可选：统一法向使其背向相机（让法向指向相机反方向）
    cv::Point3f C_w_m(
        t_wc_31.at<float>(0)/1000.0f,
        t_wc_31.at<float>(1)/1000.0f,
        t_wc_31.at<float>(2)/1000.0f
    );
    cv::Point3f CP(C_w_m.x - p_w_m.x, C_w_m.y - p_w_m.y, C_w_m.z - p_w_m.z);
    if (dot_local(n_w, CP) > 0) n_w = { -n_w.x, -n_w.y, -n_w.z };

    // 让线方向的 y 分量大于 0（约定）
    if (y_w.y < 0) y_w = { -y_w.x, -y_w.y, -y_w.z };

    cv::Point3f Xw = norm_local(n_w);
    cv::Point3f Yw = norm_local(cv::Point3f(
        y_w.x - dot_local(y_w, Xw)*Xw.x,
        y_w.y - dot_local(y_w, Xw)*Xw.y,
        y_w.z - dot_local(y_w, Xw)*Xw.z
    ));
    if (Yw.x==0 && Yw.y==0 && Yw.z==0) { // 退化兜底
        cv::Point3f ref(0,1,0);
        if (std::fabs(dot_local(ref, Xw)) > 0.95f) ref = {1,0,0};
        Yw = norm_local(cv::Point3f(
            ref.x - dot_local(ref, Xw)*Xw.x,
            ref.y - dot_local(ref, Xw)*Xw.y,
            ref.z - dot_local(ref, Xw)*Xw.z
        ));
    }
    cv::Point3f Zw = norm_local(cross_local(Xw, Yw));
    Yw = norm_local(cross_local(Zw, Xw));

    // 7) Rw（列向量 X/Y/Z） & 提取 ZYX->WPR
    Eigen::Matrix3d Rw;
    Rw << Xw.x, Yw.x, Zw.x,
          Xw.y, Yw.y, Zw.y,
          Xw.z, Yw.z, Zw.z;

    double pitch = std::asin(-Rw(2,0));
    double roll  = std::atan2(Rw(2,1), Rw(2,2));
    double yaw   = std::atan2(Rw(1,0), Rw(0,0));
    auto rad2deg = [](double v){ return v * 180.0 / 3.14159265358979323846; };

    double W = rad2deg(roll);
    double P = rad2deg(pitch);
    double R = rad2deg(yaw);

    spdlog::info("[#{}] P_world=({:.3f}, {:.3f}, {:.3f}) m, WPR=({:.1f}, {:.1f}, {:.1f}) deg, size(W,H)=({:.3f}, {:.3f})",
                 idx, p_w_m.x, p_w_m.y, p_w_m.z, W, P, R, width, height);

    // —— 输出结果填充
    out.id        = static_cast<int>(idx);
    out.xyz_m     = p_w_m;
    out.wpr_deg   = cv::Vec3f(static_cast<float>(W),
                              static_cast<float>(P),
                              static_cast<float>(R));
    out.width_m   = width;
    out.height_m  = height;
    out.obb       = rrect;
    out.bottomMidPx = midPx;
    out.p1_w_m    = p1_w_m;
    out.p2_w_m    = p2_w_m;
    out.p3_w_m    = p3_w_m;

    // —— 可视化（与原版一致：OBB + 中点 + 8 行文字）
    if (vis_io) {
        drawRotRect(*vis_io, rrect, cv::Scalar(0, 0, 0), 2);
        cv::circle(*vis_io, midPx, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        drawEightLinesCentered_(*vis_io, rrect, out.id, out.xyz_m, out.wpr_deg,
                                out.width_m, out.height_m, midPx);
    }

    return true;
}

// ===================== 文本绘制 =====================
void BoxPosePipeline::drawEightLinesCentered_(cv::Mat& vis,
                                              const cv::RotatedRect& rrect,
                                              int id,
                                              const cv::Point3f& p_w_m,
                                              const cv::Vec3f& wpr_deg,
                                              float width_m,
                                              float height_m,
                                              const cv::Point2f& midPx) {
    std::array<std::ostringstream, 8> oss;
    oss[0] << "#" << id;
    oss[1] << "x=" << std::fixed << std::setprecision(3) << p_w_m.x;
    oss[2] << "y=" << std::fixed << std::setprecision(3) << p_w_m.y;
    oss[3] << "z=" << std::fixed << std::setprecision(3) << p_w_m.z;
    oss[4] << "W=" << std::fixed << std::setprecision(1) << wpr_deg[0];
    oss[5] << "P=" << std::fixed << std::setprecision(1) << wpr_deg[1];
    oss[6] << "R=" << std::fixed << std::setprecision(1) << wpr_deg[2];
    oss[7] << std::fixed << std::setprecision(1) << width_m*1000 << "," << height_m*1000;

    const double     fontScale = 0.45;
    const int        thickness = 1;
    const cv::Scalar txtColor(0, 0, 0);
    const int        lineGap   = 4;

    std::array<cv::Size, 8> sizes;
    int base = 0;
    int totalH = 0;
    for (int i = 0; i < 8; ++i) {
        sizes[i] = cv::getTextSize(oss[i].str(),
                                   cv::FONT_HERSHEY_SIMPLEX,
                                   fontScale, thickness, &base);
        totalH  += sizes[i].height;
    }
    totalH += lineGap * 7;

    cv::Point center = rrect.center;
    int curY = (int)std::round(center.y - totalH * 0.5);

    for (int i = 0; i < 8; ++i) {
        const std::string txt = oss[i].str();
        const cv::Size& tsz   = sizes[i];
        int orgX = (int)std::round(center.x - tsz.width * 0.5);
        int orgY = curY + tsz.height;
        cv::putText(vis, txt, cv::Point(orgX, orgY),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness, cv::LINE_AA);
        curY += tsz.height + lineGap;
    }
}
