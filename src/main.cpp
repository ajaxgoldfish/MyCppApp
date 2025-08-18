#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <random>
#include <iostream>
#include <iomanip>
#include "MaskRCNNRunner.h"
#include "FusionGeometry.h"
#include <open3d/Open3D.h>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <array>
#include <algorithm>

namespace fs = std::filesystem;

// —— 快速判断像素 (u,v) 是否在旋转矩形内
inline bool inRotRectFast(const cv::RotatedRect& rr, int u, int v) {
    const float cx = rr.center.x, cy = rr.center.y;
    const float hw = rr.size.width * 0.5f, hh = rr.size.height * 0.5f;
    const float ang = rr.angle * (float)CV_PI / 180.f;
    const float ca = std::cos(ang), sa = std::sin(ang);
    const float dx = (float)u - cx, dy = (float)v - cy;
    const float x  =  dx*ca + dy*sa;
    const float y  = -dx*sa + dy*ca;
    return std::fabs(x) <= hw && std::fabs(y) <= hh;
}

struct Proj { int u, v, pid; };

// —— 小工具：画旋转矩形与带底色文字
static void drawRotRect(cv::Mat& img, const cv::RotatedRect& rr, const cv::Scalar& color, int thickness=2) {
    cv::Point2f pts[4]; rr.points(pts);
    for (int i=0;i<4;++i)
        cv::line(img, pts[i], pts[(i+1)%4], color, thickness, cv::LINE_AA);
}
static void putTextWithBg(cv::Mat& img, const std::string& text, cv::Point org,
                          double fontScale=0.6, int thickness=1,
                          const cv::Scalar& txtColor={255,255,255},
                          const cv::Scalar& bgColor={0,0,0})
{
    int base=0; cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base);
    cv::Rect r(org.x, org.y - sz.height - 4, sz.width + 8, sz.height + 6);
    r &= cv::Rect(0,0,img.cols,img.rows);
    if (r.width>0 && r.height>0) {
        cv::rectangle(img, r, bgColor, cv::FILLED, cv::LINE_AA);
    }
    cv::putText(img, text, {org.x+4, org.y-4}, cv::FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness, cv::LINE_AA);
}

int main() {
    const std::string model_path = "models/end2end.onnx";
    const std::string image_path = "2486/rgb.jpg";
    const std::string param_path = "2486/params.xml";
    const std::string pcd_path   = "2486/pcAll.pcd";

    FusionGeometry::initIntrinsic(param_path);
    FusionGeometry::initExtrinsic(param_path);

    const float score_thr = 0.8f, mask_thr = 0.6f;

    // 1) 读取内参
    cv::Mat K = FusionGeometry::getIntrinsic();
    cv::Mat Kinv = K.inv();

    // 2) 读取点云
    open3d::geometry::PointCloud pc_cam;
    if (!open3d::io::ReadPointCloud(pcd_path, pc_cam) || pc_cam.points_.empty()) {
        std::cerr << "点云读取失败或为空: " << pcd_path << "\n";
        return 1;
    }

    // 3) 读取图片并推理
    MaskRCNNRunner runner(model_path);
    cv::Mat orig = cv::imread(image_path);
    if (orig.empty()) {
        std::cerr << "无法读取图片: " << image_path << "\n";
        return 1;
    }

    std::vector<Ort::Value> outs = runner.inferRaw(orig);

    // 用原有的可视化作为底图（若为空则退回原图）
    cv::Mat vis = runner.paint(orig, outs, score_thr, mask_thr);
    if (vis.empty()) vis = orig.clone();

    // 如果你不想要遮色的 mask，只要把上一行改为：cv::Mat vis = orig.clone();

    std::vector<cv::Mat1b> masks = runner.inferMasks(orig, outs, score_thr, mask_thr);

    // —— 收集 {OBB, 底边中点像素}
    std::vector<std::pair<cv::RotatedRect, cv::Point2f>> rect_and_mid;
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

    // 6) 投影所有 3D 点到像素
    const int W = orig.cols, H = orig.rows;
    const double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2), cy = K.at<double>(1,2);
    std::vector<Proj> proj; proj.reserve(pc_cam.points_.size());
    for (int i = 0; i < (int)pc_cam.points_.size(); ++i) {
        const auto& p = pc_cam.points_[i];
        if (p.z() <= 0) continue;
        int u = (int)std::round(fx * p.x()/p.z() + cx);
        int v = (int)std::round(fy * p.y()/p.z() + cy);
        if ((unsigned)u >= (unsigned)W || (unsigned)v >= (unsigned)H) continue;
        proj.push_back({u, v, i});
    }

    std::vector<cv::Rect> usedLabelBoxes;  // 存放已放置文字的占用矩形

    // —— 逐实例求解与标注
    for (size_t i = 0; i < rect_and_mid.size(); ++i) {
        const cv::RotatedRect& rrect = rect_and_mid[i].first;
        const cv::Point2f&     midPx = rect_and_mid[i].second;

        // 0) 收集该 OBB 内的点云（相机系，单位 m）
        std::vector<Eigen::Vector3d> rect_points;
        rect_points.reserve(4096);
        for (const auto& pr : proj) {
            if (inRotRectFast(rrect, pr.u, pr.v)) {
                rect_points.push_back(pc_cam.points_[pr.pid]);
            }
        }
        if (rect_points.size() < 30) {
            std::cout << "[#" << i << "] 框内点过少，跳过\n";
            continue;
        }

        // 1) OBB 底边两个像素点
        cv::Point2f p0, p1;
        if (!FusionGeometry::bottomEdgePoints(rrect, p0, p1)) {
            std::cout << "[#" << i << "] 无法获得底边两点\n";
            continue;
        }

        // 2) 像素(u,v) -> 相机系单位视线
        auto pix2dir = [&](const cv::Point2f& px)->Eigen::Vector3d {
            cv::Vec3d hv(
                Kinv.at<double>(0,0)*px.x + Kinv.at<double>(0,1)*px.y + Kinv.at<double>(0,2),
                Kinv.at<double>(1,0)*px.x + Kinv.at<double>(1,1)*px.y + Kinv.at<double>(1,2),
                Kinv.at<double>(2,0)*px.x + Kinv.at<double>(2,1)*px.y + Kinv.at<double>(2,2)
            );
            Eigen::Vector3d v(hv[0], hv[1], hv[2]);
            return v.normalized();
        };
        Eigen::Vector3d ray1_cam = pix2dir(p0);
        Eigen::Vector3d ray2_cam = pix2dir(p1);

        // 3) 拟合平面 + 两射线与平面求交（相机系：中点/法向/线方向）
        cv::Point3f xyz_cam;
        cv::Vec3f   n_cam, line_cam;
        if (!FusionGeometry::computeBottomLineMidInfo(
                rect_points, ray1_cam, ray2_cam, xyz_cam, n_cam, line_cam)) {
            std::cout << "[#" << i << "] 平面/交点求解失败\n";
            continue;
        }

        // 4) 轴重排（相机 -> 厂商相机轴）：(x,y,z)->(z,-x,-y)
        auto reorder_vec3f = [](const cv::Vec3f& v)->cv::Vec3f { return { v[2], -v[0], -v[1] }; };
        cv::Vec3f n_cam_re    = reorder_vec3f(n_cam);
        cv::Vec3f line_cam_re = reorder_vec3f(line_cam);

        // 点（相机系 m）先重排再转 mm（与外参匹配）
        cv::Vec4f p_cam_re_mm(
            xyz_cam.z * 1000.0f,
           -xyz_cam.x * 1000.0f,
           -xyz_cam.y * 1000.0f,
            1.0f
        );

        // 5) 外参：World <- 厂商相机（mm）
        cv::Mat T_wc = FusionGeometry::getExtrinsic().clone(); // 4x4
        CV_Assert(T_wc.rows==4 && T_wc.cols==4);
        if (T_wc.type()!=CV_32F) T_wc.convertTo(T_wc, CV_32F);
        cv::Mat R_wc_33 = T_wc(cv::Rect(0,0,3,3)).clone();     // 3x3
        cv::Mat t_wc_31 = T_wc(cv::Rect(3,0,1,3)).clone();     // 3x1

        // —— 法向/线方向：只旋转
        cv::Mat n_w_cv    = R_wc_33 * cv::Mat(n_cam_re);
        cv::Mat line_w_cv = R_wc_33 * cv::Mat(line_cam_re);
        cv::Point3f n_w(n_w_cv.at<float>(0), n_w_cv.at<float>(1), n_w_cv.at<float>(2));
        cv::Point3f y_w(line_w_cv.at<float>(0), line_w_cv.at<float>(1), line_w_cv.at<float>(2));

        // —— 中点：p_world = T_wc * p_cam_re_mm（mm）→ m
        cv::Mat p_w_h = T_wc * cv::Mat(p_cam_re_mm);
        cv::Point3f p_w_m(
            p_w_h.at<float>(0)/1000.0f,
            p_w_h.at<float>(1)/1000.0f,
            p_w_h.at<float>(2)/1000.0f
        );

        // 6) 在世界系构姿态：X=法向；Y=线方向的平面投影；Z=X×Y；再正交
        auto norm = [](cv::Point3f v)->cv::Point3f{
            float L = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
            if (L < 1e-9f) return cv::Point3f(0,0,0);
            return { v.x/L, v.y/L, v.z/L };
        };
        auto dot = [](const cv::Point3f& a, const cv::Point3f& b)->float{
            return a.x*b.x + a.y*b.y + a.z*b.z;
        };
        auto cross = [](const cv::Point3f& a, const cv::Point3f& b)->cv::Point3f{
            return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
        };

        // 可选：统一法向使其背向相机
        cv::Point3f C_w_m(
            t_wc_31.at<float>(0)/1000.0f,
            t_wc_31.at<float>(1)/1000.0f,
            t_wc_31.at<float>(2)/1000.0f
        );
        cv::Point3f CP(C_w_m.x - p_w_m.x, C_w_m.y - p_w_m.y, C_w_m.z - p_w_m.z);
        if (dot(n_w, CP) > 0) n_w = { -n_w.x, -n_w.y, -n_w.z };

        // 让线方向的 y 分量大于 0
        if (y_w.y < 0) y_w = { -y_w.x, -y_w.y, -y_w.z };

        cv::Point3f Xw = norm(n_w);
        cv::Point3f Yw = norm(cv::Point3f(
            y_w.x - dot(y_w, Xw)*Xw.x,
            y_w.y - dot(y_w, Xw)*Xw.y,
            y_w.z - dot(y_w, Xw)*Xw.z
        ));
        if (Yw.x==0 && Yw.y==0 && Yw.z==0) { // 退化兜底
            cv::Point3f ref(0,1,0);
            if (std::fabs(dot(ref, Xw)) > 0.95f) ref = {1,0,0};
            Yw = norm(cv::Point3f(
                ref.x - dot(ref, Xw)*Xw.x,
                ref.y - dot(ref, Xw)*Xw.y,
                ref.z - dot(ref, Xw)*Xw.z
            ));
        }
        cv::Point3f Zw = norm(cross(Xw, Yw));
        Yw = norm(cross(Zw, Xw));

        // 7) Rw（列向量 X/Y/Z） & 提取 ZYX->WPR
        Eigen::Matrix3d Rw;
        Rw << Xw.x, Yw.x, Zw.x,
              Xw.y, Yw.y, Zw.y,
              Xw.z, Yw.z, Zw.z;

        double pitch = std::asin(-Rw(2,0));
        double roll  = std::atan2(Rw(2,1), Rw(2,2));
        double yaw   = std::atan2(Rw(1,0), Rw(0,0));
        auto rad2deg = [](double v){ return v * 180.0 / 3.1415926; };

        double W = rad2deg(roll);
        double P = rad2deg(pitch);
        double R = rad2deg(yaw);

        // —— 控制台输出（保持原样）
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[#" << i << "] P_world = (" << p_w_m.x << ", " << p_w_m.y << ", " << p_w_m.z << ") m";
        std::cout << "           WPR = (" << W << ", " << P << ", " << R << ") deg\n";

        // =========================
        // 可视化：在 vis 上画 OBB + 底边中点 + 文本（无底色，小号字体）
        // =========================

        // 画 OBB
        drawRotRect(vis, rrect, cv::Scalar(0, 0, 0), 2);

        // 画底边中点（像素）
        cv::circle(vis, midPx, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

        // 在中点旁边标注 xyz（m）与 WPR（deg）
        std::ostringstream oss1, oss2;
        oss1 << std::fixed << std::setprecision(3)
             << "xyz(m)=(" << p_w_m.x << ", " << p_w_m.y << ", " << p_w_m.z << ")";
        oss2 << std::fixed << std::setprecision(1)
             << "WPR(deg)=(" << W << ", " << P << ", " << R << ")";

        // 字体参数
        const double fontScale = 0.45;
        const int    thickness = 1;
        const cv::Scalar txtColor(0, 0, 0);

        // 计算两行文字尺寸
        int base1 = 0, base2 = 0;
        cv::Size sz1 = cv::getTextSize(oss1.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base1);
        cv::Size sz2 = cv::getTextSize(oss2.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base2);
        const int lineGap = 4;                         // 两行之间的间隔像素
        int boxW = (sz1.width > sz2.width) ? sz1.width : sz2.width;
        const int boxH = sz1.height + lineGap + sz2.height;

        // OBB 的外接轴对齐矩形（用于简单碰撞检测）
        cv::Rect obbBox = rrect.boundingRect() & cv::Rect(0,0,vis.cols,vis.rows);

        // 生成候选放置位置（文字包围框左上角）
        std::array<cv::Point,4> candidates = {
            cv::Point((int)std::round(midPx.x + 8),                (int)std::round(midPx.y - 8 - boxH)), // 右上
            cv::Point((int)std::round(midPx.x + 8),                (int)std::round(midPx.y + 8)),        // 右下
            cv::Point((int)std::round(midPx.x - 8 - boxW),         (int)std::round(midPx.y - 8 - boxH)), // 左上
            cv::Point((int)std::round(midPx.x - 8 - boxW),         (int)std::round(midPx.y + 8))         // 左下
        };

        // 选择一个不与已占用区域/OBB 外接矩形相交的候选，并裁剪到图内
        cv::Point chosenTL = candidates[0];
        bool placed = false;
        for (const auto& tl : candidates) {
            cv::Rect box(tl.x, tl.y, boxW, boxH);
            // 裁剪到图像范围内（避免越界）
            if (box.x < 0) box.x = 0;
            if (box.y < 0) box.y = 0;
            if (box.x + box.width  > vis.cols) box.x = vis.cols - box.width;
            if (box.y + box.height > vis.rows) box.y = vis.rows - box.height;

            bool overlap = false;
            // 与已放置文字框碰撞检测
            for (const auto& used : usedLabelBoxes) {
                if ( (box & used).area() > 0 ) { overlap = true; break; }
            }
            // 与当前 OBB 外接矩形碰撞检测
            if (!overlap && (box & obbBox).area() > 0) overlap = true;

            if (!overlap) { chosenTL = box.tl(); placed = true; break; }
        }
        if (!placed) {
            // 全部相交也没关系，就用右上角（已裁剪）强制放置
            cv::Rect box(candidates[0].x, candidates[0].y, boxW, boxH);
            if (box.x < 0) box.x = 0;
            if (box.y < 0) box.y = 0;
            if (box.x + box.width  > vis.cols) box.x = vis.cols - box.width;
            if (box.y + box.height > vis.rows) box.y = vis.rows - box.height;
            chosenTL = box.tl();
        }

        // 依据包围框左上角，计算两行文字的基线位置
        cv::Point line1Org(chosenTL.x,               chosenTL.y + sz1.height);
        cv::Point line2Org(chosenTL.x,               chosenTL.y + sz1.height + lineGap + sz2.height);

        // 实际绘制（无底色）
        cv::putText(vis, oss1.str(), line1Org + cv::Point(2,0),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness, cv::LINE_AA);
        cv::putText(vis, oss2.str(), line2Org + cv::Point(2,0),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness, cv::LINE_AA);

        // 把本次文字包围框加入占用列表（稍加边距，减少紧贴）
        usedLabelBoxes.emplace_back(chosenTL.x, chosenTL.y, boxW, boxH);

    }

    // 仅保存覆盖到原图的可视化
    cv::imwrite("vis_on_orig.jpg", vis);
    return 0;
}
