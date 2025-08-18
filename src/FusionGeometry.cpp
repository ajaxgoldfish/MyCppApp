#include "FusionGeometry.h"
#include <limits>
#include <cmath>
#include <algorithm>      // for std::max
#include <spdlog/spdlog.h>

cv::Mat FusionGeometry::intrinsic_;
cv::Mat FusionGeometry::T_world_cam_;


//=================== 基础：mask -> OBB ===================//
bool FusionGeometry::maskToObb(const cv::Mat& mask, cv::RotatedRect& outRect) {
    outRect = cv::RotatedRect();
    if (mask.empty()) return false;

    // 1) 做一个“非零即前景”的二值图（不做阈值筛选、不做形态学）
    cv::Mat1b bin;
    if (mask.type() == CV_8U) {
        bin = mask != 0;                      // 只看非零
    } else {
        cv::Mat tmp;
        cv::compare(mask, 0, tmp, cv::CMP_GT); // >0 视为前景；结果是 8U 0/255
        bin = tmp;
    }

    // 2) 收集所有前景像素坐标（把所有连通块一起算）
    std::vector<cv::Point> pts;
    cv::findNonZero(bin, pts);

    // 3) 没有前景像素就返回 false；否则直接对所有点做最小外接矩形
    if (pts.empty()) return false;

    outRect = cv::minAreaRect(pts);           // angle ∈ (-90, 0]
    return true;
}


//=================== 底边中点 ===================//
bool FusionGeometry::bottomMidpointCircle(const cv::RotatedRect& rect,
                                          cv::Point2f& center,
                                          int& radius) {
    if (rect.size.width <= 0 || rect.size.height <= 0) return false;

    cv::Point2f pts[4]; rect.points(pts);
    int ei = 0, ej = 1; float bestAvgY = -1e30f;

    auto consider_edge = [&](int a, int b) {
        float avgY = (pts[a].y + pts[b].y) * 0.5f;
        if (avgY > bestAvgY) { bestAvgY = avgY; ei = a; ej = b; }
    };
    consider_edge(0,1); consider_edge(1,2); consider_edge(2,3); consider_edge(3,0);

    center = (pts[ei] + pts[ej]) * 0.5f;
    float major = (rect.size.width > rect.size.height) ? rect.size.width : rect.size.height;
    radius = cvRound(major * 0.02f);
    if (radius < 2) radius = 2;
    return true;
}
// 点是否在旋转矩形内
static inline bool inRotRect(const cv::RotatedRect& rr, int u, int v) {
    cv::Point2f pts[4]; rr.points(pts);
    std::vector<cv::Point2f> poly(pts, pts + 4);
    return cv::pointPolygonTest(poly, cv::Point2f((float)u, (float)v), false) >= 0;
}

bool FusionGeometry::computePoseAtBottomMid(
        const std::vector<Eigen::Vector3d>& rect_points,
        const Eigen::Vector3d&              ray_dir_cam,
        PoseXYZABC&                         out
    )
{
    if (rect_points.size() < 30) return false;

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
    if (inliers.size() < 20) return false;

    // 2) 规范化平面参数 [n | d]，并做方向规整（让法向朝向相机 +Z，稳定）
    Eigen::Vector3d n(plane[0], plane[1], plane[2]);
    double d = plane[3];

    const double L = n.norm();
    if (L <= 0.0) return false;
    n /= L; d /= L;                 // n 单位化，d 同步缩放

    if (n.z() < 0) { n = -n; d = -d; } // 翻转时 d 也要翻，保证 n·X + d = 0 不变

    // 3) 射线-平面求交：X = t * dir； n·X + d = 0 => t = -d / (n·dir)
    if (ray_dir_cam.norm() == 0.0) return false;
    const Eigen::Vector3d dir = ray_dir_cam.normalized();

    const double ndotdir = n.dot(dir);

    if (std::abs(ndotdir) < 1e-8) return false;   // 射线与平面平行
    const double t = -d / ndotdir;
    if (t <= 0) return false;                     // 交点在相机后方/不合理

    const Eigen::Vector3d P = t * dir;            // 交点（相机系, m）

    // 4) 输出（仅 P_cam 与 n_cam）
    out.xyz  = cv::Point3f(static_cast<float>(P.x()),
                           static_cast<float>(P.y()),
                           static_cast<float>(P.z()));
    out.n_cam = cv::Vec3f(static_cast<float>(n.x()),
                          static_cast<float>(n.y()),
                          static_cast<float>(n.z()));

    spdlog::info("P_cam = ({:.6f}, {:.6f}, {:.6f})",
             out.xyz.x, out.xyz.y, out.xyz.z);

    spdlog::info("n_cam = ({}, {}, {})",
             out.n_cam[0],
             out.n_cam[1],
             out.n_cam[2]);

    return true;
}

bool FusionGeometry::initIntrinsic(const std::string& calibPath) {
    cv::FileStorage fs(calibPath, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["intrinsicRGB"] >> intrinsic_;
    if (intrinsic_.empty() || intrinsic_.rows!=3 || intrinsic_.cols!=3) return false;
    if (intrinsic_.type() != CV_64F) intrinsic_.convertTo(intrinsic_, CV_64F);
    return true;
}


const cv::Mat& FusionGeometry::getIntrinsic() {
    return intrinsic_;
}
const cv::Mat& FusionGeometry::getExtrinsic() {
    return T_world_cam_;
}

bool FusionGeometry::initExtrinsic(const std::string& calibPath) {
    cv::FileStorage fs(calibPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[initExtrinsic] 打不开外参文件: " << calibPath << "\n";
        return false;
    }

    fs["extrinsicRGB"] >> T_world_cam_;
    fs.release();

    if (T_world_cam_.empty()) {
        std::cerr << "[initExtrinsic] extrinsicRGB 节点不存在或为空\n";
        return false;
    }

    if (T_world_cam_.rows != 4 || T_world_cam_.cols != 4) {
        std::cerr << "[initExtrinsic] extrinsicRGB 必须是 4x4 矩阵\n";
        return false;
    }

    if (T_world_cam_.type() != CV_64F) {
        T_world_cam_.convertTo(T_world_cam_, CV_64F);
    }

    return true;
}

//=================== 底边两个端点 ===================//
bool FusionGeometry::bottomEdgePoints(const cv::RotatedRect& rect,
                                      cv::Point2f& p0,
                                      cv::Point2f& p1) {
    if (rect.size.width <= 0 || rect.size.height <= 0) return false;

    cv::Point2f pts[4];
    rect.points(pts);  // OpenCV 返回顺时针 4 顶点

    // 参考上面的选择规则：取“平均 y 最大”的那条边为底边（图像坐标 y 向下）
    int ei = 0, ej = 1;
    float bestAvgY = -1e30f;

    auto consider_edge = [&](int a, int b) {
        float avgY = 0.5f * (pts[a].y + pts[b].y);
        if (avgY > bestAvgY) { bestAvgY = avgY; ei = a; ej = b; }
    };
    consider_edge(0,1);
    consider_edge(1,2);
    consider_edge(2,3);
    consider_edge(3,0);

    p0 = pts[ei];
    p1 = pts[ej];

    // 可选：保证从左到右输出（按 x 升序）
    if (p0.x > p1.x) std::swap(p0, p1);

    return true;
}


//=================== 底边直线中点（由两条射线与拟合平面求交） ===================//
// 输入：
//   rect_points   —— 该目标区域内的 3D 点（相机系, m），用于 RANSAC 拟合平面
//   ray_dir1_cam  —— 底边端点 p1 的相机系视线方向（未必归一化）
//   ray_dir2_cam  —— 底边端点 p2 的相机系视线方向（未必归一化）
// 输出：
//   xyz_cam       —— 两交点的中点 M（相机系, m）
//   n_cam         —— 拟合平面的单位法向（相机系，方向已规整）
//   line_dir_cam  —— 底边方向的单位向量（相机系，沿 p1→p2）
//
// 说明：得到 (n_cam, line_dir_cam, xyz_cam) 后，后续可用
//       n_cam 作为 Z 轴，line_dir_cam 作为 Y(或 X) 轴构建姿态。
bool FusionGeometry::computeBottomLineMidInfo(
        const std::vector<Eigen::Vector3d>& rect_points,
        const Eigen::Vector3d&              ray_dir1_cam,
        const Eigen::Vector3d&              ray_dir2_cam,
        cv::Point3f&                        xyz_cam,
        cv::Vec3f&                          n_cam,
        cv::Vec3f&                          line_dir_cam
    )
{
    if (rect_points.size() < 30) return false;

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
    if (inliers.size() < 20) return false;

    // 2) 规范化平面参数 [n | d]，并做方向规整（让法向朝向相机 +Z）
    Eigen::Vector3d n(plane[0], plane[1], plane[2]);
    double d = plane[3];

    const double L = n.norm();
    if (L <= 0.0) return false;
    n /= L; d /= L;

    if (n.z() < 0) { n = -n; d = -d; } // 翻转时同步翻 d，保持 n·X + d = 0

    // 3) 两条射线与平面求交
    if (ray_dir1_cam.norm() == 0.0 || ray_dir2_cam.norm() == 0.0) return false;
    const Eigen::Vector3d dir1 = ray_dir1_cam.normalized();
    const Eigen::Vector3d dir2 = ray_dir2_cam.normalized();

    const double nd1 = n.dot(dir1);
    const double nd2 = n.dot(dir2);
    if (std::abs(nd1) < 1e-8 || std::abs(nd2) < 1e-8) return false; // 视线与平面近乎平行

    const double t1 = -d / nd1;
    const double t2 = -d / nd2;
    if (t1 <= 0 || t2 <= 0) return false; // 交点在相机后方或不合理

    const Eigen::Vector3d P1 = t1 * dir1;
    const Eigen::Vector3d P2 = t2 * dir2;

    // 4) 中点与底边方向
    const Eigen::Vector3d M  = 0.5 * (P1 + P2);
    Eigen::Vector3d edge_dir = P2 - P1;
    const double edge_len = edge_dir.norm();
    if (edge_len < 1e-8) return false; // 两交点过近/数值不稳定
    edge_dir /= edge_len;

    // （可选）方向一致性：让 (edge_dir × n).z >= 0，避免符号跳变
    // 如果不需要固定方向，可删除下面三行
    Eigen::Vector3d x_tmp = edge_dir.cross(n);
    if (x_tmp.z() < 0) edge_dir = -edge_dir;

    // 5) 输出
    xyz_cam      = cv::Point3f(static_cast<float>(M.x()),
                               static_cast<float>(M.y()),
                               static_cast<float>(M.z()));
    n_cam        = cv::Vec3f(static_cast<float>(n.x()),
                             static_cast<float>(n.y()),
                             static_cast<float>(n.z()));
    line_dir_cam = cv::Vec3f(static_cast<float>(edge_dir.x()),
                             static_cast<float>(edge_dir.y()),
                             static_cast<float>(edge_dir.z()));
    return true;
}
