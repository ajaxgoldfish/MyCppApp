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

//=================== 底边两个端点 + 另一个顶点 ===================//
bool FusionGeometry::bottomEdgeWithThirdPoint(const cv::RotatedRect& rect,
                                              cv::Point2f& p0,  // 底边左端
                                              cv::Point2f& p1,  // 底边右端
                                              cv::Point2f& p3)  // 非底边两点中的一个（取更靠近 p0 的那个）
{
    if (rect.size.width <= 0 || rect.size.height <= 0) return false;

    cv::Point2f pts[4];
    rect.points(pts);  // OpenCV 返回顺时针 4 顶点

    // 1) 选“平均 y 最大”的那条边为底边（图像坐标 y 向下）
    int ei = 0, ej = 1;        // 底边的两个顶点下标
    float bestAvgY = -1e30f;

    auto consider_edge = [&](int a, int b) {
        float avgY = 0.5f * (pts[a].y + pts[b].y);
        if (avgY > bestAvgY) { bestAvgY = avgY; ei = a; ej = b; }
    };
    consider_edge(0,1);
    consider_edge(1,2);
    consider_edge(2,3);
    consider_edge(3,0);

    // 2) 取出底边两个点
    cv::Point2f b0 = pts[ei];
    cv::Point2f b1 = pts[ej];

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

    const cv::Point2f& q0 = pts[rIdx[0]];
    const cv::Point2f& q1 = pts[rIdx[1]];

    float d0 = cv::norm(q0 - p0);
    float d1 = cv::norm(q1 - p0);
    p3 = (d0 <= d1) ? q0 : q1;

    return true;
}

//=================== 底边直线中点（两条射线求中点 + 第三条射线交点） ===================//
bool FusionGeometry::computeBottomLineMidInfo3(
        const std::vector<Eigen::Vector3d>& rect_points,
        const Eigen::Vector3d&              ray_dir1_cam,
        const Eigen::Vector3d&              ray_dir2_cam,
        const Eigen::Vector3d&              ray_dir3_cam,   // ← 新增：第三条射线
        cv::Point3f&                        xyz_cam,        // 中点（仍仅用 1、2 两条射线计算）
        cv::Vec3f&                          n_cam,
        cv::Vec3f&                          line_dir_cam,
        cv::Point3f&                        xyz1_cam,       // ← 新增输出：射线1与平面交点
        cv::Point3f&                        xyz2_cam,       // ← 新增输出：射线2与平面交点
        cv::Point3f&                        xyz3_cam        // ← 新增输出：射线3与平面交点
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

    // 3) 三条射线与平面求交（原逻辑对 1、2 射线保持不变，仅新增第 3 条）
    if (ray_dir1_cam.norm() == 0.0 || ray_dir2_cam.norm() == 0.0 || ray_dir3_cam.norm() == 0.0) return false;
    const Eigen::Vector3d dir1 = ray_dir1_cam.normalized();
    const Eigen::Vector3d dir2 = ray_dir2_cam.normalized();
    const Eigen::Vector3d dir3 = ray_dir3_cam.normalized();   // 新增

    const double nd1 = n.dot(dir1);
    const double nd2 = n.dot(dir2);
    const double nd3 = n.dot(dir3);                           // 新增
    if (std::abs(nd1) < 1e-8 || std::abs(nd2) < 1e-8 || std::abs(nd3) < 1e-8) return false; // 任一近乎平行则失败

    const double t1 = -d / nd1;
    const double t2 = -d / nd2;
    const double t3 = -d / nd3;                               // 新增
    if (t1 <= 0 || t2 <= 0 || t3 <= 0) return false;          // 交点在相机后方或不合理

    const Eigen::Vector3d P1 = t1 * dir1;
    const Eigen::Vector3d P2 = t2 * dir2;
    const Eigen::Vector3d P3 = t3 * dir3;                     // 新增

    // 4) 中点与底边方向（保持原逻辑：仅由 P1、P2 计算）
    const Eigen::Vector3d M  = 0.5 * (P1 + P2);
    Eigen::Vector3d edge_dir = P2 - P1;
    const double edge_len = edge_dir.norm();
    if (edge_len < 1e-8) return false; // 两交点过近/数值不稳定
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
    line_dir_cam = cv::Vec3f(static_cast<float>(edge_dir.x()),
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

    return true;
}


bool FusionGeometry::calcWidthHeightFrom3Points(
    const cv::Point3f& p1_w_m,
    const cv::Point3f& p2_w_m,
    const cv::Point3f& p3_w_m,
    float& width,
    float& height)
{
    // 以 p1 为公共顶点
    const cv::Point3f v12 = p2_w_m - p1_w_m; // 宽方向向量
    const cv::Point3f v13 = p3_w_m - p1_w_m; // 高方向向量

    const float w = std::sqrt(v12.x * v12.x + v12.y * v12.y + v12.z * v12.z);
    const float h = std::sqrt(v13.x * v13.x + v13.y * v13.y + v13.z * v13.z);

    // 基本健壮性检查
    if (!std::isfinite(w) || !std::isfinite(h)) return false;

    width  = w;
    height = h;
    return true;
}


