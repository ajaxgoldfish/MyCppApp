#pragma once
#include <opencv2/opencv.hpp>
 #include <open3d/Open3D.h>
#include <string>

// PoseXYZABC.h
#include <opencv2/core.hpp>
#include <Eigen/Core>

struct PoseXYZABC {
    cv::Point3f xyz;   // 射线与平面交点 P_cam（相机系, 单位 m）
    cv::Vec3f   n_cam; // 平面法向 n_cam（相机系, 单位向量）
};


class FusionGeometry final {
public:
    // 初始化，读取相机内参
    static bool initIntrinsic(const std::string& calibPath);
    static bool initExtrinsic(const std::string& calibPath);


    // 从二值/概率 mask 求 OBB（最小外接旋转矩形）
    static bool maskToObb(const cv::Mat& mask, cv::RotatedRect& outRect);

    // 取“底边”的中点（图像坐标系 y 向下）：返回圆心 + 建议半径
    static bool bottomMidpointCircle(const cv::RotatedRect& rect,cv::Point2f& center,int& radius);

    static bool computePoseAtBottomMid(const std::vector<Eigen::Vector3d>& rect_points,const Eigen::Vector3d& ray_dir_cam,PoseXYZABC& out);

    static bool bottomEdgePoints(const cv::RotatedRect& rect,
                                      cv::Point2f& p0,
                                      cv::Point2f& p1);
    static bool computeBottomLineMidInfo(
        const std::vector<Eigen::Vector3d>& rect_points,
        const Eigen::Vector3d&              ray_dir1_cam,
        const Eigen::Vector3d&              ray_dir2_cam,
        cv::Point3f&                        xyz_cam,
        cv::Vec3f&                          n_cam,
        cv::Vec3f&                          line_dir_cam
    );

    static bool computeBottomLineMidInfo3(
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
    );

    //=================== 底边两个端点 + 另一个顶点 ===================//
    static bool bottomEdgeWithThirdPoint(const cv::RotatedRect& rect,
                                                  cv::Point2f& p0,  // 底边左端
                                                  cv::Point2f& p1,  // 底边右端
                                                  cv::Point2f& p3);  // 非底边两点中的一个（取更靠近 p0 的那个）

    static bool calcWidthHeightFrom3Points(
    const cv::Point3f& p1_w_m,   // 公共顶点
    const cv::Point3f& p2_w_m,   // 与 p1 构成“宽”的另一点
    const cv::Point3f& p3_w_m,   // 与 p1 构成“高”的另一点
    float& width,                // 输出：宽（与输入单位相同）
    float& height                // 输出：高（与输入单位相同）
    );



    // 获取内参
    static const cv::Mat& getIntrinsic();

    static const cv::Mat& getExtrinsic();

private:
    static cv::Mat intrinsic_; // 3x3 相机内参矩阵
    static cv::Mat T_world_cam_; // 3x3 相机内参矩阵

};
