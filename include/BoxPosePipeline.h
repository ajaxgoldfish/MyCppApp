// BoxPosePipeline.h
#pragma once
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <memory>
#include <vector>
#include <string>

// 只做前置声明
class MaskRCNNRunner;

struct BoxPoseResult {
    int id = -1;
    cv::Point3f xyz_m{};
    cv::Vec3f   wpr_deg{};
    float width_m = 0.f, height_m = 0.f;
    cv::RotatedRect obb;
    cv::Point2f bottomMidPx{};
    cv::Point3f p1_w_m{}, p2_w_m{}, p3_w_m{};
    Eigen::Matrix3d Rw;                 // 旋转矩阵 (世界系下的朝向)
};

// ========================
// Box 姿态结果（旋转矩阵版）
// ========================
struct BoxPoseRotationResult {
    int id = -1;                     // 箱子编号
    cv::Point3f xyz_m{};             // 世界坐标系下中心点 (米)
    cv::Matx33f R;                   // 旋转矩阵 (世界系下的朝向)
    float width_m = 0.f;             // 箱子宽度 (米)
    float height_m = 0.f;            // 箱子高度 (米)
    cv::RotatedRect obb;             // 原始2D OBB
    cv::Point2f bottomMidPx{};       // 图像底边中点
    cv::Point3f p1_w_m{}, p2_w_m{}, p3_w_m{}; // 三个基准点 (米)
};


class BoxPosePipeline {
public:
    struct Options {
        std::string model_path;
        std::string calib_path;
        float score_thr = 0.8f;
        float mask_thr  = 0.6f;
        bool  paint_masks_on_vis = true;
    };

    explicit BoxPosePipeline(const Options& opt);

    // 关键：这里只声明，不内联定义
    ~BoxPosePipeline();

    bool initialize();
    bool run(const cv::Mat& rgb,
             const open3d::geometry::PointCloud& pc_cam,
             std::vector<BoxPoseResult>& results,
             cv::Mat* vis_out = nullptr);

    void setThresholds(float s, float m) { options_.score_thr = s; options_.mask_thr = m; }
    void setPaintMasks(bool on) { options_.paint_masks_on_vis = on; }

    const cv::Mat& getK()    const { return K_; }
    const cv::Mat& getKinv() const { return Kinv_; }
    const cv::Mat& getTcw()  const { return T_wc_; }
    const Options& getOptions() const { return options_; }

private:
    struct Proj { int u, v, pid; };

    static inline bool inRotRectFast(const cv::RotatedRect& rr, int u, int v);
    static inline void drawRotRect(cv::Mat& img, const cv::RotatedRect& rr,
                                   const cv::Scalar& color, int thickness=2);
    inline Eigen::Vector3d pix2dir(const cv::Point2f& px) const;
    static inline cv::Vec3f reorder_vec3f(const cv::Vec3f& v) { return { v[2], -v[0], -v[1] }; }

    bool loadCalibrationAndModel_();
    bool inferMasks_(const cv::Mat& rgb, cv::Mat& vis, std::vector<cv::Mat1b>& masks);
    void collectRectsAndBottomMids_(const std::vector<cv::Mat1b>& masks,
                                    std::vector<std::pair<cv::RotatedRect, cv::Point2f>>& rect_and_mid) const;
    void projectPointCloud_(const open3d::geometry::PointCloud& pc_cam, int W, int H, std::vector<Proj>& proj) const;
    bool solveOneBox_(size_t idx,
                      const std::pair<cv::RotatedRect, cv::Point2f>& rect_mid,
                      const std::vector<Proj>& proj,
                      const open3d::geometry::PointCloud& pc_cam,
                      BoxPoseResult& out) const;

    bool solveOneBoxR_(size_t idx,
                       const std::pair<cv::RotatedRect, cv::Point2f>& rect_mid,
                       const std::vector<Proj>& proj,
                       const open3d::geometry::PointCloud& pc_cam,
                       BoxPoseRotationResult& out) const;

    static void drawEightLinesCentered_(cv::Mat& vis, const cv::RotatedRect& rrect, int id,
                                        const cv::Point3f& p_w_m, const cv::Vec3f& wpr_deg,
                                        float width_m, float height_m);

private:
    Options options_;
    std::unique_ptr<MaskRCNNRunner> runner_; // 前置声明 + unique_ptr 没问题（析构放到 .cpp）
    cv::Mat K_, Kinv_, T_wc_;
    bool ready_ = false;
};

// ======= 内联小函数（与之前相同）=======
inline bool BoxPosePipeline::inRotRectFast(const cv::RotatedRect& rr, int u, int v) {
    const float cx = rr.center.x, cy = rr.center.y;
    const float hw = rr.size.width * 0.5f, hh = rr.size.height * 0.5f;
    const float ang = rr.angle * (float)CV_PI / 180.f;
    const float ca = std::cos(ang), sa = std::sin(ang);
    const float dx = (float)u - cx, dy = (float)v - cy;
    const float x  =  dx*ca + dy*sa;
    const float y  = -dx*sa + dy*ca;
    return std::fabs(x) <= hw && std::fabs(y) <= hh;
}
inline void BoxPosePipeline::drawRotRect(cv::Mat& img, const cv::RotatedRect& rr,
                                         const cv::Scalar& color, int thickness) {
    cv::Point2f pts[4]; rr.points(pts);
    for (int i=0;i<4;++i) cv::line(img, pts[i], pts[(i+1)%4], color, thickness, cv::LINE_AA);
}
inline Eigen::Vector3d BoxPosePipeline::pix2dir(const cv::Point2f& px) const {
    cv::Vec3d hv(
        Kinv_.at<double>(0,0)*px.x + Kinv_.at<double>(0,1)*px.y + Kinv_.at<double>(0,2),
        Kinv_.at<double>(1,0)*px.x + Kinv_.at<double>(1,1)*px.y + Kinv_.at<double>(1,2),
        Kinv_.at<double>(2,0)*px.x + Kinv_.at<double>(2,1)*px.y + Kinv_.at<double>(2,2)
    );
    Eigen::Vector3d v(hv[0], hv[1], hv[2]);
    return v.normalized();
}
