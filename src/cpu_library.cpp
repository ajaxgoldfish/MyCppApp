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
    // ==========================================
    // 全局配置与资源 (Global Configuration)
    // ==========================================
    std::string g_model_path;
    std::string g_calib_path;
    float g_score_threshold = 0.8f;   // 置信度阈值
    float g_mask_threshold = 0.6f;    // Mask二值化阈值
    bool g_paint_masks_on_vis = true; // 是否在可视化图中绘制Mask

    // 相机内参和外参
    cv::Mat g_intrinsic;
    cv::Mat g_transform_world_cam; // T_wc: 相机到世界的变换矩阵

    // 本地计算结果结构体
    struct LocalBoxPoseResult {
        int id = -1;
        cv::Point3f xyz_m{};          // 中心点坐标 (米)
        cv::Vec3f wpr_deg{};          // 欧拉角 (度)
        float width_m = 0.f;          // 宽度 (米)
        float height_m = 0.f;         // 高度 (米)
        cv::RotatedRect obb;          // 2D 旋转矩形
        cv::Point2f bottom_mid_px{};  // 底边中点 (像素)
        cv::Point3f p1_w_m{}, p2_w_m{}, p3_w_m{}; // 关键点 (世界坐标系)
        Eigen::Matrix3d rotation_matrix_world;    // 旋转矩阵 (世界坐标系)
    };

    // ==========================================
    // ONNX Runtime 全局状态
    // ==========================================
    std::unique_ptr<Ort::Env> g_env;
    std::unique_ptr<Ort::Session> g_session;
    std::string g_input_name;
    std::vector<std::string> g_output_names_str;
    std::vector<const char *> g_output_names_ptr;
    
    // 缓存矩阵 (K: 内参, Twc: 外参)
    cv::Mat g_mat_k, g_mat_k_inv, g_mat_twc; 
    bool g_is_pipeline_ready = false;
    std::string g_root_output_dir = "res"; // 可视化输出根目录

    // 点云投影映射结构
    struct ProjectionMap {
        int u, v;  // 像素坐标
        int point_idx; // 对应的点云索引
    };

} // namespace


/**
 * @brief 初始化算法流水线
 * @param is_debug 是否开启调试日志
 */
int bs_yzx_init(const bool is_debug) {
    // 1. 配置日志
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(is_debug ? spdlog::level::debug : spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);

    // 2. 设置默认参数
    g_model_path = "models/end2end.onnx";
    g_calib_path = "config/params.xml";
    g_score_threshold = 0.7f;
    g_mask_threshold = 0.5f;
    g_paint_masks_on_vis = true;

    // 3. 初始化资源
    if (!g_is_pipeline_ready) {
        try {
            // 读取内参
            cv::FileStorage fs_intrinsic(g_calib_path, cv::FileStorage::READ);
            if (!fs_intrinsic.isOpened()) return -25;
            fs_intrinsic["intrinsicRGB"] >> g_intrinsic;
            if (g_intrinsic.empty() || g_intrinsic.rows != 3 || g_intrinsic.cols != 3) return -26;
            if (g_intrinsic.type() != CV_64F) g_intrinsic.convertTo(g_intrinsic, CV_64F);
            
            // 读取外参
            cv::FileStorage fs_extrinsic(g_calib_path, cv::FileStorage::READ);
            if (!fs_extrinsic.isOpened()) {
                spdlog::error("[initExtrinsic] Cannot open extrinsic file: {}", g_calib_path);
                return -27;
            }
            
            fs_extrinsic["extrinsicRGB"] >> g_transform_world_cam;
            fs_extrinsic.release();
            
            if (g_transform_world_cam.empty()) {
                spdlog::error("[initExtrinsic] extrinsicRGB node not found or empty");
                return -28;
            }
            
            // 确保外参是 4x4 矩阵
            if (g_transform_world_cam.rows != 4 || g_transform_world_cam.cols != 4) {
                spdlog::error("[initExtrinsic] extrinsicRGB must be 4x4 matrix");
                return -29;
            }
            
            if (g_transform_world_cam.type() != CV_64F) {
                g_transform_world_cam.convertTo(g_transform_world_cam, CV_64F);
            }
            
            // 缓存矩阵
            g_mat_k = g_intrinsic.clone(); // CV_64F
            g_mat_k_inv = g_mat_k.inv();
            g_mat_twc = g_transform_world_cam.clone(); // 4x4
            if (g_mat_twc.type() != CV_32F) g_mat_twc.convertTo(g_mat_twc, CV_32F);

            // 初始化 ONNX Runtime
            if (!g_env) {
                g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "mrcnn");
            }
            
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 辅助函数：std::string 转 std::wstring (Windows下路径需要)
            auto to_wstring = [](const std::string& s) -> std::wstring {
                return { s.begin(), s.end() };
            };
            
            g_session = std::make_unique<Ort::Session>(*g_env, to_wstring(g_model_path).c_str(), session_options);
            
            Ort::AllocatorWithDefaultOptions allocator;
            g_input_name = g_session->GetInputNameAllocated(0, allocator).get();
            
            size_t output_count = g_session->GetOutputCount();
            g_output_names_str.reserve(output_count);
            for (size_t i = 0; i < output_count; ++i) {
                g_output_names_str.emplace_back(g_session->GetOutputNameAllocated(i, allocator).get());
            }
            for (auto &s: g_output_names_str) g_output_names_ptr.push_back(s.c_str());
            
            g_is_pipeline_ready = true;
        } catch (const std::exception &e) {
            spdlog::critical("Pipeline initialization failed: {}", e.what());
            g_is_pipeline_ready = false;
            return -1;
        }
    }

    return 0;
}

/**
 * @brief 执行目标检测与位姿估计
 * @param task_id 任务ID，用于查找输入数据目录
 * @param box_array 输出结果数组
 * @return 检测到的目标数量
 */
int bs_yzx_object_detection_lanxin(int task_id, zzb::Box box_array[]) {
    if (!g_is_pipeline_ready) return -10; // Pipeline 未初始化

    auto time_start = std::chrono::steady_clock::now();

    // ==========================================
    // 1. 数据加载 (Data Loading)
    // ==========================================
    const fs::path case_dir = fs::path(g_root_output_dir) / std::to_string(task_id);
    if (!fs::exists(case_dir)) {
        spdlog::error("Data directory does not exist: {}", case_dir.string());
        return -21;
    }

    cv::Mat image_rgb;
    open3d::geometry::PointCloud point_cloud;
    
    // 1.1 读取 RGB 图像
    const fs::path rgb_path = case_dir / "rgb.jpg";
    if (!fs::exists(rgb_path)) {
        spdlog::error("RGB file does not exist: {}", rgb_path.string());
        return -22;
    }
    
    image_rgb = cv::imread(rgb_path.string(), cv::IMREAD_COLOR);
    if (image_rgb.empty()) {
        spdlog::error("Cannot read RGB image: {}", rgb_path.string());
        return -22;
    }
    spdlog::info("Read RGB from file: {}", rgb_path.string());
    
    // 1.2 读取点云数据
    const fs::path pcd_path = case_dir / "pcAll.pcd";
    if (!fs::exists(pcd_path)) {
        spdlog::error("Point cloud file does not exist: {}", pcd_path.string());
        return -23;
    }
    
    if (!open3d::io::ReadPointCloud(pcd_path.string(), point_cloud) || point_cloud.points_.empty()) {
        spdlog::error("Cannot read point cloud data: {}", pcd_path.string());
        return -23;
    }
    spdlog::info("Read point cloud from file: {}", pcd_path.string());

    // 检查输入有效性
    if (image_rgb.empty() || point_cloud.points_.empty()) {
        spdlog::warn("Input is empty: rgb.empty()={} pc.size()={}", image_rgb.empty(), point_cloud.points_.size());
        return -24;
    }

    // ==========================================
    // 2. 图像预处理与推理 (Inference)
    // ==========================================
    std::vector<cv::Mat1b> detected_masks;
    std::vector<Ort::Value> ort_outputs;
    
    cv::Mat rgb_float;
    cv::cvtColor(image_rgb, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32F);
    
    // 归一化参数 (ImageNet mean/std)
    static const cv::Scalar kMean(123.675, 116.28, 103.53);
    static const cv::Scalar kStdv(58.395, 57.12, 57.375);
    cv::subtract(rgb_float, kMean, rgb_float);
    cv::divide(rgb_float, kStdv, rgb_float);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_float, blob, 1.0, cv::Size(), {}, false, false, CV_32F);
    
    std::vector<int64_t> input_shape = {1, 3, blob.size[2], blob.size[3]};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float *>(blob.data), static_cast<size_t>(blob.total()), input_shape.data(), input_shape.size());
    
    const char *input_names[] = {g_input_name.c_str()};
    ort_outputs = g_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                          g_output_names_ptr.data(), g_output_names_ptr.size());
    
    if (ort_outputs.size() < 3) {
        spdlog::error("Insufficient outputs, expected 3");
        return -27;
    }

    // ==========================================
    // 3. 结果解析与可视化 (Parsing & Visualization)
    // ==========================================
    cv::Mat vis_image = image_rgb.clone();
    const cv::Scalar kVisColor(0, 255, 0);

    // 解析输出张量形状
    auto shape_dets = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto shape_masks = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_detections = shape_dets[1];
    int mask_h = (int) shape_masks[2];
    int mask_w = (int) shape_masks[3];
    
    const float *det_data = ort_outputs[0].GetTensorData<float>();
    const float *mask_tensor_data = ort_outputs[2].GetTensorData<float>();

    // 第一次遍历：仅用于绘制2D可视化 (Bounding Box & Mask)
    for (int64_t i = 0; i < num_detections; ++i) {
        const float *curr_det = det_data + i * 5;
        float score = curr_det[4];
        if (score < g_score_threshold) continue;

        int x1 = std::clamp((int)std::lround(curr_det[0]), 0, vis_image.cols - 1);
        int y1 = std::clamp((int)std::lround(curr_det[1]), 0, vis_image.rows - 1);
        int x2 = std::clamp((int)std::lround(curr_det[2]), 0, vis_image.cols - 1);
        int y2 = std::clamp((int)std::lround(curr_det[3]), 0, vis_image.rows - 1);

        // 绘制矩形框
        cv::rectangle(vis_image, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), kVisColor, 2);
        
        // 绘制分数
        char score_buf[32];
        std::snprintf(score_buf, sizeof(score_buf), "%.2f", score);
        cv::putText(vis_image, score_buf, {x1, std::max(0, y1 - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, kVisColor, 1);
        
        // 处理Mask可视化
        int roi_w = std::max(1, x2 - x1);
        int roi_h = std::max(1, y2 - y1);
        cv::Rect roi_rect(x1, y1, roi_w, roi_h);
        roi_rect &= cv::Rect(0, 0, vis_image.cols, vis_image.rows);
        
        if (roi_rect.area() > 0) {
            const float *curr_mask_ptr = mask_tensor_data + i * (mask_h * mask_w);
            cv::Mat mask_float(mask_h, mask_w, CV_32F, const_cast<float *>(curr_mask_ptr));
            cv::Mat mask_resized;
            cv::resize(mask_float, mask_resized, roi_rect.size(), 0, 0, cv::INTER_LINEAR);
            
            cv::Mat1b mask_bin;
            cv::compare(mask_resized, g_mask_threshold, mask_bin, cv::CMP_GT);
            
            cv::Mat roi_view = vis_image(roi_rect);
            cv::Mat overlay = roi_view.clone();
            overlay.setTo(kVisColor, mask_bin);
            cv::addWeighted(roi_view, 1.0, overlay, 0.5, 0, roi_view);
        }
    }

    if (!g_paint_masks_on_vis) vis_image = image_rgb.clone();

    // ==========================================
    // 4. 提取有效 Mask (Extract Masks)
    // ==========================================
    detected_masks.clear();
    detected_masks.reserve((size_t)num_detections);
    
    for (int64_t i = 0; i < num_detections; ++i) {
        const float *curr_det = det_data + i * 5;
        float score = curr_det[4];
        if (score < g_score_threshold) continue;
        
        int x1 = std::clamp((int)std::lround(curr_det[0]), 0, image_rgb.cols - 1);
        int y1 = std::clamp((int)std::lround(curr_det[1]), 0, image_rgb.rows - 1);
        int x2 = std::clamp((int)std::lround(curr_det[2]), 0, image_rgb.cols - 1);
        int y2 = std::clamp((int)std::lround(curr_det[3]), 0, image_rgb.rows - 1);
        
        cv::Rect roi_rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
        roi_rect &= cv::Rect(0, 0, image_rgb.cols, image_rgb.rows);
        if (roi_rect.area() <= 0) continue;
        
        const float *curr_mask_ptr = mask_tensor_data + i * (mask_h * mask_w);
        cv::Mat mask_float(mask_h, mask_w, CV_32F, const_cast<float *>(curr_mask_ptr));
        cv::Mat mask_resized;
        cv::resize(mask_float, mask_resized, roi_rect.size(), 0, 0, cv::INTER_LINEAR);
        
        cv::Mat1b mask_bin;
        cv::compare(mask_resized, g_mask_threshold, mask_bin, cv::CMP_GT);
        
        // 恢复到全图尺寸的Mask
        cv::Mat1b full_mask(image_rgb.rows, image_rgb.cols, (uchar)0);
        full_mask(roi_rect).setTo(255, mask_bin);
        detected_masks.emplace_back(std::move(full_mask));
    }

    // ==========================================
    // 5. 最小外接矩形拟合 (MinAreaRect)
    // ==========================================
    std::vector<std::pair<cv::RotatedRect, cv::Point2f>> candidates_rects;
    candidates_rects.reserve(detected_masks.size());
    
    for (const auto &mask : detected_masks) {
        if (mask.empty()) continue;
        
        // 查找所有非零点
        std::vector<cv::Point> non_zero_pts;
        cv::findNonZero(mask, non_zero_pts);
        
        if (!non_zero_pts.empty()) {
            // 计算最小外接矩形
            cv::RotatedRect obb = cv::minAreaRect(non_zero_pts);
            
            if (obb.size.width > 0 && obb.size.height > 0) {
                // 寻找底边中点 (假设在图像中y最大的边为底边)
                cv::Point2f pts_obb[4];
                obb.points(pts_obb);
                
                int idx1 = 0, idx2 = 1; 
                float max_avg_y = -1e30f;

                auto update_best_edge = [&](int a, int b) {
                    float avg_y = (pts_obb[a].y + pts_obb[b].y) * 0.5f;
                    if (avg_y > max_avg_y) { max_avg_y = avg_y; idx1 = a; idx2 = b; }
                };
                
                update_best_edge(0, 1); update_best_edge(1, 2); 
                update_best_edge(2, 3); update_best_edge(3, 0);

                cv::Point2f bottom_mid = (pts_obb[idx1] + pts_obb[idx2]) * 0.5f;
                candidates_rects.emplace_back(obb, bottom_mid);
            }
        }
    }

    // ==========================================
    // 6. 点云投影与 ROI 过滤 (Projection & Filtering)
    // ==========================================
    std::vector<ProjectionMap> projected_points;
    const double fx = g_mat_k.at<double>(0, 0), fy = g_mat_k.at<double>(1, 1);
    const double cx = g_mat_k.at<double>(0, 2), cy = g_mat_k.at<double>(1, 2);
    
    projected_points.reserve(point_cloud.points_.size());
    for (int i = 0; i < (int)point_cloud.points_.size(); ++i) {
        const auto &p = point_cloud.points_[i];
        if (p.z() <= 0) continue; // 过滤 z<=0 的无效点
        
        int u = (int)std::round(fx * p.x() / p.z() + cx);
        int v = (int)std::round(fy * p.y() / p.z() + cy);
        
        if ((unsigned)u < (unsigned)image_rgb.cols && (unsigned)v < (unsigned)image_rgb.rows) {
            projected_points.push_back({u, v, i});
        }
    }

    // ==========================================
    // 7. 3D 位姿估算 (Pose Estimation)
    // ==========================================
    std::vector<LocalBoxPoseResult> results;
    results.reserve(candidates_rects.size());

    for (size_t i = 0; i < candidates_rects.size(); ++i) {
        LocalBoxPoseResult res;
        bool is_solved = false;
        
        const cv::RotatedRect &curr_rect = candidates_rects[i].first;
        const cv::Point2f &bottom_mid_px = candidates_rects[i].second;

        // 7.1 收集矩形内的 3D 点
        std::vector<Eigen::Vector3d> points_in_box;
        points_in_box.reserve(4096);
        
        // 几何预计算，判断点是否在旋转矩形内
        const float center_x = curr_rect.center.x, center_y = curr_rect.center.y;
        const float half_w = curr_rect.size.width * 0.5f;
        const float half_h = curr_rect.size.height * 0.5f;
        const float angle_rad = curr_rect.angle * (float)CV_PI / 180.f;
        const float cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

        for (const auto &proj : projected_points) {
            const float dx = (float)proj.u - center_x;
            const float dy = (float)proj.v - center_y;
            // 旋转到矩形局部坐标系
            const float local_x = dx * cos_a + dy * sin_a;
            const float local_y = -dx * sin_a + dy * cos_a;
            
            if (std::fabs(local_x) <= half_w && std::fabs(local_y) <= half_h) {
                points_in_box.push_back(point_cloud.points_[proj.point_idx]);
            }
        }

        if (points_in_box.size() < 30) {
            spdlog::info("[#{}] Too few points in box, skipping ({} points)", i, points_in_box.size());
        } else {
            // 7.2 确定 3D 关键点 (底边两点 + 第三点)
            cv::Point2f base_pt_a, base_pt_b, third_pt;
            bool edge_found = false;
            
            if (curr_rect.size.width > 0 && curr_rect.size.height > 0) {
                cv::Point2f rect_pts[4];
                curr_rect.points(rect_pts);

                // 找底边 (y最大)
                int ei = 0, ej = 1;
                float max_avg_y = -1e30f;
                auto check_edge = [&](int a, int b) {
                    float val = 0.5f * (rect_pts[a].y + rect_pts[b].y);
                    if (val > max_avg_y) { max_avg_y = val; ei = a; ej = b; }
                };
                check_edge(0,1); check_edge(1,2); check_edge(2,3); check_edge(3,0);

                cv::Point2f b0 = rect_pts[ei];
                cv::Point2f b1 = rect_pts[ej];
                // 保证从左到右
                if (b0.x > b1.x) std::swap(b0, b1);
                
                base_pt_a = b0;
                base_pt_b = b1;

                // 找第三点 (离 base_pt_a 最近的剩余点)
                bool used[4] = {false};
                used[ei] = true; used[ej] = true;
                int remain_idx[2], r_cnt = 0;
                for(int k=0; k<4; ++k) if(!used[k]) remain_idx[r_cnt++] = k;
                
                const cv::Point2f& q0 = rect_pts[remain_idx[0]];
                const cv::Point2f& q1 = rect_pts[remain_idx[1]];
                third_pt = (cv::norm(q0 - base_pt_a) <= cv::norm(q1 - base_pt_a)) ? q0 : q1;
                
                edge_found = true;
            }
            
            if (!edge_found) {
                spdlog::info("[#{}] Cannot get bottom edge/third point", i);
            } else {
                // 7.3 计算射线方向 (Pixel -> Camera Ray)
                auto get_ray_dir = [](const cv::Point2f &px) -> Eigen::Vector3d {
                    cv::Vec3d vec(
                        g_mat_k_inv.at<double>(0, 0) * px.x + g_mat_k_inv.at<double>(0, 1) * px.y + g_mat_k_inv.at<double>(0, 2),
                        g_mat_k_inv.at<double>(1, 0) * px.x + g_mat_k_inv.at<double>(1, 1) * px.y + g_mat_k_inv.at<double>(1, 2),
                        g_mat_k_inv.at<double>(2, 0) * px.x + g_mat_k_inv.at<double>(2, 1) * px.y + g_mat_k_inv.at<double>(2, 2)
                    );
                    return Eigen::Vector3d(vec[0], vec[1], vec[2]).normalized();
                };

                Eigen::Vector3d ray_dir_1 = get_ray_dir(base_pt_a);
                Eigen::Vector3d ray_dir_2 = get_ray_dir(base_pt_b);
                Eigen::Vector3d ray_dir_3 = get_ray_dir(third_pt);

                // 7.4 RANSAC 平面拟合
                bool plane_solved = false;
                cv::Point3f center_cam;
                cv::Vec3f normal_cam, direction_cam;
                cv::Point3f pt1_cam, pt2_cam, pt3_cam;

                auto temp_cloud = std::make_shared<open3d::geometry::PointCloud>();
                temp_cloud->points_.assign(points_in_box.begin(), points_in_box.end());

                Eigen::Vector4d plane_param; 
                std::vector<size_t> inliers;
                std::tie(plane_param, inliers) = temp_cloud->SegmentPlane(0.004, 3, 300); // 4mm 阈值

                if (inliers.size() >= 20) {
                    Eigen::Vector3d n(plane_param[0], plane_param[1], plane_param[2]);
                    double d = plane_param[3];
                    double norm_l = n.norm();

                    if (norm_l > 0.0) {
                        n /= norm_l; d /= norm_l;
                        // 统一法向量朝向相机 (-Z方向不是绝对，这里逻辑是调整符号)
                        if (n.z() < 0) { n = -n; d = -d; }

                        // 7.5 射线与平面求交
                        auto get_intersection = [&](const Eigen::Vector3d& dir, double& out_t) -> bool {
                            double nd = n.dot(dir);
                            if (std::abs(nd) < 1e-8) return false;
                            out_t = -d / nd;
                            return (out_t > 0);
                        };

                        double t1, t2, t3;
                        if (get_intersection(ray_dir_1, t1) && 
                            get_intersection(ray_dir_2, t2) && 
                            get_intersection(ray_dir_3, t3)) {
                            
                            Eigen::Vector3d P1 = t1 * ray_dir_1;
                            Eigen::Vector3d P2 = t2 * ray_dir_2;
                            Eigen::Vector3d P3 = t3 * ray_dir_3;

                            // 计算中心点 (底边中点)
                            Eigen::Vector3d M = 0.5 * (P1 + P2);
                            Eigen::Vector3d edge_vec = P2 - P1;
                            double edge_len = edge_vec.norm();
                            
                            if (edge_len >= 1e-8) {
                                edge_vec /= edge_len;
                                // 保证方向一致性 (叉乘z分量)
                                if (edge_vec.cross(n).z() < 0) edge_vec = -edge_vec;

                                center_cam = { (float)M.x(), (float)M.y(), (float)M.z() };
                                normal_cam = { (float)n.x(), (float)n.y(), (float)n.z() };
                                direction_cam = { (float)edge_vec.x(), (float)edge_vec.y(), (float)edge_vec.z() };
                                
                                pt1_cam = { (float)P1.x(), (float)P1.y(), (float)P1.z() };
                                pt2_cam = { (float)P2.x(), (float)P2.y(), (float)P2.z() };
                                pt3_cam = { (float)P3.x(), (float)P3.y(), (float)P3.z() };
                                
                                plane_solved = true;
                            }
                        }
                    }
                }

                if (!plane_solved) {
                    spdlog::info("[#{}] Plane solving failed", i);
                } else {
                    // ==========================================
                    // 7.6 坐标转换 (Camera -> World)
                    // ==========================================
                    // 注意：此处已去除原有的轴交换逻辑，保持直接映射
                    cv::Vec3f normal_cam_final = normal_cam;
                    cv::Vec3f dir_cam_final = direction_cam;

                    // 构造齐次坐标 (单位：毫米 -> 保持与原逻辑一致的数值量级处理)
                    cv::Vec4f pos_cam_mm(
                        center_cam.x * 1000.0f,
                        center_cam.y * 1000.0f,
                        center_cam.z * 1000.0f,
                        1.0f
                    );

                    // 旋转矩阵部分
                    cv::Mat mat_r_wc = g_mat_twc(cv::Rect(0, 0, 3, 3));
                    
                    cv::Mat normal_world_mat = mat_r_wc * cv::Mat(normal_cam_final);
                    cv::Mat dir_world_mat = mat_r_wc * cv::Mat(dir_cam_final);
                    
                    cv::Point3f normal_world(normal_world_mat.at<float>(0), normal_world_mat.at<float>(1), normal_world_mat.at<float>(2));
                    cv::Point3f dir_world(dir_world_mat.at<float>(0), dir_world_mat.at<float>(1), dir_world_mat.at<float>(2));

                    // 位置转换
                    cv::Mat pos_world_homo = g_mat_twc * cv::Mat(pos_cam_mm);
                    cv::Point3f pos_world_m(
                        pos_world_homo.at<float>(0) / 1000.0f,
                        pos_world_homo.at<float>(1) / 1000.0f,
                        pos_world_homo.at<float>(2) / 1000.0f
                    );

                    // 辅助函数：转换点到世界坐标
                    auto transform_point_to_world = [](const cv::Point3f &p) -> cv::Point3f {
                         // 注意：这里的 vector 构造顺序保持 x,y,z
                        cv::Vec4f p_homo(p.x * 1000.0f, p.y * 1000.0f, p.z * 1000.0f, 1.0f);
                        cv::Mat res = g_mat_twc * cv::Mat(p_homo);
                        return cv::Point3f(res.at<float>(0)/1000.f, res.at<float>(1)/1000.f, res.at<float>(2)/1000.f);
                    };

                    cv::Point3f p1_wm = transform_point_to_world(pt1_cam);
                    cv::Point3f p2_wm = transform_point_to_world(pt2_cam);
                    cv::Point3f p3_wm = transform_point_to_world(pt3_cam);

                    // 7.7 计算尺寸 (Width, Height)
                    float obj_width = 0.f, obj_height = 0.f;
                    cv::Point3f vec_w = p2_wm - p1_wm;
                    cv::Point3f vec_h = p3_wm - p1_wm;
                    
                    float w_val = std::sqrt(vec_w.dot(vec_w));
                    float h_val = std::sqrt(vec_h.dot(vec_h));
                    
                    if (std::isfinite(w_val) && std::isfinite(h_val)) {
                        obj_width = w_val;
                        obj_height = h_val;
                    }

                    // 7.8 构建最终旋转矩阵 (Construct Rotation Matrix)
                    // X轴: 法向量 normal_world
                    // Y轴: 正交化后的 dir_world
                    // Z轴: X cross Y
                    
                    auto normalize = [](cv::Point3f v) -> cv::Point3f {
                        float l = std::sqrt(v.dot(v));
                        return (l < 1e-9f) ? cv::Point3f(0,0,0) : v * (1.0f/l);
                    };
                    
                    // 确保轴向正方向
                    if (normal_world.x < 0) normal_world = -normal_world;
                    if (dir_world.y < 0) dir_world = -dir_world;

                    cv::Point3f axis_x = normalize(normal_world);
                    // Gram-Schmidt 正交化
                    cv::Point3f axis_y = normalize(dir_world - axis_x * axis_x.dot(dir_world));
                    
                    // 防止退化 (若 Y 变为 0)
                    if (axis_y == cv::Point3f(0,0,0)) {
                        cv::Point3f ref(0,1,0);
                        if (std::abs(axis_x.dot(ref)) > 0.95f) ref = {1,0,0};
                        axis_y = normalize(ref - axis_x * axis_x.dot(ref));
                    }
                    
                    cv::Point3f axis_z = normalize(axis_x.cross(axis_y));
                    axis_y = normalize(axis_z.cross(axis_x)); // 再次校正Y

                    Eigen::Matrix3d rotation_matrix;
                    rotation_matrix << axis_x.x, axis_y.x, axis_z.x,
                                       axis_x.y, axis_y.y, axis_z.y,
                                       axis_x.z, axis_y.z, axis_z.z;

                    // 计算欧拉角
                    double pitch = std::asin(-rotation_matrix(2, 0));
                    double roll = std::atan2(rotation_matrix(2, 1), rotation_matrix(2, 2));
                    double yaw = std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
                    auto rad2deg = [](double v) { return v * 180.0 / 3.14159265358979323846; };

                    // 7.9 填充结果
                    res.id = static_cast<int>(i);
                    res.xyz_m = pos_world_m;
                    res.wpr_deg = cv::Vec3f((float)rad2deg(roll), (float)rad2deg(pitch), (float)rad2deg(yaw));
                    res.width_m = obj_width;
                    res.height_m = obj_height;
                    res.obb = curr_rect;
                    res.bottom_mid_px = bottom_mid_px;
                    res.p1_w_m = p1_wm;
                    res.p2_w_m = p2_wm;
                    res.p3_w_m = p3_wm;
                    res.rotation_matrix_world = rotation_matrix;
                    
                    is_solved = true;
                }
            }
        }
        
        if (is_solved) {
            results.push_back(std::move(res));
            
            // 7.10 绘制 3D 结果文本
            const auto &r = results.back();
            
            // 绘制旋转矩形框
            cv::Point2f rect_pts[4];
            r.obb.points(rect_pts);
            for (int j = 0; j < 4; ++j) {
                cv::line(vis_image, rect_pts[j], rect_pts[(j + 1) % 4], cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            }
            cv::circle(vis_image, r.bottom_mid_px, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            
            // 准备文本信息
            std::array<std::string, 8> info_lines;
            std::ostringstream oss;
            
            oss.str(""); oss << "#" << r.id; info_lines[0] = oss.str();
            oss.str(""); oss << "x=" << std::fixed << std::setprecision(3) << r.xyz_m.x; info_lines[1] = oss.str();
            oss.str(""); oss << "y=" << std::fixed << std::setprecision(3) << r.xyz_m.y; info_lines[2] = oss.str();
            oss.str(""); oss << "z=" << std::fixed << std::setprecision(3) << r.xyz_m.z; info_lines[3] = oss.str();
            oss.str(""); oss << "W=" << std::fixed << std::setprecision(1) << r.wpr_deg[0]; info_lines[4] = oss.str();
            oss.str(""); oss << "P=" << std::fixed << std::setprecision(1) << r.wpr_deg[1]; info_lines[5] = oss.str();
            oss.str(""); oss << "R=" << std::fixed << std::setprecision(1) << r.wpr_deg[2]; info_lines[6] = oss.str();
            oss.str(""); oss << std::fixed << std::setprecision(1) << r.width_m * 1000 << "," << r.height_m * 1000; info_lines[7] = oss.str();

            // 绘制多行文本
            int base_line = 0;
            int total_h = 0;
            const int line_gap = 4;
            std::vector<cv::Size> text_sizes(8);
            
            for (int j = 0; j < 8; ++j) {
                text_sizes[j] = cv::getTextSize(info_lines[j], cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &base_line);
                total_h += text_sizes[j].height;
            }
            total_h += line_gap * 7;
            
            int cur_y = (int)std::round(r.obb.center.y - total_h * 0.5);
            for (int j = 0; j < 8; ++j) {
                int org_x = (int)std::round(r.obb.center.x - text_sizes[j].width * 0.5);
                int org_y = cur_y + text_sizes[j].height;
                cv::putText(vis_image, info_lines[j], cv::Point(org_x, org_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,0), 1, cv::LINE_AA);
                cur_y += text_sizes[j].height + line_gap;
            }
        }
    }

    // ==========================================
    // 8. 结果输出 (Output)
    // ==========================================
    const fs::path vis_out_path = case_dir / "vis_on_orig.jpg";
    if (!cv::imwrite(vis_out_path.string(), vis_image)) {
        spdlog::error("Failed to write visualization file: {}", vis_out_path.string());
    }

    auto time_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();

    // 填充输出数组
    int total_results = static_cast<int>(results.size());
    int num_to_write = std::min(total_results, YZX_MAX_BOX);
    
    for (int i = 0; i < num_to_write; ++i) {
        const auto &src = results[i];
        auto &dst = box_array[i];
        
        dst.x = src.xyz_m.x;
        dst.y = src.xyz_m.y;
        dst.z = src.xyz_m.z;
        
        dst.width = src.width_m;
        dst.height = src.height_m;
        
        dst.angle_a = src.wpr_deg[0];
        dst.angle_b = src.wpr_deg[1];
        dst.angle_c = src.wpr_deg[2];
        
        const auto &rot = src.rotation_matrix_world;
        dst.rw1 = rot(0, 0); dst.rw2 = rot(0, 1); dst.rw3 = rot(0, 2);
        dst.rw4 = rot(1, 0); dst.rw5 = rot(1, 1); dst.rw6 = rot(1, 2);
        dst.rw7 = rot(2, 0); dst.rw8 = rot(2, 1); dst.rw9 = rot(2, 2);
    }

    // 写入 JSON 调试文件
    json json_root;
    json_root["taskId"] = task_id;
    json_root["elapsed_ms"] = elapsed_ms;
    json_root["total"] = total_results;
    json_root["n_write"] = num_to_write;
    json_root["out_image"] = vis_out_path.string();
    json_root["input_rgb"] = rgb_path.string();
    json_root["input_pcd"] = pcd_path.string();

    auto &json_boxes = json_root["boxes"] = json::array();
    for (int i = 0; i < num_to_write; ++i) {
        const auto &b = box_array[i];
        json_boxes.push_back({
            {"x", b.x}, {"y", b.y}, {"z", b.z},
            {"width", b.width}, {"height", b.height},
            {"angle_a", b.angle_a}, {"angle_b", b.angle_b}, {"angle_c", b.angle_c},
            {"rw1", b.rw1}, {"rw2", b.rw2}, {"rw3", b.rw3},
            {"rw4", b.rw4}, {"rw5", b.rw5}, {"rw6", b.rw6},
            {"rw7", b.rw7}, {"rw8", b.rw8}, {"rw9", b.rw9}
        });
    }
    
    const fs::path json_path = case_dir / "boxes.json";
    std::ofstream ofs(json_path);
    if (ofs.is_open()) {
        ofs << std::setw(2) << json_root;
        ofs.close();
    } else {
        spdlog::error("Failed to write JSON: {}", json_path.string());
    }

    return num_to_write;
}
