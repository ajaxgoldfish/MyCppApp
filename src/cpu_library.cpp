#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cpu_library.h"
#include "LanxinCamera.h"
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <array>
#include <iomanip>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <optional>
#include <fstream>

using nlohmann::json;
namespace fs = std::filesystem;

#ifndef YZX_MAX_BOX
#define YZX_MAX_BOX 100
#endif

namespace {
    // ==========================================
    // 全局配置与资源 (Global Configuration)
    // ==========================================
    // 运行模式: 0 = 本地文件模式, 1 = 相机在线模式
    int g_run_mode = 0; 
    // 计算设备: 1 = CPU, 2 = GPU
    int g_compute_device = 1; 

    std::string g_model_path;
    std::string g_calib_path;
    float g_score_threshold = 0.7f;   // 置信度阈值
    float g_mask_threshold = 0.5f;    // Mask二值化阈值
    bool g_paint_masks_on_vis = true; // 是否在可视化图中绘制Mask

    // 相机内参和外参
    cv::Mat g_intrinsic;
    cv::Mat g_transform_world_cam; // T_wc: 相机到世界的变换矩阵

    // 相机设备 (相机模式使用)
    std::unique_ptr<LanxinCamera> g_camera;

    // 本地计算结果结构体
    struct LocalBoxPoseResult {
        int id = -1;
        cv::Point3f xyz_mm{};         // 中心点坐标 (毫米)
        cv::Vec3f wpr_deg{};          // 欧拉角 (度)
        float width_mm = 0.f;         // 宽度 (毫米)
        float height_mm = 0.f;        // 高度 (毫米)
        cv::RotatedRect obb;          // 2D 旋转矩形
        cv::Point2f bottom_mid_px{};  // 底边中点 (像素)
        cv::Point3f p1_w_mm{}, p2_w_mm{}, p3_w_mm{}; // 关键点 (世界坐标系, 毫米)
        Eigen::Matrix3d rotation_matrix_world;       // 旋转矩阵 (世界坐标系)
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

    struct DetectionResult2D {
        cv::RotatedRect obb;
        cv::Point2f bottom_mid_px;
        cv::Mat1b mask; // Optional: 如果不需要存储原始mask可移除
    };

} // namespace


/**
 * @brief 初始化算法流水线
 * @param is_debug 是否开启调试日志
 */
int bs_yzx_init(const bool is_debug) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(is_debug ? spdlog::level::debug : spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);

    // 默认配置
    g_run_mode = 0;
    g_compute_device = 1;
    g_model_path = "models/end2end.onnx";
    g_calib_path = "config/params.xml";
    g_score_threshold = 0.7f;
    g_mask_threshold = 0.5f;
    g_paint_masks_on_vis = true;

    if (!g_is_pipeline_ready) {
        try {
            // 读取配置文件
            cv::FileStorage fs_config(g_calib_path, cv::FileStorage::READ);
            if (!fs_config.isOpened()) {
                spdlog::error("[init] Cannot open config file: {}", g_calib_path);
                return -25;
            }

            // 读取运行模式和计算设备配置
            // 如果配置文件中没有这些字段，保留默认值
            if (!fs_config["RunMode"].empty()) {
                fs_config["RunMode"] >> g_run_mode;
            }
            if (!fs_config["DeviceType"].empty()) {
                fs_config["DeviceType"] >> g_compute_device;
            }

            spdlog::info("Initializing... RunMode={} (0=File, 1=Camera), DeviceType={} (1=CPU, 2=GPU)", 
                         g_run_mode, g_compute_device);

            // 读取内参
            fs_config["intrinsicRGB"] >> g_intrinsic;
            if (g_intrinsic.empty() || g_intrinsic.rows != 3 || g_intrinsic.cols != 3) return -26;
            if (g_intrinsic.type() != CV_64F) g_intrinsic.convertTo(g_intrinsic, CV_64F);
            
            // 读取外参
            fs_config["extrinsicRGB"] >> g_transform_world_cam;
            
            // 释放文件句柄
            fs_config.release(); // 重要：如果复用 fs 对象需注意
            
            if (g_transform_world_cam.empty()) {
                spdlog::error("[initExtrinsic] extrinsicRGB node not found or empty");
                return -28;
            }
            
            if (g_transform_world_cam.rows != 4 || g_transform_world_cam.cols != 4) {
                spdlog::error("[initExtrinsic] extrinsicRGB must be 4x4 matrix");
                return -29;
            }
            
            if (g_transform_world_cam.type() != CV_64F) {
                g_transform_world_cam.convertTo(g_transform_world_cam, CV_64F);
            }
            
            g_mat_k = g_intrinsic.clone(); // CV_64F
            g_mat_k_inv = g_mat_k.inv();
            g_mat_twc = g_transform_world_cam.clone(); // 4x4
            if (g_mat_twc.type() != CV_32F) g_mat_twc.convertTo(g_mat_twc, CV_32F);

            if (!g_env) {
                g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "mrcnn");
            }
            
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // 如果是 GPU 计算模式，配置 CUDA Provider
            if (g_compute_device == 2) {
                try {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    cuda_options.arena_extend_strategy = 0;
                    cuda_options.gpu_mem_limit = SIZE_MAX;
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cuda_options.do_copy_in_default_stream = 1;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    spdlog::info("CUDA Execution Provider appended.");
                } catch (const std::exception& e) {
                    spdlog::error("Failed to append CUDA provider: {}", e.what());
                    // Fallback to CPU or return error? Let's proceed, ORT might fallback.
                }
            }
            
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

    // 如果是 相机模式，初始化相机 (无论计算用CPU还是GPU，只要数据源是相机就需要)
    if (g_run_mode == 1) {
        if (!g_camera || !g_camera->isOpened()) {
            g_camera = std::make_unique<LanxinCamera>();
            if (!g_camera->isOpened()) {
                spdlog::critical("LanxinCamera connection failed");
                g_camera.reset();
                return -2;
            }
            spdlog::info("LanxinCamera connected");
        }
    }

    return 0;
}

    // ------------------------------------------------------------------------------------------------
    // 辅助函数: 数据预处理 (Preprocessing)
    // ------------------------------------------------------------------------------------------------

    /**
     * @brief 执行 ONNX Runtime 推理并解析结果
     */
    static std::vector<cv::Mat1b> run_inference_and_get_masks(const cv::Mat& image_rgb) {
        std::vector<cv::Mat1b> detected_masks;
        if (image_rgb.empty()) return detected_masks;

        cv::Mat rgb_float;
        cv::cvtColor(image_rgb, rgb_float, cv::COLOR_BGR2RGB);
        rgb_float.convertTo(rgb_float, CV_32F);
        
        // 归一化参数 (Normalize)
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
        auto ort_outputs = g_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                              g_output_names_ptr.data(), g_output_names_ptr.size());
        
        if (ort_outputs.size() < 3) return detected_masks;

        // 解析输出 (Parse outputs)
        auto shape_dets = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto shape_masks = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        int64_t num_detections = shape_dets[1];
        int mask_h = (int) shape_masks[2];
        int mask_w = (int) shape_masks[3];
        
        const float *det_data = ort_outputs[0].GetTensorData<float>();
        const float *mask_tensor_data = ort_outputs[2].GetTensorData<float>();

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
            
            cv::Mat1b full_mask(image_rgb.rows, image_rgb.cols, (uchar)0);
            full_mask(roi_rect).setTo(255, mask_bin);
            detected_masks.emplace_back(std::move(full_mask));
        }

        return detected_masks;
    }

/**
 * @brief 将点云投影到图像平面
 */
static std::vector<ProjectionMap> project_point_cloud_to_image(
    const open3d::geometry::PointCloud& pc, const cv::Size& img_size) 
{
    std::vector<ProjectionMap> proj_map;
    if (g_mat_k.empty()) return proj_map;

    const double fx = g_mat_k.at<double>(0, 0), fy = g_mat_k.at<double>(1, 1);
    const double cx = g_mat_k.at<double>(0, 2), cy = g_mat_k.at<double>(1, 2);
    
    proj_map.reserve(pc.points_.size());
    for (int i = 0; i < (int)pc.points_.size(); ++i) {
        const auto &p = pc.points_[i];
        if (p.z() <= 0) continue; 
        
        int u = (int)std::round(fx * p.x() / p.z() + cx);
        int v = (int)std::round(fy * p.y() / p.z() + cy);
        
        if ((unsigned)u < (unsigned)img_size.width && (unsigned)v < (unsigned)img_size.height) {
            proj_map.push_back({u, v, i});
        }
    }
    return proj_map;
}

/**
 * @brief 从 Mask 提取 2D 旋转矩形和底边中点
 */
static std::optional<DetectionResult2D> extract_rect_from_mask(const cv::Mat1b& mask) {
    if (mask.empty()) return std::nullopt;
    
    std::vector<cv::Point> non_zero_pts;
    cv::findNonZero(mask, non_zero_pts);
    
    if (non_zero_pts.empty()) return std::nullopt;
    
    cv::RotatedRect obb = cv::minAreaRect(non_zero_pts);
    if (obb.size.width <= 0 || obb.size.height <= 0) return std::nullopt;
    
    // 找到底边中点 (Find bottom edge mid point)
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

    DetectionResult2D res;
    res.obb = obb;
    res.bottom_mid_px = (pts_obb[idx1] + pts_obb[idx2]) * 0.5f;
    res.mask = mask; // 拷贝引用
    return res;
}

// ------------------------------------------------------------------------------------------------
// 核心函数: 位姿解算 (Pose Estimation)
// ------------------------------------------------------------------------------------------------

/**
 * @brief 对单个物体进行 6D 位姿解算 (含 RANSAC 平面拟合与坐标变换)
 */
static std::optional<LocalBoxPoseResult> solve_pose_for_single_object(
    const DetectionResult2D& det_2d,
    const std::vector<ProjectionMap>& proj_map,
    const open3d::geometry::PointCloud& global_pc) 
{
    // 1. 过滤旋转矩形框内的点 (Filter points inside RotatedRect)
    std::vector<Eigen::Vector3d> points_in_box;
    points_in_box.reserve(4096);
    
    const float center_x = det_2d.obb.center.x, center_y = det_2d.obb.center.y;
    const float half_w = det_2d.obb.size.width * 0.5f;
    const float half_h = det_2d.obb.size.height * 0.5f;
    const float angle_rad = det_2d.obb.angle * (float)CV_PI / 180.f;
    const float cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

    for (const auto &proj : proj_map) {
        const float dx = (float)proj.u - center_x;
        const float dy = (float)proj.v - center_y;
        // 旋转到局部坐标系 (Rotate to local frame)
        const float local_x = dx * cos_a + dy * sin_a;
        const float local_y = -dx * sin_a + dy * cos_a;
        
        if (std::fabs(local_x) <= half_w && std::fabs(local_y) <= half_h) {
            points_in_box.push_back(global_pc.points_[proj.point_idx]);
        }
    }

    if (points_in_box.size() < 30) return std::nullopt;

    // 2. 识别关键像素点：底边 + 第三点 (Identify key pixels: Base edge + Third point)
    cv::Point2f base_pt_a, base_pt_b, third_pt;
    {
        cv::Point2f rect_pts[4];
        det_2d.obb.points(rect_pts);

        int ei = 0, ej = 1;
        float max_avg_y = -1e30f;
        auto check_edge = [&](int a, int b) {
            float val = 0.5f * (rect_pts[a].y + rect_pts[b].y);
            if (val > max_avg_y) { max_avg_y = val; ei = a; ej = b; }
        };
        check_edge(0,1); check_edge(1,2); check_edge(2,3); check_edge(3,0);

        cv::Point2f b0 = rect_pts[ei];
        cv::Point2f b1 = rect_pts[ej];
        if (b0.x > b1.x) std::swap(b0, b1);
        base_pt_a = b0;
        base_pt_b = b1;

        bool used[4] = {false};
        used[ei] = true; used[ej] = true;
        int remain_idx[2], r_cnt = 0;
        for(int k=0; k<4; ++k) if(!used[k]) remain_idx[r_cnt++] = k;
        
        const cv::Point2f& q0 = rect_pts[remain_idx[0]];
        const cv::Point2f& q1 = rect_pts[remain_idx[1]];
        third_pt = (cv::norm(q0 - base_pt_a) <= cv::norm(q1 - base_pt_a)) ? q0 : q1;
    }

    // 3. 像素转射线 (Pixel to Ray)
    auto get_ray_dir = [](const cv::Point2f &px) -> Eigen::Vector3d {
        cv::Vec3d vec(
            g_mat_k_inv.at<double>(0, 0) * px.x + g_mat_k_inv.at<double>(0, 1) * px.y + g_mat_k_inv.at<double>(0, 2),
            g_mat_k_inv.at<double>(1, 0) * px.x + g_mat_k_inv.at<double>(1, 1) * px.y + g_mat_k_inv.at<double>(1, 2),
            g_mat_k_inv.at<double>(2, 0) * px.x + g_mat_k_inv.at<double>(2, 1) * px.y + g_mat_k_inv.at<double>(2, 2)
        );
        return Eigen::Vector3d(vec[0], vec[1], vec[2]).normalized();
    };
    Eigen::Vector3d ray_1 = get_ray_dir(base_pt_a);
    Eigen::Vector3d ray_2 = get_ray_dir(base_pt_b);
    Eigen::Vector3d ray_3 = get_ray_dir(third_pt);

    // 4. RANSAC 平面拟合 (RANSAC Plane Fitting)
    auto temp_cloud = std::make_shared<open3d::geometry::PointCloud>();
    temp_cloud->points_.assign(points_in_box.begin(), points_in_box.end());
    
    Eigen::Vector4d plane_param; 
    std::vector<size_t> inliers;
    std::tie(plane_param, inliers) = temp_cloud->SegmentPlane(0.004, 3, 300);

    if (inliers.size() < 20) return std::nullopt;

    Eigen::Vector3d n(plane_param[0], plane_param[1], plane_param[2]);
    double d = plane_param[3];
    double norm_l = n.norm();
    if (norm_l < 1e-9) return std::nullopt;

    n /= norm_l; d /= norm_l;
    if (n.z() < 0) { n = -n; d = -d; } // 对齐法向量方向 (Align normal)

    // 5. 射线与平面求交 (Ray-Plane Intersection)
    auto get_intersection = [&](const Eigen::Vector3d& dir, double& out_t) -> bool {
        double nd = n.dot(dir);
        if (std::abs(nd) < 1e-8) return false;
        out_t = -d / nd;
        return (out_t > 0);
    };

    double t1, t2, t3;
    if (!get_intersection(ray_1, t1) || !get_intersection(ray_2, t2) || !get_intersection(ray_3, t3)) {
        return std::nullopt;
    }

    Eigen::Vector3d P1 = t1 * ray_1;
    Eigen::Vector3d P2 = t2 * ray_2;
    Eigen::Vector3d P3 = t3 * ray_3;

    Eigen::Vector3d M = 0.5 * (P1 + P2);
    Eigen::Vector3d edge_vec = P2 - P1;
    double edge_len = edge_vec.norm();
    if (edge_len < 1e-8) return std::nullopt;
    
    edge_vec /= edge_len;
    if (edge_vec.cross(n).z() < 0) edge_vec = -edge_vec;

    // 6. 坐标系转换：相机 -> 世界 (Coordinate Transformation: Camera -> World)
    cv::Vec3f normal_cam = { (float)n.x(), (float)n.y(), (float)n.z() };
    cv::Vec3f dir_cam = { (float)edge_vec.x(), (float)edge_vec.y(), (float)edge_vec.z() };
    cv::Point3f center_cam = { (float)M.x(), (float)M.y(), (float)M.z() };

    // 变换旋转部分 (Transform Rotation part)
    cv::Mat mat_r_wc = g_mat_twc(cv::Rect(0, 0, 3, 3));
    cv::Mat normal_world_mat = mat_r_wc * cv::Mat(normal_cam);
    cv::Mat dir_world_mat = mat_r_wc * cv::Mat(dir_cam);
    
    cv::Point3f normal_world(normal_world_mat.at<float>(0), normal_world_mat.at<float>(1), normal_world_mat.at<float>(2));
    cv::Point3f dir_world(dir_world_mat.at<float>(0), dir_world_mat.at<float>(1), dir_world_mat.at<float>(2));

    // 变换位置部分 (Transform Position part)
    // 注意：输入是米，输出转换为毫米
    auto transform_point_to_world_mm = [](const cv::Point3f &p) -> cv::Point3f {
        cv::Vec4f p_homo(p.x * 1000.0f, p.y * 1000.0f, p.z * 1000.0f, 1.0f); // Convert to mm first
        cv::Mat res = g_mat_twc * cv::Mat(p_homo);
        // 不再除以1000，保持毫米单位
        return cv::Point3f(res.at<float>(0), res.at<float>(1), res.at<float>(2));
    };

    cv::Point3f pos_world_mm = transform_point_to_world_mm(center_cam);
    cv::Point3f p1_w_mm = transform_point_to_world_mm({(float)P1.x(), (float)P1.y(), (float)P1.z()});
    cv::Point3f p2_w_mm = transform_point_to_world_mm({(float)P2.x(), (float)P2.y(), (float)P2.z()});
    cv::Point3f p3_w_mm = transform_point_to_world_mm({(float)P3.x(), (float)P3.y(), (float)P3.z()});

    // 7. 计算尺寸 (Calculate Dimensions in mm)
    cv::Point3f vec_w = p2_w_mm - p1_w_mm;
    cv::Point3f vec_h = p3_w_mm - p1_w_mm;
    float w_val = std::sqrt(vec_w.dot(vec_w));
    float h_val = std::sqrt(vec_h.dot(vec_h));
    
    if (!std::isfinite(w_val) || !std::isfinite(h_val)) return std::nullopt;

    // 8. 构建旋转矩阵 (Construct Rotation Matrix)
    auto normalize = [](cv::Point3f v) -> cv::Point3f {
        float l = std::sqrt(v.dot(v));
        return (l < 1e-9f) ? cv::Point3f(0,0,0) : v * (1.0f/l);
    };
    
    if (normal_world.x < 0) normal_world = -normal_world;
    if (dir_world.y < 0) dir_world = -dir_world;

    cv::Point3f axis_x = normalize(normal_world);
    cv::Point3f axis_y = normalize(dir_world - axis_x * axis_x.dot(dir_world));
    
    if (axis_y == cv::Point3f(0,0,0)) {
        cv::Point3f ref(0,1,0);
        if (std::abs(axis_x.dot(ref)) > 0.95f) ref = {1,0,0};
        axis_y = normalize(ref - axis_x * axis_x.dot(ref));
    }
    cv::Point3f axis_z = normalize(axis_x.cross(axis_y));
    axis_y = normalize(axis_z.cross(axis_x));

    Eigen::Matrix3d rotation_matrix;
    rotation_matrix << axis_x.x, axis_y.x, axis_z.x,
                       axis_x.y, axis_y.y, axis_z.y,
                       axis_x.z, axis_y.z, axis_z.z;

    double pitch = std::asin(-rotation_matrix(2, 0));
    double roll = std::atan2(rotation_matrix(2, 1), rotation_matrix(2, 2));
    double yaw = std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
    auto rad2deg = [](double v) { return v * 180.0 / 3.14159265358979323846; };

    LocalBoxPoseResult res;
    res.xyz_mm = pos_world_mm;
    res.wpr_deg = cv::Vec3f((float)rad2deg(roll), (float)rad2deg(pitch), (float)rad2deg(yaw));
    res.width_mm = w_val;
    res.height_mm = h_val;
    res.obb = det_2d.obb;
    res.bottom_mid_px = det_2d.bottom_mid_px;
    res.p1_w_mm = p1_w_mm;
    res.p2_w_mm = p2_w_mm;
    res.p3_w_mm = p3_w_mm;
    res.rotation_matrix_world = rotation_matrix;
    return res;
}


/**
 * @brief 在图像上绘制结果
 */
static void visualize_results(cv::Mat& vis_image, const std::vector<LocalBoxPoseResult>& results) {
    for (const auto& r : results) {
        cv::Point2f rect_pts[4];
        r.obb.points(rect_pts);
        for (int j = 0; j < 4; ++j) {
            cv::line(vis_image, rect_pts[j], rect_pts[(j + 1) % 4], cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
        cv::circle(vis_image, r.bottom_mid_px, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        
        std::array<std::string, 8> info_lines;
        std::ostringstream oss;
        oss.str(""); oss << "#" << r.id; info_lines[0] = oss.str();
        oss.str(""); oss << "x=" << std::fixed << std::setprecision(1) << r.xyz_mm.x; info_lines[1] = oss.str();
        oss.str(""); oss << "y=" << std::fixed << std::setprecision(1) << r.xyz_mm.y; info_lines[2] = oss.str();
        oss.str(""); oss << "z=" << std::fixed << std::setprecision(1) << r.xyz_mm.z; info_lines[3] = oss.str();
        oss.str(""); oss << "W=" << std::fixed << std::setprecision(1) << r.wpr_deg[0]; info_lines[4] = oss.str();
        oss.str(""); oss << "P=" << std::fixed << std::setprecision(1) << r.wpr_deg[1]; info_lines[5] = oss.str();
        oss.str(""); oss << "R=" << std::fixed << std::setprecision(1) << r.wpr_deg[2]; info_lines[6] = oss.str();
        oss.str(""); oss << std::fixed << std::setprecision(1) << r.width_mm << "," << r.height_mm; info_lines[7] = oss.str();

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


int bs_yzx_object_detection_lanxin(int task_id, zzb::Box box_array[], float y_left_mm, float y_right_mm) {
    if (!g_is_pipeline_ready) return -10;
    // 相机模式下必须有相机
    if (g_run_mode == 1 && (!g_camera || !g_camera->isOpened())) return -11;

    auto time_start = std::chrono::steady_clock::now();

    cv::Mat image_rgb;
    open3d::geometry::PointCloud point_cloud;
    fs::path case_dir = fs::path(g_root_output_dir) / std::to_string(task_id);

    // 1. 数据加载 (Data Loading)
    if (g_run_mode == 0) {
        // 本地文件模式
        if (!fs::exists(case_dir)) {
            spdlog::error("Data directory does not exist: {}", case_dir.string());
            return -21;
        }

        const fs::path rgb_path = case_dir / "rgb.jpg";
        image_rgb = cv::imread(rgb_path.string(), cv::IMREAD_COLOR);
        if (image_rgb.empty()) {
            spdlog::error("Cannot read RGB image: {}", rgb_path.string());
            return -22;
        }

        const fs::path pcd_path = case_dir / "pcAll.pcd";
        if (!open3d::io::ReadPointCloud(pcd_path.string(), point_cloud) || point_cloud.points_.empty()) {
            spdlog::error("Cannot read point cloud data: {}", pcd_path.string());
            return -23;
        }
    } else {
        // 相机模式
        // 确保输出目录存在
        std::error_code ec;
        fs::create_directories(case_dir, ec);

        if (g_camera->CapFrame(image_rgb) != 0 || image_rgb.empty()) {
            spdlog::error("Failed to capture RGB frame");
            return -22;
        }

        // 保存原始 RGB
        const fs::path rgbPath = case_dir / "rgb_orig.jpg";
        if (!cv::imwrite(rgbPath.string(), image_rgb)) {
            spdlog::warn("Failed to save original RGB");
        }

        if (g_camera->CapFrame(point_cloud) != 0 || point_cloud.points_.empty()) {
            spdlog::error("Failed to capture point cloud or empty");
            return -23;
        }

        // 保存原始点云
        const fs::path pcdPath = case_dir / "cloud_orig.pcd";
        open3d::io::WritePointCloudOption opt;
        opt.write_ascii = open3d::io::WritePointCloudOption::IsAscii::Ascii;
        opt.compressed = open3d::io::WritePointCloudOption::Compressed::Uncompressed;
        opt.print_progress = false;
        open3d::io::WritePointCloud(pcdPath.string(), point_cloud, opt);
    }

    // 2. 推理检测 (Inference)
    auto detected_masks = run_inference_and_get_masks(image_rgb);
    
    // 3. 预计算投影 (Precompute Projection)
    auto proj_map = project_point_cloud_to_image(point_cloud, image_rgb.size());

    // 4. 主处理循环 (Main Processing Loop)
    std::vector<LocalBoxPoseResult> results;
    cv::Mat vis_image = image_rgb.clone(); 
    
    // 如果需要，在可视化图像上绘制掩码
    if (g_paint_masks_on_vis) {
        const cv::Scalar kVisColor(0, 255, 0);
        for (const auto& mask : detected_masks) {
            std::vector<cv::Point> pts;
            cv::findNonZero(mask, pts);
            if (!pts.empty()) {
                cv::Rect roi = cv::boundingRect(pts);
                cv::Mat roi_view = vis_image(roi);
                cv::Mat overlay = roi_view.clone();
                cv::Mat1b mask_roi = mask(roi);
                overlay.setTo(kVisColor, mask_roi);
                cv::addWeighted(roi_view, 1.0, overlay, 0.5, 0, roi_view);
            }
        }
    }

    int idx_counter = 0;
    for (const auto &mask : detected_masks) {
        if (auto det_2d = extract_rect_from_mask(mask)) {
            if (auto pose_res = solve_pose_for_single_object(*det_2d, proj_map, point_cloud)) {
                pose_res->id = idx_counter++;
                results.push_back(std::move(*pose_res));
            }
        }
    }

    // 5. 可视化结果 (Visualize Results)
    visualize_results(vis_image, results);

    // [新增] 绘制世界坐标系 Y 边界辅助线 (Y=1000, Y=-1200)
    if (!g_mat_twc.empty() && !g_mat_k.empty()) {
        // float y_left_mm = 1000.0f;  <-- 从参数传入，不再写死
        // float y_right_mm = -1200.0f; <-- 从参数传入，不再写死
        
        // 1. 计算 World -> Camera 变换矩阵 (T_cw = T_wc^-1)
        // g_mat_twc 是 Camera -> World (在 solve_pose 中: res = g_mat_twc * p_camera)
        cv::Mat mat_twc_64;
        g_mat_twc.convertTo(mat_twc_64, CV_64F);
        cv::Mat mat_tcw = mat_twc_64.inv();

        // 2. 确定参考点：取图像中心对应的一条射线上的点
        // 假设在深度 Z_cam = 2000mm 处寻找对应的世界坐标 Xw 和 Zw
        double cx = g_mat_k.at<double>(0, 2);
        // double cy = g_mat_k.at<double>(1, 2);
        double fx = g_mat_k.at<double>(0, 0);
        // double fy = g_mat_k.at<double>(1, 1);
        
        double z_ref_mm = 2000.0; // 2米参考深度
        // 图像中心点对应的归一化相机坐标是 (0, 0, 1) * z_ref
        cv::Mat p_cam_center = (cv::Mat_<double>(4, 1) << 0, 0, z_ref_mm, 1.0);
        
        // 转到世界坐标系
        cv::Mat p_world_center = mat_twc_64 * p_cam_center;
        double xw = p_world_center.at<double>(0, 0);
        double zw = p_world_center.at<double>(2, 0);
        // double yw = p_world_center.at<double>(1, 0); // 这个值我们将用目标 Y 替换
        
        auto get_u_from_world_y = [&](double target_y) -> int {
            // 构造目标世界坐标点 (保持 Xw, Zw 不变，修改 Yw)
            cv::Mat p_w = (cv::Mat_<double>(4, 1) << xw, target_y, zw, 1.0);
            
            // 转回相机坐标
            cv::Mat p_c = mat_tcw * p_w;
            double z_c = p_c.at<double>(2, 0);
            
            // 防止除以零或投影到相机背面
            if (z_c < 1.0) return -10000; 
            
            double x_c = p_c.at<double>(0, 0);
            // 投影 u = fx * x / z + cx
            return std::round(fx * x_c / z_c + cx);
        };

        int u_l = get_u_from_world_y(y_left_mm);
        int u_r = get_u_from_world_y(y_right_mm);

        auto draw_v_line = [&](int u, const cv::Scalar& color, const std::string& txt) {
            if (u >= 0 && u < vis_image.cols) {
                cv::line(vis_image, cv::Point(u, 0), cv::Point(u, vis_image.rows), color, 2);
                cv::putText(vis_image, txt, cv::Point(u + 5, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
        };

        draw_v_line(u_l, cv::Scalar(255, 0, 0), "Y=" + std::to_string((int)y_left_mm));   // 蓝色
        draw_v_line(u_r, cv::Scalar(0, 0, 255), "Y=" + std::to_string((int)y_right_mm));  // 红色
    }

    // 6. 结果输出 (Output)
    const fs::path vis_out_path = case_dir / "vis_on_orig.jpg";
    cv::imwrite(vis_out_path.string(), vis_image);

    auto time_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();

    int total_results = static_cast<int>(results.size());
    int num_to_write = std::min(total_results, YZX_MAX_BOX);
    
    for (int i = 0; i < num_to_write; ++i) {
        const auto &src = results[i];
        auto &dst = box_array[i];
        
        dst.id = src.id;
        // 直接赋值，单位已是毫米 (Direct assignment, unit is already mm)
        dst.x = src.xyz_mm.x; 
        dst.y = src.xyz_mm.y; 
        dst.z = src.xyz_mm.z;
        
        dst.width = src.width_mm; 
        dst.height = src.height_mm;
        
        dst.angle_a = src.wpr_deg[0]; dst.angle_b = src.wpr_deg[1]; dst.angle_c = src.wpr_deg[2];
        const auto &rot = src.rotation_matrix_world;
        dst.rw1 = rot(0,0); dst.rw2 = rot(0,1); dst.rw3 = rot(0,2);
        dst.rw4 = rot(1,0); dst.rw5 = rot(1,1); dst.rw6 = rot(1,2);
        dst.rw7 = rot(2,0); dst.rw8 = rot(2,1); dst.rw9 = rot(2,2);
    }

    // 写入 JSON
    json json_root;
    json_root["taskId"] = task_id;
    json_root["elapsed_ms"] = elapsed_ms;
    json_root["total"] = total_results;
    json_root["run_mode"] = g_run_mode;
    json_root["device_type"] = g_compute_device;
    json_root["boxes"] = json::array();
    for (int i = 0; i < num_to_write; ++i) {
        const auto &b = box_array[i];
        json_root["boxes"].push_back({ 
            {"id", b.id},
            {"x", b.x}, {"y", b.y}, {"z", b.z},
            {"w", b.width}, {"h", b.height},
            {"W", b.angle_a}, {"P", b.angle_b}, {"R", b.angle_c}
        });
    }
    
    std::ofstream ofs(case_dir / "boxes.json");
    if (ofs.is_open()) ofs << std::setw(2) << json_root;

    spdlog::info("[ OK ] taskId={} -> {}, targets={} (written {}), time={:.3f} ms, Mode={}, Device={}",
             task_id, vis_out_path.string(), total_results, num_to_write, elapsed_ms, 
             (g_run_mode==0 ? "File" : "Camera"), (g_compute_device==1 ? "CPU" : "GPU"));

    return num_to_write;
}
