#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cpu_library.h"
#include "LanxinCamera.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <array>
#include <iomanip>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <optional>
#include <fstream>
#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include "LanxinCamera.h"
#include <nlohmann/json.hpp>

using namespace std;
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
    std::string g_cnn_config_path;
    float g_score_threshold = 0.65f;   // 置信度阈值
    float g_mask_threshold = 0.5f;    // Mask二值化阈值
    bool g_paint_masks_on_vis = true; // 是否在可视化图中绘制Mask
    int g_rfdetr_input_width = 560;
    int g_rfdetr_input_height = 560;
    bool g_rfdetr_exclude_last_class = false;
    bool g_rfdetr_clip_masks_to_boxes = false;
    std::string g_rfdetr_score_activation = "sigmoid";

    // 相机内参和外参
    cv::Mat g_intrinsic;
    cv::Mat g_transform_world_cam; // T_wc: 相机到世界的变换矩阵

    // 相机设备 (相机模式使用)
    std::unique_ptr<LanxinCamera> g_camera;

    // 本地计算结果结构体
    struct LocalBoxPoseResult {
        int id = -1;
        cv::Point3f xyz_mm{};
        cv::Vec3f wpr_deg{};
        float width_mm = 0.f;
        float height_mm = 0.f;
        std::array<cv::Point2f, 4> quad_pts;  // 凸包四点 (像素)
        cv::Point2f bottom_mid_px{};
        cv::Point3f p1_w_mm{}, p2_w_mm{}, p3_w_mm{};
        Eigen::Matrix3d rotation_matrix_world;
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
        std::array<cv::Point2f, 4> quad_pts;  // 凸包四点 (顺时针)
        cv::Point2f bottom_mid_px;
    };

    std::string trim_copy(const std::string& text) {
        const auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) {
            return std::isspace(ch);
        });
        const auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) {
            return std::isspace(ch);
        }).base();

        if (begin >= end) return {};
        return {begin, end};
    }

    std::string lower_copy(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return text;
    }

    bool parse_bool_value(const std::string& value, bool default_value) {
        const auto lowered = lower_copy(trim_copy(value));
        if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") return true;
        if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") return false;
        return default_value;
    }

    void load_cnn_config() {
        std::ifstream file(g_cnn_config_path);
        if (!file.is_open()) {
            spdlog::warn("[init] Cannot open CNN config file: {}, use default score_threshold={}",
                         g_cnn_config_path, g_score_threshold);
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            const auto comment_pos = line.find_first_of("#;");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            line = trim_copy(line);
            if (line.empty() || (line.front() == '[' && line.back() == ']')) continue;

            const auto equals_pos = line.find('=');
            if (equals_pos == std::string::npos) continue;

            const auto key = trim_copy(line.substr(0, equals_pos));
            const auto value = trim_copy(line.substr(equals_pos + 1));
            const auto key_lower = lower_copy(key);

            try {
                if (key_lower == "model_path") {
                    if (!value.empty()) g_model_path = value;
                } else if (key_lower == "score_threshold") {
                    const float parsed_value = std::stof(value);
                    if (!std::isfinite(parsed_value) || parsed_value < 0.0f || parsed_value > 1.0f) {
                        spdlog::warn("[init] Invalid score_threshold in {}: {}, use default {}",
                                     g_cnn_config_path, value, g_score_threshold);
                        continue;
                    }
                    g_score_threshold = parsed_value;
                } else if (key_lower == "mask_threshold") {
                    const float parsed_value = std::stof(value);
                    if (!std::isfinite(parsed_value) || parsed_value < 0.0f || parsed_value > 1.0f) {
                        spdlog::warn("[init] Invalid mask_threshold in {}: {}, use default {}",
                                     g_cnn_config_path, value, g_mask_threshold);
                        continue;
                    }
                    g_mask_threshold = parsed_value;
                } else if (key_lower == "input_width" || key_lower == "rfdetr_input_width") {
                    const int parsed_value = std::stoi(value);
                    if (parsed_value > 0) g_rfdetr_input_width = parsed_value;
                } else if (key_lower == "input_height" || key_lower == "rfdetr_input_height") {
                    const int parsed_value = std::stoi(value);
                    if (parsed_value > 0) g_rfdetr_input_height = parsed_value;
                } else if (key_lower == "score_activation" || key_lower == "rfdetr_score_activation") {
                    const auto activation = lower_copy(value);
                    if (activation == "sigmoid" || activation == "softmax") {
                        g_rfdetr_score_activation = activation;
                    } else {
                        spdlog::warn("[init] Unsupported score_activation '{}', use {}",
                                     value, g_rfdetr_score_activation);
                    }
                } else if (key_lower == "exclude_last_class" || key_lower == "rfdetr_exclude_last_class") {
                    g_rfdetr_exclude_last_class = parse_bool_value(value, g_rfdetr_exclude_last_class);
                } else if (key_lower == "clip_masks_to_boxes" || key_lower == "rfdetr_clip_masks_to_boxes") {
                    g_rfdetr_clip_masks_to_boxes = parse_bool_value(value, g_rfdetr_clip_masks_to_boxes);
                }
            } catch (const std::exception& e) {
                spdlog::warn("[init] Failed to parse {} in {}: {}", key, g_cnn_config_path, e.what());
            }
        }

        spdlog::info("[init] RF-DETR model={}, input={}x{}, score_threshold={}, mask_threshold={}, activation={}, exclude_last_class={}, clip_masks_to_boxes={}",
                     g_model_path, g_rfdetr_input_width, g_rfdetr_input_height,
                     g_score_threshold, g_mask_threshold, g_rfdetr_score_activation,
                     g_rfdetr_exclude_last_class, g_rfdetr_clip_masks_to_boxes);
    }

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
    g_model_path = "models/rfdetr.onnx";
    g_calib_path = "config/params.xml";
    g_cnn_config_path = "cnn.ini";
    g_score_threshold = 0.7f;
    g_mask_threshold = 0.5f;
    g_paint_masks_on_vis = true;

    if (!g_is_pipeline_ready) {
        try {
            load_cnn_config();

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
                g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rfdetr");
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

int bs_yzx_box_sizeof() {
    return static_cast<int>(sizeof(zzb::Box));
}

    // ------------------------------------------------------------------------------------------------
    // 辅助函数: 数据预处理 (Preprocessing)
    // ------------------------------------------------------------------------------------------------

    float sigmoid(float value) {
        return 1.0f / (1.0f + std::exp(-value));
    }

    size_t find_output_index(const std::vector<std::string>& preferred_names, size_t fallback_index) {
        for (const auto& preferred : preferred_names) {
            const auto preferred_lower = lower_copy(preferred);
            for (size_t i = 0; i < g_output_names_str.size(); ++i) {
                if (lower_copy(g_output_names_str[i]) == preferred_lower) {
                    return i;
                }
            }
        }
        return fallback_index;
    }

    cv::Rect box_cxcywh_to_rect(const float* box_data, const cv::Size& image_size) {
        float cx = box_data[0];
        float cy = box_data[1];
        float w = box_data[2];
        float h = box_data[3];

        if (std::max({std::abs(cx), std::abs(cy), std::abs(w), std::abs(h)}) <= 2.0f) {
            cx *= static_cast<float>(image_size.width);
            cy *= static_cast<float>(image_size.height);
            w *= static_cast<float>(image_size.width);
            h *= static_cast<float>(image_size.height);
        }

        const int x1 = std::clamp(static_cast<int>(std::lround(cx - w * 0.5f)), 0, image_size.width - 1);
        const int y1 = std::clamp(static_cast<int>(std::lround(cy - h * 0.5f)), 0, image_size.height - 1);
        const int x2 = std::clamp(static_cast<int>(std::lround(cx + w * 0.5f)), 0, image_size.width - 1);
        const int y2 = std::clamp(static_cast<int>(std::lround(cy + h * 0.5f)), 0, image_size.height - 1);
        return cv::Rect(cv::Point(x1, y1), cv::Point(std::max(x1 + 1, x2), std::max(y1 + 1, y2))) &
               cv::Rect(0, 0, image_size.width, image_size.height);
    }

    float max_class_score(const float* logits, int class_count) {
        const int score_class_count = g_rfdetr_exclude_last_class ? std::max(0, class_count - 1) : class_count;
        if (score_class_count <= 0) return 0.0f;

        if (g_rfdetr_score_activation == "softmax") {
            float max_logit = logits[0];
            for (int i = 1; i < class_count; ++i) max_logit = std::max(max_logit, logits[i]);

            double sum_exp = 0.0;
            for (int i = 0; i < class_count; ++i) {
                sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
            }
            if (sum_exp <= 0.0) return 0.0f;

            float best = 0.0f;
            for (int i = 0; i < score_class_count; ++i) {
                const float score = static_cast<float>(std::exp(static_cast<double>(logits[i] - max_logit)) / sum_exp);
                best = std::max(best, score);
            }
            return best;
        }

        float best = 0.0f;
        for (int i = 0; i < score_class_count; ++i) {
            best = std::max(best, sigmoid(logits[i]));
        }
        return best;
    }

    int64_t query_count_from_shape(const std::vector<int64_t>& shape, size_t trailing_dims) {
        if (shape.size() <= trailing_dims) return 0;

        int64_t count = 0;
        for (size_t i = 0; i < shape.size() - trailing_dims; ++i) {
            count = std::max(count, shape[i]);
        }
        return count;
    }

    std::string shape_to_string(const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ",";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }

    std::string make_timestamp_dir_name() {
        const auto now = std::chrono::system_clock::now();
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

        std::tm local_tm{};
#ifdef _WIN32
        localtime_s(&local_tm, &now_time);
#else
        localtime_r(&now_time, &local_tm);
#endif

        std::ostringstream oss;
        oss << std::put_time(&local_tm, "%Y%m%d%H%M%S")
            << std::setw(3) << std::setfill('0') << millis.count();
        return oss.str();
    }

    std::optional<fs::path> find_latest_input_case_dir() {
        const fs::path root = g_root_output_dir;
        if (!fs::exists(root) || !fs::is_directory(root)) {
            return std::nullopt;
        }

        std::optional<fs::path> latest_dir;
        fs::file_time_type latest_time{};
        for (const auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_directory()) continue;

            const fs::path case_dir = entry.path();
            const fs::path rgb_path = case_dir / "rgb_orig.jpg";
            const fs::path pcd_path = case_dir / "cloud_orig.pcd";
            if (!fs::exists(rgb_path) || !fs::exists(pcd_path)) continue;

            std::error_code ec;
            const auto write_time = fs::last_write_time(case_dir, ec);
            if (ec) continue;

            if (!latest_dir || write_time > latest_time) {
                latest_dir = case_dir;
                latest_time = write_time;
            }
        }
        return latest_dir;
    }

    /**
     * @brief 执行 RF-DETR Seg ONNX 推理并返回原图尺寸的实例Mask
     */
    static std::vector<cv::Mat1b> run_inference_and_get_masks(const cv::Mat& image_rgb) {
        std::vector<cv::Mat1b> detected_masks;
        if (image_rgb.empty()) return detected_masks;

        cv::Mat resized;
        cv::resize(image_rgb, resized, cv::Size(g_rfdetr_input_width, g_rfdetr_input_height), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb_float;
        cv::cvtColor(resized, rgb_float, cv::COLOR_BGR2RGB);
        rgb_float.convertTo(rgb_float, CV_32F, 1.0 / 255.0);

        static const cv::Scalar kMean(0.485, 0.456, 0.406);
        static const cv::Scalar kStdv(0.229, 0.224, 0.225);
        cv::subtract(rgb_float, kMean, rgb_float);
        cv::divide(rgb_float, kStdv, rgb_float);

        cv::Mat blob;
        cv::dnn::blobFromImage(rgb_float, blob, 1.0, cv::Size(), {}, false, false, CV_32F);

        std::vector<int64_t> input_shape = {1, 3, blob.size[2], blob.size[3]};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, reinterpret_cast<float *>(blob.data), static_cast<size_t>(blob.total()),
            input_shape.data(), input_shape.size());

        const char *input_names[] = {g_input_name.c_str()};
        spdlog::info("RF-DETR running ONNX inference, input={}x{}", g_rfdetr_input_width, g_rfdetr_input_height);
        auto ort_outputs = g_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                              g_output_names_ptr.data(), g_output_names_ptr.size());
        spdlog::info("RF-DETR ONNX inference finished, output_count={}", ort_outputs.size());

        if (ort_outputs.size() < 3) {
            spdlog::error("RF-DETR Seg expects 3 outputs: dets, labels, masks. Actual output count={}", ort_outputs.size());
            return detected_masks;
        }

        const size_t boxes_idx = find_output_index({"dets", "pred_boxes", "boxes"}, 0);
        const size_t logits_idx = find_output_index({"labels", "pred_logits", "logits"}, 1);
        const size_t masks_idx = find_output_index({"masks", "pred_masks"}, 2);

        auto info_boxes = ort_outputs[boxes_idx].GetTensorTypeAndShapeInfo();
        auto info_logits = ort_outputs[logits_idx].GetTensorTypeAndShapeInfo();
        auto info_masks = ort_outputs[masks_idx].GetTensorTypeAndShapeInfo();
        auto shape_boxes = info_boxes.GetShape();
        auto shape_logits = info_logits.GetShape();
        auto shape_masks = info_masks.GetShape();
        spdlog::info("RF-DETR outputs: boxes {}={}, logits {}={}, masks {}={}",
                     g_output_names_str[boxes_idx], shape_to_string(shape_boxes),
                     g_output_names_str[logits_idx], shape_to_string(shape_logits),
                     g_output_names_str[masks_idx], shape_to_string(shape_masks));

        if (info_boxes.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info_logits.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info_masks.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            spdlog::error("RF-DETR outputs must be float tensors.");
            return detected_masks;
        }

        if (shape_boxes.empty() || shape_logits.empty() || shape_masks.size() < 3 ||
            shape_boxes.back() != 4 || shape_logits.back() <= 0) {
            spdlog::error("Unexpected RF-DETR output shapes: boxes dims={}, logits dims={}, masks dims={}",
                          shape_boxes.size(), shape_logits.size(), shape_masks.size());
            return detected_masks;
        }

        const int64_t box_count = query_count_from_shape(shape_boxes, 1);
        const int64_t logit_count = query_count_from_shape(shape_logits, 1);
        const int64_t mask_count = query_count_from_shape(shape_masks, 2);
        const int64_t query_count = std::min({box_count, logit_count, mask_count});
        const int class_count = static_cast<int>(shape_logits.back());
        const int mask_h = static_cast<int>(shape_masks[shape_masks.size() - 2]);
        const int mask_w = static_cast<int>(shape_masks[shape_masks.size() - 1]);

        if (query_count <= 0 || mask_h <= 0 || mask_w <= 0) {
            spdlog::warn("RF-DETR produced empty outputs: queries={}, mask={}x{}", query_count, mask_w, mask_h);
            return detected_masks;
        }

        const int64_t mask_area = static_cast<int64_t>(mask_h) * mask_w;
        const int64_t required_box_elements = query_count * 4;
        const int64_t required_logit_elements = query_count * class_count;
        const int64_t required_mask_elements = query_count * mask_area;
        if (info_boxes.GetElementCount() < required_box_elements ||
            info_logits.GetElementCount() < required_logit_elements ||
            info_masks.GetElementCount() < required_mask_elements) {
            spdlog::error("RF-DETR output buffers are smaller than expected: boxes {}/{}, logits {}/{}, masks {}/{}",
                          info_boxes.GetElementCount(), required_box_elements,
                          info_logits.GetElementCount(), required_logit_elements,
                          info_masks.GetElementCount(), required_mask_elements);
            return detected_masks;
        }

        const float *box_data = ort_outputs[boxes_idx].GetTensorData<float>();
        const float *logit_data = ort_outputs[logits_idx].GetTensorData<float>();
        const float *mask_data = ort_outputs[masks_idx].GetTensorData<float>();

        detected_masks.reserve(static_cast<size_t>(query_count));
        for (int64_t i = 0; i < query_count; ++i) {
            const float score = max_class_score(logit_data + i * class_count, class_count);
            if (score < g_score_threshold) continue;

            const cv::Rect box_rect = box_cxcywh_to_rect(box_data + i * 4, image_rgb.size());
            if (box_rect.area() <= 0) continue;

            const float *curr_mask_ptr = mask_data + i * mask_area;
            cv::Mat mask_raw(mask_h, mask_w, CV_32F, const_cast<float *>(curr_mask_ptr));

            double min_value = 0.0, max_value = 0.0;
            cv::minMaxLoc(mask_raw, &min_value, &max_value);
            const bool mask_is_logits = min_value < 0.0 || max_value > 1.0;

            cv::Mat mask_prob(mask_h, mask_w, CV_32F);
            if (mask_is_logits) {
                for (int y = 0; y < mask_h; ++y) {
                    const float *src = mask_raw.ptr<float>(y);
                    float *dst = mask_prob.ptr<float>(y);
                    for (int x = 0; x < mask_w; ++x) {
                        dst[x] = sigmoid(src[x]);
                    }
                }
            } else {
                mask_raw.copyTo(mask_prob);
            }

            cv::Mat mask_resized;
            cv::resize(mask_prob, mask_resized, image_rgb.size(), 0, 0, cv::INTER_LINEAR);

            cv::Mat1b full_mask;
            cv::compare(mask_resized, g_mask_threshold, full_mask, cv::CMP_GT);

            if (g_rfdetr_clip_masks_to_boxes) {
                cv::Mat1b clipped(image_rgb.rows, image_rgb.cols, static_cast<uchar>(0));
                full_mask(box_rect).copyTo(clipped(box_rect));
                full_mask = std::move(clipped);
            }

            detected_masks.emplace_back(std::move(full_mask));
        }

        spdlog::debug("RF-DETR Seg kept {} masks from {} queries", detected_masks.size(), query_count);
        return detected_masks;
    }

/**
 * @brief 将点云投影到图像平面 (PCL 版本)
 */
static std::vector<ProjectionMap> project_point_cloud_to_image(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, const cv::Size& img_size) 
{
    std::vector<ProjectionMap> proj_map;
    if (g_mat_k.empty() || !pc) return proj_map;

    const double fx = g_mat_k.at<double>(0, 0), fy = g_mat_k.at<double>(1, 1);
    const double cx = g_mat_k.at<double>(0, 2), cy = g_mat_k.at<double>(1, 2);
    
    proj_map.reserve(pc->points.size());
    
    #pragma omp parallel
    {
        std::vector<ProjectionMap> local_map;
        // 预分配足够的空间，避免频繁扩容
        local_map.reserve(pc->points.size() / 4); 
        
        #pragma omp for nowait
        for (int i = 0; i < (int)pc->points.size(); ++i) {
            const auto &p = pc->points[i];
            if (p.z <= 0) continue; 
            
            int u = (int)std::round(fx * p.x / p.z + cx);
            int v = (int)std::round(fy * p.y / p.z + cy);
            
            if ((unsigned)u < (unsigned)img_size.width && (unsigned)v < (unsigned)img_size.height) {
                local_map.push_back({u, v, i});
            }
        }
        
        #pragma omp critical
        {
            proj_map.insert(proj_map.end(), local_map.begin(), local_map.end());
        }
    }
    return proj_map;
}

/**
 * @brief 从 Mask 提取凸包四点及底边中点
 */
static std::optional<DetectionResult2D> extract_rect_from_mask(const cv::Mat1b& mask) {
    if (mask.empty()) return std::nullopt;

    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);
    if (pts.empty()) return std::nullopt;

    std::vector<cv::Point> hull;
    cv::convexHull(pts, hull);

    std::vector<cv::Point2f> quad;
    if (hull.size() == 4) {
        for (const auto& p : hull) quad.push_back(cv::Point2f(p));
    } else if (hull.size() >= 3) {
        float arc_len = cv::arcLength(hull, true);
        if (arc_len < 1e-6f) return std::nullopt;
        // 尝试多种 epsilon，放宽拟合限制
        const float eps_ratios[] = {0.01f, 0.02f, 0.03f, 0.05f, 0.1f, 0.2f};
        for (float r : eps_ratios) {
            cv::approxPolyDP(hull, quad, r * arc_len, true);
            if (quad.size() == 4) break;
        }
        if (quad.size() != 4) {
            // 仍无 4 点，用最小外接矩形兜底
            cv::RotatedRect rr = cv::minAreaRect(hull);
            cv::Point2f pts4[4];
            rr.points(pts4);
            quad.assign(pts4, pts4 + 4);
        }
    } else {
        return std::nullopt;
    }

    // 找底边 (y 最大)
    int i0 = 0, i1 = 1;
    float max_y = -1e30f;
    auto check = [&](int a, int b) {
        float y = (quad[a].y + quad[b].y) * 0.5f;
        if (y > max_y) { max_y = y; i0 = a; i1 = b; }
    };
    check(0, 1); check(1, 2); check(2, 3); check(3, 0);

    DetectionResult2D res;
    for (int i = 0; i < 4; ++i) res.quad_pts[i] = quad[i];
    res.bottom_mid_px = (quad[i0] + quad[i1]) * 0.5f;
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
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& global_pc) 
{
    // 1. 过滤四边形内的点 (加入 Bounding Box 快速过滤)
    std::vector<Eigen::Vector3d> points_in_box;
    points_in_box.reserve(4096);
    std::vector<cv::Point2f> quad(det_2d.quad_pts.begin(), det_2d.quad_pts.end());

    // 计算四边形的外接矩形
    cv::Rect bounding_rect = cv::boundingRect(quad);

    for (const auto& proj : proj_map) {
        // 先进行快速的 AABB 盒过滤
        if (proj.u >= bounding_rect.x && proj.u <= bounding_rect.x + bounding_rect.width &&
            proj.v >= bounding_rect.y && proj.v <= bounding_rect.y + bounding_rect.height) {
            
            // 只有在框内的点，才进行昂贵的多边形测试
            if (cv::pointPolygonTest(quad, cv::Point2f((float)proj.u, (float)proj.v), false) >= 0) {
                const auto& pt = global_pc->points[proj.point_idx];
                points_in_box.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    }
    if (points_in_box.size() < 30) return std::nullopt;

    // 2. 关键像素点：底边两点 + 第三点
    const auto& q = det_2d.quad_pts;
    int ei = 0, ej = 1;
    float max_y = -1e30f;
    auto check = [&](int a, int b) {
        float y = (q[a].y + q[b].y) * 0.5f;
        if (y > max_y) { max_y = y; ei = a; ej = b; }
    };
    check(0, 1); check(1, 2); check(2, 3); check(3, 0);

    cv::Point2f base_pt_a = q[ei], base_pt_b = q[ej];
    if (base_pt_a.x > base_pt_b.x) std::swap(base_pt_a, base_pt_b);

    bool used[4] = {false};
    used[ei] = true; used[ej] = true;
    int r0 = -1, r1 = -1;
    for (int k = 0; k < 4; ++k) if (!used[k]) { if (r0 < 0) r0 = k; else { r1 = k; break; } }
    cv::Point2f third_pt = (cv::norm(q[r0] - base_pt_a) <= cv::norm(q[r1] - base_pt_a)) ? q[r0] : q[r1];

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

    // 4. RANSAC 平面拟合 (RANSAC Plane Fitting) - PCL Implementation
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    temp_cloud->points.reserve(points_in_box.size());
    for(const auto& pt : points_in_box) {
        temp_cloud->points.emplace_back(pt.x(), pt.y(), pt.z());
    }
    temp_cloud->width = temp_cloud->points.size();
    temp_cloud->height = 1;
    temp_cloud->is_dense = false;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.004); // 4mm
    seg.setMaxIterations(300);
    seg.setInputCloud(temp_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < 20) return std::nullopt;

    Eigen::Vector3d n(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    double d = coefficients->values[3];
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

    auto transform_point_to_world_mm = [](const cv::Point3f &p) -> cv::Point3f {
        // 输入点 p 是米 (m)，外参 g_mat_twc 是毫米 (mm)
        // 必须先将 p 转为毫米才能与矩阵相乘
        cv::Vec4f p_homo(p.x * 1000.0f, p.y * 1000.0f, p.z * 1000.0f, 1.0f);
        cv::Mat res = g_mat_twc * cv::Mat(p_homo);
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
    res.quad_pts = det_2d.quad_pts;
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
        for (int j = 0; j < 4; ++j) {
            cv::line(vis_image, r.quad_pts[j], r.quad_pts[(j + 1) % 4], cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
        cv::circle(vis_image, r.bottom_mid_px, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

        cv::Point2f center(0, 0);
        for (const auto& p : r.quad_pts) center += p;
        center *= 0.25f;

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

        int cur_y = (int)std::round(center.y - total_h * 0.5);
        for (int j = 0; j < 8; ++j) {
            int org_x = (int)std::round(center.x - text_sizes[j].width * 0.5);
            int org_y = cur_y + text_sizes[j].height;
            cv::putText(vis_image, info_lines[j], cv::Point(org_x, org_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,0), 1, cv::LINE_AA);
            cur_y += text_sizes[j].height + line_gap;
        }
    }
}


int bs_yzx_object_detection_lanxin(zzb::Box box_array[]) {
    try {
    if (!g_is_pipeline_ready) return -10;
    if (box_array == nullptr) return -12;
    // 相机模式下必须有相机
    if (g_run_mode == 1 && (!g_camera || !g_camera->isOpened())) return -11;

    auto time_start = std::chrono::steady_clock::now();

    cv::Mat image_rgb;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const std::string timestamp = make_timestamp_dir_name();
    fs::path output_dir = fs::path(g_root_output_dir) / timestamp;

    // 1. 数据加载 (Data Loading)
    if (g_run_mode == 0) {
        // 本地文件模式
        const auto input_dir = find_latest_input_case_dir();
        if (!input_dir) {
            spdlog::error("No input directory found under {}. Expected rgb_orig.jpg and cloud_orig.pcd",
                          g_root_output_dir);
            return -21;
        }
        spdlog::info("Using input directory: {}", input_dir->string());

        const fs::path rgb_path = *input_dir / "rgb_orig.jpg";
        image_rgb = cv::imread(rgb_path.string(), cv::IMREAD_COLOR);
        if (image_rgb.empty()) {
            spdlog::error("Cannot read RGB image: {}", rgb_path.string());
            return -22;
        }

        const fs::path pcd_path = *input_dir / "cloud_orig.pcd";
        if (pcl::io::loadPCDFile(pcd_path.string(), *point_cloud) == -1 || point_cloud->empty()) {
            spdlog::error("Cannot read point cloud data: {}", pcd_path.string());
            return -23;
        }
        std::error_code ec;
        fs::create_directories(output_dir, ec);
    } else {
        // 相机模式
        // 确保输出目录存在
        std::error_code ec;
        fs::create_directories(output_dir, ec);

        if (g_camera->CapFrame(image_rgb) != 0 || image_rgb.empty()) {
            spdlog::error("Failed to capture RGB frame");
            return -22;
        }

        // 异步保存原始 RGB
        const fs::path rgbPath = output_dir / "rgb_orig.jpg";
        // 深拷贝图像，防止主线程后续处理修改了 image_rgb 导致保存出错
        cv::Mat image_rgb_clone = image_rgb.clone(); 
        std::thread([rgbPath, image_rgb_clone]() {
            if (!cv::imwrite(rgbPath.string(), image_rgb_clone)) {
                spdlog::warn("Failed to save original RGB");
            }
        }).detach(); // detach() 实现“阅后即焚”，不阻塞主线程

        if (g_camera->CapFrame(*point_cloud) != 0 || point_cloud->empty()) {
            spdlog::error("Failed to capture point cloud or empty");
            return -23;
        }

        // 异步保存原始点云
        const fs::path pcdPath = output_dir / "cloud_orig.pcd";
        // 深拷贝点云，防止主线程后续处理修改了 point_cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clone(new pcl::PointCloud<pcl::PointXYZ>(*point_cloud));
        std::thread([pcdPath, cloud_clone]() {
            pcl::io::savePCDFileASCII(pcdPath.string(), *cloud_clone);
            // 额外建议：如果不需要人类可读，强烈建议改成二进制保存，速度快得多，文件也小：
            // pcl::io::savePCDFileBinary(pcdPath.string(), *cloud_clone);
        }).detach();
    }

    // 2. 推理检测 (Inference)
    auto t1 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 1: Data Loading took {:.3f} ms", std::chrono::duration<double, std::milli>(t1 - time_start).count());

    auto detected_masks = run_inference_and_get_masks(image_rgb);
    
    // 3. 预计算投影 (Precompute Projection)
    auto t2 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 2: Inference took {:.3f} ms", std::chrono::duration<double, std::milli>(t2 - t1).count());

    auto proj_map = project_point_cloud_to_image(point_cloud, image_rgb.size());

    // 4. 主处理循环 (Main Processing Loop)
    auto t3 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 3: Precompute Projection took {:.3f} ms", std::chrono::duration<double, std::milli>(t3 - t2).count());
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
    #pragma omp parallel for
    for (int i = 0; i < (int)detected_masks.size(); ++i) {
        const auto &mask = detected_masks[i];
        if (auto det_2d = extract_rect_from_mask(mask)) {
            if (auto pose_res = solve_pose_for_single_object(*det_2d, proj_map, point_cloud)) {
                #pragma omp critical
                {
                    pose_res->id = idx_counter++;
                    results.push_back(std::move(*pose_res));
                }
            }
        }
    }

    // 5. 可视化结果 (Visualize Results)
    auto t4 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 4: Main Processing Loop took {:.3f} ms", std::chrono::duration<double, std::milli>(t4 - t3).count());

    visualize_results(vis_image, results);

    // 6. 结果输出 (Output)
    auto t5 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 5: Visualize Results took {:.3f} ms", std::chrono::duration<double, std::milli>(t5 - t4).count());

    const fs::path vis_out_path = output_dir / "vis_on_orig.jpg";
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

    auto t6 = std::chrono::steady_clock::now();
    spdlog::info("[Timing] Step 6: Output {:.3f} ms", std::chrono::duration<double, std::milli>(t6 - t5).count());

    spdlog::info("[ OK ] timestamp={} -> {}, targets={} (written {}), time={:.3f} ms, Mode={}, Device={}",
             timestamp, vis_out_path.string(), total_results, num_to_write, elapsed_ms, 
             (g_run_mode==0 ? "File" : "Camera"), (g_compute_device==1 ? "CPU" : "GPU"));

    return num_to_write;
    } catch (const Ort::Exception& e) {
        spdlog::critical("bs_yzx_object_detection_lanxin Ort exception: {}", e.what());
        return -91;
    } catch (const cv::Exception& e) {
        spdlog::critical("bs_yzx_object_detection_lanxin OpenCV exception: {}", e.what());
        return -92;
    } catch (const std::exception& e) {
        spdlog::critical("bs_yzx_object_detection_lanxin exception: {}", e.what());
        return -93;
    } catch (...) {
        spdlog::critical("bs_yzx_object_detection_lanxin unknown exception");
        return -94;
    }
}
