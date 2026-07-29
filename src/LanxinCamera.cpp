#include "LanxinCamera.h"
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {
    int g_capture_retry_timeout_ms = 3000;

    bool is_accepted_frame_state(const LX_STATE state) {
        return state == LX_SUCCESS ||
               state == LX_E_FRAME_ID_NOT_MATCH ||
               state == LX_E_FRAME_MULTI_MACHINE;
    }
}

static void checkTC(LX_STATE val) {
    if (val != LX_SUCCESS) {
        std::string message = std::string("LanxinCamera error: ") + DcGetErrorString(val);
        spdlog::error("{}", message);
        throw std::runtime_error(message);
    }
}

static long long elapsed_ms_since(const std::chrono::steady_clock::time_point& start_time) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
}

std::vector<std::string> LanxinCamera::DiscoverCameraIps() {
    LxDeviceInfo *device_list = nullptr;
    int device_num = 0;
    checkTC(DcGetDeviceList(&device_list, &device_num));

    std::vector<std::string> camera_ips;
    std::unordered_set<std::string> seen_ips;
    if (device_num > 0) {
        camera_ips.reserve(static_cast<size_t>(device_num));
    }

    for (int i = 0; i < device_num; ++i) {
        const std::string camera_ip = device_list[i].ip;
        if (!camera_ip.empty() && seen_ips.insert(camera_ip).second) {
            camera_ips.push_back(camera_ip);
        }
    }

    spdlog::info("Discovered {} LanxinCamera device(s)", camera_ips.size());
    return camera_ips;
}

void LanxinCamera::SetCaptureRetryTimeoutMs(const int timeoutMs) {
    g_capture_retry_timeout_ms = timeoutMs > 0 ? timeoutMs : 3000;
    spdlog::info("LanxinCamera capture retry timeout set to {}ms", g_capture_retry_timeout_ms);
}

int LanxinCamera::GetCaptureRetryTimeoutMs() {
    return g_capture_retry_timeout_ms;
}

int LanxinCamera::connect() {
    // 连接指定 IP 的蓝芯相机，并准备 RGB 与深度数据流。
    // 成功后缓存图像尺寸、数据类型和相机内参，为后续取图和点云转换提供基础参数。
    LxDeviceInfo device_info;
    const auto open_mode = static_cast<LX_OPEN_MODE>(OPEN_BY_IP);
    const char *open_param = camera_ip_.c_str();
    spdlog::info("Opening LanxinCamera by ip: {}", open_param);

    const LX_STATE lx_state = DcOpenDevice(open_mode, open_param, &handle, &device_info);
    if (LX_SUCCESS != lx_state) {
        spdlog::error("打开 LanxinCamera 设备失败, open_param={}, error={}", open_param, DcGetErrorString(lx_state));
        return -1;
    }

    bool stream_started = false;
    auto cleanup_open_handle = [&]() {
        if (handle != 0) {
            if (callback_registered_) {
                DcUnregisterFrameCallback(handle);
                callback_registered_ = false;
            }
            if (stream_started) {
                DcStopStream(handle);
            }
            DcCloseDevice(handle);
            handle = 0;
        }
        isConnect = false;
    };

    try {
        spdlog::info("device_info\n cameraid:{}\n uniqueid:{}\n cameraip:{}\n firmware_ver:{}\n sn:{}\n name:{}\n img_algor_ver:{}",
                     device_info.id, handle, device_info.ip, device_info.firmware_ver, device_info.sn,
                     device_info.name, device_info.algor_ver);

        // 只开启本算法需要的深度流和 RGB 流，减少无关数据传输。
        bool test_depth = true, test_amp = false, test_rgb = true;
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, test_depth));
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, test_amp));
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, test_rgb));

        // 显式关闭算法未使用的强度流，只保留深度和 RGB。
        checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, &test_depth));
        checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, &test_amp));
        checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, &test_rgb));
        spdlog::info("test_depth:{} test_amp:{} test_rgb:{}", test_depth, test_amp, test_rgb);

        // 打开深度到 RGB 坐标的对齐，使 RGB mask 与深度像素直接对应。
        checkTC(DcSetIntValue(handle, LX_INT_RGBD_ALIGN_MODE, DEPTH_TO_RGB));
        LxIntValueInfo align_mode;
        checkTC(DcGetIntValue(handle, LX_INT_RGBD_ALIGN_MODE, &align_mode));
        spdlog::info("RGBD align mode: {}", align_mode.cur_value);

        // 使用起流模式：回调会持续到来，CapFrame 只等待下一帧回调数据。
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_MODE, LX_TRIGGER_MODE_OFF));
        spdlog::info("Stream trigger mode configured: LX_TRIGGER_MODE_OFF");

        // 读取图像参数后，CapFrame 可以按正确尺寸和格式构造 RGB 与深度矩阵。
        LxIntValueInfo int_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_IMAGE_WIDTH, &int_value));
        this->tof_width = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_IMAGE_HEIGHT, &int_value));
        this->tof_height = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_DEPTH_DATA_TYPE, &int_value));
        this->tof_depth_type = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_2D_IMAGE_WIDTH, &int_value));
        this->rgb_width = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_2D_IMAGE_HEIGHT, &int_value));
        this->rgb_height = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_2D_IMAGE_CHANNEL, &int_value));
        this->rgb_channles = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_2D_IMAGE_DATA_TYPE, &int_value));
        this->rgb_data_type = int_value.cur_value;

        checkTC(DcRegisterFrameCallback(handle, &LanxinCamera::FrameCallback, this));
        callback_registered_ = true;
        spdlog::info("DcRegisterFrameCallback 完成");

        // 数据流启动后即可连续获取 RGB 图像和对齐深度图。
        checkTC(DcStartStream(handle));
        stream_started = true;
        spdlog::info("DcStartStream 完成");

        float *param_data = nullptr;
        if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_2D_INTRIC_PARAM, reinterpret_cast<void **>(&param_data))) {
            spdlog::error("获取相机内参失败");
            cleanup_open_handle();
            return -2;
        }

        param = cv::Mat::zeros(3, 3, CV_32FC1);
        param.at<float>(0) = *(param_data + 0);
        param.at<float>(2) = *(param_data + 2);
        param.at<float>(4) = *(param_data + 1);
        param.at<float>(5) = *(param_data + 3);
        param.at<float>(8) = 1;

        isConnect = true;
        return 0;
    } catch (...) {
        cleanup_open_handle();
        throw;
    }
}

void LanxinCamera::FrameCallback(FrameInfo* frame, void* usrData) {
    if (usrData == nullptr) return;
    static_cast<LanxinCamera*>(usrData)->HandleFrame(frame);
}

void LanxinCamera::HandleFrame(FrameInfo* frame) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    if (frame == nullptr) {
        spdlog::warn("[FrameCallback] ip={}, null frame", camera_ip_);
        return;
    }

    int depth_frame_id = -1;
    int rgb_frame_id = -1;
    if (frame->reserve_data != nullptr) {
        const auto* extend = static_cast<const FrameExtendInfo*>(frame->reserve_data);
        depth_frame_id = static_cast<int>(extend->depth_frame_id);
        rgb_frame_id = static_cast<int>(extend->rgb_frame_id);
    } else {
        spdlog::warn("[FrameCallback] ip={}, frame_state={}, no FrameExtendInfo",
                     camera_ip_, static_cast<int>(frame->frame_state));
    }

    last_depth_frame_id_ = depth_frame_id;
    last_rgb_frame_id_ = rgb_frame_id;

    if (!waiting_frame_) {
        return;
    }

    spdlog::info("[FrameCallback] ip={}, accepted callback frame_state={}, depth_frame_id={}, rgb_frame_id={}",
                 camera_ip_, static_cast<int>(frame->frame_state),
                 depth_frame_id, rgb_frame_id);

    async_frame_state_ = frame->frame_state;
    async_error_code_ = -1;

    if (!is_accepted_frame_state(frame->frame_state)) {
        async_error_code_ = static_cast<int>(frame->frame_state);
        spdlog::warn("[FrameCallback] ip={}, frame_state rejected, error={}",
                     camera_ip_, DcGetErrorString(frame->frame_state));
        return;
    }

    bool rgb_ok = false;
    cv::Mat rgb_copy;
    const auto& rgb_data = frame->rgb_data;
    if (rgb_data.frame_data != nullptr &&
        rgb_data.frame_width > 0 &&
        rgb_data.frame_height > 0 &&
        rgb_data.frame_channel > 0) {
        cv::Mat rgb_view(rgb_data.frame_height, rgb_data.frame_width,
                         CV_MAKETYPE(static_cast<int>(rgb_data.frame_data_type), rgb_data.frame_channel),
                         rgb_data.frame_data);
        if (!rgb_view.empty()) {
            rgb_copy = rgb_view.clone();
            rgb_ok = !rgb_copy.empty();
        }
    }

    bool depth_ok = false;
    cv::Mat depth_copy;
    const auto& depth_data = frame->depth_data;
    if (depth_data.frame_data != nullptr &&
        depth_data.frame_width > 0 &&
        depth_data.frame_height > 0) {
        const int depth_channels = std::max(depth_data.frame_channel, 1);
        const int depth_type = static_cast<int>(depth_data.frame_data_type);
        if (depth_channels == 1 && depth_type >= CV_8U && depth_type <= CV_64F) {
            cv::Mat depth_view(depth_data.frame_height, depth_data.frame_width,
                               CV_MAKETYPE(depth_type, depth_channels),
                               depth_data.frame_data);
            if (!depth_view.empty()) {
                depth_copy = depth_view.clone();
                depth_ok = !depth_copy.empty();
            }
        }
    }

    if (rgb_ok) {
        latest_rgb_ = std::move(rgb_copy);
        async_has_rgb_ = true;
    }
    if (depth_ok) {
        latest_depth_ = std::move(depth_copy);
        async_has_depth_ = true;
    }

    if (async_has_rgb_ && async_has_depth_) {
        async_has_frame_ = true;
        async_error_code_ = 0;
        waiting_frame_ = false;
        frame_cv_.notify_one();
    } else {
        if (!async_has_depth_) {
            async_error_code_ = -2;
        } else if (!async_has_rgb_) {
            async_error_code_ = -3;
        }
        spdlog::warn("[FrameCallback] ip={}, partial data, callback rgb_ok={}, depth_ok={}, accumulated rgb_ok={}, depth_ok={}, error_code={}, continue waiting",
                     camera_ip_, rgb_ok, depth_ok, async_has_rgb_, async_has_depth_, async_error_code_);
    }
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat) {
    cv::Mat depth;
    return CapFrame(rgbMat, depth);
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat, cv::Mat &depthMat) {
    spdlog::info("[CapFrame][FrameData] start ip={}, connected={}, handle={}",
                 camera_ip_, isConnect, handle);
    if (!isConnect) {
        spdlog::warn("[CapFrame][FrameData] ip={} is not connected, try reconnect", camera_ip_);
        if (const auto code = connect(); code != 0) {
            spdlog::error("[CapFrame][FrameData] reconnect failed, ip={}, code={}", camera_ip_, code);
            return -5;
        }
    }

    const auto start_time = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        waiting_frame_ = true;
        async_has_frame_ = false;
        async_has_rgb_ = false;
        async_has_depth_ = false;
        async_frame_state_ = LX_ERROR;
        async_error_code_ = -1;
        spdlog::info("[CapFrame][FrameData] waiting callback frame, latest depth={}, rgb={}",
                     last_depth_frame_id_, last_rgb_frame_id_);
    }

    std::unique_lock<std::mutex> lock(frame_mutex_);
    while (!async_has_frame_) {
        frame_cv_.wait_for(lock, std::chrono::seconds(1));
        if (!async_has_frame_) {
            const long long elapsed_ms = elapsed_ms_since(start_time);
            spdlog::info("[CapFrame][FrameData] still waiting callback frame, accumulated rgb_ok={}, depth_ok={}, latest depth={}, rgb={}, elapsed={}ms",
                         async_has_rgb_, async_has_depth_,
                         last_depth_frame_id_, last_rgb_frame_id_, elapsed_ms);
            if (elapsed_ms >= g_capture_retry_timeout_ms) {
                waiting_frame_ = false;
                const int error_code = !async_has_depth_ ? -2 : -3;
                spdlog::error("[CapFrame][FrameData] timeout ip={}, rgb_ok={}, depth_ok={}, elapsed={}ms, error_code={}",
                              camera_ip_, async_has_rgb_, async_has_depth_,
                              elapsed_ms, error_code);
                return error_code;
            }
        }
    }

    rgbMat = latest_rgb_.clone();
    depthMat = latest_depth_.clone();
    const LX_STATE frame_state = async_frame_state_;
    spdlog::info("[CapFrame][FrameData] success ip={}, elapsed={}ms, frame_state={}, rgb={}x{}, depth={}x{}, depth_type={}",
                 camera_ip_, elapsed_ms_since(start_time), static_cast<int>(frame_state),
                 rgbMat.cols, rgbMat.rows, depthMat.cols, depthMat.rows, depthMat.type());
    return 0;
}
