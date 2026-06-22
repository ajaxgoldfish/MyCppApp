#include "LanxinCamera.h"
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
        bool test_depth = true, test_amp = true, test_rgb = true;
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, test_depth));
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, test_rgb));

        // 幅值流当前不参与检测，保留状态变量便于后续需要时恢复调试。
        checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, &test_depth));
        checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, &test_rgb));
        spdlog::info("test_depth:{} test_amp:{} test_rgb:{}", test_depth, test_amp, test_rgb);

        // 打开深度到 RGB 坐标的对齐，使 RGB mask 能和点云建立空间对应关系。
        checkTC(DcSetIntValue(handle, LX_INT_RGBD_ALIGN_MODE, DEPTH_TO_RGB));
        LxIntValueInfo align_mode;
        checkTC(DcGetIntValue(handle, LX_INT_RGBD_ALIGN_MODE, &align_mode));
        spdlog::info("RGBD align mode: {}", align_mode.cur_value);

        // 使用软触发模式：每次 CapFrame 主动触发一次曝光，再读取对应的一帧数据。
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_MODE, LX_TRIGGER_SOFTWARE));

        spdlog::info("Software trigger configured: delay=0us, min_period=100000us, frame_count=1");

        // 读取图像参数后，CapFrame 可以按正确尺寸和格式构造 OpenCV/PCL 数据。
        LxIntValueInfo int_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_IMAGE_WIDTH, &int_value));
        this->tof_width = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_IMAGE_HEIGHT, &int_value));
        this->tof_height = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_DEPTH_DATA_TYPE, &int_value));
        this->tof_depth_type = int_value.cur_value;
        checkTC(DcGetIntValue(handle, LX_INT_3D_AMPLITUDE_DATA_TYPE, &int_value));
        this->tof_amp_type = int_value.cur_value;
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

        // 数据流启动后即可连续获取 RGB 图像和 XYZ 点云。
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
        if (waiting_frame_) {
            async_frame_arrived_ = true;
            async_has_frame_ = false;
            async_frame_state_ = LX_ERROR;
            async_error_code_ = -1;
            waiting_frame_ = false;
            frame_cv_.notify_one();
        }
        return;
    }

    unsigned int depth_frame_id = 0;
    unsigned int rgb_frame_id = 0;
    if (frame->reserve_data != nullptr) {
        const auto* extend = static_cast<const FrameExtendInfo*>(frame->reserve_data);
        depth_frame_id = extend->depth_frame_id;
        rgb_frame_id = extend->rgb_frame_id;
        spdlog::info("[FrameCallback] ip={}, frame_state={}, depth_frame_id={}, rgb_frame_id={}",
                     camera_ip_, static_cast<int>(frame->frame_state),
                     depth_frame_id, rgb_frame_id);
    } else {
        spdlog::warn("[FrameCallback] ip={}, frame_state={}, no FrameExtendInfo, ignore frame",
                     camera_ip_, static_cast<int>(frame->frame_state));
        return;
    }

    last_depth_frame_id_ = depth_frame_id;
    last_rgb_frame_id_ = rgb_frame_id;

    const bool is_new_frame =
        depth_frame_id != wait_depth_frame_id_ &&
        rgb_frame_id != wait_rgb_frame_id_;

    if (!waiting_frame_ || !is_new_frame) {
        return;
    }

    async_frame_arrived_ = true;
    async_has_frame_ = false;
    async_frame_state_ = frame->frame_state;
    async_error_code_ = -1;

    if (!is_accepted_frame_state(frame->frame_state)) {
        waiting_frame_ = false;
        frame_cv_.notify_one();
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

    bool pc_ok = false;
    pcl::PointCloud<pcl::PointXYZ> cloud_copy;
    const auto& depth_data = frame->depth_data;
    if (depth_data.frame_data != nullptr &&
        depth_data.frame_width > 0 &&
        depth_data.frame_height > 0) {
        float* xyz_data = nullptr;
        const auto xyz_ret = DcGetPtrValue(frame->handle, LX_PTR_XYZ_DATA, reinterpret_cast<void**>(&xyz_data));
        spdlog::info("[FrameCallback] ip={}, DcGetPtrValue(LX_PTR_XYZ_DATA) ret={}, error={}, ptr={}",
                     camera_ip_, static_cast<int>(xyz_ret), DcGetErrorString(xyz_ret),
                     static_cast<const void*>(xyz_data));
        if (LX_SUCCESS == xyz_ret && xyz_data != nullptr) {
            const int total = depth_data.frame_width * depth_data.frame_height;
            cloud_copy.points.reserve(total);
            for (int i = 0; i < total; ++i) {
                const float x = xyz_data[i * 3];
                const float y = xyz_data[i * 3 + 1];
                const float z = xyz_data[i * 3 + 2];
                if (x == 0 && y == 0 && z == 0) {
                    continue;
                }
                cloud_copy.points.emplace_back(x / 1000, y / 1000, z / 1000);
            }
            cloud_copy.width = cloud_copy.points.size();
            cloud_copy.height = 1;
            cloud_copy.is_dense = false;
            pc_ok = !cloud_copy.empty();
            spdlog::info("[FrameCallback] ip={}, converted point cloud, total_pixels={}, valid_points={}",
                         camera_ip_, total, cloud_copy.points.size());
        }
    }

    if (rgb_ok && pc_ok) {
        latest_rgb_ = std::move(rgb_copy);
        latest_cloud_ = std::move(cloud_copy);
        async_has_frame_ = true;
        async_error_code_ = 0;
    } else {
        async_error_code_ = rgb_ok ? -2 : -3;
        spdlog::warn("[FrameCallback] ip={}, incomplete data, rgb_ok={}, pc_ok={}, error_code={}",
                     camera_ip_, rgb_ok, pc_ok, async_error_code_);
    }

    waiting_frame_ = false;
    frame_cv_.notify_one();
}

int LanxinCamera::CapFrame(pcl::PointCloud<pcl::PointXYZ> &pc) {
    spdlog::info("[CapFrame][PointCloud] start ip={}, connected={}, handle={}, cached_tof={}x{}",
                 camera_ip_, isConnect, handle, tof_width, tof_height);
    if (!isConnect) {
        spdlog::warn("[CapFrame][PointCloud] ip={} is not connected, try reconnect", camera_ip_);
        if (const auto code = connect(); code != 0) {
            spdlog::error("[CapFrame][PointCloud] reconnect failed, ip={}, code={}", camera_ip_, code);
            return -5;
        }
    }

    const auto start_time = std::chrono::steady_clock::now();
    const int timeout_ms = GetCaptureRetryTimeoutMs();
    const auto deadline = start_time + std::chrono::milliseconds(timeout_ms);
    int last_error = -1;
    int attempt = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        ++attempt;
        spdlog::info("[CapFrame][PointCloud] attempt={}, elapsed={}ms, call LX_CMD_SOFTWARE_TRIGGER",
                     attempt, elapsed_ms_since(start_time));
        const auto trigger_ret = DcSetCmd(handle, LX_CMD_SOFTWARE_TRIGGER);
        spdlog::info("[CapFrame][PointCloud] attempt={}, LX_CMD_SOFTWARE_TRIGGER ret={}, error={}",
                     attempt, static_cast<int>(trigger_ret), DcGetErrorString(trigger_ret));
        if (LX_SUCCESS != trigger_ret) {
            if (LX_E_RECONNECTING == trigger_ret) {
                spdlog::warn("设备正在重连中");
            }
            spdlog::warn("[CapFrame][PointCloud] attempt={}, software trigger failed, ret={}, elapsed={}ms",
                         attempt, static_cast<int>(trigger_ret), elapsed_ms_since(start_time));
            last_error = -1;
            spdlog::info("[CapFrame][PointCloud] attempt={}, sleep 200ms before retry, elapsed={}ms",
                         attempt, elapsed_ms_since(start_time));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        spdlog::info("[CapFrame][PointCloud] attempt={}, elapsed={}ms, call LX_CMD_GET_NEW_FRAME",
                     attempt, elapsed_ms_since(start_time));
        const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
        spdlog::info("[CapFrame][PointCloud] attempt={}, LX_CMD_GET_NEW_FRAME ret={}, error={}",
                     attempt, static_cast<int>(ret), DcGetErrorString(ret));
        if (LX_SUCCESS != ret) {
            spdlog::warn("LX_CMD_GET_NEW_FRAME returned {}, error={}",
                         static_cast<int>(ret), DcGetErrorString(ret));
        }
        if ((LX_SUCCESS != ret) && (LX_E_FRAME_ID_NOT_MATCH != ret) && (LX_E_FRAME_MULTI_MACHINE != ret)) {
            if (LX_E_RECONNECTING == ret) {
                spdlog::warn("设备正在重连中");
            }
            spdlog::warn("[CapFrame][PointCloud] attempt={}, frame status rejected, ret={}, elapsed={}ms",
                         attempt, static_cast<int>(ret), elapsed_ms_since(start_time));
            last_error = -1;
        } else {
            spdlog::info("[CapFrame][PointCloud] attempt={}, frame status accepted, ret={}, read LX_PTR_XYZ_DATA",
                         attempt, static_cast<int>(ret));
            // 读取 SDK 输出的 XYZ 深度数据，并转换成以米为单位的 PCL 点云。
            float *xyz_data = nullptr;
            const auto get_ptr_ret =
                DcGetPtrValue(handle, LX_PTR_XYZ_DATA, reinterpret_cast<void **>(&xyz_data));
            spdlog::info("[CapFrame][PointCloud] attempt={}, DcGetPtrValue(LX_PTR_XYZ_DATA) ret={}, error={}, ptr={}",
                         attempt, static_cast<int>(get_ptr_ret), DcGetErrorString(get_ptr_ret),
                         static_cast<const void *>(xyz_data));
            if (LX_SUCCESS == get_ptr_ret && xyz_data != nullptr) {
                pc.clear();
                const int total = tof_width * tof_height;
                pc.points.reserve(total);
                for (int i = 0; i < total; ++i) {
                    float x = xyz_data[i * 3];
                    float y = xyz_data[i * 3 + 1];
                    float z = xyz_data[i * 3 + 2];
                    if (x == 0 && y == 0 && z == 0) {
                        continue;
                    }
                    pc.points.emplace_back(x / 1000, y / 1000, z / 1000);
                }
                pc.width = pc.points.size();
                pc.height = 1;
                pc.is_dense = false;
                spdlog::info("[CapFrame][PointCloud] attempt={}, converted point cloud, total_pixels={}, valid_points={}",
                             attempt, total, pc.points.size());
                if (!pc.empty()) {
                    spdlog::info("[CapFrame][PointCloud] success ip={}, attempt={}, elapsed={}ms, valid_points={}",
                                 camera_ip_, attempt, elapsed_ms_since(start_time), pc.points.size());
                    return 0;
                }
                spdlog::warn("[CapFrame][PointCloud] attempt={}, XYZ pointer is valid but point cloud is empty",
                             attempt);
            } else {
                spdlog::warn("DcGetPtrValue(LX_PTR_XYZ_DATA) returned {}, error={}",
                             static_cast<int>(get_ptr_ret), DcGetErrorString(get_ptr_ret));
                spdlog::warn("[CapFrame][PointCloud] attempt={}, failed to get XYZ pointer, ret={}, ptr={}",
                             attempt, static_cast<int>(get_ptr_ret), static_cast<const void *>(xyz_data));
            }
            last_error = -2;
        }

        spdlog::info("[CapFrame][PointCloud] attempt={}, sleep 200ms before retry, elapsed={}ms",
                     attempt, elapsed_ms_since(start_time));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    spdlog::error("获取点云数据失败，{}ms内重试仍未成功", timeout_ms);
    spdlog::error("[CapFrame][PointCloud] failed ip={}, attempts={}, elapsed={}ms, timeout={}ms, last_error={}",
                  camera_ip_, attempt, elapsed_ms_since(start_time), timeout_ms, last_error);
    return last_error;
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat) {
    spdlog::info("[CapFrame][RGB] start ip={}, connected={}, handle={}, cached_rgb={}x{}, channels={}, type={}",
                 camera_ip_, isConnect, handle, rgb_width, rgb_height, rgb_channles, rgb_data_type);
    if (!isConnect) {
        spdlog::warn("[CapFrame][RGB] ip={} is not connected, try reconnect", camera_ip_);
        if (const auto code = connect(); code != 0) {
            spdlog::error("[CapFrame][RGB] reconnect failed, ip={}, code={}", camera_ip_, code);
            return -5;
        }
    }

    const auto start_time = std::chrono::steady_clock::now();
    const int timeout_ms = GetCaptureRetryTimeoutMs();
    const auto deadline = start_time + std::chrono::milliseconds(timeout_ms);
    int last_error = -1;
    int attempt = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        ++attempt;
        spdlog::info("[CapFrame][RGB] attempt={}, elapsed={}ms, call LX_CMD_SOFTWARE_TRIGGER",
                     attempt, elapsed_ms_since(start_time));
        const auto trigger_ret = DcSetCmd(handle, LX_CMD_SOFTWARE_TRIGGER);
        spdlog::info("[CapFrame][RGB] attempt={}, LX_CMD_SOFTWARE_TRIGGER ret={}, error={}",
                     attempt, static_cast<int>(trigger_ret), DcGetErrorString(trigger_ret));
        if (LX_SUCCESS != trigger_ret) {
            if (LX_E_RECONNECTING == trigger_ret) {
                spdlog::warn("设备正在重连中");
            }
            spdlog::warn("[CapFrame][RGB] attempt={}, software trigger failed, ret={}, elapsed={}ms",
                         attempt, static_cast<int>(trigger_ret), elapsed_ms_since(start_time));
            last_error = -1;
            spdlog::info("[CapFrame][RGB] attempt={}, sleep 200ms before retry, elapsed={}ms",
                         attempt, elapsed_ms_since(start_time));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        spdlog::info("[CapFrame][RGB] attempt={}, elapsed={}ms, call LX_CMD_GET_NEW_FRAME",
                     attempt, elapsed_ms_since(start_time));
        const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
        spdlog::info("[CapFrame][RGB] attempt={}, LX_CMD_GET_NEW_FRAME ret={}, error={}",
                     attempt, static_cast<int>(ret), DcGetErrorString(ret));
        if (LX_SUCCESS != ret) {
            spdlog::warn("LX_CMD_GET_NEW_FRAME returned {}, error={}",
                         static_cast<int>(ret), DcGetErrorString(ret));
        }
        if ((LX_SUCCESS != ret) && (LX_E_FRAME_ID_NOT_MATCH != ret) && (LX_E_FRAME_MULTI_MACHINE != ret)) {
            if (LX_E_RECONNECTING == ret) {
                spdlog::warn("设备正在重连中");
            }
            spdlog::warn("[CapFrame][RGB] attempt={}, frame status rejected, ret={}, elapsed={}ms",
                         attempt, static_cast<int>(ret), elapsed_ms_since(start_time));
            last_error = -1;
        } else {
            spdlog::info("[CapFrame][RGB] attempt={}, frame status accepted, ret={}, read LX_PTR_2D_IMAGE_DATA",
                         attempt, static_cast<int>(ret));
            // 读取 SDK 当前 RGB 缓冲区，并封装为 OpenCV Mat 供检测模型使用。
            unsigned char *data_ptr = nullptr;
            const auto get_ptr_ret =
                DcGetPtrValue(handle, LX_PTR_2D_IMAGE_DATA, reinterpret_cast<void **>(&data_ptr));
            spdlog::info("[CapFrame][RGB] attempt={}, DcGetPtrValue(LX_PTR_2D_IMAGE_DATA) ret={}, error={}, ptr={}",
                         attempt, static_cast<int>(get_ptr_ret), DcGetErrorString(get_ptr_ret),
                         static_cast<const void *>(data_ptr));
            if (LX_SUCCESS == get_ptr_ret && data_ptr != nullptr) {
                rgbMat = cv::Mat(rgb_height, rgb_width, CV_MAKETYPE(rgb_data_type, rgb_channles), data_ptr);
                spdlog::info("[CapFrame][RGB] attempt={}, wrapped cv::Mat rows={}, cols={}, channels={}, empty={}",
                             attempt, rgbMat.rows, rgbMat.cols, rgbMat.channels(), rgbMat.empty());
                if (!rgbMat.empty()) {
                    spdlog::info("[CapFrame][RGB] success ip={}, attempt={}, elapsed={}ms, size={}x{}, channels={}",
                                 camera_ip_, attempt, elapsed_ms_since(start_time),
                                 rgbMat.cols, rgbMat.rows, rgbMat.channels());
                    return 0;
                }
                spdlog::warn("[CapFrame][RGB] attempt={}, RGB pointer is valid but cv::Mat is empty",
                             attempt);
            } else {
                spdlog::warn("DcGetPtrValue(LX_PTR_2D_IMAGE_DATA) returned {}, error={}",
                             static_cast<int>(get_ptr_ret), DcGetErrorString(get_ptr_ret));
                spdlog::warn("[CapFrame][RGB] attempt={}, failed to get RGB pointer, ret={}, ptr={}",
                             attempt, static_cast<int>(get_ptr_ret), static_cast<const void *>(data_ptr));
            }
            last_error = -3;
        }

        spdlog::info("[CapFrame][RGB] attempt={}, sleep 200ms before retry, elapsed={}ms",
                     attempt, elapsed_ms_since(start_time));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    spdlog::error("获取 RGB 数据失败，{}ms内重试仍未成功", timeout_ms);
    spdlog::error("[CapFrame][RGB] failed ip={}, attempts={}, elapsed={}ms, timeout={}ms, last_error={}",
                  camera_ip_, attempt, elapsed_ms_since(start_time), timeout_ms, last_error);
    return last_error;
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat, pcl::PointCloud<pcl::PointXYZ> &pc) {
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
    const int timeout_ms = GetCaptureRetryTimeoutMs();
    const auto deadline = start_time + std::chrono::milliseconds(timeout_ms);
    int last_error = -1;
    int attempt = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        ++attempt;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            waiting_frame_ = true;
            async_frame_arrived_ = false;
            async_has_frame_ = false;
            async_frame_state_ = LX_ERROR;
            async_error_code_ = -1;
            wait_depth_frame_id_ = last_depth_frame_id_;
            wait_rgb_frame_id_ = last_rgb_frame_id_;
        }

        spdlog::info("[CapFrame][FrameData] attempt={}, elapsed={}ms, call LX_CMD_SOFTWARE_TRIGGER",
                     attempt, elapsed_ms_since(start_time));
        const auto trigger_ret = DcSetCmd(handle, LX_CMD_SOFTWARE_TRIGGER);
        spdlog::info("[CapFrame][FrameData] attempt={}, LX_CMD_SOFTWARE_TRIGGER ret={}, error={}",
                     attempt, static_cast<int>(trigger_ret), DcGetErrorString(trigger_ret));
        if (LX_SUCCESS != trigger_ret) {
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                waiting_frame_ = false;
            }
            if (LX_E_RECONNECTING == trigger_ret) {
                spdlog::warn("设备正在重连中");
            }
            last_error = -1;
            spdlog::info("[CapFrame][FrameData] attempt={}, sleep 200ms before retry, elapsed={}ms",
                         attempt, elapsed_ms_since(start_time));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        spdlog::info("[CapFrame][FrameData] attempt={}, waiting for frame callback, elapsed={}ms",
                     attempt, elapsed_ms_since(start_time));
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait(lock, [&] {
            return async_frame_arrived_;
        });

        const LX_STATE frame_state = async_frame_state_;
        const int error_code = async_error_code_;
        if (async_has_frame_) {
            rgbMat = latest_rgb_.clone();
            pc = latest_cloud_;
            spdlog::info("[CapFrame][FrameData] success ip={}, attempt={}, elapsed={}ms, frame_state={}, rgb={}x{}, valid_points={}",
                         camera_ip_, attempt, elapsed_ms_since(start_time), static_cast<int>(frame_state),
                         rgbMat.cols, rgbMat.rows, pc.points.size());
            return 0;
        }
        lock.unlock();

        last_error = error_code;
        spdlog::warn("[CapFrame][FrameData] attempt={}, callback frame failed, frame_state={}, error={}, last_error={}",
                     attempt, static_cast<int>(frame_state), DcGetErrorString(frame_state), last_error);
        spdlog::info("[CapFrame][FrameData] attempt={}, sleep 200ms before retry, elapsed={}ms",
                     attempt, elapsed_ms_since(start_time));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    spdlog::error("获取 FrameData RGB/点云数据失败，{}ms内重试仍未成功", timeout_ms);
    spdlog::error("[CapFrame][FrameData] failed ip={}, attempts={}, elapsed={}ms, timeout={}ms, last_error={}",
                  camera_ip_, attempt, elapsed_ms_since(start_time), timeout_ms, last_error);
    return last_error;
}
