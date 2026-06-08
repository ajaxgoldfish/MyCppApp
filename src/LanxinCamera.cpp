#include "LanxinCamera.h"
#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <thread>
#include <unordered_set>

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

        // 打开 RGB 到深度坐标的对齐，使 2D mask 能和点云建立空间对应关系。
        checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_TO_DEPTH, true));

        // 使用软触发模式：每次 CapFrame 主动触发一次曝光，再读取对应的一帧数据。
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_MODE, LX_TRIGGER_SOFTWARE));
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_DELAY_TIME, 0));
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_MIN_PERIOD_TIME, 100000));
        checkTC(DcSetIntValue(handle, LX_INT_TRIGGER_FRAME_COUNT, 1));
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
    const auto deadline = start_time + std::chrono::seconds(3);
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
    spdlog::error("获取点云数据失败，3秒内重试仍未成功");
    spdlog::error("[CapFrame][PointCloud] failed ip={}, attempts={}, elapsed={}ms, last_error={}",
                  camera_ip_, attempt, elapsed_ms_since(start_time), last_error);
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
    const auto deadline = start_time + std::chrono::seconds(3);
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
    spdlog::error("获取 RGB 数据失败，3秒内重试仍未成功");
    spdlog::error("[CapFrame][RGB] failed ip={}, attempts={}, elapsed={}ms, last_error={}",
                  camera_ip_, attempt, elapsed_ms_since(start_time), last_error);
    return last_error;
}
