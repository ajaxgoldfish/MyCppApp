#include "LanxinCamera.h"
#include <spdlog/spdlog.h>

static void checkTC(LX_STATE val) {
    if (val != LX_SUCCESS) {
        std::string string = std::string("LanxinCamera error :") + DcGetErrorString(val);
        spdlog::error(string);
        throw std::runtime_error(string);
    }
}

int LanxinCamera::connect() {
    // 查找相机
    int device_num = 0;
    LxDeviceInfo *p_device_list = nullptr;
    checkTC(DcGetDeviceList(&p_device_list, &device_num));
    if (device_num <= 0) {
        spdlog::warn("未发现任何设备");
        return -1;
    }

    LxDeviceInfo device_info;
    int open_mode = OPEN_BY_INDEX;
    const LX_STATE lx_state = DcOpenDevice(static_cast<LX_OPEN_MODE>(open_mode), "0", &handle, &device_info);
    if (LX_SUCCESS != lx_state) {
        spdlog::error("打开 LanxinCamera 设备失败");
        return -1;
    }

    spdlog::info("device_info\n cameraid:{}\n uniqueid:{}\n cameraip:{}\n firmware_ver:{}\n sn:{}\n name:{}\n img_algor_ver:{}",
                 device_info.id, handle, device_info.ip, device_info.firmware_ver, device_info.sn,
                 device_info.name, device_info.algor_ver);

    // 设置数据流
    bool test_depth = true, test_amp = true, test_rgb = true;
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, test_depth));
    //checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, test_amp));
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, test_rgb));

    checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, &test_depth));
    //checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, &test_amp));
    checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, &test_rgb));
    spdlog::info("test_depth:{} test_amp:{} test_rgb:{}", test_depth, test_amp, test_rgb);

    // RGBD 对齐
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_TO_DEPTH, true));

    // 获取图像参数
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

    // 开启数据流
    checkTC(DcStartStream(handle));
    spdlog::info("DcStartStream 完成");

    float *param_data = nullptr;
    if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_2D_INTRIC_PARAM, reinterpret_cast<void **>(&param_data))) {
        spdlog::error("获取相机内参失败");
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
}

int LanxinCamera::CapFrame(open3d::geometry::PointCloud &pc) {
    if (!isConnect) {
        if (const auto code = connect(); code != 0) {
            return -5;
        }
    }
    const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
    if ((LX_SUCCESS != ret) && (LX_E_FRAME_ID_NOT_MATCH != ret) && (LX_E_FRAME_MULTI_MACHINE != ret)) {
        if (LX_E_RECONNECTING == ret) {
            spdlog::warn("设备正在重连中");
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return -1;
    }

    // depth
    float *xyz_data = nullptr;
    if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_XYZ_DATA, reinterpret_cast<void **>(&xyz_data))) {
        spdlog::error("获取点云数据失败");
        return -2;
    }

    const int total = tof_width * tof_height;
    for (int i = 0; i < total; ++i) {
        float x = xyz_data[i * 3];
        float y = xyz_data[i * 3 + 1];
        float z = xyz_data[i * 3 + 2];
        if (x == 0 && y == 0 && z == 0) {
            continue;
        }
        pc.points_.emplace_back(x / 1000, y / 1000, z / 1000);
    }
    return 0;
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat) {
    if (!isConnect) {
        if (const auto code = connect(); code != 0) {
            return -5;
        }
    }
    const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
    if ((LX_SUCCESS != ret) && (LX_E_FRAME_ID_NOT_MATCH != ret) && (LX_E_FRAME_MULTI_MACHINE != ret)) {
        if (LX_E_RECONNECTING == ret) {
            spdlog::warn("设备正在重连中");
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return -1;
    }

    // rgb
    unsigned char *data_ptr = nullptr;
    if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_2D_IMAGE_DATA, reinterpret_cast<void **>(&data_ptr))) {
        spdlog::error("获取 RGB 数据失败");
        return -3;
    }
    rgbMat = cv::Mat(rgb_height, rgb_width, CV_MAKETYPE(rgb_data_type, rgb_channles), data_ptr);
    return 0;
}
