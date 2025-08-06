//
// Created by zwj on 2025/1/13.
//

#include "LanxinCamera.h"
#include <iostream>
#include <spdlog/spdlog.h>

static void checkTC(LX_STATE val) {
    if (val != LX_SUCCESS) {
        std::string string = std::string("LanxinCamera error :") + DcGetErrorString(val);
        throw std::exception(string.data());
    }
}

int LanxinCamera::connect() {
    //查找相机
    int device_num = 0;
    LxDeviceInfo *p_device_list = nullptr;
    checkTC(DcGetDeviceList(&p_device_list, &device_num));
    if (device_num <= 0) {
        std::cout << "not found any device" << std::endl;
        return -1;
    }

    LxDeviceInfo device_info;
    int open_mode = OPEN_BY_INDEX;
    const LX_STATE lx_state = DcOpenDevice(static_cast<LX_OPEN_MODE>(open_mode), "0", &handle, &device_info);
    if (LX_SUCCESS != lx_state) {
        std::cout << "open LanxinCamera device failed" << std::endl;
        return -1;
    }

    std::cout << "device_info\n cameraid:" << device_info.id << "\n uniqueid:" << handle
            << "\n cameraip:" << device_info.ip << "\n firmware_ver:" << device_info.firmware_ver << "\n sn:" <<
            device_info.sn
            << "\n name:" << device_info.name << "\n img_algor_ver:" << device_info.algor_ver << std::endl;

    //设置数据流
    bool test_depth = true, test_amp = true, test_rgb = true;
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, test_depth));
    //checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, test_amp));
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, test_rgb));

    checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_DEPTH_STREAM, &test_depth));
    //checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_3D_AMP_STREAM, &test_amp));
    checkTC(DcGetBoolValue(handle, LX_BOOL_ENABLE_2D_STREAM, &test_rgb));
    std::cout << "test_depth:" << test_depth << " test_amp:" << test_amp << " test_rgb:" << test_rgb << std::endl;

    //RGBD对齐，TOF的图像尺寸和像素会扩展到与RGB一致，开启后建议关闭强度流
    checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_2D_TO_DEPTH, true));

    //获取图像参数，设置ROI BINNING RGBD对齐之后需要重新获取图像尺寸
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

    //可以根据需要,是否开启帧同步模式, 开启该模式, 内部会对每一帧做同步处理后返回
    //默认若不需要tof与rgb数据同步, 则不需要开启此功能, 内部会优先保证数据实时性
    //checkTC(DcSetBoolValue(handle, LX_BOOL_ENABLE_SYNC_FRAME, true));
    //开启数据流
    checkTC(DcStartStream(handle));
    spdlog::info("DcStartStream finish");

    float *param_data = nullptr;
    if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_2D_INTRIC_PARAM, reinterpret_cast<void **>(&param_data))) {
        return -2;
    }

    param = cv::Mat::zeros(3, 3,CV_32FC1);
    param.at<float>(0) = *(param_data + 0);
    //param.at<float>(1) = 0;
    param.at<float>(2) = *(param_data + 2);
    //param.at<float>(3) = 0;
    param.at<float>(4) = *(param_data + 1);
    param.at<float>(5) = *(param_data + 3);
    //param.at<float>(6) = 0;
    //param.at<float>(7) = 0;
    param.at<float>(8) = 1;
    std::cout << "相机内参：" << param << std::endl;
    isConnect = true;
    return 0;
}

int LanxinCamera::CapFrame(pcl::PointCloud<pcl::PointXYZ> &pc) {
    if (!isConnect) {
        if (const auto code = connect(); code != 0) {
            return -5;
        }
    }
    //更新数据
    const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
    if ((LX_SUCCESS != ret)
        && (LX_E_FRAME_ID_NOT_MATCH != ret)
        && (LX_E_FRAME_MULTI_MACHINE != ret)) {
        if (LX_E_RECONNECTING == ret) {
            std::cout << "device reconnecting" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return -1;
    }


    //depth
    {
        //第yRows行xCol列点云数据
        float *xyz_data = nullptr;
        if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_XYZ_DATA, reinterpret_cast<void **>(&xyz_data))) {
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
            pcl::PointXYZ point;
            point.x = x / 1000.0f;
            point.y = y / 1000.0f;
            point.z = z / 1000.0f;
            pc.points.push_back(point);
        }
    }
    return 0;
}

int LanxinCamera::CapFrame(cv::Mat &rgbMat) {
    if (!isConnect) {
        if (const auto code = connect(); code != 0) {
            return -5;
        }
    }
    //更新数据
    const auto ret = DcSetCmd(handle, LX_CMD_GET_NEW_FRAME);
    if ((LX_SUCCESS != ret)
        && (LX_E_FRAME_ID_NOT_MATCH != ret)
        && (LX_E_FRAME_MULTI_MACHINE != ret)) {
        if (LX_E_RECONNECTING == ret) {
            std::cout << "device reconnecting" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return -1;
    }

    //rgb
    {
        unsigned char *data_ptr = nullptr;
        if (LX_SUCCESS != DcGetPtrValue(handle, LX_PTR_2D_IMAGE_DATA, reinterpret_cast<void **>(&data_ptr))) {
            return -3;
        }
        rgbMat = cv::Mat(rgb_height, rgb_width, CV_MAKETYPE(rgb_data_type, rgb_channles), data_ptr);
    }
    return 0;
}
