//
// Created by zwj on 2025/1/13.
//

#ifndef LANXINCAMERA_H
#define LANXINCAMERA_H
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "lx_camera_api.h"

class LanxinCamera final {
public:
    LanxinCamera() {
        connect();
    }
    int CapFrame(pcl::PointCloud<pcl::PointXYZ> &pc);

    bool isOpened() const {
        return isConnect;
    }

    int CapFrame(cv::Mat &rgbMat);

    ~LanxinCamera() {
        DcStopStream(handle);
        DcCloseDevice(handle);
    }

    [[nodiscard]] cv::Mat get_param() const {
        return param;
    }

private:
    int connect();

    LxDeviceInfo *p_device_list = nullptr;
    DcHandle handle = 0;
    int rgb_data_type = 0;
    int rgb_channles = 0;
    int rgb_height = 0;
    int tof_width = 0;
    int tof_height = 0;
    int tof_depth_type = 0;
    int tof_amp_type = 0;
    int rgb_width = 0;
    cv::Mat param;
    bool isConnect = false;
};


#endif //LANXINCAMERA_H
