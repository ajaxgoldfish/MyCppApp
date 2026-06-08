#ifndef LANXINCAMERA_H
#define LANXINCAMERA_H
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <utility>
#include <vector>
#include "lx_camera_api.h"

class LanxinCamera final {
public:
    // 枚举当前 SDK 能发现的全部相机 IP，供初始化阶段一次性建立连接池。
    static std::vector<std::string> DiscoverCameraIps();

    // 构造时立即按 IP 建立连接，使调用方只需关注取 RGB 或点云数据。
    explicit LanxinCamera(std::string cameraIp) : camera_ip_(std::move(cameraIp)) {
        connect();
    }

    int CapFrame(pcl::PointCloud<pcl::PointXYZ> &pc);

    bool isOpened() const {
        return isConnect;
    }

    [[nodiscard]] const std::string& getCameraIp() const {
        return camera_ip_;
    }

    int CapFrame(cv::Mat &rgbMat);

    int CapFrame(cv::Mat &rgbMat, pcl::PointCloud<pcl::PointXYZ> &pc);

    ~LanxinCamera() {
        if (isConnect) {
            DcStopStream(handle);
            DcCloseDevice(handle);
        }
    }

    [[nodiscard]] cv::Mat get_param() const {
        return param;
    }

private:
    // 保存 SDK 句柄和图像参数，后续每次取帧都复用这些连接状态。
    int connect();

    DcHandle handle = 0;
    std::string camera_ip_;
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


#endif // 头文件保护
