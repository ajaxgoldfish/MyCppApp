#ifndef LANXINCAMERA_H
#define LANXINCAMERA_H
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
#include "lx_camera_api.h"

class LanxinCamera final {
public:
    // 枚举当前 SDK 能发现的全部相机 IP，供初始化阶段一次性建立连接池。
    static std::vector<std::string> DiscoverCameraIps();
    static void SetCaptureRetryTimeoutMs(int timeoutMs);
    static int GetCaptureRetryTimeoutMs();

    // 构造时立即按 IP 建立连接，使调用方只需关注获取 RGB 与对齐深度。
    explicit LanxinCamera(std::string cameraIp) : camera_ip_(std::move(cameraIp)) {
        connect();
    }

    bool isOpened() const {
        return isConnect;
    }

    [[nodiscard]] const std::string& getCameraIp() const {
        return camera_ip_;
    }

    int CapFrame(cv::Mat &rgbMat);

    int CapFrame(cv::Mat &rgbMat, cv::Mat &depthMat);

    ~LanxinCamera() {
        if (isConnect) {
            if (callback_registered_) {
                DcUnregisterFrameCallback(handle);
            }
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
    static void FrameCallback(FrameInfo* frame, void* usrData);
    void HandleFrame(FrameInfo* frame);

    DcHandle handle = 0;
    std::string camera_ip_;
    int rgb_data_type = 0;
    int rgb_channles = 0;
    int rgb_height = 0;
    int tof_width = 0;
    int tof_height = 0;
    int tof_depth_type = 0;
    int rgb_width = 0;
    cv::Mat param;
    bool isConnect = false;
    bool callback_registered_ = false;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    bool waiting_frame_ = false;
    bool async_has_frame_ = false;
    bool async_has_rgb_ = false;
    bool async_has_depth_ = false;
    LX_STATE async_frame_state_ = LX_ERROR;
    int async_error_code_ = -1;
    int last_depth_frame_id_ = -1;
    int last_rgb_frame_id_ = -1;
    cv::Mat latest_rgb_;
    cv::Mat latest_depth_;
};


#endif // 头文件保护
