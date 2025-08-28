#ifndef MY_VISION_LIBRARY_H
#define MY_VISION_LIBRARY_H

#include <chrono>
#include <open3d/Open3D.h>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "BoxPosePipeline.h"
#include "BoxPosePipeline.h"
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include "LanxinCamera.h"
#include "BoxPosePipeline.h"
#include <nlohmann/json.hpp>
#include <open3d/io/PointCloudIO.h>

using namespace std;
using namespace open3d::geometry;

namespace zzb {
    struct Box {
        double x;
        double y;
        double z;
        double width;
        double height;
        double angle_a;
        double angle_b;
        double angle_c;
        float rw1;
        float rw2;
        float rw3;
        float rw4;
        float rw5;
        float rw6;
        float rw7;
        float rw8;
        float rw9;
    };
}
extern "C" {
__declspec(dllexport) int bs_yzx_init(bool _isDebug);

__declspec(dllexport) int bs_yzx_object_detection_lanxin(int taskId, zzb::Box boxArr[]);
}

#endif //MY_VISION_LIBRARY_H
