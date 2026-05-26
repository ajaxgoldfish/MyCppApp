#include <iostream>
#include "../include/cpu_library.h"

int main() {
    std::cout << "Start testing yzx_vision_zzb_gpu" << std::endl;

    const int init_result = bs_yzx_init(true);
    if (init_result != 0) {
        std::cout << "bs_yzx_init failed, return value: " << init_result << std::endl;
        return -1;
    }

    zzb::Box boxes[100]{};
    const int detection_result = bs_yzx_object_detection_lanxin(boxes);

    if (detection_result < 0) {
        std::cout << "Detection failed, error code: " << detection_result << std::endl;
        return -1;
    }

    std::cout << "Detection succeeded, count: " << detection_result << std::endl;
    std::cout << "Results are saved under res/<timestamp>/" << std::endl;

    for (int i = 0; i < detection_result; ++i) {
        std::cout << "\nObject #" << (i + 1) << std::endl;
        std::cout << "  ID: " << boxes[i].id << std::endl;
        std::cout << "  Position (x, y, z): "
                  << boxes[i].x << ", "
                  << boxes[i].y << ", "
                  << boxes[i].z << " mm" << std::endl;
        std::cout << "  Size (w, h): "
                  << boxes[i].width << " x "
                  << boxes[i].height << " mm" << std::endl;
        std::cout << "  Angle (a, b, c): "
                  << boxes[i].angle_a << ", "
                  << boxes[i].angle_b << ", "
                  << boxes[i].angle_c << std::endl;
    }

    return 0;
}
