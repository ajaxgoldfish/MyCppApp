#ifndef MY_VISION_LIBRARY_H
#define MY_VISION_LIBRARY_H



namespace zzb {
    struct Box {
        int id;
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

__declspec(dllexport) int bs_yzx_box_sizeof();

__declspec(dllexport) int bs_yzx_object_detection_lanxin(zzb::Box boxArr[]);
}

#endif //MY_VISION_LIBRARY_H
