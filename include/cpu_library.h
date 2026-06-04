#ifndef MY_VISION_LIBRARY_H
#define MY_VISION_LIBRARY_H



namespace zzb {
    // DLL 对外返回的单个目标结果。
    // 位置和尺寸使用毫米，角度使用度，旋转矩阵用于调用方继续做机器人坐标或抓取姿态计算。
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
// 初始化模型和运行时资源，并一次性打开全部已发现相机；检测前必须调用一次。
__declspec(dllexport) int bs_yzx_init(bool _isDebug);

// 返回 Box 结构体大小，便于外部语言绑定校验 ABI 是否一致。
__declspec(dllexport) int bs_yzx_box_sizeof();

// 按 IP 选择启动阶段已打开的相机，采集 RGB 与点云，完成检测和位姿解算。
__declspec(dllexport) int bs_yzx_object_detection_lanxin(zzb::Box boxArr[],
                                                         const char *cameraIp,
                                                         const char *intrinsicExtrinsicPath);
}

#endif // 头文件保护
