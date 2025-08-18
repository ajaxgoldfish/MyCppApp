//
// Created by zwj on 2024/6/5.
//

#ifndef YZX_VISION_COMMON_H
#define YZX_VISION_COMMON_H

struct Box {
    double x;
    double y;
    double z;
    double width;
    double height;
    double angle_a;
    double angle_b;
    double angle_c;
};

struct MeasureInfo {
    double width;
    double height;
    double offset;
    double width1;
    double offset1;
    double angle;
    double y1;
    double y2;
    double ceilingZ;
    double floorZ;
    int isInCar;
};

//typedef void (*frameCallBack)(unsigned char *data, const int length);

constexpr int ERROR_CODE_CAMERA_CONNECT_FAIL = -1;
constexpr int ERROR_CODE_CAMERA_READ_FAIL = -2;
constexpr int ERROR_CODE_CAMERA_SAVE_FAIL = -3;

constexpr int ERROR_CODE_LIDAR_INIT_FAIL = -10;
constexpr int ERROR_CODE_LIDAR_CONNECT_FAIL = -11;
constexpr int ERROR_CODE_LIDAR_CONNECT_TIMEOUT = -12;
constexpr int ERROR_CODE_LIDAR_START_SAMPLE_FAIL = -13;
constexpr int ERROR_CODE_LIDAR_STOP_SAMPLE_FAIL = -14;
constexpr int ERROR_CODE_LIDAR_EMPTY_POINT_CLOUD = -15;

constexpr int ERROR_CODE_ONNX_INIT_FAIL = -100;
constexpr int ERROR_CODE_PADDLE_INIT_FAIL = -101;

typedef void (*FrameCallBack)(unsigned char *data,int length);

typedef void (*RecognitionResultCallBack)(int type,int classify,float score,int x,int y,int w,int h);

#endif //YZX_VISION_COMMON_H
