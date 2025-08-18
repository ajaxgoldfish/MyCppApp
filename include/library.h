#ifndef YZX_VISION_LIBRARY_H
#define YZX_VISION_LIBRARY_H

#include <open3d/Open3D.h>
#include "common.h"

using namespace std;
using namespace open3d::geometry;

extern "C" {
__declspec(dllexport) int bs_yzx_init(bool _isDebug);

__declspec(dllexport) int bs_yzx_object_detection_lanxin(int taskId, Box boxArr[]);

__declspec(dllexport) int bs_yzx_video_frame_callback(FrameCallBack onFrameCallback);

__declspec(dllexport) int bs_yzx_recognition_result_callback(RecognitionResultCallBack onRecognitionResultCallBack);

__declspec(dllexport) int bs_yzx_destroy();

__declspec(dllexport) int bs_yzx_get_video_status();

__declspec(dllexport) int bs_yzx_record_rtsp(const char * rtsp,const char * file_name,int frameNum);
}

#endif //YZX_VISION_LIBRARY_H
