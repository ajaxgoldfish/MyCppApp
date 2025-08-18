//
// Created by zwj on 2024/5/31.
//

#ifndef YZX_VISION_YOLO_H
#define YZX_VISION_YOLO_H

#include <spdlog/spdlog.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <numbers>
#include <RecognitionRes.h>

class Yolo {
public:
    void inference(Mat &rgbMat, std::vector<RecognitionRes> &resultBoxes) const;

    Yolo(const wchar_t *modelPath, int gpuMemSize = 2);

    ~Yolo();

private:
    Ort::Env *env;
    Ort::Session *session;
    std::vector<int64_t> inputNodeDims;
    std::vector<int64_t> outputNodeDims;
    bool _isObb = false;
    string inputNodeName;
    string outputNodeName;
    Ort::MemoryInfo memory_info_in = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};


#endif //YZX_VISION_YOLO_H
