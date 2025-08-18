//
// Created by zwj on 2024/5/31.
//

#include "Yolo.h"

#include <omp.h>

using namespace std;
using namespace cv;


static void paddingResize(const cv::Mat &srcImg, const Size size, cv::Mat &oImg) {
    if (srcImg.rows > srcImg.cols) {
        Mat m;
        cv::copyMakeBorder(srcImg, m, 0, 0, 0, srcImg.rows - srcImg.cols, cv::BORDER_CONSTANT, {255, 255, 255});
        cv::resize(m, oImg, size);
    } else if (srcImg.rows < srcImg.cols) {
        Mat m;
        cv::copyMakeBorder(srcImg, m, 0, srcImg.cols - srcImg.rows, 0, 0, cv::BORDER_CONSTANT, {255, 255, 255});
        cv::resize(m, oImg, size);
    } else {
        cv::resize(srcImg, oImg, size);
    }
}

Yolo::Yolo(const wchar_t *modelPath, const int gpuMemSize) {
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
    Ort::SessionOptions sessionOption;
    OrtCUDAProviderOptions cudaOption;
    cudaOption.device_id = 0;
    cudaOption.gpu_mem_limit = gpuMemSize * 1024LL * 1024LL * 1024LL;
    sessionOption.AppendExecutionProvider_CUDA(cudaOption);

    sessionOption.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(omp_get_num_procs());
    sessionOption.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    session = new Ort::Session(*env, modelPath, sessionOption);

    const Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr taskValuePtr = session->GetModelMetadata().LookupCustomMetadataMapAllocated(
        "task", allocator);
    if (taskValuePtr != nullptr) {
        if (string("obb") == taskValuePtr.get()) {
            this->_isObb = true;
        }
    }

    if (session->GetInputCount() == 0) {
        throw std::exception("模型格式错误：没有对应的输入信息。");
    }
    if (session->GetInputCount() != 1) {
        throw std::exception("模型格式错误：只能有一个输入信息。");
    }
    const auto inputNameAllocated = session->GetInputNameAllocated(0, allocator);
    this->inputNodeName = inputNameAllocated.get();

    this->inputNodeDims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inputNodeDims.size() != 4) {
        throw std::exception("模型格式错误：不是Yolo的标准输入格式。");
    }

    if (session->GetOutputCount() == 0) {
        throw std::exception("模型格式错误：没有对应的输出信息。");
    }
    if (session->GetOutputCount() != 1) {
        throw std::exception("模型格式错误：只能有一个输出信息。");
    }
    const auto outputNameAllocated = session->GetOutputNameAllocated(0, allocator);
    this->outputNodeName = outputNameAllocated.get();

    this->outputNodeDims = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (outputNodeDims.size() != 3) {
        throw std::exception("模型格式错误：不是Yolo的标准输出格式。");
    }
}

void Yolo::inference(Mat &rgbMat, std::vector<RecognitionRes> &resultBoxes) const {
    Mat oMat;
    paddingResize(rgbMat, {
                      static_cast<int>(this->inputNodeDims[2]),
                      static_cast<int>(this->inputNodeDims[3])
                  }, oMat);
    spdlog::debug("paddingResize finish");
    const auto factor = static_cast<float>(std::max(rgbMat.rows, rgbMat.cols) / static_cast<double>(oMat.rows));
    cv::Mat blob;
    cv::dnn::blobFromImage(oMat, blob, 1 / 255.0, {oMat.rows, oMat.cols}, cv::Scalar(0, 0, 0), true, false);

    std::vector inputNodeNames = {this->inputNodeName.data()};
    std::vector outputNodeNames = {this->outputNodeName.data()};


    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<float>::type>(
        this->memory_info_in,
        reinterpret_cast<float *>(blob.data), inputNodeDims[0] * inputNodeDims[1] * inputNodeDims[2] * inputNodeDims[3],
        inputNodeDims.data(), inputNodeDims.size());

    const auto opt = Ort::RunOptions{nullptr};
    spdlog::debug("yolo onnx run 开始识别");
    std::vector<Ort::Value> outputTensor =
            session->Run(opt, inputNodeNames.data(), &inputTensor,
                         inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());
    inputTensor.release();
    spdlog::debug("yolo onnx run 结束识别");


    const auto data = outputTensor.front().GetTensorMutableData<float>();

    const int signalResultNum = static_cast<int>(this->outputNodeDims[1]); //84
    const int strideNum = static_cast<int>(this->outputNodeDims[2]); //8400
    std::vector<float> confidences;
    std::vector<cv::RotatedRect> boxes;

    for (int i = 0; i < strideNum; ++i) {
        float score = data[strideNum * 4 + i];
        if (score < 0.7) {
            continue;
        }
        float x = data[strideNum * 0 + i];
        float y = data[strideNum * 1 + i];
        float w = data[strideNum * 2 + i];
        float h = data[strideNum * 3 + i];
        float r = 0;
        if (_isObb) {
            r = data[strideNum * (signalResultNum - 1) + i];
        }
        float left = x * factor;
        float top = y * factor;
        float width = w * factor;
        float height = h * factor;
        confidences.push_back(score);
        boxes.push_back({{left, top}, {width, height}, r * (180 / (float) numbers::pi)});
    }
    spdlog::debug("NMSBoxes begin");
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, 0.7, 0.6, nmsResult);
    spdlog::debug("NMSBoxes end");
    for (const int idx: nmsResult) {
        resultBoxes.emplace_back(boxes[idx], confidences[idx], rgbMat.rows, rgbMat.cols);
    }
}

Yolo::~Yolo() {
    session->release();
    delete session;
    env->release();
    delete env;
}
