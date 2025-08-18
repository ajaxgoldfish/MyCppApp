//
// Created by zhangzongbo on 2025/8/13.
//

#ifndef MASKRCNNRUNNER_H
#define MASKRCNNRUNNER_H
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class MaskRCNNRunner {
public:

    // 构造函数：加载 ONNX 模型
    explicit MaskRCNNRunner(const std::string& model_path);

    // 推理函数：输入 Mat，返回绘制了检测框和掩膜的 Mat
    cv::Mat paint(const cv::Mat& orig,const std::vector<Ort::Value>& outs, float score_thr,float mask_thr);

    // 提取每个实例的二值掩膜（原图大小，0/255）
    std::vector<cv::Mat1b> inferMasks(const cv::Mat& orig,const std::vector<Ort::Value>& outs,float score_thr,float mask_thr);

    std::vector<Ort::Value> inferRaw(const cv::Mat& orig);

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_;
    std::vector<std::string> out_names_s_;
    std::vector<const char*> out_names_;

    static std::wstring to_wstring(const std::string& s);
    static inline void pixel_normalize_mmdet_rgb(cv::Mat& rgb_f32);
};




#endif //MASKRCNNRUNNER_H
