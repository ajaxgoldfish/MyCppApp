//
// Created by zwj on 2024/6/10.
//

#include "MaskRCNN.h"


MaskRCNN::MaskRCNN(const string &modePath) {
    this->paddleModel = init_model(modePath + "\\model.yml",
                                   "",
                                   "paddlex",
                                   modePath + "\\model.pdmodel",
                                   modePath + "\\model.pdiparams",
                                   true, 0);
}

int MaskRCNN::inference(const cv::Mat &img, vector<RecognitionRes> &resList, const float threshold) const {
    vector<PaddleDeploy::Result> results;
    spdlog::info("准备开始maskRCNN预测");
    const int code = predict(img, this->paddleModel, results);
    if (code != 0) {
        return code;
    }
    spdlog::info("完成maskRCNN预测，结果：{}", results[0].det_result->boxes.size());
    for (auto &[category_id, category, score, coordinate, mask]
         : results[0].det_result->boxes) {
        if (score <= threshold) {
            continue;
        }
        auto &[data, shape] = mask;
        cv::Mat mm(shape[0], shape[1], CV_8UC1, data.data());
        resList.emplace_back(mm, score);
    }
    return code;
}


MaskRCNN::~MaskRCNN() {
    if (this->paddleModel != nullptr) {
        destroy_model(this->paddleModel);
    }
}
