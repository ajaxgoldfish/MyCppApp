//
// Created by zwj on 2024/6/10.
//

#ifndef MASKRCNN_H
#define MASKRCNN_H

#include <spdlog/spdlog.h>
#include <RecognitionRes.h>
#include <string>
#include "libpaddlex.h"
using namespace std;

class MaskRCNN {

public:
    explicit MaskRCNN(const string& modePath);

    int inference(const cv::Mat &img, vector<RecognitionRes> &resList, float threshold=0.3) const;

    ~MaskRCNN();

private:
    PaddleDeploy::Model *paddleModel = nullptr;
};


#endif //MASKRCNN_H
