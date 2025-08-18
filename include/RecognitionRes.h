//
// Created by zwj on 2024/5/21.
//

#ifndef YZX_VISION_RECOGNITIONRES_H
#define YZX_VISION_RECOGNITIONRES_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class RecognitionRes {

public:
    static void iouFilter(vector<RecognitionRes> &recognitionResVec, vector<RecognitionRes> &resList, float threshold);

    RecognitionRes(Mat &mask, double score);

    RecognitionRes(RotatedRect _rotatedRect, double score, int rows, int cols);

    cv::Mat mask;
    cv::RotatedRect rotatedRect;

    double score;

    int weight = 0;

    void drawResult(Mat &mat) const;

    double operator/(RecognitionRes &res);

    double iou(const RecognitionRes &res);



private:

    cv::RotatedRect getMinRect(const cv::Mat &mat);

    void getMaskByRRect(const cv::RotatedRect &rect, cv::Mat &mat);
};


#endif //YZX_VISION_RECOGNITIONRES_H
