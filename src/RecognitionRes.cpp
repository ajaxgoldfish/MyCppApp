//
// Created by zwj on 2024/5/21.
//

#include "Utils.h"
#include "RecognitionRes.h"
#include <algorithm>
#include <utility>
#include <spdlog/spdlog.h>

using namespace std;
using namespace cv;


void
RecognitionRes::iouFilter(vector<RecognitionRes> &recognitionResVec, vector<RecognitionRes> &resList, float threshold) {
    std::copy_if(recognitionResVec.begin(), recognitionResVec.end(),
                 std::back_inserter(resList), [&](RecognitionRes &recognitionRes) {
                     //没有一个 比当前结果得分高的 相似的结果
                     return !std::any_of(recognitionResVec.begin(), recognitionResVec.end(), [&](RecognitionRes &item) {
                         return recognitionRes.score < item.score && item / recognitionRes > threshold;
                     });
                 });
}


RecognitionRes::RecognitionRes(Mat &mask, double score) {
    this->mask = mask.clone();
    this->rotatedRect = getMinRect(this->mask);
    this->score = score;
}

RecognitionRes::RecognitionRes(RotatedRect _rotatedRect, double score, int rows, int cols) {
    this->mask = Mat(rows, cols, CV_8UC1, cv::Scalar(0));
    getMaskByRRect(_rotatedRect, this->mask);
    this->rotatedRect = _rotatedRect;
    this->score = score;
}

void RecognitionRes::getMaskByRRect(const cv::RotatedRect &rect, Mat &mat) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    // 将顶点坐标转换为整型，以便绘图
    std::vector<cv::Point> verticesInt;
    for (const auto &pt: vertices) {
        verticesInt.emplace_back(cvRound(pt.x), cvRound(pt.y));
    }
    // 使用白色填充旋转矩形的多边形区域
    cv::fillConvexPoly(mat, verticesInt.data(),
                       static_cast<int>(verticesInt.size()),
                       cv::Scalar(1));
}

cv::RotatedRect RecognitionRes::getMinRect(const cv::Mat &mat) {
    // 查找轮廓
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 确保找到了轮廓
    if (contours.empty()) {
        return {};
    }
    // 选择最大的轮廓作为示例
    auto maxContourIt = std::max_element(contours.begin(), contours.end(),
                                         [](const std::vector<cv::Point> &contour1,
                                            const std::vector<cv::Point> &contour2) {
                                             return cv::contourArea(contour1) < cv::contourArea(contour2);
                                         });
    return cv::minAreaRect(*maxContourIt);
}

void RecognitionRes::drawResult(Mat &mat) const{
    const double b = rand() % 256;
    const double r = rand() % 256;
    const double g = rand() % 256;
    mat.setTo(Scalar{b, g, r}, this->mask.clone());

    // 绘制旋转矩形框
    cv::Point2f vertices[4];
    this->rotatedRect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(mat, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }
}

// 计算两个矩形的交集区域
static Rect intersectRect(const Rect &r1, const Rect &r2) {
    int x1 = max(r1.x, r2.x);
    int y1 = max(r1.y, r2.y);
    int x2 = min(r1.x + r1.width, r2.x + r2.width);
    int y2 = min(r1.y + r1.height, r2.y + r2.height);

    if (x1 < x2 && y1 < y2) {
        return Rect(x1, y1, x2 - x1, y2 - y1);
    } else {
        return Rect(); // 如果没有交集，返回空矩形
    }
}


double RecognitionRes::iou(const RecognitionRes &res) {
    auto r1 = this->rotatedRect.boundingRect();
    auto r2 = res.rotatedRect.boundingRect();
    Rect inter = intersectRect(r1, r2);
    double intersection_area = inter.area();
    double union_area = r1.area() + r2.area() - intersection_area;
    return union_area > 0 ? intersection_area / union_area : 0; // 防止除以0
}

double RecognitionRes::operator/(RecognitionRes &res) {
    auto &rect1 = this->rotatedRect;
    auto &rect2 = res.rotatedRect;
    vector<Point2f> intersectingRegion;
    rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
    if (intersectingRegion.empty()) {
        return 0;
    }

    vector<Point2f> sort_intersectingRegion;
    convexHull(intersectingRegion, sort_intersectingRegion);
    double inter_area = contourArea(sort_intersectingRegion);
    double area_r1 = rect1.size.width * rect1.size.height;
    double area_r2 = rect2.size.width * rect2.size.height;
    return max(inter_area / area_r1, inter_area / area_r2);
}
