#ifndef YZX_VISION_DETECTOR_H
#define YZX_VISION_DETECTOR_H

#include <string>
#include <open3d/Open3D.h>
#include <memory>
#include "Utils.h"
#include "RecognitionRes.h"
#include "common.h"

using namespace std;
using namespace cv;
using namespace open3d::geometry;


class Detector {

public:
    explicit Detector(const string &configFileName, bool isShow_ = false);

    int detection(const std::string &paramKey, PointCloud &pointCloud, Mat &rgbMat,
                  vector<RecognitionRes> &resVec, Box *boxArr, const string &dateStr);

    void set_intrinsic_rgb(const Mat &intrinsic_rgb) {
        intrinsicRGB = intrinsic_rgb;
    }

    //从相机到机械臂的外参
    Mat getExtrinsicRGB(const std::string &key) const;

    //从机械臂到相机的外参
    Mat getInverseExtrinsicRGB(const Mat &ext);

    void showPcl(const vector<shared_ptr<const Geometry>> &list, const string &title = "open3d") const;
private:
    bool isShow;
    //相机内参
    Mat intrinsicRGB;
    std::mutex boxesLocker;
    std::shared_ptr<cv::FileStorage> configReader;
    void oneMaskHandle(const PointCloud &pc, vector<Box> &boxes,
                       vector<shared_ptr<const Geometry>> &showList,RecognitionRes &rectRes);

    void showResultOnMat(const Mat &rgbMat, const vector<Box> &boxes,Box *boxArr) const;

    int detection2(const Mat &depthMat, vector<Box> &boxes,
                   vector<shared_ptr<const Geometry>> &showList);

    void projectCloudPoint(PointCloud &pointCloud, Mat &depthMat) const;

    void showImg(Mat &img, const string &title = "image") const;
};

#endif //YZX_VISION_DETECTOR_H
