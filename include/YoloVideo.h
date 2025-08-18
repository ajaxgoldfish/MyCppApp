//
// Created by zwj on 2024/6/15.
//

#ifndef YOLOVIDEO_H
#define YOLOVIDEO_H
#include <common.h>
#include <Detector.h>
#include <VideoEncoder.h>
#include <Yolo.h>
#include "LanxinCamera.h"


class YoloVideo {
public:
    int captureMat(Mat &mat);

    void capturePointCloud(open3d::geometry::PointCloud & pc) const;

    explicit YoloVideo(const shared_ptr<Yolo> &yolo,const shared_ptr<Yolo> &yolo_person, bool isDebug);

    ~YoloVideo();

    void startRecord();

    std::vector<RecognitionRes> stopRecord();

    void setOnFrameCallBack(const FrameCallBack _frameCallBack) {
        onFrameCallBack = _frameCallBack;
    }

    void setOnRecognitionResultCallBack(const RecognitionResultCallBack _on_recognition_result_call_back) {
        on_recognition_result_call_back_ = _on_recognition_result_call_back;
    }

    [[nodiscard]] bool is_connect() const {
        return videoCapture_->isOpened();
    }

    void read_mat(Mat &mat) const {
        mat = *readMat;
    }

    [[nodiscard]] cv::Mat get_param() const {
        return videoCapture_->get_param();
    }

private:
    shared_ptr<Mat> readMat;
    shared_ptr<Yolo> yolo_;
    shared_ptr<Yolo> yolo_person_;
    shared_ptr<VideoEncoder> encoder_;
    bool isDebug_;
    shared_ptr<LanxinCamera> videoCapture_;
    bool isRun = true;
    bool isRecord = false;
    std::mutex vecLocker;
    std::vector<RecognitionRes> allBoxes;
    std::mutex personResultLocker;
    std::vector<RecognitionRes> personResultList;
    std::thread thread_;
    std::thread inferenceThread_;
    std::thread inferencePersonThread_;
    FrameCallBack onFrameCallBack = nullptr;
    RecognitionResultCallBack on_recognition_result_call_back_ =nullptr;
    std::mutex mutex_{};
};


#endif //YOLOVIDEO_H
