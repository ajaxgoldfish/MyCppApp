//
// Created by zwj on 2024/6/15.
//

#include "YoloVideo.h"
#include <VideoEncoder.h>
#include <common.h>


int YoloVideo::captureMat(Mat &mat) {
    std::lock_guard guard(this->mutex_);

    Mat mm;
    spdlog::debug("videoCapture_ read begin");
    const int code = videoCapture_->CapFrame(mm);
    spdlog::info("videoCapture_ read code ={}", code);
    if (code != 0 || mm.empty()) {
        spdlog::error("YoloVideo拍照失败");
        return ERROR_CODE_CAMERA_CONNECT_FAIL;
    }
    mat = mm;
    return 0;
}

void YoloVideo::capturePointCloud(open3d::geometry::PointCloud &pc) const {
    videoCapture_->CapFrame(pc);
}

YoloVideo::YoloVideo(const shared_ptr<Yolo> &yolo, const shared_ptr<Yolo> &yolo_person,
                     const bool isDebug) {
    this->isDebug_ = isDebug;
    this->yolo_ = yolo;
    this->yolo_person_ = yolo_person;
    this->encoder_ = make_shared<VideoEncoder>();
    this->videoCapture_ = make_shared<LanxinCamera>();
    //推流线程
    // this->thread_ = std::thread([&]() {
    //     spdlog::debug("thread start");
    //     if (isDebug_) {
    //         cv::namedWindow("yolo_video", WINDOW_NORMAL);
    //     }
    //
    //     while (this->isRun) {
    //         try {
    //             if (this->encoder_ == nullptr) {
    //                 const auto matPtr = make_shared<Mat>();
    //                 if (const int code = captureMat(*matPtr); code != 0) {
    //                     spdlog::error("实时识别相机读取失败");
    //                     continue;
    //                 }
    //                 int code = this->encoder_->init(matPtr->cols, matPtr->rows,
    //                                                 matPtr->cols / 4, matPtr->rows / 4);
    //                 spdlog::info("encoder_ init code ={}", code);
    //                 if (code != 0) {
    //                     continue;
    //                 }
    //             }
    //
    //             spdlog::debug("captureMat begin");
    //             auto matPtr = make_shared<Mat>();
    //             if (const int code = captureMat(*matPtr); code != 0) {
    //                 continue;
    //             }
    //             this->readMat = matPtr;
    //             spdlog::debug("视频识别-读取图片完成");
    //             Mat showMat = matPtr->clone();
    //             if (isRecord) {
    //                 std::lock_guard locker(this->vecLocker);
    //                 for (auto &item: allBoxes) {
    //                     // 绘制旋转矩形框
    //                     cv::Point2f vertices[4];
    //                     item.rotatedRect.points(vertices);
    //                     for (int i = 0; i < 4; ++i) {
    //                         cv::line(showMat, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    //                     }
    //                     cv::putText(showMat, std::to_string(static_cast<int>(round(item.score * 100))),
    //                                 item.rotatedRect.center, 1, 2,
    //                                 {255, 0, 0}, 2);
    //                     cv::putText(showMat, std::to_string(item.weight),
    //                                 {
    //                                     static_cast<int>(item.rotatedRect.center.x),
    //                                     static_cast<int>(item.rotatedRect.center.y) + 30
    //                                 },
    //                                 1, 2, {0, 0, 255}, 2);
    //                 }
    //             }
    //             std::lock_guard locker(personResultLocker); {
    //                 for (auto &item: this->personResultList) {
    //                     // 绘制旋转矩形框
    //                     cv::Point2f vertices[4];
    //                     item.rotatedRect.points(vertices);
    //                     for (int i = 0; i < 4; ++i) {
    //                         cv::line(showMat, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
    //                     }
    //                 }
    //             }
    //             if (isDebug_) {
    //                 cv::imshow("yolo_video", showMat);
    //                 cv::waitKey(3);
    //             }
    //             if (this->onFrameCallBack != nullptr) {
    //                 if (!showMat.empty()) {
    //                     if (const int code = this->encoder_->encode(showMat); code == 0) {
    //                         auto &&av_packet = this->encoder_->getAVPacket();
    //                         onFrameCallBack(av_packet->data, av_packet->size);
    //                     }
    //                 }
    //             }
    //         } catch (std::exception &ex) {
    //             spdlog::error("yolo loop error={}", ex.what());
    //         }
    //     }
    // });

    //box识别线程
    this->inferenceThread_ = std::thread([&] {
        while (this->isRun) {
            if (!isRecord) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            auto matPtr = readMat;
            if (matPtr == nullptr) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            if (matPtr->rows == 0 || matPtr->cols == 0) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            Mat mat = matPtr->clone();
            if (mat.empty()) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            std::vector<RecognitionRes> vec;
            yolo_->inference(mat, vec);
            spdlog::debug("yolo->inference size={}", vec.size());

            //加锁 控制vector
            {
                std::lock_guard locker(this->vecLocker);
                //不是记录模式下清除allBoxes
                if (!this->isRecord) {
                    this->allBoxes.clear();
                }
                for (auto &item: vec) {
                    auto rec = ranges::find_if(this->allBoxes, [&](auto &i) { return i.iou(item) > 0.5; });
                    if (rec != allBoxes.end()) {
                        if (rec->score > item.score) {
                            //之前的分数比现在高
                            rec->weight = rec->weight + 1;
                        } else {
                            //之前的分数比现在低，
                            //添加现在的
                            item.weight = rec->weight + 1;
                            // 删除之前的
                            allBoxes.erase(rec);
                            allBoxes.push_back(item);
                        }
                    } else {
                        //之前没有 则新增
                        allBoxes.push_back(item);
                    }
                }
            } //end lock
        }
    });

    //人员识别线程
    this->inferencePersonThread_ = std::thread([&] {
        if (yolo_person_ == nullptr) {
            return;
        }
        while (this->isRun) {
            if (on_recognition_result_call_back_ == nullptr) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            auto matPtr = readMat;
            if (matPtr == nullptr) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            if (matPtr->rows == 0 || matPtr->cols == 0) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            Mat mat = matPtr->clone();
            if (mat.empty()) {
                std::this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            std::vector<RecognitionRes> vec;
            yolo_person_->inference(mat, vec);
            spdlog::debug("yolo_person->inference size={}", vec.size());
            std::lock_guard locker(personResultLocker); {
                this->personResultList = vec;
            }
            if (on_recognition_result_call_back_ != nullptr) {
                for (const auto &item: vec) {
                    if (item.score <= 0.9) {
                        continue;
                    }
                    on_recognition_result_call_back_(0, 1, item.score,
                                                     item.rotatedRect.center.x, item.rotatedRect.center.y,
                                                     item.rotatedRect.size.width, item.rotatedRect.size.height);
                }
            }
        }
    });
}

YoloVideo::~YoloVideo() {
    this->isRun = false;
}

void YoloVideo::startRecord() {
    lock_guard lock(vecLocker);
    this->allBoxes.clear();
    this->isRecord = true;
}

std::vector<RecognitionRes> YoloVideo::stopRecord() {
    lock_guard lock(vecLocker);
    this->isRecord = false;
    return allBoxes;
}
