#include "Detector.h"
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <spdlog/spdlog.h>
#include "RecognitionRes.h"
#include "Utils.h"
#include "library.h"
#include <filesystem>

#include "MaskRCNN.h"
#include "YoloVideo.h"

using namespace std;
using namespace open3d;
using namespace open3d::io;

static shared_ptr<Detector> bs_detector = nullptr;
static shared_ptr<YoloVideo> yoloVideo;
static shared_ptr<Yolo> yolo_box;
static shared_ptr<Yolo> yolo_person;
static shared_ptr<MaskRCNN> maskRCNN;

static bool isDebug;

static void showImg(const Mat &img, const string &title = "image") {
    if (!isDebug) {
        return;
    }
    cv::namedWindow(title, WINDOW_NORMAL);
    cv::imshow(title, img);
    cv::waitKey(0);
    //cv::destroyWindow(title);
}

int bs_yzx_init(const bool _isDebug) {
    isDebug = _isDebug;

    if (!std::filesystem::exists("res")) {
        std::filesystem::create_directory("res");
    }

    spdlog::set_level(spdlog::level::level_enum::info);
    const std::filesystem::path currentPath = std::filesystem::current_path();
    spdlog::info("工作目录：{}", currentPath.string());
    //初始化深度学习模型
    yolo_box = make_shared<Yolo>(L"model\\yolov8n.onnx");
    if (std::filesystem::exists("model\\yolov8_person.onnx")) {
        yolo_person = make_shared<Yolo>(L"model\\yolov8_person.onnx");
    }
    spdlog::info("加载yolo完成");
    maskRCNN = make_shared<MaskRCNN>("model");
    spdlog::info("加载MaskRCNN完成");
    yoloVideo = make_shared<YoloVideo>(yolo_box, yolo_person, isDebug);
    bs_detector = make_shared<Detector>("config\\params.xml", isDebug);
    bs_detector->set_intrinsic_rgb(yoloVideo->get_param());

    Mat m;
    if (const auto code = yoloVideo->captureMat(m); code == 0) {
        spdlog::info("拍照图片尺寸：{}*{}", m.cols, m.rows);
        //预热paddle-maskrcnn
        vector<RecognitionRes> vec;
        maskRCNN->inference(m, vec);
        //预热yolo实时识别相机打开失败
        yolo_box->inference(m, vec);
    }
    return 0;
}

static void showMaskOnMat(const cv::Mat &undistortMat, const vector<RecognitionRes> &resList, const string &imgFile) {
    Mat rgbShowMaskRcnn = undistortMat.clone();
    for (int i = 0; i < resList.size(); ++i) {
        auto &item = resList[i];
        item.drawResult(rgbShowMaskRcnn);
        cv::putText(rgbShowMaskRcnn, std::to_string(item.score),
                    {
                        static_cast<int>(item.rotatedRect.center.x - 30),
                        static_cast<int>(item.rotatedRect.center.y)
                    },
                    1, 1, {255, 0, 0}, 1);
        cv::putText(rgbShowMaskRcnn, std::to_string(i), item.rotatedRect.center, 1, 3, {0, 0, 255}, 2);
    }
    cv::addWeighted(rgbShowMaskRcnn, 0.5, undistortMat, 1 - 0.5, 0, rgbShowMaskRcnn);
    cv::imwrite(imgFile, rgbShowMaskRcnn);
}

static bool isSameBox(const vector<Size> &sizeVec, const float w, const float h) {
    for (const auto size: sizeVec) {
        if (std::abs(size.height - h) < size.height * 0.2
            && std::abs(size.width - w) < size.width * 0.2) {
            return true;
        }
    }
    return false;
}

int bs_yzx_object_detection_lanxin(int taskId, Box boxArr[]) {
    //extrinsicRGB
    spdlog::info("bs_yzx_object_detection taskId={}", taskId);
    Mat undistortMat;
    vector<RecognitionRes> resList;
    int imgHandleCode = 0;
    const string &dirStr = std::to_string(taskId);
    std::filesystem::create_directory("res/" + dirStr);
    std::thread imgThread([&]() {
        spdlog::info("imgThread start");
        //Mat rgbMat;
        const string imgFilePath2 = std::format("res/{}/rgb.jpg", dirStr);
        if (const int code = yoloVideo->captureMat(undistortMat); code != 0) {
            imgHandleCode = ERROR_CODE_CAMERA_READ_FAIL;
            return;
        }
        spdlog::info("imwrite start");
        cv::imwrite(imgFilePath2, undistortMat);
        spdlog::info("拍照完成");
        omp_set_num_threads(1);
        maskRCNN->inference(undistortMat, resList);
        spdlog::info("maskRCNN结果过滤完成size={}", resList.size());
        showMaskOnMat(undistortMat, resList, "res/" + dirStr + "/mask_rcnn.jpg");
    });

    yoloVideo->startRecord();
    PointCloud pc;
    yoloVideo->capturePointCloud(pc);
    if (pc.points_.empty()) {
        imgThread.join();
        spdlog::info("识别结束，无点云数据。");
        return ERROR_CODE_LIDAR_EMPTY_POINT_CLOUD;
    }
    WritePointCloudToPCD(std::format("res/{}/pcAll.pcd", dirStr), pc, {});
    spdlog::info("PCD-ALL点云保存完成");
    //等待图片识别
    imgThread.join();
    if (imgHandleCode != 0) {
        return imgHandleCode;
    }

    vector<RecognitionRes> allBoxes = yoloVideo->stopRecord();
    showMaskOnMat(undistortMat, allBoxes, "res/" + dirStr + "/yolo.jpg");
    for (auto &item: allBoxes) {
        item.score -= 0.6;
        resList.push_back(item);
    }
    if (resList.empty()) {
        spdlog::info("识别结束，未识别到箱子。");
        return 0;
    }

    spdlog::info("video yolo识别size={},合并后size={}", allBoxes.size(), resList.size());
    int num_threads = omp_get_num_procs();
    spdlog::info("开始点云融合使用{}线程处理", num_threads);
    omp_set_num_threads(num_threads);
    bs_detector->detection("extrinsicRGB", pc, undistortMat, resList, boxArr, dirStr);
    spdlog::info("点云融合结束");
    return 0;
}

int bs_yzx_video_frame_callback(const FrameCallBack onFrameCallback) {
    if (yoloVideo == nullptr) {
        return -1;
    }
    yoloVideo->setOnFrameCallBack(onFrameCallback);
    return 0;
}

int bs_yzx_recognition_result_callback(const RecognitionResultCallBack onRecognitionResultCallBack) {
    if (yoloVideo == nullptr) {
        return -1;
    }
    yoloVideo->setOnRecognitionResultCallBack(onRecognitionResultCallBack);
    return 0;
}

int bs_yzx_destroy() {
    return 0;
}

int bs_yzx_get_video_status() {
    if (yoloVideo == nullptr) {
        return -1;
    }
    return yoloVideo->is_connect();
}

int bs_yzx_record_rtsp(const char *rtsp, const char *file_name, int frameNum) {
    cv::VideoCapture vc{rtsp};

    Mat m;
    if (!vc.isOpened()) {
        return ERROR_CODE_CAMERA_CONNECT_FAIL;
    }
    if (!vc.read(m)) {
        return ERROR_CODE_CAMERA_READ_FAIL;
    }
    const auto fps = std::min(25, static_cast<int>(vc.get(CAP_PROP_FPS)));
    cv::VideoWriter writer(file_name,
                           cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                           cv::Size(m.cols, m.rows));
    while (vc.isOpened() && frameNum > 0) {
        frameNum--;
        if (!vc.read(m)) {
            break;
        }
        if (m.empty()) {
            break;
        }
        writer.write(m);
        int ms = min(10, 1000 / fps - 20);
        std::this_thread::sleep_for(chrono::milliseconds(ms));
    }
    writer.release();
    return 0;
}
