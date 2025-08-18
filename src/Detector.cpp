#include "Detector.h"
#include "RecognitionRes.h"
#include "Utils.h"
#include <cmath>
#include <opencv2/core/mat.hpp>
#include <numbers>
#include <map>
#include <spdlog/spdlog.h>

void Detector::showImg(Mat &img, const string &title) const {
    if (!isShow) {
        return;
    }
    cv::namedWindow(title, WINDOW_NORMAL);
    cv::imshow(title, img);
    cv::waitKey(0);
    //cv::destroyWindow(title);
}

Mat Detector::getExtrinsicRGB(const std::string &key) const {
    Mat ext;
    (*configReader)[key] >> ext;
    return ext;
}

Mat Detector::getInverseExtrinsicRGB(const Mat &ext) {
    Mat invExt;
    cv::invert(ext, invExt, cv::DECOMP_LU);
    return invExt;
}

void Detector::showPcl(const vector<shared_ptr<const Geometry> > &list, const string &title) const {
    if (!this->isShow) {
        return;
    }
    auto coordinate_frame = TriangleMesh::CreateCoordinateFrame(1);
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow(title, 1500, 900);
    vis.AddGeometry(coordinate_frame);

    for (const auto &item: list) {
        vis.AddGeometry(item);
    }
    vis.GetRenderOption().background_color_ = {0, 0, 0};
    vis.GetRenderOption().SetPointSize(2.5);
    vis.Run();
}


Detector::Detector(const string &configFileName, const bool isShow_) : isShow(isShow_) {
    this->configReader = std::make_shared<cv::FileStorage>(configFileName, cv::FileStorage::READ);
    if (!configReader->isOpened()) {
        throw;
    }
}


int Detector::detection2(const Mat &depthMat, vector<Box> &boxes,
                         vector<shared_ptr<const open3d::geometry::Geometry> > &showList) {
    //只通过点云二次识别
    spdlog::info("【点云二次识别】");
    Mat mask = cv::Mat::ones(depthMat.rows, depthMat.cols, CV_8UC1);
    PointCloud lastPc;
    Utils::copyTo(lastPc, depthMat, mask);
    showPcl({make_shared<PointCloud>(lastPc)}, "二次识别-扣除已经识别的箱子");
    if (lastPc.points_.size() <= 100) {
        return 0;
    }

    //点云降采样
    //lastPc = *lastPc.VoxelDownSample(0.01);
    //spdlog::info("点云降采样");

    //计算法向量
    lastPc.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(10));
    spdlog::info("点云二次识别-计算法向量");

    PointCloud boxPC;
    Utils::removeZAngleGtCloud(boxPC, 40, lastPc);
    spdlog::info("点云二次识别-删除大于40度的点");
    if (boxPC.points_.size() <= 100) {
        return 0;
    }
    showPcl({make_shared<PointCloud>(boxPC)}, "二次识别-删除大于40度的点");
    //点云聚类
    double eps = 0.03; // 邻域半径
    int min_points = 10; // 构成核心对象的邻域点数下限
    // DBSCAN聚类
    vector<int> labels = boxPC.ClusterDBSCAN(eps, min_points);
    map<int, PointCloud> mapPc;
    for (int i = 0; i < labels.size(); ++i) {
        auto pcIndex = labels[i];
        if (pcIndex == -1) {
            continue;
        }
        if (!mapPc.contains(pcIndex)) {
            mapPc[pcIndex] = PointCloud();
        }
        mapPc[pcIndex].points_.push_back(boxPC.points_[i]);
    }
    spdlog::info("点云二次识别-聚类完成");
    if (mapPc.empty()) {
        return 0;
    }

    //map的值转vector
    vector<PointCloud> pcs;
    for (const auto &item: mapPc) {
        pcs.push_back(item.second);
    }

#pragma omp parallel for
    for (int i = 0; i < pcs.size(); ++i) {
        auto item = pcs[i];
        if (item.points_.size() < 300) {
            continue;
        }
        //统计滤波
        auto spc = item.RemoveStatisticalOutliers(30, 2);
        auto anAuto = *get<0>(spc);
        PointCloud boxPc;
        Utils::removeZAngleGtCloud(boxPc, 20, anAuto);
        if (boxPc.points_.size() < 100) {
            continue;
        }
        const Eigen::Vector3d &extent = boxPc.GetAxisAlignedBoundingBox().GetExtent();
        if (extent.x() > 1 || extent.y() > 1 || extent.z() > 0.3 ||
            extent.x() < 0.1 || extent.y() < 0.1 ||
            extent.x() / extent.y() > 7 ||
            extent.y() / extent.x() > 7) {
            continue;
        }
        showPcl({make_shared<PointCloud>(boxPc)}, "二次识别-处理第" + std::to_string(i) + "个箱子");
        //oneMaskHandle(boxPc, boxes, showList);
    }
}


int Detector::detection(const std::string &paramKey, PointCloud &pointCloud, Mat &rgbMat,
                        vector<RecognitionRes> &resVec,
                        Box *boxArr, const string &dateStr) {
    spdlog::debug("保存识别图片");

    //showPcl({std::make_shared<PointCloud>(pointCloud)}, "");

    //通过IOU过滤mask
    vector<RecognitionRes> resList;
    RecognitionRes::iouFilter(resVec, resList, 0.15);
    spdlog::info("通过IOU过滤mask完成size={} -> size={}", resVec.size(), resList.size());

    //点云投影
    Mat depthMat = Mat::zeros(rgbMat.rows, rgbMat.cols, CV_32FC3); //有序点云CV_32FC1
    projectCloudPoint(pointCloud, depthMat);
    spdlog::debug("点云投影");
    vector<shared_ptr<const Geometry> > showList = {make_shared<PointCloud>(pointCloud)};
    vector<Box> boxes;
    //循环处理，并行
#pragma omp parallel for
    for (int i = 0; i < resList.size(); ++i) {
        PointCloud boxPc;
        Utils::copyTo(boxPc, depthMat, resList[i].mask);
        spdlog::debug("mat点图转pc点云");
        oneMaskHandle(boxPc, boxes, showList, resList[i]);
    }
    //boxes简单排序
    ranges::sort(boxes, [](auto &a, auto &b) {
        if (a.y * 1000 / 100 == b.y * 1000 / 100) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });
    //将结果映射到原图并显示
    if (this->isShow) {
        pointCloud.EstimateNormals();
    }
    showPcl(showList);
    spdlog::info("识别数量:{}", boxes.size());

    spdlog::info("转换前-----------------------------------------");
    for (int i = 0; i < boxes.size(); ++i) {
        spdlog::info("sn={} box point:x={},y={},z={},w={},h={}",
                      i, boxes[i].x, boxes[i].y, boxes[i].z, boxes[i].width, boxes[i].height);
    }
    spdlog::info("转换后-----------------------------------------");

#pragma omp parallel for
    for (int i = 0; i < boxes.size(); ++i) {
        //将相机坐标系的点转到雷达坐标系
        auto robotPoint = Utils::pointTransform(
            {
                static_cast<float>(boxes[i].z * 1000),
                -static_cast<float>(boxes[i].x * 1000),
                -static_cast<float>(boxes[i].y * 1000)
            },
            this->getExtrinsicRGB(paramKey));
        boxArr[i] = boxes[i];
        boxArr[i].x = robotPoint.x;
        boxArr[i].y = robotPoint.y;
        boxArr[i].z = robotPoint.z;
        boxArr[i].width = boxArr[i].width * 1000;
        boxArr[i].height = boxArr[i].height * 1000;
        spdlog::info("sn={} box point:x={},y={},z={},w={},h={},a={},c={}",
                     i, boxArr[i].x, boxArr[i].y, boxArr[i].z,
                     boxArr[i].width, boxArr[i].height,
                     boxArr[i].angle_a, boxArr[i].angle_c);
    }
    spdlog::info("处理完成，共{}条数据", boxes.size());

    showResultOnMat(rgbMat, boxes, boxArr);
    cv::imwrite("res/" + dateStr + "/result.jpg", rgbMat);
    showImg(rgbMat);
    return 0;
}

void Detector::projectCloudPoint(PointCloud &pointCloud, Mat &depthMat) const {
    std::vector<cv::Point3f> points;
    points.reserve(pointCloud.points_.size());
    for (const auto &item: pointCloud.points_) {
        points.emplace_back(item.x(), item.y(), item.z());
    }
    std::cout <<"intrinsicRGB:"<< intrinsicRGB << std::endl;
    vector<Point2f> imagePoints;
    const Mat rvec = Mat::zeros(3, 1, CV_64FC1); // 旋转矢量，默认无旋转
    const Mat tvec = Mat::zeros(3, 1, CV_64FC1); // 平移矢量，默认无平移
    const Mat distortion = Mat::zeros(1, 5, CV_32FC1); // 无畸变
    cv::projectPoints(points, rvec, tvec, intrinsicRGB, distortion, imagePoints);

    //Mat mat = Mat::zeros(depthMat.rows, depthMat.cols, CV_8UC1);
#pragma omp parallel for
    for (int i = 0; i < imagePoints.size(); i++) {
        const Point2f &point = imagePoints[i];
        const int px = static_cast<int>(round(point.y));
        const int py = static_cast<int>(round(point.x));
        if (px > 0 && px < depthMat.rows && py > 0 && py < depthMat.cols) {
            auto &v = depthMat.at<Vec3f>(px, py);
            v[0] = static_cast<float>(pointCloud.points_.at(i).x());
            v[1] = static_cast<float>(pointCloud.points_.at(i).y());
            v[2] = static_cast<float>(pointCloud.points_.at(i).z());
        }
    }
}


void Detector::showResultOnMat(const Mat &rgbMat, const vector<Box> &boxes, Box *boxArr) const {
    if (boxes.empty()) {
        return;
    }
    Mat rvec = Mat::zeros(3, 1, CV_64FC1); // 旋转矢量，默认无旋转
    Mat tvec = Mat::zeros(3, 1, CV_64FC1); // 平移矢量，默认无平移
    const Mat distortion = Mat::zeros(1, 5, CV_32FC1); // 无畸变
    vector<Point3f> box3dPoints;
    for (const auto &box: boxes) {
        box3dPoints.emplace_back(box.x, box.y, box.z);
    }
    vector<Point2f> box2dPoints;
    projectPoints(box3dPoints, rvec, tvec, intrinsicRGB, distortion, box2dPoints);
#pragma omp parallel for
    for (int i = 0; i < box2dPoints.size(); ++i) {
        vector<Point3d> line3dPoints;
        vector<Point2d> line2dPoints;
        const auto &box = boxes[i];
        //相机坐标系->机械手坐标系
        //x->y
        //y->-z
        //z->x
        line3dPoints.emplace_back(box.x - box.width / 2.0, box.y - box.height, box.z);
        line3dPoints.emplace_back(box.x + box.width / 2.0, box.y - box.height, box.z);
        line3dPoints.emplace_back(box.x + box.width / 2.0, box.y, box.z);
        line3dPoints.emplace_back(box.x - box.width / 2.0, box.y, box.z);
        projectPoints(line3dPoints, rvec, tvec, intrinsicRGB, distortion, line2dPoints);

        for (int p = 0; p < 4; ++p) {
            cv::line(rgbMat, line2dPoints[p], line2dPoints[(p + 1) % 4], {255, 0, 0}, 2, LINE_AA);
        }
        auto &item = box2dPoints[i];
        circle(rgbMat,
               {static_cast<int>(round(item.x)), static_cast<int>(round(item.y))},
               5, {255, 0, 0}, -1);

        putText(rgbMat, std::format("[{}]", i),
                  {static_cast<int>(round(item.x)) - 25, static_cast<int>(round(item.y)) - 100},
                  1, 1, {0, 255, 0}, 2);
        putText(rgbMat, std::format("{}", round(boxArr[i].x)),
          {static_cast<int>(round(item.x)) - 25, static_cast<int>(round(item.y)) - 80},
          1, 1, {0, 255, 0}, 2);
        putText(rgbMat, std::format("{}", round(boxArr[i].y)),
          {static_cast<int>(round(item.x)) - 25, static_cast<int>(round(item.y)) - 60},
          1, 1, {0, 255, 0}, 2);
        putText(rgbMat, std::format("{}", round(boxArr[i].z)),
          {static_cast<int>(round(item.x)) - 25, static_cast<int>(round(item.y)) - 40},
          1, 1, {0, 255, 0}, 2);

        ostringstream sb2;
        sb2 << "(" << round(boxArr[i].width) << "/" << round(boxArr[i].height) << ")";
        putText(rgbMat, sb2.str(),
                {static_cast<int>(round(item.x)) - 40, static_cast<int>(round(item.y)) - 20},
                1, 1, {0, 0, 255}, 2);
    }
}

void Detector::oneMaskHandle(const PointCloud &pc, vector<Box> &boxes,
                             vector<shared_ptr<const Geometry> > &showList,
                             RecognitionRes &rectRes) {
    if (pc.points_.size() < 100) {
        return;
    }
    //点云降采样
    auto boxPc = *pc.VoxelDownSample(0.001);
    spdlog::debug("点云降采样");
    //showPcl({make_shared<PointCloud>(boxPc)});

    // 过滤大角度的单点到boxPc2
    PointCloud boxPc2;
    Utils::removeZAngleGtCloud(boxPc2, 40, boxPc);
    spdlog::debug("过滤大角度的单点到boxPc2");
    //showPcl({make_shared<PointCloud>(boxPc2)});


    //点云聚类 获取最大点云到boxPc3
    PointCloud boxPc3;
    double eps = 0.03; // 邻域半径
    int min_points = 3; // 构成核心对象的邻域点数下限
    // DBSCAN聚类
    vector<int> labels = boxPc2.ClusterDBSCAN(eps, min_points);
    int mostIndex = Utils::getMostFrequent(labels);
    if (mostIndex == -1) {
        spdlog::error("点云无法聚类");
        return;
    }
    for (int i = 0; i < labels.size(); ++i) {
        if (labels[i] != mostIndex) {
            continue;
        }
        Eigen::Vector3d &mx = boxPc2.points_[i];
        boxPc3.points_.push_back(mx);
    }
    spdlog::debug("聚类完成");
    if (boxPc3.points_.size() < 100) {
        return;
    }
    //showPcl({make_shared<PointCloud>(boxPc3)});

    // 使用RANSAC方法进行平面检测
    auto [plane,pcIdx] = boxPc3.SegmentPlane(0.02, 3, 100);
    auto boxPcPlane = boxPc3.SelectByIndex(pcIdx);
    //showPcl({boxPcPlane});
    spdlog::debug("平面检测");
    Eigen::Vector3d planeVec = {plane[0], plane[1], plane[2]};
    auto xyRotationMatrix = Utils::rotationMatrixBetweenVectors(planeVec, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d bestRotationMatrix = Eigen::Matrix3d::Zero();
    double minArea = 0;
    double angle_c = Utils::getAngle(planeVec, Eigen::Vector3d::UnitX()) * 180 / std::numbers::pi;
    double angle_a = 0;
    for (int i = -50; i < 50; i += 1) {
        auto rpc = boxPc3;
        double r = (numbers::pi / 360) * (static_cast<double>(i) / 10);
        auto zRotationMatrix = Eigen::AngleAxisd(r, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        //合并三个轴的旋转向量
        const Eigen::Matrix3d &xyzRotationMatrix = xyRotationMatrix * zRotationMatrix;
        rpc.Rotate(xyzRotationMatrix, rpc.GetCenter());
        const AxisAlignedBoundingBox aabbBox = rpc.GetAxisAlignedBoundingBox();
        const Eigen::Vector3d &extent = aabbBox.GetExtent();
        double area = extent.x() * extent.y();
        if (area < minArea || minArea == 0) {
            minArea = area;
            bestRotationMatrix = xyzRotationMatrix;
            angle_a = r;
        }
    }

    boxPcPlane->Rotate(bestRotationMatrix, boxPcPlane->GetCenter());
    auto aabb = boxPcPlane->GetAxisAlignedBoundingBox();
    PointCloud boxPc4 = boxPc3;
    boxPc4.Rotate(bestRotationMatrix, boxPc4.GetCenter());
    auto aabb2 = boxPc4.GetAxisAlignedBoundingBox();
    // boxPcPlane->Rotate(bestRotationMatrix, boxPcPlane->GetCenter());
    // auto aabbPlane = boxPcPlane->GetAxisAlignedBoundingBox();
    // auto &max_bound = aabb.max_bound_;
    // max_bound.z() = aabbPlane.max_bound_.z();
    // auto &min_bound = aabb.min_bound_;
    // min_bound.z() = aabbPlane.min_bound_.z();

    Eigen::Vector3d catchPoint = {
        aabb.GetCenter().x(),
        aabb.GetCenter().y() + aabb.GetExtent().y() / 2,
        aabb.GetCenter().z()
    };
    double width = aabb2.GetExtent().x();
    double height = aabb2.GetExtent().y();
    catchPoint = (bestRotationMatrix.inverse() * (catchPoint - aabb.GetCenter())) + aabb.GetCenter();

    spdlog::debug("完成检测，准备返回结果");
    //W=(DP)/F 计算箱子在图片上的识别宽度，高度
    //误差太大
    double boxWidth;
    double boxHeight;
    if (rectRes.rotatedRect.angle < 10) {
        boxWidth = rectRes.rotatedRect.size.width * catchPoint.z() / this->intrinsicRGB.at<float>(0, 0);
        boxHeight = rectRes.rotatedRect.size.height * catchPoint.z() / this->intrinsicRGB.at<float>(1, 1);
    } else {
        boxWidth = rectRes.rotatedRect.size.height * catchPoint.z() / this->intrinsicRGB.at<float>(0, 0);
        boxHeight = rectRes.rotatedRect.size.width * catchPoint.z() / this->intrinsicRGB.at<float>(1, 1);
    }
    if (std::abs(boxWidth - width) > boxWidth * 0.2) {
        spdlog::info("剔除boxWidth={}，width={}", boxWidth, width);
        //return;
    }
    if (std::abs(boxHeight - height) > boxHeight * 0.2) {
        spdlog::info("剔除boxHeight={}，height={}", boxHeight, height);
        //return;
    }
    //加锁
    {
        std::lock_guard<std::mutex> locker(boxesLocker);
        if (this->isShow) {
            const shared_ptr<LineSet> &lineSet = LineSet::CreateFromAxisAlignedBoundingBox(aabb);
            lineSet->Rotate(bestRotationMatrix.inverse(), aabb.GetCenter());
            showList.push_back(lineSet);
            auto pbox = TriangleMesh::CreateBox(0.001, 0.001, 0.001);
            pbox->Translate(catchPoint);
            showList.push_back(pbox);
            //showPcl({lineSet, make_shared<PointCloud>(boxPc3), pbox});
        }
        boxes.push_back({
            catchPoint.x(), catchPoint.y(), catchPoint.z(), width, height,
            angle_a, angle_c
        });
    }
    spdlog::debug("完成检测，返回结果");
}
