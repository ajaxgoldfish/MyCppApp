
//
// Created by zwj on 2024/5/7.
//


#include "Utils.h"
//#include <liblas/reader.hpp>
#include <fstream>
#include <numbers>

using namespace open3d::geometry;

string Utils::getDateStr() {
    stringstream ss;
    chrono::system_clock::time_point a = chrono::system_clock::now();      //时间点可以做差
    time_t t1 = chrono::system_clock::to_time_t(a);                  //time_t可以格式化
    ss << std::put_time(localtime(&t1), "%Y-%m-%d %H-%M-%S");
    return ss.str();
}


// int Utils::readPointCloudFromLas(PointCloud &pc, const string &filePath) {
//     std::ifstream fileStream(filePath, std::ios::in | std::ios::binary);
//     liblas::Reader reader(fileStream);
//     if (!fileStream.is_open()) {
//         return -1;
//     }
//     size_t nbPoints = reader.GetHeader().GetPointRecordsCount();
//     pc.points_.reserve(nbPoints);
//     while (reader.ReadNextPoint()) {
//         // 获取las数据的x，y，z信息
//         const liblas::Point &point = reader.GetPoint();
//         auto i = point.GetIntensity();
//         if (i <= 20) {
//             continue;
//         }
//         pc.points_.emplace_back(point.GetX(), point.GetY(), point.GetZ());
//     }
//     return 0;
// }

int Utils::getMostFrequent(vector<int> &vec) {
    std::unordered_map<int, int> count_map;
    int max_count = 0;
    int most_frequent = -1;

    for (int num: vec) {
        if (num == -1) {
            continue;
        }
        ++count_map[num];
        if (count_map[num] > max_count) {
            max_count = count_map[num];
            most_frequent = num;
        }
    }
    return most_frequent;
}

void Utils::filterPointsByBoundingBox(open3d::geometry::PointCloud &pcd,
                                      const Eigen::Vector3d &min_point,
                                      const Eigen::Vector3d &max_point,
                                      bool flag) {
    PointCloud pc;
    pc.points_.reserve(pcd.points_.size());
    for (auto &point: pcd.points_) {
        bool inside = (point.x() >= min_point.x() && point.x() <= max_point.x() &&
                       point.y() >= min_point.y() && point.y() <= max_point.y() &&
                       point.z() >= min_point.z() && point.z() <= max_point.z());
        if (flag) {
            if (!inside) {
                pc.points_.push_back(point);
            }
        } else {
            if (inside) {
                pc.points_.push_back(point);
            }
        }
    }
    pcd = pc;
}


void Utils::copyTo(PointCloud &cloud, const Mat &src, const Mat &mask) {
    auto *data = (float *) src.data;
    int size = src.rows * src.cols;
    for (int i = 0; i < size; ++i) {
        int baseIndex = i * 3; // 当前像素的第一个通道的索引
        if (mask.data[i] == 0) {
            continue;
        }
        if (data[baseIndex + 0] == 0 ||
            data[baseIndex + 1] == 0 ||
            data[baseIndex + 2] == 0) {
            continue;
        }
        cloud.points_.emplace_back(data[baseIndex + 0], data[baseIndex + 1], data[baseIndex + 2]);
        data[baseIndex + 0] = 0;
        data[baseIndex + 1] = 0;
        data[baseIndex + 2] = 0;
    }
}

// 计算从向量v1到v2的旋转矩阵
double Utils::getAngle(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    // 确保向量为单位向量
    const Eigen::Vector3d u1 = v1.normalized();
    const Eigen::Vector3d u2 = v2.normalized();
    // 计算旋转角（使用反余弦）
    return acos(u1.dot(u2));
}

// 计算从向量v1到v2的旋转矩阵
Eigen::Matrix3d Utils::rotationMatrixBetweenVectors(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    // 确保向量为单位向量
    Eigen::Vector3d u1 = v1.normalized();
    Eigen::Vector3d u2 = v2.normalized();

    // 计算旋转轴（叉积）
    Eigen::Vector3d axis = u1.cross(u2);

    // 如果向量已对齐或反向对齐，则无需旋转
    if (axis.norm() == 0) {
        if (u1.dot(u2) < 0) { // 反向，手动设置180度旋转轴
            axis = Eigen::Vector3d::UnitZ();
        } else {
            return Eigen::Matrix3d::Identity(); // 已经对齐
        }
    } else {
        axis.normalize(); // 确保是单位向量
    }

    // 计算旋转角（使用反余弦）
    double angleRad = acos(u1.dot(u2));

    // 使用罗德里格斯公式生成旋转矩阵
    return Eigen::AngleAxisd(angleRad, axis).matrix();
}


static cv::Mat calWeight(int d, double k) {
    cv::Mat x(1, d, CV_32F);
    for (int i = 0; i < d; i++) {
        x.at<float>(0, i) = (i - d / 2.0) / 2.0;
    }

    cv::Mat y(1, d, CV_32F);
    for (int i = 0; i < d; i++) {
        y.at<float>(0, i) = 1 / (1 + exp(-k * x.at<float>(0, i)));
    }

    return y;
}

cv::Mat Utils::imageFusion(cv::Mat &rgbImage1, cv::Mat &rgbImage2, int overlap, int dividingLine) {

    //再拼接彩色图,使用加权融合，使拼接缝看起来更自然
    const Rect2i &rect1 = cv::Rect(0, 0, rgbImage1.cols, dividingLine + overlap / 2);
    const Rect2i &rect2 = cv::Rect(0, dividingLine - overlap / 2,
                                   rgbImage2.cols, rgbImage2.rows - (dividingLine - overlap / 2));

    cv::Mat img1 = rgbImage1(rect1);
    cv::Mat img2 = rgbImage2(rect2);

    int row1 = img1.rows;
    int col1 = img1.cols;
    int row2 = img2.rows;
    int col2 = img2.cols;

    cv::Mat wMat = calWeight(overlap, 0.05);

    cv::Mat img_new(row1 + row2 - overlap, col1, img1.type());
    img1.copyTo(img_new(cv::Rect(0, 0, col1, row1)));

    cv::Mat w_expand(overlap, col1, wMat.type());
    for (int i = 0; i < overlap; i++) {
        for (int j = 0; j < col1; j++) {
            w_expand.at<float>(i, j) = wMat.at<float>(0, i);
        }
    }
    for (int i = 0; i < overlap; i++) {
        for (int j = 0; j < col1; j++) {
            float w = w_expand.at<float>(i, j);
            img_new.at<cv::Vec3b>(row1 - overlap + i, j) =
                    (1 - w) * img1.at<cv::Vec3b>(row1 - overlap + i, j) +
                    w * img2.at<cv::Vec3b>(i, j);
        }
    }

    img2(cv::Rect(0, overlap, col2, row2 - overlap))
            .copyTo(img_new(cv::Rect(0, row1, col2, row2 - overlap)));
    return img_new;
}

Point3f Utils::pointTransform(Point3f p, const Mat &mat) {
    Mat rgbPoint = (cv::Mat_<float>(4, 1) << p.x, p.y, p.z, 1);
    Mat robotPoint = mat * rgbPoint;
    return Point3f{
            robotPoint.at<float>(0, 0),
            robotPoint.at<float>(1, 0),
            robotPoint.at<float>(2, 0)
    };
}

void Utils::removeZAngleGtCloud(PointCloud &boxPc2, float angle, PointCloud &boxPc) {
    double fNormalAngleRange_rad = angle * numbers::pi / 180.0;
    if (!boxPc.HasNormals()) {
        boxPc.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }
    for (int i = 0; i < boxPc.normals_.size(); ++i) {
        auto n = boxPc.normals_[i];
        double normal_x = n.x();
        double normal_y = n.y();
        double normal_z = n.z();
        //平面与向量的角度
        double z_normal_angle = abs(atan((sqrt(normal_x * normal_x + normal_y * normal_y) / normal_z)));
        if (z_normal_angle < fNormalAngleRange_rad) {
            auto p = boxPc.points_[i];
            boxPc2.points_.push_back(p);
        }
    }
}

void Utils::removeXAngleGtCloud(PointCloud &distPc, float angle, PointCloud &srcPc) {
    double fNormalAngleRange_rad = angle * numbers::pi / 180.0;
    if (!srcPc.HasNormals()) {
        srcPc.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }
    for (int i = 0; i < srcPc.normals_.size(); ++i) {
        auto n = srcPc.normals_[i];
        double normal_x = n.x();
        double normal_y = n.y();
        double normal_z = n.z();
        //平面与向量的角度
        double x_normal_angle = abs(atan((sqrt(normal_z * normal_z + normal_y * normal_y) / normal_x)));
        if (x_normal_angle < fNormalAngleRange_rad) {
            auto p = srcPc.points_[i];
            distPc.points_.push_back(p);
        }
    }
}

void Utils::removeYAngleGtCloud(PointCloud &boxPc2, float angle, PointCloud &boxPc) {
    double fNormalAngleRange_rad = angle * numbers::pi / 180.0;
    if (!boxPc.HasNormals()) {
        boxPc.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }
    for (int i = 0; i < boxPc.normals_.size(); ++i) {
        auto n = boxPc.normals_[i];
        double normal_x = n.x();
        double normal_y = n.y();
        double normal_z = n.z();
        //平面与向量的角度
        double y_normal_angle = abs(atan((sqrt(normal_x * normal_x + normal_z * normal_z) / normal_y)));
        if (y_normal_angle < fNormalAngleRange_rad) {
            auto p = boxPc.points_[i];
            boxPc2.points_.push_back(p);
        }
    }
}