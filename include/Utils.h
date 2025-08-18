//
// Created by zwj on 2024/5/7.
//
#ifndef YZX_VISION_UTILS_H
#define YZX_VISION_UTILS_H

#include <string>
#include <vector>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


using namespace std;
using namespace open3d::geometry;
using namespace cv;

class Utils {
public:

    static int getMostFrequent(vector<int> &vec);

    static void filterPointsByBoundingBox(open3d::geometry::PointCloud &pcd,
                                          const Eigen::Vector3d &min_point,
                                          const Eigen::Vector3d &max_point,
                                          bool flag = false);

    // static int readPointCloudFromLas(PointCloud &pc, const string &filePath);

    static Eigen::Matrix3d rotationMatrixBetweenVectors(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);

    static void copyTo(PointCloud &cloud, const Mat &src, const Mat &mask);

    static string getDateStr();

    static Mat imageFusion(Mat &rgbImage1, Mat &rgbImage2, int overlap, int dividingLine);

    static void removeZAngleGtCloud(PointCloud &boxPc2, float angle, PointCloud &boxPc);

    static Point3f pointTransform(Point3f p, const Mat &mat);

    static double getAngle(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);

    static void removeYAngleGtCloud(PointCloud &boxPc2, float angle, PointCloud &boxPc);

    static void removeXAngleGtCloud(PointCloud &distPc, float angle, PointCloud &srcPc);
};

#endif //YZX_VISION_UTILS_H
