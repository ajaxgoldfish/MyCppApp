//
// Created by zhangzongbo on 2025/8/6.
//
// tests/pcl_test/test_pcl.cpp
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <iostream>

int main() {
    // 创建点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 填充点云数据（随机）
    cloud->width = 100;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (auto &point : cloud->points) {
        point.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        point.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        point.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }

    // 保存到 PCD 文件
    if (pcl::io::savePCDFileASCII("random_points.pcd", *cloud) == -1) {
        std::cerr << "" << std::endl;
        return -1;
    }

    std::cout << " " << cloud->points.size() << " " << std::endl;

    // 从文件中读取回来
    pcl::PointCloud<pcl::PointXYZ>::Ptr loadedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile("random_points.pcd", *loadedCloud) == -1) {
        std::cerr << "" << std::endl;
        return -2;
    }

    std::cout << "" << loadedCloud->points.size() << " " << std::endl;

    return 0;
}
