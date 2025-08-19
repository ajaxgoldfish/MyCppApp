#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "BoxPosePipeline.h"

namespace fs = std::filesystem;

int main() {
    // ====== 1) 初始化 Pipeline（只初始化一次）======
    BoxPosePipeline::Options opt;
    opt.model_path = "models/end2end.onnx";
    opt.calib_path = "config/params.xml";
    opt.score_thr  = 0.7f;
    opt.mask_thr   = 0.5f;
    opt.paint_masks_on_vis = true;

    BoxPosePipeline pipeline(opt);
    if (!pipeline.initialize()) {
        std::cerr << "[FATAL] pipeline.initialize() 失败\n";
        return -1;
    }

    // ====== 2) 遍历 res 目录 ======
    const fs::path root = "res";
    if (!fs::exists(root) || !fs::is_directory(root)) {
        std::cerr << "[FATAL] 目录不存在: " << root << "\n";
        return -1;
    }

    size_t ok_cnt = 0, skip_cnt = 0, err_cnt = 0;

    for (const auto& dirEnt : fs::directory_iterator(root)) {

        if (!dirEnt.is_directory()) continue;

        const fs::path caseDir = dirEnt.path();
        const fs::path rgbPath = caseDir / "rgb.jpg";
        const fs::path pcdPath = caseDir / "pcAll.pcd";

        if (!fs::exists(rgbPath) || !fs::exists(pcdPath)) {
            std::cout << "[SKIP] 缺少文件: " << caseDir << " （需要 rgb.jpg 与 pcAll.pcd）\n";
            ++skip_cnt;
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();  // ⏱ 开始计时

        // ====== 3) 读取输入 ======
        cv::Mat rgb = cv::imread(rgbPath.string(), cv::IMREAD_COLOR);
        if (rgb.empty()) {
            std::cout << "[ERR ] 无法读取图片: " << rgbPath << "\n";
            ++err_cnt; continue;
        }

        // —— 用 PCL 读取 .pcd（仅 XYZ）再转 open3d::geometry::PointCloud —— //
        open3d::geometry::PointCloud pc;
        bool ok = false;

        pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdPath.string(), cloud_xyz) == 0 &&
            !cloud_xyz.empty()) {
            pc.Clear();
            pc.points_.reserve(cloud_xyz.size());
            for (const auto& p : cloud_xyz.points) {
                pc.points_.emplace_back(double(p.x), double(p.y), double(p.z));
            }
            ok = true;
            }

        if (!ok || pc.points_.empty()) {
            std::cerr << "[ERR ] PCL 读取失败或点云为空: " << pcdPath << "\n";
            ++err_cnt;
            continue;
        }

        spdlog::info("read");

        // ====== 4) 执行 Pipeline ======
        std::vector<BoxPoseResult> results;
        cv::Mat vis;
        bool ok1 = pipeline.run(rgb, pc, results, &vis);
        if (!ok1 || vis.empty()) {
            std::cout << "[ERR ] pipeline.run 失败: " << caseDir << "\n";
            ++err_cnt; continue;
        }

        // ====== 5) 写结果到同目录 ======
        const fs::path outPath = caseDir / "vis_on_orig.jpg";
        if (!cv::imwrite(outPath.string(), vis)) {
            std::cout << "[ERR ] 写文件失败: " << outPath << "\n";
            ++err_cnt; continue;
        }

        auto t1 = std::chrono::steady_clock::now();  // ⏱ 结束计时
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[ OK ] " << caseDir.filename().string()
                  << " -> " << outPath
                  << "，目标数=" << results.size()
                  << "，耗时=" << elapsed_ms << " ms\n";

        ++ok_cnt;
    }


    std::cout << "完成：OK=" << ok_cnt
              << ", SKIP=" << skip_cnt
              << ", ERR="  << err_cnt << "\n";
    return (err_cnt > 0 && ok_cnt == 0) ? -1 : 0;
}
