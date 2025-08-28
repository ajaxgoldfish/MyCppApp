#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <spdlog/spdlog.h>
// 如果需要输出到文件，可额外引入：#include <spdlog/sinks/basic_file_sink.h>
#include "BoxPosePipeline.h"

namespace fs = std::filesystem;

int main() {
    // ====== 0) spdlog 基础配置（可按需调整）======
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // 彩色等级 + 时间
    spdlog::set_level(spdlog::level::info);                    // 运行时日志门槛
    spdlog::flush_on(spdlog::level::err);                      // 遇到 error 立刻 flush

    // ====== 1) 初始化 Pipeline（只初始化一次）======
    BoxPosePipeline::Options opt;
    opt.model_path = "models/end2end.onnx";
    opt.calib_path = "config/params.xml";
    opt.score_thr  = 0.7f;
    opt.mask_thr   = 0.5f;
    opt.paint_masks_on_vis = true;

    BoxPosePipeline pipeline(opt);
    if (!pipeline.initialize()) {
        spdlog::critical("pipeline.initialize() 失败");
        return -1;
    }

    // ====== 2) 遍历 res 目录 ======
    const fs::path root = "res";
    if (!fs::exists(root) || !fs::is_directory(root)) {
        spdlog::critical("目录不存在: {}", root.string());
        return -1;
    }

    size_t ok_cnt = 0, skip_cnt = 0, err_cnt = 0;

    for (const auto& dirEnt : fs::directory_iterator(root)) {
        if (!dirEnt.is_directory()) continue;

        const fs::path caseDir = dirEnt.path();
        const fs::path rgbPath = caseDir / "rgb.jpg";
        const fs::path pcdPath = caseDir / "pcAll.pcd";

        if (!fs::exists(rgbPath) || !fs::exists(pcdPath)) {
            spdlog::warn("[SKIP] 缺少文件: {} （需要 rgb.jpg 与 pcAll.pcd）", caseDir.string());
            ++skip_cnt;
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();  // ⏱ 开始计时

        // ====== 3) 读取输入 ======
        cv::Mat rgb = cv::imread(rgbPath.string(), cv::IMREAD_COLOR);
        if (rgb.empty()) {
            spdlog::error("无法读取图片: {}", rgbPath.string());
            ++err_cnt;
            continue;
        }

        open3d::geometry::PointCloud pc;
        if (!open3d::io::ReadPointCloud(pcdPath.string(), pc) || pc.points_.empty()) {
            spdlog::error("点云读取失败或为空: {}", pcdPath.string());
            ++err_cnt;
            continue;
        }

        // ====== 4) 执行 Pipeline ======
        std::vector<BoxPoseResult> results;
        cv::Mat vis;
        bool ok1 = pipeline.run(rgb, pc, results, &vis);
        if (!ok1 || vis.empty()) {
            spdlog::error("pipeline.run 失败: {}", caseDir.string());
            ++err_cnt;
            continue;
        }

        // 输出结果到控制台
        for (size_t i = 0; i < results.size(); ++i) {
            std::ostringstream oss;
            oss << results[i].Rw;   // Eigen::Matrix 支持 operator<< 到 ostream
            spdlog::info("Rw = \n{}", oss.str());

        }

        // ====== 5) 写结果到同目录 ======
        const fs::path outPath = caseDir / "vis_on_orig.jpg";
        if (!cv::imwrite(outPath.string(), vis)) {
            spdlog::error("写文件失败: {}", outPath.string());
            ++err_cnt;
            continue;
        }

        auto t1 = std::chrono::steady_clock::now();  // ⏱ 结束计时
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        spdlog::info("[ OK ] {} -> {}，目标数={}，耗时={:.3f} ms",
                     caseDir.filename().string(),
                     outPath.string(),
                     results.size(),
                     elapsed_ms);

        ++ok_cnt;
    }

    spdlog::info("完成：OK={}, SKIP={}, ERR={}", ok_cnt, skip_cnt, err_cnt);
    return (err_cnt > 0 && ok_cnt == 0) ? -1 : 0;
}
