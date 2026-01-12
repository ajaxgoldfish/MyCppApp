#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>
#include <spdlog/spdlog.h>
#include "cpu_library.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

int main() {

    // 设置控制台为UTF-8编码，解决中文乱码问题
#ifdef _WIN32
    // Windows系统设置
    SetConsoleOutputCP(CP_UTF8);           // 设置控制台输出代码页为UTF-8
    SetConsoleCP(CP_UTF8);                 // 设置控制台输入代码页为UTF-8
    // 注意：不使用_setmode，因为它可能与spdlog冲突
#endif

    // ====== 1) 初始化 CPU Library ======
    bool isDebug = true;  // 可以设置为 false 来减少调试信息
    // 自动读取 config/params.xml 配置
    int init_result = bs_yzx_init(isDebug);
    if (init_result != 0) {
        spdlog::critical("bs_yzx_init 失败，错误码: {}", init_result);
        return -1;
    }
    // ====== 2) 遍历 res 目录，处理每个任务 ======
    const fs::path root = "res";
    if (!fs::exists(root) || !fs::is_directory(root)) {
        spdlog::critical("目录不存在: {}", root.string());
        return -1;
    }

    size_t ok_cnt = 0, skip_cnt = 0, err_cnt = 0;

    for (const auto& dirEnt : fs::directory_iterator(root)) {
        if (!dirEnt.is_directory()) continue;

        const fs::path caseDir = dirEnt.path();
        const std::string dirName = caseDir.filename().string();
        
        // 尝试解析目录名为 taskId（数字）
        int taskId;
        try {
            taskId = std::stoi(dirName);
        } catch (const std::exception& e) {
            spdlog::warn("[SKIP] 目录名不是数字: {}", dirName);
            ++skip_cnt;
            continue;
        }

        // 检查必需的文件是否存在
        const fs::path rgbPath = caseDir / "rgb_orig.jpg";
        const fs::path pcdPath = caseDir / "cloud_orig.pcd";
        
        if (!fs::exists(rgbPath) || !fs::exists(pcdPath)) {
            spdlog::warn("[SKIP] 缺少必需文件: {} （需要 rgb.jpg 与 pcAll.pcd）", dirName);
            ++skip_cnt;
            continue;
        }

        // ====== 3) 调用目标检测函数 ======
        constexpr int MAX_BOXES = 100;  // 与 YZX_MAX_BOX 保持一致
        zzb::Box boxArr[MAX_BOXES];
        
        // 初始化数组
        for (int i = 0; i < MAX_BOXES; ++i) {
            boxArr[i] = {};  // 零初始化
        }

        auto t0 = std::chrono::steady_clock::now();
        // 传入 Y 轴边界值 (例如: 1000, -1200)
        int detectionResult = bs_yzx_object_detection_lanxin(taskId, boxArr, 1000.0f, -1600.0f);
        auto t1 = std::chrono::steady_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (detectionResult < 0) {
            spdlog::error("目标检测失败，taskId={}, 错误码={}", taskId, detectionResult);
            ++err_cnt;
            continue;
        }

        // ====== 4) 输出结果 ======
        int detectedCount = detectionResult;  // 返回值是检测到的目标数量

        spdlog::info("[ OK ] taskId={}, 检测到 {} 个目标，耗时={:.3f} ms",
                     taskId, detectedCount, elapsed_ms);

        ++ok_cnt;
    }

    return (err_cnt > 0 && ok_cnt == 0) ? -1 : 0;
}
