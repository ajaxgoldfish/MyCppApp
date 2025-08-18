//
// Created by zhangzongbo on 2025/8/5.
//
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <windows.h>


int main() {
    // 设置控制台为 UTF-8 编码
    SetConsoleOutputCP(CP_UTF8);

    // 创建一个彩色控制台日志器（自动带时间戳）
    auto console = spdlog::stdout_color_mt("console");

    console->info("欢迎使用 spdlog!");
    console->warn("这是一个警告信息");
    console->error("发生错误，代码：{}", 404);

    // 日志等级过滤测试
    spdlog::set_level(spdlog::level::debug); // 默认 info 级别以下不会输出
    spdlog::debug("这是调试信息");

    // 创建一个文件日志器（日志输出到 logs/test.log）
    auto file_logger = spdlog::basic_logger_mt("file_logger", "logs/test.log");
    file_logger->info("日志写入文件成功");

    // 你也可以直接使用全局日志器（默认 console）
    spdlog::info("使用默认日志器输出");

    return 0;
}
