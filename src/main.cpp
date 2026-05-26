#include <chrono>
#include <spdlog/spdlog.h>
#include "cpu_library.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    const bool isDebug = true;
    const int init_result = bs_yzx_init(isDebug);
    if (init_result != 0) {
        spdlog::critical("bs_yzx_init failed, error code: {}", init_result);
        return -1;
    }

    constexpr int MAX_BOXES = 100;
    zzb::Box boxArr[MAX_BOXES]{};

    const auto t0 = std::chrono::steady_clock::now();
    const int detection_result = bs_yzx_object_detection_lanxin(boxArr);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (detection_result < 0) {
        spdlog::error("Object detection failed, error code={}", detection_result);
        return -1;
    }

    spdlog::info("[ OK ] detected {} targets, elapsed={:.3f} ms",
                 detection_result, elapsed_ms);
    return 0;
}
