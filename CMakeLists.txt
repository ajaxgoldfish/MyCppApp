cmake_minimum_required(VERSION 3.29)
project(MyCppApp)

# 统一使用 C++23，保证主库和测试程序在同一语言标准下编译。
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# 这里集中维护头文件目录，主 DLL 和测试程序共享同一套依赖路径。
set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/lanxin/include
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-gpu-1.25.1/include
    ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/include
)

# 这里集中维护第三方库，保证检测库和测试程序链接到完全一致的运行依赖。
set(COMMON_LIBS
    ${OpenCV_LIBS}
    spdlog::spdlog
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib
    ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avcodec.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avdevice.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avfilter.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avformat.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avutil.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swresample.lib
    ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swscale.lib
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-win-x64-gpu-1.25.1/lib/onnxruntime.lib
)

# 生成对外使用的视觉检测 DLL，内部完成相机取数、ONNX 推理和 3D 位姿解算。
add_library(yzx_vision_zzb_gpu SHARED
    src/cpu_library.cpp
    src/LanxinCamera.cpp
)

target_include_directories(yzx_vision_zzb_gpu PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${COMMON_LIBS})

# 测试程序直接调用 DLL 接口，用于验证相机、标定和检测流程是否能完整跑通。
add_executable(test_gpu_dll src/test_gpu_dll.cpp)

target_include_directories(test_gpu_dll PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_gpu_dll PRIVATE yzx_vision_zzb_gpu ${COMMON_LIBS})

set_target_properties(test_gpu_dll PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 纯相机 SDK 采集程序：直接保存 RGB BMP 和对齐深度 PGM，只链接蓝芯相机库。
add_executable(camera_capture
    src/camera_capture.cpp
)

target_include_directories(camera_capture PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/lanxin/include
)
target_link_libraries(camera_capture PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib
)

if(MSVC)
    target_compile_options(camera_capture PRIVATE /utf-8)
endif()

set_target_properties(camera_capture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
