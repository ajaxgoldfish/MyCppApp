cmake_minimum_required(VERSION 3.29)
project(MyCppApp)

# 使用最新 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(PCL CONFIG REQUIRED)

set(Open3D_DIR "${CMAKE_SOURCE_DIR}/third_party/open3d/open3d-devel-windows-amd64-0.19.0/CMake")
find_package(Open3D CONFIG REQUIRED)


# MyCppApp
add_executable(MyCppApp src/main.cpp
        src/cpu_library.cpp
)



target_include_directories(MyCppApp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(MyCppApp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/lanxin/include)
target_include_directories(MyCppApp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
target_include_directories(MyCppApp PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/include)



target_link_libraries(MyCppApp PRIVATE ${OpenCV_LIBS})
target_link_libraries(MyCppApp PRIVATE spdlog::spdlog)
target_link_libraries(MyCppApp PRIVATE ${PCL_LIBRARIES})
target_link_libraries(MyCppApp PRIVATE Open3D::Open3D)
target_link_libraries(MyCppApp PRIVATE ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib)
target_link_libraries(MyCppApp PRIVATE ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avcodec.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avdevice.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avfilter.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avformat.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avutil.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swresample.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swscale.lib)
target_link_libraries(MyCppApp PRIVATE ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib)


# 生成动态库
add_library(yzx_vision_zzb_gpu SHARED
        src/gpu_library.cpp
        src/LanxinCamera.cpp)

target_include_directories(yzx_vision_zzb_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(yzx_vision_zzb_gpu PRIVATE ${CMAKE_SOURCE_DIR}/third_party/lanxin/include)
target_include_directories(yzx_vision_zzb_gpu PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-gpu/include)
target_include_directories(yzx_vision_zzb_gpu PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/include)


target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${OpenCV_LIBS})
target_link_libraries(yzx_vision_zzb_gpu PRIVATE spdlog::spdlog)
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${PCL_LIBRARIES})
target_link_libraries(yzx_vision_zzb_gpu PRIVATE Open3D::Open3D)
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib)
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avcodec.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avdevice.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avfilter.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avformat.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avutil.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swresample.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swscale.lib)
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu/lib/onnxruntime.lib)


# 测试 GPU DLL 的可执行文件
add_executable(test_gpu_dll src/test_gpu_dll.cpp)


target_include_directories(test_gpu_dll PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_gpu_dll PRIVATE ${CMAKE_SOURCE_DIR}/third_party/lanxin/include)
target_include_directories(test_gpu_dll PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-gpu/include)
target_include_directories(test_gpu_dll PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/include)

target_link_libraries(test_gpu_dll PRIVATE yzx_vision_zzb_gpu)
target_link_libraries(test_gpu_dll PRIVATE ${OpenCV_LIBS})
target_link_libraries(test_gpu_dll PRIVATE ${OpenCV_LIBS})
target_link_libraries(test_gpu_dll PRIVATE spdlog::spdlog)
target_link_libraries(test_gpu_dll PRIVATE ${PCL_LIBRARIES})
target_link_libraries(test_gpu_dll PRIVATE Open3D::Open3D)
target_link_libraries(test_gpu_dll PRIVATE ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib)
target_link_libraries(test_gpu_dll PRIVATE ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avcodec.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avdevice.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avfilter.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avformat.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avutil.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swresample.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swscale.lib)
target_link_libraries(test_gpu_dll PRIVATE ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu/lib/onnxruntime.lib)

set_target_properties(test_gpu_dll PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)