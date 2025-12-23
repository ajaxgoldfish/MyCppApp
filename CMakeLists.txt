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

# Common settings
set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/lanxin/include
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-gpu/include
    ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/include
)

set(COMMON_LIBS
    ${OpenCV_LIBS}
    spdlog::spdlog
    ${PCL_LIBRARIES}
    Open3D::Open3D
    ${PROJECT_SOURCE_DIR}/third_party/lanxin/lib/win_x64/LxCameraApi.lib
    ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avcodec.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avdevice.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avfilter.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avformat.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/avutil.lib
        ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swresample.lib
    ${PROJECT_SOURCE_DIR}/third_party/ffmpeg/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1/lib/swscale.lib
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu/lib/onnxruntime.lib
)

# 1. MyCppApp (Unified executable)
# Compiles sources directly
add_executable(MyCppApp 
    src/main.cpp
    src/cpu_library.cpp
    src/LanxinCamera.cpp
)

target_include_directories(MyCppApp PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(MyCppApp PRIVATE ${COMMON_LIBS})

# 2. Shared Library (Unified DLL)
# Previously yzx_vision_zzb_gpu, now just yzx_vision_zzb or keep name for compat
add_library(yzx_vision_zzb_gpu SHARED
    src/cpu_library.cpp
    src/LanxinCamera.cpp
)

target_include_directories(yzx_vision_zzb_gpu PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(yzx_vision_zzb_gpu PRIVATE ${COMMON_LIBS})

# 3. Test Executable (Links against DLL)
add_executable(test_gpu_dll src/test_gpu_dll.cpp)

target_include_directories(test_gpu_dll PRIVATE ${COMMON_INCLUDE_DIRS})
# Link against the shared library
target_link_libraries(test_gpu_dll PRIVATE yzx_vision_zzb_gpu ${COMMON_LIBS})

set_target_properties(test_gpu_dll PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
