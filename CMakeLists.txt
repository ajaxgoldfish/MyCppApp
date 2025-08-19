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

# spdlog
add_subdirectory(tests/spdlog_test)
# sqlite3
add_subdirectory(tests/sqlite_test)
# nlohmann_json
add_subdirectory(tests/nlohmann_test)
# opencv
add_subdirectory(tests/opencv_test)
# cpp-httplib
add_subdirectory(tests/httplib_test/server)
add_subdirectory(tests/httplib_test/client)
# pcl
add_subdirectory(tests/pcl_test)


# MyCppApp
add_executable(MyCppApp src/main.cpp
                        src/LanxinCamera.cpp
                        src/MaskRCNNRunner.cpp
                        src/FusionGeometry.cpp
        src/BoxPosePipeline.cpp
        include/BoxPosePipeline.h
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