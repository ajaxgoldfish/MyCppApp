cmake_minimum_required(VERSION 3.29)
project(MyCppApp)

# 使用最新 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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


# MyCppApp
add_executable(MyCppApp src/main.cpp)

