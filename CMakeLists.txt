cmake_minimum_required(VERSION 3.29)
project(MyCppApp)

# 使用最新 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 先添加 spdlog 库
add_subdirectory(external/spdlog)
# 再测试 spdlog 库
add_subdirectory(tests/spdlog_test)


# 添加 SQLite 测试
add_subdirectory(tests/sqlite_test)

# 添加 nlohmann_json 测试
add_subdirectory(tests/nlohmann_test)

# 添加你的可执行文件
add_executable(MyCppApp main.cpp
        tests/nlohmann_test/json_test.cpp)

# 链接 spdlog 库
target_link_libraries(MyCppApp PRIVATE spdlog::spdlog)


