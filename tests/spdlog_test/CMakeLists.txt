find_package(spdlog CONFIG REQUIRED)
# 声明一个可执行文件
add_executable(spdlog_test spdlog_test.cpp)
# 链接 spdlog 库
target_link_libraries(spdlog_test PRIVATE spdlog::spdlog)
