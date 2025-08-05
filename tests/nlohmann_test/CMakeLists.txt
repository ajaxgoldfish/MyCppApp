# test/json_test/CMakeLists.txt

add_executable(nlohmann_test json_test.cpp)

# 设置 JSON 头文件路径
target_include_directories(nlohmann_test PRIVATE
        ${PROJECT_SOURCE_DIR}/external/nlohmann/include/include
)
