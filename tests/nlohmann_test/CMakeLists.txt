# test/json_test/CMakeLists.txt

find_package(nlohmann_json CONFIG REQUIRED)

add_executable(nlohmann_test json_test.cpp)

target_link_libraries(nlohmann_test PRIVATE nlohmann_json::nlohmann_json)
