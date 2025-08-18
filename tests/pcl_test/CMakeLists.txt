# tests/pcl_test/CMakeLists.txt
find_package(PCL CONFIG REQUIRED)
add_executable(pcl_test pcl_test.cpp)
target_link_libraries(pcl_test PRIVATE ${PCL_LIBRARIES})
target_include_directories(pcl_test PRIVATE ${PCL_INCLUDE_DIRS})
