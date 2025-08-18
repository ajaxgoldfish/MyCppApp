# tests/opencv_test/CMakeLists.txt

add_executable(opencv_test opencv_test.cpp)

find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(opencv_test PRIVATE ${OpenCV_LIBS})