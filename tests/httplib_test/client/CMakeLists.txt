find_package(httplib CONFIG REQUIRED)
add_executable(http_client client.cpp)
target_link_libraries(http_client PRIVATE httplib::httplib)