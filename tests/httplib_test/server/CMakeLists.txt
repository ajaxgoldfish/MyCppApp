find_package(httplib CONFIG REQUIRED)

add_executable(http_server server.cpp)
target_link_libraries(http_server PRIVATE httplib::httplib)