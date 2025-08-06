find_package(unofficial-sqlite3 CONFIG REQUIRED)

# 声明一个可执行文件
add_executable(sqlite_test
        sqlite_test.cpp
)

target_link_libraries(sqlite_test PRIVATE unofficial::sqlite3::sqlite3)
