
# 声明一个可执行文件
add_executable(sqlite_test
        sqlite_test.cpp
        ${PROJECT_SOURCE_DIR}/external/sqlite/sqlite3.c
)

# 设置头文件路径
target_include_directories(sqlite_test PRIVATE
        ${PROJECT_SOURCE_DIR}/external/sqlite
)
