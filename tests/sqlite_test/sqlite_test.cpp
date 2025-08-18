//
// Created by zhangzongbo on 2025/8/5.
//
#include <iostream>
#include <sqlite3.h>

void check(int rc, char* errMsg) {
    if (rc != SQLITE_OK) {
        std::cerr << "SQLite error: " << (errMsg ? errMsg : sqlite3_errstr(rc)) << std::endl;
        sqlite3_free(errMsg);
        exit(1);
    }
}

int main() {
    sqlite3* db = nullptr;
    char* errMsg = nullptr;

    // 1. 打开数据库（不存在则创建）
    int rc = sqlite3_open("test.db", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Can't open DB: " << sqlite3_errmsg(db) << std::endl;
        return -1;
    }

    // 2. 创建表
    const char* createSQL = "CREATE TABLE IF NOT EXISTS person (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);";
    rc = sqlite3_exec(db, createSQL, nullptr, nullptr, &errMsg);
    check(rc, errMsg);

    // 3. 插入一条记录
    const char* insertSQL = "INSERT INTO person (name, age) VALUES ('张三', 28);";
    rc = sqlite3_exec(db, insertSQL, nullptr, nullptr, &errMsg);
    check(rc, errMsg);
    std::cout << "Inserted one record." << std::endl;

    // 4. 查询所有记录
    std::cout << "Reading all records:" << std::endl;
    const char* selectSQL = "SELECT id, name, age FROM person;";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, selectSQL, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Prepare failed: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return -1;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const unsigned char* name = sqlite3_column_text(stmt, 1);
        int age = sqlite3_column_int(stmt, 2);
        std::cout << "id: " << id << ", name: " << name << ", age: " << age << std::endl;
    }
    sqlite3_finalize(stmt);

    // 5. 更新记录
    const char* updateSQL = "UPDATE person SET age = 30 WHERE name = '张三';";
    rc = sqlite3_exec(db, updateSQL, nullptr, nullptr, &errMsg);
    check(rc, errMsg);
    std::cout << "Updated 张三's age to 30." << std::endl;

    // 6. 删除记录
    const char* deleteSQL = "DELETE FROM person WHERE name = '张三';";
    rc = sqlite3_exec(db, deleteSQL, nullptr, nullptr, &errMsg);
    check(rc, errMsg);
    std::cout << "Deleted 张三." << std::endl;

    // 7. 关闭数据库
    sqlite3_close(db);

    return 0;
}