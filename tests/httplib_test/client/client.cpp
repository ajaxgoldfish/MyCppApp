//
// Created by zhangzongbo on 2025/8/6.
//
#include <httplib.h>
#include <iostream>

int main() {
    httplib::Client cli("localhost", 8080);

    if (auto res = cli.Get("/")) {
        std::cout << "Status: " << res->status << "\n";
        std::cout << "Body: " << res->body << "\n";
    } else {
        std::cerr << "Failed to connect\n";
    }
}
