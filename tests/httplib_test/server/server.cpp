//
// Created by zhangzongbo on 2025/8/6.
//
#include <httplib.h>
#include <iostream>

int main() {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("Hello from server!", "text/plain");
    });

    std::cout << "Listening on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);
}
