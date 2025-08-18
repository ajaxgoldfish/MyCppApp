//
// Created by zhangzongbo on 2025/8/5.
//
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using json = nlohmann::json;

int main() {
    json user = {
        {"name", "zhangzongbo"},
        {"age", 30},
        {"skills", {"C++", "JSON", "cmake"}}
    };

    std::cout << user.dump(4) << std::endl;

    // 反序列化
    std::string raw = R"({"project":"MyCppApp","stars":999})";
    json parsed = json::parse(raw);

    std::cout << "Project name: " << parsed["project"] << std::endl;

    return 0;
}
