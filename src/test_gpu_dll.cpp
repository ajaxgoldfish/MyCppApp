#include <iostream>
#include "../include/cpu_library.h"

int main() {
    std::cout << "开始测试 yzx_vision_zzb 库 (GPU Mode)" << std::endl;
    
    // 1. 调用 bs_yzx_init 方法初始化
    // 参数1: true 表示开启调试模式
    // 参数2: 已移除，改为从 config/params.xml 读取 RunMode 和 DeviceType
    std::cout << "\n=== 步骤1: 初始化系统 ===" << std::endl;
    int result = bs_yzx_init(true);
    
    if (result == 0) {
        std::cout << "✓ bs_yzx_init 调用成功！返回值: " << result << std::endl;
    } else {
        std::cout << "✗ bs_yzx_init 调用失败！返回值: " << result << std::endl;
        std::cout << "初始化失败，程序退出" << std::endl;
        return -1;
    }
    
    // 2. 调用拍照和物体检测方法
    std::cout << "\n=== 步骤2: 拍照并进行物体检测 ===" << std::endl;
    
    // 准备存储检测结果的数组（最多100个box）
    zzb::Box boxes[100];
    
    // taskId 用于标识任务，结果会保存在 res/<taskId>/ 目录下
    int taskId = 9999;  // 可以改成你想要的任务ID
    
    std::cout << "开始拍照和检测（taskId=" << taskId << "）..." << std::endl;
    int detection_result = bs_yzx_object_detection_lanxin(taskId, boxes);
    
    if (detection_result >= 0) {
        std::cout << "✓ 拍照和检测成功！检测到 " << detection_result << " 个物体" << std::endl;
        
        // 3. 输出检测结果
        if (detection_result > 0) {
            std::cout << "\n=== 检测结果详情 ===" << std::endl;
            for (int i = 0; i < detection_result; i++) {
                std::cout << "\n物体 #" << (i + 1) << ":" << std::endl;
                std::cout << "  ID: " << boxes[i].id << std::endl;
                std::cout << "  位置 (x, y, z): (" 
                          << boxes[i].x << ", " 
                          << boxes[i].y << ", " 
                          << boxes[i].z << ") 毫米" << std::endl;
                std::cout << "  尺寸 (宽x高): " 
                          << boxes[i].width << " x " 
                          << boxes[i].height << " 毫米" << std::endl;
                std::cout << "  角度 (a, b, c): (" 
                          << boxes[i].angle_a << "°, " 
                          << boxes[i].angle_b << "°, " 
                          << boxes[i].angle_c << "°)" << std::endl;
            }
        }
        
        std::cout << "\n结果已保存到: res/" << taskId << "/ 目录" << std::endl;
        std::cout << "  - rgb_orig.jpg: 原始RGB图像" << std::endl;
        std::cout << "  - cloud_orig.pcd: 原始点云" << std::endl;
        std::cout << "  - vis_on_orig.jpg: 检测结果可视化" << std::endl;
        std::cout << "  - boxes.json: 检测结果JSON" << std::endl;
        
    } else {
        std::cout << "✗ 拍照和检测失败！错误码: " << detection_result << std::endl;
        // ... (error codes remain same)
        switch (detection_result) {
            case -10: std::cout << "  原因: Pipeline 未初始化" << std::endl; break;
            case -11: std::cout << "  原因: 相机未初始化或未打开" << std::endl; break;
            case -21: std::cout << "  原因: 数据目录不存在" << std::endl; break;
            case -22: std::cout << "  原因: 无法捕获/读取 RGB 图像" << std::endl; break;
            case -23: std::cout << "  原因: 无法捕获/读取 点云" << std::endl; break;
            case -25: std::cout << "  原因: 标定文件错误" << std::endl; break;
            default: std::cout << "  原因: 未知错误" << std::endl; break;
        }
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}
