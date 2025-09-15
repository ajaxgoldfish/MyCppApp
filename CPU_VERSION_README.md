# CPU版本目标检测使用说明

## 概述
`main_cpu.cpp` 提供了一个使用CPU Library进行目标检测的完整示例程序。

## 函数调用流程
1. `bs_yzx_init(bool isDebug)` - 初始化CPU版本的目标检测库
2. `bs_yzx_object_detection_lanxin(int taskId, zzb::Box boxArr[])` - 执行目标检测

## 目录结构要求
程序会自动扫描 `res/` 目录下的所有数字命名的子目录，每个子目录需要包含：

```
res/
├── 1/
│   ├── rgb.jpg      # RGB彩色图像
│   └── pcAll.pcd    # 点云数据文件
├── 2/
│   ├── rgb.jpg
│   └── pcAll.pcd
└── ...
```

## 输出文件
程序会在每个输入目录中生成：
- `vis_on_orig.jpg` - 可视化结果图像（带检测框和标注）
- `boxes.json` - 详细的检测结果JSON文件

## 编译和运行
确保你的CMakeLists.txt包含了cpu_library的编译设置，然后：

```bash
# 编译
mkdir build && cd build
cmake ..
make

# 运行
./main_cpu
```

## 日志输出示例
```
[2024-XX-XX XX:XX:XX.XXX] [info] === CPU Library 版本目标检测演示 ===
[2024-XX-XX XX:XX:XX.XXX] [info] CPU Library 初始化成功
[2024-XX-XX XX:XX:XX.XXX] [info] [ OK ] taskId=1, 检测到 3 个目标，耗时=245.123 ms
[2024-XX-XX XX:XX:XX.XXX] [info]   目标 #1: 位置=(0.123, 0.456, 0.789), 尺寸=(0.050m, 0.030m), 角度=(12.3°, 45.6°, 78.9°)
[2024-XX-XX XX:XX:XX.XXX] [info] === 处理完成 ===
[2024-XX-XX XX:XX:XX.XXX] [info] 成功: 5, 跳过: 2, 错误: 0
```

## 错误码说明
- `返回值 < 0`: 发生错误
- `返回值 >= 0`: 成功，返回值为检测到的目标数量

常见错误码：
- `-10`: Pipeline 未初始化
- `-21`: 数据目录不存在
- `-22`: RGB图像文件问题
- `-23`: 点云文件问题
