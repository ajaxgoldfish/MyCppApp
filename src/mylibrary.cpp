
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "mylibrary.h"          // 声明：bs_yzx_init / bs_yzx_object_detection_lanxin
using nlohmann::json;
namespace fs = std::filesystem;

// ====== 配置：最大写入目标数 ======
#ifndef YZX_MAX_BOX
#define YZX_MAX_BOX 100
#endif

// ====== 全局（仅本编译单元可见） ======
namespace {
    std::unique_ptr<BoxPosePipeline> g_pipeline;
    std::unique_ptr<LanxinCamera>    g_camera;

    BoxPosePipeline::Options         g_opt;
    std::string                      g_root_dir = "res"; // 仅用于输出可视化

    

    // 将一次 pipeline 结果写入 Box（按你给的字段）
    static inline void write_one_box(::zzb::Box& dst, const BoxPoseResult& src) {
        // 坐标（米）
        dst.x = static_cast<double>(src.xyz_m.x);
        dst.y = static_cast<double>(src.xyz_m.y);
        dst.z = static_cast<double>(src.xyz_m.z);

        // 尺寸（米）——若你的 BoxPoseResult 没有对应字段，请改成 0.0
        dst.width  = static_cast<double>(src.width_m);
        dst.height = static_cast<double>(src.height_m);

        // 角度（度）：W、P、R → angle_a、angle_b、angle_c
        dst.angle_a = static_cast<double>(src.wpr_deg[0]);
        dst.angle_b = static_cast<double>(src.wpr_deg[1]);
        dst.angle_c = static_cast<double>(src.wpr_deg[2]);

        // 旋转矩阵（行优先展开为 r1..r9）
        const Eigen::Matrix3d& R = src.Rw;
        dst.rw1 = static_cast<double>(R(0,0));
        dst.rw2 = static_cast<double>(R(0,1));
        dst.rw3 = static_cast<double>(R(0,2));
        dst.rw4 = static_cast<double>(R(1,0));
        dst.rw5 = static_cast<double>(R(1,1));
        dst.rw6 = static_cast<double>(R(1,2));
        dst.rw7 = static_cast<double>(R(2,0));
        dst.rw8 = static_cast<double>(R(2,1));
        dst.rw9 = static_cast<double>(R(2,2));
    }
} // namespace

// =====================
// 导出 API 实现
// =====================

int bs_yzx_init(const bool isDebug) {
    // spdlog 基础配置
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(isDebug ? spdlog::level::debug : spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);

    // Pipeline 参数（与原 main 保持一致；如需改为相机内参可在此读取 g_camera->get_param()）
    g_opt.model_path = "models/end2end.onnx";
    g_opt.calib_path = "config/params.xml";
    g_opt.score_thr  = 0.7f;
    g_opt.mask_thr   = 0.5f;
    g_opt.paint_masks_on_vis = true;

    // 初始化 pipeline
    if (!g_pipeline) {
        g_pipeline = std::make_unique<BoxPosePipeline>(g_opt);
        if (!g_pipeline->initialize()) {
            spdlog::critical("BoxPosePipeline.initialize() 失败");
            g_pipeline.reset();
            return -1;
        }
    }

    // 初始化 camera
    if (!g_camera || !g_camera->isOpened()) {
        g_camera = std::make_unique<LanxinCamera>(); // 构造里会 connect()
        if (!g_camera->isOpened()) {
            spdlog::critical("LanxinCamera 连接失败");
            g_camera.reset();
            return -2;
        }
        spdlog::info("LanxinCamera 已连接");
    }

    spdlog::info("bs_yzx_init 完成（debug={}）", isDebug);
    return 0;
}

int bs_yzx_object_detection_lanxin(int taskId, zzb::Box boxArr[]) {
    if (!g_pipeline) return -10; // 未初始化 pipeline
    if (!g_camera || !g_camera->isOpened()) return -11; // 未初始化 camera

    auto t0 = std::chrono::steady_clock::now();

    // 0) 结果目录：res/<taskId>/
    const fs::path caseDir = fs::path(g_root_dir) / std::to_string(taskId);
    std::error_code ec_mkdir;
    fs::create_directories(caseDir, ec_mkdir); // 目录不存在则创建

    // 1) 从相机抓图像
    cv::Mat rgb;
    if (g_camera->CapFrame(rgb) != 0 || rgb.empty()) {
        spdlog::error("相机获取 RGB 帧失败");
        return -22;
    }

    // 1.1 保存原始 RGB
    const fs::path rgbPath = caseDir / "rgb_orig.jpg";
    if (!cv::imwrite(rgbPath.string(), rgb)) {
        spdlog::warn("保存原始 RGB 失败: {}", rgbPath.string()); // 非致命
    } else {
        spdlog::info("原始 RGB 已保存: {}", rgbPath.string());
    }

    // 2) 从相机抓点云
    open3d::geometry::PointCloud pc;
    if (g_camera->CapFrame(pc) != 0 || pc.points_.empty()) {
        spdlog::error("相机获取点云帧失败或为空");
        return -23;
    }

    const fs::path pcdPath = caseDir / "cloud_orig.pcd";

    open3d::io::WritePointCloudOption opt;
    opt.write_ascii    = open3d::io::WritePointCloudOption::IsAscii::Ascii;          // ASCII
    opt.compressed     = open3d::io::WritePointCloudOption::Compressed::Uncompressed; // 不压缩
    opt.print_progress = false;

    bool ok_pcd = open3d::io::WritePointCloud(pcdPath.string(), pc, opt);
    if (!ok_pcd) {
        spdlog::warn("保存原始点云失败: {}", pcdPath.string());
    } else {
        spdlog::info("原始点云已保存: {}", pcdPath.string());
    }

    // 3) 执行 Pipeline
    std::vector<BoxPoseResult> results;
    cv::Mat vis;
    const bool ok = g_pipeline->run(rgb, pc, results, &vis);
    if (!ok || vis.empty()) {
        spdlog::error("pipeline.run 失败");
        return -24;
    }

    // 4) 输出可视化到 res/<taskId>/vis_on_orig.jpg
    const fs::path outPath = caseDir / "vis_on_orig.jpg";
    if (!cv::imwrite(outPath.string(), vis)) {
        spdlog::error("写可视化文件失败: {}", outPath.string()); // 非致命
    }

    auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5) 写回 boxArr（最多 YZX_MAX_BOX 个）
    const int total   = static_cast<int>(results.size());
    const int n_write = (total < YZX_MAX_BOX) ? total : YZX_MAX_BOX;
    for (int i = 0; i < n_write; ++i) {
        write_one_box(boxArr[i], results[i]);
    }

    spdlog::info("[ OK ] taskId={} -> {}，目标数={}（写入 {} 个），耗时={:.3f} ms",
                 taskId, outPath.string(), total, n_write, elapsed_ms);

    // 6) 写入 JSON（增加原始数据文件路径）
    json j;
    j["taskId"]     = taskId;
    j["elapsed_ms"] = elapsed_ms;
    j["total"]      = total;
    j["n_write"]    = n_write;
    j["out_image"]  = outPath.string();
    j["raw_rgb"]    = rgbPath.string();
    j["raw_pcd"]    = pcdPath.string();

    auto& arr = j["boxes"] = json::array();
    for (int i = 0; i < n_write; ++i) {
        const auto& b = boxArr[i];
        arr.push_back({
            {"x", b.x}, {"y", b.y}, {"z", b.z},
            {"width", b.width}, {"height", b.height},
            {"angle_a", b.angle_a}, {"angle_b", b.angle_b}, {"angle_c", b.angle_c},
            {"rw1", b.rw1}, {"rw2", b.rw2}, {"rw3", b.rw3},
            {"rw4", b.rw4}, {"rw5", b.rw5}, {"rw6", b.rw6},
            {"rw7", b.rw7}, {"rw8", b.rw8}, {"rw9", b.rw9}
        });
    }
    const fs::path jsonPath = caseDir / "boxes.json";
    std::ofstream ofs(jsonPath);
    if (!ofs) {
        spdlog::error("写 JSON 失败（无法打开文件）: {}", jsonPath.string());
    } else {
        ofs << std::setw(2) << j;
        ofs.close();
        spdlog::info("JSON 已写入: {}", jsonPath.string());
    }
    return n_write; // 返回写入个数
}
