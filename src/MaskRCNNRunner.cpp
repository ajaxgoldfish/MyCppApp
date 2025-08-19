#include "MaskRCNNRunner.h"
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
/**
[2025-08-12 15:59:32.437] [info] Input shape = [1, 3, 1088, 1920]
[2025-08-12 15:59:37.265] [info] Output[0] shape = [1, 48, 5]
[2025-08-12 15:59:37.265] [info] Output[1] shape = [1, 48]
[2025-08-12 15:59:37.265] [info] Output[2] shape = [1, 48, 56, 56]
[2025-08-12 15:59:37.266] [info] 掩膜大小: 56 x 56
 */
MaskRCNNRunner::MaskRCNNRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "mrcnn")
{
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, to_wstring(model_path).c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;

    in_name_ = session_->GetInputNameAllocated(0, alloc).get();

    size_t out_count = session_->GetOutputCount();
    out_names_s_.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        out_names_s_.emplace_back(session_->GetOutputNameAllocated(i, alloc).get());
    }
    for (auto& s : out_names_s_) {
        out_names_.push_back(s.c_str());
    }
}

cv::Mat MaskRCNNRunner::paint(
    const cv::Mat& orig,
    const std::vector<Ort::Value>& outs, // 外部传进来的推理结果
    float score_thr,
    float mask_thr)
{
    auto t0 = std::chrono::steady_clock::now();  // ⏱ 开始计时

    if (orig.empty()) {
        throw std::runtime_error("输入图像为空");
    }
    if (outs.size() < 3) {
        throw std::runtime_error("输出不足，期望3个");
    }

    // 固定颜色和透明度
    const cv::Scalar color(0, 255, 0); // 绿色
    const double alpha = 0.5;          // 透明度

    // 读取 shape
    auto sh0 = outs[0].GetTensorTypeAndShapeInfo().GetShape(); // dets: [1, N, 5]
    auto sh2 = outs[2].GetTensorTypeAndShapeInfo().GetShape(); // masks:[1, N, mH, mW]
    int64_t N = sh0[1];
    int mH = static_cast<int>(sh2[2]);
    int mW = static_cast<int>(sh2[3]);

    const float* dets = outs[0].GetTensorData<float>();
    const float* masks = outs[2].GetTensorData<float>();

    cv::Mat vis = orig.clone();
    int kept = 0;

    for (int64_t i = 0; i < N; ++i) {
        const float* r = dets + i * 5;
        float sc = r[4];
        if (sc < score_thr) continue;

        // 直接用模型输出的坐标（假设与原图一致）
        int x1 = std::lround(r[0]);
        int y1 = std::lround(r[1]);
        int x2 = std::lround(r[2]);
        int y2 = std::lround(r[3]);

        x1 = std::clamp(x1, 0, vis.cols - 1);
        y1 = std::clamp(y1, 0, vis.rows - 1);
        x2 = std::clamp(x2, 0, vis.cols - 1);
        y2 = std::clamp(y2, 0, vis.rows - 1);

        // 绘制矩形框
        cv::rectangle(vis, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);

        // 画分数
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", sc);
        cv::putText(vis, buf, { x1, std::max(0, y1 - 5) },
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

        // ROI 区域
        int ow = std::max(1, x2 - x1);
        int oh = std::max(1, y2 - y1);
        cv::Rect roi(x1, y1, ow, oh);
        roi &= cv::Rect(0, 0, vis.cols, vis.rows);
        if (roi.area() <= 0) continue;

        // mask
        const float* mptr = masks + i * (mH * mW);
        cv::Mat mask_f32(mH, mW, CV_32F, const_cast<float*>(mptr));

        cv::Mat mask_up;
        cv::resize(mask_f32, mask_up, roi.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat1b mask8;
        cv::compare(mask_up, mask_thr, mask8, cv::CMP_GT);

        cv::Mat roi_img = vis(roi);
        cv::Mat overlay = roi_img.clone();
        overlay.setTo(color, mask8);
        cv::addWeighted(roi_img, 1.0, overlay, alpha, 0, roi_img);

        ++kept;
    }

    spdlog::info("绘制 {} 个实例", kept);

    auto t1 = std::chrono::steady_clock::now();  // ⏱ 结束计时
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    spdlog::info("paint{}",elapsed_ms);
    return vis;
}


// 提取每个实例的二值掩膜（原图大小，0/255）
std::vector<cv::Mat1b> MaskRCNNRunner::inferMasks(
    const cv::Mat& orig,
    const std::vector<Ort::Value>& outs,
    float score_thr,
    float mask_thr)
{
    if (orig.empty()) {
        throw std::runtime_error("输入图像为空");
    }
    if (outs.size() < 3) {
        throw std::runtime_error("输出不足，期望3个");
    }

    auto sh0 = outs[0].GetTensorTypeAndShapeInfo().GetShape(); // dets: [1, N, 5]
    auto sh2 = outs[2].GetTensorTypeAndShapeInfo().GetShape(); // masks:[1, N, mH, mW]
    int64_t N = sh0[1];
    int mH = static_cast<int>(sh2[2]);
    int mW = static_cast<int>(sh2[3]);

    const float* dets  = outs[0].GetTensorData<float>();
    const float* masks = outs[2].GetTensorData<float>();

    std::vector<cv::Mat1b> result;
    result.reserve(static_cast<size_t>(N));

    for (int64_t i = 0; i < N; ++i) {
        const float* r = dets + i * 5;
        float sc = r[4];
        if (sc < score_thr) continue;

        int x1 = static_cast<int>(std::lround(r[0]));
        int y1 = static_cast<int>(std::lround(r[1]));
        int x2 = static_cast<int>(std::lround(r[2]));
        int y2 = static_cast<int>(std::lround(r[3]));

        x1 = std::clamp(x1, 0, orig.cols - 1);
        y1 = std::clamp(y1, 0, orig.rows - 1);
        x2 = std::clamp(x2, 0, orig.cols - 1);
        y2 = std::clamp(y2, 0, orig.rows - 1);

        int ow = std::max(1, x2 - x1);
        int oh = std::max(1, y2 - y1);
        cv::Rect roi(x1, y1, ow, oh);
        roi &= cv::Rect(0, 0, orig.cols, orig.rows);
        if (roi.area() <= 0) continue;

        const float* mptr = masks + i * (mH * mW);
        cv::Mat mask_f32(mH, mW, CV_32F, const_cast<float*>(mptr));

        cv::Mat mask_up;
        cv::resize(mask_f32, mask_up, roi.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat1b mask8;
        cv::compare(mask_up, mask_thr, mask8, cv::CMP_GT);

        cv::Mat1b fullMask(orig.rows, orig.cols, (uchar)0);
        fullMask(roi).setTo(255, mask8);

        result.emplace_back(std::move(fullMask));
    }

    spdlog::info("共导出 {} 个实例掩膜", result.size());
    return result;
}


std::wstring MaskRCNNRunner::to_wstring(const std::string& s) {
    return { s.begin(), s.end() };
}

inline void MaskRCNNRunner::pixel_normalize_mmdet_rgb(cv::Mat& rgb_f32) {
    static const cv::Scalar mean(123.675, 116.28, 103.53);
    static const cv::Scalar stdv(58.395, 57.12, 57.375);
    cv::subtract(rgb_f32, mean, rgb_f32);
    cv::divide(rgb_f32, stdv, rgb_f32);
}

std::vector<Ort::Value> MaskRCNNRunner::inferRaw(const cv::Mat& orig)
{
    if (orig.empty()) {
        throw std::runtime_error("输入图像为空");
    }

    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(orig, rgb, cv::COLOR_BGR2RGB);

    // 2. 转 float32
    rgb.convertTo(rgb, CV_32F);

    // 3. 调用 mmdet 风格归一化
    pixel_normalize_mmdet_rgb(rgb);

    // 4. 转 blob [1, 3, H, W]
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb, blob, 1.0, cv::Size(), {}, false, false, CV_32F);
    std::vector<int64_t> ishape = {1, 3, blob.size[2], blob.size[3]};

    // 5. 创建 ORT 输入
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mi,
        reinterpret_cast<float*>(blob.data),
        static_cast<size_t>(blob.total()),
        ishape.data(),
        ishape.size()
    );

    const char* in_names[] = { in_name_.c_str() };

    // 6. 推理
    auto outs = session_->Run(
        Ort::RunOptions{nullptr},
        in_names,
        &input,
        1,
        out_names_.data(),
        out_names_.size()
    );

    // 7. 打印输出 shape
    for (size_t idx = 0; idx < outs.size(); ++idx) {
        auto shape = outs[idx].GetTensorTypeAndShapeInfo().GetShape();
        std::ostringstream oss;
        oss << "outs[" << idx << "] shape = [";
        for (size_t j = 0; j < shape.size(); ++j) {
            oss << shape[j];
            if (j + 1 < shape.size()) oss << ", ";
        }
        oss << "]";
        spdlog::info("{}", oss.str());
    }

    return outs;
}


