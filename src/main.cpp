#include "BoxPosePipeline.h"
#include <open3d/Open3D.h>

int main() {
    BoxPosePipeline::Options opt;
    opt.model_path = "models/end2end.onnx";
    opt.calib_path = "2486/params.xml";
    opt.score_thr = 0.7;
    opt.mask_thr =0.5;

    BoxPosePipeline pipeline(opt);
    if (!pipeline.initialize()) return -1;

    // 输入
    cv::Mat rgb = cv::imread("2486/rgb.jpg");
    open3d::geometry::PointCloud pc;
    open3d::io::ReadPointCloud("2486/pcAll.pcd", pc);

    // 输出
    std::vector<BoxPoseResult> results;
    cv::Mat vis;
    pipeline.run(rgb, pc, results, &vis);

    cv::imwrite("2486/vis_on_orig.jpg", vis);
}
