//
// Created by zwj on 2024/7/8.
//

#include "VideoEncoder.h"

using namespace std;

int VideoEncoder::encode(const cv::Mat &mat) const {
    int cvLinesizes[1];
    cvLinesizes[0] = mat.step1();
    int code = sws_scale(sws_ctx, &mat.data, cvLinesizes, 0,
                         mat.rows, frame->data, frame->linesize);
    frame->pts++;
    // 编码帧
    code = avcodec_send_frame(codec_ctx, frame);
    if (code != 0) {
        return code;
    }
    return avcodec_receive_packet(codec_ctx, pkt);
}

void VideoEncoder::uninit() {
    sws_freeContext(sws_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
}

AVPacket *VideoEncoder::getAVPacket() const {
    return this->pkt;
}

int VideoEncoder::init(const int input_width, const int input_height, const int output_width, const int output_height) {
    auto dst_format = AV_PIX_FMT_NV12;
    avdevice_register_all();
    // 查找编码器

    //mjpeg mjpeg_qsv h264_nvenc h264_qsv
    //const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    const AVCodec *codec = avcodec_find_encoder_by_name("mjpeg_qsv");
    if (!codec) {
        std::cerr << "Codec not found" << std::endl;
        return -1;
    }
    sws_ctx = sws_getContext(input_width, input_height, AVPixelFormat::AV_PIX_FMT_BGR24, output_width, output_height,
                             dst_format, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ctx == nullptr) {
        cout << "sws_getContext error" << endl;
        return -2;
    }

    // 创建编码器上下文
    codec_ctx = avcodec_alloc_context3(codec);
    codec_ctx->bit_rate = 3000*1000; // 比特率
    codec_ctx->width = output_width;
    codec_ctx->height = output_height;
    codec_ctx->time_base = AVRational{1, 25}; // 帧率
    codec_ctx->framerate = AVRational{25, 1};
    codec_ctx->gop_size = 10; // GOP大小
    codec_ctx->max_b_frames = 3;
    codec_ctx->pix_fmt = dst_format;
    codec_ctx->global_quality = 50;

    // 打开编码器
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return -3;
    }
    frame->format = dst_format;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;
    av_frame_get_buffer(frame, 32);
    return 0;
}
