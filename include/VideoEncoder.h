//
// Created by zwj on 2024/7/8.
//

#ifndef H264ENCODER_H
#define H264ENCODER_H


#include <opencv2/opencv.hpp>
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#ifdef __cplusplus
}
#endif


class VideoEncoder {
public:
    int encode(const cv::Mat &frame) const;

    void uninit();

    int init(int input_width, int input_height, int output_width, int output_height);

    AVPacket *getAVPacket() const;

private:
    SwsContext *sws_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
};


#endif //H264ENCODER_H
