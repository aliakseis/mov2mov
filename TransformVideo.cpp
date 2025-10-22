#include "TransformVideo.h"

#include <stdint.h>
#include <sys/stat.h>

extern "C" {
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
}

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <errno.h>

#include "makeguard.h"

struct AVFrameDeleter {
    void operator()(AVFrame* frame) const { av_frame_free(&frame); };
};

typedef std::unique_ptr<AVFrame, AVFrameDeleter> AVFramePtr;

static void ReportError(int ret) {
    char errBuf[AV_ERROR_MAX_STRING_SIZE]{};
    fprintf(stderr, "Error occurred: %s\n",
        av_make_error_string(errBuf, AV_ERROR_MAX_STRING_SIZE, ret));
}

static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

int TransformVideo(const char* in_filename,
    const char* out_filename,
    std::function<void(cv::Mat&)> callback,
    int upscale, int downscale)
{
    AVFormatContext* input_format_context = nullptr, * output_format_context = nullptr;
    int ret;
    int stream_index = 0;

    if ((ret = avformat_open_input(&input_format_context, in_filename, nullptr, nullptr)) < 0) {
        fprintf(stderr, "Could not open input file '%s'\n", in_filename);
        return 1;
    }
    auto input_format_context_guard = MakeGuard(&input_format_context, avformat_close_input);

    if ((ret = avformat_find_stream_info(input_format_context, nullptr)) < 0) {
        fprintf(stderr, "Failed to retrieve input stream information\n");
        return 1;
    }

    // --- handle .bak ---
    std::string bak_filename;
    if (file_exists(out_filename)) {
        bak_filename = std::string(out_filename) + ".bak";
        if (rename(out_filename, bak_filename.c_str()) != 0) {
            fprintf(stderr, "Warning: failed to rename existing output to %s: %s\n",
                bak_filename.c_str(), strerror(errno));
            bak_filename.clear();
        }
    }

    // --- output setup ---
    avformat_alloc_output_context2(&output_format_context, nullptr, "matroska", out_filename);
    if (!output_format_context) {
        fprintf(stderr, "Could not create output context\n");
        return 1;
    }
    auto output_format_context_guard = MakeGuard(output_format_context, avformat_free_context);

    const int number_of_streams = input_format_context->nb_streams;
    std::vector<int> streams_list(number_of_streams, -1);

    int videoStreamNumber = -1;
    AVStream* videoStream = nullptr;
    AVStream* outputVideoStream = nullptr;

    for (int i = 0; i < number_of_streams; i++) {
        AVStream* in_stream = input_format_context->streams[i];
        AVCodecParameters* in_codecpar = in_stream->codecpar;
        const bool isVideo = in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO;

        if (isVideo) {
            videoStreamNumber = i;
            videoStream = in_stream;
        }
        else if (in_codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            streams_list[i] = -1;
            continue;
        }

        streams_list[i] = stream_index++;
        AVStream* out_stream = avformat_new_stream(output_format_context, nullptr);
        if (!out_stream) {
            fprintf(stderr, "Failed allocating output stream\n");
            return 1;
        }

        if ((ret = avcodec_parameters_copy(out_stream->codecpar, in_codecpar)) < 0) {
            fprintf(stderr, "Failed to copy codec parameters\n");
            return 1;
        }

        out_stream->codecpar->codec_tag = 0;
        if (isVideo) outputVideoStream = out_stream;
    }

    output_format_context->flags |= AVFMT_FLAG_NOBUFFER;

    if (!(output_format_context->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&output_format_context->pb, out_filename, AVIO_FLAG_WRITE)) < 0) {
            fprintf(stderr, "Could not open output file '%s'\n", out_filename);
            return 1;
        }
    }

    // --- input video decoder ---
    AVCodecContext* videoCodecContext = avcodec_alloc_context3(nullptr);
    if (!videoCodecContext) return 1;
    auto videoCodecContextGuard = MakeGuard(&videoCodecContext, avcodec_free_context);

    if (avcodec_parameters_to_context(videoCodecContext, videoStream->codecpar) < 0) return 1;

    auto videoCodec = avcodec_find_decoder(videoCodecContext->codec_id);
    if (!videoCodec || avcodec_open2(videoCodecContext, videoCodec, nullptr) < 0) {
        fprintf(stderr, "Error opening decoder\n");
        return 1;
    }

    // --- output encoder ---
    auto encoder = avcodec_find_encoder(videoCodecContext->codec_id);
    if (!encoder) {
        av_log(nullptr, AV_LOG_FATAL, "Necessary encoder not found\n");
        return 1;
    }
    AVCodecContext* enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) return 1;

    const auto out_height = (videoCodecContext->height * upscale / downscale) & ~7;
    const auto out_width = (videoCodecContext->width * upscale / downscale) & ~7;
    enc_ctx->height = out_height;
    enc_ctx->width = out_width;
    enc_ctx->sample_aspect_ratio = videoCodecContext->sample_aspect_ratio;
    enc_ctx->pix_fmt = (encoder->pix_fmts) ? encoder->pix_fmts[0] : videoCodecContext->pix_fmt;
    enc_ctx->time_base = videoStream->time_base;
    enc_ctx->gop_size = 1;
    enc_ctx->max_b_frames = 2;

    if (output_format_context->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if ((ret = avcodec_open2(enc_ctx, encoder, nullptr)) < 0) {
        ReportError(ret);
        return 1;
    }

    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, enc_ctx);
    if (ret < 0) return 1;
    outputVideoStream->time_base = enc_ctx->time_base;

    // --- write header ---
    if ((ret = avformat_write_header(output_format_context, nullptr)) < 0) {
        fprintf(stderr, "Error writing output header\n");
        return 1;
    }

    std::vector<int64_t> last_written_pts(output_format_context->nb_streams, AV_NOPTS_VALUE);

    // --- single-pass backup remux ---
    if (!bak_filename.empty()) {
        AVFormatContext* bak_fmt = nullptr;
        if (avformat_open_input(&bak_fmt, bak_filename.c_str(), nullptr, nullptr) == 0) {
            avformat_find_stream_info(bak_fmt, nullptr);
            AVPacket pkt{};
            //av_init_packet(&pkt);
            std::vector<int64_t> max_pts(output_format_context->nb_streams, AV_NOPTS_VALUE);

            while (av_read_frame(bak_fmt, &pkt) >= 0) {
                int out_idx = pkt.stream_index;
                if (out_idx < 0 || out_idx >= (int)output_format_context->nb_streams) {
                    av_packet_unref(&pkt);
                    continue;
                }

                AVStream* in_st = bak_fmt->streams[pkt.stream_index];
                AVStream* out_st = output_format_context->streams[out_idx];

                if (pkt.pts != AV_NOPTS_VALUE)
                    pkt.pts = av_rescale_q_rnd(pkt.pts, in_st->time_base, out_st->time_base,
                        AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                if (pkt.dts != AV_NOPTS_VALUE)
                    pkt.dts = av_rescale_q_rnd(pkt.dts, in_st->time_base, out_st->time_base,
                        AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                pkt.duration = av_rescale_q(pkt.duration, in_st->time_base, out_st->time_base);
                pkt.stream_index = out_idx;
                pkt.pos = -1;

                ret = av_interleaved_write_frame(output_format_context, &pkt);
                if (ret < 0) {
                    fprintf(stderr, "Warning: remux write error: %d\n", ret);
                    av_packet_unref(&pkt);
                    continue;
                }

                int64_t written_ts = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
                if (written_ts != AV_NOPTS_VALUE)
                    max_pts[out_idx] = (max_pts[out_idx] == AV_NOPTS_VALUE) ?
                    written_ts : std::max(max_pts[out_idx], written_ts);

                av_packet_unref(&pkt);
            }

            for (size_t i = 0; i < last_written_pts.size() && i < max_pts.size(); ++i)
                if (max_pts[i] != AV_NOPTS_VALUE) last_written_pts[i] = max_pts[i];

            avformat_close_input(&bak_fmt);
        }
        else {
            fprintf(stderr, "Warning: could not open backup file %s for remux\n", bak_filename.c_str());
        }
    }

    // --- seek input near last-written video frame ---
    int output_video_stream_index = -1;
    for (unsigned i = 0; i < output_format_context->nb_streams; ++i)
        if (output_format_context->streams[i] == outputVideoStream)
            output_video_stream_index = (int)i;

    if (output_video_stream_index >= 0 &&
        last_written_pts[output_video_stream_index] != AV_NOPTS_VALUE) {
        int64_t last_out_pts = last_written_pts[output_video_stream_index];
        int64_t seek_target = av_rescale_q(last_out_pts,
            outputVideoStream->time_base,
            videoStream->time_base);
        if (av_seek_frame(input_format_context, videoStreamNumber, seek_target,
            AVSEEK_FLAG_BACKWARD) >= 0)
            avcodec_flush_buffers(videoCodecContext);
        else
            fprintf(stderr, "Warning: av_seek_frame failed when seeking to resume point\n");
    }

    // --- prepare frames ---
    AVFramePtr videoFrame(av_frame_alloc());
    AVFramePtr videoFrameOut(av_frame_alloc());
    videoFrameOut->format = videoCodecContext->pix_fmt;
    videoFrameOut->width = out_width;
    videoFrameOut->height = out_height;
    av_frame_get_buffer(videoFrameOut.get(), 16);

    SwsContext* to_bgr_ctx = sws_getContext(
        videoCodecContext->width, videoCodecContext->height, videoCodecContext->pix_fmt,
        out_width, out_height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    SwsContext* from_bgr_ctx = sws_getContext(
        out_width, out_height, AV_PIX_FMT_BGR24,
        out_width, out_height, videoCodecContext->pix_fmt,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    AVPacket enc_pkt{};
    //av_init_packet(&enc_pkt);

    // --- main decode/encode loop ---
    while (av_read_frame(input_format_context, &enc_pkt) >= 0) {
        if (enc_pkt.stream_index != videoStreamNumber) {
            av_packet_unref(&enc_pkt);
            continue;
        }

        if (avcodec_send_packet(videoCodecContext, &enc_pkt) < 0) {
            av_packet_unref(&enc_pkt);
            break;
        }
        av_packet_unref(&enc_pkt);

        while (avcodec_receive_frame(videoCodecContext, videoFrame.get()) == 0) {
            cv::Mat img(out_height, out_width, CV_8UC3);
            int stride = img.step[0];

            sws_scale(to_bgr_ctx, videoFrame->data, videoFrame->linesize,
                0, videoCodecContext->height, &img.data, &stride);

            callback(img);

            sws_scale(from_bgr_ctx, (const uint8_t**)&img.data, &stride,
                0, out_height, videoFrameOut->data, videoFrameOut->linesize);

            videoFrameOut->pts = videoFrame->pts;
            if (avcodec_send_frame(enc_ctx, videoFrameOut.get()) < 0) break;

            AVPacket pkt{};
            //av_init_packet(&pkt);
            while (avcodec_receive_packet(enc_ctx, &pkt) == 0) {
                if (pkt.pts != AV_NOPTS_VALUE)
                    pkt.pts = av_rescale_q(pkt.pts, enc_ctx->time_base, outputVideoStream->time_base);
                if (pkt.dts != AV_NOPTS_VALUE)
                    pkt.dts = av_rescale_q(pkt.dts, enc_ctx->time_base, outputVideoStream->time_base);
                pkt.stream_index = output_video_stream_index;
                pkt.pos = -1;

                ret = av_interleaved_write_frame(output_format_context, &pkt);
                av_packet_unref(&pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error while writing encoded packet\n");
                    break;
                }
            }
        }
    }

    av_write_trailer(output_format_context);
    if (!(output_format_context->oformat->flags & AVFMT_NOFILE))
        avio_closep(&output_format_context->pb);

    sws_freeContext(to_bgr_ctx);
    sws_freeContext(from_bgr_ctx);

    return 0;
}
