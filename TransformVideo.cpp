#include "TransformVideo.h"

#include <stdint.h>
#include <sys/stat.h>

extern "C"
{
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

struct AVFrameDeleter
{
    void operator()(AVFrame *frame) const { av_frame_free(&frame); };
};

typedef std::unique_ptr<AVFrame, AVFrameDeleter> AVFramePtr;

void ReportError(int ret)
{
    char errBuf[AV_ERROR_MAX_STRING_SIZE]{};
    fprintf(stderr, "Error occurred: %s\n", av_make_error_string(errBuf, AV_ERROR_MAX_STRING_SIZE, ret));
}

static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

int TransformVideo(const char *in_filename,
    const char *out_filename,
    std::function<void(cv::Mat&)>  callback,
    int upscale, int downscale)
{
    AVFormatContext *input_format_context = NULL, *output_format_context = NULL;

    int ret;
    int stream_index = 0;

    if ((ret = avformat_open_input(&input_format_context, in_filename, NULL, NULL)) < 0) {
        fprintf(stderr, "Could not open input file '%s'\n", in_filename);
        return 1;
    }

    auto input_format_context_guard = MakeGuard(&input_format_context, avformat_close_input);

    if ((ret = avformat_find_stream_info(input_format_context, NULL)) < 0) {
        fprintf(stderr, "Failed to retrieve input stream information\n");
        return 1;
    }

    // If output exists, rename to .bak
    std::string bak_filename;
    if (file_exists(out_filename)) {
        bak_filename = std::string(out_filename) + ".bak";
        if (rename(out_filename, bak_filename.c_str()) != 0) {
            fprintf(stderr, "Warning: failed to rename existing output to %s: %s\n", bak_filename.c_str(), strerror(errno));
            // continue anyway (we may overwrite)
            bak_filename.clear();
        }
    }

    avformat_alloc_output_context2(&output_format_context, NULL, "matroska", out_filename);
    if (!output_format_context) {
        fprintf(stderr, "Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        return 1;
    }

    auto output_format_context_guard = MakeGuard(output_format_context, avformat_free_context);

    const auto number_of_streams = input_format_context->nb_streams;
    std::vector<int> streams_list(number_of_streams, -1);

    int videoStreamNumber = -1;
    AVStream* videoStream = nullptr;
    AVStream* outputVideoStream = nullptr;

    for (int i = 0; i < input_format_context->nb_streams; i++) {
        AVStream *out_stream;
        AVStream *in_stream = input_format_context->streams[i];
        AVCodecParameters *in_codecpar = in_stream->codecpar;

        const bool isVideoStream = in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
        if (isVideoStream)
        {
            videoStreamNumber = i;
            videoStream = in_stream;
        }
        else if (in_codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            streams_list[i] = -1;
            continue;
        }
        streams_list[i] = stream_index++;
        out_stream = avformat_new_stream(output_format_context, NULL);
        if (!out_stream) {
            fprintf(stderr, "Failed allocating output stream\n");
            ret = AVERROR_UNKNOWN;
            return 1;
        }
        ret = avcodec_parameters_copy(out_stream->codecpar, in_codecpar);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy codec parameters\n");
            return 1;
        }

        out_stream->codecpar->codec_tag = 0;

        if (isVideoStream)
        {
            outputVideoStream = out_stream;
        }
    }

    output_format_context->flags |= AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;
    output_format_context->flush_packets = 1;

    av_dump_format(output_format_context, 0, out_filename, 1);

    if (!(output_format_context->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&output_format_context->pb, out_filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "Could not open output file '%s'\n", out_filename);
            return 1;
        }
    }

    // input video context
    auto videoCodecContext = avcodec_alloc_context3(nullptr);
    if (!videoCodecContext)
        return 1;

    auto videoCodecContextGuard = MakeGuard(&videoCodecContext, avcodec_free_context);

    if (avcodec_parameters_to_context(videoCodecContext, videoStream->codecpar) < 0)
        return 1;

    auto videoCodec = avcodec_find_decoder(videoCodecContext->codec_id);
    if (videoCodec == nullptr)
    {
        fprintf(stderr, "No such codec found\n");
        return 1;  // Codec not found
    }

    // Open codec
    if (avcodec_open2(videoCodecContext, videoCodec, nullptr) < 0)
    {
        fprintf(stderr, "Error on codec opening\n");
        return 1;  // Could not open codec
    }

    // output encoder
    auto encoder = avcodec_find_encoder(videoCodecContext->codec_id);
    if (!encoder) {
        av_log(NULL, AV_LOG_FATAL, "Necessary encoder not found\n");
        return 1;
    }
    auto enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the encoder context\n");
        return 1;
    }

    const auto out_height = (videoCodecContext->height * upscale / downscale) & ~7;
    const auto out_width = (videoCodecContext->width * upscale / downscale) & ~7;

    enc_ctx->height = out_height;
    enc_ctx->width = out_width;
    enc_ctx->sample_aspect_ratio = videoCodecContext->sample_aspect_ratio;
    enc_ctx->pix_fmt = (encoder->pix_fmts != nullptr)? encoder->pix_fmts[0] : videoCodecContext->pix_fmt;
    enc_ctx->time_base = videoStream->time_base;

    if (output_format_context->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // sane defaults; tune as needed
    enc_ctx->gop_size = 1;
    enc_ctx->max_b_frames = 2;

    ret = avcodec_open2(enc_ctx, encoder, NULL);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder for stream\n");
        ReportError(ret);
        return 1;
    }
    // copy encoder params to output stream (find the right stream index)
    // find output stream index that corresponds to original videoStreamNumber
    int output_video_stream_index = -1;
    for (unsigned i = 0; i < output_format_context->nb_streams; ++i) {
        if (output_format_context->streams[i] == outputVideoStream) {
            output_video_stream_index = (int)i;
            break;
        }
    }
    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, enc_ctx);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to copy encoder parameters to output stream\n");
        return 1;
    }
    outputVideoStream->time_base = enc_ctx->time_base;

    AVDictionary* opts = NULL;
    av_dict_set(&opts, "flush_packets", "1", 0);

    // write header (do this before trying to remux bak so stream layout/timebases exist)
    ret = avformat_write_header(output_format_context, &opts);
    if (ret < 0) {
        fprintf(stderr, "Error occurred when opening output file\n");
        return 1;
    }

    // prepare last-written PTS tracker per output stream (in output stream timebase)
    std::vector<int64_t> last_written_pts(output_format_context->nb_streams, AV_NOPTS_VALUE);

    // If a backup exists, remux salvageable packets into new output and set last_written_pts
    if (!bak_filename.empty()) {
        AVFormatContext* bak_fmt = nullptr;
        if (avformat_open_input(&bak_fmt, bak_filename.c_str(), NULL, NULL) == 0) {
            avformat_find_stream_info(bak_fmt, NULL);
            AVPacket pkt;
            av_init_packet(&pkt);
            while (av_read_frame(bak_fmt, &pkt) >= 0) {
                int out_idx = pkt.stream_index;
                if (out_idx < 0 || out_idx >= (int)output_format_context->nb_streams) {
                    av_packet_unref(&pkt);
                    continue;
                }
                AVStream* in_st = bak_fmt->streams[pkt.stream_index];
                AVStream* out_st = output_format_context->streams[out_idx];

                // save original pts/dts for later use
                int64_t orig_pts = pkt.pts;
                int64_t orig_dts = pkt.dts;

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
                    av_packet_unref(&pkt);
                    fprintf(stderr, "Warning: stopped remuxing backup due to write error\n");
                    break;
                }

                // prefer pts, fallback to dts; use the rescaled values we've just written
                int64_t written_ts = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
                if (written_ts != AV_NOPTS_VALUE)
                    last_written_pts[out_idx] = written_ts;

                av_packet_unref(&pkt);
            }
            avformat_close_input(&bak_fmt);
        } else {
            fprintf(stderr, "Warning: could not open backup file %s for remux\n", bak_filename.c_str());
        }
        // NOTE: the above loop set last_written_pts only in limited ways; to ensure robust last pts,
        // we will compute last_written_pts for the video stream from the actual file position by re-opening the new output
        // and reading its streams' last DTS if needed. For simplicity below we will compute last_written_pts for video by using
        // the time of the last remuxed packet tracked by the last_packet_pts variable.
    }

    // To provide correct skip behavior we must know last written video pts in output timebase.
    // We attempt to set last_written_pts[output_video_stream_index] to the greatest pts we wrote.
    // For the remux loop above we didn't store per-packet pts in variables; rebuild a minimal remux that stores last pts.

    // Rewind: if there was a bak, perform remux again but tracking last pts properly (safer approach)
    if (!bak_filename.empty()) {
        // Reset output file to state after header (we already wrote header). We'll remux and update last_written_pts properly.
        // Note: this second pass will append additional duplicates if the first pass already wrote them.
        // To avoid duplicates, only perform the detailed tracked remux if last_written_pts still all AV_NOPTS_VALUE.
        bool need_detailed_remux = true;
        for (auto v : last_written_pts) { if (v != AV_NOPTS_VALUE) { need_detailed_remux = false; break; } }
        if (need_detailed_remux) {
            AVFormatContext* bak_fmt = nullptr;
            if (avformat_open_input(&bak_fmt, bak_filename.c_str(), NULL, NULL) == 0) {
                avformat_find_stream_info(bak_fmt, NULL);
                AVPacket pkt;
                av_init_packet(&pkt);
                while (av_read_frame(bak_fmt, &pkt) >= 0) {
                    int out_idx = pkt.stream_index;
                    if (out_idx < 0 || out_idx >= (int)output_format_context->nb_streams) {
                        av_packet_unref(&pkt);
                        continue;
                    }
                    AVStream* in_st = bak_fmt->streams[pkt.stream_index];
                    AVStream* out_st = output_format_context->streams[out_idx];

                    int64_t pts_saved = pkt.pts;
                    int64_t dts_saved = pkt.dts;

                    if (pkt.pts != AV_NOPTS_VALUE)
                        pkt.pts = av_rescale_q_rnd(pkt.pts, in_st->time_base, out_st->time_base,
                                                   AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    if (pkt.dts != AV_NOPTS_VALUE)
                        pkt.dts = av_rescale_q_rnd(pkt.dts, in_st->time_base, out_st->time_base,
                                                   AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    pkt.duration = av_rescale_q(pkt.duration, in_st->time_base, out_st->time_base);
                    pkt.stream_index = out_idx;
                    pkt.pos = -1;

                    ret = av_interleaved_write_frame(output_format_context, &pkt);
                    if (ret < 0) {
                        av_packet_unref(&pkt);
                        fprintf(stderr, "Warning: stopped remuxing backup due to write error\n");
                        break;
                    }

                    // store last written pts (prefer pts, fallback to dts)
                    int64_t written_ts = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
                    if (written_ts != AV_NOPTS_VALUE)
                        last_written_pts[out_idx] = written_ts;

                    av_packet_unref(&pkt);
                }
                avformat_close_input(&bak_fmt);
            } else {
                fprintf(stderr, "Warning: could not open backup file %s for detailed remux\n", bak_filename.c_str());
            }
        }
    }

    // If we have remuxed any video packets, seek original input a bit before the last_written_pts to ensure keyframe availability
    if (output_video_stream_index >= 0 && last_written_pts.size() > (size_t)output_video_stream_index &&
        last_written_pts[output_video_stream_index] != AV_NOPTS_VALUE)
    {
        int64_t last_out_pts = last_written_pts[output_video_stream_index];
        // convert from output timebase to input (video) timebase
        int64_t seek_target = av_rescale_q(last_out_pts, outputVideoStream->time_base, videoStream->time_base);
        // seek slightly back to keyframe boundary
        int seek_flags = AVSEEK_FLAG_BACKWARD;
        if (av_seek_frame(input_format_context, videoStreamNumber, seek_target, seek_flags) < 0) {
            fprintf(stderr, "Warning: av_seek_frame failed when seeking to resume point\n");
        } else {
            // flush decoder buffers so decoding restarts cleanly
            avcodec_flush_buffers(videoCodecContext);
        }
    }

    // Prepare frames / buffers
    AVFramePtr videoFrame(av_frame_alloc());
    AVFramePtr videoFrameOut(av_frame_alloc());
    videoFrameOut->format = videoCodecContext->pix_fmt;
    videoFrameOut->width = out_width;
    videoFrameOut->height = out_height;
    av_frame_get_buffer(videoFrameOut.get(), 16);

    // We will need an encoder packet for encoded output
    AVPacket enc_pkt;
    av_init_packet(&enc_pkt);
    enc_pkt.data = NULL;
    enc_pkt.size = 0;

    // main read loop
    while (true) {
        AVPacket packet;
        av_init_packet(&packet);
        ret = av_read_frame(input_format_context, &packet);
        if (ret < 0)
            break;

        // If this input stream is not being copied, skip
        if (packet.stream_index >= number_of_streams || streams_list[packet.stream_index] < 0) {
            av_packet_unref(&packet);
            continue;
        }

        // Map input stream index to output stream index
        int out_stream_index = streams_list[packet.stream_index];
        AVStream* out_stream = output_format_context->streams[out_stream_index];
        AVStream* in_stream = input_format_context->streams[packet.stream_index];

        // Compute packet time in output timebase to compare to last_written_pts
        int64_t pkt_pts_out = AV_NOPTS_VALUE;
        if (packet.pts != AV_NOPTS_VALUE) {
            pkt_pts_out = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base,
                                           AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        } else if (packet.dts != AV_NOPTS_VALUE) {
            pkt_pts_out = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base,
                                           AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        }

        // If this stream already had data remuxed and this packet is <= last remuxed pts, skip it to avoid duplicates
        if (last_written_pts[out_stream_index] != AV_NOPTS_VALUE && pkt_pts_out != AV_NOPTS_VALUE) {
            if (pkt_pts_out <= last_written_pts[out_stream_index]) {
                av_packet_unref(&packet);
                continue;
            }
        }

        // Now handle video stream (re-encode with transform)
        if (packet.stream_index == videoStreamNumber)
        {
            const int ret_send = avcodec_send_packet(videoCodecContext, &packet);
            if (ret_send < 0) {
                av_packet_unref(&packet);
                fprintf(stderr, "Error sending packet to decoder\n");
                break;
            }

            while (avcodec_receive_frame(videoCodecContext, videoFrame.get()) == 0)
            {
                // transform: convert to BGR, callback, convert back
                cv::Mat img(out_height, out_width, CV_8UC3);
                int stride = img.step[0];

                auto img_convert_ctx = sws_getCachedContext(
                    NULL,
                    videoCodecContext->width,
                    videoCodecContext->height,
                    videoCodecContext->pix_fmt,
                    out_width,
                    out_height,
                    AV_PIX_FMT_BGR24,
                    SWS_FAST_BILINEAR, NULL, NULL, NULL);

                sws_scale(img_convert_ctx, videoFrame->data, videoFrame->linesize, 0, videoCodecContext->height,
                    (uint8_t**)&img.data, &stride);

                callback(img);

                stride = img.step[0];

                auto reverse_convert_ctx = sws_getCachedContext(
                    NULL,
                    out_width,
                    out_height,
                    AV_PIX_FMT_BGR24,
                    out_width,
                    out_height,
                    videoCodecContext->pix_fmt,
                    SWS_FAST_BILINEAR, NULL, NULL, NULL);

                sws_scale(reverse_convert_ctx,
                    (const uint8_t**)&img.data,
                    &stride,
                    0, out_height,
                    videoFrameOut->data, videoFrameOut->linesize
                );

                // preserve pts/dts from decoded frame (rescale if necessary)
                videoFrameOut->pts = videoFrame->pts;

                // send to encoder
                ret = avcodec_send_frame(enc_ctx, videoFrameOut.get());
                if (ret < 0) {
                    fprintf(stderr, "Error sending frame to encoder\n");
                    break;
                }

                // receive packets from encoder
                while (avcodec_receive_packet(enc_ctx, &enc_pkt) == 0) {
                    // Rescale encoder packet timestamps to output stream timebase if needed
                    if (enc_pkt.pts != AV_NOPTS_VALUE)
                        enc_pkt.pts = av_rescale_q(enc_pkt.pts, enc_ctx->time_base, out_stream->time_base);
                    if (enc_pkt.dts != AV_NOPTS_VALUE)
                        enc_pkt.dts = av_rescale_q(enc_pkt.dts, enc_ctx->time_base, out_stream->time_base);
                    enc_pkt.stream_index = out_stream_index;
                    enc_pkt.pos = -1;

                    // Ensure monotonic PTS relative to last_written_pts: if necessary bump
                    if (last_written_pts[out_stream_index] != AV_NOPTS_VALUE && enc_pkt.pts != AV_NOPTS_VALUE) {
                        if (enc_pkt.pts <= last_written_pts[out_stream_index]) {
                            int64_t shift = last_written_pts[out_stream_index] - enc_pkt.pts + 1;
                            enc_pkt.pts += shift;
                            enc_pkt.dts = (enc_pkt.dts != AV_NOPTS_VALUE) ? enc_pkt.dts + shift : enc_pkt.dts;
                        }
                    }

                    // write packet
                    ret = av_interleaved_write_frame(output_format_context, &enc_pkt);
                    if (ret < 0) {
                        fprintf(stderr, "Error while writing encoded packet\n");
                        av_packet_unref(&enc_pkt);
                        break;
                    }

                    // update last written pts
                    int64_t written_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
                    if (written_ts != AV_NOPTS_VALUE)
                        last_written_pts[out_stream_index] = written_ts;

                    av_packet_unref(&enc_pkt);
                }
            }
        }
        else
        {
            // stream-copy other streams (audio/subs) with rescaled timestamps
            packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base, AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
            packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base, AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
            packet.duration = av_rescale_q(packet.duration, in_stream->time_base, out_stream->time_base);
            packet.stream_index = out_stream_index;
            packet.pos = -1;

            ret = av_interleaved_write_frame(output_format_context, &packet);
            if (ret < 0) {
                fprintf(stderr, "Error muxing packet\n");
                av_packet_unref(&packet);
                break;
            }

            // update last written pts for this stream
            int64_t written_ts = (packet.pts != AV_NOPTS_VALUE) ? packet.pts : packet.dts;
            if (written_ts != AV_NOPTS_VALUE)
                last_written_pts[out_stream_index] = written_ts;
        }
        av_packet_unref(&packet);
    }

    // flush encoder
    if (videoCodec->capabilities & AV_CODEC_CAP_DELAY)
    {
        av_init_packet(&enc_pkt);
        enc_pkt.data = NULL;
        enc_pkt.size = 0;
        while ((ret = avcodec_send_frame(enc_ctx, nullptr)) >= 0)
        {
            while (avcodec_receive_packet(enc_ctx, &enc_pkt) == 0) {
                if (enc_pkt.pts != AV_NOPTS_VALUE)
                    enc_pkt.pts = av_rescale_q(enc_pkt.pts, enc_ctx->time_base, outputVideoStream->time_base);
                if (enc_pkt.dts != AV_NOPTS_VALUE)
                    enc_pkt.dts = av_rescale_q(enc_pkt.dts, enc_ctx->time_base, outputVideoStream->time_base);
                enc_pkt.stream_index = output_video_stream_index;
                enc_pkt.pos = -1;

                // ensure monotonicity
                if (last_written_pts[output_video_stream_index] != AV_NOPTS_VALUE && enc_pkt.pts != AV_NOPTS_VALUE) {
                    if (enc_pkt.pts <= last_written_pts[output_video_stream_index]) {
                        int64_t shift = last_written_pts[output_video_stream_index] - enc_pkt.pts + 1;
                        enc_pkt.pts += shift;
                        enc_pkt.dts = (enc_pkt.dts != AV_NOPTS_VALUE) ? enc_pkt.dts + shift : enc_pkt.dts;
                    }
                }

                av_interleaved_write_frame(output_format_context, &enc_pkt);
                int64_t written_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
                if (written_ts != AV_NOPTS_VALUE)
                    last_written_pts[output_video_stream_index] = written_ts;

                av_packet_unref(&enc_pkt);
            }
        }
    }

    av_write_trailer(output_format_context);

    if (output_format_context && !(output_format_context->oformat->flags & AVFMT_NOFILE))
        avio_closep(&output_format_context->pb);

    return 0;
}
