#include "TransformVideo.h"

#include <stdint.h>
#include <sys/stat.h>
#include <cstdio>
#include <ctime>

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

// --- Smart deleter for AVFrame (RAII) ---
struct AVFrameDeleter { void operator()(AVFrame* f) const { av_frame_free(&f); } };
using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

// --- Convenience error printer for FFmpeg ---
static void ReportError(int ret) {
    char buf[AV_ERROR_MAX_STRING_SIZE]{};
    fprintf(stderr, "Error: %s\n",
        av_make_error_string(buf, AV_ERROR_MAX_STRING_SIZE, ret));
}

// --- Helper: check if file exists ---
static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

// Returns true if file looks complete within allowed tail loss.
// max_loss_ms: allowed missing tail in milliseconds (e.g. 2000 = allow up to 2s missing).
// Behavior:
//  - Opens file, finds stream info, requires at least one video stream.
//  - Probes up to probe_packet_read_limit packets and records the largest packet PTS per stream.
//  - If container reports duration, compares reported end time to largest observed packet PTS for video stream.
//    Accepts file when (reported_end - observed_last_packet) <= max_loss_ms.
//  - If duration is unknown, accepts when we observe at least one video packet (fragmented mp4/mkv case).
//  - Performs an optional seek-check near end only when duration is known and the observed gap is slightly larger than tolerance.
static bool is_output_complete(const char* path, int64_t max_loss_ms = 500) {
    if (!path) return false;

    AVFormatContext* fmt = nullptr;
    int ret = avformat_open_input(&fmt, path, nullptr, nullptr);
    if (ret < 0) return false;
    auto fmt_guard = MakeGuard(&fmt, avformat_close_input);

    if ((ret = avformat_find_stream_info(fmt, nullptr)) < 0) return false;
    if (fmt->nb_streams == 0) return false;

    // Find primary video stream (choose first video stream)
    int video_idx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        if (fmt->streams[i]->codecpar && fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = (int)i;
            break;
        }
    }
    if (video_idx < 0) return false;

    // Prepare probe: scan up to N packets and record max packet PTS (in stream timebase) for the video stream.
    const int probe_packet_read_limit = 1000;
    AVPacket pkt{};
    //av_init_packet(&pkt);
    int packets_read = 0;
    int64_t max_pkt_pts_video = AV_NOPTS_VALUE;
    while (packets_read < probe_packet_read_limit && (ret = av_read_frame(fmt, &pkt)) >= 0) {
        if (pkt.stream_index == video_idx) {
            if (pkt.pts != AV_NOPTS_VALUE) {
                if (max_pkt_pts_video == AV_NOPTS_VALUE || pkt.pts > max_pkt_pts_video)
                    max_pkt_pts_video = pkt.pts;
            }
            else if (pkt.dts != AV_NOPTS_VALUE) {
                if (max_pkt_pts_video == AV_NOPTS_VALUE || pkt.dts > max_pkt_pts_video)
                    max_pkt_pts_video = pkt.dts;
            }
        }
        ++packets_read;
        av_packet_unref(&pkt);
    }
    if (ret < 0 && ret != AVERROR_EOF) return false;

    // If we didn't observe any video packet during probe and duration unknown -> consider incomplete
    if (max_pkt_pts_video == AV_NOPTS_VALUE && fmt->duration == AV_NOPTS_VALUE) return false;

    // Convert observed last packet PTS to microseconds
    int64_t observed_last_us = AV_NOPTS_VALUE;
    if (max_pkt_pts_video != AV_NOPTS_VALUE) {
        observed_last_us = av_rescale_q(max_pkt_pts_video,
            fmt->streams[video_idx]->time_base,
            AVRational{ 1, AV_TIME_BASE });
    }

    // If container reports duration, compare and allow tolerance
    if (fmt->duration != AV_NOPTS_VALUE && fmt->duration > 0) {
        int64_t container_duration_us = fmt->duration; // already in AV_TIME_BASE units (microseconds)
        // If no observed packets, rely on duration (accept only if duration > 0 and small files)
        if (observed_last_us == AV_NOPTS_VALUE) {
            // we probed but didn't see video packets; treat as suspicious => incomplete
            return false;
        }
        int64_t gap_us = container_duration_us - observed_last_us;
        // If observed_last_us can be slightly greater than duration due to rounding, clamp
        if (gap_us < 0) gap_us = 0;
        if (gap_us <= max_loss_ms * 1000) {
            return true;
        }
        // If gap is slightly larger than tolerance, perform a seek/read near end to verify presence of late packets.
        const int64_t verify_seek_back_us = (max_loss_ms + 1000) * 1000; // seek ~ (tolerance + 1s) back
        int64_t seek_target_us = container_duration_us > verify_seek_back_us ? container_duration_us - verify_seek_back_us : 0;
        AVRational tb_time = { 1, AV_TIME_BASE };
        int64_t target_ts = av_rescale_q(seek_target_us, tb_time, fmt->streams[video_idx]->time_base);
        if (av_seek_frame(fmt, video_idx, target_ts, AVSEEK_FLAG_BACKWARD) < 0) return false;
        //av_init_packet(&pkt);
        pkt = {};
        if (av_read_frame(fmt, &pkt) < 0) return false;
        bool got_video_after_seek = (pkt.stream_index == video_idx);
        av_packet_unref(&pkt);
        return got_video_after_seek;
    }

    // If duration unknown (fragmented/streaming-friendly file), accept if we observed at least one video packet
    return (observed_last_us != AV_NOPTS_VALUE);
}

// ============================================================================
//  MAIN FUNCTION
// ============================================================================
//
//  This function safely transforms a video file using OpenCV callback,
//  supports automatic backup/resume logic, and ensures crash recovery.
//
//  Key behaviors:
//   - If output is already finished, do nothing.
//   - If output is incomplete, rename it to .bak.
//   - If .bak exists, reuse its content (remux) and resume from last frame.
//   - Delete .bak only once new frames are successfully written.
//
int TransformVideo(const char* in_filename,
    const char* out_filename,
    std::function<void(cv::Mat&)> callback,
    int upscale, int downscale)
{
    std::string bak_filename = std::string(out_filename) + ".bak";

    // ------------------------------------------------------------------------
    // Step 1: If output file already exists and looks finished, skip entirely
    // ------------------------------------------------------------------------
    if (file_exists(out_filename) && is_output_complete(out_filename)) {
        fprintf(stderr, "Output '%s' already complete. Nothing to do.\n", out_filename);
        return 0;
    }

    // ------------------------------------------------------------------------
    // Step 2: Handle backup (.bak) and incomplete output files
    // ------------------------------------------------------------------------
    // If a .bak already exists, we'll use it directly.
    // Otherwise, if output file exists but is incomplete, rename it to .bak
    if (file_exists(bak_filename.c_str())) {
        fprintf(stderr, "Using existing backup file '%s'\n", bak_filename.c_str());
    }
    else if (file_exists(out_filename)) {
        fprintf(stderr, "Renaming incomplete output to '%s'\n", bak_filename.c_str());
        if (rename(out_filename, bak_filename.c_str()) != 0) {
            fprintf(stderr, "Warning: failed to rename output to %s: %s\n",
                bak_filename.c_str(), strerror(errno));
            bak_filename.clear();  // continue safely without backup
        }
    }

    // ------------------------------------------------------------------------
    // Step 3: Open input (source) video
    // ------------------------------------------------------------------------
    AVFormatContext* input_fmt = nullptr;
    int ret = avformat_open_input(&input_fmt, in_filename, nullptr, nullptr);
    if (ret < 0) { ReportError(ret); return 1; }
    auto input_guard = MakeGuard(&input_fmt, avformat_close_input);
    if (avformat_find_stream_info(input_fmt, nullptr) < 0) {
        fprintf(stderr, "Cannot get input stream info\n");
        return 1;
    }

    // ------------------------------------------------------------------------
    // Step 4: Create output context
    // ------------------------------------------------------------------------
    AVFormatContext* out_fmt = nullptr;
    avformat_alloc_output_context2(&out_fmt, nullptr, "matroska", out_filename);
    if (!out_fmt) { fprintf(stderr, "Cannot allocate output context\n"); return 1; }
    auto out_guard = MakeGuard(out_fmt, avformat_free_context);
    out_fmt->flags |= AVFMT_FLAG_NOBUFFER;

    // Prepare stream mapping
    int video_idx = -1;
    AVStream* in_video = nullptr;
    AVStream* out_video = nullptr;
    std::vector<int> stream_map(input_fmt->nb_streams, -1);
    int out_stream_count = 0;

    // Copy all streams (video + audio/subtitle) structure into output
    for (unsigned i = 0; i < input_fmt->nb_streams; i++) {
        AVStream* in_st = input_fmt->streams[i];
        AVCodecParameters* par = in_st->codecpar;
        if (par->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = i;
            in_video = in_st;
        }
        else if (par->codec_type != AVMEDIA_TYPE_AUDIO &&
            par->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            continue;
        }

        stream_map[i] = out_stream_count++;
        AVStream* out_st = avformat_new_stream(out_fmt, nullptr);
        avcodec_parameters_copy(out_st->codecpar, par);
        out_st->codecpar->codec_tag = 0;

        if (par->codec_type == AVMEDIA_TYPE_VIDEO)
            out_video = out_st;
    }

    // Open output file handle if needed
    if (!(out_fmt->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&out_fmt->pb, out_filename, AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Cannot open output file\n");
            return 1;
        }
    }

    // ------------------------------------------------------------------------
    // Step 5: Initialize decoder and encoder
    // ------------------------------------------------------------------------
    AVCodecContext* dec_ctx = avcodec_alloc_context3(nullptr);
    avcodec_parameters_to_context(dec_ctx, in_video->codecpar);
    auto dec_guard = MakeGuard(&dec_ctx, avcodec_free_context);

    auto dec = avcodec_find_decoder(dec_ctx->codec_id);
    if (!dec || avcodec_open2(dec_ctx, dec, nullptr) < 0) {
        fprintf(stderr, "Decoder error\n"); return 1;
    }

    // Encoder (same codec as input)
    auto enc = avcodec_find_encoder(dec_ctx->codec_id);
    if (!enc) { fprintf(stderr, "No encoder found\n"); return 1; }

    AVCodecContext* enc_ctx = avcodec_alloc_context3(enc);
    enc_ctx->width = (dec_ctx->width * upscale / downscale) & ~7;
    enc_ctx->height = (dec_ctx->height * upscale / downscale) & ~7;
    enc_ctx->time_base = in_video->time_base;
    enc_ctx->sample_aspect_ratio = dec_ctx->sample_aspect_ratio;
    enc_ctx->pix_fmt = enc->pix_fmts ? enc->pix_fmts[0] : dec_ctx->pix_fmt;
    enc_ctx->gop_size = 1;
    enc_ctx->max_b_frames = 2;
    if (out_fmt->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(enc_ctx, enc, nullptr) < 0) { fprintf(stderr, "Encoder open error\n"); return 1; }

    // Copy encoder parameters into output stream
    avcodec_parameters_from_context(out_video->codecpar, enc_ctx);
    out_video->time_base = enc_ctx->time_base;

    // Write container header
    avformat_write_header(out_fmt, nullptr);

    // Track last PTS written per stream
    std::vector<int64_t> last_written_pts(out_fmt->nb_streams, AV_NOPTS_VALUE);

    // ------------------------------------------------------------------------
    // Step 6: If .bak exists, remux its contents into the new output file
    // ------------------------------------------------------------------------
    // This ensures we retain all previously finished frames before resuming.
    if (!bak_filename.empty() && file_exists(bak_filename.c_str())) {
        AVFormatContext* bak_fmt = nullptr;
        if (avformat_open_input(&bak_fmt, bak_filename.c_str(), nullptr, nullptr) == 0) {
            avformat_find_stream_info(bak_fmt, nullptr);
            AVPacket pkt{}; //av_init_packet(&pkt);

            while (av_read_frame(bak_fmt, &pkt) >= 0) {
                int oidx = pkt.stream_index;
                if (oidx < 0 || oidx >= (int)out_fmt->nb_streams) {
                    av_packet_unref(&pkt);
                    continue;
                }

                AVStream* in_st = bak_fmt->streams[pkt.stream_index];
                AVStream* out_st = out_fmt->streams[oidx];

                // Rescale timestamps between backup and output
                if (pkt.pts != AV_NOPTS_VALUE)
                    pkt.pts = av_rescale_q_rnd(pkt.pts, in_st->time_base, out_st->time_base,
                        AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                if (pkt.dts != AV_NOPTS_VALUE)
                    pkt.dts = av_rescale_q_rnd(pkt.dts, in_st->time_base, out_st->time_base,
                        AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                pkt.duration = av_rescale_q(pkt.duration, in_st->time_base, out_st->time_base);
                pkt.stream_index = oidx;
                pkt.pos = -1;

                // Write packet and remember last PTS
                if (av_interleaved_write_frame(out_fmt, &pkt) >= 0) {
                    int64_t ts = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
                    if (ts != AV_NOPTS_VALUE) last_written_pts[oidx] = ts;
                }
                av_packet_unref(&pkt);
            }
            avformat_close_input(&bak_fmt);
        }
    }

    // ------------------------------------------------------------------------
    // Step 7: Resume decoding from where we left off (seek input)
    // ------------------------------------------------------------------------
    // We use last_written_pts from the video stream to resume decoding
    // near the frame after the last one written during remux.
    int out_video_idx = out_video->index;
    if (out_video_idx >= 0 && last_written_pts[out_video_idx] != AV_NOPTS_VALUE) {
        int64_t last_out_pts = last_written_pts[out_video_idx];
        int64_t seek_target = av_rescale_q(last_out_pts,
            out_video->time_base,
            in_video->time_base);
        if (av_seek_frame(input_fmt, video_idx, seek_target, AVSEEK_FLAG_BACKWARD) >= 0) {
            avcodec_flush_buffers(dec_ctx);
            fprintf(stderr, "Resuming input near pts %" PRId64 "\n", last_out_pts);
        }
        else {
            fprintf(stderr, "Warning: failed to seek for resume\n");
        }
    }

    // ------------------------------------------------------------------------
    // Step 8: Initialize scaling + frame buffers
    // ------------------------------------------------------------------------
    AVFramePtr frame(av_frame_alloc()), frame_out(av_frame_alloc());
    frame_out->format = dec_ctx->pix_fmt;
    frame_out->width = enc_ctx->width;
    frame_out->height = enc_ctx->height;
    av_frame_get_buffer(frame_out.get(), 16);

    SwsContext* to_bgr = sws_getContext(
        dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        enc_ctx->width, enc_ctx->height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    SwsContext* from_bgr = sws_getContext(
        enc_ctx->width, enc_ctx->height, AV_PIX_FMT_BGR24,
        enc_ctx->width, enc_ctx->height, dec_ctx->pix_fmt,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    bool bak_deleted = false; // will become true once we output new frames

    // ------------------------------------------------------------------------
    // Step 9: Main decode / process / encode loop
    // ------------------------------------------------------------------------
    AVPacket pkt{};
    //av_init_packet(&pkt);

    while (av_read_frame(input_fmt, &pkt) >= 0) {
        if (pkt.stream_index != video_idx) { av_packet_unref(&pkt); continue; }

        if (avcodec_send_packet(dec_ctx, &pkt) < 0) {
            av_packet_unref(&pkt);
            break;
        }
        av_packet_unref(&pkt);

        while (avcodec_receive_frame(dec_ctx, frame.get()) == 0) {
            // Convert decoded frame to OpenCV Mat for user callback
            cv::Mat img(enc_ctx->height, enc_ctx->width, CV_8UC3);
            int stride = img.step[0];
            sws_scale(to_bgr, frame->data, frame->linesize, 0, dec_ctx->height,
                &img.data, &stride);

            // User-defined frame transformation
            callback(img);

            // Convert processed Mat back into AVFrame for encoding
            sws_scale(from_bgr, (const uint8_t**)&img.data, &stride, 0, enc_ctx->height,
                frame_out->data, frame_out->linesize);
            frame_out->pts = frame->pts;

            // Encode + write to output
            if (avcodec_send_frame(enc_ctx, frame_out.get()) < 0) break;
            AVPacket enc_pkt{}; //av_init_packet(&enc_pkt);

            while (avcodec_receive_packet(enc_ctx, &enc_pkt) == 0) {
                enc_pkt.stream_index = out_video_idx;
                enc_pkt.pts = av_rescale_q(enc_pkt.pts, enc_ctx->time_base, out_video->time_base);
                enc_pkt.dts = av_rescale_q(enc_pkt.dts, enc_ctx->time_base, out_video->time_base);
                enc_pkt.pos = -1;

                if ((ret = av_interleaved_write_frame(out_fmt, &enc_pkt)) >= 0) {
                    // --- Step 10: Delete .bak only after actual new frames are written ---
                    if (!bak_deleted && file_exists(bak_filename.c_str())) {
                        if (std::remove(bak_filename.c_str()) == 0)
                            fprintf(stderr, "Deleted backup %s after writing new frames\n", bak_filename.c_str());
                        bak_deleted = true;
                    }
                }
                else {
                    ReportError(ret);
                }
                av_packet_unref(&enc_pkt);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Step 11: Finalize and clean up
    // ------------------------------------------------------------------------
    av_write_trailer(out_fmt);
    if (!(out_fmt->oformat->flags & AVFMT_NOFILE))
        avio_closep(&out_fmt->pb);

    sws_freeContext(to_bgr);
    sws_freeContext(from_bgr);

    fprintf(stderr, "Transform finished successfully.\n");
    return 0;
}
