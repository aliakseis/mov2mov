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

// Improved tolerant completeness check (keeps original semantics)
static bool is_output_complete(const char* path, int64_t max_loss_ms = 500) {
    if (!path) return false;

    AVFormatContext* fmt = nullptr;
    int ret = avformat_open_input(&fmt, path, nullptr, nullptr);
    if (ret < 0) return false;
    auto fmt_guard = MakeGuard(&fmt, avformat_close_input);

    if ((ret = avformat_find_stream_info(fmt, nullptr)) < 0) return false;
    if (fmt->nb_streams == 0) return false;

    int video_idx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        if (fmt->streams[i]->codecpar && fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = (int)i;
            break;
        }
    }
    if (video_idx < 0) return false;

    const int probe_packet_read_limit = 1000;
    AVPacket pkt{};
    int packets_read = 0;
    int64_t max_pkt_pts_video = AV_NOPTS_VALUE;
    int tmp;
    while (packets_read < probe_packet_read_limit && (tmp = av_read_frame(fmt, &pkt)) >= 0) {
        if (pkt.stream_index == video_idx) {
            int64_t cand = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
            if (cand != AV_NOPTS_VALUE && (max_pkt_pts_video == AV_NOPTS_VALUE || cand > max_pkt_pts_video))
                max_pkt_pts_video = cand;
        }
        ++packets_read;
        av_packet_unref(&pkt);
    }
    if (tmp < 0 && tmp != AVERROR_EOF) return false;

    if (max_pkt_pts_video == AV_NOPTS_VALUE && fmt->duration == AV_NOPTS_VALUE) return false;

    int64_t observed_last_us = AV_NOPTS_VALUE;
    if (max_pkt_pts_video != AV_NOPTS_VALUE) {
        observed_last_us = av_rescale_q(max_pkt_pts_video,
            fmt->streams[video_idx]->time_base,
            AVRational{ 1, AV_TIME_BASE });
    }

    if (fmt->duration != AV_NOPTS_VALUE && fmt->duration > 0) {
        int64_t container_duration_us = fmt->duration;
        if (observed_last_us == AV_NOPTS_VALUE) return false;
        int64_t gap_us = container_duration_us - observed_last_us;
        if (gap_us < 0) gap_us = 0;
        if (gap_us <= max_loss_ms * 1000) return true;

        const int64_t verify_seek_back_us = (max_loss_ms + 1000) * 1000;
        int64_t seek_target_us = container_duration_us > verify_seek_back_us ? container_duration_us - verify_seek_back_us : 0;
        AVRational tb_time = { 1, AV_TIME_BASE };
        int64_t target_ts = av_rescale_q(seek_target_us, tb_time, fmt->streams[video_idx]->time_base);
        if (av_seek_frame(fmt, video_idx, target_ts, AVSEEK_FLAG_BACKWARD) < 0) return false;
        pkt = {};
        if (av_read_frame(fmt, &pkt) < 0) return false;
        bool got_video_after_seek = (pkt.stream_index == video_idx);
        av_packet_unref(&pkt);
        return got_video_after_seek;
    }

    return (observed_last_us != AV_NOPTS_VALUE);
}

// ============================================================================
//  MAIN FUNCTION
// ============================================================================
int TransformVideo(const char* in_filename,
    const char* out_filename,
    std::function<void(cv::Mat&)> callback,
    int upscale, int downscale)
{
    if (!in_filename || !out_filename) return 1;
    std::string bak_filename = std::string(out_filename) + ".bak";

    // If output exists and looks finished, skip
    if (file_exists(out_filename) && is_output_complete(out_filename)) {
        fprintf(stderr, "Output '%s' already complete. Nothing to do.\n", out_filename);
        return 0;
    }

    // If .bak exists keep it, else rename incomplete output to .bak
    if (file_exists(bak_filename.c_str())) {
        fprintf(stderr, "Using existing backup file '%s'\n", bak_filename.c_str());
    }
    else if (file_exists(out_filename)) {
        fprintf(stderr, "Renaming incomplete output to '%s'\n", bak_filename.c_str());
        if (rename(out_filename, bak_filename.c_str()) != 0) {
            fprintf(stderr, "Warning: failed to rename output to %s: %s\n",
                bak_filename.c_str(), strerror(errno));
            bak_filename.clear();
        }
    }
    else {
        bak_filename.clear();
    }

    // Open input
    AVFormatContext* input_fmt = nullptr;
    int ret = avformat_open_input(&input_fmt, in_filename, nullptr, nullptr);
    if (ret < 0) { ReportError(ret); return 1; }
    auto input_guard = MakeGuard(&input_fmt, avformat_close_input);
    if ((ret = avformat_find_stream_info(input_fmt, nullptr)) < 0) { ReportError(ret); return 1; }

    // Create output context (Matroska)
    AVFormatContext* out_fmt = nullptr;
    if ((ret = avformat_alloc_output_context2(&out_fmt, nullptr, "matroska", out_filename)) < 0 || !out_fmt) {
        fprintf(stderr, "Cannot allocate output context\n"); return 1;
    }
    auto out_guard = MakeGuard(out_fmt, avformat_free_context);
    out_fmt->flags |= AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;
    out_fmt->flush_packets = 1;

    // Build stream map input->output
    int video_idx = -1;
    AVStream* in_video = nullptr;
    AVStream* out_video = nullptr;
    std::vector<int> stream_map(input_fmt->nb_streams, -1);
    int out_stream_count = 0;
    for (unsigned i = 0; i < input_fmt->nb_streams; ++i) {
        AVStream* in_st = input_fmt->streams[i];
        AVCodecParameters* par = in_st->codecpar;
        if (!par) continue;
        if (par->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = i;
            in_video = in_st;
        }
        else if (par->codec_type != AVMEDIA_TYPE_AUDIO && par->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            continue;
        }
        stream_map[i] = out_stream_count++;
        AVStream* out_st = avformat_new_stream(out_fmt, nullptr);
        if (!out_st) { fprintf(stderr, "Failed allocating output stream\n"); return 1; }
        if ((ret = avcodec_parameters_copy(out_st->codecpar, par)) < 0) { ReportError(ret); return 1; }
        out_st->codecpar->codec_tag = 0;
        if (par->codec_type == AVMEDIA_TYPE_VIDEO) out_video = out_st;
    }

    if (!in_video || !out_video) { fprintf(stderr, "No video stream found in input\n"); return 1; }

    // Open output file handle if needed
    if (!(out_fmt->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&out_fmt->pb, out_filename, AVIO_FLAG_WRITE)) < 0) { ReportError(ret); return 1; }
    }

    // Decoder init
    AVCodecContext* dec_ctx = avcodec_alloc_context3(nullptr);
    if (!dec_ctx) return 1;
    auto dec_guard = MakeGuard(&dec_ctx, avcodec_free_context);
    if ((ret = avcodec_parameters_to_context(dec_ctx, in_video->codecpar)) < 0) { ReportError(ret); return 1; }
    auto dec = avcodec_find_decoder(dec_ctx->codec_id);
    if (!dec) { fprintf(stderr, "Decoder not found\n"); return 1; }
    if ((ret = avcodec_open2(dec_ctx, dec, nullptr)) < 0) { ReportError(ret); return 1; }

    // Encoder init (use same codec id)
    auto enc = avcodec_find_encoder(dec_ctx->codec_id);
    if (!enc) { fprintf(stderr, "Encoder not found\n"); return 1; }
    AVCodecContext* enc_ctx = avcodec_alloc_context3(enc);
    if (!enc_ctx) { fprintf(stderr, "Failed to alloc encoder ctx\n"); return 1; }
    auto enc_guard = MakeGuard(&enc_ctx, avcodec_free_context);

    enc_ctx->width = (dec_ctx->width * upscale / downscale) & ~7;
    enc_ctx->height = (dec_ctx->height * upscale / downscale) & ~7;
    enc_ctx->time_base = in_video->time_base;
    enc_ctx->sample_aspect_ratio = dec_ctx->sample_aspect_ratio;
    enc_ctx->pix_fmt = enc->pix_fmts ? enc->pix_fmts[0] : dec_ctx->pix_fmt;
    enc_ctx->gop_size = 1;
    enc_ctx->max_b_frames = 2;
    if (out_fmt->oformat->flags & AVFMT_GLOBALHEADER) enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if ((ret = avcodec_open2(enc_ctx, enc, nullptr)) < 0) { ReportError(ret); return 1; }

    // Copy encoder params into output video stream and capture its index
    if ((ret = avcodec_parameters_from_context(out_video->codecpar, enc_ctx)) < 0) { ReportError(ret); return 1; }
    out_video->time_base = enc_ctx->time_base;
    int out_video_idx = out_video->index;

    // write header
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "flush_packets", "1", 0);
    if ((ret = avformat_write_header(out_fmt, &opts)) < 0) { ReportError(ret); return 1; }

    // Prepare last-written pts map (indexed by output stream index)
    std::vector<int64_t> last_written_pts(out_fmt->nb_streams, AV_NOPTS_VALUE);

    // Remux .bak if present — use stream_map to map backup input indexes to new output indexes
    if (!bak_filename.empty()) {
        AVFormatContext* bak_fmt = nullptr;
        if (avformat_open_input(&bak_fmt, bak_filename.c_str(), nullptr, nullptr) == 0) {
            if (avformat_find_stream_info(bak_fmt, nullptr) >= 0) {
                AVPacket pkt{};
                while (av_read_frame(bak_fmt, &pkt) >= 0) {
                    int in_idx = pkt.stream_index;
                    if (in_idx < 0 || (unsigned)in_idx >= stream_map.size()) { av_packet_unref(&pkt); continue; }
                    int out_idx = stream_map[in_idx];
                    if (out_idx < 0 || out_idx >= (int)out_fmt->nb_streams) { av_packet_unref(&pkt); continue; }

                    AVStream* in_st = bak_fmt->streams[in_idx];
                    AVStream* out_st = out_fmt->streams[out_idx];

                    if (pkt.pts != AV_NOPTS_VALUE)
                        pkt.pts = av_rescale_q_rnd(pkt.pts, in_st->time_base, out_st->time_base,
                            AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                    if (pkt.dts != AV_NOPTS_VALUE)
                        pkt.dts = av_rescale_q_rnd(pkt.dts, in_st->time_base, out_st->time_base,
                            AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                    pkt.duration = av_rescale_q(pkt.duration, in_st->time_base, out_st->time_base);
                    pkt.stream_index = out_idx;
                    pkt.pos = -1;

                    ret = av_interleaved_write_frame(out_fmt, &pkt);
                    if (ret < 0) {
                        ReportError(ret);
                        av_packet_unref(&pkt);
                        break;
                    }
                    int64_t ts = (pkt.pts != AV_NOPTS_VALUE) ? pkt.pts : pkt.dts;
                    if (ts != AV_NOPTS_VALUE) last_written_pts[out_idx] = ts;
                    av_packet_unref(&pkt);
                }
            }
            avformat_close_input(&bak_fmt);
        }
        else {
            fprintf(stderr, "Warning: could not open backup '%s' to remux\n", bak_filename.c_str());
        }
    }

    // Seek input to resume point if we have last written video pts
    if (out_video_idx >= 0 && last_written_pts[out_video_idx] != AV_NOPTS_VALUE) {
        int64_t last_out_pts = last_written_pts[out_video_idx];
        int64_t seek_target = av_rescale_q(last_out_pts, out_video->time_base, in_video->time_base);
        if (av_seek_frame(input_fmt, video_idx, seek_target, AVSEEK_FLAG_BACKWARD) >= 0) {
            avcodec_flush_buffers(dec_ctx);
            fprintf(stderr, "Resuming input near pts %" PRId64 "\n", last_out_pts);
        }
        else {
            fprintf(stderr, "Warning: failed to seek for resume\n");
        }
    }

    // Prepare frames and scalers
    AVFramePtr frame(av_frame_alloc()), frame_out(av_frame_alloc());
    if (!frame || !frame_out) { fprintf(stderr, "Failed to alloc frames\n"); return 1; }

    frame_out->format = enc_ctx->pix_fmt;
    frame_out->width = enc_ctx->width;
    frame_out->height = enc_ctx->height;
    if ((ret = av_frame_get_buffer(frame_out.get(), 16)) < 0) { ReportError(ret); return 1; }

    SwsContext* to_bgr = sws_getContext(
        dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        enc_ctx->width, enc_ctx->height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    SwsContext* from_bgr = sws_getContext(
        enc_ctx->width, enc_ctx->height, AV_PIX_FMT_BGR24,
        enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!to_bgr || !from_bgr) { fprintf(stderr, "sws_getContext failed\n"); return 1; }

    bool bak_deleted = false;
    AVPacket pkt{};
    while ((ret = av_read_frame(input_fmt, &pkt)) >= 0) {
        if (pkt.stream_index != video_idx) { av_packet_unref(&pkt); continue; }

        // compute packet PTS in output timebase and skip if <= last_written (avoid duplicates)
        int out_idx_for_packet = stream_map[pkt.stream_index];
        if (out_idx_for_packet >= 0 && last_written_pts[out_idx_for_packet] != AV_NOPTS_VALUE) {
            int64_t pkt_pts_out = AV_NOPTS_VALUE;
            if (pkt.pts != AV_NOPTS_VALUE)
                pkt_pts_out = av_rescale_q(pkt.pts, input_fmt->streams[pkt.stream_index]->time_base, out_fmt->streams[out_idx_for_packet]->time_base);
            else if (pkt.dts != AV_NOPTS_VALUE)
                pkt_pts_out = av_rescale_q(pkt.dts, input_fmt->streams[pkt.stream_index]->time_base, out_fmt->streams[out_idx_for_packet]->time_base);
            if (pkt_pts_out != AV_NOPTS_VALUE && pkt_pts_out <= last_written_pts[out_idx_for_packet]) {
                av_packet_unref(&pkt);
                continue;
            }
        }

        if ((ret = avcodec_send_packet(dec_ctx, &pkt)) < 0) {
            ReportError(ret);
            av_packet_unref(&pkt);
            break;
        }
        av_packet_unref(&pkt);

        while ((ret = avcodec_receive_frame(dec_ctx, frame.get())) == 0) {
            // convert decoded frame -> BGR cv::Mat
            cv::Mat img(enc_ctx->height, enc_ctx->width, CV_8UC3);
            uint8_t* dst_data[1] = { img.data };
            int dst_linesize[1] = { static_cast<int>(img.step[0]) };
            if (sws_scale(to_bgr, frame->data, frame->linesize, 0, dec_ctx->height, dst_data, dst_linesize) <= 0) {
                fprintf(stderr, "sws_scale to_bgr produced no data\n");
            }

            // user callback
            callback(img);

            // convert back from BGR -> encoder pix_fmt (frame_out)
            uint8_t* src_data[1] = { img.data };
            int src_linesize[1] = { static_cast<int>(img.step[0]) };
            if (sws_scale(from_bgr, src_data, src_linesize, 0, enc_ctx->height, frame_out->data, frame_out->linesize) <= 0) {
                fprintf(stderr, "sws_scale from_bgr produced no data\n");
            }

            // preserve timing
            frame_out->pts = frame->pts;

            if ((ret = avcodec_send_frame(enc_ctx, frame_out.get())) < 0) {
                ReportError(ret);
                break;
            }

            AVPacket enc_pkt{};
            while ((ret = avcodec_receive_packet(enc_ctx, &enc_pkt)) == 0) {
                // rescale encoded packet timestamps to output stream timebase
                if (enc_pkt.pts != AV_NOPTS_VALUE)
                    enc_pkt.pts = av_rescale_q(enc_pkt.pts, enc_ctx->time_base, out_video->time_base);
                if (enc_pkt.dts != AV_NOPTS_VALUE)
                    enc_pkt.dts = av_rescale_q(enc_pkt.dts, enc_ctx->time_base, out_video->time_base);
                enc_pkt.stream_index = out_video_idx;
                enc_pkt.pos = -1;

                // Ensure monotonic PTS relative to previously remuxed data
                int64_t pkt_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
                if (pkt_ts != AV_NOPTS_VALUE && last_written_pts[out_video_idx] != AV_NOPTS_VALUE && pkt_ts <= last_written_pts[out_video_idx]) {
                    int64_t shift = last_written_pts[out_video_idx] - pkt_ts + 1;
                    if (enc_pkt.pts != AV_NOPTS_VALUE) enc_pkt.pts += shift;
                    if (enc_pkt.dts != AV_NOPTS_VALUE) enc_pkt.dts += shift;
                }

                if ((ret = av_interleaved_write_frame(out_fmt, &enc_pkt)) < 0) {
                    ReportError(ret);
                    av_packet_unref(&enc_pkt);
                    goto finalize;
                }

                // update last written pts
                int64_t written_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
                if (written_ts != AV_NOPTS_VALUE) last_written_pts[out_video_idx] = written_ts;

                // delete backup only after we successfully wrote at least one new encoded packet
                if (!bak_deleted && !bak_filename.empty()) {
                    if (std::remove(bak_filename.c_str()) == 0) {
                        fprintf(stderr, "Deleted backup %s after writing new frames\n", bak_filename.c_str());
                    }
                    else {
                        fprintf(stderr, "Warning: failed to delete backup %s: %s\n", bak_filename.c_str(), strerror(errno));
                    }
                    bak_deleted = true;
                }

                av_packet_unref(&enc_pkt);
            } // receive_packet loop

            if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF && ret < 0) {
                ReportError(ret);
                goto finalize;
            }
        } // receive_frame loop

        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF && ret < 0) {
            ReportError(ret);
            break;
        }
    } // read_frame loop

finalize:
    // Flush encoder
    if (enc && enc_ctx) {
        avcodec_send_frame(enc_ctx, nullptr);
        AVPacket enc_pkt{};
        while (avcodec_receive_packet(enc_ctx, &enc_pkt) == 0) {
            if (enc_pkt.pts != AV_NOPTS_VALUE)
                enc_pkt.pts = av_rescale_q(enc_pkt.pts, enc_ctx->time_base, out_video->time_base);
            if (enc_pkt.dts != AV_NOPTS_VALUE)
                enc_pkt.dts = av_rescale_q(enc_pkt.dts, enc_ctx->time_base, out_video->time_base);
            enc_pkt.stream_index = out_video_idx;
            enc_pkt.pos = -1;

            // Ensure monotonicity
            int64_t pkt_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
            if (pkt_ts != AV_NOPTS_VALUE && last_written_pts[out_video_idx] != AV_NOPTS_VALUE && pkt_ts <= last_written_pts[out_video_idx]) {
                int64_t shift = last_written_pts[out_video_idx] - pkt_ts + 1;
                if (enc_pkt.pts != AV_NOPTS_VALUE) enc_pkt.pts += shift;
                if (enc_pkt.dts != AV_NOPTS_VALUE) enc_pkt.dts += shift;
            }

            if (av_interleaved_write_frame(out_fmt, &enc_pkt) < 0) {
                ReportError(ret);
                av_packet_unref(&enc_pkt);
                break;
            }
            int64_t written_ts = (enc_pkt.pts != AV_NOPTS_VALUE) ? enc_pkt.pts : enc_pkt.dts;
            if (written_ts != AV_NOPTS_VALUE) last_written_pts[out_video_idx] = written_ts;
            av_packet_unref(&enc_pkt);
        }
    }

    av_write_trailer(out_fmt);
    if (!(out_fmt->oformat->flags & AVFMT_NOFILE)) avio_closep(&out_fmt->pb);

    sws_freeContext(to_bgr);
    sws_freeContext(from_bgr);

    fprintf(stderr, "Transform finished.\n");
    return 0;
}
