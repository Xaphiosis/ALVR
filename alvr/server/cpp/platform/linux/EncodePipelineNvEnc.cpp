#include "EncodePipelineNvEnc.h"
#include "ALVR-common/packet_types.h"
#include "alvr_server/Settings.h"
#include "ffmpeg_helper.h"
#include <chrono>
#include "alvr_server/Logger.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace {

const char *encoder(ALVR_CODEC codec) {
    switch (codec) {
    case ALVR_CODEC_H264:
        return "h264_nvenc";
    case ALVR_CODEC_H265:
        return "hevc_nvenc";
    }
    throw std::runtime_error("invalid codec " + std::to_string(codec));
}

} // namespace

/*
#define CUDA_DRVAPI_CALL( call ) \
    do \
    { \
        CUresult err__ = call; \
        if (err__ != CUDA_SUCCESS) \
        { \
            const char *szErrName = NULL; \
            cuGetErrorName(err__, &szErrName); \
            std::ostringstream errorLog; \
            errorLog << "CUDA driver API error " << szErrName ; \
            throw NVENCException::makeNVENCException(errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__); \
        } \
    } \
    while (0)
*/

int getVulkanMemoryHandle(VkDevice device,
        VkDeviceMemory memory) {
    // Get handle to memory of the VkImage

    int fd = -1;
    VkMemoryGetFdInfoKHR fdInfo = { };
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device,
            "vkGetMemoryFdKHR");

    if (!func) {
        throw std::runtime_error("Failed to locate function vkGetMemoryFdKHR\n");
        return -1;
    }

    VkResult r = func(device, &fdInfo, &fd);
    if (r != VK_SUCCESS) {
        // FIXME std::runtime_error isn't printf
        throw std::runtime_error("Failed executing vkGetMemoryFdKHR " + std::to_string(r));
    }

    return fd;
}

alvr::EncodePipelineNvEnc::EncodePipelineNvEnc(Renderer *render,
                                               VkContext &vk_ctx,
                                               VkFrame &input_frame,
                                               VkFrameCtx &vk_frame_ctx,
                                               uint32_t width,
                                               uint32_t height) {
    int err;
    r = render;
    const auto &settings = Settings::Instance();

    auto input_frame_ctx = (AVHWFramesContext *)vk_frame_ctx.ctx->data;
    assert(input_frame_ctx->sw_format == AV_PIX_FMT_BGRA);
    vk_frame = input_frame.make_av_frame(vk_frame_ctx);

    // derive CUDA device context from Vulkan one
    err = av_hwdevice_ctx_create_derived(&hw_ctx, AV_HWDEVICE_TYPE_CUDA,
                                         vk_ctx.ctx, 0);
    if (err < 0) {
        throw alvr::AvException("Failed to derive CUDA device:", err);
    }
    // dig down to get the actual CUcontext we'll be using with CUDA
    AVHWDeviceContext *hwDevContext = (AVHWDeviceContext*)(hw_ctx->data);
    AVCUDADeviceContext *cudaDevCtx = (AVCUDADeviceContext*)(hwDevContext->hwctx);
    m_cuContext = &(cudaDevCtx->cuda_ctx); // FIXME: unref in destructor?

    // abstract cuda buffer for use with FFMPEG functions
    AVBufferRef *cuda_frame_ctx = av_hwframe_ctx_alloc(hw_ctx); // will become owned by encoder
    if (cuda_frame_ctx == NULL) {
        throw std::runtime_error("Failed to allocate frame context for CUDA device");
    }
    AVHWFramesContext* frameCtxPtr = (AVHWFramesContext*)(cuda_frame_ctx->data);
    frameCtxPtr->width = width;
    frameCtxPtr->height = height;
    frameCtxPtr->format = AV_PIX_FMT_CUDA;
    // FIXME: does nvenc really not accept BGRA? why BGRx when vulkan seems to have RGBA?
    // why not AV_PIX_FMT_0BGR32 ?
    frameCtxPtr->sw_format = AV_PIX_FMT_BGR0; // compatible with VK_FORMAT_R8G8B8A8_UNORM?
    frameCtxPtr->device_ref = hw_ctx;
    frameCtxPtr->device_ctx = (AVHWDeviceContext*)hw_ctx->data;

    // allocate the CUDA buffer
    Warn("TRY: av_hwframe_ctx_init\n");
    if ((err = av_hwframe_ctx_init(cuda_frame_ctx)) < 0) {
      av_buffer_unref(&cuda_frame_ctx);
      throw alvr::AvException("Failed to initialize frame context for CUDA device:", err);
    }

    // retrieve CUDA frame buffer (this is the frame we'll send to the encoder
    // after data gets copied across from the vulkan side)
    m_cudaFrame = av_frame_alloc(); // FIXME: cleanup?
    Warn("TRY: av_hwframe_get_buffer\n");
    if ((err = av_hwframe_get_buffer(cuda_frame_ctx, m_cudaFrame, 0)) < 0) {
      av_buffer_unref(&cuda_frame_ctx);
      throw alvr::AvException("Failed to get CUDA frame's buffer:", err);
    }
    Warn("OK: av_hwframe_get_buffer\n");

    // TODO; cast vulkan data to CUDA pointer / import the memory
    Warn("TRY: getVulkanMemoryHandle\n");
    auto output = r->GetOutput();
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
    memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memDesc.handle.fd = getVulkanMemoryHandle(vk_ctx.device, output.memory);
    memDesc.size = output.size; // this is memRequirements.size from the code creating the vkIimage, different than extent.width*extent.height*4;
    Warn("OK: getVulkanMemoryHandle\n");

    CUresult cu_err = CUDA_SUCCESS;
    CUcontext cu_old_ctx;
    cu_err = cuCtxPopCurrent(&cu_old_ctx); // FIXME check
    cu_err = cuCtxPushCurrent(*m_cuContext);
    if (cu_err != CUDA_SUCCESS) {
        throw std::runtime_error("CUDA: failed to bind context to this thread.");
    }

    Warn("TRY: cuImportExternalMemory\n");
    CUexternalMemory externalMem;
    // FIXME if this works, can abstract some kind of CUDA call or at least an exception wrapper
    if ((cu_err = cuImportExternalMemory(&externalMem, &memDesc)) != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(cu_err, &szErrName);
        throw std::runtime_error(std::string("CUDA driver API error: ") + szErrName);
    }
    Warn("OK: cuImportExternalMemory\n");

    // we have the external memory imported, now need to produce a CUDA array for the source frame

    CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
    arrayDesc.Width = width;
    arrayDesc.Height = height;
    arrayDesc.Depth = 0; // FIXME not clear why 0 and not 1
    arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    arrayDesc.NumChannels = 4;
    arrayDesc.Flags = CUDA_ARRAY3D_COLOR_ATTACHMENT;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapArrayDesc = {};
    mipmapArrayDesc.arrayDesc = arrayDesc;
    mipmapArrayDesc.numLevels = 1;
    mipmapArrayDesc.offset = 0;

    Warn("TRY: cuExternalMemoryGetMappedMipmappedArray\n");
    if ((cu_err = cuExternalMemoryGetMappedMipmappedArray(&m_cuSrcMipmapArray,
                    externalMem, &mipmapArrayDesc)) != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(cu_err, &szErrName);
        throw std::runtime_error(std::string("CUDA driver API error: ") + szErrName);
    }

    Warn("TRY: cuMipmappedArrayGetLevel\n");
    if ((cu_err = cuMipmappedArrayGetLevel(&m_cuSrcArray, m_cuSrcMipmapArray, 0)) != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(cu_err, &szErrName);
        throw std::runtime_error(std::string("CUDA driver API error: ") + szErrName);
    }

    cu_err = cuCtxPopCurrent(&cu_old_ctx); // restore previous thread CUDA context FIXME check

    auto codec_id = ALVR_CODEC(settings.m_codec);
    const char *encoder_name = encoder(codec_id);
    const AVCodec *codec = avcodec_find_encoder_by_name(encoder_name);
    if (codec == nullptr) {
        throw std::runtime_error(std::string("Failed to find encoder ") + encoder_name);
    }

    encoder_ctx = avcodec_alloc_context3(codec);
    if (not encoder_ctx) {
        throw std::runtime_error("failed to allocate NvEnc encoder");
    }

    switch (codec_id) {
    case ALVR_CODEC_H264:
        switch (settings.m_entropyCoding) {
        case ALVR_CABAC:
            av_opt_set(encoder_ctx->priv_data, "coder", "ac", 0);
            break;
        case ALVR_CAVLC:
            av_opt_set(encoder_ctx->priv_data, "coder", "vlc", 0);
            break;
        }
        break;
    case ALVR_CODEC_H265:
        break;
    }

    switch (settings.m_rateControlMode) {
    case ALVR_CBR:
        av_opt_set(encoder_ctx->priv_data, "rc", "cbr", 0);
        break;
    case ALVR_VBR:
        av_opt_set(encoder_ctx->priv_data, "rc", "vbr", 0);
        break;
    }

    char preset[] = "p0";
    // replace 0 with preset number
    preset[1] += settings.m_nvencQualityPreset;
    av_opt_set(encoder_ctx->priv_data, "preset", preset, 0);

    if (settings.m_nvencAdaptiveQuantizationMode == 1) {
        av_opt_set_int(encoder_ctx->priv_data, "spatial_aq", 1, 0);
    } else if (settings.m_nvencAdaptiveQuantizationMode == 2) {
        av_opt_set_int(encoder_ctx->priv_data, "temporal_aq", 1, 0);
    }

    if (settings.m_nvencEnableWeightedPrediction) {
        av_opt_set_int(encoder_ctx->priv_data, "weighted_pred", 1, 0);
    }

    av_opt_set_int(encoder_ctx->priv_data, "tune", settings.m_nvencTuningPreset, 0);
    av_opt_set_int(encoder_ctx->priv_data, "zerolatency", 1, 0);
    // Delay isn't actually a delay instead its how many surfaces to encode at a time
    av_opt_set_int(encoder_ctx->priv_data, "delay", 1, 0);
    av_opt_set_int(encoder_ctx->priv_data, "forced-idr", 1, 0);

    /**
     * We will recieve a frame from HW as AV_PIX_FMT_VULKAN which will converted to AV_PIX_FMT_BGRA
     * as SW format when we get it from HW.
     * But NVEnc support only BGR0 format and we easy can just to force it
     * Because:
     * AV_PIX_FMT_BGRA - 28  ///< packed BGRA 8:8:8:8, 32bpp, BGRABGRA...
     * AV_PIX_FMT_BGR0 - 123 ///< packed BGR 8:8:8,    32bpp, BGRXBGRX...   X=unused/undefined
     *
     * We just to ignore the alpha channel and it's done
     */
    encoder_ctx->pix_fmt = AV_PIX_FMT_CUDA; // must be same as cuda_frame_ctx->format
    // hw_frames_ctx must be set when using GPU frames as input
    encoder_ctx->hw_frames_ctx = av_buffer_ref(cuda_frame_ctx);
    encoder_ctx->width = width;
    encoder_ctx->height = height;
    encoder_ctx->time_base = {1, (int)1e9};
    encoder_ctx->framerate = AVRational{settings.m_refreshRate, 1};
    encoder_ctx->sample_aspect_ratio = AVRational{1, 1};
    encoder_ctx->max_b_frames = 0;
    encoder_ctx->gop_size = INT16_MAX;
    auto params = FfiDynamicEncoderParams {};
    params.updated = true;
    params.bitrate_bps = 30'000'000;
    params.framerate = 60.0;
    SetParams(params);

    err = avcodec_open2(encoder_ctx, codec, NULL);
    if (err < 0) {
        throw alvr::AvException("Cannot open video encoder codec:", err);
    }

    hw_frame = av_frame_alloc();
}

alvr::EncodePipelineNvEnc::~EncodePipelineNvEnc() {
    av_buffer_unref(&hw_ctx);
    av_frame_free(&hw_frame);
}

void alvr::EncodePipelineNvEnc::PushFrame(uint64_t targetTimestampNs, bool idr) {
    r->Sync();
    timestamp.cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // copy vulkan output frame data to cuda-backed frame
    CUDA_MEMCPY2D copy;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; // srcArray specifies the handle of the source data. srcHost, srcDevice and srcPitch are ignored.
    copy.srcArray = m_cuSrcArray;
    copy.srcXInBytes = 0;
    copy.srcY = 0;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE; // dstDevice and dstPitch specify the (device) base address of the destination data and the bytes per row to apply. dstArray is ignored
    copy.dstDevice = (CUdeviceptr)m_cudaFrame->data[0]; // copy dev ptr as it can change upon resize
    copy.dstPitch = m_cudaFrame->linesize[0];
    copy.dstXInBytes = 0;
    copy.dstY = 0;
    copy.WidthInBytes = m_cudaFrame->width * 4; // RGBA
    copy.Height = m_cudaFrame->height;

    CUresult cu_err = CUDA_SUCCESS;
    CUcontext cu_old_ctx;
    cu_err = cuCtxPopCurrent(&cu_old_ctx); // FIXME check
    cu_err = cuCtxPushCurrent(*m_cuContext);
    if (cu_err != CUDA_SUCCESS) {
        throw std::runtime_error("CUDA: failed to bind context to this thread.");
    }

    // try perform the copy, and do nothing else for now FIXME

    /* printf("dstDevice = %p, dstPitch = %u, width = %u, height = %u\n", */
    /*         copy.dstDevice, (unsigned)copy.dstPitch, */
    /*         (unsigned)copy.WidthInBytes, (unsigned)copy.Height); */
    /* printf("dst accelerated: %s\n", m_cudaFrame->hw_frames_ctx ? "yes" : "no"); */
    if ((cu_err = cuMemcpy2D(&copy)) != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(cu_err, &szErrName);
        throw std::runtime_error(std::string("CUDA driver API error: ") + szErrName);
    }

    cu_err = cuCtxPopCurrent(&cu_old_ctx); // restore previous thread CUDA context FIXME check

    m_cudaFrame->pict_type = idr ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
    m_cudaFrame->pts = targetTimestampNs;

    int err;
    if ((err = avcodec_send_frame(encoder_ctx, m_cudaFrame)) < 0) {
        throw alvr::AvException("avcodec_send_frame failed:", err);
    }
}
