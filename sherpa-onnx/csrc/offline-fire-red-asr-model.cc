// sherpa-onnx/csrc/offline-fire-red-asr-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-model.h"
#include "sherpa-onnx/csrc/ort-env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if SHERPA_ONNX_ENABLE_GPU
// Minimal CUDA runtime ABI declaration, so that no CUDA toolkit headers are
// required at build time (libcudart is linked from csrc/CMakeLists.txt).
extern "C" int cudaMemcpy(void *dst, const void *src, size_t count, int kind);
extern "C" int cudaMemset(void *dst, int value, size_t count);
extern "C" int cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height,
                            int kind);
#endif

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

static inline bool IsCudaProvider(const std::string &provider) {
  return provider == "cuda";
}

#if SHERPA_ONNX_ENABLE_GPU
// cudaMemcpyKind::cudaMemcpyDefault; UVA resolves the copy direction from
// the pointers, so both host and device sources/destinations work.
constexpr int kCudaMemcpyDefault = 4;

static void CudaCopy(void *dst, const void *src, size_t nbytes) {
  int err = cudaMemcpy(dst, src, nbytes, kCudaMemcpyDefault);
  if (err != 0) {
    SHERPA_ONNX_LOGE("cudaMemcpy failed with error code %d", err);
    SHERPA_ONNX_EXIT(-1);
  }
}

static bool CudaGraphEnvEnabled() {
  const char *p = std::getenv("SHERPA_ONNX_CUDA_GRAPH");
  return p != nullptr && p[0] == '1' && p[1] == '\0';
}

static void CudaMemsetZero(void *dst, size_t nbytes) {
  int err = cudaMemset(dst, 0, nbytes);
  if (err != 0) {
    SHERPA_ONNX_LOGE("cudaMemset failed with error code %d", err);
    SHERPA_ONNX_EXIT(-1);
  }
}

static void CudaCopy2D(void *dst, size_t dpitch, const void *src,
                       size_t spitch, size_t width, size_t height) {
  int err = cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                         kCudaMemcpyDefault);
  if (err != 0) {
    SHERPA_ONNX_LOGE("cudaMemcpy2D failed with error code %d", err);
    SHERPA_ONNX_EXIT(-1);
  }
}

static int32_t CudaGraphEnvBucket(const char *name, int32_t default_value) {
  const char *p = std::getenv(name);
  if (p == nullptr || p[0] == '\0') {
    return default_value;
  }
  int32_t v = std::atoi(p);
  return v > 0 ? v : default_value;
}

static inline int64_t RoundUp(int64_t v, int64_t multiple) {
  return (v + multiple - 1) / multiple * multiple;
}
#endif

}  // namespace

class OfflineFireRedAsrModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    InitCudaGraphOptions();
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.fire_red_asr.encoder),
        sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.fire_red_asr.decoder),
        DecoderSessOpts());
    InitDecoder(nullptr, 0);

    InitCudaIOBinding();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    InitCudaGraphOptions();
    {
      auto buf = ReadFile(mgr, config.fire_red_asr.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.fire_red_asr.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    InitCudaIOBinding();
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value> ForwardEncoder(
      Ort::Value features, Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs{std::move(features),
                                     std::move(features_length)};

    std::vector<Ort::Value> encoder_out;

    if (use_cuda_iobinding_) {
      // Encoder outputs (cross_k, cross_v) are used multiple times in decoder
      // steps, so keep them on GPU to avoid device<->host copies.
      Ort::IoBinding binding(*encoder_sess_);
      binding.BindInput(encoder_input_names_ptr_[0], inputs[0]);
      binding.BindInput(encoder_input_names_ptr_[1], inputs[1]);

      binding.BindOutput(encoder_output_names_ptr_[0], *cuda_mem_info_);
      binding.BindOutput(encoder_output_names_ptr_[1], *cuda_mem_info_);
      if (encoder_output_names_ptr_.size() > 2) {
        // enc_mask is small; keep it on GPU as well since the decoder
        // consumes it.
        binding.BindOutput(encoder_output_names_ptr_[2], *cuda_mem_info_);
      }

      binding.SynchronizeInputs();
      encoder_sess_->Run(GetRunOptionsWithArenaShrinkage(), binding);
      binding.SynchronizeOutputs();
      encoder_out = binding.GetOutputValues();
    } else {
      encoder_out = encoder_sess_->Run(
          GetRunOptionsWithArenaShrinkage(), encoder_input_names_ptr_.data(),
          inputs.data(), inputs.size(), encoder_output_names_ptr_.data(),
          encoder_output_names_ptr_.size());
    }

    // Models exported before the cross_mask support have only 2 outputs.
    Ort::Value enc_mask =
        encoder_out.size() > 2 ? std::move(encoder_out[2]) : Ort::Value{};

    return {std::move(encoder_out[0]), std::move(encoder_out[1]),
            std::move(enc_mask)};
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value, Ort::Value>
  ForwardDecoder(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
                 Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
                 Ort::Value n_layer_cross_v, Ort::Value offset,
                 Ort::Value cross_mask) {
#if SHERPA_ONNX_ENABLE_GPU
    if (use_cuda_graph_) {
      auto self_shape =
          n_layer_self_k_cache.GetTensorTypeAndShapeInfo().GetShape();
      auto cross_shape =
          n_layer_cross_k.GetTensorTypeAndShapeInfo().GetShape();
      if (CudaGraphAvailableFor(static_cast<int32_t>(self_shape[1]),
                                static_cast<int32_t>(self_shape[2]),
                                static_cast<int32_t>(cross_shape[2]))) {
        return ForwardDecoderCudaGraph(
            std::move(tokens), std::move(n_layer_self_k_cache),
            std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
            std::move(n_layer_cross_v), std::move(offset),
            std::move(cross_mask));
      }
      // Context budget exhausted: fall through to the regular path.
    }
#endif
    std::vector<Ort::Value> decoder_input;
    decoder_input.reserve(7);
    decoder_input.push_back(std::move(tokens));
    decoder_input.push_back(std::move(n_layer_self_k_cache));
    decoder_input.push_back(std::move(n_layer_self_v_cache));
    decoder_input.push_back(std::move(n_layer_cross_k));
    decoder_input.push_back(std::move(n_layer_cross_v));
    decoder_input.push_back(std::move(offset));

    if (has_cross_mask_input_) {
      if (!cross_mask) {
        // The decoder requires a cross_mask but the encoder did not provide
        // one (mismatched model pair). Fall back to an all-ones mask so
        // that behavior matches the pre-cross_mask models.
        auto shape = decoder_input[3].GetTensorTypeAndShapeInfo().GetShape();
        // cross_k is (num_decoder_layers, N, T, d_model)
        std::array<int64_t, 2> mask_shape{shape[1], shape[2]};
        cross_mask = Ort::Value::CreateTensor<float>(Allocator(),
                                                     mask_shape.data(),
                                                     mask_shape.size());
        float *p = cross_mask.GetTensorMutableData<float>();
        std::fill(p, p + shape[1] * shape[2], 1.0f);
      }
      decoder_input.push_back(std::move(cross_mask));
    }

    std::vector<Ort::Value> decoder_out;

    if (use_cuda_iobinding_) {
      // CPU-side sampling needs logits on CPU, while self KV cache should
      // remain on GPU to avoid large device<->host copies between decode steps.
      Ort::IoBinding binding(*decoder_sess_);
      for (size_t i = 0; i != decoder_input.size(); ++i) {
        binding.BindInput(decoder_input_names_ptr_[i], decoder_input[i]);
      }

      binding.BindOutput(decoder_output_names_ptr_[0], cpu_mem_info_);
      binding.BindOutput(decoder_output_names_ptr_[1], *cuda_mem_info_);
      binding.BindOutput(decoder_output_names_ptr_[2], *cuda_mem_info_);

      binding.SynchronizeInputs();
      decoder_sess_->Run(GetDecoderRegularRunOptions(), binding);
      binding.SynchronizeOutputs();
      decoder_out = binding.GetOutputValues();
    } else {
      decoder_out = decoder_sess_->Run(
          GetDecoderRegularRunOptions(), decoder_input_names_ptr_.data(),
          decoder_input.data(), decoder_input.size(),
          decoder_output_names_ptr_.data(),
          decoder_output_names_ptr_.size());
    }

    Ort::Value cross_mask_out = Ort::Value{};
    if (has_cross_mask_input_) {
      cross_mask_out = std::move(decoder_input[6]);
    }

    return std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value,
                      Ort::Value, Ort::Value, Ort::Value>{
        std::move(decoder_out[0]),   std::move(decoder_out[1]),
        std::move(decoder_out[2]),   std::move(decoder_input[3]),
        std::move(decoder_input[4]), std::move(decoder_input[5]),
        std::move(cross_mask_out)};
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache(int32_t batch_size,
                                                          int32_t alloc_len) {
    if (alloc_len <= 0 || alloc_len > meta_data_.max_len) {
      alloc_len = meta_data_.max_len;
    }
#if SHERPA_ONNX_ENABLE_GPU
    if (use_cuda_graph_) {
      // Round up to the cache bucket so that segments share captured CUDA
      // graphs. Positions past the write offset are zero-filled and masked,
      // so the extra capacity does not change results.
      alloc_len = static_cast<int32_t>(
          RoundUp(alloc_len, cuda_graph_cache_bucket_));
      if (alloc_len > meta_data_.max_len) {
        alloc_len = meta_data_.max_len;
      }
    }
#endif

    std::array<int64_t, 5> shape{meta_data_.num_decoder_layers, batch_size,
                                 alloc_len, meta_data_.num_head,
                                 meta_data_.head_dim};

    Ort::Value n_layer_self_k_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    Ort::Value n_layer_self_v_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];

    float *p_k = n_layer_self_k_cache.GetTensorMutableData<float>();
    float *p_v = n_layer_self_v_cache.GetTensorMutableData<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);

    return {std::move(n_layer_self_k_cache), std::move(n_layer_self_v_cache)};
  }

  OrtAllocator *Allocator() { return allocator_; }

  const OfflineFireRedAsrModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  // Return true if the decoder model supports batch decoding, i.e., the
  // batch dimension of its tokens input is dynamic. Note that the released
  // FireRedASR decoder models hard-code a batch size of 1 there.
  bool SupportBatch() const { return support_batch_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the encoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_decoder_layers,
                               "num_decoder_layers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_head, "num_head");
    SHERPA_ONNX_READ_META_DATA(meta_data_.head_dim, "head_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sos_id, "sos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.eos_id, "eos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.max_len, "max_len");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.mean, "cmvn_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev,
                                         "cmvn_inv_stddev");
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, DecoderSessOpts());
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    for (size_t i = 0; i != decoder_input_names_.size(); ++i) {
      if (decoder_input_names_[i] == "tokens") {
        auto shape = decoder_sess_->GetInputTypeInfo(i)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();
        // a dynamic batch dimension is -1
        support_batch_ = !shape.empty() && shape[0] == -1;
      }
      if (decoder_input_names_[i] == "cross_mask") {
        has_cross_mask_input_ = true;
      }
    }
  }

  void InitCudaIOBinding() {
    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      // Use device 0 by default. SessionOptions() in sherpa-onnx usually
      // configures the CUDA EP device; binding here only affects output memory.
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
  }

  const Ort::SessionOptions &DecoderSessOpts() const {
    return decoder_graph_sess_opts_ ? *decoder_graph_sess_opts_ : sess_opts_;
  }

  void InitCudaGraphOptions() {
#if SHERPA_ONNX_ENABLE_GPU
    // CUDA graph for the decoder is opt-in: it requires the CUDA provider and
    // SHERPA_ONNX_CUDA_GRAPH=1. It is disabled by default; CPU and other
    // providers are completely unaffected.
    if (!IsCudaProvider(config_.provider) || !CudaGraphEnvEnabled()) {
      return;
    }

    auto available_providers = Ort::GetAvailableProviders();
    if (std::find(available_providers.begin(), available_providers.end(),
                  "CUDAExecutionProvider") == available_providers.end()) {
      SHERPA_ONNX_LOGE(
          "SHERPA_ONNX_CUDA_GRAPH=1 but the CUDA execution provider is not "
          "available; CUDA graph is disabled");
      return;
    }

    // The decoder session needs its own session options since
    // enable_cuda_graph must be set when the CUDA EP is appended, and the
    // encoder session (sharing sess_opts_) must not use CUDA graph.
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(config_.num_threads);
    opts.SetInterOpNumThreads(config_.num_threads);
    opts.AddConfigEntry("session.intra_op.spin_backoff_max", "8");
    opts.AddConfigEntry("session.inter_op.spin_backoff_max", "8");

    Ort::CUDAProviderOptions cuda_opts;
    std::unordered_map<std::string, std::string> cuda_option_map{
        {"device_id", "0"},
        {"cudnn_conv_algo_search", "HEURISTIC"},
        {"enable_cuda_graph", "1"},
    };
    cuda_opts.Update(cuda_option_map);
    opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);

    decoder_graph_sess_opts_ =
        std::make_unique<Ort::SessionOptions>(std::move(opts));
    cuda_arena_mem_info_ = std::make_unique<Ort::MemoryInfo>(
        "Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    // Bucket the per-segment shape dimensions so that segments share captured
    // graphs: every new shape signature costs one graph capture, while padded
    // cache positions / padded cross frames are exactly masked out (self
    // attention only reads positions <= offset; padded cross frames get
    // cross_mask 0), so results are unchanged.
    cuda_graph_cache_bucket_ =
        CudaGraphEnvBucket("SHERPA_ONNX_CUDA_GRAPH_CACHE_BUCKET", 32);
    cuda_graph_cross_bucket_ =
        CudaGraphEnvBucket("SHERPA_ONNX_CUDA_GRAPH_CROSS_BUCKET", 64);
    cuda_graph_max_contexts_ =
        CudaGraphEnvBucket("SHERPA_ONNX_CUDA_GRAPH_MAX_CONTEXTS", 8);
    use_cuda_graph_ = true;
    SHERPA_ONNX_LOGE(
        "FireRedASR decoder CUDA graph is enabled (SHERPA_ONNX_CUDA_GRAPH=1, "
        "cache_bucket=%d, cross_bucket=%d, max_contexts=%d)",
        cuda_graph_cache_bucket_, cuda_graph_cross_bucket_,
        cuda_graph_max_contexts_);
#endif
  }

#if SHERPA_ONNX_ENABLE_GPU
  // ORT replays a captured CUDA graph purely by its annotation id
  // (the "gpu_graph_id" run option); it does not validate input shapes or
  // buffer addresses at replay time. Therefore each distinct input shape
  // signature gets its own context holding fixed device buffers plus an
  // IoBinding, and its own graph id. Buffers of a context are never freed,
  // so a replay always reads/writes the addresses captured for its graph.
  struct CudaGraphDecoderContext {
    int32_t n = 0;
    int32_t cache_len = 0;
    int32_t cross_t = 0;
    int32_t graph_id = 0;
    size_t self_cache_bytes = 0;

    Ort::Value in_tokens{nullptr};
    Ort::Value in_offset{nullptr};
    Ort::Value in_self_k{nullptr};
    Ort::Value in_self_v{nullptr};
    Ort::Value in_cross_k{nullptr};
    Ort::Value in_cross_v{nullptr};
    Ort::Value in_cross_mask{nullptr};
    Ort::Value out_self_k{nullptr};
    Ort::Value out_self_v{nullptr};

    // CPU-side copy of the logits. ORT's replay path skips
    // CopyOutputsAcrossDevices, so a CPU-bound output would go stale;
    // instead logits stay bound to a device buffer held by the binding and
    // are copied here explicitly after every run.
    Ort::Value logits_cpu{nullptr};
    size_t logits_bytes = 0;

    std::unique_ptr<Ort::IoBinding> binding;
  };

  CudaGraphDecoderContext *FindCudaGraphContext(int32_t n, int32_t cache_len,
                                                int32_t cross_t_raw) const {
    int32_t cross_t = static_cast<int32_t>(
        RoundUp(cross_t_raw, cuda_graph_cross_bucket_));
    for (auto &c : cuda_graph_contexts_) {
      if (c->n == n && c->cache_len == cache_len && c->cross_t == cross_t) {
        return c.get();
      }
    }
    return nullptr;
  }

  // Whether the graph path can serve this shape signature: true if a context
  // already exists, or if the context budget is not exhausted yet. Every
  // context permanently pins its fixed buffers plus one captured graph in the
  // CUDA EP, so without a cap long audio with many distinct bucket shapes
  // would exhaust GPU memory. Shapes beyond the budget fall back to the
  // regular IOBinding path with identical numerics.
  bool CudaGraphAvailableFor(int32_t n, int32_t cache_len,
                             int32_t cross_t_raw) const {
    if (FindCudaGraphContext(n, cache_len, cross_t_raw) != nullptr) {
      return true;
    }
    return static_cast<int32_t>(cuda_graph_contexts_.size()) <
           cuda_graph_max_contexts_;
  }

  CudaGraphDecoderContext *FindOrCreateCudaGraphContext(
      int32_t n, int32_t cache_len, int32_t cross_t_raw,
      const std::vector<int64_t> &self_shape,
      const std::vector<int64_t> &cross_shape) {
    int32_t cross_t = static_cast<int32_t>(
        RoundUp(cross_t_raw, cuda_graph_cross_bucket_));
    if (auto *found = FindCudaGraphContext(n, cache_len, cross_t_raw)) {
      return found;
    }

    auto ctx = std::make_unique<CudaGraphDecoderContext>();
    ctx->n = n;
    ctx->cache_len = cache_len;
    ctx->cross_t = cross_t;
    ctx->graph_id = next_cuda_graph_id_++;

    Ort::Allocator dev_alloc(*decoder_sess_, *cuda_arena_mem_info_);

    std::array<int64_t, 2> token_shape{n, 1};
    ctx->in_tokens =
        Ort::Value::CreateTensor<int64_t>(dev_alloc, token_shape.data(), 2);

    std::array<int64_t, 1> offset_shape{n};
    ctx->in_offset =
        Ort::Value::CreateTensor<int64_t>(dev_alloc, offset_shape.data(), 1);

    ctx->in_self_k = Ort::Value::CreateTensor<float>(
        dev_alloc, self_shape.data(), self_shape.size());
    ctx->in_self_v = Ort::Value::CreateTensor<float>(
        dev_alloc, self_shape.data(), self_shape.size());
    ctx->out_self_k = Ort::Value::CreateTensor<float>(
        dev_alloc, self_shape.data(), self_shape.size());
    ctx->out_self_v = Ort::Value::CreateTensor<float>(
        dev_alloc, self_shape.data(), self_shape.size());

    std::vector<int64_t> bucketed_cross_shape = cross_shape;
    bucketed_cross_shape[2] = cross_t;
    ctx->in_cross_k = Ort::Value::CreateTensor<float>(
        dev_alloc, bucketed_cross_shape.data(), bucketed_cross_shape.size());
    ctx->in_cross_v = Ort::Value::CreateTensor<float>(
        dev_alloc, bucketed_cross_shape.data(), bucketed_cross_shape.size());

    if (has_cross_mask_input_) {
      std::array<int64_t, 2> mask_shape{n, cross_t};
      ctx->in_cross_mask =
          Ort::Value::CreateTensor<float>(dev_alloc, mask_shape.data(), 2);
    }

    size_t numel = 1;
    for (auto d : self_shape) {
      numel *= static_cast<size_t>(d);
    }
    ctx->self_cache_bytes = numel * sizeof(float);

    ctx->binding = std::make_unique<Ort::IoBinding>(*decoder_sess_);
    ctx->binding->BindInput(decoder_input_names_ptr_[0], ctx->in_tokens);
    ctx->binding->BindInput(decoder_input_names_ptr_[1], ctx->in_self_k);
    ctx->binding->BindInput(decoder_input_names_ptr_[2], ctx->in_self_v);
    ctx->binding->BindInput(decoder_input_names_ptr_[3], ctx->in_cross_k);
    ctx->binding->BindInput(decoder_input_names_ptr_[4], ctx->in_cross_v);
    ctx->binding->BindInput(decoder_input_names_ptr_[5], ctx->in_offset);
    if (has_cross_mask_input_) {
      ctx->binding->BindInput(decoder_input_names_ptr_[6], ctx->in_cross_mask);
    }

    // All outputs stay on the device in buffers held by the binding: ORT's
    // CUDA graph replay path skips CopyOutputsAcrossDevices, so CPU-bound
    // outputs would never be refreshed on replay. The updated self kv cache
    // uses fixed pre-allocated buffers so it can be copied back into the
    // fixed input buffers after each step; logits are copied to logits_cpu
    // explicitly after every run.
    ctx->binding->BindOutput(decoder_output_names_ptr_[0], *cuda_mem_info_);
    ctx->binding->BindOutput(decoder_output_names_ptr_[1], ctx->out_self_k);
    ctx->binding->BindOutput(decoder_output_names_ptr_[2], ctx->out_self_v);

    SHERPA_ONNX_LOGE(
        "Created CUDA graph decoder context: graph_id=%d N=%d cache_len=%d "
        "cross_t=%d (total contexts: %d)",
        ctx->graph_id, n, cache_len, cross_t,
        static_cast<int32_t>(cuda_graph_contexts_.size() + 1));

    cuda_graph_contexts_.push_back(std::move(ctx));
    return cuda_graph_contexts_.back().get();
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value, Ort::Value>
  ForwardDecoderCudaGraph(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
                          Ort::Value n_layer_self_v_cache,
                          Ort::Value n_layer_cross_k,
                          Ort::Value n_layer_cross_v, Ort::Value offset,
                          Ort::Value cross_mask) {
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);

    auto self_shape =
        n_layer_self_k_cache.GetTensorTypeAndShapeInfo().GetShape();
    auto cross_shape =
        n_layer_cross_k.GetTensorTypeAndShapeInfo().GetShape();
    int32_t n = static_cast<int32_t>(self_shape[1]);
    int32_t cache_len = static_cast<int32_t>(self_shape[2]);
    int32_t cross_t_raw = static_cast<int32_t>(cross_shape[2]);

    if (has_cross_mask_input_ && !cross_mask) {
      // Same fallback as the non-graph path: the decoder requires a
      // cross_mask but the encoder did not provide one.
      std::array<int64_t, 2> mask_shape{n, cross_t_raw};
      cross_mask = Ort::Value::CreateTensor<float>(
          Allocator(), mask_shape.data(), mask_shape.size());
      float *p = cross_mask.GetTensorMutableData<float>();
      std::fill(p, p + static_cast<size_t>(n) * cross_t_raw, 1.0f);
    }

    CudaGraphDecoderContext *ctx = FindOrCreateCudaGraphContext(
        n, cache_len, cross_t_raw, self_shape, cross_shape);

    // tokens and offset change value (but not shape) every step
    CudaCopy(ctx->in_tokens.GetTensorMutableData<int64_t>(),
             tokens.GetTensorData<int64_t>(), sizeof(int64_t) * n);
    CudaCopy(ctx->in_offset.GetTensorMutableData<int64_t>(),
             offset.GetTensorData<int64_t>(), sizeof(int64_t) * n);

    if (n_layer_self_k_cache.GetTensorData<float>() !=
        ctx->in_self_k.GetTensorData<float>()) {
      // First step of a new segment: stage the self kv cache and the cross
      // tensors into the fixed device buffers. Later steps of the segment
      // receive back views of in_self_k/in_self_v (returned below), so this
      // branch runs exactly once per segment.
      CudaCopy(ctx->in_self_k.GetTensorMutableData<float>(),
               n_layer_self_k_cache.GetTensorData<float>(),
               ctx->self_cache_bytes);
      CudaCopy(ctx->in_self_v.GetTensorMutableData<float>(),
               n_layer_self_v_cache.GetTensorData<float>(),
               ctx->self_cache_bytes);

      // Stage cross tensors into the (possibly larger) bucketed buffers.
      // The padded frames are zero-filled and get cross_mask 0, so cross
      // attention assigns them exactly zero weight.
      int64_t d_model = cross_shape[3];
      size_t row_bytes = sizeof(float) * cross_t_raw * d_model;
      size_t dst_pitch = sizeof(float) * ctx->cross_t * d_model;
      size_t slab_count = static_cast<size_t>(cross_shape[0]) * cross_shape[1];
      size_t cross_bytes_total = dst_pitch * slab_count;

      CudaMemsetZero(ctx->in_cross_k.GetTensorMutableData<float>(),
                     cross_bytes_total);
      CudaCopy2D(ctx->in_cross_k.GetTensorMutableData<float>(), dst_pitch,
                 n_layer_cross_k.GetTensorData<float>(), row_bytes, row_bytes,
                 slab_count);
      CudaMemsetZero(ctx->in_cross_v.GetTensorMutableData<float>(),
                     cross_bytes_total);
      CudaCopy2D(ctx->in_cross_v.GetTensorMutableData<float>(), dst_pitch,
                 n_layer_cross_v.GetTensorData<float>(), row_bytes, row_bytes,
                 slab_count);
      if (has_cross_mask_input_) {
        CudaMemsetZero(ctx->in_cross_mask.GetTensorMutableData<float>(),
                       sizeof(float) * n * ctx->cross_t);
        CudaCopy2D(ctx->in_cross_mask.GetTensorMutableData<float>(),
                   sizeof(float) * ctx->cross_t,
                   cross_mask.GetTensorData<float>(),
                   sizeof(float) * cross_t_raw, sizeof(float) * cross_t_raw,
                   n);
      }
    }

    Ort::RunOptions run_options;
    std::string graph_id_str = std::to_string(ctx->graph_id);
    run_options.AddConfigEntry("gpu_graph_id", graph_id_str.c_str());

    ctx->binding->SynchronizeInputs();
    decoder_sess_->Run(run_options, *ctx->binding);
    ctx->binding->SynchronizeOutputs();

    // Feed the updated self kv cache back into the fixed input buffers, so
    // the next graph replay reads the new state. All copies here are
    // synchronous cudaMemcpy calls, so they are ordered both with the
    // completed run above and with the next run below.
    CudaCopy(ctx->in_self_k.GetTensorMutableData<float>(),
             ctx->out_self_k.GetTensorData<float>(), ctx->self_cache_bytes);
    CudaCopy(ctx->in_self_v.GetTensorMutableData<float>(),
             ctx->out_self_v.GetTensorData<float>(), ctx->self_cache_bytes);

    std::vector<Ort::Value> decoder_out = ctx->binding->GetOutputValues();

    if (!ctx->logits_cpu) {
      auto logits_shape =
          decoder_out[0].GetTensorTypeAndShapeInfo().GetShape();
      ctx->logits_cpu = Ort::Value::CreateTensor<float>(
          Allocator(), logits_shape.data(), logits_shape.size());
      size_t numel = 1;
      for (auto d : logits_shape) {
        numel *= static_cast<size_t>(d);
      }
      ctx->logits_bytes = numel * sizeof(float);
    }
    CudaCopy(ctx->logits_cpu.GetTensorMutableData<float>(),
             decoder_out[0].GetTensorData<float>(), ctx->logits_bytes);

    Ort::Value cross_mask_out = Ort::Value{};
    if (has_cross_mask_input_) {
      cross_mask_out = std::move(cross_mask);
    }

    return std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value,
                      Ort::Value, Ort::Value, Ort::Value>{
        View(&ctx->logits_cpu),      View(&ctx->in_self_k),
        View(&ctx->in_self_v),       View(&ctx->in_cross_k),
        View(&ctx->in_cross_v),      std::move(offset),
        std::move(cross_mask_out)};
  }
#endif  // SHERPA_ONNX_ENABLE_GPU

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  Ort::MemoryInfo cpu_mem_info_;
  std::unique_ptr<Ort::MemoryInfo> cuda_mem_info_;
  bool use_cuda_iobinding_ = false;
  bool is_cpu_provider_ = false;

  // Decoder-only CUDA graph state. Populated only when built with
  // SHERPA_ONNX_ENABLE_GPU, provider is cuda, and SHERPA_ONNX_CUDA_GRAPH=1.
  bool use_cuda_graph_ = false;
  std::unique_ptr<Ort::SessionOptions> decoder_graph_sess_opts_;
#if SHERPA_ONNX_ENABLE_GPU
  std::unique_ptr<Ort::MemoryInfo> cuda_arena_mem_info_;
  int32_t cuda_graph_cache_bucket_ = 32;
  int32_t cuda_graph_cross_bucket_ = 64;
  int32_t cuda_graph_max_contexts_ = 8;
  std::vector<std::unique_ptr<CudaGraphDecoderContext>> cuda_graph_contexts_;
  int32_t next_cuda_graph_id_ = 1;
  std::mutex cuda_graph_mutex_;
#endif

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  OfflineFireRedAsrModelMetaData meta_data_;

  bool support_batch_ = false;

  // true if the decoder model has a "cross_mask" input, i.e., it was
  // exported with cross-attention padding mask support
  bool has_cross_mask_input_ = false;

  // Run counter for arena shrinkage: every kArenaShrinkageInterval runs,
  // trigger one arena shrinkage to keep memory stable without destroying
  // the session (which would require reloading 1.2GB weights).
  int32_t run_count_ = 0;
  static constexpr int32_t kArenaShrinkageInterval = 10;

  // Get RunOptions with arena shrinkage enabled every kArenaShrinkageInterval runs.
  Ort::RunOptions GetRunOptionsWithArenaShrinkage() {
    Ort::RunOptions run_options;
    run_count_++;
    if (run_count_ % kArenaShrinkageInterval == 0) {
      // Shrink arena to free unused memory blocks back to the OS.
      // This is cheaper than destroying and recreating the session.
      run_options.AddConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0");
    }
    return run_options;
  }

  // RunOptions for the decoder's regular (non-CUDA-graph) path. When the
  // decoder session was created with enable_cuda_graph=1, a Run without a
  // graph annotation defaults to annotation id 0: ORT counts such runs and
  // eventually captures a graph for id 0 bound to this call's temporary
  // buffers, then replays it on later runs. gpu_graph_id=-1
  // (kCudaGraphAnnotationSkip) opts the run out of both capture and replay.
  Ort::RunOptions GetDecoderRegularRunOptions() {
    Ort::RunOptions run_options = GetRunOptionsWithArenaShrinkage();
    if (use_cuda_graph_) {
      run_options.AddConfigEntry("gpu_graph_id", "-1");
    }
    return run_options;
  }
};

OfflineFireRedAsrModel::OfflineFireRedAsrModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFireRedAsrModel::OfflineFireRedAsrModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFireRedAsrModel::~OfflineFireRedAsrModel() = default;

std::tuple<Ort::Value, Ort::Value, Ort::Value>
OfflineFireRedAsrModel::ForwardEncoder(Ort::Value features,
                                       Ort::Value features_length) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
           Ort::Value, Ort::Value>
OfflineFireRedAsrModel::ForwardDecoder(Ort::Value tokens,
                                       Ort::Value n_layer_self_k_cache,
                                       Ort::Value n_layer_self_v_cache,
                                       Ort::Value n_layer_cross_k,
                                       Ort::Value n_layer_cross_v,
                                       Ort::Value offset,
                                       Ort::Value cross_mask) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset), std::move(cross_mask));
}

std::pair<Ort::Value, Ort::Value>
OfflineFireRedAsrModel::GetInitialSelfKVCache(int32_t batch_size,
                                              int32_t alloc_len) const {
  return impl_->GetInitialSelfKVCache(batch_size, alloc_len);
}

OrtAllocator *OfflineFireRedAsrModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineFireRedAsrModelMetaData &OfflineFireRedAsrModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

bool OfflineFireRedAsrModel::SupportBatch() const {
  return impl_->SupportBatch();
}

#if __ANDROID_API__ >= 9
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
