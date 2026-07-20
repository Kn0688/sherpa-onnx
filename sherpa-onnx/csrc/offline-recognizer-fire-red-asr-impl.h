// sherpa-onnx/csrc/offline-recognizer-fire-red-asr-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/offline-fire-red-asr-decoder.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-fire-red-asr-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-fire-red-asr-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

// When batching streams of different lengths, start a new bucket whenever
// max_num_frames / min_num_frames within the current bucket would exceed
// this ratio. Larger values allow more padding waste inside a batch;
// smaller values create more, smaller batches.
static constexpr float kMaxBucketLengthRatio = 1.2f;

static OfflineRecognitionResult Convert(
    const OfflineFireRedAsrDecoderResult &src, const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    if (!sym_table.Contains(i)) {
      continue;
    }

    const auto &s = sym_table[i];
    text += s;
    r.tokens.push_back(s);
  }

  r.text = std::move(text);

  return r;
}

class OfflineRecognizerFireRedAsrImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerFireRedAsrImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineFireRedAsrModel>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerFireRedAsrImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineFireRedAsrModel>(mgr,
                                                        config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineFireRedAsrGreedySearchDecoder>(model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for FireRedAsr. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &meta_data = model_->GetModelMetadata();

    config_.feat_config.normalize_samples = false;
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    if (n > 1 && !model_->SupportBatch()) {
      // The released FireRedASR decoder models hard-code a batch size of 1
      // for the tokens input, so we fall back to decoding one stream at a
      // time for such models.
      for (int32_t i = 0; i != n; ++i) {
        DecodeStreams(ss + i, 1);
      }

      return;
    }

    if (n <= 1) {
      DecodeStreamsBatch(ss, n);
      return;
    }

    // Sort the streams by their number of feature frames and cut them into
    // buckets of similar lengths, so that a short utterance is never padded
    // to a much longer one. Padding waste is quadratic in the encoder and
    // linear in the decoder cross-attention, so it dominates the batch cost
    // for mixed lengths.
    int32_t feat_dim = ss[0]->FeatureDim();
    std::vector<int64_t> num_frames(n);
    for (int32_t i = 0; i != n; ++i) {
      num_frames[i] = ss[i]->GetFrames().size() / feat_dim;
    }

    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return num_frames[a] < num_frames[b];
    });

    std::vector<OfflineStream *> bucket;
    bucket.reserve(n);
    int64_t bucket_min_frames = 0;
    int32_t num_buckets = 0;
    for (int32_t i = 0; i != n; ++i) {
      if (!bucket.empty() &&
          num_frames[order[i]] >
              kMaxBucketLengthRatio * bucket_min_frames) {
        if (config_.model_config.debug) {
          SHERPA_ONNX_LOGE("FireRedASR bucket %d: %d streams, frames %d..%d",
                           num_buckets, static_cast<int32_t>(bucket.size()),
                           static_cast<int32_t>(bucket_min_frames),
                           static_cast<int32_t>(num_frames[order[i - 1]]));
        }
        DecodeStreamsBatch(bucket.data(), bucket.size());
        ++num_buckets;
        bucket.clear();
      }
      if (bucket.empty()) {
        bucket_min_frames = num_frames[order[i]];
      }
      bucket.push_back(ss[order[i]]);
    }
    if (!bucket.empty()) {
      if (config_.model_config.debug) {
        SHERPA_ONNX_LOGE("FireRedASR bucket %d: %d streams, frames %d..%d",
                         num_buckets, static_cast<int32_t>(bucket.size()),
                         static_cast<int32_t>(bucket_min_frames),
                         static_cast<int32_t>(num_frames[order[n - 1]]));
      }
      DecodeStreamsBatch(bucket.data(), bucket.size());
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStreamsBatch(OfflineStream **ss, int32_t n) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<std::vector<float>> features_vec(n);
    std::vector<Ort::Value> features;
    features.reserve(n);

    std::vector<int64_t> features_length_vec(n);

    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> f = ss[i]->GetFrames();
      ApplyCMVN(&f);

      int64_t num_frames = f.size() / feat_dim;
      features_vec[i] = std::move(f);
      features_length_vec[i] = num_frames;

      std::array<int64_t, 2> shape{num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer, 0);

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), features_length_vec.size(),
        features_length_shape.data(), features_length_shape.size());

    auto cross_kv = model_->ForwardEncoder(std::move(x), std::move(x_length));

    int32_t max_num_frames = static_cast<int32_t>(*std::max_element(
        features_length_vec.begin(), features_length_vec.end()));

    auto results = decoder_->Decode(
        std::move(std::get<0>(cross_kv)), std::move(std::get<1>(cross_kv)),
        std::move(std::get<2>(cross_kv)), max_num_frames);

    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_);

      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

 private:
  void ApplyCMVN(std::vector<float> *v) const {
    const auto &meta_data = model_->GetModelMetadata();
    const auto &mean_vec = meta_data.mean;
    const auto &inv_stddev_vec = meta_data.inv_stddev;
    int32_t feat_dim = static_cast<int32_t>(mean_vec.size());
    int32_t num_frames = static_cast<int32_t>(v->size()) / feat_dim;
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat(v->data(), num_frames, feat_dim);
    Eigen::Map<const Eigen::RowVectorXf> mean(mean_vec.data(), feat_dim);
    Eigen::Map<const Eigen::RowVectorXf> inv_std(inv_stddev_vec.data(),
                                                 feat_dim);

    mat.array() =
        (mat.array().rowwise() - mean.array()).rowwise() * inv_std.array();
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineFireRedAsrModel> model_;
  std::unique_ptr<OfflineFireRedAsrDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_
