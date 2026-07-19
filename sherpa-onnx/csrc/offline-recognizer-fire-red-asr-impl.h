// sherpa-onnx/csrc/offline-recognizer-fire-red-asr-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FIRE_RED_ASR_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
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

    auto results =
        decoder_->Decode(std::move(cross_kv.first), std::move(cross_kv.second),
                         max_num_frames);

    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_);

      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

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
