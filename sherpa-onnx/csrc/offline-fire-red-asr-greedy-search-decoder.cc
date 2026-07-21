// sherpa-onnx/csrc/offline-fire-red-asr-greedy-search-decoder.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-greedy-search-decoder.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

std::vector<OfflineFireRedAsrDecoderResult>
OfflineFireRedAsrGreedySearchDecoder::Decode(Ort::Value cross_k,
                                             Ort::Value cross_v,
                                             Ort::Value enc_mask,
                                             int32_t num_feature_frames) {
  const auto &meta_data = model_->GetModelMetadata();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // cross_k is of shape (num_decoder_layers, N, T, d_model)
  auto cross_k_shape = cross_k.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(cross_k_shape[1]);

  std::array<int64_t, 2> token_shape = {batch_size, 1};
  std::vector<int64_t> cur_token(batch_size, meta_data.sos_id);

  Ort::Value tokens = Ort::Value::CreateTensor(
      memory_info, cur_token.data(), cur_token.size(), token_shape.data(),
      token_shape.size());

  std::array<int64_t, 1> offset_shape{batch_size};
  Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
      model_->Allocator(), offset_shape.data(), offset_shape.size());
  int64_t *p_offset = offset.GetTensorMutableData<int64_t>();
  std::fill(p_offset, p_offset + batch_size, 0);

  std::vector<OfflineFireRedAsrDecoderResult> ans(batch_size);

  // finished[b] is non-zero if the b-th utterance has predicted eos_id
  std::vector<int32_t> finished(batch_size, 0);
  int32_t num_finished = 0;

  // assume at most 8 tokens per second (raised from 6 for fast-speech
  // headroom; costs ~31% larger cache and a few percent decoder time)
  int32_t num_possible_tokens = num_feature_frames / 100.0 * 8;
  num_possible_tokens =
      std::min<int32_t>(num_possible_tokens, meta_data.max_len / 2);

  // The self k/v cache is written only at positions [0, num_possible_tokens),
  // so there is no need to allocate max_len frames for it. A smaller cache
  // reduces both memory usage and decoding time. The margin 4 is more than
  // enough since the write position is strictly less than num_possible_tokens.
  int32_t cache_len = std::min(meta_data.max_len, num_possible_tokens + 4);

  auto self_kv_cache = model_->GetInitialSelfKVCache(batch_size, cache_len);

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value, Ort::Value>
      decoder_out = {Ort::Value{nullptr},
                     std::move(self_kv_cache.first),
                     std::move(self_kv_cache.second),
                     std::move(cross_k),
                     std::move(cross_v),
                     std::move(offset),
                     std::move(enc_mask)};

  for (int32_t i = 0;
       i < num_possible_tokens && num_finished != batch_size; ++i) {
    decoder_out = model_->ForwardDecoder(View(&tokens),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)),
                                         std::move(std::get<6>(decoder_out)));

    const auto &logits = std::get<0>(decoder_out);
    const float *p_logits = logits.GetTensorData<float>();

    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int32_t vocab_size = logits_shape[2];

    for (int32_t b = 0; b != batch_size; ++b) {
      if (finished[b]) {
        // keep feeding eos_id to a finished utterance; its output is ignored
        cur_token[b] = meta_data.eos_id;
        continue;
      }

      const float *p = p_logits + b * vocab_size;
      int32_t max_token_id = static_cast<int32_t>(
          std::distance(p, std::max_element(p, p + vocab_size)));

      cur_token[b] = max_token_id;
      if (max_token_id == meta_data.eos_id) {
        finished[b] = 1;
        ++num_finished;
      } else {
        ans[b].tokens.push_back(max_token_id);
      }
    }

    // increment offset
    int64_t *p_cur_offset =
        std::get<5>(decoder_out).GetTensorMutableData<int64_t>();
    for (int32_t b = 0; b != batch_size; ++b) {
      p_cur_offset[b] += 1;
    }
  }

  return ans;
}

}  // namespace sherpa_onnx
