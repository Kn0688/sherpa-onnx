# sherpa-onnx 语音识别（ASR）核心链路代码导读

> 面向想深入理解 sherpa-onnx C++ 核心的开发者。
> 所有路径相对 `sherpa-onnx/csrc/`，基于版本 1.13.4。

## 0. 全局认知：一张图看懂两条链路

sherpa-onnx 的 ASR 分两条完全独立的链路，共享底层的特征提取、ONNX Session 管理和符号表：

```
非流式（offline/*）：整段音频一次性识别
  音频文件 → OfflineStream（一次性算出全部 fbank 特征）
           → 补齐/padding → 一次 batched ONNX 推理
           → 解码（CTC greedy / transducer / FST）→ 文本 + 时间戳

流式（online/*）：边录边识别
  音频块（如 20ms）→ OnlineStream（增量产生 fbank 帧，帧缓冲可弹出）
                  → IsReady() 攒够一个 chunk？
                  → RunEncoder（携带上一 chunk 的缓存状态 states）
                  → 逐帧 decoder+joiner（greedy / modified beam search）
                  → IsEndpoint()？→ 出最终结果 → Reset() 继续下一句
```

最重要的设计模式：**Impl / 工厂模式**。公开类（`OfflineRecognizer` / `OnlineRecognizer`）只是外壳，内部持有抽象 `*Impl` 指针，其静态 `Create()` 根据配置里哪个模型路径非空，实例化对应模型家族的实现。新增一种模型 = 加一个 `*-impl.h` + `*-model.cc` + 在 `Create()` 里加一个分支。

---

## 1. 非流式 ASR 链路（OfflineRecognizer）

以 **SenseVoice**（单 ONNX 文件 + CTC greedy 解码，实现最干净）为深入样例。

### 1.1 入口与配置

典型调用流程见 CLI 示例 `sherpa-onnx-offline.cc:120-182`：

```cpp
OfflineRecognizerConfig config;
config.Register(&po);                    // 解析命令行
config.Validate();                       // 校验
OfflineRecognizer recognizer(config);    // 构造 → 工厂分发
auto stream = recognizer.CreateStream();
stream->AcceptWaveform(sr, samples, n);  // 喂整段音频（只能调一次）
recognizer.DecodeStream(stream.get());   // 解码
stream->GetResult().text;                // 取结果
```

核心配置结构（聚合在 `OfflineRecognizerConfig`，`offline-recognizer.h:25-75`）：

| 配置 | 定义位置 | 关键字段 |
|---|---|---|
| `FeatureExtractorConfig` | `features.h:16-91` | `sampling_rate=16000`、`feature_dim=80`、`normalize_samples`（false 时采样点 ×32768） |
| `OfflineModelConfig` | `offline-model-config.h:30-68` | 每个模型家族一个子结构（`sense_voice`/`whisper`/`paraformer`/…）+ 公共的 `tokens`、`num_threads=2`、`provider="cpu"`、`model_type` |
| `decoding_method` 等 | `offline-recognizer.h` | `"greedy_search"`、`max_active_paths`、`hotwords_file`、`rule_fsts`（ITN） |

`Validate()` 层层委托：`OfflineRecognizerConfig::Validate()`（`offline-recognizer.cc:72-131`）→ `OfflineModelConfig::Validate()`（`offline-model-config.cc:67`）→ 第一个非空的家族配置自己的 `Validate()`（如 SenseVoice 校验语言 ∈ {auto, zh, en, ja, ko, yue}，`offline-sense-voice-model-config.cc:31-72`）。

### 1.2 工厂分发

`OfflineRecognizerImpl::Create()`（`offline-recognizer-impl.cc:81-439`）的分支顺序：

1. NPU provider 预分支：`rknn` / `axera` / `axcl` / `ascend` / `qnn`（各有 `SHERPA_ONNX_ENABLE_*` 宏保护）
2. `sense_voice.model` 非空 → `OfflineRecognizerSenseVoiceImpl`（:237）
3. `paraformer.model` → ParaformerImpl；`whisper.encoder` → WhisperImpl；`moonshine.*` / `canary.*` / `fire_red_asr.*` 等
4. 所有 CTC 家族（`nemo_ctc`/`zipformer_ctc`/`tdnn`/`wenet_ctc`/`dolphin`/…）→ 共用 `OfflineRecognizerCtcImpl`（:253-262）
5. 兜底：用 `model_type` 字符串；再不行就**临时打开 encoder ONNX 读元数据**再判断（:316-352）——所以不显式设置 `model_type` 时模型文件可能被解析两次

### 1.3 特征提取：在 AcceptWaveform 里"急切"完成

`OfflineStream`（`offline-stream.cc:27-319`）构造时按配置创建一个 kaldi-native-fbank 提取器（`knf::OnlineFbank`，Whisper 用 `OnlineWhisperFbank`）。

`AcceptWaveform()`（`offline-stream.cc:115-177`）做三件事：

1. **采样缩放**：若 `normalize_samples == false`（由模型元数据决定，如 SenseVoice），所有采样点先 ×32768
2. **重采样**：输入采样率 ≠ 16k 时用 `LinearResample`（`resample.h`）现场重采样
3. **立即算完全部 fbank 帧**（不是惰性计算！），并调用 `InputFinished()`

注意：offline 流的 `AcceptWaveform` 语义是**只能调一次**（`offline-stream.h:93-94`），整段音频必须一次给齐。

归一化分散在两处，容易困惑：

- NeMo 模型的 per-feature CMVN 在 `OfflineStream::GetFrames()` 里做（`offline-stream.cc:187-218`）
- SenseVoice/Paraformer 的全局 CMVN + LFR（低帧率堆叠）在 recognizer impl 里、推理前一刻做

### 1.4 模型加载（SenseVoice 为例）

`OfflineSenseVoiceModel`（`offline-sense-voice-model.cc:31-167`）：

- `Ort::Env` 由 `CreateOrtEnv()`（`ort-env.h`）创建；`Ort::SessionOptions` 由 `GetSessionOptions(config)`（`session.cc:135-430`）创建——`num_threads` 同时设置 intra/inter op 线程数，`provider` 字符串在此映射成 CUDA/CoreML/NNAPI 等 EP，不可用时**静默回退 CPU**（只打日志）
- 输入/输出名**不写死**，从 session 动态读取（:92-94）
- 元数据用 `SHERPA_ONNX_READ_META_DATA*` 宏（`macros.h:67-120`）读入 `OfflineSenseVoiceModelMetaData`（`offline-sense-voice-model-meta-data.h`）：`vocab_size`、`lfr_window_size=7`、`lfr_window_shift=6`、`normalize_samples`、CMVN 向量 `neg_mean`/`inv_stddev`、各语言 id 等。缺必需键 → 直接退出进程
- 前向签名：`Forward(features (N,T,C), features_length, language, text_norm) -> logits (N,T,C)`——注意**语言和是否做 ITN 是模型的输入张量**

impl 构造时还会加载 `SymbolTable`（`tokens.txt`，`symbol-table.h`）和 `OfflineCtcGreedySearchDecoder`，并在 `PostInit()` 里**用模型元数据覆盖用户的 fbank 配置**（`offline-recognizer-sense-voice-impl.h:375-382`）——命令行传的 fbank 参数可能被静默忽略。

### 1.5 批解码

`OfflineRecognizerSenseVoiceImpl::DecodeStreams(ss, n)`（`offline-recognizer-sense-voice-impl.h:112-247`）：

1. 每个流 `GetFrames()` 取出全部特征 → `ApplyLFR()`（7 帧堆叠、6 帧平移，特征维从 80 → 560）→ `ApplyCMVN()`（Eigen 逐行 `(x + neg_mean) * inv_stddev`）
2. `PadSequence` 右侧补齐到最长句 → `(N, max_T, 560)` 一个张量
3. 语言 id 和 ITN 开关（可按流用 `stream->SetOption("use_itn", ...)` 覆盖）映射成整数输入
4. **一次 batched `model_->Forward()`** ——这就是 API 叫 `DecodeStreams`（复数）的原因；单数 `DecodeStream` 只是 1 元素包装
5. SenseVoice 输出前 4 帧不是文本：语言/情感/事件的 embedding 查询帧，所以 `logits_length = features_length + 4`

### 1.6 CTC 解码 → 文本

- `OfflineCtcGreedySearchDecoder::Decode()`（`offline-ctc-greedy-search-decoder.cc:15-52`）：逐帧 argmax，跳过 blank 和连续重复，同时记录每个 token 的帧号（时间戳）
- `ConvertSenseVoiceResult()`（`offline-recognizer-sense-voice-impl.h:26-66`）：**跳过前 4 个 token**（解析回 `lang`/`emotion`/`event` 字段），其余 id 经 `SymbolTable` 转字符串拼接（`▁` 自动转空格，支持 byte-fallback `<0xXX>`）
- 文本级后处理（基类 `offline-recognizer-impl.cc`）：`ApplyInverseTextNormalization`（加载 `rule_fsts` 时做 ITN，:906-917）、`ApplyHomophoneReplacer`（中文同音字纠错，:919-926）

其他解码器对照：CTC 家族可选 `OfflineCtcFstDecoder`（TLG 图）；transducer 家族有 greedy / modified beam search，且**只有 transducer 支持热词**（`CreateStream(hotwords)` 构建 `ContextGraph` token-trie，`offline-recognizer-transducer-impl.h:163-203`）。

### 1.7 结果

`OfflineRecognitionResult`（`offline-stream.h:19-59`）：`text`、`tokens`、`timestamps`（秒）、`lang`/`emotion`/`event`（SenseVoice 特有）、`durations`（TDT 模型）等。结果存在 stream 里（`SetResult`），`GetResult()` 读取；recognizer 本身无状态。

---

## 2. 流式 ASR 链路（OnlineRecognizer）

以 **icefall streaming zipformer transducer** 为深入样例。

### 2.1 配置

`OnlineRecognizerConfig`（`online-recognizer.h:83-159`）聚合：feat_config、`OnlineModelConfig model_config`（`online-model-config.h:19-86`，含 `transducer.{encoder,decoder,joiner}` 三个文件路径）、`EndpointConfig endpoint_config`、以及 `enable_endpoint`、`decoding_method`、`hotwords_*` 等。

端点检测三条规则默认值（`endpoint.h:51-52`）：

- rule1 `{false, 2.4s, 0}`：静音 2.4 秒就算端点，哪怕什么都没识别出来
- rule2 `{true, 1.2s, 0}`：识别出内容后，尾随静音 1.2 秒出端点
- rule3 `{false, 0, 20s}`：单句最长 20 秒强制切断

### 2.2 工厂分发（两层）

第一层 `OnlineRecognizerImpl::Create()`（`online-recognizer-impl.cc:62-147`）：RKNN/QNN 预分支后，`transducer.encoder` 非空时会**临时打开 decoder ONNX** 判断变体——输出数为 1 的是 icefall 系 → `OnlineRecognizerTransducerImpl`，否则 NeMo 系；`paraformer` / `wenet_ctc` / `zipformer2_ctc` 等各有 impl。

第二层在 transducer impl 内部：`OnlineTransducerModel::Create()`（`online-transducer-model.cc:145-189`）读 encoder 元数据里的 `model_type`，实例化 `Online{Zipformer,Zipformer2,Conformer,Lstm,Ebranchformer}TransducerModel`。

### 2.3 流状态：每句话的全部状态都在 OnlineStream 里

`CreateStream()`（`online-recognizer-transducer-impl.h:201-206`）创建 `OnlineStream` 并把**解码器空结果**和**encoder 初始状态**存进流里（`InitOnlineStream`，:500-513）——recognizer 无状态，可以同时服务任意多个流。

`OnlineStream`（`online-stream.h:24-137`）关键字段：

- `feat_extractor_`：带锁的特征提取器
- `num_processed_frames_`：已送入 encoder 的帧数
- `start_frame_index_`：本句起点（全局时间轴，**Reset 不清零**）
- `states_`：encoder 缓存状态（`std::vector<Ort::Value>`）
- `result_`：当前解码结果（含 token 上下文）

增量特征提取在 `features.cc`：音频到达即喂给 `knf::OnlineFbank`，帧随来随算；`GetFrames(idx, n)`（:185-215）会**弹出并丢弃已消费的旧帧**——这是流式内存有界的关键。

### 2.4 解码主循环

典型驱动循环（麦克风版 `sherpa-onnx-microphone.cc:144-177`）：

```cpp
while (recognizer.IsReady(s.get())) {       // 攒够一个 chunk？
  recognizer.DecodeStream(s.get());
  text = recognizer.GetResult(s.get()).text; // 中间结果（partial）
}
if (recognizer.IsEndpoint(s.get())) {        // 端点？
  /* 输出最终结果 */
  recognizer.Reset(s.get());                 // 提交帧计数，开始下一句
}
```

`IsReady()` 语义（`online-recognizer-transducer-impl.h:248-251`）：

```cpp
return s->GetNumProcessedFrames() + model_->ChunkSize() < s->NumFramesReady();
```

即缓冲的**未处理帧**必须凑满一个 chunk（zipformer 通常 39 帧 ≈ 390ms）。

`DecodeStreams()`（`online-recognizer-transducer-impl.h:287-358`）每轮：

1. 每流取 `GetFrames(num_processed_frames, ChunkSize())`；推进量是 `ChunkShift()`，**ChunkSize − ChunkShift = 右上下文**（下一 chunk 会重复看到这段）
2. 拼成批张量 + `StackStates` 把各流的 encoder 状态堆成批
3. `RunEncoder` → 新的 encoder_out 和 next_states
4. `decoder_->Decode(encoder_out, &results)` 逐帧解码
5. `UnStackStates` 拆回各流，写回 `SetResult` / `SetStates`

### 2.5 zipformer 模型的状态缓存机制（流式的精髓）

`OnlineZipformerTransducerModel`（`online-zipformer-transducer-model.cc`）持有 **encoder/decoder/joiner 三个 Ort::Session**。

- 构造时从元数据读 `T`（chunk 大小）、`decode_chunk_len`、`num_encoder_layers`、`left_context_len` 等（:79-146）
- `GetEncoderInitStates()`（:387-478）：每层 7 个零张量——`cached_len`/`cached_avg`/`cached_key`/`cached_val`/`cached_val2`（注意力左上下文）+ `cached_conv1`/`cached_conv2`（卷积左上下文），共 `num_layers × 7` 个
- **zipformer 的全部历史上下文都通过这些显式 ONNX 输入/输出张量跨 chunk 传递**——模型本身无状态，状态在 `OnlineStream::states_` 里，每轮 `StackStates`（`cat.h`）/`UnStackStates`（`unbind.h`）
- `RunDecoder` 是**无状态 decoder**：输入只是最近 `context_size` 个 token id（`BuildDecoderInput`，`online-transducer-model.cc:191-207`）
- `RunJoiner`：encoder 帧 + decoder 输出 → logits（未过 softmax）

### 2.6 解码策略

**Greedy search**（`online-transducer-greedy-search-decoder.cc:74-172`）：
- 初始上下文为 `context_size` 个种子 token（结尾是 blank=0）
- 每个 encoder 帧：joiner → argmax；非 blank 且非 unk → 记录 token + 时间戳，`num_trailing_blanks` 清零；否则 `++num_trailing_blanks`（**尾随 blank 计数是端点检测的输入**）
- 有缓存优化：整批流都没发声时不重跑 decoder

**Modified beam search**（`online-transducer-modified-beam-search-decoder.cc:78-259`）：
- 逐帧展开所有假设（hyp），全局 top-k（`num_hyps × vocab_size`）
- `Hypotheses` 按 token 序列去重合并（log-sum-exp）
- 热词：`ContextGraph::ForwardOneStep` 给命中 token 加 `context_score`；可选 LM 浅融合

### 2.7 端点检测与 Reset

`IsEndpoint()`（`online-recognizer-transducer-impl.h:374-389`）：`trailing_silence_frames = num_trailing_blanks × 4`（下采样率硬编码 4）→ 交给 `endpoint.cc:76-94` 判定三条规则。

`Reset()`（:391-439）做的事：

- `segment_++`；新解码上下文只保留最后 `ContextSize()` 个 token（**encoder states 保留**，声学上下文不断）
- `s->Reset()`（`online-stream.cc:49-54`）：`start_frame_index_ += num_processed_frames_`，**不丢弃音频、不重建特征提取器**——只是一个"提交点"

### 2.8 结果

`GetResult()`（`online-transducer` impl :360-372）：`SymbolTable` 转文本 → 去 CJK 间空格 → 时间戳 × `0.01s × 4`（硬编码）→ 填 `segment`/`start_time` → ITN / 同音字纠错。两次端点之间读到的是不断增长的 partial 结果，`IsEndpoint()` 为真时即为该句最终结果。

`OnlineRecognizerResult`（`online-recognizer.h:24-81`）：`text`、`tokens`、`timestamps`、`segment`、`start_time`、`ys_probs` 等。

---

## 3. 容易踩坑的点（值得记住）

1. **offline 的特征在 `AcceptWaveform` 里就全算完了**，online 才是增量算；两者都叫 `AcceptWaveform`，语义不同（offline 只能调一次）
2. **采样缩放由模型元数据决定**（`normalize_samples`）：SenseVoice 要求 ×32768，用户无感知
3. **impl 会用模型元数据覆盖用户的 fbank 配置**（`InitFeatConfig`），命令行参数可能不生效
4. **不显式设 `model_type` 时模型可能被解析两次**（工厂探测 + impl 加载）
5. `DecodeStreams` 是真 batch：padding 后一次 `sess->Run`；没有应用层线程池
6. **provider 不可用时静默回退 CPU**——GPU 没生效时去看日志
7. online 的 `GetResult`/`IsEndpoint` 里 `frame_shift=10ms`、下采样率 4 是**硬编码**的，非 icefall 标准模型的时间戳可能不准
8. 错误处理是 `_Exit()` 而非异常（`SHERPA_ONNX_EXIT`，`macros.h:59-64`）；只有 `Forward` 外围 catch 了 `Ort::Exception`（出错返回空结果）
9. 热词（ContextGraph）只有 transducer 家族支持，且需 `modified_beam_search`

## 4. 建议的阅读顺序

1. `sherpa-onnx-offline.cc` / `sherpa-onnx.cc` / `sherpa-onnx-microphone.cc` —— 先看用法
2. `offline-recognizer.h` / `online-recognizer.h` —— 公开 API 与配置
3. `offline-recognizer-impl.cc` / `online-recognizer-impl.cc` 的 `Create()` —— 工厂分发
4. `offline-stream.cc` / `online-stream.cc` + `features.cc` —— 音频到特征
5. `offline-sense-voice-model.cc` / `online-zipformer-transducer-model.cc` —— ONNX 加载与推理
6. `offline-ctc-greedy-search-decoder.cc` / `online-transducer-greedy-search-decoder.cc` —— 解码
7. `endpoint.cc`（online 特有）—— 端点检测

## 5. 和绑定的关系

iOS/Android/Python 等所有绑定都只是这条 C++ 链路的薄封装：`c-api/c-api.cc` 把上述类包成 flat C 函数（`SherpaOnnxCreateOnlineRecognizer` 等），各语言再包 C API。所以绑定层的怪行为（参数被忽略、时间戳偏差等）答案几乎都在 csrc 里。
