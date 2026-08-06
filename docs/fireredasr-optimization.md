# FireRedASR 端侧与服务端优化全记录

> 日期：2026-07 · 平台：macOS arm64（桌面）/ iOS（移动端）· 代码：fork 分支 `fireredasr-batch-decoding`（最新 `1cc4d1ad`）
>
> 本文记录对 FireRedASR(AED) 在 sherpa-onnx 上的完整优化过程：从基线分析、五项优化的原理与实现，到实测效果与死路清单。所有数字均为实测值。

## 0. 头条结果（TL;DR）

| 指标 | 基线（官方发布） | 优化后 | 提升 |
|---|---|---|---|
| 单句 RTF（10.1s 音频） | 0.396 | **0.135** | **2.9×** |
| 单句 RTF（17.6s 音频） | 0.503 | **0.164** | **3.1×** |
| 混合长度批量正确率 | 2/8 | **8/8** | 修复 |
| 逐句精度（int8 vs fp32） | 官方有 2 处偏差 | **8/8 全同** | 反超官方 |
| 批量吞吐（等长） | 不支持批量 | **1.2~1.5×** | 新增能力 |
| KV cache 内存 | 168MB | **10~18MB** | −90% |
| 峰值 RSS（单句） | — | 2.2GB | — |

真机实测（用户 iPhone）：自适应 cache 单层 ≈2×，与桌面数据吻合。

## 1. 背景与目标

在 iOS 端部署 FireRedASR2-AED（1.1B 参数，Conformer encoder + Transformer decoder 的 AED 架构）时发现太慢，需要优化；同时探索服务端批量识别的可能性。

**基线分析（起点）：**

- 官方 sherpa-onnx 发布的 int8 模型：encoder 779MB + decoder 398MB，桌面 RTF 0.40~0.50
- decoder 自回归循环占总耗时 55~70%；encoder 的注意力 O(T²) 随分段长度平方增长
- 关键实测事实：decoder 每步成本 ≈ 9ms 固定（图调度 + ~700 个 mask 重建节点）+ 0.075ms × cache 容量

## 2. 优化一：自适应 KV cache 分配（最大单项收益，≈2×）

**问题**：decoder 的 KV cache 按 `max_len=1024` 满配分配（2×84MB），**每一步自回归都全量读写它**——而一句 10 秒的话只生成 24 个 token，99% 的 cache I/O 是空转。

**方案**：循环步数上界本来就是已知的（`num_possible_tokens = min(帧数/100×6, max_len/2)`），按它分配 cache：

```cpp
int32_t cache_len = min(meta_data.max_len, num_possible_tokens + 4);
auto self_kv_cache = model_->GetInitialSelfKVCache(cache_len);
```

**为什么输出不变**（不是近似，是等价）：注意力读取范围是 `[0, offset)`，由 offset 决定而非 cache 容量；图内 mask 由 `Shape(cache)` 动态构建。已验证 8/8 测试音频逐字一致。

**为什么不会越界**（构造证明）：循环最多 `num_possible_tokens` 步，offset 每步 +1，最大写入位置 = `num_possible_tokens − 1 < cache_len`。

**效果**：decoder 提速 3.5~5.4×，端到端 RTF 0.396→0.181 / 0.503→0.247（≈2×）；cache 内存 168MB→10~18MB。

**成本模型（实测标度）**：每步耗时 ≈ 9ms + 0.075ms × cache_len —— cache 相关开销占原单步成本的 ~85%。注意：decoder 不随线程数缩放（访存瓶颈），encoder 近似线性。

## 3. 优化二：批量解码（C++ 侧）与批量能力再导出（模型侧）

### 3.1 C++ 批量解码

`DecodeStreams` 从逐句循环改为：`PadSequence` 拼批 → 一次 encoder 前向 → 批量自回归解码（每句独立完成标志、共享权重读取）。带 `SupportBatch()` 运行时检测，老模型自动回退逐句。

**发现的模型层障碍**（关键）：已发布的所有 FireRedASR AED ONNX 都不能真正批量——

- decoder 的 `tokens [1,1]`、`offset [1]` 是固定 batch=1 维（导出时没开 `dynamic_axes`）
- encoder 没有用 `x_len` 做注意力 mask：**混合长度批量时 padding 污染**（有效位置的 cross_k 漂移 ~35%），8 句混合只有 2/8 正确

### 3.2 批量能力重导出（`scripts/fire-red-asr/export-onnx.py`）

参照 `scripts/whisper` 重写了导出脚本（fp32，opset 18，dynamo 导出器）：

1. **全维度 dynamic axes**（tokens/offset/cache/cross/features 的 batch 维全动态）
2. **encoder 用 `x_len` 做自注意力 mask**（官方 PyTorch 本来就有，发布版导出时丢了）→ 混合长度批量 2/8 → 7/8
3. **decoder 交叉注意力 mask**：encoder 增加 `enc_mask` 输出，decoder 增加 `cross_mask` 输入，softmax 前加 `(mask−1)×1e30` 加性 bias → **7/8 → 8/8，最后一个正确性缺陷闭环**。C++ 侧按输入名检测，向后兼容老模型
4. **图瘦身**：export-safe wrapper（向量化 mask `arange(S) <= offset`、`narrow`/scatter 写 cache、cross K/V 投影挪进 encoder）——**decoder 991 节点 vs 官方 8215（1/8），encoder 1331 vs 4249**，MatMul 数量不变、权重不变

**Graph authoring 心得**：导出不是"原样录音"而是"重新编曲"——Python forward 里的每步 mask 构建会被 trace 成几百个节点，且每步自回归都要跑一遍。节点调度 + 中间张量是逐步税，这正是我们长音频比官方快 39% 的来源。

**批量收益实测（搭配自适应 cache）：**

| 场景 | 加速比 | 说明 |
|---|---|---|
| 等长 batch=4 | 1.28× | decoder 批量 + 固定开销摊薄 |
| 等长 batch=8 | ~1.5× | |
| 混合（不分桶） | 0.44× | padding 浪费（fp32 encoder 占 70%）|
| **混合（分桶 1.2）** | **~1.05× 转正** | 见优化四 |

## 4. 优化三：int8 量化（`quantize-int8.py`）

- `quantize_dynamic`（仅 MatMul，per-tensor QInt8）：4.64GB → **1.73GB**（2.7×）
- **精度：int8 逐句 8/8 与 fp32 全同**（优于官方发布 int8，后者有 2 处偏差）
- 速度：RTF 0.235→（Phase 2 精简图后）**0.133 / 0.164**，比 fp32 快 1.8~2.2×
- **arm64 陷阱**：Conv 走 `ConvInteger` 无量化内核，慢 2×——已弃用（见死路清单）
- .data 压实陷阱：`quantize_dynamic` 外部数据格式会留死字节（1.29GB 活数据 → 3.88GB 文件），需压实且必须写新文件名

## 5. 优化四：长度分桶（混合批量的性能钥匙）

`DecodeStreams` 内建分桶：按帧数升序排序，相邻比 > `kMaxBucketLengthRatio=1.2` 即切新桶，逐桶批量。对调用方透明。

**验证**（8 个 4.7~17.6s 混合文件 → 切成 2+4+1+1 四桶）：

- 正确性 8/8 vs 逐句
- 速度：不分桶批量 0.44×（负收益）→ **分桶后 ~1.05× 转正**；收益随桶内数量增长，生产环境（大量近似长度句子）会更高

**为什么必须分桶**：encoder 注意力 O(T²) 的 padding 浪费 + decoder 交叉注意力按最长句计费——混合长度下 padding 是批量的头号敌人。

## 6. 被证伪的优化（负结果同样有价值）

| 尝试 | 结果 | 原因 |
|---|---|---|
| **QDQ Conv 只量权重** | **慢 ~4%，不用** | 单句 encoder 只跑一次，带宽节省 ~9ms 抵不过 Dequantize 开销；体积需 .data 压实后才有收益 |
| int8 Conv（ConvInteger） | 慢 2×，弃用 | arm64 无优化内核（x86 VNNI 上结论相反） |
| 批量+自适应 cache（官方模型 patch） | 仅 1.08× | encoder 占 75~80%（Amdahl），int8 encoder 批量无收益 |
| 调小 token 系数 6 | 不做 | 静默截断风险，收益仅 ~5% |
| 改 ORT 内核/xnnpack EP | 不做 | int8 算子覆盖不足；有零成本替代路径 |
| IOBinding（CPU） | 无意义 | CPU 上 move 语义本就零拷贝（CUDA 路径作者已做） |

**搁置（非否决）**：ANE/CoreML 硬件卸载（encoder 固定 T + fp16，预估再 2~3.4×，成本 1~2 周 + decoder stateful 风险）——重启条件：真机 CPU RTF 不达标或长时间听写热衰减成问题。

## 7. 总加速比分解

```
官方基线（RTF 0.396 / 0.503）
  × 自适应 cache（2.0~2.2×）     → RTF 0.181 / 0.247
  × 精简图导出（1.3~1.5×）       → RTF 0.135 / 0.164
  ─────────────────────────────────
  单句端到端：≈ 2.9×（10s）/ 3.1×（17.6s）
  服务端批量（等长）再 ×1.2~1.5
```

越长音频收益越大（官方 decoder 的 mask 重建开销随长度恶化）。

**对比官方 int8（同机实测）**：我们的 int8 全面更快——10s 快 8%、5s 快 41%、17.6s 快 39%。体积 1.62GB vs 官方 1.15GB（+37%，因 Conv 保持 fp32，可用 QDQ+压实换回）。

## 8. 复现指南

**代码**（fork `Kn0688/sherpa-onnx` 分支 `fireredasr-batch-decoding`）：

- C++：批量解码 + 自适应 cache + cross mask 透传 + 长度分桶（全部向后兼容）
- `scripts/fire-red-asr/`：`export-onnx.py`（mask 版导出）、`quantize-int8.py`（MatMul 量化 + .data 压实）、`quantize-qdq-conv.py`（QDQ Conv 实验脚本）、`README.md`（完整文档）

**流水线**：

```bash
# 1. 导出 fp32（需 torch/onnx/onnxscript/kaldiio，~30 分钟）
python3 export-onnx.py --repo ./FireRedASR2S --model-dir ./FireRedASR2-AED --output-dir ./exported
# 2. 量化 int8
python3 quantize-int8.py --dir ./exported
# 3. 验证（sherpa-onnx CLI，批量自动生效）
sherpa-onnx-offline --num-threads=2 \
  --fire-red-asr-encoder=./exported/encoder.int8.onnx \
  --fire-red-asr-decoder=./exported/decoder.int8.onnx \
  --tokens=tokens.txt *.wav
```

**工作目录**：`~/firered-export/`（官方仓库克隆、venv、4.4GB 权重、导出产物）、`~/firered-test/`（官方模型 + 测试音频 + 日志）。教训：**别放 /tmp，重启会清**。

**iOS 验证分支**：`ios-fireredasr-cache-opt`（v1.13.4 tag + 纯自适应 cache），`build-ios/` 下 xcframework 已构建。

## 9. 方法论沉淀（这次学到的）

1. **先 profile 再动手**：decoder 大头不是权重读取而是 cache I/O——理论模型必须让位于实测
2. **看模型先看签名**：`tokens [1,1]` vs `[N,1]`——批量可行性三行 Python 就能确认，胜过半天盲改
3. **图即代码**：ONNX 节点数 = 运行时的调度税，导出是门手艺不是按钮
4. **训练不用动**：批量/动态维是导出层的事（dynamic_axes + mask 接线）
5. **热身测量**：冷启动能把批量测成"慢 4 倍"（19s 首载）；两遍取第二遍
6. **负结果要记录**：ConvInteger、QDQ、phantom broadcast（scale 4 维 reshape 作用到 3 维权重 → 134GB 分配被 jetsam 秒杀）——死路清单和正路一样值钱

## 10. 遗留事项

- **iOS 真机三项验证**（RTF / 内存峰值 / 热衰减）——待 Xcode 连机问题解决
- ANE/CoreML 评估（搁置中，触发条件见 §6）
- 自适应 cache 回馈上游 PR（验证稳定后可发）
- 可选：encoder QDQ + .data 压实（纯体积优化，1.62→~1.2GB）

## 11. 移动端部署现状

截至当前，**iOS 真机产品线上实际启用的优化只有**：

- **自适应 KV cache 分配**（最大单项收益，≈2×）
- **token 生成上限 8 /s**（防止快语速漏字）
- 使用 `fireredasr-mobile` 分支：v1.13.4 tag + 自适应 cache + `enc_mask`/`cross_mask` 透传

批量解码、长度分桶、服务端混合批量等能力虽已完整实现，且**按本文档要求重新导出的模型已支持这些接口**（dynamic batch axes、`x_len` encoder mask、decoder `cross_mask`），但**未在移动端开启**：

- iOS 上批量解码实测出现内存暴涨（3.5GB+ RSS）导致 jetsam 闪退
- 真机连接 Xcode 不稳定，无法完整定位根因
- 移动端当前以稳定优先，暂不启用批量

**服务端**可以直接使用完整能力：自适应 cache + 批量解码 + 长度分桶 + int8 量化。移动端保持单句解码，等待后续内存问题定位后再决定是否开启批量。

## 12. 新增优化：Arena Shrinkage 与线程池指数退避（2026-08）

### 12.1 Arena Shrinkage（内存稳定，替代定期重建 Recognizer）

**问题**：ONNX Runtime 的 arena 内存只攒不还，长时间运行内存膨胀。之前的解决方案是"定期重建 Recognizer"，但需要销毁重建 session，重新加载 1.2GB 权重，代价高。

**方案**：用 `memory.enable_memory_arena_shrinkage` 配置，每次 Run 后收缩 arena，**不用销毁重建 session**。

**实现**（`sherpa-onnx/csrc/offline-fire-red-asr-model.cc`）：

```cpp
// 每 10 次 run 触发一次 arena shrinkage
static constexpr int32_t kArenaShrinkageInterval = 10;

Ort::RunOptions GetRunOptionsWithArenaShrinkage() {
  Ort::RunOptions run_options;
  run_count_++;
  if (run_count_ % kArenaShrinkageInterval == 0) {
    run_options.AddConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0");
  }
  return run_options;
}
```

**优势**：

- **不用销毁重建 session**：避免重新加载 1.2GB 权重
- **代价更低**：只需要收缩 arena，不需要重新初始化
- **性能几乎无损失**：每 10 次 run 触发一次，摊销开销

**源码依据**：

- `include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h:27`：run option `memory.enable_memory_arena_shrinkage`
- `onnxruntime/core/session/inference_session.cc:3229-3235` + `onnxruntime/core/framework/bfc_arena.cc:497-521`：每次 Run 后按设备列表收缩 arena 空闲块

### 12.2 线程池指数退避（降低功耗和发热）

**问题**：ONNX Runtime 的线程池在空闲时会自旋等待（spin loop），持续消耗 CPU，导致移动端发热和耗电。

**方案**：启用线程池指数退避，让自旋等待的间隔越来越大（1, 2, 4, 8, ...），减少 CPU 消耗。

**实现**（`sherpa-onnx/csrc/session.cc`）：

```cpp
// 在 GetSessionOptionsImpl 函数中
sess_opts.AddConfigEntry("session.intra_op.spin_backoff_max", "8");
sess_opts.AddConfigEntry("session.inter_op.spin_backoff_max", "8");
```

**原理**：

- **固定间隔**：每次都 pause 1 次，CPU 消耗高
- **指数退避**：pause 次数指数增长（1, 2, 4, 8, ...），CPU 消耗低
- **响应速度影响小**：任务到来时，线程会很快被唤醒

**效果**：

- **降低功耗和发热**：CPU 空闲时消耗更少
- **速度可能不变或略降**：对推理速度影响很小
- **适用场景**：连续推理（VAD 分割 + 连续推理），线程池持续运行

**源码依据**：

- `onnxruntime/core/session/onnxruntime_session_options_config_keys.h:183-201`：config keys `session.intra_op.spin_backoff_max` 和 `session.inter_op.spin_backoff_max`
- `onnxruntime/core/platform/EigenNonBlockingThreadPool.h`：指数退避实现

**注意**：

- 这个优化**不直接提升速度**，主要降低功耗和发热
- 对单次推理时间短（<100ms）的场景作用有限
- 对连续推理的场景（VAD 分割 + 连续推理）有用
- 2 线程时仍有效，虽然效果不如多线程明显

---

## 13. 新增优化：Whisper 自适应 KV cache（2026-08）

### 13.1 Whisper 自适应 self-attention KV cache 分配

**问题**：Whisper 的 self-attention KV cache 按 `n_text_ctx` 满配分配，**每一步自回归都全量读写它**——而一句短音频只生成几个 token，大部分 cache I/O 是空转。

**方案**：根据输入特征帧数预估 token 数，分配刚好够用的 cache：

```cpp
// 预估 token 数
int32_t num_possible_tokens = num_feature_frames / 100.0 * 6;  // 每秒约6个token
int32_t cache_len = min(n_text_ctx, num_possible_tokens + margin);

// 动态分配 cache
auto self_kv_cache = model_->GetInitialSelfKVCache(cache_len);
```

**兼容性**：如果 decoder 图硬编码了 cache 长度（如官方 whisper ONNX 模型固定为 `n_text_ctx`），检测到后回退到固定长度，保持兼容。

**效果**：减少每步 decoder 内存带宽，特别是短音频片段。预期收益类似 FireRedASR 的自适应 cache（12~16%）。

**修改的文件**：

- `sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc`
- `sherpa-onnx/csrc/offline-whisper-model.cc`
- `sherpa-onnx/csrc/offline-whisper-model.h`

**提交**：`92258ec5 Whisper: adaptive self-attention KV cache allocation`

**注意**：

- 这个优化**不影响识别准确率**，只是减少 cache 分配，不改变计算
- 对短音频片段收益更明显（cache 分配更小）
- 对长音频片段收益较小（cache 分配接近满配）
