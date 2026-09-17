# FireRedASR CUDA 平台优化记录（GTX 1660 SUPER 服务端）

> 日期：2026-09 · 平台：Linux x86_64 + NVIDIA GTX 1660 SUPER 6GB（TU116，**无 tensor core**）· 代码：fork 分支 `fireredasr-batch-decoding`（`ae0063f3`）
>
> 本文是 CUDA 平台的专项记录，与 `docs/fireredasr-optimization.md`（macOS arm64 / iOS 视角）互补：那边讲模型结构与 C++ 解码优化，这边讲**把同一份模型跑上 CUDA 时的量化格式适配问题**、已落地的两项优化，以及剩余路线图。所有数字均为实测值。

## 0. 头条结果（TL;DR）

服务端 zh 链路（FireRedASR AED，160s 生产音频，jobs 接口端到端）：

| 阶段 | RTF（160s job） | 说明 |
|---|---|---|
| 官方发布 int8，逐句 | 0.318 | 基线 |
| fork int8 + 批量 8 | 0.181 | macOS 侧批量优化移植 |
| fork **fp16** + 批量 8 | 0.154 | 优化一 |
| fork **fp16 + 逐句（MAX_BATCH=0）** | **0.098** | 优化二，累计 **3.2×** |

- 对照参照系：iOS MLX 4h 长音频 RTF ≈ 0.125 —— **服务端首次反超手机端**
- 精度：160s / 800 字文本 md5 三轮全同（`8bd84d39…`），与 int8 逐字一致，零损失
- 显存峰值：逐句 3306 MiB / 批量 8 为 4322 MiB（6GB 卡内安全）
- **本轮没有修改 sherpa-onnx 的 C++ 代码**：fp16 是模型侧转换（新增 `scripts/fire-red-asr/convert-fp16.py`，导出脚本 `export-onnx.py` 未改动），MAX_BATCH 是服务端配置项

## 1. 平台与环境

- GPU：GTX 1660 SUPER 6GB，**TU116 架构无 tensor core**（RTX 20xx 起才有）。旧文档中"GTX 1660 有 INT8 tensor core (IMMA)"的说法是错的，已更正：int8 GEMM 在这张卡上没有硬件加速红利，这直接决定了量化路线的选择
- ORT：fork 构建的 onnxruntime 1.27 CUDA EP（`build-gpu/_deps/onnxruntime-src`）；TensorRT EP 在构建里列出但系统无 libnvinfer，实际不可用
- 服务：`/home/kn/asr-service`（GTX 1660 机器，driver 580.126.09），zh 链路 = silero VAD → FireRedASR fp16 → 标点
- 测量工具（均在远程机 `~/firered-work/`）：
  - `bench/bench_fp16`、`bench/bench_enc2`、`bench/bench_dec` —— 直连 fork ORT CUDA lib 的自建 C++ 基准二进制（g++ 链接 `build-gpu/_deps/onnxruntime-src/lib`），用于隔离单模型/单阶段计时
  - `run_ab.py` —— jobs 接口 3 轮 A/B：RTF 中位 + 文本 md5 + `nvidia-smi` 显存峰值
  - 服务端 `server.py` 已加 stage/batch 计时日志（保留在生产），可直接看 encoder/decoder/每步耗时拆分

## 2. 根因：分离式 int8 与 CUDA EP 不适配

**反常现象**：iOS MLX（手机 CPU/GPU）4h 音频约 30min 识别完（RTF ≈ 0.125），而 GTX 1660 服务端用官方 int8 模型 RTF ≈ 0.27 —— 桌面 GPU 比手机慢一倍多。

定位结论：瓶颈不在算力，在**量化格式与 CUDA EP 的适配**：

- 分离式动态量化（`quantize_dynamic`）的 encoder 图含 **209 MatMulInteger + 163 DynamicQuantizeLinear + 212 Cast**。ORT CUDA EP 只在 MatMulInteger 主算子有 GPU 内核，**量化外围算子（量化/反量化/Cast）全部掉回 CPU**，逼出 **535 个 host↔device Memcpy 节点**（ORT 日志原文计数）。
- decoder 同样（96 MatMulInteger + 96 DQL + 96 Cast），且这些拷贝在**每一步自回归（~290 步）都重跑一遍**。
- 实证：encoder T=500 单跑，CPU 1213ms vs CUDA 1055ms —— **GPU 只快 13%**，PCIe 拷贝税吃光了 GPU 的算力优势。

这也解释了为什么 macOS/iOS 上 int8 是正确答案（CPU EP 上 MatMulInteger 走 sdot/IMMA 类内核，全图无跨设备拷贝），而同一份 int8 模型上 CUDA 就是错误答案。

## 3. 优化一：fp16 转换（encoder 2.67×，文本逐字一致）

**思路（路线 A，不改 onnxruntime 源码）**：既然 int8 的外围算子掉 CPU，就换成 CUDA EP 原生全图支持的格式 —— fp16。TU116 无 tensor core，但 fp16 CUDA core 吞吐仍是 fp32 的 ~2×，且图里不再有任何量化外围算子，零 memcpy。

**流水线**：

```
~/firered-work/model.pth.tar (4.7GB fp32 torch checkpoint)
  → scripts/fire-red-asr/export-onnx.py   # 未改动，重导 fp32（保留动态轴/mask/merge-QKV 全部优化）
  → scripts/fire-red-asr/convert-fp16.py  # 本轮新增，单趟完成转换+修复+压实
```

`convert-fp16.py` 用 `onnxruntime.transformers.float16` 转换，`keep_io_types=True`（IO 保持 fp32，sherpa 侧不用改），`op_block_list` 把 LayerNormalization / Softmax / Sigmoid / Exp / Pow / ReduceMean 等数值敏感算子留在 fp32。

**转换器三个坑（已在脚本内闭环修复）**：

1. graph output 被 Cast 接走后原输出悬空（`graph_output_cast_N` 无生产者）→ 重新接线；
2. 同名 Cast 节点/张量重复（一个名字被 47 次复用）→ 全图重名清扫；
3. 多次 save 在 .data 外部权重文件里留死字节（encoder 一度膨胀到 9.3GB）→ 重写压实到 1.55GB。

**实测（GTX 1660 SUPER，provider=cuda）**：

| 指标 | int8（原） | fp16（现） | 变化 |
|---|---|---|---|
| encoder T=500 单跑 | 1055 ms | **396 ms** | **2.67×** |
| 20s 整段解码 e2e | 5446 ms | **2042 ms** | **2.67×** |
| 160s 生产 job RTF（3 轮中位，批量 8） | 0.181 | **0.154** | −15% |
| 文本 md5（160s，800 字） | `8bd84d39…` | `8bd84d39…`（逐字一致） | 零损失 |
| 显存峰值（批量 8） | ~1010 MiB | 4322 MiB | +3.3GB |
| 模型磁盘 | 1.26 GB | 2.32 GB | +1.1GB |

e2e 收益（−15%）小于单段收益（2.67×）的原因：批量解码已把 int8 的 per-run 拷贝税摊薄到 1/8，fp16 是摊薄后的净算力提升。**单段/小批量场景收益最大**（这直接引出了优化二）。

**回滚**：生产保留 `server.py.bak_int8` 与 `models/zh/*.int8.onnx`，`_build_recognizer` 的 zh 分支改回 int8 文件名即可。

## 4. 优化二：MAX_BATCH=0 —— fp16 之后批量从收益变拖累

macOS 侧的结论"批量 1.2~1.5×"在 fp16 + CUDA 上**翻转**了。实测扫描（160s / 64 段）：

| MAX_BATCH | stage decode | RTF | 结论 |
|---|---|---|---|
| **0（逐句）** | **14.93s** | **0.098** | **最快** |
| 2 | 14.95s | 0.100 | 与逐句持平（分桶退化为逐句） |
| 4 | 24.46s | 0.159 | 变差 |
| 8（fp16） | 23.92s | 0.155 | 变差 |
| 8（int8 时代） | ~28s | 0.181 | int8 时批量 8 曾是最优 |

**机理**：批量 ≥4 时 decoder 每步自回归成本爆炸（批量 ~95ms/步 vs 单句 ~7ms/步）。定位：fork C++ 批量路径的 self-KV cache（N×S 大张量）与 cross K/V 分配在 CPU 内存，**每步自回归往 GPU 搬数百 MB，PCIe 成为瓶颈**。int8 时代 encoder 太慢（~1s/段），批量摊薄 encoder 的收益盖过了这个税；fp16 把 encoder 压到 ~200ms/段后，税凸显，批量净亏。

**处置**：生产默认 `ASR_MAX_BATCH=0`（逐句）。逐句下文本 md5 与批量逐字一致（3 轮 `8bd84d39…`），显存峰值 3306 MiB（比批量 8 还低 1GB）。

**教训**：批量是不是优化，取决于"摊薄的收益"与"批量的每步税"的相对大小 —— 任何一层变快后都要重新测量，不能沿用旧结论。

## 5. 死路清单（已证伪，勿重复尝试）

- **TU116 上指望 int8 硬件加速**：无 tensor core，int8 GEMM 不占便宜；且分离式 int8 在 CUDA EP 必掉 CPU（535 memcpy），打平 CPU（1055 vs 1213ms）
- **VAD（silero v5）优化**：三条路径全部证伪。生产模型是 silero **v5**（非 v4）；官方 sequence 版与生产非等效（160s 中文 max prob diff 0.75，段数 127 vs 113）；非 512 chunk 喂入改变切分（64→23 段）。零数值损耗约束下无可做空间，收益上限本来就只有 ~0.6s/job
- **TensorRT EP**：fork ORT 构建列出了 TRT EP，但系统无 libnvinfer，实际不可用；要用需先装 TensorRT
- **`gpu_mem_limit` 封顶显存（两轮实测均不成立，代码已回滚）**：曾在 fork `session.cc` 加 `SHERPA_ONNX_CUDA_GPU_MEM_LIMIT` / `SHERPA_ONNX_CUDA_ARENA_EXTEND_STRATEGY` 环境变量开关做封顶实验（验证完已 revert 删除）。实测：① 4GiB 无效——**BFCArena 按 OrtSession 独立**，encoder/decoder 各一个 arena，总量照样涨到 5337 MiB;② 2GiB/session 直接任务失败——**这版 ORT 1.27 超限没有 cudaMalloc 回退，Run 硬失败**(BFCArena "Available memory of 0")。且 arena 需求随段长变化，硬上限等于把显存增长换成随机挂任务。要封总量的正路是让 encoder/decoder session 共享同一个 CUDA allocator(ORT 支持 `CreateAllocator`+`RegisterAllocator`)，一个 arena 一个上限

## 6. CUDA 平台剩余路线图（按预期收益排序）

profiling 实证（2026-09-16,encoder fp16 T=500 CUDA,ORT profiler）：单次前向 kernel 时间 ~208ms vs 墙钟 396ms（**~47% 消耗在 kernel 间隙/launch 开销**）;kernel 时间内 Cast **33.7%**、Conv 17.1%、MatMul 仅 **13.7%**、Add/LN/mask 类 ~28%。并用 `SetOptimizedModelFilePath` dump 优化后图确认：**encoder 0 个 com.microsoft 融合节点**（相对位置编码注意力不匹配 ORT MHA fusion pattern）,decoder 有 32 个（16 层 self/cross 注意力已融合）。

1. **消除 Cast 税（kernel 内的最大单块，33.7%）**:fp16 转换护栏（LayerNorm/Sigmoid/Softmax 保 fp32）造成 1554 次/3runs 的 Cast 边界。路径：逐算子验证精度后放宽 `op_block_list`（如 LN 允许 fp16 累加 fp32 的 CUDA 内核），或在导出层把敏感算子改为 fp16 安全写法。属 `convert-fp16.py`/`export-onnx.py` 层面。
2. **CUDA Graph 消除 launch 开销（墙钟的 ~47%）**:encoder/decoder 每步形状固定即可捕获；分段 T 变化可用分桶 padding 解决。ORT provider option `enable_cuda_graph` + fork C++ 侧 IOBinding，不改 ORT 源码。
3. **encoder 注意力融合**:dump 证实当前 0 融合；导出层改造让注意力匹配 ORT fused MHA pattern（或按 SDPA 形式导出）。当前注意力相关 elementwise/Transpose 占比可观，但 Softmax 本身仅 0.4%，收益需实测验证，排在 Cast 和 CUDA Graph 之后。
4. **fork C++ 批量路径 GPU-resident KV cache**:`GetInitialSelfKVCache` / `ForwardDecoder` 的分配器改 CUDA allocator，消除每步 PCIe 搬运，修好后批量收益可恢复（长任务吞吐再上一层）。fork 内改动。
5. **TensorRT EP**：安装 libnvinfer 后可用 TRT EP 试 encoder（静态形状分段或 profile 化）。
6. **阶段 B（用户明确暂缓）**：改 onnxruntime 源码 —— 量化外围算子 GPU 化 / Memcpy 融合 / 显存池复用策略。只有在前几项穷尽后才有必要。

另发现（长程任务实测，3h48/3637 段）：ORT CUDA arena 对变化的分段 shape 只保留不释放，显存从 3305 MiB 长到 5354 MiB 稳定（6GB 卡内安全，更长任务需注意）——与 iOS 端 jetsam 观察到的 arena retention 同源；长音频 RTF 0.131 高于短音频 0.098，原因是语音占比更高（83% vs 69%）+ 分段长尾（均值 3.1s vs 1.7s，p99 14s）下 decoder 成本呈 a·d+b·d² 超线性（二次项来自每步 cross-attention 随段长增长 × 步数随段长正比；encoder 实测近似线性 ~79ms/秒音频，T=250~2000）。

## 7. 复现指引

- fp16 模型转换：fork 仓库 `scripts/fire-red-asr/convert-fp16.py --dir <fp32 导出目录>`（提交 `ae0063f3`）
- 服务端 A/B：远程机 `~/firered-work/run_ab.py`（3 轮 jobs RTF 中位 + md5 + VRAM 峰值）；单模型基准用 `~/firered-work/bench/` 下二进制
- asr-service 侧的完整生产记录（环境变量、显存曲线、VAD 实测、卸载策略）见 asr-service README，本文只收 CUDA 平台优化主线
