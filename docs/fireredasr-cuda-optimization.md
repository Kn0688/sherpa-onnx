# FireRedASR CUDA 平台优化记录（GTX 1660 SUPER 服务端）

> 日期：2026-09 · 平台：Linux x86_64 + NVIDIA GTX 1660 SUPER 6GB（TU116，**无 tensor core**）· 代码：fork 分支 `fireredasr-batch-decoding`（`ae0063f3`）
>
> 本文是 CUDA 平台的专项记录，与 `docs/fireredasr-optimization.md`（macOS arm64 / iOS 视角）互补：那边讲模型结构与 C++ 解码优化，这边讲**把同一份模型跑上 CUDA 时的量化格式适配问题**、已落地的三项优化，以及剩余路线图。所有数字均为实测值。

## 0. 头条结果（TL;DR）

服务端 zh 链路（FireRedASR AED，160s 生产音频，jobs 接口端到端）：

| 阶段 | RTF（160s job） | 说明 |
|---|---|---|
| 官方发布 int8，逐句 | 0.318 | 基线 |
| fork int8 + 批量 8 | 0.181 | macOS 侧批量优化移植 |
| fork fp16 + 批量 8 | 0.154 | 优化一 |
| fork fp16 + 逐句（MAX_BATCH=0） | 0.098 | 优化二，累计 3.2× |
| fork **fp32 encoder + fp16 decoder + 逐句** | **0.036** | 优化三（换模型文件），累计 **8.8×** |
| fork fp32 encoder + fp16 decoder + 逐句 + **禁 BFC arena** | **0.037** | 优化四（fork 首个 CUDA 代码改动），根治长任务 OOM，累计 **8.6×** |
| fork **fp32 encoder + fp32 decoder + 批量 8 + 硬切 20s + 禁 arena** | **0.027** | 批量复活（服务端硬切+攒批，fork 零改动），累计 **11.8×** |

- 对照参照系：iOS MLX 4h 长音频 RTF ≈ 0.125 —— 服务端反超手机端后进一步拉开到 ~4.6×
- **3h48 长音频终验**：fp16 1720s(RTF 0.126)→ fp32+arena 56% 处 OOM → fp32+禁 arena 797s(RTF 0.057)→ **批量链路 858s(RTF 0.061，比 fp16 快 2.0×；比逐句慢 7%——长段被帧预算压缩，但短音频快 42%，按用户决策采纳）**，显存峰值 5420 MiB 无 OOM；74597 字，与逐句链路文本相似度 99.89%（差异仅在 3 个硬切口附近）
- 精度：160s / 800 字文本 md5 三轮全同（`8bd84d39…`），fp32 与 fp16/int8 链路**逐字节一致**，零损失
- 显存峰值：fp32 链路 160s job 4365 MiB（逐句）/ 5232 MiB（批量 8），均禁 arena；3h48 峰值 5420 MiB
- **2026-09-18 批量复活（已采纳）**：批量路径本就 GPU-resident（"每步 PCIe 搬运税"的前提证伪，见 §8）；真瓶颈是 fp16 decoder GEMM 在 N≥4 慢 ~6×。最终方案：fp32 decoder + 长度感知攒批（3000 帧×N 预算）+ **服务端硬切 20s**（VAD `max_speech_duration` 是软上限，连续语音实测冲到 27.65s；硬切在 19-20s 窗口找能量最低点递归切分，零样本丢失）。曾因需截断到 10s 被拒，硬切把截断影响收敛到超长段本身（3h48 仅 3 段触发）后采纳
- **2026-09-17 decoder CUDA Graph 实验：已实现、验证、移除**。墙钟零收益（生产 decoder 早已 GPU-resident 6.1ms/步），3h48 长任务反而 +4.6%，双路径复杂度不抵收益，代码已移除（见死路清单）。fork C++ 保持无功能性改动
- **2026-09-17 重要更正**：此前"encoder 47% 墙钟消耗在 kernel 间隙"的结论是**测量错误**（ORT profiler 口径问题，见 §5）。真实根因是 fp16 GEMM 在无 tensor core 的 TU116 上比 fp32 慢 ~6×，直接催生了优化三

## 1. 平台与环境

- GPU：GTX 1660 SUPER 6GB，**TU116 架构无 tensor core**（RTX 20xx 起才有）。旧文档中"GTX 1660 有 INT8 tensor core (IMMA)"的说法是错的，已更正：int8 GEMM 在这张卡上没有硬件加速红利，这直接决定了量化路线的选择
- ORT：fork 构建的 onnxruntime 1.27 CUDA EP（`build-gpu/_deps/onnxruntime-src`）；TensorRT EP 在构建里列出但系统无 libnvinfer，实际不可用
- 服务：`/home/kn/asr-service`（GTX 1660 机器，driver 580.126.09），zh 链路 = silero VAD → FireRedASR（encoder fp32 + decoder fp16，见 §5）→ 标点
- 测量工具（均在远程机 `~/firered-work/`）：
  - `bench/bench_fp16`、`bench/bench_enc2`、`bench/bench_dec` —— 直连 fork ORT CUDA lib 的自建 C++ 基准二进制（g++ 链接 `build-gpu/_deps/onnxruntime-src/lib`），用于隔离单模型/单阶段计时
  - `cublas_probe/bench_cublas` —— cublas/cublasLt GEMM 微基准（fp16/fp32/algo 遍历）
  - `run_ab.py` —— jobs 接口 3 轮 A/B：RTF 中位 + 文本 md5 + `nvidia-smi` 显存峰值
  - 服务端 `server.py` 已加 stage/batch 计时日志（保留在生产），可直接看 encoder/decoder/每步耗时拆分

## 2. 根因：分离式 int8 与 CUDA EP 不适配

**反常现象**：iOS MLX（手机 CPU/GPU）4h 音频约 30min 识别完（RTF ≈ 0.125），而 GTX 1660 服务端用官方 int8 模型 RTF ≈ 0.27 —— 桌面 GPU 比手机慢一倍多。

定位结论：瓶颈不在算力，在**量化格式与 CUDA EP 的适配**：

- 分离式动态量化（`quantize_dynamic`）的 encoder 图含 **209 MatMulInteger + 163 DynamicQuantizeLinear + 212 Cast**。ORT CUDA EP 只在 MatMulInteger 主算子有 GPU 内核，**量化外围算子（量化/反量化/Cast）全部掉回 CPU**，逼出 **535 个 host↔device Memcpy 节点**（ORT 日志原文计数）。
- decoder 同样（96 MatMulInteger + 96 DQL + 96 Cast），且这些拷贝在**每一步自回归（~290 步）都重跑一遍**。
- 实证：encoder T=500 单跑，CPU 1213ms vs CUDA 1055ms —— **GPU 只快 13%**，PCIe 拷贝税吃光了 GPU 的算力优势。

这也解释了为什么 macOS/iOS 上 int8 是正确答案（CPU EP 上 MatMulInteger 走 sdot/IMMA 类内核，全图无跨设备拷贝），而同一份 int8 模型上 CUDA 就是错误答案。

## 3. 优化一：fp16 转换（相对 int8 2.67×，文本逐字一致）

**思路（路线 A，不改 onnxruntime 源码）**：既然 int8 的外围算子掉 CPU，就换成 CUDA EP 原生全图支持的格式 —— fp16。图里不再有任何量化外围算子，零 memcpy。

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

**回滚**:`server.py.bak_int8` 保留;`models/zh/*.int8.onnx` 已于 2026-09-18 磁盘清理中删除，官方版可从 `backup_pre_fork_20260914/models_zh/` 拷回后把 `_build_recognizer` 的 zh 分支改回 int8 文件名。

> **2026-09-17 更正**：本节原文假设"fp16 CUDA core 吞吐仍是 fp32 的 ~2×"——**对 GEMM 不成立**。cublas 在 TU116（无 tensor core）上 fp16 GEMM 实测仅 ~0.5 TFLOPS，而 fp32 SGEMM 有 3-4.5 TFLOPS，fp16 encoder 比 fp32 慢 ~6×（见 §5）。fp16 当时的 2.67× 是相对烂到底的 int8（535 memcpy）而言的；相对 fp32，fp16 是负优化。encoder 已换 fp32；decoder 因生产逐句 N=1 形态是带宽 bound（fp16 权重字节减半占优）维持 fp16。

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

## 5. 优化三：encoder 换 fp32（2026-09-17，RTF 0.098 → 0.036，2.72×，零代码）

**起源**：追查"encoder kernel ~208ms vs 墙钟 396ms，47% 间隙"的过程中，nsys 实测证明**间隙不存在**——ORT 1.27 profiler 的 `*_kernel_time` 事件是 CPU 侧时间戳包住 `OpKernel::Compute()`（async launch），不是 GPU kernel 时长（v1.27.0 `sequential_executor.cc` KernelScope 源码确认）。nsys ground truth（encoder fp16 T=500，6 次 run）：9826 个 kernel，GPU **97.8% 时间忙碌**，每 run kernel busy ≈ 391ms ≈ 墙钟 397ms；GEMM 占 96.7%，Cast 全部仅 ~3.4ms/run，H2D/DtoH 拷贝合计 <1%。`CUDA_LAUNCH_BLOCKING=1` 对照实验（node dur 总和 408.7ms，其中 MatMul 386.5ms）交叉印证。这同时解释了此前两个"无效"：CUDA Graph 无效是因为**没有 launch 间隙可消**；Cast 削减 -77% 墙钟不动是因为 Cast kernel 本来就只占 ~1%。

**真正的慢因**：cublas fp16 GEMM 在 TU116 上只有 ~0.5 TFLOPS，fp32 SGEMM 有 3-4.5 TFLOPS。cublas 微基准（`~/firered-work/cublas_probe/bench_cublas`，M=125 K=1280 N=5120，50 次均值）：

| 路径 | 耗时 |
|---|---|
| fp32 SGEMM | **0.46 ms** |
| fp16 compute32 | 3.83 ms |
| fp16 compute16 | 2.95 ms |
| cublasLt 遍历全部候选 algo 最优 | 3.14 ms |

tensor-op math 无效（无 tensor core），ORT TunableOp 无效（396/397/397ms）——硬件+库层面死局，ORT 层面无解。fp16 T 扫描还有平台期（T=300≈381ms ≈ T=500≈400ms）：这些 GEMM 是 skinny-M（M=75~125），kernel 时间由 N×K 决定、几乎与 M 无关，fp16 在短段上浪费更狠。

**T 扫描对比（ms，3 次中位数）**：

| T | 100 | 300 | 500 | 700 | 1000 | 1400 |
|---|---|---|---|---|---|---|
| fp16 | 163 | 381 | 400 | 634 | 796 | 1230 |
| fp32 | 20 | 56 | **66** | 106 | 147 | 207 |

**处置（零代码，只换模型文件）**：

- 干净重导出 fp32 encoder：onnx API load + save（单外部数据文件），产物 `~/firered-work/export_clean/encoder.fp32.onnx` + `.data` = **3,103,167,616 字节 ≈ 3.10GB**（与 775.8M params × 4B 精确吻合；原 `out/encoder.onnx.data` 6.2GB 确系重复 append 的坏文件）。转换器坑 #3 的又一次印证：外部数据文件多次 save 会留死字节，导出后必须核对文件大小 ≈ 参数量 × 字节数
- 数值验证：与生产 fp16 encoder 输出 cosine = **1.000000 / 0.999999 / 1.000000**（cross_k/cross_v/mask，T=300）；输出 dtype（fp32）与 decoder.fp16 输入完全匹配
- 部署：`/home/kn/asr-service/models/zh/encoder.fp32.onnx`(+`.data`)，`server.py:127` 一行改指向（备份 `server.py.bak_fp32enc`;fp16 文件已于 2026-09-18 清理，回滚需用 `convert-fp16.py` 重转）

**实测（160s 生产 job，run_ab ×3）**：

| 指标 | fp16 encoder | fp32 encoder |
|---|---|---|
| RTF（3 次） | 0.0980 / 0.0980 / 0.0970 | **0.0360 / 0.0360 / 0.0360** |
| 处理耗时 | 15.6-15.8s | **5.791s（2.72×）** |
| 文本 md5 | `8bd84d39…` | `8bd84d39…`（**逐字节相同**，800 字零差异） |
| 显存峰值 | ~3300 MiB | 5337 / 6144 MiB |

冒烟：cn_2min（120s，lang=zh）RTF 0.052，文本连贯正常。

**decoder 维持 fp16（实测后主动决策）**：生产是逐句解码（N=1），此形态 decoder 每步是带宽 bound（385M 参数权重读取），fp16 权重字节减半反而占优——N=1, S=100, Tc=500 实测 fp16 **21.0** vs fp32 22.7 ms/step（fp32 慢 8%）。只有批量形态 fp32 才占优（N=8, S=150：fp16 223 vs fp32 **151** ms/step，1.47×）。将来 GPU-resident KV cache 落地、批量解码重启后需重新评估（届时精度选择可能再次翻转）。

**教训**：

1. **选精度前先用 cublas 微基准测硬件实际 GEMM 吞吐**，不要按"fp16 有 2× CUDA core 吞吐"的纸面规格推断——无 tensor core 的卡上 cublas fp16 GEMM 走非 tensor-op 路径，实测只有 ~0.5 TFLOPS。有 tensor core 的卡（RTX 20xx+）结论会反过来
2. **ORT profiler 的 kernel_time 是 CPU 侧 launch 视角**，判断 GPU 间隙必须上 nsys/CUPTI；"profiler 里某类节点占比高"不等于"它是墙钟瓶颈"（Cast 税假象与此同源）
3. int8 → fp16 → fp32 三连换的完整弧线：每换一次基线，上一轮的"正确答案"都可能被推翻

## 6. 优化四：禁用 BFC arena（2026-09-17，根治长任务 OOM，fork 首个 CUDA 代码改动）

**起因**：fp32 encoder 上线后 3h48 长音频在 decode 56% 处 OOM（encoder `Where` 节点申请 2.5MB 失败），而 160s 短音频无恙（峰值 5337）。

**量化归因**（bench memsweep + 生产 VRAM 采样，结论先行：**是 arena 状态问题，不是段长问题**）：

- fp32 encoder 加载基线 3343 MiB；**单段瞬时需求 1026 MiB，与段长无关**（T=500~2760 相同）；空卡跑任意单段峰值 5401 MiB，放得下
- 生产 OOM 时平台 5695 MiB = 权重 3.9GB + context ~240MB + **arena retention ~1.6GB**，余量仅 449 MiB
- BFC arena 默认 kNextPowerOfTwo，扩展粒度 ~1GB/跳；449 < 1024 → 任何 shape 组合碰巧不复用 retained 块的普通段都会 OOM。实锤：挂掉的是 6.25s 普通段，而 27.65s 最长段早已顺利解码；失败在中段（56%）而非遇到大段时。VAD 砍段长方案据此否决（单段显存与 T 无关）
- arena retention 只涨不跌，bench 进程复现（跑完 T=2760 回 T=500 保持 5401 不回落）

**排除的路线**：

- `arena_extend_strategy=kSameAsRequested`：bench 实测与默认**逐字节一致**（~1GB 跳变是 cudnn workspace 等策略外分配），无收益证据
- **CudaMempoolArena**（cudaMallocAsync，`CreateArenaCfgV2` 的 `use_cuda_mempool`）：bench 里曲线完美（锯齿回落 3249、峰值 3953、耗时持平），但**服务内实测崩溃**（ORT 1.27 `cuda_mempool_arena.cc:179` cudaFreeAsync illegal memory access，CUDA 700，进程终止）。不可用，代码未合入

**落地**（fork `session.cc` + `ort-env.h` + `python/csrc/CMakeLists.txt`，提交 `c95b3d8e`，默认行为不变）：

`SHERPA_ONNX_CUDA_USE_ARENA=0` → `CreateOrtEnv` 注册 raw cudaMalloc/cudaFree 分配器（extern "C" 运行时从 libcudart 解析）+ session 加 `session.use_env_allocators=1`，完全绕过 arena。生产 `run.sh` 已 export 该开关。

**实测**：

| 指标 | fp32+arena | fp32+禁 arena |
|---|---|---|
| encoder T=500/1400/2760 | 78/205/420ms | 78/207/421ms（持平） |
| decoder fp16 N=1 | 21.21 ms/step | 23.60 ms/step（+11.2%） |
| 160s RTF ×3 | 0.0360 | **0.0370（+2.8%）**，md5 `8bd84d39…` 逐字节不变 |
| 160s VRAM 峰值 | 5337 | **4365（-972 MiB）** |
| 3h48 | 56% OOM | **797s 完成（RTF 0.057）**，74514 字 |
| 3h48 VRAM | 平台 5695 → OOM | **锯齿 4337~4797，结束回落** |

**代价与边界**：decoder 每步 +11%（每步数百次真实 cudaMalloc/Free 的同步开销），encoder 主导下端到端仅 +2.8%；换来显存"用多少占多少"，任意长音频不再有 arena 累积 OOM 风险。

## 7. 死路清单（已证伪，勿重复尝试）

- **TU116 上指望 int8 硬件加速**：无 tensor core，int8 GEMM 不占便宜；且分离式 int8 在 CUDA EP 必掉 CPU（535 memcpy），打平 CPU（1055 vs 1213ms）
- **TU116 上抢救 fp16 GEMM（2026-09-17 全部证伪）**：tensor-op math 无效（无 tensor core）、cublasLt 全 algo 遍历最优 3.14ms（fp32 0.46ms）、ORT TunableOp 无效（396/397/397ms）、自研 fp16 SIMT GEMM 相对 fp32 无优势。正解就是 fp32
- **VAD（silero v5）优化**：三条路径全部证伪。生产模型是 silero **v5**（非 v4）；官方 sequence 版与生产非等效（160s 中文 max prob diff 0.75，段数 127 vs 113）；非 512 chunk 喂入改变切分（64→23 段）。零数值损耗约束下无可做空间，收益上限本来就只有 ~0.6s/job
- **TensorRT EP**：fork ORT 构建列出了 TRT EP，但系统无 libnvinfer；且 TU116 无 tensor core，TRT fp16 不会比 fp32 快，此卡上放弃
- **`gpu_mem_limit` 封顶显存（两轮实测均不成立，代码已回滚）**：曾在 fork `session.cc` 加 `SHERPA_ONNX_CUDA_GPU_MEM_LIMIT` / `SHERPA_ONNX_CUDA_ARENA_EXTEND_STRATEGY` 环境变量开关做封顶实验（验证完已 revert 删除）。实测：① 4GiB 无效——**BFCArena 按 OrtSession 独立**，encoder/decoder 各一个 arena，总量照样涨到 5337 MiB;② 2GiB/session 直接任务失败——**这版 ORT 1.27 超限没有 cudaMalloc 回退，Run 硬失败**(BFCArena "Available memory of 0")。且 arena 需求随段长变化，硬上限等于把显存增长换成随机挂任务。要封总量的正路是让 encoder/decoder session 共享同一个 CUDA allocator(ORT 支持 `CreateAllocator`+`RegisterAllocator`)，一个 arena 一个上限
- **CudaMempoolArena（cudaMallocAsync，2026-09-17 证伪）**：bench 里显存曲线完美（锯齿回落、耗时持平），但服务内 160s decode 实测**进程级崩溃**（ORT 1.27 `cuda_mempool_arena.cc:179` cudaFreeAsync illegal memory access，CUDA 700）。bench 复现不了生产崩溃条件，勿再启用；arena-off 的正解是 raw allocator（§6）
- **decoder CUDA Graph（2026-09-17 完整落地验证后移除）**：在 fork `offline-fire-red-asr-model.cc` 实现了 decoder 自回归步的 CUDA Graph 捕获/回放（按桶后形状缓存 graph、固定 device buffer、D2D 回喂 self KV、MAX_CONTEXTS 上限 + 回退），数值验证完全正确（160s md5 `8bd84d39…` ×3;cn_2min 图开/关/回退触发三模式 md5 一致；3h48 74490 字、显存 5375 MiB 平台期不 OOM)。**但墙钟零收益**——生产 decoder 早已是 IOBinding + KV cache GPU-resident 的 6.1ms/步（GPU kernel-work 下限），graph replay 6.06ms/步；探针报告的 3.0× 对照组是非驻留路径，生产不存在那份税。唯一收益是 CPU 发射 991→1 次/步，单 worker 场景用不上；3h48 长任务反而 1799s vs 基线 1720s(+4.6%，每步 D2D 回喂拷贝的开销）。复杂度（双路径 + 下述 ORT 坑）不抵收益，代码已移除，如需复活见 git 历史 `9ae5c52a`+`af3abb12`。**ORT 1.27 关键坑（留给未来）**:graph 会话上不带 `gpu_graph_id` 的 Run 默认 annotation id=0,ORT 跑够 `min_num_runs_before_cuda_graph_capture` 次后会**静默为 id 0 捕获图**（绑定当次 Run 的临时 buffer)，后续 replay 输出不刷新 → `OrtValue Get<Tensor> on null` 崩溃；不参与 graph 的 Run 必须显式 `gpu_graph_id=-1`(`kCudaGraphAnnotationSkip`)

## 8. 现状与剩余路线图（fp32 + 禁 arena 落地后，2026-09-18 更新）

fp32 encoder 落地后 160s decode 从 ~15s 降到 5.79s，encoder 占比大幅下降，decoder（fp16，逐句 6.1ms/步）成为相对大头。剩余方向按现状排序：

1. ~~**fork C++ 批量路径 GPU-resident KV cache**~~ **（2026-09-18 前提证伪 → 同日以另一形态落地，已上线）**：读码+实测确认 fork 批量路径**本就 GPU-resident**（上游 `711acccd` 的 IOBinding 已在部署代码里：encoder cross K/V 直绑 CUDA 内存、decoder self-KV 零拷贝回喂，每步 PCIe 仅 tokens/offset ~128B + logits D2H N=8 时 277KB）。批量慢的真因仍是 **fp16 GEMM 在无 tensor core 卡上慢 ~6×**（N=8 时 decoder fp16 96.5ms/步 vs fp32 21.8ms/步；逐句 N=1 带宽 bound 所以 fp16 才对）。**落地形态（全部服务端改动，fork 零改动）**：fp32 decoder + 长度感知攒批（3000 帧×N 预算，长段自动缩批）+ **硬切 20s**（VAD 分段后 >20s 的段在 19-20s 窗口找 50ms 帧能量最低点递归切开，单元自测零样本丢失、时间戳连续）。实测：160s RTF **0.027**（逐句 0.037 的 1.4×，md5 逐字节一致，硬切不触发）；3h48 858s/RTF 0.061（比逐句 797s 慢 7%，长段被帧预算压缩 + N=1 步 fp32 慢 8%；比 fp16 时代 1720s 快 2.0×），74597 字，与逐句链路相似度 99.89%（3 个硬切口），VRAM 峰值 5420 MiB 无 OOM。**关键机制认知**：silero `max_speech_duration` 是软上限不是硬切（`voice-activity-detector.cc:50-65`：超时后 threshold 0.5→0.9、min_silence 0.25→0.1，找切点但不硬切），连续语音实测可长至 27.65s——显存规划必须按"段长无硬顶"做，这正是硬切的由来。教训：先以"截 10s"形态被拒（影响所有 10-20s 段），把截断收敛为只影响超长段本身的硬切后才可接受——**约束的形态决定方案的可采纳性**
2. **decoder 逐句形态进一步压缩**：N=1 带宽 bound 下，减少每步权重读取是唯一方向（权重 int8 量化只省带宽不要 int8 GEMM——TU116 上 int8 无硬件加速但字节数减半，待测；注意与 §2 分离式 int8 的 memcpy 坑区分，这里指 weight-only QDQ 类方案）。禁 arena 后 decoder 每步还有 +11% 的真实分配开销，自定义轻量 pool 分配器是潜在回收点
3. **encoder 侧**：fp32 后 T=500 仅 66ms，绝对空间已小；0 个 com.microsoft 融合节点（相对位置编码注意力不匹配 ORT MHA fusion pattern）的问题仍在，但生产均值段 T≈311 下收益有限，优先级低
4. **阶段 B（ORT 源码级）**：最大杠杆（fp16 GEMM 死局）已被 fp32 绕过；显存封顶需求已被禁 arena 解决（§6）。剩余可选：量化外围算子 GPU 化（仅当 decoder 走 int8 权重路径时相关）

另发现（长程任务实测）：ORT CUDA arena 对变化的分段 shape 只保留不释放——fp16 链路 3h48 显存从 3305 MiB 长到 5354 MiB 平台（侥幸安全）；**fp32 链路同一现象直接顶穿 6GB 卡（56% 处 OOM），最终靠禁 arena 解决（§6）**——与 iOS 端 jetsam 观察到的 arena retention 同源。长音频 RTF 高于短音频（fp16 时代 0.131 vs 0.098；fp32 时代 0.057 vs 0.037），原因是语音占比更高（83% vs 69%）+ 分段长尾（均值 3.1s vs 1.7s，p99 14s）下 decoder 成本呈 a·d+b·d² 超线性（二次项来自每步 cross-attention 随段长增长 × 步数随段长正比；encoder 实测近似线性，fp16 ~79ms/秒音频、fp32 ~15ms/秒音频，T=250~2760）。

## 9. 复现指引

- 禁 arena：fork `c95b3d8e` 起，`SHERPA_ONNX_CUDA_USE_ARENA=0` 环境变量（生产 `run.sh` 已内置）；不设则保持原 BFC arena 行为
- fp32 encoder：远端 `~/firered-work/export_clean/encoder.fp32.onnx`（+3.10GB `.data`），onnx API load + save 单外部文件压实导出；生产位于 `models/zh/encoder.fp32.onnx`，回滚 = `server.py` 改回 `encoder.fp16.onnx`（或 cp `server.py.bak_fp32enc`;fp16 文件已清理，需先重转）
- fp16 模型转换：fork 仓库 `scripts/fire-red-asr/convert-fp16.py --dir <fp32 导出目录>`（提交 `ae0063f3`）
- 服务端 A/B：远程机 `~/firered-work/run_ab.py`（3 轮 jobs RTF 中位 + md5 + VRAM 峰值）；单模型基准用 `~/firered-work/bench/` 下二进制；GEMM 微基准 `~/firered-work/cublas_probe/bench_cublas`
- asr-service 侧的完整生产记录（环境变量、显存曲线、VAD 实测、卸载策略）见 asr-service README，本文只收 CUDA 平台优化主线
