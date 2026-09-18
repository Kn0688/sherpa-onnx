# Qwen3-ASR CUDA 平台优化记录(GTX 1660 SUPER 服务端)

> 日期:2026-09-18 · 平台:Linux x86_64 + NVIDIA GTX 1660 SUPER 6GB(TU116,**无 tensor core**)· 代码:fork 分支 `fireredasr-batch-decoding`(commits `02235571..3e3b28ea`)
>
> 对象:asr-service `lang=other`/`auto` 链路的 Qwen3-ASR-0.6B(conv_frontend + encoder + decoder 三件套)。与 `docs/fireredasr-cuda-optimization.md`(zh 链路)互补:根因同类(分离式 int8 与 CUDA EP 不适配),但解法形态不同——这边没有自写导出脚本,用的是社区导出工具 + fork C++ 两处 KV 路径改造。所有数字均为实测值。

## 0. 头条结果(TL;DR)

生产 A/B(jobs 接口端到端,RTF 中位):

| 阶段 | codeswitch(6.2s)×3 | en_90s ×2 | 显存峰值(en_90s) |
|---|---|---|---|
| 官方 int8(基线) | 0.709 | 0.815 | 1088 MiB |
| Phase 1:fp32 重导出 + env 修复 | 0.398 | 0.503 | 4570 MiB |
| Phase 2:+ GPU 驻留 KV | 0.136 | 0.1915 | 4538 MiB |
| Phase 3:+ adaptive KV | **0.136** | **0.191** | **4315 MiB** |

- **累计加速:codeswitch 5.2×,en_90s 4.3×(对官方 int8)**
- 文本验收:fp32 导出与 PyTorch 参考 `max_diff=0`,用户批准 fp32 文本为新基线;Phase 2/3 与 fp32 基线**逐字节全等**(codeswitch `5fbc4802…`、en_90s `4de21e74…`、en_60s 直解 `86b1109e…`,各轮 md5 不变)
- 长音频:12.5min 英文(en_3h40.m4a 截取)生产链路 RTF **0.158**,峰值 5135 MiB,无 OOM,文本 11374 字连贯
- zh 回归红线:long_zh_3min ×3 md5 全部 `8bd84d39…`(FireRedASR 链路未受影响)

## 1. 根因:分离式 int8 与 CUDA EP 不适配(与 zh 链路同病)

官方 csukuangfj 转换的 int8 三件套是 `quantize_dynamic` 分离式量化:

- encoder:**110 MatMulInteger + 74 DynamicQuantizeLinear**;decoder:8016 节点,**197 MMI + 856 Cast**。CUDA EP 只有 MMI 主算子有 GPU 内核,量化外围算子全掉 CPU,逐节点乒乓。
- 实证(codeswitch 6.2s 直解):int8+cuda 3.99s vs **CPU 4 线程 1.34s —— CPU 比 CUDA 快 ~3×**。
- 第二宗罪(调查期发现):sherpa 上游 qwen3 实现的 KV cache 分配在 **CPU 内存**,28 层 × [1, 2048, 8, 128] fp32(K+V)≈ **每步 ~470MB H2D**;60s 语音 ~450 步自回归,合计 ~210GB PCIe 搬运。这是比量化格式更大的税,但 int8 时代被 encoder 的乒乓掩盖。

## 2. Phase 1:fp32 重导出 + env 修复(RTF 0.709→0.398 / 0.815→0.503)

零 C++ 功能改动,只换模型文件 + 一处 env 接线。

- **导出**:社区工具 [Wasser1462/Qwen3-ASR-onnx](https://github.com/Wasser1462/Qwen3-ASR-onnx),`python export_qwen3_asr_onnx.py --model ~/qwen3-work/Qwen3-ASR-0.6B --outdir ~/qwen3-work/export_fp32 --max-total-len 2048 --no-int8 --verify`(conda env `qwen3-asr`:torch 2.9.1 / transformers 4.57.6)。`--verify` 对 PyTorch 参考 **max_diff < 2e-5**。
- **签名对齐**:导出前先用 `~/qwen3-work/dump_signature.py` 存档 int8 三件套 I/O 签名,导出后逐一比对(名字+shape)强制全等才许部署。期间修过 key_delta 输出 dim 符号名不一致的问题(导出侧修复,不动 sherpa 侧)。
- **产物**(`models/other/fp32/`):conv_frontend 44MB + encoder 707MB + decoder 3.0GB。decoder 比"参数量×4"大是因为 embed/lm_head tie 权重未合并导出,可接受(磁盘不紧缺)。
- **env 修复**(fork commit `02235571`):qwen3 模型类此前自建 `Ort::Env`,导致 `SHERPA_ONNX_CUDA_USE_ARENA=0`(zh 链路的 raw-allocator 开关,`c95b3d8e`)对它不生效;改为共享 `CreateOrtEnv()` 后生效。
- **文本裁决**:int8→fp32 有个别 token 分叉(codeswitch 的法语段、en_90s 的标点/大小写),逐条列出后用户批准 fp32 文本为新基线(int8 与 PyTorch 参考本就有量化误差,fp32 `max_diff=0` 更贴近参考)。
- **实测**:生产 A/B codeswitch 0.398 / en_90s 0.503(对 int8 为 1.78×/1.62×);直解 codeswitch 2.30s(cuda)。

## 3. Phase 2:GPU 驻留 KV(commit `271e0506`,RTF →0.136/0.1915)

模板:同仓库 FireRedASR 的 IOBinding GPU-resident 写法。只改 `sherpa-onnx/csrc/offline-qwen3-asr-model.cc`(+100/-20),仅影响 `use_cuda_iobinding_` 路径,CPU 路径原样:

1. `CreateEmptyKVCache`:CUDA 时用 `OrtApi::CreateAllocator(decoder_sess, cuda_mem_info)` 的 device allocator 分配(尊重 env 注册的 raw allocator),cudaMemset 清零 + 一次 `cudaDeviceSynchronize`;
2. `ForwardLLM`:KV cache 以 device 张量 View 后 BindInput,零 H2D;KV delta 输出绑 device;
3. 写回:`ApplyKvDeltaInplace` CUDA 分支用 `cudaMemcpy(DeviceToDevice)`(偏移数学与 CPU 路径完全一致),层循环后一次 `cudaDeviceSynchronize()`(ORT 用自己的 stream,必须同步防竞争);
4. **有意偏离原计划契约**:encoder 输出 `audio_features` 仍绑回 CPU——`TrimAudioFeatures`(impl 侧)需要在 CPU 上扫数据找有效帧;audio_features 每步重传仅 ~6MB/step(vs KV 470MB/step),代价可忽略。impl 文件零改动;
5. CPU memcpy 路径保留在非 CUDA 分支。

- **实测**:直解 codeswitch 2.30→1.15s(2.0×)、en_60s 32.03→15.26s(2.1×),文本 md5 逐字节全等;生产 A/B codeswitch 0.136 / en_90s 0.1915。显存峰值基本持平(4434/4538 MiB)——KV 驻留的 +470MB 被每步 H2D 暂存缓冲的消除抵消。
- 部署坑(记录在案):venv 实际加载的是 `site-packages/sherpa_onnx/lib/_sherpa_onnx*.so`(`__init__.py` 写 `from sherpa_onnx.lib._sherpa_onnx import`),不是包根目录那份;装错位置会静默继续跑旧 .so——验证 C++ 改动是否生效要看 debug 日志的**行号**是否移位。

## 4. Phase 3:adaptive KV 分配(commit `3e3b28ea`,RTF 持平,显存 −170~220MiB)

思路同 FireRedASR 的 `min(max_len, estimated+4)`:每流 KV 分配长度 = `min(max_total_len, context_len + max_new_tokens + 8)`(8 为余量;context_len = prompt + audio pads,在 prompt 截断逻辑之后计算),替代固定 2048。decoder cache 轴在导出的 ONNX 里是动态符号轴,ORT 侧零改动。`ApplyKvDeltaInplace` 的写回 stride 与越界夹取改用 cache 张量自身 dim1(原用 `max_total_len_`,缩容后会写飞)。CPU/CUDA 两路径同时受益。

- **实测**:RTF 持平(codeswitch 0.1360 / en_90s 0.1910,变化 <1% 在噪音内)——KV 已 GPU 驻留后,attention 读量减少被 prefill logits 等固定开销淹没,与预期管理一致。**收益在显存**:峰值 4434→4261 / 4538→4315 MiB;单流 KV 从固定 ~470MB 变为按需(60s 语音 alloc_len=1315,~300MB),为多路并发留空间。
- **边界自测**(`~/qwen3-work/qwen3_boundary.py`,debug 日志实证):60s 音频 alloc_len=1315(max_new=512)/ 819(=795+16+8,max_new=16,公式精确);max_new=16 撞上限时截断告警正常、无越界,13 个有效 token 与全量输出前 13 个 token 级全等。

## 5. 长音频验证(12.5min,生产链路)

en_3h40.m4a 截取前 12.5 分钟(ffmpeg,-ar 16000 mono),lang=other 走生产 jobs:RTF **0.158**,显存峰值 5135 MiB,无 OOM,文本 11374 字首尾连贯。VAD 把音频切成 ≤60s 段,单段都在 fp32 直解天花板内。

## 6. 遗留与死路(勿重复尝试 / 后续可做)

- **fp32+cuda 直解 65-75s 显存天花板(已知风险,本计划未含)**:prefill logits [1, seq, 151936] 是主犯(seq 即 context_len)。直解实测 60s OK、75s OOM。生产 VAD other 链路是 60s **软**上限(silero `max_speech_duration` 不硬切,zh 链路实测过冲到 27.65s 的案例),软上限与天花板之间缓冲很薄——连续语音切出 >~65s 的段会 OOM **报错**(单段失败,服务不挂)。建议后续给 other 链路加 zh 同款硬切,或把 other 的 `VAD_MAX_SPEECH` 降到 ~55s。
- **CPU-only 构建链接断裂(遗留债务)**:从 `c95b3d8e` 起 `ort-env.h` 的 raw-allocator 注册,到本轮 `offline-qwen3-asr-model.cc` 的 cudaMemset/cudaMemcpy/cudaDeviceSynchronize,extern "C" CUDA 符号都是**无条件引用**,非 GPU 构建(无 libcudart)会链接失败。修法:`#if SHERPA_ONNX_ENABLE_GPU` 包裹或 dlopen 运行时解析。
- **audio_features 残余 H2D**:每步 ~6MB(TrimAudioFeatures 在 CPU 扫有效帧),要消掉需要把 trim 逻辑搬上 GPU 或容忍 padding,收益小,未做。
- **CPU 4 线程 vs CUDA 的反转弧**:调查期 codeswitch 上 CPU 4t(1.34s)比 int8+cuda(3.99s)快 ~3×;fp32 后 CUDA 反超(RTF 0.37 vs CPU 0.45 量级),CUDA 链路自此全面占优。
- **TU116 上 fp16/int8 GEMM 无硬件红利**:与 zh 链路同结论(无 tensor core),fp32 就是终点,不要再试 fp16 decoder 或 int8 权重(zh 侧已全部证伪,见 fireredasr-cuda-optimization.md §7)。

## 7. 复现指引

全部在远程机 kn@100.64.0.2:

- **导出 fp32 三件套**:conda env `qwen3-asr`,`cd ~/qwen3-work/export-tool && python export_qwen3_asr_onnx.py --model ~/qwen3-work/Qwen3-ASR-0.6B --outdir ~/qwen3-work/export_fp32 --max-total-len 2048 --no-int8 --verify`
- **签名比对**:`~/qwen3-work/dump_signature.py`(int8 基准存档 `qwen3_int8_signature.json`)
- **直解验证**(独立进程,绕开服务):`~/qwen3-work/qwen3_fp32_direct.py <wav> <model_dir> cuda 1 <out.txt>`;边界测试 `~/qwen3-work/qwen3_boundary.py`(debug=True 可看 `adaptive KV alloc_len` 日志)。必须带:
  ```bash
  SITE=$(venv/bin/python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/curand/lib"
  export SHERPA_ONNX_CUDA_USE_ARENA=0
  ```
- **生产 A/B**(~/firered-work/):`run_qwen3_baseline.py`(codeswitch ×3)、`run_qwen3_en90.py`(en_90s ×2)、`run_qwen3_long.py`(12.5min ×1)、`run_ab.py`(zh 回归红线 long_zh_3min ×3,md5 必须 `8bd84d39…`)。脚本自带 nvidia-smi 0.5s 采样报峰值
- **fork 构建/安装**:远端 `~/Documents/github/sherpa-onnx-fireredasr/build-gpu && make -j8 _sherpa_onnx`,产物 cp 到 `venv/lib/python3.13/site-packages/sherpa_onnx/lib/`(**是 lib/ 子目录,不是包根**);venv 备份点 `sherpa_onnx.bak_qwen3kv`(Phase 1)/ `sherpa_onnx.bak_adaptivekv`(Phase 2)
- **服务回滚 other 链路**:`cp app/server.py.bak_qwen3int8 app/server.py` + 重启;int8 三件套已于 2026-09-18 磁盘清理中删除,回滚需重新下载官方发布包(csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25)到 `models/other/`;`.so` 回滚用 `sherpa_onnx.bak_adaptivekv/lib/` 覆盖
