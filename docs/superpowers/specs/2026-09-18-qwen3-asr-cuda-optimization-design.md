# Qwen3-ASR(other 链路)CUDA 优化设计

> 日期:2026-09-18 · 状态:设计已批准 · 平台:Linux x86_64 + NVIDIA GTX 1660 SUPER 6GB(TU116,无 tensor core)
>
> 前置阅读:`docs/fireredasr-cuda-optimization.md`(fireredasr 全部优化结论与教训,本文的借鉴来源)。

## 1. 目标与约束

- **对象**:asr-service(`kn@100.64.0.2:/home/kn/asr-service`)`lang=other` 链路的 sherpa-onnx ONNX 版 Qwen3-ASR-0.6B(conv_frontend + encoder.int8 + decoder.int8,29 语种 + 中文方言)。
- **约束**:识别文本零变化(验收规则见 §6)、每阶段独立可回滚、改动分阶段上线。
- **明确排除**:官方 PyTorch/vLLM 链路(用户已决策,sm_75 对 vllm/flashinfer 支持存疑);fp16 路线(TU116 无 tensor core,cublas fp16 GEMM 比 fp32 慢 ~6×,fireredasr 实测死局);批量 DecodeStreams(Phase 3 后再评估)。

## 2. 现状(2026-09-18 调查,全部有实测/代码证据)

- 基线:codeswitch.wav 6.23s → RTF **0.709**;en_90s.wav 90s → RTF **0.815**(生产 jobs 接口,lang=other,3/2 次文本 md5 逐字节一致)。基线文本存 `kn@100.64.0.2:~/firered-work/qwen3_codeswitch_baseline.txt`、`qwen3_en90s_baseline.txt`。
- **int8 是分离式动态量化**(与 fireredasr 同病):encoder.int8.onnx = 110 MatMulInteger + 74 DynamicQuantizeLinear + 122 Cast;decoder.int8.onnx = 8016 节点,含 197 MatMulInteger + 113 DQL + 856 Cast。这些算子 CUDA EP 无 kernel → CPU/CUDA 混合分区逐层乒乓。
- 实测印证:codeswitch 直解 **cuda 3.99s vs cpu/4线程 1.34s(CPU 快 3×)**;en_90s 直解 cpu/4 68.6s ≈ server cuda 73s。
- **代码级热点**:`offline-qwen3-asr-model.cc` 的 KV cache 用 `Ort::AllocatorWithDefaultOptions` 分配在 **CPU 内存**(28 层 × (K,V) × [batch, 2048, 8, 128] fp32 ≈ **470MB/流**),`ForwardLLM` 每步自回归 BindInput 全量 H2D、输出绑回 CPU,`ApplyKvDeltaInplace` 纯 CPU memcpy。en_90s ≈ 450 步 × 470MB ≈ 210GB H2D。
- `DecodeStreams` 无批量(`offline-recognizer-qwen3-asr-impl.cc:1026-1031` 逐条 Decode 循环)。
- 显存余量足:int8 链路加载驻留 442 MiB,解码峰值 ~1088 MiB(KV 在 CPU 内存)。fp32 模型 ~2.6GB + GPU KV 470MB/流,6GB 卡安全。
- 禁 arena(`SHERPA_ONNX_CUDA_USE_ARENA=0`)是进程级开关,qwen3 链路已自动受益,无需再做。
- 采样默认 `temperature=1e-6`(greedy),确定性解码,文本对比可行。注意:对比必须走同一管线(VAD 分段 vs 直解文本不同;cuda vs cpu 文本也不同),同配置重复跑完全一致。

## 3. Phase 1 — fp32 重导出(零代码,只换模型文件)

- 用远端 `~/Documents/github/Qwen3-ASR`(已 clone)+ `qwen3-asr` conda env(torch 2.9.1 / transformers 4.57.6),从 ModelScope(优先,远端网络差)或 HuggingFace 拉 `Qwen/Qwen3-ASR-0.6B` 权重(~1.2GB)。
- 导出参照社区脚本 `Wasser1462/Qwen3-ASR-onnx`(export_qwen3_asr_onnx.py),**严格对齐 sherpa-onnx int8 版的 I/O 签名**:conv_frontend.onnx / encoder.onnx / decoder.onnx 三件套;decoder 的 28 层 KV `[batch, 2048, 8, 128]` fp32 输入/增量输出、动态 batch 维;`server.py` 只改文件名。
- 顺手做图瘦身(fireredasr 经验:export-safe wrapper,decoder 8016 节点压缩空间大),但数值正确优先,瘦身为可选项。
- 验收:encoder 输出与 int8 版 cosine 相似度;测试语料端到端文本对比;RTF A/B。

## 4. Phase 2 — fork C++ GPU 驻留 KV(消 470MB/步 H2D)

- 改 fork `sherpa-onnx/csrc/offline-qwen3-asr-model.{h,cc}`(及 impl 如有需要):
  - 28 层 KV cache 分配到 CUDA 显存(session 的 CUDA allocator);
  - 每步 decoder 输出的 KV 增量直接绑到 device 侧 cache 对应偏移(batch=1 时 seq 维连续,`Ort::Value::CreateTensor` 包 device 指针 + offset 可零拷贝),或以轻量 D2D 写回替代 `ApplyKvDeltaInplace` 的 CPU memcpy;
  - encoder 输出 / audio_features 保持 device 驻留,不再回拷 CPU。
- 模板:fork 内 FireRedASR 的 IOBinding GPU-resident 写法(上游 `711acccd`)。
- 改动只触 qwen3-asr model 类,其他模型路径零影响;CPU provider 路径保持原行为。

## 5. Phase 3 — adaptive KV 分配

- 每流按 `audio_tokens + prompt + max_new_tokens` 估算分配 KV 长度,替代固定 2048(fireredasr 同款,那边单项 2×;此处收益待测)。
- 副作用收益:单流显存下降,为将来批量留空间。

## 6. 测试与验收

- 每阶段:`~/firered-work/run_qwen3_baseline.py` / `run_qwen3_en90.py` / `bench_qwen3_direct.py` 风格 A/B(RTF 中位 + nvidia-smi 峰值采样)+ 文本 md5 对比。
- 测试语料:codeswitch(6.2s)+ en_90s(90s)+ 补 2~3 条不同语种/长度(含 1 条长音频)。
- **文本验收规则**:fp32 链路 vs int8 基线逐字节一致为通过;若个别 token 分叉(fp32 数值更贴近原始模型,greedy 可能分叉),逐条列出差异交用户裁决,不擅自放行。
- 显存红线:峰值 ≤ ~5GB(ModelManager 单模型驻留,zh/other 不同时加载)。
- 不承诺具体 RTF 目标值,每阶段实测汇报。

## 7. 回滚

- Phase 1:int8 模型文件保留,`server.py` 一行回退(参照 fireredasr 的 `server.py.bak_*` 惯例)。
- Phase 2/3:fork git 分支隔离(`fireredasr-batch-decoding` 上提交),失败则 revert + 重装 venv 包。

## 8. 风险

| 风险 | 应对 |
|---|---|
| int8→fp32 文本分叉 | §6 验收规则,逐条上报用户裁决 |
| 导出脚本与 sherpa 签名不匹配 | 以 int8 版 onnx 图的 I/O 名为准逐一对齐;先小样本数值验证再上线 |
| HF 权重下载慢 | 优先 ModelScope 镜像;必要时 Mac 本地下载后 scp |
| GPU 驻留 KV 引入显存增长 | 470MB/流可预期;Phase 3 进一步下降;峰值采样监控 |
