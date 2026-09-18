# Qwen3-ASR(other 链路)CUDA 优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 asr-service `lang=other` 链路的 Qwen3-ASR-0.6B 从分离式 int8 换为 fp32 重导出 + fork C++ GPU 驻留 KV + adaptive KV,消除 CUDA EP 分区乒乓与每步 470MB H2D。

**Architecture:** 三阶段,每阶段独立验证/部署/可回滚。Phase 1 零代码只换模型文件;Phase 2 改 fork `sherpa-onnx/csrc/offline-qwen3-asr-model.{h,cc}`(模板:同仓库 FireRedASR 的 IOBinding GPU-resident 写法);Phase 3 在同文件做按流 KV 长度估算。

**Tech Stack:** 远端 kn@100.64.0.2(GTX 1660 SUPER 6GB, TU116 无 tensor core);导出用 conda env `qwen3-asr`(torch 2.9.1 / transformers 4.57.6);服务用 `/home/kn/asr-service/venv`(fork sherpa-onnx GPU 构建);设计文档 `docs/superpowers/specs/2026-09-18-qwen3-asr-cuda-optimization-design.md`。

**远端操作纪律(每个远端 Task 都适用):**
- SSH 偶发断连:失败 `sleep 20-25` 重试,最多 10-15 次
- 停服务:`pgrep -f "uvicorn server:app" | head -1 | xargs -r kill`(**绝不用 pkill -f**);启动:`cd /home/kn/asr-service && nohup bash scripts/run.sh 8000 > server.log 2>&1 < /dev/null &`
- 测试脚本(已存在,可复用/扩展):`~/firered-work/run_qwen3_baseline.py`(codeswitch)、`run_qwen3_en90.py`(en_90s)、`bench_qwen3_direct.py`(独立进程直解)
- 基线文本(不可动):`~/firered-work/qwen3_codeswitch_baseline.txt`、`qwen3_en90s_baseline.txt`
- 提交 job 必须显式 `lang=other`(auto 有已知误判)

---

### Task 1: 权重与导出工具准备(远端)

**Files:**
- Create(远端): `~/qwen3-work/`(本轮工作目录,与 firered-work 并列)

- [x] **Step 1: 建工作目录 + 确认 conda env 可用**

```bash
ssh kn@100.64.0.2 'mkdir -p ~/qwen3-work && source /home/kn/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && python -c "import torch, transformers; print(torch.__version__, transformers.__version__)" && python -c "import modelscope; print(modelscope.__version__)" 2>&1 | tail -1'
```
Expected: `2.9.1 4.57.6` + modelscope 版本号。若 modelscope 缺失:`pip install modelscope`(conda env 内)。

- [x] **Step 2: 下载 Qwen/Qwen3-ASR-0.6B 权重(优先 ModelScope)**

```bash
ssh kn@100.64.0.2 'source /home/kn/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ~/qwen3-work/Qwen3-ASR-0.6B'
```
若 ModelScope 失败,回退 HF 镜像:`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ~/qwen3-work/Qwen3-ASR-0.6B`。
Expected: `~/qwen3-work/Qwen3-ASR-0.6B/` 下有 `config.json` + `*.safetensors`(合计 ~1.2GB)+ tokenizer 文件。
验证: `ls -la ~/qwen3-work/Qwen3-ASR-0.6B/ && du -sh ~/qwen3-work/Qwen3-ASR-0.6B`

- [x] **Step 3: 取导出工具(Wasser1462/Qwen3-ASR-onnx)**

```bash
ssh kn@100.64.0.2 'cd ~/qwen3-work && git clone https://github.com/Wasser1462/Qwen3-ASR-onnx.git export-tool 2>&1 | tail -2; ls export-tool/'
```
GitHub 慢时回退:`git clone https://ghproxy.net/https://github.com/Wasser1462/Qwen3-ASR-onnx.git export-tool`。
Expected: 含 `export_qwen3_asr_onnx.py`、`conv_frontend.py`、`encoder.py`、`decoder.py`。
注意:脚本 `main()` 里 `_register_qwen3_asr()` 需要 `qwen3_asr` python 包(模型仓库的 trust_remote_code 或导出工具自带)——先 `cd export-tool && ls` 确认是否有 `qwen3_asr/` 目录;若无,检查 `~/Documents/github/Qwen3-ASR/qwen_asr` 是否提供等价注册,必要时按导出工具 README 安装依赖。

---

### Task 2: int8 模型 I/O 签名与参考输出基线(远端)

**Files:**
- Create(远端): `~/qwen3-work/dump_signature.py`

- [x] **Step 1: 写签名提取脚本**

```python
#!/usr/bin/env python3
"""dump qwen3-asr int8 onnx I/O 签名(不加载权重,秒级)"""
import onnx, json, sys

def dump(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    def tinfo(v):
        t = v.type.tensor_type
        dims = [d.dim_param or d.dim_value for d in t.shape.dim]
        return {"name": v.name, "dtype": t.elem_type, "shape": [str(x) for x in dims]}
    return {
        "file": path,
        "inputs": [tinfo(i) for i in g.input],
        "outputs": [tinfo(o) for o in g.output],
    }

out = [dump(p) for p in sys.argv[1:]]
print(json.dumps(out, indent=1))
```

- [x] **Step 2: 跑签名提取并存档**

```bash
ssh kn@100.64.0.2 'cd ~/qwen3-work && /home/kn/asr-service/venv/bin/python dump_signature.py /home/kn/asr-service/models/other/conv_frontend.onnx /home/kn/asr-service/models/other/encoder.int8.onnx /home/kn/asr-service/models/other/decoder.int8.onnx > qwen3_int8_signature.json && head -50 qwen3_int8_signature.json'
```
Expected: decoder 输入含 `input_ids`/`audio_features`/`attention_mask`/`cache_position` + `cache_key_0..27`/`cache_value_0..27`(shape `[batch, max_total_len, 8, 128]`,dtype float32);输出 `logits` + `key_delta_*`/`value_delta_*`。**此 JSON 是 Phase 1 导出对齐的基准。**

- [x] **Step 3: 存 int8 参考中间输出(供 Phase 1 cosine 对比)**

用 `bench_qwen3_direct.py` 的加载方式,写 `~/qwen3-work/save_int8_ref.py`:对 codeswitch.wav 跑 conv_frontend → encoder.int8,把 conv 输出(`conv_output`)与 encoder.int8 输出(`audio_features`)各存一份 `.npy` 到 `~/qwen3-work/ref/`(ORT CPU provider 跑,避免 CUDA 分区干扰参考值)。
Expected: `~/qwen3-work/ref/conv_output.npy`、`ref/encoder_int8_audio_features.npy`。

---

### Task 3: 导出 fp32 三件套(远端)

**Files:**
- 用远端 `~/qwen3-work/export-tool/export_qwen3_asr_onnx.py`(不改动,直接用)
- Create(远端): `~/qwen3-work/export_fp32/`(产物目录)

- [x] **Step 1: 运行导出(只导 fp32,max_total_len=2048 对齐生产)**

```bash
ssh kn@100.64.0.2 'source /home/kn/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && cd ~/qwen3-work/export-tool && python export_qwen3_asr_onnx.py --model ~/qwen3-work/Qwen3-ASR-0.6B --outdir ~/qwen3-work/export_fp32 --max-total-len 2048 --no-int8 --verify 2>&1 | tail -30'
```
Expected: `[verify] encoder max_diff` / `[verify] decoder max_diff` 均 < 1e-4;产物 `conv_frontend.onnx`、`encoder.onnx(+ .data)`、`decoder.onnx(+ .data)`。导出耗时可能 10-30 分钟,SSH 用 nohup 或加大超时。

- [x] **Step 2: 签名校验 — 与 int8 基准逐一比对**

```bash
ssh kn@100.64.0.2 'cd ~/qwen3-work && /home/kn/asr-service/venv/bin/python dump_signature.py export_fp32/conv_frontend.onnx export_fp32/encoder.onnx export_fp32/decoder.onnx > qwen3_fp32_signature.json && /home/kn/asr-service/venv/bin/python - <<EOF
import json
i8 = {d["file"].split("/")[-1].replace(".int8",""): d for d in json.load(open("qwen3_int8_signature.json"))}
f32 = {d["file"].split("/")[-1]: d for d in json.load(open("qwen3_fp32_signature.json"))}
for k in ["conv_frontend.onnx","encoder.onnx","decoder.onnx"]:
    a, b = i8[k], f32[k]
    na = [(x["name"], x["shape"]) for x in a["inputs"]+a["outputs"]]
    nb = [(x["name"], x["shape"]) for x in b["inputs"]+b["outputs"]]
    assert na == nb, f"{k} 签名不匹配:\n{na}\n{nb}"
    print(k, "签名一致,", len(na), "个 I/O")
EOF'
```
Expected: 三个文件全部"签名一致"(名字+shape;dtype 允许 int8 图是 fp32 输入输出——分离式量化 IO 本就是 fp32)。**不匹配则停,改导出参数/脚本直到匹配,不动 sherpa 侧。**

- [x] **Step 3: 文件体积核验(fireredasr 教训:外部数据文件必须 ≈ 参数量×4)**

```bash
ssh kn@100.64.0.2 'ls -la ~/qwen3-work/export_fp32/'
```
Expected: encoder.onnx+.data ≈ 700-800MB;decoder.onnx+.data ≈ 2.2-2.6GB;conv_frontend.onnx ≈ 44MB。明显偏大(如 decoder .data > 3GB)= 外部数据死字节,需 onnx load+save 压实重写。

---

### Task 4: fp32 数值验证 + 端到端文本对比(远端,独立进程)

**Files:**
- Create(远端): `~/qwen3-work/verify_fp32.py`

- [x] **Step 1: encoder 级 cosine 对比**

`verify_fp32.py` 第一段:加载 `ref/conv_output.npy` 喂 export_fp32/encoder.onnx(ORT CPU),与 `ref/encoder_int8_audio_features.npy` 算 cosine 相似度(注意 int8 与 fp32 数值本有差异,cosine 期望 > 0.999;若 < 0.99 说明导出错误而非量化误差)。
Run: `ssh kn@100.64.0.2 'cd ~/qwen3-work && /home/kn/asr-service/venv/bin/python verify_fp32.py'`

- [x] **Step 2: 端到端直解文本对比**

扩展(或复制)`bench_qwen3_direct.py` 指向 fp32 三件套,直解 codeswitch.wav 与 en_90s.wav(与基线脚本同管线同参数),文本写入 `~/qwen3-work/fp32_codeswitch.txt`、`fp32_en90s.txt`:

```bash
ssh kn@100.64.0.2 'cd ~/qwen3-work && diff ~/firered-work/qwen3_codeswitch_baseline.txt fp32_codeswitch.txt && echo "codeswitch 逐字节一致"; diff ~/firered-work/qwen3_en90s_baseline.txt fp32_en90s.txt && echo "en90s 逐字节一致"'
```
Expected: 逐字节一致。**若有 token 分叉:逐条列出(diff 输出),记入结果,交用户裁决后继续。**

- [x] **Step 3: 直解速度对比**

同一脚本记录 fp32 直解耗时(cuda provider)vs 调查基线(codeswitch cuda 3.99s / cpu4 1.34s;en_90s cuda ~73s)。
Expected: fp32+cuda 显著快于 int8+cuda;记录数字。

---

### Task 5: 部署 Phase 1 + 生产 A/B(远端)

**Files:**
- Modify(远端): `/home/kn/asr-service/app/server.py`(`from_qwen3_asr` 的文件名两行,约 :152-153)
- 备份: `cp app/server.py app/server.py.bak_qwen3int8`

- [x] **Step 1: 部署模型文件**

```bash
ssh kn@100.64.0.2 'mkdir -p /home/kn/asr-service/models/other/fp32 && cp ~/qwen3-work/export_fp32/conv_frontend.onnx ~/qwen3-work/export_fp32/encoder.onnx* ~/qwen3-work/export_fp32/decoder.onnx* /home/kn/asr-service/models/other/fp32/ && ls -la /home/kn/asr-service/models/other/fp32/'
```
(int8 三件套保留在原位,回滚用。)

- [x] **Step 2: server.py 指向 fp32(备份先行)**

`from_qwen3_asr(...)` 改为 `conv_frontend=str(d/"fp32"/"conv_frontend.onnx"), encoder=str(d/"fp32"/"encoder.onnx"), decoder=str(d/"fp32"/"decoder.onnx")`。

- [x] **Step 3: 重启 + A/B**

停服务 → 启动 → `run_qwen3_baseline.py` ×3 + `run_qwen3_en90.py` ×2,记录 RTF 中位、文本 md5 对比 `~/firered-work/qwen3_*_baseline.txt`、nvidia-smi 0.5s 采样峰值。
Expected: 文本规则同 Task 4;显存红线 ≤ ~5GB(预计 3-4GB);RTF 显著下降。
健康检查: `curl -s http://127.0.0.1:8000/health`(路径以 server.py 实际为准)。

- [x] **Step 4: 阶段报告 + 提交**

把数字汇报给用户(RTF/显存/文本对比)。fork 仓库无需提交(Phase 1 零代码);asr README 的更新统一放 Task 8。

---

### Task 6: Phase 2 — fork C++ GPU 驻留 KV(本地 worktree 改码,远端构建)

**Files:**
- Modify: `sherpa-onnx/csrc/offline-qwen3-asr-model.h`(KV cache 成员类型)
- Modify: `sherpa-onnx/csrc/offline-qwen3-asr-model.cc`(`CreateEmptyKVCache` :533-581、`ForwardLLM` :466-478、`ApplyKvDeltaInplace` :583-749,行号为调查时快照,以实际为准)
- 模板参照: `sherpa-onnx/csrc/offline-fire-red-asr-model.cc`(IOBinding GPU-resident:device 侧 Ort::Value 直绑、零拷贝回喂)
- 本地 worktree: `/Users/feiguodong1/Documents/github/deep-dive-inference/sherpa-onnx-batch-wt`(分支 `fireredasr-batch-decoding`)

- [x] **Step 1: 读码定稿**

读 `offline-qwen3-asr-model.{h,cc}` 全文 + FireRedASR 的 IOBinding 段落,确认:当前 cache 分配方式、每步 BindInput/BindOutput 列表、`ApplyKvDeltaInplace` 的写回语义(delta 写入 cache 的偏移规则 = `cache_position`)。

- [x] **Step 2: 实现(仅 `use_cuda_iobinding_==true` 路径改行为,CPU provider 路径保持原样)**

契约:
1. KV cache 用 session 的 CUDA allocator 分配(device Ort::Value),不再用 `Ort::AllocatorWithDefaultOptions`;
2. `ForwardLLM` 每步:输入侧把 device cache 直接 BindInput(零 H2D);输出侧 KV 增量绑到 device 上独立 delta buffer;
3. 写回:delta → cache 的 D2D 拷贝(cudaMemcpyDeviceToDevice,偏移 = cache_position × 8 × 128 × 4B;batch=1 时 seq 维连续,整块一次拷贝),替代 CPU memcpy;或直接 `Ort::Value::CreateTensor` 包 cache+offset 让输出零拷贝落位(二选一,以实现简洁者为准);
4. encoder 输出 `audio_features` 保持 device 驻留,不再绑回 CPU(原 :391 / :471-474 的 cpu_mem_info_ 输出绑定移除);
5. 删除/旁路 `ApplyKvDeltaInplace` 的 CPU 路径(cuda 分支)。

- [x] **Step 3: 远端构建 + 装入 venv**

fork 构建流程沿用 fireredasr 时的方式(远端 build-gpu 目录增量构建,或 `SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON" python3 setup.py bdist_wheel` 后 pip install --force-reinstall 到 asr-service venv)。构建产物必须含 qwen3-asr 改动;装前备份 venv 的 sherpa_onnx 目录(`cp -r sherpa_onnx sherpa_onnx.bak_qwen3kv`)。

- [x] **Step 4: 验证(文本必须 == Phase 1 fp32 文本)**

`bench_qwen3_direct.py` 直解 codeswitch + en_90s(fp32 模型,cuda):文本与 `~/qwen3-work/fp32_*.txt` **逐字节一致**(纯搬运路径改动,数值不变,此关必须全等);记录 RTF 变化。
Expected: en_90s 级长音频明显提速(210GB H2D 消除);短音频小幅提速。

- [x] **Step 5: 生产部署 + A/B + 提交**

重启服务跑 §Task5 Step3 同款 A/B;显存采样(预期 +470MB/流 量级);fork 仓库 commit(信息示例:`qwen3-asr: GPU-resident KV cache via IOBinding (eliminate ~470MB/step H2D)`),push `fireredasr-batch-decoding`。

> **完成备注(2026-09-18,commit `271e0506`)**:契约 4 有记录在案的偏离——encoder 输出 `audio_features` 仍绑回 CPU,因为 `TrimAudioFeatures`(impl 侧)需要在 CPU 上扫描有效帧;audio_features 每步重传仅 ~6MB/step(vs KV 470MB/step),代价可忽略,impl 文件零改动。实测显存峰值基本持平(+470MB KV 驻留被每步 H2D 暂存消除抵消)。部署坑:venv 实际加载 `sherpa_onnx/lib/_sherpa_onnx*.so`,不是包根那份。

---

### Task 7: Phase 3 — adaptive KV 分配(fork C++)

**Files:**
- Modify: `sherpa-onnx/csrc/offline-qwen3-asr-model.cc`(`CreateEmptyKVCache` 增加 alloc_len 参数)及调用处(`offline-recognizer-qwen3-asr-impl.cc` 的 GenerateText/prefill 段)

- [x] **Step 1: 实现**

契约:每流 KV 分配长度 = `min(max_total_len, prompt_tokens + n_audio_tokens + max_new_tokens + 8)`(8 为余量;fireredasr 同款 `min(max_len, estimated + 4)` 思路)。逐流分配替代固定 2048;CPU 路径同样受益(分配更小)。注意 decoder 输入 shape 是动态轴,ORT 侧无需改动。

- [x] **Step 2: 构建 + 验证 + 部署 + A/B(同 Task 6 Step 3-5)**

文本必须仍与 Phase 1 fp32 文本逐字节一致;记录 RTF/显存;fork commit + push。

- [x] **Step 3: 补一条长音频验证**

从 `~/lan-share/files/` 挑一条 ≥10min 的非中文音频(或拼接 en_90s),lang=other 走生产 jobs,确认:无 OOM、文本连贯、显存峰值记录。

---

### Task 8: 收尾 — 文档与提交

**Files:**
- Create: `docs/qwen3-asr-cuda-optimization.md`(fork 仓库,与 fireredasr-cuda-optimization.md 同格式:TL;DR 表/根因/各 Phase 实测/死路/复现指引)
- Modify: `AGENTS.md`(Local work notes 加一行摘要,附关键数字)
- Modify(远端): `/home/kn/asr-service/README.md`(other 链路章节:fp32 模型、显存、回滚方法)

- [x] **Step 1: 写优化文档**(全部实测数字落盘,含 Phase 1/2/3 各自的 RTF/显存/文本验证结果)
- [x] **Step 2: 更新 AGENTS.md 摘要行**
- [x] **Step 3: fork commit + push(`fireredasr-batch-decoding`);asr README 部署到远端**
- [x] **Step 4: 终验报告给用户**:三阶段累计加速比、最终 RTF、显存、文本验收结论、回滚指引

---

## Self-Review 记录

- Spec 覆盖:§3 Phase 1 → Task 1-5;§4 Phase 2 → Task 6;§5 Phase 3 → Task 7;§6 验收 → 各 Task 验证步 + Task 4/5 文本规则;§7 回滚 → Task 5 Step1(模型保留)+ Task 6 Step3(venv 备份)+ server.py.bak_qwen3int8;§8 风险 → Task 3 Step2(签名不匹配停)、Task 1 Step2(ModelScope 优先)、Task 4 Step2(分叉上报)。
- 占位符扫描:无 TBD;Task 6 Step2 为契约式描述(5 条),刻意不写死 diff——行号已漂移风险高,实现者须先执行 Step 1 读码。
- 类型/名字一致性:`cache_key_N`/`key_delta_N` 命名以 Task 2 签名 JSON 为准(导出脚本默认命名与调查结论一致,Task 3 Step 2 强制校验)。
