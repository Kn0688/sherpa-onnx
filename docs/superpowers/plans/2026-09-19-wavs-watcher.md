# wavs 目录监听识别(watcher)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 asr-service 增加一个长期运行的 watcher 进程:轮询 `wavs/` 顶层音频,按文件名前缀路由语种,调本机 jobs API 识别,结果写 `wavs/results/`,文件名+内容 md5 双键去重,重启可恢复。

**Architecture:** 独立 Python 进程(`app/watcher.py`,生产 server.py 零改动),HTTP 客户端串行处理(一次一个 job);state 持久化在 `asr-service/watcher/state.json`(不放 wavs);`scripts/run_watcher.sh` nohup 启动。

**Tech Stack:** 远端 kn@100.64.0.2:/home/kn/asr-service,venv Python 3.13(已有 fastapi/uvicorn/numpy/soundfile;**requests 有无需在 Task 1 确认,标准是只用 stdlib urllib**)。设计文档:`docs/superpowers/specs/2026-09-19-wavs-watcher-design.md`。

**生产 API 契约(已从 server.py 核实,勿改动服务端):**
- `POST /jobs`:multipart form,字段 `file`(二进制)、`lang`(zh|en|other|auto)、`with_timestamps`(默认 True)、`punctuate`(默认 True)→ `202 {"job_id": "...", "status": "..."}`;解码失败同步 400
- `GET /jobs/{job_id}` → JobStatus:`status` ∈ pending/processing/done/failed/cancelled;完成后 `result: {"text": str, "lang": str, "duration": float, "rtf": float, "segments": [...]}`,失败时 `error: str`
- job id 是建议值(服务端自分配 `job-NNNNN`)

**远端纪律:** SSH 断连 sleep 20-25s 重试 ≤12 次;停服务 `pgrep -f "uvicorn server:app" | head -1 | xargs -r kill`(绝不用 pkill -f);启动 `cd /home/kn/asr-service && nohup bash scripts/run.sh 8000 > server.log 2>&1 < /dev/null &`;服务在跑时**不需要**为 watcher 重启它。

---

### Task 1: watcher.py 实现(本地 Mac 写码 → 部署远端)

**Files:**
- Create(远端): `/home/kn/asr-service/app/watcher.py`
- Create(远端): `/home/kn/asr-service/watcher/`(目录,state 落盘处)
- 本地暂存: `/Users/feiguodong1/Documents/github/temp/watcher.py`(scp 前在这里写)

- [ ] **Step 1: 环境确认**

```bash
ssh kn@100.64.0.2 '/home/kn/asr-service/venv/bin/python -c "import requests" 2>&1; echo rc=$?'
```
`requests` 有则用之,无则 stdlib `urllib.request`(multipart POST 手写 ~20 行)。把结论记进实现注释。

- [ ] **Step 2: 实现 watcher.py**

单文件,结构(全部函数可独立 import 供测试):

```python
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".opus"}
VALID_LANGS = {"zh", "en", "other", "auto"}
LANG_ALIASES = {"cn": "zh"}
```

1. `parse_lang(filename) -> str`:stem 第一个 `_` 之前小写化;查别名表;不在 VALID_LANGS → 返回 `"auto"`(调用处记 warning)。无 `_` → `"auto"`。
2. `md5_of(path, chunk=1MB) -> str`:流式读。
3. `DedupState(dir_path)`:加载/保存 `state.json`(tmp+rename 原子写);结构 `{"files": {name: {"md5": str, "status": "done|failed|processing", "job_id": str|None, "ts": float, "error": str|None}}, "hashes": {md5: first_done_name}}`;方法:
   - `decide(name, md5) -> "skip"|"recognize"|"copy"`:files 里同名且 md5 同 & status done/failed → skip(failed 也 skip,避免循环重投);同名 md5 变 → recognize(覆盖写);不同名但 md5 在 hashes → copy。
   - `mark_processing(name, md5)` / `mark_done(name, md5, job_id)`(同时登记 hashes[md5]=name)/ `mark_failed(name, md5, error)`。
   - 启动恢复:`recover(results_dir)`:把 `results/*.txt` 对应 stem 已存在但 state 无记录的文件标 done(md5 现算);`status=processing` 的条目若对应 result 已存在则改 done,否则改 failed("watcher restart")。
4. `stable_file(path) -> bool`:间隔 2s 读两次 `os.stat().st_size` 与 `st_mtime_ns`,相等为真。主循环里改为"本轮快照在下一轮仍不变才入队"(低频天然防抖),极简实现:扫描时记录 `(name, size, mtime_ns)`,入队条件 = 与 state 里记录的候选项连续两轮一致。
5. `submit_job(api, path, lang)`:按 Step 1 结论用 requests 或 urllib;`data = open(path,'rb').read()`;lang 原样传(`auto` 传给服务端,服务端 `_check_lang` 会转 other)。
6. `run_once(root, api, state)`:扫 `root/wavs/` 顶层 `os.scandir`(跳过目录/非音频扩展/以 `.` 开头);按 name 排序;对每个候选:stable 两轮 → parse_lang → md5_of → state.decide → skip/copy/recognize 分支;recognize 分支串行执行(提交→轮询→写结果),完成一个再处理下一个。
   - copy 分支:找 `results/{hashes首个stem}.txt`,复制为当前 stem 的 result;源 result 缺失则降级为 recognize 并把 hashes 条目清掉。
7. `main()`:env `WAVS_DIR`(默认 `ROOT/wavs`)、`ASR_API`(默认 `http://127.0.0.1:8000`)、`WATCH_INTERVAL`(默认 10s);循环 `run_once`;`requests.ConnectionError/urllib.error.URLError` → log + sleep 重试;其他单文件异常 → try/except 记 log + mark_failed(可识别为文件级错误),不退出。结果写入 `wavs/results/{stem}.txt`(只写 `result.text`,UTF-8;results 目录 `mkdir(exist_ok=True)`)。
8. 日志:`logging` 到 stdout(由 run_watcher.sh 重定向),格式 `%(asctime)s %(levelname)s %(message)s`。

约束:不用 watchdog 等三方库;atomic 写 result(tmp+rename);整个 watcher 单线程。

- [ ] **Step 3: 本地语法/import 自检**

```bash
python3 -m py_compile /Users/feiguodong1/Documents/github/temp/watcher.py
python3 - <<'EOF'
import sys; sys.path.insert(0, "/tmp")
import importlib.util
spec = importlib.util.spec_from_file_location("watcher", "/Users/feiguodong1/Documents/github/temp/watcher.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.parse_lang("zh_1.wav") == "zh" and m.parse_lang("cn_x.m4a") == "zh"
assert m.parse_lang("EN_a.mp3") == "en" and m.parse_lang("other_9.flac") == "other"
assert m.parse_lang("noprefix.wav") == "auto" and m.parse_lang("fr_2.wav") == "auto"
s = m.DedupState("/tmp/watcher_selftest")
import pathlib; pathlib.Path("/tmp/watcher_selftest").mkdir(exist_ok=True)
pathlib.Path("/tmp/watcher_selftest/a.bin").write_bytes(b"hello")
h = m.md5_of("/tmp/watcher_selftest/a.bin")
assert s.decide("a.bin", h) == "recognize"          # 首见 → 识别
s.mark_done("a.bin", h, "job-x")
assert s.decide("a.bin", h) == "skip"               # 同名同值 → 跳过
assert s.decide("b.bin", h) == "copy"               # 异名同内容 → 复制结果
assert s.decide("a.bin", h[:-1] + "0") == "recognize"  # 内容变了 → 重识别
print("selftest ok")
EOF
```
(自检代码按实现的真实语义微调,但必须覆盖 6 个前缀用例 + state 基本读写。)

- [ ] **Step 4: 部署到远端**

```bash
scp /Users/feiguodong1/Documents/github/temp/watcher.py kn@100.64.0.2:/home/kn/asr-service/app/watcher.py
ssh kn@100.64.0.2 'mkdir -p /home/kn/asr-service/watcher && /home/kn/asr-service/venv/bin/python -m py_compile /home/kn/asr-service/app/watcher.py'
```

- [ ] **Step 5: 函数级单测(远端,独立进程,不碰服务)**

在远端建 `~/watcher-test/test_watcher.py`,用临时目录构造:前后缀解析表、md5 双键(skip/copy/覆盖重识别三条路径)、recover(results 已存在补登记 + processing→failed)、state 原子落盘(读回校验 JSON)。用 venv python `-m pytest` 或裸 assert 脚本跑通。期望全绿。

---

### Task 2: run_watcher.sh + 启动

**Files:**
- Create(远端): `/home/kn/asr-service/scripts/run_watcher.sh`

- [ ] **Step 1: 写脚本(仿 run.sh 风格)**

```bash
#!/usr/bin/env bash
# 启动 wavs  watcher: venv + uvicorn 无关, 纯 python 循环
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/venv/bin/activate"
mkdir -p "$ROOT/logs" "$ROOT/watcher"
exec python "$ROOT/app/watcher.py"
```

- [ ] **Step 2: 启动并让 watcher 跑空一轮**

```bash
ssh kn@100.64.0.2 'cd /home/kn/asr-service && nohup bash scripts/run_watcher.sh > logs/watcher.log 2>&1 < /dev/null & sleep 3 && tail -5 logs/watcher.log && pgrep -af watcher.py'
```
Expected: 日志有启动行;进程在列。

---

### Task 3: 端到端验收(远端,真实服务)

**Files:** 无新增代码;测试产物放 `~/watcher-test/`

- [ ] **Step 1: 准备测试音频**(复用现成小文件:`models/other/test_wavs/codeswitch.wav`(en/other)、`data/` 里挑个小的中文片段(如从 long_zh_3min 截取 5s,ffmpeg 或 soxr 现做);造 3 个投放文件:`zh_t1.wav`(中文)、`en_t2.wav`(codeswitch 拷一份)、`cn_t3.wav`(同 zh_t1 内容)**

- [ ] **Step 2: 核心验收矩阵**(每步验证 watcher.log + results/ 内容)
  1. 投 `zh_t1.wav` → 出 `results/zh_t1.txt`(非空、内容为中文识别结果,lang 走 zh 链路——从 server log 或结果文本特征确认);
  2. `cp zh_t1.wav cn_t3.wav` → 不重复识别,`results/cn_t3.txt` 与 `results/zh_t1.txt` 内容一致(copy 分支);
  3. 再投同名 `zh_t1.wav`(未变)→ watcher.log 出现 skip,无新 job(观察服务侧 jobs 列表计数不变);
  4. 修改 `zh_t1.wav` 内容(尾部追加静音或换另一个音频拷过去)→ 重新识别,results 更新;
  5. 投 `en_t2.wav` → `results/en_t2.txt`(前缀 en → 服务端 en/parakeet 链路,文本与该链路特征一致)。

- [ ] **Step 3: 健壮性验收**
  1. kill watcher(`pgrep -f watcher.py | head -1 | xargs -r kill`)→ 重启 → 对已有 results 的文件不重复识别(state recover);
  2. 停服务 → 投新文件 → watcher.log 出现连接重试、不退出 → 启服务 → 任务最终完成;
  3. 半成品:`cp` 一个大文件到 wavs 的同时观察(或分批 `dd` 追加)→ 两轮稳定前不提交。

- [ ] **Step 4: 清理测试文件 + 服务回归**

删除 wavs/ 里的测试投放件与 results 测试产物(保留目录);提交 zh 回归:`run_ab.py` 或手动 POST `data/long_zh_3min.wav` lang=zh,md5 必须 `8bd84d3915be595dcef60024685d6039`。

---

### Task 4: README + 收尾

**Files:**
- Modify(远端): `/home/kn/asr-service/README.md`(母版 `/Users/feiguodong1/Documents/github/temp/asr_README.md` 先改再 scp)
- Modify(fork): `docs/superpowers/plans/2026-09-19-wavs-watcher.md` checkbox 勾选

- [ ] **Step 1: README 加 "wavs 目录监听(watcher)" 章节**:用途、文件命名约定(`{lang}_{任意}.wav`,别名 cn→zh,未知→auto)、去重规则(文件名+md5,state 位置)、结果目录、启停命令、故障排查(watcher.log 看点)。

- [ ] **Step 2: README 部署 + fork 提交推送**

```bash
scp /Users/feiguodong1/Documents/github/temp/asr_README.md kn@100.64.0.2:/home/kn/asr-service/README.md
cd /Users/feiguodong1/Documents/github/deep-dive-inference/sherpa-onnx-batch-wt && git add -A && git commit -m "docs: wavs watcher (plan checked, README section)" && git push origin fireredasr-batch-decoding
```

---

## Self-Review 记录

- Spec 覆盖:§4 流程 1-5 → Task 1 Step 2;§5 错误处理 → Task 1 Step 2.7 + Task 3 Step 3;§6 测试 → Task 1 Step 5 + Task 3;§7 部署 → Task 2 + Task 4;§2 需求(cn 别名/未知→auto、双键去重、state 不在 wavs、results 子目录、源文件保留、不递归)逐条落在 Step 2 契约与验收矩阵。
- 占位符扫描:Step 2 为函数级契约(8 个具名函数 + 语义),刻意不写整文件 full code——watcher 是脚本级新文件,实现者按契约直写;全部接口已具名,语义已固定,无 "TBD/handle edge cases" 类空话。
- 类型一致性:`decide` 返回值集合 skip/recognize/copy 在 §Step 2.3 与 Task 3 验收一致;state JSON 键名 files/hashes 全程一致。
