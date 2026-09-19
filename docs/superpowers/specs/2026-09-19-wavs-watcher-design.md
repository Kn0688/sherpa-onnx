# wavs 目录监听识别(watcher)设计

> 日期:2026-09-19 · 状态:设计已批准 · 目标机:kn@100.64.0.2:/home/kn/asr-service

## 1. 目标与约束

- `asr-service/wavs/` 目录会被不定时写入音频文件,需要一个长期运行的 watcher 自动识别其中的音频。
- 约束:生产服务(app/server.py)零改动;watcher 可独立启停;去重状态持久化且**不放 wavs 目录**;源音频保留不动;只扫描 wavs 顶层,**不递归**。

## 2. 需求(用户已确认)

- **语种**:取文件名第一个 `_` 之前的前缀作为 lang(如 `zh_001.m4a`、`en_002.wav`);别名 `cn→zh`;合法集合 {zh, en, other, auto},不认识的按 `auto` 处理并记 warning。
- **去重**:文件名 + 内容 md5 双键去重,state 文件放 wavs 之外。
- **输出**:识别完成后,把结果从 jobs 体系拷贝/写入 `wavs/results/{原文件名去扩展名}.txt`。
- **源文件**:识别后保留。
- **架构**:独立 watcher 进程,通过本机 HTTP 调 jobs API。

## 3. 架构

- 新增 `app/watcher.py`(单文件,标准库 + `requests`/venv 已有依赖),`scripts/run_watcher.sh` 以 nohup 模式启动(与 `scripts/run.sh` 同款),日志 `logs/watcher.log`。
- 每 10s 轮询 `wavs/` 顶层,通过 `http://127.0.0.1:8000` 的 jobs API(POST /jobs → 轮询 GET /jobs/{id} → 取结果)串行处理,一次一个 job(匹配服务单 worker、单模型驻留的现状)。

## 4. 处理流程

1. **发现**:listdir 顶层;过滤音频扩展名(.wav/.m4a/.mp3/.flac/.ogg/.aac/.opus,大小写不敏感);跳过子目录(含 results/);文件须连续两次扫描 size+mtime 不变才视为写完(防半成品被识别)。
2. **语种解析**:按 §2 规则;无 `_` 的文件名按 `auto` + warning。
3. **去重**(state:`asr-service/watcher/state.json`):
   - 结构:`{"files": {name: {"md5": ..., "status": done|failed|processing, "job_id": ..., "ts": ..., "error": ...}}, "hashes": {md5: first_name}}`;每次状态变更后原子落盘(写临时文件 + rename)。
   - 同名且 md5 相同 → 跳过;同名但 md5 变化(覆盖写入)→ 重新识别。
   - 不同名但 md5 相同 → 不重复识别,把首个文件的结果文本拷成新名字的 result(并登记 state)。
4. **执行**:POST /jobs(lang 来自前缀)→ 轮询至 done/failed → 成功则结果文本写入 `wavs/results/{stem}.txt`(results 目录自动创建);失败则 state 记 failed + error。
5. 串行队列:发现顺序按文件名排序处理;单个文件失败不阻塞后续。

## 5. 错误处理

- 服务不可达(connection refused 等):等待 10s 重试,watcher 不退出。
- job failed:state 持久标记,重启不会反复重投;文件 md5 变化才重投。
- watcher 崩溃/被杀:重启后从 state 恢复,status=processing 的条目按文件名+md5 重新判定(结果文件已存在则直接补登记)。
- state.json 损坏:备份为 state.json.bad 后从空 state 重建(已存在的 results/*.txt 不会被重投——启动时把已有 result 对应的文件名标记 done,md5 现算)。

## 6. 测试

- **函数级**(watcher 自带的自测函数或独立脚本):前缀解析(zh/cn/en/other/auto/未知/无下划线);dedup state 读写与原子落盘;同 md5 不同名的结果复用;覆盖写入(md5 变)触发重识别。
- **端到端**(生产机,真实服务):
  1. 投放 `zh_*.wav` / `en_*.wav` / `cn_*.wav` 小文件,验证 results/*.txt 产出且语种路由正确;
  2. 重复投放同名文件 → 跳过;改名同内容文件 → 不重复识别但 result 拷出;
  3. kill watcher 重启 → state 恢复,不重复处理;
  4. 服务重启场景 → watcher 等待恢复后继续;
  5. 半成品文件(投放过程中扫描)→ 等写完才识别。
- 回归:zh 链路红线不受影响(watcher 只是 HTTP 客户端)。

## 7. 部署与运维

- 启动:`cd /home/kn/asr-service && nohup bash scripts/run_watcher.sh > logs/watcher.log 2>&1 < /dev/null &`
- 停止:`pgrep -f "watcher.py" | head -1 | xargs -r kill`(与服务的停法同款纪律,不用 pkill -f)。
- README 增加 watcher 章节(用途/启停/state 位置/去重规则/结果目录)。
