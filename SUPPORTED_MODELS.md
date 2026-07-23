# sherpa-onnx 支持的模型清单

> 本清单依据 `sherpa-onnx/csrc/` 源码核实(各功能的 `*-model-config.h` 聚合结构与 `Create()` 工厂分支)。
> 每个模型对应 `csrc/` 中一组 `*-model.cc/.h` + config,并在相应功能的工厂里占一个分支。
> 具体预训练权重、支持语言与用法以官方文档为准:https://k2-fsa.github.io/sherpa/onnx/

所有模型默认通过 **ONNX Runtime** 推理;部分模型另有 NPU 原生后端(见文末)。

---

## 一、语音识别 ASR

### 非流式(整段音频一次解码,精度通常更高)

来源:`offline-model-config.h`

| 模型 | 架构/来源 | 说明 |
|---|---|---|
| **Transducer** | Zipformer,k2-fsa/icefall | 中英等多语言均有强模型,支持 int8 量化 |
| **Paraformer** | 阿里 FunASR | 非自回归,中文精度高、速度快 |
| **SenseVoice** | 阿里 | 多语种(中/英/日/韩/粤),非自回归,小而快 |
| **Whisper** | OpenAI | 多语种精度标杆,tiny→large 多档 |
| **Moonshine** | 轻量英文 | 面向端侧的小模型 |
| **Dolphin** | 东亚/南亚多语种及方言 | — |
| **Canary** | NVIDIA NeMo | 多语种、多任务(识别+翻译) |
| **FireRedASR** / **FireRedASR-CTC** | 小红书 | 中文高精度(AED 版较重) |
| **NeMo CTC** | NVIDIA NeMo(EncDec CTC) | — |
| **Zipformer-CTC** | k2-fsa | CTC 解码分支 |
| **TDNN** | icefall | 传统小模型(如 yesno) |
| **WeNet-CTC** | WeNet 社区 | — |
| **Cohere Transcribe** | Cohere | — |
| **Omnilingual ASR CTC** | Meta | 超多语种 |
| **FunASR-Nano** | FunASR 系 | 轻量 |
| **MedASR-CTC** | 医疗领域 | — |

### 流式(实时逐块解码,低延迟)

来源:`online-model-config.h`

| 模型 | 架构/来源 | 说明 |
|---|---|---|
| **Transducer** | 流式 Zipformer / NeMo transducer | 流式主力,精度与延迟平衡好 |
| **Paraformer** | 阿里 FunASR | 流式版 |
| **Zipformer2-CTC** | k2-fsa | — |
| **WeNet-CTC** | WeNet 社区 | — |
| **NeMo-CTC** | NVIDIA NeMo | — |
| **T-one CTC** | 流式 CTC | — |

---

## 二、语音合成 TTS

来源:`offline-tts-model-config.h`

| 模型 | 说明 |
|---|---|
| **VITS** | 通用架构,涵盖 Piper、MeloTTS 等多语言模型 |
| **Matcha** | Matcha-TTS,flow-matching 声学模型 |
| **Kokoro** | 多语种,音质高 |
| **ZipVoice** | k2-fsa,zero-shot 音色克隆 |
| **Kitten** | 超轻量 |
| **Pocket** | 轻量 |
| **Supertonic** | — |

---

## 三、其他功能

| 功能 | 支持的模型 | 源码线索 |
|---|---|---|
| **VAD 语音活动检测** | Silero-VAD、TEN-VAD | `silero-vad-model.*`、`ten-vad-model.*` |
| **说话人 embedding**(识别/验证) | 通用(WeSpeaker / 3D-Speaker)、NeMo(TitaNet) | `speaker-embedding-extractor-{general,nemo}-impl.h` |
| **说话人分离 diarization** | Pyannote 分段 + 快速聚类 | `offline-speaker-segmentation-pyannote-model.h`、`fast-clustering.h` |
| **关键词唤醒 KWS** | Transducer(Zipformer) | `keyword-spotter-transducer-impl.h` |
| **语种识别 SLID** | Whisper | `spoken-language-identification-whisper-impl.h` |
| **音频标注** | CED、Zipformer | `audio-tagging-{ced,zipformer}-impl.h` |
| **标点恢复** | CT-Transformer(离线)、CNN-BiLSTM(流式) | `offline-punctuation-ct-transformer-impl.h`、`online-punctuation-cnn-bilstm-impl.h` |
| **语音增强/降噪** | GTCRN、DPDFNet(均有离线+流式版) | `*-speech-denoiser-{gtcrn,dpdfnet}-impl.h` |
| **音源分离** | Spleeter、UVR(MDX) | `offline-source-separation-{spleeter,uvr}-impl.h` |
| **变音符号还原**(阿拉伯语) | CATT | `offline-diacritization-catt-impl.h` |

---

## 四、推理后端覆盖

- **ONNX Runtime**(默认):覆盖上述全部模型。通过 `provider` 选择执行后端:`cpu`(默认)、`cuda`、`coreml`、`xnnpack`、`nnapi`、`trt`、`directml`、`spacemit`。
- **NPU 原生后端**(非 ONNX,编译开关启用,仅覆盖部分模型):
  - **RKNN**(Rockchip):SenseVoice、Paraformer、Zipformer transducer/CTC
  - **Axera / AXCL**:SenseVoice
  - **QNN**(Qualcomm):Zipformer transducer / NeMo transducer

---

## 五、选型推荐:iOS 中文语音识别(整段转写,精度优先)

> 场景:iOS 平台、**非流式整段转写**、**中文为主**、**精度优先**(可接受较大模型/较慢速度)。
> 下表三个模型均已在离线识别工厂 `offline-recognizer-impl.cc` 注册,且 C API 暴露配置(iOS/Swift 可调)。

| 优先级 | 模型 | 定位 | 项目内成熟度 |
|---|---|---|---|
| **起步基线(推荐先用)** | **Paraformer-large** | 非自回归,中文精度高、速度快一个量级,iOS 上从容,有 int8 量化版 | ✅ 有官方导出脚本 `scripts/paraformer/` + WASM demo,拿来即用 |
| **追极致中文精度** | **FireRedASR-AED** | 中文精度天花板级;大模型、自回归解码较慢 | ⚠️ 引擎/C API 支持确凿(`offline-recognizer-fire-red-asr-impl.h`、`c-api.h:902`),但仓库内**无导出脚本**,预训练模型需从官方渠道获取 |
| **需多语种时** | **SenseVoice** | 中/英/日/韩/粤,非自回归,极快;纯中文不如上两者 | ✅ 有官方导出脚本 `scripts/sense-voice/` + WASM demo |

**建议路径**:先用 **Paraformer-large** 跑通 iOS 全流程做基线(生态最成熟、最快见效);若中文精度仍不满足,再换 **FireRedASR-AED**,识别链路可直接复用。

**精度加分项(整段转写场景收益大,强烈建议叠加)**:
- **VAD**(Silero-VAD)先切分长音频再送识别 → 降内存峰值 + 常提升长音频准确率
- **标点恢复**(CT-Transformer)→ 输出带标点的可读文本
- **逆文本正则化 / 热词** → 数字规整、注入领域词汇

**iOS 现实约束**:FireRedASR-AED、Whisper-large 等大模型在 iPhone 上内存/延迟吃紧,适合"录一段等几秒出结果"的交互,不建议对超长音频一次性喂入;Paraformer 在设备上从容得多。集成参考仓库 `build-ios.sh`、`ios-swift/`、`ios-swiftui/`。
