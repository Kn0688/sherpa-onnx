# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Project overview

sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx) is a local (offline, no network) speech-processing engine built on ONNX Runtime. It provides:

- Speech recognition (streaming + non-streaming ASR)
- Text-to-speech (TTS)
- Speaker diarization / identification / verification
- Voice activity detection (VAD)
- Keyword spotting
- Spoken language identification
- Audio tagging
- Punctuation restoration
- Speech enhancement and source separation

It runs on Linux, macOS, Windows, Android, iOS, HarmonyOS, and WebAssembly, on x64/x86/arm64/arm32/riscv64, and exposes APIs for C++, C, Python, Java, Kotlin, Swift, Go, C#, Rust, Dart, Pascal, and JavaScript (Node.js + WASM).

**The C++ core in `sherpa-onnx/csrc/` is the single source of truth.** Everything else — the C API and all language bindings — wraps that core. A change to behavior almost always starts in `csrc/`.

Current version: `1.13.4` (see `SHERPA_ONNX_VERSION` in the root `CMakeLists.txt`).

## Repository layout

- `sherpa-onnx/csrc/` — C++ core library, CLI binaries, and C++ tests
- `sherpa-onnx/c-api/` — flat C ABI (`c-api.cc/.h`) over the core; the stable exported surface that most bindings build on. `c-api/cxx-api.cc/.h` is an idiomatic C++ RAII wrapper on top of the C API
- `sherpa-onnx/jni/`, `java-api/`, `kotlin-api/`, `python/` (pybind11), `rust/`, `pascal-api/` — language bindings
- `wasm/` — WebAssembly builds (one per feature)
- `cmake/` — dependency-fetching CMake modules (`onnxruntime`, `kaldi-native-fbank`, `kaldi-decoder`, `piper-phonemize`, `portaudio`, etc.)
- `c-api-examples/`, `cxx-api-examples/`, `python-api-examples/`, `go-api-examples/`, `java-api-examples/`, `kotlin-api-examples/`, `nodejs-examples/`, `nodejs-addon-examples/`, `swift-api-examples/`, `rust-api-examples/`, `dotnet-examples/`, `dart-api-examples/`, `flutter-examples/`, `pascal-api-examples/`, `mfc-examples/`, `ffmpeg-examples/`, `tauri-examples/`, `lazarus-examples/` — per-language/per-framework usage examples
- `android/`, `ios-swift/`, `ios-swiftui/`, `flutter/`, `harmony-os/` — mobile/platform projects
- `scripts/<model-family>/` — Python export/conversion tooling that turns upstream checkpoints (icefall, nemo, whisper, piper, vits, kokoro, sense-voice, ...) into the ONNX + metadata format sherpa-onnx consumes. Relevant when debugging why a model won't load — the metadata keys read in `csrc/` must match what these export scripts write
- `toolchains/` — CMake toolchain files for cross-compilation
- `build-*.sh` (repo root) — per-target build scripts, each creating its own `build-<target>/` directory

## Build

The build is CMake-driven (minimum CMake 3.15, C++17). Dependencies (onnxruntime, kaldi-native-fbank, kaldi-decoder, openfst, piper-phonemize, portaudio, etc.) are fetched automatically via `cmake/*.cmake` — no manual install. The correct onnxruntime prebuilt is chosen per platform (`cmake/onnxruntime-*.cmake`).

For a plain desktop build:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TESTS=ON ..
make -j
```

Each cross-compile / platform target has a dedicated `build-*.sh` script at the repo root (e.g. `build-android-arm64-v8a.sh`, `build-ios.sh`, `build-wasm-simd-asr.sh`, `build-swift-macos.sh`). Read the relevant script before building for a platform — it sets the exact CMake flags.

Key CMake options (defaults in parentheses; see top of `CMakeLists.txt`):

- `SHERPA_ONNX_ENABLE_TESTS` (OFF) — build the GoogleTest suite
- `SHERPA_ONNX_ENABLE_PYTHON` (OFF), `SHERPA_ONNX_ENABLE_JNI` (OFF)
- `SHERPA_ONNX_ENABLE_TTS` (ON), `SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION` (ON) — turning these OFF drops whole subsystems and their tests/deps
- `SHERPA_ONNX_ENABLE_GPU` (OFF, CUDA), NPU backends: `SHERPA_ONNX_ENABLE_RKNN`, `_AXERA`, `_AXCL`, `_ASCEND_NPU`, `_QNN`
- `BUILD_SHARED_LIBS` (OFF) — static by default
- `SHERPA_ONNX_ENABLE_WASM*` — WebAssembly variants, one per feature
- `SHERPA_ONNX_ENABLE_PORTAUDIO` (ON) — only used by demo binaries; `sherpa-onnx-core` does not depend on it

Python wheel: `python3 setup.py bdist_wheel` (honors `SHERPA_ONNX_CMAKE_ARGS` / `SHERPA_ONNX_MAKE_ARGS` env vars). Set `SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON"` for the CUDA wheel.

Library targets produced: `sherpa-onnx-core` (static core), `sherpa-onnx-c-api`, `sherpa-onnx-cxx-api`.

CLI tools are the `sherpa-onnx-*.cc` files in `csrc/`, each registered as an `add_executable` in `csrc/CMakeLists.txt` (e.g. `sherpa-onnx-offline`, `sherpa-onnx` streaming, `sherpa-onnx-offline-tts`, `sherpa-onnx-vad`, `sherpa-onnx-microphone`).

## Testing instructions

C++ tests use GoogleTest and are only built when `SHERPA_ONNX_ENABLE_TESTS=ON`. Each `*-test.cc` in `csrc/` becomes its own executable and CTest case.

```bash
cd build
ctest                        # run all tests
ctest -R text-utils-test     # run one test by name
./bin/text-utils-test        # or run the executable directly
```

Test source files are listed explicitly in the `sherpa_onnx_test_srcs` block of `sherpa-onnx/csrc/CMakeLists.txt` — add new `*-test.cc` files there. Each test is registered via the `sherpa_onnx_add_test()` function in the same file.

CI (GitHub Actions, `.github/workflows/`) builds and tests many platform/binding combinations (Linux/macOS/Windows, Android APKs, iOS, WASM, wheels, style checks).

## Code style guidelines

- C++ follows Google style (`.clang-format`, `BasedOnStyle: Google`), enforced in CI by cpplint:
  - `./scripts/check_style_cpplint.sh` — check files from the last commit
  - `./scripts/check_style_cpplint.sh 1` — check uncommitted changes
  - `./scripts/check_style_cpplint.sh 2` — check the whole project
- clang-tidy: `make clang-tidy-check` (or `make check`) against `.clang-tidy`
- Python: flake8, max line length 120 (`.flake8`)

## Architecture notes

### The Impl / factory pattern (most important thing to understand)

Every major feature is a thin public class backed by an abstract `*Impl` with a static `Create()` factory that picks a concrete subclass at runtime based on which model paths are set in the config. This is how one API surface supports dozens of model architectures.

Example — `OfflineRecognizer` (non-streaming ASR):

- `offline-recognizer.h/.cc` — public class, holds a `unique_ptr<OfflineRecognizerImpl>`
- `offline-recognizer-impl.h/.cc` — abstract base + `Create()` dispatch
- `offline-recognizer-<model>-impl.h` — one concrete impl per model family (whisper, paraformer, sense-voice, transducer, moonshine, canary, fire-red-asr, qwen3-asr, dolphin, ...)

`Create()` inspects `config.model_config` (e.g. `if (!config.model_config.whisper.encoder.empty())`) to decide which impl to instantiate. NPU providers (`provider == "rknn"/"axera"/"axcl"`) are branched first and select `*TplImpl<...ModelRknn>` template instantiations, guarded by `#if SHERPA_ONNX_ENABLE_*`.

The same pattern repeats for `OnlineRecognizer` (streaming ASR), `OfflineTts`, `KeywordSpotter`, `OfflinePunctuation`, `AudioTagging`, `OfflineSourceSeparation`, `OfflineSpeakerDiarization`, VAD, etc. When adding support for a new model, you add a new `*-impl.h`, a `*-model.cc/.h` (+ optional `*-model-config.cc/.h` and `*-model-meta-data.h`), and a branch in the relevant `Create()`.

### File naming conventions in csrc/

- `offline-*` = non-streaming (whole audio at once); `online-*` = streaming (chunk by chunk)
- `*-model.cc/.h` = ONNX session wrapper for one architecture (loads the model, runs `Ort::Session`)
- `*-model-config.cc/.h` = config struct + CLI arg registration + validation
- `*-model-meta-data.h` = metadata baked into the ONNX file, read via the `SHERPA_ONNX_READ_META_DATA*` macros in `macros.h`
- `*-decoder*.cc/.h` = decoding strategy (greedy-search, modified-beam-search, ctc-*, etc.), independent of the model
- `*-rknn.*`, `*-qnn.*` etc. = NPU-specific model implementations using a native (non-ONNX) runtime

### Runtime backends

Default backend is ONNX Runtime, selected via the `provider` string (`provider.cc`): `cpu`, `cuda`, `coreml`, `xnnpack`, `nnapi`, `trt`, `directml`, `spacemit`. Separately, several NPU vendors have their own native runtimes (RKNN, QNN, Axera/AXCL, Ascend) compiled in behind `SHERPA_ONNX_ENABLE_*` flags — these only cover a subset of models.

### ASR stream lifecycle

`OfflineStream` / `OnlineStream` carry per-utterance state. Typical flow: `recognizer.CreateStream()` → `stream->AcceptWaveform(sample_rate, samples)` → `recognizer.DecodeStream(s)` → `recognizer.GetResult(s)`. Streaming recognizers add `IsReady()`, `IsEndpoint()`, and `Reset()`. Feature extraction (fbank) is in `features.cc` via kaldi-native-fbank.

## Conventions

- Version lives in `CMakeLists.txt` (`SHERPA_ONNX_VERSION`). Bumping a release touches `CMakeLists.txt`, `CHANGELOG.md`, and `new-release.sh` together (see the comment above the version line)
- Logging/error macros are in `csrc/macros.h`: `SHERPA_ONNX_LOGE(...)`, `SHERPA_ONNX_EXIT(code)`, and the `SHERPA_ONNX_READ_META_DATA*` family for pulling values out of ONNX model metadata
- Model documentation and pretrained models are hosted at https://k2-fsa.github.io/sherpa/onnx/ (referenced throughout error messages and the README)
- `SUPPORTED_MODELS.md` lists the supported model families

## Security considerations

- This project runs fully offline — no network calls at inference time. Do not add telemetry or runtime downloads to core code
- Dependencies are pinned and fetched from fixed URLs with checksums in `cmake/*.cmake`; keep that pattern when adding or upgrading a dependency
- The C API (`c-api/c-api.h`) is the stable ABI boundary: changes there affect every downstream binding, so extend rather than break existing structs/functions
- When enabling GPU/NPU builds, no extra privileges should be required; the build scripts intentionally avoid `sudo`

## Local work notes (2026-07, this checkout only)

- `docs/asr-pipeline.md` — developer walkthrough of the offline/online ASR pipelines (Impl/factory dispatch, stream lifecycle, SenseVoice CTC and zipformer transducer deep dives, common pitfalls)
- `docs/fireredasr-optimization.md` — **the complete FireRedASR optimization report**: baseline analysis, all five optimizations (adaptive cache / batch decode / re-export / int8 / bucketing), measured results, falsified attempts, reproduction guide, methodology lessons
- Branch `fireredasr-batch-decoding` (local + fork, `1cc4d1ad`) — FireRedASR AED work based on `4392c456`:
- **HEADLINE RESULTS (all measured, macOS arm64, threads=2)**: total single-utterance speedup vs official released int8 ≈ **2.9x (10.1s audio: RTF 0.396→0.135) / 3.1x (17.6s: 0.503→0.164)**; layers = adaptive cache 2.0-2.2x × slim-graph export ~1.3-1.5x. Server batching adds 1.2-1.5x (equal-length) on top. Our int8 beats the official int8 everywhere (8%/41%/39% on 10s/5s/17.6s). Correctness: per-utterance 8/8 == fp32, mixed-length batch 8/8. Memory: KV cache 168MB→10-18MB, peak RSS 2.2GB, model 1.62GB (official 1.15GB — bigger because our Conv stays fp32; see QDQ note). On-device (user-measured): adaptive cache alone ≈2x, matches desktop
- **Batch-capable re-export (Phase 0/1/4 done, `/tmp/firered-export/`)**: `scripts/fire-red-asr/export-onnx.py` (committed on `fireredasr-batch-decoding`) re-exports FireRedASR2-AED fp32 with dynamic batch axes + x_len encoder mask. Numerics vs official PyTorch ≤3e-5; mixed-length batch 2/8→7/8. Speed (fp32, threads=2): equal-length batch=4 → 1.28x, batch=8 → ~1.5x; **mixed-length batch is a loss (0.57x) — always sort/bucket by length**. venv + weights + reports in `~/firered-export/` (survives reboot). Findings: official encoder HAS x_len mask (released ONNX lost it); official "KV cache" stores per-layer hidden states, mathematically equivalent to true KV cache; 8k.wav diff vs official was a resampling difference, not int8. **Phase 5 done (`2d47f08e`)**: `quantize_dynamic(MatMul)` int8, 4.64→1.73GB, 8/8 per-utterance identical to fp32. **Phase 2 done (`861fb4c1`)**: encoder `enc_mask` output + decoder `cross_mask` input (additive bias, backward-compatible with released models) — **mixed-length batch 7/8 → 8/8 on both fp32 and int8, all correctness defects closed**; int8 RTF 0.133 (10s) / 0.168 (17.6s), ~5% mask overhead; new `scripts/fire-red-asr/quantize-int8.py` handles .data compaction pitfalls. **Bucketing done (`1cc4d1ad`)** — see Open backlog line below
  - **Graph authoring IS optimization**: our export has 8x fewer decoder nodes than the official one (991 vs 8215; encoder 1331 vs 4249) with identical weights and identical MatMul count. Causes: (1) write export-safe wrapper modules instead of tracing raw forward — vectorized masks (`arange(S) <= offset`, `narrow`/scatter for cache) replace the official per-step Python mask construction (~700 Shape/Unsqueeze/Expand/ConstantOfShape nodes per decode step); (2) the dynamo exporter does DCE/constant-folding that legacy tracing does not; (3) cross-K/V projections moved into the encoder. Why it matters: ORT per-node dispatch + intermediate tensors are a per-step tax in the autoregressive loop — this is why our decoder beats the official int8 by 39% on long audio. Lesson: tracing a raw forward records its inefficiencies; author the graph you want.
  - Batched `DecodeStreams` for FireRedASR AED + `SupportBatch()` runtime fallback. **All currently released FireRedASR AED ONNX models (v1/asr2) cannot batch** — decoder `tokens`/`offset` inputs are fixed to batch=1; encoder does not mask self-attention with `x_len`, so mixed-length batches are corrupted by padding (valid-position cross_k drifts ~35%). True batching needs re-exported models (dynamic axes + attention masks)
  - **Adaptive KV cache allocation** (`GetInitialSelfKVCache` with `alloc_len`): cache sized `min(max_len, estimated_tokens + 4)` instead of full `max_len=1024`. Output is token-identical; decoder is 3.5-5.4x faster, end-to-end RTF 0.342→0.182 (10 s) / 0.495→0.260 (17.6 s) on macOS arm64. This is the single biggest known FireRedASR AED optimization
  - Measured facts: decoder loop was 55-70% of e2e latency before the cache fix; per-step cost ≈ 9 ms fixed (graph rebuilds ~700 mask nodes per step) + 0.075 ms × cache_len; decoder does NOT scale with `num_threads` (memory/ops-bound), encoder scales near-linearly
  - **Batch + adaptive cache combo (measured with patched dynamic decoder)**: turns positive but only ~1.08x for N≥4 (breakeven at N=2). Per-step superlinearity dropped from ~23x to 4-7x, but after the cache fix the encoder is 75-80% of e2e and int8 encoder gains nothing from batch (Amdahl). Server-side batch payoff requires re-export fixing all three: dynamic dims, batch-friendly encoder export (static quant scales or fp32) + padding mask, and the ~700 per-step mask-rebuild nodes; then N=7 could reach 3x+
- Branch `ios-fireredasr-cache-opt` (local only, based on tag `v1.13.4`) — cache-only port of the adaptive KV cache optimization (no batch code), validated RTF 0.396→0.181 / 0.503→0.247 with token-identical output. **This is the branch for iOS on-device validation**; `build-ios/sherpa-onnx.xcframework` + `build-ios/ios-onnxruntime/onnxruntime.xcframework` are already built from it
- Branch `fireredasr-mobile` (fork, `d00c7927`, based on `ios-fireredasr-cache-opt`) — **current on-device working branch**: adaptive cache + enc_mask/cross_mask passthrough (works with released AND batch-capable models) + 8 tokens/s cap, NO batch code. Batch is parked on `fireredasr-batch-decoding` after iOS jetsam kills (bug_type 298) at 3.54GB RSS on long transcriptions — desktop batch peak is only ~1.9GB, so device growth is likely ORT arena retention across varied batch shapes; planned fix if revived: periodic recognizer recreation to flush arena. CPU-watchdog theory (bug_type 202 cpu_resource report, 66%/50%) was ruled out — 1-thread batch also died. PR to upstream: branch `fireredasr-adaptive-kv-cache` (cache-only, minimal)
- Local test artifacts: `build/sherpa-onnx-offline-baseline` (pre-change binary for A/B), `~/firered-test/` (released models + experiment logs; moved off /tmp after a reboot wiped it)
- Rejected/deferred paths (do not re-litigate):
  - **Deferred (parked, not rejected)**: ANE/CoreML offload (parked 2026-07-20 — est. 2-3.4x on top of cache opt but 1-2 weeks effort + decoder stateful risk; revisit when: on-device CPU RTF proves insufficient, or long-form continuous dictation makes thermal throttling a real problem. Encoder-only hybrid (fixed-T export + provider=coreml, decoder on CPU) is the recommended 80/20 shape if revived)
  - **Rejected (measured dead ends)**: int8 Conv via ConvInteger on arm64 (2x slower, no optimized kernel — use QDQ weight-only quantization for Conv instead); xnnpack EP on iOS (user-tested, no benefit); ORT kernel patches; decoder per-step IOBinding on CPU (no-op by design, CUDA path already exists); batch>16 (plateau); lowering the 6 tokens/s decode cap below 6 (truncation risk). NOTE: the cap was later RAISED 6→8 (`48fa8c88`) for fast-speech safety — ~3% e2e cost, texts token-identical
- Open backlog: length bucketing for mixed batches **(done `1cc4d1ad`: DecodeStreams sorts by frames, cuts buckets at ratio 1.2 — mixed batch 0.44x → ~1.05x, 8/8 correct; fuller real-world buckets gain more)**, on-device iOS validation (RTF/memory/thermal)
- QDQ weight-only Conv quantization (**measured NOT worth it**: 8/8 correct but ~4% slower — single encoder forward amortizes weight bandwidth to nothing, dequantize overhead dominates; the "official int8 30% short-utterance edge" is already closed by the slim decoder graph — our int8 is faster than official everywhere: 8%/41%/39% on 10s/5s/17.6s). NOTE: the manual QDQ script's first version had a broadcasting bug (scale reshaped to 4 dims on 3-D conv1d weights → 134GB phantom allocation → jetsam kill, masqueraded as machine memory pressure)
- Local work dirs moved off /tmp after a reboot wiped it: `~/firered-test` (released models + logs), `~/firered-export` (FireRedASR2S clone, venv, 4.4GB weights, exported fp32/int8 models)
