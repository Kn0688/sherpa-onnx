# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

sherpa-onnx is a local (offline, no network) speech-processing engine built on ONNX Runtime. It provides speech recognition (streaming + non-streaming ASR), text-to-speech, speaker diarization/identification/verification, VAD, keyword spotting, spoken language identification, audio tagging, punctuation, speech enhancement, and source separation.

The C++ core in `sherpa-onnx/csrc/` is the single source of truth. Everything else — the C API, and all language bindings (Python, Go, C#, Java, Kotlin, Swift, Rust, Dart, Pascal, JavaScript/WASM) — wraps that core. A change to behavior almost always starts in `csrc/`.

## Build

The build is CMake-driven. Each cross-compile / platform target has a dedicated `build-*.sh` script at the repo root (e.g. `build-android-arm64-v8a.sh`, `build-ios.sh`, `build-wasm-simd-asr.sh`, `build-swift-macos.sh`) that creates its own `build-<target>/` directory. Read the relevant script before building for a platform — they set the exact CMake flags.

For a plain desktop build:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TESTS=ON ..
make -j
```

Key CMake options (all default as noted; see top of `CMakeLists.txt`):
- `SHERPA_ONNX_ENABLE_TESTS` (OFF) — build the gtest suite
- `SHERPA_ONNX_ENABLE_PYTHON` (OFF), `SHERPA_ONNX_ENABLE_JNI` (OFF)
- `SHERPA_ONNX_ENABLE_TTS` (ON), `SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION` (ON) — turning these OFF drops whole subsystems and their tests/deps
- `SHERPA_ONNX_ENABLE_GPU` (OFF, CUDA), NPU backends: `SHERPA_ONNX_ENABLE_RKNN`, `_AXERA`, `_AXCL`, `_ASCEND_NPU`, `_QNN`
- `BUILD_SHARED_LIBS` (OFF) — static by default
- `SHERPA_ONNX_ENABLE_WASM*` — WebAssembly variants, one per feature

Dependencies (onnxruntime, kaldi-native-fbank, kaldi-decoder, openfst, piper-phonemize, portaudio, etc.) are fetched automatically via `cmake/*.cmake` — no manual install. The correct onnxruntime prebuilt is chosen per platform (`cmake/onnxruntime-*.cmake`).

Python: `python3 setup.py bdist_wheel` (honors `SHERPA_ONNX_CMAKE_ARGS` / `SHERPA_ONNX_MAKE_ARGS` env vars). Set `SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON"` for the CUDA wheel.

## Test

C++ tests use GoogleTest and are only built when `SHERPA_ONNX_ENABLE_TESTS=ON`. Each `*-test.cc` in `csrc/` becomes its own executable and CTest case.

```bash
cd build
ctest                        # run all tests
ctest -R text-utils-test     # run one test by name
./bin/text-utils-test        # or run the executable directly
```

Test source files are listed explicitly in the `sherpa_onnx_test_srcs` block of `sherpa-onnx/csrc/CMakeLists.txt` — add new `*-test.cc` files there.

## Lint / style

- C++ follows Google style (`.clang-format`, `BasedOnStyle: Google`, C++11). Enforced in CI by cpplint.
  - `./scripts/check_style_cpplint.sh`    — check files from the last commit
  - `./scripts/check_style_cpplint.sh 1`  — check uncommitted changes
  - `./scripts/check_style_cpplint.sh 2`  — check the whole project
- clang-tidy: `make clang-tidy-check` (or `make check`) against `.clang-tidy`.
- Python: flake8, max line length 120 (`.flake8`).

## Architecture

### The Impl / factory pattern (most important thing to understand)

Every major feature is a thin public class backed by an abstract `*Impl` with a static `Create()` factory that picks a concrete subclass at runtime based on which model paths are set in the config. This is how one API surface supports dozens of model architectures.

Example — `OfflineRecognizer` (non-streaming ASR):
- `offline-recognizer.h/.cc` — public class, holds a `unique_ptr<OfflineRecognizerImpl>`
- `offline-recognizer-impl.h/.cc` — abstract base + `Create()` dispatch
- `offline-recognizer-<model>-impl.h` — one concrete impl per model family (whisper, paraformer, sense-voice, transducer, moonshine, canary, fire-red-asr, qwen3-asr, dolphin, ...)

`Create()` inspects `config.model_config` (e.g. `if (!config.model_config.whisper.encoder.empty())`) to decide which impl to instantiate. NPU providers (`provider == "rknn"/"axera"/"axcl"`) are branched first and select `*TplImpl<...ModelRknn>` template instantiations, guarded by `#if SHERPA_ONNX_ENABLE_*`.

The same pattern repeats for: `OnlineRecognizer` (streaming ASR), `OfflineTts`, `KeywordSpotter`, `OfflinePunctuation`, `AudioTagging`, `OfflineSourceSeparation`, `OfflineSpeakerDiarization`, VAD, etc. When adding support for a new model, you add a new `*-impl.h`, a `*-model.cc/.h` (+ optional `*-model-config.cc/.h` and `*-model-meta-data.h`), and a branch in the relevant `Create()`.

### File naming conventions in csrc/

- `offline-*` = non-streaming (whole audio at once); `online-*` = streaming (chunk by chunk).
- `*-model.cc/.h` = ONNX session wrapper for one architecture (loads the model, runs `Ort::Session`).
- `*-model-config.cc/.h` = config struct + CLI arg registration + validation.
- `*-model-meta-data.h` = metadata baked into the ONNX file, read via the `SHERPA_ONNX_READ_META_DATA*` macros in `macros.h`.
- `*-decoder*.cc/.h` = decoding strategy (greedy-search, modified-beam-search, ctc-*, etc.), independent of the model.
- `*-rknn.*`, `*-qnn.*` etc. = NPU-specific model implementations using a native (non-ONNX) runtime.

### Runtime backends

Default backend is ONNX Runtime, selected via the `provider` string (`provider.cc`): `cpu`, `cuda`, `coreml`, `xnnpack`, `nnapi`, `trt`, `directml`, `spacemit`. Separately, several NPU vendors have their own native runtimes (RKNN, QNN, Axera/AXCL, Ascend) compiled in behind `SHERPA_ONNX_ENABLE_*` flags — these only cover a subset of models.

### ASR stream lifecycle

`OfflineStream` / `OnlineStream` carry per-utterance state. Typical flow: `recognizer.CreateStream()` → `stream->AcceptWaveform(sample_rate, samples)` → `recognizer.DecodeStream(s)` → `recognizer.GetResult(s)`. Streaming recognizers add `IsReady()`, `IsEndpoint()`, and `Reset()`. Feature extraction (fbank) is in `features.cc` via kaldi-native-fbank.

### Binding & API layers (all wrap csrc/)

- `sherpa-onnx/c-api/c-api.cc/.h` — flat C ABI over the core; the foundation for most other bindings and the stable exported surface.
- `sherpa-onnx/c-api/cxx-api.cc/.h` — idiomatic C++ RAII wrapper on top of the C API (move-only handles, plain config structs).
- `sherpa-onnx/jni/` — JNI (Android/Java/Kotlin); `sherpa-onnx/python/` — pybind11; plus `java-api/`, `kotlin-api/`, `rust/`, `pascal-api/`.
- `wasm/` — WebAssembly builds (one per feature).
- The library-produced targets: `sherpa-onnx-core` (static core), `sherpa-onnx-c-api`, `sherpa-onnx-cxx-api`.

### Runnable binaries & examples

CLI tools are the `sherpa-onnx-*.cc` files in `csrc/`, each registered as an `add_executable` in `csrc/CMakeLists.txt` (e.g. `sherpa-onnx-offline`, `sherpa-onnx` streaming, `sherpa-onnx-offline-tts`, `sherpa-onnx-vad`, `sherpa-onnx-microphone`). Per-language usage examples live in top-level `*-api-examples/` and `*-examples/` directories (`c-api-examples/`, `python-api-examples/`, `go-api-examples/`, `nodejs-addon-examples/`, etc.).

### scripts/

`scripts/<model-family>/` holds the Python export/conversion tooling that turns upstream checkpoints (icefall, nemo, whisper, piper, vits, kokoro, sense-voice, ...) into the ONNX + metadata format sherpa-onnx consumes. Relevant when debugging why a model won't load — the metadata keys read in `csrc/` must match what these export scripts write.

## Conventions

- Version lives in `CMakeLists.txt` (`SHERPA_ONNX_VERSION`). Bumping a release touches `CMakeLists.txt`, `CHANGELOG.md`, and `new-release.sh` together (see the comment above the version line).
- Logging/error macros are in `csrc/macros.h`: `SHERPA_ONNX_LOGE(...)`, `SHERPA_ONNX_EXIT(code)`, and the `SHERPA_ONNX_READ_META_DATA*` family for pulling values out of ONNX model metadata.
- Model documentation and pretrained models are hosted at https://k2-fsa.github.io/sherpa/onnx/ (referenced throughout error messages and the README).
