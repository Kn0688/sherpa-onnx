// sherpa-onnx/csrc/ort-env.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ORT_ENV_H_
#define SHERPA_ONNX_CSRC_ORT_ENV_H_

#include <cstdlib>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "sherpa-onnx/csrc/macros.h"  // NOLINT

// Resolved at runtime from libcudart already loaded by the CUDA EP, so the
// csrc target does not need to link cudart explicitly.
extern "C" {
int cudaMalloc(void **ptr, size_t size);
int cudaFree(void *ptr);
}

namespace sherpa_onnx {

// Raw cudaMalloc/cudaFree allocator used when SHERPA_ONNX_CUDA_USE_ARENA=0.
// Bypasses the BFC arena entirely: every ORT allocation is a real cudaMalloc
// and every free is a real cudaFree, so GPU memory follows actual usage
// instead of being retained by the arena. Registered on every Ort::Env;
// sessions must opt in with session.use_env_allocators=1 (done in session.cc
// for the CUDA provider when the same env var is set).
inline void *RawCudaAlloc(OrtAllocator *, size_t size) {
  void *p = nullptr;
  if (size == 0 || cudaMalloc(&p, size) != 0) return nullptr;
  return p;
}
inline void RawCudaFree(OrtAllocator *, void *p) {
  if (p) cudaFree(p);
}
inline const OrtMemoryInfo *RawCudaInfo(const OrtAllocator *) {
  static Ort::MemoryInfo *mi =
      new Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  return *mi;
}
inline void *RawCudaReserve(OrtAllocator *t, size_t size) {
  return RawCudaAlloc(t, size);
}

inline void MaybeRegisterRawCudaAllocator(Ort::Env &env) {
  const char *v = std::getenv("SHERPA_ONNX_CUDA_USE_ARENA");
  if (v == nullptr || std::atoi(v) != 0) return;
  static OrtAllocator raw_cuda_allocator = [] {
    OrtAllocator a{};
    a.version = ORT_API_VERSION;
    a.Alloc = RawCudaAlloc;
    a.Free = RawCudaFree;
    a.Info = RawCudaInfo;
    a.Reserve = RawCudaReserve;
    return a;
  }();
  env.RegisterAllocator(&raw_cuda_allocator);
  SHERPA_ONNX_LOGE("CUDA BFC arena disabled (SHERPA_ONNX_CUDA_USE_ARENA=0), "
                   "using raw cudaMalloc/cudaFree");
}

// Create an Ort::Env with appropriate threading configuration.
// In WASM builds, onnxruntime's default thread pool creation can cause
// abort() from background pthreads. Using CreateEnvWithGlobalThreadPools
// with single-threaded pools avoids this. Per-session threading is
// configured separately via session.cc SetIntraOpNumThreads.
inline Ort::Env CreateOrtEnv() {
#if SHERPA_ONNX_ENABLE_WASM
  Ort::ThreadingOptions tp;
  auto &api = Ort::GetApi();
  Ort::ThrowOnError(api.SetGlobalIntraOpNumThreads(tp, 1));
  Ort::ThrowOnError(api.SetGlobalInterOpNumThreads(tp, 1));
  OrtEnv *env = nullptr;
  Ort::ThrowOnError(api.CreateEnvWithGlobalThreadPools(
      ORT_LOGGING_LEVEL_ERROR, "sherpa-onnx", tp, &env));
  return Ort::Env(env);
#else
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR);
  MaybeRegisterRawCudaAllocator(env);
  return env;
#endif
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ORT_ENV_H_
