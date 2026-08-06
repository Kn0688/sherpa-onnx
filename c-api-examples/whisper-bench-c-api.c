// c-api-examples/whisper-bench-c-api.c
//
// Benchmark whisper decoding: measure only the DecodeOfflineStream time,
// excluding model loading.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sherpa-onnx/c-api/c-api.h"

static double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <encoder> <decoder> <tokens> <wav> <tail_paddings>\n", argv[0]);
    return -1;
  }

  const char *encoder_filename = argv[1];
  const char *decoder_filename = argv[2];
  const char *tokens_filename = argv[3];
  const char *wav_filename = argv[4];
  int32_t tail_paddings = atoi(argv[5]);

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config.debug = 0;
  recognizer_config.model_config.num_threads = 1;
  recognizer_config.model_config.provider = "cpu";
  recognizer_config.model_config.tokens = tokens_filename;
  recognizer_config.model_config.whisper.decoder = decoder_filename;
  recognizer_config.model_config.whisper.encoder = encoder_filename;
  recognizer_config.model_config.whisper.language = "en";
  recognizer_config.model_config.whisper.tail_paddings = tail_paddings;
  recognizer_config.model_config.whisper.task = "transcribe";

  const SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  // Warm up
  for (int32_t i = 0; i < 3; ++i) {
    const SherpaOnnxOfflineStream *stream =
        SherpaOnnxCreateOfflineStream(recognizer);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                    wave->num_samples);
    SherpaOnnxDecodeOfflineStream(recognizer, stream);
    const SherpaOnnxOfflineRecognizerResult *result =
        SherpaOnnxGetOfflineStreamResult(stream);
    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
  }

  // Benchmark
  int32_t N = 20;
  double total = 0;
  for (int32_t i = 0; i < N; ++i) {
    const SherpaOnnxOfflineStream *stream =
        SherpaOnnxCreateOfflineStream(recognizer);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                    wave->num_samples);

    double start = now_sec();
    SherpaOnnxDecodeOfflineStream(recognizer, stream);
    double end = now_sec();

    total += (end - start);

    const SherpaOnnxOfflineRecognizerResult *result =
        SherpaOnnxGetOfflineStreamResult(stream);
    if (i == 0) {
      fprintf(stderr, "Text: %s\n", result->text);
    }
    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
  }

  fprintf(stderr, "Average decode time: %.4f s\n", total / N);

  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
