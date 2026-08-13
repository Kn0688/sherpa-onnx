# parakeet-tdt-0.6b-v3 ONNX export + int8 encoder optimization

Export https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 to the
encoder/decoder/joiner ONNX triple consumed by sherpa-onnx
(`--model-type=nemo_transducer`).

- `./run.sh` — download the `.nemo` checkpoint, export to ONNX, quantize to
  int8, and smoke-test en/de/fr/es wavs.
- `./optimize_encoder_int8.py` — **post-quantization graph surgery on the
  int8 encoder** (no re-export, no NeMo needed). See below.

## Why optimize the released int8 encoder

`onnxruntime.quantization.quantize_dynamic` converts every Conv it can into
`ConvInteger`. On arm64 ONNX Runtime has **no optimized ConvInteger kernel**
(it falls back to the reference implementation), so in the released
`sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` model the 77 ConvInteger ops
take ~28% of the encoder time.

Measured per encoder run (Apple Silicon, T=385 mel frames, threads=2):

| op                       | ConvInteger | after rewrite          |
|--------------------------|-------------|------------------------|
| depthwise (k=9), 24x     | 29.1 ms     | 1.8 ms (fp32 Conv)     |
| subsampling, 5x          | 5.9 ms      | fp32 Conv              |
| pointwise 1x1, 48x       | 14.3 ms     | MatMulInteger (u8 GEMM)|

The rewrite is split by op type:

1. **depthwise + subsampling -> fp32 Conv**: the u8 weights are dequantized
   offline and the DynamicQuantizeLinear activation chain is bypassed
   (fp32 Conv on arm64 is 16x faster than ConvInteger for depthwise).
2. **pointwise 1x1 -> Transpose + MatMulInteger + Transpose**: keeps the
   ORIGINAL u8 weights / zero-point / scale. Integer accumulation is exact
   (1024 \* 255 \* 255 < 2^31), so this path is bit-exact vs the released
   ConvInteger — it just runs on the fast MLAS u8 GEMM.

The output file is the same size as the input (dead quantization chains are
eliminated). Only the encoder changes; decoder/joiner/tokens are reused
as-is.

## Usage

```bash
pip install onnx numpy

# after ./run.sh (or with the released encoder.int8.onnx):
python3 ./optimize_encoder_int8.py encoder.int8.onnx encoder.int8.opt.onnx
# then rename encoder.int8.opt.onnx -> encoder.int8.onnx in the model dir
```

## Measured results (macOS arm64, threads=2)

- single utterance: RTF 0.041 -> 0.033 (-20%)
- 102.5 s long audio: RTF 0.055 -> 0.049 (-11%)
- batch mode: RTF 0.271 -> 0.136 (2x)
- accuracy, 200 utterances LibriSpeech test-clean: WER 2.42% -> 2.33%
  (156/200 identical transcripts; no regression)
