# Introduction

This folder contains code showing how to convert [Whisper][whisper] to onnx
and use onnxruntime to replace PyTorch for speech recognition.

You can use [sherpa-onnx][sherpa-onnx] to run the converted model.

Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html
for details.

## Finding Alignment Heads for Word Timestamps

The `export-onnx-with-attention.py` script exports Whisper models with
cross-attention weights for word-level timestamps. It requires knowing which
attention heads are "alignment heads" - heads that show monotonically increasing
attention patterns useful for aligning audio to text.

For standard OpenAI Whisper models, alignment heads are defined in the
`ALIGNMENT_HEADS` dict in the export script. For new or custom models (like
distil-whisper variants), you can discover alignment heads using:

```bash
python find_alignment_heads.py --model <model-name> --audio <test-audio.wav>
```

This script analyzes all attention heads and ranks them by:
- **Monotonicity**: Whether attention peaks move forward as tokens are decoded
- **Diagonal score**: Correlation with expected diagonal attention pattern

Example output:
```
Top 15 alignment head candidates:
------------------------------------------------------------
 Layer   Head    Monotonic     Diagonal     Combined
------------------------------------------------------------
     3      2        0.846        0.985        0.915
     0      0        0.962        0.617        0.789
     ...
```

Heads with high combined scores (>0.7) are good candidates. A single head with
a very high diagonal score (>0.9) is often sufficient for accurate timestamps.

## Decoder Export Optimizations

`export-onnx.py` exports the decoder through `TextDecoderSlim`, an
export-safe wrapper that emits a minimal graph with the same inputs/outputs
as the legacy traced wrapper (so the C++ runtime needs no change):

- The causal mask is built vectorized on the fly (`arange(S) <= offset +
  arange(s)`, applied as an additive bias) instead of slicing a traced-in
  -inf mask buffer, and the KV cache is written with `scatter` instead of
  slice-assign. This removes the traced graph's control-flow scaffolding
  (Shape/Gather/Slice/Where/ScatterND chains): 13825 -> 3261 nodes for
  medium (-76%).
- The logits projection uses a registered transposed-embedding buffer, so
  the weight is a constant initializer at MatMul `input[1]` and quantizers
  can see it. (For the legacy traced layout — embedding behind an `Identity`
  at `input[0]` — `fold_logits_projection()` rewrites the graph to the same
  effect; it is kept as a safety net and is a no-op for the slim graph.)
- After the usual int8 `quantize_dynamic`, the decoder is also quantized to
  int4 with `MatMulNBitsQuantizer` (bits=4, block_size=32, symmetric,
  accuracy_level=4, i.e. the SQNBIT/KleidiAI int4 GEMV ukernel on arm64).
  Use `--no-int4` to skip it. The encoder stays int8-only: its MatMuls are
  compute-bound GEMMs where int4 does not pay off.

Measured on whisper medium (dense English speech, ~70 generated tokens,
end-to-end including the encoder): the slim int4 decoder is -28.3% vs the
previous int8 decoder (~-35% for the decoder part alone), with recognition
text token-identical, and the decoder file shrinks from 544.6 to 447.9 MiB
(-17.7%). On whisper tiny, where the fp32 logits projection dominated
per-step weight traffic, quantizing it to int8 alone gave -35% decode time,
also token-identical.

[whisper]: https://github.com/openai/whisper
[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
