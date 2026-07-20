# FireRedASR2-AED ONNX export (batch-capable)

`export-onnx.py` re-exports [FireRedASR2-AED](https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED)
(PyTorch) to `encoder.onnx` + `decoder.onnx` with **dynamic batch dimensions**,
fixing two limitations of the currently released sherpa-onnx models:

1. `tokens`/`offset` inputs are fixed to batch=1 in the released decoder
   (this export declares dynamic batch axes everywhere, so sherpa-onnx's
   `SupportBatch()` detection enables true batched decoding).
2. The released encoder does not mask self-attention with `x_len`, so
   mixed-length batches are corrupted by padding. This export masks
   self-attention with `x_len` (mixed-length batch went 2/8 → 7/8 vs
   per-utterance results).

Known remaining limitation: the decoder cross-attention has no mask input,
so heavily padded rows can still see garbage encoder frames (observed as a
rare single-token flip). A cross-attention mask input is planned (Phase 2).

## Usage

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch onnx onnxruntime onnxscript soundfile kaldi_native_fbank modelscope

# Download FireRedASR2-AED weights (model.pth.tar, ~4.7 GB) from
# https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED
# and the official code from https://github.com/FireRedTeam/FireRedASR2S

python3 export-onnx.py   # see the script's top-level constants for paths
```

Outputs (fp32): `encoder.onnx` (+ external `.data`, ~3.1 GB),
`decoder.onnx` (+ `.data`, ~1.5 GB), opset 18. Metadata required by
sherpa-onnx (num_decoder_layers, num_head, head_dim, sos, eos, max_len,
cmvn_mean, cmvn_inv_stddev) is written into `encoder.onnx`.

## Validation (macOS arm64, ORT 1.27)

- Numerics vs official PyTorch: encoder cross_k/v max|Δ| ≈ 2e-5; decoder
  logits max|Δ| ≤ 2.9e-5, identical argmax
- batch=3 identical files: rows identical, match PyTorch ground truth
- Equal-length batch: 4/4 == per-utterance
- Mixed-length batch (padding up to 3.5x): 7/8 == per-utterance (released
  model: 2/8); remaining diff traced to fp32-vs-int8 and one cross-attn
  token flip
- Speed with sherpa-onnx `fireredasr-batch-decoding` branch (fp32,
  threads=2): equal-length batch=4 → 1.28x, batch=8 → ~1.5x; mixed-length
  batch is a loss (0.57x) due to padding waste in the fp32 encoder —
  **always sort/bucket by length before batching**

## Quantization (Phase 5)

Planned: `quantize_dynamic` on MatMul (mirroring `scripts/whisper`), mainly
for the encoder (70-80% of e2e time). Re-validate text equality after
quantization; expect minor token diffs vs fp32.
