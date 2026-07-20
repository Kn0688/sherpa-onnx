# FireRedASR2-AED ONNX export

Export https://github.com/FireRedTeam/FireRedASR2S (FireRedASR2-AED) to the
encoder/decoder ONNX pair consumed by
`sherpa-onnx/csrc/offline-fire-red-asr-model.cc`.

Differences from the originally released `sherpa-onnx-fire-red-asr2-*` models:

1. **Dynamic batch** on every input (the released decoder hard-codes batch=1
   in `tokens`/`offset`; `OfflineFireRedAsrModel::SupportBatch()` returns
   false for it and sherpa falls back to per-stream decoding).
2. **The encoder masks self-attention with `x_len`** (faithful to the official
   PyTorch code; the released encoder did not, so valid frames were polluted
   by padding in mixed-length batches).
3. **Decoder cross-attention mask**: the encoder gains a third output
   `enc_mask (N, Tc)` and the decoder gains an input `cross_mask (N, Tc)`;
   padded encoder frames get a `-1e30` additive bias before the cross-attn
   softmax. sherpa detects the `cross_mask` input by name and stays
   backward-compatible with old models that do not have it.

## Usage

```bash
pip install torch onnx onnxscript onnxruntime kaldiio
git clone https://github.com/FireRedTeam/FireRedASR2S
# download weights: https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED
# (model.pth.tar, cmvn.ark, dict.txt)

python3 ./scripts/fire-red-asr/export-onnx.py \
  --repo ./FireRedASR2S \
  --model-dir ./FireRedASR2-AED \
  --output-dir ./out

python3 ./scripts/fire-red-asr/quantize-int8.py --dir ./out
# -> ./out/{encoder,decoder}.int8.onnx (+ .data)
```

Note: exporting requires `onnxscript` (torch dynamo exporter) and writes
external-data `.data` files (models > 2GB). Do NOT add `Conv` to
`op_types_to_quantize` on arm64 — ConvInteger is slower than fp32 Conv there.

## Validation (2026-07, macOS arm64, num-threads=2)

Test set: 8 wavs from the released model package (mixed lengths, up to 3.5x
padding in a batch of 8).

- mixed-length batch == per-stream: **8/8** (released model: 2/8;
  pre-cross_mask re-export: 7/8 with one token flip from unmasked
  cross-attention)
- equal-length batch == per-stream: 4/4
- int8 per-stream == fp32 per-stream: 8/8
- single-file RTF (int8): 0.133 (10.1s) / 0.168 (17.6s); fp32: 0.43 / 0.36;
  the mask pass-through costs ~5% on batch-of-8 end-to-end
- old released models (no cross_mask input) keep working unchanged
