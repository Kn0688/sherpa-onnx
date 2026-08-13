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
# -> ./out/{encoder,decoder}.onnx (decoder has merged Q/K/V; encoder has its
#    32 pointwise convs rewritten to MatMul so int8 quantizes them to sdot —
#    see "Encoder: pointwise conv -> MatMul" below. Always on.)

python3 ./scripts/fire-red-asr/quantize-int8.py --dir ./out
# -> ./out/{encoder,decoder}.int8.onnx (+ .data)

python3 ./scripts/fire-red-asr/quantize-int4.py --dir ./out
# -> ./out/decoder.int4.onnx  (recommended decoder; use encoder.int8.onnx)
```

Note: exporting requires `onnxscript` (torch dynamo exporter) and writes
external-data `.data` files (models > 2GB). Do NOT add `Conv` to
`op_types_to_quantize` on arm64 — ConvInteger is slower than fp32 Conv there.

## Optimizations

### Encoder: pointwise conv -> MatMul at export time (default) — int8 ~30% faster, ~37% smaller, token-identical

`export-onnx.py` rewrites the Conformer conv module's 32 pointwise (1x1,
bias-free) Conv1d — `pointwise_conv1` (1280->5120) and `pointwise_conv2`
(2560->1280), 16 layers each — as `Transpose + MatMul + Transpose`. This is
**always on** (not a flag): it is a prerequisite for a fast int8 encoder, not
an optional tweak.

- **Why**: `quantize-int8.py` quantizes MatMul but deliberately NOT Conv
  (ConvInteger has no fast arm64 kernel, ~2x slower than fp32 — see the Note
  above). Left as Conv, these 32 layers stay **fp32** through int8
  quantization and become the int8 encoder's bottleneck (both compute and
  size). Rewritten to MatMul, `quantize_dynamic` lowers them to int8
  `MatMulInteger`, which uses the arm64 sdot ukernel.
- **How / layout**: Conv1d is NCW — `out[n,o,t] = sum_i W[o,i,0]*in[n,i,t]`.
  Equivalent MatMul: transpose in to (N,T,Cin), matmul by W2(Cin,Cout) ->
  (N,T,Cout), transpose back. Weight (O,I,1) -> (I,O).
- **CRITICAL — weight must be MatMul input[1]**: `quantize_dynamic` hard-codes
  input[0]=activation, input[1]=weight
  (onnxruntime `quantization/operators/matmul.py`), and with
  `MatMulConstBOnly=True` (the dynamic default) only quantizes a MatMul whose
  input[1] is a constant initializer. So `x @ W` (weight on input[1]) IS
  quantized to int8; the cleaner-looking `W @ x` (weight on input[0]) would
  NOT be. Keep W on the right.
- **Measured** (Apple M3 Pro, arm64, strict interleaved A/B vs the original
  int8 encoder with fp32 convs, 41 runs/config, no profiling):

  | metric | original int8 | conv->MatMul int8 | gain |
  |--------|--------------|-------------------|------|
  | speed (median, threads=1) | baseline | — | **-30.0%** (3s/10s/17s: -31.0/-31.2/-30.1) |
  | speed (median, threads=4) | baseline | — | **-29.4%** (3s/10s/17s: -29.0/-30.0/-27.3) |
  | encoder live weights | 1292 MB | 820 MB | **-36.5%** |
  | recognition | — | — | token-identical, CER 0.0 on 8 wavs |

- **Where the 472 MB goes**: the 32 conv weights (688 MB of the fp32 total)
  become int8 (fp32 688->59 MB, int8 604->761 MB); same weights at 1/4 the
  bytes.
- **Not bit-exact** vs Conv (fp32 accumulation order differs), but the fp32
  graph is verified cos=1.0 / max|Δ|=7.6e-6 vs the Conv encoder, and int8
  end-to-end is token-identical (8 wavs: zh/en-mixed, Mandarin, Sichuan,
  Tianjin, Henan dialects, 8k).
- **Timing is what matters**: the SAME conv->MatMul rewrite applied
  **post-quantization** (on `encoder.int8.onnx`) is a no-op rename — the convs
  are already fp32 there and ORT's CPU already lowers 1x1 conv to a GEMM, so it
  is speed-neutral. The gain exists ONLY when the rewrite happens **before**
  quantization — i.e. here, at export — so that `quantize_dynamic` actually
  turns these 32 layers into int8 MatMulInteger.

### Encoder: ONNX Runtime transformers optimizer (REMOVED - verified ineffective)

- **Status**: **REMOVED** — strict A/B testing (no profiling, 50 runs) shows
  the optimized encoder is **slower or equal** to the original int8 encoder
  (-0.6% / -2.7% / -1.0% on 3s/10s/15s audio). The graph is almost identical
  (1992 vs 1993 nodes, only 1 `Not` node difference), so the transformers
  optimizer does not actually optimize the int8 model.
- **Previous claim (28%) was a measurement error** caused by profiling
  overhead in the test harness. Do NOT use this optimization.

### Decoder: merged Q/K/V + MatMulNBits int4 (22% + half weight memory)

- **Why**: each decoder step is an M=1 GEMV, i.e. memory-**bandwidth**
  bound, and the three separate Q/K/V projections mean 3x the kernel
  launches (96 small self-attn MatMuls across 16 layers).
- **How**: (1) `export-onnx.py` pre-concatenates `w_qs/w_ks/w_vs` into one
  `(3*D, D)` `w_qkv` per layer (the bias-less K projection gets a
  zero-padded bias slice, so the merge is exact) and splits the result back
  into Q/K/V — 96 -> 32 MatMuls, numerically verified against the original
  projections (max abs err ~7e-6, float32 rounding). (2)
  `quantize-int4.py` converts the fp32 decoder weights to int4 with
  `MatMulNBitsQuantizer(bits=4, block_size=32, is_symmetric=True,
  accuracy_level=4)`; `accuracy_level=4` selects the KleidiAI/NEON SQNBIT
  GEMV ukernel. All decoder Linears have K=1280, and 1280 % 32 == 0.
- **Measured** (Apple Silicon, vs original int8 decoder):

  | config | decoder step | MatMul count | weights |
  |--------|--------------|--------------|---------|
  | int8 decoder | 7.491 ms | 193 | ~418 MB |
  | int4 decoder | 6.001 ms (-20%) | 128 | ~209 MB |
  | **merged QKV + int4** | **5.811 ms (-22%)** | **96** | **~209 MB** |

- **Accuracy**: recognition output 100% token-identical to int8 on
  test-10s/60s/120s/180s clips. End-to-end (optimized encoder + int4
  decoder): 2.233s -> 1.280s on 10s audio (**43%**), ~5%/1% on 60s/120s
  (encoder share dilutes the decoder gain).
- **Caveats**: the input to `quantize-int4.py` must be the **fp32** decoder
  (not an int8 one). Encoder int4 is intentionally not used — encoder
  MatMuls are compute-bound GEMMs, so int4 only helps very short audio.
  iOS xcframework builds of onnxruntime may lack KleidiAI; the plain NEON
  kernel still works.

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

## Length bucketing in sherpa-onnx (done)

`OfflineRecognizerFireRedAsrImpl::DecodeStreams` sorts streams by frame
count and cuts buckets at `kMaxBucketLengthRatio = 1.2` before batching
(transparent to callers). Validated on the 8-file mixed set (4.7–17.6s,
split into 4 buckets of 2+4+1+1): correctness 8/8 vs per-stream, bucketed
batch ≈1.05x vs sequential (unbucketed mixed batch was 0.44x). Gains grow
with bucket fullness; small ad-hoc sets stay near breakeven.

## Conv quantization: measured dead ends

These are dead ends for quantizing conv **as conv**. The path that DOES work —
rewrite the pointwise convs to MatMul before quantization so they become
`MatMulInteger` (sdot) instead of `ConvInteger` — is the first section above
("Encoder: pointwise conv -> MatMul at export time").

- `quantize_dynamic` with `Conv` → `ConvInteger`: ~2x SLOWER than fp32 on
  arm64 (no optimized kernel). Do not use.
- `quantize-qdq-conv.py` (manual weight-only QDQ: per-output-channel int8 +
  DequantizeLinear, compute stays fp32): verified 8/8 token-identical and
  ConvInteger-free, but **~4% slower at runtime, NOT recommended** — a
  single encoder forward amortizes weight-load bandwidth to ~nothing, so
  dequantize overhead dominates. Kept in the repo only for disk-size
  experiments (needs .data compaction to realize savings).
- The earlier "official int8 is ~30% faster on short utterances" is already
  closed by the slim decoder graph: this export is FASTER than the official
  int8 everywhere (8% on 10s, 41% on 5s, 39% on 17.6s audio).
