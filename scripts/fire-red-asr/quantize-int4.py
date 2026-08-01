#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""Quantize the exported FireRedASR2 decoder to int4 (MatMulNBits).

The decoder runs one M=1 GEMV per generated token, so it is memory-BANDWIDTH
bound: int4 weights halve the weight traffic vs int8. On arm64, onnxruntime's
KleidiAI/NEON SQNBIT GEMV ukernel (selected when accuracy_level=4) makes the
decoder step ~22% faster than int8 with half the weight memory, and the
recognition output is token-identical (validated on test-10s/60s/120s/180s).

Use the merged-QKV fp32 decoder from export-onnx.py as input: the merged
(3*D, D) QKV projection quantizes as ONE MatMulNBits node instead of three,
which is ~3% faster than int4 without the merge (96 vs 128 quantized nodes).

NOTE: the input must be the FP32 decoder, not an already-int8-quantized one.
The encoder is intentionally NOT quantized here: encoder MatMuls are
compute-bound GEMMs (M = num frames), so int4 only helps very short audio
(see optimization_roadmap.md 1.2.1). Keep using the transformers-optimized
fp32/int8 encoder.

Usage:
  python3 ./scripts/fire-red-asr/quantize-int4.py --dir ./out
  # -> ./out/decoder.int4.onnx
"""
import argparse
import os

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="dir containing decoder.onnx (fp32, merged QKV)")
    p.add_argument("--input", default=None,
                   help="input model path (default: <dir>/decoder.onnx)")
    p.add_argument("--output", default=None,
                   help="output model path (default: <dir>/decoder.int4.onnx)")
    p.add_argument("--block-size", type=int, default=32,
                   help="quantization block size along K; all decoder "
                        "Linears have K=1280, and 1280 %% 32 == 0")
    args = p.parse_args()

    in_file = args.input or os.path.join(args.dir, "decoder.onnx")
    out_file = args.output or os.path.join(args.dir, "decoder.int4.onnx")

    quantizer = MatMulNBitsQuantizer(
        onnx.load(in_file),
        bits=4,
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=4,  # SQNBIT_CompInt8: KleidiAI/NEON int4 GEMV ukernel
    )
    quantizer.process()
    quantizer.model.save_model_to_file(out_file)
    print("saved", out_file)

    model = onnx.load(out_file)
    n_nbits = sum(1 for n in model.graph.node if n.op_type == "MatMulNBits")
    n_matmul = sum(1 for n in model.graph.node if n.op_type == "MatMul")
    print(f"MatMulNBits nodes: {n_nbits}, remaining fp32 MatMul: {n_matmul}")

    onnx.checker.check_model(out_file)
    print("checker ok:", out_file)


if __name__ == "__main__":
    main()
