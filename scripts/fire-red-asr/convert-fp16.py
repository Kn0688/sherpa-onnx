#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""Convert the exported FireRedASR2 fp32 encoder/decoder to fp16 for CUDA GPUs.

WHY THIS EXISTS (target platform: NVIDIA CUDA EP, esp. GTX 16xx / Turing TU116
which has NO int8 tensor cores):
  quantize-int8.py uses quantize_dynamic (separated dynamic quantization), whose
  per-MatMul entourage (DynamicQuantizeLinear + Cast + Mul) has no CUDA kernel
  and stays on CPU. On the CUDA EP that forces a host<->device memcpy around
  every quantized MatMul — measured: 535 Memcpy nodes inserted into the encoder
  graph, encoder on CUDA only 13% faster than CPU (1055 vs 1213 ms at T=500).
  fp16 weights+activations keep MatMul/Conv/LayerNorm fused on GPU end to end:
  same encoder 396 ms (2.7x), zero memcpy warnings.

ACCURACY GUARDRAILS:
  - keep_io_types=True: graph inputs/outputs stay float32, so the C++ side
    (offline-fire-red-asr-model.cc) needs no changes.
  - op_block_list keeps numerically sensitive ops in fp32 (LayerNormalization,
    Softmax, Sigmoid, Exp, Pow, ReduceMean, ...), so fp16 error stays ~1e-3.
    Verified: recognition text token-identical to the int8 production model on
    the 160s zh benchmark (md5-equal, 3 runs).

ONNXRUNTIME float16 CONVERTER BUGS WORKED AROUND HERE (ort 1.30, dynamo-exported
opset-18 input graph):
  1. Graph outputs are rewired to boundary Cast nodes whose input
     ("graph_output_cast_N") is left dangling — no node produces it. repair()
     points it at the (renamed) original producer output.
  2. Duplicate tensor output names and duplicate node names (one reused 47x).
     ONNX rejects both; we uniquify. Node names are labels only; tensor renames
     keep referential integrity because each duplicate gets a fresh name and no
     consumer could have referenced the ambiguous name.
  3. Never re-save an external-data model more than once: each save appends
     dead bytes to .data (encoder ballooned 6.2GB -> 9.3GB before compaction).
     This script saves exactly once per model.

Usage:
  python3 ./scripts/fire-red-asr/convert-fp16.py --dir ./out
  # -> ./out/{encoder,decoder}.fp16.onnx (+ .data), from {encoder,decoder}.onnx
"""
import argparse
import os

import onnx
from onnxruntime.transformers import float16

# Ops kept in fp32 (numerically sensitive / reduction ops / control flow).
OP_BLOCK_LIST = [
    "LayerNormalization", "Softmax", "Sigmoid", "Exp", "Pow",
    "ReduceMean", "Where", "Range", "CumSum", "NonZero",
]


def repair_dangling_graph_output_casts(model):
    """Fix converter bug 1: dangling `graph_output_cast_N` Cast inputs."""
    produced = set()
    for n in model.graph.node:
        produced.update(n.output)
    graph_outputs = {o.name for o in model.graph.output}
    fixed = 0
    for n in model.graph.node:
        for i, inp in enumerate(list(n.input)):
            if not inp.startswith("graph_output_cast_") or inp in produced:
                continue
            orig = next((o for o in n.output if o in graph_outputs), None)
            if orig is None:
                continue
            candidates = [o for nn in model.graph.node for o in nn.output
                          if o.startswith(orig) and o != orig]
            if not candidates:
                raise RuntimeError(f"cannot repair dangling input {inp} for {orig}")
            n.input[i] = candidates[0]
            produced.add(inp)
            fixed += 1
    return fixed


def uniquify_names(model):
    """Fix converter bug 2: duplicate tensor output names and node names."""
    graph_outputs = {o.name for o in model.graph.output}
    used = (set(graph_outputs) | {i.name for i in model.graph.input}
            | {t.name for t in model.graph.initializer})
    tensor_fixes = 0
    for n in model.graph.node:
        for j, onm in enumerate(list(n.output)):
            if not onm:
                continue
            if onm in used and onm not in graph_outputs:
                base, k = onm, 1
                while f"{base}_u{k}" in used:
                    k += 1
                n.output[j] = f"{base}_u{k}"
                tensor_fixes += 1
            used.add(n.output[j])
    seen = set()
    node_fixes = 0
    for n in model.graph.node:
        if not n.name:
            continue
        if n.name in seen:
            base, k = n.name, 1
            while f"{base}_d{k}" in seen:
                k += 1
            n.name = f"{base}_d{k}"
            node_fixes += 1
        seen.add(n.name)
    return tensor_fixes, node_fixes


def convert(src_dir, name):
    src = os.path.join(src_dir, f"{name}.onnx")
    dst = os.path.join(src_dir, f"{name}.fp16.onnx")
    model = onnx.load(src, load_external_data=True)
    fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=OP_BLOCK_LIST,
        disable_shape_infer=True,  # shape inference chokes on >2GB external data
    )
    n_repaired = repair_dangling_graph_output_casts(fp16)
    t_fixes, n_fixes = uniquify_names(fp16)
    # Save exactly once, fresh .data file (bug 3: no dead bytes).
    data_name = f"{name}.fp16.onnx.data"
    data_path = os.path.join(src_dir, data_name)
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(fp16, dst, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_name)
    gb = os.path.getsize(data_path) / 1e9
    print(f"{name}: repaired {n_repaired} dangling casts, "
          f"uniquified {t_fixes} tensors + {n_fixes} nodes -> {dst} ({gb:.2f} GB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="dir containing fp32 encoder.onnx and decoder.onnx "
                        "(from export-onnx.py)")
    args = p.parse_args()
    convert(args.dir, "encoder")
    convert(args.dir, "decoder")


if __name__ == "__main__":
    main()
