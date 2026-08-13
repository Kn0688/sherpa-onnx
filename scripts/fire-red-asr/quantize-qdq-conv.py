#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""Manual weight-only QDQ quantization for Conv ops (arm64-safe).

Why manual: onnxruntime 1.27's quantize_dynamic has no quant_format option,
and operator-oriented ConvInteger has NO optimized kernel on arm64 (~2x
slower than fp32). This script quantizes every Conv weight to
per-output-channel int8 and inserts DequantizeLinear, so weights are stored
as int8 (75% smaller, 75% less load bandwidth) while compute stays fp32.

Memory note: this works PROTO-ONLY. The >1GB .data file is never loaded
whole; each Conv weight is read lazily via np.memmap from the external
.data file, quantized, and stored back as a small int8 tensor inside the
proto. The old fp32 Conv tensors stay as dead bytes in the shared .data
file (harmless; the proto no longer references them).

Usage:
  ./quantize-qdq-conv.py --dir /path/to/exported

Reads encoder.int8.onnx (from quantize-int8.py), writes
encoder.qdqconv.int8.onnx which SHARES encoder.int8.onnx.data.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DTYPE_SIZE = {
    TensorProto.FLOAT: 4,
    TensorProto.INT8: 1,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT8: 1,
}
DTYPE_NP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.INT8: np.int8,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
}


def read_external_tensor(t, data_dir):
    """Read an external-data tensor's values via memmap (never whole-file)."""
    loc, offset, length = None, 0, None
    for e in t.external_data:
        if e.key == "location":
            loc = e.value
        elif e.key == "offset":
            offset = int(e.value)
        elif e.key == "length":
            length = int(e.value)
    path = os.path.join(data_dir, loc)
    dtype = DTYPE_NP[t.data_type]
    count = int(np.prod(t.dims))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    if length is None:
        length = count * DTYPE_SIZE[t.data_type]
    raw = mm[offset:offset + length]
    arr = np.frombuffer(raw, dtype=dtype, count=count)
    return arr.reshape(t.dims)


def quantize_conv_weights(model, data_dir):
    """Replace each fp32 Conv weight W with int8 W_q + DequantizeLinear."""
    name2init = {t.name: t for t in model.graph.initializer}
    new_initializers = []
    dq_insertions = []  # (consumer node, dq_node)

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        w_name = node.input[1]
        w = name2init.get(w_name)
        if w is None or w.data_type != TensorProto.FLOAT:
            continue

        if w.data_location == TensorProto.EXTERNAL:
            arr = read_external_tensor(w, data_dir).astype(np.float32)
        else:
            arr = numpy_helper.to_array(w).astype(np.float32)

        out_ch = arr.shape[0]
        scale = np.abs(arr).reshape(out_ch, -1).max(axis=1) / 127.0
        scale = np.maximum(scale, 1e-8).astype(np.float32)

        # weight can be 3-D (conv1d) or 4-D (conv2d) — reshape the scale
        # generically; hard-coding 4 dims would broadcast 3-D weights into a
        # bogus (out_ch, out_ch, C, k) allocation of >100 GB (OOM kill)
        scale_shape = (out_ch,) + (1,) * (arr.ndim - 1)
        w_q = np.round(arr / scale.reshape(scale_shape))
        w_q = np.clip(w_q, -127, 127).astype(np.int8)
        del arr

        q_name = w_name + "_int8"
        s_name = w_name + "_scale"
        z_name = w_name + "_zero"
        dq_out = w_name + "_deq"

        new_initializers.append(numpy_helper.from_array(w_q, name=q_name))
        new_initializers.append(numpy_helper.from_array(scale, name=s_name))
        new_initializers.append(numpy_helper.from_array(
            np.zeros(out_ch, dtype=np.int8), name=z_name))

        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, s_name, z_name],
            outputs=[dq_out],
            name=f"DQ_Conv_{len(dq_insertions)}",
            axis=0,
        )
        node.input[1] = dq_out
        dq_insertions.append(dq_node)

        # remove the old fp32 weight from the initializer list (its bytes
        # remain in the shared .data file but become unreferenced)
        model.graph.initializer.remove(w)

    # insert each DQ node right before its consumer Conv node, in document
    # order, so the graph stays topologically sorted
    dq_iter = iter(dq_insertions)
    merged = []
    for node in model.graph.node:
        if node.op_type == "Conv" and node.input[1].endswith("_deq"):
            merged.append(next(dq_iter))
        merged.append(node)
    del model.graph.node[:]
    model.graph.node.extend(merged)

    model.graph.initializer.extend(new_initializers)
    return len(dq_insertions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="dir containing encoder.int8.onnx")
    args = p.parse_args()

    src = os.path.join(args.dir, "encoder.int8.onnx")
    dst = os.path.join(args.dir, "encoder.qdqconv.int8.onnx")

    # proto-only load: external references stay as references
    model = onnx.load(src, load_external_data=False)
    n = quantize_conv_weights(model, args.dir)
    print(f"rewrote {n} Conv nodes to int8 weight + DequantizeLinear")

    # proto-only save: keeps referencing the SAME .data file
    # (encoder.int8.onnx.data). The dst model shares it with the src model.
    onnx.save(model, dst)
    print(f"saved {dst} (shares {os.path.basename(src)}.data)")


if __name__ == "__main__":
    main()
