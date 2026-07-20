#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""Quantize the exported FireRedASR2 encoder/decoder to int8.

Mirrors scripts/whisper/export-onnx.py: quantize_dynamic over MatMul with
QInt8 weights. NOTE: do NOT add "Conv" to op_types_to_quantize on arm64 —
onnxruntime's ConvInteger has no optimized kernel there and runs ~2x slower
than fp32 Conv.

Two gotchas handled here:
1. quantize_dynamic with use_external_data_format leaves dead bytes in the
   .data file (observed: 1.29GB live -> 3.88GB file); compact_external_data()
   rewrites it to the live size. The re-save must go to a NEW path first —
   re-writing the same filename grows the file instead of truncating it.
2. quantize_dynamic drops metadata_props; the encoder metadata is re-added
   with a proto-only save (never re-save an external-data model with
   save_as_external_data=True just to edit metadata — that duplicates every
   tensor in the .data file).
"""
import argparse
import os

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def add_meta_data(filename, meta_data):
    model = onnx.load(filename, load_external_data=False)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)  # proto only; external refs preserved


def get_meta(filename):
    m = onnx.load(filename, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


def compact_external_data(filename):
    """Rewrite the .data file keeping only live (referenced) tensors.

    quantize_dynamic with use_external_data_format leaves dead bytes in the
    .data file (observed: 1.29GB live -> 3.88GB file). Round-tripping every
    initializer through raw_data and re-saving compacts it to the live size.

    Two gotchas:
    - the re-save must use a NEW file base name (and its own location name);
      saving with location set to the final .data name doubles the tensors
    - after moving the files into place, the location entries inside the
      proto still point to the temporary name and must be patched with a
      proto-only save
    """
    tmp = filename + ".compact.tmp"
    tmp_data = os.path.basename(tmp) + ".data"
    model = onnx.load(filename, load_external_data=True)
    for t in model.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            arr = onnx.numpy_helper.to_array(t)
            t.ClearField("external_data")
            t.data_location = onnx.TensorProto.DEFAULT
            t.raw_data = arr.tobytes()
    onnx.save(model, tmp, save_as_external_data=True,
              all_tensors_to_one_file=True, location=tmp_data,
              size_threshold=100)

    final_data = os.path.basename(filename) + ".data"
    model2 = onnx.load(tmp, load_external_data=False)
    for t in model2.graph.initializer:
        for e in t.external_data:
            if e.key == "location" and e.value != final_data:
                e.value = final_data
    onnx.save(model2, tmp)  # proto only; refs patched to the final name

    os.replace(tmp, filename)
    os.replace(tmp + ".data", filename + ".data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="dir containing encoder.onnx and decoder.onnx")
    args = p.parse_args()

    enc_meta = get_meta(os.path.join(args.dir, "encoder.onnx"))

    for name in ["encoder", "decoder"]:
        quantize_dynamic(
            model_input=os.path.join(args.dir, f"{name}.onnx"),
            model_output=os.path.join(args.dir, f"{name}.int8.onnx"),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
            use_external_data_format=True,
        )
        compact_external_data(os.path.join(args.dir, f"{name}.int8.onnx"))
        print("saved", os.path.join(args.dir, f"{name}.int8.onnx"))

    add_meta_data(os.path.join(args.dir, "encoder.int8.onnx"), enc_meta)
    print("metadata restored on encoder.int8.onnx")


if __name__ == "__main__":
    main()
