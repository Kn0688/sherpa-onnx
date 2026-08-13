#!/usr/bin/env python3
"""Compact external data for encoder.qdqconv.int8.onnx.

The QDQ proto shares encoder.int8.onnx.data, which still contains the
dead fp32 Conv weights (no longer referenced). This script copies only
the referenced byte ranges into a fresh encoder.qdqconv.int8.onnx.data
and rewrites offsets, shrinking encoder total size by ~37%.

Usage:
  ./compact-qdq-data.py --dir /path/to/exported
"""
import argparse
import os

import onnx
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="dir containing encoder.qdqconv.int8.onnx")
    args = p.parse_args()

    d = args.dir
    proto_path = os.path.join(d, "encoder.qdqconv.int8.onnx")
    new_data_name = "encoder.qdqconv.int8.onnx.data"
    new_data_path = os.path.join(d, new_data_name)

    model = onnx.load(proto_path, load_external_data=False)

    # collect referenced (location, offset, length) per initializer
    out = open(new_data_path, "wb")
    written = 0
    n_ext = 0
    cache = {}  # (loc) -> memmap
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        loc, offset, length = None, 0, None
        for e in init.external_data:
            if e.key == "location":
                loc = e.value
            elif e.key == "offset":
                offset = int(e.value)
            elif e.key == "length":
                length = int(e.value)
        assert loc is not None, init.name
        if loc not in cache:
            cache[loc] = np.memmap(os.path.join(d, loc), dtype=np.uint8, mode="r")
        mm = cache[loc]
        if length is None:
            length = mm.size - offset
        new_offset = written
        out.write(mm[offset:offset + length])
        written += length
        n_ext += 1

        # rewrite external_data entries
        del init.external_data[:]
        for k, v in (("location", new_data_name),
                     ("offset", str(new_offset)),
                     ("length", str(length))):
            kv = init.external_data.add()
            kv.key = k
            kv.value = v
    out.close()

    onnx.save_model(model, proto_path, save_as_external_data=False)
    print("external initializers rewritten:", n_ext)
    print("new data file: %.1f MB" % (os.path.getsize(new_data_path) / 1e6))
    print("proto: %.1f MB" % (os.path.getsize(proto_path) / 1e6))


if __name__ == "__main__":
    main()
