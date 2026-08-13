#!/usr/bin/env python3
"""Optimize the released parakeet-tdt int8 encoder (graph surgery, no re-export).

Measured on Apple Silicon (T=385 mel frames, threads=2), per encoder run:
  ConvInteger total        49.3 ms
    depthwise (k=9)        29.1 ms   <- reference kernel, pathological
    pointwise_conv1 (1x1)   9.3 ms
    pointwise_conv2 (1x1)   5.0 ms
    subsampling             5.9 ms
  fp32 Conv total          57.0 ms
    depthwise               1.8 ms   <- 16x faster than ConvInteger
    pointwise              57.1 ms   <- 4x SLOWER than ConvInteger

So the optimal assignment is split:
  1. depthwise + subsampling ConvInteger -> fp32 Conv
     (dequantize u8 weights offline; activation quantization chain bypassed)
  2. pointwise 1x1 ConvInteger -> Transpose + MatMulInteger + Transpose,
     keeping the ORIGINAL u8 weights/zero-point/scale: integer accumulation
     is exact (1024*255*255 < 2^31), so this path is bit-exact vs the
     released ConvInteger while running on the fast MLAS u8 GEMM (measured
     same speed as u8xs8 sdot on arm64).

Usage:
  python3 ./optimize_encoder_int8.py encoder.int8.onnx encoder.int8.opt.onnx

Only numpy + onnx are required. The input is the int8 encoder produced by
./run.sh (or downloaded from the released model repo); the output replaces it.
decoder/joiner/tokens are untouched.
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def find_scale_zp(init, w_q_name):
    scale = init.get(w_q_name.replace("_quantized", "_scale"))
    zp = init.get(w_q_name.replace("_quantized", "_zero_point"))
    return scale, zp


class Graph:
    def __init__(self, model):
        self.g = model.graph
        self.init = {t.name: t for t in self.g.initializer}
        self.producers = {}
        for n in self.g.node:
            for o in n.output:
                self.producers[o] = n
        self.consumers = {}
        for n in self.g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)

    def single_consumer(self, node):
        cs = self.consumers.get(node.output[0], [])
        return cs[0] if len(cs) == 1 else None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"  # all | nodw | nopw
    model = onnx.load(src)
    gr = Graph(model)
    g = gr.g

    replace_map = {}   # id(ConvInteger) -> list of new nodes
    remove_ids = set()
    new_inits = []
    n_fp32 = n_mm = n_skip = 0

    for n in list(g.node):
        if n.op_type != "ConvInteger":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        w_q_t = gr.init[n.input[1]]
        w_shape = tuple(w_q_t.dims)
        group = attrs.get("group", 1)
        kernel = attrs.get("kernel_shape")
        # MatMulInteger rewrite only for 1-D (3-D weight) pointwise convs;
        # 2-D 1x1 subsampling convs go the fp32 Conv route (cheap anyway)
        is_pointwise = group == 1 and all(k == 1 for k in kernel) and len(w_shape) == 3

        x_q, w_q_name = n.input[0], n.input[1]
        x_zp_name = n.input[2] if len(n.input) > 2 else ""
        dql = gr.producers.get(x_q)
        cast = gr.single_consumer(n)
        mul = gr.single_consumer(cast) if cast is not None else None
        if (dql is None or dql.op_type != "DynamicQuantizeLinear"
                or cast is None or cast.op_type != "Cast"
                or mul is None or mul.op_type != "Mul"):
            print(f"  skip {n.name}: unexpected neighborhood")
            n_skip += 1
            continue

        scale_t, zp_t = find_scale_zp(gr.init, w_q_name)
        w_q = numpy_helper.to_array(w_q_t).astype(np.float32)
        w_scale = numpy_helper.to_array(scale_t).astype(np.float32)
        w_zp = numpy_helper.to_array(zp_t).astype(np.float32) if zp_t is not None else np.float32(0)
        w_fp32 = (w_q - float(w_zp.reshape(()))) * float(w_scale.reshape(()))

        base = n.name.strip("/").replace("/", ".")

        # downstream chain: ConvInteger -> Cast -> Mul(x_scale*w_scale) -> out
        scale_input = mul.input[1] if mul.input[0] == cast.output[0] else mul.input[0]
        scales_mul = gr.producers.get(scale_input)
        chain = {id(cast), id(mul)}
        if scales_mul is not None and scales_mul.op_type == "Mul":
            chain.add(id(scales_mul))

        if not is_pointwise:
            if mode == "nopw":
                pass  # fall through to fp32 Conv
            elif mode == "nodw":
                continue
            # ---- depthwise / subsampling: fp32 Conv, bypass quant chain ----
            w_name = base + ".w_deq_fp32"
            new_inits.append(numpy_helper.from_array(w_fp32, w_name))
            new_conv = onnx.helper.make_node(
                "Conv", [dql.input[0], w_name], [mul.output[0]],
                name=n.name + "_fp32", **attrs)
            replace_map[id(n)] = [new_conv]
            remove_ids.update(chain)
            n_fp32 += 1
        else:
            if mode == "nodw":
                pass  # fall through to MatMulInteger
            elif mode == "nopw":
                continue
            # ---- pointwise 1x1: Transpose + MatMulInteger + Transpose ----
            # Keep the ORIGINAL u8 weight / zero point / scale: integer
            # accumulation is exact (1024 * 255 * 255 < 2^31), so this path
            # is bit-exact vs the released ConvInteger while running on the
            # fast MLAS u8 GEMM (verified same speed as u8xs8 sdot).
            O, I = w_shape[0], w_shape[1]
            w_q_u8 = numpy_helper.to_array(w_q_t)          # (O,I,1) u8
            w2 = np.ascontiguousarray(
                w_q_u8.reshape(O, I).T)                    # (I,O) -> input[1]
            w2_name = base + ".w_u8_IO"
            new_inits.append(numpy_helper.from_array(w2, w2_name))
            w_zp_name = n.input[3] if len(n.input) > 3 else ""

            t_in = base + "_ntc"
            mm_out = base + "_mm"
            mm_inputs = [t_in, w2_name]
            if x_zp_name:
                mm_inputs.append(x_zp_name)
                if w_zp_name:
                    mm_inputs.append(w_zp_name)
            nodes = [
                onnx.helper.make_node("Transpose", [x_q], [t_in],
                                      name=base + "_pre", perm=[0, 2, 1]),
                onnx.helper.make_node("MatMulInteger", mm_inputs,
                                      [mm_out], name=base + "_mmint"),
                onnx.helper.make_node("Transpose", [mm_out], [n.output[0]],
                                      name=base + "_post", perm=[0, 2, 1]),
            ]
            replace_map[id(n)] = nodes
            # Cast/Mul dequant chain untouched (same scale, same zp)
            n_mm += 1

    nodes = []
    for x in g.node:
        if id(x) in remove_ids:
            continue
        if id(x) in replace_map:
            nodes.extend(replace_map[id(x)])
        else:
            nodes.append(x)
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_inits)

    # dead-code elimination (drops orphaned DQLs, old u8 weights/scales/zps)
    consumed = set()
    for n in g.node:
        consumed.update(i for i in n.input if i)
    consumed.update(o.name for o in g.output)
    nodes = list(g.node)
    changed = True
    while changed:
        changed = False
        live = []
        for n in nodes:
            if all((o in consumed) or (o == "") for o in n.output):
                live.append(n)
            else:
                changed = True
        if changed:
            consumed = set()
            for n in live:
                consumed.update(i for i in n.input if i)
            consumed.update(o.name for o in g.output)
            nodes = live
    del g.node[:]
    g.node.extend(live)

    used = set()
    for n in g.node:
        used.update(n.input)
    kept = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"fp32 Conv: {n_fp32}, pointwise->MatMulInteger: {n_mm}, skipped: {n_skip}")
    print(f"nodes {len(g.node)}, saved {dst}")


if __name__ == "__main__":
    main()
