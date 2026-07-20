#!/usr/bin/env python3
"""Export FireRedASR2-AED encoder/decoder to ONNX with dynamic batch axes (fp32).

I/O contract mirrors the released sherpa-onnx FireRedASR2 models
(see sherpa-onnx/csrc/offline-fire-red-asr-model.cc), but with dynamic batch:

encoder.onnx
  in : x (N,T,80) f32 [CMVN-normalized fbank], x_len (N,) i64
  out: n_layer_cross_k (16,N,Tc,1280), n_layer_cross_v (16,N,Tc,1280)

decoder.onnx
  in : tokens (N,1) i64,
       in_n_layer_self_k_cache / in_n_layer_self_v_cache (16,N,S,20,64),
       n_layer_cross_k / n_layer_cross_v (16,N,Tc,1280),
       offset (N,) i64
  out: logits (N,1,8667), out_n_layer_self_k_cache, out_n_layer_self_v_cache

Metadata (on encoder): num_decoder_layers, num_head, head_dim, sos, eos,
max_len, cmvn_mean, cmvn_inv_stddev (+ informational keys), copied from the
released sherpa encoder.int8.onnx so values are byte-identical.

Wrappers re-implement the official forward in an export-safe way:
- vectorized padding mask (official builds it with a python loop)
- true KV cache with write-at-offset + arange<=offset key mask
  (official caches layer outputs and recomputes k/v; equivalent because
  positions are causal/frozen)
- cross-attention mask is NOT included (mirrors the released sherpa decoder);
  mixed-length batches will suffer padding pollution -> Phase 2.
"""
import argparse
import os
import sys

REPO = "/tmp/firered-export/FireRedASR2S"
sys.path.insert(0, os.path.join(REPO, "fireredasr2s"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from fireredasr2.asr import load_fireredasr_aed_model  # noqa: E402

MODEL_DIR = "/tmp/firered-export/pretrained_models/FireRedASR2-AED"
RELEASED_ENCODER = ("/tmp/firered-test/sherpa-onnx-fire-red-asr2-zh_en-int8-"
                    "2026-02-26/encoder.int8.onnx")
OUT_DIR = "/tmp/firered-export/exported"

NUM_LAYERS = 16
NUM_HEAD = 20
HEAD_DIM = 64
D_MODEL = 1280


def rel_pos_emb(pe: torch.Tensor, T: int) -> torch.Tensor:
    # pe: (1, 2*max_len-1, D). Mirrors RelPositionalEncoding.forward.
    # Use narrow with an explicit symbolic length: a plain dynamic slice
    # pe[:, c-T+1:c+T] yields an UNBACKED symint for the length, which then
    # breaks view_as() inside RelPosMultiHeadAttention._rel_shift under
    # torch.export. narrow() keeps the output length a backed expression
    # 2*T-1.
    c = pe.shape[1] // 2
    return pe.narrow(1, c - T + 1, 2 * T - 1)


class FireRedEncoderWrapper(nn.Module):
    """Conformer encoder + per-decoder-layer cross k/v projections."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        # borrow the decoder's cross-attn projections (w_ks: no bias, w_vs: bias)
        self.cross_attn = nn.ModuleList(
            [layer.cross_attn for layer in model.decoder.layer_stack])

    def forward(self, x, x_len):
        # x: (N,T,80) CMVN-normalized; x_len: (N,) int64, x_len <= T
        x = F.pad(x, (0, 0, 0, self.encoder.input_preprocessor.context - 1))
        T = x.shape[1]
        # vectorized version of ConformerEncoder.padding_position_is_0
        mask = torch.arange(T, device=x.device).unsqueeze(0) < x_len.unsqueeze(1)
        src_mask = mask.unsqueeze(1)  # (N,1,T), bool, 1=valid
        embed, out_len, sub_mask = self.encoder.input_preprocessor(x, src_mask)
        h = self.encoder.dropout(embed)
        pos_emb = self.encoder.dropout(
            rel_pos_emb(self.encoder.positional_encoding.pe, embed.shape[1]))
        for layer in self.encoder.layer_stack:
            h = layer(h, pos_emb, slf_attn_mask=sub_mask, pad_mask=sub_mask)
        cross_k = torch.stack([m.w_ks(h) for m in self.cross_attn])
        cross_v = torch.stack([m.w_vs(h) for m in self.cross_attn])
        return cross_k, cross_v


class FireRedDecoderWrapper(nn.Module):
    """Single greedy step with true KV cache (write at offset)."""

    def __init__(self, model):
        super().__init__()
        self.dec = model.decoder

    def forward(self, tokens, in_self_k, in_self_v, cross_k, cross_v, offset):
        # tokens: (N,1) i64; in_self_k/v: (L,N,S,H,d) f32;
        # cross_k/v: (L,N,Tc,D) f32; offset: (N,) i64
        dec = self.dec
        N = tokens.shape[0]
        H, d = NUM_HEAD, HEAD_DIM
        S = in_self_k.shape[2]

        h = dec.dropout(
            dec.tgt_word_emb(tokens) * dec.scale
            + dec.positional_encoding.pe[0].index_select(0, offset).unsqueeze(1))

        # key mask: positions <= offset are valid (includes the just-written one)
        rng = torch.arange(S, device=tokens.device)
        key_mask = rng.unsqueeze(0) <= offset.unsqueeze(1)  # (N,S) bool
        neg = torch.tensor(float("-inf"), dtype=h.dtype, device=h.device)

        idx = offset.view(N, 1, 1, 1).expand(N, 1, H, d)

        out_ks, out_vs = [], []
        for i, layer in enumerate(dec.layer_stack):
            # --- self attention (last-position query, KV cache) ---
            residual = h
            x = layer.self_attn_norm(h)
            q = layer.self_attn.w_qs(x).view(N, 1, H, d)
            k = layer.self_attn.w_ks(x).view(N, 1, H, d)
            v = layer.self_attn.w_vs(x).view(N, 1, H, d)
            k_cache = in_self_k[i].scatter(1, idx, k)
            v_cache = in_self_v[i].scatter(1, idx, v)
            out_ks.append(k_cache)
            out_vs.append(v_cache)
            qh = q.transpose(1, 2)                    # (N,H,1,d)
            kh = k_cache.transpose(1, 2)              # (N,H,S,d)
            vh = v_cache.transpose(1, 2)              # (N,H,S,d)
            attn = torch.matmul(qh, kh.transpose(2, 3)) / (d ** 0.5)
            attn = attn.masked_fill(
                ~key_mask.view(N, 1, 1, S), neg)
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, vh)                # (N,H,1,d)
            o = o.transpose(1, 2).reshape(N, 1, H * d)
            h = residual + layer.self_attn.dropout(layer.self_attn.fc(o))

            # --- cross attention (no mask; mirrors released sherpa decoder) ---
            residual = h
            x = layer.cross_attn_norm(h)
            q = layer.cross_attn.w_qs(x).view(N, 1, H, d).transpose(1, 2)
            ck = cross_k[i].view(N, -1, H, d).transpose(1, 2)  # (N,H,Tc,d)
            cv = cross_v[i].view(N, -1, H, d).transpose(1, 2)
            attn = torch.matmul(q, ck.transpose(2, 3)) / (d ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, cv)                # (N,H,1,d)
            o = o.transpose(1, 2).reshape(N, 1, H * d)
            h = residual + layer.cross_attn.dropout(layer.cross_attn.fc(o))

            # --- mlp ---
            h = h + layer.mlp(layer.mlp_norm(h))

        h = dec.layer_norm_out(h)
        logits = dec.tgt_word_prj(h)  # (N,1,V)
        return logits, torch.stack(out_ks), torch.stack(out_vs)


def add_meta_data(filename, meta_data):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    base = os.path.basename(filename)
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=base + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )


def released_encoder_metadata():
    m = onnx.load(RELEASED_ENCODER, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


@torch.no_grad()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--opset", type=int, default=18)
    args.add_argument("--dynamo", type=int, default=1)
    args = args.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_fireredasr_aed_model(os.path.join(MODEL_DIR, "model.pth.tar"))
    model.eval()

    enc = FireRedEncoderWrapper(model)
    dec = FireRedDecoderWrapper(model)
    enc.eval()
    dec.eval()

    # ---- dummy inputs ----
    N, T = 2, 100
    x = torch.randn(N, T, 80)
    x_len = torch.tensor([T, T - 30], dtype=torch.int64)

    cross_k, cross_v = enc(x, x_len)
    Tc = cross_k.shape[2]
    print("cross_k:", tuple(cross_k.shape), "cross_v:", tuple(cross_v.shape))
    assert cross_k.shape == (NUM_LAYERS, N, Tc, D_MODEL)

    S = 32
    tokens = torch.tensor([[3], [3]], dtype=torch.int64)
    offset = torch.zeros(N, dtype=torch.int64)
    in_k = torch.zeros(NUM_LAYERS, N, S, NUM_HEAD, HEAD_DIM)
    in_v = torch.zeros_like(in_k)
    logits, out_k, out_v = dec(tokens, in_k, in_v, cross_k, cross_v, offset)
    print("logits:", tuple(logits.shape), "out_k:", tuple(out_k.shape))
    assert logits.shape == (N, 1, 8667)
    assert out_k.shape == in_k.shape

    export_kw = dict(
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=bool(args.dynamo),
    )

    # ---- encoder ----
    enc_file = os.path.join(OUT_DIR, "encoder.onnx")
    torch.onnx.export(
        enc, (x, x_len), enc_file,
        input_names=["x", "x_len"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_len": {0: "N"},
            "n_layer_cross_k": {1: "N", 2: "T"},
            "n_layer_cross_v": {1: "N", 2: "T"},
        },
        **export_kw,
    )
    print("saved", enc_file)

    # ---- decoder ----
    dec_file = os.path.join(OUT_DIR, "decoder.onnx")
    torch.onnx.export(
        dec,
        (tokens, in_k, in_v, cross_k, cross_v, offset),
        dec_file,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=[
            "logits",
            "out_n_layer_self_k_cache",
            "out_n_layer_self_v_cache",
        ],
        dynamic_axes={
            "tokens": {0: "N"},
            "in_n_layer_self_k_cache": {1: "N", 2: "T"},
            "in_n_layer_self_v_cache": {1: "N", 2: "T"},
            "n_layer_cross_k": {1: "N", 2: "T"},
            "n_layer_cross_v": {1: "N", 2: "T"},
            "offset": {0: "N"},
            "logits": {0: "N"},
            "out_n_layer_self_k_cache": {1: "N", 2: "T"},
            "out_n_layer_self_v_cache": {1: "N", 2: "T"},
        },
        **export_kw,
    )
    print("saved", dec_file)

    # ---- metadata (values copied verbatim from the released sherpa encoder) ----
    meta = released_encoder_metadata()
    keep = [
        "model_type", "version", "model_author", "maintainer", "feat_dim",
        "cmvn_mean", "cmvn_inv_stddev", "num_decoder_layers", "num_head",
        "head_dim", "max_len", "sos", "eos", "url", "url-2", "comment",
    ]
    enc_meta = {k: v for k, v in meta.items() if k in keep and k != "onnx.infer"}
    enc_meta["comment"] = (
        "FireRedASR2-AED fp32 re-export with dynamic batch axes. "
        + enc_meta.get("comment", ""))
    add_meta_data(enc_file, enc_meta)
    print("metadata written:", {k: v[:60] for k, v in enc_meta.items()})

    for f in (enc_file, dec_file):
        onnx.checker.check_model(f)
        print("checker ok:", f)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 8)
    main()
