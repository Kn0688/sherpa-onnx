#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
"""Export FireRedASR2-AED encoder/decoder to ONNX with dynamic batch axes.

Unlike the originally released FireRedASR2 models (which hard-code batch=1 in
the decoder and do not mask the encoder with x_len), this script exports:

encoder.onnx
  in : x (N,T,80) f32 [CMVN-normalized fbank], x_len (N,) i64
  out: n_layer_cross_k (16,N,Tc,1280), n_layer_cross_v (16,N,Tc,1280),
       enc_mask (N,Tc) f32, 1.0 = valid frame, 0.0 = padding

decoder.onnx
  in : tokens (N,1) i64,
       in_n_layer_self_k_cache / in_n_layer_self_v_cache (16,N,S,20,64),
       n_layer_cross_k / n_layer_cross_v (16,N,Tc,1280),
       offset (N,) i64,
       cross_mask (N,Tc) f32 (same layout as enc_mask; masked positions get a
       -1e30 additive bias before the cross-attention softmax)
  out: logits (N,1,8667), out_n_layer_self_k_cache, out_n_layer_self_v_cache

Metadata (on encoder): num_decoder_layers, num_head, head_dim, sos, eos,
max_len, cmvn_mean, cmvn_inv_stddev — see
sherpa-onnx/csrc/offline-fire-red-asr-model.cc InitEncoder.

Wrappers re-implement the official forward in an export-safe way:
- vectorized padding mask (the official code builds it with a python loop)
- true KV cache with write-at-offset + arange<=offset key mask (the official
  "cache" stores per-layer outputs and recomputes k/v; equivalent because
  positions are causal/frozen)
- cross-attention bias is ADDITIVE, not masked_fill: a second Where node with
  a symbolic shape confuses onnxruntime's buffer-reuse planner
  ("Shape mismatch attempting to re-use buffer")

Usage:
  pip install torch onnx onnxscript kaldiio
  git clone https://github.com/FireRedTeam/FireRedASR2S
  # download FireRedASR2-AED weights (model.pth.tar, cmvn.ark, dict.txt)
  python3 ./scripts/fire-red-asr/export-onnx.py \
    --repo ./FireRedASR2S --model-dir ./FireRedASR2-AED --output-dir ./
"""
import argparse
import math
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

NUM_LAYERS = 16
NUM_HEAD = 20
HEAD_DIM = 64
D_MODEL = 1280


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="path to the FireRedASR2S git repo")
    p.add_argument("--model-dir", required=True,
                   help="dir containing model.pth.tar, cmvn.ark, dict.txt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


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
        # (N,1,Tc) bool -> (N,Tc) float, 1.0 = valid frame, 0.0 = padding
        enc_mask = sub_mask.squeeze(1).to(h.dtype)
        return cross_k, cross_v, enc_mask


class FireRedDecoderWrapper(nn.Module):
    """Single greedy step with true KV cache (write at offset)."""

    def __init__(self, model):
        super().__init__()
        self.dec = model.decoder

    def forward(self, tokens, in_self_k, in_self_v, cross_k, cross_v, offset,
                cross_mask):
        # tokens: (N,1) i64; in_self_k/v: (L,N,S,H,d) f32;
        # cross_k/v: (L,N,Tc,D) f32; offset: (N,) i64;
        # cross_mask: (N,Tc) f32, 1.0 = valid encoder frame, 0.0 = padding
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

        # (N,Tc) -> (N,1,1,Tc) additive bias: 0 for valid, -1e30 for padding.
        # NOTE: additive bias instead of masked_fill; see the module docstring.
        cross_bias = (cross_mask - 1.0)[:, None, None, :] * 1e30

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

            # --- cross attention (masked via additive bias) ---
            residual = h
            x = layer.cross_attn_norm(h)
            q = layer.cross_attn.w_qs(x).view(N, 1, H, d).transpose(1, 2)
            ck = cross_k[i].view(N, -1, H, d).transpose(1, 2)  # (N,H,Tc,d)
            cv = cross_v[i].view(N, -1, H, d).transpose(1, 2)
            attn = torch.matmul(q, ck.transpose(2, 3)) / (d ** 0.5)
            attn = attn + cross_bias
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
    """Edit metadata WITHOUT re-serializing tensor data.

    Loading an external-data model and re-saving it with
    save_as_external_data=True duplicates every tensor in the .data file.
    Loading with load_external_data=False keeps the external references
    intact, and a proto-only save updates just the metadata.
    """
    model = onnx.load(filename, load_external_data=False)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


def read_cmvn(kaldi_cmvn_file):
    """Same math as fireredasr2s/fireredasr2/data/asr_feat.py CMVN."""
    import kaldiio

    stats = kaldiio.load_mat(kaldi_cmvn_file)
    assert stats.shape[0] == 2
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    assert count >= 1
    floor = 1e-20
    means = []
    inv_stddev = []
    for d in range(dim):
        mean = stats[0, d] / count
        means.append(mean.item())
        var = (stats[1, d] / count) - mean * mean
        if var < floor:
            var = floor
        inv_stddev.append(1.0 / math.sqrt(var))
    return means, inv_stddev


@torch.no_grad()
def main():
    args = get_args()
    sys.path.insert(0, os.path.join(os.path.abspath(args.repo), "fireredasr2s"))
    from fireredasr2.asr import load_fireredasr_aed_model

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_fireredasr_aed_model(
        os.path.join(args.model_dir, "model.pth.tar"))
    model.eval()

    enc = FireRedEncoderWrapper(model).eval()
    dec = FireRedDecoderWrapper(model).eval()

    # ---- dummy inputs ----
    N, T = 2, 100
    x = torch.randn(N, T, 80)
    x_len = torch.tensor([T, T - 30], dtype=torch.int64)

    cross_k, cross_v, enc_mask = enc(x, x_len)
    Tc = cross_k.shape[2]
    assert cross_k.shape == (NUM_LAYERS, N, Tc, D_MODEL)
    assert enc_mask.shape == (N, Tc)

    S = 32
    tokens = torch.tensor([[3], [3]], dtype=torch.int64)
    offset = torch.zeros(N, dtype=torch.int64)
    in_k = torch.zeros(NUM_LAYERS, N, S, NUM_HEAD, HEAD_DIM)
    in_v = torch.zeros_like(in_k)
    logits, out_k, out_v = dec(tokens, in_k, in_v, cross_k, cross_v, offset,
                               enc_mask)
    assert logits.shape == (N, 1, 8667)
    assert out_k.shape == in_k.shape

    export_kw = dict(opset_version=args.opset, do_constant_folding=True,
                     dynamo=True)

    enc_file = os.path.join(args.output_dir, "encoder.onnx")
    torch.onnx.export(
        enc, (x, x_len), enc_file,
        input_names=["x", "x_len"],
        output_names=["n_layer_cross_k", "n_layer_cross_v", "enc_mask"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_len": {0: "N"},
            "n_layer_cross_k": {1: "N", 2: "T"},
            "n_layer_cross_v": {1: "N", 2: "T"},
            "enc_mask": {0: "N", 1: "T"},
        },
        **export_kw,
    )
    print("saved", enc_file)

    dec_file = os.path.join(args.output_dir, "decoder.onnx")
    torch.onnx.export(
        dec,
        (tokens, in_k, in_v, cross_k, cross_v, offset, enc_mask),
        dec_file,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
            "cross_mask",
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
            "cross_mask": {0: "N", 1: "T"},
            "logits": {0: "N"},
            "out_n_layer_self_k_cache": {1: "N", 2: "T"},
            "out_n_layer_self_v_cache": {1: "N", 2: "T"},
        },
        **export_kw,
    )
    print("saved", dec_file)

    mean, inv_stddev = read_cmvn(os.path.join(args.model_dir, "cmvn.ark"))
    enc_meta = {
        "model_type": "fire-red-asr-aed",
        "version": "2",
        "model_author": "FireRedTeam",
        "maintainer": "k2-fsa",
        "feat_dim": 80,
        "cmvn_mean": ",".join(map(str, mean)),
        "cmvn_inv_stddev": ",".join(map(str, inv_stddev)),
        "num_decoder_layers": NUM_LAYERS,
        "num_head": NUM_HEAD,
        "head_dim": HEAD_DIM,
        "max_len": 1024,
        "sos": 3,
        "eos": 4,
        "url": "https://github.com/FireRedTeam/FireRedASR2S",
        "comment": "FireRedASR2-AED re-export with dynamic batch axes, "
                   "x_len-masked encoder, and decoder cross-attention mask",
    }
    add_meta_data(enc_file, enc_meta)
    print("metadata written")

    for f in (enc_file, dec_file):
        onnx.checker.check_model(f)
        print("checker ok:", f)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 8)
    main()
