"""Export trained latent weights to the packed `.bnai` ternary artifact
(spec Section 7). This is the format `runtime/src/packed_format.rs` parses --
see that file's module doc for the Rust-side mirror of this contract.

File layout (little-endian throughout):
    magic            4 bytes   b"BNAI"
    version          u8
    metadata_len     u32
    metadata         `metadata_len` bytes of UTF-8 JSON (architecture
                     hyperparams + param count + this format's pack scheme)
    embedding table  vocab_size * d_model fp16 values, row-major
                     (tied with the LM head -- kept dense/fp16, never
                     quantized, since it's a lookup not a matmul)
    per layer, repeated n_layers times, each projection in this fixed order
    [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]:
        attn_norm weight (once per layer, before the 4 attn projections)
        for each of q/k/v/o: out_features u32, in_features u32,
                              packed ternary bytes, fp16 scale
        ffn_norm weight (once per layer, before the 3 ffn projections)
        for each of gate/up/down: out_features u32, in_features u32,
                                   packed ternary bytes, fp16 scale
    final_norm weight (once, at the very end)

Ternary packing: 5 values per byte in base-3 -- `byte = t0 + 3*t1 + 9*t2 +
27*t3 + 81*t4`, each ti in {0,1,2} mapping to weight {-1,0,+1} (ti = weight+1).
Denser than the naive 2-bit/value packing (1.6 bits/weight vs 2 bits/weight)
at the cost of needing a 256-entry lookup table to unpack -- which is exactly
the "ROM" the architecture is named for: unpacking a byte is a single table
read, not arithmetic.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

from architecture import BNAIConfig, BNAILanguageModel, BitLinear  # noqa: E402
from bpe import BPETokenizer  # noqa: E402

MAGIC = b"BNAI"
FORMAT_VERSION = 1
PROJECTION_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _build_unpack_lut() -> np.ndarray:
    lut = np.zeros((256, 5), dtype=np.int8)
    for byte in range(256):
        v = byte
        for i in range(5):
            lut[byte, i] = (v % 3) - 1
            v //= 3
    return lut


_UNPACK_LUT = _build_unpack_lut()


def pack_ternary(w_ternary: np.ndarray) -> bytes:
    """w_ternary: any-shape int array with values in {-1,0,1}. Returns the
    5-per-byte base-3 packed bytes (padded with ternary-0 up to a multiple of 5)."""
    flat = w_ternary.reshape(-1).astype(np.int16)
    digits = (flat + 1).astype(np.uint16)  # {-1,0,1} -> {0,1,2}
    pad = (-len(digits)) % 5
    if pad:
        digits = np.concatenate([digits, np.full(pad, 1, dtype=np.uint16)])  # pad -> ternary 0
    digits = digits.reshape(-1, 5)
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
    packed = (digits * powers).sum(axis=1).astype(np.uint8)
    return packed.tobytes()


def unpack_ternary(data: bytes, n: int) -> np.ndarray:
    """Inverse of pack_ternary: returns the first `n` int8 values in {-1,0,1}."""
    byte_arr = np.frombuffer(data, dtype=np.uint8)
    unpacked = _UNPACK_LUT[byte_arr].reshape(-1)
    return unpacked[:n]


def _write_fp16_array(f, arr: np.ndarray):
    f.write(arr.astype(np.float16).tobytes())


def _write_scale(f, scale: float):
    f.write(struct.pack("<e", float(scale)))


def export_model_to_bnai(model: BNAILanguageModel, tokenizer: BPETokenizer, out_path: str) -> dict:
    """Writes `out_path` (.bnai) plus a sibling `<out_path>.tokenizer.json`.
    Returns a small dict of summary stats (param_count, file_size_bytes) that
    callers (train.py, the CLI below) log/print."""
    if not model.cfg.ternary_weights:
        raise ValueError("export_model_to_bnai requires a ternary model (cfg.ternary_weights=True)")

    model.eval()
    cfg = model.cfg
    metadata = {
        "format_version": FORMAT_VERSION,
        "pack_scheme": "base3_5_per_byte",
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "head_dim": cfg.head_dim,
        "ffn_hidden": cfg.ffn_hidden,
        "context_len": cfg.context_len,
        "rope_theta": cfg.rope_theta,
        "rms_eps": cfg.rms_eps,
        "param_count": model.num_parameters(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", FORMAT_VERSION))
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)

        with torch.no_grad():
            embedding = model.embed_tokens.weight.detach().cpu().numpy()
            _write_fp16_array(f, embedding)

            for layer in model.layers:
                _write_fp16_array(f, layer.attn_norm.weight.detach().cpu().numpy())
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    module: BitLinear = getattr(layer.attn, name)
                    w_ternary, scale = module.export_ternary()
                    w_np = w_ternary.cpu().numpy()
                    f.write(struct.pack("<II", module.out_features, module.in_features))
                    f.write(pack_ternary(w_np))
                    _write_scale(f, scale.item())

                _write_fp16_array(f, layer.ffn_norm.weight.detach().cpu().numpy())
                for name in ("gate_proj", "up_proj", "down_proj"):
                    module: BitLinear = getattr(layer.ffn, name)
                    w_ternary, scale = module.export_ternary()
                    w_np = w_ternary.cpu().numpy()
                    f.write(struct.pack("<II", module.out_features, module.in_features))
                    f.write(pack_ternary(w_np))
                    _write_scale(f, scale.item())

            _write_fp16_array(f, model.final_norm.weight.detach().cpu().numpy())

    tokenizer.save(out_path + ".tokenizer.json")

    file_size = os.path.getsize(out_path)
    return {"param_count": metadata["param_count"], "file_size_bytes": file_size}


def load_bnai_metadata(path: str) -> dict:
    """Reads just the header+metadata -- used by eval.py / the web app build
    step to report packed size and param count without loading full weights."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"not a .bnai file (bad magic): {path}")
        (version,) = struct.unpack("<B", f.read(1))
        (meta_len,) = struct.unpack("<I", f.read(4))
        metadata = json.loads(f.read(meta_len).decode("utf-8"))
        metadata["format_version_on_disk"] = version
        return metadata


def _build_placeholder_model(vocab_size_override: int | None = None) -> BNAILanguageModel:
    # Uses BNAIConfig's own defaults for every dim except vocab_size, so this
    # never drifts out of sync with the real spec'd architecture.
    cfg = BNAIConfig(ternary_weights=True)
    if vocab_size_override is not None:
        cfg.vocab_size = vocab_size_override
    return BNAILanguageModel(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="path to a train.py/sft.py .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="path to the trained tokenizer JSON")
    parser.add_argument("--out", required=True, help="output .bnai path")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="export a randomly-initialized placeholder artifact (no --checkpoint needed) "
        "-- used to exercise the runtime/web pipeline before real training completes",
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)

    if args.placeholder:
        print("[export] WARNING: exporting a PLACEHOLDER model with random (untrained) weights.")
        # Full spec vocab_size (32000) regardless of the placeholder tokenizer's
        # actual learned vocab -- the point is to measure the real target file
        # size/param count, not shrink the model to match a toy tokenizer.
        model = _build_placeholder_model()
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --placeholder is set")
        payload = torch.load(args.checkpoint, map_location="cpu")
        model_cfg = BNAIConfig(**payload["config"]["model"])
        model = BNAILanguageModel(model_cfg)
        model.load_state_dict(payload["model_state_dict"])

    stats = export_model_to_bnai(model, tokenizer, args.out)
    print(
        f"[export] wrote {args.out} "
        f"({stats['file_size_bytes'] / 1e6:.2f} MB packed, {stats['param_count']:,} params)"
    )


if __name__ == "__main__":
    main()
