import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))

from architecture import BNAIConfig, BNAILanguageModel  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from export import (  # noqa: E402
    _build_placeholder_model,
    export_model_to_bnai,
    load_bnai_metadata,
    pack_ternary,
    unpack_ternary,
)


def test_pack_unpack_round_trip_exact_multiple_of_5():
    w = np.array([-1, 0, 1, 1, -1, 0, 0, 0, 1, -1], dtype=np.int8)
    packed = pack_ternary(w)
    assert len(packed) == 2
    unpacked = unpack_ternary(packed, len(w))
    assert np.array_equal(unpacked, w)


def test_pack_unpack_round_trip_needs_padding():
    w = np.array([-1, 0, 1, 1, -1, 0, 1], dtype=np.int8)  # 7 values, not a multiple of 5
    packed = pack_ternary(w)
    assert len(packed) == 2  # ceil(7/5) = 2
    unpacked = unpack_ternary(packed, len(w))
    assert np.array_equal(unpacked, w)


def test_pack_unpack_random_round_trip():
    rng = np.random.default_rng(0)
    w = rng.integers(-1, 2, size=(37, 41)).astype(np.int8)
    packed = pack_ternary(w)
    unpacked = unpack_ternary(packed, w.size).reshape(w.shape)
    assert np.array_equal(unpacked, w)


def _tiny_config():
    return BNAIConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=4, ffn_hidden=32, context_len=32)


def _tiny_tokenizer():
    texts = ["hello world " * 5] * 20
    return BPETokenizer.train(texts, vocab_size=64)


def test_export_writes_valid_header_and_metadata(tmp_path):
    model = BNAILanguageModel(_tiny_config())
    tok = _tiny_tokenizer()
    out_path = str(tmp_path / "model.bnai")

    stats = export_model_to_bnai(model, tok, out_path)

    assert os.path.exists(out_path)
    assert os.path.exists(out_path + ".tokenizer.json")
    assert stats["param_count"] == model.num_parameters()
    assert stats["file_size_bytes"] == os.path.getsize(out_path)

    meta = load_bnai_metadata(out_path)
    assert meta["d_model"] == 16
    assert meta["n_layers"] == 2
    assert meta["n_heads"] == 4
    assert meta["ffn_hidden"] == 32
    assert meta["pack_scheme"] == "base3_5_per_byte"


def test_placeholder_model_matches_full_spec_param_budget():
    """The placeholder export must use the full spec vocab_size (32000), not
    shrink to whatever tiny vocab a toy tokenizer happens to produce -- the
    point of the placeholder artifact is to measure the real target file size."""
    model = _build_placeholder_model()
    target = 74_200_000
    assert abs(model.num_parameters() - target) / target < 0.05


def test_export_rejects_fp16_baseline_model(tmp_path):
    cfg = _tiny_config()
    cfg.ternary_weights = False
    model = BNAILanguageModel(cfg)
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError):
        export_model_to_bnai(model, tok, str(tmp_path / "model.bnai"))


def test_exported_packed_weights_match_export_ternary_source_of_truth(tmp_path):
    """Parses the raw bytes back out of the .bnai file for one projection and
    checks they match what BitLinear.export_ternary() produced -- this is the
    numerical-parity check Phase 6 calls for, done here on the Python side;
    runtime/tests mirrors it in Rust against the same file."""
    import struct

    model = BNAILanguageModel(_tiny_config())
    tok = _tiny_tokenizer()
    out_path = str(tmp_path / "model.bnai")
    export_model_to_bnai(model, tok, out_path)

    expected_w, expected_scale = model.layers[0].attn.q_proj.export_ternary()
    expected_w = expected_w.numpy()

    with open(out_path, "rb") as f:
        f.read(4)  # magic
        (version,) = struct.unpack("<B", f.read(1))
        (meta_len,) = struct.unpack("<I", f.read(4))
        f.read(meta_len)  # metadata

        cfg = _tiny_config()
        embedding_bytes = cfg.vocab_size * cfg.d_model * 2  # fp16
        f.read(embedding_bytes)

        norm_bytes = cfg.d_model * 2  # fp16
        f.read(norm_bytes)  # attn_norm for layer 0

        out_features, in_features = struct.unpack("<II", f.read(8))
        n_weights = out_features * in_features
        packed_len = (n_weights + 4) // 5
        packed_bytes = f.read(packed_len)
        (scale,) = struct.unpack("<e", f.read(2))

    actual_w = unpack_ternary(packed_bytes, n_weights).reshape(out_features, in_features)
    assert np.array_equal(actual_w, expected_w)
    assert abs(scale - expected_scale.item()) < 1e-3
