import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture import BNAIConfig, BNAILanguageModel, BitLinear, _absmean_quantize_weight


def default_config() -> BNAIConfig:
    return BNAIConfig(
        vocab_size=32000,
        d_model=768,
        n_layers=14,
        n_heads=12,
        ffn_hidden=2048,
        context_len=2048,
    )


def test_param_count_matches_budget():
    model = BNAILanguageModel(default_config())
    total = model.num_parameters()
    # Target: ~125M ("125M-class", GPT-2-small width), within +/-5%.
    # Precise value for this exact config is 123,688,704.
    target = 125_000_000
    assert abs(total - target) / target < 0.05, f"param count {total} outside +/-5% of {target}"


PARAMETER_LADDER = [
    # (stage name, vocab_size, d_model, n_heads, n_layers, ffn_hidden, target)
    ("stage1_25m", 10000, 384, 6, 12, 1024, 25_000_000),
    ("stage2_50m", 16000, 640, 10, 8, 1728, 50_000_000),
    ("stage3_75m", 32000, 576, 9, 14, 1536, 75_000_000),
    ("stage4_125m", 32000, 768, 12, 14, 2048, 125_000_000),
]


@pytest.mark.parametrize("name,vocab,d_model,n_heads,n_layers,ffn_hidden,target", PARAMETER_LADDER)
def test_parameter_ladder_stages_within_tolerance(name, vocab, d_model, n_heads, n_layers, ffn_hidden, target):
    """Locks in docs/model_card.md's 25M->50M->75M->125M ladder table --
    each stage's exact config must land within +/-5% of its target."""
    cfg = BNAIConfig(
        vocab_size=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers, ffn_hidden=ffn_hidden, context_len=2048
    )
    total = BNAILanguageModel(cfg).num_parameters()
    err = abs(total - target) / target
    assert err < 0.05, f"{name}: param count {total:,} outside +/-5% of {target:,} (err={err:.2%})"


def test_param_count_breakdown():
    cfg = default_config()
    model = BNAILanguageModel(cfg)

    embedding_params = cfg.vocab_size * cfg.d_model
    attn_params_per_layer = 4 * cfg.d_model * cfg.d_model
    ffn_params_per_layer = 3 * cfg.d_model * cfg.ffn_hidden
    norm_params_per_layer = 2 * cfg.d_model
    expected = embedding_params + cfg.n_layers * (
        attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    ) + cfg.d_model  # final norm

    actual = model.num_parameters()
    assert actual == expected


def test_head_dim_divides_evenly():
    cfg = default_config()
    assert cfg.head_dim == 64
    assert cfg.d_model == cfg.n_heads * cfg.head_dim


def test_absmean_quantize_produces_only_ternary_values():
    w = torch.randn(64, 64)
    w_ternary, scale = _absmean_quantize_weight(w)
    unique_vals = set(w_ternary.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert scale.item() > 0


def test_bitlinear_forward_shape():
    layer = BitLinear(16, 32)
    x = torch.randn(4, 10, 16)
    out = layer(x)
    assert out.shape == (4, 10, 32)


def test_bitlinear_ste_gradient_flows_to_latent_weight():
    layer = BitLinear(8, 8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_bitlinear_weight_sparsity_in_range():
    layer = BitLinear(128, 128)
    sparsity = layer.weight_sparsity()
    assert 0.0 <= sparsity <= 1.0


def test_bitlinear_export_ternary_only_has_valid_values():
    layer = BitLinear(32, 32)
    w_ternary, scale = layer.export_ternary()
    assert w_ternary.dtype == torch.int8
    assert set(w_ternary.unique().tolist()).issubset({-1, 0, 1})
    assert scale.item() > 0


def test_model_forward_produces_finite_logits_and_loss():
    cfg = BNAIConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, context_len=16)
    model = BNAILanguageModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


def test_model_backward_updates_all_bitlinear_weights():
    cfg = BNAIConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=4, ffn_hidden=32, context_len=16)
    model = BNAILanguageModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))

    _, loss = model(input_ids, targets)
    loss.backward()

    for m in model.bitlinear_modules():
        assert m.weight.grad is not None
        assert torch.isfinite(m.weight.grad).all()


def test_causality_future_tokens_do_not_affect_earlier_logits():
    cfg = BNAIConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=4, ffn_hidden=32, context_len=16)
    model = BNAILanguageModel(cfg)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_full, _ = model(input_ids)

    modified = input_ids.clone()
    modified[0, -1] = (modified[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_modified, _ = model(modified)

    # Changing the last token must not change logits at earlier positions.
    assert torch.allclose(logits_full[:, :-1], logits_modified[:, :-1], atol=1e-5)


def test_initial_loss_is_close_to_uniform_random_baseline():
    """A freshly initialized model should predict roughly uniformly over the
    vocab, i.e. initial loss ~= ln(vocab_size). A much larger loss usually
    means an init scale bug (e.g. an oversized embedding table blowing up
    logit magnitude through the tied LM head)."""
    cfg = BNAIConfig(vocab_size=200, d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, context_len=32)
    model = BNAILanguageModel(cfg)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        _, loss = model(input_ids, targets)

    uniform_baseline = math.log(cfg.vocab_size)
    assert loss.item() < uniform_baseline * 2.5, f"initial loss {loss.item():.2f} far above uniform baseline {uniform_baseline:.2f}"


def test_tied_embedding_and_lm_head_share_storage():
    model = BNAILanguageModel(default_config())
    assert model.lm_head.weight is model.embed_tokens.weight


def test_fp16_baseline_config_uses_plain_linear_not_bitlinear():
    cfg = BNAIConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=4, ffn_hidden=32, context_len=16, ternary_weights=False)
    model = BNAILanguageModel(cfg)
    assert list(model.bitlinear_modules()) == []
    assert model.mean_weight_sparsity() == 0.0

    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))
    _, loss = model(input_ids, targets)
    assert torch.isfinite(loss)

    # Same architecture/param count as the ternary model -- only the
    # projection implementation differs, isolating the quantization cost.
    ternary_model = BNAILanguageModel(BNAIConfig(**{**cfg.__dict__, "ternary_weights": True}))
    assert model.num_parameters() == ternary_model.num_parameters()


def test_context_len_overflow_raises():
    cfg = BNAIConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=4, ffn_hidden=32, context_len=8)
    model = BNAILanguageModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(input_ids)
