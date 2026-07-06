"""Ternary-ROM transformer architecture (BNAI / "Benny").

Decoder-only, LLaMA-family block shape, with BitLinear (ternary {-1,0,+1}
weight) projections in place of dense nn.Linear for attention and FFN. See
docs/model_card.md for the full spec this implements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BNAIConfig:
    # ~123.7M params (marketed as "125M"): d_model=768/n_heads=12 matches
    # GPT-2-small's width, head_dim=64 and n_layers=14 carried over from the
    # original ~74.2M design. See docs/model_card.md for the compute-scaling
    # reasoning behind this size (Chinchilla token budget x per-token compute
    # scale as params^2, not params, so this isn't just "make it bigger").
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 14
    n_heads: int = 12
    ffn_hidden: int = 2048
    context_len: int = 2048
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    act_bits: int = 8  # activation quantization bit-width inside BitLinear
    ternary_weights: bool = True  # False builds the fp16 baseline (Section 6) with plain nn.Linear

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def _absmean_quantize_weight(w: torch.Tensor, eps: float = 1e-5):
    """BitNet b1.58 weight quantization: absmean-scaled round-clip to {-1,0,1}.

    Returns (ternary_weight_in_float, scale) where ternary_weight_in_float
    contains only values in {-1, 0, 1} and scale is the scalar such that
    w ~= ternary_weight_in_float * scale.
    """
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_ternary = w_scaled.round().clamp(-1, 1)
    return w_ternary, scale


def _absmax_quantize_activation(x: torch.Tensor, bits: int = 8, eps: float = 1e-5):
    """Per-token int8 (or n-bit) absmax activation quantization (BitNet b1.58)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / qmax
    x_q = (x / scale).round().clamp(-qmax - 1, qmax)
    return x_q, scale


class _RoundSTE(torch.autograd.Function):
    """Straight-through estimator: identity gradient through a non-differentiable op."""

    @staticmethod
    def forward(ctx, x, quantized):
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _ste(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x, quantized)


class BitLinear(nn.Module):
    """Linear layer with ternary {-1,0,+1} weights and int8 activations.

    Training: maintains a latent full-precision (fp32 param, trained in bf16
    autocast) shadow weight. The forward pass quantizes weights to ternary via
    absmean scaling and activations to int8 via per-token absmax scaling, with
    a straight-through estimator so gradients flow to the latent weights as if
    quantization were the identity.

    Inference (post-export): this module is not used directly -- the exported
    `.bnai` artifact stores only the ternary weights + one scale per layer, and
    `runtime/` implements the matmul as accumulate/select against that table.
    """

    def __init__(self, in_features: int, out_features: int, act_bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=in_features ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary, w_scale = _absmean_quantize_weight(self.weight)
        w_q = _ste(self.weight, w_ternary)

        x_q_int, x_scale = _absmax_quantize_activation(x, bits=self.act_bits)
        x_q = _ste(x, x_q_int)

        out = F.linear(x_q, w_q)
        return out * (w_scale * x_scale)

    def weight_sparsity(self) -> float:
        """Fraction of weights that quantize to exactly 0 -- logged during training."""
        with torch.no_grad():
            w_ternary, _ = _absmean_quantize_weight(self.weight)
            return (w_ternary == 0).float().mean().item()

    def export_ternary(self):
        """Returns (int8 tensor in {-1,0,1}, fp32 scalar scale) for the export step."""
        with torch.no_grad():
            w_ternary, w_scale = _absmean_quantize_weight(self.weight)
            return w_ternary.to(torch.int8), w_scale.detach()


def make_projection(cfg: BNAIConfig, in_features: int, out_features: int) -> nn.Module:
    """BitLinear for the ternary model, or a plain bias-free nn.Linear for the
    fp16 baseline (Section 6) -- same architecture and training recipe
    otherwise, so the comparison isolates the cost of ternary quantization."""
    if cfg.ternary_weights:
        return BitLinear(in_features, out_features, cfg.act_bits)
    linear = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(linear.weight, mean=0.0, std=in_features ** -0.5)
    return linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, n_heads, seq_len, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BNAIConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = make_projection(cfg, cfg.d_model, cfg.d_model)
        self.k_proj = make_projection(cfg, cfg.d_model, cfg.d_model)
        self.v_proj = make_projection(cfg, cfg.d_model, cfg.d_model)
        self.o_proj = make_projection(cfg, cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(out)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg: BNAIConfig):
        super().__init__()
        self.gate_proj = make_projection(cfg, cfg.d_model, cfg.ffn_hidden)
        self.up_proj = make_projection(cfg, cfg.d_model, cfg.ffn_hidden)
        self.down_proj = make_projection(cfg, cfg.ffn_hidden, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BNAIConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLUFeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BNAILanguageModel(nn.Module):
    """Ternary-ROM SLM. Embedding and LM head are tied and kept at full precision."""

    def __init__(self, cfg: BNAIConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # nn.Embedding defaults to std=1, which is far too large here: this
        # table is tied to the LM head, so an oversized init blows up logit
        # magnitude (and therefore initial loss) through the output matmul.
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=cfg.d_model ** -0.5)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        # Weight tying: lm_head.weight is the same tensor as embed_tokens.weight.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self._rope_cache_len = 0
        self._cos = None
        self._sin = None

    def _rope(self, seq_len: int, device, dtype):
        if self._cos is None or self._rope_cache_len < seq_len or self._cos.device != device:
            cos, sin = build_rope_cache(seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype)
            self._cos, self._sin = cos, sin
            self._rope_cache_len = seq_len
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = input_ids.shape
        if t > self.cfg.context_len:
            raise ValueError(f"sequence length {t} exceeds context_len {self.cfg.context_len}")

        x = self.embed_tokens(input_ids)
        cos, sin = self._rope(t, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        total = 0
        seen = set()
        for p in self.parameters():
            if id(p) in seen:  # skip tied lm_head.weight double count
                continue
            seen.add(id(p))
            if exclude_embedding and p is self.embed_tokens.weight:
                continue
            total += p.numel()
        return total

    def bitlinear_modules(self):
        for layer in self.layers:
            for m in (
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                layer.attn.o_proj,
                layer.ffn.gate_proj,
                layer.ffn.up_proj,
                layer.ffn.down_proj,
            ):
                if isinstance(m, BitLinear):
                    yield m

    def mean_weight_sparsity(self) -> float:
        """Only meaningful for the ternary model; 0.0 for the fp16 baseline
        (Section 5's per-step diagnostic logging)."""
        vals = [m.weight_sparsity() for m in self.bitlinear_modules()]
        return sum(vals) / len(vals) if vals else 0.0
