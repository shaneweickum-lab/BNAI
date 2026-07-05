# Model Card — Benny (BNAI V1.0)

**Status: architecture and pipeline complete; no real training run has
happened yet.** Every number below that depends on actual training (loss,
perplexity, benchmark scores, ternary-vs-fp16 comparison) is marked TBD. The
only measured numbers here are the ones that don't require training: param
count and packed file size, both taken from the placeholder export in this
repo (`web/public/model/benny-placeholder.bnai`).

## What this is

A 75M-parameter decoder-only transformer with ternary ({-1, 0, +1}) weights
in every attention and feed-forward projection ("BitLinear", following the
BitNet b1.58 approach), trained compute-optimally per the Chinchilla scaling
law (~20 tokens/parameter) and served entirely client-side via a from-scratch
Rust→WASM inference engine — no inference server, no API calls, generation
runs on the visitor's own machine.

The engineering point of this project isn't raw model capability (a 75M
model cannot compete with frontier assistants on knowledge or reasoning) —
it's the systems work: quantization-aware training implemented from scratch,
a dependency-light WASM runtime that never dequantizes ternary weights back
to float, and honest measurement of what that buys you.

## Architecture

| Hyperparameter | Value |
|---|---|
| d_model | 768 |
| n_layers | 14 |
| n_heads | 12 (head_dim 64) |
| attention | causal multi-head self-attention, RoPE, no bias |
| feed-forward | SwiGLU, hidden dim 2048 |
| normalization | RMSNorm, pre-norm |
| vocab size | 32,000 (from-scratch byte-level BPE, see below) |
| context length | 2048 tokens |
| weight tying | input embedding == output (LM head) projection |
| weight precision | ternary {-1,0,+1} via BitLinear; embedding/LM-head excluded |
| activation precision | int8 per-token (absmax) during training; plain float at inference (see runtime/) |

**Measured parameter count: 123,688,704** ("125M-class" — `d_model`=768 /
`n_heads`=12 deliberately matches GPT-2-small's width; `head_dim`=64 and
`n_layers`=14 carried over from the original ~74.2M design point). See
`model/tests/test_architecture.py::test_param_count_matches_budget` and
`docs/training_log.md`'s note on why this size and not bigger: Chinchilla
token budget (`20 × params`) combined with per-token compute that's also
roughly linear in `params` means total training compute scales as
`params²` — this size already costs ~2.8x the compute of the original 75M
design point, and going much bigger trades away laptop-training
feasibility for a bigger number that the engineering story doesn't need.

## Tokenizer

A fresh byte-level BPE tokenizer (`model/tokenizer/bpe.py`), trained on a
sample of the actual pretraining corpus rather than reusing an existing
(likely oversized) vocabulary — see spec Section 4's reasoning: an inherited
vocab wastes embedding-table capacity on a 75M model. Includes chat special
tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) from the start.
GPT-2-style byte-to-unicode mapping guarantees full UTF-8 coverage with no
`<unk>` fallback for raw bytes.

**The tokenizer currently shipped (`model/tokenizer/bnai_tokenizer.json`) is
trained on a small synthetic placeholder corpus, not the real pretraining
data.** Retrain it on the real corpus per the README before a real training
run.

## Packed ternary format (`.bnai`)

Documented in full in `model/export.py`'s module docstring and mirrored in
`runtime/src/packed_format.rs`. Short version: ternary weights are packed
5-per-byte in base-3 (1.6 bits/weight); the embedding/LM-head table stays
dense fp16 (it's a lookup, not a matmul, so quantizing it buys no inference
speedup and would cost quality). Unpacking a byte is a single 256-entry
lookup-table read — this is the "ROM" the project is named for.

**Measured, on the placeholder (random-weight) export at full architecture:**

- Packed size: **69.02 MB**
- fp16-equivalent size (if nothing were quantized): ~247.4 MB
- **Measured compression ratio: ~3.58x** — meaningfully below a naive
  "1.58 bits vs 16 bits ≈ 10x" intuition, or even the 5-8x that intuition
  might get discounted to, though slightly better than the ~3.1x measured at
  the original 74.2M design point. The reason: the tied embedding table is
  ~24.6M of the 123.7M total params (~19.9%) and stays fp16, so only the
  remaining ~80% (the attention/FFN projections) compress ~10x; overall
  ratio works out to `1 / (0.199 + 0.801 * (1.6/16)) ≈ 3.58x`. Widening
  `d_model` (embedding grows linearly with it) while attention/FFN params
  grow quadratically with it is *why* the ratio improves at this larger
  size — a real, measured, architecture-dependent effect, not a
  coincidence. This is a real tradeoff (spec Section 2 explicitly keeps the
  embedding at higher precision for quality), not a packing-format bug —
  stated here per the spec's own instruction to measure and report the real
  number rather than assume the optimistic one.

This also matters for the web demo's cold-load story (Section 7): 48MB is a
real download, not a trivial one, especially on mobile networks — the demo
UI shows real download progress rather than downplaying this.

## Training recipe (Chinchilla-optimal, Stage A + Stage B)

See `model/configs/base_ternary.yaml` / `base_fp16_baseline.yaml` /
`sft.yaml` for exact hyperparameters. Summary:

- **Stage A (base pretrain):** ~2.47B tokens (20 tokens/param), FineWeb-Edu,
  AdamW (β 0.9/0.95, wd 0.1), peak LR 3e-4 with linear warmup (2%) + cosine
  decay to 10%, gradient clipping at 1.0, bf16 mixed precision.
- **Stage B (SFT):** UltraChat-200k subset + OASST2, ~65M tokens (not
  counted in the Chinchilla budget), lower LR (5e-5), loss masked to
  assistant turns only, monitored for "alignment tax" (regression on Stage
  A's held-out perplexity).
- **fp16 baseline:** identical architecture/recipe with `ternary_weights:
  false` (`base_fp16_baseline.yaml`), trained on the same data/token budget,
  to isolate the cost of quantization from every other variable.

**Training has not run yet** (see README for why — this environment has no
GPU/MPS; the real run happens on the target M5 MacBook laptop). All
loss/perplexity/benchmark numbers below are TBD until then.

## Evaluation (TBD — pending real training)

| Metric | Ternary | fp16 baseline | Target |
|---|---|---|---|
| Held-out validation perplexity | TBD | TBD | ternary within 10-15% of fp16 |
| Reduced HellaSwag/LAMBADA-style accuracy | TBD | TBD | modest, stated honestly |
| Inference tokens/sec (this repo's Python/PyTorch reference bench) | TBD | TBD | — |
| Inference tokens/sec (real WASM runtime, browser) | TBD | n/a (WASM engine is ternary-only) | faster than fp16 WASM baseline |

Run `model/eval.py` against real checkpoints to fill this in (see README).

## Known limitations

- Small-model reasoning/knowledge limits — this is not a frontier-scale
  assistant and the demo UI says so explicitly.
- Context length capped at 2048 tokens trained / possibly less as served,
  pending the mobile-Safari KV-cache memory check (spec Section 11).
- The shipped tokenizer and model weights in this repo, until a real
  training run completes, are both placeholders (synthetic corpus /
  untrained random weights respectively) — used only to prove the
  export → WASM runtime → web demo pipeline end-to-end.

## Dataset licensing (confirm at real-training time)

- FineWeb-Edu — Open Data Commons Attribution license (ODC-By) at time of
  writing this spec; reconfirm before use.
- UltraChat 200k — permissive license at time of writing; reconfirm.
- OASST2 (OpenAssistant) — permissive (Apache 2.0) at time of writing;
  reconfirm.

Re-verify all three at implementation/training time — license terms and
dataset availability can change.
