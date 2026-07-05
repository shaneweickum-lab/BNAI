# Benny — BNAI V1.0

A 75M-parameter, ternary-weight ("BitNet b1.58"-style) language model,
trained Chinchilla-optimally and served entirely client-side in the browser
via a from-scratch Rust→WASM inference engine. No inference server — once
the page loads, generation runs on the visitor's own machine.

The point of this project is the systems engineering, not raw model
capability: a 75M model can't compete with frontier assistants on knowledge
or reasoning, and the demo says so. What it demonstrates instead: a
quantization-aware training loop implemented from scratch (BitLinear +
straight-through estimator), a packed weight format designed around a
literal read-only-memory lookup table (hence "Ternary-ROM"), and a
dependency-light WASM engine that never dequantizes ternary weights back to
float.

**Current status: architecture, data pipeline, training/export/eval code,
the Rust WASM runtime, and the Next.js demo are built and tested end-to-end
against a small local smoke test. No real training run has happened yet —
the model currently shipped in `web/public/model/` has random, untrained
weights.** See "Status" below for exactly what's real vs. placeholder.

## Repository layout

```
model/          Python: architecture, tokenizer, data pipeline, train/sft/export/eval
runtime/        Rust, compiled to WASM: ternary inference engine + BPE tokenizer
web/            Next.js app (landing / demo / about), deployed on Vercel
docs/           model_card.md, training_log.md, benchmarks.md
```

## Quick start

```bash
# Python side (training/export/eval)
pip install -r model/requirements.txt
python -m pytest model/tests/ -q          # 34 tests, no GPU/network needed

# Rust side (WASM runtime)
cd runtime && cargo test                  # native tests
cargo build --target wasm32-unknown-unknown --release   # confirm it compiles to wasm
# wasm-pack build --target web --out-dir pkg   # produces the browser-loadable .wasm + JS glue

# Web side (demo)
cd web && npm install && npm run build && npm run dev
```

## Architecture summary

Decoder-only transformer, LLaMA-family block shape: d_model 576, 14 layers,
9 heads (head_dim 64), SwiGLU FFN (hidden 1536), RMSNorm pre-norm, RoPE, no
biases, 32k vocab, 2048 context. Every attention/FFN projection is a
`BitLinear` layer — weights are quantized to {-1,0,+1} via absmean scaling
during the forward pass, trained with a straight-through estimator so
gradients flow to a full-precision latent weight as if quantization were the
identity. The tied embedding/LM-head table stays fp16 (it's a lookup, not a
matmul — quantizing it buys no speed, costs quality). Full details:
`docs/model_card.md`.

**Measured parameter count: 74,187,072** (`model/tests/test_architecture.py`
verifies this against the ~74.2M target, ±5%).

## Retrain / re-export / redeploy, end to end

This is the full path from empty checkpoints to a live demo with a real,
trained model. Every step below is a script in this repo — no step is a
"figure it out yourself" gap.

### 0. Where this needs to run

Real training (Phase 3/4/5 below) needs a machine with enough compute to
push ~1.5B+ tokens through a 75M-param model in a reasonable number of
sessions — this repo was built and spec'd against a MacBook Pro M5 (24GB
unified memory, PyTorch MPS backend, no CUDA). All the code auto-detects
`cuda` → `mps` → `cpu` and runs on any of them, but CPU-only will be far too
slow for the real budget — it's only meant for the small local smoke tests
below. See "Hardware notes" for the M5-specific throughput expectations.

### 1. Tokenizer

Train fresh on a representative sample of your actual pretraining corpus
(don't reuse the placeholder tokenizer already in the repo — it was trained
on synthetic text purely to exercise the pipeline):

```bash
python model/tokenizer/train_tokenizer.py \
    --hf-dataset HuggingFaceFW/fineweb-edu --hf-split train --max-docs 200000 \
    --vocab-size 32000 --out model/tokenizer/bnai_tokenizer.json
```

(Or `--input "path/to/*.jsonl"` for a local corpus.) Confirm FineWeb-Edu's
current license terms before downloading at scale — see `docs/model_card.md`.

### 2. Phase 3 smoke test — do this before committing to a multi-day run

```bash
python model/train.py --config model/configs/base_ternary.yaml \
    --tokenizer model/tokenizer/bnai_tokenizer.json \
    --smoke-test-steps 200
```

Watch the printed `tok/s`. Compare against the measured fp16 baseline of
5,600 tok/s on the M5 and the scenario ladder in `model/configs/base_ternary.yaml`'s
comments. Record the result in `docs/training_log.md`. If the resulting
full-run wall-clock estimate is impractical, use the fallback ladder there
(trim context/width, reduce token budget and document it, or move to rented
cloud compute) rather than silently changing the recipe.

### 3. Full base pretrain (+ fp16 baseline for comparison)

```bash
python model/train.py --config model/configs/base_ternary.yaml \
    --tokenizer model/tokenizer/bnai_tokenizer.json

python model/train.py --config model/configs/base_fp16_baseline.yaml \
    --tokenizer model/tokenizer/bnai_tokenizer.json
```

Both checkpoint frequently (`checkpoint_interval_steps` in the config) and
resume with `--resume` — training is designed as a series of resumable
sessions, not one uninterrupted run, since a laptop will sleep/thermal
throttle/need to be used for other things mid-run. Loss, val perplexity,
grad norm, weight sparsity, LR, and tokens/sec are logged every
`log_interval_steps` to `checkpoints/*/training_log.jsonl`.

### 4. SFT (Stage B)

```bash
python model/sft.py --config model/configs/sft.yaml \
    --tokenizer model/tokenizer/bnai_tokenizer.json \
    --base-checkpoint checkpoints/base_ternary/latest.pt
```

Watches base-domain held-out perplexity throughout (the "alignment tax"
check) — if it regresses more than `max_base_ppl_regression` in
`configs/sft.yaml`, reduce SFT epochs/LR rather than continuing blindly.

### 5. Evaluate

```bash
python model/eval.py --checkpoint checkpoints/sft/latest.pt \
    --tokenizer model/tokenizer/bnai_tokenizer.json \
    --compare-fp16 checkpoints/base_fp16_baseline/latest.pt \
    --benchmark-inference
```

Fill the results into `docs/benchmarks.md` and `docs/model_card.md`.

### 6. Export the real model for the web demo

```bash
python model/export.py --checkpoint checkpoints/sft/latest.pt \
    --tokenizer model/tokenizer/bnai_tokenizer.json \
    --out web/public/model/benny.bnai
```

This replaces the placeholder `benny-placeholder.bnai` with the real,
trained artifact. Update `web/` to point at the new filename (and update the
size/param-count figures across `web/app/*` and `docs/*` — they're currently
the placeholder's measured numbers, not invented ones, but they'll change
once this is a real trained model with a real weight distribution... though
note the *packed size* is fixed by the format regardless of weight values,
so 48.05MB / 74.2M params will stay accurate unless you change the
architecture).

### 7. Build the WASM runtime and deploy

```bash
cd runtime && wasm-pack build --target web --out-dir ../web/public/wasm
cd ../web && npm run build
```

Then deploy `web/` to Vercel (connect the repo, or `vercel deploy` from
`web/`) — there's no inference API route, so Vercel only serves static
assets (the Next.js app, the WASM binary, the packed model file).

## Hardware notes (base pretrain / SFT, spec-locked target: MacBook Pro M5)

- **Backend**: PyTorch MPS, not CUDA. `train.py`/`sft.py` auto-detect
  `cuda → mps → cpu`. Before trusting MPS throughput, run the Phase 3 smoke
  test above — some quantization/STE ops can silently fall back to CPU on
  MPS and quietly wreck throughput; watch for that rather than assuming it
  away.
- **Memory**: 24GB is shared across OS/CPU/GPU. The configs use gradient
  accumulation (`micro_batch_size` × `grad_accum_steps` to reach
  `effective_batch_tokens`) rather than one large batch — tune
  `micro_batch_size` down in the config if you hit OOM.
- **Throughput / wall-clock**: measured fp16 baseline on the M5 is 5,600
  tok/s. Ternary training adds quantization overhead per forward/backward
  pass that fp16 doesn't have, and MPS lacks CUDA's fused ternary-training
  kernels — expect somewhere between ~2,800 and ~5,000 tok/s depending on
  overhead, i.e. roughly 3.5-6 days of *continuous* compute for the full
  1.5B-token budget, and correspondingly more real calendar time given a
  laptop isn't run at 100% utilization nonstop. Measure, don't assume — the
  Phase 3 smoke test above gives you the real number for this machine.
- **No distributed training** — single-device only, by design; there's no
  multi-GPU scaffolding to configure or debug.

## What's real vs. placeholder right now

| Piece | Status |
|---|---|
| Architecture + params (74.2M) | Real, tested (`model/tests/`) |
| Tokenizer algorithm | Real, tested |
| Tokenizer *vocabulary* shipped in this repo | Placeholder — trained on synthetic text |
| Training/SFT/export/eval code | Real, tested end-to-end on tiny local smoke tests |
| Actual trained model weights | **Not real yet** — `web/public/model/benny-placeholder.bnai` has random weights |
| Packed file size (48.05MB) / compression ratio (~3.1x) | Real measurement (packing is weight-value-independent) |
| Rust WASM runtime | See `runtime/` — built against the placeholder artifact |
| Web demo | See `web/` — built against the placeholder artifact and a documented worker interface |
| Any loss/perplexity/eval/benchmark number | **TBD** — needs the real training run |

## Open questions to confirm before/at real-training time (spec Section 11)

- Measured ternary tok/s from the Phase 3 smoke test on the actual M5,
  compared against the estimated ranges above.
- Current license terms for FineWeb-Edu, UltraChat 200k, and OASST2 — verify
  before downloading and training on them at scale.
- Mobile Safari's real KV-cache memory ceiling at `context_len=2048` — may
  push the *served* context window below the *trained* one; test on a real
  iOS device, not a simulator.
