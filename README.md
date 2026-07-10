# Benny — BNAI V1.0

A 125M-parameter (123.7M measured), ternary-weight ("BitNet b1.58"-style) language model,
trained Chinchilla-optimally and served entirely client-side in the browser
via a from-scratch Rust→WASM inference engine. No inference server — once
the page loads, generation runs on the visitor's own machine.

The point of this project is the systems engineering, not raw model
capability: a 125M model can't compete with frontier assistants on knowledge
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
model/          Python: architecture, tokenizer, data pipeline, train/phase2/sft/export/eval
aiml/           Deterministic pattern-matching layer: category XML, bootstrap+compile tooling
runtime/        Rust, compiled to WASM: ternary inference engine + BPE tokenizer
web/            Next.js app (landing / demo / about), deployed on Vercel
prototypes/     Standalone, non-neural UX prototypes (context-folding/ -- simulation only)
docs/           model_card.md, context_folding.md, training_log.md, benchmarks.md
```

## Quick start

```bash
# Python side (training/export/eval, plus the AIML compiler/bootstrap tooling)
pip install -r model/requirements.txt
python -m pytest model/tests/ aiml/tests/ -q     # ~69 tests, no GPU/network needed

# Rust side (WASM runtime)
cd runtime && cargo test                  # native tests
cargo build --target wasm32-unknown-unknown --release   # confirm it compiles to wasm
# wasm-pack build --target web --out-dir pkg   # produces the browser-loadable .wasm + JS glue

# Web side (demo, incl. the AIML matcher + dialogue manager)
cd web && npm install && npm test && npm run build && npm run dev
```

## Architecture summary

Decoder-only transformer, LLaMA-family block shape: d_model 768, 14 layers,
12 heads (head_dim 64), SwiGLU FFN (hidden 2048), RMSNorm pre-norm, RoPE, no
biases, 32k vocab, 2048 context. Every attention/FFN projection is a
`BitLinear` layer — weights are quantized to {-1,0,+1} via absmean scaling
during the forward pass, trained with a straight-through estimator so
gradients flow to a full-precision latent weight as if quantization were the
identity. The tied embedding/LM-head table stays fp16 (it's a lookup, not a
matmul — quantizing it buys no speed, costs quality). Full details:
`docs/model_card.md`.

**Measured parameter count: 123,688,704** ("125M-class" —
`model/tests/test_architecture.py` verifies this against a ~125M target,
±5%). `d_model`=768/`n_heads`=12 deliberately match GPT-2-small's width;
see `docs/model_card.md` for why this size and not bigger: Chinchilla token
budget and per-token compute both scale with params, so total training
compute scales as roughly `params²` — this size already costs ~2.8x the
compute of the original ~74.2M design point.

## Hybrid deterministic architecture

Benny is two layers, not one:

1. **Deterministic layer** (`aiml/`) — an AIML-style pattern matcher, tried
   first on every turn. Same input + same dialogue state always produces
   the same match: cheap, fast, no neural inference required.
2. **Neural fallback** (`model/`, `runtime/`) — the ternary GPT transformer
   above, invoked only when the deterministic layer finds no unambiguous
   match: the long tail of open-ended input the pattern set doesn't cover.

This is a deliberate cost/latency/predictability engineering decision —
deterministic where determinism is sufficient, neural only where necessary
— not a fallback to older chatbot technology because the neural approach
is inadequate. A pattern that would produce multiple divergent valid
replies (what AIML calls a `<random>` block) is treated as ambiguous and
routed to the neural fallback too, rather than picked from randomly — the
matcher never guesses, it only returns responses that are actually
reproducible. See `aiml/README.md` for the matching rules and
`web/lib/dialogue/dialogueManager.ts` for the routing logic itself.

## Parameter ladder (25M → 50M → 75M → 125M)

125M is the ceiling/default for this project phase — not trained directly,
but reached via a 4-stage validation ladder, since Chinchilla-optimal
training compute scales roughly as params² (token budget scales with
params, and per-token cost does too), not linearly:

| Stage | params | purpose |
|---|---|---|
| 1 | 25,083,264 | validate the whole pipeline cheaply before scaling up |
| 2 | 49,900,160 | first quality scale-up |
| 3 | 74,187,072 | the original design point from an earlier pass of this project |
| 4 | 123,688,704 | ceiling for this phase (current default) |

Configs: `model/configs/stage{1,2,3,4}_{25m,50m,75m,125m}_{ternary,fp16_baseline}.yaml`.
Full rationale and exact architecture per stage: `docs/model_card.md`.

## Context folding (research extension, architecture built, not yet trained)

A learned compression head extends Benny's effective context beyond its
2048-token window: text is chunked into blocks, each followed by a handful
of "gist tokens" (ordinary vocabulary entries, no separate encoder/decoder)
that a segment-conditioning attention mask forces the model to rely on once
a block is folded, instead of that block's raw tokens. Ternary-native by
design — the gist pathway trains through the exact same BitLinear-quantized
stack as everything else, not a full-precision side-channel. Full
positioning, prior-art comparison, and risk/fallback plan:
`docs/context_folding.md`.

Two artifacts exist for this, kept deliberately separate:
- `model/phase2_train.py` — the real, trained version (architecture +
  training scaffold built and unit-tested; the actual Phase 2 training run
  needs real long-document data + compute on the M5, same constraint as
  Phase 1, and has not happened yet).
- `prototypes/context-folding/index.html` — a standalone, vanilla-JS UX
  simulation (no real compression, no model weights) used to validate the
  fold/unfold interaction and telemetry design before the real version was
  built. Open it directly in a browser; it has no build step and touches
  nothing else in this repo.

## Web demo: chat app shell

`/demo` is a Claude/Gemini-style app shell, not a single hardcoded chat:
a left drawer for new-chat/history/projects, a right drawer that turns
this project's own engineering facts (params, packed size, compression
ratio, live tokens/sec, AIML-resolved ratio, an energy-vs-cloud-tier
comparison) into a live showcase panel rather than static prose, and
support for attaching plain-text-ish files (`.txt`/`.md`/`.json`/`.csv`/
`.log`) whose content is folded into the conversation.

**Chat history and projects persist locally, via the browser's
`localStorage`** (`web/lib/store/`) — a small but real change from earlier
in this project's build: previously nothing persisted at all. Nothing
about the "100% client-side, no server calls" story changes — persistence
here means *only* "survives a page reload," not "saved anywhere off your
device." Clearing your browser's site data for this page removes it, same
as any other client-side-only web app. There are still no accounts, no
server-side persistence, and no analytics.

## Retrain / re-export / redeploy, end to end

This is the full path from empty checkpoints to a live demo with a real,
trained model. Every step below is a script in this repo — no step is a
"figure it out yourself" gap.

### 0. Where this needs to run

Real training (Phase 3/4/5 below) needs a machine with enough compute to
push ~2.47B+ tokens through a 123.7M-param model in a reasonable number of
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

Watch the printed `tok/s`. Compare against a freshly-measured fp16 baseline
run at this same ~123.7M architecture — the spec's original 5,600 tok/s
figure was measured at the original ~74.2M design point and doesn't
directly transfer after the resize (see `docs/training_log.md`). Record the
result there. If the resulting
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
so 69.02MB / 123.7M params will stay accurate unless you change the
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
- **Throughput / wall-clock**: the spec's original fp16 baseline of 5,600
  tok/s on the M5 was measured at the original ~74.2M design point, not the
  current ~123.7M ("125M-class") architecture — re-measure it here rather
  than reusing that number (a rough scaling estimate is ~3,360 tok/s, but
  treat that as a placeholder guess, not a plan input). Ternary training
  adds quantization overhead per forward/backward pass that fp16 doesn't
  have, and MPS lacks CUDA's fused ternary-training kernels. At this size,
  expect somewhere in the range of ~10-17 days of *continuous* compute for
  the full 2.47B-token budget (~2.8x the original design point's estimated
  3.5-6 days, since Chinchilla-optimal compute scales roughly as params²),
  and correspondingly more real calendar time given a laptop isn't run at
  100% utilization nonstop. Measure, don't assume — the Phase 3 smoke test
  above gives you the real number for this machine at this size.
- **No distributed training** — single-device only, by design; there's no
  multi-GPU scaffolding to configure or debug.

## What's real vs. placeholder right now

| Piece | Status |
|---|---|
| Architecture + params (123.7M, "125M-class") | Real, tested (`model/tests/`) |
| Tokenizer algorithm | Real, tested |
| Tokenizer *vocabulary* shipped in this repo | Placeholder — trained on synthetic text |
| Training/SFT/export/eval code | Real, tested end-to-end on tiny local smoke tests |
| Actual trained model weights | **Not real yet** — `web/public/model/benny-placeholder.bnai` has random weights |
| Packed file size (69.02MB) / compression ratio (~3.58x) | Real measurement (packing is weight-value-independent) |
| Rust WASM runtime | See `runtime/` — built against the placeholder artifact |
| Web demo | See `web/` — built against the placeholder artifact and a documented worker interface |
| AIML matcher engine + dialogue manager | Real, tested (`aiml/tests/`, `web/lib/aiml/*.test.ts`) |
| AIML category *content* shipped in this repo | Placeholder — 95 hand-curated seed categories, not bootstrapped from real UltraChat/OASST2 (see `aiml/README.md`) |
| Context folding (gist tokens / Vault / two-phase curriculum) | **Not implemented** — referenced spec docs were never shared; dialogue manager does simple oldest-turn truncation instead |
| Any loss/perplexity/eval/benchmark number | **TBD** — needs the real training run |

## Open questions to confirm before/at real-training time (spec Section 11)

- Measured ternary tok/s from the Phase 3 smoke test on the actual M5,
  compared against the estimated ranges above.
- Current license terms for FineWeb-Edu, UltraChat 200k, and OASST2 — verify
  before downloading and training on them at scale.
- Mobile Safari's real KV-cache memory ceiling at `context_len=2048` — may
  push the *served* context window below the *trained* one; test on a real
  iOS device, not a simulator.
- **Context-folding reconciliation (not attempted, flagged only):** a later
  spec referenced two documents this repo has never had —
  `benny-folding-head-architecture-spec.txt` (gist tokens, a "Vault", a
  two-phase training curriculum for the neural fallback's conversation
  history) and `benny-positioning-narrative.txt` (a README/positioning
  draft). Neither exists in this repo or has been shared. The current
  neural fallback (`model/`, `runtime/`) and dialogue manager
  (`web/lib/dialogue/`) do **not** implement context folding — the
  dialogue manager truncates oldest turns on overflow, nothing fancier.
  Reconcile once those documents are available; don't guess their content.
