# Training Log — Benny (BNAI V1.0)

This is a template to fill in as real training runs happen. It exists now
(before any real run) so the format is decided in advance rather than
invented retroactively.

## Environment

| | |
|---|---|
| Machine | MacBook Pro, Apple M5, 24GB unified memory |
| Backend | PyTorch MPS (no CUDA) |
| PyTorch version | TBD |
| OS | TBD |

## Phase 3 — MPS smoke test (fill in before committing to the full run)

Purpose: confirm BitLinear's quantization/STE ops run correctly (and at
reasonable speed) on MPS before starting a multi-day run, and measure real
tokens/sec on this machine to pick the right wall-clock scenario from the
spec's ladder (optimistic ~4,800-5,000 tok/s / likely ~3,600-4,200 / conservative
~2,800-3,100, against a measured fp16 baseline of 5,600 tok/s).

| Check | Result |
|---|---|
| All BitLinear ops (quantize weight, quantize activation, STE backward) run on MPS without silent CPU fallback | TBD |
| Measured ternary tok/s (few hundred steps, full architecture) | TBD |
| Measured fp16-baseline tok/s (same architecture, `ternary_weights: false`) | TBD |
| Which wall-clock scenario applies | TBD |
| Recomputed full-run estimate (1.5B tokens / measured tok/s) | TBD |
| Fallback taken, if any (trim context/width, reduce token budget, rent cloud compute) | TBD |

Command used:
```
python model/train.py --config model/configs/base_ternary.yaml \
    --tokenizer model/tokenizer/bnai_tokenizer.json \
    --smoke-test-steps 200
```

## Phase 4 — Full base pretrain run

| | Ternary (`base_ternary.yaml`) | fp16 baseline (`base_fp16_baseline.yaml`) |
|---|---|---|
| Start date | TBD | TBD |
| End date | TBD | TBD |
| Total tokens seen | TBD | TBD |
| Final train loss | TBD | TBD |
| Final held-out val perplexity | TBD | TBD |
| Total wall-clock (sum of sessions) | TBD | TBD |
| Number of resumed sessions | TBD | TBD |
| Notable interruptions (sleep/thermal/reboot) | TBD | TBD |

Loss/perplexity/grad-norm/sparsity/tokens-per-sec curves: see
`checkpoints/*/training_log.jsonl` (written automatically by `train.py`) —
plot and embed here once real runs exist.

## Phase 5 — SFT (Stage B)

| | |
|---|---|
| Base checkpoint used | TBD |
| SFT dataset(s) / subset size (tokens) | TBD |
| Start / end date | TBD |
| Final SFT train perplexity | TBD |
| Base-domain val perplexity: pre-SFT reference | TBD |
| Base-domain val perplexity: post-SFT | TBD |
| Alignment tax (regression %) | TBD |
| Action taken if regression exceeded `max_base_ppl_regression` | TBD |

## Deviations from the spec (if any)

Record anything actually different from `docs/model_card.md`'s recipe here,
with the reason — e.g. if the Section 5 fallback ladder had to be used
(reduced token budget, trimmed architecture, moved to rented compute).
