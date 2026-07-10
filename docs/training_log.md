# Training Log — Benny (BNAI V1.0)

This is a template to fill in as real training runs happen. It exists now
(before any real run) so the format is decided in advance rather than
invented retroactively.

Training happens across the 4-stage parameter ladder (see
`docs/model_card.md`), not as a single run — duplicate the Phase 3/4/5
sections below per stage as each one is actually trained. Stage 4
(~123.7M) is the ceiling/default; Stages 1-3 are earlier validation
checkpoints on the way there.

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
tokens/sec on this machine to pick the right wall-clock scenario.

**Note on the fp16 baseline number:** the spec's original ladder
(optimistic ~4,800-5,000 tok/s / likely ~3,600-4,200 / conservative
~2,800-3,100 tok/s, against a measured fp16 baseline of 5,600 tok/s) was
measured at the *original* ~74.2M architecture. This repo's architecture is
now ~123.7M params ("125M-class", resized from the original design point
-- see `docs/model_card.md`), so per-token compute is higher and that
5,600 tok/s figure does not directly apply. Re-measure the fp16 baseline at
the current architecture rather than reusing the old number; a rough
scaling estimate (throughput ∝ 1/params) puts it around
`5,600 × (74.2/123.7) ≈ 3,360 tok/s`, but treat that as a placeholder guess
to replace with a real measurement, not a plan input.

| Check | Result |
|---|---|
| All BitLinear ops (quantize weight, quantize activation, STE backward) run on MPS without silent CPU fallback | TBD |
| Measured ternary tok/s (few hundred steps, full ~123.7M architecture) | TBD |
| Measured fp16-baseline tok/s (same architecture, `ternary_weights: false`) | TBD |
| Which wall-clock scenario applies | TBD |
| Recomputed full-run estimate (2.47B tokens / measured tok/s) | TBD |
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
