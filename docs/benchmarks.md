# Benchmarks — Benny (BNAI V1.0)

Template, filled in as real numbers exist. Two independent things are
measured here — don't conflate them:

1. **Training-side reference** (`model/eval.py --benchmark-inference`): a
   plain PyTorch forward pass, ternary vs fp16, on whatever machine you run
   it on. Useful for a quick sanity check, *not* representative of the real
   browser number.
2. **Runtime-side (the actual headline result)**: the Rust/WASM engine in
   `runtime/`, running in an actual browser, on the actual target devices
   (spec Section 7's locked browser matrix: desktop/mobile Chrome,
   desktop/mobile Edge, mobile Safari ≥16.4). This is what the demo UI's
   live "tokens/sec, running locally in your browser" readout shows, and
   what should be published on the landing page.

## Packed model size (measured now, doesn't need training)

| | Value |
|---|---|
| Total parameters | 123,688,704 ("125M-class") |
| Packed `.bnai` size (ternary + fp16 embedding) | 69.02 MB |
| fp16-equivalent size (nothing quantized) | ~247.4 MB |
| **Measured compression ratio** | **~3.58x** |

See `docs/model_card.md` for why this is ~3.58x rather than the 5-8x a naive
"1.58 bits vs 16 bits" estimate might suggest (short version: the tied fp16
embedding table is ~19.9% of total params and doesn't compress -- a smaller
share than at the original 74.2M design point, which is why the ratio
improved rather than worsened when the model got bigger).

## 1. Training-side reference benchmark (TBD)

Run: `python model/eval.py --checkpoint <ternary.pt> --compare-fp16 <fp16.pt> --tokenizer <tok.json> --benchmark-inference`

| | Ternary | fp16 baseline |
|---|---|---|
| Device | TBD | TBD |
| Tokens/sec | TBD | TBD |
| Peak memory (MB) | TBD | TBD |
| Speedup (ternary/fp16) | TBD | — |

## 2. Runtime/browser benchmark (TBD — the real headline number)

Fill in once `runtime/` + `web/` are wired together and deployed. Test on
each browser in the locked matrix (spec Section 7); mobile Safari is the
binding memory constraint, test on a real iOS device, not just desktop
Safari or a simulator.

| Browser / device | WASM SIMD supported? | Cold load time (69MB model + runtime) | Time to first token | Steady-state tokens/sec | Peak memory |
|---|---|---|---|---|---|
| Desktop Chrome | TBD | TBD | TBD | TBD | TBD |
| Desktop Edge | TBD | TBD | TBD | TBD | TBD |
| Mobile Chrome | TBD | TBD | TBD | TBD | TBD |
| Mobile Edge | TBD | TBD | TBD | TBD | TBD |
| Mobile Safari (real device, ≥16.4) | TBD | TBD | TBD | TBD | TBD |

## 3. Model quality (TBD — pending real training)

| Metric | Ternary | fp16 baseline | Gap |
|---|---|---|---|
| Held-out validation perplexity | TBD | TBD | TBD (target: ternary within 10-15% of fp16) |
| Reduced HellaSwag/LAMBADA-style accuracy | TBD | TBD | — |

## Open measurement questions (spec Section 11 — confirm before/at deploy)

- Real ternary tok/s on the M5 during Phase 3's smoke test, vs a freshly
  re-measured fp16 baseline at the current ~123.7M architecture (the
  spec's original 5,600 tok/s figure was measured at the original ~74.2M
  design point and doesn't directly apply after the resize -- see
  `docs/training_log.md`).
- Mobile Safari's actual KV-cache memory ceiling at context_len=2048 — may
  push the *served* context window below the *trained* 2048, tested for
  real on-device rather than assumed.
