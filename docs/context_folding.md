# Context Folding — Benny's learned compression head

**Status: architecture + training scaffold built and unit-tested; real
Phase 2 training has not run.** This document covers the trained,
real-forward-pass version of context folding — not the standalone UX
prototype (`prototypes/context-folding/index.html`), which is a
deliberately non-neural simulation used to validate state-management and
telemetry UX before this real version existed. Do not confuse the two: the
prototype's "gist tokens" are 4 heuristically-picked words with a
placeholder comment; the real version's gist tokens are learned embeddings
trained end-to-end through the actual ternary backbone.

## Positioning and prior art

This builds directly on an established line of research in learned context
compression:

- **Gist Tokens** (Mu et al., 2023) — train a model to distill a prompt
  into a small set of tokens that carry its information forward.
- **AutoCompressors** (Chevalier et al., 2023) — recursively compress long
  context into "summary vectors" used as soft prompts, extending effective
  context length.
- **In-context Autoencoder / ICAE** (Ge et al., 2024) — learnable encoder,
  fixed decoder, compresses context into fixed memory slots.
- **Sentence-Anchored Gist Compression** (2025) — end-to-end training via
  the standard language-modeling objective alone, no auxiliary
  reconstruction loss; compression tokens are added to the vocabulary and
  an attention mask enforces segment-conditioning. This is the most
  directly reusable design for Benny, and the one this implementation
  follows most closely.
- **Adaptive KV-cache quantization** (2026, Huffman-inspired) — variable
  bit-width allocation by token importance. Cited as an *acknowledged
  inspiration* for a possible future variable-ratio extension, not claimed
  as original to this project.

What appears to be a genuine gap (stated with appropriate hedging — this
reflects a survey at design time, not an exhaustive literature search):

1. All of the above operate on continuous, full-precision backbones. None
   train a natively ternary/BitLinear backbone jointly with a compression
   head. **Benny's contribution: what does gisting look like when the
   entire model — weights and memory both — is 1.58-bit?**
2. All of the above treat compressed memory as ephemeral, single-session
   KV-cache state. The prototype's Vault design (content-addressable,
   hash-keyed, exportable) borrows a systems-engineering pattern
   (content-addressable storage, e.g. git objects) and applies it to
   compressed neural memory — portable across sessions/devices, not just
   an inference-time cache. **This axis is validated in the UX prototype
   only; the real trained version's export format is not yet built (see
   "What's not built yet" below).**

Portfolio framing: name the prior art explicitly, then name these two axes
as the specific extension. **Do not claim to have "solved long context"** —
this is a specific, narrow, defensible combination, not a general result.

## Reality check on compression ratio

Recent work found compression capacity scales with model size: an 8B model
can condense up to ~1568 tokens into a single vector via prompt tuning,
while smaller models manage far less. Benny is ~123.7M params (see the
parameter ladder in `docs/model_card.md`) — treat the "100 tokens → N gist
tokens" ratio as a **hypothesis**, tuned empirically via the gist-token-count
sweep (4/8/16 — `model/configs/phase2_gist{4,8,16}.yaml`), not a spec.

## Architecture

- **Gist tokens are ordinary vocabulary entries** (`model/folding.py`'s
  `add_gist_tokens`) — learned embeddings like any other token, requiring
  no separate encoder/decoder network. They occupy previously-unused rows
  in the model's fixed-size embedding table (the same headroom that already
  lets an undersized placeholder tokenizer work elsewhere in this repo), so
  moving from Phase 1 to Phase 2 needs **zero architecture change or
  checkpoint surgery** — a Phase 1 checkpoint's `model_state_dict` loads
  directly.
- **Segment-conditioning attention mask** (`model/folding.py`'s
  `build_segment_conditioning_mask` / `build_batched_segment_conditioning_mask`,
  threaded through `model/architecture.py`'s `attn_mask` parameter): text is
  chunked into fixed-size blocks (`fold_block_size`, starts at 100), with
  `gist_token_count` gist ids inserted after every block. The mask enforces
  ordinary causal ordering, *plus* one extra exclusion: raw (non-gist)
  tokens of a strictly earlier block are masked out entirely, while that
  block's gist tokens stay visible regardless of position. `attn_mask=None`
  (Phase 1's default) reproduces the exact prior `is_causal=True` behavior
  — verified by a regression test
  (`test_attn_mask_none_matches_prior_is_causal_behavior`).
- **Ternary-native gist representation** (the primary novel axis): no
  full-precision side-channel. Gist token embeddings and the attention
  computation that produces them go through the exact same BitLinear
  ternary-quantized path as every other token, from the start of Phase 2
  training — not quantized post-hoc. This is expected to be the hardest
  part to stabilize; budget real debugging time here (spec's own words:
  "the highest-research-risk component of the whole project").
- **Loss function**: standard next-token prediction only. No auxiliary
  reconstruction loss, no second decoder path — keeps scope achievable at
  this model size on a single MacBook.
- **Two-tier memory**: the *neural* path (gist tokens) carries forward only
  enough signal for good next-token prediction — it will **not** reconstruct
  exact original text, and no cited prior art claims otherwise. A separate,
  non-neural *verbatim* path (the prototype's Vault) stores original block
  text for UI display / exact-recall / audit purposes, explicitly decoupled
  from the model's real attention path. Any UI built on the real version
  must keep this distinction visible — "folded" badges show what the model
  sees (gist tokens); "inspect original" pulls from the separate Vault
  cache. The model does not reconstruct text from gist tokens.

### Honesty note: what "reduces compute" actually means here

During *training*, PyTorch's `scaled_dot_product_attention` still computes
the full O(seq_len²) attention matrix and masks it — passing a boolean
mask alone doesn't skip that work. The mask's job during training is to
teach the model that only prior blocks' gist tokens matter, which is what
makes it *safe and correct* to literally evict a folded block's raw K/V
cache entries at **inference** time in a real serving runtime. That
eviction — not the training-time mask — is where the real compute/memory
reduction is realized. **Building that inference-time runtime is out of
scope for this pass** (see "What's not built yet" below); this repo's
`runtime/` (Rust/WASM) does not yet implement folding-aware KV-cache
eviction.

## Two-phase training curriculum (why two phases, not one)

- **Phase 1** (`model/train.py`, already spec'd/built pre-folding): stable
  base pretraining, no gist tokens, no segment masking. This is the
  fallback deliverable if Phase 2 runs into trouble.
- **Phase 2** (`model/phase2_train.py`): resumes from a Phase 1 checkpoint,
  introduces gist-token vocabulary + the segment-conditioning mask +
  long/concatenated-document training data (spec Section 6: concatenation
  chosen over a filtered-long-document subset — simpler, no new
  licensing/sourcing question). Trains with standard LM loss until gist
  token behavior stabilizes.

Baking folding into pretraining from token zero would compound two hard
problems (BitLinear QAT stability + learned compression stability) on a
first from-scratch run, under a tight compute budget. Splitting them means
a Phase 1 failure is "normal QAT debugging" (already scoped), and a Phase 2
failure is isolated to the folding mechanism. **If Phase 2 doesn't converge
in time, Phase 1 alone is still a complete, shippable model** — this is
exactly the ladder/ceiling framing already used for the parameter sweep.

## Evaluation plan (spec Section 4) — harness built, real numbers pending

`model/eval.py --folding-eval` runs:
- **Extended-context perplexity**, folded vs. an ordinary truncated
  baseline of the same held-out long documents (`perplexity_at_extended_context`).
- **Needle-in-haystack retrieval**: a synthetic fact placed in an early
  block, queried after several folds, measured against the same truncated
  baseline (`needle_in_haystack_eval`) — synthetic/procedural by
  construction, so the harness runs today without a real corpus. With the
  current untrained/placeholder checkpoint, expect near-chance accuracy on
  both arms; the point of this pass is a **correct, runnable harness**, not
  a real result — that needs Phase 2 training to have actually happened.
- The gist-token-count sweep (4/8/16) results are themselves portfolio
  evidence of empirical rigor, reported alongside the final chosen ratio,
  not just the final number.
- A ternary-vs-full-precision gist head comparison (if time allows) is the
  number that actually substantiates the "ternary-native" novelty claim —
  not yet built; would need a second Phase 2 training path with
  `ternary_weights: false` restricted to just the gist-producing layers,
  which the current fp16-baseline toggle doesn't support at that
  granularity.

## Risks and fallback (spec Section 5)

- **Highest risk**: ternary quantization of the compression pathway fails
  to converge or degrades quality sharply. Fallback: a full-precision gist
  head bolted onto the ternary backbone (loses the "fully ternary" claim,
  keeps a working folding demo).
- **Second risk**: Phase 2 training time exceeds budget. Fallback: ship the
  Phase 1 base model, document Phase 2 as in-progress/next-milestone work
  with this architecture spec as evidence of design completeness.
- Phase 2 risk must not block the Phase 1 deliverable — checkpoint and
  evaluate Phase 1 independently before starting Phase 2 (already how
  `phase2_train.py` is structured: it requires an existing Phase 1
  checkpoint as input, never trains one itself).

## What's built vs. not built yet

| Piece | Status |
|---|---|
| Gist-token vocabulary (`model/folding.py::add_gist_tokens`) | Real, tested |
| Segment-conditioning mask (single + batched) | Real, tested — including a correctness test that changing an early raw token doesn't move later-block logits directly, only through the gist token's legitimate pooled representation |
| Folded data pipeline (`FoldedTokenBlockDataset`) | Real, tested |
| Phase 2 training script (`phase2_train.py`) | Real, tested end-to-end on tiny local smoke tests (fresh + resume) |
| Sweep configs (gist-count 4/8/16) | Real, ready to run |
| Extended-context perplexity + needle-in-haystack eval | Real harness, tested — no real result yet (needs Phase 2 training) |
| **Real Phase 2 training run** | **Not run** — needs real long-document data + real compute on the M5, same constraint as Phase 1 |
| Standalone UX prototype (`prototypes/context-folding/`) | Real, verified in-browser — explicitly a simulation, not the trained version |
| Portable Vault export format (real trained version) | **Not built** — only exists in the prototype's in-memory, non-persistent form |
| Inference-time KV-cache eviction (the actual compute-savings mechanism) | **Not built** — `runtime/` (Rust/WASM) doesn't implement folding-aware serving yet |

## Open decisions carried over from the spec

1. Gist tokens per 100-token block — sweep 4/8/16, decide from real Phase 2
   eval results (`docs/benchmarks.md`), not the prototype's assumed value.
2. Long-document data source — concatenation (chosen, simpler) vs. a
   filtered-long-document FineWeb-Edu subset; revisit if quality suffers.
3. Portable session export format — JSON, matching the prototype's
   existing shapes; revisit only if size becomes a problem, and only once
   the real trained version's export path is actually built.
