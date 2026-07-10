# Benny's deterministic layer (AIML)

Benny is a two-layer conversational system (see the top-level README's
"Hybrid deterministic architecture" section for the full framing):

1. **Deterministic layer** (this directory): AIML-style pattern matching,
   tried first on every turn. Same input + same dialogue state always
   produces the same match -- cheap, fast, no neural inference required.
2. **Neural fallback** (`model/`, `runtime/`): the ternary GPT transformer,
   invoked only when the deterministic layer finds no unambiguous match.

This is a cost/latency/predictability engineering decision -- deterministic
where determinism is sufficient, neural only where necessary -- not a
regression to older chatbot technology. See `web/lib/dialogue/dialogueManager.ts`
for where that decision is actually implemented.

## Format

Standard-flavored AIML XML, one or more `<category>` elements per file:

```xml
<category>
  <pattern>MY NAME IS *</pattern>
  <that>...</that>            <!-- optional: scopes to the bot's last utterance -->
  <topic>...</topic>          <!-- optional: scopes to the current dialogue topic -->
  <setTopic>...</setTopic>    <!-- optional: sets the topic when this category wins -->
  <template>Nice to meet you, <star/>!</template>
</category>
```

- `<pattern>`/`<that>` words are normalized (uppercased, punctuation
  stripped) and may contain `*` (greedy, one-or-more words) or `_` (exactly
  one word) wildcards. Exact words beat `_` beat `*`; longer/more-specific
  matches beat shorter ones; `<that>`/`<topic>`-scoped categories that match
  the current state are checked and preferred *before* unscoped ones, even
  at equal word-specificity.
- A category with exactly one `<template>` (or a `<random>` with exactly
  one `<li>`) is single-template and returned deterministically. A
  `<random>` with 2+ `<li>` variants is multi-template -- the matcher
  treats this as **ambiguous** and does not pick among them; the dialogue
  manager routes to the GPT fallback instead (spec Section 1.3). This is a
  deliberate design choice: a cluster of divergent valid replies to the
  same input is exactly the case the neural model is better suited to than
  static template selection.
- `<setTopic>` is **not** standard AIML. The project spec requires
  `<topic>`-scoped categories but never specifies how a category sets the
  topic going forward, and generic AIML `<set>`/`<get>` predicate variables
  are explicitly out of scope. This is a minimal, single-purpose addition:
  when a category with `<setTopic>VALUE</setTopic>` wins a match, the
  dialogue manager (not the matcher) sets the topic to `VALUE`.
- `<srai>`, pronoun/person substitution, and generic sets/maps are **not**
  supported, on purpose -- keeping the feature surface small.

**Safety rule: no catch-all patterns.** A category whose pattern has zero
literal words and no `<that>`/`<topic>` scope (a bare `*`/`_`, or any
all-wildcard pattern like `* *`) would match every input and defeat the
entire point of having a fallback system. `aiml/tools/aiml_compiler.py`
hard-fails the whole compile if one exists -- this is enforced at compile
time, not worked around at runtime.

## Pipeline: bootstrap -> review -> compile

```
aiml/tools/bootstrap.py  -->  candidate clusters (JSON, for human review)
        |                              |
        |                     (human curation: generalize a pattern,
        |                      write a PARAPHRASED template -- never
        |                      copy source text verbatim)
        v                              v
aiml/categories/*.aiml  <-------  reviewed .aiml files
        |
        v
aiml/tools/aiml_compiler.py  -->  web/lib/aiml/generated/categories.json
```

1. **`bootstrap.py`** mines short, single-turn-shaped user utterances from
   conversational training data (UltraChat 200k / OASST2, or any local
   `{"messages": [...]}` jsonl in the same shape `model/sft.py` expects),
   filters out turns that look like they need unique factual/named-entity
   context (those should keep falling through to the GPT fallback), groups
   near-duplicates by lightweight normalized-token-set similarity (no
   embedding model), ranks by frequency, and writes the top-N clusters to a
   JSON file for review. It does **not** generate AIML XML -- turning a
   cluster into a real category needs a human to generalize a wildcard
   pattern and write a paraphrased template, which is a judgment call this
   script doesn't make.

   ```bash
   python aiml/tools/bootstrap.py --hf-dataset HuggingFaceH4/ultrachat_200k \
       --hf-split train_sft --max-conversations 200000 --top-n 300 --out candidates.json
   ```

2. **Human review** (or, for this repo's current seed set, Claude acting
   as the initial reviewer -- see "Current status" below): read the
   candidate clusters, generalize a pattern, write a paraphrased template,
   add it to `aiml/categories/*.aiml`. Reject or merge clusters that don't
   generalize well. No catch-all patterns, ever.

3. **`aiml_compiler.py`** parses, validates, and compiles `aiml/categories/*.aiml`
   into the JSON the browser loads statically:

   ```bash
   python aiml/tools/aiml_compiler.py --categories "aiml/categories/*.aiml" \
       --out web/lib/aiml/generated/categories.json
   ```

   This is the single source of truth for compilation -- the TypeScript
   runtime (`web/lib/aiml/*.ts`) never parses XML itself, it only
   implements the matching algorithm against this compiled JSON. Keep the
   two in sync by hand if you change either's shape (the same way
   `runtime/src/tokenizer.rs` is kept in sync with `model/tokenizer/bpe.py`).

## Current status

The category set currently in `aiml/categories/` (95 categories: greetings,
small talk, emotional check-ins, meta questions about the bot, farewells,
honest capability/limitation deflections, politeness/compliments/mild-
insult handling) is a **hand-curated seed set**, not a bootstrapped-
from-real-data corpus -- this sandbox has no network access to
UltraChat/OASST2 (confirmed still blocked: `huggingface.co` is explicitly
policy-denied by this environment's egress proxy, not merely
unconfigured). It exists to demonstrate the pipeline end-to-end (data
format -> compiler -> matcher -> dialogue manager -> web demo), the same
way the shipped model checkpoint and tokenizer are currently placeholders
pending real training (see the top-level README).

The `capabilities.aiml` file is worth calling out specifically: every
pattern in it is a question where the honest deterministic answer is "no"
or "I can't verify that" (a real clock, live weather, browsing, persistent
cross-session memory) -- a fixed template must never assert something
false just because a fixed string is easy to write, per this project's
honesty ethos.

To build the real set: run `bootstrap.py` against the actual datasets
(from an environment with network access to Hugging Face), do a real
human review pass over the candidate clusters, then recompile. Same
licensing question already flagged for FineWeb-Edu/UltraChat/OASST2 in
`docs/model_card.md` applies here too -- confirm terms once, it covers
both uses.
