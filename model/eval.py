"""Evaluation harness (spec Section 6 / Phase 7).

Four things, all runnable independently via CLI flags:
  1. Held-out validation perplexity for a single checkpoint.
  2. A reduced HellaSwag/LAMBADA-style multiple-choice next-token accuracy
     check (log-likelihood of the correct continuation vs distractors).
  3. Side-by-side ternary-vs-fp16 comparison: two checkpoints trained with an
     identical recipe/data budget (one ternary, one fp16 baseline), reporting
     the perplexity gap that isolates the cost of quantization.
  4. Inference benchmarking: tokens/sec and peak memory for a plain PyTorch
     forward pass on this machine's CPU. This is *not* the WASM number the
     web demo publishes (spec Section 7/8) -- it's the training-side
     reference point; the browser-measured number is a separate, manual step
     recorded in docs/benchmarks.md once the runtime is deployed.

Usage:
    python eval.py --checkpoint checkpoints/base_ternary/latest.pt \
        --tokenizer tokenizer/bnai_tokenizer.json --local-val-corpus /path/*.jsonl

    python eval.py --checkpoint checkpoints/base_ternary/latest.pt \
        --tokenizer tokenizer/bnai_tokenizer.json --compare-fp16 checkpoints/base_fp16_baseline/latest.pt \
        --local-val-corpus /path/*.jsonl

    python eval.py --checkpoint checkpoints/base_ternary/latest.pt \
        --tokenizer tokenizer/bnai_tokenizer.json --benchmark-inference

    python eval.py --checkpoint checkpoints/base_ternary/latest.pt \
        --tokenizer tokenizer/bnai_tokenizer.json --multiple-choice-examples examples.jsonl
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import resource
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

from architecture import BNAIConfig, BNAILanguageModel  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from data.pipeline import TokenBlockDataset, iter_hf_streaming_documents, iter_local_jsonl_documents  # noqa: E402
from train_utils import estimate_perplexity, select_device  # noqa: E402


def load_checkpoint(path: str, device: torch.device) -> BNAILanguageModel:
    payload = torch.load(path, map_location=device)
    cfg = BNAIConfig(**payload["config"]["model"])
    model = BNAILanguageModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def make_val_batches(args, model_cfg: BNAIConfig, tokenizer: BPETokenizer, micro_batch_size: int):
    if args.local_val_corpus:
        paths = sorted(glob.glob(args.local_val_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-val-corpus {args.local_val_corpus}")
        documents = itertools.chain.from_iterable(iter_local_jsonl_documents(p) for p in paths)
    else:
        documents = iter_hf_streaming_documents(args.hf_val_dataset, args.hf_val_split, args.hf_val_text_field)

    val_dataset = TokenBlockDataset(documents, tokenizer, model_cfg.context_len, val_fraction=args.val_fraction, split="val")
    return iter(DataLoader(val_dataset, batch_size=micro_batch_size))


# ---------------------------------------------------------------------------
# 2. Multiple-choice (HellaSwag/LAMBADA-style) accuracy
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sequence_log_likelihood(model: BNAILanguageModel, tokenizer: BPETokenizer, context: str, continuation: str, device) -> float:
    context_ids = tokenizer.encode(context, add_bos=True)
    continuation_ids = tokenizer.encode(continuation)
    if not continuation_ids:
        return float("-inf")

    full_ids = context_ids + continuation_ids
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    logits, _ = model(input_ids)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    continuation_start = len(context_ids) - 1  # index into targets where the continuation begins
    continuation_targets = targets[0, continuation_start:]
    continuation_log_probs = log_probs[0, continuation_start:]
    token_ll = continuation_log_probs.gather(-1, continuation_targets.unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item() / len(continuation_ids)  # length-normalized, standard for multiple-choice eval


@torch.no_grad()
def multiple_choice_accuracy(model: BNAILanguageModel, tokenizer: BPETokenizer, examples: list, device) -> dict:
    """`examples`: list of {"context": str, "choices": [str, ...], "correct_idx": int}
    (this is the HellaSwag/LAMBADA-style format -- pass a jsonl of these, sourced
    from the real datasets at eval time, per spec Section 6's "reduced" battery)."""
    correct = 0
    for ex in examples:
        scores = [
            _sequence_log_likelihood(model, tokenizer, ex["context"], choice, device) for choice in ex["choices"]
        ]
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        if predicted == ex["correct_idx"]:
            correct += 1
    accuracy = correct / len(examples) if examples else float("nan")
    return {"accuracy": accuracy, "n_examples": len(examples)}


# ---------------------------------------------------------------------------
# 4. Inference benchmarking (training-side reference, not the WASM number)
# ---------------------------------------------------------------------------


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps" and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory() / 1e6
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # ru_maxrss is KB on Linux, bytes on macOS*


@torch.no_grad()
def benchmark_inference(model: BNAILanguageModel, tokenizer: BPETokenizer, device: torch.device, prompt_len=32, gen_len=64, n_runs=3) -> dict:
    """Naive full-forward-per-token generation (no KV-cache -- architecture.py
    is the training-time module; the KV-cached, no-multiply inference path
    lives in runtime/, exercised separately from the browser). This gives a
    directional ternary-vs-fp16 speed comparison on the training side, not
    an absolute throughput number to compare against the WASM demo."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    timings = []
    for _ in range(n_runs):
        ids = input_ids.clone()
        start = time.time()
        for _ in range(gen_len):
            logits, _ = model(ids[:, -model.cfg.context_len :])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        timings.append((time.time() - start, gen_len))

    total_tokens = sum(t[1] for t in timings)
    total_time = sum(t[0] for t in timings)
    return {
        "tokens_per_sec": total_tokens / max(total_time, 1e-9),
        "peak_memory_mb": _peak_memory_mb(device),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Context-folding eval (spec Section 4): perplexity at extended context,
# with vs without folding; needle-in-haystack retrieval.
# ---------------------------------------------------------------------------


def _folding_document_source(args):
    if args.local_val_corpus:
        paths = sorted(glob.glob(args.local_val_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-val-corpus {args.local_val_corpus}")
        return itertools.chain.from_iterable(iter_local_jsonl_documents(p) for p in paths)
    return iter_hf_streaming_documents(args.hf_val_dataset, args.hf_val_split, args.hf_val_text_field)


def perplexity_at_extended_context(
    model: BNAILanguageModel,
    tokenizer,
    documents,
    gist_ids: list,
    fold_block_size: int,
    window_len: int,
    device,
    micro_batch_size: int = 4,
    max_batches: int = 20,
) -> dict:
    """Spec Section 4, bullet 1: perplexity on held-out long documents, with
    folding (extended effective context via the segment-conditioning mask)
    vs. without (an ordinary fixed-`context_len` window that simply can't
    see anything beyond it -- the standard baseline, not a strawman)."""
    from data.pipeline import FoldedTokenBlockDataset  # local import: only needed for this folding-specific path
    from folding import build_batched_segment_conditioning_mask
    from train_utils import estimate_perplexity as _estimate_perplexity

    documents = list(documents)  # both datasets below consume the same held-out documents

    folded_dataset = FoldedTokenBlockDataset(
        iter(documents), tokenizer, window_len=window_len, gist_ids=gist_ids, fold_block_size=fold_block_size,
        val_fraction=0.0, split="train",
    )
    folded_batches = iter(DataLoader(folded_dataset, batch_size=micro_batch_size))
    folded_ppl = _estimate_perplexity(
        model, folded_batches, device, max_batches=max_batches,
        mask_fn=lambda b: build_batched_segment_conditioning_mask(b["block_ids"], b["is_gist"]),
    )

    truncated_dataset = TokenBlockDataset(
        iter(documents), tokenizer, model.cfg.context_len, val_fraction=0.0, split="train"
    )
    truncated_batches = iter(DataLoader(truncated_dataset, batch_size=micro_batch_size))
    truncated_ppl = _estimate_perplexity(model, truncated_batches, device, max_batches=max_batches)

    return {
        "folded_perplexity": folded_ppl,
        "truncated_baseline_perplexity": truncated_ppl,
        "window_len": window_len,
        "context_len": model.cfg.context_len,
    }


def _random_filler_text(rng, n_words: int) -> str:
    vocab = ["the", "quick", "brown", "fox", "runs", "over", "hills", "and", "through", "valleys", "while", "birds", "sing"]
    return " ".join(rng.choice(vocab) for _ in range(n_words))


@torch.no_grad()
def _greedy_continuation_matches(model: BNAILanguageModel, tokenizer, prompt_ids: list, needle_ids: list, device) -> bool:
    """Feeds `prompt_ids`, then greedily generates len(needle_ids) tokens
    and checks for an exact match against `needle_ids`."""
    ids = list(prompt_ids)
    generated = []
    for _ in range(len(needle_ids)):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        ids.append(next_id)
    return generated == needle_ids


def needle_in_haystack_eval(
    model: BNAILanguageModel,
    tokenizer,
    gist_ids: list,
    fold_block_size: int,
    window_len: int,
    device,
    n_trials: int = 10,
    filler_blocks: int = 20,
    seed: int = 0,
) -> dict:
    """Spec Section 4, bullet 2: place a specific fact in an early
    (foldable) block, query for it after several folds have occurred,
    measure retrieval accuracy vs an un-folded baseline of the same
    effective length. Synthetic/procedural by construction (no real corpus
    needed), so this runs today -- though with an untrained (placeholder)
    checkpoint, expect near-chance accuracy on both arms; the point of this
    pass is a correct, runnable harness, not a real result yet (spec
    Section 4's real numbers need Phase 2 training to have actually run)."""
    from folding import build_segment_conditioning_mask, fold_document

    rng = __import__("random").Random(seed)
    codes = [f"CODE{rng.randint(1000, 9999)}" for _ in range(n_trials)]

    folded_correct = 0
    truncated_correct = 0

    for code in codes:
        needle_sentence = f"The secret code is {code}."
        filler = " ".join(_random_filler_text(rng, fold_block_size) for _ in range(filler_blocks))
        document = needle_sentence + " " + filler
        query = "The secret code is"

        needle_ids = tokenizer.encode(" " + code)  # leading space matches BPE's typical word-boundary tokenization
        doc_ids = tokenizer.encode(document)
        query_ids = tokenizer.encode(" " + query.strip())

        # Folded arm: fold the whole document, then append the (unfolded) query.
        folded = fold_document(doc_ids, fold_block_size, gist_ids)
        folded_prompt = folded.token_ids + query_ids
        block_ids = torch.tensor(folded.block_ids + [folded.block_ids[-1] + 1] * len(query_ids))
        is_gist = torch.tensor(folded.is_gist + [False] * len(query_ids))
        # Greedy-generates needle_ids one token at a time; since the query
        # tail's block id / gist-ness is already known ahead of time (query
        # tokens and the generated continuation both belong to one final
        # "current" block), build one mask sized to the full prompt +
        # continuation budget up front, then slice a growing prefix of it
        # per generation step instead of rebuilding it every step.
        pad_block = block_ids[-1].item()
        block_ids_full = torch.cat([block_ids, torch.full((len(needle_ids),), pad_block, dtype=torch.long)])
        is_gist_full = torch.cat([is_gist, torch.zeros(len(needle_ids), dtype=torch.bool)])
        mask_full = build_segment_conditioning_mask(block_ids_full, is_gist_full).to(device)

        folded_generated = []
        ids = list(folded_prompt)
        for step in range(len(needle_ids)):
            cur_len = len(ids)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(input_ids, attn_mask=mask_full[:cur_len, :cur_len].unsqueeze(0).unsqueeze(0))
            next_id = logits[0, -1].argmax().item()
            folded_generated.append(next_id)
            ids.append(next_id)
        if folded_generated == needle_ids:
            folded_correct += 1

        # Truncated baseline: ordinary context window, no folding at all --
        # if the needle + filler exceeds context_len, the needle is simply
        # gone (this is the realistic failure mode folding is meant to help).
        # Leave room for the needle_ids continuation itself, since the prompt
        # grows by one token per generation step and must never exceed
        # context_len (model.forward enforces this with no mask supplied).
        truncated_prompt = (doc_ids + query_ids)[-(model.cfg.context_len - len(needle_ids)) :]
        if _greedy_continuation_matches(model, tokenizer, truncated_prompt, needle_ids, device):
            truncated_correct += 1

    return {
        "folded_accuracy": folded_correct / n_trials,
        "truncated_baseline_accuracy": truncated_correct / n_trials,
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--local-val-corpus", default=None)
    parser.add_argument("--hf-val-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-val-split", default="train")
    parser.add_argument("--hf-val-text-field", default="text")
    parser.add_argument("--val-fraction", type=float, default=0.008)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--compare-fp16", default=None, help="path to an fp16-baseline checkpoint for side-by-side comparison")
    parser.add_argument("--multiple-choice-examples", default=None, help="jsonl of {context, choices, correct_idx}")
    parser.add_argument("--benchmark-inference", action="store_true")
    parser.add_argument("--folding-eval", action="store_true", help="run Phase 2 context-folding eval (spec Section 4): extended-context perplexity + needle-in-haystack")
    parser.add_argument("--gist-token-count", type=int, default=8)
    parser.add_argument("--fold-block-size", type=int, default=100)
    parser.add_argument("--fold-window-len", type=int, default=4096)
    parser.add_argument("--needle-trials", type=int, default=10)
    args = parser.parse_args()

    device = select_device()
    print(f"[eval] device={device}")

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_checkpoint(args.checkpoint, device)
    model_cfg = model.cfg

    val_ppl = estimate_perplexity(
        model, make_val_batches(args, model_cfg, tokenizer, args.micro_batch_size), device, max_batches=args.val_batches
    )
    print(f"[eval] held-out validation perplexity: {val_ppl:.3f}")

    if args.compare_fp16:
        fp16_model = load_checkpoint(args.compare_fp16, device)
        fp16_val_ppl = estimate_perplexity(
            fp16_model,
            make_val_batches(args, fp16_model.cfg, tokenizer, args.micro_batch_size),
            device,
            max_batches=args.val_batches,
        )
        gap = (val_ppl - fp16_val_ppl) / fp16_val_ppl
        print(f"[eval] fp16 baseline perplexity: {fp16_val_ppl:.3f}")
        print(f"[eval] ternary penalty vs fp16: {gap:+.1%} (spec target: within 10-15%)")

    if args.multiple_choice_examples:
        with open(args.multiple_choice_examples) as f:
            examples = [json.loads(line) for line in f if line.strip()]
        result = multiple_choice_accuracy(model, tokenizer, examples, device)
        print(f"[eval] multiple-choice accuracy: {result['accuracy']:.1%} (n={result['n_examples']})")

    if args.benchmark_inference:
        bench = benchmark_inference(model, tokenizer, device)
        print(f"[eval] inference benchmark: {bench['tokens_per_sec']:.1f} tok/s, {bench['peak_memory_mb']:.1f} MB peak ({bench['device']})")
        if args.compare_fp16:
            fp16_bench = benchmark_inference(fp16_model, tokenizer, device)
            speedup = bench["tokens_per_sec"] / fp16_bench["tokens_per_sec"]
            print(
                f"[eval] fp16 baseline: {fp16_bench['tokens_per_sec']:.1f} tok/s, "
                f"{fp16_bench['peak_memory_mb']:.1f} MB peak -- ternary speedup: {speedup:.2f}x"
            )

    if args.folding_eval:
        from folding import add_gist_tokens, gist_token_ids

        folding_tokenizer = add_gist_tokens(tokenizer, args.gist_token_count, model_cfg.vocab_size)
        gist_ids = gist_token_ids(folding_tokenizer, args.gist_token_count)

        documents = list(_folding_document_source(args))
        ppl_result = perplexity_at_extended_context(
            model, folding_tokenizer, documents, gist_ids, args.fold_block_size, args.fold_window_len, device,
            micro_batch_size=args.micro_batch_size, max_batches=args.val_batches,
        )
        print(
            f"[eval] folding perplexity: folded={ppl_result['folded_perplexity']:.2f} "
            f"truncated_baseline={ppl_result['truncated_baseline_perplexity']:.2f} "
            f"(window_len={ppl_result['window_len']}, context_len={ppl_result['context_len']})"
        )

        needle_result = needle_in_haystack_eval(
            model, folding_tokenizer, gist_ids, args.fold_block_size, args.fold_window_len, device,
            n_trials=args.needle_trials,
        )
        print(
            f"[eval] needle-in-haystack: folded_accuracy={needle_result['folded_accuracy']:.1%} "
            f"truncated_baseline_accuracy={needle_result['truncated_baseline_accuracy']:.1%} "
            f"(n_trials={needle_result['n_trials']})"
        )


if __name__ == "__main__":
    main()
