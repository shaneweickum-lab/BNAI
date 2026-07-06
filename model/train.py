"""Base pretrain loop (spec Section 5 / Phase 3-4).

Single-device (no distributed training scaffolding -- this targets a single
MacBook M5 via PyTorch MPS, with CPU/CUDA fallback for portability). Reaches
the configured effective batch size via gradient accumulation rather than a
single large forward/backward pass, since the full 256x2048-token batch in
the spec almost certainly doesn't fit in 24GB of shared unified memory.

Usage:
    # Real run (needs network access to stream FineWeb-Edu + a trained tokenizer):
    python train.py --config configs/base_ternary.yaml --tokenizer tokenizer/bnai_tokenizer.json

    # Phase 3 smoke test -- confirms loss decreases and reports measured
    # tokens/sec on *this* machine before committing to the full run:
    python train.py --config configs/base_ternary.yaml --tokenizer tokenizer/bnai_tokenizer.json \
        --local-corpus /path/to/*.jsonl --smoke-test-steps 200

    # Resume an interrupted run:
    python train.py --config configs/base_ternary.yaml --tokenizer tokenizer/bnai_tokenizer.json --resume
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

from architecture import BNAIConfig, BNAILanguageModel  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from data.pipeline import TokenBlockDataset, iter_hf_streaming_documents, iter_local_jsonl_documents  # noqa: E402
from export import export_model_to_bnai  # noqa: E402
from train_utils import (  # noqa: E402
    build_lr_schedule,
    cycle_iterable,
    estimate_perplexity,
    load_config,
    save_checkpoint,
    select_device,
)


def build_model_config(cfg: dict) -> BNAIConfig:
    return BNAIConfig(**cfg["model"])


def make_document_source(args, cfg: dict):
    if args.local_corpus:
        paths = sorted(glob.glob(args.local_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-corpus {args.local_corpus}")

        def make_iterator():
            return itertools.chain.from_iterable(iter_local_jsonl_documents(p) for p in paths)

    else:
        data_cfg = cfg["data"]

        def make_iterator():
            return iter_hf_streaming_documents(
                data_cfg["train_dataset"], data_cfg.get("train_split", "train"), data_cfg.get("text_field", "text")
            )

    return make_iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--local-corpus", default=None, help="glob of local .jsonl files, for smoke tests / offline runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test-steps", type=int, default=None, help="cap total optimizer steps (Phase 3 smoke test)")
    parser.add_argument("--val-interval-steps", type=int, default=100, help="how often to evaluate held-out perplexity")
    parser.add_argument("--val-batches", type=int, default=20, help="number of val batches per evaluation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    model_cfg = build_model_config(cfg)
    train_cfg = cfg["training"]
    output_dir = cfg["output_dir"]

    device = select_device()
    print(f"[train] device={device}")

    tokenizer = BPETokenizer.load(args.tokenizer)
    assert tokenizer.vocab_size <= model_cfg.vocab_size, "tokenizer vocab larger than model vocab_size"

    model = BNAILanguageModel(model_cfg).to(device)
    param_count = model.num_parameters()
    print(f"[train] model params: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )

    grad_accum_steps = max(
        1, train_cfg["effective_batch_tokens"] // (train_cfg["micro_batch_size"] * model_cfg.context_len)
    )
    total_optimizer_steps = train_cfg["target_tokens"] // train_cfg["effective_batch_tokens"]
    if args.smoke_test_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.smoke_test_steps)
    lr_at = build_lr_schedule(
        train_cfg["peak_lr"], train_cfg["warmup_fraction"], train_cfg["min_lr_fraction"], total_optimizer_steps
    )

    start_step = 0
    tokens_seen = 0
    latest_ckpt = os.path.join(output_dir, "latest.pt")
    if args.resume and os.path.exists(latest_ckpt):
        payload = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = payload["step"]
        tokens_seen = payload["tokens_seen"]
        print(f"[train] resumed from step {start_step} ({tokens_seen:,} tokens seen)")

    make_iterator = make_document_source(args, cfg)
    val_fraction = cfg["data"]["val_fraction"]

    train_dataset = TokenBlockDataset(
        cycle_iterable(make_iterator), tokenizer, model_cfg.context_len, val_fraction=val_fraction, split="train"
    )
    loader = DataLoader(train_dataset, batch_size=train_cfg["micro_batch_size"])
    data_iter = iter(loader)

    def make_val_batches():
        val_dataset = TokenBlockDataset(
            make_iterator(), tokenizer, model_cfg.context_len, val_fraction=val_fraction, split="val"
        )
        return iter(DataLoader(val_dataset, batch_size=train_cfg["micro_batch_size"]))

    amp_dtype = getattr(torch, train_cfg.get("amp_dtype", "bfloat16"))
    use_amp = device.type in ("cuda", "cpu")  # MPS autocast support varies by op -- verified live, not assumed

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")

    model.train()
    step = start_step
    while step < total_optimizer_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_tokens = 0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, targets)
            (loss / grad_accum_steps).backward()

            accum_loss += loss.item()
            micro_tokens += input_ids.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])

        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        tokens_seen += micro_tokens
        step += 1
        elapsed = time.time() - step_start
        tok_per_sec = micro_tokens / max(elapsed, 1e-9)

        if step % train_cfg["log_interval_steps"] == 0 or step == total_optimizer_steps:
            mean_loss = accum_loss / grad_accum_steps
            ppl = math.exp(min(mean_loss, 20))
            sparsity = model.mean_weight_sparsity()
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": mean_loss,
                "train_perplexity": ppl,
                "grad_norm": grad_norm.item(),
                "weight_sparsity": sparsity,
                "lr": lr,
                "tokens_per_sec": tok_per_sec,
            }
            print(
                f"[train] step={step}/{total_optimizer_steps} tokens={tokens_seen:,} "
                f"loss={mean_loss:.4f} ppl={ppl:.2f} grad_norm={grad_norm:.3f} "
                f"sparsity={sparsity:.3f} lr={lr:.2e} tok/s={tok_per_sec:.0f}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if step % args.val_interval_steps == 0 or step == total_optimizer_steps:
            val_ppl = estimate_perplexity(model, make_val_batches(), device, amp_dtype, args.val_batches)
            print(f"[train] step={step} held-out val perplexity={val_ppl:.2f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "val_perplexity": val_ppl}) + "\n")

        if step % train_cfg["checkpoint_interval_steps"] == 0 or step == total_optimizer_steps:
            save_checkpoint(model, optimizer, step, tokens_seen, cfg, output_dir)
            print(f"[train] checkpoint saved at step {step}")

        if model_cfg.ternary_weights and (
            step % train_cfg["export_interval_steps"] == 0 or step == total_optimizer_steps
        ):
            export_path = os.path.join(output_dir, f"exported_step_{step}.bnai")
            export_model_to_bnai(model, tokenizer, export_path)
            print(f"[train] exported ternary checkpoint -> {export_path}")

    print("[train] done")


if __name__ == "__main__":
    main()
