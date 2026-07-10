"""Phase 2: continued pretraining with context-folding introduced (spec
Section 3). Resumes from a stable Phase 1 checkpoint (train.py) and adds
gist-token vocabulary + the segment-conditioning attention mask
(model/folding.py) on long/concatenated-document training data.

No architecture change or checkpoint surgery is needed to move from Phase 1
to Phase 2: the model's `vocab_size` already has headroom above the base
tokenizer's real vocab (the same headroom that lets an undersized
placeholder tokenizer work in this repo -- see model/export.py), so the new
gist tokens simply occupy previously-unused embedding rows. This script
loads the Phase 1 checkpoint's `model_state_dict` directly.

Loss function: standard next-token prediction only (same as Phase 1) -- no
auxiliary reconstruction loss, no second decoder path (spec 2.4).

Usage:
    # Real run (resumes from a Phase 1 checkpoint, needs a long-document/
    # concatenated corpus):
    python phase2_train.py --config configs/phase2_gist8.yaml \
        --tokenizer tokenizer/bnai_tokenizer.json \
        --base-checkpoint checkpoints/stage4_125m_ternary/latest.pt

    # Smoke test with a local corpus:
    python phase2_train.py --config configs/phase2_gist8.yaml \
        --tokenizer tokenizer/bnai_tokenizer.json \
        --base-checkpoint checkpoints/stage4_125m_ternary/latest.pt \
        --local-corpus /path/to/*.jsonl --smoke-test-steps 200

    # Resume an interrupted Phase 2 run:
    python phase2_train.py --config configs/phase2_gist8.yaml \
        --tokenizer tokenizer/bnai_tokenizer.json \
        --base-checkpoint checkpoints/stage4_125m_ternary/latest.pt --resume
"""
from __future__ import annotations

import argparse
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
from data.pipeline import FoldedTokenBlockDataset  # noqa: E402
from folding import add_gist_tokens, build_batched_segment_conditioning_mask, gist_token_ids  # noqa: E402
from train import make_document_source  # noqa: E402  (reuses Phase 1's local-jsonl/HF-streaming source builder)
from train_utils import (  # noqa: E402
    build_lr_schedule,
    cycle_iterable,
    estimate_perplexity,
    load_config,
    save_checkpoint,
    select_device,
)


def _mask_fn(batch):
    return build_batched_segment_conditioning_mask(batch["block_ids"], batch["is_gist"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, help="path to the BASE (Phase 1) tokenizer JSON, no gist tokens yet")
    parser.add_argument("--base-checkpoint", required=True, help="Phase 1 checkpoint to resume from (ignored if --resume finds a Phase 2 checkpoint)")
    parser.add_argument("--local-corpus", default=None, help="glob of local .jsonl files, for smoke tests / offline runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test-steps", type=int, default=None)
    parser.add_argument("--val-interval-steps", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    folding_cfg = cfg["folding"]
    train_cfg = cfg["training"]
    output_dir = cfg["output_dir"]

    device = select_device()
    print(f"[phase2] device={device}")

    base_tokenizer = BPETokenizer.load(args.tokenizer)

    start_step = 0
    tokens_seen = 0
    latest_ckpt = os.path.join(output_dir, "latest.pt")
    resumed = args.resume and os.path.exists(latest_ckpt)

    if resumed:
        payload = torch.load(latest_ckpt, map_location=device)
        model_cfg = BNAIConfig(**payload["config"]["model"])
        model = BNAILanguageModel(model_cfg).to(device)
        model.load_state_dict(payload["model_state_dict"])
        start_step = payload["step"]
        tokens_seen = payload["tokens_seen"]
        print(f"[phase2] resumed from step {start_step} ({tokens_seen:,} tokens seen)")
    else:
        base_payload = torch.load(args.base_checkpoint, map_location=device)
        model_cfg = BNAIConfig(**base_payload["config"]["model"])
        model = BNAILanguageModel(model_cfg).to(device)
        model.load_state_dict(base_payload["model_state_dict"])
        print(
            f"[phase2] loaded Phase 1 checkpoint from {args.base_checkpoint} "
            f"({model_cfg.d_model=}, {model_cfg.n_layers=}) -- no architecture change, "
            "gist tokens occupy previously-unused embedding rows"
        )

    tokenizer = add_gist_tokens(base_tokenizer, folding_cfg["gist_token_count"], model_cfg.vocab_size)
    gist_ids = gist_token_ids(tokenizer, folding_cfg["gist_token_count"])
    print(
        f"[phase2] gist_token_count={folding_cfg['gist_token_count']} "
        f"fold_block_size={folding_cfg['fold_block_size']} window_len={folding_cfg['window_len']} "
        f"gist_ids={gist_ids}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    if resumed:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    window_len = folding_cfg["window_len"]
    fold_block_size = folding_cfg["fold_block_size"]

    grad_accum_steps = max(1, train_cfg["effective_batch_tokens"] // (train_cfg["micro_batch_size"] * window_len))
    total_optimizer_steps = train_cfg["target_tokens"] // train_cfg["effective_batch_tokens"]
    if args.smoke_test_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.smoke_test_steps)
    lr_at = build_lr_schedule(
        train_cfg["peak_lr"], train_cfg["warmup_fraction"], train_cfg["min_lr_fraction"], total_optimizer_steps
    )

    make_iterator = make_document_source(args, cfg)
    val_fraction = cfg["data"]["val_fraction"]

    train_dataset = FoldedTokenBlockDataset(
        cycle_iterable(make_iterator),
        tokenizer,
        window_len=window_len,
        gist_ids=gist_ids,
        fold_block_size=fold_block_size,
        val_fraction=val_fraction,
        split="train",
    )
    loader = DataLoader(train_dataset, batch_size=train_cfg["micro_batch_size"])
    data_iter = iter(loader)

    def make_val_batches():
        val_dataset = FoldedTokenBlockDataset(
            make_iterator(),
            tokenizer,
            window_len=window_len,
            gist_ids=gist_ids,
            fold_block_size=fold_block_size,
            val_fraction=val_fraction,
            split="val",
        )
        return iter(DataLoader(val_dataset, batch_size=train_cfg["micro_batch_size"]))

    amp_dtype = getattr(torch, train_cfg.get("amp_dtype", "bfloat16"))
    use_amp = device.type in ("cuda", "cpu")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")
    tokenizer.save(os.path.join(output_dir, "tokenizer_with_gist.json"))

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
            attn_mask = _mask_fn(batch).to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, targets, attn_mask=attn_mask)
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
                "phase2_train_loss": mean_loss,
                "phase2_train_perplexity": ppl,
                "grad_norm": grad_norm.item(),
                "weight_sparsity": sparsity,
                "lr": lr,
                "tokens_per_sec": tok_per_sec,
            }
            print(
                f"[phase2] step={step}/{total_optimizer_steps} tokens={tokens_seen:,} "
                f"loss={mean_loss:.4f} ppl={ppl:.2f} grad_norm={grad_norm:.3f} "
                f"sparsity={sparsity:.3f} lr={lr:.2e} tok/s={tok_per_sec:.0f}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if step % args.val_interval_steps == 0 or step == total_optimizer_steps:
            val_ppl = estimate_perplexity(
                model, make_val_batches(), device, amp_dtype, args.val_batches, mask_fn=_mask_fn
            )
            print(f"[phase2] step={step} held-out folded-val perplexity={val_ppl:.2f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "phase2_val_perplexity": val_ppl}) + "\n")

        if step % train_cfg["checkpoint_interval_steps"] == 0 or step == total_optimizer_steps:
            save_checkpoint(model, optimizer, step, tokens_seen, cfg, output_dir)
            print(f"[phase2] checkpoint saved at step {step}")

    print("[phase2] done")


if __name__ == "__main__":
    main()
