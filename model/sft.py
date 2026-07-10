"""Conversational fine-tune, Stage B (spec Section 4 / Phase 5).

Continues training from a Stage A base-pretrain checkpoint on multi-turn
chat data (UltraChat 200k + OASST2), with the loss masked to only score
assistant turns (see data/pipeline.py's ChatSFTDataset). Not counted in the
Chinchilla pretraining budget.

Watches for "alignment tax": periodically re-evaluates perplexity on the same
held-out split of the *base* pretrain corpus (not the chat data) and warns if
it regresses more than `max_base_ppl_regression` from the step-0 measurement,
per spec Section 4's guidance to reduce SFT epochs/LR if that happens.

Usage:
    python sft.py --config configs/sft.yaml --tokenizer tokenizer/bnai_tokenizer.json \
        --base-checkpoint checkpoints/base_ternary/latest.pt

    # Smoke test with local jsonl corpora for both chat data and the base-val check:
    python sft.py --config configs/sft.yaml --tokenizer tokenizer/bnai_tokenizer.json \
        --base-checkpoint checkpoints/base_ternary/latest.pt \
        --local-chat-corpus /path/to/chat/*.jsonl --local-base-val-corpus /path/to/base/*.jsonl \
        --smoke-test-steps 50
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
from data.pipeline import (  # noqa: E402
    ChatSFTDataset,
    TokenBlockDataset,
    iter_hf_streaming_conversations,
    iter_hf_streaming_documents,
    iter_local_jsonl_conversations,
    iter_local_jsonl_documents,
)
from export import export_model_to_bnai  # noqa: E402
from train_utils import (  # noqa: E402
    build_lr_schedule,
    cycle_iterable,
    estimate_perplexity,
    load_config,
    save_checkpoint,
    select_device,
)


def make_chat_source(args, cfg: dict):
    if args.local_chat_corpus:
        paths = sorted(glob.glob(args.local_chat_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-chat-corpus {args.local_chat_corpus}")

        def make_iterator():
            return itertools.chain.from_iterable(iter_local_jsonl_conversations(p) for p in paths)

    else:
        data_cfg = cfg["data"]

        def make_iterator():
            return iter_hf_streaming_conversations(
                data_cfg["sft_dataset"], data_cfg.get("sft_split", "train"), data_cfg.get("turns_field", "messages")
            )

    return make_iterator


def make_base_val_source(args, cfg: dict):
    if args.local_base_val_corpus:
        paths = sorted(glob.glob(args.local_base_val_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-base-val-corpus {args.local_base_val_corpus}")

        def make_iterator():
            return itertools.chain.from_iterable(iter_local_jsonl_documents(p) for p in paths)

    else:
        data_cfg = cfg["data"]

        def make_iterator():
            return iter_hf_streaming_documents(
                data_cfg["base_val_dataset"], data_cfg.get("base_val_split", "train"), data_cfg.get("base_val_text_field", "text")
            )

    return make_iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", required=True, help="Stage A checkpoint (train.py's latest.pt or step_N.pt)")
    parser.add_argument("--local-chat-corpus", default=None, help="glob of local .jsonl conversation files")
    parser.add_argument("--local-base-val-corpus", default=None, help="glob of local .jsonl base-domain files, for the alignment-tax check")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test-steps", type=int, default=None)
    parser.add_argument("--base-val-interval-steps", type=int, default=100)
    parser.add_argument("--base-val-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    output_dir = cfg["output_dir"]

    device = select_device()
    print(f"[sft] device={device}")

    tokenizer = BPETokenizer.load(args.tokenizer)

    # Architecture always comes from a checkpoint (resumed SFT run, or the
    # base pretrain checkpoint being fine-tuned), never from sft.yaml -- SFT
    # continues training whatever architecture the checkpoint already is,
    # so one sft.yaml works across the whole parameter ladder (Stage 1-4)
    # instead of needing a duplicate per-stage config that must be kept in
    # sync with the base config by hand.
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
        print(f"[sft] resumed from step {start_step} ({tokens_seen:,} tokens seen)")
    else:
        base_payload = torch.load(args.base_checkpoint, map_location=device)
        model_cfg = BNAIConfig(**base_payload["config"]["model"])
        model = BNAILanguageModel(model_cfg).to(device)
        model.load_state_dict(base_payload["model_state_dict"])
        print(f"[sft] loaded base checkpoint from {args.base_checkpoint} ({model_cfg.d_model=}, {model_cfg.n_layers=})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    if resumed:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    grad_accum_steps = max(
        1, train_cfg["effective_batch_tokens"] // (train_cfg["micro_batch_size"] * model_cfg.context_len)
    )
    total_optimizer_steps = train_cfg["target_tokens"] // train_cfg["effective_batch_tokens"]
    if args.smoke_test_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.smoke_test_steps)
    lr_at = build_lr_schedule(
        train_cfg["peak_lr"], train_cfg["warmup_fraction"], train_cfg["min_lr_fraction"], total_optimizer_steps
    )

    make_chat_iterator = make_chat_source(args, cfg)
    val_fraction = cfg["data"]["val_fraction"]
    train_dataset = ChatSFTDataset(
        cycle_iterable(make_chat_iterator), tokenizer, model_cfg.context_len, val_fraction=val_fraction, split="train"
    )
    loader = DataLoader(train_dataset, batch_size=train_cfg["micro_batch_size"])
    data_iter = iter(loader)

    make_base_val_iterator = make_base_val_source(args, cfg)
    base_val_fraction = cfg["data"]["base_val_fraction"]

    def make_base_val_batches():
        val_dataset = TokenBlockDataset(
            make_base_val_iterator(), tokenizer, model_cfg.context_len, val_fraction=base_val_fraction, split="val"
        )
        return iter(DataLoader(val_dataset, batch_size=train_cfg["micro_batch_size"]))

    amp_dtype = getattr(torch, train_cfg.get("amp_dtype", "bfloat16"))
    use_amp = device.type in ("cuda", "cpu")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")

    # The alignment-tax reference must be the *pre-SFT* base checkpoint's
    # perplexity, fixed once and reused across resumed sessions -- recomputing
    # it against a partially-fine-tuned model on every --resume would make
    # the regression comparison drift against itself over a multi-session run.
    reference_path = os.path.join(output_dir, "reference_base_ppl.json")
    if args.resume and os.path.exists(reference_path):
        with open(reference_path) as f:
            reference_base_ppl = json.load(f)["reference_base_ppl"]
    else:
        reference_base_ppl = estimate_perplexity(model, make_base_val_batches(), device, amp_dtype, args.base_val_batches)
        os.makedirs(output_dir, exist_ok=True)
        with open(reference_path, "w") as f:
            json.dump({"reference_base_ppl": reference_base_ppl}, f)
    print(f"[sft] reference base-domain val perplexity (pre-SFT): {reference_base_ppl:.2f}")
    max_regression = train_cfg.get("max_base_ppl_regression", 0.10)

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
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "sft_train_loss": mean_loss,
                "sft_train_perplexity": ppl,
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "tokens_per_sec": tok_per_sec,
            }
            print(
                f"[sft] step={step}/{total_optimizer_steps} tokens={tokens_seen:,} "
                f"loss={mean_loss:.4f} ppl={ppl:.2f} grad_norm={grad_norm:.3f} lr={lr:.2e} tok/s={tok_per_sec:.0f}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if step % args.base_val_interval_steps == 0 or step == total_optimizer_steps:
            base_ppl = estimate_perplexity(model, make_base_val_batches(), device, amp_dtype, args.base_val_batches)
            regression = (base_ppl - reference_base_ppl) / reference_base_ppl
            flag = " *** ALIGNMENT TAX WARNING ***" if regression > max_regression else ""
            print(f"[sft] step={step} base-domain val perplexity={base_ppl:.2f} (regression={regression:+.1%}){flag}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "base_val_perplexity": base_ppl, "base_ppl_regression": regression}) + "\n")

        if step % train_cfg["checkpoint_interval_steps"] == 0 or step == total_optimizer_steps:
            save_checkpoint(model, optimizer, step, tokens_seen, cfg, output_dir)
            print(f"[sft] checkpoint saved at step {step}")

        if model_cfg.ternary_weights and (
            step % train_cfg["export_interval_steps"] == 0 or step == total_optimizer_steps
        ):
            export_path = os.path.join(output_dir, f"exported_step_{step}.bnai")
            export_model_to_bnai(model, tokenizer, export_path)
            print(f"[sft] exported ternary checkpoint -> {export_path}")

    print("[sft] done")


if __name__ == "__main__":
    main()
