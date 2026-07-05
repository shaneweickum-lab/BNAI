"""Shared helpers between train.py (Stage A) and sft.py (Stage B) -- device
selection, LR schedule, checkpointing, and held-out perplexity evaluation are
identical between the two loops, only the data source and loss masking differ.
"""
from __future__ import annotations

import math
import os

import torch
import yaml


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_lr_schedule(peak_lr: float, warmup_fraction: float, min_lr_fraction: float, total_steps: int):
    warmup_steps = max(1, int(warmup_fraction * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return peak_lr * (min_lr_fraction + (1 - min_lr_fraction) * cosine)

    return lr_at


def cycle_iterable(make_iterator):
    """Restarts a stream when exhausted -- needed for small local smoke-test
    corpora; a real multi-billion-token streaming corpus is large enough
    that this loop body only ever runs once in practice."""
    while True:
        yielded_any = False
        for item in make_iterator():
            yielded_any = True
            yield item
        if not yielded_any:
            raise RuntimeError("data source produced zero examples")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, step, tokens_seen, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": cfg,
    }
    path = os.path.join(output_dir, "latest.pt")
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # never leaves a half-written "latest.pt" if interrupted
    torch.save(payload, os.path.join(output_dir, f"step_{step}.pt"))


@torch.no_grad()
def estimate_perplexity(model, val_batches, device, amp_dtype=torch.bfloat16, max_batches: int = 50) -> float:
    """Mean cross-entropy over up to `max_batches` batches from `val_batches`
    (an iterable of {"input_ids","targets"} dicts), converted to perplexity."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = device.type in ("cuda", "cpu")
    for i, batch in enumerate(val_batches):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(input_ids, targets)
        total_loss += loss.item()
        n += 1
    if was_training:
        model.train()
    if n == 0:
        return float("nan")
    return math.exp(min(total_loss / n, 20))
