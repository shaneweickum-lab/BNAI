"""Streaming data pipeline for both training stages (spec Section 4).

Stage A (base pretrain): a stream of raw documents (FineWeb-Edu) is tokenized
on the fly and packed into fixed-length `context_len` blocks -- the corpus
never needs to be materialized in memory, which matters at the ~1.5-2B token
scale this model trains on.

Stage B (SFT): a stream of multi-turn conversations (UltraChat 200k / OASST2)
is rendered through the chat template, packed the same way, with a loss mask
so the model is only trained to predict assistant turns, not the user/system
turns it's conditioned on.

Both stages reserve a deterministic, hash-based held-out validation split
*before* any training starts, so the same document never appears in both
train and val across resumed runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))
from bpe import BPETokenizer, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, TURN_END_ID  # noqa: E402


def _split_bucket(key: str, num_buckets: int = 10_000) -> int:
    """Deterministic hash bucket in [0, num_buckets) -- used to carve out a
    stable held-out split without needing to know the dataset size upfront."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def is_val_split(key: str, val_fraction: float) -> bool:
    return _split_bucket(key) < int(val_fraction * 10_000)


# ---------------------------------------------------------------------------
# Stage A: base pretrain
# ---------------------------------------------------------------------------


def iter_local_jsonl_documents(path: str, text_field: str = "text") -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if text:
                yield text


def iter_hf_streaming_documents(
    dataset_name: str, split: str = "train", text_field: str = "text", **load_kwargs
) -> Iterator[str]:
    """Yields raw document text from an HF `datasets` streaming source
    (e.g. dataset_name="HuggingFaceFW/fineweb-edu"). Imported lazily since
    `datasets` is an optional, heavy dependency only needed for real training
    runs, not for the architecture/tokenizer unit tests."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs)
    for example in ds:
        text = example.get(text_field)
        if text:
            yield text


class TokenBlockDataset(IterableDataset):
    """Tokenizes a stream of documents and packs them into contiguous
    `block_len`-token blocks (input_ids, targets), with `<bos>`/`<eos>`
    inserted between documents so the model learns document boundaries."""

    def __init__(
        self,
        documents: Iterable[str],
        tokenizer: BPETokenizer,
        block_len: int,
        val_fraction: float = 0.0,
        split: str = "train",
    ):
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.block_len = block_len
        self.val_fraction = val_fraction
        self.split = split

    def __iter__(self) -> Iterator[dict]:
        buffer: list[int] = []
        for doc_idx, text in enumerate(self.documents):
            key = f"{doc_idx}:{text[:64]}"
            in_val = is_val_split(key, self.val_fraction)
            if self.split == "val" and not in_val:
                continue
            if self.split == "train" and in_val:
                continue

            buffer.append(BOS_ID)
            buffer.extend(self.tokenizer.encode(text))
            buffer.append(EOS_ID)

            while len(buffer) >= self.block_len + 1:
                block = buffer[: self.block_len + 1]
                buffer = buffer[self.block_len :]
                input_ids = torch.tensor(block[:-1], dtype=torch.long)
                targets = torch.tensor(block[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}


# ---------------------------------------------------------------------------
# Stage B: conversational SFT
# ---------------------------------------------------------------------------

_ROLE_TO_TOKEN_ID = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}


@dataclass
class ChatExample:
    turns: list  # list of {"role": "system"|"user"|"assistant", "content": str}


def iter_local_jsonl_conversations(path: str, turns_field: str = "messages") -> Iterator[ChatExample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = obj.get(turns_field, [])
            if turns:
                yield ChatExample(turns=turns)


def iter_hf_streaming_conversations(
    dataset_name: str, split: str = "train", turns_field: str = "messages", **load_kwargs
) -> Iterator[ChatExample]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs)
    for example in ds:
        turns = example.get(turns_field)
        if turns:
            yield ChatExample(turns=turns)


class ChatSFTDataset(IterableDataset):
    """Renders conversations through the chat template and packs them into
    `block_len`-token blocks. `targets` masks non-assistant tokens with -100
    so the loss (spec Section 4's F.cross_entropy `ignore_index=-100`) only
    scores assistant turns -- the model is trained to generate replies, not
    to reproduce the prompts it's conditioned on."""

    def __init__(
        self,
        conversations: Iterable[ChatExample],
        tokenizer: BPETokenizer,
        block_len: int,
        val_fraction: float = 0.0,
        split: str = "train",
    ):
        super().__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.block_len = block_len
        self.val_fraction = val_fraction
        self.split = split

    def _render(self, example: ChatExample):
        ids: list[int] = [BOS_ID]
        loss_mask: list[bool] = [False]
        for turn in example.turns:
            role = turn["role"]
            content = turn["content"]
            role_token = _ROLE_TO_TOKEN_ID[role]

            ids.append(role_token)
            loss_mask.append(False)

            content_ids = self.tokenizer.encode(content)
            ids.extend(content_ids)
            loss_mask.extend([role == "assistant"] * len(content_ids))

            ids.append(TURN_END_ID)
            loss_mask.append(role == "assistant")
        return ids, loss_mask

    def __iter__(self) -> Iterator[dict]:
        buffer_ids: list[int] = []
        buffer_mask: list[bool] = []
        for conv_idx, example in enumerate(self.conversations):
            key = f"{conv_idx}:{example.turns[0]['content'][:64] if example.turns else conv_idx}"
            in_val = is_val_split(key, self.val_fraction)
            if self.split == "val" and not in_val:
                continue
            if self.split == "train" and in_val:
                continue

            ids, mask = self._render(example)
            buffer_ids.extend(ids)
            buffer_mask.extend(mask)

            while len(buffer_ids) >= self.block_len + 1:
                id_block = buffer_ids[: self.block_len + 1]
                mask_block = buffer_mask[: self.block_len + 1]
                buffer_ids = buffer_ids[self.block_len :]
                buffer_mask = buffer_mask[self.block_len :]

                input_ids = torch.tensor(id_block[:-1], dtype=torch.long)
                raw_targets = torch.tensor(id_block[1:], dtype=torch.long)
                target_mask = torch.tensor(mask_block[1:], dtype=torch.bool)
                targets = torch.where(target_mask, raw_targets, torch.tensor(-100, dtype=torch.long))
                yield {"input_ids": input_ids, "targets": targets}
