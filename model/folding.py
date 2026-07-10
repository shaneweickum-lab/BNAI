"""Context-folding utilities for Benny's Phase 2 continued-pretraining
curriculum (spec: "folding head" architecture -- gist tokens as ordinary
vocabulary entries, trained via a segment-conditioning attention mask, no
separate encoder/decoder network and no auxiliary reconstruction loss).

Gist tokens are learned embeddings like any other token in
`model/architecture.py`'s tied embedding table, and flow through the exact
same BitLinear-ternary-quantized attention/FFN stack as every other
token -- this is the deliberate departure from prior art (Gist Tokens,
AutoCompressors, ICAE), all of which compress on a full-precision
backbone. The only architectural change this requires is *attention
masking*: a "segment-conditioning" mask that lets each block's tokens see
prior blocks' gist tokens but not their raw tokens. That's what actually
reduces the attended context (fewer keys/values participate in each
softmax), not just a display simplification -- see
`build_segment_conditioning_mask` below and
`architecture.py::CausalSelfAttention`'s `attn_mask` parameter for where
it plugs into the forward pass.

Loss function: standard next-token prediction only, same as Phase 1 --
no auxiliary reconstruction loss, no second decoder path.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def gist_token_string(index: int) -> str:
    return f"<|GIST_{index}|>"


def add_gist_tokens(tokenizer, gist_token_count: int, model_vocab_size: int):
    """Returns a new tokenizer (same class as `tokenizer`) with
    `gist_token_count` extra vocabulary entries appended after the base
    tokenizer's existing vocab.

    Gist tokens are plain vocabulary entries, not `special_tokens` --
    they decode to their literal placeholder text (`<|GIST_0|>`, ...),
    which is useful for seeing where folds happened in eval output rather
    than having them silently vanish the way `<pad>`/`<bos>` do.

    Requires `model_vocab_size` (the model's fixed embedding-table size,
    e.g. `BNAIConfig.vocab_size`) to have enough headroom above the base
    tokenizer's vocab_size. When it does, a Phase 1 checkpoint loads
    directly into Phase 2 with *zero* architecture change or embedding-
    table surgery: the new gist tokens simply occupy previously-unused
    embedding rows -- the same headroom that already lets an undersized
    placeholder tokenizer work against a fixed model vocab_size elsewhere
    in this repo (see model/export.py's placeholder export).
    """
    needed = tokenizer.vocab_size + gist_token_count
    if needed > model_vocab_size:
        raise ValueError(
            f"model_vocab_size={model_vocab_size} has no room for {gist_token_count} gist tokens on top "
            f"of the base tokenizer's {tokenizer.vocab_size} entries (needs {needed}). Reduce "
            "gist_token_count, retrain the base tokenizer with a smaller vocab_size, or increase "
            "the model's vocab_size (which requires re-pretraining from scratch, not just Phase 2)."
        )

    new_vocab = dict(tokenizer.vocab)
    base_id = len(new_vocab)
    for i in range(gist_token_count):
        new_vocab[gist_token_string(i)] = base_id + i

    return type(tokenizer)(vocab=new_vocab, merges=list(tokenizer.merges))


def gist_token_ids(tokenizer_with_gist, gist_token_count: int) -> list:
    """IDs of the gist tokens in a tokenizer built by `add_gist_tokens`."""
    ids = [tokenizer_with_gist.vocab[gist_token_string(i)] for i in range(gist_token_count)]
    return ids


@dataclass
class FoldedSequence:
    token_ids: list
    block_ids: list  # same length as token_ids -- which block each position belongs to
    is_gist: list  # same length -- True for gist-token positions


def fold_document(token_ids: list, block_size: int, gist_ids: list) -> FoldedSequence:
    """Chunks a flat (already BPE-encoded) token-id sequence into
    fixed-size blocks and inserts `gist_ids` after every block, including
    the last (spec 2.2: "After each block, insert the N gist token IDs"):

        [raw tokens of block 0 (<= block_size)] [gist_ids]
        [raw tokens of block 1 (<= block_size)] [gist_ids]
        ...

    The last block may be shorter than `block_size` (not padded). Every
    block gets its gist tokens appended, including a short final block --
    at inference time the most-recently-completed block is exactly the
    one that would get folded next, so the model needs to learn to
    produce a useful gist summary after any block, not just full ones.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    new_ids: list = []
    block_ids: list = []
    is_gist: list = []

    current_block = 0
    for start in range(0, len(token_ids), block_size):
        chunk = token_ids[start : start + block_size]
        for tid in chunk:
            new_ids.append(tid)
            block_ids.append(current_block)
            is_gist.append(False)
        for gid in gist_ids:
            new_ids.append(gid)
            block_ids.append(current_block)
            is_gist.append(True)
        current_block += 1

    return FoldedSequence(token_ids=new_ids, block_ids=block_ids, is_gist=is_gist)


def build_segment_conditioning_mask(block_ids: torch.Tensor, is_gist: torch.Tensor) -> torch.Tensor:
    """Boolean `[seq_len, seq_len]` mask suitable for
    `F.scaled_dot_product_attention`'s `attn_mask` (`True` = query position
    may attend to key position).

    `allowed[q, k] = causal(q, k) AND (block_ids[k] == block_ids[q] OR is_gist[k])`

    In words: ordinary causal ordering, PLUS an extra exclusion -- raw
    (non-gist) tokens belonging to a strictly earlier block than the
    query's own block are masked out entirely, even though plain causal
    ordering would otherwise allow attending to them. A block's gist
    tokens remain visible to every later position regardless of the
    query's position within its own block. This is the actual
    compute-reduction mechanism (fewer keys/values enter each softmax as
    more blocks get folded), not merely a UI-level compression.

    `block_ids`/`is_gist`: 1-D tensors of length `seq_len`, as produced by
    `fold_document` (converted to tensors by the caller).

    Honesty note on "reduces compute": during *training*, PyTorch's
    `scaled_dot_product_attention` still computes the full O(seq_len^2)
    QK^T matrix and masks it -- a boolean mask alone doesn't skip that
    work. This mask's job during training is to teach the model that only
    prior blocks' gist tokens matter (so it produces a gist representation
    good enough to rely on), which is what makes it *safe and correct* to
    literally evict a folded block's raw K/V cache entries at *inference*
    time in a real serving runtime -- that eviction, not this training-time
    mask, is where the real compute/memory reduction is realized. Building
    that inference-time runtime is out of scope for this training pass.
    """
    seq_len = block_ids.shape[0]
    device = block_ids.device

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    same_block = block_ids.unsqueeze(1) == block_ids.unsqueeze(0)  # [seq_len, seq_len]
    gist_visible = is_gist.unsqueeze(0).expand(seq_len, seq_len)  # key is a gist token

    return causal & (same_block | gist_visible)


def build_batched_segment_conditioning_mask(block_ids: torch.Tensor, is_gist: torch.Tensor) -> torch.Tensor:
    """Batched version of `build_segment_conditioning_mask` for training:
    `block_ids`/`is_gist` are `[batch, seq_len]` (each row may have
    different block boundaries -- e.g. different documents concatenated
    at different offsets within the window). Returns a
    `[batch, 1, seq_len, seq_len]` boolean mask, broadcastable across
    attention heads when passed as `CausalSelfAttention`'s `attn_mask`."""
    batch, seq_len = block_ids.shape
    device = block_ids.device

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0)
    same_block = block_ids.unsqueeze(2) == block_ids.unsqueeze(1)  # [batch, seq_len, seq_len]
    gist_visible = is_gist.unsqueeze(1).expand(batch, seq_len, seq_len)

    mask = causal & (same_block | gist_visible)  # [batch, seq_len, seq_len]
    return mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
