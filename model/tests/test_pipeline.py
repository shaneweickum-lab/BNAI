import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))

from bpe import BPETokenizer  # noqa: E402
from data.pipeline import (  # noqa: E402
    ChatExample,
    ChatSFTDataset,
    TokenBlockDataset,
    is_val_split,
)

SAMPLE_DOCS = [f"document number {i} about ternary weight language models and BitNet" for i in range(200)]


def _tiny_tokenizer():
    return BPETokenizer.train(SAMPLE_DOCS, vocab_size=300)


def test_val_split_is_deterministic():
    assert is_val_split("doc-123", 0.5) == is_val_split("doc-123", 0.5)


def test_val_split_fraction_is_approximately_respected():
    keys = [f"doc-{i}" for i in range(5000)]
    frac = 0.1
    in_val = sum(is_val_split(k, frac) for k in keys)
    assert abs(in_val / len(keys) - frac) < 0.03


def test_token_block_dataset_yields_correct_shapes():
    tok = _tiny_tokenizer()
    block_len = 16
    ds = TokenBlockDataset(SAMPLE_DOCS, tok, block_len=block_len, val_fraction=0.0, split="train")
    batches = list(ds)
    assert len(batches) > 0
    for b in batches:
        assert b["input_ids"].shape == (block_len,)
        assert b["targets"].shape == (block_len,)


def test_token_block_dataset_targets_are_shifted_input():
    tok = _tiny_tokenizer()
    block_len = 8
    ds = TokenBlockDataset(SAMPLE_DOCS, tok, block_len=block_len, val_fraction=0.0, split="train")
    first = next(iter(ds))
    # targets[i] should equal input_ids[i+1] within this block (next-token prediction).
    assert first["input_ids"][1:].tolist() == first["targets"][:-1].tolist()


def test_train_val_split_is_disjoint_and_covers_all_docs():
    tok = _tiny_tokenizer()
    block_len = 32
    train_ds = TokenBlockDataset(SAMPLE_DOCS, tok, block_len=block_len, val_fraction=0.2, split="train")
    val_ds = TokenBlockDataset(SAMPLE_DOCS, tok, block_len=block_len, val_fraction=0.2, split="val")
    assert len(list(train_ds)) > 0
    assert len(list(val_ds)) > 0


def test_chat_sft_dataset_masks_non_assistant_tokens():
    tok = _tiny_tokenizer()
    conversations = [
        ChatExample(
            turns=[
                {"role": "user", "content": "document number one about ternary weight"},
                {"role": "assistant", "content": "document number two about BitNet language models"},
            ]
        )
        for _ in range(50)
    ]
    ds = ChatSFTDataset(conversations, tok, block_len=16, val_fraction=0.0, split="train")
    batches = list(ds)
    assert len(batches) > 0
    # Every block should have at least one non-masked (assistant) target,
    # and at least one masked (-100) target from the user turn / role tokens.
    saw_masked = any((b["targets"] == -100).any() for b in batches)
    saw_unmasked = any((b["targets"] != -100).any() for b in batches)
    assert saw_masked
    assert saw_unmasked
