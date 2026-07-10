import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))

from bpe import BPETokenizer  # noqa: E402
from folding import (  # noqa: E402
    add_gist_tokens,
    build_batched_segment_conditioning_mask,
    build_segment_conditioning_mask,
    fold_document,
    gist_token_ids,
    gist_token_string,
)


def _tiny_tokenizer(vocab_size=300):
    texts = ["the quick brown fox jumps over the lazy dog " * 3] * 20
    return BPETokenizer.train(texts, vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# add_gist_tokens / gist_token_ids
# ---------------------------------------------------------------------------


def test_add_gist_tokens_appends_after_existing_vocab():
    tok = _tiny_tokenizer()
    base_size = tok.vocab_size
    tok_with_gist = add_gist_tokens(tok, gist_token_count=4, model_vocab_size=base_size + 100)

    assert tok_with_gist.vocab_size == base_size + 4
    for i in range(4):
        assert tok_with_gist.vocab[gist_token_string(i)] == base_size + i


def test_add_gist_tokens_preserves_base_vocab_ids():
    tok = _tiny_tokenizer()
    tok_with_gist = add_gist_tokens(tok, gist_token_count=4, model_vocab_size=tok.vocab_size + 100)
    for token_str, token_id in tok.vocab.items():
        assert tok_with_gist.vocab[token_str] == token_id


def test_add_gist_tokens_raises_without_headroom():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="no room"):
        add_gist_tokens(tok, gist_token_count=4, model_vocab_size=tok.vocab_size + 2)


def test_add_gist_tokens_decode_shows_literal_placeholder():
    tok = _tiny_tokenizer()
    tok_with_gist = add_gist_tokens(tok, gist_token_count=2, model_vocab_size=tok.vocab_size + 10)
    gid = gist_token_ids(tok_with_gist, 2)[0]
    assert tok_with_gist.decode([gid]) == gist_token_string(0)


def test_gist_token_ids_are_contiguous_and_ordered():
    tok = _tiny_tokenizer()
    tok_with_gist = add_gist_tokens(tok, gist_token_count=5, model_vocab_size=tok.vocab_size + 10)
    ids = gist_token_ids(tok_with_gist, 5)
    assert ids == list(range(tok.vocab_size, tok.vocab_size + 5))


# ---------------------------------------------------------------------------
# fold_document
# ---------------------------------------------------------------------------


def test_fold_document_inserts_gist_ids_after_every_block():
    raw = list(range(1, 251))  # 250 raw tokens
    folded = fold_document(raw, block_size=100, gist_ids=[901, 902])

    # 3 blocks: 100 + 100 + 50 raw tokens, each followed by 2 gist ids.
    assert len(folded.token_ids) == 250 + 3 * 2
    # First block: 100 raw tokens then the 2 gist ids.
    assert folded.token_ids[0:100] == raw[0:100]
    assert folded.token_ids[100:102] == [901, 902]
    # Last (short) block is 50 raw tokens, not padded, still gets gist ids appended.
    assert folded.token_ids[-52:] == raw[-50:] + [901, 902]


def test_fold_document_last_block_not_padded():
    raw = list(range(1, 51))  # 50 raw tokens, one short block
    folded = fold_document(raw, block_size=100, gist_ids=[999])
    assert folded.token_ids == raw + [999]
    assert folded.block_ids == [0] * 50 + [0]
    assert folded.is_gist == [False] * 50 + [True]


def test_fold_document_block_ids_increment_per_block():
    raw = list(range(1, 251))
    folded = fold_document(raw, block_size=100, gist_ids=[901])
    # positions 0-99 (block 0), 100 (gist for block 0), 101-200 (block 1 raw), 201 (gist), 202-251 (block 2 raw), 252 (gist)
    assert folded.block_ids[0] == 0
    assert folded.block_ids[100] == 0  # gist token still belongs to the block it summarizes
    assert folded.block_ids[101] == 1
    assert folded.block_ids[-1] == 2  # final gist token belongs to the last (short) block


def test_fold_document_is_gist_flags_only_gist_positions():
    raw = list(range(1, 11))
    folded = fold_document(raw, block_size=5, gist_ids=[100, 101])
    # block 0: 5 raw + 2 gist, block 1: 5 raw + 2 gist
    assert folded.is_gist == [False] * 5 + [True, True] + [False] * 5 + [True, True]


def test_fold_document_rejects_non_positive_block_size():
    with pytest.raises(ValueError):
        fold_document([1, 2, 3], block_size=0, gist_ids=[9])


# ---------------------------------------------------------------------------
# build_segment_conditioning_mask
# ---------------------------------------------------------------------------


def test_mask_allows_within_block_causal_attention():
    # Single block, no gist tokens yet visible from elsewhere.
    block_ids = torch.tensor([0, 0, 0])
    is_gist = torch.tensor([False, False, False])
    mask = build_segment_conditioning_mask(block_ids, is_gist)
    expected = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_mask_blocks_raw_tokens_of_earlier_block():
    # block 0: raw, raw, gist (positions 0,1,2); block 1: raw, raw (positions 3,4)
    block_ids = torch.tensor([0, 0, 0, 1, 1])
    is_gist = torch.tensor([False, False, True, False, False])
    mask = build_segment_conditioning_mask(block_ids, is_gist)

    # Position 3 (block 1, first raw token) must NOT see positions 0,1 (block 0's raw tokens)...
    assert not mask[3, 0]
    assert not mask[3, 1]
    # ...but MUST see position 2 (block 0's gist token).
    assert mask[3, 2]
    # And must see itself (causal, same block).
    assert mask[3, 3]


def test_mask_still_respects_causal_ordering_for_gist_tokens():
    block_ids = torch.tensor([0, 0, 0, 1, 1])
    is_gist = torch.tensor([False, False, True, False, False])
    mask = build_segment_conditioning_mask(block_ids, is_gist)
    # Gist token (position 2) cannot see the future (position 3, 4).
    assert not mask[2, 3]
    assert not mask[2, 4]


def test_mask_within_block_raw_tokens_see_each_other_causally():
    block_ids = torch.tensor([0, 0, 0])
    is_gist = torch.tensor([False, False, False])
    mask = build_segment_conditioning_mask(block_ids, is_gist)
    assert mask[1, 0]  # position 1 sees position 0 (same block, causal)
    assert not mask[0, 1]  # but not vice versa (future)


def test_mask_gist_tokens_of_two_blocks_back_remain_visible():
    # block 0: raw, gist; block 1: raw, gist; block 2: raw
    block_ids = torch.tensor([0, 0, 1, 1, 2])
    is_gist = torch.tensor([False, True, False, True, False])
    mask = build_segment_conditioning_mask(block_ids, is_gist)
    # Position 4 (block 2's raw token) sees both prior gist tokens (1 and 3)
    # but neither prior block's raw tokens (0 and 2).
    assert mask[4, 1]
    assert mask[4, 3]
    assert not mask[4, 0]
    assert not mask[4, 2]


# ---------------------------------------------------------------------------
# build_batched_segment_conditioning_mask
# ---------------------------------------------------------------------------


def test_batched_mask_matches_single_sequence_mask_per_row():
    block_ids_row = torch.tensor([0, 0, 0, 1, 1])
    is_gist_row = torch.tensor([False, False, True, False, False])
    single = build_segment_conditioning_mask(block_ids_row, is_gist_row)

    batched = build_batched_segment_conditioning_mask(block_ids_row.unsqueeze(0), is_gist_row.unsqueeze(0))
    assert batched.shape == (1, 1, 5, 5)
    assert torch.equal(batched[0, 0], single)


def test_batched_mask_handles_different_block_boundaries_per_row():
    # Row 0: blocks split at position 2. Row 1: blocks split at position 3.
    block_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
    is_gist = torch.tensor([[False, True, False, False, False], [False, False, True, False, False]])
    batched = build_batched_segment_conditioning_mask(block_ids, is_gist)

    expected_row0 = build_segment_conditioning_mask(block_ids[0], is_gist[0])
    expected_row1 = build_segment_conditioning_mask(block_ids[1], is_gist[1])
    assert torch.equal(batched[0, 0], expected_row0)
    assert torch.equal(batched[1, 0], expected_row1)
