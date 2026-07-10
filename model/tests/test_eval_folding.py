import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))

from architecture import BNAIConfig, BNAILanguageModel  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from eval import needle_in_haystack_eval, perplexity_at_extended_context  # noqa: E402
from folding import add_gist_tokens, gist_token_ids  # noqa: E402

SAMPLE_DOCS = [f"the quick brown fox jumps over the lazy dog number {i} while birds sing" for i in range(30)]


def _tiny_model_and_tokenizer():
    tok = BPETokenizer.train(SAMPLE_DOCS, vocab_size=300)
    cfg = BNAIConfig(vocab_size=350, d_model=16, n_layers=2, n_heads=4, ffn_hidden=32, context_len=24)
    model = BNAILanguageModel(cfg)
    model.eval()
    tok_with_gist = add_gist_tokens(tok, gist_token_count=3, model_vocab_size=cfg.vocab_size)
    gids = gist_token_ids(tok_with_gist, 3)
    return model, tok_with_gist, gids


def test_perplexity_at_extended_context_returns_both_arms():
    model, tokenizer, gids = _tiny_model_and_tokenizer()
    result = perplexity_at_extended_context(
        model, tokenizer, SAMPLE_DOCS, gids, fold_block_size=6, window_len=30, device=torch.device("cpu"),
        micro_batch_size=2, max_batches=3,
    )
    assert "folded_perplexity" in result
    assert "truncated_baseline_perplexity" in result
    assert result["folded_perplexity"] > 0
    assert result["truncated_baseline_perplexity"] > 0
    assert result["window_len"] == 30
    assert result["context_len"] == model.cfg.context_len


def test_needle_in_haystack_eval_returns_valid_structure():
    model, tokenizer, gids = _tiny_model_and_tokenizer()
    result = needle_in_haystack_eval(
        model, tokenizer, gids, fold_block_size=6, window_len=30, device=torch.device("cpu"),
        n_trials=3, filler_blocks=2, seed=0,
    )
    assert result["n_trials"] == 3
    assert 0.0 <= result["folded_accuracy"] <= 1.0
    assert 0.0 <= result["truncated_baseline_accuracy"] <= 1.0


def test_needle_in_haystack_eval_is_deterministic_given_seed():
    model, tokenizer, gids = _tiny_model_and_tokenizer()
    result_a = needle_in_haystack_eval(
        model, tokenizer, gids, fold_block_size=6, window_len=30, device=torch.device("cpu"),
        n_trials=3, filler_blocks=2, seed=42,
    )
    result_b = needle_in_haystack_eval(
        model, tokenizer, gids, fold_block_size=6, window_len=30, device=torch.device("cpu"),
        n_trials=3, filler_blocks=2, seed=42,
    )
    assert result_a == result_b
