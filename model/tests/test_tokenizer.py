import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"))

from bpe import BPETokenizer, SPECIAL_TOKENS, BOS_ID, EOS_ID, UNK_ID  # noqa: E402


SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world, this is a test of the BNAI tokenizer",
    "ternary weight language models use absmean scaling",
    "the quick brown fox jumps over the lazy dog again and again",
] * 20


def test_special_token_ids_are_reserved_first():
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.vocab[name] == i


def test_vocab_size_does_not_exceed_target():
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    assert tok.vocab_size <= 300


def test_encode_decode_round_trip():
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    text = "the quick brown fox jumps over the lazy dog"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_encode_decode_round_trip_unseen_unicode():
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    text = "emoji test \U0001F600 and accents café"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(isinstance(i, int) for i in ids)


def test_bos_eos_added_correctly():
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_save_load_round_trip(tmp_path):
    tok = BPETokenizer.train(SAMPLE_TEXTS, vocab_size=300)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "the quick brown fox jumps over the lazy dog"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
