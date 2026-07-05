"""From-scratch byte-level BPE tokenizer (GPT-2 style), trained fresh on the
BNAI pretraining corpus per spec Section 4 (a 32k vocabulary sized for a 75M
model, rather than reusing an oversized inherited tokenizer).

This is the single source of truth for tokenization: `train_tokenizer.py`
trains it, `model/data/pipeline.py` and `sft.py` use it to tokenize training
data, `export.py` bundles its vocab+merges JSON into the web asset, and
`runtime/src/tokenizer.rs` re-implements this exact algorithm in Rust so the
WASM runtime tokenizes identically without any external tokenizer dependency.

Byte-level (not word-level) so any UTF-8 input has full coverage with no
<unk> fallback needed for the base vocabulary.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

# GPT-2's pretokenization regex needs the `regex` package for \p{L}/\p{N}
# support that stdlib `re` lacks. Fall back to a plain-`re`-compatible
# approximation if `regex` isn't installed, so this module has no hard
# dependency beyond the stdlib.
try:
    import regex

    _GPT2_SPLIT_PATTERN = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def _pretokenize(text: str):
        return _GPT2_SPLIT_PATTERN.findall(text)

except ImportError:
    _SIMPLE_SPLIT_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\W]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+"""
    )

    def _pretokenize(text: str):
        return _SIMPLE_SPLIT_PATTERN.findall(text)


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, TURN_END_ID = range(8)


def _bytes_to_unicode() -> dict:
    """Reversible byte -> printable-unicode-char mapping (GPT-2 trick).

    BPE merges operate over unicode symbols; mapping every byte value to a
    printable character means BPE never has to special-case whitespace or
    control bytes, and any input byte sequence round-trips exactly.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _word_to_symbols(word: str) -> tuple:
    return tuple(_BYTE_ENCODER[b] for b in word.encode("utf-8"))


@dataclass
class BPETokenizer:
    vocab: dict = field(default_factory=dict)  # token string -> id
    merges: list = field(default_factory=list)  # ordered [ (a, b), ... ] pair merges

    def __post_init__(self):
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self._id_to_token = {i: t for t, i in self.vocab.items()}

    # ---- training ----

    @classmethod
    def train(cls, texts, vocab_size: int = 32000, special_tokens=None, min_pair_freq: int = 2):
        special_tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS
        word_freqs = Counter()
        for text in texts:
            for word in _pretokenize(text):
                if word:
                    word_freqs[_word_to_symbols(word)] += 1

        # Base vocab: all special tokens, then all 256 byte symbols.
        vocab = {}
        for tok in special_tokens:
            vocab[tok] = len(vocab)
        for b in range(256):
            vocab.setdefault(_BYTE_ENCODER[b], len(vocab))

        merges = []
        splits = {word: list(word) for word in word_freqs}

        target_merges = vocab_size - len(vocab)
        for _ in range(max(0, target_merges)):
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_freq:
                break

            merged_token = best_pair[0] + best_pair[1]
            vocab[merged_token] = len(vocab)
            merges.append(best_pair)

            for word in list(splits.keys()):
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and symbols[i] == best_pair[0]
                        and symbols[i + 1] == best_pair[1]
                    ):
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

            if len(vocab) >= vocab_size:
                break

        return cls(vocab=vocab, merges=merges)

    # ---- encode / decode ----

    def _bpe_word(self, symbols: list) -> list:
        symbols = list(symbols)
        while len(symbols) > 1:
            ranked = [
                (self._merge_rank.get((symbols[i], symbols[i + 1]), None), i)
                for i in range(len(symbols) - 1)
            ]
            ranked = [(r, i) for r, i in ranked if r is not None]
            if not ranked:
                break
            _, merge_idx = min(ranked)
            symbols[merge_idx : merge_idx + 2] = [symbols[merge_idx] + symbols[merge_idx + 1]]
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list:
        ids = []
        if add_bos:
            ids.append(BOS_ID)
        for word in _pretokenize(text):
            if not word:
                continue
            symbols = self._bpe_word(list(_word_to_symbols(word)))
            for sym in symbols:
                ids.append(self.vocab.get(sym, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list) -> str:
        chars = []
        for i in ids:
            tok = self._id_to_token.get(i)
            if tok is None or tok in SPECIAL_TOKENS:
                continue
            chars.append(tok)
        text = "".join(chars)
        byte_arr = bytes(_BYTE_DECODER[c] for c in text)
        return byte_arr.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- persistence ----

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "merges": [[a, b] for a, b in self.merges],
                    "special_tokens": SPECIAL_TOKENS,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(pair) for pair in data["merges"]]
        return cls(vocab=data["vocab"], merges=merges)
