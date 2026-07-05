"""Train the BNAI tokenizer (32k byte-level BPE + chat special tokens).

Per spec Section 4: train fresh on a representative sample of the actual
pretraining corpus, not an inherited vocabulary, so the embedding table isn't
oversized for a 75M-parameter model.

Usage:
    # From local text files (one doc per line, or one doc per file):
    python train_tokenizer.py --input "corpus/*.txt" --vocab-size 32000 \
        --out tokenizer/bnai_tokenizer.json

    # From a streaming Hugging Face dataset (e.g. FineWeb-Edu), sampling
    # up to --max-docs documents so training stays memory-bounded:
    python train_tokenizer.py --hf-dataset HuggingFaceFW/fineweb-edu \
        --hf-split train --max-docs 200000 --out tokenizer/bnai_tokenizer.json
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe import BPETokenizer, SPECIAL_TOKENS  # noqa: E402


def iter_local_texts(pattern: str):
    """Reads .jsonl files as {"text": ...} documents (matching
    data/pipeline.py's iter_local_jsonl_documents), and any other file as
    one document per non-empty line."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    from pipeline import iter_local_jsonl_documents  # noqa: E402

    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no files matched --input pattern: {pattern}")
    for path in paths:
        if path.endswith(".jsonl"):
            yield from iter_local_jsonl_documents(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def iter_hf_texts(dataset_name: str, split: str, text_field: str, max_docs: int):
    from datasets import load_dataset  # imported lazily: optional heavy dependency

    ds = load_dataset(dataset_name, split=split, streaming=True)
    for i, example in enumerate(ds):
        if i >= max_docs:
            break
        text = example.get(text_field)
        if text:
            yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="glob pattern for local text files")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset name, e.g. HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--max-docs", type=int, default=200_000, help="cap on docs sampled for tokenizer training")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out", required=True, help="output path for tokenizer JSON")
    args = parser.parse_args()

    if not args.input and not args.hf_dataset:
        parser.error("must pass either --input or --hf-dataset")

    if args.input:
        texts = iter_local_texts(args.input)
    else:
        texts = iter_hf_texts(args.hf_dataset, args.hf_split, args.hf_text_field, args.max_docs)

    print(f"Training BPE tokenizer, target vocab_size={args.vocab_size} ...")
    tokenizer = BPETokenizer.train(texts, vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tokenizer.save(args.out)
    print(f"Saved tokenizer with {tokenizer.vocab_size} tokens ({len(tokenizer.merges)} merges) -> {args.out}")


if __name__ == "__main__":
    main()
