"""Mines candidate AIML categories from conversational training data
(spec Section 1.4's bootstrap pipeline, steps 1-3): extract short,
single-turn-shaped user utterances, cluster near-duplicates, rank by
frequency, and write the top-N clusters out for human review.

This is a one-time data-processing utility -- it does not ship with Benny
or run at inference time. It deliberately stops short of generating AIML
XML: turning a cluster into a real `<category>` requires generalizing a
wildcard pattern and writing a PARAPHRASED template (spec steps 4-5), which
needs human judgment (or an LLM call this script doesn't make) -- the
output here is the *input* to that review step, not a finished category
set. See `aiml/categories/*.aiml` for the reviewed, hand-curated result.

Usage:
    # Real corpus (needs network access to stream UltraChat/OASST2):
    python bootstrap.py --hf-dataset HuggingFaceH4/ultrachat_200k --hf-split train_sft \
        --max-conversations 200000 --top-n 300 --out candidates.json

    # Local corpus (same {"messages": [...]} jsonl shape as model/sft.py expects):
    python bootstrap.py --local-chat-corpus /path/to/*.jsonl --top-n 100 --out candidates.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "data"))
from pipeline import ChatExample, iter_hf_streaming_conversations, iter_local_jsonl_conversations  # noqa: E402

# Common openers for turns that need unique factual/entity context -- these
# are exactly the turns that should keep falling through to the GPT
# fallback rather than getting a canned deterministic reply (spec 1.4).
_FACTUAL_OPENER_RE = re.compile(
    r"^(who (is|was|are|were)|when (did|was|is|were)|what (is|was) the capital|"
    r"what year|how many|where (is|was|are)|which (country|city|president|company))\b",
    re.IGNORECASE,
)
_PROPER_NOUN_RUN_RE = re.compile(r"(?:\b[A-Z][a-z]+\b\s+){2,}\b[A-Z][a-z]+\b")


def is_entity_heavy(text: str) -> bool:
    """Heuristic exclusion for turns likely to require unique factual/named-
    entity context -- not a precise NER system, just enough to keep obvious
    factual-lookup turns out of the deterministic candidate pool."""
    if _FACTUAL_OPENER_RE.search(text.strip()):
        return True
    # A run of 3+ consecutive capitalized words (after the sentence-initial
    # word, which is capitalized regardless of content) usually indicates a
    # multi-word proper noun (a person/place/organization name).
    rest = text.split(" ", 1)[1] if " " in text else ""
    if _PROPER_NOUN_RUN_RE.search(rest):
        return True
    return False


def passes_length_filter(text: str, min_words: int = 2, max_words: int = 14) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


def extract_candidate_utterances(conversations, min_words: int = 2, max_words: int = 14):
    for conv in conversations:
        for turn in conv.turns:
            if turn.get("role") != "user":
                continue
            content = turn.get("content", "").strip()
            if not content or not passes_length_filter(content, min_words, max_words):
                continue
            if is_entity_heavy(content):
                continue
            yield content


def _cluster_key(text: str) -> frozenset:
    normalized = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return frozenset(normalized.split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Cluster:
    key: frozenset
    examples: list = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.examples)

    @property
    def representative(self) -> str:
        # Shortest example reads best as a generalization starting point.
        return min(self.examples, key=len)


def cluster_utterances(utterances, similarity_threshold: float = 0.6, max_examples_per_cluster: int = 8):
    """Greedy near-duplicate clustering via normalized token-set Jaccard
    similarity -- deliberately lightweight (spec: "a lightweight similarity
    method... is enough"), no embedding model required."""
    clusters: list[Cluster] = []
    for text in utterances:
        key = _cluster_key(text)
        if not key:
            continue
        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = _jaccard(key, cluster.key)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is not None and best_score >= similarity_threshold:
            if len(best_cluster.examples) < max_examples_per_cluster:
                best_cluster.examples.append(text)
        else:
            clusters.append(Cluster(key=key, examples=[text]))
    return clusters


def make_conversation_source(args):
    if args.local_chat_corpus:
        paths = sorted(glob.glob(args.local_chat_corpus))
        if not paths:
            raise FileNotFoundError(f"no files matched --local-chat-corpus {args.local_chat_corpus}")
        return itertools.chain.from_iterable(iter_local_jsonl_conversations(p) for p in paths)
    return iter_hf_streaming_conversations(args.hf_dataset, args.hf_split, args.turns_field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-chat-corpus", default=None, help="glob of local {'messages': [...]} jsonl files")
    parser.add_argument("--hf-dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--hf-split", default="train_sft")
    parser.add_argument("--turns-field", default="messages")
    parser.add_argument("--max-conversations", type=int, default=200_000)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=14)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--min-cluster-size", type=int, default=2, help="drop clusters smaller than this")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--out", required=True, help="output path for ranked candidate clusters (JSON)")
    args = parser.parse_args()

    if not args.local_chat_corpus and not args.hf_dataset:
        parser.error("must pass either --local-chat-corpus or --hf-dataset")

    conversations = make_conversation_source(args)
    conversations = itertools.islice(conversations, args.max_conversations)

    utterances = extract_candidate_utterances(conversations, args.min_words, args.max_words)
    clusters = cluster_utterances(utterances, args.similarity_threshold)

    clusters = [c for c in clusters if c.count >= args.min_cluster_size]
    clusters.sort(key=lambda c: c.count, reverse=True)
    clusters = clusters[: args.top_n]

    payload = {
        "note": (
            "Candidate clusters for human review (spec Section 1.4 steps 1-3). "
            "Each cluster needs a generalized wildcard pattern and a PARAPHRASED "
            "template written by a human reviewer before it becomes a real AIML "
            "category in aiml/categories/ -- do not copy `representative` verbatim "
            "into a <template>."
        ),
        "clusters": [
            {"rank": i + 1, "count": c.count, "representative": c.representative, "examples": c.examples}
            for i, c in enumerate(clusters)
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[bootstrap] wrote {len(clusters)} candidate clusters -> {args.out}")


if __name__ == "__main__":
    main()
