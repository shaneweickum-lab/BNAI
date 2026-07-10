import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from bootstrap import (  # noqa: E402
    Cluster,
    _cluster_key,
    _jaccard,
    cluster_utterances,
    is_entity_heavy,
    passes_length_filter,
)


def test_factual_openers_are_entity_heavy():
    assert is_entity_heavy("who is the president of France")
    assert is_entity_heavy("when did World War Two end")
    assert is_entity_heavy("what is the capital of Japan")
    assert is_entity_heavy("how many moons does Jupiter have")


def test_proper_noun_run_is_entity_heavy():
    assert is_entity_heavy("tell me about Barack Hussein Obama please")


def test_plain_smalltalk_is_not_entity_heavy():
    assert not is_entity_heavy("how are you doing today")
    assert not is_entity_heavy("hello there")
    assert not is_entity_heavy("i feel sad today")


def test_length_filter():
    assert passes_length_filter("hi there", min_words=2, max_words=14)
    assert not passes_length_filter("hi", min_words=2, max_words=14)
    assert not passes_length_filter("a " * 20, min_words=2, max_words=14)


def test_jaccard_identical_sets_is_one():
    a = frozenset({"hello", "there"})
    assert _jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_cluster_utterances_groups_near_duplicates():
    utterances = ["how are you", "how are you today", "how are you doing", "goodbye now"]
    clusters = cluster_utterances(utterances, similarity_threshold=0.5)
    sizes = sorted(c.count for c in clusters)
    assert sizes[-1] >= 2  # the "how are you" variants should cluster together


def test_cluster_representative_is_shortest_example():
    cluster = Cluster(key=_cluster_key("x"), examples=["how are you doing today", "how are you"])
    assert cluster.representative == "how are you"
