import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from aiml_compiler import (  # noqa: E402
    AimlCompileError,
    compile_categories_from_files,
    normalize_word,
    tokenize_input,
    tokenize_pattern,
)


def _write(tmp_path, name: str, xml: str) -> str:
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


def _wrap(categories_xml: str) -> str:
    return f"<categories>{categories_xml}</categories>"


# ---------------------------------------------------------------------------
# normalization / tokenization
# ---------------------------------------------------------------------------


def test_normalize_word_uppercases_and_strips_punctuation():
    assert normalize_word("Hello!") == "HELLO"
    assert normalize_word("don't") == "DON'T"
    assert normalize_word("...") == ""


def test_tokenize_input_produces_plain_words():
    assert tokenize_input("Hello, world!") == ["HELLO", "WORLD"]


def test_tokenize_pattern_recognizes_wildcards():
    tokens = tokenize_pattern("MY NAME IS *")
    assert tokens == [
        {"kind": "literal", "word": "MY"},
        {"kind": "literal", "word": "NAME"},
        {"kind": "literal", "word": "IS"},
        {"kind": "wildcard", "wildcard": "*"},
    ]


def test_tokenize_pattern_recognizes_low_wildcard():
    tokens = tokenize_pattern("_ NAME IS HERE")
    assert tokens[0] == {"kind": "wildcard", "wildcard": "_"}


# ---------------------------------------------------------------------------
# catch-all rejection (the core safety rule)
# ---------------------------------------------------------------------------


def test_bare_high_wildcard_with_no_scope_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.aiml", _wrap("<category><pattern>*</pattern><template>anything</template></category>"))
    with pytest.raises(AimlCompileError, match="catch-all|no literal words"):
        compile_categories_from_files([path])


def test_bare_low_wildcard_with_no_scope_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.aiml", _wrap("<category><pattern>_</pattern><template>anything</template></category>"))
    with pytest.raises(AimlCompileError):
        compile_categories_from_files([path])


def test_all_wildcard_multiword_pattern_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.aiml", _wrap("<category><pattern>* _</pattern><template>anything</template></category>"))
    with pytest.raises(AimlCompileError):
        compile_categories_from_files([path])


def test_all_wildcard_pattern_with_that_scope_is_allowed(tmp_path):
    xml = _wrap(
        "<category><pattern>*</pattern><that>DID YOU DO IT</that>"
        "<template>Good to hear.</template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert len(compiled["categories"]) == 1


def test_all_wildcard_pattern_with_topic_scope_is_allowed(tmp_path):
    xml = _wrap(
        "<category><pattern>*</pattern><topic>FREEFORM</topic>"
        "<template>Tell me more.</template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert len(compiled["categories"]) == 1


def test_normal_literal_pattern_is_allowed(tmp_path):
    xml = _wrap("<category><pattern>HELLO</pattern><template>Hi there!</template></category>")
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert compiled["categories"][0]["literalCount"] == 1


# ---------------------------------------------------------------------------
# single vs multi-template classification
# ---------------------------------------------------------------------------


def test_single_template_category(tmp_path):
    xml = _wrap("<category><pattern>HELLO</pattern><template>Hi there!</template></category>")
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert len(compiled["categories"][0]["templates"]) == 1


def test_multi_template_category_via_random_li(tmp_path):
    xml = _wrap(
        "<category><pattern>HOW ARE YOU</pattern>"
        "<template><random><li>Doing great!</li><li>Just a pile of if-statements.</li></random></template>"
        "</category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert len(compiled["categories"][0]["templates"]) == 2


def test_incidental_single_li_random_is_not_treated_as_ambiguous(tmp_path):
    """Classification is by actual variant count, not by the presence of
    the <random> tag -- a <random> with exactly one <li> is single-template."""
    xml = _wrap(
        "<category><pattern>HELLO</pattern>"
        "<template><random><li>Hi there!</li></random></template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    assert len(compiled["categories"][0]["templates"]) == 1


def test_random_with_sibling_text_is_rejected(tmp_path):
    xml = _wrap(
        "<category><pattern>HELLO</pattern>"
        "<template>Well, <random><li>hi</li><li>hey</li></random></template></category>"
    )
    path = _write(tmp_path, "bad.aiml", xml)
    with pytest.raises(AimlCompileError):
        compile_categories_from_files([path])


def test_random_with_no_li_children_is_rejected(tmp_path):
    xml = _wrap("<category><pattern>HELLO</pattern><template><random></random></template></category>")
    path = _write(tmp_path, "bad.aiml", xml)
    with pytest.raises(AimlCompileError):
        compile_categories_from_files([path])


# ---------------------------------------------------------------------------
# template tag validation
# ---------------------------------------------------------------------------


def test_star_without_index_defaults_to_1(tmp_path):
    xml = _wrap("<category><pattern>MY NAME IS *</pattern><template>Hi <star/>!</template></category>")
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    star_node = [n for n in compiled["categories"][0]["templates"][0] if n["kind"] == "star"][0]
    assert star_node["index"] == 1


def test_star_with_explicit_index(tmp_path):
    xml = _wrap(
        "<category><pattern>* NAME IS *</pattern>"
        "<template>Hi <star index=\"2\"/>!</template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    star_node = [n for n in compiled["categories"][0]["templates"][0] if n["kind"] == "star"][0]
    assert star_node["index"] == 2


def test_unrecognized_template_tag_is_rejected(tmp_path):
    xml = _wrap("<category><pattern>HELLO</pattern><template><sr/></template></category>")
    path = _write(tmp_path, "bad.aiml", xml)
    with pytest.raises(AimlCompileError):
        compile_categories_from_files([path])


# ---------------------------------------------------------------------------
# that / topic / setTopic
# ---------------------------------------------------------------------------


def test_that_and_topic_and_set_topic_are_parsed(tmp_path):
    xml = _wrap(
        "<category><pattern>GOODBYE</pattern><that>SEE YOU LATER</that>"
        "<topic>FAREWELL</topic><setTopic>NONE</setTopic>"
        "<template>Bye!</template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    cat = compiled["categories"][0]
    assert cat["that"] == [
        {"kind": "literal", "word": "SEE"},
        {"kind": "literal", "word": "YOU"},
        {"kind": "literal", "word": "LATER"},
    ]
    assert cat["topic"] == "FAREWELL"
    assert cat["setTopic"] == "NONE"


def test_missing_that_topic_set_topic_are_null(tmp_path):
    xml = _wrap("<category><pattern>HELLO</pattern><template>Hi!</template></category>")
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    cat = compiled["categories"][0]
    assert cat["that"] is None
    assert cat["topic"] is None
    assert cat["setTopic"] is None


# ---------------------------------------------------------------------------
# ids and multi-file compilation
# ---------------------------------------------------------------------------


def test_ids_are_assigned_in_order_across_multiple_files(tmp_path):
    path_a = _write(tmp_path, "a.aiml", _wrap("<category><pattern>HELLO</pattern><template>Hi!</template></category>"))
    path_b = _write(tmp_path, "b.aiml", _wrap("<category><pattern>GOODBYE</pattern><template>Bye!</template></category>"))
    compiled = compile_categories_from_files([path_a, path_b])
    ids = [c["id"] for c in compiled["categories"]]
    assert ids == [0, 1]


def test_missing_pattern_raises():
    import xml.etree.ElementTree as ET

    from aiml_compiler import compile_category

    el = ET.fromstring("<category><template>Hi!</template></category>")
    with pytest.raises(AimlCompileError):
        compile_category(el, 0)


def test_wildcard_tier_computed_correctly(tmp_path):
    xml = _wrap(
        "<category><pattern>MY NAME IS *</pattern><template>Hi <star/>!</template></category>"
        "<category><pattern>MY NAME IS _</pattern><that>X</that><template>Hi!</template></category>"
        "<category><pattern>MY NAME IS BENNY</pattern><template>Hi!</template></category>"
    )
    path = _write(tmp_path, "ok.aiml", xml)
    compiled = compile_categories_from_files([path])
    tiers = [c["wildcardTier"] for c in compiled["categories"]]
    assert tiers == [0, 1, 2]
