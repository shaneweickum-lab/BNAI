"""Compiles `aiml/categories/*.aiml` into the compact JSON the browser's
matching engine (`web/lib/aiml/*.ts`) loads statically at build time.

This is the single source of truth for compilation -- the TypeScript
runtime never parses XML itself, it only implements the *matching*
algorithm (normalize/tokenize input, rank survivors, render templates)
against the JSON this module produces. See `web/lib/aiml/types.ts` for the
TypeScript mirror of the shapes emitted here; keep them in sync by hand
the same way `runtime/src/tokenizer.rs` is kept in sync with
`model/tokenizer/bpe.py`.

AIML XML schema this compiler accepts, per category:
    <category>
      <pattern>MY NAME IS *</pattern>
      <that>...</that>            optional
      <topic>...</topic>          optional
      <setTopic>...</setTopic>    optional (not standard AIML -- see below)
      <template>...</template>    required; either plain content (text +
                                   <star/>/<star index="N"/>) or a single
                                   top-level <random><li>...</li>...</random>
                                   block with no sibling content
    </category>

`<setTopic>` is a deliberate, minimal addition beyond standard AIML: the
project spec requires `<topic>`-scoped categories but never specifies how a
category *sets* the topic going forward, and generic AIML `<set>`/`<get>`
predicate variables are explicitly out of scope. Rather than add that whole
subsystem for one string, a category can declare `<setTopic>VALUE</setTopic>`
as a sibling of `<pattern>`/`<template>` -- the caller (the dialogue
manager, not the matcher) applies it after a single-template win.

Validation (fails the whole compile, not just one category):
  - No category may have a pattern with zero literal words AND no
    `<that>`/`<topic>` scope -- this is the catch-all safety rule (a bare
    `*`/`_`, or any all-wildcard pattern like `* *`, would match every
    input and defeat the entire point of a fallback system).
  - No `<template>` may contain anything beyond plain text, `<star/>`,
    `<star index="N"/>`, or a single top-level `<random><li>...</li></random>`
    block -- hard boundary on the "no srai/pronoun/sets" scope limit.
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


class AimlCompileError(Exception):
    pass


# ---------------------------------------------------------------------------
# Normalization / tokenization -- must match web/lib/aiml/patternMatcher.ts's
# normalize()/tokenize() exactly, since one runs at compile time (here) and
# the other at runtime (in the browser) against live user input.
# ---------------------------------------------------------------------------

_PUNCTUATION_RE = re.compile(r"[^A-Z0-9']")


def normalize_word(raw: str) -> str:
    """Uppercase and strip punctuation from a single word. Returns "" if
    nothing is left (e.g. the token was pure punctuation), which callers
    must filter out."""
    return _PUNCTUATION_RE.sub("", raw.upper())


def tokenize_input(text: str) -> list:
    """For live text (user input, last-bot-utterance) -- never contains
    wildcard tokens, just normalized words."""
    words = []
    for raw in text.split():
        w = normalize_word(raw)
        if w:
            words.append(w)
    return words


def tokenize_pattern(text: str) -> list:
    """For <pattern>/<that> source text -- recognizes bare `*`/`_` tokens
    as wildcards before normalizing the remaining literal words."""
    tokens = []
    for raw in text.split():
        if raw == "*":
            tokens.append({"kind": "wildcard", "wildcard": "*"})
        elif raw == "_":
            tokens.append({"kind": "wildcard", "wildcard": "_"})
        else:
            w = normalize_word(raw)
            if w:
                tokens.append({"kind": "literal", "word": w})
    return tokens


# ---------------------------------------------------------------------------
# Template parsing
# ---------------------------------------------------------------------------

_ALLOWED_TEMPLATE_TAGS = {"star", "random", "li"}


def _parse_star_index(el: ET.Element, source: str) -> int:
    idx_attr = el.get("index")
    if idx_attr is None:
        return 1
    try:
        idx = int(idx_attr)
    except ValueError:
        raise AimlCompileError(f"<star index=\"{idx_attr}\"/> in category \"{source}\": index must be an integer")
    if idx < 1:
        raise AimlCompileError(f"<star index=\"{idx_attr}\"/> in category \"{source}\": index must be >= 1")
    return idx


def _parse_template_nodes(container: ET.Element, source: str) -> list:
    """Parses the flat text+<star/> node sequence of a single template
    variant (either the top-level <template> when there's no <random>, or
    a single <li>'s content)."""
    nodes = []
    if container.text:
        nodes.append({"kind": "text", "text": container.text})
    for child in container:
        tag = child.tag
        if tag not in _ALLOWED_TEMPLATE_TAGS:
            raise AimlCompileError(
                f"category \"{source}\": <template> contains unsupported tag <{tag}> -- "
                "only plain text, <star/>, and a single top-level <random><li> block are allowed"
            )
        if tag == "star":
            nodes.append({"kind": "star", "index": _parse_star_index(child, source)})
        else:
            raise AimlCompileError(
                f"category \"{source}\": <{tag}> may not appear nested inside template content "
                "(a <random> block must be the template's only top-level content)"
            )
        if child.tail:
            nodes.append({"kind": "text", "text": child.tail})
    return nodes


def parse_template(template_el: ET.Element, source: str) -> list:
    """Returns `templates`: a list of template variants (each a node list).
    Length 1 = single-template category; length > 1 = multi-template
    ("ambiguous") -- classified by actual <li> count, not by whether a
    <random> tag is present at all."""
    random_children = [c for c in template_el if c.tag == "random"]
    other_children = [c for c in template_el if c.tag != "random"]

    if random_children:
        if len(random_children) > 1 or other_children or (template_el.text and template_el.text.strip()):
            raise AimlCompileError(
                f"category \"{source}\": <random> must be the template's only top-level content "
                "(no sibling text or tags)"
            )
        random_el = random_children[0]
        li_children = [c for c in random_el if c.tag == "li"]
        non_li = [c for c in random_el if c.tag != "li"]
        if non_li:
            raise AimlCompileError(f"category \"{source}\": <random> may only contain <li> children")
        if not li_children:
            raise AimlCompileError(f"category \"{source}\": <random> must contain at least one <li>")
        return [_parse_template_nodes(li, source) for li in li_children]

    return [_parse_template_nodes(template_el, source)]


# ---------------------------------------------------------------------------
# Category compilation
# ---------------------------------------------------------------------------

@dataclass
class CompiledCategory:
    id: int
    pattern_source: str
    pattern: list
    that: list | None
    topic: str | None
    set_topic: str | None
    templates: list
    literal_count: int
    low_count: int
    high_count: int
    wildcard_tier: int  # 0 = contains "*", 1 = contains "_" but no "*", 2 = fully literal

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "patternSource": self.pattern_source,
            "pattern": self.pattern,
            "that": self.that,
            "topic": self.topic,
            "setTopic": self.set_topic,
            "templates": self.templates,
            "literalCount": self.literal_count,
            "lowCount": self.low_count,
            "highCount": self.high_count,
            "wildcardTier": self.wildcard_tier,
        }


def _pattern_stats(tokens: list) -> tuple:
    literal_count = sum(1 for t in tokens if t["kind"] == "literal")
    low_count = sum(1 for t in tokens if t["kind"] == "wildcard" and t["wildcard"] == "_")
    high_count = sum(1 for t in tokens if t["kind"] == "wildcard" and t["wildcard"] == "*")
    if high_count > 0:
        wildcard_tier = 0
    elif low_count > 0:
        wildcard_tier = 1
    else:
        wildcard_tier = 2
    return literal_count, low_count, high_count, wildcard_tier


def compile_category(el: ET.Element, category_id: int) -> CompiledCategory:
    pattern_el = el.find("pattern")
    if pattern_el is None or not (pattern_el.text or "").strip():
        raise AimlCompileError(f"category #{category_id}: missing or empty <pattern>")
    pattern_source = pattern_el.text.strip()
    pattern = tokenize_pattern(pattern_source)

    that_el = el.find("that")
    that = tokenize_pattern(that_el.text.strip()) if that_el is not None and (that_el.text or "").strip() else None

    # topic may be multi-word in principle; normalize each word and rejoin.
    topic_el = el.find("topic")
    topic = None
    if topic_el is not None and (topic_el.text or "").strip():
        topic = " ".join(tokenize_input(topic_el.text.strip()))

    set_topic_el = el.find("setTopic")
    set_topic = None
    if set_topic_el is not None and (set_topic_el.text or "").strip():
        set_topic = " ".join(tokenize_input(set_topic_el.text.strip()))

    template_el = el.find("template")
    if template_el is None:
        raise AimlCompileError(f"category \"{pattern_source}\": missing <template>")
    templates = parse_template(template_el, pattern_source)

    literal_count, low_count, high_count, wildcard_tier = _pattern_stats(pattern)

    if literal_count == 0 and that is None and topic is None:
        raise AimlCompileError(
            f"category \"{pattern_source}\": pattern has no literal words and no <that>/<topic> scope -- "
            "this is a catch-all that would match every input and is rejected. Add a literal word or a "
            "<that>/<topic> scope."
        )

    return CompiledCategory(
        id=category_id,
        pattern_source=pattern_source,
        pattern=pattern,
        that=that,
        topic=topic,
        set_topic=set_topic,
        templates=templates,
        literal_count=literal_count,
        low_count=low_count,
        high_count=high_count,
        wildcard_tier=wildcard_tier,
    )


def compile_categories_from_files(paths: list) -> dict:
    categories = []
    next_id = 0
    for path in paths:
        tree = ET.parse(path)
        root = tree.getroot()
        category_els = root.findall(".//category") if root.tag != "category" else [root]
        for el in category_els:
            categories.append(compile_category(el, next_id))
            next_id += 1

    if not categories:
        raise AimlCompileError("no categories found in any input file")

    return {"categories": [c.to_json() for c in categories]}


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="aiml/categories/*.aiml", help="glob of source AIML XML files")
    parser.add_argument("--out", default="web/lib/aiml/generated/categories.json", help="output compiled JSON path")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.categories))
    if not paths:
        raise FileNotFoundError(f"no files matched --categories {args.categories}")

    compiled = compile_categories_from_files(paths)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, ensure_ascii=False)

    n_single = sum(1 for c in compiled["categories"] if len(c["templates"]) == 1)
    n_multi = len(compiled["categories"]) - n_single
    print(
        f"[aiml] compiled {len(compiled['categories'])} categories "
        f"({n_single} single-template, {n_multi} multi-template) from {len(paths)} file(s) -> {args.out}"
    )


if __name__ == "__main__":
    main()
