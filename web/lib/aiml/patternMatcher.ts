/**
 * Deterministic pattern matching primitives for the AIML-style layer.
 *
 * This is the cheap, predictable half of Benny's hybrid routing story: a
 * plain string/word matcher that runs synchronously on the main thread with
 * no model weights involved. When it can find one unambiguous category, we
 * never have to pay for a GPT Web Worker round trip -- "deterministic where
 * determinism is sufficient, neural only where necessary."
 *
 * Normalization here must exactly mirror the Python compiler's
 * `normalize_word`/`tokenize_input` (aiml/tools/aiml_compiler.py), since that
 * compiler is what produced the literal words baked into
 * lib/aiml/generated/categories.json -- if this drifted from the Python
 * rules, compiled patterns would never line up with live user input.
 */

import type { PatternToken } from "./types";

// Mirrors Python's `_PUNCTUATION_RE = re.compile(r"[^A-Z0-9']")`: only
// uppercase letters, digits, and apostrophes survive.
const PUNCTUATION_RE = /[^A-Z0-9']/g;

function normalizeWord(raw: string): string {
  return raw.toUpperCase().replace(PUNCTUATION_RE, "");
}

/**
 * Uppercase, strip everything except A-Z/0-9/', collapse whitespace, split
 * into words. Used both for live text (user input, last-bot-utterance) and,
 * conceptually, for the same normalization the Python compiler already
 * applied to literal pattern words -- so a normalized live word can be
 * compared directly against a compiled `literal` token's `word`.
 */
export function normalize(text: string): string[] {
  const words: string[] = [];
  for (const raw of text.split(/\s+/)) {
    const w = normalizeWord(raw);
    if (w) words.push(w);
  }
  return words;
}

export interface MatchTokensResult {
  captures: string[];
}

/**
 * Full-string match of a compiled pattern/that token array against a list
 * of already-normalized words.
 *
 * - `literal` must equal the word at that position exactly.
 * - `_` consumes exactly one word.
 * - `*` is greedy: it tries the longest possible remaining span first,
 *   backtracking to shorter spans only if the rest of the pattern then
 *   fails against the rest of the words. Like `_`, `*` must consume at
 *   least one word (standard AIML wildcard semantics) -- it does not match
 *   an empty span.
 *
 * Captures (both `_` and `*`) are collected left-to-right in pattern order,
 * `*` captures being the joined-by-space span of words it consumed. This is
 * exactly what template `<star index="N">` (1-based) indexes into.
 */
export function matchTokens(pattern: PatternToken[], words: string[]): MatchTokensResult | null {
  // Recursive descent with backtracking. `tryMatch` returns the list of
  // captures contributed by pattern[pIdx..] when matched against
  // words[wIdx..], or null if no match is possible from this point.
  function tryMatch(pIdx: number, wIdx: number): string[] | null {
    if (pIdx === pattern.length) {
      return wIdx === words.length ? [] : null;
    }

    const token = pattern[pIdx];

    if (token.kind === "literal") {
      if (wIdx >= words.length || words[wIdx] !== token.word) return null;
      return tryMatch(pIdx + 1, wIdx + 1);
    }

    if (token.wildcard === "_") {
      if (wIdx >= words.length) return null;
      const rest = tryMatch(pIdx + 1, wIdx + 1);
      if (rest === null) return null;
      return [words[wIdx], ...rest];
    }

    // "*": greedy, longest-span-first, with backtracking.
    for (let end = words.length; end >= wIdx + 1; end--) {
      const rest = tryMatch(pIdx + 1, end);
      if (rest !== null) {
        return [words.slice(wIdx, end).join(" "), ...rest];
      }
    }
    return null;
  }

  const captures = tryMatch(0, 0);
  return captures === null ? null : { captures };
}
