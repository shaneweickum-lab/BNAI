/**
 * Top-level category selection: given normalized input + dialogue context,
 * finds every eligible compiled category, ranks the survivors, and reports
 * either a single deterministic winner, an "ambiguous" winner (a <random>
 * category whose variant we deliberately do not pick here), or "no-match".
 *
 * This -- plus patternMatcher.ts -- is the entire deterministic layer that
 * lib/dialogue/dialogueManager.ts tries first, before ever waking up the
 * GPT Web Worker. The point is cost/latency/predictability: deterministic
 * where determinism is sufficient, neural only where necessary.
 */

import { matchTokens, normalize } from "./patternMatcher";
import { render } from "./render";
import type { CompiledCategory, MatchResult, MatcherContext, PatternToken } from "./types";

function normalizeTopicLabel(topic: string | null): string {
  return normalize(topic ?? "").join(" ");
}

function contextTier(category: CompiledCategory): 0 | 1 | 2 {
  const hasThat = category.that !== null;
  const hasTopic = category.topic !== null;
  if (hasThat && hasTopic) return 2;
  if (hasThat || hasTopic) return 1;
  return 0;
}

function firstTokenIsLiteral(pattern: PatternToken[]): boolean {
  return pattern.length > 0 && pattern[0].kind === "literal";
}

interface Candidate {
  category: CompiledCategory;
  captures: string[];
}

/**
 * Ranking comparator implementing the exact priority order:
 *   1. contextTier, descending (context-scoping beats raw specificity)
 *   2. wildcardTier, descending (literal > `_` > `*`)
 *   3. literalCount, descending
 *   4. lowCount, descending
 *   5. at contextTier 1: that-only beats topic-only; then literal-first
 *      pattern beats wildcard-first pattern
 *   6. lower id wins (source definition order)
 * Each row only breaks ties left by the row above.
 */
function compareCandidates(a: Candidate, b: Candidate): number {
  const ca = a.category;
  const cb = b.category;

  const tierA = contextTier(ca);
  const tierB = contextTier(cb);
  if (tierA !== tierB) return tierB - tierA;

  if (ca.wildcardTier !== cb.wildcardTier) return cb.wildcardTier - ca.wildcardTier;

  if (ca.literalCount !== cb.literalCount) return cb.literalCount - ca.literalCount;

  if (ca.lowCount !== cb.lowCount) return cb.lowCount - ca.lowCount;

  if (tierA === 1) {
    const aThatOnly = ca.that !== null && ca.topic === null;
    const bThatOnly = cb.that !== null && cb.topic === null;
    if (aThatOnly !== bThatOnly) return aThatOnly ? -1 : 1;
  }

  const aLiteralFirst = firstTokenIsLiteral(ca.pattern);
  const bLiteralFirst = firstTokenIsLiteral(cb.pattern);
  if (aLiteralFirst !== bLiteralFirst) return aLiteralFirst ? -1 : 1;

  return ca.id - cb.id;
}

export function match(input: string, context: MatcherContext, categories: CompiledCategory[]): MatchResult {
  const inputWords = normalize(input);
  const thatWords = normalize(context.lastBotUtterance);
  const normalizedContextTopic = normalizeTopicLabel(context.topic);

  const candidates: Candidate[] = [];

  for (const category of categories) {
    const patternMatch = matchTokens(category.pattern, inputWords);
    if (patternMatch === null) continue;

    if (category.that !== null) {
      const thatMatch = matchTokens(category.that, thatWords);
      if (thatMatch === null) continue;
    }

    if (category.topic !== null) {
      if (normalizedContextTopic !== category.topic) continue;
    }

    candidates.push({ category, captures: patternMatch.captures });
  }

  if (candidates.length === 0) return { kind: "no-match" };

  candidates.sort(compareCandidates);
  const winner = candidates[0];

  if (winner.category.templates.length > 1) {
    return {
      kind: "ambiguous",
      category: winner.category,
      captures: winner.captures,
      candidateCount: winner.category.templates.length,
    };
  }

  const response = render(winner.category.templates[0], winner.captures);
  return { kind: "single", category: winner.category, captures: winner.captures, response };
}
