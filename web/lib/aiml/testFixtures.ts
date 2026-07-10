/**
 * Small helpers shared by the aiml/*.test.ts files for hand-building
 * CompiledCategory fixtures without repeating the verbose JSON shape in
 * every test. Not part of the runtime matcher -- test-only support code.
 */

import { normalize } from "./patternMatcher";
import type { CompiledCategory, PatternToken, TemplateAst } from "./types";

export function tokenizePatternSource(source: string): PatternToken[] {
  return source
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .map((w): PatternToken => (w === "*" || w === "_" ? { kind: "wildcard", wildcard: w } : { kind: "literal", word: w }));
}

function patternStats(pattern: PatternToken[]): { literalCount: number; lowCount: number; highCount: number; wildcardTier: 0 | 1 | 2 } {
  let literalCount = 0;
  let lowCount = 0;
  let highCount = 0;
  for (const token of pattern) {
    if (token.kind === "literal") literalCount += 1;
    else if (token.wildcard === "_") lowCount += 1;
    else highCount += 1;
  }
  const wildcardTier: 0 | 1 | 2 = highCount > 0 ? 0 : lowCount > 0 ? 1 : 2;
  return { literalCount, lowCount, highCount, wildcardTier };
}

export function textTemplate(text: string): TemplateAst {
  return [{ kind: "text", text }];
}

/** Build a single-template CompiledCategory from a plain-English pattern source string. */
export function makeCategory(opts: {
  id: number;
  pattern: string;
  templates?: TemplateAst[];
  that?: string | null;
  topic?: string | null;
  setTopic?: string | null;
}): CompiledCategory {
  const pattern = tokenizePatternSource(opts.pattern);
  const stats = patternStats(pattern);
  return {
    id: opts.id,
    patternSource: opts.pattern,
    pattern,
    that: opts.that != null ? tokenizePatternSource(opts.that) : null,
    // Mirrors the compiler: topic is stored as an already-normalized exact-match
    // label, not a pattern, so tests build it the same way match.ts's runtime
    // normalization does.
    topic: opts.topic != null ? normalize(opts.topic).join(" ") : null,
    setTopic: opts.setTopic ?? null,
    templates: opts.templates ?? [textTemplate(`response for ${opts.pattern}`)],
    ...stats,
  };
}
