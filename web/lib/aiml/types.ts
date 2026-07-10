/**
 * Type definitions for the deterministic AIML-style pattern-matching layer.
 *
 * These types mirror the schema emitted by the Python compiler
 * (aiml/tools/aiml_compiler.py) into lib/aiml/generated/categories.json.
 * This module only defines shapes -- the actual matching algorithm lives in
 * patternMatcher.ts / match.ts / render.ts.
 */

export type PatternToken = { kind: "literal"; word: string } | { kind: "wildcard"; wildcard: "*" | "_" };

export interface CompiledCategory {
  id: number;
  patternSource: string;
  pattern: PatternToken[];
  that: PatternToken[] | null;
  topic: string | null;
  setTopic: string | null;
  templates: TemplateAst[];
  literalCount: number;
  lowCount: number;
  highCount: number;
  wildcardTier: 0 | 1 | 2;
}

export type TemplateNode = { kind: "text"; text: string } | { kind: "star"; index: number };

export type TemplateAst = TemplateNode[];

export interface MatcherContext {
  lastBotUtterance: string;
  topic: string | null;
}

export type MatchResult =
  | { kind: "no-match" }
  | { kind: "single"; category: CompiledCategory; captures: string[]; response: string }
  | { kind: "ambiguous"; category: CompiledCategory; captures: string[]; candidateCount: number };
