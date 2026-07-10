/**
 * Dialogue manager: pure logic (no React) that runs on every turn to decide
 * whether the deterministic AIML matcher can answer, or whether the turn
 * must fall back to the ternary GPT transformer running in the Web Worker.
 *
 * This is the crux of the hybrid design: deterministic where determinism is
 * sufficient (a cheap, synchronous, fully-predictable pattern match), neural
 * only where necessary (no unambiguous deterministic answer exists). Every
 * turn tries the cheap path first; the expensive path is a fallback, not the
 * default.
 */

import { match } from "../aiml/match";
import type { CompiledCategory, MatcherContext } from "../aiml/types";
import type { DialogueState, DialogueTurn } from "./types";

export type RouteResult =
  | { path: "aiml"; response: string; newState: DialogueState }
  | { path: "gpt-no-match" | "gpt-ambiguous"; gptPrompt: string; newState: DialogueState };

// Chat special tokens (see model/tokenizer/bpe.py): rendered here as literal
// token *text* -- the tokenizer re-encodes these strings back to their fixed
// IDs, so the dialogue manager never needs to know the numeric IDs.
const SYSTEM_TOKEN = "<|system|>";
const USER_TOKEN = "<|user|>";
const ASSISTANT_TOKEN = "<|assistant|>";
const END_TOKEN = "<|end|>";

const SYSTEM_PROMPT = "You are Benny, a small helpful assistant.";

// No token budget passed in: unbounded (renders full history). Callers that
// care about the model's actual context window (app/demo/page.tsx, from
// useInferenceWorker's modelStats.contextLen) should pass a real value.
const UNBOUNDED_MAX_CONTEXT_TOKENS = Number.POSITIVE_INFINITY;

function renderTurnTokens(role: "system" | "user" | "assistant", content: string): string {
  const token = role === "system" ? SYSTEM_TOKEN : role === "user" ? USER_TOKEN : ASSISTANT_TOKEN;
  return `${token}\n${content}\n${END_TOKEN}\n`;
}

/** Word-count proxy for token-count -- deliberately not the real BPE
 * tokenizer, which would be over-engineering for a simple truncation
 * heuristic that only needs to be roughly right. */
function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/**
 * Renders the full multi-turn prompt (pre-rendered, complete text handed to
 * the worker as-is -- see workers/inference.worker.ts) from a system turn
 * plus the given history, ending with a trailing "<|assistant|>\n" to
 * prompt generation. If it would exceed `maxContextTokens` (a word-count
 * proxy), drops the oldest turns first until it fits -- simple truncation,
 * no summarization.
 */
function buildGptPrompt(history: DialogueTurn[], maxContextTokens: number): string {
  const systemTurn = renderTurnTokens("system", SYSTEM_PROMPT);
  const trailingPrompt = `${ASSISTANT_TOKEN}\n`;

  let turns = history;
  for (;;) {
    const body = turns.map((t) => renderTurnTokens(t.role, t.content)).join("");
    const fullPrompt = systemTurn + body + trailingPrompt;
    if (wordCount(fullPrompt) <= maxContextTokens || turns.length <= 1) {
      return fullPrompt;
    }
    turns = turns.slice(1);
  }
}

function lastBotUtterance(history: DialogueTurn[]): string {
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === "assistant") return history[i].content;
  }
  return "";
}

export function route(
  state: DialogueState,
  userInput: string,
  categories: CompiledCategory[],
  maxContextTokens: number = UNBOUNDED_MAX_CONTEXT_TOKENS,
): RouteResult {
  const context: MatcherContext = { lastBotUtterance: lastBotUtterance(state.history), topic: state.topic };
  const result = match(userInput, context, categories);

  if (result.kind === "single") {
    const newHistory: DialogueTurn[] = [
      ...state.history,
      { role: "user", content: userInput },
      { role: "assistant", content: result.response },
    ];
    const newTopic = result.category.setTopic !== null ? result.category.setTopic : state.topic;
    return {
      path: "aiml",
      response: result.response,
      newState: { topic: newTopic, history: newHistory },
    };
  }

  // "no-match" / "ambiguous": the assistant turn isn't known yet (GPT hasn't
  // responded), so only the user turn is appended here -- the caller appends
  // the assistant turn once generation finishes.
  const newHistory: DialogueTurn[] = [...state.history, { role: "user", content: userInput }];
  const newState: DialogueState = { topic: state.topic, history: newHistory };
  const gptPrompt = buildGptPrompt(newHistory, maxContextTokens);

  return {
    path: result.kind === "no-match" ? "gpt-no-match" : "gpt-ambiguous",
    gptPrompt,
    newState,
  };
}
