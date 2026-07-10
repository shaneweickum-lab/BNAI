import { describe, expect, it } from "vitest";
import { makeCategory, textTemplate } from "../aiml/testFixtures";
import type { CompiledCategory } from "../aiml/types";
import { route } from "./dialogueManager";
import type { DialogueState } from "./types";

const emptyState: DialogueState = { topic: null, history: [] };

describe("dialogueManager.route", () => {
  it("resolves a single-match turn deterministically, with no gptPrompt", () => {
    const categories: CompiledCategory[] = [makeCategory({ id: 0, pattern: "HELLO", templates: [textTemplate("Hi there!")] })];

    const result = route(emptyState, "hello", categories);

    expect(result.path).toBe("aiml");
    if (result.path !== "aiml") throw new Error("expected aiml path");
    expect(result.response).toBe("Hi there!");
    expect(result.newState.history).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "Hi there!" },
    ]);
    expect("gptPrompt" in result).toBe(false);
  });

  it("routes a no-match turn to gpt-no-match with a history-inclusive prompt", () => {
    const categories: CompiledCategory[] = [makeCategory({ id: 0, pattern: "HELLO" })];
    const state: DialogueState = {
      topic: null,
      history: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello there" },
      ],
    };

    const result = route(state, "tell me about the weather", categories);

    expect(result.path).toBe("gpt-no-match");
    if (result.path !== "gpt-no-match") throw new Error("expected gpt-no-match path");
    // Only the user turn is appended -- the assistant turn is appended later
    // by the caller once GPT actually responds.
    expect(result.newState.history).toEqual([
      ...state.history,
      { role: "user", content: "tell me about the weather" },
    ]);
    expect(result.gptPrompt).toContain("<|system|>");
    expect(result.gptPrompt).toContain("hi");
    expect(result.gptPrompt).toContain("hello there");
    expect(result.gptPrompt).toContain("tell me about the weather");
    expect(result.gptPrompt.endsWith("<|assistant|>\n")).toBe(true);
  });

  it("routes an ambiguous-match turn to gpt-ambiguous", () => {
    const categories: CompiledCategory[] = [
      makeCategory({
        id: 0,
        pattern: "HOW ARE YOU",
        templates: [textTemplate("Great, thanks!"), textTemplate("Doing well!")],
      }),
    ];

    const result = route(emptyState, "how are you", categories);

    expect(result.path).toBe("gpt-ambiguous");
    if (result.path !== "gpt-ambiguous") throw new Error("expected gpt-ambiguous path");
    expect(result.gptPrompt).toContain("how are you");
  });

  it("updates newState.topic when the winning category sets one", () => {
    const categories: CompiledCategory[] = [
      makeCategory({ id: 0, pattern: "GOODBYE", templates: [textTemplate("Bye!")], setTopic: "FAREWELL" }),
    ];

    const result = route(emptyState, "goodbye", categories);

    expect(result.path).toBe("aiml");
    expect(result.newState.topic).toBe("FAREWELL");
  });

  it("leaves topic unchanged when the winning category has no setTopic", () => {
    const categories: CompiledCategory[] = [makeCategory({ id: 0, pattern: "HELLO", templates: [textTemplate("Hi!")] })];
    const state: DialogueState = { topic: "SOME_TOPIC", history: [] };

    const result = route(state, "hello", categories);

    expect(result.newState.topic).toBe("SOME_TOPIC");
  });

  it("truncates by dropping the oldest turns first when history exceeds maxContextTokens", () => {
    const categories: CompiledCategory[] = [makeCategory({ id: 0, pattern: "HELLO" })];
    const history = [
      { role: "user" as const, content: "first message from long ago" },
      { role: "assistant" as const, content: "first reply from long ago" },
      { role: "user" as const, content: "second message" },
      { role: "assistant" as const, content: "second reply" },
    ];
    const state: DialogueState = { topic: null, history };

    // A tiny token budget that can't possibly fit the whole history plus the
    // system turn and new user turn.
    const result = route(state, "final question", categories, 12);

    expect(result.path).toBe("gpt-no-match");
    if (result.path !== "gpt-no-match") throw new Error("expected gpt-no-match path");
    expect(result.gptPrompt).not.toContain("first message from long ago");
    expect(result.gptPrompt).not.toContain("first reply from long ago");
    // The most recent turns (and the new user turn) should still be present.
    expect(result.gptPrompt).toContain("final question");

    // newState.history is NOT truncated -- only the rendered prompt is. Full
    // conversation history is preserved for the app/UI layer.
    expect(result.newState.history).toEqual([...history, { role: "user", content: "final question" }]);
  });
});
