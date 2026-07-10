import { describe, expect, it } from "vitest";
import { match } from "./match";
import { makeCategory, textTemplate } from "./testFixtures";
import type { CompiledCategory, MatcherContext } from "./types";

const noContext: MatcherContext = { lastBotUtterance: "", topic: null };

describe("match", () => {
  // Worked example 1: higher literalCount wins between two otherwise-unscoped,
  // equally-eligible wildcard patterns.
  it("prefers the category with more literal words when both match", () => {
    const categories: CompiledCategory[] = [
      makeCategory({ id: 0, pattern: "MY NAME IS *", templates: [textTemplate("Hi *!")] }),
      makeCategory({ id: 1, pattern: "* NAME IS *", templates: [textTemplate("Hello, *.")] }),
    ];

    const result = match("MY NAME IS SHANE", noContext, categories);

    expect(result.kind).toBe("single");
    if (result.kind !== "single") throw new Error("expected single");
    expect(result.category.id).toBe(0);
    expect(result.captures).toEqual(["SHANE"]);
  });

  // Worked example 2: a <that>-scoped category beats an identical unscoped
  // pattern when the scope matches; it's excluded entirely when it doesn't.
  it("prefers a that-scoped category when the that context matches, and excludes it when it doesn't", () => {
    const categories: CompiledCategory[] = [
      makeCategory({ id: 2, pattern: "WHAT IS YOUR NAME", that: "I ASKED YOUR NAME" }),
      makeCategory({ id: 3, pattern: "WHAT IS YOUR NAME" }),
    ];

    const matchingContext: MatcherContext = { lastBotUtterance: "I ASKED YOUR NAME", topic: null };
    const matchingResult = match("WHAT IS YOUR NAME", matchingContext, categories);
    expect(matchingResult.kind).toBe("single");
    if (matchingResult.kind !== "single") throw new Error("expected single");
    expect(matchingResult.category.id).toBe(2);

    const otherContext: MatcherContext = { lastBotUtterance: "something else", topic: null };
    const otherResult = match("WHAT IS YOUR NAME", otherContext, categories);
    expect(otherResult.kind).toBe("single");
    if (otherResult.kind !== "single") throw new Error("expected single");
    expect(otherResult.category.id).toBe(3);
  });

  // Worked example 3: a <topic>-scoped category beats an identical unscoped
  // pattern when the topic matches exactly; excluded for any other topic
  // (including null).
  it("prefers a topic-scoped category when the topic matches, and excludes it otherwise", () => {
    const categories: CompiledCategory[] = [
      makeCategory({ id: 4, pattern: "HELLO", topic: "GREETING_TOPIC" }),
      makeCategory({ id: 5, pattern: "HELLO" }),
    ];

    const matchingResult = match("HELLO", { lastBotUtterance: "", topic: "GREETING_TOPIC" }, categories);
    expect(matchingResult.kind).toBe("single");
    if (matchingResult.kind !== "single") throw new Error("expected single");
    expect(matchingResult.category.id).toBe(4);

    const otherTopicResult = match("HELLO", { lastBotUtterance: "", topic: "SOMETHING_ELSE" }, categories);
    expect(otherTopicResult.kind).toBe("single");
    if (otherTopicResult.kind !== "single") throw new Error("expected single");
    expect(otherTopicResult.category.id).toBe(5);

    const nullTopicResult = match("HELLO", { lastBotUtterance: "", topic: null }, categories);
    expect(nullTopicResult.kind).toBe("single");
    if (nullTopicResult.kind !== "single") throw new Error("expected single");
    expect(nullTopicResult.category.id).toBe(5);
  });

  // Worked example 4: contextTier 2 (both that and topic set+matched) beats
  // contextTier 0 (neither), even at identical word-specificity.
  it("ranks a category with both that and topic scoping above one with neither, at equal specificity", () => {
    const categories: CompiledCategory[] = [
      makeCategory({ id: 6, pattern: "GOOD MORNING", that: "DID YOU SLEEP WELL", topic: "MORNING_CHAT" }),
      makeCategory({ id: 7, pattern: "GOOD MORNING" }),
    ];

    const context: MatcherContext = { lastBotUtterance: "DID YOU SLEEP WELL", topic: "MORNING_CHAT" };
    const result = match("GOOD MORNING", context, categories);

    expect(result.kind).toBe("single");
    if (result.kind !== "single") throw new Error("expected single");
    expect(result.category.id).toBe(6);
  });

  // Worked example 5: a category compiled from a <random> block (2+ templates)
  // must report "ambiguous" and must not render or pick a variant.
  it("reports ambiguous (without rendering) for a category with multiple templates", () => {
    const categories: CompiledCategory[] = [
      makeCategory({
        id: 8,
        pattern: "HOW ARE YOU",
        templates: [textTemplate("I'm doing well, thanks!"), textTemplate("Pretty good, how about you?")],
      }),
    ];

    const result = match("HOW ARE YOU", noContext, categories);

    expect(result.kind).toBe("ambiguous");
    if (result.kind !== "ambiguous") throw new Error("expected ambiguous");
    expect(result.category.id).toBe(8);
    expect(result.candidateCount).toBe(2);
    // Ambiguous results carry no rendered response field.
    expect((result as unknown as { response?: string }).response).toBeUndefined();
  });

  it("reports no-match when nothing is eligible", () => {
    const categories: CompiledCategory[] = [makeCategory({ id: 9, pattern: "HELLO" })];
    expect(match("SOMETHING COMPLETELY DIFFERENT", noContext, categories)).toEqual({ kind: "no-match" });
  });
});
