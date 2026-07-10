/**
 * Sanity-checks the matcher against the real, compiled
 * lib/aiml/generated/categories.json (44 hand-authored categories spanning
 * greetings/small-talk/emotional-checkins/meta-questions/farewells) -- not
 * just the hand-built fixtures used elsewhere in this directory's tests.
 * This is what actually ships to app/demo/page.tsx, so it's worth a direct
 * check that real data behaves as expected end-to-end.
 */

import { describe, expect, it } from "vitest";
import categoriesData from "./generated/categories.json";
import { match } from "./match";
import type { CompiledCategory } from "./types";

const categories = categoriesData.categories as unknown as CompiledCategory[];

describe("match against the real compiled categories.json", () => {
  it("resolves HELLO deterministically", () => {
    const result = match("HELLO", { lastBotUtterance: "", topic: null }, categories);
    expect(result.kind).toBe("single");
    if (result.kind !== "single") throw new Error("expected single");
    expect(typeof result.response).toBe("string");
    expect(result.response.length).toBeGreaterThan(0);
  });

  it("reports HOW ARE YOU as ambiguous (a <random> block in the real data)", () => {
    const result = match("HOW ARE YOU", { lastBotUtterance: "", topic: null }, categories);
    expect(result.kind).toBe("ambiguous");
  });

  it("GOODBYE then SEE YOU demonstrates setTopic/topic-scoping working end-to-end", () => {
    // Turn 1: GOODBYE has no <that>/<topic> scope of its own, but its
    // winning category sets topic -> FAREWELL for the next turn.
    const turn1 = match("GOODBYE", { lastBotUtterance: "", topic: null }, categories);
    expect(turn1.kind).toBe("single");
    if (turn1.kind !== "single") throw new Error("expected single");
    expect(turn1.category.setTopic).toBe("FAREWELL");

    const topicAfterGoodbye = turn1.category.setTopic;

    // Turn 2, in the FAREWELL topic: the FAREWELL-scoped "SEE YOU" category
    // must win over the generic unscoped "SEE YOU" category.
    const turn2InFarewell = match("SEE YOU", { lastBotUtterance: "", topic: topicAfterGoodbye }, categories);
    expect(turn2InFarewell.kind).toBe("single");
    if (turn2InFarewell.kind !== "single") throw new Error("expected single");
    expect(turn2InFarewell.category.topic).toBe("FAREWELL");

    // Without that topic in context, "SEE YOU" should fall to the generic,
    // unscoped category instead.
    const turn2NoTopic = match("SEE YOU", { lastBotUtterance: "", topic: null }, categories);
    expect(turn2NoTopic.kind).toBe("single");
    if (turn2NoTopic.kind !== "single") throw new Error("expected single");
    expect(turn2NoTopic.category.topic).toBeNull();

    // The two "SEE YOU" categories must actually be different categories.
    expect(turn2InFarewell.category.id).not.toBe(turn2NoTopic.category.id);
  });
});
