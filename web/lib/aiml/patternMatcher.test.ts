import { describe, expect, it } from "vitest";
import { matchTokens, normalize } from "./patternMatcher";
import { tokenizePatternSource } from "./testFixtures";

describe("normalize", () => {
  it("uppercases, strips punctuation (keeping apostrophes), and splits on whitespace", () => {
    expect(normalize("Hello, world!")).toEqual(["HELLO", "WORLD"]);
    expect(normalize("What's   your   name?")).toEqual(["WHAT'S", "YOUR", "NAME"]);
  });

  it("drops words that are pure punctuation", () => {
    expect(normalize("Hi -- there")).toEqual(["HI", "THERE"]);
  });

  it("returns an empty array for empty/whitespace-only input", () => {
    expect(normalize("")).toEqual([]);
    expect(normalize("   ")).toEqual([]);
  });
});

describe("matchTokens", () => {
  it("matches a fully literal pattern exactly", () => {
    const pattern = tokenizePatternSource("HELLO THERE");
    expect(matchTokens(pattern, ["HELLO", "THERE"])).toEqual({ captures: [] });
  });

  it("fails on a fully literal pattern that doesn't match (not a substring match)", () => {
    const pattern = tokenizePatternSource("HELLO THERE");
    expect(matchTokens(pattern, ["HELLO", "THERE", "FRIEND"])).toBeNull();
    expect(matchTokens(pattern, ["HELLO"])).toBeNull();
  });

  it("`_` consumes exactly one word and captures it", () => {
    const pattern = tokenizePatternSource("MY NAME IS _");
    const result = matchTokens(pattern, ["MY", "NAME", "IS", "SHANE"]);
    expect(result).toEqual({ captures: ["SHANE"] });
  });

  it("`_` does not match zero or multiple words", () => {
    const pattern = tokenizePatternSource("MY NAME IS _");
    expect(matchTokens(pattern, ["MY", "NAME", "IS"])).toBeNull();
    expect(matchTokens(pattern, ["MY", "NAME", "IS", "SHANE", "SMITH"])).toBeNull();
  });

  it("`*` greedily captures the longest possible span at the end of a pattern", () => {
    const pattern = tokenizePatternSource("MY NAME IS *");
    const result = matchTokens(pattern, ["MY", "NAME", "IS", "SHANE", "SMITH"]);
    expect(result).toEqual({ captures: ["SHANE SMITH"] });
  });

  it("`*` backtracks from a failed longest-first attempt to find a valid split", () => {
    // "* IS *" against "A IS B IS C": the greedy first attempt for the first
    // `*` consumes everything ("A IS B IS C"), which leaves nothing for the
    // literal "IS" or the second `*` -- that fails, and every shorter first
    // span fails too until the first `*` backs off to "A IS B", leaving
    // "IS" to match words[3] and the second `*` to capture "C".
    const pattern = tokenizePatternSource("* IS *");
    const result = matchTokens(pattern, ["A", "IS", "B", "IS", "C"]);
    expect(result).toEqual({ captures: ["A IS B", "C"] });
  });

  it("`*` requires at least one word (does not match an empty span)", () => {
    const pattern = tokenizePatternSource("HELLO *");
    expect(matchTokens(pattern, ["HELLO"])).toBeNull();
  });

  it("multiple wildcards capture in left-to-right pattern order", () => {
    const pattern = tokenizePatternSource("* NAME IS *");
    const result = matchTokens(pattern, ["MY", "NAME", "IS", "SHANE"]);
    expect(result).toEqual({ captures: ["MY", "SHANE"] });
  });
});
