import { describe, expect, it } from "vitest";
import { injectProjectInstructions } from "./promptBuilder";

const BASE_PROMPT = "<|system|>\nYou are Benny, a small helpful assistant.\n<|end|>\n<|user|>\nhi\n<|end|>\n<|assistant|>\n";

describe("injectProjectInstructions", () => {
  it("returns the prompt unchanged when there are no instructions", () => {
    expect(injectProjectInstructions(BASE_PROMPT, undefined)).toBe(BASE_PROMPT);
    expect(injectProjectInstructions(BASE_PROMPT, null)).toBe(BASE_PROMPT);
    expect(injectProjectInstructions(BASE_PROMPT, "")).toBe(BASE_PROMPT);
    expect(injectProjectInstructions(BASE_PROMPT, "   ")).toBe(BASE_PROMPT);
  });

  it("inserts a second system turn right after the base system turn", () => {
    const result = injectProjectInstructions(BASE_PROMPT, "Always answer in haiku.");

    expect(result).toBe(
      "<|system|>\nYou are Benny, a small helpful assistant.\n<|end|>\n" +
        "<|system|>\nAlways answer in haiku.\n<|end|>\n" +
        "<|user|>\nhi\n<|end|>\n<|assistant|>\n",
    );
  });

  it("trims the instructions text before inserting", () => {
    const result = injectProjectInstructions(BASE_PROMPT, "  Be terse.  ");
    expect(result).toContain("<|system|>\nBe terse.\n<|end|>\n");
  });

  it("places the instructions turn before any user/assistant turns", () => {
    const result = injectProjectInstructions(BASE_PROMPT, "Be terse.");
    const instructionsIdx = result.indexOf("Be terse.");
    const firstUserIdx = result.indexOf("<|user|>");
    expect(instructionsIdx).toBeGreaterThan(-1);
    expect(instructionsIdx).toBeLessThan(firstUserIdx);
  });

  it("leaves the prompt untouched if the expected end-turn marker is missing", () => {
    const malformed = "no chat tokens here";
    expect(injectProjectInstructions(malformed, "Be terse.")).toBe(malformed);
  });
});
