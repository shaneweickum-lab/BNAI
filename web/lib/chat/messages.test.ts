import { describe, expect, it } from "vitest";
import type { DialogueTurn } from "../dialogue/types";
import { deriveMessages } from "./messages";

describe("deriveMessages", () => {
  it("returns an empty list for empty history", () => {
    expect(deriveMessages([], {})).toEqual([]);
  });

  it("carries resolvedBy only for assistant turns present in the index map", () => {
    const history: DialogueTurn[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "how are you" },
      { role: "assistant", content: "great" },
    ];
    const resolvedByIndex = { 1: "deterministic", 3: "generated" } as const;

    const messages = deriveMessages(history, resolvedByIndex, "conv1");

    expect(messages).toEqual([
      { id: "conv1-0", role: "user", content: "hi", resolvedBy: undefined },
      { id: "conv1-1", role: "assistant", content: "hello", resolvedBy: "deterministic" },
      { id: "conv1-2", role: "user", content: "how are you", resolvedBy: undefined },
      { id: "conv1-3", role: "assistant", content: "great", resolvedBy: "generated" },
    ]);
  });

  it("leaves resolvedBy undefined for an assistant turn missing from the index (e.g. after a reload)", () => {
    const history: DialogueTurn[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ];

    const messages = deriveMessages(history, {});

    expect(messages[1].resolvedBy).toBeUndefined();
  });

  it("produces unique, stable ids across different id prefixes", () => {
    const history: DialogueTurn[] = [{ role: "user", content: "hi" }];
    const a = deriveMessages(history, {}, "convA");
    const b = deriveMessages(history, {}, "convB");
    expect(a[0].id).not.toBe(b[0].id);
  });
});
