import { describe, expect, it } from "vitest";
import { render } from "./render";

describe("render", () => {
  it("concatenates text nodes", () => {
    expect(render([{ kind: "text", text: "Hello there" }], [])).toBe("Hello there");
  });

  it("substitutes star nodes (1-based index) with the matching capture", () => {
    const ast = [
      { kind: "text" as const, text: "Nice to meet you, " },
      { kind: "star" as const, index: 1 },
      { kind: "text" as const, text: "!" },
    ];
    expect(render(ast, ["Shane"])).toBe("Nice to meet you, Shane!");
  });

  it("substitutes an empty string for an out-of-range star index", () => {
    const ast = [{ kind: "star" as const, index: 2 }];
    expect(render(ast, ["only-one"])).toBe("");
  });
});
