/**
 * Renders a single compiled template (one entry of `CompiledCategory.templates`)
 * into plain text, substituting `<star index="N">` placeholders with the
 * captures produced by patternMatcher.matchTokens.
 */

import type { TemplateNode } from "./types";

export function render(templateAst: TemplateNode[], captures: string[]): string {
  let out = "";
  for (const node of templateAst) {
    if (node.kind === "text") {
      out += node.text;
    } else {
      out += captures[node.index - 1] ?? "";
    }
  }
  return out;
}
