/**
 * Injects a project's custom instructions (Claude-Projects-style, see
 * StoredProject.instructions in lib/store/types.ts) into an already-rendered
 * GPT prompt string produced by lib/dialogue/dialogueManager.ts's route().
 *
 * Design decision (see task writeup / final report): dialogueManager.route()
 * does NOT take a system-instructions parameter today, and its signature is
 * out of scope to change here. Rather than touch that shared, tested module,
 * this does the injection at the string level, in the chat UI layer, AFTER
 * route() has already decided the AIML-vs-GPT path -- so project
 * instructions can only ever affect the GPT-fallback prompt, never whether
 * the deterministic AIML matcher fires (route() is called with the same
 * categories/state regardless of which project the conversation belongs to).
 *
 * The rendered prompt looks like:
 *   <|system|>\nYou are Benny...\n<|end|>\n<|user|>\n...<|end|>\n...<|assistant|>\n
 * This inserts a second <|system|> turn (the project instructions) right
 * after the base system turn's closing <|end|>\n, i.e. before any user/
 * assistant turns -- "extra system-turn text", per the task's own framing.
 */

const SYSTEM_TOKEN = "<|system|>";
const END_TOKEN = "<|end|>";
const END_TURN_MARKER = `${END_TOKEN}\n`;

export function injectProjectInstructions(gptPrompt: string, instructions: string | null | undefined): string {
  const trimmed = instructions?.trim();
  if (!trimmed) return gptPrompt;

  const insertAt = gptPrompt.indexOf(END_TURN_MARKER);
  // Fail safe: if the expected chat-token delimiter isn't found (e.g. the
  // dialogue manager's prompt format ever changes), leave the prompt
  // untouched rather than risk corrupting it with a misplaced insert.
  if (insertAt === -1) return gptPrompt;

  const afterFirstSystemTurn = insertAt + END_TURN_MARKER.length;
  const instructionsTurn = `${SYSTEM_TOKEN}\n${trimmed}\n${END_TURN_MARKER}`;

  return gptPrompt.slice(0, afterFirstSystemTurn) + instructionsTurn + gptPrompt.slice(afterFirstSystemTurn);
}
