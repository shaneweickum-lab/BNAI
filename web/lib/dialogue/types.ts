/**
 * Types for the dialogue manager: the thin layer that decides, per turn,
 * whether the deterministic AIML matcher (lib/aiml/match.ts) can answer, or
 * whether the turn needs to fall back to the GPT Web Worker.
 */

export interface DialogueTurn {
  role: "user" | "assistant";
  content: string;
}

export interface DialogueState {
  topic: string | null;
  // `lastBotUtterance` for the matcher is derived on demand from the last
  // `role: "assistant"` entry here (or "" if there isn't one yet) -- it is
  // deliberately not stored redundantly alongside history.
  history: DialogueTurn[];
}
