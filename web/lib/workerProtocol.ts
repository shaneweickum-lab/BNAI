/**
 * Message protocol between the main thread (app/demo/page.tsx via
 * lib/useInferenceWorker.ts) and the inference Web Worker
 * (workers/inference.worker.ts).
 *
 * Keeping this in one file means both sides import the same types, so the
 * postMessage contract can't silently drift.
 */

export type MainToWorkerMessage =
  | { type: "init" }
  | { type: "generate"; requestId: string; prompt: string; maxNewTokens: number }
  | { type: "cancel"; requestId: string }
  | { type: "reset" };

export type WorkerToMainMessage =
  | { type: "download-progress"; loadedBytes: number; totalBytes: number; fromCache: boolean }
  | { type: "ready"; paramCount: number; fileSizeBytes: number; contextLen: number }
  // `textSoFar` is the full decoded text of every token generated in this
  // request so far (not just the newest token) -- re-decoding the whole
  // token-id sequence on every step, rather than decoding one new token in
  // isolation, avoids splitting a multi-byte UTF-8 codepoint across two
  // "token" messages (which would otherwise render as mojibake/replacement
  // characters whenever a BPE token boundary falls mid-codepoint).
  | { type: "token"; requestId: string; textSoFar: string; tokensPerSecond: number }
  | { type: "done"; requestId: string; totalTokens: number; elapsedMs: number }
  | { type: "error"; message: string; requestId?: string };
