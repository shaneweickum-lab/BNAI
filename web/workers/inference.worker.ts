/**
 * Inference Web Worker.
 *
 * Owns the WASM model load and the entire generation loop, off the main
 * thread, so the chat UI (app/demo/page.tsx via lib/useInferenceWorker.ts)
 * never blocks. Talks to the main thread purely via postMessage using the
 * message shapes defined in lib/workerProtocol.ts.
 *
 * All model-specific logic (tokenization, sampling) lives behind
 * worker/wasmEngine.ts -- this file only owns fetching, timing, and the
 * postMessage protocol.
 */

import { fetchWithProgress } from "../lib/modelCache";
import { MODEL_FILE_PATH, TOKENIZER_FILE_PATH } from "../lib/modelInfo";
import type { MainToWorkerMessage, WorkerToMainMessage } from "../lib/workerProtocol";
import { decode, encode, generateNextToken, loadSession, resetSession, type WasmSession } from "../worker/wasmEngine";

// Cast to the dom-lib `Worker` interface purely to get a conveniently typed
// single-argument postMessage()/addEventListener() without pulling in the
// `webworker` lib (which would conflict with the `dom` lib the rest of the
// app's tsconfig uses).
const ctx = self as unknown as Worker;

let session: WasmSession | null = null;
const cancelledRequests = new Set<string>();

function post(message: WorkerToMainMessage) {
  ctx.postMessage(message);
}

async function init() {
  try {
    const modelBytes = await fetchWithProgress(MODEL_FILE_PATH, (p) =>
      post({ type: "download-progress", loadedBytes: p.loadedBytes, totalBytes: p.totalBytes, fromCache: p.fromCache }),
    );
    const tokenizerBytes = await fetchWithProgress(TOKENIZER_FILE_PATH, () => {});
    const tokenizerJson = new TextDecoder("utf-8").decode(tokenizerBytes);

    session = await loadSession(modelBytes, tokenizerJson);
    const stats = session.getStats();
    post({ type: "ready", paramCount: stats.paramCount, fileSizeBytes: stats.fileSizeBytes, contextLen: stats.contextLen });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
}

async function generate(requestId: string, prompt: string, maxNewTokens: number) {
  if (!session) {
    post({ type: "error", requestId, message: "Session not initialized yet." });
    return;
  }

  // `prompt` is treated as pre-rendered, complete multi-turn text (built by
  // lib/dialogue/dialogueManager.ts using the model's <|system|>/<|user|>/
  // <|assistant|>/<|end|> chat tokens) -- the worker no longer wraps it in
  // its own single-turn template, it just encodes what it's given.
  const promptIds = encode(session, prompt);
  if (promptIds.length === 0) {
    post({ type: "error", requestId, message: "Prompt encoded to zero tokens." });
    return;
  }

  // Feed the whole prompt through the (mock) session, one token at a time,
  // to prime its KV-cache/position state -- this mirrors how the real
  // autoregressive Session is expected to work.
  for (let i = 0; i < promptIds.length - 1; i++) {
    if (cancelledRequests.has(requestId)) {
      cancelledRequests.delete(requestId);
      return;
    }
    // Feed each prompt token to advance the KV-cache position; the
    // "predicted next token" returned here is discarded because we already
    // know the real next prompt token (standard prefill behavior).
    generateNextToken(session, promptIds[i]);
  }
  let lastId = promptIds[promptIds.length - 1];
  const generatedIds: number[] = [];

  const startTime = performance.now();
  let tokensGenerated = 0;

  for (let i = 0; i < maxNewTokens; i++) {
    if (cancelledRequests.has(requestId)) {
      cancelledRequests.delete(requestId);
      break;
    }

    // TODO(runtime): this setTimeout is standing in for real per-token
    // compute cost. The mock engine samples in well under a millisecond,
    // which would otherwise report an absurd (tens of thousands of
    // tokens/sec) rate that no CPU-bound WASM decode loop could hit. Remove
    // this once feedToken() is doing real forward-pass work.
    await new Promise((resolve) => setTimeout(resolve, 25 + Math.random() * 25));

    const nextId = generateNextToken(session, lastId);
    lastId = nextId;
    tokensGenerated += 1;
    generatedIds.push(nextId);

    // Re-decode the *entire* generated-so-far id sequence rather than just
    // the newest token: a single BPE token's bytes don't always align to a
    // full UTF-8 codepoint, so decoding one new token in isolation and
    // string-concatenating the results can split a multi-byte character
    // across two messages and render as replacement-character mojibake.
    const textSoFar = decode(session, generatedIds);
    const elapsedSeconds = (performance.now() - startTime) / 1000;
    const tokensPerSecond = elapsedSeconds > 0 ? tokensGenerated / elapsedSeconds : 0;

    post({ type: "token", requestId, textSoFar, tokensPerSecond });

    const stats = session.getStats();
    if (stats.contextRemaining <= 0) break;

    // Stop on end-of-turn/end-of-sequence-style special tokens.
    const newTokenText = decode(session, [nextId]);
    if (["<eos>", "<|end|>"].includes(newTokenText)) break;
  }

  const elapsedMs = performance.now() - startTime;
  post({ type: "done", requestId, totalTokens: tokensGenerated, elapsedMs });
}

ctx.addEventListener("message", (event: MessageEvent<MainToWorkerMessage>) => {
  const msg = event.data;
  switch (msg.type) {
    case "init":
      void init();
      break;
    case "generate":
      void generate(msg.requestId, msg.prompt, msg.maxNewTokens);
      break;
    case "cancel":
      cancelledRequests.add(msg.requestId);
      break;
    case "reset":
      if (session) resetSession(session);
      break;
  }
});

export {};
