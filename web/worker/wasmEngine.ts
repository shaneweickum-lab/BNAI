/**
 * wasmEngine.ts -- the integration seam between the chat UI's Web Worker
 * (workers/inference.worker.ts) and the Rust/WASM inference engine that
 * `runtime/` will eventually build.
 *
 * ============================================================================
 * CONTRACT FOR THE REAL RUST/WASM INTEGRATION (read this before wiring it up)
 * ============================================================================
 *
 * Everything in this file that touches actual model weights is a stand-in.
 * When `runtime/pkg` (the wasm-bindgen build output) exists, replace the
 * `MockWasmSession` class and `loadSession()` body below with real bindings
 * that satisfy the exact same `WasmSession` interface and `loadSession`
 * signature exported from this module:
 *
 *   export interface WasmSession {
 *     encode(text: string): number[];
 *     decode(ids: number[]): string;
 *     feedToken(tokenId: number): number;   // one autoregressive step
 *     resetKvCache(): void;
 *     getStats(): SessionStats;
 *   }
 *
 *   export async function loadSession(
 *     modelBytes: ArrayBuffer,
 *     tokenizerJson: string,
 *   ): Promise<WasmSession>
 *
 * Concretely, the expected real-world shape (per the runtime's design doc)
 * is a wasm-bindgen `Session` class constructed from the raw `.bnai` bytes
 * plus the tokenizer JSON string, e.g.:
 *
 *   import init, { Session } from "../../runtime/pkg/bnai_runtime.js";
 *   await init();
 *   const session = new Session(new Uint8Array(modelBytes), tokenizerJson);
 *   // Session must expose: encode(text), decode(ids), feedToken(tokenId),
 *   // resetKvCache(), and getStats() (or equivalent getters for
 *   // paramCount / fileSizeBytes / contextLen / contextRemaining).
 *
 * If the real bindings use slightly different method names (e.g. `feed_token`
 * from wasm-bindgen's default snake_case-to-camelCase behavior, or separate
 * getters instead of one getStats() bundle), adapt them with a thin wrapper
 * class here that implements `WasmSession` -- nothing outside this file
 * should need to change. `workers/inference.worker.ts` only ever calls the
 * five functions exported from this module, never the underlying bindings
 * directly.
 *
 * ============================================================================
 * WHAT'S MOCKED TODAY
 * ============================================================================
 * - Tokenization (encode/decode) is REAL: it implements the documented
 *   byte-level BPE format against the actual vocab + merges shipped in
 *   benny-placeholder.bnai.tokenizer.json (see lib/bpeTokenizer.ts).
 * - Model metadata (param count, file size, context length) is REAL: it's
 *   parsed straight out of the .bnai file header (see lib/bnaiHeader.ts).
 * - Generation (feedToken) is MOCKED: there are no trained weights yet
 *   (benny-placeholder.bnai has random ternary weights), and the Rust WASM
 *   kernel doesn't exist yet either, so `feedToken` just samples a
 *   pseudo-random token id from the real vocabulary. This is intentionally
 *   NOT trying to look smart -- it exists purely so the chat UI, worker
 *   plumbing, and tokens/sec measurement all have something real to drive
 *   against before the runtime lands.
 */

import { parseBnaiHeader } from "../lib/bnaiHeader";
import { ByteLevelBpeTokenizer, type TokenizerJson } from "../lib/bpeTokenizer";

export interface SessionStats {
  paramCount: number;
  fileSizeBytes: number;
  contextLen: number;
  contextRemaining: number;
}

export interface WasmSession {
  encode(text: string): number[];
  decode(ids: number[]): string;
  /** Feed one token id in, get the next sampled token id out (one autoregressive step). */
  feedToken(tokenId: number): number;
  /** Clear the KV cache and reset the position counter to 0. */
  resetKvCache(): void;
  getStats(): SessionStats;
}

// TODO(runtime): swap for the real wasm-bindgen bindings once runtime/pkg exists.
class MockWasmSession implements WasmSession {
  private tokenizer: ByteLevelBpeTokenizer;
  private paramCount: number;
  private fileSizeBytes: number;
  private contextLen: number;
  private position = 0;
  private vocabIds: number[];
  private endOfTurnIds: number[];

  constructor(tokenizer: ByteLevelBpeTokenizer, paramCount: number, fileSizeBytes: number, contextLen: number) {
    this.tokenizer = tokenizer;
    this.paramCount = paramCount;
    this.fileSizeBytes = fileSizeBytes;
    this.contextLen = contextLen;

    // Precompute a sampling pool that excludes control/padding tokens and
    // any byte-level BPE merge piece that isn't valid UTF-8 on its own (see
    // isCleanStandaloneToken doc comment) -- since this mock samples
    // uniformly at random with no actual language model behind it, gluing
    // together arbitrary raw-byte merge pieces would otherwise render as
    // "�" mojibake noise. Restricting to clean standalone tokens keeps
    // the mock's output readable (if meaningless) "word salad" instead.
    const allIds: number[] = [];
    for (let id = 0; id < tokenizer.vocabSize; id++) {
      if (tokenizer.isSpecialToken(id)) continue;
      if (tokenizer.isCleanStandaloneToken(id)) allIds.push(id);
    }
    this.vocabIds = allIds.length > 0 ? allIds : [0];

    // The specific tokens that should end a turn -- NOT just "any special
    // token" (that previously included <pad>/<unk>/<bos>, which could get
    // emitted mid-reply and show up literally as the text "<pad>").
    this.endOfTurnIds = ["<eos>", "<|end|>"]
      .map((t) => tokenizer.idForToken(t))
      .filter((id): id is number => id !== undefined);
  }

  encode(text: string): number[] {
    return this.tokenizer.encode(text);
  }

  decode(ids: number[]): string {
    return this.tokenizer.decode(ids);
  }

  feedToken(tokenId: number): number {
    // The mock engine samples randomly rather than actually conditioning on
    // the input token -- there are no trained weights to condition with yet.
    // The real Rust/WASM Session will use this argument for real.
    void tokenId;
    this.position += 1;

    // Small, deterministic-ish chance to emit an end-of-turn token once the
    // reply has some length, so mock generations naturally terminate.
    if (this.position > 12 && this.endOfTurnIds.length > 0 && Math.random() < 0.04) {
      return this.endOfTurnIds[Math.floor(Math.random() * this.endOfTurnIds.length)];
    }

    const idx = Math.floor(Math.random() * this.vocabIds.length);
    return this.vocabIds[idx];
  }

  resetKvCache(): void {
    this.position = 0;
  }

  getStats(): SessionStats {
    return {
      paramCount: this.paramCount,
      fileSizeBytes: this.fileSizeBytes,
      contextLen: this.contextLen,
      contextRemaining: Math.max(0, this.contextLen - this.position),
    };
  }
}

/**
 * Construct a session from the raw .bnai bytes + tokenizer JSON string.
 * Mirrors the constructor signature the real wasm-bindgen `Session` is
 * expected to expose.
 */
export async function loadSession(modelBytes: ArrayBuffer, tokenizerJson: string): Promise<WasmSession> {
  const parsed = parseBnaiHeader(modelBytes);
  const tokenizerData = JSON.parse(tokenizerJson) as TokenizerJson;
  const tokenizer = new ByteLevelBpeTokenizer(tokenizerData);

  return new MockWasmSession(
    tokenizer,
    parsed.header.param_count,
    modelBytes.byteLength,
    parsed.header.context_len,
  );
}

export function encode(session: WasmSession, text: string): number[] {
  return session.encode(text);
}

export function decode(session: WasmSession, ids: number[]): string {
  return session.decode(ids);
}

/** One autoregressive generation step: feed a token id, get the next token id back. */
export function generateNextToken(session: WasmSession, tokenId: number): number {
  return session.feedToken(tokenId);
}

export function resetSession(session: WasmSession): void {
  session.resetKvCache();
}
