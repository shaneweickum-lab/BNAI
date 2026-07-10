/**
 * Feature detection for running the WASM inference engine in-browser.
 *
 * Checks:
 *  - WebAssembly + WASM SIMD support (the real Rust runtime is expected to
 *    be compiled with SIMD enabled for acceptable throughput on a 123.7M-param
 *    model; without it, generation would be unusably slow).
 *  - Safari version on iOS: pre-16.4 iOS Safari lacks WASM SIMD entirely,
 *    so we explicitly flag it as unsupported rather than let it silently
 *    fail or run at a crawl.
 *  - Web Worker + Cache API availability (needed for off-main-thread
 *    inference and for not re-downloading the ~69MB model every visit).
 */

export interface BrowserSupportResult {
  supported: boolean;
  reasons: string[];
  details: {
    hasWasm: boolean;
    hasWasmSimd: boolean;
    hasWebWorker: boolean;
    hasCacheApi: boolean;
    isIosSafari: boolean;
    iosSafariVersion: number | null;
  };
}

// A minimal, valid WASM module byte sequence that includes a SIMD (v128)
// instruction (i32x4 splat of a constant), used purely to probe whether
// WebAssembly.validate() accepts SIMD opcodes in this browser. This is the
// same byte sequence used by the `wasm-feature-detect` project's `simd()`
// check.
// prettier-ignore
const WASM_SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
  0x00, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
  0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b,
]);

function detectWasmSimd(): boolean {
  try {
    if (typeof WebAssembly === "undefined" || !WebAssembly.validate) return false;
    return WebAssembly.validate(WASM_SIMD_PROBE);
  } catch {
    return false;
  }
}

function detectIosSafari(userAgent: string): { isIosSafari: boolean; version: number | null } {
  const isIos = /iP(hone|od|ad)/.test(userAgent);
  if (!isIos) return { isIosSafari: false, version: null };

  // Exclude other iOS browsers (Chrome/Firefox/Edge on iOS all use the
  // system WebKit under the hood but report their own product name too);
  // we only special-case actual Safari since that's what the spec calls out.
  const isChromeOrOtherOnIos = /CriOS|FxiOS|EdgiOS|OPiOS/.test(userAgent);
  if (isChromeOrOtherOnIos) return { isIosSafari: false, version: null };

  const match = userAgent.match(/Version\/(\d+)\.(\d+)/);
  if (!match) return { isIosSafari: true, version: null };
  const version = parseFloat(`${match[1]}.${match[2]}`);
  return { isIosSafari: true, version };
}

export function checkBrowserSupport(userAgent: string = typeof navigator !== "undefined" ? navigator.userAgent : ""): BrowserSupportResult {
  const reasons: string[] = [];

  const hasWasm = typeof WebAssembly !== "undefined";
  if (!hasWasm) reasons.push("WebAssembly is not available in this browser.");

  const hasWasmSimd = hasWasm && detectWasmSimd();
  if (hasWasm && !hasWasmSimd) {
    reasons.push("WebAssembly SIMD is not supported, which the inference engine requires for usable performance.");
  }

  const hasWebWorker = typeof Worker !== "undefined";
  if (!hasWebWorker) reasons.push("Web Workers are not available in this browser.");

  const hasCacheApi = typeof caches !== "undefined";
  // Not fatal on its own -- we can still fetch the model without caching --
  // but worth surfacing.

  const { isIosSafari, version } = detectIosSafari(userAgent);
  if (isIosSafari && version !== null && version < 16.4) {
    reasons.push(
      `iOS Safari ${version} is not supported (WASM SIMD requires iOS Safari 16.4+). Please update iOS or use a different browser.`,
    );
  }

  return {
    supported: reasons.length === 0,
    reasons,
    details: {
      hasWasm,
      hasWasmSimd,
      hasWebWorker,
      hasCacheApi,
      isIosSafari,
      iosSafariVersion: version,
    },
  };
}
