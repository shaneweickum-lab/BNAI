/**
 * Shared, measured facts about the Benny (BNAI V1.0) placeholder model artifact.
 *
 * These numbers come from actually inspecting the checked-in artifacts in
 * `public/model/` (see `lib/bnaiHeader.ts` for the runtime header parser that
 * reads the param count directly out of the .bnai file), not from assumptions
 * about what ternary quantization "should" achieve. In particular the overall
 * compression ratio is ~3.1x, not the 5-8x you'd get if every parameter were
 * ternary-packed -- the tied input/output embedding table is kept at fp16 for
 * quality and is NOT ternary-compressed, only the attention/FFN projections are.
 */

export const MODEL_NAME = "Benny";
export const MODEL_SUBTITLE = "BNAI V1.0";

export const MODEL_FILE_PATH = "/model/benny-placeholder.bnai";
export const TOKENIZER_FILE_PATH = "/model/benny-placeholder.bnai.tokenizer.json";

// --- Architecture (from model/architecture.py BNAIConfig defaults) ---
export const ARCHITECTURE = {
  vocabSize: 32000,
  dModel: 768,
  nLayers: 14,
  nHeads: 12,
  headDim: 64,
  ffnHidden: 2048,
  contextLen: 2048,
  weightScheme: "ternary {-1, 0, +1} via BitLinear (BitNet b1.58-style)",
} as const;

// --- Measured artifact facts (public/model/benny-placeholder.bnai) ---
// Total parameter count, read straight out of the .bnai header at runtime
// (param_count field) -- repeated here as a static fallback for pages that
// render before the file has been fetched (e.g. the landing page).
export const TOTAL_PARAMS = 123_688_704;

// ~24.6M of the 123.7M total params are the tied embedding table, kept at
// fp16 (not ternary-compressed) to protect quality on such a small model.
export const EMBEDDING_PARAMS = 24_576_000;
export const TERNARY_PARAMS = TOTAL_PARAMS - EMBEDDING_PARAMS; // ~99.1M

// Measured on-disk file size of the packed .bnai artifact.
export const PACKED_FILE_SIZE_BYTES = 69_015_916; // ~69.02 MB (measured)
export const PACKED_FILE_SIZE_MB = PACKED_FILE_SIZE_BYTES / 1_000_000;

// fp16-baseline-equivalent size: every one of the 123.7M params stored as an
// IEEE754 half (2 bytes), with no ternary packing anywhere. Expressed in
// decimal MB (1 MB = 1,000,000 bytes), matching how the 69.02 MB packed
// file size above is conventionally reported, so the two numbers are
// directly comparable on the landing page.
export const FP16_BASELINE_SIZE_BYTES = TOTAL_PARAMS * 2; // ~247.4 MB
export const FP16_BASELINE_SIZE_MB = FP16_BASELINE_SIZE_BYTES / 1_000_000;

// Measured compression ratio: fp16 baseline / actual packed file size.
// This comes out to ~3.58x -- well short of the 5-8x someone might assume
// from "ternary = ~5x smaller than fp16", precisely because the embedding
// table isn't ternary at all.
export const MEASURED_COMPRESSION_RATIO =
  FP16_BASELINE_SIZE_BYTES / PACKED_FILE_SIZE_BYTES;

export const TRAINING_BUDGET_TOKENS = "~2.47B tokens (Chinchilla-optimal for this param count)";

export function formatParams(n: number): string {
  return `${(n / 1_000_000).toFixed(1)}M`;
}

// Decimal MB (1 MB = 1,000,000 bytes) -- matches how the measured 69.02 MB
// packed file size is reported, so this stays consistent everywhere it's
// used (landing page, demo download progress, demo model badge).
export function formatMB(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(2)} MB`;
}
