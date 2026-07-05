/**
 * Parser for the .bnai packed-model file header.
 *
 * File layout (observed in public/model/benny-placeholder.bnai, written by
 * model/export.py on the training side):
 *
 *   bytes [0..4)   "BNAI" magic
 *   byte  [4]      format version (u8)
 *   bytes [5..9)   header length, u32 little-endian
 *   bytes [9..9+N) UTF-8 JSON metadata header, N = header length above
 *   bytes [9+N..)  packed weight payload
 *
 * This is real parsing of real bytes -- not mocked -- since the header
 * format is simple and stable regardless of which inference engine
 * (mock JS or the real Rust/WASM runtime) ends up reading the payload.
 */

export interface BnaiHeader {
  format_version: number;
  pack_scheme: string;
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  head_dim: number;
  ffn_hidden: number;
  context_len: number;
  rope_theta: number;
  rms_eps: number;
  param_count: number;
  tokenizer_vocab_size: number;
  [key: string]: unknown;
}

export interface ParsedBnaiFile {
  header: BnaiHeader;
  headerByteLength: number;
  payloadOffset: number;
  payloadByteLength: number;
  totalFileByteLength: number;
}

const MAGIC = "BNAI";

export function parseBnaiHeader(buffer: ArrayBuffer): ParsedBnaiFile {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== MAGIC) {
    throw new Error(`Invalid .bnai file: expected magic "BNAI", got "${magic}"`);
  }

  const view = new DataView(buffer);
  const formatVersion = bytes[4];
  const headerLength = view.getUint32(5, true /* little-endian */);

  const jsonStart = 9;
  const jsonBytes = bytes.subarray(jsonStart, jsonStart + headerLength);
  const jsonText = new TextDecoder("utf-8").decode(jsonBytes);
  const header = JSON.parse(jsonText) as BnaiHeader;
  header.format_version = header.format_version ?? formatVersion;

  const payloadOffset = jsonStart + headerLength;
  const payloadByteLength = buffer.byteLength - payloadOffset;

  return {
    header,
    headerByteLength: headerLength,
    payloadOffset,
    payloadByteLength,
    totalFileByteLength: buffer.byteLength,
  };
}
