//! Parser + codec for the `.bnai` packed ternary model format.
//!
//! This is the Rust-side mirror of `model/export.py` (that file is the
//! authoritative source of truth for the byte layout -- read its module
//! docstring first if these two ever appear to disagree).
//!
//! File layout (little-endian throughout):
//! ```text
//! magic            4 bytes   b"BNAI"
//! version          u8
//! metadata_len     u32
//! metadata         `metadata_len` bytes of UTF-8 JSON (see `Metadata`)
//! embedding table  vocab_size * d_model fp16 values, row-major (tied with
//!                  the LM head; dense, never quantized)
//! per layer, repeated n_layers times, projections in this fixed order
//! [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]:
//!     attn_norm weight (d_model fp16 values, before the 4 attn projections)
//!     for each of q/k/v/o proj: out_features u32, in_features u32,
//!         packed ternary bytes (ceil(out*in/5) bytes), fp16 scale
//!     ffn_norm weight (d_model fp16 values, before the 3 ffn projections)
//!     for each of gate/up/down proj: same out/in/packed/scale layout
//! final_norm weight (once, at the very end, d_model fp16 values)
//! ```
//!
//! Ternary packing: 5 values per byte in base-3 -- `byte = t0 + 3*t1 + 9*t2 +
//! 27*t3 + 81*t4`, each `ti` in `{0,1,2}` mapping to weight `{-1,0,+1}` via
//! `ti = weight + 1`. This module keeps the packed bytes as a single owned
//! buffer and records `(offset, len)` for each projection/norm rather than
//! eagerly unpacking to a dense f32 matrix -- unpacking happens lazily,
//! per-element, during the matmul (see `model.rs::ternary_matmul`), which is
//! the entire point of the packed format (a 256-entry lookup table standing
//! in for a weight-side multiply -- the "ROM" in the project's name).

use serde::Deserialize;
use std::fmt;
use std::sync::LazyLock;

pub const MAGIC: &[u8; 4] = b"BNAI";

/// Architecture hyperparameters + bookkeeping stored in the file header.
/// Field names/types intentionally mirror the JSON dict built in
/// `export.py::export_model_to_bnai`.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub format_version: u32,
    pub pack_scheme: String,
    pub vocab_size: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub ffn_hidden: u32,
    pub context_len: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub param_count: u64,
    pub tokenizer_vocab_size: u32,
}

/// One packed ternary projection: its shape, where its packed bytes live in
/// the owning `BnaiFile`'s raw buffer, and its single dequantization scale.
#[derive(Debug, Clone, Copy)]
pub struct ProjMeta {
    pub out_features: u32,
    pub in_features: u32,
    pub packed_offset: usize,
    pub packed_len: usize,
    pub scale: f32,
}

/// Byte offsets/shapes for every tensor in one transformer block.
#[derive(Debug, Clone, Copy)]
pub struct LayerMeta {
    /// Byte offset of the attn_norm weight (d_model fp16 values).
    pub attn_norm_offset: usize,
    pub q: ProjMeta,
    pub k: ProjMeta,
    pub v: ProjMeta,
    pub o: ProjMeta,
    /// Byte offset of the ffn_norm weight (d_model fp16 values).
    pub ffn_norm_offset: usize,
    pub gate: ProjMeta,
    pub up: ProjMeta,
    pub down: ProjMeta,
}

/// A fully-parsed `.bnai` file: header/metadata plus byte offsets into the
/// still-packed raw buffer for every tensor. No ternary weight is ever
/// unpacked into a dense array here -- only fp16 norm/embedding rows are
/// decoded on demand (those are already dense/small in the file).
pub struct BnaiFile {
    pub data: Vec<u8>,
    pub metadata: Metadata,
    pub embedding_offset: usize,
    pub layers: Vec<LayerMeta>,
    pub final_norm_offset: usize,
}

// Hand-written (not derived) so debug-printing a `BnaiFile` never dumps its
// (potentially tens-of-MB) raw byte buffer.
impl fmt::Debug for BnaiFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BnaiFile")
            .field("data_len", &self.data.len())
            .field("metadata", &self.metadata)
            .field("embedding_offset", &self.embedding_offset)
            .field("layers", &self.layers.len())
            .field("final_norm_offset", &self.final_norm_offset)
            .finish()
    }
}

#[derive(Debug)]
pub enum ParseError {
    BadMagic,
    UnexpectedEof,
    Utf8(std::str::Utf8Error),
    Json(serde_json::Error),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::BadMagic => write!(f, "not a .bnai file (bad magic bytes)"),
            ParseError::UnexpectedEof => write!(f, "unexpected end of file while parsing .bnai"),
            ParseError::Utf8(e) => write!(f, "metadata is not valid UTF-8: {e}"),
            ParseError::Json(e) => write!(f, "failed to parse .bnai metadata JSON: {e}"),
        }
    }
}

impl std::error::Error for ParseError {}

fn require(cond: bool) -> Result<(), ParseError> {
    if cond {
        Ok(())
    } else {
        Err(ParseError::UnexpectedEof)
    }
}

fn read_proj(data: &[u8], pos: &mut usize) -> Result<ProjMeta, ParseError> {
    require(data.len() >= *pos + 8)?;
    let out_features = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    let in_features = u32::from_le_bytes(data[*pos + 4..*pos + 8].try_into().unwrap());
    *pos += 8;

    let n = out_features as usize * in_features as usize;
    let packed_len = n.div_ceil(5);
    require(data.len() >= *pos + packed_len)?;
    let packed_offset = *pos;
    *pos += packed_len;

    require(data.len() >= *pos + 2)?;
    let scale_bits = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    let scale = f16_to_f32(scale_bits);
    *pos += 2;

    Ok(ProjMeta {
        out_features,
        in_features,
        packed_offset,
        packed_len,
        scale,
    })
}

impl BnaiFile {
    pub fn parse(data: Vec<u8>) -> Result<Self, ParseError> {
        require(data.len() >= 4 + 1 + 4)?;
        if &data[0..4] != MAGIC {
            return Err(ParseError::BadMagic);
        }
        let mut pos = 4;
        let _version = data[pos];
        pos += 1;
        let meta_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        require(data.len() >= pos + meta_len)?;
        let meta_str = std::str::from_utf8(&data[pos..pos + meta_len]).map_err(ParseError::Utf8)?;
        let metadata: Metadata = serde_json::from_str(meta_str).map_err(ParseError::Json)?;
        pos += meta_len;

        let embedding_offset = pos;
        let embed_bytes = metadata.vocab_size as usize * metadata.d_model as usize * 2;
        require(data.len() >= pos + embed_bytes)?;
        pos += embed_bytes;

        let mut layers = Vec::with_capacity(metadata.n_layers as usize);
        for _ in 0..metadata.n_layers {
            let attn_norm_offset = pos;
            let norm_bytes = metadata.d_model as usize * 2;
            require(data.len() >= pos + norm_bytes)?;
            pos += norm_bytes;

            let q = read_proj(&data, &mut pos)?;
            let k = read_proj(&data, &mut pos)?;
            let v = read_proj(&data, &mut pos)?;
            let o = read_proj(&data, &mut pos)?;

            let ffn_norm_offset = pos;
            require(data.len() >= pos + norm_bytes)?;
            pos += norm_bytes;

            let gate = read_proj(&data, &mut pos)?;
            let up = read_proj(&data, &mut pos)?;
            let down = read_proj(&data, &mut pos)?;

            layers.push(LayerMeta {
                attn_norm_offset,
                q,
                k,
                v,
                o,
                ffn_norm_offset,
                gate,
                up,
                down,
            });
        }

        let final_norm_offset = pos;
        let norm_bytes = metadata.d_model as usize * 2;
        require(data.len() >= pos + norm_bytes)?;
        pos += norm_bytes;
        // Trailing bytes beyond `pos` (there shouldn't be any) are tolerated;
        // we only require the file to be at least this long.
        let _ = pos;

        Ok(BnaiFile {
            data,
            metadata,
            embedding_offset,
            layers,
            final_norm_offset,
        })
    }

    /// Decode `count` consecutive fp16 values starting at `byte_offset` into
    /// f32. Used for norm weights (small: `d_model` values) and single
    /// embedding-table rows -- never for a whole ternary projection.
    pub fn read_fp16_row(&self, byte_offset: usize, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = byte_offset + i * 2;
            let bits = u16::from_le_bytes([self.data[off], self.data[off + 1]]);
            out.push(f16_to_f32(bits));
        }
        out
    }

    /// One row of the dense embedding table, upcast to f32.
    pub fn embedding_row(&self, token_id: usize) -> Vec<f32> {
        let d = self.metadata.d_model as usize;
        self.read_fp16_row(self.embedding_offset + token_id * d * 2, d)
    }
}

/// 256-entry unpack lookup table: byte value -> its 5 decoded ternary
/// weights (each in `{-1,0,1}`), least-significant base-3 digit first.
/// Mirrors `export.py::_build_unpack_lut` exactly.
pub type Lut = [[i8; 5]; 256];

fn build_lut() -> Lut {
    let mut lut = [[0i8; 5]; 256];
    for byte in 0..256usize {
        let mut v = byte;
        for slot in lut[byte].iter_mut() {
            *slot = (v % 3) as i8 - 1;
            v /= 3;
        }
    }
    lut
}

static LUT: LazyLock<Lut> = LazyLock::new(build_lut);

pub fn lut() -> &'static Lut {
    &LUT
}

/// Pack a flat array of ternary weights (`{-1,0,1}`) 5-per-byte in base-3,
/// padding the tail with ternary-0 up to a multiple of 5. Mirrors
/// `export.py::pack_ternary` exactly.
pub fn pack_ternary(weights: &[i8]) -> Vec<u8> {
    let n = weights.len();
    let pad = (5 - n % 5) % 5;
    let powers: [u16; 5] = [1, 3, 9, 27, 81];

    let mut out = Vec::with_capacity((n + pad) / 5);
    let mut chunk = [1u16; 5]; // padding digit for ternary-0 is (0+1)=1
    let mut idx = 0usize;
    for &w in weights {
        chunk[idx % 5] = (w + 1) as u16;
        idx += 1;
        if idx.is_multiple_of(5) {
            out.push(chunk.iter().zip(powers.iter()).map(|(d, p)| d * p).sum::<u16>() as u8);
            chunk = [1u16; 5];
        }
    }
    if !idx.is_multiple_of(5) {
        // Remaining slots in `chunk` beyond `idx % 5` are already the
        // padding digit (1 == weight 0) from initialization/reset above.
        out.push(chunk.iter().zip(powers.iter()).map(|(d, p)| d * p).sum::<u16>() as u8);
    }
    out
}

/// Inverse of `pack_ternary`: returns the first `n` int8 values in
/// `{-1,0,1}`. Mirrors `export.py::unpack_ternary` exactly.
pub fn unpack_ternary(data: &[u8], n: usize) -> Vec<i8> {
    let lut = lut();
    let mut out = Vec::with_capacity(n);
    for &byte in data {
        for &comp in &lut[byte as usize] {
            if out.len() == n {
                return out;
            }
            out.push(comp);
        }
    }
    out
}

/// Decode a single ternary component of a packed projection without
/// unpacking the surrounding bytes: `idx` is the flat `out*in_features +
/// in` index into the (conceptual) dense weight matrix.
#[inline]
pub fn unpack_ternary_at(data: &[u8], idx: usize) -> i8 {
    let byte = data[idx / 5];
    lut()[byte as usize][idx % 5]
}

/// IEEE-754 binary16 -> binary32. Handles zero, subnormals, normals, inf and
/// NaN. Written by hand (see runtime task notes) rather than pulling in the
/// `half` crate, since the only fp16 data in this format is embedding rows,
/// norm weights and per-projection scales -- a handful of small, well-tested
/// conversions, not a hot path worth a dependency.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    let bits32: u32 = if exponent == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            // Subnormal half -> normalize into a binary32 normal number.
            let mut shift = 0u32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                shift += 1;
            }
            m &= 0x3ff;
            // Half subnormal value = mantissa * 2^-24. After normalizing
            // (m << shift) to have its leading 1 at bit 10, the effective
            // binary32 exponent field is 127 - 14 - shift = 113 - shift.
            let exp32 = 113 - shift;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exponent == 0x1f {
        // Inf or NaN: exponent all-ones, mantissa carried through.
        (sign << 31) | (0xffu32 << 23) | (mantissa << 13)
    } else {
        // Normal half: rebias exponent (half bias 15, float bias 127).
        let exp32 = exponent + (127 - 15);
        (sign << 31) | (exp32 << 23) | (mantissa << 13)
    };
    f32::from_bits(bits32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip_exact_multiple_of_5() {
        let w: [i8; 10] = [-1, 0, 1, 1, -1, 0, 0, 0, 1, -1];
        let packed = pack_ternary(&w);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary(&packed, w.len());
        assert_eq!(unpacked, w.to_vec());
    }

    #[test]
    fn pack_unpack_round_trip_needs_padding() {
        let w: [i8; 7] = [-1, 0, 1, 1, -1, 0, 1];
        let packed = pack_ternary(&w);
        assert_eq!(packed.len(), 2); // ceil(7/5) == 2
        let unpacked = unpack_ternary(&packed, w.len());
        assert_eq!(unpacked, w.to_vec());
    }

    #[test]
    fn pack_unpack_random_round_trip() {
        // Simple deterministic PRNG so this test has no extra dependency.
        let mut state: u64 = 0xC0FFEE;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 3) as i8) - 1
        };
        let w: Vec<i8> = (0..(37 * 41)).map(|_| next()).collect();
        let packed = pack_ternary(&w);
        assert_eq!(packed.len(), w.len().div_ceil(5));
        let unpacked = unpack_ternary(&packed, w.len());
        assert_eq!(unpacked, w);
    }

    #[test]
    fn pack_ternary_pads_with_zero_weight() {
        // 6 values -> 1 full byte + 1 padded byte where the padding digit
        // must decode back to weight 0 (ti=1), not garbage.
        let w: [i8; 6] = [1, 1, 1, 1, 1, -1];
        let packed = pack_ternary(&w);
        assert_eq!(packed.len(), 2);
        // second byte holds w[5]=-1 followed by 4 padding zeros
        let second_byte_decoded = lut()[packed[1] as usize];
        assert_eq!(second_byte_decoded, [-1, 0, 0, 0, 0]);
    }

    #[test]
    fn unpack_ternary_at_matches_bulk_unpack() {
        let w: [i8; 23] = [
            1, -1, 0, 0, 1, -1, -1, 1, 0, 1, 1, -1, 0, -1, 1, 0, 0, 1, -1, -1, 0, 1, 1,
        ];
        let packed = pack_ternary(&w);
        for (idx, &expected) in w.iter().enumerate() {
            assert_eq!(unpack_ternary_at(&packed, idx), expected, "idx={idx}");
        }
    }

    #[test]
    fn lut_matches_pack_formula() {
        // byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4, ti = weight+1
        let lut = lut();
        // All-zero-weight byte: every ti=1 -> byte = 1+3+9+27+81 = 121
        assert_eq!(lut[121], [0, 0, 0, 0, 0]);
        // All +1 weights: every ti=2 -> byte = 2+6+18+54+162 = 242
        assert_eq!(lut[242], [1, 1, 1, 1, 1]);
        // All -1 weights: every ti=0 -> byte = 0
        assert_eq!(lut[0], [-1, -1, -1, -1, -1]);
    }

    #[test]
    fn f16_to_f32_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert_eq!(f16_to_f32(0x8000), -0.0f32);
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
        assert_eq!(f16_to_f32(0xBC00), -1.0f32);
        assert_eq!(f16_to_f32(0x4000), 2.0f32);
        assert_eq!(f16_to_f32(0x3800), 0.5f32);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest positive subnormal: mantissa=1, exponent=0 -> 2^-24.
        let smallest = f16_to_f32(0x0001);
        assert!((smallest - 5.9604645e-8).abs() < 1e-12);
        // Largest subnormal: mantissa=0x3ff, exponent=0 -> 1023 * 2^-24.
        let largest_sub = f16_to_f32(0x03ff);
        assert!((largest_sub - (1023.0 * 2f32.powi(-24))).abs() < 1e-12);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let data = b"NOPE1234".to_vec();
        let err = BnaiFile::parse(data).unwrap_err();
        matches!(err, ParseError::BadMagic);
    }

    #[test]
    fn parse_rejects_truncated_file() {
        let data = MAGIC.to_vec(); // magic only, no version/metadata_len
        let err = BnaiFile::parse(data).unwrap_err();
        matches!(err, ParseError::UnexpectedEof);
    }
}
