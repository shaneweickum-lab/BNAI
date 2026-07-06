//! Inference-only forward pass: RMSNorm, RoPE, causal self-attention with a
//! per-layer KV-cache, SwiGLU FFN, embedding lookup / tied LM head, and the
//! ternary matmul itself. Mirrors `model/architecture.py`'s
//! `BNAILanguageModel` forward pass (see that file for the training-time
//! counterpart -- this module only ever runs inference against an already
//! ternary-quantized, packed `.bnai` file).

use crate::packed_format::{lut, BnaiFile, LayerMeta, Lut, ProjMeta};

/// Growing per-layer KV-cache. `k`/`v` store one flat `d_model`-length
/// vector per cached position (heads packed contiguously, matching the
/// projection output layout), so a new token is an O(1) `extend` rather
/// than recomputing anything for earlier positions.
#[derive(Default, Clone)]
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl KvCache {
    fn len(&self, d_model: usize) -> usize {
        self.k.len() / d_model
    }

    fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }
}

#[derive(Debug)]
pub enum ModelError {
    ContextFull,
    InvalidTokenId,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::ContextFull => write!(f, "KV-cache is full: context_len exhausted"),
            ModelError::InvalidTokenId => write!(f, "token id is out of vocab_size range"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Ternary matmul without weight-side multiplication: for each output
/// feature, walk the packed byte stream (5 ternary components per byte,
/// decoded via the 256-entry LUT) and accumulate/select/skip the matching
/// input activation. `out[o] = scale * sum_i select(weight(o,i), input[i])`.
///
/// The packed bytes are read directly out of the file's raw buffer (no
/// dense f32 weight matrix is ever materialized) -- this is the "ROM" the
/// architecture is named for.
pub fn ternary_matmul(input: &[f32], data: &[u8], proj: &ProjMeta, lut: &Lut) -> Vec<f32> {
    let out_features = proj.out_features as usize;
    let in_features = proj.in_features as usize;
    let mut output = vec![0f32; out_features];
    if out_features == 0 || in_features == 0 {
        return output;
    }
    let packed = &data[proj.packed_offset..proj.packed_offset + proj.packed_len];

    let mut byte_pos = 0usize;
    let mut sub = 0usize;
    let mut byte = packed[0];
    for o in 0..out_features {
        let mut acc = 0f32;
        let row = &lut[byte as usize];
        let mut components = row;
        for i in 0..in_features {
            match components[sub] {
                1 => acc += input[i],
                -1 => acc -= input[i],
                _ => {}
            }
            sub += 1;
            if sub == 5 {
                sub = 0;
                byte_pos += 1;
                if byte_pos < packed.len() {
                    byte = packed[byte_pos];
                }
                components = &lut[byte as usize];
            }
        }
        output[o] = acc * proj.scale;
    }
    output
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / d;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary position embedding, standard rotate-half formulation, applied
/// in-place to one head's `head_dim`-length slice of Q or K.
pub fn apply_rope_inplace(vec: &mut [f32], position: usize, theta: f32) {
    let head_dim = vec.len();
    let half = head_dim / 2;
    for j in 0..half {
        let inv_freq = 1.0 / theta.powf((2 * j) as f32 / head_dim as f32);
        let angle = position as f32 * inv_freq;
        let (s, c) = angle.sin_cos();
        let x1 = vec[j];
        let x2 = vec[j + half];
        vec[j] = x1 * c - x2 * s;
        vec[j + half] = x2 * c + x1 * s;
    }
}

fn forward_layer(
    x: &[f32],
    layer: &LayerMeta,
    file: &BnaiFile,
    cache: &mut KvCache,
    position: usize,
    lut: &Lut,
) -> Vec<f32> {
    let d_model = file.metadata.d_model as usize;
    let n_heads = file.metadata.n_heads as usize;
    let head_dim = file.metadata.head_dim as usize;
    let eps = file.metadata.rms_eps;
    let theta = file.metadata.rope_theta;

    let attn_norm_w = file.read_fp16_row(layer.attn_norm_offset, d_model);
    let normed = rms_norm(x, &attn_norm_w, eps);

    let mut q = ternary_matmul(&normed, &file.data, &layer.q, lut);
    let mut k = ternary_matmul(&normed, &file.data, &layer.k, lut);
    let v = ternary_matmul(&normed, &file.data, &layer.v, lut);

    for h in 0..n_heads {
        apply_rope_inplace(&mut q[h * head_dim..(h + 1) * head_dim], position, theta);
        apply_rope_inplace(&mut k[h * head_dim..(h + 1) * head_dim], position, theta);
    }

    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let cache_len = cache.len(d_model); // includes the position just appended

    let mut attn_out = vec![0f32; d_model];
    let inv_sqrt_head_dim = 1.0 / (head_dim as f32).sqrt();
    let mut scores = Vec::with_capacity(cache_len);
    for h in 0..n_heads {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        scores.clear();
        for pos in 0..cache_len {
            let base = pos * d_model + h * head_dim;
            let k_h = &cache.k[base..base + head_dim];
            let dot: f32 = q_h.iter().zip(k_h).map(|(a, b)| a * b).sum();
            scores.push(dot * inv_sqrt_head_dim);
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        for pos in 0..cache_len {
            let base = pos * d_model + h * head_dim;
            let v_h = &cache.v[base..base + head_dim];
            let weight = scores[pos];
            for d in 0..head_dim {
                attn_out[h * head_dim + d] += weight * v_h[d];
            }
        }
    }

    let o = ternary_matmul(&attn_out, &file.data, &layer.o, lut);
    let mut x1: Vec<f32> = x.iter().zip(o.iter()).map(|(&a, &b)| a + b).collect();

    let ffn_norm_w = file.read_fp16_row(layer.ffn_norm_offset, d_model);
    let normed2 = rms_norm(&x1, &ffn_norm_w, eps);
    let gate = ternary_matmul(&normed2, &file.data, &layer.gate, lut);
    let up = ternary_matmul(&normed2, &file.data, &layer.up, lut);
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();
    let down = ternary_matmul(&hidden, &file.data, &layer.down, lut);
    for i in 0..d_model {
        x1[i] += down[i];
    }
    x1
}

/// Owns the parsed `.bnai` file plus per-layer KV-caches and the current
/// sequence position. `feed_token` is the only state-mutating forward-pass
/// entry point: embed -> N transformer blocks -> final norm -> LM head.
pub struct Model {
    pub file: BnaiFile,
    caches: Vec<KvCache>,
    seq_len: usize,
}

impl Model {
    pub fn new(file: BnaiFile) -> Self {
        let n_layers = file.metadata.n_layers as usize;
        Model {
            file,
            caches: vec![KvCache::default(); n_layers],
            seq_len: 0,
        }
    }

    pub fn reset(&mut self) {
        for c in self.caches.iter_mut() {
            c.clear();
        }
        self.seq_len = 0;
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn remaining_context(&self) -> usize {
        (self.file.metadata.context_len as usize).saturating_sub(self.seq_len)
    }

    /// Embed `token_id`, run it through every transformer block (appending
    /// to each layer's KV-cache), and return the full vocab-sized logits
    /// vector from the tied LM head. O(1) in the number of *new* tokens:
    /// earlier positions' K/V are read from the cache, never recomputed.
    pub fn feed_token(&mut self, token_id: u32) -> Result<Vec<f32>, ModelError> {
        if self.remaining_context() == 0 {
            return Err(ModelError::ContextFull);
        }
        if token_id as usize >= self.file.metadata.vocab_size as usize {
            return Err(ModelError::InvalidTokenId);
        }
        let d_model = self.file.metadata.d_model as usize;
        let lut = lut();

        let mut x = self.file.embedding_row(token_id as usize);
        let position = self.seq_len;
        for (layer, cache) in self.file.layers.iter().zip(self.caches.iter_mut()) {
            x = forward_layer(&x, layer, &self.file, cache, position, lut);
        }

        let final_norm_w = self.file.read_fp16_row(self.file.final_norm_offset, d_model);
        let normed = rms_norm(&x, &final_norm_w, self.file.metadata.rms_eps);
        let logits = lm_head(&self.file, &normed);

        self.seq_len += 1;
        Ok(logits)
    }
}

/// Dense (non-quantized) matmul against the tied embedding table:
/// `logits[v] = sum_d x[d] * embedding[v, d]`. Small relative to the
/// ternary layers (see task notes) -- a plain float dot product per vocab
/// row, no LUT involved.
fn lm_head(file: &BnaiFile, x: &[f32]) -> Vec<f32> {
    let vocab = file.metadata.vocab_size as usize;
    let d = x.len();
    let mut logits = vec![0f32; vocab];
    for (v, logit) in logits.iter_mut().enumerate() {
        let row = file.embedding_row(v);
        let mut acc = 0f32;
        for i in 0..d {
            acc += x[i] * row[i];
        }
        *logit = acc;
    }
    logits
}

/// Deterministic, stateless temperature + top-k sampling from a logits
/// vector. `seed` drives a tiny built-in PRNG (splitmix-style) -- callers
/// (e.g. the JS Web Worker) should pass a fresh seed (e.g. derived from
/// `crypto.getRandomValues`) on each call to get non-repeating samples;
/// the same seed always yields the same draw, which is convenient for tests.
pub fn sample_from_logits(logits: &[f32], temperature: f32, top_k: u32, seed: u32) -> u32 {
    assert!(!logits.is_empty(), "sample_from_logits requires a non-empty logits vector");
    let t = if temperature <= 0.0 { 1e-6 } else { temperature };

    let mut scaled: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l / t)).collect();
    scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = if top_k == 0 || (top_k as usize) > scaled.len() {
        scaled.len()
    } else {
        top_k as usize
    };
    scaled.truncate(k);

    let max = scaled.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&(_, v)| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let r = next_rand_unit(seed);
    let mut cumulative = 0f32;
    for (i, &e) in exps.iter().enumerate() {
        cumulative += e / sum;
        if r <= cumulative {
            return scaled[i].0 as u32;
        }
    }
    scaled.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

/// splitmix32-style hash of `seed` mapped to `[0, 1)`.
fn next_rand_unit(seed: u32) -> f32 {
    let mut x = seed.wrapping_add(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    (x as f64 / u32::MAX as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed_format::lut;

    /// Hand-built 2x2 ternary projection with a known weight matrix:
    /// w = [[1, -1], [0, 1]] (row-major, out_features=2, in_features=2).
    /// Flat digits (ti = weight+1): [2,0,1,2] + 1 padding digit (=1, weight
    /// 0). byte = 2*1 + 0*3 + 1*9 + 2*27 + 1*81 = 146.
    #[test]
    fn ternary_matmul_hand_built_2x2() {
        let data = vec![146u8];
        let proj = ProjMeta {
            out_features: 2,
            in_features: 2,
            packed_offset: 0,
            packed_len: 1,
            scale: 1.0,
        };
        let input = [3.0f32, 4.0f32];
        let out = ternary_matmul(&input, &data, &proj, lut());
        // out[0] = select(1,3) + select(-1,4) = 3 - 4 = -1
        // out[1] = select(0,3) + select(1,4) = 0 + 4 = 4
        assert_eq!(out, vec![-1.0, 4.0]);
    }

    #[test]
    fn ternary_matmul_applies_scale() {
        let data = vec![146u8];
        let proj = ProjMeta {
            out_features: 2,
            in_features: 2,
            packed_offset: 0,
            packed_len: 1,
            scale: 0.5,
        };
        let input = [3.0f32, 4.0f32];
        let out = ternary_matmul(&input, &data, &proj, lut());
        assert_eq!(out, vec![-0.5, 2.0]);
    }

    #[test]
    fn ternary_matmul_crosses_byte_boundary_within_a_row() {
        // out_features=1, in_features=7: one row spans 2 packed bytes
        // (ceil(7/5)=2). weights = [1,1,1,1,1,-1,1] -> digits [2,2,2,2,2 |
        // 0,2,+pad1,1,1]. byte0 = 2+6+18+54+162=242. byte1 digits
        // [0,2,1,1,1] -> 0+6+9+27+81=123.
        let data = vec![242u8, 123u8];
        let proj = ProjMeta {
            out_features: 1,
            in_features: 7,
            packed_offset: 0,
            packed_len: 2,
            scale: 1.0,
        };
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let out = ternary_matmul(&input, &data, &proj, lut());
        // sum of all inputs except the 6th (-1 weight) flips sign: 1+2+3+4+5-6+7=16
        assert_eq!(out, vec![16.0]);
    }

    #[test]
    fn rms_norm_matches_manual_computation() {
        let x = [3.0f32, 4.0f32];
        let w = [1.0f32, 1.0f32];
        let eps = 1e-5;
        let out = rms_norm(&x, &w, eps);
        let mean_sq = (9.0 + 16.0) / 2.0;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        assert!((out[0] - 3.0 * inv_rms).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv_rms).abs() < 1e-6);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        apply_rope_inplace(&mut v, 0, 10000.0);
        // angle=0 for every freq at position 0 -> cos=1, sin=0 -> unchanged
        for (a, b) in v.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_vector_norm() {
        // RoPE is a rotation, so the L2 norm of each head must be invariant.
        let mut v = [1.0f32, -2.0, 0.5, 3.0, -1.5, 2.5, 0.25, -0.75];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_inplace(&mut v, 5, 10000.0);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn sample_from_logits_is_deterministic_for_a_given_seed() {
        let logits = vec![1.0, 5.0, 2.0, 0.1, 3.0];
        let a = sample_from_logits(&logits, 1.0, 0, 42);
        let b = sample_from_logits(&logits, 1.0, 0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_from_logits_top_1_is_argmax() {
        let logits = vec![1.0, 5.0, 2.0, 0.1, 3.0];
        for seed in 0..20 {
            let picked = sample_from_logits(&logits, 1.0, 1, seed);
            assert_eq!(picked, 1); // index of max logit (5.0)
        }
    }

    #[test]
    fn sample_from_logits_respects_top_k() {
        let logits = vec![1.0, 5.0, 2.0, 0.1, 3.0];
        // top_k=2 restricts to indices {1 (5.0), 4 (3.0)}
        for seed in 0..50 {
            let picked = sample_from_logits(&logits, 1.0, 2, seed);
            assert!(picked == 1 || picked == 4, "picked unexpected index {picked}");
        }
    }
}
