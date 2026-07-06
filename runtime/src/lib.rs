//! BNAI ("Benny") runtime: ternary-weight transformer inference compiled to
//! WASM for a browser Web Worker.
//!
//! The public surface is deliberately small -- `Session` is the only
//! `#[wasm_bindgen]` type a JS caller needs:
//!
//! ```text
//! const session = new Session(modelBytes, tokenizerJson);
//! const ids = session.encode("hello");
//! let logits = session.feed_token(ids[0]);
//! // ... feed remaining prompt tokens ...
//! const nextId = session.sample(logits, /*temperature*/ 0.8, /*top_k*/ 40, /*seed*/ Math.random() * 2**32);
//! logits = session.feed_token(nextId);
//! session.decode([nextId]);
//! session.reset(); // clears the KV-cache to start a new conversation
//! ```
//!
//! Internals live in three modules, each independently unit-tested:
//! - `packed_format`: parses the `.bnai` container and the 5-per-byte
//!   ternary packing scheme (no dense weight matrix is ever materialized).
//! - `model`: RoPE, RMSNorm, causal attention with a KV-cache, SwiGLU FFN,
//!   the ternary (multiplication-free) matmul, and sampling.
//! - `tokenizer`: from-scratch byte-level BPE, matching
//!   `model/tokenizer/bpe.py` exactly.

pub mod model;
pub mod packed_format;
pub mod tokenizer;

use wasm_bindgen::prelude::*;

/// A loaded model + tokenizer, with its own KV-cache and sequence position.
/// Constructed once per conversation (or call `reset()` to start a new one
/// without re-parsing the `.bnai` file).
#[wasm_bindgen]
pub struct Session {
    model: model::Model,
    tokenizer: tokenizer::BpeTokenizer,
}

#[wasm_bindgen]
impl Session {
    /// Parse a `.bnai` file (as raw bytes, e.g. from `fetch().arrayBuffer()`)
    /// and its sidecar tokenizer JSON string. Fails if the magic bytes,
    /// metadata JSON, or any tensor's declared shape doesn't fit inside the
    /// provided bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, tokenizer_json: &str) -> Result<Session, JsValue> {
        let file = packed_format::BnaiFile::parse(model_bytes)
            .map_err(|e| JsValue::from_str(&format!("failed to parse .bnai file: {e}")))?;
        let tokenizer = tokenizer::BpeTokenizer::from_json(tokenizer_json)
            .map_err(|e| JsValue::from_str(&format!("failed to parse tokenizer JSON: {e}")))?;
        Ok(Session {
            model: model::Model::new(file),
            tokenizer,
        })
    }

    /// Encode text into token ids (no BOS/EOS added -- callers that need
    /// turn markers should encode `<|user|>`-style special tokens
    /// explicitly via their fixed ids, see `bos_id()`/etc below, or rely on
    /// the chat-template convention the web app builds on top of this).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, false, false)
    }

    /// Decode token ids back to text. Special tokens are dropped (matching
    /// the Python tokenizer's `decode`).
    pub fn decode(&self, ids: Vec<u32>) -> String {
        self.tokenizer.decode(&ids)
    }

    /// Feed one token through the full forward pass, appending to every
    /// layer's KV-cache, and return the vocab-sized logits vector for the
    /// *next* token. Errors if the context window is already full (see
    /// `remaining_context()`) or if `token_id` is outside the vocabulary.
    pub fn feed_token(&mut self, token_id: u32) -> Result<Vec<f32>, JsValue> {
        self.model
            .feed_token(token_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Sample a token id from a logits vector (as returned by
    /// `feed_token`) using temperature scaling and top-k filtering.
    /// `top_k = 0` disables the top-k filter (samples from the full
    /// distribution). Pass a fresh `seed` each call (e.g. derived from
    /// `crypto.getRandomValues`) for non-repeating generation; the same
    /// seed always reproduces the same draw.
    pub fn sample(&self, logits: Vec<f32>, temperature: f32, top_k: u32, seed: u32) -> u32 {
        model::sample_from_logits(&logits, temperature, top_k, seed)
    }

    /// Clear every layer's KV-cache and reset the sequence position to 0,
    /// to start a fresh conversation without re-parsing the model file.
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Tokens of context left before the KV-cache is full (`context_len -
    /// current sequence length`). The web app uses this to warn before
    /// hitting mobile Safari's tighter memory ceiling.
    pub fn remaining_context(&self) -> u32 {
        self.model.remaining_context() as u32
    }

    /// Number of tokens fed so far (i.e. current KV-cache length).
    pub fn seq_len(&self) -> u32 {
        self.model.seq_len() as u32
    }

    /// Model architecture's max context length (from file metadata).
    pub fn context_len(&self) -> u32 {
        self.model.file.metadata.context_len
    }

    /// Embedding/LM-head vocabulary size (from file metadata; this is the
    /// architecture's vocab_size, which may exceed the tokenizer's actual
    /// trained vocab -- see `model/export.py`'s placeholder-export note).
    pub fn vocab_size(&self) -> u32 {
        self.model.file.metadata.vocab_size
    }

    /// Total ternary+dense parameter count, for the UI's model-size badge.
    /// Returned as f64 (not u64/BigInt) since JS numbers are f64 anyway.
    pub fn param_count(&self) -> f64 {
        self.model.file.metadata.param_count as f64
    }

    /// Size in bytes of the raw `.bnai` file this session was built from,
    /// for the UI's model-size badge.
    pub fn file_size_bytes(&self) -> f64 {
        self.model.file.data.len() as f64
    }
}
