//! From-scratch byte-level BPE tokenizer, mirroring `model/tokenizer/bpe.py`
//! (`BPETokenizer`) test-for-test: same byte<->unicode mapping, same
//! pretokenization behavior, same greedy-lowest-merge-rank encode loop, same
//! special-token id assignment. No external tokenizer crate.
//!
//! One deliberate deviation from the Python source: the GPT-2 split regex
//! there uses a negative lookahead (`\s+(?!\S)`) to decide how a run of
//! whitespace attaches to the following word. Rust's `regex` crate is a
//! linear-time (non-backtracking) engine and does not support lookaround at
//! all, so instead of pulling in `fancy-regex` (a whole extra dependency for
//! one feature), whitespace runs are split by hand in `pretokenize` --
//! reasoning through the backtracking semantics of `\s+(?!\S)` shows it
//! always resolves to "consume the whole run, except leave the very last
//! whitespace character for the next token" unless the run reaches end of
//! string (in which case the whole run is consumed). See `pretokenize`'s
//! doc comment for the derivation. The `regex` crate is still used for the
//! content-token alternatives (`\p{L}+`/`\p{N}+`/etc), which need no
//! lookaround.

use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

pub const SPECIAL_TOKENS: [&str; 8] = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
];
pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const SYSTEM_ID: u32 = 4;
pub const USER_ID: u32 = 5;
pub const ASSISTANT_ID: u32 = 6;
pub const TURN_END_ID: u32 = 7;

/// GPT-2's reversible byte -> printable-unicode-char mapping. Printable
/// ASCII/Latin-1 ranges map to themselves; every other byte value (control
/// characters, whitespace, etc) is assigned the next available codepoint
/// starting at 256. Mirrors `bpe.py::_bytes_to_unicode` exactly, including
/// iteration order (which determines *which* byte gets which of the 256+n
/// codepoints).
fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);

    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut encoder = HashMap::with_capacity(256);
    let mut decoder = HashMap::with_capacity(256);
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("byte-to-unicode codepoints are always valid");
        encoder.insert(b as u8, ch);
        decoder.insert(ch, b as u8);
    }
    (encoder, decoder)
}

static BYTE_ENCODER: LazyLock<HashMap<u8, char>> = LazyLock::new(|| bytes_to_unicode().0);
static BYTE_DECODER: LazyLock<HashMap<char, u8>> = LazyLock::new(|| bytes_to_unicode().1);

/// The lookaround-free subset of GPT-2's split pattern: apostrophe
/// contractions, then " ?<letters>", " ?<numbers>", " ?<other non-space>".
/// Anchored at the start of the haystack so callers can tell "no content
/// match here" apart from "matched later in the string".
static CONTENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+)")
        .expect("static regex is valid")
});

/// Pretokenize `text` into GPT-2-style word-ish chunks, replicating
/// `bpe.py::_pretokenize` (the `regex`-backed branch, which is what
/// `train_tokenizer.py`/this fixture's tokenizer were built with).
///
/// Derivation of the whitespace-run rule (replacing `\s+(?!\S)`): by the
/// time a whitespace run is reached, either (a) it is followed immediately
/// by a non-whitespace char -- but then the earlier ` ?<class>+`
/// alternatives already consumed exactly one leading space plus that
/// content, so we never actually see this case reach whitespace-handling en
/// masse for a *single* space -- or (b) the run has length >= 2, or (c) the
/// run reaches end-of-string. Backtracking `\s+(?!\S)` against a run of N
/// whitespace chars followed by a non-whitespace char always settles on
/// consuming N-1 chars (leaving exactly one trailing whitespace char to
/// prefix the next word); against a run that reaches end-of-string, it
/// consumes the whole run.
pub fn pretokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let rest = &text[pos..];
        if let Some(m) = CONTENT_RE.find(rest) {
            if m.start() == 0 {
                tokens.push(m.as_str().to_string());
                pos += m.end();
                continue;
            }
        }

        // Not a content chunk: `rest` must start with whitespace (the three
        // content classes above partition every non-whitespace character).
        let mut run_end = 0usize;
        let mut run_chars = 0usize;
        for (idx, ch) in rest.char_indices() {
            if ch.is_whitespace() {
                run_end = idx + ch.len_utf8();
                run_chars += 1;
            } else {
                break;
            }
        }
        if run_chars == 0 {
            // Defensive fallback (should not happen given class coverage):
            // consume one character verbatim to guarantee progress.
            let ch = rest.chars().next().unwrap();
            tokens.push(ch.to_string());
            pos += ch.len_utf8();
            continue;
        }

        if run_end == rest.len() {
            // Run reaches end of string: consume it all.
            tokens.push(rest[..run_end].to_string());
            pos += run_end;
        } else if run_chars == 1 {
            // A lone whitespace char followed by more text that the
            // content alternatives couldn't attach to (only possible if
            // that next char is itself whitespace, i.e. run_chars would
            // have been >1 -- kept only as a safety net).
            tokens.push(rest[..run_end].to_string());
            pos += run_end;
        } else {
            let last_char = rest[..run_end].chars().last().unwrap();
            let truncated_end = run_end - last_char.len_utf8();
            tokens.push(rest[..truncated_end].to_string());
            pos += truncated_end;
        }
    }
    tokens
}

fn word_to_symbols(word: &str) -> Vec<String> {
    word.bytes().map(|b| BYTE_ENCODER[&b].to_string()).collect()
}

#[derive(Deserialize)]
struct TokenizerFile {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    #[allow(dead_code)]
    special_tokens: Vec<String>,
}

/// A loaded BPE tokenizer: vocab (token string -> id), its inverse, and the
/// merge-rank table used to greedily apply merges in trained-priority order.
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merge_rank: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        let file: TokenizerFile = serde_json::from_str(json_str)?;
        let merge_rank = file
            .merges
            .iter()
            .enumerate()
            .map(|(i, pair)| (pair.clone(), i))
            .collect();
        let id_to_token = file.vocab.iter().map(|(tok, &id)| (id, tok.clone())).collect();
        Ok(Self {
            vocab: file.vocab,
            id_to_token,
            merge_rank,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Greedily merge adjacent symbol pairs in lowest-merge-rank-first
    /// order until no known merge applies. Mirrors
    /// `bpe.py::BPETokenizer._bpe_word`, including its tie-break (lowest
    /// rank, then lowest position -- which is what `min()` over `(rank,
    /// index)` tuples gives, and what "keep the first-seen index on a rank
    /// tie" gives here too).
    fn bpe_word(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            if symbols.len() <= 1 {
                break;
            }
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.merge_rank.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, idx)) => {
                    let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                    symbols.splice(idx..idx + 2, [merged]);
                }
            }
        }
        symbols
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(BOS_ID);
        }
        for word in pretokenize(text) {
            if word.is_empty() {
                continue;
            }
            let symbols = self.bpe_word(word_to_symbols(&word));
            for sym in symbols {
                ids.push(*self.vocab.get(&sym).unwrap_or(&UNK_ID));
            }
        }
        if add_eos {
            ids.push(EOS_ID);
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(&id) {
                if SPECIAL_TOKENS.contains(&tok.as_str()) {
                    continue;
                }
                text.push_str(tok);
            }
        }
        let bytes: Vec<u8> = text.chars().filter_map(|c| BYTE_DECODER.get(&c).copied()).collect();
        String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> BpeTokenizer {
        // vocab: specials + all 256 byte symbols (no merges) -- enough to
        // exercise byte-level fallback without needing a trained merge list.
        let mut vocab = HashMap::new();
        for (i, tok) in SPECIAL_TOKENS.iter().enumerate() {
            vocab.insert(tok.to_string(), i as u32);
        }
        let encoder = bytes_to_unicode().0;
        let mut next_id = SPECIAL_TOKENS.len() as u32;
        let mut bytes_sorted: Vec<u8> = (0..=255).collect();
        bytes_sorted.sort_by_key(|&b| encoder[&b] as u32); // stable arbitrary order
        for b in bytes_sorted {
            let ch = encoder[&b].to_string();
            vocab.entry(ch).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        }
        BpeTokenizer {
            vocab: vocab.clone(),
            id_to_token: vocab.into_iter().map(|(t, i)| (i, t)).collect(),
            merge_rank: HashMap::new(),
        }
    }

    #[test]
    fn pretokenize_basic_sentence() {
        let toks = pretokenize("Hello, world!");
        assert_eq!(toks, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn pretokenize_multiple_leading_spaces_attach_last_space_to_word() {
        let toks = pretokenize("a   b");
        // "a", then 2 bare spaces, then " b" (space attaches to b)
        assert_eq!(toks, vec!["a", "  ", " b"]);
    }

    #[test]
    fn pretokenize_trailing_whitespace_is_one_token() {
        let toks = pretokenize("hi   ");
        assert_eq!(toks, vec!["hi", "   "]);
    }

    #[test]
    fn byte_roundtrip_ascii() {
        let tok = tiny_tokenizer();
        let text = "Hello, world! 123";
        let ids = tok.encode(text, false, false);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn byte_roundtrip_non_ascii() {
        let tok = tiny_tokenizer();
        let text = "café \u{1F980} \u{4F60}\u{597D} na\u{00EF}ve r\u{00E9}sum\u{00E9}";
        let ids = tok.encode(text, false, false);
        assert!(ids.iter().all(|&id| id != UNK_ID), "byte-level fallback should avoid <unk> entirely");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn special_tokens_have_fixed_ids() {
        assert_eq!(PAD_ID, 0);
        assert_eq!(UNK_ID, 1);
        assert_eq!(BOS_ID, 2);
        assert_eq!(EOS_ID, 3);
        assert_eq!(SYSTEM_ID, 4);
        assert_eq!(USER_ID, 5);
        assert_eq!(ASSISTANT_ID, 6);
        assert_eq!(TURN_END_ID, 7);
    }

    #[test]
    fn decode_skips_special_tokens() {
        let tok = tiny_tokenizer();
        let mut ids = vec![BOS_ID];
        ids.extend(tok.encode("hi", false, false));
        ids.push(EOS_ID);
        assert_eq!(tok.decode(&ids), "hi");
    }
}
