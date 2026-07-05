//! Integration tests against the real (random-weight) placeholder fixture
//! at `tests/fixtures/benny-placeholder.bnai`. These exercise the full
//! parse -> Session -> forward pass pipeline; since the weights are
//! untrained/random, assertions are limited to structural/numerical
//! sanity (shapes, no NaN/Inf, context bookkeeping) -- never output
//! *quality*.

use bnai_runtime::model::Model;
use bnai_runtime::packed_format::BnaiFile;
use bnai_runtime::tokenizer::BpeTokenizer;

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/benny-placeholder.bnai")
}

fn tokenizer_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/benny-placeholder.bnai.tokenizer.json")
}

fn load_fixture_bytes() -> Vec<u8> {
    std::fs::read(fixture_path()).expect("fixture .bnai file must exist")
}

#[test]
fn fixture_header_and_metadata_match_expected_spec() {
    let data = load_fixture_bytes();
    let file = BnaiFile::parse(data).expect("fixture must parse as a valid .bnai file");

    assert_eq!(file.metadata.vocab_size, 32000);
    assert_eq!(file.metadata.d_model, 576);
    assert_eq!(file.metadata.n_layers, 14);
    assert_eq!(file.metadata.n_heads, 9);
    assert_eq!(file.metadata.head_dim, 64);
    assert_eq!(file.metadata.ffn_hidden, 1536);
    assert_eq!(file.metadata.context_len, 2048);
    assert_eq!(file.metadata.pack_scheme, "base3_5_per_byte");

    let expected_params = 74_187_072u64;
    let diff = (file.metadata.param_count as i64 - expected_params as i64).unsigned_abs();
    assert!(
        diff < expected_params / 100, // within 1%
        "param_count {} too far from expected {}",
        file.metadata.param_count,
        expected_params
    );

    assert_eq!(file.layers.len(), 14);
    for layer in &file.layers {
        assert_eq!(layer.q.out_features, 576);
        assert_eq!(layer.q.in_features, 576);
        assert_eq!(layer.gate.out_features, 1536);
        assert_eq!(layer.gate.in_features, 576);
        assert_eq!(layer.down.out_features, 576);
        assert_eq!(layer.down.in_features, 1536);
    }
}

#[test]
fn fixture_parses_full_file_length_exactly() {
    let data = load_fixture_bytes();
    let file_len = data.len();
    let file = BnaiFile::parse(data).expect("fixture must parse");
    // final_norm_offset + d_model*2 bytes should land exactly at EOF: the
    // packed layout leaves no slack, so a mis-parse anywhere upstream
    // would show up here as a mismatch.
    let end = file.final_norm_offset + file.metadata.d_model as usize * 2;
    assert_eq!(end, file_len, "parsed layout doesn't consume the whole file");
}

#[test]
fn session_forward_pass_produces_sane_logits_and_advances_context() {
    let data = load_fixture_bytes();
    let file = BnaiFile::parse(data).expect("fixture must parse");
    let vocab_size = file.metadata.vocab_size as usize;
    let context_len = file.metadata.context_len as usize;
    let mut model = Model::new(file);

    assert_eq!(model.remaining_context(), context_len);

    // A short pseudo-random sequence of token ids within vocab range.
    let prompt_ids: [u32; 5] = [42, 1000, 7, 31999, 12345];

    for (i, &tok) in prompt_ids.iter().enumerate() {
        let logits = model.feed_token(tok).expect("feed_token should succeed within context");
        assert_eq!(logits.len(), vocab_size, "logits must cover the full vocab");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must contain no NaN/Inf (step {i}, token {tok})"
        );
        assert_eq!(model.remaining_context(), context_len - (i + 1));
        assert_eq!(model.seq_len(), i + 1);
    }

    // Sampling from the final logits should yield a valid in-vocab id.
    let last_logits = model.feed_token(prompt_ids[0]).unwrap();
    let sampled = bnai_runtime::model::sample_from_logits(&last_logits, 0.8, 40, 1234);
    assert!((sampled as usize) < vocab_size);

    let greedy = bnai_runtime::model::sample_from_logits(&last_logits, 1.0, 1, 0);
    let argmax = last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(greedy, argmax, "top_k=1 must be exactly argmax");
}

#[test]
fn session_reset_clears_context() {
    let data = load_fixture_bytes();
    let file = BnaiFile::parse(data).expect("fixture must parse");
    let context_len = file.metadata.context_len as usize;
    let mut model = Model::new(file);

    model.feed_token(1).unwrap();
    model.feed_token(2).unwrap();
    assert_eq!(model.seq_len(), 2);

    model.reset();
    assert_eq!(model.seq_len(), 0);
    assert_eq!(model.remaining_context(), context_len);

    // After reset, feeding should behave exactly like a fresh model (no
    // leftover KV-cache entries skewing attention).
    let logits = model.feed_token(1).unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn feed_token_rejects_out_of_vocab_id() {
    let data = load_fixture_bytes();
    let file = BnaiFile::parse(data).expect("fixture must parse");
    let vocab_size = file.metadata.vocab_size;
    let mut model = Model::new(file);
    let err = model.feed_token(vocab_size + 1000);
    assert!(err.is_err());
}

#[test]
fn feed_token_rejects_once_context_is_full() {
    let data = load_fixture_bytes();
    let mut file = BnaiFile::parse(data).expect("fixture must parse");
    // Shrink context_len so the test doesn't need to run 2048 forward
    // passes to observe the ceiling.
    file.metadata.context_len = 3;
    let mut model = Model::new(file);

    for _ in 0..3 {
        model.feed_token(1).expect("should succeed under the cap");
    }
    assert_eq!(model.remaining_context(), 0);
    let err = model.feed_token(1);
    assert!(err.is_err(), "feeding beyond context_len must error");
}

#[test]
fn tokenizer_round_trip_ascii_from_fixture_json() {
    let json = std::fs::read_to_string(tokenizer_path()).expect("tokenizer fixture must exist");
    let tok = BpeTokenizer::from_json(&json).expect("tokenizer JSON must parse");

    let text = "Hello, world! This is Benny.";
    let ids = tok.encode(text, false, false);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn tokenizer_round_trip_non_ascii_from_fixture_json() {
    let json = std::fs::read_to_string(tokenizer_path()).expect("tokenizer fixture must exist");
    let tok = BpeTokenizer::from_json(&json).expect("tokenizer JSON must parse");

    let text = "caf\u{e9} \u{1f980} \u{4f60}\u{597d} na\u{ef}ve r\u{e9}sum\u{e9}";
    let ids = tok.encode(text, false, false);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}
