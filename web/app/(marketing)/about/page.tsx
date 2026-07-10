import type { Metadata } from "next";
import styles from "./about.module.css";
import {
  ARCHITECTURE,
  EMBEDDING_PARAMS,
  FP16_BASELINE_SIZE_MB,
  MEASURED_COMPRESSION_RATIO,
  MODEL_NAME,
  PACKED_FILE_SIZE_MB,
  TERNARY_PARAMS,
  TOTAL_PARAMS,
  TRAINING_BUDGET_TOKENS,
  formatParams,
} from "../../../lib/modelInfo";

const REPO_URL = "https://github.com/shaneweickum-lab/bnai";

export const metadata: Metadata = {
  title: `About — ${MODEL_NAME}`,
  description: "Dataset, training recipe, architecture, and evaluation write-up for Benny (BNAI V1.0).",
};

export default function AboutPage() {
  return (
    <div>
      <section className={styles.hero}>
        <h1 className={styles.title}>About {MODEL_NAME}</h1>
        <p className={styles.subtitle}>
          The dataset, training recipe, and architecture behind {MODEL_NAME} (BNAI V1.0), plus where to
          find the evaluation numbers once the training run has actually happened.
        </p>
      </section>

      <section className={styles.section}>
        <h2>Dataset</h2>
        <p>
          <strong>Base pretraining:</strong> a subset of{" "}
          <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu" target="_blank" rel="noopener noreferrer">
            FineWeb-Edu
          </a>{" "}
          totaling roughly 1.5&ndash;2B tokens, chosen for education-filtered web text quality relative
          to raw Common Crawl at this token budget.
        </p>
        <p>
          <strong>Supervised fine-tuning (SFT):</strong> subsets of{" "}
          <a href="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k" target="_blank" rel="noopener noreferrer">
            UltraChat-200k
          </a>{" "}
          and{" "}
          <a href="https://huggingface.co/datasets/OpenAssistant/oasst2" target="_blank" rel="noopener noreferrer">
            OASST2
          </a>
          , used to teach the base model the <code>&lt;|user|&gt;</code> / <code>&lt;|assistant|&gt;</code>{" "}
          chat turn format used by the tokenizer and the demo.
        </p>
      </section>

      <section className={styles.section}>
        <h2>Architecture &amp; training budget</h2>
        <ul>
          <li>Decoder-only transformer, {ARCHITECTURE.nLayers} layers, {ARCHITECTURE.dModel}-dim, {ARCHITECTURE.nHeads} heads ({ARCHITECTURE.headDim}-dim per head)</li>
          <li>FFN hidden size {ARCHITECTURE.ffnHidden}, context length {ARCHITECTURE.contextLen} tokens</li>
          <li>Attention/FFN projections use <code>BitLinear</code>: {ARCHITECTURE.weightScheme}</li>
          <li>Tied input/output embedding table (&asymp;{formatParams(EMBEDDING_PARAMS)} of {formatParams(TOTAL_PARAMS)} total params) kept at fp16</li>
          <li>Total parameters: {formatParams(TOTAL_PARAMS)} ({TOTAL_PARAMS.toLocaleString()} exactly)</li>
          <li>Target training budget: {TRAINING_BUDGET_TOKENS}</li>
        </ul>
        <p>
          The packed <code>.bnai</code> artifact currently measures {PACKED_FILE_SIZE_MB.toFixed(2)} MB,
          versus an fp16-equivalent baseline of {FP16_BASELINE_SIZE_MB.toFixed(1)} MB &mdash; a measured{" "}
          {MEASURED_COMPRESSION_RATIO.toFixed(1)}x reduction (see the landing page for why that’s lower
          than a naive “everything is ternary” estimate: the {formatParams(EMBEDDING_PARAMS)}-param
          embedding table stays fp16, only the {formatParams(TERNARY_PARAMS)} attention/FFN params are
          ternary-packed).
        </p>
      </section>

      <section className={styles.section}>
        <h2>
          Evaluation<span className={styles.tbd}>TBD</span>
        </h2>
        <p>
          The Chinchilla-optimal pretraining run for this model has not happened yet &mdash; it runs on
          separate training hardware, outside this repo’s CI, and the artifact currently shipped in the
          demo (<code>benny-placeholder.bnai</code>) has random, untrained weights. The sections below
          are placeholders for real numbers once that run completes; nothing here is invented.
        </p>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Ternary (BitNet b1.58)</th>
              <th>fp16 baseline</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Validation perplexity (FineWeb-Edu held-out)</td>
              <td>TBD &mdash; pending training run</td>
              <td>TBD &mdash; pending training run</td>
            </tr>
            <tr>
              <td>SFT eval / win-rate vs. baseline</td>
              <td>TBD &mdash; pending SFT run</td>
              <td>TBD &mdash; pending SFT run</td>
            </tr>
            <tr>
              <td>Packed size</td>
              <td>{PACKED_FILE_SIZE_MB.toFixed(2)} MB (measured)</td>
              <td>{FP16_BASELINE_SIZE_MB.toFixed(1)} MB (measured-equivalent)</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className={styles.section}>
        <h2>Source</h2>
        <p>
          All training code (<code>model/</code>), the Rust WASM inference runtime (<code>runtime/</code>),
          and this web app (<code>web/</code>) live in one repository.
        </p>
        <p>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            {REPO_URL}
          </a>
        </p>
      </section>
    </div>
  );
}
