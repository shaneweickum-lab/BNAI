"use client";

/**
 * Right-drawer content: the engineering showcase. Organized into four
 * sections -- model facts (from lib/modelInfo.ts, never re-derived here),
 * live session stats for the active conversation, a "100% client-side"
 * callout, and the energy-comparison panel (adapted from
 * prototypes/context-folding/index.html's energy-comparison section).
 */

import {
  ARCHITECTURE,
  FP16_BASELINE_SIZE_MB,
  MEASURED_COMPRESSION_RATIO,
  PACKED_FILE_SIZE_MB,
  TOTAL_PARAMS,
  TRAINING_BUDGET_TOKENS,
  formatMB,
  formatParams,
} from "../../lib/modelInfo";
import type { ModelStats } from "../../lib/useInferenceWorker";
import type { StoredConversation } from "../../lib/store/types";
import {
  BENNY_WATTS_ASSUMPTION,
  CLOUD_ESTIMATE_DISCLAIMER,
  CLOUD_REFERENCE_ROWS,
  computeBennyWh,
  computeDeltaPct,
} from "../../lib/chat/energy";
import styles from "./MetricsPanel.module.css";

interface MetricsPanelProps {
  modelStats: ModelStats | null;
  activeConversation: StoredConversation | null;
  tokensPerSecond: number;
  isGenerating: boolean;
  lastElapsedMs: number | null;
}

export default function MetricsPanel({
  modelStats,
  activeConversation,
  tokensPerSecond,
  isGenerating,
  lastElapsedMs,
}: MetricsPanelProps) {
  const paramCount = modelStats?.paramCount ?? TOTAL_PARAMS;
  const packedSizeLabel = modelStats ? formatMB(modelStats.fileSizeBytes) : `${PACKED_FILE_SIZE_MB.toFixed(2)} MB`;

  const routingStats = activeConversation?.routingStats;
  const totalTurns = routingStats ? routingStats.deterministic + routingStats.noMatch + routingStats.ambiguous : 0;

  const bennyWh = lastElapsedMs != null ? computeBennyWh(lastElapsedMs) : null;
  const energyRows =
    bennyWh != null
      ? [
          { label: "Benny (this reply)", wh: bennyWh },
          ...CLOUD_REFERENCE_ROWS,
        ]
      : [];
  const maxWh = energyRows.length > 0 ? Math.max(...energyRows.map((r) => r.wh)) : 1;

  return (
    <div className={styles.panel}>
      <section className={styles.section}>
        <span className={styles.sectionTitle}>Model</span>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>Parameters</span>
          <span className={styles.statValue}>{formatParams(paramCount)}</span>
        </div>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>Packed size (ternary)</span>
          <span className={styles.statValue}>{packedSizeLabel}</span>
        </div>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>fp16-equivalent size</span>
          <span className={styles.statValue}>{FP16_BASELINE_SIZE_MB.toFixed(1)} MB</span>
        </div>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>Measured compression</span>
          <span className={styles.statValue}>{MEASURED_COMPRESSION_RATIO.toFixed(2)}x</span>
        </div>
        <p className={styles.architectureBlurb}>
          {ARCHITECTURE.dModel}d model &middot; {ARCHITECTURE.nLayers} layers &middot; {ARCHITECTURE.nHeads} heads &middot;{" "}
          {ARCHITECTURE.contextLen}-token context &middot; {ARCHITECTURE.weightScheme}. Trained on{" "}
          {TRAINING_BUDGET_TOKENS}.
        </p>
      </section>

      <section className={styles.section}>
        <span className={styles.sectionTitle}>Live session</span>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>Tokens/sec</span>
          <span className={styles.statValue}>
            {isGenerating
              ? `${tokensPerSecond.toFixed(1)} (live)`
              : tokensPerSecond > 0
                ? `${tokensPerSecond.toFixed(1)} (last run)`
                : "—"}
          </span>
        </div>
        <div className={styles.statRow}>
          <span className={styles.statLabel}>Deterministic routing</span>
          <span className={styles.statValue}>
            {totalTurns > 0 ? `${routingStats!.deterministic}/${totalTurns} turns` : "—"}
          </span>
        </div>
      </section>

      <div className={`callout ${styles.calloutSmall}`}>100% client-side &mdash; zero server calls, ever.</div>

      <section className={styles.section}>
        <span className={styles.sectionTitle}>Energy (simulated vs. illustrative)</span>
        {bennyWh == null ? (
          <p className={styles.energyEmpty}>Generate a GPT-fallback reply to see this comparison.</p>
        ) : (
          <>
            <p className={styles.energyDisclaimer}>
              All figures below are simulated or illustrative reference numbers, not measured hardware power or
              vendor-published cloud energy costs.
            </p>
            {energyRows.map((row, idx) => {
              const pct = Math.max(2, (row.wh / maxWh) * 100);
              const barClass =
                idx === 0 ? styles.energyBarBenny : idx === 1 ? styles.energyBarFrontier : styles.energyBarReasoning;
              return (
                <div className={styles.energyRow} key={row.label}>
                  <div className={styles.energyRowHeader}>
                    <span>{row.label}</span>
                    <span className="mono">{row.wh.toFixed(3)} Wh</span>
                  </div>
                  <div className={styles.energyBarTrack}>
                    <div className={`${styles.energyBarFill} ${barClass}`} style={{ width: `${pct}%` }} />
                  </div>
                  {idx === 0 ? (
                    <p className={styles.energyNote}>
                      Simulated local CPU draw, {BENNY_WATTS_ASSUMPTION}W assumption &middot; this reply took{" "}
                      {lastElapsedMs!.toFixed(0)}ms wall-clock. NOT a measured hardware power reading &mdash; it&apos;s
                      (seconds &times; {BENNY_WATTS_ASSUMPTION}W) / 3600.
                    </p>
                  ) : (
                    <p className={styles.energyNote}>
                      ~{computeDeltaPct(bennyWh, row.wh)}% less than this row&apos;s simulated Benny figure, per the
                      illustrative estimate above &mdash; {CLOUD_ESTIMATE_DISCLAIMER}
                    </p>
                  )}
                </div>
              );
            })}
          </>
        )}
      </section>
    </div>
  );
}
