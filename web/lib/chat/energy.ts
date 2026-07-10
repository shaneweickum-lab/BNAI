/**
 * Energy-comparison numbers for the MetricsPanel showcase, adapted from the
 * standalone prototype at prototypes/context-folding/index.html (see its
 * "ENERGY COMPARISON ENGINE" section). Every number here is either:
 *   (a) derived from an actual measured elapsedMs (from the inference
 *       worker's onDone callback) combined with a stated, made-up wattage
 *       assumption, or
 *   (b) a static "illustrative estimate" for cloud tiers, explicitly
 *       labeled as NOT vendor-published data.
 * None of this should ever be read as a real hardware power measurement or
 * a real vendor energy-cost figure -- see DISCLAIMER_TEXT, which every
 * cloud-tier row must carry verbatim.
 */

// Simulated local CPU draw assumption, in watts -- not a measured reading.
export const BENNY_WATTS_ASSUMPTION = 15;

export interface EnergyReferenceRow {
  label: string;
  wh: number;
}

// Static reference figures for comparison, not measured for this session.
export const CLOUD_REFERENCE_ROWS: EnergyReferenceRow[] = [
  { label: "Frontier API tier", wh: 0.34 },
  { label: "Reasoning-model tier", wh: 4.0 },
];

// Exact disclaimer text every cloud-tier row must carry -- a bare percentage
// delta reads as a marketing claim; this project's whole positioning is
// engineering honesty over hype, so the percentage only ever appears
// directly next to this string, never standalone.
export const CLOUD_ESTIMATE_DISCLAIMER = "(illustrative estimate, not vendor-published data)";

/** Wh for a real generation, from its measured elapsedMs, under the stated
 * BENNY_WATTS_ASSUMPTION -- NOT a measured hardware power reading. */
export function computeBennyWh(elapsedMs: number): number {
  const seconds = elapsedMs / 1000;
  return (seconds * BENNY_WATTS_ASSUMPTION) / 3600;
}

/** Percentage Benny's simulated Wh is below a reference row's Wh, clamped to
 * [0, 100]. Only ever meant to be rendered directly beside CLOUD_ESTIMATE_DISCLAIMER. */
export function computeDeltaPct(bennyWh: number, referenceWh: number): number {
  if (referenceWh <= 0) return 0;
  return Math.max(0, Math.round((1 - bennyWh / referenceWh) * 100));
}
