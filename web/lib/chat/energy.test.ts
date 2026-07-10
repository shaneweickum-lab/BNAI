import { describe, expect, it } from "vitest";
import { BENNY_WATTS_ASSUMPTION, computeBennyWh, computeDeltaPct } from "./energy";

describe("computeBennyWh", () => {
  it("computes (seconds * watts) / 3600 from elapsedMs", () => {
    const elapsedMs = 3600 * 1000; // exactly 1 hour
    expect(computeBennyWh(elapsedMs)).toBeCloseTo(BENNY_WATTS_ASSUMPTION, 6);
  });

  it("returns 0 for 0ms elapsed", () => {
    expect(computeBennyWh(0)).toBe(0);
  });

  it("scales linearly with elapsed time", () => {
    expect(computeBennyWh(2000)).toBeCloseTo(computeBennyWh(1000) * 2, 10);
  });
});

describe("computeDeltaPct", () => {
  it("computes the percent Benny is below a reference value", () => {
    expect(computeDeltaPct(0.17, 0.34)).toBe(50);
  });

  it("clamps to 0 rather than going negative when Benny is above the reference", () => {
    expect(computeDeltaPct(1, 0.34)).toBe(0);
  });

  it("returns 0 for a non-positive reference rather than dividing by zero", () => {
    expect(computeDeltaPct(0.1, 0)).toBe(0);
  });
});
