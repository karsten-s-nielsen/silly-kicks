# TF-19 PR-3b — xS-arm GK-substitution probe

**Probe verdict:** `no_valid_placebo`
**Re-gate (arm=shot, entanglement=inside_band):** `unmeasurable_at_dose`
**Frames used:** 123430   **Matches:** 64   **Games:** 64   **seed:** 42

## Which branch? (read first — the ghost-accuracy paradox)
- gated_band_n: 10203  (needs >= 100 to be measurable)
- `unmeasurable_at_dose`: band under-filled (couldn't reach 2 m; NOT a null effect)
- `no_valid_placebo`: band fills but the random-outfielder CONTROL can't be certified (re-gates to `unmeasurable_at_dose`; see Placebo below; NOT a null effect)
- `gated_clean_fail`: band fills, GK flat (closes the arm; names the GK-feature lever)

## The effect (does the keeper's position move xS at all?)
- dose ladder (median |ΔxS| by ghost displacement): 2 m: 0.0154   3 m: 0.0200   4 m: 0.0222
- unbanded median |ΔxS|: 0.002978019416332245   nearest-defender control: 0.0049907974898815155
- effect vs control ratio (2 m / nearest-def): 3.09x

## Placebo (random-outfielder control — the certification blocker)
- placebo_p95: 0.0   placebo_zero_fraction: 0.6645104381064393
- an all-zero placebo_p95 + a high zero-fraction => the aggregate xS features barely respond to one distant player moving 2 m => the control is degenerate => `no_valid_placebo` (prongs not run).

## Targets -> used -> band reconciliation
- total targets: 123430   n_frames_used: 123430   distinct games: 64   gated_band_n: 10203
- targets->used drop frac: 0.0  (baseline structural drop; see note)
- note: a drop is EXPECTED — the ghost engine resolves the carrier with the ghost model's carrier_params, the probe with the xs model's; the flag fires only ABOVE that baseline.

## Rule prongs  (omitted — no_valid_placebo)
- gated_band_median: n/a (unmeasurable)
- nearest_def_median: 0.0049907974898815155
- placebo_p95: 0.0
- ratio rule: gated_band_median >= 2.0 * max(nearest_def_median, placebo_p95)
- absolute floor (TF19): gated_band_median vs 0.01
- dose_response rho / p: n/a (unmeasurable) / n/a (unmeasurable)

