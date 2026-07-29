# TF-19 PR-3b xS-arm probe — v1 (random) vs v2 (model-relevant defenders)

**Entanglement:** inside_band   **seed:** 42   **Matches:** 64   **Games:** 64
**Lock commit:** `78ffc70`   **Run commit:** `d1fc18d2fe5404cccd5ee9eb34ea7e1049e9f323`   (blindness: constants locked before the run; verify any intervening diff is inert)

## The honest framing
- v2 changes EXACTLY ONE thing vs v1: the placebo pool (random outfielder -> ball-nearest defenders). The defender placebo is a WEAKER control than nearest_def, so it is INERT in the ratio (`max()` pins to nearest_def); its job is to clear the no_valid_placebo gate with a principled null, not to move the bar.
- The ratio prong is therefore a 'beat nearest_def by 2x' test, near-certain to pass. v2's REAL decider is the clustered dose-response permutation, which v1 never reached.
- The attacker diagnostic is reported (non-gating): the nearest attacker is the shooter, so gating on attackers would answer a model-sensitivity question, not a deterrence one.

## v1 (frozen random placebo)

### v1: `no_valid_placebo`   re-gate: `unmeasurable_at_dose`   (random placebo)
- gated_band_n: 10180 (needs >= 100)   frames_used: 123430
- dose ladder (median |ΔxS| by ghost displacement): 2 m: 0.0155   3 m: 0.0200   4 m: 0.0222
- nearest_def control: 0.005025619640946388   placebo_p95: 0.0   gated_band_median: n/a (unmeasurable)
- effect vs control ratio (2 m / nearest-def): 3.08x
- dose_response rho / p: n/a (unmeasurable) / n/a (unmeasurable)   (prongs omitted — unmeasurable)

## v2 (relevance-matched)

### v2: `pass`   re-gate: `joins_with_caveat`   (model_relevant_def placebo)
- gated_band_n: 10180 (needs >= 100)   frames_used: 123430
- dose ladder (median |ΔxS| by ghost displacement): 2 m: 0.0155   3 m: 0.0200   4 m: 0.0222
- nearest_def control: 0.005025619640946388   placebo_p95: 0.0005699987057596445   gated_band_median: 0.015477672219276428
- effect vs control ratio (2 m / nearest-def): 3.08x
- dose_response rho / p: 0.4361283091408954 / 0.001
- attacker diagnostic p95 (non-gating): 0.0

## Targets -> used -> band reconciliation
- total targets: 123430   n_frames_used: 123430   distinct games: 64   gated_band_n: 10180
- targets->used drop frac: 0.0 (a drop is EXPECTED — ghost vs xs carrier-resolver mismatch; read as 'above that baseline').

