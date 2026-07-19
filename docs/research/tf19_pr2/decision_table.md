# TF-19 PR-2 decision-table verdict

Records the ADR-037 §4 `regate_verdict` outcome for the attempt-arm re-gate, using the
completed DGX Stage A / Stage B retrains. This is the PR-2 verdict record referenced by
CHANGELOG 4.51.0 and ADR-040; it does not re-derive the probe machinery (that shipped in
PR-1, `silly_kicks/tracking/_model_eval.py` / `_xcross_eval.py`, ADR-037).

## Source data

Authoritative numbers below were read directly from the DGX training box
(`karsten@192.168.68.73:~/Development/sk_stageB_448/xcross/xcross_attempt_v1/metrics.json`)
during this PR, not relayed from memory. The bundled `silly_kicks/tracking/_xcross_weights/default/metrics.json`
(Stage A, `public`) was cross-checked against the repository worktree and independently
reproduces the Stage A figures below.

## Cross arm — xCross GK-substitution probe (frozen rule, ADR-037 §3 frozen constants)

Frozen constants: `TF19_PROBE_RATIO=2.0`, `TF19_PROBE_ABS_FLOOR=0.01`
(`silly_kicks/tracking/_xcross_eval.py`). `tf19_ready` requires BOTH prongs to clear.

| Metric | Stage A (`public`, bundled default) | Stage B (`sc_extended`, +98 owner SkillCorner matches) |
|---|---|---|
| `gk_median_abs_delta` | 0.002417 | 0.009697 (~0.0097) |
| `nearest_def_median_abs_delta` | 0.001718 | 0.004380 (~0.00438) |
| ratio (gk / nearest_def) | ~1.41x | ~2.21x |
| clears `ratio >= 2.0` prong | no | **yes** |
| clears `abs_floor >= 0.01` prong | no | no (0.0097 < 0.01) |
| dose ratio (4m band / 2m band) | ~2.36x | ~1.66x |
| `gk_zero_fraction` | n/a (not the reported gate figure) | 0.0 |
| `tf19_ready` | `false` | `false` |
| `tf19_reason` | ratio + floor both miss | "GK \|delta\| did not clear ratio>=2.0 x control AND abs-floor>=0.01" |

Stage B strengthened the GK-substitution signal materially: the ratio prong went from
failing (~1.41x) to clearing (~2.21x, above the 2.0x bar), and the median absolute delta
nearly quadrupled (0.0024 to 0.0097). The gate still does not open, because `tf19_ready`
is an AND of both prongs and the absolute-floor prong (>= 0.01) is missed narrowly, at
0.0097 versus the 0.01 floor. This is a **10% relative miss on the floor prong alone**,
not the order-of-magnitude gap Stage A showed on both prongs.

## Trap: the BUNDLED artifact's paired block is Stage A, not Stage B

`silly_kicks/tracking/_xcross_weights/default/metrics.json` (and the xS equivalent) record the
**Stage A** paired test, whose corpus contained **no owner-tier SkillCorner rows**. There
`cand_masks["sc_extended"] = is_public | is_sc_private` with `is_sc_private` **empty**, so the
`sc_extended` candidate was the *same row mask* as `public` — the same model scored against
itself — and its deltas are `[0.0, 0.0, 0.0, 0.0, 0.0]` **by construction**. Since
`clears_rule` demands strictly positive values, zero fails and the fixed sequence stops, which
is why `public` is the bundled default.

**Those zeros are a degenerate self-comparison, not a measured null, and NOT a verdict on
`sc_extended`.** The Stage B run — the one with the owner SkillCorner rows actually present —
records `sc_extended` deltas `[-0.008, +0.009, +0.005, +0.029, +0.023]` (4/5 positive, mean
+0.0117), `clears_rule = True`, and `"shipped": "sc_extended"`. `sc_extended` is HF-only
because it is **not redistributable** (ADR-038), not because it lost a test.

Recorded because a reader (2026-07-18) went to the bundled artifact rather than this table,
misread the zeros as a failure, and published that claim on the public HF model cards before it
was caught. Corpus sizes are the quickest tell: Stage A xCross is 718,005 rows, Stage B is
1,209,333.

Both Stage A and Stage B probes were run against the same held-out GS pair
(`gradientsports` matches `10502` / `10503`); the Stage B run's own `probe_sample_in_training_folds`
record confirms neither match entered the `sc_extended` training folds (both `false`),
i.e. the probe measurement is genuinely out-of-fold.

## Cross arm — decision-table row

Applying `regate_verdict(arm="cross", probe_verdict=..., entanglement=...)`
(`silly_kicks/tracking/_model_eval.py`):

- `probe_verdict` = `"fail"` — `tf19_ready=false` on the gated (Stage B) run, since the AND
  of the ratio and floor prongs is not satisfied (`_tf19_ready` in `_xcross_eval.py`).
- `entanglement` = `"inside_band"` — the registered expected outcome per ADR-037 §2 point 2
  (the causal-harness GK-confounder-entanglement result was expected to land inside the
  placebo band after the retrain, not to clear it; this is the banked entanglement input
  for the cross arm, carried forward from the ADR-037 registration).
- `regate_verdict(arm="cross", probe_verdict="fail", entanglement="inside_band")` =
  **`gated_clean_fail`**.

Per ADR-037 §4's table, `probe_verdict == "fail"` maps to `gated_clean_fail` regardless of
the entanglement value (the function checks `fail` before consulting `entanglement`) — this
is the "clean fail" the ADR anticipated as the registered expected outcome (ADR-037 §2 point 2:
"the frozen cross gate is therefore EXPECTED to hold"), now measured on corrected, chirality-fixed
frames instead of the mis-served ones the gate was originally read against.

**Cross-arm verdict: `gated_clean_fail`.** The Delta-attempt cross arm stays gated; per
ADR-037 §4 this routes to GK feature engineering, not to "no signal."

## Shot arm — status: PENDING (PR-3-gated)

The xS dose-banded probe rule (`evaluate_xs_probe`, `xs_substitution_probe`) and its
registry entry (`PROBE_WRAPPERS["xs"]`) are already implemented in `_model_eval.py` (PR-1,
shipped 4.47.0). Verified during this PR:

```
grep -n "def evaluate_xs_probe|def regate_verdict|PROBE_WRAPPERS" silly_kicks/tracking/_model_eval.py
```

confirms `evaluate_xs_probe`, `regate_verdict`, and both the `"xcross"` and `"xs"` entries
in `PROBE_WRAPPERS` are present and importable. What is **not** available is the input the
xS probe consumes: `xs_substitution_probe(model, frames, targets, ...)` requires
ghost-substitution `targets` produced by the GK-substitution engine, which is the
`silly_kicks/gkdv/` package named in ADR-037's PR sequencing as **PR-3** ("PR-3 = the
`gkdv/` package (engine + physics arms + validation)"). That package does not exist yet in
this checkout (`silly_kicks/gkdv/` is absent). There is therefore no substituted-targets
input to feed `xs_substitution_probe` in this PR.

**This PR records no xS probe result.** No number is fabricated for the shot arm. The shot
row of the decision table is **PENDING PR-3** — it will be filled in once the `gkdv/`
ghost-substitution engine lands and can produce the `targets` frame the registered xS probe
rule (ADR-037 §3, all of its locked constants — twelve as registered in
`PROBE_WRAPPERS["xs"]["rule_constants"]`; the pre-exec-review set of eight grew by four)
consumes.

| arm | probe verdict | entanglement | decision-table row |
|---|---|---|---|
| cross | `fail` | `inside_band` | `gated_clean_fail` |
| shot | *(not run — PR-3-gated)* | *(not run — PR-3-gated)* | **PENDING** |

## Recommendation

The Stage A -> Stage B jump (ratio 1.41x -> 2.21x, clearing the 2.0x bar) is a real,
substantially strengthened GK-responsiveness signal even though the gate stays closed on
the absolute-floor prong. Per ADR-037's routing rule, a probe fail is not "no signal" — it
sends the arm to GK feature engineering. Given how close the miss is (0.0097 against a 0.01
floor, a 10% relative gap versus Stage A's order-of-magnitude gap on both prongs), this is
flagged for owner / Eyestone attention as a candidate for the next feature-engineering
pass on the GK-distribution feature block, rather than a settled dead end.
