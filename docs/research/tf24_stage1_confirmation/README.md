# TF-24 Stage-1 confirmation (ADR-060)

Authoritative confirmation that TF-24's Stage-1 recommendation stands on corrected (post-ADR-028)
geometry, run with the redesigned `scripts/check_stage1_argmax.py` (prefer-incumbent selection over the
indistinguishable set + a standing fold-stability diagnostic).

- **Code:** `run_commit 2cecd2b` (clean tree, `run_tree_dirty: false`).
- **Store:** `~/tf24-store/s1.db` (the `balanced_confirm_tol3` study) — **150 trials over `beta`/`gamma`
  only; `tolerance_m` was held at 3.0, never swept**. This discharges the spec §7 store-reconciliation
  gate: the checker's "only beta/gamma vary in the store" was correct, and item C's removal of
  `tolerance_m` from `stage1_config` simply aligns the config with what this store already did.
- **Corpus:** 179 matches (SkillCorner + IDSSE/DFL + Gradient Sports).

## Result

- **Invariance (prong 1):** `shipped_point` and `recorded_optimum` both `stands` at 0.999999 (≥ the
  pre-registered 0.999 threshold) — carrier assignment is orientation-invariant under an exact point
  reflection.
- **Selection (prong 2):** `moved = False`. The shipped default (`beta=0, gamma=0.25`, mean 0.541346)
  is the outright **highest-mean** of the six candidates; no candidate clears the effect-size floor
  **and** the paired-SE test. `carrier_selected.json` = `{beta: 0.0, gamma: 0.25}` (no `tolerance_m` —
  held constant, sourced downstream from `DEFAULT_CARRIER_PARAMS`).
- **Fold-stability:** `fold_to_point_var_ratio = 68849.6` (between-fold noise dwarfs between-point
  differences), 3 distinct fold winners → verdict `no_discriminating_evidence`. The six points span a
  mean range of ~4.4e-4 against a CV SE of ~0.0175 (≈ 1/40 of one SE).
- **δ robustness:** the keep-incumbent recommendation is invariant to `MIN_EFFECT_SIZE` across
  `[0, 0.1]` (`moved=False` at every δ), so the frozen `δ = 0.005` is not load-bearing here.

Both spec §7 pre-land gates (store reconciliation; δ derivation + robustness) are discharged, so
ADR-060 is **Accepted**.

## Files

- `metrics.json` — full confirmation output (invariance + selection + fold_stability + run provenance).
- `carrier_selected.json` — the recommended carrier point `{beta, gamma}` + provenance; the Stage-2 input.
