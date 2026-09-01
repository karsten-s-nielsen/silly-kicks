# ADR-083: TF-60 PR3 — Rest-Defense GK-Ghost Re-fit (Extended-Grid `sweeper` Variant)

**Status:** Accepted (2026-09-01, silly-kicks 4.105.0, PR-S176)

## Context

TF-60's rest-defense × GK arc (spec `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md`)
values keeper POSITIONING counterfactually: how much does the actual in-possession keeper change the
attacking team's space and threat versus a league-average "ghost" keeper. §9 of that spec made the
ghost's in-possession validity a **GATE, not an assumption** — and the gate **fired**.

The shipped `GhostGkModel` **hard-saturates at its trained-label ceiling `GRID_X_MAX = 30 m`**:
`prepare_ghost_gk_training_data` drops every keeper label with `gk_x > 30 m` as a "sweeper rush", so
`predict_mean` cannot place an in-possession high sweeper (30–45 m off its own goal — precisely the
rest-defense regime the arms must reason about). `ghost_out_of_box` is **blind** to this, because the
model CLIPS its output to ≤ 30 m while out-of-domain is an *input* property no output flag can see.
A controlled extrapolation probe confirmed it (`docs/research/tf60_ghost_gk_in_possession_validity/`).

A ghost that cannot represent a high keeper would make the PR4 GK arms measure the wrong
counterfactual. So a GK-ghost re-fit was inserted BEFORE the arms — reshaping the arc to
PR1 (Layer-1 KPIs) → PR2 (Layer-2 danger) → **PR3 (this: GK-ghost re-fit)** → PR4 (GK arms) →
PR5 (ghost-outfield model) → PR6 (outfield arm).

The hard constraint: the shipped `default` / `position_only` / `full` ghost variants are consumed by
GKDV and VAEP; changing them is a retrain/re-materialize trigger. The re-fit must therefore be
**purely additive**.

## Decision

Ship an **additive** extended-grid `GhostGkModel` variant pair — `sweeper` (faithful, 26 features) and
`sweeper_position_only` (velocity-less SB360, 21 features) — grid ceiling `x_max = 52.5 m` (halfway to
midfield), `y` and resolution unchanged. The PR4 arms consume `from_variant("sweeper")`; the frozen
variants are untouched.

1. **The grid becomes a first-class per-model `GhostGridSpec`** (frozen dataclass: `x_min, x_max,
   y_min, y_max, resolution`; derived `nx`/`ny`; `to_metadata_dict()` emits the exact current 7-key
   shape). `DEFAULT_GHOST_GRID` equals the old module constants. Threaded through `__init__` / `save`
   (writes `grid_spec`) / `load` (restores it; a pre-refactor artifact with no `grid_spec` key loads
   `DEFAULT_GHOST_GRID`) / the label-domain filter / `serve_ghost_gk_positions`'s `ghost_out_of_box`.

2. **`GhostGkVariant` gains `sweeper` / `sweeper_position_only`.** `_resolve_ghost_model_for_frames`
   gets an ADDITIVE `model == "sweeper"` branch placed BEFORE the `model is not None` custom
   short-circuit; it velocity-keys within the family (faithful on velocity-bearing frames,
   position_only on declared velocity-less frames). A missing `sweeper_position_only` returns
   `(None, "sweeper_position_only")` — the honest NaN degrade, NEVER the (invalid-on-velocity-less)
   faithful default (the ADR-067 asymmetry). GKDV passes `None` / `"default"` / an instance and can
   NEVER reach this branch.

3. **`predict_density` fail-loud on a non-default grid** (raises `ValueError`). The KDE density path
   stays on `DEFAULT_GHOST_GRID` this cycle; the mean/serve path (`predict_mean`,
   `compute_ghost_gk`, `serve_ghost_gk_positions`) is grid-independent and works on the extended grid.

4. **Trainer / publisher.** `train_ghost_gk.py` gains `--grid-x-max` (keyed into the
   shard-generation token, the 4.77.1 stale-shard rule) and extends `--variant` to name
   `sweeper` / `sweeper_position_only`, plus `>30 m`-stratum MAE and per-provider `>30 m` coverage in
   `metrics.json`. `publish_ghost_gk.py` builds its sanity sample from the metadata `feature_names`.

**Byte-identity for the frozen variants is proven THREE ways** (a full metadata-SHA round-trip is
*unachievable* — `save()` recomputes `feature_contract.probe_sha256`): (1) the pre-existing golden /
chirality / feature-contract / KDE-density tests pass unchanged; (2)
`DEFAULT_GHOST_GRID.to_metadata_dict()` equals `default`'s committed `grid_spec`; (3) a re-save of
`default` differs only in `feature_contract`.

**Bundled artifacts** were re-fit on the same 179-match public corpus as `default`, from a CLEAN
`training_commit` (no `--allow-dirty`; `run_tree_dirty = false`). Results: CV MAE euclid 1.142 m
(faithful) / 1.164 m (position_only); boosted-reconstruction parity 1.2e-13 (exact — safe to publish);
all acceptance criteria PASS. **§9 payoff (REPORTED, not gated, per spec §6/§7.3):** `>30 m`-stratum
MAE 2.06 m / 2.03 m — the sweeper *places* high keepers where the default is blind (probe: default
saturates at ~29.8 m while the sweeper tracks a scene translated +25 m to ~45.9 m). `>30 m` coverage
is IDSSE/Sportec-dominated (11.5 %); SkillCorner 0.24 %; **Gradient Sports 0.0 %**.

## Alternatives considered

- **Re-fit `default` with the cap lifted (in place).** Rejected: `default` is consumed by GKDV/VAEP;
  changing its weights is a retrain + re-materialize trigger for the whole downstream, for a capability
  only the rest-defense arms need. Additivity confines the blast radius to a new opt-in variant.
- **A separate `GhostSweeperModel` class.** Rejected as over-engineered: the model, extractor,
  serialization, and load-guards are identical to `GhostGkModel`; only the grid ceiling differs. A
  first-class `grid_spec` on the existing class captures the one real difference with no code
  duplication (and reuses every load-guard unchanged).
- **Change the module-global `GRID_*` constants to 52.5.** Rejected: it would move `default`'s label
  domain and break its byte-identity (retrain trigger), and it conflates "the label grid this model
  was trained on" (a per-model fact) with "the KDE density grid" (still 30 m). The per-model
  `GhostGridSpec` separates them.
- **Extend the KDE density grid to 52.5 too.** Deferred (density is fail-loud on the extended grid).
  Density (`predict_density`) is a fit-only diagnostic that the arms do not consume, and extending it
  would enlarge the golden surface for no consumer this cycle. Fail-loud keeps the door explicit.
- **Gate the PR4 arms without a high-capable ghost** (accept the saturating default). Rejected: §9
  made in-possession validity a real requirement — a saturating ghost makes the arm's counterfactual
  systematically wrong in exactly the rest-defense regime the metric is about.

## Consequences

### Positive
- The PR4 GK arms get a ghost that can represent the in-possession high-sweeper regime (2 m stratum
  accuracy where the default was structurally blind).
- Purely additive: **no GKDV retrain, no VAEP retrain, no re-materialize, C4 unchanged** (no new
  emitted feature columns; the variant serves the existing `ghost_gk_x/y`).
- The grid-first-class refactor closes a latent gap (`load` never restored the grid) and makes the
  ghost grid auditable per artifact.
- The bundled sweeper artifacts are *more* thoroughly verified than the frozen siblings: their
  `feature_contract` probe is fresh at the training commit, so the fingerprint is fully verified
  (the older `default` probe is stale → `UnverifiableFeatureContractWarning`, unchanged).

### Negative
- Two more bundled artifacts (~1.7 MB npz each) to maintain and, if published, upload.
- The high-regime training signal is **almost entirely IDSSE/Sportec** (11.5 % coverage) — Gradient
  Sports contributes **0.0 %** keepers beyond 30 m and SkillCorner 0.24 %. The sweeper is valid (parity
  exact, MAE in line with siblings). The GS 0.0 % was investigated and is a **Gradient Sports
  source-data limitation**: GS clamps the goalkeeper's tracked position to a hard **27.5 m from goal**
  (universal and exact across matches / keepers on the raw provider data; keeper-specific — outfielders
  roam the full pitch), which silly-kicks faithfully passes through (native keeper identity, no
  coordinate clamp). So **any GS goalkeeper-depth analysis is invalid past 27.5 m** — the PR4 GK arms,
  `xt_gk`, GK influence, and the factual keeper position. Documented in
  `docs/research/gs_keeper_clamp/findings.md` and flagged at conversion time by the new
  `validate_gk_position_clamp` / `GoalkeeperClampWarning` (auto-wired into
  `tracking.gradientsports.convert_to_frames`).

### Neutral
- `predict_density` is fail-loud on the extended grid; a future cycle may extend the KDE grid if a
  consumer needs sweeper density.
- Naming: `sweeper` is capability-descriptive (cross-layer asymmetry with the arms' `rest_defense_*`
  naming is expected — this is a `tracking`/GhostGkModel artifact, the ENABLER for the arms, not a
  `restdefense` column).

## CLAUDE.md Amendment

Add to the ghost-GK / trained-artifacts area a GhostGkModel-variant contract: the grid is a
first-class per-model `GhostGridSpec` (threaded through save/load); `default`/`position_only`/`full`
are FROZEN and byte-identical; `sweeper`/`sweeper_position_only` are additive extended-grid
(`x_max = 52.5`) variants selected only by an explicit `"sweeper"` request (gkdv never selects them →
no retrain); `predict_density` is default-grid-only (fail-loud otherwise); the mean/serve path is
grid-independent. Update the restdefense bullet's PR-arc line: PR3 (GK-ghost re-fit) shipped;
PR4 (GK arms) consumes `from_variant("sweeper")`.

## Related

- Spec: `docs/superpowers/specs/2026-08-30-tf60-restdefense-gk-ghost-refit-design.md` (sub-spec);
  parent `2026-08-30-tf60-rest-defense-structure-and-gk-design.md` §9.
- Plan: `docs/superpowers/plans/2026-08-31-tf60-pr3-gk-ghost-refit.md`.
- Finding + probes: `docs/research/tf60_ghost_gk_in_possession_validity/`.
- Gradient Sports keeper-clamp investigation: `docs/research/gs_keeper_clamp/findings.md` (the GS 0.0 %
  finding, run down to the raw provider data; the `validate_gk_position_clamp` / `GoalkeeperClampWarning`
  detector added this cycle).
- ADR-080 (TF-60 PR1 Layer-1 KPIs), ADR-081 (PR2 Layer-2 danger).
- ADR-067 (velocity-keyed model auto-select; the position_only variant + the missing-fallback-to-NaN
  asymmetry this reuses).
- ADR-011 / ADR-016 / ADR-040 / ADR-044 / ADR-050 / ADR-076 (trained-artifact discipline: npz + JSON +
  SHA256SUMS, pickle-free, chirality + feature-contract load-guards, numba leaf traversal — all reused
  unchanged).

## Notes

- The three-way byte-identity gate replaces an unachievable metadata-SHA round-trip: `save()`
  recomputes `feature_contract.probe_sha256` from the current extractor, so `default`'s metadata SHA
  differs on re-save independent of this refactor (impl-review P3I-02; owner-ratified 2026-08-31).
- Cycle mechanics: TWO commits, `gh pr merge --merge` (NOT squash — a rewritten SHA would orphan the
  weights' `training_commit` provenance); commit 1 = code + tests + docs (the clean tree the DGX
  trained from), commit 2 = the DGX weights + this ADR + version + CHANGELOG + TODO + CLAUDE.md.
