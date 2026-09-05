# ADR-087: TF-60 PR5 — ghost-outfield rearguard-positioning model

| Field | Value |
|---|---|
| **Date** | 2026-09-05 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

TF-60 (rest defense) needs a **counterfactual** for outfield rearguard positioning — "how much does the actual rest-defense shape differ from a league-average one?" — the outfield sibling of the frozen ghost-GK model (ADR-083). PR6 (the outfield counterfactual arm) consumes it; PR5 is the trained model itself.

The forcing constraints: it must (1) be **leakage-safe** — no feature may encode the modeled team's own rearguard coordinates, which are the prediction target; (2) serve BOTH tactical regimes — the ball-carrier's rest-defense rearguard AND the defending line — from ONE model; (3) reuse the established ghost-model artifact conventions (pickle-free, fail-closed load, velocity-keyed variants, FOV honest-NaN) so it composes with the rest of `tracking`; and (4) be **additive** — a new model in no default xfn list, changing no existing feature and triggering no retrain. Trained on the same 179-match public corpus as the bundled ghost-GK default (64 Gradient Sports + 108 SkillCorner + 7 IDSSE).

## Decision

Ship `silly_kicks/tracking/_ghost_outfield.py::GhostOutfieldModel` — **gradient-boosted MEAN x/y ensembles** (HGBR, not the ghost-GK RFCDE/KDE density path) predicting a league-average rest-defense rearguard position per `(frame, team, lateral slot)`, from a **leakage-safe 20/16-feature vector**, **possession-conditioned** (trained on BOTH teams' deepest-`n` per frame with a live `team_in_possession` discriminator; served for the in-possession slice by default), with the standard fail-closed load-guards, velocity-keyed `default`/`position_only` variants, and FOV cropped-honest-NaN serving. Bundle both variants trained at **1 frame/second** (mirroring the ghost-GK trainer). Serve via `serve_ghost_outfield_positions` (a serve seam, not an `add_*` aggregator) — additive, C4-free, no retrain.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. RFCDE/KDE density (copy ghost-GK) | Full positional distribution; reuse ghost-GK KDE code | The outfield arm needs a POINT estimate (the ghost's position), not a density; density adds cost and a KDE-grid contract for no consumer | Over-engineered for the consumer (PR6 differences a position) |
| B. One model per regime (in-possession + defending) | Each regime cleanly separated | Two artifacts, two training runs, doubled maintenance; the regimes share nearly all geometry | Possession-conditioning gives one model both regimes at no accuracy cost (measured: in-poss 6.97 m / out-of-poss 5.04 m from ONE model) |
| C. Train on every 25 fps frame | Maximum data | ~82M rows → infeasible fit on the DGX (single-core-effective HGBR, >9 h and not converging), and statistically redundant (consecutive frames near-identical) | 1 fps is ~25× fewer rows with no meaningful signal loss for a mean-positioning model; matches the shipped ghost-GK methodology |
| D. **Boosted-mean, possession-conditioned, 1 fps** (chosen) | Point estimate the arm needs; one model both regimes; tractable fit; mirrors ghost-GK conventions | Loses the full distribution (not needed) | — |

## Consequences

### Positive

- PR6's outfield counterfactual arm has its league-average baseline.
- One model serves both possession regimes; `position_only` covers velocity-less SB360 freeze-frames.
- Reuses the ghost-GK numba leaf-walk, pickle-free serialization, and fail-closed load-guards (chirality + feature-contract), so it inherits their cross-version stability.

### Negative

- A third bundled ghost artifact to maintain (npz + metadata + SHA256SUMS × 2 variants), HF-published like ghost-GK.
- The trainer reads the whole 25 fps corpus to subsample to 1 fps (extraction still touches every frame); acceptable (~7 s/match).

### Neutral

- `score_diff` is emitted home-perspective (not re-signed per modeled team); a reviewed feature-encoding choice, flagged for a future refinement if it proves weak.
- Additive: no existing feature changes, no VAEP/tracking retrain, no re-materialize, C4 aggregator count unchanged (33).

## Related

- **Specs:** `docs/superpowers/specs/2026-09-03-tf60-pr5-ghost-outfield-model-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-04-tf60-pr5-ghost-outfield-model.md`
- **Issues / PRs:** #233 (PR-S180); `training_commit` `b68328a`
- **ADRs:** sibling of ADR-083 (ghost-GK sweeper re-fit); inherits ADR-067 (velocity-keyed variants), ADR-077 (FOV observability), ADR-011/016/040/044/050 (trained-artifact conventions), ADR-076 (numba leaf-walk).
- **External references:** Le et al. 2017, "Data-Driven Ghosting" (see `NOTICE`).

## Notes

Trained on `training_commit` `b68328a` (clean tree), DGX-Spark aarch64, sklearn 1.9.0, 179 matches, 1 fps (4.17M rows). Held-out CV euclidean MAE: **6.00 m** (faithful `default`) / **6.07 m** (`position_only`); per-slot 5.96–6.11 m; per-possession in 6.97 m / out 5.04 m; rearguard coherence `ordering_fraction=1.0`. The `>30 m` high-sweeper corpus characteristic that motivated ADR-083 does not apply here (the rearguard is the deepest outfield line, well-sampled on every provider).
