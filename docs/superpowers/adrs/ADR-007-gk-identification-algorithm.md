# ADR-007: GK Identification Algorithm

| Field | Value |
|---|---|
| **Date** | 2026-05-04 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

Kloppy's `is_goalkeeper` flag for Metrica and SkillCorner tracking data is unreliable, with empirical coverage of 21-50% per (frame, team). The root causes vary by provider: Metrica's CSV parser hardcodes `Position.Unknown`, SkillCorner extrapolates from sparse starting-position data, and jersey-number heuristics (GK = #1) fail on 6/6 sampled lakehouse matches.

PR-S26 requires 100% GK identification for downstream features (TF-12 GK angles, TF-13/TF-14 defensive line). A silly-kicks-side positional algorithm is needed that works regardless of provider metadata quality.

The algorithm must handle: (1) standard GKs who stay in the penalty area, (2) sweeper-keepers who play high, (3) GK substitutions mid-match, and (4) brief outfielder appearances near goal.

## Decision

Implement a B+ filtered algorithm with agreement-based source resolution:

1. **Always run** the positional algorithm (not count-triggered)
2. **Multi-GK detection** via strict criteria (pa_dwell ≥ 0.40 AND dist < 20m)
3. **Sweeper-keeper fallback** via rank-sum scoring when strict fails
4. **Agreement-based source**: `is_goalkeeper_source="native"` iff algorithm picks == kloppy's native picks

Thresholds locked: `_GK_N_FRAMES_FRAC=0.30`, `_GK_PA_DWELL_MIN=0.40`, `_GK_DIST_MAX_M=20.0`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Trust kloppy's native flag | Zero implementation cost | 21-50% coverage on Metrica/SkillCorner | Coverage too low for reliable features |
| B. Count-triggered fallback (run algorithm only when kloppy ≠ 1 GK) | Lower compute cost | N=2 after sub falsely triggers | Q26 lakehouse review showed count-based is fragile |
| C. Always-run B+ with agreement-based source | Consistent detection; provenance tracking | Slightly higher compute | — (chosen) |

## Consequences

### Positive

- 100% GK identification rate on all tracking providers
- Multi-GK substitution detection without false positives
- Sweeper-keeper detection via fallback when strict criteria fail
- Provenance tracking via `is_goalkeeper_source` column enables downstream debugging

### Negative

- ~20ms overhead per match for algorithm execution (acceptable given tracking volumes)
- Native adapters (Sportec, Gradient Sports) must emit `is_goalkeeper_source="native"` for schema consistency

### Neutral

- `TrackingConversionReport` gains two fields: `n_teams_gk_derived`, `derived_gk_picks`

### Validation

- **Sportec** (native ground truth): 14/14 teams — Tier-1.
- **SkillCorner** (external `match.json` roster via pining, `player_role.acronym == "GK"`):
  20/20 team-GKs across the 10 public A-League matches — **Tier-1** (PR-S86, 4.19.1; gated by
  `tests/tracking/test_gk_skillcorner_roster_e2e.py`, exact-set-equality). The algorithm
  required no change.
- **Metrica**: external-roster verification is impossible on public data (anonymized players,
  no roster to anchor against; licensed EPTS metadata unavailable) — remains **Tier-2**
  algorithm self-consistency, a documented permanent limitation.

## Related

- **Specs:** `docs/superpowers/specs/2026-05-04-pr-s26-kloppy-gk-hardening-design.md`
- **ADRs:** extends ADR-004 (tracking namespace charter), informs TF-13/TF-14 (defensive line features)

## Notes

Penalty area definition (SPADL coordinates):
- x < 16.5 OR x > 88.5 (symmetric goal ends)
- y ∈ [13.84, 54.16] (18-yard box width)

Distance to goal: `min(x, 105 - x)` — symmetric for both goal ends.

## Amendment (4.38.0, PR-S105) — SkillCorner trusts the native roster; positional derivation is per-match, not per-batch

**Context.** `tracking.skillcorner.convert_to_frames` discarded the (Tier-1, roster-validated) native
`is_goalkeeper` and **re-derived it positionally via `derive_goalkeepers` on every call** (using the
native flag only for the `is_goalkeeper_source` label). Positional derivation is sound on a full match
but **unstable on a short window**: the lakehouse's AC dispatch builds frames in **250-frame (~25 s)
batches**, and in such a window a transiently goal-parked outfielder (defending CB / camped forward)
meets the symmetric PA-dwell criterion. Across ~164 per-batch builds the **union of flagged players
reached ~15/team** (real bronze; the metric `xt_gk` was then computed for 19–24 players/match instead
of ~1–2 keepers). GS/sportec/idsse were immune — their converters already trust the native roster.

**Decision.**
1. **SkillCorner trusts the native roster `is_goalkeeper`** when present (≥1 GK per `(game_id, team_id)`):
   use it as-is, `is_goalkeeper_source = "native"`, **skip `derive_goalkeepers`**. Derive only as a
   fallback for a `(game, team)` whose native flag is absent (a data-quality edge). This makes SkillCorner
   **batching-immune** — the roster flag is identical in every batch (verified: per-batch union 15/13 → 1/1
   per team) — and aligns SkillCorner with the existing GS/sportec roster-trust.
2. **S2 guard (machine-observable):** `warnings.warn` + `TrackingConversionReport.n_implausible_gk_teams`
   when a resolved per-`(game, team)` GK count is implausible (`>2` or `0`). The contamination went
   undetected until a downstream cohort anomaly; a converter-level count check catches it immediately.

**Consequence / follow-up.** This is SkillCorner-only (GS/sportec/idsse/metrica identification unchanged);
it is an `xt_gk` serve-output change for SkillCorner → the lakehouse re-materializes. **Metrica (anonymized,
no roster) is contaminated the same way and roster-trust cannot fix it** — it needs positional derivation
run **once per full match**, not per 250-frame batch (a separable derive-once silly-kicks API + a lakehouse
change). Tracked as a follow-up; the S2 guard surfaces it in the meantime. The core `derive_goalkeepers`
algorithm is unchanged (still Tier-1/Tier-2 as above); only WHEN/whether it runs changed for SkillCorner.
