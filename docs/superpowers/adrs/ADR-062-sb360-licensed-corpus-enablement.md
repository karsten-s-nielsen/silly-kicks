# ADR-062: SB360 licensed-corpus enablement — loader path, opt-in visibility companions, and a leak-safe validation driver

| Field | Value |
|---|---|
| **Date** | 2026-08-17 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |

## Context

`pining-for-the-data` now serves a **30-match licensed StatsBomb 360 corpus** under provider
`statsbomb`, each match carrying four artifacts (`events`, `freeze_frames`, `metadata`, `roster`). The
library already had the shaping half — `providers/statsbomb.shape_snapshots` +
`tracking.snapshot_to_tracking_frames` (ADR-054) — but no path from raw pining artifacts to SPADL
actions + tracking frames, no way to report *how much of the pitch a partial-visibility provider
actually observed* around a count feature, and no validation driver for the corpus. This ADR records
the enablement, which is **additive** end-to-end (no default xfn list changes, no retrain, no
re-materialization).

## Decisions

**1. Loader shape — widen the INTERNAL tuple, never the public one.** `scripts/_loader_pining._build_match`
widens from a 4-tuple to a 5-tuple (adds `visible_area`), and a parallel PUBLIC `load_statsbomb_matches`
yields a 6-tuple carrying the SB360-only `visible_area`. The public `load_matches` 5-tuple and its every
unpack site stay byte-for-byte untouched (Hyrum's Law: the tracking providers have no per-action visible
area, so widening their iterator would break every consumer for a field they can't populate). The
raw-JSON flattener is scripts-side (`scripts/_sb_raw.py::flatten_events`, single-sourcing SIX copies of
the `_adapt_events` body; plus `parse_freeze_frames`/`parse_metadata`/`parse_roster`); the
`providers/statsbomb` port stays pure-shaping (ADR-054), so raw parsing never enters the library.
Fidelity (`xy_fidelity_version`) is threaded from `metadata` — coordinates without it are silently
mis-scaled — and roster identity is joined onto actions.

**2. Visibility companions are OPT-IN and ADDITIVE (ADR-009).** `add_action_context(actions, frames, *,
visible_area=None)` emits six companion columns (`<feature>_observed_fraction`/`_observed_source` for the
three region-based counts) ONLY when `visible_area` is supplied. The primary count columns are
byte-identical with and without it, and the per-Series functions + `tracking_default_xfns` are untouched,
so opting the coverage in is the consumer's choice and no VAEP feature changes. `classify_region_observation`
+ the FEATURE-level `REGION_OBSERVATION_SOURCE_VALUES` (`observed`/`no_polygon`/`degenerate_polygon`/
`degenerate_region`) REUSE the ADR-055 polygon tokens and ADD `degenerate_region` — they are **not** a
widening of the pinned `VISIBLE_AREA_SOURCE_VALUES`. The region of interest is an inscribed disk (radius =
the measured distance) or the triangle-to-goal; a NaN radius (no visible opponent) → `degenerate_region`,
never a fabricated fraction. `unlinked` is overlaid by the caller from the action↔frame link.

**3. ONE call convention, single-sourced under `scripts/`.** The ADR-053 SB360 audit and the new validation
driver both run the whole `add_*` battery on freeze-frames, and that per-aggregator adapter layer is
exactly what silent-empty-blocked once (ADR-053). To stop it forking, the adapters (`tests/sb360/_calls.py`)
+ the adapter map (`_registry._adapters()`) are MOVED to `scripts/_sb_battery.py`; `_calls.py` becomes a
re-export shim so the committed `_entries` round-trip stays byte-identical. Layering is `tests → scripts`
(scripts imports no tests), asserted by a leaf-invariant test.

**4. NO observed-region ADR-053 audit axis.** The design spec proposed a third audit axis for the companion
columns. It was **rejected on measurement**: ADR-053 is a two-leg (Leg A vs Leg B) fabrication detector,
and the companions depend on the polygon + action geometry — not kinematics or roster — so on the
full-coverage audit fixture both legs are byte-identical and every verdict would be `identical → works`. A
gate that records `works` without ever exercising partial visibility is the "coverage denominator
masquerading as a signal" / "a gate that certifies the failure it catches is worse than none" trap the
codebase names elsewhere. The companions are verified where it is meaningful — `tests/tracking/test_visibility.py`,
`tests/tracking/test_add_action_context.py`, and `scripts/validate_sb360_licensed_corpus.py` (all five
degradation tokens observed on the real licensed corpus). The scope note lives at
`tests/sb360/_registry.py::audited_surface`, where a maintainer tempted to add the axis meets the reasoning.

**5. A leak-safe validation driver.** `scripts/validate_sb360_licensed_corpus.py` measures per-column
coverage, the honest `honest_nan`/`silent_degrade`/`raises` distribution, the three count features'
`observed_source`/`fraction` distributions, and the real-`visible_area` pitch-coverage distribution.
Licensed per-match shards go to a GITIGNORED root; only the reconciled aggregate lands under
`docs/research/` (ADR-052 shards; ADR-037 provenance stamped, `require_clean_tree`); enrolled in
`ARTIFACT_DRIVERS`. The citation-quality artifact is a clean-tree run.

**6. Snapshot team ids are RESOLVED to real match ids, not left synthetic.** The validation driver
(Decision 5) surfaced a latent 4.76.0 defect: `providers/statsbomb.shape_snapshots` emitted a synthetic
actor-relative `{0,1}` team id on the principle "SB360 records no team identity." But the port receives
`actions` — the event's actor team — and a match has exactly two teams, so the `teammate` flag fully
DETERMINES each player's real team; resolving it is a derivation from context the port already consumes,
not a fabricated identity. The synthetic ids broke the action↔frame team join (`acting_team_attacks_rtl`,
ADR-028) → all-`<NA>` → ADR-051 D3 honest-NaN for every direction-dependent tracking feature. `shape_snapshots`
now resolves the real ids, falling back to the synthetic pair ONLY when the two teams cannot be resolved
(no `team_id` column, or not exactly two distinct teams), so the "don't claim identity we don't have"
principle is preserved exactly where it applies. This is the tracking-feature half of the enablement
working on SB360 — verified on real match 3986784 (1795/1795 `<NA>` → 1795/1795 resolved) and
regression-guarded from both sides in `tests/providers/statsbomb/test_parse.py`. No retrain: no bundled
model consumes SB360 snapshot frames.

## Consequences

- **No retrain, no re-materialization** — additive opt-in features; no default xfn list changed; the
  audit round-trip is byte-identical. The `shape_snapshots` team-id fix (Decision 6) changes SB360
  snapshot output but no bundled model trains on it.
- **Validated on real licensed data** — the loader parsers (home_team_id, fidelity threaded without
  inference, roster identity 100% resolved), `measure_match`, and the full driver CLI path
  (`main → for_each → reconcile → provenance`) all ran cleanly on the licensed corpus.
- **Player identity via roster, not freeze-frames** — SB360 freeze-frame rows stay anonymous
  (`snapshot_to_tracking_frames` numbers them), but `roster` + `events` give per-keeper identity.
- Decision 4 reinterprets §9 of the design spec; the reinterpretation is routed back to review.
