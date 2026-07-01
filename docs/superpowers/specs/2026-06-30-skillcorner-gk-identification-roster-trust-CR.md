# Change request: SkillCorner GK identification — trust the native roster, don't re-derive per batch

**Date:** 2026-06-30 · **From:** xT-GK analysis side (Karsten) · **For:** silly-kicks session to plan + implement (TDD)
**Type:** change request — requirements + acceptance. The *how* is yours to plan; this is the *what* and *why*.
**Separate from** the 4.37.0 keeper-*origin* fix (that was *where* the keeper is). This is *which actions are keeper
actions* — a distinct, pre-existing defect surfaced when the public SkillCorner `xt_gk` was first inspected post-recompute.
**Related:** `docs/superpowers/specs/2026-06-30-skillcorner-keeper-origin-resolution-CR.md` (the origin fix);
`silly_kicks/tracking/skillcorner.py`, `_gk_identification.py`, `_xt_gk.py` (`_gk_distribution_mask`).

## The bug (data-grounded)
On the recomputed public SkillCorner gold, `xt_gk` (a **goalkeeper** metric) is computed for **19–24 distinct players
per match** — essentially both full squads — instead of the ~1–2 keepers. Gradient Sports scores ~1 player/match
(correct). So the **SkillCorner GK-distribution identification is broken**: `_gk_distribution_mask` scores open-play
passes by ~15 non-keepers/team because the frames' `is_goalkeeper` flag is set on ~15 players/team.

## Root cause (verified on real data, not inferred)
The lakehouse AC dispatch builds frames in **250-frame batches** and calls `skillcorner.convert_to_frames` **per
batch**. SkillCorner's `convert_to_frames` (`skillcorner.py` ~L216–228) **discards the clean native roster
`is_goalkeeper` and re-derives it positionally** via `derive_goalkeepers` (using the native flag only for the
`is_goalkeeper_source` label). Positional derivation is fine on a full match but **unstable on a 25-second window**:

| Check (real SkillCorner bronze, 10 matches) | Result |
|---|---|
| Roster (`skillcorner_matches`, `position_acronym='GK'`) | **1 GK/team, every match** (clean) |
| `derive_goalkeepers` Stage 2a, **full match** | **1/team** (clean — the algorithm is *not* broken) |
| Same Stage 2a, **per 250-frame batch** | **15.2 distinct players/team** flagged (14.2 non-keepers) |
| `xt_gk`-scored distinct players / SkillCorner match (gold) | **19–24** (matches the per-batch union) |

Mechanism: Stage 2a flags every candidate with `pa_dwell ≥ 0.40` and `dist_mean < 20 m`, using a **symmetric**
(both-penalty-areas, distance-to-nearest-goal) test. In a 25 s window with sparse broadcast detection, whoever is
transiently parked near *either* goal (a defending CB, an attacking forward) qualifies; across ~280 batches/period
~15 different players/team get flagged. **GS and sportec/idsse are immune because their converters trust the native
roster GK** (no re-derivation → stable across batches).

## The fix (S1) — trust the native roster `is_goalkeeper` for SkillCorner
SkillCorner ships a reliable per-player GK role (verified exactly 1/team). **Prefer it; derive only as a fallback** —
exactly as `gradientsports.py` and `sportec.py` already do (set `is_goalkeeper` from the roster, source `"native"`, no
`derive_goalkeepers`). Concretely, in `skillcorner.convert_to_frames`:
- When the input `is_goalkeeper` is a valid native roster flag (≥1 GK per `(game_id, team_id)`), **use it as-is**,
  set `is_goalkeeper_source = "native"`, and **skip `derive_goalkeepers`**.
- Only fall back to `derive_goalkeepers` for a `(game, team)` whose native flag is **absent/empty** (0 GK) — a
  data-quality edge, not the norm.
- This makes SkillCorner **batching-immune**: the roster flag is identical in every batch, so per-batch frame building
  yields the same `is_goalkeeper` as full-match.
- Robust to GK subs: trusting the roster flags both the starting and sub keeper (both *are* keepers) — correct.

## The guard (S2) — loud, observable, so this can't silently recur
Emit a **machine-observable** signal when `is_goalkeeper` resolves to an implausible per-`(game, team)` count
(e.g. `> 2` or `0`): `warnings.warn(stacklevel=2)` **plus a countable field** on the conversion report
(`TrackingConversionReport`), never a bare log. The whole-squad contamination went undetected until a downstream
cohort anomaly — a per-(game,team) GK-count check at the converter would have caught it immediately. (Naturally never
fires once S1 trusts the 1/team roster, but it protects every provider/path that *does* derive.)

## Acceptance
- SkillCorner `is_goalkeeper` = the roster GK (~1–2/team), **identical under full-match vs small-window (batched)
  invocation** — add a test that a 250-frame slice yields the same `is_goalkeeper` set as the full match.
- The GK-distribution mask scores only keepers (~1–2 acting players/match), not the squad.
- `is_goalkeeper_source = "native"` for SkillCorner when the roster flag is present (derive only when absent).
- S2 guard: a synthetic frame set with an over-flagged GK count warns + increments the report count.
- **Regression gate:** GS / sportec / idsse / **metrica** keeper identification **unchanged** (this is SkillCorner-only;
  Metrica still derives — see below).

## Out of scope (tracked separately — do NOT fix here)
The deeper latent issue: **per-batch positional derivation is unsound for providers that *must* derive (Metrica —
anonymized, no roster).** Metrica is contaminated the same way and trusting-the-roster can't fix it; it needs GK
derivation run **once per full match**, not per 250-frame batch — a split silly-kicks (separable derive-once API /
accept pre-derived picks) + lakehouse (derive once, feed `is_goalkeeper` into the per-batch builds) change. Flag it as
a follow-up; this CR is the SkillCorner roster-trust fix only.

## Boundary / ownership
silly-kicks owns converter GK identification (GS/sportec already do roster-trust here; SkillCorner is the outlier that
re-derives). The lakehouse needs **no** SkillCorner change — it already feeds the native `is_goalkeeper` in the
post-join bronze (`skillcorner_matches` roster, `position_acronym='GK'`); confirm that flag arrives clean and this CR
simply stops `convert_to_frames` from discarding it.
