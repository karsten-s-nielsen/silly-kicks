# Change request: `acting_gk_from_frames` — resolve the acting team's GK from tracking (mirror of `defending_gk_from_frames`)

**Date:** 2026-07-01 · **From:** xT-GK analysis side (Karsten) · **For:** silly-kicks session to plan + implement (TDD)
**Type:** change request — requirements + acceptance. Small, composable utility.
**Related:** `silly_kicks/tracking/_gk_resolve.py` (`defending_gk_from_frames`, TF-13 — the exact mirror of this);
the 4.37.0 keeper-origin + 4.38.0 GK-identification fixes. **This is one half of a split fix** — the lakehouse-side
consumer (goal-kick actor override) is a separate handoff.

## Why (the problem this enables fixing)
On tracking providers, a **goal-kick's SPADL taker (`player_id`) is NULL**, so the AC layer fills the actor from the
**ball-carrier at the linked frame** (`ball_carrier_at_action`). For a goal-kick the linked frame has the ball at the
**downfield event location** (the same scatter 4.37.0 fixed for the *origin*), so the "carrier" is whatever outfielder
is near the ball 14–20 m downfield — varying every time. Result (verified on the recomputed public SkillCorner gold):
goal-kick `xt_gk` is credited to **29–35 different outfielders, ~1 each**, essentially never the keeper (0% match the
receiver, 0% the real keeper). Value + origin are correct; only the **credit** is wrong.

A goal-kick's taker is **unambiguously the acting team's keeper**. The lakehouse will override the goal-kick actor with
that keeper (mirroring how it already overrides carrier-derived *possession* for set-piece restarts via
`_fill_possession_from_set_piece_actions`, PR-S67). To do that it needs a **resolver for the acting team's GK from
frames** — which silly-kicks should own (GK-from-frames is domain logic that already lives in `_gk_resolve.py`).

## The requirement (S1)
Add a public resolver — the **mirror of `defending_gk_from_frames`** — that returns, per action, the **acting team's**
goalkeeper `player_id`:

```python
def acting_gk_from_frames(actions, frames, *, tolerance_seconds=0.2) -> pd.Series: ...
```
- Aligned to `actions.index`; dtype matches frames' `player_id` (object for kloppy/SkillCorner/Metrica, Int64 for GS).
- Same structure as `defending_gk_from_frames`, with **one inversion**: pick the GK whose `gk_team_id == action.team_id`
  (the acting team), instead of `!=` (the opposing team). Consider factoring the shared body so the two functions
  differ only by the team predicate.
- **Robustness to partial detection (important — this is broadcast tracking):** the goal is the acting team's keeper's
  **identity**, not their position at that instant. Post-4.38.0, `is_goalkeeper` is roster-trusted and set on the
  keeper's rows **throughout** the match, so resolve the acting team's GK **even when the keeper isn't detected in the
  exact linked frame** — e.g. fall back to the team's `is_goalkeeper` identity for that `(game_id, team_id[, period])`
  when the linked frame has no acting-team GK row. (A pure per-frame link, like `defending_gk_from_frames`, would
  return NaN on the ~40% of goal-kicks where the keeper isn't detected at the event frame — too lossy for this use.)
- **GK-sub safety:** if a `(game, team)` has more than one `is_goalkeeper` identity across the match (a keeper sub),
  prefer the one active around the action's time. Simplest acceptable fallback if that's hard: the GK whose rows are
  nearest-in-time to the action.

## Acceptance
- For a frame set with a known acting-team GK, `acting_gk_from_frames` returns that GK's `player_id` for the acting
  team's actions — **including goal-kicks where the keeper is not detected in the linked frame** (identity fallback).
- NaN where the action can't be resolved at all (no acting-team GK identity anywhere / NaN `team_id`).
- dtype matches frames' `player_id`; mirror the existing `defending_gk_from_frames` tests, plus:
  - a golden where the keeper is undetected at the goal-kick frame → still resolves (identity fallback, not NaN);
  - a GK-sub case (two keepers) → returns the time-appropriate one.
- **Regression:** `defending_gk_from_frames` unchanged (if you factor a shared helper, prove byte-identical output).

## Boundary / scope (what this CR is NOT)
silly-kicks provides the **resolver only** — a pure composable lookup, exactly like `defending_gk_from_frames`. It does
**not** fill or mutate actions (silly-kicks stays a pure `player_id` pass-through, PR-S67). **Deciding when to apply it**
— i.e. overriding the goal-kick actor — is the **lakehouse's** synthesis step (a separate handoff), scoped to
goal-kicks only. Do not add goal-kick-specific logic here; keep the resolver general (any acting-team GK lookup).
