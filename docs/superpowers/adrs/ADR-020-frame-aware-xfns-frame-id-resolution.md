# ADR-020: Frame-aware xfns resolve frame_id by position, never via `.at` on a non-unique `action_id`

| Field | Value |
|---|---|
| **Date** | 2026-06-07 |
| **Status** | Accepted (silly-kicks 4.16.0) |
| **Deciders** | Karsten Nielsen (maintainer), silly-kicks part-deux review sessions |

(ADR-015 reserved by TF-17 PR-C. 016 = ghost-GK served estimator; 017 = time-base contract; 018 =
own-goal VAEP labels; 019 = id-dtype contract. This ADR is 020.)

## Context

VAEP gamestates (`silly_kicks.vaep.feature_framework.gamestates`) build the lagged slots `states[1]`,
`states[2]` by `actions.shift(i)` with the period/game boundary filled from the group's first action.
So a shifted slot **repeats the boundary action** — its `action_id` column is **non-unique**
(empirically `gamestates(actions)[1].action_id == [a0, a0, a1, a2, …]`).

Frame-aware xfns transformers (and the per-Series helpers lifted via `lift_to_states`) resolved each
action's linked frame with `pointers.set_index("action_id").at[aid, "frame_id"]`. On a non-unique
index `.at[aid]` returns a **Series**, so `pd.isna(...)` / `int(float(...))` raise
`ValueError: The truth value of a Series is ambiguous`. The same non-uniqueness makes provenance
merges (`actions[["action_id"]].merge(pointers, on="action_id")`) **fan out**
(`ValueError: Length of values (N) does not match length of index (M)`).

A behavioral probe over **every** `*_xfns` factory (run through real `gamestates()`) confirmed the
bug in **8 families**: `pitch_control`, `obso`, `pausa`, `space_creation`, `pressure`,
`cover_shadow`, `gk_influence`, `player_influence`. A `grep` for `.at[…]` did **not** capture all of
them (`pausa` reaches the bug through `add_obso`; `pressure` through an `action_id`-indexed kernel),
which is why a hand-listed seam inventory was rejected in favour of a behavioral gate. The
production/lakehouse path was unaffected because it uses the `add_*` aggregators on full action
streams (unique `action_id`), not the `*_xfns`/gamestates path — so the bug was latent.

## Decision

**Frame-aware code MUST resolve a linked `frame_id` by position, never via `set_index("action_id")`
`.at[aid]` on a possibly-non-unique `action_id`; and provenance/id merges on a slot MUST be
dup-safe.**

- The shared resolver `silly_kicks.tracking._kernels.resolve_frame_ids_by_position(actions, frames, *,
  links=None)` returns a positionally-aligned `float64` array of linked frame_ids (NaN if unlinked).
  Caller-supplied `links` (the `add_*` aggregator path, unique ids) are reindexed by `action_id`; the
  internal-link path re-keys to a unique positional surrogate before linking. It is **byte-equivalent**
  to the old `.at` lookup on unique ids (locked by `tests/tracking/test_structural_pass.py::
  TestResolveFrameIdsByPosition`), so retrofitting it into a helper shared by a safe aggregator and a
  broken xfns leaves the aggregator's behavior unchanged.
- Provenance merges on a slot dedup `pointers` on `action_id` first; `action_id`-indexed kernels
  (pressure) re-key to a unique positional surrogate on the internal-link path.
- A red-first **conformance gate** `tests/tracking/test_frame_aware_xfns_dup_action_id.py`
  auto-enumerates every `*_xfns` via `dir(features)`, runs each through a real dup-`action_id`
  gamestate, and a meta-assertion proves the parametrization equals the registered surface — so any
  future xfns is auto-covered and cannot reintroduce the bug. The gate discriminates the two dup
  symptoms from a fixture gap (so the retrofit fixes the bug, not the fixture) and **fails** (never
  skips) on an unconstructable factory.

## Consequences

- **Behavior change (Hyrum):** the 8 affected `*_xfns` previously raised when composed into a VAEP
  model via gamestates; they now produce values. This changes the VAEP feature matrix for any consumer
  using the xfns path → **retrain trigger**. The `add_*` aggregator path is byte-unchanged.
- New invariant is discoverable: the gate is the conformance test; the resolver docstring cross-links
  this ADR.
- The `_defensive_line_at_actions` kernel already used a positional (`_row_idx`) pattern for the same
  reason; this ADR generalizes that discipline to the whole frame-aware surface.

## Alternatives considered

- **Hand-listed seam inventory + per-site fix.** Rejected: incomplete (the `.at` grep missed `pausa`/
  `pressure`); a behavioral gate enumerates failures and guards the future surface.
- **Fix only the originally-reported families (cover_shadow / player_influence).** Rejected: the
  empirical sweep showed the bug is systemic (8 families); a partial fix leaves latent crashes.
