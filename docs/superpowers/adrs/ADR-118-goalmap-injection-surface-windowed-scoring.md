# ADR-118: GoalMap injection surface for windowed ghost scoring (extends ADR-055)

| Field | Value |
|---|---|
| **Date** | 2026-10-11 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

`resolve_defended_goals(frames) -> GoalMap` is a **whole-unit estimator**: the defended-goal end is the mean GK x per `(game_id, period_id, team_id)` over all the unit's frames, and the mean is the robustness. ADR-055 already states the rule — *"build ONCE per match, from the FULL frames … thread it into the per-frame functions"* — and records the cost of a smaller-sample estimator: a per-frame map disagrees with the per-match one on **7.1% of SkillCorner team-frames** (2.2% metrica, 0.0% sportec/gradientsports; `tracking/_gk_resolve.py`). The damage concentrates in sparse broadcast detection, which is exactly the provider class most likely to be scored in a resource-constrained path.

A downstream consumer (the luxury-lakehouse) must **window** Stage-2 ghost scoring to stay under a serverless per-group memory cap, so each scorer sees one frame-window rather than the whole unit. If the goal map is re-resolved inside each window it becomes a per-window estimator and the metric changes on those 7.1% of frames — not noise, a real output shift.

Implementing the request surfaced a **latent ADR-055 violation**: several public ghost-serving and decision entry points re-resolve the goal map **internally** for the goal-relative → frame-coordinate flip, independent of any map the engine had already built. `build_ghost_frames` resolved it **twice** — once for the §4.1 domain filter (`gkdv/_engine.py`) and again inside `serve_ghost_gk_positions` for the flip (`tracking/_ghost_gk.py`). Rest-defense mirrored the pattern (`build_restdefense_ghost_frames` + both its serves). Within a whole-unit call the two resolutions agree; under windowing they would both drift, and patching only one would leave the domain and the served positions keyed to different maps — a third, incoherent behaviour.

## Decision

Add an optional, keyword-only `goal_map: GoalMap | None = None` to the five public entry points that resolve the defended-goal end for ghost scoring — `gkdv.build_ghost_frames`, `tracking.serve_ghost_gk_positions`, `tracking.serve_ghost_outfield_positions`, `restdefense.build_restdefense_ghost_frames`, and `gk_decision.ReconstructedOptionSet` (the `match_ltr` branch) — and thread one injected instance into **both** the domain filter **and** every serve's frame-coordinate flip. `None` resolves internally from the frames and is **byte-identical** for every existing caller. A windowed caller pins the whole-unit `resolve_defended_goals(frames)` once and injects the same immutable `GoalMap` into every window.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. The original 2-edit ask (`build_ghost_frames` + `ReconstructedOptionSet` only) | Minimal diff | Leaves `serve_ghost_gk_positions`/`serve_ghost_outfield_positions` re-resolving per window; domain and served positions keyed to different maps | Does not achieve window-stable output and fails its own byte-identical intent on the 7.1% frames |
| B. Pass a scalar attacking-direction instead of a `GoalMap` | Narrower argument | Forks a second representation of "which way does this team attack" | Violates ADR-055's single-implementation rule directly |
| C. (chosen) Optional `goal_map=` threaded into every re-resolution site; `None` = recompute | One injection surface; honours "build once, thread in"; byte-identical default | Five public signatures grow a param; threading is a convention, not a gate | — |

## Consequences

### Positive

- Windowed Stage-2 ghost scoring is byte-identical to whole-unit scoring.
- ADR-055's "build once per match, thread in" rule now holds **through the serve layer**, where it was silently broken.
- The injection surface is discoverable in one place rather than inferred from five docstrings.

### Negative

- Five public signatures carry a new optional parameter.
- No gate enforces the threading: a future ghost-serving or flip site that re-resolves `resolve_defended_goals` internally, without exposing/accepting `goal_map=`, re-introduces the per-window drift. The convention is documented here and in each docstring; a maintainer adding such a site must honour it.
- Downstream windowed callers (the lakehouse) must precompute and inject the whole-unit map per window, and floor-bump to 4.132.0.

### Neutral

- `compute_rest_defense`'s Layer-1 `goal_map=` seam (which feeds only the sampler) is unchanged and was already complete.
- `ReconstructedOptionSet`'s `per_action_ltr` branch never touches the goal map, so it is unaffected.
- Byte-identical holds precisely because, on the `None` path, the engine threads `resolve_defended_goals(src)` into serves that previously recomputed `resolve_defended_goals(frames=src)` — the same value.

## Related

- **ADRs:** extends `ADR-055` (goal-map unification: one implementation, build once, thread in); relies on `ADR-019` (canonical id keys in `GoalMap`).
- **Requesting document:** `D:\Development\_reviews\2026-10-11-silly-kicks-goalmap-injection-handoff.md` (luxury-lakehouse windowing request) — no in-repo spec; handoff-driven.
- **Code:** `silly_kicks/gkdv/_engine.py`, `silly_kicks/tracking/_ghost_gk.py`, `silly_kicks/tracking/_ghost_outfield.py`, `silly_kicks/restdefense/_counterfactual.py`, `silly_kicks/gk_decision/_reconstruct.py`.
- **Release:** 4.132.0.

## Notes

The two-place recompute was, concretely, `gkdv/_engine.py` (`_pin_defended_goal(src)` for the domain filter) plus `tracking/_ghost_gk.py` (`resolve_defended_goals(frames)` for the `_defended > 50.0` flip); rest-defense mirrored it at `restdefense/_counterfactual.py` plus `tracking/_ghost_outfield.py` (the rearguard flip) and the keeper serve. Each is now gated `if goal_map is None: goal_map = resolve_defended_goals(...)` and, where the engine owns the map, the single instance is threaded into the serve call. Verified by TDD: a byte-identical leg (`None` and injected-whole-unit both equal the default) and a load-bearing leg (an inverted `GoalMap` moves the output; reverting the implementation turns the tests red with `TypeError: unexpected keyword argument 'goal_map'`).
