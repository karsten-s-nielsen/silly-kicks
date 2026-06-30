# Design — SkillCorner keeper-origin resolution (broadcast-tracking domain fix)

**Date:** 2026-06-30 · **Author:** silly-kicks session · **Status:** design, pending cross-session review
**Implements:** `docs/superpowers/specs/2026-06-30-skillcorner-keeper-origin-resolution-CR.md` (the change request — the *what/why*; this doc is the *how*).
**Related:** `docs/superpowers/specs/2026-06-29-xtgk-pre-jeff-verification-handoff.md`, ADR-024 (xT-GK), ADR-019 (id-dtype), ADR-025 (restart enrichment), ADR-028 (action-LTR re-projection), ADR-034 (TF-23 SkillCorner builder).

---

## ⚑ POST-VALIDATION REVISION (4.37.0, supersedes §2/§3 below where they conflict)

Real-data validation (Databricks bronze + pining, owner-run) **revised the design before ship**:

1. **Distrust scope narrowed to GOAL-KICKS ONLY.** §3's ladder distrusted all GK distributions. The data
   showed the native origin of an **open-play GK pass IS the keeper** (native-vs-detected-keeper **0.4 m** — the
   ball is at the keeper when they release it), unlike a goal-kick (native is the broadcast ball **14–20 m**
   downfield). So open-play GK passes/throws **keep their native origin** (no distrust, no `unresolved` discard);
   distrust applies to **goal-kicks only**. The `gk_distribution_mask` param is retired; the engine distrusts
   `is_gk` rows. `unresolved` is now a rare residual.
2. **ADR-028 orientation fix added.** The detected keeper is sampled in the frame's home-attacks-right convention
   but consumed in action-LTR — `_tracking_gk_xy_detected` now point-reflects it for away-team actions
   (`acting_team_attacks_rtl`) before the clamp. The original home-only fixtures hid this; tests now cover home
   AND away across every tier.

Everything else in §1/§4–§11 (the fail-safe allowlist, S1 layered invariant, S4 guard, C1, regression gate)
ships as designed. ADR-024's 4.37.0 amendment is the canonical record.

---

## 1. Problem (restated, data-grounded)

For SkillCorner **broadcast** tracking, the SPADL action's native origin is the **broadcast ball-detection
location, not the keeper's position** — scattered across the pitch (goal-kick `start_x` min 0.8 / max 98.4 / SD 23.2;
own-box rate 51 % vs ~100 % for every other provider; passes 60 % own-box). `resolve_gk_geometry` trusts any
non-NaN `start_x` as `origin_source="native"` (confidence 1.0), so the keeper-distribution origin — and the
`xt_gk` `base`/`dzv` and keeper pressure/PEV computed at that origin — is corrupted.

The destination is **not** affected: the broadcast ball-event endpoint *is* genuinely where the ball went; the bug
is specifically "ball-location ≠ keeper-location" at the **origin**.

## 2. The trustworthiness trigger (the one new concept)

Native-origin trustworthiness is **provider-dependent**:

- **Full-tracking** (Gradient Sports / idsse / metrica): native GK origin is trustworthy. **Frozen path, unchanged.**
- **Broadcast** (SkillCorner): native GK origin is a ball-detection artifact → **distrusted** → resolved via a
  detection-aware ladder.

The trigger is the **provider**, not the action type. The *policy* decision (which providers distrust native) is
made at the `compute_xt_gk` edge — which already reads `frames["source_provider"]` to pick the completion-model
variant (`_xt_gk.py` ~L367). The *mechanism* (the ladder) lives in the geometry engine. This matches the established
"policy at the edge, mechanism in the shared engine" discipline (cf. `add_restart_coordinates` tripwire vs the pure
`resolve_restart_geometry`).

### 2.1 Seam

`resolve_gk_geometry` / `resolve_restart_geometry` gain an explicit opt-in:

```python
def resolve_restart_geometry(actions, *, frames=None, links=None,
                             impute_types=None, distrust_native_origin=False) -> pd.DataFrame: ...
def resolve_gk_geometry(actions, *, frames=None, links=None,
                        distrust_native_origin=False) -> pd.DataFrame: ...
```

`distrust_native_origin=False` (default) ⇒ **byte-identical to today**. `compute_xt_gk` sets it `True` when the
single resolved provider distrusts native. A small pure helper holds the provider→trust mapping (mirrors
`variant_key_for_provider`), so a future broadcast provider is covered by default.

**Fail-safe default (allowlist, not denylist).** Unknown / `None` / future providers default to **distrust**; only
the known **full-tracking** providers are explicitly allowlisted as trusted. A denylist (`!= "skillcorner"`) is the
unsafe default — a new broadcast source (or `None`) would silently use native ball-event origins until someone
notices the S4 warnings. This mirrors the `access_tier` privacy default (unknown → most-restrictive):

```python
_NATIVE_ORIGIN_TRUSTED = frozenset({"gradientsports", "idsse", "metrica", "sportec", "statsbomb", "wyscout"})

def native_origin_is_trusted(provider: str | None) -> bool:
    return provider is not None and str(provider).lower() in _NATIVE_ORIGIN_TRUSTED  # default unknown -> distrust
```

`compute_xt_gk` computes `distrust = not native_origin_is_trusted(resolved_provider)` from the same
`source_provider` lookup it already performs.

- **Regression gate preserved:** the allowlist names every currently-tested full-tracking provider → those stay on
  the frozen native-first path → byte-identical. A future full-tracking provider that defaults to distrust is *safe*,
  not wrong: the ladder's tier-1 resolves the keeper from tracking ≈ the native position anyway. S4 stays the
  net, but the **default** is now safe too.
- **Fixture note:** all production frames carry `source_provider` (the converters set it — `skillcorner.py` L141 =
  `"skillcorner"`, the GS/sportec adapters set theirs). Hand-built synthetic xt_gk fixtures **must set an explicit
  `source_provider`** so a `None`-defaults-to-distrust frame doesn't silently flip a "GS" golden test onto the
  broadcast path. The discrimination test (§7) asserts the allowlist→frozen-path equivalence explicitly.

## 3. S3 — the resolution ladder (core change)

When `distrust_native_origin=True`, GK-distribution rows **skip the native tier entirely** and run a
**detection-aware** ladder. (When `False`, the existing native-first ladder runs unchanged.)

### 3.1 Ladder, by action type (broadcast mode only)

| Action type | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Goal-kick (`type_id == goalkick`) | detected GK position, **clamped** to goal area `x ≤ 16.5` | rule-point `(5.5, 34)` | — |
| Open-play GK pass / throw-in (actor is the acting team's GK) | detected GK position, **no clamp** | — | `unresolved` (NaN; **no impute**) |

- Goal-kick has an unambiguous physical constraint (taken from the goal area), so an off-position detected keeper is
  clamped out → rule-point. Open-play has no such constraint (a sweeper-keeper at halfway is a legitimate origin),
  so it is accepted as-is or left `unresolved` — **never imputed with a guess** (analysis-side confirmed: tier-3 is
  flag-`unresolved`, do not add a weaker prior).
- The **acting-team GK** for the open-play branch is identified exactly as `_gk_distribution_mask` already does
  (frames `is_goalkeeper` ∧ actor `(game,team,player)` match; ADR-019 dtype-safe ids).

### 3.2 The detection-aware helper

New helper (sibling of the existing single-frame `_tracking_gk_xy`):

```python
def _tracking_gk_xy_detected(actions, frames, links, *, window_s=1.0, clamp_goal_area: bool) -> np.ndarray:
    # for each in-scope row:
    #   linked frame -> search +/- round(frame_rate * window_s) frames within (game_id, period_id)
    #   keep only frames where the acting-team GK's OWN row has `visibility` truthy (real detection, not interpolation)
    #   pick nearest-in-time, ties -> at-or-before (origin wants pre-release keeper position)
    #   clamp_goal_area=True  -> require gx <= 16.5 else NaN (goal-kick)
    #   clamp_goal_area=False -> accept (open-play); only the S1 within-pitch invariant applies
    #   none detected in window -> NaN (fall through)
```

- **Detected** = the keeper's own frame row has `visibility` truthy. `visibility` is coerced with the existing
  `_truthy_bool` (never `.astype(bool)` a provider string — ADR-019 trap).
- `window_s` (default `1.0`) is a **tunable param**, chosen to match measured recovery (~58 % detected at the action
  frame → ~70 % within ±1 s) without much position drift.
- Frame-rate → frame-count uses the frames' `frame_rate` column (SkillCorner ~10 Hz); the window is
  `round(frame_rate * window_s)` frames each side.
- The **at-or-before bias** is the tie-break only (nearest-in-time wins; ties resolve to the earlier/at-or-before
  frame). No additional asymmetry unless validation shows post-release drift matters.

### 3.3 Provenance

Per-row `origin_source ∈ {tracking_gk, goalkick_prior, unresolved}` (the `resolve_gk_geometry` shim already maps the
engine's `restart_prior` → `goalkick_prior`). `origin_confidence` keeps the existing scale (`tracking_gk` 0.7,
`goalkick_prior` 0.2, `unresolved` 0.0). `xt_gk_origin_x/_y` (Item-1 audit columns, already shipped 4.36.0) surface
the resolved coords for every row.

### 3.4 Validate-then-maybe: open-play misdetection bound

The open-play no-clamp path ships **bare**. After the fix, validate that pass origins localize to the own half
(today's ~15 % opp-half pass origins are a ball-event artifact and should largely vanish once tier-1 uses keeper
detection). **Only if** attacking-half origins persist, add a *generous* own-half sanity bound (keeper origin within
the defensive ~60 % of the pitch; beyond → `unresolved`/flagged, **never clamped** — same don't-mask rule). **Do not
pre-build this.** Let the post-fix distribution decide.

### 3.5 Destination — unchanged

Distrust-native is **origin-only**. The destination keeps the existing `native → next_event` ladder for all
providers (the broadcast ball endpoint is correct). NaN / off-camera destinations are already handled by that ladder;
no new logic.

## 4. S1 — transform invariant (verification + invariant, not a rewrite)

The existing transform in `tracking/skillcorner.py` (`x + 52.5`, `y + 34.0`) is **already correct** (center-origin
±52.5 / ±34 → SPADL [0,105] × [0,68]). The gap is the **within-pitch invariant** — and it must be **warn-and-flag
per-row + a batch rate-gate**, NOT a hard per-row assertion (consistent with S4: one outlier noise point or an
out-of-play ball must not crash a whole match's frame conversion).

**Mechanism (two tiers, mirrors S4):**
- **Per-row:** a player/ball coord outside `[−TOL_xy, 105 + TOL_xy] × [−TOL_xy, 68 + TOL_xy]` →
  `warnings.warn(...)` (`stacklevel=2`) + a machine-observable count in `TrackingConversionReport`
  (new field, e.g. `n_gross_off_pitch`). **Never clamp**, never crash on a single row.
- **Batch / CI rate-gate:** a *systematic fraction* off-pitch (a 123-type transform break — wrong sign, wrong
  origin) → **hard fail**. This is where "the transform is actually broken" shows up, vs a single noisy row. Lives
  as a CI test asserting the off-pitch rate stays below a small threshold on the pining corpus.

**TOL is calibrated from the real bronze range, NOT hand-picked.** The earlier "7.5 m floor" was wrong: review
measured the real SkillCorner bronze at native min −60.1 / max 63.5 (→ SPADL ≈ −7.6 / 116) for players, and the
**ball legitimately goes further** off-pitch (out-of-play for throw-ins / goal-kicks). So:
- **Re-measure** the real bronze player + ball off-pitch range on the pining corpus (DGX) before fixing `TOL_xy`.
- Provisional from the review measurement: players reach ~11 m past the goal line → `TOL_xy ≳ 15 m` (player) with
  margin; the **ball gets a wider bound** (or is excluded from the player bound and rate-gated separately).
  Finalize both empirically — this is a §11 deferred-calibration item, not a guessed constant.
- Scope: the AC/xt_gk frame path (`convert_to_frames`) only. The lakehouse `fct_tracking_frames` transform (L4) is
  out of scope.

## 5. S2 — detection bit preservation (already carried — add a guard)

`is_visible` already survives `convert_to_frames` as the `visibility` column (`tracking/schema.py:28` defines it on
`TRACKING_FRAMES_COLUMNS`; converter L109 `players["visibility"] = players.pop("is_visible")`). No new plumbing.

The gap is **protection**: add an invariant test that `visibility` survives `convert_to_frames` end-to-end and is
**not interpolated/smoothed away** by `preprocess` (which operates on x/y/speed, not `visibility` — confirm and
guard). This is what S3's tier-1 detection gate depends on.

## 6. S4 — loud-but-observable validation

A `native` goal-kick origin beyond the penalty area (`x > 16.5` in LTR own-half coords) is physically implausible.
New guard helper next to the resolver, run on the `compute_xt_gk` path (not only the `add_restart_coordinates` edge,
where the existing tripwire lives):

- **Runtime:** `warnings.warn(...)` (`stacklevel=2`) **+ a machine-observable count** — `XtGkReport` gains
  `n_native_goalkick_out_of_region: int`. The flag is **never a bare log** (a warning nobody aggregates is
  effectively silent — the failure mode behind the stale-grid / mocked-Spark incidents). Warn-and-flag, **not** a
  hard exception (one bad row must not crash a match/dataset).
- **CI hard backstop (optional, recommended):** a rate-gate test asserting no provider exceeds a small % of
  out-of-region native goal-kick origins — catches a systematic ingestion regression (a provider silently feeding
  ball-as-origin, or a GS ingestion break) loudly, where a hard failure belongs.
- **Goal-kick-only** (open-play has no physical origin constraint — covered by §3's detection path + the
  validate-then-maybe own-half bound). Since SkillCorner goalkicks no longer use native, this mainly defends
  **unknown future providers** and surfaces GS data-quality outliers as warnings (not reverts).

## 7. Regression gate (hard requirement)

GS / idsse / metrica keeper-origin resolution **and** `xt_gk` values must be **byte-identical** after this change.
Guaranteed structurally by the default-off `distrust_native_origin` flag. Pinned by:

- the existing `resolve_gk_geometry` parity tests (frozen path untouched);
- a new discrimination test: same actions/frames, broadcast-mode vs full-tracking-mode → only the broadcast path
  changes; full-tracking path identical to the no-flag baseline.

## 8. Testing

- **Golden tier test** (production-realistic SkillCorner geometry): synthetic center-origin frames + `visibility`
  bits + scattered ball-event native origins, exercising each tier:
  1. goal-kick, keeper detected in box → `tracking_gk` (clamped);
  2. goal-kick, keeper not detected → `goalkick_prior` `(5.5,34)`;
  3. open-play GK pass, keeper detected at halfway → `tracking_gk` (no clamp, accepted);
  4. open-play GK pass, no detection → `unresolved` (NaN, no impute).
- **S1 invariant test**: in-tolerance behind-goal keeper + out-of-play ball pass (no crash); gross off-pitch row
  warns + increments `TrackingConversionReport.n_gross_off_pitch`; a systematically-broken transform trips the
  batch rate-gate.
- **S2 guard test**: `visibility` survives `convert_to_frames` and `preprocess`.
- **S4 test**: out-of-region native goalkick origin warns + increments the `XtGkReport` count; optional rate-gate.
- **Coherence test (review low-1)**: assert the *resolved* origin feeds **all** of `compute_xt_gk` — not just
  `base`/`dzv` but also the keeper **pressure**/PEV and the **RAV completion** features (that's the whole point: they
  were computed at the wrong origin). `compute_xt_gk` reads `sub_geom` for both, so this is likely already correct —
  the test pins it: changing the resolved origin must move pressure/PEV + completion, not only base/dzv.
- **Regression**: GS/idsse/metrica/sportec byte-identical (parity + discrimination tests; the allowlist→frozen-path
  equivalence is asserted explicitly per §2.1).
- **Perf (review low-2)**: the per-row ±window detection search is fine at ~100 keeper actions/match, but confirm it
  is not a Python-loop hotspot across the full pining corpus; **vectorize if it is** (the existing `_tracking_gk_xy`
  is already a per-row loop, so this is the same cost profile — measure before optimizing).
- Full `[test]` install · `ruff format --check` · `ruff check` · `pyright silly_kicks/` (full package) ·
  `/final-review` before the single commit.

## 9. Release & ownership

- **Minor version bump** + **ADR-024 amendment** (xT-GK geometry contract: provider-aware origin trust).
- Not a forced VAEP retrain (xt_gk is opt-in, in no default xfn list) but an **xt_gk serve-output change for
  SkillCorner** → lakehouse re-materializes (same framing as PR-S91). silly-kicks **ships first**; the lakehouse
  adopts the release.
- **Out of scope (lakehouse L4):** the `xt_gk_origin_source` mart enum, `unresolved`→NULL rendering, `access_tier`,
  and the `fct_tracking_frames` re-point. This change stays in the converter (`tracking/skillcorner.py`) + the
  resolver (`tracking/_gk_geometry.py`) + the `compute_xt_gk` seam (`tracking/_xt_gk.py`) + `XtGkReport`.

## 10. Files touched

- `silly_kicks/tracking/_gk_geometry.py` — `distrust_native_origin` param; `_tracking_gk_xy_detected`; the broadcast
  ladder branch; the S4 out-of-region guard helper.
- `silly_kicks/tracking/_xt_gk.py` — `native_origin_is_trusted` mapping; pass the flag; `XtGkReport`
  `n_native_goalkick_out_of_region`.
- `silly_kicks/tracking/skillcorner.py` — S1 within-pitch invariant (per-row warn-and-flag, no crash).
- `silly_kicks/tracking/schema.py` (or the report dataclass) — `TrackingConversionReport.n_gross_off_pitch`.
- `tests/tracking/` — golden tier test, S1 invariant, S2 guard, S4 guard + optional rate-gate, regression
  discrimination.
- `pyproject.toml` / `__init__.py` / `TODO.md` / `CHANGELOG.md` — version bump (all four must match).
- `docs/superpowers/adrs/ADR-024-xt-gk.md` — amendment.

## 11. Open / deferred (not pre-built)

- Open-play own-half misdetection bound (§3.4) — validate-then-maybe.
- `window_s` and the at-or-before bias are tunable; defaults `1.0` / ties-earlier ship as-is.
- **S1 `TOL_xy` (player) + the ball bound (§4)** — calibrate from the re-measured real bronze player+ball off-pitch
  range on the pining corpus (DGX); do not ship a guessed constant.
- Exact CI rate-gate thresholds (§4 off-pitch rate, §6 out-of-region rate) — pick from the observed rates on the
  pining corpus.
