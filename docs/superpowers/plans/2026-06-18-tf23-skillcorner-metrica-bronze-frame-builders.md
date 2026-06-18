# TF-23 SkillCorner + Metrica bronze→frame builders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the owner runs inline, not via subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two pure, bronze-consuming tracking frame builders (`tracking.skillcorner` / `tracking.metrica`) that single-source the SkillCorner/Metrica coordinate+orientation+clock transform currently duplicated three ways, recover SkillCorner `ball_z`, and orient via a geometric net promoted from lakehouse ADR-053 — proven correct by a kloppy-independent event-anchored y-identity gate.

**Architecture:** Each builder is a pure `pd.DataFrame → (pd.DataFrame, TrackingConversionReport)` function mirroring `tracking.sportec`/`tracking.gradientsports`: shape bronze → rescale to SPADL 105×68 → recover z/visibility → period-relative clock (single-sourced constant) → id-namespace → GK-derive → speed → orient. Orientation is a new public `orient_frames_to_ltr_by_geometry` (schema-adapted port of ADR-053's `correct_frames_to_home_ltr`: per-period home-GK-median-x anchor, point-reflect mis-oriented periods, idempotent). The kloppy gateway stays the structural oracle; the event-anchored action↔frame y-identity check (via silly-kicks' own `link_actions_to_frames`) is the primary Gate-C closer.

**Tech Stack:** Python, pandas, numpy. silly-kicks `tracking` namespace. pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-06-18-tf23-skillcorner-metrica-bronze-frame-builders-design.md` (Revision 3).

**Decision record to author in this PR:** `docs/superpowers/adrs/ADR-034-skillcorner-metrica-bronze-builders.md`.

**Branch / version:** `pr-s98-tf23-skillcorner-metrica-bronze-builders` off `main`; version bump **4.32.0 → 4.33.0** (additive minor). NO worktree (project policy). **ONE commit at the end after `/final-review` + explicit owner approval** (project commit policy — no per-task commits).

**Test command:** `python -m pytest tests/ -m "not e2e" -v --tb=short` (full suite); per-test `python -m pytest <path>::<name> -v`.

### Review round 2 — changes integrated (2026-06-18)

Lakehouse plan-review fixes, all verified against real source/data:
1. **Metrica builder vectorized** — the `iterrows()` + per-row `json.loads` shape (the exact anti-pattern the lakehouse just removed; banned at tracking scale) is replaced by a vectorized `.map(json.loads)` + `explode` + `pd.DataFrame(...tolist())` path, guarded by an **iterrows-spy structural test** (no wall-clock assert, per the project's structural-guard convention). — *Task 4*
2. **Task 0 precondition** — a signature/existence check of every assumed silly-kicks symbol the builders + keystone gate depend on (`link_actions_to_frames`, `_action_orientation.*`, `derive_goalkeepers`, `_derive_speed`, `orient_frames_to_ltr`, `require_et_direction`, `TrackingConversionReport` fields), so drift is caught on line one, not at Task 6. — *Task 0*
3. **Metrica clock fixed (real-data-verified)** — Databricks query showed Metrica sample games use **mixed** raw clocks (SG1/SG2 continuous P2≈2850/2717; SG3 period-relative P2≈0). The SkillCorner nominal offset is wrong for all three; "no rebasing" breaks SG1/SG2. Metrica now rebases via **per-`(period)` min-timestamp subtraction** (→ period-relative, matches kloppy); `metrica.py` no longer imports `spadl.skillcorner._PERIOD_START_SECONDS` (resolves the cross-wire smell). SkillCorner keeps the nominal constant (matches its events converter; SK P2 min is exactly 2700). — *Tasks 3, 4*
4. **SkillCorner native-GK-survives test** — orientation anchors on home-GK x; a unit test pins that `derive_goalkeepers` keeps SkillCorner's authoritative native GK (Tier-1 roster-validated, PR-S86), so the anchor is sound. — *Task 3*
5. **Gate asserts per-`(team, period)` residuals** — not a global median (which a sub-50% ET mis-orientation could slip under). — *Task 6*
6. **Fixture provenance pinned** — the post-`⋈ skillcorner_matches` SkillCorner bronze + Metrica bronze fixtures are **extracted from Databricks** `soccer_analytics.bronze` (documented `SOURCE_SHA`, DFL-slice precedent), NOT reconstructed in silly-kicks (which would drag the O5 join upstream). — *Task 6*
7. Idempotence test adds a **pure-coordinate** assertion (second pass, label excluded); velocity-flip-untested-against-default-builder noted. — *Task 2*

### Review round 3 — changes integrated (2026-06-18)

Final plan-review fixes, verified against real source/data:
1. **Metrica GK flag bug fixed (the must-fix).** Verified `bronze.metrica_tracking.gk_jersey_numbers` is a **flat list** (`["11","25"]`, no team split) AND that `derive_goalkeepers` **ORs, never clears** (`_gk_identification.py:163-169`) — so a team-agnostic `jersey.isin(...)` mis-flag would persist into the orientation anchor. Fix (aligned with "only bend so far for Metrica"): **Metrica seeds NO native GK** — `is_goalkeeper=False`, the validated positional algorithm derives it (Metrica is Tier-2/anonymized, ADR-007; positional is its GK authority), source always `"derived"`. The flat list becomes an **observability count cross-check** (warns on disagreement), not a flagging input. SkillCorner keeps its authoritative per-player roster flag. New **shared-GK-number collision fixture + test** (`test_gk_not_flagged_by_shared_jersey_number`) proves the deep players, not the jersey-collision outfielders, are flagged. — *Task 4*
2. **SkillCorner nominal-clock caveat documented** — SK intentionally follows its events converter's nominal convention (period-relative-from-0 not strictly held under stoppage); benign because frames/events offset identically (shared-constant guard). — *Task 3 (Confirm 1)*
3. **Metrica clock↔events parity** — commented that the event-anchored gate is the *sole* guard (no structural test, by nature; a mismatch fails it loud). — *Task 4 (Confirm 2)*
4. Structural guard also catches `apply(axis=1)`; malformed-player drop documented as accepted for frozen clean Metrica data. — *Task 4 (nits)*

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `silly_kicks/tracking/direction.py` | Modify | Add `orient_frames_to_ltr_by_geometry` (the promoted geometric orienter). |
| `silly_kicks/tracking/schema.py` | Modify | Add `SKILLCORNER_TRACKING_FRAMES_COLUMNS` / `METRICA_TRACKING_FRAMES_COLUMNS` aliases. |
| `silly_kicks/tracking/skillcorner.py` | Create | SkillCorner bronze→frame builder + `EXPECTED_INPUT_COLUMNS`. |
| `silly_kicks/tracking/metrica.py` | Create | Metrica bronze→frame builder + `EXPECTED_INPUT_COLUMNS`. |
| `silly_kicks/tracking/__init__.py` | Modify | Export `skillcorner`, `metrica` submodules + `orient_frames_to_ltr_by_geometry` + the two schema aliases. |
| `tests/tracking/test_orient_by_geometry.py` | Create | Orienter unit + idempotence + ET + mirrored-ADR-053 behavior. |
| `tests/tracking/test_skillcorner_builder.py` | Create | SkillCorner builder unit/invariants (rescale, ball_z recovery, clock single-source, guards). |
| `tests/tracking/test_metrica_builder.py` | Create | Metrica builder unit/invariants (rescale, JSON explode, z=NaN, GK-from-jerseys). |
| `tests/tracking/test_builder_event_anchored_gate.py` | Create | **Keystone**: committed-fixture action↔frame y-identity (both teams, off-centre, metrica-y named). |
| `tests/tracking/test_builder_kloppy_parity_e2e.py` | Create | Owner-gated `@e2e` parity-to-oracle (kloppy vs builder, same match). |
| `tests/datasets/tracking/sk_metrica_builder/` | Create | Committed bronze slices + golden frames + actions slice (captured during execution). |
| `docs/superpowers/adrs/ADR-034-skillcorner-metrica-bronze-builders.md` | Create | Decision record. |
| `NOTICE` | Modify | Attribute the ADR-053 geometric-orientation method (cross-repo provenance). |
| `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` | Modify | Version bump 4.33.0 (hard gate: all four match). |

---

## Task 0: Branch + environment

- [ ] **Step 1: Create the feature branch off main**

Run:
```bash
git checkout main && git pull && git checkout -b pr-s98-tf23-skillcorner-metrica-bronze-builders
```

- [ ] **Step 2: Sync the dev environment (session-start policy)**

Run:
```bash
pip install -e ".[test]"
```
Expected: installs cleanly; `python -c "import silly_kicks.tracking as t; print(t.__name__)"` prints `silly_kicks.tracking`.

- [ ] **Step 3: Precondition — verify every assumed symbol exists with the expected shape**

The builders and the keystone gate are written against existing silly-kicks symbols. Confirm them BEFORE writing any code (drift here makes the plan's code dead-on-arrival). Run:

```bash
python -c "
import inspect, dataclasses as dc
from silly_kicks.tracking.utils import link_actions_to_frames, _derive_speed, orient_frames_to_ltr, play_left_to_right
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
from silly_kicks.tracking._gk_identification import derive_goalkeepers
from silly_kicks.tracking.direction import require_et_direction, home_attacks_right_per_period
from silly_kicks.tracking.schema import TrackingConversionReport, KLOPPY_TRACKING_FRAMES_COLUMNS
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS
from silly_kicks.spadl import config as cfg
# arity / kw-only checks
s = inspect.signature(link_actions_to_frames); assert {'actions','frames','tolerance_seconds'} <= set(s.parameters)
s = inspect.signature(reproject_to_action_ltr); assert {'df','flip_mask','x_cols','y_cols'} <= set(s.parameters)
s = inspect.signature(derive_goalkeepers); assert 'frames' in s.parameters
# report dataclass fields the builders construct
fields = {f.name for f in dc.fields(TrackingConversionReport)}
need = {'provider','total_input_frames','total_output_rows','n_periods','frame_coverage_per_period','ball_out_seconds_per_period','nan_rate_per_column','derived_speed_rows','unrecognized_player_ids','n_teams_gk_derived','derived_gk_picks'}
assert need <= fields, need - fields
assert len(KLOPPY_TRACKING_FRAMES_COLUMNS) == 20
assert _PERIOD_START_SECONDS[2] == 2700.0
assert cfg.field_length == 105.0 and cfg.field_width == 68.0
print('PRECONDITION OK')
"
```
Expected: `PRECONDITION OK`. If any assertion fails, STOP and reconcile the plan's code against the real signature before proceeding.

---

## Task 1: Schema aliases

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (after `SPORTEC_TRACKING_FRAMES_COLUMNS`, ~line 43)
- Test: `tests/tracking/test_skillcorner_builder.py` (created in Task 3; alias is asserted there)

- [ ] **Step 1: Add the two schema aliases**

In `silly_kicks/tracking/schema.py`, immediately after the `SPORTEC_TRACKING_FRAMES_COLUMNS` definition, add:

```python
SKILLCORNER_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""SkillCorner native bronze→frame output: object identifiers (SkillCorner numeric
ids are stringified to match the SPADL ``player_id_native`` convention and the
kloppy-gateway oracle). Same shape as the kloppy variant."""

METRICA_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Metrica native bronze→frame output: object identifiers (``"Home"``/``"Away"`` team
labels + roster-mapped player ids). Same shape as the kloppy variant."""
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from silly_kicks.tracking.schema import SKILLCORNER_TRACKING_FRAMES_COLUMNS, METRICA_TRACKING_FRAMES_COLUMNS; print(len(SKILLCORNER_TRACKING_FRAMES_COLUMNS))"
```
Expected: `20`

---

## Task 2: The geometric orienter (`orient_frames_to_ltr_by_geometry`)

Schema-adapted port of lakehouse `analytics.action_context.pipeline.correct_frames_to_home_ltr` (ADR-053). **Acceptance oracle = ADR-053's behavior** (mirrored below as concrete cases). Differs from the lakehouse original only in: constants source from `spadlconfig`; **silent on normal per-period flips** (in the library orientation is the builder's owned operation, not a surfaced correction) — warns only on a missing-GK-anchor period, raises only on zero-home-match (ADR-019).

**Files:**
- Modify: `silly_kicks/tracking/direction.py`
- Test: `tests/tracking/test_orient_by_geometry.py`

- [ ] **Step 1: Write the failing tests (asymmetric+extreme ground-truth, idempotence, ET, guards)**

Create `tests/tracking/test_orient_by_geometry.py`:

```python
"""orient_frames_to_ltr_by_geometry — promoted ADR-053 geometric frame-LTR net."""
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry


def _frame(period, team, player, x, y, is_gk=False, is_ball=False):
    return {
        "game_id": "g1", "period_id": period, "frame_id": 0, "time_seconds": 0.0,
        "frame_rate": 10.0, "player_id": player, "team_id": team,
        "is_ball": is_ball, "is_goalkeeper": is_gk, "x": x, "y": y, "z": np.nan,
        "speed": 1.0, "speed_source": "derived", "ball_state": None,
        "team_attacking_direction": None, "confidence": None, "visibility": None,
        "source_provider": "skillcorner", "is_goalkeeper_source": "native",
    }


def _two_period_match(home_gk_x_p1, home_gk_x_p2):
    """home GK + an away outfield marker per period; asymmetric/extreme positions."""
    rows = [
        _frame(1, "H", "hgk", home_gk_x_p1, 5.0, is_gk=True),
        _frame(1, "A", "afw", 100.0, 60.0),
        _frame(1, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(2, "H", "hgk", home_gk_x_p2, 5.0, is_gk=True),
        _frame(2, "A", "afw", 5.0, 60.0),
        _frame(2, "A", "agk", 5.0, 34.0, is_gk=True),
    ]
    return pd.DataFrame(rows)


def test_home_gk_on_attacking_half_period_is_flipped():
    # P1: home GK at x=100 (>52.5) => mis-oriented => flip. P2: home GK at x=5 => keep.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    p1_hgk = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    p2_hgk = out[(out.period_id == 2) & (out.player_id == "hgk")].iloc[0]
    # Both periods: home GK now at LOW x (home defends x=0 in the canonical LTR frame).
    assert p1_hgk.x == pytest.approx(5.0)   # 105 - 100
    assert p1_hgk.y == pytest.approx(63.0)  # 68 - 5
    assert p2_hgk.x == pytest.approx(5.0)   # unchanged


def test_labels_populated_ltr_for_home_rtl_for_away_after_orient():
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    home = out[(~out.is_ball) & (out.team_id == "H")]
    away = out[(~out.is_ball) & (out.team_id == "A")]
    assert (home.team_attacking_direction == "ltr").all()
    assert (away.team_attacking_direction == "rtl").all()


def test_idempotent_label_rederivation():
    # Re-orienting after clearing labels reproduces the same labels AND coords.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    once = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    twice = orient_frames_to_ltr_by_geometry(once.assign(team_attacking_direction=None), home_team_id="H")
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True), check_dtype=False)


def test_idempotent_pure_coordinates():
    # The property that matters: orienting an ALREADY-oriented frame (no label reset) is a
    # pure no-op — home GK is already low-x so no period flips, labels stay put.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    once = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    twice = orient_frames_to_ltr_by_geometry(once, home_team_id="H")  # NO reset
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True), check_dtype=False)


def test_extra_time_periods_flip_independently():
    rows = [
        _frame(3, "H", "hgk", 100.0, 5.0, is_gk=True), _frame(3, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(4, "H", "hgk", 5.0, 5.0, is_gk=True),   _frame(4, "A", "agk", 5.0, 34.0, is_gk=True),
    ]
    out = orient_frames_to_ltr_by_geometry(pd.DataFrame(rows), home_team_id="H")
    assert out[(out.period_id == 3) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(5.0)
    assert out[(out.period_id == 4) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(5.0)


def test_ball_rows_flipped_with_their_period():
    rows = [
        _frame(1, "H", "hgk", 100.0, 5.0, is_gk=True),
        _frame(1, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(1, None, None, 80.0, 10.0, is_ball=True),
    ]
    out = orient_frames_to_ltr_by_geometry(pd.DataFrame(rows), home_team_id="H")
    ball = out[out.is_ball].iloc[0]
    assert ball.x == pytest.approx(25.0)   # 105 - 80
    assert ball.y == pytest.approx(58.0)   # 68 - 10


def test_velocity_components_negated_on_flip():
    # NOTE: the default builders emit no vx/vy (only `speed`); vx/vy exist only under
    # `preprocess`. This test injects them to exercise the negate-on-flip path directly,
    # since the default-builder output wouldn't reach it.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    frames["vx"] = 2.0
    frames["vy"] = -3.0
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    p1 = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    p2 = out[(out.period_id == 2) & (out.player_id == "hgk")].iloc[0]
    assert p1.vx == pytest.approx(-2.0) and p1.vy == pytest.approx(3.0)   # flipped
    assert p2.vx == pytest.approx(2.0) and p2.vy == pytest.approx(-3.0)   # unchanged


def test_zero_home_match_raises():
    frames = _two_period_match(100.0, 5.0)
    with pytest.raises(ValueError, match="matched ZERO"):
        orient_frames_to_ltr_by_geometry(frames, home_team_id="NOPE")


def test_missing_required_column_raises():
    frames = _two_period_match(100.0, 5.0).drop(columns=["is_goalkeeper"])
    with pytest.raises(ValueError, match="required column"):
        orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py -v`
Expected: FAIL with `ImportError: cannot import name 'orient_frames_to_ltr_by_geometry'`.

- [ ] **Step 3: Implement the orienter**

In `silly_kicks/tracking/direction.py`, add at the top (after existing imports):

```python
import warnings

from silly_kicks.spadl import config as _spadlconfig

_PITCH_LENGTH_M: float = _spadlconfig.field_length  # 105.0
_PITCH_WIDTH_M: float = _spadlconfig.field_width    # 68.0
_PITCH_MID_X: float = _PITCH_LENGTH_M / 2.0          # 52.5
_LTR_KNOWN_PERIODS: tuple[int, ...] = (1, 2, 3, 4)   # period 5 (PSO) direction undefined
```

Then append the function:

```python
def orient_frames_to_ltr_by_geometry(
    frames: pd.DataFrame,
    *,
    home_team_id: Any,
    source: str = "",
    game_id: Any = None,
) -> pd.DataFrame:
    """Flag-free geometric frame-LTR orientation: ensure home attacks +x every period.

    Per-period directional anchor = the home goalkeeper's median x. A GK sits deepest
    in its own half, so in the canonical home-attacks-right (LTR) frame the home GK
    must sit at LOW x (home defends x=0). Any period whose home-GK median x is on the
    attacking half (``> 52.5``) is mis-oriented; ALL its rows are point-reflected
    (``x->105-x``, ``y->68-y``, ``vx->-vx``, ``vy->-vy`` when present; ``speed`` is a
    magnitude, unchanged). ``team_attacking_direction`` is populated where null.

    Unlike :func:`orient_frames_to_ltr` (flag-based), this reads orientation from the
    DATA, so it is robust to absent/defaulted ``home_team_start_left`` (no bronze field
    carries it) and to per-feed ET coordinate flips. **Idempotent** — a no-op on
    already-correctly-oriented frames (home GK already at low x). Promoted from
    luxury-lakehouse ADR-053 ``correct_frames_to_home_ltr``; see NOTICE.

    Orientation is the builder's owned, normal operation (every match flips ~half its
    periods), so normal flips are SILENT (unlike ADR-053's correctness-net logging);
    a period with no GK anchor warns; a ``home_team_id`` matching no player raises
    (ADR-019 — mis-orienting is worse than failing).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form frames. Required: ``x``, ``y``, ``team_id``, ``period_id``,
        ``is_ball``, ``is_goalkeeper``. ``vx``/``vy`` flipped when present.
    home_team_id : Any
        Home-team id matching ``frames["team_id"]`` (compared via ADR-019 ``ids_match``).
    source, game_id : Any
        Diagnostic context only (warning messages).

    Returns
    -------
    pd.DataFrame
        New DataFrame in home-attacks-right convention.

    Raises
    ------
    ValueError
        Missing required column, or ``home_team_id`` matches zero player rows.

    Examples
    --------
    Orient absolute metrica/skillcorner frames built from bronze::

        from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry
        oriented = orient_frames_to_ltr_by_geometry(frames, home_team_id="Home")
    """
    required = {"x", "y", "team_id", "period_id", "is_ball", "is_goalkeeper"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"orient_frames_to_ltr_by_geometry: required column(s) missing: {sorted(missing)}")
    if len(frames) == 0:
        return frames.copy()

    out = frames.copy()
    is_ball = out["is_ball"].astype(bool)
    is_player = ~is_ball
    is_home = ids_match(out["team_id"], home_team_id).fillna(False)
    is_gk = out["is_goalkeeper"].astype(bool)

    if not bool((is_player & is_home).any()):
        raise ValueError(
            f"orient_frames_to_ltr_by_geometry: home_team_id={home_team_id!r} matched ZERO "
            f"player rows ({source} game={game_id}) — refusing to guess orientation."
        )

    x_arr = out["x"].to_numpy(dtype="float64")
    period_arr = out["period_id"].to_numpy()
    home_arr = is_home.to_numpy(dtype=bool)
    player_arr = is_player.to_numpy(dtype=bool)
    gk_arr = is_gk.to_numpy(dtype=bool)

    def _gk_median(mask: np.ndarray) -> float:
        vals = x_arr[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    has_vx, has_vy = "vx" in out.columns, "vy" in out.columns
    for period in pd.Series(period_arr[player_arr]).dropna().unique():
        psel = player_arr & (period_arr == period)
        home_gk_x = _gk_median(psel & home_arr & gk_arr)
        if not np.isnan(home_gk_x):
            needs_flip = home_gk_x > _PITCH_MID_X
        else:
            away_gk_x = _gk_median(psel & ~home_arr & gk_arr)
            if np.isnan(away_gk_x):
                warnings.warn(
                    f"orient_frames_to_ltr_by_geometry: {source} game={game_id} period={period} "
                    "has no GK anchor (home or away) — orientation left as-is for this period.",
                    stacklevel=2,
                )
                continue
            needs_flip = away_gk_x < _PITCH_MID_X
        if needs_flip:
            fmask = period_arr == period
            out.loc[fmask, "x"] = _PITCH_LENGTH_M - x_arr[fmask]
            out.loc[fmask, "y"] = _PITCH_WIDTH_M - out["y"].to_numpy(dtype="float64")[fmask]
            if has_vx:
                out.loc[fmask, "vx"] = -out["vx"].to_numpy(dtype="float64")[fmask]
            if has_vy:
                out.loc[fmask, "vy"] = -out["vy"].to_numpy(dtype="float64")[fmask]

    if "team_attacking_direction" in out.columns and out["team_attacking_direction"].isna().all():
        known = is_player & out["period_id"].isin(_LTR_KNOWN_PERIODS)
        out.loc[known & is_home, "team_attacking_direction"] = "ltr"
        out.loc[known & ~is_home, "team_attacking_direction"] = "rtl"
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Mirror ADR-053's golden as a cross-repo equivalence anchor**

Add to `tests/tracking/test_orient_by_geometry.py` a test that reproduces ADR-053's documented real-slice expectations (the lakehouse `test_frame_orientation_golden.py` behavior), asserting **coordinates** (not logging): a metrica-shaped period-2 flip lands home low-x, and an already-correct idsse-shaped slice is byte-unchanged (no-op). Use the same fixture shape as the lakehouse golden if portable; otherwise encode the two cases synthetically:

```python
def test_already_correct_frames_are_noop():
    # home GK already low-x in both periods => no flip, positions unchanged.
    frames = _two_period_match(home_gk_x_p1=5.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    pd.testing.assert_frame_equal(
        out.drop(columns=["team_attacking_direction"]).reset_index(drop=True),
        frames.drop(columns=["team_attacking_direction"]).reset_index(drop=True),
        check_dtype=False,
    )
```

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py -v`
Expected: PASS (10 tests).

---

## Task 3: SkillCorner bronze→frame builder

**Files:**
- Create: `silly_kicks/tracking/skillcorner.py`
- Test: `tests/tracking/test_skillcorner_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_skillcorner_builder.py`:

```python
"""tracking.skillcorner.convert_to_frames — bronze→canonical frame builder."""
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import skillcorner as sk
from silly_kicks.tracking.schema import KLOPPY_TRACKING_FRAMES_COLUMNS


def _bronze_row(frame, period, ts, player, team, x, y, is_gk, ball_x, ball_y, ball_z, is_vis):
    return {
        "match_id": "m1", "period": period, "frame": frame, "timestamp": ts,
        "player_id": player, "team_id": team, "is_goalkeeper": is_gk,
        "x": x, "y": y, "ball_x": ball_x, "ball_y": ball_y, "ball_z": ball_z,
        "is_visible": is_vis, "frame_rate": 10,
    }


def _bronze(n_frames=4):
    """Two teams (home 31, away 42), one GK each; home GK on the LEFT in P1 (low centre-x)."""
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.1 if period == 1 else 2700.0 + (f - 2) * 0.1
        # centre-origin meters: home GK near own goal (left, x≈-50) in P1
        rows += [
            _bronze_row(f, period, ts, 311, 31, -50.0, 0.0, True, 5.0, 1.0, 2.0, True),
            _bronze_row(f, period, ts, 312, 31, -10.0, 5.0, False, 5.0, 1.0, 2.0, True),
            _bronze_row(f, period, ts, 421, 42, 50.0, 0.0, True, 5.0, 1.0, 2.0, False),
            _bronze_row(f, period, ts, 422, 42, 10.0, -5.0, False, 5.0, 1.0, 2.0, True),
        ]
    return pd.DataFrame(rows)


def test_rescale_centre_origin_to_spadl():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    # away outfield at centre-origin (10, -5) -> SPADL (62.5, 29.0)
    row = frames[(frames.player_id == "422") & (frames.frame_id == 0)].iloc[0]
    assert row.x == pytest.approx(62.5) and row.y == pytest.approx(29.0)


def test_ball_z_recovered_not_nan():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    ball = frames[frames.is_ball].iloc[0]
    assert ball.z == pytest.approx(2.0)   # bronze ball_z preserved, NOT NaN


def test_player_z_is_nan_and_visibility_mapped():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    p = frames[(~frames.is_ball) & (frames.player_id == "421")].iloc[0]
    assert np.isnan(p.z)
    assert p.visibility is False   # is_goalkeeper away GK had is_visible=False


def test_period_relative_clock():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    p2 = frames[frames.period_id == 2]["time_seconds"]
    assert p2.min() == pytest.approx(0.0)   # 2700 - 2700


def test_ids_are_object_strings():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert frames["team_id"].dropna().map(type).eq(str).all()
    assert frames["player_id"].dropna().map(type).eq(str).all()


def test_output_schema_matches_kloppy_variant():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert list(frames.columns) == list(KLOPPY_TRACKING_FRAMES_COLUMNS)


def test_ltr_orientation_applied_by_default():
    # Home GK starts left (low x) in P1 -> P1 keeps; P2 home defends right -> flips to low x.
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31")  # output_convention default "ltr"
    hgk = frames[(~frames.is_ball) & (frames.player_id == "311")]
    assert (hgk.x < 52.5).all()   # home GK low-x every period post-LTR
    assert (frames[(~frames.is_ball) & (frames.team_id == "31")].team_attacking_direction == "ltr").all()


def test_clock_constant_is_single_sourced():
    # Regression guard for duplicated-truth #3: the builder must import the SPADL constant.
    import inspect
    from silly_kicks.spadl import skillcorner as sk_spadl
    assert sk._PERIOD_START_SECONDS is sk_spadl._PERIOD_START_SECONDS


def test_native_skillcorner_gk_survives_derivation():
    # Orientation anchors on home-GK median x; SkillCorner has an AUTHORITATIVE native GK
    # (skillcorner_matches.position_acronym). derive_goalkeepers is Tier-1 roster-validated
    # (PR-S86, 20/20) and must keep the native pick — a wrong overwrite silently mirrors
    # orientation for the one provider with ground-truth GK identity.
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    home_gk = set(frames[(~frames.is_ball) & (frames.team_id == "31") & frames.is_goalkeeper]["player_id"])
    assert home_gk == {"311"}   # the native SkillCorner GK (bronze is_goalkeeper=True), survived


def test_missing_input_column_raises():
    bad = _bronze().drop(columns=["ball_z"])
    with pytest.raises(ValueError, match="ball_z"):
        sk.convert_to_frames(bad, home_team_id="31")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_skillcorner_builder.py -v`
Expected: FAIL (`ModuleNotFoundError: silly_kicks.tracking.skillcorner`).

- [ ] **Step 3: Implement the SkillCorner builder**

Create `silly_kicks/tracking/skillcorner.py`:

```python
"""SkillCorner bronze→frame builder (TF-23, ADR-034).

Pure ``pd.DataFrame -> (pd.DataFrame, TrackingConversionReport)`` builder consuming the
post-join SkillCorner bronze (``bronze.skillcorner_tracking`` joined with
``bronze.skillcorner_matches`` for team/GK), parallel to ``tracking.sportec``. Single-
sources the coordinate (centre-origin -> SPADL 105x68), ``ball_z`` recovery, period-
relative clock, id-namespacing, GK derivation, speed, and geometric LTR orientation
that the luxury-lakehouse previously duplicated. See spec
``2026-06-18-tf23-skillcorner-metrica-bronze-frame-builders-design.md``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

# Same-provider single-source: spadl.skillcorner (the SkillCorner EVENTS converter) owns the
# nominal period offsets; SK tracking imports the SAME constant so frames match events (kills
# duplicated-truth #3). SK P2 raw clock starts exactly at the nominal 2700 (verified). NB: this
# is NOT the metrica cross-wire the review flagged — metrica.py has its own per-period-min clock.
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS

from ._gk_identification import derive_goalkeepers
from .direction import orient_frames_to_ltr_by_geometry, require_et_direction
from .schema import SKILLCORNER_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .utils import _derive_speed, orient_frames_to_ltr

EXPECTED_INPUT_COLUMNS: tuple[str, ...] = (
    "match_id", "period", "frame", "timestamp", "player_id", "team_id",
    "is_goalkeeper", "x", "y", "ball_x", "ball_y", "ball_z", "is_visible", "frame_rate",
)


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: Any | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert post-join SkillCorner bronze tracking to canonical SPADL frames.

    Parameters
    ----------
    bronze : pd.DataFrame
        SkillCorner tracking bronze joined with ``skillcorner_matches`` (team/GK).
        Required columns: see ``EXPECTED_INPUT_COLUMNS``. ``x``/``y`` are centre-origin
        meters; ``ball_z`` is real ball height; ``timestamp`` is the continuous
        broadcast clock; ``is_goalkeeper`` is the native (roster) flag.
    home_team_id : Any
        Home team id (matches ``team_id`` after stringification).
    output_convention : {"absolute_frame", "ltr"}, default "ltr"
        ``"ltr"`` orients via the geometric net (``home_team_start_left=None``) or the
        flag-based ``orient_frames_to_ltr`` (when a flag is supplied). ``"absolute_frame"``
        leaves frames unoriented (``team_attacking_direction=None``).
    home_team_start_left, home_team_start_left_extratime : bool | None
        Optional authoritative orientation flags; ``None`` => geometric orientation.
    preprocess : PreprocessConfig | None
        Optional smoothing/velocity; off by default.

    Returns
    -------
    tuple[pd.DataFrame, TrackingConversionReport]

    Examples
    --------
    Build LTR frames from bronze::

        from silly_kicks.tracking import skillcorner
        frames, report = skillcorner.convert_to_frames(bronze_df, home_team_id="31")
    """
    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in bronze.columns]
    if missing:
        raise ValueError(f"skillcorner.convert_to_frames: bronze missing column(s): {missing}")
    if home_team_start_left is None and output_convention == "ltr":
        # geometric path needs no ET flag; flag path validates it
        pass
    elif output_convention == "ltr":
        require_et_direction(bronze["period"], home_team_start_left_extratime, source="skillcorner convert_to_frames")

    src = bronze.copy()
    game_id = str(src["match_id"].iloc[0])
    frame_rate = float(src["frame_rate"].iloc[0]) if "frame_rate" in src else 10.0

    # --- player rows ---
    players = src[
        ["frame", "period", "timestamp", "player_id", "team_id", "is_goalkeeper", "x", "y", "is_visible"]
    ].copy()
    players = players.rename(
        columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds"}
    )
    players["x"] = players["x"] + 52.5
    players["y"] = players["y"] + 34.0
    players["z"] = np.nan
    players["visibility"] = players.pop("is_visible")
    players["is_ball"] = False
    players["player_id"] = players["player_id"].astype(str)
    players["team_id"] = players["team_id"].astype(str)

    # --- ball rows (one per (frame, period); recover ball_z) ---
    ball = src[["frame", "period", "timestamp", "ball_x", "ball_y", "ball_z"]].drop_duplicates(
        subset=["frame", "period"]
    ).copy()
    ball = ball.rename(
        columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds",
                 "ball_x": "x", "ball_y": "y", "ball_z": "z"}
    )
    ball["x"] = ball["x"] + 52.5
    ball["y"] = ball["y"] + 34.0
    ball["player_id"] = None
    ball["team_id"] = None
    ball["is_goalkeeper"] = False
    ball["visibility"] = None
    ball["is_ball"] = True

    df = pd.concat([players, ball], ignore_index=True)
    df["game_id"] = game_id
    df["frame_rate"] = frame_rate
    df["source_provider"] = "skillcorner"
    df["ball_state"] = None
    df["team_attacking_direction"] = None
    df["confidence"] = None
    df["speed"] = np.nan
    df["speed_source"] = None

    # period-relative clock via the SINGLE-SOURCED nominal constant (matches the SK events
    # converter, so action<->frame linkage is exact regardless of the absolute value). NB:
    # SkillCorner intentionally follows the events NOMINAL convention, NOT a strict
    # period-relative-from-0 — if a real broadcast carries 1st-half stoppage, P2 may start
    # slightly off 0. Benign: frames and events offset IDENTICALLY (the shared-constant guard
    # test proves it); this is the ADR-017 caveat for SkillCorner. (Metrica DOES hold
    # period-relative-from-0 via its per-period-min rebase.)
    df["time_seconds"] = df["time_seconds"] - df["period_id"].map(_PERIOD_START_SECONDS).fillna(0.0).astype(float)

    df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    df = _derive_speed(df)

    # GK derivation + agreement-based source (mirror the kloppy gateway)
    native_gk = {
        (str(g), str(t)): set(grp.loc[grp["is_goalkeeper"], "player_id"].dropna().astype(str))
        for (g, t), grp in df[~df["is_ball"]].groupby(["game_id", "team_id"], sort=False)
    }
    df, derived_picks = derive_goalkeepers(df)
    n_derived = 0
    df["is_goalkeeper_source"] = None
    for (g, t), algo in derived_picks.items():
        source_val = "native" if set(algo) == native_gk.get((g, t), set()) else "derived"
        n_derived += source_val == "derived"
        m = (df["game_id"] == g) & (df["team_id"] == t) & ~df["is_ball"]
        df.loc[m, "is_goalkeeper_source"] = source_val

    final = pd.DataFrame({c: df[c] for c in SKILLCORNER_TRACKING_FRAMES_COLUMNS})
    for c, dt in SKILLCORNER_TRACKING_FRAMES_COLUMNS.items():
        if dt == "bool":
            final[c] = final[c].astype("bool")
        elif dt in {"int64", "float64"}:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype(dt)
        else:
            final[c] = final[c].astype(object)

    if output_convention == "ltr":
        if home_team_start_left is None:
            final = orient_frames_to_ltr_by_geometry(final, home_team_id=str(home_team_id), source="skillcorner", game_id=game_id)
        else:
            final = orient_frames_to_ltr(
                final, home_team_id=str(home_team_id), home_team_start_left=home_team_start_left,
                home_team_start_left_extratime=home_team_start_left_extratime,
            )

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess
        cfg = resolve_preprocess(preprocess, provider="skillcorner")
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    report = TrackingConversionReport(
        provider="skillcorner",
        total_input_frames=int(src[["frame", "period"]].drop_duplicates().shape[0]),
        total_output_rows=len(final),
        n_periods=int(final["period_id"].nunique()),
        frame_coverage_per_period={int(p): 1.0 for p in final["period_id"].unique()},
        ball_out_seconds_per_period={},
        nan_rate_per_column={c: float(final[c].isna().mean()) for c in final.columns},
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
        n_teams_gk_derived=n_derived,
        derived_gk_picks=derived_picks,
    )
    return final, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_skillcorner_builder.py -v`
Expected: PASS (10 tests). If `test_player_z_is_nan_and_visibility_mapped` fails on the `visibility is False` identity, adjust to `bool(p.visibility) is False` (object-column boolean).

---

## Task 4: Metrica bronze→frame builder

**Files:**
- Create: `silly_kicks/tracking/metrica.py`
- Test: `tests/tracking/test_metrica_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_metrica_builder.py`:

```python
"""tracking.metrica.convert_to_frames — bronze→canonical frame builder."""
import json
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import metrica as mt
from silly_kicks.tracking.schema import KLOPPY_TRACKING_FRAMES_COLUMNS


def _bronze(n_frames=4, p2_start=2850.0):
    """Metrica frame-level bronze: 0-1 normalized coords, JSON player columns.

    ``p2_start`` sets the raw P2 clock origin: 2850.0 mimics a CONTINUOUS sample game
    (SG1-like, NOT the nominal 2700 — so per-period-min rebasing is distinguishable from
    a nominal-offset subtraction); pass 0.0 for a PERIOD-RELATIVE game (SG3-like).
    """
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.04 if period == 1 else p2_start + (f - 2) * 0.04
        # home GK jersey "1" near own goal (left, x≈0.05) in P1
        home = {"1": {"x": 0.05, "y": 0.50}, "9": {"x": 0.40, "y": 0.55}}
        away = {"1": {"x": 0.95, "y": 0.50}, "9": {"x": 0.60, "y": 0.45}}
        rows.append({
            "period": period, "frame": f, "timestamp": ts,
            "ball_x": 0.50, "ball_y": 0.50,
            "home_players": json.dumps(home), "away_players": json.dumps(away),
            "gk_jersey_numbers": json.dumps(["1"]), "frame_rate": 25,
        })
    return pd.DataFrame(rows)


def _roster():
    return {"Home": {"1": "h_gk", "9": "h_fw"}, "Away": {"1": "a_gk", "9": "a_fw"}}


def test_rescale_0_1_to_spadl_no_flip():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    fw = frames[(frames.player_id == "h_fw") & (frames.frame_id == 0)].iloc[0]
    assert fw.x == pytest.approx(0.40 * 105.0) and fw.y == pytest.approx(0.55 * 68.0)


def test_ball_z_is_nan():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert np.isnan(frames[frames.is_ball].iloc[0].z)


def _bronze_gk_collision(n_frames=4):
    """Teams have DIFFERENT GK numbers (home #1, away #16), each reusing the OTHER's GK
    number on an outfielder (home #16 outfielder, away #1 outfielder) — the exact case a
    team-agnostic jersey.isin(gk_jersey_numbers) mis-flags."""
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.04 if period == 1 else 2850.0 + (f - 2) * 0.04
        home = {"1": {"x": 0.04, "y": 0.50}, "16": {"x": 0.45, "y": 0.55}}   # GK #1 deep, OF #16
        away = {"16": {"x": 0.96, "y": 0.50}, "1": {"x": 0.55, "y": 0.45}}    # GK #16 deep, OF #1
        rows.append({
            "period": period, "frame": f, "timestamp": ts, "ball_x": 0.50, "ball_y": 0.50,
            "home_players": json.dumps(home), "away_players": json.dumps(away),
            "gk_jersey_numbers": json.dumps(["1", "16"]), "frame_rate": 25,
        })
    return pd.DataFrame(rows)


def test_gk_derived_positionally():
    # GK comes from positional derivation, NOT the flat jersey list. Home GK (jersey "1",
    # deepest home player) is flagged.
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert frames[frames.player_id == "h_gk"].is_goalkeeper.all()


def test_gk_not_flagged_by_shared_jersey_number():
    # The team-agnostic bug would flag home #16 + away #1 outfielders as GK (both numbers are
    # in the flat gk_jersey_numbers). Positional derivation must flag ONLY the deep players.
    roster = {"Home": {"1": "h_gk", "16": "h_of"}, "Away": {"16": "a_gk", "1": "a_of"}}
    frames, _ = mt.convert_to_frames(_bronze_gk_collision(), jersey_to_player_id=roster, output_convention="absolute_frame")
    gk_ids = set(frames[(~frames.is_ball) & frames.is_goalkeeper]["player_id"])
    assert gk_ids == {"h_gk", "a_gk"}, gk_ids   # NOT h_of / a_of


def test_clock_rebased_per_period_min_continuous_game():
    # Continuous raw P2 (starts 2850, NOT nominal 2700) -> per-period-min rebases to ~0.
    # A nominal-2700 subtraction would WRONGLY leave P2 at ~150.
    frames, _ = mt.convert_to_frames(_bronze(p2_start=2850.0), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert frames[frames.period_id == 2]["time_seconds"].min() == pytest.approx(0.0, abs=0.05)


def test_clock_rebased_per_period_min_period_relative_game():
    # Already period-relative raw P2 (starts 0) -> stays ~0 (no spurious negative times).
    frames, _ = mt.convert_to_frames(_bronze(p2_start=0.0), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    p2 = frames[frames.period_id == 2]["time_seconds"]
    assert p2.min() == pytest.approx(0.0, abs=0.05)
    assert (p2 >= -1e-6).all()   # never negative


def test_builder_does_not_iterate_rows():
    # Structural perf guard (no wall-clock assert): the vectorized shape must not iterrows
    # NOR apply(axis=1) — both are the tracking-scale row-wise cliff.
    import pandas as _pd
    orig_iter, orig_apply = _pd.DataFrame.iterrows, _pd.DataFrame.apply
    def _boom_iter(self):
        raise AssertionError("metrica builder must not call DataFrame.iterrows (tracking-scale anti-pattern)")
    def _spy_apply(self, func, *a, **k):
        axis = k.get("axis", a[0] if a else 0)
        if axis in (1, "columns"):
            raise AssertionError("metrica builder must not call DataFrame.apply(axis=1) (row-wise cliff)")
        return orig_apply(self, func, *a, **k)
    _pd.DataFrame.iterrows = _boom_iter
    _pd.DataFrame.apply = _spy_apply
    try:
        mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    finally:
        _pd.DataFrame.iterrows = orig_iter
        _pd.DataFrame.apply = orig_apply


def test_output_schema_matches_kloppy_variant():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert list(frames.columns) == list(KLOPPY_TRACKING_FRAMES_COLUMNS)


def test_ltr_orientation_home_low_x_every_period():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), home_team_id="Home")
    hgk = frames[(~frames.is_ball) & (frames.player_id == "h_gk")]
    assert (hgk.x < 52.5).all()


def test_missing_input_column_raises():
    bad = _bronze().drop(columns=["home_players"])
    with pytest.raises(ValueError, match="home_players"):
        mt.convert_to_frames(bad, jersey_to_player_id=_roster())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_metrica_builder.py -v`
Expected: FAIL (`ModuleNotFoundError: silly_kicks.tracking.metrica`).

- [ ] **Step 3: Implement the Metrica builder**

Create `silly_kicks/tracking/metrica.py`:

```python
"""Metrica bronze→frame builder (TF-23, ADR-034).

Pure builder consuming Metrica frame-level bronze (``bronze.metrica_tracking``: 0-1
normalized coords, JSON ``home_players``/``away_players``), parallel to
``tracking.skillcorner``. Metrica has NO ball z (``z=NaN`` is correct). 0-1 -> SPADL
105x68 is a pure standardization (no y-flip — Metrica y is bottom-to-top). See spec.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

# NOTE: Metrica does NOT import the SkillCorner nominal clock constant. Metrica sample
# games use MIXED raw clocks (some continuous, some period-relative — verified on real
# bronze 2026-06-18), so the clock is rebased per-(period) min (see convert_to_frames).

from ._gk_identification import derive_goalkeepers
from .direction import orient_frames_to_ltr_by_geometry, require_et_direction
from .schema import METRICA_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .utils import _derive_speed, orient_frames_to_ltr

EXPECTED_INPUT_COLUMNS: tuple[str, ...] = (
    "period", "frame", "timestamp", "ball_x", "ball_y",
    "home_players", "away_players", "gk_jersey_numbers", "frame_rate",
)


def _to_player_tuples(raw: Any) -> list[tuple[str, float, float]]:
    """Parse one frame's player JSON blob to ``[(jersey, x, y), ...]`` (one json.loads/row).

    Malformed/positionless player entries are dropped. Acceptable here: Metrica is 3 frozen,
    hand-curated, well-formed sample games (no live feed); gross data loss would surface as a
    short ``total_output_rows`` in the report. (If a live Metrica feed is ever added, fold a
    dropped-player count into the report — out of scope for the frozen public data.)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    d = json.loads(raw) if isinstance(raw, str) else raw
    return [
        (str(j), float(c["x"]), float(c["y"]))
        for j, c in d.items()
        if isinstance(c, dict) and "x" in c and "y" in c
    ]


def _explode_team(bronze: pd.DataFrame, col: str, team_label: str) -> pd.DataFrame:
    """Vectorized JSON explode (NO iterrows): frame×players long-form for one team."""
    base = bronze[["frame", "period", "timestamp"]].reset_index(drop=True).copy()
    base["_pl"] = bronze[col].map(_to_player_tuples).to_numpy()  # one json.loads per row
    out = base.explode("_pl", ignore_index=True).dropna(subset=["_pl"])
    if out.empty:
        return out.assign(jersey=[], x=[], y=[])
    out[["jersey", "x", "y"]] = pd.DataFrame(out["_pl"].tolist(), index=out.index)
    out["team_id"] = team_label
    return out.drop(columns="_pl")


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any = "Home",
    jersey_to_player_id: dict[str, dict[str, str]] | None = None,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: Any | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert Metrica frame-level bronze to canonical SPADL frames.

    Parameters
    ----------
    bronze : pd.DataFrame
        Metrica tracking bronze; required columns: see ``EXPECTED_INPUT_COLUMNS``.
    home_team_id : Any, default "Home"
        Team label used for the home rows (Metrica is anonymized — "Home"/"Away").
    jersey_to_player_id : dict[str, dict[str, str]] | None
        ``{"Home": {jersey: player_id}, "Away": {...}}`` from the consumer's roster;
        ``None`` => synthetic ``f"{team}_{jersey}"`` ids.
    output_convention, home_team_start_left, home_team_start_left_extratime, preprocess
        As in ``tracking.skillcorner.convert_to_frames``.

    Returns
    -------
    tuple[pd.DataFrame, TrackingConversionReport]

    Examples
    --------
    Build LTR frames from Metrica bronze::

        from silly_kicks.tracking import metrica
        frames, report = metrica.convert_to_frames(bronze_df, jersey_to_player_id=roster)
    """
    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in bronze.columns]
    if missing:
        raise ValueError(f"metrica.convert_to_frames: bronze missing column(s): {missing}")
    if home_team_start_left is not None and output_convention == "ltr":
        require_et_direction(bronze["period"], home_team_start_left_extratime, source="metrica convert_to_frames")

    roster = jersey_to_player_id or {}
    frame_rate = float(bronze["frame_rate"].iloc[0]) if "frame_rate" in bronze else 25.0
    gk_raw = bronze["gk_jersey_numbers"].dropna()
    gk_jerseys: set[str] = set()
    if not gk_raw.empty:
        parsed = json.loads(gk_raw.iloc[0]) if isinstance(gk_raw.iloc[0], str) else gk_raw.iloc[0]
        gk_jerseys = {str(j) for j in parsed} if parsed else set()

    src = bronze[~bronze["period"].isna()].copy()
    # --- vectorized player explode (NO iterrows) ---
    players = pd.concat(
        [_explode_team(src, "home_players", "Home"), _explode_team(src, "away_players", "Away")],
        ignore_index=True,
    )
    players = players.rename(columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds"})
    players["x"] = players["x"] * 105.0
    players["y"] = players["y"] * 68.0
    players["z"] = np.nan
    players["is_goalkeeper"] = False  # NO native GK seed for Metrica (see GK-derivation note below)
    players["is_ball"] = False
    players["visibility"] = None
    # roster map (team, jersey) -> player_id via a VECTORIZED merge (no per-row python loop);
    # synthetic f"{team}_{jersey}" fallback for unmapped jerseys.
    roster_df = pd.DataFrame(
        [(t, j, p) for t, d in roster.items() for j, p in d.items()],
        columns=["team_id", "jersey", "player_id"],
    )
    if not roster_df.empty:
        players = players.merge(roster_df, on=["team_id", "jersey"], how="left")
    else:
        players["player_id"] = np.nan
    players["player_id"] = players["player_id"].fillna(players["team_id"] + "_" + players["jersey"])
    players = players.drop(columns="jersey")

    # --- ball rows (one per frame; Metrica has no ball z) ---
    ball = src[["frame", "period", "timestamp", "ball_x", "ball_y"]].dropna(subset=["ball_x", "ball_y"]).copy()
    ball = ball.rename(columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds",
                                "ball_x": "x", "ball_y": "y"})
    ball["x"] = ball["x"] * 105.0
    ball["y"] = ball["y"] * 68.0
    ball["z"] = np.nan
    ball["player_id"] = None
    ball["team_id"] = None
    ball["is_goalkeeper"] = False
    ball["is_ball"] = True
    ball["visibility"] = None

    df = pd.concat([players, ball], ignore_index=True)
    df["game_id"] = "metrica"
    df["frame_id"] = df["frame_id"].astype(int)
    df["period_id"] = df["period_id"].astype(int)
    df["frame_rate"] = frame_rate
    df["source_provider"] = "metrica"
    df["ball_state"] = None
    df["team_attacking_direction"] = None
    df["confidence"] = None
    df["speed"] = np.nan
    df["speed_source"] = None
    # CLOCK: Metrica sample games use MIXED raw clocks (continuous vs period-relative —
    # verified on real bronze 2026-06-18). Rebase per-(period) min so every period starts at
    # ~0 (ADR-017 period-relative; matches the kloppy Metrica gateway = the parity oracle AND
    # the Metrica events clock, which kloppy also emits period-relative). NOT the SkillCorner
    # nominal offset. NB: unlike SkillCorner (shared-constant structural guard), Metrica
    # frame<->event clock parity has NO structural test — the event-anchored gate (Task 6) is
    # the SOLE guard (a mismatch -> zero links -> the gate's `len(res) >= 4` fails loud).
    df["time_seconds"] = df["time_seconds"] - df.groupby("period_id")["time_seconds"].transform("min")

    df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    df = _derive_speed(df)

    # Metrica is anonymized (Tier-2, ADR-007): gk_jersey_numbers is a FLAT list (verified:
    # e.g. ["11","25"]) with NO team split, so a native per-(team,jersey) GK flag is
    # unrecoverable — a team-agnostic flag mis-assigns when teams share a number, and
    # derive_goalkeepers ORs (never clears, _gk_identification.py:163-169), so the mis-flag
    # would reach the orientation anchor. We seed NO native GK (is_goalkeeper already False)
    # and let the validated positional algorithm derive it; source is therefore always
    # "derived". (SkillCorner, by contrast, passes its authoritative per-player roster flag.)
    df, derived_picks = derive_goalkeepers(df)
    df["is_goalkeeper_source"] = None
    df.loc[~df["is_ball"], "is_goalkeeper_source"] = "derived"
    n_derived = len(derived_picks)
    # Observability cross-check (lakehouse "never silently substitute"): the flat list's total
    # count should match the derived GK count across teams; disagreement is surfaced, not hidden.
    derived_gk_count = sum(len(v) for v in derived_picks.values())
    if gk_jerseys and derived_gk_count != len(gk_jerseys):
        warnings.warn(
            f"metrica.convert_to_frames: derived {derived_gk_count} GK(s) but gk_jersey_numbers "
            f"lists {len(gk_jerseys)} — positional GK derivation disagrees with the roster count.",
            stacklevel=2,
        )

    final = pd.DataFrame({c: df[c] for c in METRICA_TRACKING_FRAMES_COLUMNS})
    for c, dt in METRICA_TRACKING_FRAMES_COLUMNS.items():
        if dt == "bool":
            final[c] = final[c].astype("bool")
        elif dt in {"int64", "float64"}:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype(dt)
        else:
            final[c] = final[c].astype(object)

    if output_convention == "ltr":
        if home_team_start_left is None:
            final = orient_frames_to_ltr_by_geometry(final, home_team_id=str(home_team_id), source="metrica", game_id="metrica")
        else:
            final = orient_frames_to_ltr(
                final, home_team_id=str(home_team_id), home_team_start_left=home_team_start_left,
                home_team_start_left_extratime=home_team_start_left_extratime,
            )

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess
        cfg = resolve_preprocess(preprocess, provider="metrica")
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    report = TrackingConversionReport(
        provider="metrica",
        total_input_frames=int(bronze[["frame", "period"]].drop_duplicates().shape[0]),
        total_output_rows=len(final),
        n_periods=int(final["period_id"].nunique()),
        frame_coverage_per_period={int(p): 1.0 for p in final["period_id"].unique()},
        ball_out_seconds_per_period={},
        nan_rate_per_column={c: float(final[c].isna().mean()) for c in final.columns},
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
        n_teams_gk_derived=n_derived,
        derived_gk_picks=derived_picks,
    )
    return final, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_metrica_builder.py -v`
Expected: PASS (10 tests: rescale, ball_z NaN, GK-derived-positionally, GK-not-flagged-by-shared-jersey, two clock-convention cases, no-iterrows guard, schema, LTR, missing-column).

---

## Task 5: Public exports

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_public_surface_tf23.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_public_surface_tf23.py`:

```python
def test_tf23_public_surface():
    import silly_kicks.tracking as t
    assert hasattr(t, "skillcorner") and hasattr(t.skillcorner, "convert_to_frames")
    assert hasattr(t, "metrica") and hasattr(t.metrica, "convert_to_frames")
    assert "orient_frames_to_ltr_by_geometry" in t.__all__
    assert "SKILLCORNER_TRACKING_FRAMES_COLUMNS" in t.__all__
    assert "METRICA_TRACKING_FRAMES_COLUMNS" in t.__all__
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_public_surface_tf23.py -v`
Expected: FAIL.

- [ ] **Step 3: Wire the exports**

In `silly_kicks/tracking/__init__.py`:
1. Add `"metrica"`, `"skillcorner"`, `"orient_frames_to_ltr_by_geometry"`, `"METRICA_TRACKING_FRAMES_COLUMNS"`, `"SKILLCORNER_TRACKING_FRAMES_COLUMNS"` to `__all__` (alphabetical position).
2. Add to the submodule import line: `from . import (..., metrica, ..., skillcorner, ...)` (extend the existing `from . import feature_framework, features, gradientsports, ...` line).
3. Add `orient_frames_to_ltr_by_geometry` to the `from .direction import require_et_direction` line → `from .direction import orient_frames_to_ltr_by_geometry, require_et_direction`.
4. Add the two aliases to the `from .schema import (...)` block.

- [ ] **Step 4: Run to verify it passes + full tracking suite still green**

Run: `python -m pytest tests/tracking/test_public_surface_tf23.py -v && python -m pytest tests/tracking/ -m "not e2e" -q`
Expected: PASS; no regressions.

---

## Task 6: Keystone — event-anchored action↔frame y-identity gate (committed fixture)

This is the primary Gate-C closer (kloppy-independent), self-contained via silly-kicks'
own `link_actions_to_frames`. Metrica-y is a named, non-skippable case.

**Files:**
- Create: `tests/tracking/test_builder_event_anchored_gate.py`
- Create: `tests/datasets/tracking/sk_metrica_builder/{skillcorner_bronze.parquet, skillcorner_actions.parquet, metrica_bronze.parquet, metrica_actions.parquet}` (captured in Step 1)

- [ ] **Step 1: Capture committed fixtures (Databricks-extracted bronze — provenance pinned)**

**The bronze fixtures are EXTRACTED FROM the lakehouse Databricks `soccer_analytics.bronze`, NOT reconstructed in silly-kicks** — the SkillCorner builder consumes the *post-`⋈ skillcorner_matches`* narrow frame (team/GK/`ball_z`), and that join is ingestion-context (spec O5). Reconstructing it in silly-kicks would drag the join upstream. The extraction performs the same join the lakehouse prod job does, so the fixture is exactly what the lakehouse will pass the builder.

Write a local, NON-committed script `scripts/_tf23_capture_fixtures.py` (DGX/Databricks SDK access) that:
1. **SkillCorner bronze** — `SELECT ...` from `bronze.skillcorner_tracking t JOIN bronze.skillcorner_matches m ON t.match_id=m.match_id` for one match, projecting the `EXPECTED_INPUT_COLUMNS` (incl. `ball_z`, `is_visible`, and the joined `team_id`, `is_goalkeeper` via `position_acronym='GK'`, and the `home_team_id`). Record `match_id` + the `_ingested_at` watermark as `SOURCE_SHA`.
2. **Metrica bronze** — `SELECT ...` from `bronze.metrica_tracking` for one match (self-contained; no join). If the match has no ET, append a synthetic period-3 slice (reusing P1 geometry, raw clock offset) so the ET path is exercised.
3. **Actions** — build the SPADL actions for each match via `silly_kicks.spadl.skillcorner` (from the SkillCorner events bronze) / the Metrica kloppy event path, on the DGX/pining cache.
4. **Slice** — downsample each to a small set retaining **shots + passes by BOTH teams at off-centre y (|y−34|>5 m), across ≥2 periods including ET**.
5. Write `{skillcorner_bronze, skillcorner_actions, metrica_bronze, metrica_actions}.parquet` + a committed `meta.json` (`{"skillcorner_home_team_id": "...", "skillcorner_match_id": "...", "metrica_match_id": "...", "source_sha": "..."}`, read by the gate test — not hardcoded) + a committed `README.md` recording the join SQL + `SOURCE_SHA` + slice rule.

Run on the DGX; commit only the parquet fixtures + README (not the script — throwaway pattern, per the DFL parse-port slice precedent).

- [ ] **Step 2: Write the gate test (FAILS until fixtures + builders exist)**

Create `tests/tracking/test_builder_event_anchored_gate.py`:

```python
"""Event-anchored action<->frame y-identity gate — the primary Gate-C closer.

Kloppy-independent: links SPADL actions to builder frames via silly-kicks'
link_actions_to_frames, reprojects to action-LTR (ADR-028), asserts the acting
player's frame position at the action instant ~= the action start coordinate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import skillcorner as sk, metrica as mt
from silly_kicks.tracking.utils import link_actions_to_frames
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr

_FIX = Path(__file__).parent.parent / "datasets" / "tracking" / "sk_metrica_builder"
_TOL_M = 3.0          # action coord is the event tracker point; ~m tolerance
_CENTRE_BAND_M = 5.0  # exclude |y-34|<5: the |68-2y| error vanishes at centre


def _y_identity_residuals(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Per-action |frame_actor_y - action_start_y| in the action-LTR frame, off-centre only."""
    pointers, _ = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    linked = actions.merge(pointers[["action_id", "frame_id"]], on="action_id").dropna(subset=["frame_id"])
    linked["frame_id"] = linked["frame_id"].astype(int)  # pointers frame_id is float (NaN-able)
    # acting player's frame row at the linked frame (same-provider ids in committed fixtures)
    fr = frames[~frames["is_ball"].astype(bool)]
    merged = linked.merge(
        fr[["period_id", "frame_id", "player_id", "x", "y"]],
        on=["period_id", "frame_id", "player_id"], how="inner",
    ).reset_index(drop=True)
    # IMPORTANT: compute the flip mask on `merged` (post-merge index), NOT on `linked` —
    # the merge resets the index, so a flip aligned to `linked` would misalign.
    flip = acting_team_attacks_rtl(merged, frames)
    merged = reproject_to_action_ltr(merged, flip, x_cols=["x"], y_cols=["y"])
    merged = merged[(merged["start_y"] - 34.0).abs() > _CENTRE_BAND_M]   # off-centre only
    merged["dy"] = (merged["y"] - merged["start_y"]).abs()
    merged["dx"] = (merged["x"] - merged["start_x"]).abs()
    return merged


@pytest.mark.parametrize("provider", ["skillcorner", "metrica"])
def test_event_anchored_y_identity_both_teams(provider):
    import json
    bronze = pd.read_parquet(_FIX / f"{provider}_bronze.parquet")
    actions = pd.read_parquet(_FIX / f"{provider}_actions.parquet")
    if provider == "skillcorner":
        home = str(json.loads((_FIX / "meta.json").read_text())["skillcorner_home_team_id"])
        frames, _ = sk.convert_to_frames(bronze, home_team_id=home)
    else:
        frames, _ = mt.convert_to_frames(bronze, home_team_id="Home")
    res = _y_identity_residuals(actions, frames)
    assert len(res) >= 4, "fixture must retain off-centre actions for both teams"
    # BOTH teams represented (catches a one-sided mirror that passes on a single team)
    assert res["team_id"].nunique() >= 2
    # PER-(team, period) medians — NOT a global median: a single mis-oriented period
    # (e.g. ET <50% of actions) would slip under a global median, the exact subtle bug.
    grp = res.groupby(["team_id", "period_id"])[["dy", "dx"]].median()
    assert (grp["dy"] < _TOL_M).all(), f"{provider} per-(team,period) y disagreement:\n{grp}"
    assert (grp["dx"] < _TOL_M).all(), f"{provider} per-(team,period) x disagreement:\n{grp}"


def test_metrica_y_is_named_and_off_centre_and_et():
    """Metrica-y is THE historical bug + highest-risk axis — named, non-skippable."""
    bronze = pd.read_parquet(_FIX / "metrica_bronze.parquet")
    actions = pd.read_parquet(_FIX / "metrica_actions.parquet")
    frames, _ = mt.convert_to_frames(bronze, home_team_id="Home")
    res = _y_identity_residuals(actions, frames)
    assert (res["start_y"] - 34.0).abs().max() > _CENTRE_BAND_M  # genuinely off-centre
    assert res["period_id"].isin([3, 4]).any(), "metrica fixture must include an ET period"
    # per-(team, period) — the ET period must independently pass, not be masked by regulation.
    grp = res.groupby(["team_id", "period_id"])["dy"].median()
    assert (grp < _TOL_M).all(), f"metrica per-(team,period) y residual (ET must pass alone):\n{grp}"
```

- [ ] **Step 3: Run the gate**

Run: `python -m pytest tests/tracking/test_builder_event_anchored_gate.py -v`
Expected: PASS once fixtures are captured (Step 1) and builders (Tasks 3–4) exist. A FAIL here is a real coordinate defect — debug via `_y_identity_residuals` before proceeding. (If the SkillCorner home-id sidecar is needed, read it from the committed `README.md` value, not hardcoded.)

---

## Task 7: Owner-gated parity-to-oracle e2e

**Files:**
- Create: `tests/tracking/test_builder_kloppy_parity_e2e.py`

- [ ] **Step 1: Write the e2e parity test (marked e2e — not in the CI default suite)**

Create `tests/tracking/test_builder_kloppy_parity_e2e.py`:

```python
"""Parity-to-oracle: bronze builder vs kloppy gateway on the SAME match (owner-gated).

Structural agreement with the blessed kloppy path. z stays in parity (verified: kloppy
carries SkillCorner ball z as Point3D); z is also validated independently for physical
sense. Requires raw SkillCorner/Metrica data + bronze for the same match (DGX/pining).
"""
import os
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_DATA = os.environ.get("TF23_PARITY_DATA")  # dir with raw + bronze for one match each


@pytest.mark.skipif(not _DATA, reason="TF23_PARITY_DATA not set (owner-gated)")
def test_skillcorner_builder_matches_kloppy_oracle():
    import kloppy
    from silly_kicks.tracking import kloppy as gw, skillcorner as sk
    ds = kloppy.skillcorner.load(meta_data=f"{_DATA}/sc_meta.json", raw_data=f"{_DATA}/sc_tracking.json", include_empty_frames=False)
    oracle, _ = gw.convert_to_frames(ds, output_convention="ltr")
    bronze = pd.read_parquet(f"{_DATA}/sc_bronze.parquet")
    home = str(ds.metadata.teams[0].team_id)
    built, _ = sk.convert_to_frames(bronze, home_team_id=home)
    # align on (period, frame, player) and compare coordinate truth
    key = ["period_id", "frame_id", "player_id"]
    m = oracle.merge(built, on=key, suffixes=("_o", "_b"))
    assert (m["x_o"] - m["x_b"]).abs().median() < 0.5
    assert (m["y_o"] - m["y_b"]).abs().median() < 0.5
    assert (m["team_attacking_direction_o"] == m["team_attacking_direction_b"]).mean() > 0.99
    # z parity on ball rows (kloppy Point3D carries SkillCorner ball z)
    ball = oracle.merge(built, on=["period_id", "frame_id"], suffixes=("_o", "_b"))
    ball = ball[ball["is_ball_o"] & ball["is_ball_b"]]
    assert (ball["z_o"] - ball["z_b"]).abs().median() < 0.2


@pytest.mark.skipif(not _DATA, reason="TF23_PARITY_DATA not set (owner-gated)")
def test_skillcorner_ball_z_physically_sensible():
    from silly_kicks.tracking import skillcorner as sk
    bronze = pd.read_parquet(f"{_DATA}/sc_bronze.parquet")
    built, _ = sk.convert_to_frames(bronze, home_team_id=str(bronze["team_id"].iloc[0]))
    bz = built[built["is_ball"]]["z"].dropna()
    assert ((bz >= 0) & (bz <= 10)).mean() > 0.99           # physical range
    assert (bz > 0.5).any()                                  # some airborne frames
```

- [ ] **Step 2: Verify the e2e test collects + skips cleanly without data**

Run: `python -m pytest tests/tracking/test_builder_kloppy_parity_e2e.py -v`
Expected: 2 SKIPPED (no `TF23_PARITY_DATA`). On the DGX with data set, both PASS.

---

## Task 8: ADR-034 + NOTICE attribution

**Files:**
- Create: `docs/superpowers/adrs/ADR-034-skillcorner-metrica-bronze-builders.md`
- Modify: `NOTICE`

- [ ] **Step 1: Author ADR-034**

Create the ADR capturing: supersedes ADR-029's "no native converter" clause (bronze-consuming, not raw-file); harmonizes with lakehouse ADR-053 (promotes its geometric net upstream; lakehouse retires its copy); extends ADR-031 delete-and-depend to SC/Metrica; bounded-context boundary decision (§3 of the spec); `ball_z` recovery; event-anchored gate as the Gate-C closer. Cross-reference the spec.

- [ ] **Step 2: Add NOTICE attribution for the promoted method**

In `NOTICE`, under "Mathematical / Methodological References", add an entry crediting the geometric frame-LTR orientation method as promoted from luxury-lakehouse ADR-053 (`correct_frames_to_home_ltr`), and add `See NOTICE …` to the `orient_frames_to_ltr_by_geometry` docstring.

- [ ] **Step 3: Verify the Examples/attribution CI gates pass**

Run: `python -m pytest tests/ -m "not e2e" -k "examples or notice or public_api" -q`
Expected: PASS (every new public def has an Examples block + the new method is attributed).

---

## Task 9: Version bump (hard gate) + CHANGELOG + TODO

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: Bump version in all four files (must match)**

- `pyproject.toml`: `version = "4.33.0"`
- `silly_kicks/__init__.py`: `__version__ = "4.33.0"`
- `CHANGELOG.md`: new `## 4.33.0` entry summarizing the two builders, the promoted geometric orienter, `ball_z` recovery, and the lakehouse delete-and-depend consequence (its retrain trigger, not silly-kicks').
- `TODO.md`: delete the TF-23 Research-&-Future-Work entry's "Spec drafted … pending review" tail and replace with the shipped record once merged; for now mark it "plan written, in build".

- [ ] **Step 2: Verify the version-match gate**

Run:
```bash
python -c "import silly_kicks, tomllib; v=tomllib.load(open('pyproject.toml','rb'))['project']['version']; assert silly_kicks.__version__==v=='4.33.0', (silly_kicks.__version__, v)"
```
Expected: no assertion error.

---

## Task 10: Full verification + final review + single commit (approval-gated)

- [ ] **Step 1: Shift-left local gates**

Run:
```bash
ruff format --check . && ruff check . && pyright silly_kicks/ && python -m pytest tests/ -m "not e2e" -q
```
Expected: all green. Fix any failure before proceeding (do NOT commit red).

- [ ] **Step 2: Run `/final-review`**

Invoke the mandatory `/final-review` gate; address findings.

- [ ] **Step 3: Single commit — ONLY after explicit owner approval**

Do not commit without the owner's go-ahead (project policy). When approved:
```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(tracking): SkillCorner + Metrica bronze→frame builders single-source the coordinate/orientation/clock truth -- silly-kicks 4.33.0 (ADR-034, PR-S98)

Adds pure tracking.skillcorner / tracking.metrica bronze-consuming builders parallel to
sportec/gradientsports, recovering SkillCorner ball_z and orienting via a geometric net
promoted from lakehouse ADR-053. Closes ADR-031 Gate C on the shipping path via a
kloppy-independent event-anchored y-identity gate. Lakehouse delete-and-depends both its
builder copies + retires its orientation net (its retrain trigger, not silly-kicks').

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push + open PR (after approval)**

```bash
git push -u origin pr-s98-tf23-skillcorner-metrica-bronze-builders
gh pr create --fill --base main
```
Then: wait for CI green before any tag (project policy — never tag before CI green).

---

## Self-review checklist (completed by plan author)

**Spec coverage:** §4.1 surface → Tasks 3–5; §4.2 input contract → Tasks 3–4 (`EXPECTED_INPUT_COLUMNS` + missing-column guards); §4.3 orienter (promoted, schema-adapted, acceptance oracle) → Task 2; §4.4 pipeline → Tasks 3–4; §5.1(A) event-anchored gate → Task 6; §5.1(B) kloppy parity + z handling → Task 7; §5.2 unit/ET/idempotence/clock-single-source → Tasks 2–4; §5.3 migration gate → lakehouse-side (named in ADR-034, not a silly-kicks task); §2.2 ball_z recovery → Task 3 + Task 7 z-validation; §7 rollout / §8 non-goals / §9 O-decisions → ADR-034 (Task 8); version/CHANGELOG → Task 9.

**Open items deferred by design (not gaps):** TF-23b (GS-native-adapter geometric net) is a separate PR per spec §8; the lakehouse migration (delete both copies + retire net) is consumer-side per spec §7.

**Type consistency:** `convert_to_frames` signatures identical across both builders bar the Metrica `jersey_to_player_id`/`home_team_id` defaults; `orient_frames_to_ltr_by_geometry(frames, *, home_team_id, source, game_id)` called consistently; `_PERIOD_START_SECONDS` imported (not redefined) in both builders; schema aliases reference `KLOPPY_TRACKING_FRAMES_COLUMNS` so column lists match the gateway oracle (parity-critical).
