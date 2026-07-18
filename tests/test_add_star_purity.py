"""``add_*`` input-purity gate (ADR-033).

Every public ``add_*`` enrichment must be PURE: it must not mutate any
caller-supplied DataFrame / Series / ndarray, and — since every ``add_*`` adds
columns — it must return a NEW object (never the caller's input). This closes
the in-place-mutation class motivated by ``add_gk_distribution_metrics`` (which,
when ``gk_role`` was already present, assigned its four columns straight onto
the caller's frame).

Design (ADR-033):

* **Auto-enumerating + single-source.** One canonical ``PURITY_ENTRIES``
  registry keyed by ``"<package>:<add_name>"``; the parametrization, the
  per-package registered subsets, ``_resolve_fn``, and the heuristic's
  variant-count all derive from it. A new ``add_*`` that isn't registered — or
  that mutates — fails CI (two meta-assertions pin the surface to ``__all__``).
* **Build inputs ONCE and hold the reference.** The liveness gate's ``_std``
  rebuilds the input and caches ``_frames`` (so the object the function gets is
  not the one the test holds) — unusable for a mutation check. Here every
  ``build_inputs`` returns the exact objects passed to ``invoke``, and the
  harness snapshots THOSE references.
* **Per-function variants.** A helper that branches on column presence (e.g.
  ``if "gk_role" not in actions.columns``) gets one variant per branch — the
  gate otherwise only closes the default path. A best-effort AST heuristic
  nudges toward this; the real backstop is the contributor contract (CLAUDE.md).

See ADR-033 + ``docs/superpowers/specs/2026-06-16-add-star-purity-gate-design.md``.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl as sp
import silly_kicks.tracking as tracking
from silly_kicks.atomic import spadl as asp
from silly_kicks.atomic import tracking as atr
from silly_kicks.atomic.tracking import features as atf
from silly_kicks.tracking import features as F
from silly_kicks.tracking.pitch_control import PitchControlCache

# Reuse the liveness gate's UNCACHED builders (fresh each call) + the GS jersey
# fixture. We deliberately do NOT import its cached _frames/_xt/_frames_with_possession
# (passing a cached object into a pre-fix mutating helper would poison the shared
# cache: cross-test contamination + nondeterministic purity result).
from tests.tracking.test_aggregator_column_liveness import (
    _frow,
    _gs_jersey_inputs,
    make_actions,
    make_frames,
)

# ADR-041 opt-out: auto-enumerating gate -- it sweeps EVERY registered aggregator on defaults, so the OBSO
# family's synthetic-EPV notice is expected here and unrelated to what this gate asserts.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")


# ---------------------------------------------------------------------------
# The purity assertion
# ---------------------------------------------------------------------------
def _assert_pure(name, variant, inputs, invoke):
    # Split by type so each .equals() call is type-narrowed (DataFrame.equals wants a DataFrame, etc.).
    df_snaps = [(x, x.copy(deep=True)) for x in inputs if isinstance(x, pd.DataFrame)]
    sr_snaps = [(x, x.copy(deep=True)) for x in inputs if isinstance(x, pd.Series)]
    arr_snaps = [(x, x.copy()) for x in inputs if isinstance(x, np.ndarray)]
    out = invoke(inputs)
    for orig, snap in df_snaps:
        assert snap.equals(orig), f"{name}[{variant}] MUTATED a caller DataFrame in place"
    for orig, snap in sr_snaps:
        assert snap.equals(orig), f"{name}[{variant}] MUTATED a caller Series in place"
    for orig, snap in arr_snaps:
        # equal_nan requires an inexact dtype; int/object ndarrays raise -> guard it.
        eq = (
            np.array_equal(snap, orig, equal_nan=True)
            if np.issubdtype(orig.dtype, np.inexact)
            else np.array_equal(snap, orig)
        )
        assert eq, f"{name}[{variant}] MUTATED a caller ndarray in place"
    for x in inputs:
        assert out is not x, f"{name}[{variant}] returned the SAME object as an input (must return a copy)"


# ---------------------------------------------------------------------------
# Fresh, OWNED input builders
# ---------------------------------------------------------------------------
def _fresh_xt():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _fresh_links():
    links, _report = tracking.link_actions_to_frames(make_actions(), make_frames())
    return links


def _fresh_frames_with_possession():
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    frames = make_frames()
    return derive_team_in_possession(frames, infer_ball_carrier(frames))


def _spadl_actions(*, with_gk_role):
    """Standard SPADL actions; optionally carrying a pre-computed ``gk_role`` column."""
    df = make_actions()
    if with_gk_role:
        # Use REAL category members ("none" is not one -> would coerce to NaN).
        roles = (["distribution", "shot_stopping"] + ["distribution"] * 10)[: len(df)]
        df["gk_role"] = pd.Categorical(roles, categories=list(sp.utils._GK_ROLE_CATEGORIES))
    return df


def _atomic_actions(*, with_gk_role):
    """Minimal atomic-SPADL actions (x, y, dx, dy)."""
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "action_id": [0, 1],
            "team_id": [10, 10],
            "player_id": [1, 2],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
            "time_seconds": [10.0, 20.0],
            "x": [5.0, 50.0],
            "y": [34.0, 34.0],
            "dx": [55.0, 20.0],
            "dy": [0.0, 6.0],
        }
    )
    if with_gk_role:
        df["gk_role"] = pd.Categorical(
            ["distribution", "shot_stopping"], categories=list(asp.utils._GK_ROLE_CATEGORIES)
        )
    return df


def _atomic_actions_full():
    """Atomic-SPADL actions aligned to the liveness 5-window geometry (x=start, dx=end-start), so the
    atomic tracking mirrors see the same pass/shot/goalkick/cross domains the standard surface does.
    For PURITY (unlike liveness) the values needn't be exercised -- the mirror just runs + copies."""
    a = make_actions()
    return pd.DataFrame(
        {
            "game_id": a["game_id"],
            "period_id": a["period_id"],
            "action_id": a["action_id"],
            "team_id": a["team_id"],
            "player_id": a["player_id"],
            "defending_gk_player_id": a["defending_gk_player_id"],
            "type_id": a["type_id"],
            "type_name": a["type_name"],
            "result_id": a["result_id"],
            "result_name": a["result_name"],
            "bodypart_id": a["bodypart_id"],
            "bodypart_name": a["bodypart_name"],
            "time_seconds": a["time_seconds"],
            "x": a["start_x"],
            "y": a["start_y"],
            "dx": a["end_x"] - a["start_x"],
            "dy": a["end_y"] - a["start_y"],
        }
    )


def _fresh_atomic_links():
    links, _report = tracking.link_actions_to_frames(_atomic_actions(with_gk_role=False), make_frames())
    return links


def _fresh_atomic_full_links():
    links, _report = tracking.link_actions_to_frames(_atomic_actions_full(), make_frames())
    return links


def _frames_with_ghost():
    """make_frames + precomputed ghost columns on GK rows -> exercises add_ghost_gk's precompute
    short-circuit (`if "ghost_gk_x" in frames.columns`), which ALIASES ``ghost_frames = frames`` (a real
    mutation-of-the-caller risk this variant guards)."""
    f = make_frames()
    gk = f["is_goalkeeper"].astype(bool) & ~f["is_ball"].astype(bool)
    f["ghost_gk_x"] = np.where(gk, 52.5, np.nan)
    f["ghost_gk_y"] = np.where(gk, 34.0, np.nan)
    f["ghost_gk_density_spread"] = np.where(gk, 1.0, np.nan)
    return f


def _astd_inputs():
    return [_atomic_actions_full(), make_frames()]


def _axtf_inputs():
    return [_atomic_actions_full(), make_frames(), _fresh_xt()]


def _shot_goalmouth_inputs():
    """The TF-48 goalmouth fixture (post-contact ball flight straddling the goal plane)."""
    t_a = 60.0
    rows = []
    for i in range(-10, 34):  # 25 fps, t in [-0.4, +1.32] around the shot
        t = round(t_a + i / 25.0, 3)
        bt = max(t - t_a, 0.0)
        bx, by = 85.0 + 25.0 * bt, 30.0 + 2.0 * bt
        rows.append(_frow(1, 5, True, 4.0, 34.0, t))
        rows.append(_frow(2, 6, True, 101.0, 34.0, t))
        ball = _frow(pd.NA, pd.NA, False, bx, by, t, is_ball=True)
        ball["x"] = bx  # un-clamped: the crossing needs samples BEYOND x=105
        ball["z"] = max(4.0 * bt - 4.905 * bt * bt, 0.0)
        rows.append(ball)
    frames = pd.DataFrame(rows)
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["team_id"] = frames["team_id"].astype("Int64")
    actions = pd.DataFrame(
        {
            "game_id": [1],
            "action_id": [0],
            "period_id": [1],
            "time_seconds": [t_a],
            "team_id": pd.Series([5], dtype="int64"),
            "player_id": pd.Series([10], dtype="int64"),
            "start_x": [85.0],
            "start_y": [30.0],
            "end_x": [105.0],
            "end_y": [34.0],
            "type_id": [11],
            "type_name": ["shot"],
            "result_id": [1],
            "result_name": ["success"],
            "bodypart_id": [0],
            "bodypart_name": ["foot"],
        }
    )
    return [actions, frames]


# ---------------------------------------------------------------------------
# Invoke factories (default-arg binding to avoid late-closure capture)
# ---------------------------------------------------------------------------
def _std_inputs():
    return [make_actions(), make_frames()]


def _std_invoke(fn, **kw):
    return lambda inputs: fn(inputs[0], inputs[1], **kw)


def _xtf_inputs():
    return [make_actions(), make_frames(), _fresh_xt()]


def _xtf_invoke(fn, **kw):
    return lambda inputs: fn(inputs[0], inputs[1], inputs[2], home_team_id=5, **kw)


# ---------------------------------------------------------------------------
# The ONE canonical registry. Key = "<package>:<add_name>".
# Variant = (variant_name, build_inputs: () -> list, invoke: (inputs) -> result_df).
# ---------------------------------------------------------------------------
def _one(build, invoke):
    return [("default", build, invoke)]


PURITY_ENTRIES: dict[str, list[tuple]] = {
    # ---- spadl --------------------------------------------------------------
    # add_game_state branches on input FORMAT (`type_name`/`result_name` vs `type_id`/`result_id`); both
    # branches must return a copy -> a variant per input shape (also clears the AST heuristic's flag).
    "spadl:add_game_state": [
        ("with_names", lambda: [make_actions()], lambda i: sp.add_game_state(i[0])),
        (
            "ids_only",
            lambda: [make_actions().drop(columns=["type_name", "result_name", "bodypart_name"])],
            lambda i: sp.add_game_state(i[0]),
        ),
    ],
    "spadl:add_gk_role": _one(lambda: [make_actions()], lambda i: sp.add_gk_role(i[0])),
    "spadl:add_names": _one(lambda: [make_actions()], lambda i: sp.add_names(i[0])),
    "spadl:add_possessions": _one(lambda: [make_actions()], lambda i: sp.add_possessions(i[0])),
    "spadl:add_pre_shot_gk_context": _one(
        lambda: [make_actions(), make_frames()],
        lambda i: sp.add_pre_shot_gk_context(i[0], frames=i[1]),
    ),
    "spadl:add_restart_coordinates": _one(
        lambda: [make_actions(), make_frames()],
        lambda i: sp.add_restart_coordinates(i[0], frames=i[1]),
    ),
    "spadl:add_gk_distribution_metrics": [
        (
            "gk_role_present",
            lambda: [_spadl_actions(with_gk_role=True)],
            lambda i: sp.add_gk_distribution_metrics(i[0]),
        ),
        (
            "gk_role_absent",
            lambda: [_spadl_actions(with_gk_role=False)],
            lambda i: sp.add_gk_distribution_metrics(i[0]),
        ),
    ],
    # ---- atomic.spadl -------------------------------------------------------
    "atomic.spadl:add_gk_role": _one(lambda: [_atomic_actions(with_gk_role=False)], lambda i: asp.add_gk_role(i[0])),
    "atomic.spadl:add_names": _one(lambda: [_atomic_actions(with_gk_role=False)], lambda i: asp.add_names(i[0])),
    "atomic.spadl:add_possessions": _one(
        lambda: [_atomic_actions(with_gk_role=False)], lambda i: asp.add_possessions(i[0])
    ),
    "atomic.spadl:add_pre_shot_gk_context": _one(
        lambda: [_atomic_actions(with_gk_role=False)], lambda i: asp.add_pre_shot_gk_context(i[0])
    ),
    "atomic.spadl:add_gk_distribution_metrics": [
        (
            "gk_role_present",
            lambda: [_atomic_actions(with_gk_role=True)],
            lambda i: asp.add_gk_distribution_metrics(i[0]),
        ),
        (
            "gk_role_absent",
            lambda: [_atomic_actions(with_gk_role=False)],
            lambda i: asp.add_gk_distribution_metrics(i[0]),
        ),
    ],
    # ---- tracking -----------------------------------------------------------
    "tracking:add_action_context": _one(_std_inputs, _std_invoke(F.add_action_context)),
    "tracking:add_actor_pre_window": _one(_std_inputs, _std_invoke(F.add_actor_pre_window)),
    "tracking:add_cover_shadows": _one(_xtf_inputs, _xtf_invoke(F.add_cover_shadows)),
    # add_das branches on `links is not None and "frame_id" in links.columns` (caller-supplied links vs
    # internal linking); both paths must copy -> a variant per branch.
    "tracking:add_das": [
        ("internal_link", lambda: [make_actions(), _fresh_frames_with_possession()], lambda i: F.add_das(i[0], i[1])),
        (
            "supplied_links",
            lambda: [make_actions(), _fresh_frames_with_possession(), _fresh_links()],
            lambda i: F.add_das(i[0], i[1], links=i[2]),
        ),
    ],
    "tracking:add_defensive_line": _one(_std_inputs, _std_invoke(F.add_defensive_line, home_team_id=5, n=4)),
    "tracking:add_elastic_sync": _one(_std_inputs, _std_invoke(F.add_elastic_sync)),
    # add_ghost_gk branches on `"ghost_gk_x" in frames.columns` (precompute short-circuit aliasing frames)
    # -> a variant per branch, incl. the alias path.
    "tracking:add_ghost_gk": [
        ("compute", _std_inputs, _std_invoke(F.add_ghost_gk, home_team_id=5)),
        ("precomputed", lambda: [make_actions(), _frames_with_ghost()], _std_invoke(F.add_ghost_gk, home_team_id=5)),
    ],
    "tracking:add_gk_completion": _one(_std_inputs, _std_invoke(F.add_gk_completion)),
    "tracking:add_gk_influence": _one(_xtf_inputs, _xtf_invoke(F.add_gk_influence)),
    "tracking:add_gradientsports_player_ids": _one(
        lambda: list(_gs_jersey_inputs()),
        lambda i: tracking.add_gradientsports_player_ids(i[0], i[1], home_team_id=5, away_team_id=6)[0],
    ),
    "tracking:add_line_break": _one(_std_inputs, _std_invoke(F.add_line_break, home_team_id=5)),
    # Two variants: ADR-033 requires both branches of the new xt=/epv_grid= input mode
    # (ADR-041) -- the synthetic-default path and the injected-xT path build the EPV
    # grid differently and write a different provenance label.
    "tracking:add_obso": [
        ("default", _std_inputs, _std_invoke(F.add_obso)),
        ("xt_supplied", _xtf_inputs, lambda i: F.add_obso(i[0], i[1], xt=i[2])),
    ],
    "tracking:add_off_ball_context": _one(_std_inputs, _std_invoke(F.add_off_ball_context, home_team_id=5)),
    "tracking:add_off_ball_runs": _one(_std_inputs, _std_invoke(F.add_off_ball_runs, home_team_id=5)),
    # add_packing branches on params.require_secured (gates its OWN kernel-owned columns,
    # never caller inputs) -- a non-default-params variant still pins the params path.
    # TF-35 (ADR-042): the internal-link and caller-supplied-links branches take
    # different code paths through the aggregator's provenance merge, and the
    # pitch_control_cache variant additionally hands in a mutable object the helper
    # must not corrupt -- two variants per the contributor contract.
    "tracking:add_off_ball_run_values": [
        ("internal_link", _xtf_inputs, _xtf_invoke(F.add_off_ball_run_values)),
        (
            "supplied_links_and_cache",
            lambda: [make_actions(), make_frames(), _fresh_xt(), _fresh_links(), PitchControlCache()],
            lambda i: F.add_off_ball_run_values(i[0], i[1], i[2], home_team_id=5, links=i[3], pitch_control_cache=i[4]),
        ),
    ],
    "tracking:add_packing": [
        ("defaults", _std_inputs, _std_invoke(F.add_packing, home_team_id=5)),
        (
            "nondefault_params",
            _std_inputs,
            lambda i: F.add_packing(
                i[0], i[1], home_team_id=5, params=tracking.PackingParams(include_gk=True, back_line_n=3)
            ),
        ),
    ],
    "tracking:add_pausa": [
        ("default", _std_inputs, _std_invoke(F.add_pausa)),
        ("xt_supplied", _xtf_inputs, lambda i: F.add_pausa(i[0], i[1], xt=i[2])),
    ],
    "tracking:add_pitch_control": _one(_std_inputs, _std_invoke(F.add_pitch_control)),
    "tracking:add_player_influence": _one(_xtf_inputs, _xtf_invoke(F.add_player_influence)),
    "tracking:add_pre_shot_gk_angle": _one(_std_inputs, lambda i: F.add_pre_shot_gk_angle(i[0], frames=i[1])),
    "tracking:add_pre_shot_gk_position": _one(_std_inputs, _std_invoke(F.add_pre_shot_gk_position)),
    "tracking:add_pressure_on_actor": _one(_std_inputs, _std_invoke(F.add_pressure_on_actor)),
    # add_shape_graph / add_xcross_attempt / add_xshot_occurrence share the links-optimization branch
    # `links is not None and "frame_id" in links.columns` (body assigns a LOCAL frame-id set, never a
    # column) -> a supplied-links variant also confirms they don't mutate caller-supplied links.
    "tracking:add_shape_graph": [
        ("internal_link", _std_inputs, _std_invoke(F.add_shape_graph, home_team_id=5)),
        (
            "supplied_links",
            lambda: [make_actions(), make_frames(), _fresh_links()],
            lambda i: F.add_shape_graph(i[0], i[1], home_team_id=5, links=i[2]),
        ),
    ],
    "tracking:add_shot_goalmouth": _one(_shot_goalmouth_inputs, lambda i: F.add_shot_goalmouth(i[0], i[1])),
    "tracking:add_space_creation": [
        ("default", _std_inputs, _std_invoke(F.add_space_creation, home_team_id=5)),
        (
            "xt_supplied",
            _xtf_inputs,
            lambda i: F.add_space_creation(i[0], i[1], home_team_id=5, xt=i[2]),
        ),
    ],
    "tracking:add_structural_pass": _one(_std_inputs, _std_invoke(F.add_structural_pass, home_team_id=5)),
    "tracking:add_sync_score": _one(
        lambda: [make_actions(), _fresh_links()], lambda i: tracking.add_sync_score(i[0], i[1])
    ),
    "tracking:add_team_shape": _one(_std_inputs, _std_invoke(F.add_team_shape, home_team_id=5)),
    "tracking:add_xcross_attempt": [
        ("internal_link", _std_inputs, _std_invoke(tracking.add_xcross_attempt, home_team_id=5)),
        (
            "supplied_links",
            lambda: [make_actions(), make_frames(), _fresh_links()],
            lambda i: tracking.add_xcross_attempt(i[0], i[1], home_team_id=5, links=i[2]),
        ),
    ],
    "tracking:add_xshot_occurrence": [
        ("internal_link", _std_inputs, _std_invoke(tracking.add_xshot_occurrence, home_team_id=5)),
        (
            "supplied_links",
            lambda: [make_actions(), make_frames(), _fresh_links()],
            lambda i: tracking.add_xshot_occurrence(i[0], i[1], home_team_id=5, links=i[2]),
        ),
    ],
    "tracking:add_xt_gk": _one(_xtf_inputs, _xtf_invoke(F.add_xt_gk)),
    # ---- atomic.tracking (add_sync_score at the package level; 15 feature mirrors below) ----
    "atomic.tracking:add_sync_score": _one(
        lambda: [_atomic_actions(with_gk_role=False), _fresh_atomic_links()],
        lambda i: atr.add_sync_score(i[0], i[1]),
    ),
    "atomic.tracking:add_action_context": _one(_astd_inputs, _std_invoke(atf.add_action_context)),
    "atomic.tracking:add_actor_pre_window": _one(_astd_inputs, _std_invoke(atf.add_actor_pre_window)),
    "atomic.tracking:add_cover_shadows": _one(_axtf_inputs, _xtf_invoke(atf.add_cover_shadows)),
    "atomic.tracking:add_ghost_gk": [
        ("compute", _astd_inputs, _std_invoke(atf.add_ghost_gk, home_team_id=5)),
        (
            "precomputed",
            lambda: [_atomic_actions_full(), _frames_with_ghost()],
            _std_invoke(atf.add_ghost_gk, home_team_id=5),
        ),
    ],
    "atomic.tracking:add_gk_influence": _one(_axtf_inputs, _xtf_invoke(atf.add_gk_influence)),
    "atomic.tracking:add_off_ball_run_values": _one(_axtf_inputs, _xtf_invoke(atf.add_off_ball_run_values)),
    "atomic.tracking:add_packing": _one(_astd_inputs, _std_invoke(atf.add_packing, home_team_id=5)),
    "atomic.tracking:add_pitch_control": _one(_astd_inputs, _std_invoke(atf.add_pitch_control)),
    "atomic.tracking:add_player_influence": _one(_axtf_inputs, _xtf_invoke(atf.add_player_influence)),
    "atomic.tracking:add_pre_shot_gk_angle": _one(_astd_inputs, lambda i: atf.add_pre_shot_gk_angle(i[0], frames=i[1])),
    "atomic.tracking:add_pre_shot_gk_position": _one(_astd_inputs, _std_invoke(atf.add_pre_shot_gk_position)),
    "atomic.tracking:add_pressure_on_actor": _one(_astd_inputs, _std_invoke(atf.add_pressure_on_actor)),
    "atomic.tracking:add_shot_goalmouth": _one(_astd_inputs, _std_invoke(atf.add_shot_goalmouth)),
    "atomic.tracking:add_structural_pass": _one(_astd_inputs, _std_invoke(atf.add_structural_pass, home_team_id=5)),
    "atomic.tracking:add_xcross_attempt": [
        ("internal_link", _astd_inputs, _std_invoke(atf.add_xcross_attempt, home_team_id=5)),
        (
            "supplied_links",
            lambda: [_atomic_actions_full(), make_frames(), _fresh_atomic_full_links()],
            lambda i: atf.add_xcross_attempt(i[0], i[1], home_team_id=5, links=i[2]),
        ),
    ],
    "atomic.tracking:add_xshot_occurrence": [
        ("internal_link", _astd_inputs, _std_invoke(atf.add_xshot_occurrence, home_team_id=5)),
        (
            "supplied_links",
            lambda: [_atomic_actions_full(), make_frames(), _fresh_atomic_full_links()],
            lambda i: atf.add_xshot_occurrence(i[0], i[1], home_team_id=5, links=i[2]),
        ),
    ],
    "atomic.tracking:add_xt_gk": _one(_axtf_inputs, _xtf_invoke(atf.add_xt_gk)),
}

REGISTERED_NAMES = set(PURITY_ENTRIES)

_PKG_MOD = {
    "spadl": "silly_kicks.spadl",
    "atomic.spadl": "silly_kicks.atomic.spadl",
    "tracking": "silly_kicks.tracking",
    "atomic.tracking": "silly_kicks.atomic.tracking",
}


def _registered_for(prefix: str) -> set[str]:
    return {k.split(":", 1)[1] for k in PURITY_ENTRIES if k.split(":", 1)[0] == prefix}


SPADL_REGISTERED = _registered_for("spadl")
ATOMIC_SPADL_REGISTERED = _registered_for("atomic.spadl")
TRACKING_REGISTERED = _registered_for("tracking")
ATOMIC_TRACKING_REGISTERED = _registered_for("atomic.tracking")


def _exported_add_surface(prefix: str) -> set[str]:
    """The public ``add_*`` surface for a package prefix: its ``__all__`` UNION its ``.features``
    submodule's ``__all__`` (atomic.tracking exports ``add_sync_score`` at the package level but its 15
    feature mirrors only via ``atomic.tracking.features.__all__``)."""
    pkg = importlib.import_module(_PKG_MOD[prefix])
    names = {n for n in getattr(pkg, "__all__", ()) if n.startswith("add_")}
    feat = getattr(pkg, "features", None)
    if feat is not None:
        names |= {n for n in getattr(feat, "__all__", ()) if n.startswith("add_")}
    return names


def _resolve_fn(qkey: str):
    """Resolve a registered add_* by NAME via getattr on its package, falling back to the package's
    ``.features`` submodule (the atomic.tracking mirrors live there, not at the package level). The
    registry's ``invoke`` is an OPAQUE closure, NOT the bound fn, so it can't yield the source."""
    prefix, name = qkey.split(":", 1)
    pkg = importlib.import_module(_PKG_MOD[prefix])
    fn = getattr(pkg, name, None)
    if fn is None:
        feat = getattr(pkg, "features", None)
        fn = getattr(feat, name, None) if feat is not None else None
    assert fn is not None, f"{qkey} registered but not resolvable on {_PKG_MOD[prefix]} or its .features"
    return fn


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qkey,variant_name", [(q, v[0]) for q, vs in PURITY_ENTRIES.items() for v in vs])
def test_add_star_does_not_mutate_input(qkey, variant_name):
    variant = next(v for v in PURITY_ENTRIES[qkey] if v[0] == variant_name)
    _vname, build_inputs, invoke = variant
    _assert_pure(qkey, variant_name, build_inputs(), invoke)


# ---------------------------------------------------------------------------
# Meta-assertions: pin the gate surface to the public __all__ (review #1).
# ---------------------------------------------------------------------------
def _defined_add_defs(submodule) -> set[str]:
    return {
        n
        for n, o in inspect.getmembers(submodule, inspect.isfunction)
        if n.startswith("add_") and o.__module__ == submodule.__name__ and not n.startswith("_")
    }


def test_meta_registration_complete_per_package():
    """Per package: the purity-gate's registered add_* subset == the package's public add_* surface
    (``__all__`` UNION ``.features.__all__``). Mirrors the proven liveness pattern
    (test_aggregator_column_liveness.py::test_meta_surface_complete) but over the full export surface,
    so the 15 atomic.tracking.features mirrors are required to be wired."""
    for prefix, registered_subset in (
        ("spadl", SPADL_REGISTERED),
        ("atomic.spadl", ATOMIC_SPADL_REGISTERED),
        ("tracking", TRACKING_REGISTERED),
        ("atomic.tracking", ATOMIC_TRACKING_REGISTERED),
    ):
        exported = _exported_add_surface(prefix)
        assert exported == registered_subset, (
            f"{prefix}: purity-gate surface != public add_* exports "
            f"(unwired: {exported - registered_subset}, stale: {registered_subset - exported})"
        )


def test_meta_all_public_add_defs_are_exported():
    """Every public ``def add_*`` in the DEFINING submodules is exported by its owning package's public
    surface (``__all__`` UNION ``.features.__all__``) -- catches a public def added but forgotten from
    the export surface. ``o.__module__ == submodule.__name__`` is the valid discovery filter only inside
    the defining submodule (not at the re-export package)."""
    import silly_kicks.atomic.spadl.utils
    import silly_kicks.atomic.tracking.features
    import silly_kicks.spadl.utils
    import silly_kicks.tracking.features

    for submod, prefix in (
        (silly_kicks.spadl.utils, "spadl"),
        (silly_kicks.tracking.features, "tracking"),
        (silly_kicks.atomic.spadl.utils, "atomic.spadl"),
        (silly_kicks.atomic.tracking.features, "atomic.tracking"),
    ):
        missing = _defined_add_defs(submod) - _exported_add_surface(prefix)
        assert not missing, f"{submod.__name__}: public add_* defs missing from {prefix} export surface: {missing}"


# ---------------------------------------------------------------------------
# Best-effort branch-conditional heuristic (review #6: a NUDGE, not a guarantee).
# ---------------------------------------------------------------------------
def _branches_on_column_presence(fn) -> bool:
    """True iff fn has an ``if``-statement whose test compares a STRING-LITERAL column name (In/NotIn)
    against a ``*.columns`` Attribute AND whose body does something other than ``raise`` -- the
    branch-conditional-mutation shape (``if "gk_role" not in actions.columns: out = add_gk_role(out)``).
    Three discriminators keep it precise (NOT a proof -- best-effort):

    * literal-left operand -> fires on the bug shape, NOT on the idempotency provenance-skip guard
      ``if not any(c in out.columns for c in provenance_cols):`` (a ``Name``-left compare in a genexp);
    * ``ast.If`` only -> excludes validation list-comps (a ``comprehension``);
    * body is not purely ``raise`` -> excludes input-validation guards
      (``if "defending_gk_player_id" not in actions.columns: raise ValueError(...)``), which can't mutate."""
    fn = inspect.unwrap(fn)  # getsource on a DECORATED fn returns the wrapper, not the body
    try:
        tree = ast.parse(inspect.getsource(fn))
    except (OSError, TypeError):
        # dynamically-created / C-level / source-less helper -> cannot inspect; skip (don't crash).
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if node.body and all(isinstance(stmt, ast.Raise) for stmt in node.body):
            continue  # input-validation guard, not a conditional-mutation branch
        for cmp in ast.walk(node.test):
            if not (isinstance(cmp, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in cmp.ops)):
                continue
            if not isinstance(cmp.left, ast.Constant):  # literal column name -> the bug shape
                continue
            if any(isinstance(c, ast.Attribute) and c.attr == "columns" for c in cmp.comparators):
                return True
    return False


def test_meta_column_branching_helpers_have_multiple_variants():
    """BEST-EFFORT nudge (review #6), NOT a guarantee. The AST heuristic only recognizes the ONE shape
    the motivating bug took (``if <col> [not] in <df>.columns:``); a helper that branches a different
    way (a ``.get``, a try/except, a precomputed-mask flag) and mutates on one branch is NOT flagged.
    The real guarantee is per-variant coverage in PURITY_ENTRIES. Contributor contract (CLAUDE.md): any
    add_* that conditionally adds columns MUST register >=2 purity variants (present AND absent branch)."""
    _SINGLE_VARIANT_OK: dict[str, str] = {}  # qkey: reason (justified single-variant; e.g. non-mutating branch)
    for qkey, variants in PURITY_ENTRIES.items():
        if qkey in _SINGLE_VARIANT_OK or len(variants) >= 2:
            continue
        if _branches_on_column_presence(_resolve_fn(qkey)):
            raise AssertionError(
                f"{qkey} has an `if ... in <df>.columns` branch but <2 purity variants -- register the "
                f"branch's variant (the gate closes default-path mutation only), or allowlist with a reason"
            )


# ---------------------------------------------------------------------------
# Doc-accuracy: for helpers whose docstring claims an EXHAUSTIVE column list (Part C), the gate-observed
# emitted set must equal an explicit pinned set, and the docstring must NAME every column (review #4 +
# #5). Explicit frozensets (NOT a docstring-backtick parse -- parsing is fragile): the literal IS the
# contract a doc edit must keep in sync. Provenance (the 4 shared linkage columns) is subtracted.
# ---------------------------------------------------------------------------
_PROVENANCE = frozenset({"frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"})

# (qkey, variant_name) -> the feature columns the helper's docstring exhaustively claims.
_EXHAUSTIVE_EMITTED: dict[tuple[str, str], frozenset[str]] = {
    ("spadl:add_gk_distribution_metrics", "gk_role_present"): frozenset(
        {"gk_pass_length_m", "gk_pass_length_class", "is_launch", "gk_xt_delta"}
    ),
    ("tracking:add_off_ball_run_values", "internal_link"): frozenset(
        {
            "run_value_target",
            "n_disruptive_runs",
            "run_value_disruptive_sum",
            "n_valued_disruptive_runs",
            "run_value_enabled_pass",
        }
    ),
    ("tracking:add_shot_goalmouth", "default"): frozenset(
        {
            "shot_crossing_y",
            "shot_crossing_z",
            "shot_speed",
            "shot_time_to_goal_line",
            "shot_on_target_derived",
            "shot_crossing_source",
            "shot_crossing_confidence",
            "shot_fit_n_frames",
            "shot_fit_rmse",
            "shot_fit_end_reason",
            "shot_z_profile",
        }
    ),
    ("tracking:add_off_ball_context", "default"): frozenset(
        {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
            "line_break",
            "n_attackers_behind_line",
        }
    ),
}


@pytest.mark.parametrize("key", sorted(_EXHAUSTIVE_EMITTED))
def test_exhaustive_docstrings_match_emitted_columns(key):
    qkey, variant_name = key
    expected = _EXHAUSTIVE_EMITTED[key]
    variant = next(v for v in PURITY_ENTRIES[qkey] if v[0] == variant_name)
    _vname, build_inputs, invoke = variant
    inputs = build_inputs()
    out = invoke(inputs)
    emitted = (set(out.columns) - set(inputs[0].columns)) - _PROVENANCE
    assert emitted == expected, (
        f"{qkey}: gate-observed emitted feature columns {sorted(emitted)} != pinned exhaustive set "
        f"{sorted(expected)} (missing: {sorted(expected - emitted)}, extra: {sorted(emitted - expected)})"
    )
    doc = _resolve_fn(qkey).__doc__ or ""
    undocumented = {c for c in expected if c not in doc}
    assert not undocumented, f"{qkey}: docstring claims exhaustiveness but omits columns: {sorted(undocumented)}"


# ---------------------------------------------------------------------------
# Chain-purity: the real enrichment chain a consumer runs must leave the ORIGINAL inputs byte-unchanged
# (review #5 -- starts from built actions+frames, NOT a converter). Per-function purity implies chain
# purity; this documents the chained consumer contract the lakehouse exercises.
# ---------------------------------------------------------------------------
def test_enrichment_chain_does_not_mutate_original_inputs():
    actions, frames = make_actions(), make_frames()
    actions_before, frames_before = actions.copy(deep=True), frames.copy(deep=True)
    a = sp.add_gk_role(actions)
    a = sp.add_gk_distribution_metrics(a)
    a = F.add_off_ball_context(a, frames, home_team_id=5)
    a = F.add_shot_goalmouth(a, frames)
    assert actions_before.equals(actions), "chain mutated the original actions frame"
    assert frames_before.equals(frames), "chain mutated the original frames frame"
    assert len(a.columns) > len(actions.columns)  # the chain did real work
