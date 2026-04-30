"""add_gk_role goalkeeper_ids parameter — backward-compat + new behavior.

Covers the 2.5.0 opt-in coverage feature:
- Backward-compat: goalkeeper_ids=None (default) preserves byte-for-byte
  behavior of the pre-2.5.0 add_gk_role.
- Rule (a): goalkeeper_ids provided + clean player_ids → distribution
  detection extends to known-GK rows whose preceding action was keeper.
- Rule (b): goalkeeper_ids provided + NaN player_ids → coarser team-based
  fallback tags non-keeper rows after a keeper action by the same team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.atomic.spadl.config as atomic_spadlcfg
import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as std_utils


def _make_std_actions(player_ids, type_ids, team_ids, *, n_rows: int) -> pd.DataFrame:
    """Build a minimal valid standard-SPADL frame with explicit per-row inputs."""
    foot_id = spadlcfg.bodypart_id["foot"]
    success_id = spadlcfg.result_id["success"]
    return pd.DataFrame(
        {
            "game_id": [1] * n_rows,
            "period_id": [1] * n_rows,
            "action_id": list(range(n_rows)),
            "team_id": team_ids,
            "player_id": pd.array(player_ids, dtype="float64"),
            "type_id": type_ids,
            "result_id": [success_id] * n_rows,
            "result_name": ["success"] * n_rows,
            "bodypart_id": [foot_id] * n_rows,
            "bodypart_name": ["foot"] * n_rows,
            "type_name": ["pass"] * n_rows,
            "time_seconds": [float(i) for i in range(n_rows)],
            "start_x": [10.0 + i for i in range(n_rows)],
            "start_y": [10.0] * n_rows,
            "end_x": [20.0 + i for i in range(n_rows)],
            "end_y": [10.0] * n_rows,
        }
    )


# ---------------------------------------------------------------------------
# Backward-compat: goalkeeper_ids=None preserves pre-2.5.0 behavior.
# ---------------------------------------------------------------------------


def test_std_default_none_preserves_existing_behavior() -> None:
    """add_gk_role() with no goalkeeper_ids argument == add_gk_role(goalkeeper_ids=None)."""
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[100.0, 100.0, 200.0, 200.0, 100.0],
        type_ids=[pass_id, save_id, pass_id, pass_id, pass_id],
        team_ids=[10, 10, 20, 20, 10],
        n_rows=5,
    )
    out_default = std_utils.add_gk_role(actions)
    out_explicit_none = std_utils.add_gk_role(actions, goalkeeper_ids=None)
    pd.testing.assert_series_equal(
        out_default["gk_role"].astype(object),
        out_explicit_none["gk_role"].astype(object),
        check_names=False,
    )


def test_atomic_default_none_preserves_existing_behavior() -> None:
    """Atomic counterpart: goalkeeper_ids=None == no goalkeeper_ids passed."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    actions = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "period_id": [1] * 5,
            "action_id": list(range(5)),
            "team_id": [10, 10, 20, 20, 10],
            "player_id": pd.array([100.0, 100.0, 200.0, 200.0, 100.0], dtype="float64"),
            "type_id": [pass_id, save_id, pass_id, pass_id, pass_id],
            "type_name": ["pass"] * 5,
            "bodypart_id": [0] * 5,
            "bodypart_name": ["foot"] * 5,
            "time_seconds": [float(i) for i in range(5)],
            "x": [10.0 + i for i in range(5)],
            "y": [10.0] * 5,
            "dx": [10.0] * 5,
            "dy": [0.0] * 5,
        }
    )
    out_default = atomic_utils.add_gk_role(actions)
    out_explicit_none = atomic_utils.add_gk_role(actions, goalkeeper_ids=None)
    pd.testing.assert_series_equal(
        out_default["gk_role"].astype(object),
        out_explicit_none["gk_role"].astype(object),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Rule (a) — known-GK match: clean player_id, GK in goalkeeper_ids set.
# ---------------------------------------------------------------------------


def test_std_rule_a_known_gk_match_does_not_regress() -> None:
    """When same_player matching already covers a row, adding goalkeeper_ids
    must NOT regress that detection (the rule is additive, not replacing).
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[200.0, 100.0, 100.0, 200.0],
        type_ids=[pass_id, save_id, pass_id, pass_id],
        team_ids=[20, 10, 10, 20],
        n_rows=4,
    )
    out = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out.iloc[2]["gk_role"] == "distribution"


def test_std_rule_a_extends_to_gk_pass_with_no_strict_same_player_match() -> None:
    """Rule (a) is the load-bearing extension: keeper save by GK 100, then a
    pass by some-other-GK (101) on the same team — strict same_player would
    not match (different player_id), but goalkeeper_ids includes both 100 and
    101, AND prev was keeper, AND same team → tag as distribution.
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[100.0, 101.0, 200.0],
        type_ids=[save_id, pass_id, pass_id],
        team_ids=[10, 10, 20],
        n_rows=3,
    )
    # Without goalkeeper_ids: row 1 NOT tagged as distribution.
    out_no_gks = std_utils.add_gk_role(actions)
    assert pd.isna(out_no_gks.iloc[1]["gk_role"])
    # With goalkeeper_ids: row 1 IS tagged as distribution.
    out_with_gks = std_utils.add_gk_role(actions, goalkeeper_ids={100.0, 101.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


# ---------------------------------------------------------------------------
# Rule (b) — NaN-team fallback: both NaN player_ids, same team_id.
# ---------------------------------------------------------------------------


def test_std_rule_b_nan_team_fallback_tags_distribution() -> None:
    """When both current and shifted player_ids are NaN AND same team AND
    prev was keeper, tag as distribution (lakehouse coverage gap fix).
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan, 200.0],
        type_ids=[save_id, pass_id, pass_id],
        team_ids=[10, 10, 20],
        n_rows=3,
    )
    # Without goalkeeper_ids: row 1 NOT tagged (NaN==NaN is False).
    out_no_gks = std_utils.add_gk_role(actions)
    assert pd.isna(out_no_gks.iloc[1]["gk_role"])
    # With goalkeeper_ids (any non-empty set signals opt-in): row 1 IS tagged.
    out_with_gks = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


def test_std_rule_b_nan_team_fallback_respects_team_boundary() -> None:
    """Rule (b) requires same team — a NaN-pass by team 20 after NaN-keeper
    by team 10 must NOT be tagged as distribution.
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan],
        type_ids=[save_id, pass_id],
        team_ids=[10, 20],
        n_rows=2,
    )
    out = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert pd.isna(out.iloc[1]["gk_role"])  # different team — no fallback


def test_atomic_rule_b_nan_team_fallback_tags_distribution() -> None:
    """Atomic counterpart of test_std_rule_b_nan_team_fallback_tags_distribution."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    actions = pd.DataFrame(
        {
            "game_id": [1] * 3,
            "period_id": [1] * 3,
            "action_id": [0, 1, 2],
            "team_id": [10, 10, 20],
            "player_id": pd.array([np.nan, np.nan, 200.0], dtype="float64"),
            "type_id": [save_id, pass_id, pass_id],
            "type_name": ["pass"] * 3,
            "bodypart_id": [0] * 3,
            "bodypart_name": ["foot"] * 3,
            "time_seconds": [0.0, 1.0, 2.0],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 10.0, 10.0],
            "dx": [10.0, 10.0, 10.0],
            "dy": [0.0, 0.0, 0.0],
        }
    )
    out_no_gks = atomic_utils.add_gk_role(actions)
    assert pd.isna(out_no_gks.iloc[1]["gk_role"])
    out_with_gks = atomic_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


# ---------------------------------------------------------------------------
# Empty set edge case — passing an empty goalkeeper_ids should be valid.
# ---------------------------------------------------------------------------


def test_std_empty_goalkeeper_ids_set() -> None:
    """Passing an empty set should opt into the NaN-team fallback (rule b)
    but NOT match anyone via rule (a).
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan],
        type_ids=[save_id, pass_id],
        team_ids=[10, 10],
        n_rows=2,
    )
    out = std_utils.add_gk_role(actions, goalkeeper_ids=set())
    assert out.iloc[1]["gk_role"] == "distribution"
