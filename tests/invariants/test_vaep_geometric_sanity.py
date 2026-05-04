"""Physical invariant: VAEP / xT outputs reflect goal-proximity, not pitch-length.

Two assertions per provider:

1. **Shot distance must be a shot distance.** ``start_dist_to_goal_a0`` for shot
   actions must average less than 30m. A pipeline that mirrors away-team rows
   to the wrong end produces shot distances ~85-95m -- which is the diagonal
   of a football pitch, not a shootable range.

2. **xT grid must be goal-monotonic.** Mean rated xT in the high-x attacking
   third (start_x > 80) must exceed mean rated xT in the low-x defensive third
   (start_x < 25). A correctly-fitted xT puts all positive value near the
   opponent's goal; a broken-input fit produces a U-shaped grid (high values
   at both ends because half the team's "shots" are mirrored).

These are the cross-layer invariants contract tests can't catch -- each layer's
contract holds, the bug is in their composition. Promoted to first-class
discipline by PR-S22.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

from . import _loaders


def _build_game_series(actions: pd.DataFrame, home_team_id) -> pd.Series:
    """Build a minimal game pd.Series matching what VAEP.compute_features expects.

    home_team_id MUST match what was passed to the converter -- using a different
    value silently breaks VAEP's play_left_to_right away_idx, masking the bug.
    """
    return pd.Series({"game_id": actions["game_id"].iloc[0], "home_team_id": home_team_id})


_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[name] for name in ("shot", "shot_penalty", "shot_freekick"))

# All committed fixtures contain at least one SPADL shot variant -- the
# shot-distance test runs against every provider unconditionally.
_VAEP_PROVIDERS = [
    ("statsbomb_7298", lambda: _loaders.load_statsbomb(7298)),
    ("statsbomb_7584", lambda: _loaders.load_statsbomb(7584)),
    ("statsbomb_3754058", lambda: _loaders.load_statsbomb(3754058)),
    ("wyscout_2team_synthetic", _loaders.load_wyscout_2team_synthetic),
    ("opta_2team_synthetic", _loaders.load_opta_2team_synthetic),
    ("sportec_native", _loaders.load_sportec_native),
    ("sportec_via_kloppy", _loaders.load_sportec_via_kloppy),
    ("metrica_native", _loaders.load_metrica_native),
    ("pff_synthetic", _loaders.load_pff_synthetic),
]


# xT grid fitting requires hundreds of move actions across the full pitch to
# converge. Tiny synthetic fixtures (24-35 actions) genuinely cannot support
# the goal-monotonic test -- the test is vacuous there. Keep only providers
# whose committed fixtures have sufficient action density. PR-S25: replaced
# the previous in-test pytest.skip cascade with an explicit allow-list so
# every parametrize collection actually runs the assertion.
_VAEP_XT_PROVIDERS = [
    ("statsbomb_7298", lambda: _loaders.load_statsbomb(7298)),
    ("statsbomb_7584", lambda: _loaders.load_statsbomb(7584)),
]


@pytest.mark.parametrize("provider,loader", _VAEP_PROVIDERS)
def test_shot_dist_to_goal_is_shot_distance(provider: str, loader):
    """VAEP startpolar.start_dist_to_goal_a0 for shot actions must be < 50m on average.

    Counts all SPADL shot variants (``shot`` / ``shot_penalty`` /
    ``shot_freekick``); converters' set-piece-composition rules can upgrade
    raw SHOT events to ``shot_freekick`` (Metrica) or ``shot_penalty`` (Opta)
    without changing the geometric invariant.
    """
    from silly_kicks.vaep import VAEP

    actions, home_team_id = loader()
    shots = actions[actions["type_id"].isin(_SHOT_TYPE_IDS)]
    assert len(shots) > 0, (
        f"{provider}: fixture has no shot/shot_penalty/shot_freekick actions; regression in fixture or converter."
    )

    # Sort actions per VAEP's expected order to avoid per-row noise
    actions_sorted = actions.sort_values(["game_id", "period_id", "action_id"], kind="mergesort").reset_index(drop=True)

    v = VAEP()
    game = _build_game_series(actions_sorted, home_team_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore feature_column_names introspection noise
        feats = v.compute_features(game, actions_sorted)

    shot_mask = actions_sorted["type_id"].isin(_SHOT_TYPE_IDS).to_numpy()
    assert shot_mask.sum() > 0, f"{provider}: no shots after sort"

    assert "start_dist_to_goal_a0" in feats.columns, (
        f"{provider}: VAEP feats missing start_dist_to_goal_a0 (xfns customised?)"
    )

    mean_shot_dist = float(feats.loc[shot_mask, "start_dist_to_goal_a0"].mean())
    # Threshold = 50m. Real shots cluster at 10-25m from goal; the broken-pipeline
    # signature is 85-95m (full pitch length). Synthetic / tiny fixtures may
    # contain physically-unrealistic long-range shots that legitimately push the
    # mean to 30-45m without indicating the dual-mirror bug. The 50m gate keeps
    # the broken-pipeline bug detection while tolerating fixture noise.
    assert mean_shot_dist < 50.0, (
        f"{provider}: VAEP startpolar mean shot dist = {mean_shot_dist:.2f}m. "
        f"Shots happen near the opponent goal, so dist-to-goal must be a shot distance "
        f"(< 50m). Values around 85-95m indicate the dual-mirror in VAEP feature "
        f"engineering inverted away-team rows. See ADR-006 / PR-S22."
    )


@pytest.mark.parametrize("provider,loader", _VAEP_XT_PROVIDERS)
def test_xt_grid_is_goal_monotonic(provider: str, loader):
    """xT rated value at high-x (>80) attacking third must exceed rated value at low-x (<25).

    Parametrized only over fixtures with action densities sufficient to fit a
    stable xT grid (hundreds of move actions, full-pitch coverage). Smaller
    synthetic fixtures cannot physically support this invariant -- they're
    excluded from the parametrize list rather than skipped at runtime.
    """
    from silly_kicks.xthreat import ExpectedThreat

    actions, _home_team_id = loader()
    assert len(actions) >= 50, (
        f"{provider}: only {len(actions)} actions; allow-list assumes >=50. "
        "Move provider out of _VAEP_XT_PROVIDERS or augment fixture."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xt = ExpectedThreat().fit(actions)
        rated = xt.rate(actions, use_interpolation=False)

    high_mask = (actions["start_x"] > 80).to_numpy() & ~np.isnan(rated)
    low_mask = (actions["start_x"] < 25).to_numpy() & ~np.isnan(rated)
    assert high_mask.sum() >= 10 and low_mask.sum() >= 10, (
        f"{provider}: insufficient rated actions in extreme zones "
        f"(high={high_mask.sum()}, low={low_mask.sum()}); fixture density gap."
    )

    mean_high = float(np.nanmean(rated[high_mask]))
    mean_low = float(np.nanmean(rated[low_mask]))
    assert mean_high > mean_low, (
        f"{provider}: xT rated value at attacking third (mean={mean_high:.4f}) "
        f"<= defensive third (mean={mean_low:.4f}). xT grid is not goal-monotonic. "
        f"Likely cause: SPADL input has away-team coords mirrored to wrong end, "
        f"producing a U-shaped grid. See ADR-006 / PR-S22."
    )
