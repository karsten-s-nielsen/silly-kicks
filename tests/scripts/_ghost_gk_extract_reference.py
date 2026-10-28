"""The `2474b73` ghost_gk per-variant `_extract_work`, VERBATIM -- the test oracle for spec r4 D18 (a).

`scripts/train_ghost_gk.py` at `2474b73` (git blob `eb0cb2939fe7834e665a7e1781dca84dbb5e7b6d`): its body is copied
unchanged below, with the module-level `_UNIT_BIAS` it fills. The helpers it calls are imported from the trainer, where
they are unchanged. `tests/scripts/test_ghost_gk_superset_select.py` checks the superset `extract` plus the per-variant
`select` against it, unit by unit and variant by variant. Never edit the body: it is the pre-change behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_ghost_gk import (
    _BIAS_COUNTERS,
    _assert_no_column_collision,
    _carrier,
    _home_map,
    _pack,
    grid_spec_from_args,
    keeper_detection_mask_or_none,
)

#: The per-unit selection-bias sums the oracle fills (its own dict: the trainer's `_UNIT_BIAS` is `select`'s now).
_UNIT_BIAS: dict[str, float] = {}


def _extract_work(item, ctx):
    """The legacy per-game ``_work`` for every game in the unit's file (sorted), packs concatenated.

    The bias counters are summed over the file's games (one game per staged flat file, so the sums are
    the legacy per-game sums). A game with no home-team mapping is skipped, as the legacy did; a unit
    with NO mapped game is an exclusion, so the skip is reported in the stage summary."""
    from scripts._item_outcome import ItemExcluded
    from silly_kicks.tracking import prepare_ghost_gk_training_data
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, keeper_detection_mask

    _assert_no_column_collision(GHOST_GK_FEATURE_NAMES)
    _ref, file_frames, actions = item
    p = ctx.params
    cp = _carrier(p)
    grid_spec = grid_spec_from_args(p["grid_x_max"])
    home_map = _home_map(ctx)
    actions_by_game = dict(list(actions.groupby("game_id"))) if actions is not None and len(actions) else {}
    prov = str(file_frames["source_provider"].iloc[0]) if "source_provider" in file_frames.columns else "unknown"
    _UNIT_BIAS.clear()
    _UNIT_BIAS.update(dict.fromkeys(_BIAS_COUNTERS, 0.0))
    totals = _UNIT_BIAS
    packs = []
    game_ids = sorted(file_frames["game_id"].unique())
    unmapped = [g for g in game_ids if home_map.get(str(g)) is None]
    for game_id in unmapped:
        print(f"    SKIP game {game_id}: no home_team_id in mapping")
    if game_ids and len(unmapped) == len(game_ids):
        raise ItemExcluded("no home_team_id in mapping", details={"game_ids": [str(g) for g in unmapped]})
    for game_id in game_ids:
        home = home_map.get(str(game_id))
        if home is None:
            continue
        game_frames = file_frames[file_frames["game_id"] == game_id]
        game_actions = actions_by_game.get(game_id) if actions_by_game else None
        feats, labs, meta = prepare_ghost_gk_training_data(
            game_frames,
            home_team_id=home,
            actions=game_actions,
            subsample_fps=p["subsample_fps"],
            carrier_params=cp,
            feature_set=p["feature_set"],
            grid_spec=grid_spec,
            return_meta=True,
        )
        if not len(feats):
            continue
        if prov == "skillcorner" and len(meta):
            _vis = keeper_detection_mask_or_none(meta["gk_visibility"])
            if _vis is not None:
                _depth = meta["gk_x_gr"].to_numpy(dtype=float)
                _b2k = np.hypot(
                    feats["ball_x"].to_numpy() - meta["gk_x_gr"].to_numpy(),
                    feats["ball_y"].to_numpy() - meta["gk_y_gr"].to_numpy(),
                )
                _vm = _vis.to_numpy()
                for k, v in {
                    "bias_depth_detected_sum": float(np.sum(_depth[_vm])),
                    "bias_depth_detected_n": int(_vm.sum()),
                    "bias_depth_undetected_sum": float(np.sum(_depth[~_vm])),
                    "bias_depth_undetected_n": int((~_vm).sum()),
                    "bias_b2k_detected_sum": float(np.nansum(_b2k[_vm])),
                    "bias_b2k_detected_n": int((~np.isnan(_b2k[_vm])).sum()),
                    "bias_b2k_undetected_sum": float(np.nansum(_b2k[~_vm])),
                    "bias_b2k_undetected_n": int((~np.isnan(_b2k[~_vm])).sum()),
                }.items():
                    totals[k] += v
        keep = keeper_detection_mask(meta["gk_visibility"], provider=prov)
        feats = feats[keep].reset_index(drop=True)
        labs = labs[keep].reset_index(drop=True)
        meta = meta[keep].reset_index(drop=True)
        if not len(feats):
            print(f"  SKIP {prov}/{game_id}: no detected-keeper frames")
            continue
        packs.append(_pack(feats, labs, game_id=game_id, provider=prov, keepers=meta["gk_player_id"].astype(str)))
    return pd.concat(packs, ignore_index=True) if packs else None
