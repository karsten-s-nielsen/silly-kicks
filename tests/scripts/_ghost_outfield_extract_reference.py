"""The `2474b73` ghost_outfield `_extract_work`, VERBATIM -- the test oracle for spec r4 D18 (b).

`scripts/train_ghost_outfield.py` at `2474b73` (git blob `43df69965276e27b3ded626f8459f93af469e344`): copied
unchanged below. The extraction then ran with the variant's feature set;
`tests/scripts/test_position_only_projection.py` checks that the faithful extraction, projected by its one consumer,
is the old `position_only` output. Never edit the bodies.
"""

from __future__ import annotations

from scripts.train_ghost_outfield import (
    _actions_for,
    _home_map,
    _home_team_for,
    extract_match,
    feature_set_for_variant,
)


def _extract_work(item, ctx):
    from scripts._driver_sources import read_dataset_parquet

    p, ns = item
    frames = read_dataset_parquet(p)
    return extract_match(
        frames,
        _actions_for(p, ns),
        feature_set_for_variant(ctx.params["variant"]),
        n_rearguard=ctx.params["n_rearguard"],
        home_team_id=_home_team_for(p, frames, _home_map(ctx)),
        subsample_fps=ctx.params["subsample_fps"],
    )
