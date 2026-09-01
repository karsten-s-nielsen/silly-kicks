"""TF-60 PR3 Task 8: the ghost-GK publisher validates any variant, incl. the 21-feature
position_only family (`sweeper_position_only`), not just the faithful 26.

`--verify-only` runs the whole load + contract + sample-prediction validation with NO network.
"""

import sys

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GHOST_GK_FEATURE_NAMES_POSITION_ONLY,
    GhostGkModel,
    GhostGridSpec,
)


def _fit_and_save(art, *, feature_set, grid_spec):
    names = GHOST_GK_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else GHOST_GK_FEATURE_NAMES
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(names))), columns=names)
    labels = pd.DataFrame(
        {"gk_x": rng.uniform(grid_spec.x_min, grid_spec.x_max, 200), "gk_y": rng.uniform(18, 50, 200)}
    )
    m = GhostGkModel(feature_set=feature_set, n_estimators=20, max_depth=3, grid_spec=grid_spec)
    m.fit(X, labels)
    m.save(art)


def test_publisher_verify_only_handles_position_only_feature_count(tmp_path, monkeypatch):
    # A sweeper_position_only artifact has 21 features; the publisher must build its sanity sample from
    # the artifact's own feature_names (21), not the hardcoded 26, or predict_mean fails on load-verify.
    from scripts.publish_ghost_gk import main

    art = tmp_path / "sweeper_position_only"
    _fit_and_save(art, feature_set="position_only", grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    monkeypatch.setattr(sys, "argv", ["prog", "--artifact-dir", str(art), "--verify-only"])
    main()  # must NOT raise (feature_names come from metadata -> the 21-col sample matches)


def test_publisher_verify_only_handles_faithful_sweeper(tmp_path, monkeypatch):
    from scripts.publish_ghost_gk import main

    art = tmp_path / "sweeper"
    _fit_and_save(art, feature_set="faithful", grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    monkeypatch.setattr(sys, "argv", ["prog", "--artifact-dir", str(art), "--verify-only"])
    main()  # must NOT raise
