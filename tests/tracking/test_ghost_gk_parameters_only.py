"""The parameters-only artifact contract (spec 2026-07-20, §2 + §4)."""

import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


@pytest.fixture(scope="module")
def bundled() -> GhostGkModel:
    return GhostGkModel.from_variant("default")


def test_save_omits_the_three_arrays(tmp_path, bundled):
    """save() writes a 1.3.0 npz that contains none of the per-sample arrays."""
    bundled.save(tmp_path / "m")
    with np.load(tmp_path / "m" / "rfcde_weights.npz", allow_pickle=True) as z:
        files = set(z.files)
    assert "training_gk_x" not in files
    assert "training_gk_y" not in files
    assert "training_leaves" not in files
    # tree ensembles + baselines are still present
    assert "n_trees" in files and "n_trees_y" in files
    assert "baseline_x" in files and "baseline_y" in files


def test_metadata_marks_parameters_only(tmp_path, bundled):
    bundled.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["version"] == "1.3.0"
    assert meta["stores_training_data"] is False


def test_predict_mean_byte_identical_after_strip(tmp_path, bundled):
    """The served position is unchanged by dropping the arrays."""
    golden = np.load("tests/tracking/fixtures/ghost_gk_kde_golden.npz", allow_pickle=True)
    cols = [str(c) for c in golden["feature_cols"]]
    X = pd.DataFrame(golden["features"][:20], columns=cols)[GHOST_GK_FEATURE_NAMES]
    before = bundled.predict_mean(X)

    bundled.save(tmp_path / "m")
    reloaded = GhostGkModel.load(tmp_path / "m")
    after = reloaded.predict_mean(X)
    assert np.array_equal(before, after)  # byte-identical, not approx


def test_sklearn_version_preserved_not_restamped(tmp_path, bundled, monkeypatch):
    """Migration must NOT rewrite the recorded training-time sklearn version."""
    recorded = bundled._sklearn_version
    assert recorded is not None
    # Simulate a different runtime sklearn than the one the model was fit under.
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", recorded + "-different")
    bundled.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["sklearn_version"] == recorded  # preserved, not the runtime value


def test_predict_density_message_names_the_cause(tmp_path, bundled):
    """A loaded parameters-only model gives a density-specific error, not 'not fitted'."""
    bundled.save(tmp_path / "m")
    reloaded = GhostGkModel.load(tmp_path / "m")
    X = pd.DataFrame(np.zeros((1, len(GHOST_GK_FEATURE_NAMES))), columns=GHOST_GK_FEATURE_NAMES)
    with pytest.raises(RuntimeError, match=r"parameters-only|density.*not.*available|fit.*locally"):
        reloaded.predict_density(X)


def test_compute_ghost_gk_emits_two_columns_no_kde_backend():
    """compute_ghost_gk serves positions only; no density column, no kde_backend kwarg."""
    import inspect

    from silly_kicks.tracking._ghost_gk import compute_ghost_gk

    sig = inspect.signature(compute_ghost_gk)
    assert "kde_backend" not in sig.parameters
    doc = compute_ghost_gk.__doc__ or ""
    assert "ghost_gk_density_spread" not in doc


def test_add_ghost_gk_and_xfns_have_no_kde_backend():
    import inspect

    from silly_kicks.tracking.features import add_ghost_gk, ghost_gk_xfns

    assert "kde_backend" not in inspect.signature(add_ghost_gk).parameters
    assert "kde_backend" not in inspect.signature(ghost_gk_xfns).parameters


def test_ghost_gk_xfns_emits_six_columns_not_nine():
    """2 metric columns x 3 gamestate slots (spread retired) = 6, via the frames=None contract.

    frames=None is the ADR-005 no-frames path: the xfn early-returns named NaN columns WITHOUT
    needing a GK-bearing frame (verified on the pre-strip xfn: this exact call returned 9 named
    columns, 3 metrics x 3 slots). It exercises the column-NAME contract this task changes
    (col_names 3->2, slot loop stays 3), not the empty-data path. The frames-PRESENT emission
    path is covered by the integration tests + the liveness gate."""
    import pandas as pd

    from silly_kicks.tracking.features import ghost_gk_xfns

    (xfn,) = ghost_gk_xfns(home_team_id=1)
    out = xfn([pd.DataFrame(index=range(2))] * 3, None)
    assert list(out.columns) == [
        "ghost_gk_x_a0",
        "ghost_gk_y_a0",
        "ghost_gk_x_a1",
        "ghost_gk_y_a1",
        "ghost_gk_x_a2",
        "ghost_gk_y_a2",
    ]
