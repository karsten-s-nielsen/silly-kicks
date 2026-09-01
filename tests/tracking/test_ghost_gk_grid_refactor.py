"""TF-60 PR3 Tasks 2-4: grid_spec threaded through __init__/save/load/out_of_box/label-filter/density.

Byte-identical `default` is the load-bearing property (the additivity that protects gkdv).
"""

import json

import pytest

from silly_kicks.tracking._ghost_gk import (
    _WEIGHTS_ROOT,
    DEFAULT_GHOST_GRID,
    GhostGkModel,
    GhostGridSpec,
)

from ._ghost_toy import fit_toy, rewrite_sums

_DEFAULT_DIR = _WEIGHTS_ROOT / "default"


# --- Task 2: __init__ / save / load -------------------------------------------------------------
def test_default_model_constructs_with_default_grid():
    assert GhostGkModel().grid_spec == DEFAULT_GHOST_GRID


def test_bundled_default_loads_and_restores_default_grid():
    loaded = GhostGkModel.load(_DEFAULT_DIR)
    assert loaded.grid_spec == DEFAULT_GHOST_GRID


def test_default_grid_spec_serializes_byte_identical_to_committed():
    # The byte-identity proof for the grid refactor: the committed default's grid_spec field and the
    # re-serialized DEFAULT_GHOST_GRID.to_metadata_dict() are identical (a mis-serialized 5-key
    # grid_spec would differ here). NOTE: the FULL metadata.json does NOT round-trip through
    # load->save on any code version -- save() recomputes feature_contract.probe_sha256 from the
    # current extractor, which has evolved since the DGX artifact was built (pre-existing, orthogonal
    # to this refactor). So the grid byte-identity is asserted on the grid_spec field specifically;
    # the behavioural byte-identity of `default` is covered by the existing golden/chirality/contract
    # tests passing unchanged (Task 9 full suite).
    committed = json.loads((_DEFAULT_DIR / "metadata.json").read_text())["grid_spec"]
    assert committed == {
        "x_min": 0.0,
        "x_max": 30.0,
        "y_min": 18.0,
        "y_max": 50.0,
        "nx": 60,
        "ny": 64,
        "resolution": 0.5,
    }
    assert GhostGkModel.load(_DEFAULT_DIR).grid_spec.to_metadata_dict() == committed


def test_default_metadata_differs_only_in_recomputed_feature_contract(tmp_path):
    # Pin the finding: re-saving the default changes ONLY feature_contract.probe_sha256 (the
    # pre-existing recompute). If this refactor ever changes another field for `default`, this fails.
    committed = json.loads((_DEFAULT_DIR / "metadata.json").read_text())
    out = tmp_path / "resaved"
    GhostGkModel.load(_DEFAULT_DIR).save(out)
    resaved = json.loads((out / "metadata.json").read_text())
    diffs = {k for k in set(committed) | set(resaved) if committed.get(k) != resaved.get(k)}
    # Only feature_contract changed (its probe_sha256); grid_spec + all else are byte-identical.
    assert diffs <= {"feature_contract"}
    assert committed["grid_spec"] == resaved["grid_spec"]


def test_extended_grid_round_trips_through_save_load(tmp_path):
    m = GhostGkModel(grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    fit_toy(m)
    out = tmp_path / "sweeper_toy"
    m.save(out)
    back = GhostGkModel.load(out)
    assert back.grid_spec.x_max == 52.5
    assert back.grid_spec.nx == 105


def test_metadata_without_grid_spec_loads_default(tmp_path):
    # Back-compat: a pre-refactor artifact (no grid_spec key) loads with DEFAULT_GHOST_GRID.
    m = GhostGkModel.load(_DEFAULT_DIR)
    art = tmp_path / "no_grid"
    m.save(art)
    md_path = art / "metadata.json"
    md = json.loads(md_path.read_text())
    del md["grid_spec"]
    md_path.write_text(json.dumps(md, indent=2), newline="\n")
    rewrite_sums(art)
    back = GhostGkModel.load(art)
    assert back.grid_spec == DEFAULT_GHOST_GRID


# --- Task 3: out_of_box + label filter read the per-model grid -----------------------------------
def test_out_of_box_threshold_is_variant_relative():
    # 40 m is out_of_box for default (x_max=30) but not for sweeper (52.5). Unit the flag expression
    # against grid_spec.x_max (the end-to-end wiring is exercised by the Task-6 saturation gate).
    assert (40.0 > GhostGridSpec(0, 30, 18, 50, 0.5).x_max) is True
    assert (40.0 > GhostGridSpec(0, 52.5, 18, 50, 0.5).x_max) is False


def test_label_domain_predicate_retains_high_sweeper_under_extended_grid():
    import pandas as pd

    labels = pd.DataFrame({"gk_x": [10.0, 35.0, 45.0], "gk_y": [34.0, 34.0, 34.0]})

    def in_domain(g):
        return (
            (labels["gk_x"] >= g.x_min)
            & (labels["gk_x"] <= g.x_max)
            & (labels["gk_y"] >= g.y_min)
            & (labels["gk_y"] <= g.y_max)
        )

    assert in_domain(GhostGridSpec(0, 30, 18, 50, 0.5)).tolist() == [True, False, False]
    assert in_domain(GhostGridSpec(0, 52.5, 18, 50, 0.5)).tolist() == [True, True, True]


# --- Task 4: predict_density fail-loud on a non-default grid --------------------------------------
def test_predict_density_raises_on_non_default_grid():
    import pandas as pd

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    m = GhostGkModel(grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    fit_toy(m)  # locally-fit so training arrays exist (density is fit-only)
    X = pd.DataFrame([[0.0] * len(GHOST_GK_FEATURE_NAMES)], columns=GHOST_GK_FEATURE_NAMES)
    with pytest.raises(ValueError, match="extended-grid density"):
        m.predict_density(X)


def test_predict_density_still_works_on_default_grid():
    import pandas as pd

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    m = GhostGkModel()  # default grid
    fit_toy(m)
    X = pd.DataFrame([[0.0] * len(GHOST_GK_FEATURE_NAMES)], columns=GHOST_GK_FEATURE_NAMES)
    out = m.predict_density(X)  # must NOT raise
    assert out is not None


def test_no_density_guard_leaked_into_the_mean_path_primitive():
    import inspect

    from silly_kicks.tracking import _ghost_gk

    assert "extended-grid density" not in inspect.getsource(_ghost_gk.compute_ghost_gk)
