"""TF-60 Task 11: the Layer-3 arm-values corpus driver (mirrors test_build_gkdv_arm_values / _sb360)."""

from __future__ import annotations

import ast
import inspect

import scripts.build_tf60_layer3_arm_values as mod  # bare import: tests/scripts/ has NO __init__.py
from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture
from tests.tracking.test_ghost_gk import _fitted_model
from tests.tracking.test_ghost_outfield_model import _fit_toy


def _toy_models():
    return _fit_toy()[0], _fitted_model()[0]


def test_measure_match_emits_the_declared_shard_schema():
    actions, frames = make_rest_defense_fixture()
    outfield_model, gk_model = _toy_models()
    shard = mod.measure_match(
        "1",
        actions,
        frames,
        home_team_id=1,
        xt=make_fitted_xt(),
        ghost_outfield_model=outfield_model,
        ghost_gk_model=gk_model,
    )
    # The emitted keys are EXACTLY the declared schema (compare the keys the rows carry, never
    # pd.DataFrame(columns=...), which would select-and-hide a drift).
    assert tuple(shard.columns) == mod._EMITTED_SHARD_COLUMNS
    assert len(shard) >= 1  # the fixture has >=1 scored in-possession sample
    # both arms + the two Layer-1 anchor columns are carried
    for col in (
        "rd_outfield_deter_threat",
        "rd_outfield_deter_space",
        "rd_gk_deter_threat",
        "rd_gk_deter_space",
        "rd_num_superiority",
        "rd_compactness_x",
        "keeper_key",
    ):
        assert col in shard.columns


def test_empty_measure_still_carries_the_declared_columns():
    # "ran, produced nothing" must stay distinct from "not yet run" (ADR-052): a barren frame set
    # returns the declared columns, never a schema-less empty frame.
    actions, frames = make_rest_defense_fixture()
    # Put each frame's ball near the in-possession team's OWN goal -> not committed-forward -> zero
    # scored samples. Frame 102 is away (team 2, defends x=105); the rest are home (team 1, defends x=0).
    frames = frames.copy()
    ball = frames["is_ball"].astype(bool)
    frames.loc[ball & (frames["frame_id"] == 102), "x"] = 104.0
    frames.loc[ball & (frames["frame_id"] != 102), "x"] = 1.0
    outfield_model, gk_model = _toy_models()
    shard = mod.measure_match(
        "1",
        actions,
        frames,
        home_team_id=1,
        xt=make_fitted_xt(),
        ghost_outfield_model=outfield_model,
        ghost_gk_model=gk_model,
    )
    assert tuple(shard.columns) == mod._EMITTED_SHARD_COLUMNS
    assert len(shard) == 0


# --- 4.77.1 shard-schema / generation-token pin ---------------------------------------------------

_PINNED_SCHEMA = (
    "tf60-layer3-arms-1",
    (
        "game_id",
        "period_id",
        "team_id",
        "action_id",
        "keeper_key",
        "rd_num_superiority",
        "rd_compactness_x",
        "rd_outfield_deter_threat",
        "rd_outfield_deter_space",
        "rd_gk_deter_threat",
        "rd_gk_deter_space",
        "rd_outfield_source",
        "rd_gk_source",
    ),
)


def test_shard_schema_and_generation_token_move_together():
    token, columns = _PINNED_SCHEMA
    assert mod._EMITTED_SHARD_COLUMNS == columns, (
        "measure_match's emitted columns changed. Bump _SHARD_SCHEMA_VERSION AND update _PINNED_SCHEMA."
    )
    assert mod._SHARD_SCHEMA_VERSION == token


def test_the_token_is_what_actually_reaches_for_each():
    """Pinning a constant proves nothing if token_inputs hard-codes a different literal."""
    tree = ast.parse(inspect.getsource(mod.main).lstrip())
    schema_values = [
        v
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for k, v in zip(node.keys, node.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == "schema"
    ]
    assert schema_values, "no `schema` key in main()'s token_inputs -- the generation is unpinned"
    for value in schema_values:
        assert isinstance(value, ast.Name) and value.id == "_SHARD_SCHEMA_VERSION", (
            f"token_inputs['schema'] must reference _SHARD_SCHEMA_VERSION, not a literal (got {ast.dump(value)})"
        )


# --- provenance / ASCII wiring (mirrors build_gkdv / build_sb360 gates) ---------------------------


def test_driver_offers_allow_dirty_and_calls_require_clean_tree_from_main():
    tree = ast.parse(inspect.getsource(mod))
    main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "require_clean_tree" in called
    assert any("--allow-dirty" in s for s in _string_consts(main_fn))


def test_driver_never_shells_out_to_rev_parse():
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert "rev-parse" not in ast.dump(node)


def test_driver_source_is_ascii():
    src = inspect.getsource(mod)
    non_ascii = {c for c in src if ord(c) > 127}
    assert not non_ascii, f"non-ASCII in driver source (breaks --help on Windows): {non_ascii}"


def test_driver_fits_xt_on_the_corpus_like_the_established_convention():
    # The established convention for a reported-not-gated corpus measurement driver that needs xt:
    # fit ExpectedThreat on the loaded corpus (measure_cover_shadow_argmax_agreement.py). Assert the
    # fit is wired (not a refusal like the gkdv DAS-scoped physics driver).
    src = inspect.getsource(mod.main)
    assert "ExpectedThreat(" in src
    assert ".fit(" in src


def test_driver_uses_no_pickle():
    """The repo is pickle-free (ADR-011). The parallel --xt-out/--xt-in surface serializes via npz, so the
    driver must import neither pickle nor call pickle.* (AST, not substring: 'pickle-free' in a comment is
    fine, `pickle.loads(` is not)."""
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "pickle" for a in node.names), "no `import pickle` (ADR-011)"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "pickle", "no `from pickle import ...` (ADR-011)"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "pickle", "no `pickle.*` usage -- the repo is pickle-free (ADR-011)"


def test_xt_npz_roundtrip_is_exact():
    """The parallel --xt-out/--xt-in decoupling rests on this: a fitted xT written to npz and reloaded
    scores IDENTICALLY (singh_counts is deterministic, so its grids ARE the model). Fit a REAL model (the
    driver always fits, so heatmaps/matrices are populated) -> round-trip -> exact rate() parity + exact
    fitted-state equality. corpus_ids are sorted on write (workers must agree on the token order)."""
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    import silly_kicks.spadl.config as cfg
    from silly_kicks.xthreat import ExpectedThreat

    # `list.index` returns a plain int (pyright-clean, unlike int(itertuples().field) which is Scalar).
    ty = cfg.actiontypes.index
    re = cfg.results.index
    rng = np.random.default_rng(0)
    n = 600
    actions = pd.DataFrame(
        {
            "type_id": rng.choice([ty("pass"), ty("shot"), ty("dribble")], n),
            "result_id": rng.choice([re("success"), re("fail")], n),
            "start_x": rng.uniform(0, 105, n),
            "start_y": rng.uniform(0, 68, n),
            "end_x": rng.uniform(0, 105, n),
            "end_y": rng.uniform(0, 68, n),
        }
    )
    xt = ExpectedThreat()
    xt.fit(actions)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "xt.npz"
        mod._dump_xt_npz(xt, ["11", "10"], path)  # unsorted in -> sorted out
        xt2, ids = mod._load_xt_npz(path)
        assert ids == ["10", "11"]

        np.testing.assert_array_equal(np.asarray(xt.rate(actions)), np.asarray(xt2.rate(actions)))
        for k, v in vars(xt).items():
            v2 = getattr(xt2, k)
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, v2, equal_nan=True), f"array {k} drifted across the round-trip"
            elif isinstance(v, list):
                assert len(v) == len(v2) and all(
                    np.array_equal(a, b, equal_nan=True) for a, b in zip(v, v2, strict=True)
                ), f"list {k} drifted across the round-trip"
            else:
                assert repr(v) == repr(v2), f"scalar {k} drifted across the round-trip"


def test_xt_dump_refuses_a_non_default_surface():
    """_dump_xt_npz rebuilds params via the default ctor, so a non-default (e.g. kde_smoothed) surface would
    silently lose its params -> it must fail closed, not serialize a lossy surface."""
    import pytest

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(method="kde_smoothed")
    with pytest.raises(RuntimeError, match="singh_counts"):
        mod._dump_xt_npz(xt, ["1"], "unused.npz")


def _string_consts(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
