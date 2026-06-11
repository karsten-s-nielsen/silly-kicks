"""Provider-aware GK-completion variant selection (D-S2, C4) + native-only training (F1/G1)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

_GOALKICK = spadlconfig.actiontype_id["goalkick"]


# --- Task 6: pure variant_key_for_provider (C4 — exhaustive, artifact-free) ---


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("skillcorner", "skillcorner"),
        ("gradientsports", "gs"),
        ("sportec", "gs"),
        ("snapshot", "gs"),
        ("metrica", "gs"),
        (None, "gs"),
        ("unknown_x", "gs"),
        ("SkillCorner", "skillcorner"),  # case-insensitive
    ],
)
def test_variant_key_for_provider(provider, expected):
    from silly_kicks.tracking._gk_completion import variant_key_for_provider

    assert variant_key_for_provider(provider) == expected


# --- Task 9: native-only training filter (F1 + G1) ---


def _make_goalkick_fixture(result_id, result_source):
    """A minimal N-row goalkick action frame, all geometry/id-scoreable."""
    n = len(result_id)
    return pd.DataFrame(
        {
            "game_id": "g1",
            "action_id": np.arange(n, dtype="int64"),
            "period_id": 1,
            "time_seconds": np.arange(n, dtype="float64"),
            "team_id": "366",
            "player_id": "1",
            "start_x": 5.0,
            "start_y": 34.0,
            "end_x": 40.0 + np.arange(n, dtype="float64"),  # finite length, distinct
            "end_y": 34.0,
            "type_id": _GOALKICK,
            "result_id": np.asarray(result_id, dtype="int64"),
            "result_source": np.asarray(result_source, dtype=object),
        }
    )


def test_training_uses_native_label_only():
    from silly_kicks.tracking._gk_completion import prepare_gk_completion_training_data

    S, F = spadlconfig.result_id["success"], spadlconfig.result_id["fail"]
    # 4 goalkicks: 2 native (1 success, 1 fail) + 1 inferred (positive-only) + 1 stopgap.
    # Only the 2 native rows may train (G1: inferred is positive-only -> calibration bias).
    actions = _make_goalkick_fixture(
        result_id=[S, F, S, S],
        result_source=["native", "native", "inferred", "stopgap"],
    )
    X, y, _groups = prepare_gk_completion_training_data(actions, frames=None)
    assert len(y) == 2  # inferred + stopgap dropped; only native trains
    assert set(y.tolist()) == {0, 1}  # native supplies BOTH classes
    assert "result_source" not in X.columns  # not a feature


def test_training_filter_is_noop_without_result_source():
    # GS-shaped actions (no result_source column) -> all scoreable native rows kept (provider-agnostic).
    from silly_kicks.tracking._gk_completion import prepare_gk_completion_training_data

    S, F = spadlconfig.result_id["success"], spadlconfig.result_id["fail"]
    actions = _make_goalkick_fixture(result_id=[S, F, S], result_source=["native", "native", "native"])
    actions = actions.drop(columns=["result_source"])  # GS has no such column
    _X, y, _groups = prepare_gk_completion_training_data(actions, frames=None)
    assert len(y) == 3  # no filtering applied


# --- Task 7: provider-aware resolution in compute_xt_gk (the _resolve_completion_for_frames seam) ---


def _frames(providers):
    return pd.DataFrame({"source_provider": providers, "frame_id": np.arange(len(providers))})


def test_resolve_override_wins():
    from silly_kicks.tracking._gk_completion import GkCompletionModel
    from silly_kicks.tracking._xt_gk import _resolve_completion_for_frames

    m = GkCompletionModel.from_variant("default")  # bundled GS model
    model, _key = _resolve_completion_for_frames(_frames(["skillcorner"]), m)
    assert model is m  # override beats auto-select


def test_resolve_auto_selects_by_provider(monkeypatch):
    import silly_kicks.tracking._gk_completion as gc
    from silly_kicks.tracking._xt_gk import _resolve_completion_for_frames

    calls, sentinel = [], object()
    monkeypatch.setattr(
        gc.GkCompletionModel,
        "from_variant",
        classmethod(lambda cls, variant="default": (calls.append(variant), sentinel)[1]),
    )
    model, key = _resolve_completion_for_frames(_frames(["skillcorner", "skillcorner"]), None)
    assert model is sentinel and key == "skillcorner"
    assert calls == ["skillcorner"]
    calls.clear()
    _model, key = _resolve_completion_for_frames(_frames(["gradientsports"]), None)
    assert key == "gs"
    assert calls == ["gs"]


def test_resolve_multi_provider_raises():
    from silly_kicks.tracking._xt_gk import _resolve_completion_for_frames

    with pytest.raises(ValueError, match="multiple real providers"):
        _resolve_completion_for_frames(_frames(["skillcorner", "gradientsports"]), None)


def test_resolve_snapshot_excluded_from_uniqueness(monkeypatch):
    import silly_kicks.tracking._gk_completion as gc
    from silly_kicks.tracking._xt_gk import _resolve_completion_for_frames

    sentinel = object()
    monkeypatch.setattr(gc.GkCompletionModel, "from_variant", classmethod(lambda cls, variant="default": sentinel))
    # snapshot + one real provider -> no raise (snapshot is a synthetic frames-only tag, C3)
    model, key = _resolve_completion_for_frames(_frames(["snapshot", "skillcorner"]), None)
    assert model is sentinel and key == "skillcorner"


def test_resolve_missing_variant_falls_back_to_default_with_warning(monkeypatch):
    import silly_kicks.tracking._gk_completion as gc
    from silly_kicks.tracking._xt_gk import _resolve_completion_for_frames

    calls, sentinel = [], object()

    def fake(cls, variant="default"):
        calls.append(variant)
        if variant == "skillcorner":
            raise FileNotFoundError("no skillcorner weights bundled")
        return sentinel

    monkeypatch.setattr(gc.GkCompletionModel, "from_variant", classmethod(fake))
    with pytest.warns(UserWarning, match="no bundled GK-completion weights"):
        model, key = _resolve_completion_for_frames(_frames(["skillcorner"]), None)
    # fell back to the default (gs) model that ACTUALLY scored -> key reports "gs"
    assert model is sentinel and calls == ["skillcorner", "default"] and key == "gs"


# --- Task 11 (#3): pure comparability verdict logic (scripts/_xtgk_comparability.py) ---


def _import_comparability():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from _xtgk_comparability import compare_xtgk_distributions

    return compare_xtgk_distributions


def _band_df(per_band):
    """per_band: list of (dist_center, value, n) -> a long df with dist/xt_gk."""
    dist, xt = [], []
    for c, v, n in per_band:
        dist += [c] * n
        xt += [v] * n
    return pd.DataFrame({"dist": dist, "xt_gk": xt})


def test_comparability_within_tolerance():
    compare = _import_comparability()
    sc = _band_df([(5.0, 0.010, 40), (20.0, 0.020, 40)])
    gs = _band_df([(5.0, 0.011, 40), (20.0, 0.021, 40)])  # offsets -0.001 each
    _bands, verdict = compare(sc, gs, offset_tol=0.01, min_n=30)
    assert verdict == "within_tolerance"


def test_comparability_escalate_nonuniform():
    compare = _import_comparability()
    sc = _band_df([(5.0, 0.060, 40), (20.0, 0.020, 40)])  # band1 offset +0.05, band2 ~0
    gs = _band_df([(5.0, 0.010, 40), (20.0, 0.020, 40)])
    _bands, verdict = compare(sc, gs, offset_tol=0.01, min_n=30)
    assert verdict == "escalate"  # non-uniform offset -> genuine difference, do not auto-conform


def test_comparability_uniform_offset_flags_artifact():
    compare = _import_comparability()
    sc = _band_df([(5.0, 0.030, 40), (20.0, 0.040, 40)])  # uniform +0.02 each band
    gs = _band_df([(5.0, 0.010, 40), (20.0, 0.020, 40)])
    _bands, verdict = compare(sc, gs, offset_tol=0.01, min_n=30)
    assert verdict == "escalate_or_correctable_artifact"


def test_comparability_underpowered_is_insufficient():
    compare = _import_comparability()
    sc = _band_df([(5.0, 0.01, 10)])  # n < min_n
    gs = _band_df([(5.0, 0.01, 10)])
    _bands, verdict = compare(sc, gs, offset_tol=0.01, min_n=30)
    assert verdict == "insufficient_overlap"


# --- #2: calibration helpers (ECE + reliability-slope) ---


def _import_train():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_gk_completion

    return train_gk_completion


def test_reliability_slope_and_ece_on_calibrated_predictions():
    t = _import_train()
    rng = np.random.RandomState(0)
    n = 5000
    p = rng.rand(n)
    y = (rng.rand(n) < p).astype(int)  # perfectly calibrated: P(y=1)=p
    assert 0.85 <= t._reliability_slope(y, p) <= 1.15  # ~1 (diagonal)
    assert t._ece(y, p) < 0.05  # small
    # degenerate (single occupied bin) -> slope undefined (NaN), not a crash
    assert np.isnan(t._reliability_slope(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])))


def test_reliability_slope_flags_overconfidence():
    t = _import_train()
    rng = np.random.RandomState(1)
    n = 8000
    p_true = rng.rand(n)
    y = (rng.rand(n) < p_true).astype(int)
    p_over = np.clip((p_true - 0.5) * 2.5 + 0.5, 0.0, 1.0)  # predictions pushed to the extremes
    assert t._reliability_slope(y, p_over) < 0.9  # observed rises slower than predicted -> slope < 1


# --- Task 10 Step 4 (#1): does-it-run smoke of the --variant skillcorner train path (no network) ---


@pytest.mark.slow
def test_train_skillcorner_smoke(tmp_path, monkeypatch):
    """Exercises _train_skillcorner / _ece / _reliability_slope / the decision logic on a synthetic
    feature cache (NO network), with the weights dirs redirected to tmp so the real bundled artifact
    is never clobbered."""
    import importlib
    import sys
    from argparse import Namespace
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    train = importlib.import_module("train_gk_completion")
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES as FEATS

    rng = np.random.RandomState(0)
    n = 160
    data = {f: rng.randn(n) for f in FEATS}
    data["is_goalkick"] = (np.arange(n) % 4 == 0).astype(float)  # ~25% goalkicks, rest GK-pass
    data["is_throw_in"] = 0.0  # type: ignore[assignment]
    df = pd.DataFrame(data)
    df["_y"] = (rng.rand(n) < 0.6).astype(int)  # both classes
    df["_group"] = np.arange(n) % 5  # 5 groups for GroupKFold
    cache = tmp_path / "feat.parquet"
    df.to_parquet(cache)

    monkeypatch.setattr(train, "_SKILLCORNER_WEIGHTS_DIR", tmp_path / "skillcorner")
    monkeypatch.setattr(train, "_WEIGHTS_ROOT", tmp_path)
    rc = train._train_skillcorner(Namespace(max_per_provider=2, tracking_limit=10, cache_features=str(cache)))
    assert rc == 0
    # a decision artifact was written under tmp (bundle -> skillcorner/metrics.json; else root report);
    # the real bundled weights are untouched (nothing written outside tmp_path).
    assert (tmp_path / "skillcorner" / "metrics.json").exists() or (
        tmp_path / "skillcorner_remeasurement.json"
    ).exists()


# --- 4.22.1: variant-key -> bundled-weights alias (lakehouse report 2026-06-11 item 6) ---


def test_from_variant_gs_aliases_to_default():
    """ "gs" (the variant_key_for_provider key for the GS-construct providers) must load
    the bundled "default" weights -- previously FileNotFoundError, so the two public
    APIs did not compose."""
    from silly_kicks.tracking._gk_completion import GkCompletionModel

    m_gs = GkCompletionModel.from_variant("gs")
    assert m_gs is GkCompletionModel.from_variant("default")  # shared cached instance


def test_public_api_composition_never_raises():
    """from_variant(variant_key_for_provider(p)) must work for every mapped provider."""
    from silly_kicks.tracking._gk_completion import GkCompletionModel, variant_key_for_provider

    for provider in ("gradientsports", "sportec", "metrica", "skillcorner", None, "unknown_x"):
        GkCompletionModel.from_variant(variant_key_for_provider(provider))  # must not raise
