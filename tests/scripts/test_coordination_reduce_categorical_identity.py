"""D-5 (D3 leg): the D3 reduce is BYTE-IDENTICAL on an object-dtype match_tables and the same table with
its string columns cast to a sorted ``CategoricalDtype`` -- the exactness gate for the generalized
categorical combine (reduce-memory architecture, rev 3). Categorical group keys flip ``groupby``'s
``observed=`` default (pd2 False -> empty groups / pd3 True) and change ``sort_values``/``merge``/``set_index``
semantics; this gate fails until every reduce op is categorical-safe (``observed=True``, sorted categories).

Calibrate (D2/ruthless) is intentionally NOT imported, so this runs on the local pd2 env; the D2
``reliability_over_folds`` categorical leg is asserted on the DGX pd3 + ruthless-0.7 env.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _fake_corpus import make_loaded  # noqa: E402

import scripts.validate_team_coordination as d3  # noqa: E402  (NO calibrate import -> local pd2 ok)
from scripts._coordination_corpus import match_tables  # noqa: E402
from silly_kicks.coordination import CoordinationParams  # noqa: E402
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
]

_SECONDS = 240.0
_SHOTS = (36.0, 84.0, 132.0, 180.0)
_TACKLES = (60.0, 108.0, 156.0, 204.0)
_PROV = {"commit": "test", "dirty": False, "tree_state": "clean", "tree_hash": "test"}

#: CI-speed cap on the reliability + occlusion bootstraps -- ~90% of this file's runtime (614s + 143s).
#: The gate compares the object-dtype leg against the sorted-categorical leg at the SAME seed+draws, so
#: the draw count CANCELS and byte-identity is invariant to it (verified: identical pass at 25 vs the
#: prod 400). Every categorical-sensitive path -- pd.unique / rng.choice / group relabel / icc1 /
#: weighted-median groupby -- still fires each draw, so dtype-safety coverage is unchanged. Prod stays
#: 400: RELIABILITY_BOOTSTRAP_DRAWS is a DEF-TIME default (reliability_cell binds it at import, so patch
#: the FUNCTION, not the constant); _OCCLUSION_CI_DRAWS is read at CALL time inside the resample loop (so
#: setattr takes). Blast radius checked: nothing outside these two bootstraps reads either constant.
_CI_BOOTSTRAP_DRAWS = 25


@pytest.fixture(autouse=True)
def _cap_bootstrap_draws(monkeypatch):
    """Cap both bootstraps to _CI_BOOTSTRAP_DRAWS for CI cost; leg-vs-leg byte-identity is unaffected."""
    import scripts._coordination_reliability as _rel
    import scripts.derive_coordination_params as _dcp

    _orig_ci = _rel._bootstrap_ci
    monkeypatch.setattr(
        _rel,
        "_bootstrap_ci",
        lambda samples, groups, kind, seed, draws: _orig_ci(
            samples, groups, kind, seed, min(draws, _CI_BOOTSTRAP_DRAWS)
        ),
    )
    monkeypatch.setattr(_dcp, "_OCCLUSION_CI_DRAWS", min(_dcp._OCCLUSION_CI_DRAWS, _CI_BOOTSTRAP_DRAWS))


def _match(game: int, team_ids: tuple[int, int], seed: int):
    frames = make_coordination_match(
        seconds=_SECONDS, provider="sportec", periods=2, team_ids=team_ids, seed=seed, dead_intervals=[(100.0, 130.0)]
    )
    frames["game_id"] = game
    parts = []
    for period, fp in frames.groupby("period_id", sort=True):
        acts = make_coordination_actions(
            fp, restarts=[(t, "tackle") for t in _TACKLES], goals=_SHOTS, possession_every_s=12.0
        )
        acts["game_id"] = game
        acts["period_id"] = period
        parts.append(acts)
    actions = pd.concat(parts, ignore_index=True)
    actions["action_id"] = np.arange(len(actions))
    return make_loaded("sportec", str(game), frames=frames, actions=actions)


@pytest.fixture(scope="module")
def metrics_frame() -> pd.DataFrame:
    params = CoordinationParams.for_provider("sportec")
    matches = [_match(1, (1, 2), 11), _match(2, (1, 2), 12), _match(3, (1, 3), 13)]
    return pd.concat(
        [match_tables(m, params, n_surrogates=0, include_switch_events=True) for m in matches], ignore_index=True
    )


def _to_sorted_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every string/object column to a sorted ``CategoricalDtype`` -- exactly what the generalized
    categorical combine produces (shared sorted categories, NOT append-order ``union_categoricals``)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or isinstance(out[col].dtype, pd.StringDtype):
            cats = sorted(x for x in out[col].dropna().unique())
            out[col] = out[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))
    return out


def _report(metrics: pd.DataFrame, occ: pd.DataFrame | None = None, stoppage: pd.DataFrame | None = None) -> str:
    empty = metrics.iloc[:0]
    rep = d3.build_report(metrics, empty if stoppage is None else stoppage, empty if occ is None else occ, _PROV)
    rep.pop("stage_timing", None)
    return json.dumps(rep, indent=2, default=str, sort_keys=True)


@pytest.fixture(scope="module")
def occ_frame() -> pd.DataFrame:
    # RM-IMPL-01: the headline OOM is the occlusion share (occ_err). Build the REAL occ_df (occlusion_metrics
    # output) so the D1 occlusion reduce + D3 occlusion_curves are byte-identity-tested under categorical -- not
    # short-circuited on an empty frame. TWO short matches (so the match-grain per-bin CI bootstrap, ADR-112
    # follow-up, has >= 2 clusters and is genuinely exercised, not a vacuous < 2-match NaN) + a NARROW FOV (15 m):
    # obs_frac then varies across windows -> multiple obs_bins. occ_err stays small (full 465M-row scale = DGX run).
    from scripts.derive_coordination_params import occlusion_metrics

    def _occ(game: int, seed: int) -> pd.DataFrame:
        f = make_coordination_match(seconds=90.0, hz=10.0, provider="sportec", periods=1, seed=seed)
        f["game_id"] = game
        acts = make_coordination_actions(f, restarts=[], goals=(), possession_every_s=12.0)
        acts["game_id"] = game
        acts["period_id"] = 1
        acts["action_id"] = np.arange(len(acts))
        return occlusion_metrics(make_loaded("sportec", str(game), frames=f, actions=acts), 15.0)

    return pd.concat([_occ(1, 11), _occ(2, 12)], ignore_index=True)


def test_fixture_occ_is_discriminating(occ_frame):
    # the occlusion reduce's categorical-sensitive path (weighted-median per obs_bin + bootstrap, groupby
    # family/construct/obs_bin) is genuinely exercised: multiple bins + constructs, not a degenerate frame.
    assert {"occ_err", "occ_full", "bridge_rmse", "noise_rms", "gk_rate"} <= set(occ_frame["quantity"])
    err = occ_frame[occ_frame["quantity"] == "occ_err"]
    assert len(err) and err["obs_bin"].nunique() >= 2 and err["construct"].nunique() >= 2


def test_d1_occlusion_reduce_byte_identical_under_categorical(occ_frame):
    # D-5 (D1 occlusion leg): the occlusion reduce kernels build_derivation calls -- occlusion_per_construct_detail
    # (weighted-median curve + seeded bootstrap CI), _reduce_min_observed (family-MAX), _reduce_max_gap -- plus D3's
    # occlusion_curves, are byte-identical on an object occ_df vs the sorted-categorical one the combine produces.
    from scripts.derive_coordination_params import (
        _reduce_max_gap,
        _reduce_min_observed,
        occlusion_per_construct_detail,
    )

    cat = _to_sorted_categorical(occ_frame)

    def _dump(df: pd.DataFrame) -> str:
        return json.dumps(
            {
                "detail": occlusion_per_construct_detail(df, provider_neutral=True),
                "min_observed": _reduce_min_observed(df, provider_neutral=True, why={}),
                "max_gap": _reduce_max_gap(df, provider_neutral=True, why={}),
                "curves": d3.occlusion_curves(df),
            },
            sort_keys=True,
            default=str,
        )

    assert _dump(occ_frame) == _dump(cat)


def test_fixture_is_discriminating(metrics_frame):
    # ADR-032: the union is mixed + the sparse (level, window_kind) combos that trip the observed= empty-group
    # trap are present, else the identity below passes vacuously.
    assert {"pair", "pair_phase", "cluster_team", "spectral", "rsi"} <= set(metrics_frame["table"])
    assert {"period", "possession"} <= set(metrics_frame["window_kind"].dropna())
    assert {"team_team", "dyad"} <= set(metrics_frame["level"].dropna())
    assert metrics_frame.select_dtypes(include=["object"]).shape[1] >= 5
    # D-5 coverage (Task 5 / RM-SPEC-11): the reduce's order-sensitive categorical ops -- the combine's canonical
    # sort on `_ORDER_COLUMNS` (:464), `_window_join` merge, `_paired_axis` set_index, `_team_key` astype(str) --
    # are keyed on provider/match_id/table/level/window_kind. They must be CATEGORICAL in this harness or D-5 would
    # prove categorical-safety vacuously (structural invariant + load-bearing plant: test_reduce_categorical_safety).
    cat = _to_sorted_categorical(metrics_frame)
    assert {"provider", "match_id"} <= set(cat.columns), "combine sort keys absent from the metrics melt"
    for key in ("provider", "match_id", "table", "level", "window_kind"):
        assert isinstance(cat[key].dtype, pd.CategoricalDtype), f"{key!r} not categorical -> D-5 vacuous for it"


def test_d3_reduce_byte_identical_object_vs_sorted_categorical(metrics_frame, occ_frame):
    # feed a NON-empty occ_df so build_report's occlusion_curves runs under categorical too (not short-circuited).
    obj = _report(metrics_frame, occ=occ_frame)
    cat = _report(_to_sorted_categorical(metrics_frame), occ=_to_sorted_categorical(occ_frame))
    assert cat == obj  # full build_report (hypotheses + reliability + liveness + occlusion) unchanged under categorical
