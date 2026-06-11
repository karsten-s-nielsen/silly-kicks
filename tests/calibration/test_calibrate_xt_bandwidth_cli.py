"""SK-xT-3 Task 5: CLI seam smoke + manifest shape + reported cross-check + P1/N8 guards."""

import math

import silly_kicks
from scripts.calibrate_xt_bandwidth import build_manifest, run_xt_bandwidth, xt_quality_cross_check
from tests._xthreat_helpers import _corpus_with_shots, _sparse_overfit_corpus


def _multi_game_shots(n_games=6):
    import pandas as pd

    parts = []
    for g in range(n_games):
        d = _corpus_with_shots(n_per_zone=20, seed=g)
        d["game_id"] = g
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_run_xt_bandwidth_smoke_returns_finite_best(tmp_path):
    result, _objective = run_xt_bandwidth(
        actions=_sparse_overfit_corpus(seed=7, n_games=20),
        n_trials=3,
        seed=42,
        store_path=str(tmp_path / "xt.db"),
    )
    assert result.best is not None
    assert math.isfinite(result.best.metrics["xt_holdout_nll"])


def test_build_manifest_scopes_recommendation_and_versions(tmp_path):
    result, _obj = run_xt_bandwidth(
        actions=_sparse_overfit_corpus(seed=8, n_games=20),
        n_trials=3,
        seed=42,
        store_path=str(tmp_path / "xt.db"),
    )
    manifest = build_manifest(
        source="pining",
        seed=42,
        n_trials=3,
        max_points_per_zone=None,
        match_ids={"pining": ["m1"]},
        result=result,
        cross_check=None,
    )
    assert manifest["stage"] == "xt_bandwidth"
    assert manifest["applies_to_library_default"] is False
    assert "unverified" in manifest["recommendation_scope"]
    assert manifest["silly_kicks_version"] == silly_kicks.__version__
    assert manifest["recommendation"]["method"] == "kde_smoothed"
    assert set(manifest["recommendation"]["grid"]) == {"n_zones_x", "n_zones_y"}


def test_cross_check_returns_finite_rho_for_both_grids():
    cc = xt_quality_cross_check(
        _multi_game_shots(),
        recommendation={"bandwidth": 1.5, "adaptive": True, "grid": {"n_zones_x": 16, "n_zones_y": 12}},
        k=10,
        seed=42,
    )
    assert math.isfinite(cc["rho_recommended"])
    assert math.isfinite(cc["rho_singh"])


def test_scores_per_game_does_not_leak_goal_across_game_boundary():
    # P1 regression: the LAST action of game A must NOT be labelled "scored" by game B's early goal.
    import pandas as pd

    import silly_kicks.spadl.config as cfg
    from scripts.calibrate_xt_bandwidth import _scores_per_game

    cols = [
        "game_id",
        "action_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "bodypart_id",
        "type_id",
        "result_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]
    pas, shot = cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]
    succ = cfg.result_id["success"]
    rows = [
        # game A: a single trailing pass (no goal in A) -> label MUST be 0
        [1, 0, 1, 0.0, 7, 7, 0, pas, succ, 50.0, 34.0, 60.0, 34.0],
        # game B: an immediate goal by the same team within k actions of A's last row
        [2, 1, 1, 0.0, 7, 7, 0, shot, succ, 100.0, 34.0, 105.0, 34.0],
    ]
    df = pd.DataFrame(rows, columns=cols)
    y = _scores_per_game(df, k=10)
    assert y[0] == 0  # game A's pass is NOT credited with game B's goal (would be 1 if leaked)


def test_load_corpus_pining_requests_minimal_tracking(monkeypatch):
    # N8: the pining corpus load must pass tracking_limit=1 (NOT 0 — 0 is falsy and loads all frames).
    import pandas as pd

    import scripts._loader_pining as loader
    from scripts.calibrate_xt_bandwidth import _load_corpus

    captured = {}
    cols = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]

    def _fake_load_matches(*, providers, tracking_limit=None, max_per_provider=None, cache_dir=None):
        captured["tracking_limit"] = tracking_limit
        captured["cache_dir"] = cache_dir
        yield "skillcorner", "m1", pd.DataFrame([[1, 50.0, 34.0, 60.0, 34.0, 0, 1]], columns=cols), None, None

    monkeypatch.setattr(loader, "load_matches", _fake_load_matches)
    args = type(
        "A",
        (),
        {
            "source": "pining",
            "providers": ["skillcorner"],
            "max_matches_per_provider": None,
            "cache_dir": None,
            "corpus_cache": None,
            "subsample_games": None,
            "seed": 42,
        },
    )()
    actions, ids = _load_corpus(args)
    assert captured["tracking_limit"] == 1
    assert "skillcorner" in ids
    # provider-qualified unique string game_id (guards mixed-dtype + cross-provider id collision)
    assert list(actions["game_id"].unique()) == ["skillcorner:m1"]


def _corpus_args(**over):
    base = {"source": "pining", "corpus_cache": None, "subsample_games": None, "seed": 42}
    base.update(over)
    return type("A", (), base)()


def test_corpus_cache_roundtrip_skips_reassembly(tmp_path, monkeypatch):
    # The corpus parquet cache means a second load does NOT re-assemble (no re-download/parse) — the
    # basis of the corpus-size contrast (build full once, subsample cheaply).
    import pandas as pd

    import scripts.calibrate_xt_bandwidth as cli

    calls = {"n": 0}

    def _fake_assemble(args):
        calls["n"] += 1
        return pd.DataFrame(
            {
                "game_id": ["skillcorner:m1", "skillcorner:m1", "idsse:M2"],
                "start_x": [1.0, 2.0, 3.0],
                "type_id": [0, 0, 0],
            }
        )

    monkeypatch.setattr(cli, "_assemble_corpus", _fake_assemble)
    cache = tmp_path / "corpus.parquet"
    args = _corpus_args(corpus_cache=str(cache))
    _df1, ids1 = cli._load_corpus(args)  # assembles + writes cache
    _df2, ids2 = cli._load_corpus(args)  # reads cache — no reassembly
    assert calls["n"] == 1
    assert cache.exists()
    assert set(ids1) == set(ids2) == {"skillcorner", "idsse"}


def test_assemble_corpus_canonicalizes_for_parquet(tmp_path, monkeypatch):
    # DGX full-run regression: the full SPADL actions carry provider-specific/heterogeneous columns
    # (original_event_id mixed int/str; mixed-dtype team_id/player_id across providers) that break
    # pyarrow's to_parquet. _assemble_corpus must project to the canonical SPADL columns + str-cast
    # the id columns so the multi-provider corpus serializes.
    import pandas as pd

    import scripts._loader_pining as loader
    import scripts.calibrate_xt_bandwidth as cli

    cols = [
        "game_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "bodypart_id",
        "type_id",
        "result_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "original_event_id",
    ]

    def _fake_load(*, providers, tracking_limit=None, max_per_provider=None, cache_dir=None):
        # provider A: int team_id + int original_event_id
        yield (
            "skillcorner",
            "m1",
            pd.DataFrame([[1, 1, 0.0, 10, 100, 0, 0, 1, 1.0, 1.0, 2.0, 2.0, 5]], columns=cols),
            None,
            None,
        )
        # provider B: str team_id + str original_event_id -> heterogeneous object columns on concat
        yield (
            "gradientsports",
            "g1",
            pd.DataFrame([["x", 1, 0.0, "TB", "P9", 0, 0, 1, 3.0, 3.0, 4.0, 4.0, "e9"]], columns=cols),
            None,
            None,
        )

    monkeypatch.setattr(loader, "load_matches", _fake_load)
    args = type(
        "A",
        (),
        {
            "source": "pining",
            "providers": ["skillcorner", "gradientsports"],
            "max_matches_per_provider": None,
            "cache_dir": None,
        },
    )()
    df = cli._assemble_corpus(args)
    assert "original_event_id" not in df.columns  # heterogeneous provider extra dropped
    assert df["team_id"].map(type).eq(str).all()  # asymmetric ids string-cast  # type: ignore[arg-type]
    df.to_parquet(tmp_path / "c.parquet")  # must NOT raise pyarrow ArrowTypeError


def test_subsample_games_reduces_corpus(monkeypatch):
    import pandas as pd

    import scripts.calibrate_xt_bandwidth as cli

    monkeypatch.setattr(
        cli,
        "_assemble_corpus",
        lambda args: pd.DataFrame(
            {"game_id": [f"p:{g}" for g in range(5) for _ in range(4)], "start_x": 1.0, "type_id": 0}
        ),
    )
    df, ids = cli._load_corpus(_corpus_args(subsample_games=2))
    assert df["game_id"].nunique() == 2
    assert sum(len(v) for v in ids.values()) == 2
