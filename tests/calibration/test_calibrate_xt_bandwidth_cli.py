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


def test_load_corpus_pining_requests_minimal_tracking(monkeypatch, tmp_path):
    # N8: the pining corpus load must pass tracking_limit=1 (NOT 0 - 0 is falsy and loads all frames).
    import pandas as pd

    import scripts._loader_pining as loader
    from scripts.calibrate_xt_bandwidth import _load_corpus

    captured = {}
    cols = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]

    def _fake_load_matches(*, providers, match_ids=None, tracking_limit=None, max_per_provider=None, cache_dir=None):
        captured["tracking_limit"] = tracking_limit
        captured["cache_dir"] = cache_dir
        captured["match_ids"] = match_ids
        yield "skillcorner", "m1", pd.DataFrame([[1, 50.0, 34.0, 60.0, 34.0, 0, 1]], columns=cols), None, None

    monkeypatch.setattr(loader, "load_matches", _fake_load_matches)
    monkeypatch.setattr(loader, "select_match_ids", lambda **kw: [("skillcorner", "m1")])
    args = _corpus_args(
        providers=["skillcorner"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )
    actions, ids = _load_corpus(args)
    assert captured["tracking_limit"] == 1
    # The walk is INVERTED onto select_match_ids: load_matches is asked for one named match, which
    # is what puts the download+parse behind the shard check rather than in front of it.
    assert captured["match_ids"] == {"skillcorner": ["m1"]}
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

    # One row per match, keyed the way the INVERTED walk asks for them: one named match per call.
    rows = {
        # provider A: int team_id + int original_event_id
        ("skillcorner", "m1"): pd.DataFrame([[1, 1, 0.0, 10, 100, 0, 0, 1, 1.0, 1.0, 2.0, 2.0, 5]], columns=cols),
        # provider B: str team_id + str original_event_id -> heterogeneous object columns on concat
        ("gradientsports", "g1"): pd.DataFrame(
            [["x", 1, 0.0, "TB", "P9", 0, 0, 1, 3.0, 3.0, 4.0, 4.0, "e9"]], columns=cols
        ),
    }

    def _fake_load(*, providers, match_ids=None, tracking_limit=None, max_per_provider=None, cache_dir=None):
        (provider,) = providers
        # `or {}` for the type checker, not for behaviour: the INVERTED walk always names its
        # match, so an absent slice must KeyError rather than quietly load the whole manifest.
        for mid in (match_ids or {})[provider]:
            yield provider, mid, rows[(provider, mid)], None, None

    monkeypatch.setattr(loader, "load_matches", _fake_load)
    monkeypatch.setattr(loader, "select_match_ids", lambda **kw: sorted(rows))
    args = _corpus_args(
        providers=["skillcorner", "gradientsports"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )
    df = cli._assemble_corpus(args)
    assert "original_event_id" not in df.columns  # heterogeneous provider extra dropped
    assert df["team_id"].map(type).eq(str).all()  # asymmetric ids string-cast  # type: ignore[arg-type]
    assert set(df["game_id"]) == {"skillcorner:m1", "gradientsports:g1"}
    df.to_parquet(tmp_path / "c.parquet")  # must NOT raise pyarrow ArrowTypeError


def _shardable_corpus(monkeypatch, *, loaded, yields=None, raises=()):
    """Patch the pining loader with a per-match fake that RECORDS every match it is asked to load.

    ``yields`` names the matches that produce a row; anything absent yields nothing, which is how
    `load_matches` reports an S1-geometry exclusion. ``raises`` names matches that blow up, which is
    how a transient fetch failure arrives after `_build_match_with_retry` has given up.
    """
    import pandas as pd

    import scripts._loader_pining as loader

    cols = ["game_id", "period_id", "time_seconds", "team_id", "player_id", "start_x", "type_id", "result_id"]
    pairs = [("skillcorner", "m1"), ("idsse", "M2")]
    yields = set(pairs) if yields is None else set(yields)
    raises = set(raises)

    def _fake_load(*, providers, match_ids=None, tracking_limit=None, max_per_provider=None, cache_dir=None):
        (provider,) = providers
        # `or {}` for the type checker, not for behaviour: the INVERTED walk always names its
        # match, so an absent slice must KeyError rather than quietly load the whole manifest.
        for mid in (match_ids or {})[provider]:
            loaded.append((provider, mid))
            if (provider, mid) in raises:
                raise OSError(f"transient fetch failure for {provider}/{mid}")
            if (provider, mid) in yields:
                yield provider, mid, pd.DataFrame([[1, 1, 0.0, 10, 100, 5.0, 0, 1]], columns=cols), None, None

    monkeypatch.setattr(loader, "load_matches", _fake_load)
    monkeypatch.setattr(loader, "select_match_ids", lambda *, providers, **kw: [p for p in pairs if p[0] in providers])
    return pairs


def test_a_RESUMED_corpus_pass_does_not_reload_an_already_sharded_match(monkeypatch, tmp_path):
    """The whole point of inverting the walk onto `select_match_ids`.

    `for_each` skips `work(item)`, never the production of `item` -- so had the driver kept
    streaming `load_matches`, a resumed run would re-download and re-parse every match in order to
    then skip a set of trivial writes. This asserts the loader is not entered at all on the second
    pass, and (non-vacuity) that the second pass nonetheless returns the same corpus rather than
    nothing: an implementation that simply produced an empty frame would satisfy the call count.
    """
    import scripts.calibrate_xt_bandwidth as cli

    loaded: list = []
    _shardable_corpus(monkeypatch, loaded=loaded)
    args = _corpus_args(
        providers=["skillcorner", "idsse"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )

    first = cli._assemble_corpus(args)
    assert sorted(loaded) == [("idsse", "M2"), ("skillcorner", "m1")]

    loaded.clear()
    second = cli._assemble_corpus(args)
    assert loaded == [], f"resume re-entered the loader for {loaded}"
    assert set(second["game_id"]) == set(first["game_id"]) == {"skillcorner:m1", "idsse:M2"}
    assert len(second) == len(first) == 2


def test_a_CHANGED_declared_input_starts_a_NEW_generation_and_reloads(monkeypatch, tmp_path):
    """The other side of the band: resume must NOT serve shards built under different inputs.

    `_CORPUS_COLS` is a declared `token_inputs` entry precisely because it determines shard content.
    Changing it must land the pass in a different generation directory, so every match is loaded
    again -- the inverse of the test above, and the reason a stale shard is unrepresentable here
    rather than merely guarded.
    """
    import scripts.calibrate_xt_bandwidth as cli

    loaded: list = []
    _shardable_corpus(monkeypatch, loaded=loaded)
    args = _corpus_args(
        providers=["skillcorner", "idsse"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )

    cli._assemble_corpus(args)
    loaded.clear()
    monkeypatch.setattr(cli, "_CORPUS_COLS", [c for c in cli._CORPUS_COLS if c != "player_id"])
    cli._assemble_corpus(args)
    assert sorted(loaded) == [("idsse", "M2"), ("skillcorner", "m1")], "a changed token reused stale shards"

    shard_root = tmp_path / "r_corpus" / "shards"
    generations = sorted(p.name for p in shard_root.iterdir() if p.is_dir())
    assert len(generations) == 2, f"expected two generation dirs side by side, found {generations}"


def test_an_EXCLUDED_match_writes_an_EMPTY_shard_and_is_not_retried(monkeypatch, tmp_path):
    """`load_matches` DROPS a geometrically-broken skillcorner match: it yields nothing for it.

    That decision is deterministic for a given artifact, so it is recorded as an empty shard --
    "ran, produced nothing" -- rather than left absent, which would make every resume pay the
    download and the parse again to reach the same verdict.
    """
    import scripts.calibrate_xt_bandwidth as cli

    loaded: list = []
    _shardable_corpus(monkeypatch, loaded=loaded, yields=[("idsse", "M2")])
    args = _corpus_args(
        providers=["skillcorner", "idsse"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )

    first = cli._assemble_corpus(args)
    assert set(first["game_id"]) == {"idsse:M2"}
    shard_root = tmp_path / "r_corpus" / "shards"
    (generation,) = [p for p in shard_root.iterdir() if p.is_dir()]
    assert (generation / "skillcorner__m1.parquet").is_file(), "the excluded match left no shard"

    loaded.clear()
    second = cli._assemble_corpus(args)
    assert loaded == [], "the excluded match was re-attempted on resume"
    assert set(second["game_id"]) == {"idsse:M2"}


def test_a_NARROWED_corpus_does_not_inherit_the_previous_run_s_matches(monkeypatch, tmp_path):
    """The corpus must be THIS run's requested matches, never the whole generation directory.

    `--providers` and `--max-matches-per-provider` are corpus SELECTORS, deliberately absent from
    `token_inputs` so that narrowing the corpus reuses the shards it can instead of re-downloading
    them. That makes the generation directory a SUPERSET of any one run -- so combining it with
    `_driver.reconcile`, whose whole-generation read is right for a partitioned driver, silently
    returns matches nobody asked for. MEASURED against that implementation: a `--providers
    skillcorner` run following a two-provider run returned ['idsse:M2', 'skillcorner:m1'].

    The first assertion is the non-vacuity partner: the wide run must genuinely have deposited an
    idsse shard, or the narrow run has nothing to inherit and this passes for the wrong reason.
    """
    import scripts.calibrate_xt_bandwidth as cli

    loaded: list = []
    _shardable_corpus(monkeypatch, loaded=loaded)
    wide = cli._assemble_corpus(
        _corpus_args(
            providers=["skillcorner", "idsse"],
            max_matches_per_provider=None,
            cache_dir=None,
            report_out=str(tmp_path / "r"),
            shard_dir=None,
        )
    )
    assert set(wide["game_id"]) == {"skillcorner:m1", "idsse:M2"}

    loaded.clear()
    narrow = cli._assemble_corpus(
        _corpus_args(
            providers=["skillcorner"],
            max_matches_per_provider=None,
            cache_dir=None,
            report_out=str(tmp_path / "r"),
            shard_dir=None,
        )
    )
    assert set(narrow["game_id"]) == {"skillcorner:m1"}, "the narrowed run inherited a shard it did not request"
    assert loaded == [], "narrowing re-loaded a match whose shard already existed"


def test_a_FAILED_match_raises_but_KEEPS_the_shards_the_pass_did_write(monkeypatch, tmp_path):
    """`for_each` records a failing item and carries on, which is right for a corpus pass and wrong
    for a corpus that feeds a cited recommendation: silently dropping a match moves the sweep with
    nothing in the manifest to show for it. So the driver persists everything, then refuses.

    The second half is what makes the refusal cheap: the successful match's shard survives, so the
    re-run retries ONLY the failure. Asserted by call record, not by inference.
    """
    import pytest

    import scripts.calibrate_xt_bandwidth as cli

    loaded: list = []
    _shardable_corpus(monkeypatch, loaded=loaded, raises=[("idsse", "M2")])
    args = _corpus_args(
        providers=["skillcorner", "idsse"],
        max_matches_per_provider=None,
        cache_dir=None,
        report_out=str(tmp_path / "r"),
        shard_dir=None,
    )

    with pytest.raises(RuntimeError, match="failed to load"):
        cli._assemble_corpus(args)
    assert sorted(loaded) == [("idsse", "M2"), ("skillcorner", "m1")]

    loaded.clear()
    with pytest.raises(RuntimeError, match="failed to load"):
        cli._assemble_corpus(args)
    assert loaded == [("idsse", "M2")], f"the re-run should retry only the failure, it loaded {loaded}"


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
