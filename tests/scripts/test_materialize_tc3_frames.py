"""The materialized corpus must contain everything the trainer needs, not just the frames.

Two halves. `assert_frames_parity` protects the WRITE path: if the parquet round-trip drifts a
schema, dtype or value, the trainer fits on different data than the established pipeline produces.
(It cannot protect the PARSE -- this driver and `_loader_pining_to_cache.py` share one
`load_matches`, so the design note claiming "the established input comes from a different pipeline"
was wrong.)

`collect_home_team_map` protects the INPUT SET: `train_ghost_gk.py` needs a home-team mapping and
per-game actions, neither of which a `for_each` generation carries -- flat shards have no `meta.json`
sibling. Missing them, the trainer either exits 1 or silently fits on a shorter corpus.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.materialize_tc3_frames import assert_frames_parity, collect_home_team_map


def _frame(**over) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "frame_id": [1, 2],
            "player_id": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "team_id": ["H", "A"],
            "vx": [0.5, 0.6],
        }
    )
    for k, v in over.items():
        base[k] = v
    return base


def test_identical_frames_pass():
    assert_frames_parity(_frame(), _frame(), match_id="m1")


def test_missing_column_is_rejected():
    with pytest.raises(AssertionError, match="column"):
        assert_frames_parity(_frame().drop(columns=["team_id"]), _frame(), match_id="m1")


def test_row_count_mismatch_is_rejected():
    with pytest.raises(AssertionError, match="row count"):
        assert_frames_parity(_frame().iloc[:1], _frame(), match_id="m1")


def test_dtype_drift_is_rejected():
    with pytest.raises(AssertionError, match="dtype"):
        assert_frames_parity(_frame(x=[1, 2]), _frame(), match_id="m1")


def test_value_drift_is_rejected():
    """Compare CONTENT, not just shape -- a schema-equal frame with different coordinates is
    exactly the silent-skew case."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(x=[1.5, 2.0]), _frame(), match_id="m1")


def test_NON_KEY_value_drift_is_also_rejected():
    """Measured defect in the first draft: hashing only the identity columns let `vx` drift from
    0.5 to 99.0 undetected. Ghost's extractor consumes velocity and so does `infer_ball_carrier`,
    so a positions-right / velocities-wrong parse is precisely the silent skew this gate names."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(vx=[99.0, 0.6]), _frame(), match_id="m1")


def test_negative_zero_does_not_trip_a_spurious_failure():
    """`-0.0` and `0.0` hash differently unless normalised. Negative zero is reachable via the
    velocity NEGATION (`-vx` where `vx == 0.0`), and the corpus driver treats a parity failure as
    STOP -- so this would cost a corpus pass to diagnose."""
    assert_frames_parity(_frame(x=[-0.0, 2.0]), _frame(x=[0.0, 2.0]), match_id="m1")


def _sidecar(home_dir, stem: str, *, home: str, game_ids: list[str]) -> None:
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / f"{stem}.json").write_text(json.dumps({"home_team_id": home, "game_ids": game_ids}))


def test_home_map_is_keyed_by_game_id_not_match_id(tmp_path):
    """SkillCorner's `game_id` is a kloppy hash unrelated to its match id, and the trainer looks up
    `home_team_map.get(str(game_id))`. Keying by the match id would miss every SkillCorner game
    while looking populated."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1886347", home="55", game_ids=["a1b2c3hash"])
    assert collect_home_team_map(home, [("skillcorner", "1886347")]) == {"a1b2c3hash": "55"}


def test_a_match_contributing_several_game_ids_maps_all_of_them(tmp_path):
    home = tmp_path / "_home"
    _sidecar(home, "gradientsports__10502", home="7", game_ids=["g1", "g2"])
    assert collect_home_team_map(home, [("gradientsports", "10502")]) == {"g1": "7", "g2": "7"}


def test_a_missing_sidecar_RAISES_rather_than_yielding_a_short_map(tmp_path):
    """The failure this prevents is quiet: `for_each` resumes by skipping items whose SHARD exists,
    so a generation written before the side artifacts has shards and no sidecars. Returning the
    partial map would let the trainer print `SKIP game <id>` per game and fit on a SHORTER corpus
    while reporting success."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    with pytest.raises(SystemExit, match="no home-team sidecar"):
        collect_home_team_map(home, [("skillcorner", "1"), ("skillcorner", "2")])


def test_the_missing_sidecar_guard_is_not_vacuous(tmp_path):
    """Companion to the test above: the same two keys pass once BOTH sidecars exist, so the raise
    is caused by absence and not by the shape of the call."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    _sidecar(home, "skillcorner__2", home="66", game_ids=["g2"])
    assert collect_home_team_map(home, [("skillcorner", "1"), ("skillcorner", "2")]) == {"g1": "55", "g2": "66"}


def test_duplicate_identity_rows_are_order_insensitive():
    """Two rows tying on every identity column but differing on `vx` must hash the same regardless
    of order. They did not, until the sort key was widened to ALL columns -- and GS duplicate frames
    make this reachable, at the same spurious-STOP cost as negative zero."""
    dup = _frame()
    dup["frame_id"] = [1, 1]
    dup["player_id"] = ["a", "a"]
    dup["x"] = [1.0, 1.0]
    dup["y"] = [3.0, 3.0]
    dup["team_id"] = ["H", "H"]
    dup["vx"] = [0.5, 9.9]
    assert_frames_parity(dup, dup.iloc[::-1].reset_index(drop=True), match_id="m1")


def test_the_missing_sidecar_message_reports_the_real_corpus_size(tmp_path):
    """Regression: `keys` was consumed by the loop and then re-listed for the message, so a
    GENERATOR reported "N of 0 shards" -- a diagnostic that misstates the corpus size at exactly the
    moment someone is working out what went missing."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    keys = iter([("skillcorner", "1"), ("skillcorner", "2"), ("skillcorner", "3")])
    with pytest.raises(SystemExit, match="2 of 3 shards"):
        collect_home_team_map(home, keys)


def test_match_ids_json_constrains_corpus_and_keys_generation(tmp_path, monkeypatch):
    """TF-65 C2: --match-ids-json loads a {provider:[ids]} roster, threads it to pining_source
    (constraining the corpus to the F1b 179), and keys a distinct generation via a roster_sha in
    token_inputs. Without the flag, no roster_sha (the full-catalog generation)."""
    import sys

    import scripts.materialize_tc3_frames as M
    from scripts import _loader_pining as LP

    captured: dict = {}

    def fake_pining_source(providers, *, match_ids=None, max_per_provider=None, cache_dir=None, **kw):
        captured["match_ids"] = match_ids
        return [], (lambda ref: None)

    class _StopError(Exception):
        pass

    def fake_for_each(refs, **kw):
        captured["token_inputs"] = kw.get("token_inputs")
        raise _StopError

    monkeypatch.setattr(LP, "pining_source", fake_pining_source)
    monkeypatch.setattr(LP, "resolve_cache_dir", lambda d: d)
    monkeypatch.setattr(M, "for_each", fake_for_each)
    monkeypatch.setattr(M, "git_provenance", lambda: {"commit": "deadbeef", "tree_dirty": False})
    monkeypatch.setattr(M, "require_clean_tree", lambda prov, allow_dirty=False: None)

    roster = {"gradientsports": ["10502"], "skillcorner": ["1021404"]}
    rp = tmp_path / "roster.json"
    rp.write_text(json.dumps(roster))

    base = [
        "materialize_tc3_frames.py",
        "--cache-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "o"),
        "--providers",
        "gradientsports",
        "skillcorner",
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--match-ids-json", str(rp)])
    with pytest.raises(_StopError):
        M.main()
    assert captured["match_ids"] == roster
    assert "roster_sha" in captured["token_inputs"]
    sha_with = captured["token_inputs"]["roster_sha"]

    captured.clear()
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(_StopError):
        M.main()
    assert captured["match_ids"] is None
    assert "roster_sha" not in captured["token_inputs"]

    # a different roster -> a different generation key
    roster2 = {"gradientsports": ["10503"]}
    rp2 = tmp_path / "roster2.json"
    rp2.write_text(json.dumps(roster2))
    captured.clear()
    monkeypatch.setattr(sys, "argv", [*base, "--match-ids-json", str(rp2)])
    with pytest.raises(_StopError):
        M.main()
    assert captured["token_inputs"]["roster_sha"] != sha_with


def test_part2_extras_stage_the_skillcorner_native_footprint_and_profile(tmp_path, monkeypatch):
    # TF-46 Part 2 staging (CL-8): the GI events, the camera footprint built with the match's OWN frames report
    # (one orientation decision), and the profile record (season from the manifest date, opendata from the
    # public visibility, the SkillCorner pitch length).
    import json
    import shutil
    from pathlib import Path

    import scripts._loader_pining as L

    src = Path("tests/datasets/skillcorner/opendata")
    shutil.copy(src / "1886347_match.json", tmp_path / "m.json")
    shutil.copy(src / "1886347_events_slim.csv", tmp_path / "e.csv")
    meta = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    pids = [p["id"] for p in meta["players"]][:4]
    corners = {
        "x_top_left": -30.0,
        "y_top_left": 20.0,
        "x_top_right": 30.0,
        "y_top_right": 20.0,
        "x_bottom_right": 30.0,
        "y_bottom_right": -20.0,
        "x_bottom_left": -30.0,
        "y_bottom_left": -20.0,
    }
    with (tmp_path / "t_tracking_extrapolated.jsonl").open("w", encoding="utf-8") as fh:
        for f in range(1, 11):
            fh.write(
                json.dumps(
                    {
                        "period": 1,
                        "frame": f,
                        "timestamp": f"00:00:{f / 10:04.1f}",
                        "player_data": [
                            {"player_id": pid, "x": -10.0 + i, "y": 1.0 * i, "is_detected": True}
                            for i, pid in enumerate(pids)
                        ],
                        "ball_data": {"x": 0.0, "y": 0.0, "z": 0.1, "is_detected": True},
                        "image_corners_projection": corners,
                    }
                )
                + "\n"
            )
    paths = {
        "metadata": tmp_path / "m.json",
        "events": tmp_path / "e.csv",
        "tracking": tmp_path / "t_tracking_extrapolated.jsonl",
    }
    monkeypatch.setattr(L, "_download_artifacts", lambda *a, **k: paths)
    monkeypatch.setattr(L, "_resolve_token", lambda t: "tok")
    frames, report = L.build_skillcorner_frames(paths, "1886347", None, with_velocity=False)
    ref = L.MatchRef("skillcorner", "1886347")
    loaded = L.LoadedMatch("skillcorner", "1886347", pd.DataFrame(), frames, "h", None, report)
    ex = L.part2_extras(
        ref, loaded, cache_dir=tmp_path / "c", entry={"date": "2024-09-14", "visibility": "public"}, base_url="http://x"
    )
    assert len(ex["native"]) > 0 and len(ex["footprint"]) == 10
    assert {"cam_tl_x", "cam_br_y"} <= set(ex["footprint"].columns)
    prof = ex["profile"]
    assert prof["competition_season"] == "2024/2025" and prof["is_opendata"] is True
    assert prof["pitch_length"] == float(meta["pitch_length"]) and prof["tracking_release"] == "sc_jsonl"
    over = L.part2_extras(
        ref, loaded, cache_dir=tmp_path / "c", entry={}, overrides={"tracking_release": "v3"}, base_url="http://x"
    )["profile"]
    assert over["tracking_release"] == "v3" and over["is_opendata"] is False


def test_part2_main_stamps_the_run_provenance_not_a_provider_name(tmp_path, monkeypatch):
    # TF-46 Part 2 R.2 staging smoke (2026-10-23): the --part2 manifest loop reused the name `prov` and shadowed the
    # git provenance, so `manifest_all.json` crashed at prov["commit"]. The run commit must reach the manifest.
    import sys

    import scripts.materialize_tc3_frames as M
    from scripts import _loader_pining as LP

    class _Res:
        shard_keys: tuple = ()
        shard_dir = tmp_path / "o" / "shards" / "gen"

        def manifest(self):
            return {}

    monkeypatch.setattr(LP, "pining_source", lambda providers, **kw: ([], lambda ref: None))
    monkeypatch.setattr(LP, "resolve_cache_dir", lambda d: d)
    monkeypatch.setattr(LP, "_resolve_token", lambda t: "tok")
    monkeypatch.setattr(LP, "_base_url", lambda: "http://x")
    monkeypatch.setattr(LP, "_list_matches", lambda prov, tok, base: [{"id": "1"}])
    monkeypatch.setattr(M, "for_each", lambda refs, **kw: _Res())
    monkeypatch.setattr(M, "collect_home_team_map", lambda home_dir, keys: {})
    monkeypatch.setattr(M, "git_provenance", lambda: {"commit": "deadbeef", "dirty": False})
    monkeypatch.setattr(M, "require_clean_tree", lambda prov, allow_dirty=False: None)
    out = tmp_path / "o"
    argv = ["m.py", "--cache-dir", str(tmp_path), "--out", str(out), "--providers", "gradientsports", "--part2"]
    monkeypatch.setattr(sys, "argv", argv)
    M.main()
    man = json.loads((out / "manifest_all.json").read_text(encoding="utf-8"))
    assert man["run_commit"] == "deadbeef" and man["run_tree_dirty"] is False


def test_part2_extras_profile_carries_pitch_dims_for_a_native_provider(tmp_path, monkeypatch):
    # TF-46 Part 2 R.2 staging smoke: `pitch_width` was bound only on the SkillCorner branch, so IDSSE / GS units
    # failed with an unbound local. A provider already on the SPADL frame records 105 x 68.
    from scripts import _loader_pining as L

    (tmp_path / "roster.json").write_text("[]", encoding="utf-8")
    (tmp_path / "events.json").write_text("[]", encoding="utf-8")
    paths = {"roster": tmp_path / "roster.json", "events": tmp_path / "events.json"}
    monkeypatch.setattr(L, "_download_artifacts", lambda *a, **k: paths)
    monkeypatch.setattr(L, "_resolve_token", lambda t: "tok")
    monkeypatch.setattr(L, "_gs_flatten_events", lambda events, roster: pd.DataFrame())  # the dims are under test
    ref = L.MatchRef("gradientsports", "10502")
    loaded = L.LoadedMatch("gradientsports", "10502", pd.DataFrame(), pd.DataFrame(), "h", None, None)
    ex = L.part2_extras(ref, loaded, cache_dir=tmp_path / "c", entry={"date": "2024-09-14"}, base_url="http://x")
    assert ex["profile"]["pitch_length"] == 105.0 and ex["profile"]["pitch_width"] == 68.0


def test_worker_partitions_are_disjoint_and_complete():
    from scripts.materialize_tc3_frames import in_worker_partition

    keys = [("skillcorner", str(m)) for m in range(500)] + [("idsse", f"DFL-MAT-{m}") for m in range(7)]
    for n in (1, 3, 16):
        owners = [[i for i in range(n) if in_worker_partition(k, i, n)] for k in keys]
        assert all(len(o) == 1 for o in owners)  # each unit in exactly one partition
        if n > 1:
            assert len({o[0] for o in owners}) == n  # every partition receives units


def test_worker_mode_keeps_one_generation_and_defers_the_assembly(tmp_path, monkeypatch):
    # --worker does not enter the token (N workers fill ONE generation); a worker writes no home_teams/manifest;
    # the final pass without --worker sees every ref and assembles them.
    import sys
    from types import SimpleNamespace

    import scripts.materialize_tc3_frames as M
    from scripts import _loader_pining as LP

    refs = [SimpleNamespace(provider="skillcorner", match_id=str(m), key=("skillcorner", str(m))) for m in range(40)]
    seen: dict = {}

    class _Res:
        shard_keys: tuple = ()
        shard_dir = tmp_path / "o" / "shards" / "gen"

        def manifest(self):
            return {}

    def fake_for_each(items, **kw):
        seen.setdefault("calls", []).append((list(items), kw["token_inputs"]))
        return _Res()

    monkeypatch.setattr(LP, "pining_source", lambda providers, **kw: (refs, lambda ref: None))
    monkeypatch.setattr(LP, "resolve_cache_dir", lambda d: d)
    monkeypatch.setattr(M, "for_each", fake_for_each)
    monkeypatch.setattr(M, "collect_home_team_map", lambda home_dir, keys: {})
    monkeypatch.setattr(M, "git_provenance", lambda: {"commit": "deadbeef", "dirty": False})
    monkeypatch.setattr(M, "require_clean_tree", lambda prov, allow_dirty=False: None)
    out = tmp_path / "o"
    base = ["m.py", "--cache-dir", str(tmp_path), "--out", str(out), "--providers", "skillcorner"]
    for i in range(4):
        monkeypatch.setattr(sys, "argv", [*base, "--worker", f"{i}/4"])
        M.main()
    assert not (out / "home_teams.json").exists() and not (out / "manifest_all.json").exists()
    monkeypatch.setattr(sys, "argv", base)
    M.main()
    calls = seen["calls"]
    assert len({json.dumps(t, sort_keys=True) for _items, t in calls}) == 1  # ONE generation token for all
    worker_units = [r.key for items, _t in calls[:4] for r in items]
    assert sorted(worker_units) == sorted(r.key for r in refs) and len(set(worker_units)) == len(refs)
    assert len(calls[4][0]) == len(refs)  # the final pass walks (resumes over) every unit
    assert (out / "home_teams.json").exists() and (out / "manifest_all.json").exists()


def test_worker_spec_is_validated():
    from scripts.materialize_tc3_frames import parse_worker

    for bad in ("4/4", "x/4", "3", "-1/4"):
        with pytest.raises(SystemExit):
            parse_worker(bad)
