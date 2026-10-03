"""Local reduce-path + seam tests for scripts/validate_das_native_parity.py (spec 7.2, Task 14).

The clean-tree / --allow-dirty / no-rev-parse / enrolment conformance is asserted by
tests/scripts/test_provenance_wiring.py (this driver is in ARTIFACT_DRIVERS). Here: the FULL map+reduce
runs offline on an injected two-match synthetic corpus (golden scenes) with the accessible-space leg
STUBBED by the golden reference outputs, producing a schema-complete metrics.json whose counts
reconcile -- the mandatory local check before any DGX run (the trainer-locally rule).

The corpus map uses ``direction_col="dir"`` (the golden frames carry it and have no keeper rows); the
production GoalMap direction path is covered by test_das_pack / test_das_divergences (D-DIR). The numba
leg is disabled here for speed/determinism (engine parity is test_das_engine_parity / test_das_invariance).
"""

from __future__ import annotations

import sys
from pathlib import Path

import _loader_pining as lp
import numpy as np
import pandas as pd
import pytest

from scripts import validate_das_native_parity as D
from silly_kicks.tracking._das_pack import Reason
from tests.tracking._das_golden import load_golden

_CLEAN_PROV = {
    "commit": "0" * 40,
    "dirty": False,
    "tree_state": "clean",
    "platform": "test",
    "machine": "x86_64",
}


def _obj_id(v):
    """An IDSSE-style STRING player id for a golden integer id (NA stays NA: the ball row)."""
    return pd.NA if pd.isna(v) else f"DFL-OBJ-{int(v):06d}"


def _clu_id(v):
    """An IDSSE-style STRING team id for a golden integer id (NA stays NA)."""
    return pd.NA if pd.isna(v) else f"DFL-CLU-{int(v):06d}"


def _corpus(*, perturb_ref_game: int | None = None, string_ids: bool = False):
    """Two matches from golden scenes S01/S05, each a distinct game_id. Returns (refs, load, stub_ref).

    ``load(ref) -> (provider, match_id, actions, frames)``; ``stub_ref(frames)`` returns the golden
    reference arrays for that game (keys relabeled to the match's game_id). ``perturb_ref_game`` adds a
    +1.0 offset to that game's reference team_das, so a test can prove the Δ is non-vacuous.
    ``string_ids`` relabels player/team ids to IDSSE-style strings in the frames AND the reference keys
    (the reference keys as the reference leg parses them back).
    """
    g = load_golden()
    frames_by_gid: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    ref_by_gid: dict[int, dict] = {}
    for gid, scene in ((1, "S01"), (2, "S05")):
        frames = g.frames_for(scene).copy()
        frames["game_id"] = gid  # distinct game per match (S01/S05 are both game_id=1 in the fixture)
        ref = g.reference_for(scene)
        tk = ref["team_keys"].copy()
        tk[:, 0] = gid
        pk = ref["player_keys"].copy()
        pk[:, 0] = gid
        if string_ids:
            frames["player_id"] = frames["player_id"].map(_obj_id).astype(object)
            for c in ("team_id", "team_in_possession"):
                frames[c] = frames[c].map(_clu_id).astype(object)
            pk = pk.astype(object)
            pk[:, 3] = [_obj_id(v) for v in pk[:, 3]]
        team_das = ref["team_das"].copy()
        if perturb_ref_game == gid:
            team_das = team_das + 1.0
        ref_by_gid[gid] = {**ref, "team_keys": tk, "player_keys": pk, "team_das": team_das}
        actions = pd.DataFrame(
            {
                "game_id": gid,
                "period_id": int(frames["period_id"].iloc[0]),
                "frame_id": sorted(int(f) for f in frames["frame_id"].unique()),
            }
        )
        frames_by_gid[gid] = (frames, actions)

    refs = [lp.MatchRef("skillcorner", str(gid), {}) for gid in (1, 2)]

    def load(ref: lp.MatchRef):
        gid = int(ref.match_id)
        frames, actions = frames_by_gid[gid]
        return ("skillcorner", str(gid), actions, frames)

    def stub_ref(frames: pd.DataFrame) -> dict:
        return ref_by_gid[int(frames["game_id"].iloc[0])]

    return refs, load, stub_ref


def _run(tmp_path, monkeypatch, *, perturb_ref_game=None, string_ids=False):
    monkeypatch.setattr(D, "_numba_available", lambda: False)  # skip the numba compile in a unit test
    refs, load, stub_ref = _corpus(perturb_ref_game=perturb_ref_game, string_ids=string_ids)
    return D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir", reference_leg=stub_ref)


def test_full_reduce_path_is_schema_complete_and_counts_reconcile(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)

    # metrics.json written and re-readable.
    import json

    on_disk = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["run_commit"] == _CLEAN_PROV["commit"]
    assert on_disk["run_tree_dirty"] is False

    # Conservation: two matches attempted, none failed or excluded (ADR-052).
    assert out["n_attempted"] == 2
    assert out["n_failed"] == 0
    assert out["n_excluded"] == 0

    prov = out["providers"]["skillcorner"]
    assert prov["n_matches_scored"] == 2
    # Schema-complete: every spec 7.2 block present.
    for grain in ("team", "player"):
        for output in ("das", "as"):
            for kind in ("abs", "rel"):
                assert set(prov[grain][output][kind]) == {"max", "p99", "p50"}
    assert set(prov["quadrature_shift_das"]) == {"median", "p90", "max"}
    assert set(prov["timings_ms_per_frame"]) == {"ref", "numpy", "numba", "periodic"}
    assert set(prov["finite_mask_mismatches"]) == {"team", "player"}
    assert set(prov["reason_counts"]) == set(D._REASON_COLS.values())
    # No divergence-exclusion block: the reference respects offside + keys frames collision-free, so the
    # per-class NaN-degrade accounting is reason_counts (asserted above) and the headline is reason==OK.
    assert "divergences" not in prov

    # Population block: listed / scored / excluded.
    pop = out["population"]
    assert pop["listed_per_provider"]["skillcorner"] == 2
    assert pop["scored_per_provider"]["skillcorner"] == 2


def test_native_reproduces_the_reference_leg_within_parity(tmp_path, monkeypatch):
    # The native numpy engine in REFERENCE quadrature reproduces the golden library outputs (~1e-12),
    # so the parity Δ is machine-epsilon and no frame's finiteness disagrees.
    out = _run(tmp_path, monkeypatch)
    prov = out["providers"]["skillcorner"]
    assert prov["team"]["das"]["abs"]["max"] < 1e-6
    assert prov["player"]["das"]["abs"]["max"] < 1e-6
    assert prov["finite_mask_mismatches"] == {"team": 0, "player": 0}
    # S01/S05 are normal full-team single-period scenes: every frame is reason==OK, so the headline is
    # the whole corpus.
    assert prov["n_matches_scored"] == 2
    # The quadrature shift (periodic - reference) is a REAL, non-zero corpus figure (ADR-108).
    assert np.isfinite(prov["quadrature_shift_das"]["max"])


def test_string_ids_idsse_shape_are_scored_and_joined(tmp_path, monkeypatch):
    # IDSSE ids are strings (DFL-OBJ-* / DFL-CLU-*). The DGX smoke (combined-cycle Task 17) failed every IDSSE
    # match on ``int(canonical_id(v))``. With string ids the map must score both matches AND join the player
    # grain (an empty player grade would mean the native and reference keys never matched).
    out = _run(tmp_path, monkeypatch, string_ids=True)
    assert out["n_failed"] == 0
    prov = out["providers"]["skillcorner"]
    assert prov["n_matches_scored"] == 2
    assert prov["player"]["das"]["abs"]["max"] < 1e-6
    assert prov["finite_mask_mismatches"] == {"team": 0, "player": 0}


@pytest.mark.parametrize("raw", [100, 100.0, np.int64(100), "100", "DFL-OBJ-0002DR"])
def test_native_and_reference_player_keys_agree(raw):
    # One rule on both sides of the join: the reference leg's token for an id, parsed back, equals the
    # native key for the same raw id (an int for a numeric id, else the string).
    from scripts import _das_reference_leg as R

    assert R._id_key(R._id_token(raw)) == D._native_player_key(raw)


def _bench_with_stubs(monkeypatch, fake_path):
    """Run ``_bench_match`` on a tiny frame with every heavy leg stubbed; only the path legs are a real seam."""
    import silly_kicks.tracking._das_engine as E
    import silly_kicks.tracking._das_pack as P

    raw = pd.DataFrame({"game_id": [1, 1], "period_id": [1, 1], "frame_id": [1, 2], "x": [1.0, 2.0], "team_id": [1, 1]})

    def fake_possession(f):
        return f if "team_in_possession" in f.columns else f.assign(team_in_possession=1)

    monkeypatch.setattr(D, "_prepare_possession", fake_possession)
    monkeypatch.setattr(D, "_scored_frames", lambda actions, frames: frames)
    monkeypatch.setattr(D, "_direction_column", lambda f, direction_col: f.assign(**{D._DIR_COL: 1.0}))
    monkeypatch.setattr(D, "_reference_leg_subprocess", lambda *a, **k: {"compute_s": 1.0})
    monkeypatch.setattr(D, "_run_native", lambda *a, **k: None)
    monkeypatch.setattr(D, "_time_leg", lambda fn, repeat: (None, 1.0))
    monkeypatch.setattr(D, "_sweep_counts", lambda: [1])
    monkeypatch.setattr(D, "_keeper_counterfactual", lambda scored: (scored, False))
    monkeypatch.setattr(D, "_path_subprocess", fake_path)
    monkeypatch.setattr(P, "pack_frames", lambda *a, **k: None)
    monkeypatch.setattr(E, "compute_das", lambda *a, **k: None)
    return D._bench_match(
        ("gradientsports", "m", pd.DataFrame(), raw),
        reference_python="r",
        old_path_python="o",
        new_path_python="n",
        repeat=1,
    )


def test_bench_match_path_legs_get_frames_with_possession(monkeypatch):
    # The old/new path legs time add_das, which REFUSES frames without team_in_possession. Raw pining
    # frames lack it, so the benchmark failed every match on the DGX (combined-cycle Task 17). Every heavy
    # leg is stubbed: only what reaches the path legs is under test.
    seen: list[pd.DataFrame] = []

    def fake_path(frames, actions, *, python, repeat, expect_native, memory_limit_gib):
        seen.append(frames)
        return {"add_das_s": 1.0, "das_xfns_s": 1.0, "pandas": "2.3.3", "over_memory": False}

    row = _bench_with_stubs(monkeypatch, fake_path)
    assert len(seen) == 2  # new path + old path
    assert all("team_in_possession" in f.columns for f in seen)
    assert row["add_das_new_s"] == 1.0 and row["add_das_old_s"] == 1.0


def test_bench_match_records_an_old_path_that_did_not_fit(monkeypatch):
    # GS: the old path (4.127.0 + accessible-space) was OOM-killed inside a 110G cap in das_xfns, while the
    # new path needed 2.6 GB (combined-cycle Phase B). A ceiling trip is a RESULT: keep what finished.
    limits: list = []

    def fake_path(frames, actions, *, python, repeat, expect_native, memory_limit_gib):
        limits.append(memory_limit_gib)
        if expect_native:
            return {"add_das_s": 1.0, "das_xfns_s": 6.0, "pandas": "2.3.3", "over_memory": False}
        return {
            "add_das_s": 11.0,
            "pandas": "2.3.3",
            "over_memory": True,
            "phase": "das_xfns",
            "limit_gib": 100.0,
            "peak_gib": 100.4,
        }

    row = _bench_with_stubs(monkeypatch, fake_path)
    assert limits == [D._PATH_MEMORY_LIMIT_GIB] * 2
    assert row["add_das_old_s"] == 11.0 and "das_xfns_old_s" not in row
    assert row["old_path_over_memory"] == {"phase": "das_xfns", "limit_gib": 100.0, "peak_gib": 100.4}
    assert row["das_xfns_new_s"] == 6.0 and "new_path_over_memory" not in row


def test_summarize_benchmark_speedups_use_the_matches_where_both_paths_finished():
    base = {
        "n_scored_frames": 100,
        "ms_frame_ref": 30.0,
        "ms_frame_numpy": 10.0,
        "ms_frame_numba_serial": 2.0,
        "ms_frame_numpy_periodic": 11.0,
        "numba_threads_ms_frame": {"1": 2.0},
        "add_das_new_s": 1.0,
        "das_xfns_new_s": 1.0,
    }
    full = [dict(base, provider=p, add_das_old_s=20.0, das_xfns_old_s=60.0) for p in ("sk", "idsse")]
    gs = dict(
        base,
        provider="gs",
        add_das_old_s=10.0,
        old_path_over_memory={"phase": "das_xfns", "limit_gib": 100.0, "peak_gib": 100.4},
    )
    s = D.summarize_benchmark([*full, gs])
    assert s["das_xfns_speedup"] == 60.0  # sk + idsse only
    assert s["add_das_speedup"] == 20.0  # median of 20, 20, 10
    assert s["speedup_n"] == {"add_das": 3, "das_xfns": 2}
    assert s["old_path_over_memory"] == {"gs": 1}
    assert s["n_matches"] == 3  # the GS match still counts for every engine figure


def test_path_subprocess_passes_the_ceiling_and_returns_a_trip(monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd[-1] == "100.0"  # the ceiling reaches the timing module
        (Path(cmd[3]) / "timing.json").write_text(
            '{"add_das_s": 9.0, "over_memory": true, "phase": "das_xfns", "limit_gib": 100.0,'
            ' "peak_gib": 100.2, "silly_kicks": "4.127.0", "native": false}'
        )

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    t = D._path_subprocess(
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [1]}),
        python="py",
        repeat=1,
        expect_native=False,
        memory_limit_gib=100.0,
    )
    assert t["over_memory"] is True and t["phase"] == "das_xfns"


def test_the_reduce_is_non_vacuous_a_reference_gap_shows_up(tmp_path, monkeypatch):
    # Perturb one game's reference team_das by +1.0: the corpus max abs Δ must jump to ~1.0, proving the
    # parity computation is not vacuously zero (a gate that cannot see a gap protects nothing).
    out = _run(tmp_path, monkeypatch, perturb_ref_game=2)
    assert out["providers"]["skillcorner"]["team"]["das"]["abs"]["max"] == pytest.approx(1.0, abs=1e-6)


def test_input_contract_declares_a_stable_digest(tmp_path):
    ic = D.input_contract()
    assert ic["driver"] == "validate_das_native_parity"
    assert ic["digest"]  # ADR-056
    assert ic["params"]["reference_quadrature"] == "reference"
    assert ic["params"]["schema"] == D._SHARD_SCHEMA_VERSION


def test_reduce_parity_empty_is_empty():
    assert D.reduce_parity([]) == {}
    assert D.reduce_parity([pd.DataFrame(columns=D._EMITTED_SHARD_COLUMNS)]) == {}


def test_measure_match_empty_frames_returns_columns():
    # reference_leg is now a required kwarg; empty frames return early (before the leg is called), so the
    # stub is unused -- but it must be passed.
    shard = D._measure_match(("skillcorner", "1", None, pd.DataFrame()), reference_leg=lambda f: {})
    assert list(shard.columns) == D._EMITTED_SHARD_COLUMNS
    assert shard.empty


def test_scored_frames_links_by_time_when_actions_lack_frame_id(monkeypatch):
    # Real corpus actions carry time_seconds, NOT frame_id -> the driver must link (ADR-004) and score
    # ONLY the linked (period, frame) frames, never every frame in the match (~1e6). Regression for the
    # DGX finding: the old frame_id-only check fell through to scoring all frames.
    import silly_kicks.tracking as T

    frames = pd.DataFrame(
        {"game_id": 1, "period_id": 1, "frame_id": [10, 11, 12, 13], "player_id": ["A", "A", "A", "A"], "x": 1.0}
    )
    actions = pd.DataFrame({"action_id": [1, 2], "period_id": [1, 1], "time_seconds": [0.1, 0.3]})

    def fake_link(a, f, **kw):
        return pd.DataFrame({"action_id": [1, 2], "frame_id": pd.array([10, 12], dtype="Int64")}), object()

    monkeypatch.setattr(T, "link_actions_to_frames", fake_link)
    scored = D._scored_frames(actions, frames)
    assert sorted(scored["frame_id"].tolist()) == [10, 12], "scored set is the LINKED frames, not all"


def test_prepare_possession_derives_only_when_absent(monkeypatch):
    import silly_kicks.tracking as T

    # present -> pass-through (identity, no derive call).
    have = pd.DataFrame({"game_id": [1], "frame_id": [1], "team_in_possession": ["H"]})
    assert D._prepare_possession(have) is have

    # absent -> infer_ball_carrier + derive_team_in_possession are called and the result is returned.
    raw = pd.DataFrame({"game_id": [1], "frame_id": [1], "team_id": ["H"]})
    sentinel_carrier = object()
    derived = raw.assign(team_in_possession=["H"], ball_carrier_player_id=[7])
    calls = {}

    def fake_infer(f, **kw):
        calls["infer"] = True
        return sentinel_carrier

    def fake_derive(f, carrier):
        calls["derive_carrier_is_sentinel"] = carrier is sentinel_carrier
        return derived

    monkeypatch.setattr(T, "infer_ball_carrier", fake_infer)
    monkeypatch.setattr(T, "derive_team_in_possession", fake_derive)
    out = D._prepare_possession(raw)
    assert calls == {"infer": True, "derive_carrier_is_sentinel": True}
    assert "team_in_possession" in out.columns and out is derived


# --------------------------------------------------------------------------------------------------
# Reduce accounting: headline + finite-mask over reason==OK; reason_counts is the per-class accounting.
# --------------------------------------------------------------------------------------------------


def _mk_row(grain: str, **kw) -> dict:
    row: dict[str, object] = dict.fromkeys(D._EMITTED_SHARD_COLUMNS, np.nan)
    row.update(grain=grain, provider="skillcorner", game_id=1)
    row.update(kw)
    return row


def _shard(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=D._EMITTED_SHARD_COLUMNS)


def test_reduce_headline_and_finite_mask_are_over_reason_ok_rows():
    rows = [
        _mk_row(
            "team",
            period_id=1,
            frame_id=1,
            abs_das=1e-9,
            rel_das=1e-9,
            abs_as=1e-9,
            rel_as=1e-9,
            finite_ref=True,
            finite_native=True,
            quad_shift_das=0.0,
            reason=int(Reason.OK),
        ),
        # native NaNs a NaN-ball frame while the reference recorded a fictional value (D-BALLNAN): a
        # finite-mask mismatch that must NOT count -- the frame is reason != OK, so it is excluded from
        # the headline and the finite-mask, and tallied by reason_counts (on the match row).
        _mk_row(
            "team",
            period_id=1,
            frame_id=2,
            abs_das=np.nan,
            rel_das=np.nan,
            abs_as=np.nan,
            rel_as=np.nan,
            finite_ref=True,
            finite_native=False,
            quad_shift_das=np.nan,
            reason=int(Reason.BALL_NAN),
        ),
        _mk_row("match", n_scored_frames=2, **{**{c: 0 for c in D._REASON_COLS.values()}, "reason_ball_nan": 1}),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["finite_mask_mismatches"]["team"] == 0, "the BALL_NAN mismatch is excluded by the reason==OK filter"
    assert prov["reason_counts"]["reason_ball_nan"] == 1, "the NaN-degrade class is counted by reason_counts"
    assert "divergences" not in prov


def test_load_match_ids_groups_the_list_matches_shape_by_provider():
    spec = [
        {"provider": "skillcorner", "match_id": "111"},
        {"provider": "gradientsports", "match_id": "222"},
        {"provider": "skillcorner", "match_id": "333"},
    ]
    got = D._load_match_ids(spec)
    assert got == {"skillcorner": ["111", "333"], "gradientsports": ["222"]}


# --------------------------------------------------------------------------------------------------
# Reference-leg pandas-2 subprocess: locate / probe / marshalling (monkeypatched subprocess, no venv).
# --------------------------------------------------------------------------------------------------


def test_resolve_reference_python_missing_is_fatal(monkeypatch):
    monkeypatch.delenv("SK_DAS_REFERENCE_PYTHON", raising=False)
    with pytest.raises(SystemExit) as ei:
        D._resolve_reference_python(None)
    assert "accessible-space==2.0.15" in str(ei.value) and "pandas<3" in str(ei.value)


def test_resolve_reference_python_accepts_existing_file(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SK_DAS_REFERENCE_PYTHON", str(fake))
    assert D._resolve_reference_python(None) == str(fake)
    assert D._resolve_reference_python(str(fake)) == str(fake)  # explicit arg wins


def test_probe_reference_env_rejects_pandas3(monkeypatch):
    class R:
        stdout = "3.0.6\n2.5.3\n2.0.15\n3.12.0\n"  # pandas / numpy / accessible-space / python

    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        D._probe_reference_env("pyx")


def test_probe_reference_env_returns_versions(monkeypatch):
    class R:
        stdout = "2.3.3\n2.5.3\n2.0.15\n3.12.0\n"

    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: R())
    assert D._probe_reference_env("pyx") == {
        "pandas": "2.3.3",
        "numpy": "2.5.3",
        "accessible_space": "2.0.15",
        "python": "3.12.0",
    }


def test_reference_leg_subprocess_round_trips(monkeypatch):
    frames = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [7],
            "is_ball": [True],
            "player_id": ["ball"],
            "team_id": [None],
            "x": [0.0],
            "y": [0.0],
            "vx": [0.0],
            "vy": [0.0],
            "team_in_possession": ["t1"],
            "_das_parity_dir": [1.0],
        }
    )

    def fake_run(cmd, **kw):
        out = Path(cmd[3])  # [python, module, in_parquet, out_dir, repeat, infer]
        pd.DataFrame({"game_id": [1], "period_id": [1], "frame_id": [7], "as": [3.0], "das": [1.5]}).to_parquet(
            out / "team.parquet"
        )
        pd.DataFrame(
            {"game_id": [1], "period_id": [1], "frame_id": [7], "player_id": [9], "as": [2.0], "das": [1.0]}
        ).to_parquet(out / "player.parquet")
        (out / "timing.json").write_text('{"compute_s": 0.25, "repeat": 1}', encoding="utf-8")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    got = D._reference_leg_subprocess(frames, reference_python="pyx")
    assert got["team_das"].tolist() == [1.5]
    assert got["team_keys"][0].tolist() == [1, 1, 7]
    assert got["player_keys"][0].tolist() == [1, 1, 7, 9]
    assert got["player_das"].tolist() == [1.0]
    assert got["compute_s"] == 0.25


def test_input_contract_declares_reference_env_pins():
    ic = D.input_contract()
    assert ic["params"]["reference_env_pins"] == {"accessible-space": "2.0.15", "pandas": "<3"}


def test_run_corpus_stamps_reference_env(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    out = D.run_corpus(
        refs,
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        direction_col="dir",
        reference_leg=stub_ref,
        reference_env={"pandas": "2.3.3", "accessible_space": "2.0.15"},
    )
    assert out["reference_env"] == {"pandas": "2.3.3", "accessible_space": "2.0.15"}


def test_n_outside_golden_bound_applies_allclose_per_row():
    abs_d = [0.0, 5e-13, 2e-12, 1e-6]
    rel_d = [0.0, 5e-15, float("nan"), 1e-8]  # exact; inside (ref 100); ref==0 & |d|>atol; outside
    assert D._n_outside_golden_bound(abs_d, rel_d, 1e-12) == 2
    extra = [0.0, 1e-9, 0.0, 0.0]  # |numba - numpy|, triangle bound
    assert D._n_outside_golden_bound(abs_d, rel_d, 1e-10, extra=extra) == 1
    assert D._n_outside_golden_bound([float("nan")], [float("nan")], 1e-12) == 0  # finite-mask case, counted elsewhere


def test_reduce_reports_all_four_cells_and_the_new_counts():
    ok = dict(
        abs_as=0.0,
        rel_as=0.0,
        finite_ref=True,
        finite_native=True,
        quad_shift_das=0.0,
        numba_minus_numpy_das=0.0,
        numba_minus_numpy_as=0.0,
        reason=int(Reason.OK),
    )
    player_ok = {**ok, "numba_minus_numpy_as": 1e-6}
    rows = [
        _mk_row("team", period_id=1, frame_id=1, abs_das=0.0, rel_das=0.0, **ok),
        _mk_row("team", period_id=1, frame_id=2, abs_das=1e-6, rel_das=1e-8, **ok),
        _mk_row("player", period_id=1, frame_id=1, player_id=9, abs_das=0.0, rel_das=0.0, **player_ok),
        _mk_row(
            "match",
            n_scored_frames=2,
            n_dkey_frames=1,
            n_dir_compared=2,
            n_dir_disagree=1,
            **{c: 0 for c in D._REASON_COLS.values()},
        ),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    out = prov["n_outside_golden_bound"]
    assert out["numpy"]["team"] == {"das": 1, "as": 0}
    assert out["numpy"]["player"] == {"das": 0, "as": 0}
    assert out["numba"]["team"] == {"das": 1, "as": 0}
    assert out["numba"]["player"] == {"das": 0, "as": 1}
    # CCC-PLAN-23: per-cell count of FINITE numba comparisons (the gate requires every cell > 0)
    assert prov["numba_compared"] == {"team": {"das": 2, "as": 2}, "player": {"das": 1, "as": 1}}
    assert prov["finite_counts"]["team"] == {"ref": 2, "native": 2, "rows": 2}
    assert prov["d_key_frames"] == 1
    assert prov["direction"] == {"n_compared": 2, "n_disagree": 1}


def test_numba_cells_with_no_finite_comparison_are_counted_zero():
    """CCC-PLAN-23: an empty/misaligned numba player merge leaves NaN diffs -- 0 compared, never 'clean'."""
    nan_nb = dict(
        abs_as=0.0,
        rel_as=0.0,
        finite_ref=True,
        finite_native=True,
        quad_shift_das=0.0,
        numba_minus_numpy_das=np.nan,
        numba_minus_numpy_as=np.nan,
        reason=int(Reason.OK),
    )
    rows = [
        _mk_row("player", period_id=1, frame_id=1, player_id=9, abs_das=0.0, rel_das=0.0, **nan_nb),
        _mk_row("match", n_scored_frames=1, **{c: 0 for c in D._REASON_COLS.values()}),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["numba_compared"]["player"] == {"das": 0, "as": 0}


def test_n_dkey_frames_counts_frames_whose_frame_id_recurs_in_another_period():
    """CCC-PLAN-22: an exact two-period fixture. Period 2 reuses frame_ids 1..5 of period 1 (10 colliding
    frame keys) and adds frame 6 (unique); a second game reusing the same ids does NOT collide."""
    keys = pd.DataFrame(
        {
            "game_id": [1] * 11 + [2] * 5,
            "period_id": [1] * 5 + [2] * 6 + [1] * 5,
            "frame_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
        }
    )
    frames = keys.loc[keys.index.repeat(3)].reset_index(drop=True)  # several rows per frame, as real frames have
    assert D._n_dkey_frames(frames) == 10


def test_measure_match_records_dkey_and_direction(monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    item = load(refs[0])

    def inferred(frames):  # the library's own direction disagrees on every frame
        r = stub_ref(frames)
        return {**r, "team_das": r["team_das"] + 5.0}

    shard = D._measure_match(item, reference_leg=stub_ref, inferred_leg=inferred, direction_col="dir")
    m = shard[shard["grain"] == "match"].iloc[0]
    assert int(m["n_dir_compared"]) > 0 and int(m["n_dir_disagree"]) == int(m["n_dir_compared"])
    assert int(m["n_dkey_frames"]) == 0  # the golden scenes are single-period: exactly 0, not ">= 0"


def test_reduce_refuses_an_unaccounted_match(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(
        refs[:1],
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    with pytest.raises(SystemExit, match="have no shard"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")


def test_reduce_at_another_commit_is_refused_by_the_generation_check(tmp_path, monkeypatch):
    """CCC-PLAN-21: the commit-keyed generation makes this the refusal that fires."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(
        refs,
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    other = {**_CLEAN_PROV, "commit": "1" * 40}
    with pytest.raises(SystemExit, match=r"does not match this reduce's token .*commit 1{40}"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=other, direction_col="dir")


def test_a_token_mismatch_at_the_same_commit_names_the_token_inputs(tmp_path, monkeypatch):
    """B r3 CCC-PLAN-26: a direction_col mismatch is not blamed on the commit alone."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(
        refs,
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    with pytest.raises(SystemExit, match=r"does not match this reduce's token .*direction_col None"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col=None)


def test_print_generation_is_the_token_the_map_writes(tmp_path, monkeypatch, capsys):
    """B r3 CCC-PLAN-28: the launcher's done-marker is keyed on the generation this prints."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(
        refs,
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col=None,
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    assert [p.name for p in root.iterdir() if p.is_dir()] == [D._map_generation(_CLEAN_PROV["commit"])]


def test_a_planted_foreign_manifest_is_refused(tmp_path, monkeypatch):
    """Defence in depth (CCC-PLAN-21): a manifest naming another commit inside THIS commit's generation
    (e.g. copied in by hand) is refused by the commits_seen check, which this test keeps exercised."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(
        refs,
        load,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    (gen,) = [p for p in root.iterdir() if p.is_dir()]
    (gen / "manifest_rogue.json").write_text('{"n_attempted": 1, "run_commit": "' + "1" * 40 + '"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="another commit"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")


def test_population_records_accounted_and_commit(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert out["population"]["accounted"] == 2  # informational: the reduce refuses before it could differ (B r2 C7)
    assert out["commits_seen"] == [_CLEAN_PROV["commit"]]


def test_main_slices_providers_to_the_allowlist(tmp_path, monkeypatch):
    """CCC-PLAN-08: a worker slice without IDSSE ids must not process the whole IDSSE manifest."""
    seen = {}

    def fake_pining_source(providers, **kw):
        seen["providers"] = list(providers)
        raise SystemExit("stop")

    monkeypatch.setattr(D, "pining_source", fake_pining_source)
    sl = tmp_path / "s.json"
    sl.write_text('[{"provider": "skillcorner", "match_id": "1"}]')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "d",
            "--out",
            str(tmp_path / "o"),
            "--match-ids-json",
            str(sl),
            "--allow-dirty",
            "--shards-only",
            "--worker-tag",
            "w0",
        ],
    )
    with pytest.raises(SystemExit, match="stop"):
        D.main()
    assert seen["providers"] == ["skillcorner"]


def test_old_path_frames_restore_pre_f1b_dtypes():
    f = pd.DataFrame(
        {"x": pd.Series([1.0], dtype="float32"), "team_id": pd.Series([1], dtype="Int64").astype("category")}
    )
    out = D._old_path_frames(f)
    assert out["x"].dtype == "float64" and str(out["team_id"].dtype) == "Int64"


def test_thread_sweep_skips_counts_above_the_cpu_count(monkeypatch):
    """B r2 C6: _THREAD_SWEEP's own comment says counts above os.cpu_count() are skipped."""
    monkeypatch.setattr(D.os, "cpu_count", lambda: 8)
    assert D._sweep_counts() == [1, 2, 4, 8]


def test_benchmark_refs_come_from_the_sample_file_only(tmp_path):
    """B r2 C5: the sample whose SHA-256 is recorded IS the population benchmarked."""
    sample = tmp_path / "s.json"
    sample.write_text('[{"provider": "idsse", "match_id": "DFL-MAT-J03WMX"}]')
    assert D._benchmark_match_ids(sample) == {"idsse": ["DFL-MAT-J03WMX"]}


def test_path_subprocess_refuses_the_wrong_engine(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        assert "PYTHONPATH" not in kw["env"]  # the subprocess must import its OWN silly-kicks
        (Path(cmd[3]) / "timing.json").write_text(
            '{"add_das_s": 1.0, "das_xfns_s": 1.0, "silly_kicks": "4.127.0", "native": true}'
        )

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "x")
    with pytest.raises(SystemExit, match="old path"):
        D._path_subprocess(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), python="py", repeat=1, expect_native=False
        )


def test_summarize_benchmark_computes_the_spec_4_2_figures():
    row = {
        "provider": "gs",
        "n_scored_frames": 100,
        "ms_frame_ref": 30.0,
        "ms_frame_numpy": 10.0,
        "ms_frame_numba_serial": 2.0,
        "ms_frame_numpy_periodic": 11.0,
        "numba_threads_ms_frame": {str(k): 2.0 / (0.8 * k) if k > 1 else 2.0 for k in D._THREAD_SWEEP},
        "add_das_new_s": 1.0,
        "add_das_old_s": 20.0,
        "das_xfns_new_s": 1.0,
        "das_xfns_old_s": 60.0,
        "paired_s": 2.0,
        "independent_s": 2.0,
    }
    s = D.summarize_benchmark([row, dict(row, provider="sk")])
    assert s["ref_over_numba_serial"] == 15.0 and s["ref_over_numpy"] == 3.0
    assert abs(s["prange_efficiency"]["16"] - 0.8) < 1e-12
    assert s["add_das_speedup"] == 20.0 and s["das_xfns_speedup"] == 60.0
    assert s["paired_over_independent"] == 1.0
    assert s["seconds_per_frame"] == {"numba_serial": 0.002, "numpy": 0.011}
    assert s["n_matches"] == 2


def test_foreign_cpu_fraction():
    # 10 s elapsed on 4 cpus = 40 cpu-s; machine busy 30 cpu-s, of which 20 were ours -> 10/40 foreign
    assert D._foreign_cpu_fraction(busy0=0.0, busy1=30.0, own0=0.0, own1=20.0, elapsed=10.0, ncpu=4) == 0.25


def test_overlapping_map_workers_are_refused_at_the_reduce(tmp_path, monkeypatch):
    """Exclusion counts ride the worker manifests and are REPLAYED on resume, so two workers covering
    one match would count its exclusion twice (combined-cycle Phase B overlap guard)."""
    import json

    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    for tag, subset in (("w0", refs), ("w1", refs[1:])):
        D.run_corpus(
            subset,
            load,
            tmp_path / "out",
            prov=_CLEAN_PROV,
            shard_root=root,
            direction_col="dir",
            reference_leg=stub_ref,
            shards_only=True,
            worker_tag=tag,
        )
    from scripts._driver import join_key

    (w1,) = list(root.glob("*/manifest_w1.json"))
    assert json.loads(w1.read_text(encoding="utf-8"))["partition_keys"] == sorted(join_key(r.key) for r in refs[1:])
    with pytest.raises(ValueError, match="overlap"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")


def test_exclusions_are_counted_from_the_markers_not_lost_on_a_relaunch(tmp_path, monkeypatch):
    """`_parallel_launch` relaunches a killed worker with ONLY its remaining items, under the same tag, and the
    killed attempt never wrote its manifest. Summing manifests then loses every match the first attempt
    excluded (final-review finding, combined-cycle Phase B amendment 2). The reduce counts the exclusion
    MARKERS over the population instead: one per excluded match, however many passes it took."""
    from scripts._item_outcome import ItemExcluded

    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"

    def load_excluding_first(ref):
        if ref is refs[0]:
            raise ItemExcluded("planted: ball off pitch")
        return load(ref)

    # attempt 1: excludes refs[0], then is "killed" before its manifest is written
    D.run_corpus(
        refs[:1],
        load_excluding_first,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    for mf in root.glob("*/manifest_w0.json"):
        mf.unlink()
    # the relaunch: same tag, remaining items only
    D.run_corpus(
        refs[1:],
        load_excluding_first,
        tmp_path / "out",
        prov=_CLEAN_PROV,
        shard_root=root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="w0",
    )
    out = D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")
    assert out["n_excluded"] == 1
    assert out["population"]["excluded"]["excluded"] == 1
