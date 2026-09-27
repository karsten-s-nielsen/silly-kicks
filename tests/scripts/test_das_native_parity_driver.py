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

import _loader_pining as lp
import numpy as np
import pandas as pd
import pytest

from scripts import validate_das_native_parity as D
from silly_kicks.tracking._das_pack import Reason, pack_frames
from tests.tracking._das_golden import load_golden

_CLEAN_PROV = {
    "commit": "0" * 40,
    "dirty": False,
    "tree_state": "clean",
    "platform": "test",
    "machine": "x86_64",
}


def _corpus(*, perturb_ref_game: int | None = None):
    """Two matches from golden scenes S01/S05, each a distinct game_id. Returns (refs, load, stub_ref).

    ``load(ref) -> (provider, match_id, actions, frames)``; ``stub_ref(frames)`` returns the golden
    reference arrays for that game (keys relabeled to the match's game_id). ``perturb_ref_game`` adds a
    +1.0 offset to that game's reference team_das, so a test can prove the Δ is non-vacuous.
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


def _run(tmp_path, monkeypatch, *, perturb_ref_game=None):
    monkeypatch.setattr(D, "_numba_available", lambda: False)  # skip the numba compile in a unit test
    refs, load, stub_ref = _corpus(perturb_ref_game=perturb_ref_game)
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
    # Divergence accounting block (spec 6.8 / 7.2): a count per documented D-* class + exclusion tally.
    div = prov["divergences"]
    assert set(div) == {
        "d_off_frames",
        "d_key_frames",
        "d_ballnan_frames",
        "d_possabsent_frames",
        "excluded_rows",
        "max_abs_das_divergent",
    }
    assert set(div["excluded_rows"]) == {"team", "player"}
    assert set(div["max_abs_das_divergent"]) == {"team", "player"}

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
    # S01/S05 are normal full-team scenes: no D-* divergence fires, so the headline is the whole corpus.
    assert prov["divergences"]["d_off_frames"] == 0
    assert prov["divergences"]["d_key_frames"] == 0
    assert prov["divergences"]["excluded_rows"] == {"team": 0, "player": 0}
    # The quadrature shift (periodic - reference) is a REAL, non-zero corpus figure (ADR-108).
    assert np.isfinite(prov["quadrature_shift_das"]["max"])


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
    shard = D._measure_match(("skillcorner", "1", None, pd.DataFrame()))
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
# Divergence exclusion (spec 6.8 / 7.2): documented D-* frames leave the headline and are counted.
# --------------------------------------------------------------------------------------------------


def _off_frame(n_defenders: int) -> pd.DataFrame:
    """One OK frame: a ball, one attacker (team H), and ``n_defenders`` defenders (team A), dir=+1.

    Fewer than two finite defenders is the D-OFF condition (the reference applies an arbitrary offside
    line; native does not), so ``n_defenders=1`` must flag and ``n_defenders=2`` must not.
    """
    rows = [
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            player_id="ball",
            team_id=None,
            is_ball=True,
            is_goalkeeper=False,
            x=52.5,
            y=34.0,
            vx=0.0,
            vy=0.0,
            team_in_possession="H",
            dir=1.0,
        ),
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            player_id=1,
            team_id="H",
            is_ball=False,
            is_goalkeeper=False,
            x=60.0,
            y=34.0,
            vx=1.0,
            vy=0.0,
            team_in_possession="H",
            dir=1.0,
        ),
    ]
    for j in range(n_defenders):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id=100 + j,
                team_id="A",
                is_ball=False,
                is_goalkeeper=False,
                x=70.0 + j,
                y=34.0,
                vx=-1.0,
                vy=0.0,
                team_in_possession="H",
                dir=1.0,
            )
        )
    return pd.DataFrame(rows)


def test_d_off_per_frame_flags_frames_with_fewer_than_two_defenders():
    one = pack_frames(_off_frame(1), attacking_direction_col="dir")
    two = pack_frames(_off_frame(2), attacking_direction_col="dir")
    assert one.reason.tolist() == [int(Reason.OK)] and two.reason.tolist() == [int(Reason.OK)]
    assert D._d_off_per_frame(one).tolist() == [True], "one finite defender -> D-OFF"
    assert D._d_off_per_frame(two).tolist() == [False], "two finite defenders -> offside is determinable"


def _mk_row(grain: str, **kw) -> dict:
    row: dict[str, object] = dict.fromkeys(D._EMITTED_SHARD_COLUMNS, np.nan)
    row.update(grain=grain, provider="skillcorner", game_id=1)
    row.update(kw)
    return row


def _match_row() -> dict:
    return _mk_row("match", n_scored_frames=1, **{c: 0 for c in D._REASON_COLS.values()})


def _shard(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=D._EMITTED_SHARD_COLUMNS)


def test_reduce_excludes_d_off_rows_from_headline_and_counts_them():
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
            is_d_off=False,
        ),
        # a D-OFF frame with a large reference gap: excluded from the headline, surfaced in the block.
        _mk_row(
            "team",
            period_id=1,
            frame_id=2,
            abs_das=15.0,
            rel_das=1.0,
            abs_as=15.0,
            rel_as=1.0,
            finite_ref=True,
            finite_native=True,
            quad_shift_das=0.0,
            reason=int(Reason.OK),
            is_d_off=True,
        ),
        _match_row(),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["team"]["das"]["abs"]["max"] == pytest.approx(1e-9), "the 15.0 D-OFF gap must not enter the headline"
    assert prov["divergences"]["d_off_frames"] == 1
    assert prov["divergences"]["excluded_rows"]["team"] == 1
    assert prov["divergences"]["max_abs_das_divergent"]["team"] == pytest.approx(15.0), "accounting shows the gap"


def test_reduce_counts_and_excludes_d_key_frames_colliding_across_periods():
    rows = [
        _mk_row(
            "team",
            period_id=1,
            frame_id=5,
            abs_das=9.0,
            rel_das=1.0,
            abs_as=9.0,
            rel_as=1.0,
            finite_ref=True,
            finite_native=True,
            quad_shift_das=0.0,
            reason=int(Reason.OK),
            is_d_off=False,
        ),
        _mk_row(
            "team",
            period_id=2,
            frame_id=5,
            abs_das=9.0,
            rel_das=1.0,
            abs_as=9.0,
            rel_as=1.0,
            finite_ref=True,
            finite_native=True,
            quad_shift_das=0.0,
            reason=int(Reason.OK),
            is_d_off=False,
        ),
        _mk_row(
            "team",
            period_id=1,
            frame_id=6,
            abs_das=1e-9,
            rel_das=1e-9,
            abs_as=1e-9,
            rel_as=1e-9,
            finite_ref=True,
            finite_native=True,
            quad_shift_das=0.0,
            reason=int(Reason.OK),
            is_d_off=False,
        ),
        _match_row(),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["divergences"]["d_key_frames"] == 2, "frame_id 5 under two periods -> both rows are D-KEY"
    assert prov["team"]["das"]["abs"]["max"] == pytest.approx(1e-9), "only the unique-key frame is in the headline"


def test_reduce_headline_finite_mask_excludes_degrade_reason_frames():
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
            is_d_off=False,
        ),
        # native NaNs a NaN-ball frame while the reference recorded a fictional 0.0 (D-BALLNAN).
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
            is_d_off=False,
        ),
        _match_row(),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["finite_mask_mismatches"]["team"] == 0, (
        "the BALL_NAN mismatch is a divergence class, not a headline miss"
    )
    assert prov["divergences"]["d_ballnan_frames"] == 1


def test_load_match_ids_groups_the_list_matches_shape_by_provider():
    spec = [
        {"provider": "skillcorner", "match_id": "111"},
        {"provider": "gradientsports", "match_id": "222"},
        {"provider": "skillcorner", "match_id": "333"},
    ]
    got = D._load_match_ids(spec)
    assert got == {"skillcorner": ["111", "333"], "gradientsports": ["222"]}
