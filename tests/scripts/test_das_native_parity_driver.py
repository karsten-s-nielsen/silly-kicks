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
