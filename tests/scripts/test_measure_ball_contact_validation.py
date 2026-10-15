"""TF-79: the ball-contact validation driver runs offline on an injected loader and reports metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class _Ref:
    key: str


def _match_frames_and_actions():
    fps = 25.0
    dt = 1.0 / fps
    p_a, p_b = 7, 9
    ax, ay = 60.0, 34.0
    cbx, cby = 62.0, 48.0

    def frow(pid, team, is_ball, x, y, z, fid):
        return dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            time_seconds=round(fid * dt, 3),
            frame_rate=fps,
            player_id=pid,
            team_id=team,
            is_ball=is_ball,
            is_goalkeeper=False,
            x=x,
            y=y,
            z=z,
            speed=1.0,
        )

    rows = []
    for fid in range(30):
        if fid <= 10:
            bx, by, bz = 50.0 + (ax - 50.0) * (fid / 10.0), 34.0, 0.1
        elif fid <= 20:
            f = (fid - 10) / 10.0
            bx, by, bz = ax + (cbx - ax) * f, ay + (cby - ay) * f, 0.1 + 2.3 * f
        else:
            bx, by, bz = cbx, cby, 2.4
        rows.append(frow(pd.NA, pd.NA, True, bx, by, bz, fid))
        rows.append(frow(p_a, "T1", False, ax, ay, 0.0, fid))
        rows.append(frow(p_b, "T2", False, 63.0, 48.0, 0.0, fid))
    frames = pd.DataFrame(rows)
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["team_id"] = frames["team_id"].astype("category")
    for c in ("x", "y", "z"):
        frames[c] = frames[c].astype("float32")

    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "time_seconds": [0.40, 0.80],
            "player_id": pd.Series([7, 9], dtype="Int64"),
            "team_id": pd.Series([1, 2], dtype="int64"),
            "start_x": [60.0, 62.0],
            "start_y": [34.0, 48.0],
            "end_x": [62.0, 62.0],
            "end_y": [48.0, 48.0],
            "type_id": [0, 0],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
        }
    )
    return actions, frames


def test_driver_run_reports_precision_recall_timing(tmp_path):
    import scripts.measure_ball_contact_validation as d

    actions, frames = _match_frames_and_actions()
    refs = [_Ref("idsse_m1")]  # colon-free: shard keys become filenames (Windows forbids ':')

    def load(ref):
        return ("idsse", "m1", actions, frames, None)

    m = d.run(tmp_path, providers=["idsse"], provenance={"commit": "test", "dirty": False}, source=(refs, load))

    o = m["overall"]
    assert {
        "contact_rate_per_min",
        "unresolved_rate",
        "signed_offset_s",
        "agreement_vs_tolerance",
        "confidence_calibration",
        "velocity_change_ms_pct",
    } <= set(o)
    # the two planted touches sit at the action times -> agreement at 1.0 s is non-zero
    assert o["agreement_vs_tolerance"]["agree_within_1.0s"] > 0.0
    assert set(o["agreement_vs_tolerance"]) == {"agree_within_0.2s", "agree_within_0.5s", "agree_within_1.0s"}
    assert "corpus_bound" in m and m["corpus_bound"]["n_matches"] == 1
    assert "per_provider" in m and "idsse" in m["per_provider"]
    # ADR-037 committed-artifact provenance shape: top-level run_commit / run_tree_dirty (the shape
    # tests/scripts/test_artifact_provenance_output.py enforces), NOT a nested provenance.commit.
    assert m["run_commit"] == "test"
    assert m["run_tree_dirty"] is False
    assert (tmp_path / "metrics.json").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    # memo foregrounds PER-PROVIDER (authoritative) and flags the pooled block as mix-dependent
    assert "## Per provider" in report
    assert "### idsse" in report
    assert report.index("## Per provider") < report.index("## Pooled across all providers")
    assert "NOT a detector-quality score" in report
