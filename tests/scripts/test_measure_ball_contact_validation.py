"""TF-79: the ball-contact validation driver runs offline on an injected loader and reports metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class _Ref:
    key: str


@dataclass
class _GsRef:
    """A GS match ref, keyed like ``MatchRef.key``."""

    match_id: str

    @property
    def key(self) -> tuple[str, str]:
        return ("gradientsports", self.match_id)


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


# --------------------------------------------------------------------------- #
# TF-79 v2 GS truth mode (detector spec sections 2, 6 and 7)
# --------------------------------------------------------------------------- #


def test_other_provider_trigger_both_sides():
    import scripts.measure_ball_contact_validation as d

    def m(idsse_agree, sc_agree, sc_unres, idsse_unres=0.1):
        return {
            "per_provider": {
                "idsse": {"agreement_vs_tolerance": {"agree_within_0.5s": idsse_agree}, "unresolved_rate": idsse_unres},
                "skillcorner": {"agreement_vs_tolerance": {"agree_within_0.5s": sc_agree}, "unresolved_rate": sc_unres},
            }
        }

    ref = m(0.50, 0.80, 0.20)
    assert d.other_provider_trigger(ref, m(0.50, 0.80, 0.20, idsse_unres=0.40))["fires"] is False  # SkillCorner only
    assert d.other_provider_trigger(ref, m(0.41, 0.80, 0.20))["fires"] is False  # agreement falls 0.09
    assert d.other_provider_trigger(ref, m(0.39, 0.80, 0.20))["fires"] is True  # agreement falls 0.11
    assert d.other_provider_trigger(ref, m(0.50, 0.80, 0.29))["fires"] is False  # SkillCorner unresolved +0.09
    assert d.other_provider_trigger(ref, m(0.50, 0.80, 0.31))["fires"] is True  # SkillCorner unresolved +0.11


def test_gs_events_json_reads_the_cached_events(tmp_path, monkeypatch):
    import json

    import scripts._loader_pining as lp

    f = tmp_path / "events.json"
    f.write_text(json.dumps([{"gameEventId": 1}]), encoding="utf-8")
    seen = {}

    def fake_download(provider, match_id, artifacts, token, base_url, dest, *, use_cache=False, events_only=False):
        seen.update(provider=provider, events_only=events_only)
        return {"events": f}

    monkeypatch.setattr(lp, "_download_artifacts", fake_download)
    monkeypatch.setattr(lp, "_resolve_token", lambda t: "tok")
    ref = lp.MatchRef("gradientsports", "3816", {})
    assert lp.gradientsports_events_json(ref, cache_dir=tmp_path) == [{"gameEventId": 1}]
    assert seen == {"provider": "gradientsports", "events_only": True}


def test_gs_truth_dev_run_end_to_end(tmp_path):
    # The synthetic match: player 7 touches at 0.4 s, player 9 at 0.8 s. PFF says 7 releases at 0.4 s and 9 receives
    # at 0.8 s, so every grid point scores "right" and the selection takes the stricter corner (5 m/s, 0.8 m).
    import scripts.measure_ball_contact_validation as d
    from tests.tracking.test_ball_contact import _synthetic_match_frames

    # Period 1 only: the touches at frames 10 and 20 (the ball crawls on after B, so v2 sees both changes).
    frames = _synthetic_match_frames(gap_period2=False).query("period_id == 1")
    ev = [
        {"startTime": 0.0, "gameEvents": {"gameEventType": "FIRSTKICKOFF", "period": 1}},
        {
            "startTime": 0.0,
            "endTime": 0.4,
            "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 7},
            "possessionEvents": {"possessionEventType": "PA"},
        },
        {
            "startTime": 0.8,
            "endTime": 1.0,
            "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 9},
            "possessionEvents": {"possessionEventType": "IT"},
        },
    ]

    refs = [_GsRef("3816"), _GsRef("3812")]  # one development match, one held-out match (filtered out)

    def load(ref):
        return ("gradientsports", ref.match_id, None, frames, None), ev

    rep = d.run_gs_truth(tmp_path, provenance={"commit": "test", "dirty": False}, source=(refs, load))
    assert rep["matches"] == ["3816"] and len(rep["grid"]) == 20
    assert rep["chosen"] == {"contact_radius_m": 0.8, "min_velocity_change_ms": 5.0}
    assert rep["chosen_on_grid_edge"] is True
    assert (tmp_path / "truth_dev.json").exists()
    # Scorer spec r7 section 3: both populations in every grid row; player 9 is at the ball when PFF says he receives.
    assert all(g["supported_share"] == 1.0 and g["sup_n"] == g["n"] == 1 for g in rep["grid"])
    assert all(g["sup_right_share"] == g["right_share"] for g in rep["grid"])


def test_gs_truth_selects_on_the_supported_pairs(tmp_path, monkeypatch):
    # Scorer spec r7 section 3: the rule baseline selects with the bar's population.
    import scripts.measure_ball_contact_validation as d
    from tests.tracking.test_ball_contact import _synthetic_match_frames

    frames = _synthetic_match_frames(gap_period2=False).query("period_id == 1")
    ev = [
        {"startTime": 0.0, "gameEvents": {"gameEventType": "FIRSTKICKOFF", "period": 1}},
        {
            "startTime": 0.0,
            "endTime": 0.4,
            "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 7},
            "possessionEvents": {"possessionEventType": "PA"},
        },
        {
            "startTime": 0.8,
            "endTime": 1.0,
            "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 9},
            "possessionEvents": {"possessionEventType": "IT"},
        },
    ]
    seen = []
    real = d.select_point

    def spy(grid, *, prefer, population):
        seen.append(population)
        return real(grid, prefer=prefer, population=population)

    monkeypatch.setattr(d, "select_point", spy)
    from scripts import _driver

    tokens = []
    real_for_each = _driver.for_each

    def for_each_spy(*args, **kw):
        tokens.append(kw.get("token_inputs"))
        return real_for_each(*args, **kw)

    monkeypatch.setattr(_driver, "for_each", for_each_spy)
    d.run_gs_truth(
        tmp_path,
        provenance={"commit": "test", "dirty": False},
        source=([_GsRef("3816")], lambda ref: (("gradientsports", ref.match_id, None, frames, None), ev)),
    )
    assert seen == ["supported"]
    # Review B CD-SB-101 (H2): shards from before the supported counts never resume into this generation.
    assert tokens and tokens[0]["support_radius_m"] == 1.5


def test_other_provider_reports_take_the_frozen_scorer():
    # Scorer spec section 3: the IDSSE and SkillCorner reports run with the development run's frozen scorer and its
    # threshold (threshold-free peaks), so the trigger reads the detector that ships.
    import scripts.measure_ball_contact_validation as d

    p = d.scorer_params("/runs/dev", 0.35)
    assert p is not None
    assert (p.contact_scorer, p.min_contact_probability, p.min_velocity_change_ms) == ("/runs/dev", 0.35, 0.0)
    assert d.scorer_params(None, None) is None


def test_a_scorer_needs_its_threshold():
    from pytest import raises

    import scripts.measure_ball_contact_validation as d

    with raises(ValueError, match="--min-contact-probability"):
        d.scorer_params("/runs/dev", None)


def test_the_harness_token_reads_the_scorer_content(tmp_path):
    import scripts.measure_ball_contact_validation as d

    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    params = d.scorer_params(str(tmp_path), 0.4)
    first = d._scorer_sha(params)
    (tmp_path / "model.json").write_text('{"x": 1}', encoding="utf-8")
    assert d._scorer_sha(params) != first and d._scorer_sha(None) is None
    assert d._token_inputs(params)["contact_scorer_model_sha256"] == d._scorer_sha(params)  # the token reads it
    assert d._token_inputs(None)["contact_scorer_model_sha256"] is None
