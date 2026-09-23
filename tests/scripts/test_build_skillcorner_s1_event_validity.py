"""Task 0 driver tests (spec 2026-09-22 section 8): the SkillCorner S1 event-validity measurement.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path, so the driver and the
`_fake_corpus` / `_loader_pining` helpers import bare.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import _fake_corpus as fc
import build_skillcorner_s1_event_validity as drv
import pandas as pd
import pytest
from _driver import join_key, read_exclusion

from silly_kicks.spadl import config as spadlconfig


def _anchored_row(*, game_id, team_id, period_id, type_name, start_x):
    return {
        "game_id": str(game_id),
        "team_id": team_id,
        "period_id": period_id,
        "type_id": spadlconfig.actiontype_id[type_name],
        "start_x": float(start_x),
        "start_y": 34.0,
    }


# --- Step 1: anchored-consistency statistic ------------------------------------------------


def test_own_half_anchored_consistent_below_halfway():
    """goalkick / clearance are consistent iff start_x < 52.5 (action-LTR own half)."""
    actions = pd.DataFrame(
        [
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="goalkick", start_x=10.0),
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="clearance", start_x=90.0),
        ]
    )
    anchored = drv.anchored_consistency(actions)
    consistent = dict(zip(anchored["type_name"], anchored["consistent"], strict=True))
    assert consistent == {"goalkick": True, "clearance": False}


def test_attacking_half_anchored_consistent_above_halfway():
    """shot / corner_short / corner_crossed are consistent iff start_x > 52.5."""
    actions = pd.DataFrame(
        [
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="shot", start_x=95.0),
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="corner_short", start_x=5.0),
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="corner_crossed", start_x=104.0),
        ]
    )
    anchored = drv.anchored_consistency(actions)
    consistent = dict(zip(anchored["type_name"], anchored["consistent"], strict=True))
    assert consistent == {"shot": True, "corner_short": False, "corner_crossed": True}


def test_non_anchored_types_are_dropped():
    """A pass / dribble carries no half-of-the-pitch law, so it is not scored."""
    actions = pd.DataFrame(
        [
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="pass", start_x=10.0),
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="dribble", start_x=90.0),
            _anchored_row(game_id="m", team_id="A", period_id=1, type_name="goalkick", start_x=10.0),
        ]
    )
    anchored = drv.anchored_consistency(actions)
    assert list(anchored["type_name"]) == ["goalkick"]


def test_cell_counts_group_by_team_and_period_and_score_at_five():
    """A cell is (game, team, period); n = anchored actions, k = consistent; scored iff n >= 5."""
    rows = [_anchored_row(game_id="m", team_id="A", period_id=1, type_name="goalkick", start_x=10.0) for _ in range(4)]
    rows += [_anchored_row(game_id="m", team_id="A", period_id=1, type_name="clearance", start_x=95.0)]  # wrong side
    rows += [_anchored_row(game_id="m", team_id="B", period_id=1, type_name="shot", start_x=95.0) for _ in range(3)]
    anchored = drv.anchored_consistency(pd.DataFrame(rows))
    cells = {(r["team_id"], r["period_id"]): r for r in drv.cell_counts(anchored).to_dict("records")}
    a = cells[("A", 1)]
    assert (a["n"], a["k"], a["scored"]) == (5, 4, True)
    b = cells[("B", 1)]
    assert (b["n"], b["k"], b["scored"]) == (3, 3, False)


# --- Step 2: reversal + boundary statistics (both sides of the band) ------------------------


def _cell(type_name, start_xs):
    rows = [_anchored_row(game_id="m", team_id="A", period_id=1, type_name=type_name, start_x=x) for x in start_xs]
    return drv.anchored_consistency(pd.DataFrame(rows))


def test_reversed_cell_fires_both_conjuncts_its_twin_does_not():
    p_t = {"goalkick": 0.9}
    good = _cell("goalkick", [10.0] * 6)  # all own-half consistent: k=6
    mirror = _cell("goalkick", [95.0] * 6)  # reflected across 52.5: k=0
    assert drv.cell_is_reversed(mirror, p_t=p_t, m=1) is True
    assert drv.cell_is_reversed(good, p_t=p_t, m=1) is False


def test_reversal_needs_significance_not_just_minority():
    """k/n < 0.5 alone is not enough: a tiny, weak minority must not be called reversed."""
    p_t = {"goalkick": 0.55}
    cell = _cell("goalkick", [95.0, 95.0, 95.0, 10.0, 10.0])  # k=2, n=5, k/n=0.4 but weak
    # P(K<=2 | 5 Bernoulli(0.55)) is not < 0.05, so not reversed despite the minority.
    assert drv.cell_is_reversed(cell, p_t=p_t, m=1) is False


def test_cell_null_cdf_matches_brute_force_enumeration():
    import itertools

    cell = drv.anchored_consistency(
        pd.DataFrame(
            [
                _anchored_row(game_id="m", team_id="A", period_id=1, type_name="goalkick", start_x=10.0),
                _anchored_row(game_id="m", team_id="A", period_id=1, type_name="goalkick", start_x=95.0),
                _anchored_row(game_id="m", team_id="A", period_id=1, type_name="shot", start_x=95.0),
            ]
        )
    )
    p_t = {"goalkick": 0.9, "shot": 0.6}
    probs = [0.9, 0.9, 0.6]
    k = int(cell["consistent"].sum())
    brute = 0.0
    for combo in itertools.product([0, 1], repeat=3):
        if sum(combo) <= k:
            prob = 1.0
            for bit, p in zip(combo, probs, strict=True):
                prob *= p if bit else (1.0 - p)
            brute += prob
    assert drv.cell_null_cdf(cell, p_t=p_t) == pytest.approx(brute)


def test_boundary_pileup_flags_a_clamped_match_not_its_clean_twin():
    passing_b = [0.0, 0.001, 0.002, 0.0]
    tau = drv.boundary_tau(passing_b)
    assert tau == pytest.approx(0.05)  # 3 * q99.9 is tiny, so the 0.05 floor wins

    def _open_play(start_xs, start_ys):
        rows = []
        for x, y in zip(start_xs, start_ys, strict=True):
            r = _anchored_row(game_id="m", team_id="A", period_id=1, type_name="pass", start_x=x)
            r["start_y"] = float(y)
            rows.append(r)
        return pd.DataFrame(rows)

    clamped = _open_play([0.005] * 6 + [50.0] * 4, [34.0] * 10)  # 6/10 on the x=0 edge
    clean = _open_play([50.0] * 10, [34.0] * 10)
    n_c, b_c = drv.boundary_counts(clamped)
    assert (n_c, b_c) == (10, 6)
    assert drv.match_is_boundary_flagged(6 / 10, tau) is True
    n_k, b_k = drv.boundary_counts(clean)
    assert drv.match_is_boundary_flagged(b_k / n_k, tau) is False


# --- Step 3: per-match verdict precedence ---------------------------------------------------


def _match_anchored(cells_spec):
    """cells_spec: list of (team, period, n_consistent_start_x, n_inconsistent_start_x) goalkick counts."""
    rows = []
    for team, period, n_ok, n_bad in cells_spec:
        for _ in range(n_ok):
            rows.append(_anchored_row(game_id="m", team_id=team, period_id=period, type_name="goalkick", start_x=10.0))
        for _ in range(n_bad):
            rows.append(_anchored_row(game_id="m", team_id=team, period_id=period, type_name="goalkick", start_x=95.0))
    return drv.anchored_consistency(pd.DataFrame(rows))


_FULL_REG = [("A", 1, 5, 0), ("A", 2, 5, 0), ("B", 1, 5, 0), ("B", 2, 5, 0)]
_PT = {"goalkick": 0.9}


def test_verdict_sound_when_all_regulation_cells_scored_and_clean():
    m = _match_anchored(_FULL_REG)
    assert drv.match_event_verdict(m, boundary_fraction=0.0, p_t=_PT, tau=0.05, m=1) == "sound"


def test_verdict_reversed_takes_precedence_over_everything():
    # A reversed cell (6 goalkicks all on the wrong side) plus a boundary pile-up: reversed wins.
    m = _match_anchored([("A", 1, 0, 6), ("A", 2, 5, 0), ("B", 1, 5, 0), ("B", 2, 5, 0)])
    assert drv.match_event_verdict(m, boundary_fraction=0.9, p_t=_PT, tau=0.05, m=1) == "reversed"


def test_verdict_boundary_beats_insufficient():
    # No reversed cell; a regulation cell is unscored (would be insufficient) but boundary fires first.
    m = _match_anchored([("A", 1, 5, 0), ("A", 2, 2, 0), ("B", 1, 5, 0), ("B", 2, 5, 0)])
    assert drv.match_event_verdict(m, boundary_fraction=0.9, p_t=_PT, tau=0.05, m=1) == "boundary_pileup"


def test_verdict_insufficient_when_a_regulation_cell_is_unscored():
    m = _match_anchored([("A", 1, 5, 0), ("A", 2, 2, 0), ("B", 1, 5, 0), ("B", 2, 5, 0)])
    assert drv.match_event_verdict(m, boundary_fraction=0.0, p_t=_PT, tau=0.05, m=1) == "insufficient"


def test_unscored_extra_time_cell_is_reported_not_insufficient():
    m = _match_anchored([*_FULL_REG, ("A", 3, 2, 0)])  # ET cell unscored
    assert drv.match_event_verdict(m, boundary_fraction=0.0, p_t=_PT, tau=0.05, m=1) == "sound"


# --- Step 4: two for_each passes over the same ref list -------------------------------------


def _sc_actions(match_id):
    """A SkillCorner-shaped events frame: a full regulation set of anchored goalkicks per team."""
    rows = []
    for team in ("A", "B"):
        for period in (1, 2):
            for _ in range(5):
                rows.append(
                    _anchored_row(game_id=match_id, team_id=team, period_id=period, type_name="goalkick", start_x=10.0)
                )
    # a few open-play passes, none on the boundary
    for _ in range(4):
        r = _anchored_row(game_id=match_id, team_id="A", period_id=1, type_name="pass", start_x=50.0)
        rows.append(r)
    return pd.DataFrame(rows)


class _EventsOkTrackingFails:
    """load_match stand-in: events_only serves actions; a full (tracking) load fails for named keys."""

    def __init__(self, loaded, *, tracking_fail=(), s1_exclude=None):
        self.loaded = dict(loaded)
        self.tracking_fail = set(tracking_fail)
        self.s1_exclude = dict(s1_exclude or {})
        self.calls = []

    def __call__(self, ref, *, events_only, cache_dir=None, **kw):
        self.calls.append((ref.key, events_only))
        if not events_only:
            if ref.key in self.tracking_fail:
                raise RuntimeError(f"tracking load failed for {ref.key}")
            if ref.key in self.s1_exclude:
                pr, br = self.s1_exclude[ref.key]
                import _loader_pining as lp

                raise lp.MatchExcluded(
                    "S1 geometry gate", details={"player_off_pitch_rate": pr, "ball_off_pitch_rate": br}
                )
        return self.loaded[ref.key]


def _install(monkeypatch, refs, loader):
    monkeypatch.setattr(drv, "list_match_refs", lambda **_kw: list(refs))
    monkeypatch.setattr(drv, "load_match", loader)


def test_events_pass_writes_one_shard_per_match_and_resumes(tmp_path, monkeypatch):
    refs = [fc.make_ref("skillcorner", "1"), fc.make_ref("skillcorner", "2")]
    loaded = {r.key: fc.make_loaded("skillcorner", r.match_id, actions=_sc_actions(r.match_id)) for r in refs}
    loader = _EventsOkTrackingFails(loaded)
    _install(monkeypatch, refs, loader)
    token = {"schema": "s1-events-1"}
    res = drv._events_pass(refs, shard_root=tmp_path / "events", cache_dir=None, token_inputs=token)
    assert set(res.shard_keys) == {join_key(("skillcorner", "1")), join_key(("skillcorner", "2"))}
    shard = pd.read_parquet(next((res.shard_dir).glob("*.parquet")))
    assert set(shard.columns) == {"game_id", "team_id", "period_id", "kind", "type_name", "n", "k"}
    assert (shard["kind"] == "open_play").sum() == 1  # one boundary row per match
    n_calls_first = len(loader.calls)
    drv._events_pass(refs, shard_root=tmp_path / "events", cache_dir=None, token_inputs=token)
    assert len(loader.calls) == n_calls_first  # resume-before-load: nothing re-loaded


def test_tracking_pass_shards_pass_marks_exclusion_and_records_failure(tmp_path, monkeypatch):
    refs = [fc.make_ref("skillcorner", str(i)) for i in (1, 2, 3)]
    loaded = {
        r.key: fc.make_loaded(
            "skillcorner",
            r.match_id,
            report=SimpleNamespace(player_off_pitch_rate=0.0001, ball_off_pitch_rate=0.0),
        )
        for r in refs
    }
    loader = _EventsOkTrackingFails(
        loaded,
        tracking_fail={("skillcorner", "3")},
        s1_exclude={("skillcorner", "2"): (0.34, 0.002)},
    )
    _install(monkeypatch, refs, loader)
    token = {"schema": "s1-tracking-1"}
    res = drv._tracking_pass(refs, shard_root=tmp_path / "tracking", cache_dir=None, token_inputs=token)

    k1, k2, k3 = (join_key(("skillcorner", i)) for i in ("1", "2", "3"))
    # match 1: S1-passing -> a shard with its rates
    assert k1 in res.shard_keys
    shard = pd.read_parquet(res.shard_dir / f"{k1}.parquet")
    assert float(shard["player_off_pitch_rate"].iloc[0]) == pytest.approx(0.0001)

    # match 2: S1-excluded -> a marker carrying the rates, NO shard
    assert k2 in res.exclusions
    assert not (res.shard_dir / f"{k2}.parquet").exists()
    marker = read_exclusion(res.shard_dir, ("skillcorner", "2"))
    assert marker is not None
    assert marker["details"]["player_off_pitch_rate"] == pytest.approx(0.34)

    # match 3: tracking load failed -> recorded failure, NO shard, resume retries
    assert k3 in res.failures
    assert not (res.shard_dir / f"{k3}.parquet").exists()
    calls_before = len(loader.calls)
    drv._tracking_pass(refs, shard_root=tmp_path / "tracking", cache_dir=None, token_inputs=token)
    assert (("skillcorner", "3"), False) in loader.calls[calls_before:]  # retried


# --- Step 5: reduce -- refuses on failures, recomputes verdicts, never reads its own artifact -


def _run_passes(tmp_path, refs, loader, monkeypatch):
    _install(monkeypatch, refs, loader)
    ev = drv._events_pass(refs, shard_root=tmp_path / "e", cache_dir=None, token_inputs={"schema": "e"})
    tr = drv._tracking_pass(refs, shard_root=tmp_path / "t", cache_dir=None, token_inputs={"schema": "t"})
    return ev, tr


def _passing_loaded(refs):
    return {
        r.key: fc.make_loaded(
            "skillcorner",
            r.match_id,
            actions=_sc_actions(r.match_id),
            report=SimpleNamespace(player_off_pitch_rate=0.0001, ball_off_pitch_rate=0.0),
        )
        for r in refs
    }


def test_reduce_refuses_on_failures_and_allow_failed_records(tmp_path, monkeypatch):
    refs = [fc.make_ref("skillcorner", str(i)) for i in (1, 2, 3)]
    loader = _EventsOkTrackingFails(_passing_loaded(refs), tracking_fail={("skillcorner", "3")})
    ev, tr = _run_passes(tmp_path, refs, loader, monkeypatch)

    with pytest.raises(RuntimeError, match="tracking pass"):
        drv.reduce_verdicts(ev, tr, allow_failed=False)

    art = drv.reduce_verdicts(ev, tr, allow_failed=True)
    k3 = join_key(("skillcorner", "3"))
    assert art["allowed_failed"] is True
    assert art["matches"][k3]["status"] == "tracking_unloadable"
    assert "error" in art["matches"][k3]
    assert art["matches"][k3]["event_verdict"] == "sound"  # events loaded, so a verdict is still computed


def test_reduce_assigns_status_verdict_and_matching_digest(tmp_path, monkeypatch):
    from _events_admission import listing_digest

    refs = [fc.make_ref("skillcorner", "1"), fc.make_ref("skillcorner", "2")]
    loader = _EventsOkTrackingFails(_passing_loaded(refs), s1_exclude={("skillcorner", "2"): (0.34, 0.002)})
    ev, tr = _run_passes(tmp_path, refs, loader, monkeypatch)

    art = drv.reduce_verdicts(ev, tr)
    m = art["matches"]
    k1, k2 = join_key(("skillcorner", "1")), join_key(("skillcorner", "2"))
    assert m[k1]["status"] == "s1_passed"
    assert m[k2]["status"] == "s1_excluded"
    assert m[k1]["event_verdict"] == "sound"
    assert m[k2]["event_verdict"] == "sound"
    assert m[k2]["player_off_pitch_rate"] == pytest.approx(0.34)
    assert art["digest"] == listing_digest(m)  # the digest admission recomputes on load


def test_producer_recomputes_and_never_consults_verdicts_json(tmp_path, monkeypatch):
    """main() must not read its own verdicts.json, so a stale artifact cannot freeze a verdict, and it
    must never consult the admission layer (spec CDLS-SPEC-27)."""
    from _events_admission import EventsOnlyAdmission

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    k1 = join_key(("skillcorner", "1"))
    # A stale artifact whose verdict is WRONG (reversed). If the producer read it, the fresh run would
    # keep it; the recompute overwrites it with the true 'sound'.
    stale = {"matches": {k1: {"status": "s1_passed", "event_verdict": "reversed"}}, "run_commit": "stale"}
    (out_dir / "verdicts.json").write_text(json.dumps(stale), encoding="utf-8")

    def _boom(*_a, **_k):
        raise AssertionError("producer consulted the admission layer / its own artifact")

    monkeypatch.setattr(EventsOnlyAdmission, "load", classmethod(lambda *a, **k: _boom()))

    refs = [fc.make_ref("skillcorner", "1")]
    loader = _EventsOkTrackingFails(_passing_loaded(refs))
    _install(monkeypatch, refs, loader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--out-dir", str(out_dir), "--shard-root", str(tmp_path / "sh"), "--allow-dirty"],
    )
    drv.main()

    written = json.loads((out_dir / "verdicts.json").read_text(encoding="utf-8"))
    assert written["matches"][k1]["event_verdict"] == "sound"  # recomputed, not the stale 'reversed'
    assert written["run_commit"] != "stale"  # a fresh provenance stamp, not the stale file's


# --- Step 6: provenance / driver hygiene ---------------------------------------------------


def test_driver_source_is_ascii():
    """--help executes main() on parserless scripts elsewhere, and a non-ASCII byte in a driver
    breaks --help on a non-UTF-8 Windows console (the driver ASCII gate)."""
    import pathlib

    src = pathlib.Path(drv.__file__).read_text(encoding="utf-8")
    non_ascii = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
    assert not non_ascii, f"non-ASCII in driver source: {non_ascii[:5]}"
