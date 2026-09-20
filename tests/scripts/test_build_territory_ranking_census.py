"""OFFLINE census/gate/ranking tests for build_territory_ranking_census (Task 11, TF-54b).

The census driver decides -- on MEASURED evidence -- whether a per-defender RANKING is licensed on
a corpus (the ADR-099 ranking gate). Its numbers come from a networked corpus pass, but the DECISION
is a pure function of a per-``(defender, game, team)`` metric table + a lineup map, so every gate below
runs on hand-built DataFrames with NO network and NO fitted model.

The gate is tested FROM BOTH SIDES (the CLAUDE.md rule): a below-threshold design must NOT license a
ranking, and a strong-defender design must -- and neither side may pass by the other's mechanism (a
volume-only design cannot buy a license; a defender-null ICC cannot either).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from build_territory_ranking_census import (
    ICC_LOWER_FLOOR,
    MIN_MULTI_TEAM_DEFENDERS,
    MIN_PASSES_FACED,
    POWER_FLOOR,
    _fit_disjoint_models,
    _lineup_team_map,
    build_ranking,
    census_gate,
    tier1_counts,
    tier2_icc,
)

from silly_kicks.expected_passing import PassCompletionModel
from silly_kicks.id_compat import canonical_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.xthreat import ExpectedThreat

_VOLUME_COL = "territory_passes_aimed_into_hull"
_METRIC_COL = "territory_xt_prevented_above_expectation"


# --------------------------------------------------------------------------------------------------
# Synthetic per-(defender, game, team) tables.
# --------------------------------------------------------------------------------------------------
def _crossed_metric_table(*, n_def, n_team, reps, sd_def, sd_team, sd_resid, passes, seed=4, mu=0.0) -> pd.DataFrame:
    """A fully-crossed (defender x team) metric table with KNOWN variance components.

    One row per ``(defender, team, rep)`` -- a defender appears on ``n_team`` distinct teams
    (so every defender is a multi-team defender), each with ``reps`` observations. The metric
    value carries a defender effect + a team effect + residual noise, so a strong ``sd_def``
    yields a detectable defender-share ICC and ``sd_def=0`` yields a null one. ``passes`` fills
    the volume column uniformly (>= MIN_PASSES_FACED so Tier-1 counts every cell as qualifying).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd_def, n_def)
    b = rng.normal(0.0, sd_team, n_team)
    rows = []
    for d in range(n_def):
        for t in range(n_team):
            for _ in range(reps):
                y = mu + a[d] + b[t] + rng.normal(0.0, sd_resid)
                rows.append(
                    {
                        "player_id": f"d{d}",
                        "team_id": f"t{t}",
                        "game_id": f"g{d}_{t}_{rng.integers(0, 10_000)}",
                        _METRIC_COL: y,
                        _VOLUME_COL: passes,
                    }
                )
    return pd.DataFrame(rows)


def _single_team_table(*, n_def, passes, seed=1) -> pd.DataFrame:
    """Every defender on ONE team -- no cross-team observation, so no multi-team defender."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_def):
        rows.append(
            {
                "player_id": f"d{d}",
                "team_id": "t0",
                "game_id": f"g{d}",
                _METRIC_COL: float(rng.normal()),
                _VOLUME_COL: passes,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Tier-1 counting.
# --------------------------------------------------------------------------------------------------
def test_tier1_counts_distinct_defenders_and_multi_team_and_qualifying_cells():
    """Distinct defenders, multi-team defenders (>=2 distinct teams), cells clearing MIN_PASSES_FACED."""
    df = pd.DataFrame(
        {
            "player_id": ["A", "A", "B", "B", "C"],
            "team_id": ["t1", "t2", "t1", "t1", "t1"],
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            # A spans two teams; B stays on one; C stays on one.
            _METRIC_COL: [0.1, 0.2, -0.1, 0.0, 0.3],
            # cell (A,t1) passes = 40 clears; (A,t2)=10 below; (B,t1)=40+40=80 clears; (C,t1)=5 below.
            _VOLUME_COL: [40, 10, 40, 40, 5],
        }
    )
    out = tier1_counts(df, min_passes_faced=30)

    assert out["n_defenders"] == 3  # A, B, C
    assert out["n_multi_team_defenders"] == 1  # only A appears on >=2 distinct teams
    # (defender, team) cells clearing 30 passes: (A,t1)=40 yes, (A,t2)=10 no, (B,t1)=80 yes, (C,t1)=5 no
    assert out["n_qualifying_cells"] == 2


def test_tier1_conserves_defender_cell_counts():
    """Conservation: multi-team defenders <= all defenders; qualifying cells <= all cells."""
    df = _crossed_metric_table(n_def=6, n_team=3, reps=2, sd_def=1.0, sd_team=0.5, sd_resid=1.0, passes=40)
    out = tier1_counts(df, min_passes_faced=MIN_PASSES_FACED)

    assert out["n_defenders"] == 6
    assert out["n_multi_team_defenders"] <= out["n_defenders"]
    assert out["n_qualifying_cells"] <= out["n_cells"]
    assert out["n_cells"] == 6 * 3  # every (defender, team) pair is populated
    assert out["n_multi_team_defenders"] == 6  # every defender spans all three teams here


# --------------------------------------------------------------------------------------------------
# Tier-2 ICC.
# --------------------------------------------------------------------------------------------------
def test_tier2_reports_icc_lo_and_power():
    """tier2_icc returns the point ICC, the bootstrap lower bound, and a power proxy in [0, 1]."""
    df = _crossed_metric_table(n_def=30, n_team=6, reps=4, sd_def=3.0, sd_team=1.0, sd_resid=2.0, passes=40)
    out = tier2_icc(df)

    assert 0.0 <= out["icc"] <= 1.0
    assert out["lo"] <= out["icc"]
    assert 0.0 <= out["power"] <= 1.0


# --------------------------------------------------------------------------------------------------
# The gate, FROM BOTH SIDES.
# --------------------------------------------------------------------------------------------------
def test_gate_refuses_when_too_few_multi_team_defenders_and_does_not_run_tier2():
    """Side A: n_multi_team_defenders < MIN_MULTI_TEAM_DEFENDERS -> not licensed, Tier-2 absent."""
    # A big single-team design: plenty of volume, plenty of defenders, but ZERO multi-team defenders.
    df = _single_team_table(n_def=MIN_MULTI_TEAM_DEFENDERS + 50, passes=MIN_PASSES_FACED + 10)
    t1 = tier1_counts(df, min_passes_faced=MIN_PASSES_FACED)
    assert t1["n_multi_team_defenders"] == 0  # precondition: identifiability precondition fails

    census = {"tier1": t1, "tier2": None}
    verdict = census_gate(census)
    assert verdict["ranking_licensed"] is False
    assert "multi" in verdict["reason"].lower() or "identif" in verdict["reason"].lower()


def test_gate_licenses_a_strong_defender_signal_that_clears_both_tiers():
    """Side B: Tier-1 clears AND a strong defender ICC -> licensed."""
    df = _crossed_metric_table(
        n_def=MIN_MULTI_TEAM_DEFENDERS + 10,
        n_team=6,
        reps=4,
        sd_def=4.0,  # a large defender variance -> detectable ICC
        sd_team=1.0,
        sd_resid=2.0,
        passes=MIN_PASSES_FACED + 10,
    )
    t1 = tier1_counts(df, min_passes_faced=MIN_PASSES_FACED)
    assert t1["n_multi_team_defenders"] >= MIN_MULTI_TEAM_DEFENDERS  # Tier-1 clears

    t2 = tier2_icc(df)
    assert t2["lo"] > ICC_LOWER_FLOOR  # a strong, detectable defender share
    assert t2["power"] >= POWER_FLOOR  # design is powered at ICC_EFFECT_SIZE

    verdict = census_gate({"tier1": t1, "tier2": t2})
    assert verdict["ranking_licensed"] is True


def test_gate_refuses_a_defender_null_design_even_when_tier1_clears():
    """Side C: Tier-1 clears but the defender signal is null -> not licensed.

    A volume-and-multi-team-rich design with sd_def=0 cannot buy a license -- the gate is not
    satisfiable by Tier-1 alone. The null defender ICC is near zero and the design is UNPOWERED at
    ICC_EFFECT_SIZE, so the gate refuses on the ICC-lower-bound OR the power leg (both are the
    "the defender share is not detectable" verdict). Asserting the GATE contract rather than a single
    intermediate is deliberate: a null design's bootstrap `lo` can sit marginally above 0.0 by
    resampling noise (the Task-10 null test carries a `lo <= 0.05` tolerance for exactly this), yet
    the power leg -- ~0 detections of ICC >= 0.05 -- still refuses it.
    """
    df = _crossed_metric_table(
        n_def=MIN_MULTI_TEAM_DEFENDERS + 10,
        n_team=6,
        reps=4,
        sd_def=0.0,  # NO defender signal
        sd_team=1.5,
        sd_resid=3.0,
        passes=MIN_PASSES_FACED + 10,
    )
    t1 = tier1_counts(df, min_passes_faced=MIN_PASSES_FACED)
    assert t1["n_multi_team_defenders"] >= MIN_MULTI_TEAM_DEFENDERS  # Tier-1 clears (identifiable design)

    t2 = tier2_icc(df)
    assert t2["icc"] < 0.05  # the null defender share is near zero
    assert t2["power"] < POWER_FLOOR  # and the design cannot detect ICC_EFFECT_SIZE

    verdict = census_gate({"tier1": t1, "tier2": t2})
    assert verdict["ranking_licensed"] is False
    reason = verdict["reason"].lower()
    assert "icc" in reason or "power" in reason  # refused on the detectability leg, not Tier-1


def test_volume_alone_never_licenses_the_ranking():
    """The identifiability precondition is not purchasable with pass volume.

    A single-team design with unbounded volume still fails: no cross-team observation means the
    defender-vs-team confound is unidentifiable, and the gate refuses before Tier-2 is even run.
    """
    huge_volume = _single_team_table(n_def=MIN_MULTI_TEAM_DEFENDERS + 100, passes=MIN_PASSES_FACED * 100)
    t1 = tier1_counts(huge_volume, min_passes_faced=MIN_PASSES_FACED)
    assert t1["n_qualifying_cells"] >= MIN_MULTI_TEAM_DEFENDERS  # volume is abundant
    assert t1["n_multi_team_defenders"] == 0  # but zero identifying power

    verdict = census_gate({"tier1": t1, "tier2": None})
    assert verdict["ranking_licensed"] is False


# --------------------------------------------------------------------------------------------------
# build_ranking.
# --------------------------------------------------------------------------------------------------
def test_build_ranking_produces_a_table_only_when_licensed():
    """build_ranking returns None when unlicensed, and a ranked table when licensed."""
    df = _crossed_metric_table(
        n_def=MIN_MULTI_TEAM_DEFENDERS + 10, n_team=6, reps=4, sd_def=4.0, sd_team=1.0, sd_resid=2.0, passes=40
    )

    unlicensed = build_ranking(df, licensed=False, min_passes_faced=MIN_PASSES_FACED)
    assert unlicensed is None

    licensed = build_ranking(df, licensed=True, min_passes_faced=MIN_PASSES_FACED)
    assert licensed is not None
    assert not licensed.empty
    assert _METRIC_COL in licensed.columns
    # Ranked descending by the headline metric (best defender first).
    vals = pd.to_numeric(licensed[_METRIC_COL], errors="coerce").to_numpy()
    assert np.all(np.diff(vals) <= 1e-9)


def test_build_ranking_filters_below_volume_defenders():
    """A defender below the adequate pass-faced volume is excluded from the licensed ranking."""
    df = pd.DataFrame(
        {
            "player_id": ["hi", "hi", "lo"],
            "team_id": ["t1", "t2", "t1"],
            "game_id": ["g1", "g2", "g3"],
            _METRIC_COL: [0.5, 0.4, 0.9],
            # 'hi' totals 60 passes (clears 30); 'lo' totals 5 (below).
            _VOLUME_COL: [30, 30, 5],
        }
    )
    ranked = build_ranking(df, licensed=True, min_passes_faced=30)
    assert ranked is not None
    assert set(ranked["player_id"]) == {"hi"}  # 'lo' filtered out despite the higher metric value


# --------------------------------------------------------------------------------------------------
# _lineup_team_map: robust to ALL statsbombpy roster shapes + defensive on non-dict entries -- and it
# MUST MAP the players, not skip them (a skip-only "fix" reads as a match with zero defenders, which
# is worse than a crash because it looks like real data).
#
# `sb.lineups(fmt="dict")` returns `{top_key: roster}`, and the roster VARIES across statsbombpy
# versions / open-data competitions:
#
#   (1) The CONFIRMED real shape (verified on the live corpus, match 3879673): the top-level dict is
#       keyed by team_id(int) and each roster is a WRAPPER dict
#       `{"team_id":..., "team_name":..., "lineup":[player_dict, ...]}`. The players live in
#       `roster["lineup"]`; the team id is `roster["team_id"]` (or the top-level key). The prior
#       `for player in _values(roster)` iterated `[team_id(int), team_name(str), lineup(list)]` and its
#       `isinstance(player, dict)` skip dropped ALL THREE -> ZERO players mapped (silent data loss).
#   (2) The list form (WC2022, which the probe used): `{team_name: [player_dict, ...]}`. Players are the
#       list; team id is each player dict's own `team_id`.
#   (3) The pid-keyed form: `{team_name: {player_id: player_dict}}`. Players are `roster.values()`; team
#       id is each player dict's own `team_id`.
#
# The map must reach the real PLAYER DICTS in every shape, MAP them to their team id, and never crash on
# a stray non-dict entry. The shape-(1) test below asserts the players ARE MAPPED (exact contents), so
# it FAILS against a skip-only implementation.
# --------------------------------------------------------------------------------------------------


def _install_fake_lineups(monkeypatch, lineups_by_match: dict):
    """Inject a fake `statsbombpy` module so `_lineup_team_map`'s function-local import resolves
    (statsbombpy is a scripts-only network dep, not installed in CI). `sb.lineups(match_id, fmt=...)`
    returns the stubbed per-match `{top_key: roster}` payload."""
    import sys
    import types

    fake_sb = types.SimpleNamespace(lineups=lambda match_id, fmt="dict": lineups_by_match[int(match_id)])
    fake_mod = types.ModuleType("statsbombpy")
    fake_mod.sb = fake_sb  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "statsbombpy", fake_mod)


def test_lineup_team_map_team_id_keyed_lineup_wrapper_shape(monkeypatch):
    """Shape (1): the CONFIRMED real corpus shape -- top-level `{team_id(int): {"team_id","team_name",
    "lineup":[player_dict,...]}}`. EVERY player must map to its team id (a NON-EMPTY, EXACT map). This
    is the test a skip-only implementation FAILS: iterating `roster.values()` and dropping non-dicts
    would map ZERO players here, so asserting exact non-empty contents (not "did not raise") is what
    exposes the silent data loss."""
    _install_fake_lineups(
        monkeypatch,
        {
            # top-level keyed by team_id (int); roster is the "lineup"-wrapper dict.
            777: {
                230: {
                    "team_id": 230,
                    "team_name": "Team A",
                    "lineup": [
                        {"player_id": 1, "team_id": 230},
                        {"player_id": 2, "team_id": 230},
                    ],
                },
                228: {
                    "team_id": 228,
                    "team_name": "Team B",
                    "lineup": [
                        {"player_id": 3, "team_id": 228},
                    ],
                },
            }
        },
    )
    out = _lineup_team_map("777")
    # Non-empty AND exact: every player is mapped to the correct raw team_id (a skip-only fix -> {}).
    assert out == {canonical_id(1): 230, canonical_id(2): 230, canonical_id(3): 228}
    assert out  # explicit: the map is NOT empty (the skip-only-implementation defect)


def test_lineup_team_map_lineup_wrapper_missing_player_team_id_falls_back_to_roster(monkeypatch):
    """Shape (1) fallback: when a wrapped player dict lacks its own `team_id`, the roster-level
    `team_id` (and, absent that, the top-level team_id key) supplies it -- so the player still maps."""
    _install_fake_lineups(
        monkeypatch,
        {
            888: {
                # roster dict carries team_id; players do not -> fall back to roster["team_id"].
                230: {
                    "team_id": 230,
                    "team_name": "Team A",
                    "lineup": [{"player_id": 1}, {"player_id": 2}],
                },
                # roster dict LACKS team_id; the top-level key IS the team id int -> fall back to it.
                228: {
                    "team_name": "Team B",
                    "lineup": [{"player_id": 3}],
                },
            }
        },
    )
    out = _lineup_team_map("888")
    assert out == {canonical_id(1): 230, canonical_id(2): 230, canonical_id(3): 228}


def test_lineup_team_map_list_of_dicts_roster_shape(monkeypatch):
    """Shape (2): a `list[player_dict]` roster (the WC2022 form the probe used) maps via each player's
    own `team_id` (the top key is a team NAME, so no roster-level id to fall back to)."""
    _install_fake_lineups(
        monkeypatch,
        {
            111: {
                "Team A": [
                    {"player_id": 1, "team_id": 900},
                    {"player_id": 2, "team_id": 900},
                ],
                "Team B": [
                    {"player_id": 3, "team_id": 901},
                ],
            }
        },
    )
    out = _lineup_team_map("111")
    assert out == {canonical_id(1): 900, canonical_id(2): 900, canonical_id(3): 901}


def test_lineup_team_map_dict_keyed_roster_shape(monkeypatch):
    """Shape (3): a `{player_id: player_dict}` roster maps correctly -- iterate the player DICTS
    (values), never the int player_id keys; team id comes from each player dict's own `team_id`."""
    _install_fake_lineups(
        monkeypatch,
        {
            222: {
                "Team A": {
                    1: {"player_id": 1, "team_id": 900},
                    2: {"player_id": 2, "team_id": 900},
                },
                "Team B": {
                    3: {"player_id": 3, "team_id": 901},
                },
            }
        },
    )
    out = _lineup_team_map("222")
    assert out == {canonical_id(1): 900, canonical_id(2): 900, canonical_id(3): 901}


def test_lineup_team_map_skips_non_dict_entries_without_crashing(monkeypatch):
    """A stray non-dict entry INSIDE a resolved players iterable (an int / None) is skipped, no crash,
    and the OTHER real players still map -- exercised across all three roster shapes."""
    _install_fake_lineups(
        monkeypatch,
        {
            333: {
                # shape (1): a stray non-dict inside a "lineup" list.
                230: {
                    "team_id": 230,
                    "team_name": "Team A",
                    "lineup": [
                        {"player_id": 1, "team_id": 230},
                        7,  # stray non-dict entry
                        None,
                        {"player_id": 2, "team_id": 230},
                    ],
                },
                # shape (2): a bare list with a stray non-dict.
                "Team B": [
                    {"player_id": 3, "team_id": 901},
                    "junk",  # stray non-dict entry
                ],
                # shape (3): a pid-keyed roster whose values include a stray non-dict.
                "Team C": {
                    4: {"player_id": 4, "team_id": 902},
                    99: 99,  # stray non-dict value
                },
            }
        },
    )
    out = _lineup_team_map("333")
    assert out == {
        canonical_id(1): 230,
        canonical_id(2): 230,
        canonical_id(3): 901,
        canonical_id(4): 902,
    }


# --------------------------------------------------------------------------------------------------
# Broad multi-competition corpus wiring (REVIVAL delta): the census MUST run over the FULL public
# open-data manifest with ONE leakage-disjoint fit pooled across the whole corpus. A crossed
# defender+team ICC needs defenders observed on >=2 teams, which only exist ACROSS competitions --
# so a single-competition census can only ever return "not licensed" (n_multi_team_defenders=0).
#
# These pin the corpus wiring OFFLINE (monkeypatched manifest stub + tiny fixture + a fake for_each);
# the numerics / gate are covered by the pure-core tests above.
# --------------------------------------------------------------------------------------------------


def _fake_match(provider, match_id):
    """A minimal 5-tuple (provider, match_id, actions, frames, home_team_id). game_id == match_id so
    the shard the fake for_each writes carries a distinct, competition-tagged game."""
    actions = pd.DataFrame({"game_id": [match_id], "player_id": [1], "player_name": ["X"]})
    return (provider, match_id, actions, pd.DataFrame(), 10)


def _install_broad_corpus_stubs(monkeypatch, tmp_path, *, competitions, matches_per_competition=2):
    """Stub the manifest, the loader, the clean-tree guard, and for_each; return the capture handles.

    Returns ``(seen_keys, seen_tokens, loaded_competitions)`` -- the SCORED keys for_each iterated, the
    shard-generation token, and the (competition, season) tuples the loader was called for (so a test
    can prove EVERY competition was walked, independent of how the fit/score split lands). The fake
    for_each WRITES one real parquet shard per item (from the driver's own `work`) so the reduce
    (tier1 -> gate -> census.json) runs to completion off the network, pooling across every match.
    """
    import scripts._driver as driver_mod
    import scripts._provenance as prov_mod
    import scripts._sb_open_data as sbmod

    monkeypatch.setattr(sbmod, "all_open_competitions", lambda: list(competitions))
    monkeypatch.setattr(sbmod, "assert_statsbomb_open_data_mode", lambda: None)

    loaded_competitions: list[tuple[int, int]] = []

    def _fake_loader(*, competition_id, season_id, match_ids=None, max_matches=None, preserve_native=()):
        loaded_competitions.append((competition_id, season_id))
        for j in range(matches_per_competition):
            yield _fake_match("statsbomb", f"m-{competition_id}-{season_id}-{j}")

    monkeypatch.setattr(sbmod, "load_open_data_matches", _fake_loader)

    # Fit is stubbed (a real xt fit needs full SPADL columns); this test pins the CORPUS WIRING.
    monkeypatch.setattr(
        "build_territory_ranking_census._fit_disjoint_models",
        lambda fit_actions: ("XT", "CM"),
    )

    # Score each match into a tiny per-(defender, game, team) shard so the reduce has data to pool.
    def _fake_score(item, *, xt, completion_model):
        _provider, match_id, _actions, _frames, _home = item
        return pd.DataFrame(
            {
                "game_id": [str(match_id)],
                "player_id": [f"d-{match_id}"],
                "team_id": ["t0"],
                _METRIC_COL: [0.5],
                _VOLUME_COL: [MIN_PASSES_FACED + 10],
            }
        )

    monkeypatch.setattr("build_territory_ranking_census._score_match", _fake_score)

    monkeypatch.setattr(prov_mod, "git_provenance", lambda: {"commit": "0" * 40, "dirty": False, "tree_state": "clean"})
    monkeypatch.setattr(prov_mod, "require_clean_tree", lambda prov, allow_dirty=False: prov)

    seen_keys: list[str] = []
    seen_tokens: dict = {}

    class _FakeRes:
        def __init__(self, gen_dir):
            self.shard_dir = gen_dir

        def manifest(self):
            return {"generation": "fake", "n_shards": len(seen_keys)}

    def _fake_for_each(items, *, key, work, shard_root, token_inputs, label):
        gen_dir = shard_root / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        seen_tokens.update(token_inputs)
        for it in items:
            k = key(it)
            seen_keys.append(k)
            frame = work(it)
            frame.to_parquet(gen_dir / f"{k}.parquet", index=False)
        return _FakeRes(gen_dir)

    monkeypatch.setattr(driver_mod, "for_each", _fake_for_each)
    return seen_keys, seen_tokens, loaded_competitions


def test_all_competitions_pools_across_every_competition(monkeypatch, tmp_path):
    """--all-competitions draws from all_open_competitions() (a 2-tuple stub) and chains EVERY
    (comp, season)'s matches into ONE census pooled across the whole corpus. A regression to a single
    competition would load only one tuple's matches, and the census would see the wrong game count."""
    import json

    import build_territory_ranking_census as census

    competitions = [(43, 106), (11, 90)]
    seen_keys, _seen_tokens, loaded = _install_broad_corpus_stubs(
        monkeypatch, tmp_path, competitions=competitions, matches_per_competition=2
    )

    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["prog", "--all-competitions", "--out", str(out_dir), "--allow-dirty"])
    census.main()

    # EVERY competition in the manifest was LOADED (a regression to a single hard-coded competition
    # would call the loader for one tuple only) -- proven at the load seam, independent of how the
    # deterministic fit/score split happens to concentrate competitions.
    assert set(loaded) == {(43, 106), (11, 90)}, loaded

    # The reduce POOLED across the whole corpus into one census.json: all 4 loaded matches are
    # accounted for (2 fit + 2 scored), and one pooled row per scored shard.
    out = json.loads((out_dir / "census.json").read_text(encoding="utf-8"))
    assert out["n_fit_matches"] + out["n_scored_matches"] == 4  # every loaded match accounted for
    assert out["n_scored_matches"] == len(seen_keys)
    # A single-club/national-team corpus is expected NOT to license (the honest limit); what matters
    # here is that the census RAN over the pooled corpus and produced a verdict.
    assert "ranking_licensed" in out
    assert out["census"]["tier1"]["n_rows"] == len(seen_keys)  # one row per scored shard, pooled


def test_all_competitions_token_differs_from_single_competition(monkeypatch, tmp_path):
    """The --all-competitions shard-generation token carries a corpus marker so its generation digest
    DIFFERS from a single-competition run's token (the 4.77.1 stale-shard rule)."""
    import build_territory_ranking_census as census

    # Broad run.
    _keys_b, token_broad, _loaded_b = _install_broad_corpus_stubs(
        monkeypatch, tmp_path, competitions=[(43, 106), (11, 90)]
    )
    monkeypatch.setattr("sys.argv", ["prog", "--all-competitions", "--out", str(tmp_path / "b"), "--allow-dirty"])
    census.main()

    # Single-competition run (default path) -- fresh capture dicts via a fresh stub install.
    _keys_s, token_single, _loaded_s = _install_broad_corpus_stubs(monkeypatch, tmp_path, competitions=[(43, 106)])
    monkeypatch.setattr("sys.argv", ["prog", "--out", str(tmp_path / "s"), "--allow-dirty"])
    census.main()

    assert token_broad.get("source") != token_single.get("source"), (token_broad, token_single)
    # The broad marker is the full-manifest sentinel; the single path keys on nothing broad.
    assert token_broad.get("source") == "open-data-all"
    # The single-competition (default) path token is BYTE-IDENTICAL to the historical one: NO `source`
    # key at all, so its generation directory is unchanged (a stale/new-generation regression check).
    from build_territory_ranking_census import _SHARD_SCHEMA_VERSION

    assert "source" not in token_single
    assert token_single == {
        "metric": "territory_ranking_census",
        "schema": _SHARD_SCHEMA_VERSION,
        "xt": "leakage_disjoint_fit",
        "fit_fraction": 0.5,
    }


def test_competitions_json_narrows_and_keys_the_token_disjointly(monkeypatch, tmp_path):
    """--competitions-json narrows to a chosen list AND keys the token on it, so a narrowed generation
    disjoins from the full-manifest one and from a differently-narrowed one."""
    import json

    import build_territory_ranking_census as census

    import scripts._sb_open_data as sbmod

    def _must_not_call():
        raise AssertionError("all_open_competitions must not be called with --competitions-json")

    seen_keys, token, _loaded = _install_broad_corpus_stubs(monkeypatch, tmp_path, competitions=[(43, 106)])
    monkeypatch.setattr(sbmod, "all_open_competitions", _must_not_call)  # not consulted when JSON given

    comps_file = tmp_path / "comps.json"
    comps_file.write_text(json.dumps([[7, 27]]), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--out", str(tmp_path / "out"), "--competitions-json", str(comps_file), "--allow-dirty"],
    )
    census.main()

    assert {k.split("-")[1] for k in seen_keys} == {"7"}, seen_keys
    assert token.get("source") != "open-data-all"  # a narrowed generation disjoins from the full one


def test_all_competitions_is_mutually_exclusive_with_competition_id(monkeypatch, tmp_path):
    """Passing both a broad-corpus flag and a single-competition selector is a usage error (ap.error)."""
    import build_territory_ranking_census as census

    _install_broad_corpus_stubs(monkeypatch, tmp_path, competitions=[(43, 106)])

    for extra in (["--competition-id", "43"], ["--season-id", "106"]):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--all-competitions", "--out", str(tmp_path / "x"), "--allow-dirty", *extra],
        )
        with pytest.raises(SystemExit):
            census.main()


def test_competitions_json_is_mutually_exclusive_with_competition_id(monkeypatch, tmp_path):
    """--competitions-json also conflicts with an explicit single-competition selector."""
    import json

    import build_territory_ranking_census as census

    _install_broad_corpus_stubs(monkeypatch, tmp_path, competitions=[(43, 106)])
    comps_file = tmp_path / "comps.json"
    comps_file.write_text(json.dumps([[7, 27]]), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--competitions-json",
            str(comps_file),
            "--competition-id",
            "43",
            "--out",
            str(tmp_path / "x"),
            "--allow-dirty",
        ],
    )
    with pytest.raises(SystemExit):
        census.main()


# --------------------------------------------------------------------------------------------------
# #2 fit-corpus column prune -- byte-identity gate (optimization-audit).
# --------------------------------------------------------------------------------------------------
def _fittable_actions(*, n=240, seed=0) -> pd.DataFrame:
    """A small deterministic SPADL action frame fittable by BOTH xt and PassCompletionModel.

    Passes (mixed success/fail) drive the completion logistic; passes/dribbles/shots across the
    pitch drive the xt transition + scoring/move probabilities. Coordinates + results are seeded so
    the fit is reproducible; quality is irrelevant -- the gate compares two fits of the SAME data.
    """
    rng = np.random.default_rng(seed)
    _T = spadlconfig.actiontype_id
    _R = spadlconfig.result_id
    types = rng.choice([_T["pass"], _T["dribble"], _T["cross"], _T["shot"]], size=n, p=[0.6, 0.25, 0.1, 0.05])
    results = rng.choice([_R["success"], _R["fail"]], size=n, p=[0.7, 0.3])
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": rng.integers(1, 3, n),
            "action_id": np.arange(n),
            "time_seconds": np.sort(rng.uniform(0, 3000, n)),
            "team_id": rng.integers(1, 3, n),
            "player_id": rng.integers(1, 23, n),
            "start_x": rng.uniform(0, 105, n),
            "start_y": rng.uniform(0, 68, n),
            "end_x": rng.uniform(0, 105, n),
            "end_y": rng.uniform(0, 68, n),
            "type_id": types,
            "result_id": results,
            "bodypart_id": 0,
        }
    )


def test_fit_disjoint_models_column_prune_is_byte_identical():
    """The SPADL_COLUMNS prune in ``_fit_disjoint_models`` must not change the fitted xt or
    completion model: the pruned fit is byte-identical to a full-frame fit, proving the prune keeps
    every column both fits read AND that the dropped non-canonical extras never affected the fit.
    """
    acts = _fittable_actions().assign(
        preserve_native_foo=1.0,  # non-SPADL extras that the prune must drop
        enriched_start_x=lambda d: d["start_x"] + 1.0,
        tracking_join_col="z",
    )
    xt_pruned, cm_pruned = _fit_disjoint_models([acts])
    xt_full = ExpectedThreat().fit(acts)
    cm_full = PassCompletionModel().fit(acts)

    # Bind fitted attributes to locals BEFORE the array_equal calls: pyright drops member-access
    # narrowing (xt.xT / cm._coef are ndarray|None) after any intervening call, but keeps LOCAL
    # narrowing across calls. All are non-None post-fit.
    xtp, xtf = xt_pruned.xT, xt_full.xT
    tmp, tmf = xt_pruned.transition_matrix, xt_full.transition_matrix
    coefp, coeff = cm_pruned._coef, cm_full._coef
    meanp, meanf = cm_pruned._mean, cm_full._mean
    scalep, scalef = cm_pruned._scale, cm_full._scale
    assert xtp is not None and xtf is not None and tmp is not None and tmf is not None  # fitted
    assert coefp is not None and coeff is not None and meanp is not None and meanf is not None
    assert scalep is not None and scalef is not None
    assert np.array_equal(xtp, xtf)
    assert np.array_equal(tmp, tmf)
    assert np.array_equal(coefp, coeff)
    assert cm_pruned._intercept == cm_full._intercept
    assert np.array_equal(meanp, meanf)
    assert np.array_equal(scalep, scalef)
