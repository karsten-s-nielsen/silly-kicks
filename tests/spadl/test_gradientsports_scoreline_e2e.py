"""Owner-gated e2e: validate the GS goal-capture against the real WC2022 catalog.

Runs only where the pining owner token + data are reachable (public CI skips). This is the ONLY
real-coordinate validation of the Component-1 own-goal geometry tripwire — the synthetic unit tests
are circular on hand-placed coordinates. See docs/superpowers/plans/2026-06-04-gs-goal-capture.md.
"""

import importlib.util
import json
import os
import tempfile
import warnings
from pathlib import Path

import pytest

# The 3 confirmed own goals: match -> (conceding teamId, OG scorer playerId = rebounderPlayerId).
KNOWN_OWN_GOALS = {"10503": (364, 11856), "3853": (374, 4002), "3855": (368, 4602)}

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="owner-tier Gradient Sports data (PINING_FOR_THE_DATA_TOKEN)"),
]


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_nonevent_filter_drops_disallowed_goal_g3853():
    # The decisive over-count case (round-2 #8/D): g3853 = CAN 1-2 MAR (3 goals). It carries FOUR raw
    # shotOutcome=="G" events — a 2nd En-Nesyri @47:17 is nonEvent=True (VAR-disallowed). The void
    # filter must drop exactly that one: 4 raw -> 3 real, matching the official 3-goal scoreline.
    L = _load_loader()
    tok, base = L._resolve_token(None), L._base_url()
    with tempfile.TemporaryDirectory() as tmp:
        p = L._download_to_temp("gradientsports", "3853", "events", tok, base, Path(tmp))
        events = json.load(open(p, encoding="utf-8"))
    g = [ev for ev in events if (ev.get("possessionEvents") or {}).get("shotOutcomeType") == "G"]
    raw = len(g)
    real = sum(1 for ev in g if not (ev.get("possessionEvents") or {}).get("nonEvent", False))
    assert raw == 4, f"g3853: expected 4 raw shotOutcome-G events, got {raw}"
    assert real == 3, f"g3853: expected 3 real (nonEvent=False) goals (official CAN 1-2 MAR), got {real}"


def test_real_own_goals_captured_through_converter_no_tripwire_warn():
    # The ONLY real-coordinate validation of the tripwire inequality. A backwards inequality would
    # revert all 3 real own goals (keeper_save/fail) + WARN; filter to the tripwire message so
    # unrelated UserWarnings (e.g. ET filtering) don't false-fail.
    from silly_kicks.spadl import config as spadlconfig

    L = _load_loader()
    for mid, (team, scorer) in KNOWN_OWN_GOALS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _prov, _m, actions, _frames, _home = next(
                iter(
                    L.load_matches(
                        providers=["gradientsports"],
                        match_ids={"gradientsports": [mid]},
                        tracking_limit=1,
                    )
                )
            )
        tripped = [w for w in caught if "own-goal" in str(w.message) and "attacking half" in str(w.message)]
        assert not tripped, (
            f"g{mid}: tripwire reverted a real own goal — inequality likely backwards: "
            f"{[str(w.message) for w in tripped]}"
        )
        og = actions[(actions["result_id"] == spadlconfig.result_id["owngoal"]) & (actions["team_id"] == team)]
        assert len(og) == 1, f"g{mid}: expected exactly 1 owngoal for team {team}, got {len(og)}"
        assert (og["type_id"] == spadlconfig.actiontype_id["bad_touch"]).all()
        assert (og["player_id"] == scorer).all()


def test_dribble_ends_derived_on_real_wc2022():
    # PR-S116: real WC2022 GS dribbles must no longer be 100% zero-displacement (the
    # pre-fix live corpus measured 850/850 start==end). Residual placeholders are
    # period-last carries only, so >90% must now carry a real derived end.
    from silly_kicks.spadl import config as spadlconfig

    L = _load_loader()
    _prov, _m, actions, _frames, _home = next(
        iter(
            L.load_matches(
                providers=["gradientsports"],
                match_ids={"gradientsports": ["10503"]},
                tracking_limit=1,
            )
        )
    )
    dribbles = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
    if len(dribbles) == 0:
        pytest.skip("match has no dribbles")
    moved = (dribbles["end_x"] != dribbles["start_x"]) | (dribbles["end_y"] != dribbles["start_y"])
    assert moved.mean() > 0.9, f"only {moved.mean():.0%} of {len(dribbles)} GS dribbles derived an end"
