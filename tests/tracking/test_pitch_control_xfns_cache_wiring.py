"""TF-7 xfns cache -- completeness gate (mirrors #1's complete-by-construction philosophy).

Every ``add_*`` aggregator that accepts ``pitch_control_cache`` MUST have a correspondingly-named
``*_xfns`` factory that also accepts (and threads) it, so a FUTURE pitch-control-consuming family
cannot be added without the cache seam. The perf guard alone would miss an omitted family silently.
"""

from __future__ import annotations

import inspect

from silly_kicks.tracking import features as F

# Every PC-consuming xfns factory threaded in this PR. Each family's aggregator (add_*) accepts
# pitch_control_cache, so its xfns factory must too.
_PC_CONSUMING_XFNS = [
    "pitch_control_xfns",
    "obso_xfns",
    "cover_shadow_xfns",
    "gk_influence_xfns",
    "player_influence_xfns",
    "space_creation_xfns",
    "pausa_xfns",
    "off_ball_run_value_xfns",
]

# The add_<x> -> <x>_xfns derivation is literal EXCEPT where a family's public names drifted on
# a plural/singular (recorded explicitly rather than loosening the assertion):
#   add_cover_shadows      -> cover_shadow_xfns      (shadows -> shadow)
#   add_off_ball_run_values -> off_ball_run_value_xfns (values -> value)
_AGG_TO_XFNS_OVERRIDE = {
    "add_cover_shadows": "cover_shadow_xfns",
    "add_off_ball_run_values": "off_ball_run_value_xfns",
}


def _xfns_name_for(agg_name: str) -> str:
    if agg_name in _AGG_TO_XFNS_OVERRIDE:
        return _AGG_TO_XFNS_OVERRIDE[agg_name]
    return agg_name[len("add_") :] + "_xfns"


def _accepts_cache(fn) -> bool:
    return "pitch_control_cache" in inspect.signature(fn).parameters


def test_every_pc_consuming_xfns_factory_accepts_the_cache():
    missing = [n for n in _PC_CONSUMING_XFNS if not _accepts_cache(getattr(F, n))]
    assert not missing, f"PC-consuming xfns factories missing pitch_control_cache=: {missing}"


def test_pc_consuming_set_matches_aggregators_that_accept_the_cache():
    # META (anti-rot): the wired set must equal the set of xfns for add_* aggregators that accept
    # pitch_control_cache, so a NEW PC-consuming family can't be added without appearing here.
    agg_xfns = {_xfns_name_for(n) for n in dir(F) if n.startswith("add_") and _accepts_cache(getattr(F, n))}
    wired = set(_PC_CONSUMING_XFNS)

    # Every aggregator-with-cache maps to a wired (and real) xfns factory.
    missing = agg_xfns - wired
    assert not missing, f"add_* aggregators accepting pitch_control_cache with no wired xfns: {sorted(missing)}"
    # No stale entries: every wired name is backed by a real aggregator-with-cache AND exists on F.
    assert wired == agg_xfns, f"wired set diverged from aggregator-derived set: {sorted(wired ^ agg_xfns)}"
    assert all(hasattr(F, n) for n in _PC_CONSUMING_XFNS)
