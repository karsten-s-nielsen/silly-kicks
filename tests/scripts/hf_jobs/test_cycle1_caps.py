"""Gate (B32): the committed cycle1.toml MUST pin dispatch_cap=25 AND shard_cap=25 explicitly.

Owner ruling (HFU full utilization): shard_cap is 25 THIS cycle -- full-utilization sharding is NOT deferred, so
a stage may split up to 25 shards. shard_cap is still pinned EXPLICITLY (not left to the `= dispatch_cap`
default) so the value is a deliberate decision; a missing shard_cap fails loud.
"""

from __future__ import annotations

import pathlib

from scripts._toml import load_toml

CYCLE1 = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "hf_jobs" / "campaigns" / "cycle1.toml"


def test_cycle1_pins_both_caps_explicitly():
    raw = load_toml(CYCLE1)["campaign"]
    assert raw.get("dispatch_cap") == 25, "cycle1 must set dispatch_cap=25 (the probe budget)"
    assert raw.get("shard_cap") == 25, "cycle1 shard_cap must be 25 (owner ruling: full sharding this cycle)"
    assert "concurrency_cap" not in raw, "the single concurrency_cap is replaced by dispatch_cap/shard_cap"
    # INF-B-05: ties the "no dispatch back-off at shard_cap=dispatch_cap=25" claim to the committed toml --
    # dispatch_floor is absent, so it defaults to shard_cap=25 (= dispatch_cap), leaving no back-off range.
    assert "dispatch_floor" not in raw, "cycle1 must NOT set dispatch_floor (defaults to shard_cap=25; no back-off)"


def test_cycle1_target_is_30_minutes():
    """HFU full-utilization (R3): the cycle1 sizing target is 30 minutes so the planner splits more
    aggressively. The job timeout is decoupled from the target (T2), never derived from it."""
    raw = load_toml(CYCLE1)["campaign"]
    assert raw.get("target_minutes") == 30, "cycle1 target_minutes must be 30 (HFU full-utilization R3)"
