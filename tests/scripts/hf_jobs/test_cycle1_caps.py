"""Gate (B32): the committed cycle1.toml MUST pin dispatch_cap=25 AND shard_cap=20 explicitly.

The default `shard_cap = dispatch_cap` would re-shard every stage at 25 (sharding is Cycle 2); Cycle 1 raises
ONLY the dispatch budget. A missing shard_cap is therefore a silent re-shard, so this gate fails loud on it.
"""

from __future__ import annotations

import pathlib

from scripts._toml import load_toml

CYCLE1 = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "hf_jobs" / "campaigns" / "cycle1.toml"


def test_cycle1_pins_both_caps_explicitly():
    raw = load_toml(CYCLE1)["campaign"]
    assert raw.get("dispatch_cap") == 25, "cycle1 must set dispatch_cap=25 (the probe budget)"
    assert raw.get("shard_cap") == 20, "cycle1 must set shard_cap=20 explicitly (else the default re-shards at 25)"
    assert "concurrency_cap" not in raw, "the single concurrency_cap is replaced by dispatch_cap/shard_cap"


def test_cycle1_target_is_30_minutes():
    """HFU full-utilization (R3): the cycle1 sizing target is 30 minutes so the planner splits more
    aggressively. The job timeout is decoupled from the target (T2), never derived from it."""
    raw = load_toml(CYCLE1)["campaign"]
    assert raw.get("target_minutes") == 30, "cycle1 target_minutes must be 30 (HFU full-utilization R3)"
