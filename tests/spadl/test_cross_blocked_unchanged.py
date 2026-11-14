"""``cross_blocked`` keeps its pre-ADR-133 values (spec 2026-11-13-cross-definition §7.1, plan Task 3 Step 2).

ADR-133 keys ``cross_blocked`` to the provider's own cross (``cross_native``) and leaves its meaning unchanged, so
for every converter that emits it the values must equal the base commit's. The golden holds the non-``<NA>``
``cross_blocked`` values that the converters at ``0cc07e6c`` produced on three inputs, keyed by
``original_event_id``: the real StatsBomb match 7584 (31 open-play crosses, 1 blocked) and the delivery scene with
a blocked native cross for Gradient Sports and Wyscout.

Regenerate ONLY on a checkout of the base commit (the pre-ADR-133 converters)::

    python -m tests.spadl.test_cross_blocked_unchanged
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "cross_blocked_golden.json"
BASE_COMMIT = "0cc07e6c"


def _statsbomb_real() -> pd.DataFrame:
    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl import statsbomb

    raw = json.loads(pathlib.Path("tests/datasets/statsbomb/raw/events/7584.json").read_text(encoding="utf-8"))
    events = flatten_events(raw, 7584)
    return statsbomb.convert_to_actions(events, home_team_id=events["team_id"].dropna().iloc[0])[0]


def _gradientsports_scene() -> pd.DataFrame:
    from tests.spadl._delivery_scenes import _gradientsports, _gradientsports_run

    return _gradientsports_run(_gradientsports(blocked=True))


def _wyscout_scene() -> pd.DataFrame:
    from tests.spadl._delivery_scenes import _wyscout, _wyscout_run

    return _wyscout_run(_wyscout(blocked=True))


RUNS = {
    "statsbomb_real_7584": _statsbomb_real,
    "gradientsports_scene_blocked": _gradientsports_scene,
    "wyscout_scene_blocked": _wyscout_scene,
}


def _known_cross_blocked(actions: pd.DataFrame) -> dict[str, bool]:
    """``{original_event_id: cross_blocked}`` over the rows where it is known (not ``<NA>``)."""
    known = actions[actions["cross_blocked"].notna() & actions["original_event_id"].notna()]
    return {str(k): bool(v) for k, v in zip(known["original_event_id"], known["cross_blocked"], strict=True)}


@pytest.mark.parametrize("run", sorted(RUNS))
def test_cross_blocked_unchanged(run):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["base_commit"] == BASE_COMMIT
    expected = golden["runs"][run]
    assert any(expected.values()), f"precondition: the {run} golden holds a blocked cross"
    assert _known_cross_blocked(RUNS[run]()) == expected


if __name__ == "__main__":  # regenerate from the BASE commit's converters only
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    payload = {"base_commit": BASE_COMMIT, "runs": {name: _known_cross_blocked(fn()) for name, fn in RUNS.items()}}
    GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
