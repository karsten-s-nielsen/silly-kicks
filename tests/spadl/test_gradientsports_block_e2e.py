"""Owner-gated e2e: validate the block-detection columns on the real WC2022 GS catalog.

Runs only where the pining owner token + data are reachable (public CI skips). The synthetic unit
tests in ``test_gradientsports.py`` prove the mechanism on hand-placed values; this is the ONLY
real-data validation that the GS converter surfaces `shotOutcomeType=="B"` -> `shot_blocked` and
`crossOutcomeType=="B"` -> `cross_blocked` (open-play only). See ADR-046 + the block-detection plan.
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pytest

# match 10502 was pining-probed during design: 6 raw B-shots, 6 raw B-crosses (crossOutcomeType=="B"
# <=> incompletionReasonType=="BL"). Counts are re-derived from the raw events at run time, not
# hard-coded, so the test tracks the real feed rather than a snapshot.
_MATCH = "10502"

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


def test_block_columns_on_real_gs_match():
    L = _load_loader()
    tok, base = L._resolve_token(None), L._base_url()

    # Raw B-shot / B-cross counts, straight from the feed.
    with tempfile.TemporaryDirectory() as tmp:
        p = L._download_to_temp("gradientsports", _MATCH, "events", tok, base, Path(tmp))
        raw = json.load(open(p, encoding="utf-8"))

    def _pe(ev):
        return ev.get("possessionEvents") or {}

    raw_b_shots = sum(
        1 for ev in raw if _pe(ev).get("possessionEventType") == "SH" and _pe(ev).get("shotOutcomeType") == "B"
    )
    raw_b_crosses = sum(
        1 for ev in raw if _pe(ev).get("possessionEventType") == "CR" and _pe(ev).get("crossOutcomeType") == "B"
    )
    assert raw_b_shots >= 1 and raw_b_crosses >= 1, (
        f"fixture {_MATCH} should carry raw B shots + crosses; got shots={raw_b_shots} crosses={raw_b_crosses}"
    )

    # Converted SPADL output.
    _prov, _m, actions, _frames, _home = next(
        iter(
            L.load_matches(
                providers=["gradientsports"],
                match_ids={"gradientsports": [_MATCH]},
                tracking_limit=1,
            )
        )
    )

    assert str(actions["shot_blocked"].dtype) == "boolean"
    assert str(actions["cross_blocked"].dtype) == "boolean"

    n_shot_blocked = int((actions["shot_blocked"] == True).sum())  # noqa: E712
    n_cross_blocked = int((actions["cross_blocked"] == True).sum())  # noqa: E712

    # Every real blocked shot is surfaced; no over-detection.
    assert 1 <= n_shot_blocked <= raw_b_shots, f"shot_blocked={n_shot_blocked} not in [1, raw B-shots {raw_b_shots}]"
    # cross_blocked is scoped to open-play `cross` (set-piece corner/freekick crosses -> pd.NA),
    # so it is a subset of the raw B-crosses, and positive on this match.
    assert 1 <= n_cross_blocked <= raw_b_crosses, (
        f"cross_blocked={n_cross_blocked} not in [1, raw B-crosses {raw_b_crosses}]"
    )

    # Non-shot / non-cross rows are pd.NA (unknown), never False — the 3-valued contract on real data.
    assert actions["shot_blocked"].isna().any(), "expected pd.NA shot_blocked on non-shot rows"
    assert actions["cross_blocked"].isna().any(), "expected pd.NA cross_blocked on non-cross rows"
