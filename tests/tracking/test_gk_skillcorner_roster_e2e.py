"""TF-27: SkillCorner derived-GK vs external-roster Tier-1 validation (e2e).

Requires the 10 public A-League matches pre-downloaded into SKILLCORNER_SAMPLE_DIR
(run scripts/download_skillcorner_sample.py). Anchors derive_goalkeepers against the
match.json roster GK per team; exact-set-equality gate with an in-comparator sweeper
allowlist. Per match (team_ids recur across the sample — never merge truth dicts).
"""

from __future__ import annotations

import json

import pytest

from scripts._loader_pining import build_skillcorner_frames
from silly_kicks.tracking._gk_identification import derive_goalkeepers
from tests._skillcorner_sample import (
    SAMPLE_DIR,
    AgreementResult,
    available_matches,
    build_skillcorner_gk_truth,
    compare_gk_picks,
    find_artifact,
)

_FRAME_CAP = 8000  # probe-confirmed correct picks; anchors the starting GK; bounds runtime

# Genuine sweeper-keeper exceptions, (match_id, team_id) as strings, with justification.
# Empty until the e2e surfaces a real, investigated multi-pick. DO NOT add without one.
_SUBSET_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()

_REQUIRED = ("_match.json", "_tracking_extrapolated.jsonl")

pytestmark = pytest.mark.e2e


def _matches():
    return available_matches(_REQUIRED)


@pytest.mark.skipif(not _matches(), reason="SkillCorner sample (match.json + tracking) not available")
def test_skillcorner_derived_gk_matches_roster():
    overall = AgreementResult.empty()
    for mid in _matches():
        match_dir = SAMPLE_DIR / mid
        meta_path = find_artifact(match_dir, "_match.json")
        trk_path = find_artifact(match_dir, "_tracking_extrapolated.jsonl")
        with open(meta_path, encoding="utf-8") as fh:  # type: ignore[arg-type]
            meta = json.load(fh)

        truth = build_skillcorner_gk_truth(meta)
        name_map = {str(p["id"]): p.get("short_name", str(p["id"])) for p in meta.get("players", [])}

        paths = {"metadata": str(meta_path), "tracking": str(trk_path)}
        frames = build_skillcorner_frames(paths, _FRAME_CAP)

        # Join-key guard (loud, not skip): every rostered GK id must appear in frames,
        # and the overall id overlap must be substantial — a drift on an unprobed match
        # is a structural failure, never a silent "no data" or a confusing GK mismatch.
        frame_ids = {str(x) for x in frames.loc[~frames["is_ball"], "player_id"].dropna().unique()}
        roster_ids = {str(p["id"]) for p in meta.get("players", [])}
        for team, gks in truth.items():
            for gk in gks:
                assert gk in frame_ids, (
                    f"match {mid} team {team}: rostered GK {gk} "
                    f"({name_map.get(gk)}) absent from frame player_ids — id-scheme drift"
                )
        overlap = len(frame_ids & roster_ids)
        assert overlap >= min(20, len(frame_ids)), (
            f"match {mid}: only {overlap} frame ids match the roster — id-scheme drift"
        )

        _out, picks = derive_goalkeepers(frames)
        overall = overall + compare_gk_picks(
            truth, picks, match_id=mid, subset_allowlist=_SUBSET_ALLOWLIST, name_map=name_map
        )

    assert not overall.no_roster_gk, f"teams without a roster GK (unexpected):\n{overall.summary()}"
    assert overall.is_perfect, f"derived GK != roster GK:\n{overall.summary()}"
    assert len(overall.matched) >= 2, "no teams validated — sample present but empty?"
