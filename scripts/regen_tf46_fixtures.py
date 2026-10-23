"""Regenerate the TF-46 Arm 1 slim fixtures from pining (Task 2.1, spec section 10.1).

Sources the committed test fixtures from the gated pining mock-provider API (network; the public token
serves SkillCorner open data and IDSSE). NOT run in default CI: the offline hash test
(``tests/datasets/test_tf46_fixture_hashes.py``) verifies the committed fixtures against ``SHA256SUMS``
without a token. Owner/DGX-run only (``docs/context/corpus-drivers.md``).

Walks the matches through the shared ``scripts/_driver.for_each`` seam (ADR-052): each match is persisted
so a crash resumes, work already done is skipped, and progress prints -- the resilience a remote-box
corpus walk needs. ``work(ref)`` writes one match's slim fixture files as a side effect and returns a
one-row provenance frame; the committed fixtures are the files, not the shards (``<out>/_shards`` is
transient bookkeeping, never committed).

SkillCorner: ``<out>/skillcorner/opendata/<id>_events_slim.csv`` (the converter-read columns + the goal
truth) + ``<id>_match.json``; for ``_TRACKING_SLIM_IDS`` also ``<id>_tracking_bronze_slim.parquet`` (the tracking
bronze on every ``_TRACKING_SLIM_STRIDE``-th frame). IDSSE: ``<out>/idsse/<bare>_events_native.parquet`` +
``<bare>_meta.json`` (home_team_id + DFL-derived home_team_start_left / _extratime). Writes ``<out>/SHA256SUMS``
over every fixture file; re-running reproduces each fixture byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts._driver import for_each
from scripts._loader_pining import (
    MatchRef,
    _base_url,
    _download_artifacts,
    _resolve_token,
    list_match_refs,
)

#: The SkillCorner ``dynamic_events.csv`` columns the converter + goals reconciliation read. Everything
#: else of the ~294 native columns is dropped from the committed fixture.
_SKILLCORNER_KEEP = [
    "event_type",
    "event_id",
    "player_id",
    "team_id",
    "period",
    "time_start",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "player_targeted_x_reception",
    "player_targeted_y_reception",
    "end_type",
    "start_type",
    "game_interruption_before",
    "game_interruption_after",
    "player_targeted_third_pass",
    "player_targeted_channel_pass",
    "is_header",
    "hand_pass",
    "pass_outcome",
    "received",
    "team_score",
    "opponent_team_score",
    # TF-46 Arm 1 Part 2 GI facts (spec section 17.2, CL-1): the per-possession frame bracket and the
    # attacking side. Absent-column safe (the `keep` filter below drops any the feed lacks).
    "frame_start",
    "frame_end",
    "attacking_side",
]


#: Opendata matches that also get a slim TRACKING bronze (R.2 2026-10-23: the GI direction check is driven
#: through the DEFAULT ``convert_to_frames`` on real data). One match keeps the fixture small (MIT opendata).
_TRACKING_SLIM_IDS: frozenset[str] = frozenset({"1886347"})
#: Every N-th tracking frame is kept: enough keeper geometry per period for the orienter and GoalMap.
_TRACKING_SLIM_STRIDE = 100


def _skillcorner_tracking_slim(ref: MatchRef, dest: Path, token: str, base_url: str) -> int:
    """Write ``<id>_tracking_bronze_slim.parquet``: the loader's bronze (``_skillcorner_bronze``, the input of
    ``tracking.skillcorner.convert_to_frames``) on every ``_TRACKING_SLIM_STRIDE``-th frame. Returns the rows."""
    from scripts._loader_pining import _skillcorner_bronze, _skillcorner_raw_records

    with tempfile.TemporaryDirectory() as tmp:
        paths = _download_artifacts("skillcorner", ref.match_id, dict(ref.artifacts), token, base_url, Path(tmp))
        meta = _read_json(paths["metadata"])
        raw = [r for r in _skillcorner_raw_records(paths) if int(r.get("frame") or 0) % _TRACKING_SLIM_STRIDE == 0]
        raw = [r for r in raw if r.get("period") is not None and r.get("player_data")]
        bronze = _skillcorner_bronze(raw, meta, match_id=str(ref.match_id))
    bronze = bronze.sort_values(["period", "frame", "player_id"], kind="mergesort").reset_index(drop=True)
    bronze.to_parquet(dest / f"{ref.match_id}_tracking_bronze_slim.parquet", index=False)
    return len(bronze)


def _slim_match_json(meta: dict) -> dict:
    """Keep only the ``match.json`` fields the converter + goals reconciliation read."""
    out: dict = {
        "id": meta.get("id"),
        "pitch_length": meta.get("pitch_length"),
        "pitch_width": meta.get("pitch_width"),
    }
    for side in ("home_team", "away_team"):
        if meta.get(side) is not None:
            out[side] = {"id": meta[side].get("id"), "name": meta[side].get("name")}
    for key in ("home_team_score", "away_team_score"):
        if key in meta:
            out[key] = meta[key]
    players = meta.get("players")
    if isinstance(players, list):
        out["players"] = [
            {"id": p.get("id"), "team_id": p.get("team_id"), "goal": p.get("goal"), "own_goal": p.get("own_goal")}
            for p in players
        ]
    return out


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _skillcorner_item(ref: MatchRef, dest: Path, token: str, base_url: str) -> pd.DataFrame:
    """Download, slim and write one SkillCorner match; return its one-row provenance frame."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _download_artifacts(
            "skillcorner", ref.match_id, dict(ref.artifacts), token, base_url, Path(tmp), events_only=True
        )
        events = pd.read_csv(paths["events"], low_memory=False)
        keep = [c for c in _SKILLCORNER_KEEP if c in events.columns]
        slim = events[keep].copy()
        slim.to_csv(dest / f"{ref.match_id}_events_slim.csv", index=False, lineterminator="\n")
        _write_json(dest / f"{ref.match_id}_match.json", _slim_match_json(_read_json(paths["metadata"])))
    n_track = _skillcorner_tracking_slim(ref, dest, token, base_url) if str(ref.match_id) in _TRACKING_SLIM_IDS else 0
    return pd.DataFrame(
        [{"provider": "skillcorner", "match_id": ref.match_id, "rows": len(slim), "cols": len(keep), "track": n_track}]
    )


def _idsse_item(ref: MatchRef, dest: Path, token: str, base_url: str) -> pd.DataFrame:
    """Parse, shape and write one IDSSE match's native events; return its one-row provenance frame."""
    from silly_kicks.providers.sportec import (
        derive_idsse_home_team_start_left,
        derive_idsse_home_team_start_left_extratime,
        parse_dfl_events,
        parse_dfl_match_info,
        shape_events_to_native,
    )

    bare = str(ref.match_id).removeprefix("DFL-MAT-")
    with tempfile.TemporaryDirectory() as tmp:
        paths = _download_artifacts(
            "idsse", ref.match_id, dict(ref.artifacts), token, base_url, Path(tmp), events_only=True
        )
        mi = parse_dfl_match_info(str(paths["metadata"]))
        native = shape_events_to_native(parse_dfl_events(str(paths["events"]), match_info=mi, match_id=bare))
        hsl = bool(derive_idsse_home_team_start_left(native, mi.home_team_id))
        hsl_et = derive_idsse_home_team_start_left_extratime(native, mi.home_team_id)
        native.to_parquet(dest / f"{bare}_events_native.parquet", index=False)
        _write_json(
            dest / f"{bare}_meta.json",
            {
                "match_id": bare,
                "home_team_id": mi.home_team_id,
                "away_team_id": mi.away_team_id,
                "home_team_start_left": hsl,
                "home_team_start_left_extratime": None if hsl_et is None else bool(hsl_et),
            },
        )
    return pd.DataFrame([{"provider": "idsse", "match_id": bare, "events": len(native)}])


def _write_sha256sums(out_dir: Path) -> int:
    """Hash every committed fixture file (never the transient ``_shards`` dir) into ``SHA256SUMS``."""
    files = sorted(
        p
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and "_shards" not in p.relative_to(out_dir).parts
    )
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out_dir).as_posix()}" for p in files]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the TF-46 Arm 1 slim fixtures from pining.")
    parser.add_argument("--out-dir", required=True, help="Output root for the fixture tree + SHA256SUMS.")
    parser.add_argument(
        "--provider", choices=["skillcorner", "idsse", "all"], default="all", help="Which provider to regen."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root = out_dir / "_shards"  # transient resume bookkeeping; NOT committed
    token = _resolve_token(None)  # public token serves SkillCorner open data + IDSSE
    base_url = _base_url()

    if args.provider in ("skillcorner", "all"):
        dest = out_dir / "skillcorner" / "opendata"
        dest.mkdir(parents=True, exist_ok=True)
        for_each(
            list_match_refs(providers=["skillcorner"], token=token, base_url=base_url),
            key=lambda r: ("skillcorner", r.match_id),
            work=lambda r: _skillcorner_item(r, dest, token, base_url),
            shard_root=shard_root,
            token_inputs={"provider": "skillcorner", "task": "tf46-fixtures"},
            label="match",
        )
    if args.provider in ("idsse", "all"):
        dest = out_dir / "idsse"
        dest.mkdir(parents=True, exist_ok=True)
        for_each(
            list_match_refs(providers=["idsse"], token=token, base_url=base_url),
            key=lambda r: ("idsse", r.match_id),
            work=lambda r: _idsse_item(r, dest, token, base_url),
            shard_root=shard_root,
            token_inputs={"provider": "idsse", "task": "tf46-fixtures"},
            label="match",
        )

    n = _write_sha256sums(out_dir)
    print(f"wrote {n} fixture file(s) + SHA256SUMS to {out_dir}")


if __name__ == "__main__":
    main()
