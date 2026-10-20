"""Regenerate the TF-46 Arm 1 slim fixtures from pining (Task 2.1, spec section 10.1).

Sources the committed test fixtures from the gated pining mock-provider API (network; the public token
serves SkillCorner open data and IDSSE). NOT run in default CI: the offline hash test
(``tests/datasets/test_tf46_fixture_hashes.py``) verifies the committed fixtures against ``SHA256SUMS``
without a token. This driver is owner/DGX-run only (``docs/context/corpus-drivers.md``).

SkillCorner: downloads each public match's ``dynamic_events.csv`` + ``match.json``, keeps only the columns
the converter reads (``_SKILLCORNER_KEEP``) plus the goal truth in ``match.json``, and writes
``<out>/skillcorner/opendata/<id>_events_slim.csv`` + ``<id>_match.json``.

IDSSE: parses each public DFL match to the native sportec-events frame (the converter input) and writes
``<out>/idsse/<bare_id>_events_native.parquet`` + a ``<bare_id>_meta.json`` sidecar carrying the
``home_team_id`` and the DFL-derived ``home_team_start_left`` / ``_extratime`` the converter needs.

Writes ``<out>/SHA256SUMS`` over every fixture file. Re-running reproduces each fixture byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts._loader_pining import (
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
]


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
            {
                "id": p.get("id"),
                "team_id": p.get("team_id"),
                "goal": p.get("goal"),
                "own_goal": p.get("own_goal"),
            }
            for p in players
        ]
    return out


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _regen_skillcorner(out_dir: Path, token: str, base_url: str) -> list[Path]:
    dest = out_dir / "skillcorner" / "opendata"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    refs = list_match_refs(providers=["skillcorner"], token=token, base_url=base_url)
    for ref in refs:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _download_artifacts(
                "skillcorner", ref.match_id, dict(ref.artifacts), token, base_url, Path(tmp), events_only=True
            )
            events = pd.read_csv(paths["events"], low_memory=False)
            keep = [c for c in _SKILLCORNER_KEEP if c in events.columns]
            slim = events[keep].copy()
            csv_path = dest / f"{ref.match_id}_events_slim.csv"
            slim.to_csv(csv_path, index=False, lineterminator="\n")
            json_path = dest / f"{ref.match_id}_match.json"
            _write_json(json_path, _slim_match_json(_read_json(paths["metadata"])))
            written.extend([csv_path, json_path])
            print(f"skillcorner {ref.match_id}: {len(slim)} rows, {len(keep)} cols")
    return written


def _regen_idsse(out_dir: Path, token: str, base_url: str) -> list[Path]:
    from silly_kicks.providers.sportec import (
        derive_idsse_home_team_start_left,
        derive_idsse_home_team_start_left_extratime,
        parse_dfl_events,
        parse_dfl_match_info,
        shape_events_to_native,
    )

    dest = out_dir / "idsse"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    refs = list_match_refs(providers=["idsse"], token=token, base_url=base_url)
    for ref in refs:
        bare = str(ref.match_id).removeprefix("DFL-MAT-")
        with tempfile.TemporaryDirectory() as tmp:
            paths = _download_artifacts(
                "idsse", ref.match_id, dict(ref.artifacts), token, base_url, Path(tmp), events_only=True
            )
            mi = parse_dfl_match_info(str(paths["metadata"]))
            native = shape_events_to_native(parse_dfl_events(str(paths["events"]), match_info=mi, match_id=bare))
            hsl = bool(derive_idsse_home_team_start_left(native, mi.home_team_id))
            hsl_et = derive_idsse_home_team_start_left_extratime(native, mi.home_team_id)
            pq_path = dest / f"{bare}_events_native.parquet"
            native.to_parquet(pq_path, index=False)
            meta_path = dest / f"{bare}_meta.json"
            _write_json(
                meta_path,
                {
                    "match_id": bare,
                    "home_team_id": mi.home_team_id,
                    "away_team_id": mi.away_team_id,
                    "home_team_start_left": hsl,
                    "home_team_start_left_extratime": None if hsl_et is None else bool(hsl_et),
                },
            )
            written.extend([pq_path, meta_path])
            print(f"idsse {bare}: {len(native)} native events")
    return written


def _write_sha256sums(out_dir: Path, files: list[Path]) -> None:
    lines = []
    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the TF-46 Arm 1 slim fixtures from pining.")
    parser.add_argument("--out-dir", required=True, help="Output root for the fixture tree + SHA256SUMS.")
    parser.add_argument(
        "--provider", choices=["skillcorner", "idsse", "all"], default="all", help="Which provider to regen."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _resolve_token(None)  # public token serves SkillCorner open data + IDSSE
    base_url = _base_url()

    written: list[Path] = []
    if args.provider in ("skillcorner", "all"):
        written.extend(_regen_skillcorner(out_dir, token, base_url))
    if args.provider in ("idsse", "all"):
        written.extend(_regen_idsse(out_dir, token, base_url))

    _write_sha256sums(out_dir, written)
    print(f"wrote {len(written)} fixture file(s) + SHA256SUMS to {out_dir}")


if __name__ == "__main__":
    main()
