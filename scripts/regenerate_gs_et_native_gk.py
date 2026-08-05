"""Regenerate the committed `gs_et` ET fixture with NATIVE goalkeeper labels (TF-23b, ADR-035).

The prior committed `tests/regressions/extratime/gs_et/frames.parquet` carried no
`is_goalkeeper`/roster, so the round-trip test had to synthesize a roster. This script
re-extracts Gradient Sports **WC2022** match 10517, period 3, from the pining-for-the-data
cache as the **raw tracking-adapter input** (`x_centered`/`y_centered` + flags) carrying the
**native** `is_goalkeeper` from the roster join -- so `test_real_et_roundtrip.py` exercises the
production GK anchor and self-corrects a negated ET flag against a geometric ground truth.

Mirrors `scripts/_loader_pining._build_gradientsports`'s bronze->`resolved` path (the input to
`convert_to_frames`); it does NOT call `convert_to_frames` -- the fixture must be raw input so the
test can feed it a negated flag. Reads the cached artifacts (no network/token needed):
``<cache>/gradientsports/<match>/gradientsports_<match>_{metadata,roster,tracking,events}``.

Run on the DGX (canonical compute, pining cache present):

    python scripts/regenerate_gs_et_native_gk.py \
        --cache-dir ~/Development/silly-kicks/xt_bandwidth_run/artifact_cache \
        --out-dir tests/regressions/extratime/gs_et
"""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path

import pandas as pd

from silly_kicks.tracking.gradientsports import EXPECTED_INPUT_COLUMNS, add_gradientsports_player_ids

# This match's documented identity (tripwire -- fail loud if the loaded match disagrees).
_EXPECTED_MATCH_ID = 10517
_EXPECTED_HOME_TEAM_ID = 364
_PERIOD = 3


def _dedupe_gs_frame_records(frames_json: list[dict]) -> list[dict]:
    """Drop duplicate (period, frameNum) records (GS ships some 2-16x). Mirrors the loader."""
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for fr in frames_json:
        key = (int(fr["period"]), int(fr["frameNum"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(fr)
    return out


def _gradientsports_adapter_input(match_dir: Path, match_id: int) -> tuple[pd.DataFrame, dict]:
    """Parse cached GS bronze -> the raw `convert_to_frames` input (native is_goalkeeper).

    A faithful lift of `_loader_pining._build_gradientsports` up to (not including) the
    `convert_to_frames` call -- the `resolved` DataFrame is exactly the adapter input.
    """
    prefix = f"gradientsports_{match_id}_"
    paths = {role: match_dir / f"{prefix}{role}" for role in ("metadata", "roster", "tracking", "events")}
    for role, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing cached GS artifact {role!r}: {p}")

    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    meta = meta[0] if isinstance(meta, list) else meta
    home_team_id = int(meta["homeTeam"]["id"])
    away_team_id = int(meta["awayTeam"]["id"])
    home_start_left = bool(meta.get("homeTeamStartLeft", True))
    home_start_left_et = meta.get("homeTeamStartLeftExtraTime")
    game_id = int(meta.get("id", meta.get("gameId", 0)) or 0)

    roster_raw = json.loads(paths["roster"].read_text(encoding="utf-8"))
    roster = pd.DataFrame(
        {
            "team_id": [int(r["team"]["id"]) for r in roster_raw],
            "shirt_number": [str(r["shirtNumber"]) for r in roster_raw],
            "player_id": [int(r["player"]["id"]) for r in roster_raw],
            "position_group_type": [r.get("positionGroupType") for r in roster_raw],
        }
    )

    raw = paths["tracking"].read_bytes()
    text = bz2.decompress(raw).decode("utf-8") if raw[:2] == b"BZ" else raw.decode("utf-8")
    frames_json = _dedupe_gs_frame_records([json.loads(line) for line in text.splitlines() if line.strip()])

    rows: list[dict] = []
    for fr in frames_json:
        base = dict(
            game_id=game_id,
            period_id=int(fr["period"]),
            frame_id=int(fr["frameNum"]),
            time_seconds=float(fr.get("periodGameClockTime", 0.0)),
            frame_rate=29.97,
            z=0.0,
            speed_native=float("nan"),
            ball_state="alive",
        )
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for p in fr.get(key, []):
                rows.append(
                    {
                        **base,
                        "team_side": side,
                        "jersey_number": str(p["jerseyNum"]),
                        "is_ball": False,
                        "x_centered": float(p["x"]),
                        "y_centered": float(p["y"]),
                    }
                )
        for b in fr.get("balls", []):
            ball_z = b.get("z")
            rows.append(
                {
                    **base,
                    "team_side": None,
                    "jersey_number": None,
                    "is_ball": True,
                    "x_centered": float(b["x"]),
                    "y_centered": float(b["y"]),
                    "z": float(ball_z) if ball_z is not None else float("nan"),
                }
            )
    jersey_frames = pd.DataFrame(rows)
    resolved, _rep = add_gradientsports_player_ids(
        jersey_frames, roster, home_team_id=home_team_id, away_team_id=away_team_id
    )
    meta_out = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_start_left": home_start_left,
        "home_team_start_left_extratime": home_start_left_et,
    }
    return resolved, meta_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--match-id", type=int, default=_EXPECTED_MATCH_ID)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Keep only the first N distinct period-3 frame_ids (committed-fixture size cap; "
        "a few hundred frames give a stable home-GK-median anchor). 0 = full period.",
    )
    args = ap.parse_args()

    if args.match_id != _EXPECTED_MATCH_ID:
        raise SystemExit(
            f"refusing to regenerate gs_et from match {args.match_id}: the committed fixture's "
            f"identity ({_EXPECTED_MATCH_ID}) is referenced by the README + test; a substitution "
            "must be a deliberate, documented decision (edit _EXPECTED_MATCH_ID + README + meta)."
        )
    match_dir = args.cache_dir.expanduser() / "gradientsports" / str(args.match_id)
    if not match_dir.is_dir():
        raise SystemExit(f"match {args.match_id} not found in pining cache at {match_dir} -- fail loud, no fallback.")

    resolved, meta_out = _gradientsports_adapter_input(match_dir, args.match_id)

    # Tripwire: the loaded match must be the documented one.
    if meta_out["home_team_id"] != _EXPECTED_HOME_TEAM_ID:
        raise SystemExit(
            f"home_team_id={meta_out['home_team_id']} != documented {_EXPECTED_HOME_TEAM_ID} "
            f"for match {args.match_id} -- wrong match loaded; aborting."
        )

    et = resolved[resolved["period_id"] == _PERIOD].copy()
    if et.empty:
        raise SystemExit(f"no period-{_PERIOD} frames for match {args.match_id}.")

    # Cap to a small committed-fixture window (the first N distinct frame_ids).
    if args.max_frames and et["frame_id"].nunique() > args.max_frames:
        keep = sorted(et["frame_id"].unique())[: args.max_frames]
        et = et[et["frame_id"].isin(keep)].copy()

    # Keep exactly the tracking-adapter input columns (no extra restricted fields).
    cols = [c for c in EXPECTED_INPUT_COLUMNS if c in et.columns]
    missing = set(EXPECTED_INPUT_COLUMNS) - set(cols)
    if missing:
        raise SystemExit(f"adapter input missing required columns: {sorted(missing)}")
    et = et[cols].reset_index(drop=True)

    # Native GK must be present (the whole point of the regen).
    home_gk = et[(et["team_id"] == meta_out["home_team_id"]) & (et["is_goalkeeper"]) & (~et["is_ball"])]
    if home_gk["player_id"].nunique() < 1:
        raise SystemExit("no native home goalkeeper in the period-3 slice -- regen would not exercise the GK anchor.")

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    et.to_parquet(out_dir / "frames.parquet", index=False)
    pd.DataFrame([meta_out]).to_parquet(out_dir / "meta.parquet", index=False)

    print(f"wrote {len(et)} rows ({et['frame_id'].nunique()} frames) -> {out_dir / 'frames.parquet'}")
    print(f"meta: {meta_out}")
    print(f"native home GKs: {sorted(home_gk['player_id'].unique().tolist())}")


if __name__ == "__main__":
    main()
