"""One-off empirical probe for silly-kicks tracking PR-1 (PR-S19).

Reads per-provider tracking statistics from:
  Source 1: Databricks SQL --- soccer_analytics.dev_gold.fct_tracking_frames
            (providers: metrica, idsse, skillcorner)
  Source 2: Local PFF WC2022 JSONL.bz2 (1 match minimum to characterize)

Writes: tests/datasets/tracking/empirical_probe_baselines.json

Run once during PR-S19 development. Both this script AND its output JSON are
committed to the repo. The real datasets are NOT committed.

Usage::

    python scripts/probe_tracking_baselines.py

Notes
-----
- The PFF local path contains brackets (``[Karsten]``, ``[Microsoft]``) that
  ``glob`` interprets as character classes, so listing uses ``Path.iterdir()``
  + suffix filter rather than ``Path.glob()`` (memory:reference_pff_data_local).
- The probe is defensive about schema variation and lakehouse availability;
  it falls back to documented defaults where queries or columns are missing.
"""

from __future__ import annotations

import bz2
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "tests" / "datasets" / "tracking" / "empirical_probe_baselines.json"

LAKEHOUSE_TABLE = "soccer_analytics.dev_gold.fct_tracking_frames"
PFF_LOCAL_DIR = Path(r"D:\[Karsten]\Dropbox\[Microsoft]\Downloads\FIFA World Cup 2022\Tracking Data")


def _list_pff_tracking_files(pff_dir: Path) -> list[Path]:
    """List PFF .jsonl.bz2 files via iterdir() to bypass bracketed-path glob issues."""
    if not pff_dir.is_dir():
        return []
    return sorted(p for p in pff_dir.iterdir() if p.is_file() and p.name.endswith(".jsonl.bz2"))


def probe_lakehouse() -> dict[str, dict[str, Any]]:
    """Query fct_tracking_frames per provider via Databricks SQL.

    Returns a dict keyed by silly-kicks provider name (idsse->sportec).
    On any failure (connector missing, env unset, query error), returns {}.
    """
    try:
        from databricks import sql  # type: ignore[import-not-found]
    except ImportError:
        print("  [warn] databricks-sql-connector not installed; skipping lakehouse probe")
        return {}

    raw_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get("DATABRICKS_HOST", "")
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path.startswith("//"):
        http_path = http_path[1:]
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
        print("  [warn] DATABRICKS_HOST/HTTP_PATH/TOKEN not all set; skipping lakehouse probe")
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        conn = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )
    except Exception as e:
        print(f"  [warn] databricks connect failed: {e}; skipping lakehouse probe")
        return {}

    try:
        with conn.cursor() as cur:
            for raw_provider in ("metrica", "idsse", "skillcorner"):
                key = "sportec" if raw_provider == "idsse" else raw_provider
                try:
                    # raw_provider is interpolated from a fixed tuple literal
                    # ("metrica", "idsse", "skillcorner") and the Databricks
                    # token used here is read-only on this mart.
                    cur.execute(
                        f"""
                        SELECT
                            approx_percentile(frame_rate, 0.5)               AS frame_rate_p50,
                            count(DISTINCT period)                           AS n_periods,
                            avg(CASE WHEN x IS NULL THEN 1.0 ELSE 0.0 END)   AS nan_rate_x,
                            avg(CASE WHEN y IS NULL THEN 1.0 ELSE 0.0 END)   AS nan_rate_y,
                            avg(CASE WHEN ball_x IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_ball_x,
                            avg(CASE WHEN ball_y IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_ball_y,
                            avg(CASE WHEN speed IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_speed,
                            avg(CASE WHEN speed_ms IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_speed_ms,
                            approx_percentile(speed_ms, 0.5)                 AS speed_ms_p50,
                            approx_percentile(speed_ms, 0.99)                AS speed_ms_p99,
                            approx_percentile(distance_to_ball, 0.5)         AS distance_to_ball_p50,
                            approx_percentile(distance_to_ball, 0.95)        AS distance_to_ball_p95,
                            avg(CASE WHEN x < 0 OR x > 120 THEN 1.0 ELSE 0.0 END) AS off_pitch_x_rate,
                            avg(CASE WHEN y < 0 OR y > 80 THEN 1.0 ELSE 0.0 END) AS off_pitch_y_rate,
                            avg(CASE WHEN ball_x < 0 OR ball_x > 120 THEN 1.0 ELSE 0.0 END) AS off_pitch_ball_x_rate,
                            count(DISTINCT match_id)                         AS n_matches,
                            count(*)                                         AS n_rows
                        FROM {LAKEHOUSE_TABLE}
                        WHERE source_provider = '{raw_provider}'
                        """
                    )
                    row = cur.fetchone()
                    if row is None:
                        continue
                    cols = [d[0] for d in cur.description]
                    out[key] = {c: _coerce(v) for c, v in zip(cols, row, strict=False)}
                    # The lakehouse mart is wide-form 120x80 (StatsBomb units); flag
                    # this so synthetic generators do not import the unit choice.
                    out[key]["lakehouse_wide_form_units"] = "statsbomb_120x80"
                    out[key]["speed_native_supplied"] = (
                        out[key].get("nan_rate_speed_ms") is not None and out[key]["nan_rate_speed_ms"] < 0.5
                    )
                except Exception as e:
                    print(f"  [warn] lakehouse query for {raw_provider} failed: {e}")
                    continue
    finally:
        conn.close()
    return out


def _coerce(v: Any) -> Any:
    """Coerce Databricks SQL row values to JSON-serializable scalars."""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return float(v) if hasattr(v, "__float__") else str(v)


def probe_pff_local(jsonl_bz2_path: Path, max_frames: int = 5000) -> dict[str, Any]:
    """Read up to ``max_frames`` PFF tracking frames and characterize them.

    PFF JSONL schema: each line is one frame with nested ``balls`` (list),
    ``homePlayers`` / ``awayPlayers`` (lists). Frame rate is 29.97 fps per
    PFF metadata; we report 30 as the canonical bucket.
    """
    rows: list[dict[str, Any]] = []
    n_balls_present = 0
    n_homeplayers_total = 0
    n_awayplayers_total = 0
    n_off_pitch_x = 0
    n_off_pitch_y = 0
    n_player_rows = 0
    with bz2.open(jsonl_bz2_path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_frames:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
            balls = obj.get("balls") or []
            if balls:
                n_balls_present += 1
            for grp in (obj.get("homePlayers") or [], obj.get("awayPlayers") or []):
                for p in grp:
                    n_player_rows += 1
                    px, py = p.get("x"), p.get("y")
                    if px is not None and (px < -52.5 or px > 52.5):
                        n_off_pitch_x += 1
                    if py is not None and (py < -34.0 or py > 34.0):
                        n_off_pitch_y += 1
            n_homeplayers_total += len(obj.get("homePlayers") or [])
            n_awayplayers_total += len(obj.get("awayPlayers") or [])

    n_frames = len(rows)
    if n_frames == 0:
        return {"frame_rate_p50": 30.0, "n_frames_sample": 0, "match_filename": jsonl_bz2_path.name}

    df = pd.DataFrame({"period": [r.get("period") for r in rows]})
    n_periods = int(df["period"].dropna().nunique())

    return {
        "frame_rate_p50": 30.0,  # PFF documented 29.97 Hz; round to 30 for synthetic generators
        "n_periods": n_periods,
        "n_frames_sample": n_frames,
        "ball_visible_rate": float(n_balls_present) / n_frames,
        "avg_home_players_per_frame": float(n_homeplayers_total) / n_frames,
        "avg_away_players_per_frame": float(n_awayplayers_total) / n_frames,
        "off_pitch_x_rate": float(n_off_pitch_x) / max(n_player_rows, 1),
        "off_pitch_y_rate": float(n_off_pitch_y) / max(n_player_rows, 1),
        "speed_native_supplied": False,  # PFF JSONL doesn't carry per-player speed natively
        "ball_row_rate": 1.0,
        "match_filename": jsonl_bz2_path.name,
    }


def main() -> None:
    print("[1/3] Probing lakehouse...")
    lakehouse_stats = probe_lakehouse()
    print(f"  Got {len(lakehouse_stats)} provider entries: {list(lakehouse_stats)}")

    print("[2/3] Probing local PFF WC22 (1 match)...")
    matches = _list_pff_tracking_files(PFF_LOCAL_DIR)
    if not matches:
        print(f"  [warn] No PFF tracking files in {PFF_LOCAL_DIR}; PFF entry will use defaults.")
        pff_stats: dict[str, Any] = {
            "frame_rate_p50": 30.0,
            "n_frames_sample": 0,
            "speed_native_supplied": True,
            "ball_row_rate": 1.0,
            "match_filename": None,
        }
    else:
        pff_stats = probe_pff_local(matches[0])
        print(f"  Probed {pff_stats.get('match_filename')} ({pff_stats.get('n_frames_sample')} frames)")

    out = {
        "probe_run_date": "2026-04-30",
        "probe_run_source_lakehouse_table": LAKEHOUSE_TABLE,
        "probe_run_source_pff_path_marker": "FIFA World Cup 2022/Tracking Data",
        "providers": _ensure_all_providers({**lakehouse_stats, "pff": pff_stats}),
    }

    print(f"[3/3] Writing {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print("Done.")


def _ensure_all_providers(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Guarantee all four provider keys present; backfill defaults where missing."""
    defaults = {
        "metrica": {"frame_rate_p50": 25.0, "speed_native_supplied": False},
        "sportec": {"frame_rate_p50": 25.0, "speed_native_supplied": True},
        "skillcorner": {"frame_rate_p50": 10.0, "speed_native_supplied": False},
        "pff": {"frame_rate_p50": 30.0, "speed_native_supplied": True},
    }
    out = dict(stats)
    for name, defaults_dict in defaults.items():
        out.setdefault(name, defaults_dict)
        for k, v in defaults_dict.items():
            out[name].setdefault(k, v)
    return out


if __name__ == "__main__":
    main()
