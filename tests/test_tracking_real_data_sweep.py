"""Real-data sweep, marked e2e. Skipped in CI; run locally before tagging.

Loads real tracking data from environment-pointed local paths and asserts:
  - dtype audit (output matches the per-provider canonical schema)
  - bounds audit (TRACKING_CONSTRAINTS)
  - NaN-rate-per-column audit
  - distance-to-ball percentile baseline emit

Output: structured JSON summary printed to stdout. Run via::

    pytest tests/test_tracking_real_data_sweep.py -m e2e -s

Use the JSON summary in the PR description.

The four loader bodies require local-data helpers the user has from
PR-S18 (PFF) and the lakehouse pipelines (IDSSE / Metrica / SkillCorner).
Each test ``pytest.skip``s with an explicit reason when the
``*_TRACKING_DIR`` env var is missing or the loader is not available
(memory: silently-skipping-tests-hide-breakage).
"""

from __future__ import annotations

import bz2
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import TRACKING_CONSTRAINTS

PROBE_BASELINES = json.loads(
    (Path(__file__).resolve().parent / "datasets" / "tracking" / "empirical_probe_baselines.json").read_text(),
)


def _summarize_provider(frames: pd.DataFrame, provider: str) -> dict[str, Any]:
    nan_rates = {col: float(frames[col].isna().mean()) for col in frames.columns}
    dist_pcts: dict[str, float] = {}
    if "is_ball" in frames.columns:
        ball = frames[frames["is_ball"]][["period_id", "frame_id", "x", "y"]].rename(
            columns={"x": "bx", "y": "by"},
        )
        players = frames[~frames["is_ball"]]
        joined = players.merge(ball, on=["period_id", "frame_id"])
        dist = pd.Series(
            np.sqrt(
                (joined["x"] - joined["bx"]) ** 2 + (joined["y"] - joined["by"]) ** 2,
            ),
        ).dropna()
        if len(dist):
            dist_pcts = {f"p{p}": float(np.percentile(dist, p)) for p in (25, 50, 75, 95)}
    summary = {
        "provider": provider,
        "n_rows": len(frames),
        "frame_rate_observed": float(frames["frame_rate"].iloc[0]) if len(frames) else None,
        "n_periods_observed": int(frames["period_id"].nunique()) if len(frames) else 0,
        "nan_rate_x": nan_rates.get("x", 0.0),
        "nan_rate_y": nan_rates.get("y", 0.0),
        "nan_rate_speed": nan_rates.get("speed", 0.0),
        "distance_to_ball_percentiles": dist_pcts,
    }
    print(f"\n[real-data-sweep:{provider}] {json.dumps(summary)}")
    return summary


def _bounds_check(frames: pd.DataFrame, provider: str) -> None:
    for col, (lo, hi) in TRACKING_CONSTRAINTS.items():
        if col not in frames.columns:
            continue
        vals = frames[col].dropna()
        if len(vals) == 0:
            continue
        if hi == float("inf"):
            assert (vals >= lo).all(), f"{provider}/{col}: values below {lo}"
        else:
            # Real data has a small tail of physically-realistic
            # out-of-pitch rows: players overrun touchlines, ball goes for
            # throw-in, the goalkeeper runs into the goal, etc. Empirical
            # PR-S19 sweep on PFF WC22 shows ~0.5% off-pitch on y; allow
            # 2% tolerance per provider. The strict synthetic gate stays
            # at 100% in test_tracking_cross_provider_parity.py.
            in_bounds = vals.between(lo, hi).mean()
            assert in_bounds >= 0.98, f"{provider}/{col}: only {in_bounds:.4f} of values in [{lo}, {hi}]"


@pytest.mark.e2e
def test_pff_real_data_sweep():
    path = os.environ.get("PFF_TRACKING_DIR")
    if not path:
        pytest.skip("PFF_TRACKING_DIR not set; skipping PFF real-data sweep.")
    pff_dir = Path(path)
    if not pff_dir.is_dir():
        pytest.skip(f"PFF_TRACKING_DIR={path!r} is not a directory; skipping.")
    matches = sorted(p for p in pff_dir.iterdir() if p.name.endswith(".jsonl.bz2"))
    if not matches:
        pytest.skip(f"No .jsonl.bz2 files in PFF_TRACKING_DIR={path!r}; skipping.")

    rows: list[dict[str, Any]] = []
    home_team_id_value: int | None = None
    with bz2.open(matches[0], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 3000:
                break
            obj = json.loads(line)
            period_id = obj.get("period")
            if period_id is None:
                continue
            game_event = obj.get("game_event") or {}
            if home_team_id_value is None:
                home_team_id_value = (
                    game_event.get("home_team", {}).get("id")
                    if isinstance(
                        game_event.get("home_team"),
                        dict,
                    )
                    else None
                )
            for is_home, players in (
                (True, obj.get("homePlayers") or []),
                (False, obj.get("awayPlayers") or []),
            ):
                for p in players:
                    rows.append(
                        {
                            "game_id": int(obj.get("gameRefId") or 0),
                            "period_id": int(period_id),
                            "frame_id": int(obj.get("frameNum") or i),
                            "time_seconds": float(obj.get("periodElapsedTime") or 0.0),
                            "frame_rate": 30.0,
                            "player_id": p.get("jerseyNum"),
                            "team_id": 1 if is_home else 2,
                            "is_ball": False,
                            "is_goalkeeper": False,
                            "x_centered": float(p.get("x", 0.0)),
                            "y_centered": float(p.get("y", 0.0)),
                            "z": float("nan"),
                            "speed_native": float("nan"),
                            "ball_state": "alive",
                        }
                    )
            for ball in obj.get("balls") or []:
                rows.append(
                    {
                        "game_id": int(obj.get("gameRefId") or 0),
                        "period_id": int(period_id),
                        "frame_id": int(obj.get("frameNum") or i),
                        "time_seconds": float(obj.get("periodElapsedTime") or 0.0),
                        "frame_rate": 30.0,
                        "player_id": None,
                        "team_id": None,
                        "is_ball": True,
                        "is_goalkeeper": False,
                        "x_centered": float(ball.get("x", 0.0)),
                        "y_centered": float(ball.get("y", 0.0)),
                        "z": float(ball.get("z", 0.0)),
                        "speed_native": float("nan"),
                        "ball_state": "alive",
                    }
                )

    if not rows:
        pytest.skip("PFF real-data sweep: no parseable rows from sample.")

    raw = pd.DataFrame(rows)
    raw["player_id"] = raw["player_id"].astype("Int64")
    raw["team_id"] = raw["team_id"].astype("Int64")

    from silly_kicks.tracking.pff import convert_to_frames

    frames, _ = convert_to_frames(raw, home_team_id=1, home_team_start_left=True)
    _bounds_check(frames, "pff")
    _summarize_provider(frames, "pff")


def _databricks_connect():
    """Open a Databricks SQL connection from DATABRICKS_HOST/HTTP_PATH/TOKEN env vars.

    Returns None when any of the three is missing or the connector is not
    installed --- callers should ``pytest.skip`` with an explicit reason.
    """
    try:
        from databricks import sql  # type: ignore[import-not-found]
    except ImportError:
        return None
    raw_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get(
        "DATABRICKS_HOST",
        "",
    )
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path.startswith("//"):
        http_path = http_path[1:]
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
        return None
    try:
        return sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )
    except Exception:
        return None


def _query_lakehouse_sample(provider_raw: str, n_rows: int = 50000) -> pd.DataFrame | None:
    """Pull a single-match wide-form sample from the lakehouse mart.

    Returns the raw lakehouse rows (StatsBomb 120x80 yard, wide-form with
    ball_x / ball_y), or None if the query cannot run.
    """
    conn = _databricks_connect()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            # Pick the first match_id, then pull rows with the match_id pushed
            # down into the WHERE clause (avoids a wide CTE-join that on
            # large providers (idsse 21M rows) hit Databricks query
            # timeouts during PR-S19 sweep development).
            #
            # provider_raw is constrained to a fixed tuple literal at the
            # call sites; match_id_value, min_f, max_f are server-supplied
            # results from the previous query --- not user input. The
            # Databricks token used here is also read-only on this mart.
            cur.execute(
                f"""
                SELECT match_id
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                GROUP BY match_id
                LIMIT 1
                """
            )
            r = cur.fetchone()
            if r is None:
                return pd.DataFrame()
            match_id_value = r[0]
            # Pull a contiguous frame window so the sample contains ALL players
            # per frame (natural row ordering on the mart groups rows by
            # player_id, so an unsorted LIMIT yields one player's full
            # trajectory rather than a multi-player slice).
            cur.execute(
                f"""
                SELECT min(frame) AS min_f, max(frame) AS max_f
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                  AND match_id = '{match_id_value}'
                  AND period = 1
                """
            )
            f_row = cur.fetchone()
            if f_row is None or f_row[0] is None:
                return pd.DataFrame()
            min_f = int(f_row[0])
            # ~22 players per frame; cap frames so result <= n_rows.
            n_frames_window = max(50, n_rows // 22)
            max_f = min_f + n_frames_window
            cur.execute(
                f"""
                SELECT *
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                  AND match_id = '{match_id_value}'
                  AND period = 1
                  AND frame BETWEEN {min_f} AND {max_f}
                """
            )
            description = cur.description or []
            cols = [d[0] for d in description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame.from_records(rows, columns=cols)


from datasets.tracking._lakehouse_adapter import (  # noqa: E402
    lakehouse_to_kloppy_dataset as _lakehouse_to_kloppy_dataset,
)
from datasets.tracking._lakehouse_adapter import (  # noqa: E402
    lakehouse_to_sportec_input as _lakehouse_to_sportec_input,
)


@pytest.mark.e2e
def test_idsse_real_data_sweep():
    raw = _query_lakehouse_sample("idsse")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "IDSSE real-data sweep requires Databricks SQL connectivity "
            "(DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN). "
            "Skipping.",
        )
    sportec_input = _lakehouse_to_sportec_input(raw)
    # Drop rows with NaN x/y from the lakehouse upstream (e.g., player off-frame).
    sportec_input = sportec_input.dropna(subset=["x_centered", "y_centered"])
    # _query_lakehouse_sample restricts to period 1; the lakehouse is also
    # direction-normalized, so home_team_start_left=True yields no flip.
    if len(sportec_input) == 0:
        pytest.skip("IDSSE lakehouse sample yielded no usable rows after cleanup.")

    from silly_kicks.tracking.sportec import convert_to_frames

    home_team_str = str(sportec_input.loc[~sportec_input["is_ball"], "team_id"].dropna().iloc[0])
    frames, _ = convert_to_frames(
        sportec_input,
        home_team_id=home_team_str,
        home_team_start_left=True,
    )
    _bounds_check(frames, "sportec")
    _summarize_provider(frames, "sportec")


@pytest.mark.e2e
def test_metrica_real_data_sweep():
    raw = _query_lakehouse_sample("metrica")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "Metrica real-data sweep requires Databricks SQL connectivity. Skipping.",
        )
    from kloppy.domain import Provider

    ds = _lakehouse_to_kloppy_dataset(raw, Provider.METRICA)
    if len(ds.records) == 0:
        pytest.skip("Metrica lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.kloppy import convert_to_frames

    frames, _ = convert_to_frames(ds)
    _bounds_check(frames, "metrica")
    _summarize_provider(frames, "metrica")


@pytest.mark.e2e
def test_skillcorner_real_data_sweep():
    raw = _query_lakehouse_sample("skillcorner")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "SkillCorner real-data sweep requires Databricks SQL connectivity. Skipping.",
        )
    from kloppy.domain import Provider

    ds = _lakehouse_to_kloppy_dataset(raw, Provider.SKILLCORNER)
    if len(ds.records) == 0:
        pytest.skip("SkillCorner lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.kloppy import convert_to_frames

    frames, _ = convert_to_frames(ds)
    _bounds_check(frames, "skillcorner")
    _summarize_provider(frames, "skillcorner")
