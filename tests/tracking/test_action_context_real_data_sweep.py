"""e2e real-data sweep for the 4 PR-S20 action_context features.

Mirrors PR-S19's ``tests/test_tracking_real_data_sweep.py`` shape: PFF
loads from a local env-pointed directory (``PFF_TRACKING_DIR``); IDSSE /
Metrica / SkillCorner pull a single-match wide-form sample from the
lakehouse via Databricks SQL (``DATABRICKS_HOST`` / ``DATABRICKS_HTTP_PATH``
/ ``DATABRICKS_TOKEN``). For each provider:

  - convert frames to silly_kicks long-form
  - synthesize / pull actions
  - run ``add_action_context``
  - assert per-provider bounds on the 4 features
  - emit a JSON summary to stdout

Skipped in CI; run locally before tagging::

    pytest tests/tracking/test_action_context_real_data_sweep.py -m e2e -s

Each test ``pytest.skip``s with an explicit reason on missing env / data
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"


def _bounds_check(enriched: pd.DataFrame, provider: str) -> None:
    """Assert per-feature bounds AND that each feature is actually exercised
    (n_valid > 0). Without the n_valid > 0 assertion, a feature could come
    back all-NaN and the bounds check would silently pass — defeating the
    point of e2e validation."""
    nearest = enriched["nearest_defender_distance"].dropna()
    assert len(nearest) > 0, f"{provider}: nearest_defender_distance not exercised (all NaN)"
    assert (nearest >= 0).all(), f"{provider}: negative nearest_defender_distance"
    assert (nearest <= 200).all(), f"{provider}: nearest_defender_distance > 200 m"

    speed = enriched["actor_speed"].dropna()
    assert len(speed) > 0, f"{provider}: actor_speed not exercised (all NaN)"
    assert (speed >= 0).all(), f"{provider}: negative actor_speed"
    # Upper bound is permissive: 100 m/s tolerates the broadcast-tracking
    # artifacts in SkillCorner (PR-S19 baselines show speed_ms_p99 ~30 m/s
    # for SkillCorner, with a tail of frame-to-frame coordinate jumps that
    # produce spurious finite-difference speeds). The bound exists to catch
    # unit-conversion / sign errors, not validate physical plausibility.
    assert (speed <= 100).all(), f"{provider}: actor_speed > 100 m/s (likely a unit bug)"

    rz = enriched["receiver_zone_density"].dropna()
    assert len(rz) > 0, f"{provider}: receiver_zone_density not exercised (all NaN)"
    assert (rz >= 0).all(), f"{provider}: negative receiver_zone_density"

    dt = enriched["defenders_in_triangle_to_goal"].dropna()
    assert len(dt) > 0, f"{provider}: defenders_in_triangle_to_goal not exercised (all NaN)"
    assert (dt >= 0).all(), f"{provider}: negative defenders_in_triangle_to_goal"

    link_rate = enriched["frame_id"].notna().mean()
    assert link_rate >= 0.95, f"{provider}: link_rate {link_rate:.2f} below 0.95"


def _summarize(enriched: pd.DataFrame, provider: str) -> dict[str, Any]:
    """Print + return distribution stats for the 4 features."""
    summary: dict[str, Any] = {"provider": provider, "n_actions": len(enriched)}
    for col in (
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
    ):
        vals = enriched[col].dropna()
        if len(vals):
            summary[f"{col}_p50"] = float(np.percentile(vals, 50))
            summary[f"{col}_p99"] = float(np.percentile(vals, 99))
            summary[f"{col}_n_valid"] = len(vals)
        else:
            summary[f"{col}_p50"] = None
            summary[f"{col}_p99"] = None
            summary[f"{col}_n_valid"] = 0
    summary["link_rate"] = float(enriched["frame_id"].notna().mean())
    print(f"\n[action-context-sweep:{provider}] {json.dumps(summary)}")
    return summary


@pytest.mark.e2e
def test_pff_action_context_sweep() -> None:
    """PFF: load 1 match from PFF_TRACKING_DIR, build long-form via the PFF
    adapter, synthesize actions from frames at controlled times, run
    add_action_context, assert bounds + emit summary."""
    path = os.environ.get("PFF_TRACKING_DIR")
    if not path:
        pytest.skip("PFF_TRACKING_DIR not set; skipping PFF action_context sweep.")
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
            if home_team_id_value is None and isinstance(game_event.get("home_team"), dict):
                home_team_id_value = game_event.get("home_team", {}).get("id")
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
        pytest.skip("PFF action_context sweep: no parseable rows.")

    raw = pd.DataFrame(rows)
    raw["player_id"] = raw["player_id"].astype("Int64")
    raw["team_id"] = raw["team_id"].astype("Int64")

    from silly_kicks.tracking.features import add_action_context
    from silly_kicks.tracking.pff import convert_to_frames as pff_convert
    from silly_kicks.tracking.utils import _derive_speed

    frames, _ = pff_convert(raw, home_team_id=1, home_team_start_left=True)
    # PFF has speed_native_supplied=false (per PR-S19 baselines); the adapter
    # leaves speed=NaN, and finite-difference derivation is the caller's
    # responsibility per ADR-004 invariant 7.
    frames = _derive_speed(frames)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    _bounds_check(enriched, "pff")
    _summarize(enriched, "pff")


def _synthesize_actions_from_frames(frames: pd.DataFrame, n_actions: int = 10) -> pd.DataFrame:
    """Pick ``n_actions`` (period_id, frame_id, player_id) triples from the frames
    and stamp synthetic action rows whose actor + team match real frame data.

    The actor is a non-ball, non-goalkeeper player from the chosen frame; the
    action's start coords are the actor's frame x/y. End coords are anchored
    at the same point (no displacement) since this is a smoke test, not a
    pass / shot characterization.

    Skips the first 100 distinct frames so every synthesized action has prior
    frames available for finite-difference speed derivation (PFF + Metrica +
    SkillCorner have speed_native_supplied=false / partially derived; the
    first frame of the loaded slice has NaN speed because there's no t-1).
    Without this skip, actor_speed comes back NaN on every action and we
    aren't actually exercising the actor_speed feature.
    """
    candidates = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"])].copy()
    if len(candidates) < n_actions:
        candidates = frames[~frames["is_ball"]].copy()
    distinct_frames = candidates.drop_duplicates(["period_id", "frame_id"]).reset_index(drop=True)
    if len(distinct_frames) > n_actions + 100:
        # Skip the first 100 distinct frames; sample from frames 100..100+n_actions.
        skip_keys = set(
            zip(
                distinct_frames["period_id"].head(100).to_numpy(),
                distinct_frames["frame_id"].head(100).to_numpy(),
                strict=False,
            ),
        )
        per_row_keys = list(zip(candidates["period_id"], candidates["frame_id"], strict=False))
        keep = [key not in skip_keys for key in per_row_keys]
        candidates = candidates.loc[keep].reset_index(drop=True)
    sample = candidates.drop_duplicates(["period_id", "frame_id"]).head(n_actions)
    return pd.DataFrame(
        {
            "action_id": list(range(1, len(sample) + 1)),
            "period_id": sample["period_id"].to_numpy(),
            "time_seconds": sample["time_seconds"].to_numpy(),
            "team_id": sample["team_id"].to_numpy(),
            "player_id": sample["player_id"].to_numpy(),
            "start_x": sample["x"].to_numpy(),
            "start_y": sample["y"].to_numpy(),
            "end_x": sample["x"].to_numpy(),
            "end_y": sample["y"].to_numpy(),
        }
    )


def _databricks_connect():
    try:
        from databricks import sql  # type: ignore[import-not-found]
    except ImportError:
        return None
    raw_host = os.environ.get("DATABRICKS_HOST") or os.environ.get(
        "DATABRICKS_SERVER_HOSTNAME",
        "",
    )
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path and not http_path.startswith("/"):
        http_path = "/" + http_path
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
    conn = _databricks_connect()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT match_id
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                GROUP BY match_id
                ORDER BY match_id LIMIT 1
                """,
            )
            r = cur.fetchone()
            if r is None:
                return pd.DataFrame()
            match_id_value = r[0]
            cur.execute(
                f"""
                SELECT min(frame) AS min_f, max(frame) AS max_f
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                  AND match_id = '{match_id_value}'
                  AND period = 1
                """,
            )
            f_row = cur.fetchone()
            if f_row is None or f_row[0] is None:
                return pd.DataFrame()
            min_f = int(f_row[0])
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
                """,
            )
            description = cur.description or []
            cols = [d[0] for d in description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame.from_records(rows, columns=cols)


@pytest.mark.e2e
def test_idsse_action_context_sweep() -> None:
    raw = _query_lakehouse_sample("idsse")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "IDSSE action_context sweep requires Databricks SQL connectivity. Skipping.",
        )
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_sportec_input

    sportec_input = lakehouse_to_sportec_input(raw)
    sportec_input = sportec_input.dropna(subset=["x_centered", "y_centered"])
    if len(sportec_input) == 0:
        pytest.skip("IDSSE lakehouse sample yielded no usable rows after cleanup.")

    from silly_kicks.tracking.features import add_action_context
    from silly_kicks.tracking.sportec import convert_to_frames as sportec_convert

    home_team_str = str(sportec_input.loc[~sportec_input["is_ball"], "team_id"].dropna().iloc[0])
    frames, _ = sportec_convert(sportec_input, home_team_id=home_team_str, home_team_start_left=True)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    _bounds_check(enriched, "sportec")
    _summarize(enriched, "sportec")


@pytest.mark.e2e
def test_metrica_action_context_sweep() -> None:
    raw = _query_lakehouse_sample("metrica")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "Metrica action_context sweep requires Databricks SQL connectivity. Skipping.",
        )
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from kloppy.domain import Provider  # type: ignore[reportMissingImports]

    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_kloppy_dataset

    ds = lakehouse_to_kloppy_dataset(raw, Provider.METRICA)
    if len(ds.records) == 0:
        pytest.skip("Metrica lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.features import add_action_context
    from silly_kicks.tracking.kloppy import convert_to_frames as kloppy_convert

    frames, _ = kloppy_convert(ds)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    _bounds_check(enriched, "metrica")
    _summarize(enriched, "metrica")


@pytest.mark.e2e
def test_skillcorner_action_context_sweep() -> None:
    raw = _query_lakehouse_sample("skillcorner")
    if raw is None or len(raw) == 0:
        pytest.skip(
            "SkillCorner action_context sweep requires Databricks SQL connectivity. Skipping.",
        )
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from kloppy.domain import Provider  # type: ignore[reportMissingImports]

    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_kloppy_dataset

    ds = lakehouse_to_kloppy_dataset(raw, Provider.SKILLCORNER)
    if len(ds.records) == 0:
        pytest.skip("SkillCorner lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.features import add_action_context
    from silly_kicks.tracking.kloppy import convert_to_frames as kloppy_convert

    frames, _ = kloppy_convert(ds)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    _bounds_check(enriched, "skillcorner")
    _summarize(enriched, "skillcorner")


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_position e2e smoke (per provider)
# ---------------------------------------------------------------------------


def _augment_actions_for_gk_pipeline(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the columns silly_kicks.spadl.utils.add_pre_shot_gk_context requires.

    Synthesized actions from _synthesize_actions_from_frames lack game_id, type_id,
    result_id, bodypart_id (since the PR-S20 e2e bounds-check doesn't need them).
    For PR-S21 GK pipeline we need them populated; default to pass / success / foot
    (no shots → GK columns will all be NaN; this is a structural smoke test).
    """
    from silly_kicks.spadl import config as spadlconfig

    out = actions.copy()
    if "game_id" not in out.columns:
        out["game_id"] = 1
    if "type_id" not in out.columns:
        out["type_id"] = spadlconfig.actiontype_id["pass"]
    if "result_id" not in out.columns:
        out["result_id"] = spadlconfig.result_id["success"]
    if "bodypart_id" not in out.columns:
        out["bodypart_id"] = spadlconfig.bodypart_id["foot"]
    return out


_GK_PIPELINE_EXTRA_COLUMNS = (
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "frame_id",
    "time_offset_seconds",
    "link_quality_score",
    "n_candidate_frames",
)


def _gk_pipeline_smoke(actions: pd.DataFrame, frames: pd.DataFrame, provider: str) -> None:
    """Run add_pre_shot_gk_context(actions, frames=frames) and assert structural properties.

    Synthesized actions are non-shots → GK columns all NaN; this validates wiring +
    lazy-import + 8 column emission, not value distributions. Real shot validation
    is covered by tests/tracking/test_kernels.py (Tier-1) and the slim per-row gate.
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context

    actions = _augment_actions_for_gk_pipeline(actions)
    enriched = add_pre_shot_gk_context(actions, frames=frames)
    missing = [c for c in _GK_PIPELINE_EXTRA_COLUMNS if c not in enriched.columns]
    assert not missing, f"{provider}: GK pipeline missing columns {missing}"
    # GK columns should be all-NaN (synthesized non-shots).
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        assert enriched[col].isna().all(), f"{provider}: {col} has non-NaN values on non-shot synthesized actions."
    # Provenance columns should be populated for at least 95% of rows (linkage rate).
    link_rate = enriched["frame_id"].notna().mean()
    assert link_rate >= 0.95, f"{provider}: GK pipeline link rate {link_rate:.2f} < 0.95"
    print(f"\n[gk-pipeline-smoke:{provider}] link_rate={link_rate:.3f}, n={len(enriched)}")


@pytest.mark.e2e
def test_pff_pre_shot_gk_pipeline_smoke() -> None:
    """PFF: load frames + synthesize non-shot actions; run full add_pre_shot_gk_context(frames=...)."""
    path = os.environ.get("PFF_TRACKING_DIR")
    if not path:
        pytest.skip("PFF_TRACKING_DIR not set; skipping PFF GK pipeline smoke.")
    pff_dir = Path(path)
    if not pff_dir.is_dir():
        pytest.skip(f"PFF_TRACKING_DIR={path!r} is not a directory; skipping.")
    matches = sorted(p for p in pff_dir.iterdir() if p.name.endswith(".jsonl.bz2"))
    if not matches:
        pytest.skip(f"No .jsonl.bz2 files in PFF_TRACKING_DIR={path!r}; skipping.")

    rows: list[dict[str, Any]] = []
    with bz2.open(matches[0], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 3000:
                break
            obj = json.loads(line)
            period_id = obj.get("period")
            if period_id is None:
                continue
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
    if not rows:
        pytest.skip("PFF GK pipeline smoke: no parseable rows.")
    raw = pd.DataFrame(rows)
    raw["player_id"] = raw["player_id"].astype("Int64")
    raw["team_id"] = raw["team_id"].astype("Int64")

    from silly_kicks.tracking.pff import convert_to_frames as pff_convert
    from silly_kicks.tracking.utils import _derive_speed

    frames, _ = pff_convert(raw, home_team_id=1, home_team_start_left=True)
    frames = _derive_speed(frames)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    _gk_pipeline_smoke(actions, frames, "pff")


@pytest.mark.e2e
def test_idsse_pre_shot_gk_pipeline_smoke() -> None:
    raw = _query_lakehouse_sample("idsse")
    if raw is None or len(raw) == 0:
        pytest.skip("IDSSE GK pipeline smoke requires Databricks SQL connectivity.")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_sportec_input

    sportec_input = lakehouse_to_sportec_input(raw)
    sportec_input = sportec_input.dropna(subset=["x_centered", "y_centered"])
    if len(sportec_input) == 0:
        pytest.skip("IDSSE lakehouse sample yielded no usable rows.")

    from silly_kicks.tracking.sportec import convert_to_frames as sportec_convert

    home_team_str = str(sportec_input.loc[~sportec_input["is_ball"], "team_id"].dropna().iloc[0])
    frames, _ = sportec_convert(sportec_input, home_team_id=home_team_str, home_team_start_left=True)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    _gk_pipeline_smoke(actions, frames, "sportec")


@pytest.mark.e2e
def test_metrica_pre_shot_gk_pipeline_smoke() -> None:
    raw = _query_lakehouse_sample("metrica")
    if raw is None or len(raw) == 0:
        pytest.skip("Metrica GK pipeline smoke requires Databricks SQL connectivity.")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from kloppy.domain import Provider  # type: ignore[reportMissingImports]

    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_kloppy_dataset

    ds = lakehouse_to_kloppy_dataset(raw, Provider.METRICA)
    if len(ds.records) == 0:
        pytest.skip("Metrica lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.kloppy import convert_to_frames as kloppy_convert

    frames, _ = kloppy_convert(ds)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    _gk_pipeline_smoke(actions, frames, "metrica")


@pytest.mark.e2e
def test_skillcorner_pre_shot_gk_pipeline_smoke() -> None:
    raw = _query_lakehouse_sample("skillcorner")
    if raw is None or len(raw) == 0:
        pytest.skip("SkillCorner GK pipeline smoke requires Databricks SQL connectivity.")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from kloppy.domain import Provider  # type: ignore[reportMissingImports]

    from tests.datasets.tracking._lakehouse_adapter import lakehouse_to_kloppy_dataset

    ds = lakehouse_to_kloppy_dataset(raw, Provider.SKILLCORNER)
    if len(ds.records) == 0:
        pytest.skip("SkillCorner lakehouse sample yielded no usable frames.")

    from silly_kicks.tracking.kloppy import convert_to_frames as kloppy_convert

    frames, _ = kloppy_convert(ds)
    actions = _synthesize_actions_from_frames(frames, n_actions=10)
    _gk_pipeline_smoke(actions, frames, "skillcorner")
