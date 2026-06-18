"""Metrica bronze->frame builder (TF-23, ADR-034).

Pure builder consuming Metrica frame-level bronze (``bronze.metrica_tracking``: 0-1
normalized coords, JSON ``home_players``/``away_players``), parallel to
``tracking.skillcorner``. Metrica has NO ball z (``z=NaN`` is correct).

INPUT CONTRACT (verified against the kloppy oracle, TF-23): the bronze ``y`` must be in
**SPADL bottom-to-top** convention (y=0 bottom, y=1 top), so the builder's pure ``y*68``
standardization is canonical. kloppy's metrica NATIVE coordinate system is the OPPOSITE
(origin top-left, y top-to-bottom), so a consumer landing bronze directly from a kloppy
``TrackingDataset`` MUST flip y (``1 - y``) first --- the luxury-lakehouse bronze does
exactly this ("metrica y is already SPADL bottom-to-top"). Fed contract-honoring bronze,
the builder's frames match ``tracking.kloppy.convert_to_frames`` byte-for-byte (dx=dy=0
on Metrica open-data game 1; see tests/tracking/test_builder_kloppy_parity_e2e.py). 0-1 ->
SPADL 105x68 is a pure standardization with NO additional y-flip. See spec.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

# NOTE: Metrica does NOT import the SkillCorner nominal clock constant. Metrica sample
# games use MIXED raw clocks (some continuous, some period-relative --- verified on real
# bronze 2026-06-18), so the clock is rebased per-(period) min (see convert_to_frames).
from ._gk_identification import derive_goalkeepers
from .direction import orient_frames_to_ltr_by_geometry, require_et_direction
from .schema import METRICA_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .utils import _derive_speed, orient_frames_to_ltr

if TYPE_CHECKING:
    from .preprocess import PreprocessConfig

EXPECTED_INPUT_COLUMNS: tuple[str, ...] = (
    "period",
    "frame",
    "timestamp",
    "ball_x",
    "ball_y",
    "home_players",
    "away_players",
    "gk_jersey_numbers",
    "frame_rate",
)


def _to_player_tuples(raw: Any) -> list[tuple[str, float, float]]:
    """Parse one frame's player JSON blob to ``[(jersey, x, y), ...]`` (one json.loads/row).

    Malformed/positionless player entries are dropped. Acceptable here: Metrica is 3 frozen,
    hand-curated, well-formed sample games (no live feed); gross data loss would surface as a
    short ``total_output_rows`` in the report. (If a live Metrica feed is ever added, fold a
    dropped-player count into the report --- out of scope for the frozen public data.)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    d = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(d, dict):
        return []
    return [
        (str(j), float(c["x"]), float(c["y"])) for j, c in d.items() if isinstance(c, dict) and "x" in c and "y" in c
    ]


def _explode_team(bronze: pd.DataFrame, col: str, team_label: str) -> pd.DataFrame:
    """Vectorized JSON explode (NO iterrows): frame*players long-form for one team."""
    base = bronze[["frame", "period", "timestamp"]].reset_index(drop=True).copy()
    base["_pl"] = bronze[col].map(_to_player_tuples).to_numpy()  # one json.loads per row
    out = base.explode("_pl", ignore_index=True).dropna(subset=["_pl"])
    if out.empty:
        return out.assign(jersey=[], x=[], y=[], team_id=[]).drop(columns="_pl")
    out[["jersey", "x", "y"]] = pd.DataFrame(out["_pl"].tolist(), index=out.index)
    out["team_id"] = team_label
    return out.drop(columns="_pl")


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any = "Home",
    jersey_to_player_id: dict[str, dict[str, str]] | None = None,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert Metrica frame-level bronze to canonical SPADL frames.

    Parameters
    ----------
    bronze : pd.DataFrame
        Metrica tracking bronze; required columns: see ``EXPECTED_INPUT_COLUMNS``.
    home_team_id : Any, default "Home"
        Team label used for the home rows (Metrica is anonymized --- "Home"/"Away").
    jersey_to_player_id : dict[str, dict[str, str]] | None
        ``{"Home": {jersey: player_id}, "Away": {...}}`` from the consumer's roster;
        ``None`` => synthetic ``f"{team}_{jersey}"`` ids.
    output_convention, home_team_start_left, home_team_start_left_extratime, preprocess
        As in ``tracking.skillcorner.convert_to_frames``.

    Returns
    -------
    tuple[pd.DataFrame, TrackingConversionReport]

    Examples
    --------
    Build LTR frames from Metrica bronze::

        from silly_kicks.tracking import metrica
        frames, report = metrica.convert_to_frames(bronze_df, jersey_to_player_id=roster)
    """
    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in bronze.columns]
    if missing:
        raise ValueError(f"metrica.convert_to_frames: bronze missing column(s): {missing}")
    if home_team_start_left is not None and output_convention == "ltr":
        require_et_direction(bronze["period"], home_team_start_left_extratime, source="metrica convert_to_frames")

    roster = jersey_to_player_id or {}
    frame_rate = float(bronze["frame_rate"].iloc[0]) if "frame_rate" in bronze else 25.0
    gk_raw = bronze["gk_jersey_numbers"].dropna()
    gk_jerseys: set[str] = set()
    if not gk_raw.empty:
        parsed = json.loads(gk_raw.iloc[0]) if isinstance(gk_raw.iloc[0], str) else gk_raw.iloc[0]
        gk_jerseys = {str(j) for j in parsed} if parsed else set()

    src = bronze[~bronze["period"].isna()].copy()
    # --- vectorized player explode (NO iterrows) ---
    players = pd.concat(
        [_explode_team(src, "home_players", "Home"), _explode_team(src, "away_players", "Away")],
        ignore_index=True,
    )
    players = players.rename(columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds"})
    players["x"] = players["x"] * 105.0
    players["y"] = players["y"] * 68.0
    players["z"] = np.nan
    players["is_goalkeeper"] = False  # NO native GK seed for Metrica (see GK-derivation note below)
    players["is_ball"] = False
    players["visibility"] = None
    # roster map (team, jersey) -> player_id via a VECTORIZED merge (no per-row python loop);
    # synthetic f"{team}_{jersey}" fallback for unmapped jerseys.
    roster_df = pd.DataFrame(
        [(t, j, p) for t, d in roster.items() for j, p in d.items()],
        columns=["team_id", "jersey", "player_id"],
    )
    if not roster_df.empty:
        players = players.merge(roster_df, on=["team_id", "jersey"], how="left")
    else:
        players["player_id"] = np.nan
    players["player_id"] = players["player_id"].fillna(players["team_id"] + "_" + players["jersey"])
    players = players.drop(columns="jersey")

    # --- ball rows (one per frame; Metrica has no ball z) ---
    ball = src[["frame", "period", "timestamp", "ball_x", "ball_y"]].dropna(subset=["ball_x", "ball_y"]).copy()
    ball = ball.rename(
        columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds", "ball_x": "x", "ball_y": "y"}
    )
    ball["x"] = ball["x"] * 105.0
    ball["y"] = ball["y"] * 68.0
    ball["z"] = np.nan
    ball["player_id"] = None
    ball["team_id"] = None
    ball["is_goalkeeper"] = False
    ball["is_ball"] = True
    ball["visibility"] = None

    df = pd.concat([players, ball], ignore_index=True)
    df["game_id"] = "metrica"
    df["frame_id"] = df["frame_id"].astype(int)
    df["period_id"] = df["period_id"].astype(int)
    df["frame_rate"] = frame_rate
    df["source_provider"] = "metrica"
    df["ball_state"] = None
    df["team_attacking_direction"] = None
    df["confidence"] = None
    df["speed"] = np.nan
    df["speed_source"] = None
    # CLOCK: Metrica sample games use MIXED raw clocks (continuous vs period-relative ---
    # verified on real bronze 2026-06-18). Rebase per-(period) min so every period starts at
    # ~0 (ADR-017 period-relative; matches the kloppy Metrica gateway = the parity oracle AND
    # the Metrica events clock, which kloppy also emits period-relative). NOT the SkillCorner
    # nominal offset. NB: unlike SkillCorner (shared-constant structural guard), Metrica
    # frame<->event clock parity has NO structural test --- the event-anchored gate (Task 6) is
    # the SOLE guard (a mismatch -> zero links -> the gate's `len(res) >= 4` fails loud).
    df["time_seconds"] = df["time_seconds"] - df.groupby("period_id")["time_seconds"].transform("min")

    df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    df = _derive_speed(df)

    # Metrica is anonymized (Tier-2, ADR-007): gk_jersey_numbers is a FLAT list (verified:
    # e.g. ["11","25"]) with NO team split, so a native per-(team,jersey) GK flag is
    # unrecoverable --- a team-agnostic flag mis-assigns when teams share a number, and
    # derive_goalkeepers ORs (never clears, _gk_identification.py:163-169), so the mis-flag
    # would reach the orientation anchor. We seed NO native GK (is_goalkeeper already False)
    # and let the validated positional algorithm derive it; source is therefore always
    # "derived". (SkillCorner, by contrast, passes its authoritative per-player roster flag.)
    df, derived_picks = derive_goalkeepers(df)
    df["is_goalkeeper_source"] = None
    df.loc[~df["is_ball"], "is_goalkeeper_source"] = "derived"
    n_derived = len(derived_picks)
    # Observability cross-check (lakehouse "never silently substitute"): the flat list's total
    # count should match the derived GK count across teams; disagreement is surfaced, not hidden.
    derived_gk_count = sum(len(v) for v in derived_picks.values())
    if gk_jerseys and derived_gk_count != len(gk_jerseys):
        warnings.warn(
            f"metrica.convert_to_frames: derived {derived_gk_count} GK(s) but gk_jersey_numbers "
            f"lists {len(gk_jerseys)} --- positional GK derivation disagrees with the roster count.",
            stacklevel=2,
        )

    final = pd.DataFrame({c: df[c] for c in METRICA_TRACKING_FRAMES_COLUMNS})
    for c, dt in METRICA_TRACKING_FRAMES_COLUMNS.items():
        if dt == "bool":
            final[c] = final[c].astype("bool")
        elif dt in {"int64", "float64"}:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype(dt)  # type: ignore[arg-type]
        else:
            final[c] = final[c].astype(object)

    if output_convention == "ltr":
        if home_team_start_left is None:
            final = orient_frames_to_ltr_by_geometry(
                final, home_team_id=str(home_team_id), source="metrica", game_id="metrica"
            )
        else:
            final = orient_frames_to_ltr(
                final,
                home_team_id=str(home_team_id),
                home_team_start_left=home_team_start_left,
                home_team_start_left_extratime=home_team_start_left_extratime,
            )

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess

        cfg = resolve_preprocess(preprocess, provider="metrica")
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    report = TrackingConversionReport(
        provider="metrica",
        total_input_frames=int(bronze[["frame", "period"]].drop_duplicates().shape[0]),
        total_output_rows=len(final),
        n_periods=int(final["period_id"].nunique()),
        frame_coverage_per_period={int(p): 1.0 for p in final["period_id"].unique()},
        ball_out_seconds_per_period={},
        nan_rate_per_column={c: float(final[c].isna().mean()) for c in final.columns},
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
        n_teams_gk_derived=n_derived,
        derived_gk_picks=derived_picks,
    )
    return final, report
