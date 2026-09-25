"""ADR-103 F1a: category dtypes are value-neutral + memory-lean, and the value_counts sites are
guarded against the category zero-count trap (D2-SPEC-09)."""

from __future__ import annotations

import pandas as pd

from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS

# ADR-103: only the STATIC set-once low-card columns are category (dynamic ones stay object).
_CATEGORY_COLS = ("ball_state", "source_provider", "is_goalkeeper_source")


def _make_frames(n_frames: int = 400, *, source_provider_dtype: str) -> pd.DataFrame:
    """A minimal-but-valid tracking-frame set: 2 outfield + 1 ball row per frame, one provider."""
    rows = []
    for fid in range(1, n_frames + 1):
        t = float(fid) * 0.04
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=10,
                team_id=1,
                is_ball=False,
                is_goalkeeper=True,
                x=8.0,
                y=34.0,
                z=0.0,
                speed=0.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                visibility=None,
                source_provider="gradientsports",
                is_goalkeeper_source="native",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=20,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=60.0,
                y=34.0,
                z=0.0,
                speed=0.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="rtl",
                visibility=None,
                source_provider="gradientsports",
                is_goalkeeper_source="native",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=pd.NA,
                team_id=pd.NA,
                is_ball=True,
                is_goalkeeper=False,
                x=30.0,
                y=34.0,
                z=0.0,
                speed=0.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                visibility=None,
                source_provider="gradientsports",
                is_goalkeeper_source="native",
            )
        )
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64")
    df["source_provider"] = df["source_provider"].astype(source_provider_dtype)  # type: ignore[arg-type]
    for c in _CATEGORY_COLS:
        if c != "source_provider":
            df[c] = df[c].astype("category")
    return df


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                time_seconds=2.0,
                team_id=1,
                player_id=10,
                start_x=8.0,
                start_y=34.0,
                end_x=30.0,
                end_y=34.0,
            )
        ]
    )


def test_confidence_dropped_and_low_card_cols_are_category():
    assert "confidence" not in TRACKING_FRAMES_COLUMNS
    for c in _CATEGORY_COLS:
        assert TRACKING_FRAMES_COLUMNS[c] == "category", c


def test_category_frames_are_materially_smaller():
    cat = _make_frames(source_provider_dtype="category")
    obj = cat.astype({c: "object" for c in _CATEGORY_COLS})
    # Per-column: each static low-cardinality column is far smaller as category than as object
    # (int codes + tiny dict vs one interned-string pointer per row). Asserted per-column because the
    # TOTAL ratio depends on how many object columns remain (the dynamic 3 + ids stay object).
    for c in _CATEGORY_COLS:
        cat_c = int(cat[c].memory_usage(deep=True))
        obj_c = int(obj[c].memory_usage(deep=True))
        assert cat_c * 5 < obj_c, (c, cat_c, obj_c)
    assert cat.memory_usage(deep=True).sum() < obj.memory_usage(deep=True).sum()


def test_per_provider_link_rate_has_no_zero_count_categories():
    """D2-SPEC-09: a `category` source_provider whose domain is WIDER than observed (the realistic
    multi-provider-concat case) would, without the utils.py:479 astype(object) guard, make
    value_counts inject {sportec:0.0, ...} into per_provider_link_rate. The guard collapses it."""
    frames = _make_frames(source_provider_dtype="object")
    domain = ["gradientsports", "sportec", "metrica", "skillcorner", "snapshot"]
    frames["source_provider"] = pd.Categorical(frames["source_provider"], categories=domain)
    assert len(frames["source_provider"].cat.categories) == 5  # domain wider than observed
    _pointers, report = link_actions_to_frames(_actions(), frames)
    # exactly the ONE observed provider — not the full 5-value category domain
    assert set(report.per_provider_link_rate) == {"gradientsports"}


def test_link_rate_identical_object_vs_category():
    """Value-neutrality: the LinkReport is byte-identical whether source_provider is object or category."""
    _p_o, rep_o = link_actions_to_frames(_actions(), _make_frames(source_provider_dtype="object"))
    _p_c, rep_c = link_actions_to_frames(_actions(), _make_frames(source_provider_dtype="category"))
    assert rep_o.per_provider_link_rate == rep_c.per_provider_link_rate
