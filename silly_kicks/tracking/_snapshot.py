"""Convert per-event player-position snapshots to tracking frame schema.

Public API: snapshot_to_tracking_frames
Module: silly_kicks.tracking._snapshot
Spec: docs/superpowers/specs/2026-05-27-snapshot-to-tracking-frames-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SPEED_SOURCE_UNAVAILABLE, TRACKING_FRAMES_COLUMNS

#: BOTH teams are labelled with this ONE value on purpose: a snapshot shares its event's SPADL
#: action-LTR frame, so it is already action-LTR and the geometry layer must NEVER re-project it
#: (ADR-028). This is the accepted-convention case in ``validate_period_directions`` -- NOT the
#: rejected single-team self-contradiction (that guard raises only when ONE team carries both
#: directions in a period). Flipping to per-team directions reintroduces the ADR-028 mixed-frame
#: defect on all SB360 input. Pinned by ``test_snapshot_actions_are_never_reprojected``.
_SNAPSHOT_ATTACKING_DIRECTION = "ltr"


def snapshot_to_tracking_frames(
    snapshots: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-event player-position snapshots to tracking frame schema.

    Parameters
    ----------
    snapshots : pd.DataFrame
        One row per player per event. Required columns: action_id, team_id,
        is_goalkeeper, x, y. Optional: player_id (synthetic sequential int
        if absent). Coordinates must be in the current SPADL coordinate system.
    actions : pd.DataFrame
        SPADL actions DataFrame. Used to derive game_id, period_id,
        time_seconds, and ball position (start_x, start_y) per frame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (frames, links) where:
        - frames: 20-column TRACKING_FRAMES_COLUMNS schema, one synthetic
          frame per action that has snapshot data. Every row carries
          ``speed_source=SPEED_SOURCE_UNAVAILABLE``: a per-event freeze-frame
          has no per-player temporal history, so ``speed`` and the ``vx``/``vy``
          that ``derive_velocities`` would produce can never exist. Velocity
          consumers (e.g. ``add_das``) read that marker and degrade honestly
          rather than raising.
        - links: Pre-built pointer DataFrame matching the
          link_actions_to_frames output contract (action_id, frame_id,
          time_offset_seconds=0.0, n_candidate_frames=1,
          link_quality_score=1.0).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    One synthetic frame per action that has snapshot rows, plus links already pointing at it:

    >>> import pandas as pd
    >>> from silly_kicks.tracking import snapshot_to_tracking_frames
    >>> snapshots = pd.DataFrame(
    ...     {
    ...         "action_id": [0, 0],
    ...         "team_id": [1, 2],
    ...         "is_goalkeeper": [False, True],
    ...         "x": [52.5, 3.0],
    ...         "y": [34.0, 34.0],
    ...     }
    ... )
    >>> actions = pd.DataFrame(
    ...     {
    ...         "action_id": [0],
    ...         "game_id": [7],
    ...         "period_id": [1],
    ...         "time_seconds": [12.5],
    ...         "start_x": [52.5],
    ...         "start_y": [34.0],
    ...     }
    ... )
    >>> frames, links = snapshot_to_tracking_frames(snapshots, actions)
    >>> len(frames), links["link_quality_score"].tolist()
    (3, [1.0])

    The third frame row is the ball, placed at the action's ``start_x``/``start_y``. Every
    player row carries ``speed_source=SPEED_SOURCE_UNAVAILABLE`` -- a freeze-frame has no
    temporal history, so velocity consumers degrade rather than raise.
    """
    # --- empty input ---
    if len(snapshots) == 0:
        return _empty_frames(), _empty_links()

    # --- action metadata lookup ---
    action_meta = actions[["action_id", "game_id", "period_id", "time_seconds", "start_x", "start_y"]].copy()
    action_ids_with_data = snapshots["action_id"].unique()
    action_meta = action_meta[action_meta["action_id"].isin(action_ids_with_data)]

    if len(action_meta) == 0:
        return _empty_frames(), _empty_links()

    # --- player rows ---
    has_player_id = "player_id" in snapshots.columns
    player = snapshots.merge(
        action_meta[["action_id", "game_id", "period_id", "time_seconds"]],
        on="action_id",
        how="inner",
    )

    if not has_player_id:
        # Synthetic sequential int per frame
        player = player.copy()
        player["player_id"] = np.arange(len(player))

    player_frames = pd.DataFrame(
        {
            "game_id": player["game_id"],
            "period_id": player["period_id"],
            "frame_id": player["action_id"],
            "time_seconds": player["time_seconds"],
            "frame_rate": np.nan,
            "player_id": player["player_id"],
            "team_id": player["team_id"],
            "is_ball": False,
            "is_goalkeeper": player["is_goalkeeper"],
            "x": player["x"],
            "y": player["y"],
            "z": np.nan,
            "speed": np.nan,
            # A snapshot is ONE synthesised frame per action: there is no second sample of
            # the same player to differentiate, so speed (and the vx/vy derived from the
            # same history) can never exist here -- structurally, not "not yet". Velocity
            # consumers read this marker to degrade honestly instead of either crashing or
            # silently absorbing a genuine forgotten-derive_velocities() bug. See
            # SPEED_SOURCE_UNAVAILABLE.
            "speed_source": SPEED_SOURCE_UNAVAILABLE,
            "ball_state": "alive",
            "team_attacking_direction": _SNAPSHOT_ATTACKING_DIRECTION,
            "confidence": np.nan,
            "visibility": np.nan,
            "source_provider": "snapshot",
            "is_goalkeeper_source": "native",
        }
    )

    # --- ball rows (one per frame) ---
    ball_frames = pd.DataFrame(
        {
            "game_id": action_meta["game_id"].values,
            "period_id": action_meta["period_id"].values,
            "frame_id": action_meta["action_id"].values,
            "time_seconds": action_meta["time_seconds"].values,
            "frame_rate": np.nan,
            "player_id": np.nan,
            "team_id": np.nan,
            "is_ball": True,
            "is_goalkeeper": False,
            "x": action_meta["start_x"].values,
            "y": action_meta["start_y"].values,
            "z": np.nan,
            "speed": np.nan,
            # A snapshot is ONE synthesised frame per action: there is no second sample of
            # the same player to differentiate, so speed (and the vx/vy derived from the
            # same history) can never exist here -- structurally, not "not yet". Velocity
            # consumers read this marker to degrade honestly instead of either crashing or
            # silently absorbing a genuine forgotten-derive_velocities() bug. See
            # SPEED_SOURCE_UNAVAILABLE.
            "speed_source": SPEED_SOURCE_UNAVAILABLE,
            "ball_state": "alive",
            "team_attacking_direction": _SNAPSHOT_ATTACKING_DIRECTION,
            "confidence": np.nan,
            "visibility": np.nan,
            "source_provider": "snapshot",
            "is_goalkeeper_source": "native",
        }
    )

    # --- combine, enforce column order, and enforce the declared DTYPES ---
    frames = pd.concat([player_frames, ball_frames], ignore_index=True)
    frames = frames[list(TRACKING_FRAMES_COLUMNS.keys())]
    frames = _cast_to_declared_schema(frames)

    # --- links ---
    links = pd.DataFrame(
        {
            "action_id": action_meta["action_id"].values,
            "frame_id": action_meta["action_id"].values,
            "time_offset_seconds": 0.0,
            "n_candidate_frames": 1,
            "link_quality_score": 1.0,
        }
    )

    return frames, links


#: Columns whose dtype is decided by the CALLER's id domain rather than by this port. Everything
#: else is the same in every declared variant, so it is cast to the base unconditionally.
_ID_COLUMNS = ("game_id", "player_id", "team_id")


def _cast_to_declared_schema(frames: pd.DataFrame) -> pd.DataFrame:
    """Emit the schema this port claims to emit, rather than whatever ``concat`` inferred.

    Selecting the 20 columns without applying their dtypes left ``player_id``/``team_id`` as
    whatever the concat produced -- and because the synthesized ball row is NA in both, a numpy-int
    source was upcast to ``float64``, i.e. ids became FLOATS. That is the shape ADR-019 records as
    rendering ``"366.0"`` against a clean ``"366"``. It was also resolver-dependent: a nullable
    ``Int64`` source stayed ``Int64`` on pandas 2.3.3 and was promoted to ``Float64`` on 3.0.5, so
    one input had two answers depending on the leg.

    This was unimplementable until ADR-058: the base declared a non-nullable ``int64`` for the two
    columns that are NA on the ball row BY CONSTRUCTION, so the cast raised on every snapshot. With
    the base at nullable ``Int64`` the port can satisfy its own declaration.

    **Identifier columns follow the caller's domain, using the two declarations ADR-058 already
    established -- no seventh variant.** Genuinely-string ids (the kloppy family's ``object``) are
    left alone; numeric ids take the base. Deciding per COLUMN rather than per frame is deliberate:
    a caller can legitimately pair a numeric ``game_id`` with string ``team_id``s, and picking one
    variant for the whole frame would corrupt one or the other.
    """
    for col, declared in TRACKING_FRAMES_COLUMNS.items():
        if col in _ID_COLUMNS and frames[col].dtype == object:
            continue  # genuine string ids -- KLOPPY_TRACKING_FRAMES_COLUMNS declares `object`
        if str(frames[col].dtype) != declared:
            # Routed through `pandas_dtype` because the schema stores dtypes as `str` and
            # pandas-stubs types `astype`'s parameter as a literal union, not `str`.
            frames[col] = frames[col].astype(pd.api.types.pandas_dtype(declared))
    return frames


def _empty_frames() -> pd.DataFrame:
    """Return an empty DataFrame with TRACKING_FRAMES_COLUMNS schema."""
    return pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in TRACKING_FRAMES_COLUMNS.items()})


def _empty_links() -> pd.DataFrame:
    """Return an empty links DataFrame matching link_actions_to_frames contract.

    Dtypes default to int64 for the empty case (no input to infer from).
    Matches the empty-return pattern in link_actions_to_frames (utils.py:163-170).
    """
    return pd.DataFrame(
        {
            "action_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_offset_seconds": pd.Series([], dtype="float64"),
            "n_candidate_frames": pd.Series([], dtype="int64"),
            "link_quality_score": pd.Series([], dtype="float64"),
        }
    )
