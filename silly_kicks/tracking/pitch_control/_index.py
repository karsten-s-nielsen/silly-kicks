"""The columnar spearman frame index: every frame's players as contiguous arrays (spec 2026-11-08 tracking-batch §4.2).

One :class:`_SpearmanFrameIndex` holds the per-frame player rows of a batch -- the filter "not the ball, and ``x`` and
``y`` not NaN, in frame-row order" -- as contiguous arrays plus per-frame row offsets, and the first ball position of
each frame. It replaces the per-request pandas extraction of the spearman batch: the frames are filtered in ONE
vectorised pass, and extraction, ball resolution, and TTI then run once per distinct frame.

- :meth:`_SpearmanFrameIndex.from_groups` builds it from a ``group_rows`` grouping, one entry per frame key.
- :meth:`_SpearmanFrameIndex.from_slices` builds it from explicit frame slices and NEVER dedups by frame id: the
  cover-shadow batch scores a frame and its keeper-removed counterfactual, which share a frame key and differ in
  content (ADR-043).
- :meth:`_SpearmanFrameIndex.select` cuts a chunk of frames (spec W3: memory-bounded chunks over one index).

Element for element it equals the per-frame extraction of ``compute_spearman`` as at 4.139
(``_extract_frame_players``): positions and velocities upcast to float64 once (ADR-106), a NaN velocity becomes 0.0,
a frame without a ``vx`` column has zero velocity, and ``is_ball`` / ``is_goalkeeper`` go through ``astype(bool)``.
The team id of each frame is a pandas Series view in its source's own dtype, made on demand, because ``ids_match``
chooses its raw or canonical path from the content it sees (spec W2.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_COLUMNS = ["is_ball", "is_goalkeeper", "x", "y", "player_id", "team_id"]


@dataclass(frozen=True)
class _SpearmanFrameIndex:
    """The player rows of ``n_frames`` frames as contiguous arrays.

    Frame ``f`` owns rows ``offsets[f]:offsets[f+1]`` of ``pos``, ``vel`` and ``is_gk``. Its team ids and player ids
    are the rows from ``src_lo[f]`` of source part ``part[f]`` (one part per built frame set or slice), so a selected
    chunk shares its parent's sources.
    """

    pos: np.ndarray  # (n_rows, 2) float64
    vel: np.ndarray  # (n_rows, 2) float64, NaN -> 0.0
    is_gk: np.ndarray  # (n_rows,) bool
    offsets: np.ndarray  # (n_frames + 1,) int64
    ball_x: np.ndarray  # (n_frames,) float64: the first ball row's x, NaN when the frame has no ball row
    ball_y: np.ndarray  # (n_frames,) float64
    part: np.ndarray  # (n_frames,) int64: the source part of each frame
    src_lo: np.ndarray  # (n_frames,) int64: the frame's first row in its source part
    team_parts: tuple[pd.Series, ...]  # per source part: the kept rows' team ids, in the part's own dtype
    pid_parts: tuple[np.ndarray, ...]  # per source part: the kept rows' player ids

    @property
    def n_frames(self) -> int:
        return len(self.offsets) - 1

    def n_rows(self, frame: int) -> int:
        return int(self.offsets[frame + 1] - self.offsets[frame])

    def rows(self, frame: int) -> slice:
        return slice(int(self.offsets[frame]), int(self.offsets[frame + 1]))

    def _src(self, frame: int) -> tuple[int, slice]:
        lo = int(self.src_lo[frame])
        return int(self.part[frame]), slice(lo, lo + self.n_rows(frame))

    def team_ids(self, frame: int) -> pd.Series:
        """The frame's team ids: a view in its source's dtype (``ids_match`` decides its path on it)."""
        part, sl = self._src(frame)
        return self.team_parts[part].iloc[sl]

    def player_ids(self, frame: int) -> np.ndarray:
        part, sl = self._src(frame)
        return self.pid_parts[part][sl]

    def ball(self, frame: int) -> tuple[float, float] | None:
        """The frame's first ball row as ``(x, y)``; ``None`` when there is none or it is NaN."""
        bx, by = float(self.ball_x[frame]), float(self.ball_y[frame])
        return None if (np.isnan(bx) or np.isnan(by)) else (bx, by)

    @classmethod
    def from_groups(cls, groups, keys: Sequence[tuple] | None = None) -> _SpearmanFrameIndex:
        """One entry per frame key in ``keys`` (in that order; default every group), read from the ``group_rows``
        grouping ``groups``. A key that names no frame is an empty frame."""
        frame_keys: Sequence[tuple] = groups.keys() if keys is None else keys
        parts = [groups.positions(*key) for key in frame_keys]
        lengths = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
        frame = groups.frame
        taken = frame.take(np.concatenate(parts)) if parts else frame.iloc[:0]
        return cls._build(taken, lengths)

    @classmethod
    def from_slices(cls, slices: Sequence[pd.DataFrame]) -> _SpearmanFrameIndex:
        """One entry per slice, in order; two slices with the same frame key stay two entries."""
        if not slices:
            return cls._build(pd.DataFrame(columns=_COLUMNS), np.zeros(0, dtype=np.int64))
        return cls._concat([cls._build(s, np.array([len(s)], dtype=np.int64)) for s in slices])

    def select(self, frames: Sequence[int]) -> _SpearmanFrameIndex:
        """The sub-index of ``frames`` (in that order): their rows gathered contiguously, the sources shared. A frame
        number ``-1`` is an empty frame (no players, no ball)."""
        sel = np.asarray(frames, dtype=np.int64)
        real = sel >= 0
        safe = np.where(real, sel, 0)
        lengths = (
            np.where(real, self.offsets[safe + 1] - self.offsets[safe], 0)
            if self.n_frames
            else np.zeros(len(sel), np.int64)
        )
        row_index = (
            np.concatenate(
                [np.arange(int(self.offsets[f]), int(self.offsets[f + 1]), dtype=np.int64) for f in sel[real]]
            )
            if real.any()
            else np.zeros(0, dtype=np.int64)
        )
        nan = np.full(len(sel), np.nan)
        return _SpearmanFrameIndex(
            pos=np.ascontiguousarray(self.pos[row_index]),
            vel=np.ascontiguousarray(self.vel[row_index]),
            is_gk=self.is_gk[row_index],
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            ball_x=np.where(real, self.ball_x[safe], nan) if self.n_frames else nan,
            ball_y=np.where(real, self.ball_y[safe], nan) if self.n_frames else nan,
            part=np.where(real, self.part[safe], 0) if self.n_frames else np.zeros(len(sel), np.int64),
            src_lo=np.where(real, self.src_lo[safe], 0) if self.n_frames else np.zeros(len(sel), np.int64),
            team_parts=self.team_parts,
            pid_parts=self.pid_parts,
        )

    @classmethod
    def _build(cls, rows: pd.DataFrame, lengths: np.ndarray) -> _SpearmanFrameIndex:
        """The index of consecutive frames in ``rows`` (frame ``f`` is the next ``lengths[f]`` rows); one part."""
        n_frames = len(lengths)
        bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        is_ball = rows["is_ball"].astype(bool).to_numpy()
        x = rows["x"].to_numpy(dtype=np.float64)
        y = rows["y"].to_numpy(dtype=np.float64)
        keep = ~is_ball & ~np.isnan(x) & ~np.isnan(y)

        # ball: the first ball row of each frame (NaN when the frame has none; a NaN position stays NaN)
        ball_x = np.full(n_frames, np.nan)
        ball_y = np.full(n_frames, np.nan)
        ball_rows = np.flatnonzero(is_ball)
        if len(ball_rows) and n_frames:
            first = np.searchsorted(ball_rows, bounds[:-1], side="left")
            has = first < len(ball_rows)
            has[has] = ball_rows[first[has]] < bounds[1:][has]
            pick = ball_rows[first[has]]
            ball_x[has] = x[pick]
            ball_y[has] = y[pick]

        kept_rows = np.flatnonzero(keep)
        offsets = np.searchsorted(kept_rows, bounds, side="left").astype(np.int64)
        players = rows.iloc[kept_rows]
        pos = np.column_stack([x[kept_rows], y[kept_rows]]) if len(kept_rows) else np.zeros((0, 2))
        if "vx" in rows.columns:
            vel = players[["vx", "vy"]].to_numpy(dtype="float64")
        else:
            vel = np.zeros_like(pos)
        vel = np.nan_to_num(vel, nan=0.0)
        return cls(
            pos=np.ascontiguousarray(pos, dtype=np.float64),
            vel=np.ascontiguousarray(vel, dtype=np.float64),
            is_gk=np.ascontiguousarray(players["is_goalkeeper"].astype(bool).to_numpy(), dtype=bool),
            offsets=offsets,
            ball_x=ball_x,
            ball_y=ball_y,
            part=np.zeros(n_frames, dtype=np.int64),
            src_lo=offsets[:-1].copy(),
            team_parts=(players["team_id"],),
            pid_parts=(players["player_id"].to_numpy(),),
        )

    @classmethod
    def _concat(cls, parts: Sequence[_SpearmanFrameIndex]) -> _SpearmanFrameIndex:
        """Single-part indexes side by side; each keeps its own source part."""
        sizes = [int(p.offsets[-1]) for p in parts]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        offsets = np.concatenate([[0], *[p.offsets[1:] + s for p, s in zip(parts, starts, strict=True)]])
        return cls(
            pos=np.ascontiguousarray(np.concatenate([p.pos for p in parts], axis=0)),
            vel=np.ascontiguousarray(np.concatenate([p.vel for p in parts], axis=0)),
            is_gk=np.concatenate([p.is_gk for p in parts]),
            offsets=offsets.astype(np.int64),
            ball_x=np.concatenate([p.ball_x for p in parts]),
            ball_y=np.concatenate([p.ball_y for p in parts]),
            part=np.concatenate([np.full(p.n_frames, k, dtype=np.int64) for k, p in enumerate(parts)]),
            src_lo=np.concatenate([p.src_lo for p in parts]),
            team_parts=tuple(t for p in parts for t in p.team_parts),
            pid_parts=tuple(a for p in parts for a in p.pid_parts),
        )
