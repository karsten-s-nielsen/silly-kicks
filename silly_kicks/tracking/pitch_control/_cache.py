"""Per-pass pitch-control surface cache (TF-7 perf — shared surface).

``PitchControlCache`` memoizes *canonical* per-frame pitch-control surfaces so the
several enrichment families that need pitch control on overlapping frames compute
each surface once per ``(frame, team, method, params, ball_position, decompose)``
instead of once per family. It mirrors the ``links`` kwarg pattern: create one
cache per enrichment pass and thread it through the tracking aggregators; passing
the *same* cache across aggregators shares surfaces across feature families.

IMPORTANT — counterfactual safety. The cache is only valid for surfaces computed
on the *original* tracking frame. Counterfactual surfaces (a player removed or
moved, as in cover-shadow blocking or space-creation) share the canonical frame's
``(game_id, period_id, frame_id)`` but have different content — they must never be
routed through the cache; call ``compute_pitch_control`` directly for those.

No global state and not thread-safe: create one instance per pass and let it go
out of scope to free memory. See ADR-008 and
docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md.
"""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from ._dispatch import compute_pitch_control
from ._params import _METHOD_TO_PARAMS_TYPE, Method, PitchControlParams
from ._surface import PitchControlSurface


class PitchControlCache:
    """Memoizes canonical per-frame pitch-control surfaces within one pass.

    Examples
    --------
    Repeated queries on the same frame return the memoized surface::

        from silly_kicks.tracking.pitch_control import PitchControlCache
        cache = PitchControlCache()
        s1 = cache.surface(frame, attacking_team_id=1)   # computes
        s2 = cache.surface(frame, attacking_team_id=1)   # cache hit
        s1 is s2  # -> True
    """

    def __init__(self, maxsize: int | None = None) -> None:
        """``maxsize=None`` (default) is unbounded — byte-identical to the historical cache. A positive
        ``maxsize`` bounds retained surfaces with LRU eviction (ADR-103 F5): a whole-unit scorer can
        thus cap peak memory instead of accumulating one surface per distinct frame. ``decompose=True``
        surfaces are the per-surface memory driver (they retain per-player grids); pass
        ``decompose=False`` for the aggregate-only surface when per-player decomposition is not needed.
        """
        self._store: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def __len__(self) -> int:
        """Number of memoized surfaces (canonical frames only).

        The honest public observable for "was this cache actually shared?" -- callers
        threading a cache across feature families assert on it instead of reaching into
        the private store.

        Examples
        --------
        >>> from silly_kicks.tracking.pitch_control import PitchControlCache
        >>> cache = PitchControlCache()
        >>> len(cache)
        0
        """
        return len(self._store)

    def surface(
        self,
        frame: pd.DataFrame,
        attacking_team_id: int | str,
        *,
        method: Method = "spearman",
        params: PitchControlParams | None = None,
        decompose: bool = False,
        ball_position: tuple[float, float] | None = None,
    ) -> PitchControlSurface:
        """Return the (possibly cached) canonical surface for this frame + team.

        Identical in result to calling ``compute_pitch_control`` directly. Falls
        back to a direct (uncached) compute when a stable frame-identity key
        cannot be formed — e.g. the frame is not a single identifiable
        ``(game_id, period_id, frame_id)``.

        Examples
        --------
        Fetch a cached (or freshly computed) Voronoi surface for a frame::

            cache = PitchControlCache()
            surface = cache.surface(frame, 1, method="voronoi")
        """
        key = self._key(frame, attacking_team_id, method, params, decompose, ball_position)
        if key is not None and key in self._store:
            self._store.move_to_end(key)  # LRU: most-recently-used
            return self._store[key]
        surface = compute_pitch_control(
            frame,
            attacking_team_id,
            method=method,
            params=params,
            decompose=decompose,
            ball_position=ball_position,
        )
        if key is not None:
            self._store[key] = surface
            if self._maxsize is not None and len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict least-recently-used
        return surface

    def warm(
        self,
        frames: pd.DataFrame,
        requests: list,
        *,
        method: Method = "spearman",
        params: PitchControlParams | None = None,
    ) -> None:
        """Batch-populate the cache for a whole unit (ADR-103 F2/F5).

        ``requests`` is a list of ``((game_id, period_id, frame_id), attacking_team_id, decompose)``.
        Computes them via :func:`compute_pitch_control_batch` (one grouping pass, duplicates once) and
        stores each under its canonical key, so subsequent :meth:`surface` calls on those frames HIT.
        Honours ``maxsize`` (LRU). A cache miss on a later ``surface`` call is still served correctly.

        Examples
        --------
        Pre-populate a shared cache for a unit, then read surfaces back as hits::

            cache = PitchControlCache()
            cache.warm(frames, [((1, 1, 10), 1, True), ((1, 1, 11), 1, False)])
            s = cache.surface(frame_10, attacking_team_id=1, decompose=True)  # cache hit
        """
        from silly_kicks._frame_index import group_rows

        from ._dispatch import compute_pitch_control_batch

        if not requests:
            return
        surfaces = compute_pitch_control_batch(frames, requests, method=method, params=params)
        groups = group_rows(frames, ("game_id", "period_id", "frame_id"))
        for (frame_key, attacking_team_id, decompose), surface in zip(requests, surfaces, strict=True):
            frame = groups.get(*frame_key)
            key = self._key(frame, attacking_team_id, method, params, bool(decompose), None)
            if key is None:
                continue
            self._store[key] = surface
            self._store.move_to_end(key)
            if self._maxsize is not None and len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    @staticmethod
    def _key(
        frame: pd.DataFrame,
        attacking_team_id: int | str,
        method: Method,
        params: PitchControlParams | None,
        decompose: bool,
        ball_position: tuple[float, float] | None,
    ):
        """Build a hashable cache key, or None to bypass the cache.

        Returns None (compute uncached) when the method is unknown — so
        ``compute_pitch_control`` raises its own clear error — or when the frame
        does not resolve to a single ``(game_id, period_id, frame_id)``.
        """
        ptype = _METHOD_TO_PARAMS_TYPE.get(method)
        if ptype is None:
            return None  # invalid method -> bypass; let compute_pitch_control raise
        for col in ("game_id", "period_id", "frame_id"):
            if col not in frame.columns:
                return None
        gids = frame["game_id"].dropna().unique()
        pids = frame["period_id"].dropna().unique()
        fids = frame["frame_id"].dropna().unique()
        if len(gids) != 1 or len(pids) != 1 or len(fids) != 1:
            return None
        # Normalize None -> method default so a None caller and an explicit
        # default caller collide (compute_pitch_control treats them identically).
        params_norm = params if params is not None else ptype()
        try:
            hash(params_norm)
        except TypeError:
            return None
        return (
            gids[0],
            pids[0],
            int(fids[0]),
            attacking_team_id,
            method,
            params_norm,
            ball_position,
            bool(decompose),
        )
