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

    def __init__(self) -> None:
        self._store: dict = {}

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
        return surface

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
