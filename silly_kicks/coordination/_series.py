"""Raw coordination time-series (spec 7.12): relative phase / coupling angle / cluster amplitude.

A long, contract-free primitive expressed in team A's goal-relative frame (there are no windows, so no
reference-team flip). Composites, archetypes and rankings stay consumer-side (ADR-009).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from silly_kicks.coordination._catalog import METHODS_BY_LEVEL, PairSpec, resolve_pairs
from silly_kicks.coordination._columns import DEFAULT_LEVELS
from silly_kicks.coordination._compute import _amplitude_series, _cid, _cluster_inputs, _cluster_kernels, _context
from silly_kicks.coordination._kernels._vector_coding import coupling_angle_deg, stationary_mask
from silly_kicks.coordination._signals import CoordinationSignals

_PAIR_COLUMNS = (
    "game_id",
    "period_id",
    "segment_id",
    "time_s",
    "kind",
    "level",
    "signal_a",
    "signal_b",
    "axis",
    "team_a_id",
    "team_b_id",
    "player_a_id",
    "player_b_id",
    "value",
)
_CLUSTER_COLUMNS = ("game_id", "period_id", "segment_id", "time_s", "kind", "team_id", "axis", "value")


def compute_coordination_series(
    signals: CoordinationSignals,
    *,
    kind: Literal["relative_phase", "coupling_angle", "cluster_amplitude"],
    pairs: Sequence[PairSpec] | None = None,
) -> pd.DataFrame:
    """A long time-series table for one ``kind`` (raw primitive; no output contract, spec 7.12).

    Examples
    --------
    Raw per-sample series from a built ``signals`` object (needs a real match)::

        signals = build_coordination_signals(frames, windows=period_windows(frames))
        rp = compute_coordination_series(signals, kind="relative_phase")
        # rp has one row per sample per pair: columns time_s, level, signal_a, signal_b, value (degrees)
    """
    if kind == "cluster_amplitude":
        return _cluster_amplitude_series(signals)
    return _pair_series(signals, kind, pairs)


def _pair_series(signals, kind, pairs) -> pd.DataFrame:
    from silly_kicks.coordination._compute import _bindings_for  # local: avoid a public cycle

    params = signals.params
    fs = signals.fs
    # Admit only the pairs the OWNING method would score (A-51). Relative phase runs at every level; coupling angle is
    # vector coding's output, so it mirrors VC's static admission -- the level must list vector_coding (excludes dyads)
    # and the pair must be commensurate (`_compute.compute_vector_coding`'s `not_commensurate` refusal, spec 7.7).
    if kind == "relative_phase":
        specs = [p for p in resolve_pairs(pairs, DEFAULT_LEVELS) if "relative_phase" in METHODS_BY_LEVEL[p.level]]
    else:
        specs = [
            p
            for p in resolve_pairs(pairs, DEFAULT_LEVELS)
            if "vector_coding" in METHODS_BY_LEVEL[p.level] and p.commensurate
        ]
    rows: list[dict[str, Any]] = []
    for ps in signals.periods:
        t = ps.t
        # a synthetic period-spanning window row so binding resolution has a non-possession context
        wrow = pd.Series(
            {"window_kind": "period", "window_source": "period", "attacking_team_id": pd.NA, "window_id": 0}
        )
        for spec in specs:
            binds = _bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                continue
            for b in binds:
                ctx = _context(ps, spec, b, 0, t.size)
                seg_id = ps.segment_id[b.team_a_id] if spec.level != "dyad" else None
                for lo, hi in ctx.runs:
                    if kind == "relative_phase":
                        z = (ctx.sa * ctx.sb) * ctx.pha[lo:hi] * np.conj(ctx.phb[lo:hi])
                        vals = np.degrees(np.angle(z))
                        idxs = np.arange(lo, hi)
                    else:  # coupling_angle: consecutive diffs, stationary omitted
                        if hi - lo < 2:
                            continue
                        da = ctx.sa * np.diff(ctx.va[lo:hi])
                        db = ctx.sb * np.diff(ctx.vb[lo:hi])
                        keep = ~stationary_mask(
                            da, db, params.vc_epsilon.get(spec.signal_a, 0.0), params.vc_epsilon.get(spec.signal_b, 0.0)
                        )
                        vals = coupling_angle_deg(da[keep], db[keep])
                        idxs = (np.arange(lo + 1, hi))[keep]
                    for i, v in zip(idxs, vals, strict=False):
                        rows.append(_pair_series_row(ps, spec, b, kind, fs, int(i), float(v), seg_id))
    return pd.DataFrame(rows, columns=list(_PAIR_COLUMNS))


def _pair_series_row(ps, spec, b, kind, fs, i, value, seg_id) -> dict[str, Any]:
    return {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "segment_id": int(seg_id[i]) if seg_id is not None else pd.NA,
        "time_s": i / fs,
        "kind": kind,
        "level": spec.level,
        "signal_a": spec.signal_a,
        "signal_b": spec.signal_b,
        "axis": spec.axis,
        "team_a_id": _cid(b.team_a_id),
        "team_b_id": _cid(b.team_b_id),
        "player_a_id": _cid(b.player_a_id),
        "player_b_id": _cid(b.player_b_id),
        "value": value,
    }


def _cluster_amplitude_series(signals) -> pd.DataFrame:
    params = signals.params
    fs = signals.fs
    kern = _cluster_kernels()  # the same cluster arithmetic compute_cluster_phase scores with
    rows: list[dict[str, Any]] = []
    for ps in signals.periods:
        for tm in ps.team_ids:
            for axis in ("x", "y"):
                z, valid, _pids, _on = _cluster_inputs(ps, tm, axis, params)
                if z.shape[1] == 0:
                    continue
                _q, rel, usable = kern.cluster_phase(z, valid, params.min_players)
                amp = _amplitude_series(rel, valid, usable, ps.segments[tm], kern.window_cluster_stats)
                seg_id = ps.segment_id[tm]
                for i in np.flatnonzero(np.isfinite(amp)):
                    rows.append(
                        {
                            "game_id": ps.game_id,
                            "period_id": int(ps.period_id),
                            "segment_id": int(seg_id[i]),
                            "time_s": int(i) / fs,
                            "kind": "cluster_amplitude",
                            "team_id": _cid(tm),
                            "axis": axis,
                            "value": float(amp[i]),
                        }
                    )
    return pd.DataFrame(rows, columns=list(_CLUSTER_COLUMNS))
