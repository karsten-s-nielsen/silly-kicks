"""Per-pass structural primitives (TF-45): Line Bypass Score, Space Gain Metric,
Structural Disruption Index.

Quantifies how a pass deforms the opponent's defensive structure, independent of
outcome value. Library ships RAW primitives only; the TIV z-norm composite,
K-means archetypes, and passer/receiver rankings are corpus-level and live with
consumers (mirrors the frozen-exogenous-xT decision, ADR-009).

INVARIANT: post-normalization SPADL action coords (start_x/start_y, end_x/end_y;
acting team attacks +x) and LTR tracking coords (home attacks +x) share the
[0,105]x[0,68] pitch frame. Defenders are mirrored (105-x, 68-y) into the action's
attack-positive frame iff the acting team is the AWAY team. We mirror DEFENDERS
(not the action coords as _line_breaking.py:243-252 does) on purpose: LBS is only
clean in attack-positive coords (otherwise the inequality flips sign per team);
SGM/SDI are isometry-invariant so the direction does not matter for them.

CAVEAT (see NOTICE): LBS is purely 1-D along the attacking axis -- a defender whose
d_x in (start_x, end_x] is counted even if he is on the opposite touchline.
Receiver location x_r is the pass DESTINATION (end_x/end_y); SPADL has no
receiver_player_id.

See spec docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._id_compat import ids_match, same_id


@dataclass(frozen=True)
class StructuralPassParams:
    """Tunable parameters for structural-pass metrics.

    sigma: defender spatial-influence radius (m) for the SGM Gaussian density.
    Default 15.0 -- empirically tuned (2,466 real WC2022 passes): smallest sigma at
    which the faithful 1/rho is intrinsically bounded by pitch geometry (no
    eps-floor). See scripts/tune_structural_pass_sigma.py + spec D1. No is_default()
    (matches CoverShadowParams / LineBreakingParams).
    """

    sigma: float = 15.0


def _structural_pass_core(
    defenders_xy: np.ndarray,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    sigma: float,
) -> tuple[float, float, float]:
    """Pure structural-pass math. defenders_xy is (n,2) in the acting-attack-positive
    frame; passer_xy=(start_x,start_y), receiver_xy=(end_x,end_y) in the same frame.

    Returns (structural_lbs, structural_sgm, structural_sdi).
    0 defenders -> (nan, nan, nan) (only degenerate case: rho=0 / centroid undefined).
    >=1 defender -> all numeric. structural_lbs is an int-valued float (count).
    """
    d = np.asarray(defenders_xy, dtype="float64")
    if d.ndim != 2 or d.shape[0] == 0:
        return (np.nan, np.nan, np.nan)

    p = np.asarray(passer_xy, dtype="float64")
    r = np.asarray(receiver_xy, dtype="float64")

    # LBS: defenders with start_x < d_x <= end_x (forward-only by construction)
    lbs = float(np.count_nonzero((d[:, 0] > p[0]) & (d[:, 0] <= r[0])))

    # SGM: inverse Gaussian density (available space), receiver minus passer
    two_s2 = 2.0 * sigma * sigma
    rho_p = np.exp(-((d - p) ** 2).sum(axis=1) / two_s2).sum()
    rho_r = np.exp(-((d - r) ** 2).sum(axis=1) / two_s2).sum()
    sgm = (1.0 / rho_r) - (1.0 / rho_p)

    # SDI: distance-from-defensive-centroid, receiver minus passer
    c = d.mean(axis=0)
    sdi = float(np.hypot(r[0] - c[0], r[1] - c[1]) - np.hypot(p[0] - c[0], p[1] - c[1]))

    return (lbs, float(sgm), sdi)


def compute_structural_pass_metrics(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    params: StructuralPassParams | None = None,
) -> dict[str, float]:
    """Per-frame structural-pass metrics for ONE linked frame.

    Schema-agnostic: passer_xy/receiver_xy passed explicitly (re-exportable to atomic).
    Defenders = opponent outfield (~is_ball FIRST, then ~ids_match(team, attacking)
    & ~is_goalkeeper). Defenders mirrored into the acting-attack-positive frame iff
    the acting team is AWAY, then handed to _structural_pass_core.

    See NOTICE for full bibliographic citations.
    """
    if params is None:
        params = StructuralPassParams()

    nan_out = {"structural_lbs": np.nan, "structural_sgm": np.nan, "structural_sdi": np.nan}
    if frame is None or len(frame) == 0:
        return dict(nan_out)
    if not (
        np.isfinite(passer_xy[0])
        and np.isfinite(passer_xy[1])
        and np.isfinite(receiver_xy[0])
        and np.isfinite(receiver_xy[1])
    ):
        return dict(nan_out)

    players = frame[~frame["is_ball"].astype(bool)]
    opp = players[
        ~ids_match(players["team_id"], attacking_team_id).to_numpy() & ~players["is_goalkeeper"].astype(bool).to_numpy()
    ]
    dx = opp["x"].to_numpy(dtype="float64")
    dy = opp["y"].to_numpy(dtype="float64")
    ok = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = dx[ok], dy[ok]
    if dx.size == 0:
        return dict(nan_out)

    # Mirror defenders into the acting team's attack-positive frame iff AWAY.
    if not same_id(attacking_team_id, home_team_id):
        dx, dy = 105.0 - dx, 68.0 - dy
    defenders_xy = np.column_stack([dx, dy])

    lbs, sgm, sdi = _structural_pass_core(defenders_xy, passer_xy, receiver_xy, params.sigma)
    return {"structural_lbs": lbs, "structural_sgm": sgm, "structural_sdi": sdi}
