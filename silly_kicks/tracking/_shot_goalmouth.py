"""Post-shot goalmouth crossing geometry (TF-48, ADR-030).

Fits the post-contact ball trajectory from tracking frames for shot actions and
derives the goal-plane crossing (y, z), kinematics, and provenance. Pure geometry,
no model. Engine is orientation-agnostic (goal ends from the GK map); output is
canonicalized to attacked-goal-at-x=105 (full point reflection x->105-x, y->68-y).
NOT for VAEP features (post-contact outcome leakage; see ADR-030 + guard test).

See NOTICE for full bibliographic citations (Anzer & Bauer 2021 -- xGOT lineage).
Spec: docs/superpowers/specs/2026-06-10-shot-goalmouth-psxg-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_G = 9.81
_FIELD_LENGTH = spadlconfig.field_length  # 105.0
_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_GOAL_Y_C = _FIELD_WIDTH / 2.0  # 34.0
_GOAL_HALF_MOUTH = 7.32 / 2.0  # 3.66
_BAR_Z = 2.44
# Contact-refinement search window, ASYMMETRIC (pilot-calibrated 2026-06-11): GS stamps
# shot events ~0.2-1.0 s BEFORE ball contact (wind-up, not strike -- measured on real
# WC2022 distance-to-plane trajectories), so the search extends further forward.
_REFINE_BEFORE_S = 0.3
_REFINE_AFTER_S = 1.5
_REFINE_SPEED_JUMP_MS = 3.0  # provisional noise floor for a "qualifying" speed increase
_PRE_SECONDS = 0.3  # pre-window pulled ONLY for contact refinement
# Velocity-estimation baseline (pilot-calibrated 2026-06-11): per-frame finite differences
# amplify position jitter by 1/dt -- at 29.97 fps a +-0.15 m jitter is +-4.5 m/s of velocity
# noise, which tripped the speed-drop/reversal/refinement checks on virtually EVERY real
# shot (while the 10 fps sweep copies worked -- the diagnostic that exposed this). All
# kinematic checks estimate velocity over >= this many seconds (multi-frame at high rates).
_MIN_VEL_BASELINE_S = 0.1
# PILOT NOTE: _REFINE_SPEED_JUMP_MS, the reversal floor (s1 > 1.0 in _grow_segment) and the
# engine's truncation slack (0.5 s) are deliberately MODULE constants, not params (speculative
# API surface is debt) -- but they ARE on the SB pilot's sensitivity checklist: at SkillCorner's
# 10 fps (~3 m/frame at 30 m/s) they mean different things than at 25 fps. Promote to
# ShotGoalmouthParams ONLY if the pilot shows per-corpus tuning is needed.

STANDARD_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty"))


@dataclass(frozen=True)
class ShotGoalmouthParams:
    """Tuning surface for the post-shot trajectory fit. Defaults are PROVISIONAL
    pending the SB-WC2022 pilot (spec section 10.4), incl. a per-frame-rate
    sensitivity row (SkillCorner 10 fps).

    Examples
    --------
    >>> ShotGoalmouthParams(post_window_seconds=1.5).post_window_seconds
    1.5
    """

    post_window_seconds: float = 3.5  # pilot-calibrated: real crossings land up to ~+2.8 s
    # after the (early) GS event stamp; 2.0 truncated them
    min_fit_frames: int = 3
    break_residual_m: float = 0.75
    break_speed_drop_frac: float = 0.5
    max_time_to_plane_seconds: float = 3.0
    # t* may exceed the PRODUCING segment's evidence span by at most this factor
    # (v4 pilot 2026-06-11: extrapolations past 3x their span had dy median 6.22 m,
    # max 41 m -- all junk; below 3x, 2.35 m. Joins the t* bounds family above.)
    max_extrapolation_leverage: float = 3.0
    # ONE ground band, deliberately shared: "rolling" classification AND the bounce
    # detector's z-at-flip ceiling (a bounce is by definition a near-ground event).
    rolling_z_max_m: float = 0.3
    bounce_min_dz_m: float = 0.25  # hysteresis: min drop-then-rise around a vz flip
    on_target_tolerance_m: float = 0.11  # ball radius; post/bar width folded in BY DECISION
    contact_refinement: bool = True

    def __post_init__(self) -> None:
        for name in (
            "post_window_seconds",
            "break_residual_m",
            "max_time_to_plane_seconds",
            "max_extrapolation_leverage",
            "bounce_min_dz_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"ShotGoalmouthParams.{name} must be > 0")
        for name in ("rolling_z_max_m", "on_target_tolerance_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"ShotGoalmouthParams.{name} must be >= 0")
        if not 0.0 < self.break_speed_drop_frac < 1.0:
            raise ValueError("ShotGoalmouthParams.break_speed_drop_frac must be in (0, 1)")
        if self.min_fit_frames < 2:
            raise ValueError("ShotGoalmouthParams.min_fit_frames must be >= 2")


def _collapse_held_samples(t, x, y, z):
    """Collapse runs of exactly-equal consecutive ball samples to their FIRST stamp.

    GS's raw ``balls`` channel is sample-and-hold upsampled: ~15 Hz positions emitted
    twice at 29.97 Hz stamps (measured 50% consecutive-duplicate x/y/z across ALL 127
    WC2022 pilot windows; raw-artifact-confirmed channel property, NOT a loader bug --
    ``ballsSmoothed`` is true ~30 Hz but x/y-divergent). A held duplicate is a phantom
    zero-velocity observation: it phase-modulates every baseline-velocity estimate (a
    5.5 m/s carry reads alternating ~3.7/~7.3 m/s, leaking through the flight gate)
    and saw-tooths the LS fits (~0.4 m structured residual at shot speed). Exact
    equality only -- a genuinely noisy stationary ball never repeats byte-identically
    (measured <=1% on clean ~30 Hz channels).
    """
    if len(t) < 2:
        return t, x, y, z
    z_same = (z[1:] == z[:-1]) | (np.isnan(z[1:]) & np.isnan(z[:-1]))
    same = (x[1:] == x[:-1]) & (y[1:] == y[:-1]) & z_same
    keep = np.concatenate([[True], ~same])
    return t[keep], x[keep], y[keep], z[keep]


def _ls_linear(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Least-squares ``v = a + b*t``. Requires len >= 2."""
    design = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(design, v, rcond=None)
    return float(a), float(b)


def _ls_ballistic_z(t: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Least-squares ``z = z0 + vz*t - 0.5*g*t^2`` (fixed g). Requires len >= 2."""
    zz = z + 0.5 * _G * t**2
    return _ls_linear(t, zz)


def _rmse_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    x0, vx = _ls_linear(t, x)
    y0, vy = _ls_linear(t, y)
    return float(np.sqrt(np.mean((x - (x0 + vx * t)) ** 2 + (y - (y0 + vy * t)) ** 2)))


def _trim_to_flight_core(ts, xs, ys, zs, goal_x, airborne_z_m):
    """Trim leading/trailing NON-FLIGHT samples: keep a sample only when its baseline
    approach speed is >= ``_FLIGHT_MIN_APPROACH_MS`` TOWARD the plane, or >= the same
    magnitude AWAY from it (own goals / wide mishits stay and report ``no_crossing``
    honestly), OR the ball is AIRBORNE there (finite z > ``airborne_z_m`` -- a chip/lob
    decelerates horizontally below the gate while climbing; a real WC2022 chip goal was
    trimmed to 2 samples, v3 pilot 2026-06-11; carries/frozen tails are on the GROUND).
    The slow/lateral GROUND middle ground is NOT shot flight: a ball carried diagonally
    can clear a scalar-speed bar while approaching at 3-6 m/s, and fitting that
    carry-line was the measured dominant >=3 m dy error class (9/13 on the pilot;
    a scalar-speed keep-branch admitted them -- removed 2026-06-11)."""
    n = len(ts)
    if n < 3:
        return ts, xs, ys, zs
    dist = np.abs(xs - goal_x)

    def _flying(i, j):  # samples i < j spanning >= the baseline
        for k in (i, j):  # airborne at either endpoint (head protects i=lo, tail j=hi)
            if np.isfinite(zs[k]) and zs[k] > airborne_z_m:
                return True
        dt = ts[j] - ts[i]
        appr = (dist[i] - dist[j]) / dt
        return abs(appr) >= _FLIGHT_MIN_APPROACH_MS

    lo, hi = 0, n - 1
    while hi - lo + 1 >= 2:
        j = lo + 1
        while j < hi and (ts[j] - ts[lo]) < _MIN_VEL_BASELINE_S:
            j += 1
        if (ts[j] - ts[lo]) < _MIN_VEL_BASELINE_S or _flying(lo, j):
            break
        lo += 1
    while hi - lo + 1 >= 2:
        j = hi - 1
        while j > lo and (ts[hi] - ts[j]) < _MIN_VEL_BASELINE_S:
            j -= 1
        if (ts[hi] - ts[j]) < _MIN_VEL_BASELINE_S or _flying(j, hi):
            break
        hi -= 1
    sl = slice(lo, hi + 1)
    return ts[sl] - ts[lo], xs[sl], ys[sl], zs[sl]


def _vel_at(t, x, y, i, *, back: bool):
    """Horizontal velocity (vx, vy, speed) at sample ``i``, estimated over a baseline of
    at least ``_MIN_VEL_BASELINE_S`` seconds looking back (``back=True``) or forward.
    Multi-frame at high rates -- per-frame finite differences amplify position jitter by
    1/dt (the 29.97 fps failure mode; see _MIN_VEL_BASELINE_S). Returns None when the
    requested baseline does not exist."""
    n = len(t)
    if back:
        j = i - 1
        while j > 0 and (t[i] - t[j]) < _MIN_VEL_BASELINE_S:
            j -= 1
        if j < 0 or (t[i] - t[j]) < _MIN_VEL_BASELINE_S:
            return None  # STRICT: a sub-baseline estimate would re-amplify jitter
        dt = t[i] - t[j]
        vx, vy = (x[i] - x[j]) / dt, (y[i] - y[j]) / dt
    else:
        j = i + 1
        while j < n - 1 and (t[j] - t[i]) < _MIN_VEL_BASELINE_S:
            j += 1
        if j >= n or (t[j] - t[i]) < _MIN_VEL_BASELINE_S:
            return None
        dt = t[j] - t[i]
        vx, vy = (x[j] - x[i]) / dt, (y[j] - y[i]) / dt
    return float(vx), float(vy), float(np.hypot(vx, vy))


def _grow_segment(t, x, y, z, goal_x, params):
    """Incremental fit from the first post-contact sample. Returns
    ``(end_idx_exclusive, end_reason, observed_straddle | None)``.

    ``observed_straddle`` is the index pair ``(i-1, i)`` when consecutive samples
    straddle the goal plane, so the caller can interpolate y AND z at the exact
    plane time. Kinematic checks use baseline velocities (``_vel_at``), never
    per-frame differences (noise amplification at high frame rates). The speed-drop
    break is SKIPPED while the ball is airborne (finite z above the rolling band):
    a chip/lob legitimately decelerates horizontally mid-flight (real WC2022 chip
    goal, v3 pilot 2026-06-11); mid-air deflections still end the segment via the
    residual + reversal checks.
    """
    n = len(t)
    sign0 = float(np.sign(x[min(2, n - 1)] - x[0])) if n >= 2 else 0.0  # initial x direction
    for i in range(1, n):
        # observed plane straddle between i-1 and i?
        if (x[i - 1] - goal_x) * (x[i] - goal_x) <= 0 and x[i] != x[i - 1]:
            return i + 1, "plane_crossed", (i - 1, i)
        if i >= 2:
            # residual of the newest sample vs a LOCAL fit (last _LOCAL_FIT_WINDOW_S
            # before i) -- ONLY once that window is MATURE (span >= _MIN_FIT_SPAN_S):
            # an immature 2-3-point fit plus real 30 fps jitter phantom-breaks
            # virtually every flight (pilot-measured; the 10 fps sweep resolving fine
            # was the tell -- 3x the per-frame SNR). The fit is LOCAL, not anchored at
            # the segment start: a full-segment linear fit diverges from any smoothly
            # CURVING flight (chip deceleration / curl) once the segment is long, so
            # the anchored residual phantom-broke real chips ~1 s in (v3 pilot
            # 2026-06-11); a deflection violates even the local fit, so the
            # save-semantics of the break are preserved. Windows shorter than
            # _LOCAL_FIT_WINDOW_S behave exactly as the old anchored check.
            w0 = int(np.searchsorted(t[:i], t[i] - _LOCAL_FIT_WINDOW_S))
            if (t[i - 1] - t[w0]) >= _MIN_FIT_SPAN_S:
                x0, vx = _ls_linear(t[w0:i], x[w0:i])
                y0, vy = _ls_linear(t[w0:i], y[w0:i])
                px, py = x0 + vx * t[i], y0 + vy * t[i]
                if np.hypot(x[i] - px, y[i] - py) > params.break_residual_m:
                    return i, "trajectory_break", None
            # baseline-velocity speed drop / direction reversal at sample i
            before = _vel_at(t, x, y, i - 1, back=True)
            after = _vel_at(t, x, y, i - 1, back=False)
            if before is None or after is None:
                continue
            (_, _, s0), (vx1, _, s1) = before, after
            airborne = np.isfinite(z[i - 1]) and z[i - 1] > params.rolling_z_max_m
            if not airborne and s0 > 0 and s1 < s0 * (1.0 - params.break_speed_drop_frac):
                return i, "trajectory_break", None
            # reversal vs the shot's OWN initial direction (NOT vs the goal: a shot that
            # starts away from the attacked plane -- own goal / mishit -- must not break
            # here; it falls through to the fit and resolves as no_crossing)
            if sign0 != 0 and vx1 * sign0 < 0 and s1 > 1.0:
                return i, "trajectory_break", None
    return n, "window_cap", None


# Flight-run anchoring (pilot diagnostics 2026-06-11): GS shot stamps precede the actual
# flight by up to ~2.6 s (and headers off crosses can FLY before the stamp), so a
# stamp-local velocity-step search often locks onto the assist pass. The robust anchor is
# the plane-approach run itself: contiguous samples approaching the goal plane at >=
# _FLIGHT_MIN_APPROACH_MS (baseline-velocity smoothed) with cumulative drop >=
# _FLIGHT_MIN_DROP_M; among candidate runs, the SHOT is the one ending NEAREST the plane
# (an assist ends at the shooter, 10+ m out) -- tie-broken by largest drop.
_FLIGHT_MIN_APPROACH_MS = 7.0
_FLIGHT_MIN_DROP_M = 3.0
# A run "reaches" the goal plane when its end sits within this of it (the ball got to the
# goal line). Among reaching runs the EARLIEST is the shot -- the first time the ball
# reaches the plane (ADR-030; fixes a real holdout goal whose in-mouth crossing the bare
# nearest-plane tie-break anchored PAST, 10511/1089). An assist ends 10+ m out (never
# reaches), so the closest-approach fallback still rejects it.
_FLIGHT_REACH_M = 2.0
_MIN_FIT_SPAN_S = 0.15  # residual break checks need a fit at least this mature
_LOCAL_FIT_WINDOW_S = 0.4  # residual break compares vs the LAST 0.4 s, not the full segment
# Contact anchoring (v2 pilot 2026-06-11, hardened v7): the GS event x/y is an exact
# ball-track point (median shooter-to-nearest-sample distance 0.0 m; 92% of pilot
# windows within 1 m). A sample is CONTACTABLE only when it is also at playable height
# (z NaN or <= _CONTACT_MAX_Z_M) -- a cross passing 6 m OVERHEAD of the shooter is not
# the ball at the shooter (the measured 12.6 m P2 goal: the window's only plane
# straddle was the pre-contact assist arc, 9.5 m wide of the mouth, behind the goal).
# _CONTACT_RADIUS_M anchors the fit start (2.0 m covers stamp jitter; worst case trims
# ~R/v ~ 0.1 s of flight); _CONTACT_EXIST_RADIUS_M is the looser EXISTENCE bar: a
# window whose ball never comes contactably near the stamped location provably does
# not contain the shot (insufficient_frames -- see _contact_clamp).
_CONTACT_RADIUS_M = 2.0
_CONTACT_EXIST_RADIUS_M = 5.0
_CONTACT_MAX_Z_M = 2.6
# Curve-aware y extrapolation (ADR-030; holdout class B, 5 chip-curl goals): the
# constant-velocity y fit extrapolates the goal-line crossing LINEARLY, missing a
# curling/dipping flight (measured 5.4 m on a real WC2022 chip-curl goal; DGX-confirmed
# the curl is real -- linear-fit RMSE 1.1-2.3 m collapses to 0.4-1.1 m under a quadratic).
# The quadratic is used ONLY when the producing segment is long enough to estimate
# curvature (>= _QUAD_MIN_FRAMES samples spanning >= _QUAD_MIN_SPAN_S) AND it markedly
# out-fits the line (linear RMSE above the jitter floor _QUAD_MIN_LIN_RMSE_M AND quad
# RMSE <= _QUAD_RMSE_RATIO x it -- real curl, never per-frame noise), and the crossing
# lever is TIGHTER than the linear cap (_QUAD_MAX_LEVERAGE x span; curvature error grows
# as t^2). Otherwise the byte-identical linear crossing stands (straight shots unchanged).
_QUAD_MIN_FRAMES = 6
_QUAD_MIN_SPAN_S = 0.5
_QUAD_MIN_LIN_RMSE_M = 0.5
_QUAD_RMSE_RATIO = 0.7
_QUAD_MAX_LEVERAGE = 1.5


def _extrapolate_crossing_y(fts, fys, y0, vy, t_star, span) -> float:
    """Crossing y at ``t_star``: the constant-velocity line by default; a span-gated
    quadratic when the producing segment supports a curvature estimate AND the quadratic
    markedly out-fits the line (real curl, not jitter -- ADR-030 chip-curl class)."""
    y_linear = float(y0 + vy * t_star)
    if len(fts) < _QUAD_MIN_FRAMES or span < _QUAD_MIN_SPAN_S:
        return y_linear
    if (t_star - float(fts[-1])) > _QUAD_MAX_LEVERAGE * span:
        return y_linear  # too far past the evidence to trust a curvature extrapolation
    lin_rmse = float(np.sqrt(np.mean((y0 + vy * fts - fys) ** 2)))
    if lin_rmse < _QUAD_MIN_LIN_RMSE_M:
        return y_linear  # no curvature signal above the per-frame jitter floor
    c2, c1, c0 = np.polyfit(fts, fys, 2)
    quad_rmse = float(np.sqrt(np.mean((np.polyval((c2, c1, c0), fts) - fys) ** 2)))
    if quad_rmse > _QUAD_RMSE_RATIO * lin_rmse:
        return y_linear  # quadratic does not markedly out-fit -> noise, not curl
    return float(c2 * t_star**2 + c1 * t_star + c0)


def _contact_clamp(t, x, y, z, contact_xy):
    """-> ``(exists, t_min_or_None)``. ``exists`` is the contact-existence verdict;
    ``t_min`` is the earliest legal fit start (the LAST contactable sample within
    ``_CONTACT_RADIUS_M`` + the baseline shift -- the ball leaving the shooter IS the
    shot contact; a cross + header is ONE continuous plane-approach run, so run
    selection alone fits the ASSIST: measured 7.3 m dy on a real WC2022 header goal).
    No 2 m-vicinity sample but existence satisfied (sloppy stamp within 5 m) -> no
    clamp, anchor falls to the flight-run/refine path unchanged."""
    d2 = np.hypot(x - contact_xy[0], y - contact_xy[1])
    playable = np.isnan(z) | (z <= _CONTACT_MAX_Z_M)
    if not bool(((d2 <= _CONTACT_EXIST_RADIUS_M) & playable).any()):
        return False, None
    near = np.where((d2 <= _CONTACT_RADIUS_M) & playable)[0]
    if len(near) == 0:
        return True, None
    i = int(near[-1])
    return True, float(t[i]) + (_MIN_VEL_BASELINE_S if i > 0 else 0.0)


def _find_flight_run(t, x, goal_x):
    """-> start index of the shot's flight run, or None. See the anchoring note above."""
    n = len(t)
    if n < 3:
        return None
    dist = np.abs(x - goal_x)
    # forward-baseline approach speed per sample (noise-robust at high frame rates)
    appr = np.full(n, np.nan)
    for i in range(n - 1):
        j = i + 1
        while j < n - 1 and (t[j] - t[i]) < _MIN_VEL_BASELINE_S:
            j += 1
        if (t[j] - t[i]) >= _MIN_VEL_BASELINE_S:
            appr[i] = (dist[i] - dist[j]) / (t[j] - t[i])
    runs = []
    start = None
    for i in range(n):
        fast = np.isfinite(appr[i]) and appr[i] >= _FLIGHT_MIN_APPROACH_MS
        if fast and start is None:
            start = i
        elif not fast and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    runs = [(a, b) for a, b in runs if dist[a] - dist[b] >= _FLIGHT_MIN_DROP_M]
    if not runs:
        return None
    # the shot is the FIRST run that actually REACHES the goal plane; only when none
    # reaches it (a purely extrapolated shot, samples stop short) does the closest-
    # approach rule decide -- which still rejects an assist pass ending 10+ m out.
    reaching = [r for r in runs if dist[r[1]] <= _FLIGHT_REACH_M]
    if reaching:
        return min(reaching, key=lambda r: r[0])[0]
    a, _b = min(runs, key=lambda r: (dist[r[1]], -(dist[r[0]] - dist[r[1]])))
    return a


def _refine_contact(t, x, y, goal_x, params) -> float:
    """First SHOT-CONSISTENT kinematic discontinuity in ``[-_REFINE_BEFORE_S,
    +_REFINE_AFTER_S]`` (asymmetric -- GS stamps shots before contact): a horizontal speed
    INCREASE >= ``_REFINE_SPEED_JUMP_MS`` whose post-discontinuity vx points toward the
    attacked goal. Largest-discontinuity selection is REJECTED by spec (a close-range
    save inside the window would win). Returns the refined t0 (0.0 if none qualifies).
    """
    if not params.contact_refinement or len(t) < 3:
        return 0.0
    toward = 1.0 if goal_x > 50.0 else -1.0
    in_win = np.where((t >= -_REFINE_BEFORE_S) & (t <= _REFINE_AFTER_S))[0]
    # FIRST shot-consistent goalward-velocity step over baseline velocities (an argmax
    # matched-filter variant was tried at the pilot and REGRESSED resolution 0.49 -> 0.10
    # -- it locks onto late steps and over-trims flight; reverted 2026-06-11).
    for i in in_win:
        before = _vel_at(t, x, y, i, back=True)
        after = _vel_at(t, x, y, i, back=False)
        if before is None or after is None:
            continue
        gw_before = before[0] * toward
        gw_after = after[0] * toward
        if gw_after - gw_before >= _REFINE_SPEED_JUMP_MS and gw_after > 0:
            # the forward baseline straddles contact: shift by the full baseline so the
            # segment start is provably post-contact (costs <= 0.1 s of flight)
            return float(t[i] + _MIN_VEL_BASELINE_S)
    return 0.0


def _classify_z(t, z, params):
    """-> ``(profile, z_seg_start_idx)``. profile in {"rolling", "airborne", "bounced"}
    or None (z unusable: all-NaN or < 2 finite samples). ``z_seg_start_idx`` = first
    index of the LATEST z-sub-segment (0 unless bounced).

    Bounce = finite-difference vz sign flip (- -> +) at sample k where
    (i) z[k] <= rolling_z_max_m (near ground) AND
    (ii) drop >= bounce_min_dz_m before k AND rise >= bounce_min_dz_m after k (hysteresis)
    -- a noisy airborne trajectory whose vz flips at height stays "airborne".

    UNREACHABILITY NOTE (ADR-030): a detected flip at k requires z[k-1], z[k], z[k+1]
    all finite (np.diff over NaN yields NaN; NaN comparisons are False), so a detected
    bounce always leaves >= 2 finite samples in the sub-segment -- the caller's
    ``f.sum() >= 2`` guard is defensive.
    """
    ok = np.isfinite(z)
    if ok.sum() < 2:
        return None, 0
    if np.nanmax(z) <= params.rolling_z_max_m:
        return "rolling", 0
    with np.errstate(invalid="ignore"):
        vz = np.diff(z) / np.diff(t)
    start = 0
    bounced = False
    for k in range(1, len(vz)):
        if not (vz[k - 1] < 0 <= vz[k]):
            continue
        if not np.isfinite(z[k]) or z[k] > params.rolling_z_max_m:
            continue  # flip at height = noise, not a bounce
        drop = np.nanmax(z[start : k + 1]) - z[k]
        rise = np.nanmax(z[k:]) - z[k]
        if drop >= params.bounce_min_dz_m and rise >= params.bounce_min_dz_m:
            bounced, start = True, k  # recurse to the LATEST bounce
    if bounced:
        sub = z[start:]
        if np.nanmax(sub) <= params.rolling_z_max_m:
            return "rolling", start  # degenerated to rolling
        return "bounced", start
    return "airborne", 0


def _fit_one_shot(t, x, y, z, *, goal_x, params, window_truncated=False, contact_xy=None) -> dict:
    """Fit one shot's ball samples (FRAME coords) and intersect the attacked goal
    plane. Returns a plain dict consumed only by :func:`compute_shot_goalmouth`
    (canonicalization is the engine's job). Per-column segment provenance (M-1):
    ``speed`` ALWAYS from the earliest (contact) sub-segment; crossing + fit-quality
    columns from the segment that PRODUCED the crossing.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(8) / 25.0
    >>> r = _fit_one_shot(t, 85 + 25 * t, 34 + 0 * t, 1 + 0 * t,
    ...                   goal_x=105.0, params=ShotGoalmouthParams())
    >>> r["source"]
    'extrapolated'
    """
    out = dict(
        crossing_y=np.nan,
        crossing_z=np.nan,
        speed=np.nan,
        time_to_goal_line=np.nan,
        source="no_ball_frames",
        end_reason=None,
        z_profile=None,
        n_fit_frames=0,
        fit_rmse=np.nan,
    )
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(t) == 0:
        return out
    t, x, y, z = _collapse_held_samples(t, x, y, z)
    t_clamp = None
    if contact_xy is not None and params.contact_refinement:
        exists, t_clamp = _contact_clamp(t, x, y, z, contact_xy)
        if not exists:
            # the ball never comes contactably near the stamped shot location: the
            # window provably does not contain the shot (the measured 12.6 m P2 goal
            # fitted a pre-contact assist arc) -- honest no-fit, never a wrong crossing
            out["source"] = "insufficient_frames"
            return out
    if params.contact_refinement:
        run_start = _find_flight_run(t, x, goal_x)
        if run_start is not None:
            # the run's forward-baseline start straddles contact (first qualifying sample
            # can sit up to one baseline BEFORE the strike); shift by the baseline so the
            # segment is provably in-flight (costs <= 0.1 s against the 3.5 s window).
            # No shift when the run begins at the window head -- nothing earlier exists
            # to bleed in, and the shift would just trim clean flight.
            t0 = float(t[run_start]) + (_MIN_VEL_BASELINE_S if run_start > 0 else 0.0)
        else:
            t0 = _refine_contact(t, x, y, goal_x, params)
        if t_clamp is not None:
            # the fit may not start before the ball LEAVES the shooter (cross + header
            # are one continuous approach run; dribble touches precede the strike)
            t0 = max(t0, t_clamp)
    else:
        t0 = 0.0
    post = t >= t0
    t, x, y, z = t[post] - t0, x[post], y[post], z[post]
    if len(t) < 2:
        out["source"] = "insufficient_frames"
        return out
    end, reason, straddle = _grow_segment(t, x, y, z, goal_x, params)
    if reason == "window_cap" and window_truncated:
        reason = "data_end"
    out["end_reason"] = reason
    ts, xs, ys, zs = t[:end], x[:end], y[:end], z[:end]

    observed = straddle is not None
    if not observed:
        # SEGMENT-CORE TRIM (pilot 2026-06-11): the fitted core must actually be FLYING.
        # Trim leading/trailing samples whose baseline approach speed toward the plane is
        # below flight speed -- a blocked shot's frozen tail or a t0 overshoot's slow head
        # would otherwise blend into the fit (the measured ~2.3 m dy bias mechanism).
        ts, xs, ys, zs = _trim_to_flight_core(ts, xs, ys, zs, goal_x, params.rolling_z_max_m)
        # re-anchor: grow's end indexes the PRE-trim segment; every downstream consumer
        # (n_fit_frames, bounce post_n, the seg slices) works on the trimmed arrays
        # (the pilot artifact reported n_fit_frames=19 on a 12-sample core)
        end = len(ts)
    if not observed and (len(ts) < params.min_fit_frames or (ts[-1] - ts[0]) < _MIN_FIT_SPAN_S):
        # an extrapolated crossing needs BOTH enough samples and enough TIME SPAN --
        # a sub-0.15 s core (e.g. blocked-dead flights) cannot anchor a plane crossing
        out["source"] = "insufficient_frames"
        return out

    # --- speed: ALWAYS the EARLIEST (contact) sub-segment (M-1) ---
    profile, zstart = _classify_z(ts, zs, params)
    out["z_profile"] = profile
    # pre-bounce sub-segment when it can support a 2-param fit; else the full segment
    # (graceful degradation, documented: a 1-sample pre-bounce cannot anchor a speed fit)
    contact_end = zstart if zstart >= 2 else len(ts)
    cts, cxs, cys = ts[:contact_end], xs[:contact_end], ys[:contact_end]
    if len(cts) >= 2:
        _, vx0 = _ls_linear(cts, cxs)
        _, vy0 = _ls_linear(cts, cys)
        speed_h = float(np.hypot(vx0, vy0))
        czs = zs[:contact_end]
        f0 = np.isfinite(czs)
        if profile is not None and f0.sum() >= 2:
            _, vz0 = _ls_ballistic_z(cts[f0], czs[f0])
            out["speed"] = float(np.hypot(speed_h, vz0))
        else:
            out["speed"] = speed_h  # 2D fallback (documented degradation)

    # --- crossing: the segment that PRODUCES it ---
    if observed:
        i, j = straddle
        frac = (goal_x - xs[i]) / (xs[j] - xs[i])
        out["crossing_y"] = float(ys[i] + frac * (ys[j] - ys[i]))
        t_star = float(ts[i] + frac * (ts[j] - ts[i]))
        if np.isfinite(zs[i]) and np.isfinite(zs[j]):
            out["crossing_z"] = max(float(zs[i] + frac * (zs[j] - zs[i])), 0.0)
        out["time_to_goal_line"] = t_star
        out["source"] = "observed"
        out["n_fit_frames"] = int(end)
        out["fit_rmse"] = _rmse_xy(ts, xs, ys) if end >= 2 else np.nan
        return out

    # extrapolated: pick the producing segment (post-bounce supersession, 3 branches;
    # branch 3 (< 2 finite z in the sub-segment) is unreachable by construction -- see
    # _classify_z docstring; the f.sum() >= 2 guard below is defensive)
    seg = slice(0, end)
    if profile == "bounced":
        post_n = end - zstart
        if post_n >= params.min_fit_frames:
            seg = slice(zstart, end)  # full x/y+z supersession
    fts, fxs, fys = ts[seg], xs[seg], ys[seg]
    x0, vx = _ls_linear(fts, fxs)
    y0, vy = _ls_linear(fts, fys)
    # a fit RAN -- populate the diagnostics even when no crossing results (R4b: consumers
    # must not read NA as "no fit ran"; NA stays reserved for the truly-unfitted sources)
    out["n_fit_frames"] = int(seg.stop - seg.start)
    out["fit_rmse"] = _rmse_xy(fts, fxs, fys)
    toward = 1.0 if goal_x > 50.0 else -1.0
    if vx * toward <= 0:
        out["source"] = "no_crossing"
        return out
    t_star = (goal_x - x0) / vx
    span = float(fts[-1] - fts[0])
    if (
        t_star <= float(fts[0])
        or t_star > params.max_time_to_plane_seconds
        or (t_star - float(fts[0])) > params.max_extrapolation_leverage * span
    ):
        out["source"] = "no_crossing"
        return out
    out["crossing_y"] = _extrapolate_crossing_y(fts, fys, y0, vy, t_star, span)
    out["time_to_goal_line"] = float(t_star)
    # crossing z by profile
    if profile == "rolling":
        out["crossing_z"] = float(np.nanmean(zs))
    elif profile in ("airborne", "bounced"):
        zseg = slice(zstart, end) if profile == "bounced" else seg
        zt, zv = ts[zseg], zs[zseg]
        # Z-ONSET TRIM (pilot 2026-06-11): the GS z channel can lag flight onset (median
        # 0.10 s but p75 0.80 s measured) -- a flat-zero prefix drags the ballistic fit to
        # the ground (the measured crossing-z collapse). Fit z from the first SUSTAINED
        # rise above the ground band onward; a sub-2-sample remainder leaves crossing_z
        # NaN (honest -- a wrong 0 is undetectable downstream, a NaN is imputable).
        zon = _z_onset(zv, params)
        if zon > 0:
            zt, zv = zt[zon:], zv[zon:]
        f = np.isfinite(zv)
        if f.sum() >= 2:
            z0c, vz = _ls_ballistic_z(zt[f], zv[f])
            out["crossing_z"] = max(float(z0c + vz * t_star - 0.5 * _G * t_star**2), 0.0)
        # else: < 2 finite z samples in the producing z-sub-segment -> crossing_z stays NaN
    out["source"] = "extrapolated"
    return out


def _z_onset(zv, params) -> int:
    """Index of the first SUSTAINED z rise above the ground band (this sample and the
    next both exceed ``rolling_z_max_m``), or 0 when z is credible from the start
    (first finite sample already above the band, or no sustained rise exists)."""
    fin = np.isfinite(zv)
    if not fin.any():
        return 0
    first = int(np.argmax(fin))
    if zv[first] > params.rolling_z_max_m:
        return 0  # z credible from the start -- no trim
    for k in range(first, len(zv) - 1):
        if (
            np.isfinite(zv[k])
            and np.isfinite(zv[k + 1])
            and zv[k] > params.rolling_z_max_m
            and zv[k + 1] > params.rolling_z_max_m
        ):
            return k
    return 0


def _contact_anchor(row, goal_x: float):
    """Shooter position in FRAME coords for the contact anchor, or None.

    Reads the action's OWN location (standard SPADL ``start_x``/``start_y``; atomic
    ``x``/``y``) -- action space is canonical attacked-at-105, so the point reflects
    into frame coords by the same point reflection the engine applies to outputs
    (orientation-agnosticism preserved; the engine still never reads
    ``team_attacking_direction``). NaN/absent coordinates -> None (ADR-003: the fit
    degrades to the un-anchored behavior, never crashes).
    """
    for cx_col, cy_col in (("start_x", "start_y"), ("x", "y")):
        if cx_col in row.index and cy_col in row.index:
            try:
                cx, cy = float(row[cx_col]), float(row[cy_col])
            except (TypeError, ValueError):
                return None
            if not (np.isfinite(cx) and np.isfinite(cy)):
                return None
            if goal_x <= 50.0:
                cx, cy = _FIELD_LENGTH - cx, _FIELD_WIDTH - cy
            return (cx, cy)
    return None


def _confidence(r: dict, params: ShotGoalmouthParams) -> float:
    """PROVISIONAL map (ADR-025 style; calibrated at the SB pilot -- spec sections 7/10).
    Inputs include z_profile + producing-segment size because a 2-sample z refit is
    exactly determined (RMSE == 0 would out-score an honest 5-point fit; spec L-1).
    The ONE non-provisional choice: observed (1.0) STRICTLY dominates any extrapolated
    score (capped 0.9).

    Examples
    --------
    >>> _confidence({"source": "observed"}, ShotGoalmouthParams())
    1.0
    """
    if r["source"] == "observed":
        return 1.0
    if r["source"] != "extrapolated":
        return 0.0
    n_term = min(r["n_fit_frames"] / 8.0, 1.0)
    rmse = r["fit_rmse"]
    rmse_term = 1.0 / (1.0 + (rmse / params.break_residual_m if np.isfinite(rmse) else 1.0))
    z_term = 1.0 if r["z_profile"] in ("airborne", "rolling") else (0.7 if r["z_profile"] == "bounced" else 0.5)
    return float(np.clip(0.9 * n_term * rmse_term * z_term, 0.0, 0.9))


def compute_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
    shot_type_ids: frozenset[int] = STANDARD_SHOT_TYPE_IDS,
) -> pd.DataFrame:
    """PURE engine: index-aligned TF-48 output frame; never mutates ``actions``, emits no
    warnings (edge policy lives in ``add_shot_goalmouth``). Orientation-agnostic: goal ends
    come from the GK map (``defended_goal_x``), output is canonicalized to
    attacked-goal-at-x=105 (full point reflection x->105-x, y->68-y); the engine never
    reads ``team_attacking_direction``. Consumed frame columns: game_id, period_id,
    frame_id, time_seconds, team_id, is_ball, is_goalkeeper, x, y, z. The action's own
    location (``start_x``/``start_y``; atomic ``x``/``y``) is consumed OPTIONALLY as the
    contact anchor (see ``_contact_anchor``; NaN/absent -> un-anchored fit).

    ``links`` is accepted for signature parity ONLY -- the ENGINE never reads it (the
    window is time-sliced via ``slice_around_event``, link-independent); the ``add_*``
    edge uses ``links`` solely for its provenance-column merge. ``shot_type_ids`` is the
    atomic-mirror seam ({shot, shot_penalty} there).

    Examples
    --------
    The output is index-aligned with ``actions``, so it joins straight back on. Every row is
    present -- non-shots and unfittable shots carry NaN / NA rather than being dropped::

        out = compute_shot_goalmouth(actions, frames)
        crossed = out[out["shot_crossing_source"].notna()]
        crossed[["shot_crossing_y", "shot_crossing_z", "shot_on_target_derived"]]

    Read the provenance before the geometry, because a crossing point is only as good as the
    fit behind it -- ``shot_fit_end_reason`` says why the window closed and
    ``shot_crossing_confidence`` scales with the evidence::

        out.loc[out["shot_crossing_confidence"] > 0.5, "shot_crossing_y"]

    ``shot_crossing_y`` / ``_z`` are ALWAYS in the canonical attacked-goal-at-x=105 frame,
    whichever way the shooting team was playing: the engine resolves goal ends from the GK
    map, so the caller never has to orient the input or re-orient the output.

    ``shot_on_target_derived`` is deliberately ``pd.NA`` rather than ``False`` when the
    crossing height is unknown -- an unknowable bar is not a miss::

        on_target_rate = out["shot_on_target_derived"].mean()  # NA rows excluded

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    from silly_kicks.tracking._gk_geometry import _truthy_bool
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking.utils import slice_around_event

    _ = links  # signature parity only; see docstring
    params = params or ShotGoalmouthParams()
    n = len(actions)
    out = pd.DataFrame(index=actions.index)
    for c in ("shot_crossing_y", "shot_crossing_z", "shot_speed", "shot_time_to_goal_line"):
        out[c] = np.full(n, np.nan)
    out["shot_on_target_derived"] = pd.array([pd.NA] * n, dtype="boolean")
    out["shot_crossing_source"] = pd.array([pd.NA] * n, dtype="object")
    out["shot_crossing_confidence"] = np.full(n, np.nan)
    out["shot_fit_n_frames"] = pd.array([pd.NA] * n, dtype="Int64")
    out["shot_fit_rmse"] = np.full(n, np.nan)
    out["shot_fit_end_reason"] = pd.array([pd.NA] * n, dtype="object")
    out["shot_z_profile"] = pd.array([pd.NA] * n, dtype="object")
    if n == 0 or "type_id" not in actions.columns:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    is_ball = _truthy_bool(frames["is_ball"])  # ADR-019: never .astype(bool) a string column
    ball = frames[is_ball].drop_duplicates(["game_id", "period_id", "frame_id"], keep="first")
    goal_map = resolve_defended_goals(frames)

    shots = actions[is_shot]
    sl = slice_around_event(shots, ball, pre_seconds=_PRE_SECONDS, post_seconds=params.post_window_seconds)
    by_action = dict(iter(sl.groupby("action_id"))) if len(sl) else {}

    for ridx, row in shots.iterrows():
        # attacked goal = the goal defended by the OTHER team in this (game, period).
        # Three mutually exclusive resolution states:
        #   resolved   -- exactly 2 teams, action team identified (same_id), ends differ
        #   degenerate -- exactly 2 teams but the GK map puts both at the SAME end (PSO;
        #                 spec 5.5) -> fallback to the end nearer the window's mean ball x
        #   unresolved -- anything else (NaN action team -> same_id never matches -> 2
        #                 candidate "opponent" ends; or a malformed (game, period) group)
        # The seam owns the lookup: `ends_in_period` canonicalizes, and `attacked_goal` is a
        # real lookup of the opponent's entry. The former raw tuple `==` compared keys built
        # from `frames` against ids read off `actions` -- a live ADR-019 hazard.
        ends = goal_map.ends_in_period(row["game_id"], row["period_id"], allow_guess=True)
        attacked = goal_map.attacked_goal(row["game_id"], row["period_id"], row["team_id"], allow_guess=True)
        degenerate = len(ends) == 2 and len(set(ends.values())) == 1
        resolved = attacked is not None and not degenerate
        if not resolved and not degenerate:
            out.loc[ridx, "shot_crossing_source"] = "unresolved"
            out.loc[ridx, "shot_crossing_confidence"] = 0.0
            continue
        g = by_action.get(row["action_id"])
        if g is None or g.empty:
            out.loc[ridx, "shot_crossing_source"] = "no_ball_frames"
            out.loc[ridx, "shot_crossing_confidence"] = 0.0
            continue
        g = g.sort_values("time_offset_seconds")
        t = g["time_offset_seconds"].to_numpy(float)
        xv = g["x"].to_numpy(float)
        yv = g["y"].to_numpy(float)
        zv = g["z"].to_numpy(float)
        if resolved:
            goal_x = float(attacked)  # type: ignore[arg-type]
        else:  # degenerate -> PSO fallback (spec section 5.5)
            goal_x = 0.0 if float(np.nanmean(xv)) < 52.5 else _FIELD_LENGTH
        truncated = (t[-1] - max(float(t[0]), 0.0)) < params.post_window_seconds - 0.5 if len(t) else True
        r = _fit_one_shot(
            t,
            xv,
            yv,
            zv,
            goal_x=goal_x,
            params=params,
            window_truncated=truncated,
            contact_xy=_contact_anchor(row, goal_x),
        )
        # canonicalize to attacked-goal-at-105 (full point reflection x->105-x, y->68-y)
        cy = r["crossing_y"] if goal_x > 50.0 else (_FIELD_WIDTH - r["crossing_y"])
        out.loc[ridx, "shot_crossing_y"] = cy
        out.loc[ridx, "shot_crossing_z"] = r["crossing_z"]
        out.loc[ridx, "shot_speed"] = r["speed"]
        out.loc[ridx, "shot_time_to_goal_line"] = r["time_to_goal_line"]
        out.loc[ridx, "shot_crossing_source"] = r["source"]
        out.loc[ridx, "shot_fit_end_reason"] = r["end_reason"]
        out.loc[ridx, "shot_z_profile"] = r["z_profile"]
        out.loc[ridx, "shot_fit_n_frames"] = r["n_fit_frames"] if r["n_fit_frames"] else pd.NA
        out.loc[ridx, "shot_fit_rmse"] = r["fit_rmse"]
        out.loc[ridx, "shot_crossing_confidence"] = _confidence(r, params)
        if r["source"] in ("observed", "extrapolated") and np.isfinite(cy):
            if np.isfinite(r["crossing_z"]):
                tol = params.on_target_tolerance_m
                on = abs(cy - _GOAL_Y_C) <= _GOAL_HALF_MOUTH + tol and r["crossing_z"] <= _BAR_Z + tol
                out.loc[ridx, "shot_on_target_derived"] = bool(on)
            # crossing_z NaN -> on-target stays NA (bar unknowable; spec section 7)
    return out


@dataclass(frozen=True)
class ShotGoalmouthReport:
    """Aggregate provenance QA for shot-goalmouth output (convenience over value_counts;
    ``z_profile_counts`` is the corpus-scale bounce-misclassification detector -- spec L-3).

    Examples
    --------
    Run it over a corpus, not a match: the counts are a distribution check, and the failure
    it is built to catch -- z-profile misclassification -- is invisible one shot at a time::

        rep = ShotGoalmouthReport.from_frame(add_shot_goalmouth(actions, frames))
        rep.z_profile_counts  # a bounced share that drifts is the L-3 tell
        rep.end_reason_counts  # window_cap dominating means fits are running out of frames
        rep.n_shots  # shots that produced a crossing, NOT shots attempted

    ``n_shots`` counts rows with a resolved crossing source, so comparing it against the
    number of shot actions gives the coverage rate the ADR-030 floors are stated in.
    """

    n_shots: int
    source_counts: dict[str, int]
    end_reason_counts: dict[str, int]
    z_profile_counts: dict[str, int]
    n_on_target_derived: int

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> ShotGoalmouthReport:
        """Build from a ``compute_/add_shot_goalmouth`` frame.

        Reads only the four provenance columns and the on-target flag, so it accepts the
        engine output, the ``add_*`` output, or a concatenation of many matches. Rows that
        produced no crossing are counted OUT of every tally: ``n_shots`` is the resolved
        count, and the value-counts skip nulls, so an unfittable shot never masquerades as a
        profile or an end reason. ``n_on_target_derived`` counts ``True`` alone -- the
        ``pd.NA`` of an unknown crossing height is not a miss.

        Examples
        --------
        >>> import pandas as pd
        >>> from silly_kicks.tracking import ShotGoalmouthReport
        >>> enriched = pd.DataFrame(
        ...     {
        ...         "shot_crossing_source": ["fitted", "fitted", None],  # 3rd: no crossing
        ...         "shot_fit_end_reason": ["plane_straddle", "window_cap", None],
        ...         "shot_z_profile": ["airborne", "bounced", None],
        ...         "shot_on_target_derived": pd.array([True, False, pd.NA], dtype="boolean"),
        ...     }
        ... )
        >>> rep = ShotGoalmouthReport.from_frame(enriched)
        >>> rep.n_shots, rep.n_on_target_derived
        (2, 1)
        >>> rep.z_profile_counts
        {'airborne': 1, 'bounced': 1}
        """

        def _counts(col: str) -> dict[str, int]:
            return {str(k): int(v) for k, v in df[col].value_counts(dropna=True).items()}

        return cls(
            n_shots=int(df["shot_crossing_source"].notna().sum()),
            source_counts=_counts("shot_crossing_source"),
            end_reason_counts=_counts("shot_fit_end_reason"),
            z_profile_counts=_counts("shot_z_profile"),
            n_on_target_derived=int((df["shot_on_target_derived"] == True).sum()),  # noqa: E712
        )
