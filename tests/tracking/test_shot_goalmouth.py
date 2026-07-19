"""TF-48 post-shot goalmouth geometry (spec 2026-06-10-shot-goalmouth-psxg-design)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._shot_goalmouth import ShotGoalmouthParams


class TestParams:
    def test_defaults(self):
        p = ShotGoalmouthParams()
        assert p.post_window_seconds == 3.5 and p.min_fit_frames == 3  # pilot-calibrated
        assert p.rolling_z_max_m == 0.3 and p.bounce_min_dz_m == 0.25

    @pytest.mark.parametrize(
        "kw",
        [
            {"post_window_seconds": 0.0},
            {"post_window_seconds": -1.0},
            {"min_fit_frames": 1},
            {"break_residual_m": 0.0},
            {"break_speed_drop_frac": 1.5},
            {"max_time_to_plane_seconds": 0.0},
            {"rolling_z_max_m": -0.1},
            {"bounce_min_dz_m": 0.0},
            {"on_target_tolerance_m": -0.01},
        ],
    )
    def test_post_init_rejects(self, kw):
        with pytest.raises(ValueError):
            ShotGoalmouthParams(**kw)

    def test_frozen(self):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            ShotGoalmouthParams().min_fit_frames = 5  # type: ignore[misc]


from silly_kicks.tracking._shot_goalmouth import _find_flight_run, _fit_one_shot  # noqa: E402

P = ShotGoalmouthParams()


def traj(vx=25.0, vy=0.0, vz=5.0, x0=85.0, y0=34.0, z0=0.0, fps=25.0, n=12, t0=0.0, gravity=True):
    """Ballistic samples: x linear, y linear, z = z0 + vz t - 4.905 t^2 (>= 0)."""
    t = t0 + np.arange(n) / fps
    g = 9.81 if gravity else 0.0
    z = np.maximum(z0 + vz * t - 0.5 * g * t**2, 0.0)
    return t, x0 + vx * t, y0 + vy * t, z


class TestFitOneShot:
    def test_straight_drive_extrapolated(self):
        # 25 m/s from x=85 toward 105: crossing at t*=0.8 s; samples cover 0..0.44 s only.
        t, x, y, z = traj(vx=25.0, vy=2.0, vz=4.0, n=12, fps=25.0)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated"
        assert r["crossing_y"] == pytest.approx(34.0 + 2.0 * 0.8, abs=1e-6)
        assert r["crossing_z"] == pytest.approx(max(4.0 * 0.8 - 4.905 * 0.64, 0.0), abs=1e-3)
        assert r["time_to_goal_line"] == pytest.approx(0.8, abs=1e-6)
        assert r["speed"] == pytest.approx(np.hypot(np.hypot(25.0, 2.0), 4.0), rel=0.05)
        assert r["z_profile"] == "airborne" and r["end_reason"] == "window_cap"

    def test_observed_crossing_interpolated(self):
        # samples STRADDLE the plane -> source observed, crossing from interpolation
        t, x, y, z = traj(vx=30.0, x0=95.0, n=12)  # crosses 105 at t=1/3 s (within samples)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "observed" and r["end_reason"] == "plane_crossed"
        assert r["crossing_y"] == pytest.approx(34.0, abs=1e-6)

    def test_wide_miss_away_from_plane_is_no_crossing(self):
        t, x, y, z = traj(vx=-20.0, x0=85.0)  # moving AWAY from goal_x=105
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "no_crossing"
        assert np.isnan(r["crossing_y"]) and np.isnan(r["crossing_z"])

    def test_too_slow_ball_is_insufficient(self):
        # a 2 m/s GROUND ball is sub-flight-speed EVERYWHERE: the flight-core trim
        # (pilot 2026-06-11) leaves no usable flight -> insufficient_frames ("no
        # fittable flight"), more honest than no_crossing ("a fitted flight that
        # misses"). Fast away-balls (own goals / wide mishits) still report
        # no_crossing. z pinned to the ground: an AIRBORNE slow ball is a chip and
        # legitimately fits (z-aware flight, v3 pilot).
        t, x, y, _ = traj(vx=2.0, x0=85.0)
        r = _fit_one_shot(t, x, y, np.full_like(t, 0.05), goal_x=105.0, params=P)
        assert r["source"] == "insufficient_frames"

    def test_extrapolation_leverage_capped(self):
        # 0.44 s of clean 10 m/s flight, plane 20 m away: t* = 2.0 s is within the
        # absolute cap but 4.5x the evidence span. v4 pilot (2026-06-11): extrapolations
        # past 3x their span had dy median 6.22 m (max 41 m, all junk) vs 2.35 m below
        # -- claiming a crossing from that little evidence is a guess, not a fit.
        t, x, y, z = traj(vx=10.0, x0=85.0, vz=0.0, n=12, fps=25.0)
        r = _fit_one_shot(t, x, y, np.zeros_like(z), goal_x=105.0, params=P)
        assert r["source"] == "no_crossing"
        assert r["n_fit_frames"] >= P.min_fit_frames  # the fit itself is reported (R4b)

    def test_no_crossing_keeps_fit_diagnostics(self):
        # R4b: a fit RAN -> n_fit_frames / fit_rmse populated even without a crossing
        t, x, y, z = traj(vx=-20.0, x0=85.0)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["n_fit_frames"] >= P.min_fit_frames and np.isfinite(r["fit_rmse"])

    def test_insufficient_frames(self):
        t, x, y, z = traj(n=2)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "insufficient_frames"

    def test_slow_diagonal_carry_is_insufficient(self):
        # the measured dominant error class (pilot 2026-06-11, 9/13 of >=3 m dy): a ball
        # carried DIAGONALLY clears a scalar-speed bar (8.6 m/s here) while approaching
        # the plane at only ~5 m/s -- fitting that carry-line produced garbage crossings.
        # Flight requires |approach| >= 7 m/s; the slow/lateral middle ground trims away.
        t, x, y, _ = traj(vx=5.0, vy=7.0, x0=85.0, n=20)  # scalar 8.6, approach 5.0
        z = np.zeros(20)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "insufficient_frames"

    def test_noisy_high_rate_survives(self):
        # 29.97 fps + realistic broadcast position jitter (sigma 0.15 m) must NOT
        # phantom-break: per-frame finite differences amplify jitter ~30x at this
        # rate (the real-WC2022 failure mode, pilot 2026-06-11); baseline velocities fix it
        rng = np.random.default_rng(42)
        t, x, y, z = traj(vx=22.0, x0=80.0, vz=4.0, n=45, fps=29.97)
        x = x + rng.normal(0, 0.15, len(x))
        y = y + rng.normal(0, 0.15, len(y))
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] in ("extrapolated", "observed")


class TestSampleHoldDuplication:
    """GS raw ``balls`` is sample-and-hold upsampled: ~15 Hz positions emitted twice at
    29.97 Hz stamps (measured 50% consecutive-duplicate x/y/z on ALL 127 WC2022 pilot
    windows, raw-artifact-confirmed 2026-06-11 -- a channel property, not a loader bug).
    A held duplicate is a phantom zero-velocity observation: 0.1 s baselines then read a
    5.5 m/s carry as alternating ~3.7/~7.3 m/s depending on phase, leaking sub-flight
    carries through the 7 m/s flight gate (the measured 29.5 m worst-dy mechanism)."""

    @staticmethod
    def _hold2(real_xyz, fps=29.97):
        """Emit each real ~15 Hz sample twice at consecutive ~30 Hz stamps."""
        xr, yr, zr = real_xyz
        t = np.arange(2 * len(xr)) / fps
        return t, np.repeat(xr, 2), np.repeat(yr, 2), np.repeat(zr, 2)

    def test_sample_hold_carry_does_not_leak(self):
        # 5.5 m/s carry from 13 m out: REAL approach speed is sub-flight (< 7 m/s), but
        # the odd-phase duplicated baseline reads 2 real steps over ~0.1 s => ~7.3 m/s
        n = 25
        tr = np.arange(n) * 2 / 29.97
        t, x, y, z = self._hold2((92.0 + 5.5 * tr, 50.0 + 1.5 * tr, np.zeros(n)))
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "insufficient_frames"

    def test_sample_hold_real_shot_still_fits(self):
        # a genuine 24 m/s shot under the same duplication must keep fitting cleanly
        n = 8
        tr = np.arange(n) * 2 / 29.97
        zr = np.maximum(4.0 * tr - 4.905 * tr**2, 0.0)
        t, x, y, z = self._hold2((85.0 + 24.0 * tr, 31.0 + 2.0 * tr, zr))
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated"
        t_star = 20.0 / 24.0
        assert r["crossing_y"] == pytest.approx(31.0 + 2.0 * t_star, abs=0.3)
        assert r["speed"] == pytest.approx(np.hypot(np.hypot(24.0, 2.0), 4.0), rel=0.1)

    def test_n_fit_frames_reports_post_trim_core(self):
        # flight (8 m/s approach) then a 4.2 m/s tail: the speed-drop check tolerates it
        # (ratio > 0.5) so _grow_segment only breaks on residual ~5 samples in; the
        # flight-core trim then removes the leaked tail. n_fit_frames must count the
        # TRIMMED core, not the pre-trim grow end (the pilot artifact reported n=19 on
        # a 12-sample core; the bounce-supersession arithmetic shares the same offset).
        fps = 25.0
        n1, n2 = 11, 10
        t = np.arange(n1 + n2) / fps
        # close to the plane so the extrapolation stays under the leverage cap
        x = np.concatenate([99.0 + 8.0 * t[:n1], 99.0 + 8.0 * t[n1 - 1] + 4.2 * (t[n1:] - t[n1 - 1])])
        y = np.full(n1 + n2, 40.0)
        z = np.zeros(n1 + n2)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "extrapolated"
        assert r["n_fit_frames"] <= n1 + 3


class TestContactAnchor:
    """Cross + header form ONE continuous plane-approach run (the redirect keeps approach
    speed), so the flight-run anchor locked onto the ASSIST and extrapolated the cross
    line (measured 7.3 m dy on a real WC2022 header goal, v2 pilot 2026-06-11). The GS
    event x/y is an exact ball-track point (median shooter-to-nearest-sample distance
    0.0 m; 92% of pilot windows within 1 m), so the kernel re-anchors t0 at the LAST run
    sample within _CONTACT_RADIUS_M of the shooter position -- the ball leaving the
    shooter IS the shot contact."""

    @staticmethod
    def _cross_then_header(fps=25.0):
        # cross from (87, 42) arriving at the shooter (99, 32) at t=0.75, header
        # redirect to a flat goalward flight that crosses x=105 at y ~ 31.4
        t = np.arange(-0.2, 1.21, 1 / fps)
        x = np.where(t < 0.75, 87.0 + 16.0 * (t + 0.2) * (12.0 / 15.62), 99.0 + 24.0 * (t - 0.75))
        y = np.where(t < 0.75, 42.0 - 16.0 * (t + 0.2) * (10.0 / 15.62), 32.0 - 2.0 * (t - 0.75))
        z = np.where(t < 0.75, 2.0, np.maximum(2.0 + 1.0 * (t - 0.75) - 4.905 * (t - 0.75) ** 2, 0.0))
        return t, x, y, z

    def test_cross_then_header_anchors_at_shooter(self):
        t, x, y, z = self._cross_then_header()
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=(99.0, 32.0))
        assert r["source"] in ("observed", "extrapolated")
        # header line crosses at ~31.4; the assist-cross line extrapolates to ~27
        assert r["crossing_y"] > 30.0

    def test_no_contact_falls_back_to_run_start(self):
        # without the anchor the same window fits the CROSS (documents the failure mode
        # the anchor exists for -- if this ever starts passing >30, the anchor can go)
        t, x, y, z = self._cross_then_header()
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=None)
        assert r["crossing_y"] < 30.0

    def test_far_stamp_is_insufficient(self):
        # CONTACT EXISTENCE (v7, the 12.6 m P2 class): when the event provides a shot
        # location, a window whose ball NEVER comes contactably near it (2-D within
        # _CONTACT_EXIST_RADIUS_M at z <= _CONTACT_MAX_Z_M) provably does not contain
        # the shot -- fitting whatever flight happens to be in the window produced the
        # worst measured goals (a pre-contact assist arc straddling the extended plane
        # 9.5 m wide). Honest insufficient_frames; NaN-stamp (contact_xy=None) keeps
        # the un-anchored behavior (ADR-003).
        t, x, y, z = traj(vx=25.0, vy=2.0, vz=4.0, n=12, fps=25.0)
        r_far = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=(20.0, 5.0))
        assert r_far["source"] == "insufficient_frames"
        r_none = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=None)
        assert r_none["source"] == "extrapolated"  # NaN-stamp path unchanged

    def test_overhead_arc_is_not_the_shot(self):
        # replica of the measured 12.6 m P2 goal (10502/1187): a high cross (z ~ 5-6 m)
        # passes 2-D-NEAR the shooter but metres OVERHEAD, then crosses the extended
        # plane far wide of the mouth. The ball is never CONTACTABLE at the stamp ->
        # the window does not contain the shot; the wide pre-contact straddle must NOT
        # be reported as the goal's observed crossing.
        fps = 25.0
        t = np.arange(40) / fps  # 1.6 s
        x = 97.0 + 5.0 * t  # drifts toward/past the plane (crosses x=105 at t=1.6)
        y = 44.0 - 12.0 * t  # sweeping wide across the box
        z = 6.0 - 1.0 * t  # stays >= 4.4 m -- never contactable
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=(99.7, 34.0))
        assert r["source"] == "insufficient_frames"

    def test_contact_after_aerial_arc_anchors_to_shot(self):
        # the arc comes DOWN to the shooter (contactable), the header redirects
        # goalward: the fit must anchor at the low pass and produce the SHOT's
        # crossing, not the arc's wide one
        fps = 25.0
        t1 = np.arange(0, 0.72, 1 / fps)  # arc descending onto the shooter
        x1 = 92.0 + 9.7 * t1
        y1 = 44.0 - 14.0 * t1
        z1 = 4.0 - 3.0 * t1
        t2 = np.arange(0.72, 1.32, 1 / fps)  # header: flat drive to the goal
        x2 = x1[-1] + 18.0 * (t2 - t1[-1])
        y2 = y1[-1] - 1.0 * (t2 - t1[-1])
        z2 = np.maximum(z1[-1] - 2.0 * (t2 - t1[-1]), 0.1)
        t = np.concatenate([t1, t2])
        x, y, z = np.concatenate([x1, x2]), np.concatenate([y1, y2]), np.concatenate([z1, z2])
        contact = (float(x1[-1]), float(y1[-1]))  # stamp at the header point
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, contact_xy=contact)
        assert r["source"] in ("observed", "extrapolated")
        # header line: crossing y ~ contact_y - (dist-to-plane/18) * 1
        assert abs(r["crossing_y"] - (y1[-1] - (105.0 - x1[-1]) / 18.0)) < 0.6


class TestFlightRunSelection:
    """When MORE than one plane-approach run reaches the goal line, the SHOT is the
    EARLIEST run that reaches it -- the first time the ball gets to the goal plane. The
    bare 'nearest-plane' tie-break picked whichever run ended marginally closer, which on
    a real holdout goal (10511/1089) was a LATER run, anchoring t0 PAST the real in-mouth
    crossing (the kernel then fitted post-goal motion and reported the goal off-target;
    ADR-030). An assist pass ends 10+ m out and never 'reaches', so it is still rejected.
    """

    def test_prefers_earliest_reaching_run(self):
        # two qualifying approach runs both reach x=105: an EARLY one ending at dist 1.0 m
        # and a LATER one ending marginally closer (0.5 m). The shot is the early run.
        t = np.arange(40) / 25.0
        x = np.full(40, 90.0)
        x[2:12] = np.linspace(90.0, 104.0, 10)  # run 1: dist 15->1, drop 14, fast
        x[12:16] = 104.0  # dwell at the goal line (run 1 ends here, dist 1.0 -> reaches)
        x[16:27] = np.linspace(104.0, 98.0, 11)  # ball drifts back out (no run)
        x[27:37] = np.linspace(98.0, 104.5, 10)  # run 2: dist 7->0.5, drop 6.5, reaches closer
        x[37:] = 104.5
        start = _find_flight_run(t, x, 105.0)
        assert start is not None
        assert t[start] < 0.5  # the EARLY run (idx 2, t=0.08), not the later run

    def test_assist_not_reaching_falls_back(self):
        # only ONE run reaches the plane (the shot, dist->0.3); an earlier 'run' that ends
        # 9 m out (an assist) does NOT reach -> the shot is still chosen.
        t = np.arange(40) / 25.0
        x = np.full(40, 70.0)
        x[2:11] = np.linspace(70.0, 79.0, 9)  # assist: dist 35->26, drop 9, fast, ends FAR out
        x[11:24] = 79.0  # dwell at the receiver
        x[24:35] = np.linspace(79.0, 104.7, 11)  # shot: dist 26->0.3, reaches
        x[35:] = 104.7
        start = _find_flight_run(t, x, 105.0)
        assert start is not None and t[start] > 0.8  # the shot run, not the far assist


class TestChipFlight:
    """A chip/lob decelerates horizontally below the 7 m/s flight gate while CLIMBING
    (real WC2022 chip goal: x-approach 13 -> 3 m/s, z 0 -> 2.7 m, v3 pilot 2026-06-11)
    -- x-distance-only flight logic speed-drop-broke it and trimmed the core to 2
    samples (insufficient). Airborne (finite z above the rolling band) IS flight: the
    trim keeps airborne samples and the speed-drop break is skipped at airborne
    checkpoints (residual + reversal breaks stay active -- mid-air deflections still
    end the segment)."""

    def test_decelerating_chip_reaches_observed_straddle(self):
        # x = 95 + 13t - 4t^2 crosses 105 at t=1.25 with v_x=3 m/s (ratio 3/13 trips the
        # 0.5 speed-drop break today); z = 6.5t - 4.905t^2 stays above the band
        fps = 25.0
        t = np.arange(36) / fps
        x = 95.0 + 13.0 * t - 4.0 * t**2
        y = 34.0 - 1.0 * t
        z = np.maximum(6.5 * t - 4.905 * t**2, 0.0)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "observed"
        assert r["crossing_y"] == pytest.approx(34.0 - 1.25, abs=0.15)

    def test_ground_carry_still_trims(self):
        # the z-OR must not resurrect the ground-carry leak: same deceleration profile
        # ON THE GROUND is still not flight once it drops sub-gate
        fps = 25.0
        t = np.arange(36) / fps
        x = 95.0 + 5.0 * t
        y = 50.0 + 1.5 * t
        z = np.full_like(t, 0.05)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "insufficient_frames"


class TestCurveAwareExtrapolation:
    """A curling/dipping shot whose y bends within the fitted flight is extrapolated to
    the plane LINEARLY by the constant-velocity fit, missing the curl (measured 5.4 m on a
    real WC2022 chip-curl goal -- holdout class B, 5 goals; ADR-030). When the producing
    segment is long enough to estimate curvature AND a quadratic markedly out-fits the
    line (real curl, not jitter), the crossing y is taken from the span-gated quadratic.
    Straight shots keep the linear value byte-identically (test_straight_drive_extrapolated
    + test_near_straight_keeps_linear)."""

    def test_curling_chip_extrapolation_follows_the_curve(self):
        # y curls as 34 + 8 t^2 over a 1.16 s airborne flight; plane reached by
        # extrapolation at t*=1.3 s. A line cannot follow the bend -- its crossing
        # undershoots the true curved y by ~3 m; the quadratic recovers it.
        fps = 25.0
        t = np.arange(30) / fps  # span 1.16 s
        vx, x0, ay = 10.0, 92.0, 8.0
        x = x0 + vx * t
        y = 34.0 + ay * t**2
        z = np.maximum(7.0 * t - 4.905 * t**2, 0.0)  # airborne arc, stays above the band
        t_star = (105.0 - x0) / vx
        y_true = 34.0 + ay * t_star**2
        # what a pure-linear fit of the same samples would predict (the OLD behavior)
        b, a = np.polyfit(t, y, 1)[::-1]  # a + b t
        y_linear = a + b * t_star
        assert abs(y_linear - y_true) > 2.0  # the linear miss this test exists to fix
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "extrapolated"
        assert r["crossing_y"] == pytest.approx(y_true, abs=0.6)

    def test_near_straight_keeps_linear(self):
        # a shot with negligible curvature must NOT be bent by the quadratic path: the
        # curvature-signal gate (lin RMSE floor) keeps the byte-identical linear crossing.
        fps = 25.0
        t = np.arange(30) / fps
        vx, x0, vy = 10.0, 92.0, 3.0
        x = x0 + vx * t
        y = 34.0 + vy * t  # exactly linear
        z = np.maximum(7.0 * t - 4.905 * t**2, 0.0)
        t_star = (105.0 - x0) / vx
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "extrapolated"
        assert r["crossing_y"] == pytest.approx(34.0 + vy * t_star, abs=1e-6)


class TestBreakDetection:
    def test_deflection_ends_segment(self):
        t, x, y, z = traj(vx=25.0, vy=0.0, n=14)
        x[8:] = x[7] + np.arange(1, 7) * 0.2  # deflected: x slows hard
        y[8:] = y[7] + np.arange(1, 7) * 1.6  # ... and veers in y
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["end_reason"] == "trajectory_break"
        assert r["source"] == "extrapolated"  # pre-break segment still extrapolates
        assert r["n_fit_frames"] <= 9

    def test_block_short_segment_insufficient(self):
        t, x, y, z = traj(vx=25.0, n=14)
        x[2:] = x[1]  # blocked dead after 2 samples...
        z[2:] = 0.05  # ...and knocked to the ground (a frozen-x ball with ballistic z
        # would be an unphysical fixture; airborne samples count as flight since v3)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "insufficient_frames"


class TestZProfiles:
    def test_rolling_daisy_cutter(self):
        t, x, y, _ = traj(vx=20.0, n=12)
        z = np.abs(np.sin(np.arange(12))) * 0.1  # never above 0.3
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["z_profile"] == "rolling"
        assert r["crossing_z"] == pytest.approx(float(np.nanmean(z)), abs=1e-9)

    def test_bounced_full_supersession(self):
        # pre-bounce 6 samples descending; bounce at k=5 (z=0.05); post-bounce ballistic
        fps = 25.0
        t = np.arange(12) / fps
        x, y = 80.0 + 22.0 * t, 34.0 + 0.0 * t
        z = np.concatenate([np.linspace(1.0, 0.05, 6), 0.05 + 3.0 * (t[6:] - t[5]) - 4.905 * (t[6:] - t[5]) ** 2])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "bounced"
        assert r["n_fit_frames"] <= 7  # producing segment = post-bounce (M-1)

    def test_bounced_z_only_refit_branch(self):
        # post-bounce sub-segment has exactly 2 samples (>= 2 but < min_fit_frames=3):
        # z refit on the sub-segment, x/y from the FULL segment (spec L-2 branch 2).
        # detector check: flip at k=6 (vz[5]<0<=vz[6]); z[6]=0.05<=0.3; drop=1.15>=0.25;
        # rise=0.45-0.05=0.40>=0.25; sub z=[0.05,0.45], max 0.45>0.3 -> bounced (not rolling)
        fps = 25.0
        t = np.arange(8) / fps
        x, y = 92.0 + 22.0 * t, np.full(8, 34.0)  # near plane: leverage-cap headroom
        z = np.concatenate([np.linspace(1.2, 0.05, 7), [0.45]])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "bounced"
        assert r["n_fit_frames"] == 8  # x/y producing segment = FULL segment (no supersession)
        assert np.isfinite(r["crossing_z"])  # z refit on the 2-sample sub-segment

    # NOTE (spec section 6 branch 3, < 2 post-bounce samples -> crossing_z NaN): under this
    # detector the branch is UNREACHABLE BY CONSTRUCTION -- a vz sign flip at k requires
    # z[k-1], z[k], z[k+1] all finite, so a detected bounce always leaves >= 2 finite samples
    # in the sub-segment. The `f.sum() >= 2` guard in _fit_one_shot is retained DEFENSIVELY.
    # Proof recorded in ADR-030; branch 1 + branch 2 tested above.

    def test_noisy_airborne_stays_airborne(self):
        # vz sign flips AT HEIGHT (z ~ 1.5 m) from noise -> must NOT classify bounced (M-2)
        t, x, y, _ = traj(vx=22.0, n=12)
        z = 1.5 + np.array([0.0, 0.1, -0.08, 0.12, -0.1, 0.05, -0.07, 0.1, -0.05, 0.02, -0.04, 0.0])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "airborne"

    def test_dipping_ballistic_closed_form(self):
        # z DESCENDS through the window (past apex): fixed-g fit must recover the
        # closed-form crossing z. vz=8, z0=0 -> apex at 0.815 s; samples 0.4..0.84 s.
        t, x, y, z = traj(vx=20.0, x0=80.0, vz=8.0, n=12, fps=25.0, t0=0.4)
        t = t - 0.4  # window-relative; gravity term is shift-invariant
        # crossing at x=105: x = 88 + 20*t_win -> t_win* = 0.85; absolute flight 1.25 s
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        z_true = 8.0 * 1.25 - 4.905 * 1.25**2
        assert r["source"] == "extrapolated"
        assert r["crossing_z"] == pytest.approx(max(z_true, 0.0), abs=0.05)

    def test_occlusion_gap_data_end(self):
        # samples stop mid-window (occlusion): window_truncated -> end_reason data_end
        t, x, y, z = traj(vx=20.0, x0=92.0, n=10)  # near plane: leverage-cap headroom
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, window_truncated=True)
        assert r["end_reason"] == "data_end"
        assert r["source"] == "extrapolated"

    def test_z_onset_lag_trimmed(self):
        # GS z-channel onset lag (pilot-measured p75 0.8 s): z reads ~0 for the first
        # samples of a real flight, then the true ballistic appears. The flat prefix
        # must NOT drag the ballistic fit to the ground -- fit from the sustained rise.
        fps = 25.0
        n = 20
        t = np.arange(n) / fps
        x, y = 80.0 + 20.0 * t, np.full(n, 34.0)
        z_true = 0.0 + 6.0 * t - 4.905 * t**2
        z = z_true.copy()
        z[:8] = 0.02  # channel lag: first 0.32 s reads ground
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "extrapolated"
        t_star = 25.0 / 20.0  # x crosses 105 at 1.25 s
        z_at_star = max(6.0 * t_star - 4.905 * t_star**2, 0.0)
        # ungated fit (flat prefix included) lands ~50%+ low; trimmed fit must be close
        assert r["crossing_z"] == pytest.approx(z_at_star, abs=0.4)

    def test_z_all_nan_degrades_visibly(self):
        t, x, y, _ = traj(vx=25.0, vy=1.0, n=10)
        z = np.full(10, np.nan)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated" and np.isfinite(r["crossing_y"])
        assert np.isnan(r["crossing_z"]) and r["z_profile"] is None
        assert np.isfinite(r["speed"])  # 2D fallback


class TestContactRefinement:
    def test_refinement_skips_pre_contact_drift(self):
        # 0.2 s of slow drift before contact, then the shot
        fps = 25.0
        pre_t = np.arange(-5, 0) / fps
        shot_t, shot_x, shot_y, shot_z = traj(vx=25.0, n=10, fps=fps)
        t = np.concatenate([pre_t, shot_t])
        x = np.concatenate([85.0 + 0.5 * pre_t, shot_x])
        y = np.concatenate([np.full(5, 34.0), shot_y])
        z = np.concatenate([np.zeros(5), shot_z])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated"
        assert r["speed"] == pytest.approx(np.hypot(25.0, 4.0), rel=0.1)

    def test_refinement_does_not_lock_onto_save(self):
        # shot at t=0 (speed jump +25), SAVE at t=0.2 (huge reversal, inside the window):
        # refinement must pick the FIRST shot-consistent discontinuity (H-3)
        fps = 25.0
        t = np.arange(-2, 8) / fps
        x = np.where(t < 0, 95.0, 95.0 + 25.0 * t)
        x = np.where(t > 0.2, x[np.searchsorted(t, 0.2)] - 10.0 * (t - 0.2), x)
        y, z = np.full_like(t, 34.0), np.zeros_like(t)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        # the fit segment is the shot, broken at the save; never the post-save track
        assert r["end_reason"] in ("trajectory_break", "plane_crossed")


from silly_kicks.tracking._shot_goalmouth import (  # noqa: E402
    ShotGoalmouthReport,
    compute_shot_goalmouth,
)


def make_match(shot_team="A", period=1, goal_a_x=5.0, goal_b_x=100.0, flip=False, vy=2.0, game_id=1):
    """One shot at t=600.0 by ``shot_team`` toward the opponent's goal. Frames:
    GKs anchored at each end + ball trajectory (25 m/s, +vy, ballistic z).
    flip=True mirrors the WHOLE frame set (x->105-x, y->68-y) -- same physical match
    in the opposite global convention (M-2 invariance)."""
    from silly_kicks.spadl import config as spadlconfig

    rows = []
    for off in np.arange(-0.4, 1.21, 0.04):
        tt = 600.0 + off
        rows.append(
            dict(
                game_id=game_id,
                period_id=period,
                frame_id=int(tt * 25),
                time_seconds=tt,
                frame_rate=25.0,
                player_id=1,
                team_id="A",
                is_ball=False,
                is_goalkeeper=True,
                x=goal_a_x,
                y=34.0,
                z=0.0,
                source_provider="gradientsports",
            )
        )
        rows.append(
            dict(
                game_id=game_id,
                period_id=period,
                frame_id=int(tt * 25),
                time_seconds=tt,
                frame_rate=25.0,
                player_id=2,
                team_id="B",
                is_ball=False,
                is_goalkeeper=True,
                x=goal_b_x,
                y=34.0,
                z=0.0,
                source_provider="gradientsports",
            )
        )
        bt = max(off, 0.0)  # ball waits at the spot, then flies
        bx, by = 85.0 + 25.0 * bt, 30.0 + vy * bt
        bz = max(4.0 * bt - 4.905 * bt**2, 0.0)
        rows.append(
            dict(
                game_id=game_id,
                period_id=period,
                frame_id=int(tt * 25),
                time_seconds=tt,
                frame_rate=25.0,
                player_id=None,
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=bx,
                y=by,
                z=bz,
                source_provider="gradientsports",
            )
        )
    frames = pd.DataFrame(rows)
    if flip:
        frames["x"], frames["y"] = 105.0 - frames["x"], 68.0 - frames["y"]
    actions = pd.DataFrame(
        {
            "game_id": [game_id],
            "action_id": [0],
            "period_id": [period],
            "time_seconds": [600.0],
            "team_id": [shot_team],
            "player_id": [10],
            "start_x": [85.0],
            "start_y": [30.0],
            "end_x": [105.0],
            "end_y": [34.0],
            "type_id": [spadlconfig.actiontype_id["shot"]],  # from config, never a literal
            "result_id": [0],
            "bodypart_id": [0],
        }
    )
    return actions, frames


class TestComputeEngine:
    def test_home_shot_canonical_outputs(self):
        actions, frames = make_match()
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        # make_match's ball reaches x=115 by window end -> the plane crossing is OBSERVED
        # (straddled samples, exact interpolation); extrapolation is the KERNEL tests' domain
        assert r["shot_crossing_source"] == "observed"
        # crossing at t*=0.8: y = 30 + 2*0.8 = 31.6 (already attacked-goal-at-105)
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)

    def test_orientation_invariance_byte_identical(self):
        a1, f1 = make_match(flip=False)
        a2, f2 = make_match(flip=True)
        # SPADL actions are per-action LTR -- identical in both cases by construction
        o1 = compute_shot_goalmouth(a1, f1)
        o2 = compute_shot_goalmouth(a2, f2)
        pd.testing.assert_frame_equal(o1, o2)  # M-2: engine assumes nothing about orientation

    def test_away_team_attacks_low_x_goal(self):
        # team B shoots: attacked goal is A's (x~0 end). Ball flies toward LOW x.
        actions, frames = make_match(shot_team="B")
        b = frames["is_ball"].astype(bool)
        frames.loc[b, "x"] = 105.0 - frames.loc[b, "x"]
        frames.loc[b, "y"] = 68.0 - frames.loc[b, "y"]
        # action coords stay per-action-LTR: the physical strike point (20, 38) in frame
        # coords IS (105-20, 68-38) = (85, 30) in the shooter's attacked-at-105 space
        # (the contact-existence check verifies the stamp against the ball track)
        actions["start_x"], actions["start_y"] = 85.0, 30.0
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        # the away shot IS the point reflection of the home shot -> canonical output identical
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)

    def test_non_shot_rows_all_nan(self):
        actions, frames = make_match()
        actions["type_id"] = 0  # pass
        out = compute_shot_goalmouth(actions, frames)
        assert out["shot_crossing_y"].isna().all() and out["shot_crossing_source"].isna().all()

    def test_no_ball_frames(self):
        actions, frames = make_match()
        out = compute_shot_goalmouth(actions, frames[~frames["is_ball"].astype(bool)])
        assert out.iloc[0]["shot_crossing_source"] == "no_ball_frames"

    def test_duplicate_frames_deduped(self):
        actions, frames = make_match()
        out_ref = compute_shot_goalmouth(actions, frames)
        dup = pd.concat([frames, frames], ignore_index=True)  # GS dup-frame pathology
        out_dup = compute_shot_goalmouth(actions, dup)
        pd.testing.assert_frame_equal(out_ref, out_dup)

    def test_unresolved_when_goal_map_degenerate(self):
        actions, frames = make_match(goal_a_x=50.0, goal_b_x=50.0)  # both GKs mid-pitch
        frames = frames[~frames["is_ball"].astype(bool)]  # and no ball fallback either
        out = compute_shot_goalmouth(actions, frames)
        assert out.iloc[0]["shot_crossing_source"] in ("unresolved", "no_ball_frames")

    def test_own_goal_is_no_crossing(self):
        # ball flies toward the shooter's OWN goal -> intentional exclusion (spec section 8).
        # Reflect the ball's direction AROUND ITS STRIKE POINT (x=85): a real own goal still
        # starts at the shooter (the contact-existence check verifies the stamp vs the track)
        actions, frames = make_match()
        b = frames["is_ball"].astype(bool)
        frames.loc[b, "x"] = 170.0 - frames.loc[b, "x"]  # 2*85 - x: reverse direction only
        out = compute_shot_goalmouth(actions, frames)
        assert out.iloc[0]["shot_crossing_source"] == "no_crossing"

    def test_nan_team_id_unresolved_no_crash(self):
        actions, frames = make_match()
        actions["team_id"] = pd.array([pd.NA], dtype="object")
        out = compute_shot_goalmouth(actions, frames)  # ADR-003: never crash
        # pinned by same_id's contract (id_compat.py -- "False if either is NA"):
        # NaN action team -> neither goal-map team matches -> 2 candidate ends -> unresolved
        assert out.iloc[0]["shot_crossing_source"] == "unresolved"

    def test_pso_degenerate_positive_path(self):
        # degenerate GK map (both teams classified to the SAME end) + ball present:
        # the ball-mean fallback resolves the attacked end -> a real crossing (spec 5.5)
        actions, frames = make_match(goal_a_x=95.0, goal_b_x=100.0)  # both ends -> 105.0
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)  # ball flies toward 105

    def test_period2_flip_resolved_per_period(self):
        # period 2: GK anchors swap ends; ball trajectory mirrored (the physical second-half
        # shot). The GK map resolves per (game, period) -> canonical output identical to p1.
        actions, frames = make_match(period=2, goal_a_x=100.0, goal_b_x=5.0, flip=False)
        b = frames["is_ball"].astype(bool)
        frames.loc[b, "x"] = 105.0 - frames.loc[b, "x"]
        frames.loc[b, "y"] = 68.0 - frames.loc[b, "y"]
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)


class TestOnTargetDerived:
    """Boundary tests for shot_on_target_derived (the lakehouse PSxG on-target gate).
    Drive the crossing point via the ball trajectory's vy/vz; tolerance = 0.11 m."""

    def _shot_with(self, vy, vz, gravity=True):
        actions, frames = make_match(vy=vy)
        b = frames["is_ball"].astype(bool)
        if not gravity:  # replace z with a linear rise to place crossing z precisely
            bt = np.maximum(frames.loc[b, "time_seconds"].to_numpy() - 600.0, 0.0)
            frames.loc[b, "z"] = vz * bt
        return compute_shot_goalmouth(actions, frames).iloc[0]

    def test_inside_mouth_true(self):
        # crossing y = 30 + 5*0.8 = 34.0 (centre), z linear 1.0*0.8 = 0.8 -> True
        r = self._shot_with(vy=5.0, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == True  # noqa: E712

    def test_just_outside_post_false(self):
        # crossing y = 30 + 9.8*0.8 = 37.84 -> 3.84 from centre > 3.66 + 0.11 -> False
        r = self._shot_with(vy=9.8, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == False  # noqa: E712

    def test_within_post_tolerance_true(self):
        # crossing y = 30 + 9.6*0.8 = 37.68 -> 3.68 from centre <= 3.66 + 0.11 -> True
        r = self._shot_with(vy=9.6, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == True  # noqa: E712

    def test_lob_over_bar_false_with_valid_yz(self):
        # crossing z = 3.5*0.8 = 2.8 > 2.44 + 0.11 -> False, with VALID y and z (spec 11)
        r = self._shot_with(vy=0.0, vz=3.5, gravity=False)
        assert r["shot_on_target_derived"] == False  # noqa: E712
        assert np.isfinite(r["shot_crossing_y"]) and np.isfinite(r["shot_crossing_z"])

    def test_na_when_z_unavailable(self):
        actions, frames = make_match()
        frames.loc[frames["is_ball"].astype(bool), "z"] = np.nan
        r = compute_shot_goalmouth(actions, frames).iloc[0]
        assert pd.isna(r["shot_on_target_derived"])  # bar unknowable -> NA (spec 7)

    def test_na_when_not_resolved(self):
        actions, frames = make_match()
        r = compute_shot_goalmouth(actions, frames[~frames["is_ball"].astype(bool)]).iloc[0]
        assert pd.isna(r["shot_on_target_derived"])  # source no_ball_frames -> NA


def test_report_counts():
    actions, frames = make_match()
    out = compute_shot_goalmouth(actions, frames)
    rep = ShotGoalmouthReport.from_frame(out)
    assert rep.n_shots == 1
    assert rep.source_counts == {"observed": 1}
    assert rep.z_profile_counts.get("airborne", 0) == 1


class TestAddAggregator:
    def test_columns_and_passthrough(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth

        out = add_shot_goalmouth(actions, frames)
        for c in (
            "shot_crossing_y",
            "shot_crossing_z",
            "shot_speed",
            "shot_time_to_goal_line",
            "shot_on_target_derived",
            "shot_crossing_source",
            "shot_crossing_confidence",
            "shot_fit_n_frames",
            "shot_fit_rmse",
            "shot_fit_end_reason",
            "shot_z_profile",
        ):
            assert c in out.columns
        assert len(out) == len(actions)
        pd.testing.assert_frame_equal(out[actions.columns], actions)  # input never mutated

    def test_provenance_idempotent_skip(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth

        actions2 = actions.assign(frame_id=1, time_offset_seconds=0.0, n_candidate_frames=1, link_quality_score=1.0)
        out = add_shot_goalmouth(actions2, frames)
        assert not any(c.startswith("frame_id_") for c in out.columns)
        assert not any(c.endswith(("_x", "_y")) and "frame_id" in c for c in out.columns)

    def test_per_series_wrapper(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import shot_crossing_y

        s = shot_crossing_y(actions, frames)
        assert isinstance(s, pd.Series) and s.name == "shot_crossing_y"

    def test_warns_when_mostly_unresolvable(self):
        actions, frames = make_match()
        no_ball = frames[~frames["is_ball"].astype(bool)]
        from silly_kicks.tracking.features import add_shot_goalmouth

        with pytest.warns(UserWarning, match="could not be resolved"):
            add_shot_goalmouth(actions, no_ball)

    def test_no_warning_on_healthy_match(self):
        import warnings as _w

        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth

        with _w.catch_warnings():
            # scoped to UserWarning (OUR edge contract) -- an unscoped "error" filter would
            # also trip on pandas Future/DeprecationWarnings on a future pandas bump
            _w.simplefilter("error", UserWarning)
            add_shot_goalmouth(actions, frames)
