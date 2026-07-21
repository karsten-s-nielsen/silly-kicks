"""PR-S66 — ghost-GK linked-frame restriction (KDE-only, bit-identical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import compute_ghost_gk
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames

_GHOST_COLS = ["ghost_gk_x", "ghost_gk_y"]


def _make_goal_flip_velocity_fixture(home_team_id: int = 1, away_team_id: int = 2) -> tuple[pd.DataFrame, set[int]]:
    """5 frames, 1 period, engineered to exercise BOTH cross-frame deps.

    - Goal-flip dep: home GK sits at x=5 in frames 1-4 (full-period mean stays
      < 52.5 -> defends x=0), but is camped at x=60 in the lone linked frame 5
      (frame-5-alone mean >= 52.5 would flip the inferred goal to x=105).
    - Velocity dep: the home defensive line shifts every frame, so frame 5's
      one-step velocity (vs the real frame-4 predecessor) differs from a
      no-predecessor compute.

    Returns (frames, linked_frame_ids).
    """
    parts = []
    for fid in range(1, 6):
        f = _make_ghost_gk_frames(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            frame_id=fid,
            timestamp=float(fid) * 0.04,
        )
        defmask = (f["team_id"] == home_team_id) & ~f["is_goalkeeper"].astype(bool) & ~f["is_ball"].astype(bool)
        f.loc[defmask, "x"] = f.loc[defmask, "x"] + fid * 2.0  # moving back line
        if fid == 5:
            gkmask = (f["team_id"] == home_team_id) & f["is_goalkeeper"].astype(bool)
            f.loc[gkmask, "x"] = 60.0  # camp high only in the linked frame
        parts.append(f)
    return pd.concat(parts, ignore_index=True), {5}


def _linked_gk_rows(result: pd.DataFrame, link_frame_ids: set) -> pd.DataFrame:
    mask = (
        result["is_goalkeeper"].astype(bool)
        & ~result["is_ball"].astype(bool)
        & result["frame_id"].astype(int).isin(link_frame_ids)
    )
    return result.loc[mask].sort_values(["frame_id", "team_id"]).reset_index(drop=True)


class TestComputeGhostGkRestriction:
    def test_full_vs_restricted_bit_identical_with_explicit_carrier(self):
        # PR-S81/N5: supplying a precomputed carrier (on FULL frames) keeps the
        # restriction byte-identical for kept frames, same as the internal-carrier path.
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()
        carrier = infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]

        full = compute_ghost_gk(frames, model=model, home_team_id=1, carrier=carrier)
        restricted = compute_ghost_gk(frames, model=model, home_team_id=1, carrier=carrier, link_frame_ids=linked)

        f_rows = _linked_gk_rows(full, linked)
        r_rows = _linked_gk_rows(restricted, linked)
        assert len(f_rows) == len(r_rows) > 0
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(f_rows[col].to_numpy(), r_rows[col].to_numpy())

    def test_full_vs_restricted_bit_identical(self):
        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()

        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        restricted = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)

        f_rows = _linked_gk_rows(full, linked)
        r_rows = _linked_gk_rows(restricted, linked)
        assert len(f_rows) == len(r_rows) > 0
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(f_rows[col].to_numpy(), r_rows[col].to_numpy())

    def test_naive_prefilter_discriminates(self):
        """Proves the fixture actually triggers the cross-frame deps: dropping
        unlinked frames BEFORE extraction must change BOTH the extracted features
        (the deps' direct effect) and — through the model — the ghost output.
        The FEATURE-level assertion (M2) pins the fixture to the mechanism
        regardless of how feature-sensitive _fitted_model() is, so the check can't
        pass/fail vacuously on model insensitivity."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()
        fid = next(iter(linked))

        # --- Feature-level: the linked frame's extracted features MUST differ ---
        feat_full, meta_full = _extract_all_ghost_gk_features(frames, home_team_id=1)
        feat_naive, meta_naive = _extract_all_ghost_gk_features(
            frames[frames["frame_id"].astype(int).isin(linked)], home_team_id=1
        )
        row_full = feat_full[meta_full["frame_id"].astype(int) == fid].reset_index(drop=True)
        row_naive = feat_naive[meta_naive["frame_id"].astype(int) == fid].reset_index(drop=True)
        assert len(row_full) == len(row_naive) > 0
        assert not np.allclose(row_full.to_numpy(dtype=float), row_naive.to_numpy(dtype=float), equal_nan=True), (
            "fixture does not change extracted features; cross-frame deps not exercised"
        )

        # --- Output-level: the difference propagates through the model ---
        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        naive = compute_ghost_gk(
            frames[frames["frame_id"].astype(int).isin(linked)],
            model=model,
            home_team_id=1,
        )
        f_rows = _linked_gk_rows(full, linked)
        n_rows = _linked_gk_rows(naive, linked)
        differs = any(
            not np.allclose(f_rows[col].to_numpy(), n_rows[col].to_numpy(), equal_nan=True) for col in _GHOST_COLS
        )
        assert differs

    def test_restriction_shrinks_predict_set(self, monkeypatch):
        # Structural perf guard: spy the feature EXTRACTOR (the dominant remaining cost, ~18x
        # predict_mean; spec 2026-07-20 §9.5). Post-parameters-only, compute_ghost_gk no longer
        # runs predict_density; _extract_all_ghost_gk_features receives link_frame_ids and returns
        # the restricted sample set (_ghost_gk.py:2200-2213), so len(feats) IS the predict set.
        import silly_kicks.tracking._ghost_gk as _gg

        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()

        captured: list[int] = []
        orig = _gg._extract_all_ghost_gk_features

        def spy(*args, **kwargs):
            feats, meta = orig(*args, **kwargs)
            captured.append(len(feats))
            return feats, meta

        monkeypatch.setattr(_gg, "_extract_all_ghost_gk_features", spy)
        compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)
        restricted_n = captured[-1]
        captured.clear()
        compute_ghost_gk(frames, model=model, home_team_id=1)  # full
        full_n = captured[-1]

        assert restricted_n < full_n
        assert restricted_n == 2 * len(linked)  # 2 GKs (home + away) per linked frame

    def test_link_frame_ids_none_unchanged(self):
        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        a = compute_ghost_gk(frames, model=model, home_team_id=1)
        b = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=None)
        gk = a["is_goalkeeper"].astype(bool) & ~a["is_ball"].astype(bool)
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(a.loc[gk, col].to_numpy(), b.loc[gk, col].to_numpy())


class TestAddGhostGkRestriction:
    def _make_actions(self, frame_ids=(5,)):
        # One shot action per linked frame; defending GK is the AWAY team.
        rows = []
        for k, fid in enumerate(frame_ids):
            rows.append(
                {
                    "action_id": k,
                    "game_id": "100",
                    "period_id": 1,
                    "time_seconds": float(fid) * 0.04,
                    "team_id": 1,  # home attacks -> away GK is the ghost target
                    "player_id": "p99",
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "type_name": "shot",
                    "result_name": "fail",
                    "bodypart_name": "foot",
                }
            )
        return pd.DataFrame(rows)

    def test_add_ghost_gk_passes_pointers_frame_ids(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        actions = self._make_actions(frame_ids=(5,))

        captured: dict = {}
        real = ghost_mod.compute_ghost_gk

        def spy(frames_arg, **kwargs):
            captured["link_frame_ids"] = kwargs.get("link_frame_ids")
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", spy)
        add_ghost_gk(actions, frames, model=model, home_team_id=1)

        # Internally-computed pointers (no links kwarg) still drive restriction.
        assert captured["link_frame_ids"] is not None
        assert captured["link_frame_ids"] <= {1, 2, 3, 4, 5}

    def test_add_ghost_gk_output_unchanged_by_restriction(self, monkeypatch):
        """Action-coupled columns identical whether the internal compute is
        restricted (real path) or not (forced full via patched kwarg-strip)."""
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        actions = self._make_actions(frame_ids=(5,))

        restricted = add_ghost_gk(actions, frames, model=model, home_team_id=1)

        real = ghost_mod.compute_ghost_gk

        def force_full(frames_arg, **kwargs):
            kwargs["link_frame_ids"] = None
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", force_full)
        full = add_ghost_gk(actions, frames, model=model, home_team_id=1)

        for col in _GHOST_COLS:
            np.testing.assert_array_equal(restricted[col].to_numpy(), full[col].to_numpy())


class TestGhostGkXfnsRestriction:
    def _states(self):
        # 3 gamestate slots (a0/a1/a2); each a single action on a distinct frame.
        def one(action_id, fid):
            return pd.DataFrame(
                [
                    {
                        "action_id": action_id,
                        "game_id": "100",
                        "period_id": 1,
                        "time_seconds": float(fid) * 0.04,
                        "team_id": 1,
                        "player_id": "p99",
                        "start_x": 80.0,
                        "start_y": 34.0,
                        "type_name": "shot",
                        "result_name": "fail",
                        "bodypart_name": "foot",
                    }
                ]
            )

        return [one(0, 5), one(0, 4), one(0, 3)]

    def test_xfns_union_passed_to_compute(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import ghost_gk_xfns

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()

        captured: dict = {}
        real = ghost_mod.compute_ghost_gk

        def spy(frames_arg, **kwargs):
            captured.setdefault("calls", 0)
            captured["calls"] += 1
            captured["link_frame_ids"] = kwargs.get("link_frame_ids")
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", spy)
        (xfn,) = ghost_gk_xfns(model=model, home_team_id=1)
        xfn(self._states(), frames)

        # Single compute call, restricted to the UNION of the slots' linked frames.
        assert captured["calls"] == 1
        assert captured["link_frame_ids"] is not None
        assert captured["link_frame_ids"] <= {1, 2, 3, 4, 5}

    def test_xfns_output_unchanged_by_restriction(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import ghost_gk_xfns

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        states = self._states()

        (xfn,) = ghost_gk_xfns(model=model, home_team_id=1)
        restricted = xfn(states, frames)

        real = ghost_mod.compute_ghost_gk

        def force_full(frames_arg, **kwargs):
            kwargs["link_frame_ids"] = None
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", force_full)
        (xfn2,) = ghost_gk_xfns(model=model, home_team_id=1)
        full = xfn2(states, frames)

        pd.testing.assert_frame_equal(restricted, full)


def _make_dense_match(
    n_frames: int = 250, home_team_id: int = 1, away_team_id: int = 2
) -> tuple[pd.DataFrame, set[int]]:
    """Multi-frame fixture for the structural call-count guard + a bit-identical
    check at modest size. NOTE: this is the lightweight CI guard, NOT the §5 scale
    measurement — the real full-match timing (n≈3000 rows≈70k, bundled model) is
    Task 5, run manually against the per-half budget."""
    parts = []
    for fid in range(1, n_frames + 1):
        f = _make_ghost_gk_frames(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            frame_id=fid,
            timestamp=float(fid) * 0.04,
        )
        defmask = (f["team_id"] == home_team_id) & ~f["is_goalkeeper"].astype(bool) & ~f["is_ball"].astype(bool)
        f.loc[defmask, "x"] = f.loc[defmask, "x"] + (fid % 7)  # mild movement
        parts.append(f)
    # link every 25th frame (≈ one action/sec)
    linked = set(range(1, n_frames + 1, 25))
    return pd.concat(parts, ignore_index=True), linked


@pytest.mark.e2e
class TestGhostGkRestrictionStructuralGuard:
    def test_bit_identical_at_guard_scale(self):
        model, _, _ = _fitted_model()
        frames, linked = _make_dense_match()

        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        restricted = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)

        f_rows = _linked_gk_rows(full, linked)
        r_rows = _linked_gk_rows(restricted, linked)
        assert len(f_rows) == len(r_rows) > 0
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(f_rows[col].to_numpy(), r_rows[col].to_numpy())

    def test_predict_set_equals_linked_count(self, monkeypatch):
        """Structural perf guard (CI-robust): the restricted feature extraction runs on exactly
        the linked GK-sample count, far below the full-frame sample count.

        Spies the feature EXTRACTOR (the dominant remaining cost post-parameters-only, ~18x
        predict_mean; spec 2026-07-20 §9.5). _extract_all_ghost_gk_features returns the
        link_frame_ids-restricted sample set (_ghost_gk.py:2200-2213)."""
        import silly_kicks.tracking._ghost_gk as _gg

        model, _, _ = _fitted_model()
        frames, linked = _make_dense_match()

        captured: list[int] = []
        orig = _gg._extract_all_ghost_gk_features

        def spy(*args, **kwargs):
            feats, meta = orig(*args, **kwargs)
            captured.append(len(feats))
            return feats, meta

        monkeypatch.setattr(_gg, "_extract_all_ghost_gk_features", spy)
        compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)
        restricted_n = captured[-1]
        captured.clear()
        compute_ghost_gk(frames, model=model, home_team_id=1)
        full_n = captured[-1]

        # 2 GKs per linked frame; full = 2 GKs x 250 frames.
        assert restricted_n == 2 * len(linked)
        assert full_n == 2 * 250
        assert restricted_n < full_n / 5  # large reduction


class TestExtractionRestriction:
    """Task 6 (§5 variant): restricting the HEAVY extraction to linked frames —
    velocity state precomputed over full frames — is byte-identical, including the
    cross-period one-step velocity features."""

    def test_extract_restricted_bit_identical_incl_velocity(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames, linked = _make_goal_flip_velocity_fixture()
        feat_full, meta_full = _extract_all_ghost_gk_features(frames, home_team_id=1)
        feat_restr, meta_restr = _extract_all_ghost_gk_features(frames, home_team_id=1, link_frame_ids=linked)

        # Restricted extraction yields ONLY linked-frame samples.
        assert set(meta_restr["frame_id"].astype(int)) == linked

        # The linked-frame feature rows match full extraction exactly, including
        # defensive_line_speed / defending_centroid_vx (which need frame 4 as the
        # true predecessor of linked frame 5).
        fid = next(iter(linked))
        full_rows = feat_full[meta_full["frame_id"].astype(int) == fid].reset_index(drop=True)
        restr_rows = feat_restr[meta_restr["frame_id"].astype(int) == fid].reset_index(drop=True)
        pd.testing.assert_frame_equal(full_rows, restr_rows)

        # The velocity features must actually be exercised (non-trivial) here.
        assert full_rows["defensive_line_speed"].notna().any()

    def test_extract_none_unchanged(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames, _ = _make_goal_flip_velocity_fixture()
        a, ma = _extract_all_ghost_gk_features(frames, home_team_id=1)
        b, mb = _extract_all_ghost_gk_features(frames, home_team_id=1, link_frame_ids=None)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(ma, mb)


def test_atomic_reexports_add_ghost_gk():
    """Atomic mirror re-exports (no duplicate impl) — inherits the fix."""
    from silly_kicks.atomic.tracking import features as atomic_feat
    from silly_kicks.tracking import features as main_feat

    assert atomic_feat.add_ghost_gk is main_feat.add_ghost_gk
    assert atomic_feat.ghost_gk_xfns is main_feat.ghost_gk_xfns
