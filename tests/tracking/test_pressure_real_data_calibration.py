"""Per-provider real-data sanity bounds (lakehouse review item 3).

Uses ``load_provider_frames`` + ``synthesize_actions`` from
``tests.tracking._provider_inputs`` -- the same helpers the per-row
regression test uses. This avoids the slim-parquet ``__kind`` discriminator
trap (event-side columns leak into the frame slice if read directly).
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor

from ._provider_inputs import (
    PFF_DIR,
    SLIM_DIR,
    load_provider_frames,
    synthesize_actions,
)


def _load_provider_actions_and_frames(provider: str):
    if provider == "pff":
        # PFF needs the medium_halftime fixture
        if not (PFF_DIR / "medium_halftime.parquet").exists():
            pytest.skip(f"{PFF_DIR}/medium_halftime.parquet missing")
    else:
        if not (SLIM_DIR / f"{provider}_slim.parquet").exists():
            pytest.skip(f"{SLIM_DIR}/{provider}_slim.parquet missing")
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    return actions, frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_andrienko_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider_actions_and_frames(provider)
    out = pressure_on_actor(actions, frames, method="andrienko_oval")
    nan_rate = out.isna().mean()
    assert nan_rate < 0.20, f"{provider} andrienko NaN rate {nan_rate:.2%} > 20%"
    valid = out.dropna()
    in_band = ((valid >= 0) & (valid <= 200)).mean()
    assert in_band >= 0.99, f"{provider} andrienko {in_band:.2%} of values in [0, 200]"


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_link_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider_actions_and_frames(provider)
    out = pressure_on_actor(actions, frames, method="link_zones")
    nan_rate = out.isna().mean()
    assert nan_rate < 0.20
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_bekkers_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider_actions_and_frames(provider)
    if "vx" not in frames.columns:
        from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

        frames = smooth_frames(frames)
        frames = derive_velocities(frames)
    # Some slim providers (Metrica) have ~77% NaN ball coords; Bekkers will NaN
    # those actions. Don't assert a tight nan_rate cap; just check the shape.
    out = pressure_on_actor(actions, frames, method="bekkers_pi")
    if out.dropna().empty:
        pytest.skip(f"{provider}: bekkers all NaN (likely ball-row coverage gap)")
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


# ---------------------------------------------------------------------------
# Spec section 8.4: cross-provider median consistency + TF-3 sanity bounds
# ---------------------------------------------------------------------------


def _prep_frames_for_method(frames, method: str):
    """Ensure ``frames`` has vx/vy when the chosen method requires them."""
    if method == "bekkers_pi" and "vx" not in frames.columns:
        from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

        return derive_velocities(smooth_frames(frames))
    return frames


def _synthesize_actions_with_pressure(
    frames: pd.DataFrame,
    n_actions: int = 10,
    target_d: float = 2.0,
):
    """Synthesize actions whose anchor is exactly ``target_d`` meters from a
    chosen real defender at angle 0 toward the goal-mouth centre (105, 34).

    Provider-independent geometry: by anchoring the action at a deterministic
    offset from a known frame-row defender, all three pressure methods produce
    cross-provider-comparable values. Without this, the cross-provider median
    gate measures provider-dependent distance distributions rather than kernel
    behaviour.

    The chosen defender becomes the "primary presser" at d=target_d, angle 0.
    Other defenders in the same frame contribute to the sum/aggregate per
    method but those contributions vary across providers and average out at
    the median (the dominant signal is the d=2m presser).
    """
    import numpy as np

    candidates = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))].copy()
    candidates = candidates.dropna(subset=["x", "y", "team_id"])
    if candidates.empty:
        return pd.DataFrame()

    # Pick the first n_actions distinct (period_id, frame_id, team_id) triples;
    # for each, choose any defender from team_b and place the actor at
    # (defender.x - target_d, defender.y) so vec(actor->defender) = +x at angle 0
    # and dist = target_d (assuming defender x within field; we filter to ensure).
    chosen: list[dict] = []
    seen_frames: set = set()
    for _, defender_row in candidates.iterrows():
        if len(chosen) >= n_actions:
            break
        key = (int(defender_row["period_id"]), int(defender_row["frame_id"]))
        if key in seen_frames:
            continue
        dx, dy = float(defender_row["x"]), float(defender_row["y"])
        # Anchor actor at d=target_d from defender, on the side away from goal,
        # so threat direction (anchor->goal) and presser direction (anchor->defender)
        # are aligned (angle 0).
        actor_x = dx - target_d  # actor sits at defender.x - target_d, defender ahead toward goal
        actor_y = dy
        if not (0 <= actor_x <= 105 and 0 <= actor_y <= 68):
            continue
        # The actor must be on the OPPOSITE team from the defender. Pick the
        # first non-defender team in the frame.
        same_frame = candidates[
            (candidates["period_id"] == defender_row["period_id"])
            & (candidates["frame_id"] == defender_row["frame_id"])
            & (candidates["team_id"] != defender_row["team_id"])
        ]
        if same_frame.empty:
            continue
        actor_team = same_frame["team_id"].iloc[0]
        actor_player_id = same_frame["player_id"].iloc[0]
        chosen.append(
            {
                "action_id": len(chosen) + 1,
                "period_id": int(defender_row["period_id"]),
                "time_seconds": float(defender_row["time_seconds"]),
                "team_id": actor_team,
                "player_id": actor_player_id,
                "start_x": float(actor_x),
                "start_y": float(actor_y),
                "end_x": float(actor_x),
                "end_y": float(actor_y),
                "type_id": 0,
            }
        )
        seen_frames.add(key)
    _ = np  # keep numpy import alive for future helpers
    return pd.DataFrame(chosen)


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_per_method_cross_provider_median_within_2x(method: str) -> None:
    """Spec section 8.4: per-method medians per provider agree within 2x.

    Catches "Sportec medians 0.4 and Metrica medians 4.0 -- something is wrong"
    pathologies that bound checks alone miss.

    Uses ``_synthesize_actions_with_pressure`` to guarantee non-zero pressure
    distributions per provider (forces actor within 5m of a defender; without
    this, andrienko_oval / link_zones medians collapse to 0 on random actors
    and the test would have to skip).
    """
    medians: dict[str, float] = {}
    for provider in ("sportec", "metrica", "skillcorner"):
        if provider == "pff":
            from ._provider_inputs import PFF_DIR

            if not (PFF_DIR / "medium_halftime.parquet").exists():
                pytest.fail(f"{provider}: PFF fixture missing; cannot proceed without skipping.")
        else:
            if not (SLIM_DIR / f"{provider}_slim.parquet").exists():
                pytest.fail(f"{provider}: slim parquet missing; regenerate with probe_action_context_baselines.py.")
        frames = load_provider_frames(provider)
        # Deterministic geometry: actor sits 2m from a chosen defender at angle 0.
        # All three methods produce comparable values across providers since
        # the geometry is provider-independent.
        actions = _synthesize_actions_with_pressure(frames, n_actions=20, target_d=2.0)
        assert not actions.empty, f"{provider}: no near-defender actions synthesized; fixture too sparse."
        frames = _prep_frames_for_method(frames, method)
        out = pressure_on_actor(actions, frames, method=method)
        valid = out.dropna()
        assert not valid.empty, f"{provider}/{method}: all NaN after pressure-anchored synthesis."
        medians[provider] = float(valid.median())

    nonzero_medians = [m for m in medians.values() if m > 1e-9]
    assert len(nonzero_medians) >= 2, (
        f"{method}: only {len(nonzero_medians)}/3 providers produced non-zero medians "
        f"({medians}); cross-provider comparison requires >= 2."
    )
    ratio = max(nonzero_medians) / min(nonzero_medians)
    assert ratio <= 2.0, f"{method}: cross-provider medians diverge {ratio:.2f}x (want <=2x). Medians: {medians}"


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_atomic_standard_cross_provider_median_within_1pct(method: str) -> None:
    """Spec section 8.4: same method x same provider on matched fixtures should
    produce medians within 1% across atomic vs standard surfaces (catches
    namespace drift)."""
    from silly_kicks.atomic.tracking.features import (
        pressure_on_actor as atomic_pressure,
    )

    for provider in ("sportec", "metrica", "skillcorner"):
        actions, frames = _load_provider_actions_and_frames(provider)
        frames = _prep_frames_for_method(frames, method)
        atomic_actions = actions.copy()
        atomic_actions["x"] = atomic_actions["start_x"]
        atomic_actions["y"] = atomic_actions["start_y"]
        atomic_actions["dx"] = 0.0
        atomic_actions["dy"] = 0.0
        atomic_actions = atomic_actions.drop(columns=["start_x", "start_y"], errors="ignore")
        std_out = pressure_on_actor(actions, frames, method=method)
        atomic_out = atomic_pressure(atomic_actions, frames, method=method)
        std_valid = std_out.dropna()
        atomic_valid = atomic_out.dropna()
        if std_valid.empty or atomic_valid.empty:
            continue
        std_med = float(std_valid.median())
        atomic_med = float(atomic_valid.median())
        if abs(std_med) < 1e-9 and abs(atomic_med) < 1e-9:
            continue
        denom = max(abs(std_med), abs(atomic_med), 1e-9)
        delta_pct = abs(std_med - atomic_med) / denom
        assert delta_pct <= 0.01, (
            f"{provider}/{method}: atomic vs standard medians differ {delta_pct:.4f} "
            f"(want <=1%). std_median={std_med}, atomic_median={atomic_med}"
        )


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_pre_window_real_data_sanity_bounds(provider: str) -> None:
    """Spec section 8.4 TF-3 sanity bounds: arc-length >= displacement (triangle
    inequality), both >= 0, both <= 30m for the 0.5s default window (elite
    sprint upper bound 12 m/s x 0.5s x 5 = 30m generous cap)."""
    from silly_kicks.tracking.features import (
        actor_arc_length_pre_window,
        actor_displacement_pre_window,
    )

    actions, frames = _load_provider_actions_and_frames(provider)
    arc = actor_arc_length_pre_window(actions, frames)
    disp = actor_displacement_pre_window(actions, frames)

    arc_v = arc.dropna()
    disp_v = disp.dropna()
    if arc_v.empty or disp_v.empty:
        pytest.skip(f"{provider}: pre-window all-NaN, no rows to validate")

    assert (arc_v >= 0).all(), f"{provider}: negative arc_length values present"
    assert (disp_v >= 0).all(), f"{provider}: negative displacement values present"
    assert (arc_v <= 30.0).all(), (
        f"{provider}: arc_length > 30m on 0.5s window (max={float(arc_v.max()):.2f}). "
        "Likely position-coordinate noise or unit bug."
    )
    assert (disp_v <= 30.0).all(), f"{provider}: displacement > 30m on 0.5s window (max={float(disp_v.max()):.2f})"

    # Triangle inequality: arc-length >= displacement on every action where both are defined.
    paired = pd.concat({"arc": arc, "disp": disp}, axis=1).dropna()
    if not paired.empty:
        violations = paired[paired["arc"] + 1e-9 < paired["disp"]]
        assert violations.empty, (
            f"{provider}: triangle-inequality violation on {len(violations)} actions "
            f"(arc < displacement). First: {violations.head().to_dict()}"
        )
