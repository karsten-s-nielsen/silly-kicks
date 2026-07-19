"""GkdvParams' GK-BLIND construction guard (TF-19 PR-3, Task 4).

WHY THIS FILE EXISTS SEPARATELY FROM TASK 8: the guard SHIPS in Task 4 but the plan only
asserted it in Task 8's arm tests, four tasks later. A guard that is unguarded in the
interim is exactly the gap this branch keeps closing, so it is pinned here at the point of
introduction rather than at the point of first use.

The property: a GK-blind configuration must be UNREPRESENTABLE, not merely rejected later
at call time. ``lambda_gk`` exists only on ``SpearmanParams`` -- ``fernandez_bornn`` and
``voronoi`` carry no GK term at all, so a ghost-GK substitution through them silently loses
the keeper's control-rate multiplier and the threat arm measures nothing about the keeper.
Failing at construction stops that object being built and passed around.
"""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.gkdv import GkdvParams


def test_spearman_constructs_and_is_the_default():
    """Non-vacuity: the guard must ADMIT the valid method, not reject everything."""
    assert GkdvParams().pitch_control_method == "spearman"
    assert GkdvParams(pitch_control_method="spearman").lambda_gk == 3.0


@pytest.mark.parametrize("method", ["voronoi", "fernandez_bornn"])
def test_gk_blind_methods_are_rejected_AT_CONSTRUCTION(method: str):
    """Both real alternative pitch-control methods carry no GK term -> unrepresentable."""
    with pytest.raises(ValueError, match="GK-BLIND"):
        GkdvParams(pitch_control_method=method)


def test_rejection_message_names_the_mechanism_not_just_the_rule():
    """An actionable error: it must say WHY, so a reader does not simply widen the allowlist."""
    with pytest.raises(ValueError) as exc:
        GkdvParams(pitch_control_method="voronoi")
    msg = str(exc.value)
    assert "lambda_gk" in msg, "message must name the missing mechanism"
    assert "spearman" in msg, "message must name the permitted value"


def test_params_are_frozen():
    """Registered knobs are echoed into GkdvReport for traceability -- they must not drift
    after construction, or the report would misreport the run."""
    params = GkdvParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.pitch_control_method = "voronoi"  # type: ignore[misc]
