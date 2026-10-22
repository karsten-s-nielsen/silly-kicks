"""No-regression quality floor (pure). Compares the head-to-head committed vs fresh scalar; the bulk
re-fit must not silently lower the bar. Fail-closed: a missing or NaN value BLOCKS, never silently passes.

ADR-128 (spec 5.5): `certify` applies the tier's certification policy single-sourced in the registry --
``public`` = the head-to-head pair, ``owner`` = CV parity on the registry ``cv_metric``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def regression_check(committed_value: float, fresh_value: float, *, direction: str, delta: float) -> tuple[bool, dict]:
    if math.isnan(committed_value) or math.isnan(fresh_value):
        return (
            False,
            {
                "regressed": True,
                "reason": "nan",
                "committed": committed_value,
                "fresh": fresh_value,
                "delta": delta,
                "direction": direction,
            },
        )
    if direction == "lower_better":
        regressed = fresh_value > committed_value + delta
    elif direction == "higher_better":
        regressed = fresh_value < committed_value - delta
    else:
        raise ValueError(f"unknown direction {direction!r}")
    detail = {
        "committed": committed_value,
        "fresh": fresh_value,
        "delta": delta,
        "direction": direction,
        "regressed": regressed,
    }
    return (not regressed, detail)


def read_pair(run_metrics: dict, metric: str) -> tuple[float, float]:
    # KeyError is intended: an absent head-to-head pair is a hard failure, never a pass.
    return float(run_metrics[f"committed_{metric}"]), float(run_metrics[f"fresh_{metric}"])


def evaluate_spec(spec, run_metrics: dict) -> tuple[bool, dict]:
    committed_value, fresh_value = read_pair(run_metrics, spec.primary_metric)
    return regression_check(committed_value, fresh_value, direction=spec.metric_direction, delta=spec.delta)


def _dotted(metrics: Mapping[str, Any], path: str) -> float:
    """The value at a dotted ``path`` in a ``metrics.json`` dict (``{shipped_variant}`` is filled from the
    same dict). KeyError when absent: a missing metric is a hard failure, never a pass."""
    cur: Any = metrics
    for part in path.format(shipped_variant=metrics.get("shipped_variant", "")).split("."):
        cur = cur[part]
    return float(cur)


def certify(
    spec,
    tier: str,
    *,
    fresh_metrics: Mapping[str, Any],
    committed_metrics: Mapping[str, Any],
    headtohead: Mapping[str, Any] | None = None,
) -> tuple[bool, dict]:
    """Certify a fresh artifact under the tier's policy (ADR-128 spec 5.5, single-sourced in the registry).

    ``public``: the head-to-head pair (`evaluate_spec`; KeyError when absent). ``owner``: CV parity -- the
    registry ``cv_metric`` read from BOTH ``metrics.json`` files, with the registry ``delta`` and direction
    (the head-to-head is a diagnostic only). ``rebaseline`` is the identity comparison of
    `scripts.hf_jobs.identity` (decisions P3, P15), not a quality floor, so it is refused here.
    """
    basis = spec.certification_for(tier)
    if basis == "public":
        ok, detail = evaluate_spec(spec, dict(headtohead or {}))
        return ok, {**detail, "basis": "headtohead"}
    if basis == "owner":
        if not spec.cv_metric:
            raise ValueError(f"{spec.name}: owner certification needs a registry cv_metric")
        ok, detail = regression_check(
            _dotted(committed_metrics, spec.cv_metric),
            _dotted(fresh_metrics, spec.cv_metric),
            direction=spec.metric_direction,
            delta=spec.delta,
        )
        return ok, {**detail, "basis": "cv_parity", "metric": spec.cv_metric}
    raise ValueError(f"{spec.name}/{tier}: certification {basis!r} is an identity comparison (identity.compare)")
