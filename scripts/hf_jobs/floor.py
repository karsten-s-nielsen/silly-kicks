"""No-regression quality floor (pure). Compares the head-to-head committed vs fresh scalar; the bulk
re-fit must not silently lower the bar. Fail-closed: a missing or NaN value BLOCKS, never silently passes.
"""

from __future__ import annotations

import math


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
