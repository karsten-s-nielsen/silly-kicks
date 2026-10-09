"""Soft physical-plausibility guard for derived kinematics (TF-65 §4.3).

NaNs implausible velocity/acceleration, counts them, and emits a dedicated ``PlausibilityWarning``.
NOT a hard raise: out-of-pitch ball frames legitimately exceed the speed bound (~6 % of GS ball rows),
so a raise would fire every match. The guard is the in-cycle safety net that neutralises the
downstream symptom from both the gap bug (fixed at source by the dense-grid reindex) and the
separately-tracked out-of-pitch ball source.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._config_dataclass import PreprocessConfig


class PlausibilityWarning(UserWarning):
    """A built kinematic exceeded a physical-plausibility bound and was set to NaN.

    Its own category (never an umbrella / never subclassing another sk warning) so consumers can
    filter it independently.
    """


def apply_plausibility_guard(frames: pd.DataFrame, config: PreprocessConfig | None = None) -> pd.DataFrame:
    """NaN ``vx``/``vy``/``speed`` where ``speed > max_plausible_speed`` and ``ax``/``ay``/``accel``
    where ``accel > max_plausible_accel``; count + warn. Returns a copy; never raises."""
    cfg = config or PreprocessConfig.default()
    out = frames.copy()

    n_speed = 0
    if "speed" in out.columns:
        speed = out["speed"].to_numpy(dtype=float)
        bad = speed > cfg.max_plausible_speed  # NaN compares False -> untouched
        n_speed = int(np.count_nonzero(bad))
        if n_speed:
            for col in ("vx", "vy", "speed"):
                if col in out.columns:
                    out.loc[bad, col] = np.nan

    n_accel = 0
    if "accel" in out.columns:
        accel = out["accel"].to_numpy(dtype=float)
        bad = accel > cfg.max_plausible_accel
        n_accel = int(np.count_nonzero(bad))
        if n_accel:
            for col in ("accel_x", "accel_y", "accel"):
                if col in out.columns:
                    out.loc[bad, col] = np.nan

    if n_speed or n_accel:
        warnings.warn(
            f"Plausibility guard: NaN'd {n_speed} frame(s) with speed > {cfg.max_plausible_speed} m/s "
            f"and {n_accel} with |accel| > {cfg.max_plausible_accel} m/s^2.",
            PlausibilityWarning,
            stacklevel=2,
        )
    return out
