"""Behavioral chirality fingerprint (ADR-037; enforcement in load() lands in PR-2).

A y-mirrored model serves inverted signed features silently --- the 4.18.0-weights class
of bug. The fingerprint is the model's OUTPUTS on a fixed, deliberately y-ASYMMETRIC
synthetic frame: derived from behavior, so a mislabeled artifact cannot satisfy it.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

_CHIRALITY_VERSION = "chirality-probe-1"


def canonical_probe_frame() -> pd.DataFrame:
    """One synthetic frame, goal at x=105, all rows deliberately OFF the y=34 mirror axis."""
    rows = [
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="A1",
            x=80.0,
            y=20.0,
            vx=1.0,
            vy=0.5,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="A2",
            x=88.0,
            y=45.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="B1",
            x=92.0,
            y=25.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="B2",
            x=95.0,
            y=50.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="BGK",
            x=103.0,
            y=30.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=True,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="ball",
            x=82.0,
            y=21.0,
            vx=2.0,
            vy=1.0,
            is_ball=True,
            is_goalkeeper=False,
        ),
    ]
    return pd.DataFrame(rows)


def chirality_fingerprint(predict_on_frame: Callable[[pd.DataFrame], np.ndarray]) -> dict:
    """predict_on_frame: Callable[[pd.DataFrame], np.ndarray] --- the model's own feature
    extraction + predict on the canonical frame. Returns a JSON-serializable dict."""
    frame = canonical_probe_frame()
    frame_sha = hashlib.sha256(json.dumps(frame.to_dict("records"), sort_keys=True, default=str).encode()).hexdigest()
    outputs = np.asarray(predict_on_frame(frame), dtype=float).ravel()
    if not np.all(np.isfinite(outputs)):
        raise ValueError(f"chirality fingerprint produced non-finite outputs: {outputs!r}")
    return {
        "version": _CHIRALITY_VERSION,
        "frame_sha256": frame_sha,
        "outputs": [round(float(v), 10) for v in outputs],
    }


# Tolerance catches a y-mirror (gross) but tolerates cross-platform float noise (~1e-6).
# The fingerprint is computed on the DGX (aarch64) at save; load() re-verifies on x86.
# See ADR-037 § 9 + the 2026-07-17-tf19-pr2 plan's KEY RISK note.
_CHIRALITY_ATOL = 1e-3
_CHIRALITY_RTOL = 1e-2


def verify_chirality(
    recomputed: dict,
    stored: dict | None,
    *,
    legacy_override: bool,
    model_name: str,
    error_cls: type[Exception] | None = None,
) -> None:
    """Fail-closed chirality check at load() (ADR-037 § 9, TF-19 PR-2).

    ``recomputed`` = ``chirality_fingerprint`` re-run on the just-loaded model.
    ``stored`` = the ``chirality`` block from the artifact's metadata.json (``None`` if absent).
    Raises ``error_cls`` on a MISSING fingerprint (every pre-PR-2 artifact = the mis-served ones)
    unless ``legacy_override``; raises on a probe-frame change or an output mismatch beyond the
    cross-platform tolerance.

    ``error_cls`` is the exception each caller's ``load()`` raises for artifact-integrity failures,
    so the chirality error shares that ``load()``'s taxonomy (a consumer catching the model's own
    ``IntegrityError`` catches this too). Defaults to ``_xshot_occurrence.IntegrityError`` — the
    type xS and xCross use throughout; ``_ghost_gk`` passes its own module-local ``IntegrityError``.
    """
    if error_cls is None:
        from silly_kicks.tracking._xshot_occurrence import IntegrityError

        error_cls = IntegrityError

    if stored is None:
        if legacy_override:
            warnings.warn(
                f"{model_name}: loading a weights artifact with NO chirality fingerprint under "
                "legacy_override=True. Every pre-TF-19-PR-2 artifact is y-mirror-mis-served; only "
                "override for an artifact you have independently verified.",
                stacklevel=2,
            )
            return
        raise error_cls(
            f"{model_name}: weights artifact is missing its chirality fingerprint. Every "
            "pre-TF-19-PR-2 artifact is the y-mirror-mis-served class of bug (ADR-037). Refusing "
            "to load; pass legacy_override=True only if independently verified."
        )
    if recomputed.get("frame_sha256") != stored.get("frame_sha256"):
        raise error_cls(
            f"{model_name}: chirality probe frame changed (stored {stored.get('frame_sha256', '')[:8]} "
            f"vs library {recomputed.get('frame_sha256', '')[:8]}). Version skew; refusing to load."
        )
    a = np.asarray(recomputed.get("outputs", []), dtype=float)
    b = np.asarray(stored.get("outputs", []), dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=_CHIRALITY_ATOL, rtol=_CHIRALITY_RTOL):
        raise error_cls(
            f"{model_name}: chirality mismatch --- served outputs {a.tolist()} do not match the "
            f"trained fingerprint {b.tolist()} within tol (atol={_CHIRALITY_ATOL}). This is the "
            "y-mirror-mis-serving signature; refusing to load."
        )
