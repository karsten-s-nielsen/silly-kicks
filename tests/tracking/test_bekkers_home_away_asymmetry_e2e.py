"""Owner-gated: ADR-045 makes ``bekkers_pi`` mirror-invariant on REAL tracking data.

``bekkers_pi`` is a rigid-motion invariant of the on-pitch geometry: a 180-degree point
reflection of every player and the ball -- positions AND velocities -- preserves all pairwise
distances and relative velocities, so the per-action pressure must not change. The ADR-045 fix
(D1/D2) is exactly what makes the internal action-LTR re-projection reflect velocities
consistently with positions; before it, an away action's velocities were read against
re-projected positions (defenders modelled running backwards, -38.9%). This e2e confirms the
invariant on REAL data and, crucially, proves it has TEETH: an INCOMPLETE mirror that reflects
positions but NOT velocities (the D1 defect, reconstructed on the input) changes the pressure by
up to ~0.99. Measured max per-action |delta|: complete mirror ~6e-14 (machine precision) on both
IDSSE and Gradient Sports; incomplete mirror ~0.99.

Match-independent by construction, so it runs on BOTH providers and dog-foods the ADR-045
``reflect`` seam on real frames.

Why NOT an away/home pressure ratio band (an earlier draft of this file): ``bekkers_pi`` is
orientation-INVARIANT, so the away/home ratio measures GENUINE match pressing asymmetry, not
orientation. Measured across three GS matches the ratio is 0.61 / 0.91 / 1.02 and on IDSSE 1.14 --
it varies by match, i.e. GS 10502 is simply a lopsided match, not a defect and not a GS
velocity-availability gap (GS carries full per-player velocity and the re-projection fires). The
ratio was the wrong invariant; the mirror is the right one. See ADR-045.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="pining data (PINING_FOR_THE_DATA_TOKEN)"),
]

# Complete-mirror per-action agreement is machine precision (measured max |delta| ~6e-14 on both
# providers); a real position/velocity inconsistency (the D1 defect) moves it by up to ~0.99. A
# strict absolute bound (rtol=0) turns that six-order-of-magnitude gap into a clean pass/fail.
_INVARIANCE_ATOL = 1e-9


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _bekkers(actions, frames) -> np.ndarray:
    from silly_kicks.tracking.features import add_pressure_on_actor

    out = add_pressure_on_actor(actions, frames=frames, methods=("bekkers_pi",))
    col = next(c for c in out.columns if c.startswith("pressure_on_actor__bekkers"))
    return out[col].to_numpy(dtype="float64")


@pytest.mark.parametrize("provider", ["idsse", "gradientsports"])
def test_bekkers_is_invariant_under_a_complete_physical_mirror_on_real_data(provider: str) -> None:
    from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, reflect, reflect_columns

    loader = _load_loader()
    match = next(iter(loader.load_matches(providers=[provider], max_per_provider=1)), None)
    if match is None:
        pytest.skip(f"no {provider} match available from the pining listing")
    _provider, _match_id, actions, frames, _home = match

    base = _bekkers(actions, frames)
    mask = np.ones(len(frames), dtype=bool)

    # COMPLETE physical mirror via the ADR-045 seam: point-reflect x/y (+ x_smoothed/y_smoothed),
    # NEGATE vx/vy, swap the direction labels, leave speed (a magnitude) alone.
    complete = reflect(frames, mask, kinds=TRACKING_REFLECTION_KINDS)
    mirrored = _bekkers(actions, complete)

    base_finite = np.isfinite(base)
    assert base_finite.sum() >= 100, "too few scored actions for the invariant to mean anything"
    # A pure physical mirror cannot change WHICH actions are scoreable.
    assert np.array_equal(base_finite, np.isfinite(mirrored)), (
        f"{provider}: the mirror changed the scoreable-action set -- not a pure physical mirror"
    )
    assert np.allclose(base[base_finite], mirrored[base_finite], rtol=0.0, atol=_INVARIANCE_ATOL), (
        f"{provider}: bekkers_pi is NOT invariant under a complete physical mirror "
        f"(max |delta| {np.nanmax(np.abs(base - mirrored)):.3e}) -- the internal action-LTR "
        f"re-projection is treating positions and velocities inconsistently (ADR-045 D1/D2)."
    )

    # TEETH: an INCOMPLETE mirror that reflects positions + labels but NOT velocities reproduces
    # the D1 defect on the input and MUST move the pressure, else the invariance above is vacuous.
    px = [c for c in ("x", "x_smoothed") if c in frames.columns]
    py = [c for c in ("y", "y_smoothed") if c in frames.columns]
    dl = [c for c in ("team_attacking_direction",) if c in frames.columns]
    incomplete = reflect_columns(frames, mask, point_x=px, point_y=py, direction_label=dl)
    broken = _bekkers(actions, incomplete)
    both = base_finite & np.isfinite(broken)
    assert not np.allclose(base[both], broken[both], rtol=0.0, atol=_INVARIANCE_ATOL), (
        f"{provider}: the incomplete (velocity-unreflected) mirror produced identical pressure "
        f"-- this invariance test has no teeth; velocity reflection is not being exercised."
    )
