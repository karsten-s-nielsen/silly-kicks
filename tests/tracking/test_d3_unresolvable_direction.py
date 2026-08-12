"""P0.4 -- the case the corpus could not supply, PLANTED.

P0.4 asked whether ``acting_team_attacks_rtl``'s "NaN anyway" justification transfers to
``_player_influence``. The helper defaults to ``False`` (no flip) for an action whose team has
no resolvable direction, justified in its docstring by *"such actions produce NaN geometry
anyway because they cannot link to a usable position"*. That argument was made for off-ball
runs; a GRID exists whether or not any single action links, so it has to be re-verified here
rather than inherited.

RUN ON THE REAL CORPUS 2026-08-11: 3 SkillCorner matches, 3,645 actions, **zero** unresolvable
(4.3% of FRAMES lack a direction label, which does not propagate -- the lookup needs only one
labelled row per ``(game, period, team)``). The failure mode never occurred, so the corpus
CANNOT falsify the justification: it is UNTESTED, not confirmed. A sample that cannot produce
the problem has not cleared it.

So the case is BUILT. That is the only way to answer it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import add_player_influence
from tests.tracking._mirror_registry import canonical_scene, gate_xt


def _strip_direction(frames: pd.DataFrame) -> pd.DataFrame:
    """Frames whose direction labels are gone, but which still LINK to their actions.

    Both halves matter. Dropping the labels makes ``acting_team_attacks_rtl`` fall to its
    default; keeping the frames linkable is what separates this from the case the helper's
    docstring already covers ("cannot link to a usable position"). If the justification held,
    an unresolvable-direction row would be NaN anyway -- here it CAN link, so anything it emits
    is emitted on a guessed direction.
    """
    out = frames.copy()
    out["team_attacking_direction"] = None
    return out


def test_the_planted_scene_really_is_unresolvable():
    """Non-vacuity FIRST: if direction still resolves, the probe below proves nothing."""
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

    actions, frames = canonical_scene()
    blind = _strip_direction(frames)

    resolved = acting_team_attacks_rtl(actions, frames)
    defaulted = acting_team_attacks_rtl(actions, blind)

    # Read against the 4.80.0 nullable contract. `.to_numpy(dtype=bool)` -- what this asserted
    # before -- now RAISES on the blind leg, because <NA> has no bool. That raise is the contract
    # working: under the old bare-bool return the two legs were INDISTINGUISHABLE here, since an
    # unresolved action and a resolved left-to-right one both read False.
    assert resolved.notna().all(), "the base scene must resolve EVERY action, or the contrast is muddied"
    assert resolved.fillna(False).to_numpy(dtype=bool).any(), (
        "the base scene must resolve SOME action to rtl, or the contrast is empty"
    )
    assert defaulted.isna().all(), (
        "the planted scene still resolves a direction -- stripping the labels did not reach the "
        "helper, so this probe would be measuring the resolved path"
    )


def test_player_influence_on_UNRESOLVABLE_direction_is_not_PARTIALLY_guessed():
    """An unresolvable-direction row must not be silently half-answered.

    Deliberately NOT asserting the count is zero. ``_player_influence`` inherits the helper's
    documented ``False`` default, and replacing that with Constraint 5's refusal is a
    BEHAVIOURAL decision -- it belongs in an ADR and a release note, not smuggled in under a
    test written to close a measurement. What this pins is the state that is worse than either
    consistent answer: SOME rows guessed and some not, with nothing downstream able to tell
    them apart.

    The measurement itself is recorded in the plan (P0.4). This gate exists so that the next
    person to change the unresolved-direction policy finds a live case instead of a corpus that
    silently never exercises it.
    """
    actions, frames = canonical_scene()
    blind = _strip_direction(frames)

    out = add_player_influence(actions.copy(), blind, gate_xt())

    xt_cols = [c for c in out.columns if c.startswith("off_ball_xt")]
    assert xt_cols, "no xT columns emitted -- the probe would be vacuous"

    emitted = out[xt_cols].to_numpy(dtype=float)
    live_rows = int(np.isfinite(emitted).any(axis=1).sum())

    assert live_rows in (0, len(out)), (
        f"PARTIAL emission on unresolvable direction: {live_rows}/{len(out)} rows carry a value. "
        f"That is worse than either consistent answer -- some rows are computed on a guessed "
        f"direction and some are not, and no column distinguishes them downstream."
    )
