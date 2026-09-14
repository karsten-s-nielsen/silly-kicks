"""action_ltr_goal_map is a PUBLIC tracking seam (ADR-055 / TF62-SPEC-09), and territorial_defense
uses the promoted one rather than re-defining it (the ONE goal-end implementation)."""

from __future__ import annotations

import inspect


def test_action_ltr_goal_map_is_public_and_action_ltr():
    from silly_kicks.tracking import action_ltr_goal_map

    gm = action_ltr_goal_map(7, 1, acting_team_id=1, opponent_team_id=2)
    assert gm.attacked_goal(7, 1, 1, allow_guess=True) == 105.0  # acting team attacks opponent's end
    assert gm.attacked_goal(7, 1, 2, allow_guess=True) == 0.0  # opponent attacks acting team's end


def test_territorial_defense_uses_the_promoted_public_seam():
    import silly_kicks.territorial_defense._engine as eng
    from silly_kicks.tracking import action_ltr_goal_map as public

    # same object, not a re-implementation ...
    assert eng.action_ltr_goal_map is public
    # ... and the source file no longer DEFINES it
    assert "def action_ltr_goal_map(" not in inspect.getsource(eng)
