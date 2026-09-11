"""Tiny FOV-observability fixture: linked actions + a partially-cropping ``visible_area``.

Reuses the committed SB360 paired fixture's Leg A builder (``tests/sb360/_fixture.py``) -- so
``actions``/``frames`` are produced by the REAL ``snapshot_to_tracking_frames`` and link with the
REAL linker, never hand-assembled -- and supplies a deliberately partial camera polygon so the
observed-fraction parity gate is meaningful rather than all-``no_polygon``/all-``unlinked``.

The polygon crops the pitch to ``x in [0, 100]`` for most actions, which clips the
goal-side of the ``defenders_in_triangle_to_goal`` triangle (it reaches ``x = 105``) into a
partial ``observed`` fraction strictly between 0 and 1 -- a real crop, exactly what the byte-
identical retirement of the ADR-062 companions must preserve.
"""

from __future__ import annotations

import pandas as pd

from tests.sb360._fixture import build_leg_a

#: The pre-territorial_defense scene (action_ids 0-5). TF-54b (ADR-090) extended the SHARED SB360
#: fixture with actions 6-9 (interceptions + an opponent pass) for its boundary-audit entry; those are
#: irrelevant to FOV observability, so this fixture stays scoped to the original scene -- keeping the
#: byte-identical ADR-062 companion golden valid without regenerating it.
_ORIGINAL_SCENE_IDS = (0, 1, 2, 3, 4, 5)

#: action_ids that receive the cropping polygon. Action 3 is deliberately OMITTED so its regions
#: classify as ``no_polygon`` -- exercising that companion source alongside ``observed``.
_CROPPED_ACTIONS = (0, 1, 2, 4, 5)

#: A camera crop covering ``x in [0, 100]``, full pitch height. The triangle-to-goal region reaches
#: ``x = 105``, so cropping at ``x = 100`` yields a partial ``observed`` fraction (a real crop).
_CROP_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 68.0), (0.0, 68.0)]


def _legs():
    return build_leg_a()


def tiny_actions() -> pd.DataFrame:
    """Canonical-SPADL actions that all link in Leg A (the original 6-action scene; ADR-090)."""
    actions, _frames, _links = _legs()
    return actions[actions["action_id"].isin(_ORIGINAL_SCENE_IDS)].reset_index(drop=True)


def tiny_frames() -> pd.DataFrame:
    """Freeze-frame tracking frames (one per action), built by the real producer (original scene)."""
    _actions, frames, _links = _legs()
    return frames[frames["frame_id"].isin(_ORIGINAL_SCENE_IDS)].reset_index(drop=True)


def tiny_visible_area() -> pd.DataFrame:
    """Per-``action_id`` camera polygons; a partial crop on most actions, absent on one."""
    rows = [{"action_id": aid, "polygon": list(_CROP_POLYGON)} for aid in _CROPPED_ACTIONS]
    return pd.DataFrame(rows)
