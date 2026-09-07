"""TF-60 Task 5: the gkdv engine consumes the serve's frame ghost_x/_y (ADR-089 both-axes).

gkdv no longer re-derives frame coordinates with the legacy x-only formula (``ghost_y =
ghost_gr_y``), which mislocated an away-team keeper once the ghost-GK model became both-axes.
The provenance ghost_x/_y must equal serve_ghost_gk_positions' output, checked at BOTH ends.
"""

import numpy as np
import pandas as pd

from silly_kicks.gkdv import build_ghost_frames
from silly_kicks.tracking import serve_ghost_gk_positions
from tests.gkdv._fixtures import in_domain_frames
from tests.tracking.test_ghost_gk import _fitted_model


def _mirror(frames: pd.DataFrame) -> pd.DataFrame:
    """180-degree point reflection of positions + velocities (keeps the domain: ball near x=105)."""
    f = frames.copy()
    f["x"] = 105.0 - f["x"]
    f["y"] = 68.0 - f["y"]
    for c in ("vx", "vy"):
        if c in f.columns:
            f[c] = -f[c]
    return f


def test_engine_provenance_ghost_equals_serve_frame_coords() -> None:
    model = _fitted_model()[0]
    # in_domain_frames: defending keeper at flip=False; its mirror: defending keeper at flip=True
    # (where the legacy x-only y was wrong).
    for frames in (in_domain_frames(), _mirror(in_domain_frames())):
        _cf, prov, _rep = build_ghost_frames(frames, model=model, home_team_id=1)
        scored = prov[prov["drop_reason"].isna()]
        assert len(scored) >= 1
        served = serve_ghost_gk_positions(frames, model=model, home_team_id=1)
        m = scored.merge(
            served[["game_id", "period_id", "frame_id", "gk_team_id", "ghost_x", "ghost_y"]],
            on=["game_id", "period_id", "frame_id", "gk_team_id"],
            suffixes=("_p", "_s"),
        )
        assert len(m) == len(scored)
        np.testing.assert_allclose(
            m["ghost_x_p"].to_numpy(float), m["ghost_x_s"].to_numpy(float), atol=1e-9, equal_nan=True
        )
        np.testing.assert_allclose(
            m["ghost_y_p"].to_numpy(float), m["ghost_y_s"].to_numpy(float), atol=1e-9, equal_nan=True
        )
