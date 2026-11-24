"""The seeded cases that the tracking-batch value snapshots pin (owner ruling 2026-11-23, spec 2026-11-08
tracking-batch section 7).

Each case returns one DataFrame from a public function of a core that the cycle rewrote and whose other tests pin no
output values: the preprocess, the two linkers, the shape-graph core, and ``detect_line_breaking``. The DFL parser
is not here: its goldens (``tests/providers/sportec/test_parse_port_parity.py``) already pin its values.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

PREPROCESS_METHODS: tuple[str, ...] = ("savgol", "butterworth", "ema", "kalman")
_PREPROCESS_KEYS = ["game_id", "period_id", "frame_id", "is_ball", "player_id"]
_PREPROCESS_VALUES = [
    "x_smoothed",
    "y_smoothed",
    "vx",
    "vy",
    "speed",
    "accel_x",
    "accel_y",
    "accel",
    "pos_var",
    "vel_var",
    "accel_var",
]


def _quiet(fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def preprocess(method: str) -> pd.DataFrame:
    """``derive_velocities(smooth_frames(frames, method=method))`` on 48 seeded frames per period (gaps, NaN
    detections, float32 coordinates, two periods)."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from tests._prod_fixtures import make_production_frames

    frames = make_production_frames(n_frames=48, n_periods=2)
    out = _quiet(lambda: derive_velocities(smooth_frames(frames, method=method)))
    return out[[*_PREPROCESS_KEYS, *_PREPROCESS_VALUES]].reset_index(drop=True)


def _line_breaking_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    from tests.tracking.test_line_breaking_independence import _inputs

    return _inputs()


def link(kind: str, inputs: str) -> pd.DataFrame:
    """The pointer frame of ``link_actions_to_frames`` (``kind="tolerance"``) or ``link_actions_to_frames_elastic``,
    on the 120 seeded single-game actions of the line-breaking fixture (``inputs="line_breaking"``) or on the two-game
    fixture (``inputs="two_games"``, 14 actions; the elastic linker links 8 of them)."""
    from silly_kicks.tracking.utils import link_actions_to_frames, link_actions_to_frames_elastic
    from tests._prod_fixtures import make_two_game_inputs

    actions, frames = _line_breaking_inputs() if inputs == "line_breaking" else make_two_game_inputs()
    fn = link_actions_to_frames if kind == "tolerance" else link_actions_to_frames_elastic
    return _quiet(lambda: fn(actions, frames, on_low_coverage="ignore")[0]).reset_index(drop=True)


def line_breaking() -> pd.DataFrame:
    """``detect_line_breaking`` on the seeded fixture that reaches every route of the function."""
    from silly_kicks.tracking import detect_line_breaking

    actions, frames = _line_breaking_inputs()
    return _quiet(lambda: detect_line_breaking(actions, frames)).reset_index(drop=True)


def _team_frames() -> list[np.ndarray]:
    """200 seeded team-frames: the outfield players of one team in one frame, with no NaN position."""
    from tests._prod_fixtures import make_production_frames

    frames = make_production_frames(seed=20261108, n_frames=100, n_periods=1)
    outfield = frames[~frames["is_ball"].to_numpy(dtype=bool) & ~frames["is_goalkeeper"].to_numpy(dtype=bool)]
    outfield = outfield.dropna(subset=["x", "y"])
    return [
        grp[["x", "y"]].to_numpy(dtype="float64")
        for _key, grp in outfield.groupby(["frame_id", "team_id"], sort=True, observed=True)
    ]


def shape_graph_edges() -> pd.DataFrame:
    """Every edge of ``compute_shape_graph`` with its stability, sorted by (team-frame, p, q)."""
    from silly_kicks.tracking import compute_shape_graph

    rows = []
    for tf, positions in enumerate(_team_frames()):
        sg = compute_shape_graph(positions)
        for (p, q), stability in zip(sg.edges, sg.stabilities, strict=True):
            a, b = sorted((int(p), int(q)))
            rows.append((tf, a, b, float(stability)))
    return pd.DataFrame(rows, columns=["team_frame", "p", "q", "stability"]).sort_values(
        ["team_frame", "p", "q"], kind="mergesort", ignore_index=True
    )


def shape_graph_faces() -> pd.DataFrame:
    """Every face of ``compute_shape_graph`` as its sorted vertices, sorted by (team-frame, vertices)."""
    from silly_kicks.tracking import compute_shape_graph

    rows = []
    for tf, positions in enumerate(_team_frames()):
        for face in compute_shape_graph(positions).faces:
            rows.append((tf, "-".join(str(v) for v in sorted(int(i) for i in face))))
    return pd.DataFrame(rows, columns=["team_frame", "vertices"]).sort_values(
        ["team_frame", "vertices"], kind="mergesort", ignore_index=True
    )


def shape_graph_labels() -> pd.DataFrame:
    """``infer_positions`` for each team-frame and player, for a team that attacks +x and one that attacks -x."""
    from silly_kicks.tracking import compute_shape_graph, infer_positions

    rows = []
    for tf, positions in enumerate(_team_frames()):
        sg = compute_shape_graph(positions)
        for direction in (1.0, -1.0):
            for player, label in enumerate(infer_positions(sg, positions, direction)):
                rows.append((tf, direction, player, label.vertical, label.horizontal))
    return pd.DataFrame(rows, columns=["team_frame", "direction", "player", "vertical", "horizontal"])


CASES: dict[str, Callable[[], pd.DataFrame]] = {
    **{f"preprocess_{m}": (lambda m=m: preprocess(m)) for m in PREPROCESS_METHODS},
    "link_tolerance": lambda: link("tolerance", "line_breaking"),
    "link_tolerance_two_games": lambda: link("tolerance", "two_games"),
    "link_elastic_two_games": lambda: link("elastic", "two_games"),
    "line_breaking": line_breaking,
    "shape_graph_edges": shape_graph_edges,
    "shape_graph_faces": shape_graph_faces,
    "shape_graph_labels": shape_graph_labels,
}
