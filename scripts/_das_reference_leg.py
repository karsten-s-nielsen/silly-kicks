"""Faithful accessible-space 2.0.15 DAS reference leg, run in a pinned pandas-2 subprocess.

accessible-space 2.0.15 is a pandas-2-era library. Under pandas 3 Copy-on-Write the array it builds
internally (``PLAYER_POS = dfp.values.reshape(...)``) is read-only, so its offside pre-processing
(``PLAYER_POS[PLAYER_IS_OFFSIDE, :] = np.nan``) raises ``ValueError: assignment destination is
read-only``, which the library catches and silently skips offside -- keeping offside attackers the native
engine correctly removes. The parity leg therefore runs the library in a pinned pandas-2 interpreter
(provisioned separately; see the driver's ``_resolve_reference_python``). See
docs/superpowers/specs/2026-09-27-das-parity-reference-pandas2-design.md and the root-cause record.

sk-free by construction: top-level imports are pandas/numpy/stdlib only, and ``accessible_space`` is
imported LAZILY inside :func:`reference_leg_arrays` so the driver and CI can import this module for the
constants / gate / keying helper without the library present.
"""

from __future__ import annotations

import importlib.metadata
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_X_OFFSET = 52.5
_Y_OFFSET = 34.0

# Byte-for-byte the golden generator's ``_COMMON`` (tests/tracking/_fixtures/das_golden/_generate.py),
# modulo ``attacking_direction_col`` -- here the driver's shared per-frame direction column name rather
# than the generator's ``"dir"``. Gate-checked by tests/scripts/test_das_reference_leg.py.
_REFERENCE_COMMON: dict[str, Any] = dict(
    frame_col="frame_id",
    player_col="player_id",
    team_col="team_id",
    x_col="x",
    y_col="y",
    vx_col="vx",
    vy_col="vy",
    team_in_possession_col="team_in_possession",
    ball_player_id="ball",
    period_col="period_id",
    attacking_direction_col="_das_parity_dir",
    infer_attacking_direction=False,
    use_progress_bar=False,
)


def _reference_lib_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Library-input copy: centre coords to the accessible-space origin, float64, ball ``player_id`` ->
    ``"ball"`` / ball team ``None``, string ``"p<id>"`` / ``"t<id>"`` ids, numpy-object string columns.

    The object coercion is a no-op under pandas 2 (no arrow default) but keeps the recipe byte-identical
    to the frozen golden generator's ``_lib_frames``.
    """
    out = frames.copy()
    out["x"] = out["x"].astype("float64") - _X_OFFSET
    out["y"] = out["y"].astype("float64") - _Y_OFFSET
    out["vx"] = out["vx"].astype("float64")
    out["vy"] = out["vy"].astype("float64")
    is_ball = out["is_ball"].to_numpy(dtype=bool)
    pid = out["player_id"].astype("object")
    pid[is_ball] = "ball"
    pid[~is_ball] = ["p" + str(int(v)) for v in out.loc[~is_ball, "player_id"]]
    out["player_id"] = pid
    tid = out["team_id"].astype("object")
    tid[is_ball] = None
    tid[~is_ball] = ["t" + str(int(v)) for v in out.loc[~is_ball, "team_id"]]
    out["team_id"] = tid
    out["team_in_possession"] = ["t" + str(int(v)) if pd.notna(v) else None for v in out["team_in_possession"]]
    for c in ("player_id", "team_id", "team_in_possession"):
        out[c] = out[c].astype(object)
    return out


def _add_unique_frame_col(lib: pd.DataFrame, col: str = "_uframe") -> pd.DataFrame:
    """Add a dense group code over ``(game_id, period_id, frame_id)``: one code per frame, shared by its
    rows, distinct across periods.

    accessible-space's DAS entry points pivot on ``frame_col`` ALONE, so feeding this collision-free key
    (instead of ``frame_id``) stops the library merging frames that reuse a ``frame_id`` between periods.
    Value-neutral for the DAS itself (a per-frame-independent computation).
    """
    out = lib.copy()
    out[col] = out.groupby(["game_id", "period_id", "frame_id"], sort=True, observed=True).ngroup()
    return out


def _check_reference_env(pandas_version: str, asp_version: str) -> None:
    """Raise unless the reference environment is the one accessible-space 2.0.15 requires. Explicit
    ``raise`` (never ``assert`` -- ``-O`` strips asserts)."""
    if int(pandas_version.split(".")[0]) >= 3:
        raise RuntimeError(
            f"reference leg requires pandas<3 (accessible-space 2.0.15 silently disables offside under "
            f"pandas-3 Copy-on-Write); got pandas {pandas_version}"
        )
    if asp_version != "2.0.15":
        raise RuntimeError(f"reference leg requires accessible-space==2.0.15 (frozen oracle); got {asp_version}")


def reference_leg_arrays(frames: pd.DataFrame) -> dict[str, np.ndarray]:
    """Faithful accessible-space DAS reference for one match (all periods).

    Fail-loud: the offside-skip warning becomes an error, and the environment is checked, BEFORE any
    scoring -- a silent offside skip or an env regression aborts the run rather than comparing the native
    engine against a broken oracle.
    """
    warnings.filterwarnings("error", message="Offside not properly detectable")
    _check_reference_env(pd.__version__, importlib.metadata.version("accessible-space"))
    import accessible_space as asp  # pyright: ignore[reportMissingImports]  # lazy: das-reference dev extra, pandas-2 subprocess only

    lib = _reference_lib_frames(frames).reset_index(drop=True)
    lib = _add_unique_frame_col(lib)  # collision-free frame key (feeds accessible-space's frame_col)
    common = {**_REFERENCE_COMMON, "frame_col": "_uframe"}
    team = asp.get_dangerous_accessible_space(lib.copy(), **common)
    ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
    as_t = np.asarray(team.acc_space, dtype=float)
    das_t = np.asarray(team.das, dtype=float)
    as_p = np.asarray(ind.player_acc_space, dtype=float)
    das_p = np.asarray(ind.player_das, dtype=float)

    kept = lib["team_in_possession"].notna().to_numpy()
    per_frame: dict[tuple, tuple[float, float]] = {}
    player_rows: list[tuple] = []
    ti = 0
    for i in range(len(lib)):
        if not kept[i]:
            continue
        r = lib.iloc[i]
        key = (r["game_id"], r["period_id"], r["frame_id"])
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        if not bool(r["is_ball"]):
            player_rows.append((*key, int(str(r["player_id"])[1:]), as_p[ti], das_p[ti]))
        ti += 1

    team_keys = sorted(per_frame)
    pid_sorted = sorted(player_rows, key=lambda t: (t[0], t[1], t[2], t[3]))
    return {
        "team_keys": np.array(team_keys, dtype=object),
        "team_as": np.array([per_frame[k][0] for k in team_keys], dtype=float),
        "team_das": np.array([per_frame[k][1] for k in team_keys], dtype=float),
        "player_keys": np.array([r[:4] for r in pid_sorted], dtype=object),
        "player_as": np.array([r[4] for r in pid_sorted], dtype=float),
        "player_das": np.array([r[5] for r in pid_sorted], dtype=float),
    }


def _main(in_parquet: str, out_dir: str) -> None:
    frames = pd.read_parquet(in_parquet)
    arrays = reference_leg_arrays(frames)
    out = Path(out_dir)
    tk = arrays["team_keys"]
    pd.DataFrame(
        {
            "game_id": tk[:, 0],
            "period_id": tk[:, 1],
            "frame_id": tk[:, 2],
            "as": arrays["team_as"],
            "das": arrays["team_das"],
        }
    ).to_parquet(out / "team.parquet")
    pk = arrays["player_keys"]
    pd.DataFrame(
        {
            "game_id": pk[:, 0],
            "period_id": pk[:, 1],
            "frame_id": pk[:, 2],
            "player_id": pk[:, 3],
            "as": arrays["player_as"],
            "das": arrays["player_das"],
        }
    ).to_parquet(out / "player.parquet")


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
