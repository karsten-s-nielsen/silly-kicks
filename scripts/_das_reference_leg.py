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
import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

_T = TypeVar("_T")

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


#: The ball-carrier column native DAS reads to exclude the passer from offside (``_das_pack``'s default
#: ``player_in_possession_col``; pinned equal by tests/scripts/test_das_reference_leg.py). When the frames
#: carry it, the library is told the carrier too: without it accessible-space can mark a carrier who is
#: beyond the defensive line offside, which native never does (combined-cycle Phase B, F6).
_CARRIER_COL = "ball_carrier_player_id"


def _id_token(v: Any) -> str:
    """The token body for one non-null id. An integral id keeps the frozen golden generator's
    ``str(int(v))``; a string id (IDSSE ``DFL-OBJ-*`` / ``DFL-CLU-*``) passes through unchanged."""
    if isinstance(v, str):
        return v
    return str(int(v))


def _id_key(body: str) -> int | str:
    """Parse a token body back to the player join key: an int for an all-digit body, else the string.

    The native leg applies the same rule to ``canonical_id`` (``validate_das_native_parity._native_player_key``),
    so both sides of the parity join key a given raw id identically.
    """
    return int(body) if body.isdigit() else body


def _reference_lib_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Library-input copy: centre coords to the accessible-space origin, float64, ball ``player_id`` ->
    ``"ball"`` / ball team ``None``, string ``"p<id>"`` / ``"t<id>"`` ids (:func:`_id_token`), numpy-object
    string columns. A ball-carrier column (:data:`_CARRIER_COL`), when present, gets the carrier's own
    ``"p<id>"`` token (``None`` where no carrier).

    The object coercion is a no-op under pandas 2 (no arrow default) but keeps the recipe byte-identical
    to the frozen golden generator's ``_lib_frames`` for integral ids (the golden frames carry no carrier).
    """
    out = frames.copy()
    out["x"] = out["x"].astype("float64") - _X_OFFSET
    out["y"] = out["y"].astype("float64") - _Y_OFFSET
    out["vx"] = out["vx"].astype("float64")
    out["vy"] = out["vy"].astype("float64")
    is_ball = out["is_ball"].to_numpy(dtype=bool)
    pid = out["player_id"].astype("object")
    pid[is_ball] = "ball"
    pid[~is_ball] = ["p" + _id_token(v) for v in out.loc[~is_ball, "player_id"]]
    out["player_id"] = pid
    tid = out["team_id"].astype("object")
    tid[is_ball] = None
    tid[~is_ball] = ["t" + _id_token(v) for v in out.loc[~is_ball, "team_id"]]
    out["team_id"] = tid
    out["team_in_possession"] = ["t" + _id_token(v) if pd.notna(v) else None for v in out["team_in_possession"]]
    id_cols = ["player_id", "team_id", "team_in_possession"]
    if _CARRIER_COL in out.columns:
        out[_CARRIER_COL] = ["p" + _id_token(v) if pd.notna(v) else None for v in out[_CARRIER_COL]]
        id_cols.append(_CARRIER_COL)
    for c in id_cols:
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


def best_of(fn: Callable[[], _T], repeat: int) -> tuple[_T, float]:
    """``(result, best_seconds)`` over ``max(1, repeat)`` calls of ``fn`` -- the minimum wall time;
    the result is the last call's."""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    for _ in range(max(1, repeat) - 1):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _common(*, infer_direction: bool, carrier: bool = False) -> dict:
    """The library call recipe: the shared direction column, or the library's OWN inference (the
    das-native 7.2 'GoalMap direction versus reference inference' leg). ``carrier`` forwards the
    ball carrier (:data:`_CARRIER_COL`) as ``player_in_possession_col``, as native uses it."""
    base = {**_REFERENCE_COMMON, "frame_col": "_uframe"}
    if carrier:
        base["player_in_possession_col"] = _CARRIER_COL
    if infer_direction:
        return {**base, "attacking_direction_col": None, "infer_attacking_direction": True}
    return base


def _row_values(result: Any, index: pd.Index, grain: str) -> np.ndarray:
    """One accessible-space result as float64, refusing any shape off the 2.0.15 contract measured on the DGX
    (combined-cycle Phase B, F5): ``team`` results cover the rows WITH possession, ``player`` results cover
    EVERY row, each on the input's own index. A silent positional misread would scramble the comparison."""
    if len(result) != len(index):
        raise RuntimeError(f"accessible-space returned {len(result)} {grain} values for {len(index)} rows")
    if isinstance(result, pd.Series) and not result.index.equals(index):
        raise RuntimeError(f"accessible-space returned {grain} values on an index other than the input rows'")
    return np.asarray(result, dtype=float)


def reference_leg_arrays(
    frames: pd.DataFrame, *, repeat: int = 1, infer_direction: bool = False
) -> dict[str, np.ndarray]:
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
    common = _common(infer_direction=infer_direction, carrier=_CARRIER_COL in lib.columns)

    def _library():
        team = asp.get_dangerous_accessible_space(lib.copy(), **common)
        # the inferred-direction leg needs team DAS only (the comparison is per frame)
        ind = None if infer_direction else asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
        return team, ind

    # Timed in-process: the library compute only -- no interpreter start-up, imports or parquet I/O.
    (team, ind), compute_s = best_of(_library, repeat)
    kept = lib["team_in_possession"].notna().to_numpy()
    as_t = _row_values(team.acc_space, lib.index[kept], "team")
    das_t = _row_values(team.das, lib.index[kept], "team")
    as_p = _row_values(ind.player_acc_space, lib.index, "player") if ind is not None else None
    das_p = _row_values(ind.player_das, lib.index, "player") if ind is not None else None

    per_frame: dict[tuple, tuple[float, float]] = {}
    player_rows: list[tuple] = []
    ti = 0  # team results cover the possession rows only; player results cover every row (read by i)
    for i in range(len(lib)):
        if not kept[i]:
            continue
        r = lib.iloc[i]
        key = (r["game_id"], r["period_id"], r["frame_id"])
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        if as_p is not None and das_p is not None and not bool(r["is_ball"]):
            player_rows.append((*key, _id_key(str(r["player_id"])[1:]), as_p[i], das_p[i]))
        ti += 1

    team_keys = sorted(per_frame)
    pid_sorted = sorted(player_rows, key=lambda t: (t[0], t[1], t[2], t[3]))
    return {
        "team_keys": np.array(team_keys, dtype=object),
        "team_as": np.array([per_frame[k][0] for k in team_keys], dtype=float),
        "team_das": np.array([per_frame[k][1] for k in team_keys], dtype=float),
        # (0, 4) when no player rows (the inferred-direction leg): the writer slices four key columns
        "player_keys": np.array([r[:4] for r in pid_sorted], dtype=object).reshape(len(pid_sorted), 4),
        "player_as": np.array([r[4] for r in pid_sorted], dtype=float),
        "player_das": np.array([r[5] for r in pid_sorted], dtype=float),
        "compute_s": np.asarray(compute_s, dtype=float),
    }


def _main(in_parquet: str, out_dir: str, repeat: int = 1, infer_direction: bool = False) -> None:
    frames = pd.read_parquet(in_parquet)
    arrays = reference_leg_arrays(frames, repeat=repeat, infer_direction=infer_direction)
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
    (out / "timing.json").write_text(
        json.dumps({"compute_s": float(arrays["compute_s"]), "repeat": repeat}), encoding="utf-8"
    )


if __name__ == "__main__":
    _main(
        sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1, len(sys.argv) > 4 and sys.argv[4] == "1"
    )
