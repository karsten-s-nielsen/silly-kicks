"""Contract test: the DAS reference leg against the REAL accessible-space 2.0.15 (combined-cycle Phase B).

The parity driver compares native DAS with accessible-space run through ``scripts/_das_reference_leg.py``.
Three facts about the library carry that comparison, and ``test_das_reference_leg.py`` can only fake them:

* team results cover the rows WITH possession, player results cover EVERY row (F5);
* the library counts a ball carrier beyond the defensive line as offside unless it is told who the
  carrier is, while native never does (F6);
* string provider ids (IDSSE ``DFL-OBJ-*``) pass through as ids (F1).

Here the real library runs on small synthetic scenes, and the reference leg must reproduce native DAS
within the golden bound, with a negative control proving the F6 scene really exercises carrier offside.

The library is the dev-only ``das-reference`` extra, absent from the normal CI legs, where this module
skips. The dedicated ``das-reference-contract`` CI job sets ``SK_REQUIRE_DAS_REFERENCE=1``, which turns a
missing library or a pandas-3 runtime into a FAILURE, never a silent skip
(``tests/test_ci_das_reference_contract_wired.py`` pins that job).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

_REQUIRED = os.environ.get("SK_REQUIRE_DAS_REFERENCE") == "1"
if _REQUIRED:
    # the contract job: a missing library FAILS, never skips (dev-only extra, so absent for pyright in CI lint)
    import accessible_space  # pyright: ignore[reportMissingImports]  # noqa: F401

    if int(pd.__version__.split(".")[0]) >= 3:
        raise RuntimeError(f"the das-reference contract needs pandas<3 (ADR-107); got {pd.__version__}")
else:
    pytest.importorskip("accessible_space")
    if int(pd.__version__.split(".")[0]) >= 3:
        pytest.skip("accessible-space 2.0.15 needs pandas<3 (ADR-107)", allow_module_level=True)

from scripts import _das_reference_leg as R  # noqa: E402
from scripts import validate_das_native_parity as D  # noqa: E402

_K: list[str] = list(D._FRAME_KEYS)

# One deterministic 11-v-11 layout (SPADL metres, team 1 attacks +x). Team 2 holds a line at x <= 95 plus
# a keeper at 103, so the second-last defender sits at x = 95.
_ATTACK = [(40, 20), (45, 50), (55, 34), (60, 10), (62, 58), (70, 25), (72, 45), (80, 34), (85, 15), (88, 55), (90, 34)]
_DEFEND = [
    (75, 34),
    (78, 20),
    (78, 48),
    (82, 30),
    (82, 40),
    (86, 18),
    (86, 52),
    (90, 28),
    (92, 40),
    (95, 34),
    (103, 34),
]


def _frame(fid: int, *, possession: int | None, ball=(60.0, 34.0), moves: dict | None = None) -> list[dict]:
    moves = moves or {}
    rows = []
    for team, layout, vx in ((1, _ATTACK, 2.0), (2, _DEFEND, -1.0)):
        for k, (x, y) in enumerate(layout):
            pid = team * 100 + k + 1
            x, y = moves.get(pid, (x, y))
            rows.append(
                {"player_id": pid, "team_id": team, "is_ball": False, "x": x, "y": y, "vx": vx, "vy": 0.5 * (k % 3 - 1)}
            )
    rows.append({"player_id": None, "team_id": None, "is_ball": True, "x": ball[0], "y": ball[1], "vx": 3.0, "vy": 0.0})
    for r in rows:
        r.update({"game_id": 1, "period_id": 1, "frame_id": fid, "team_in_possession": possession})
    return rows


def _frames(rows: list[dict]) -> pd.DataFrame:
    f = pd.DataFrame(rows)
    for c in ("player_id", "team_id", "team_in_possession"):
        f[c] = f[c].astype("Int64")
    f["is_ball"] = f["is_ball"].astype(bool)
    f[D._DIR_COL] = 1.0  # team 1 attacks +x in every frame
    return f


def _with_string_ids(f: pd.DataFrame) -> pd.DataFrame:
    """IDSSE-shaped ids: DFL-OBJ-* players, DFL-CLU-* teams (the ball row keeps NA)."""
    out = f.copy()
    out["player_id"] = [pd.NA if pd.isna(v) else f"DFL-OBJ-{int(v):06d}" for v in f["player_id"]]
    for c in ("team_id", "team_in_possession"):
        out[c] = [pd.NA if pd.isna(v) else f"DFL-CLU-{int(v):06d}" for v in f[c]]
    for c in ("player_id", "team_id", "team_in_possession"):
        out[c] = out[c].astype(object)
    return out


def _carrier_scene() -> pd.DataFrame:
    """Ball near the attacked byline; attacker 111 carries it, 0.8 m AHEAD of the ball and beyond the
    second-last defender (x = 95): offside by position, but the carrier can never be offside."""
    f = _frames(_frame(20, possession=1, ball=(100.0, 30.0), moves={111: (100.8, 30.0)}))
    f["ball_carrier_player_id"] = pd.array([111] * len(f), dtype="Int64")
    return f


def _reference(frames: pd.DataFrame) -> dict:
    with warnings.catch_warnings():  # reference_leg_arrays installs an "error" filter; keep it scoped
        warnings.simplefilter("ignore", UserWarning)  # the library's own advisory warnings
        return R.reference_leg_arrays(frames)


def _parity(frames: pd.DataFrame, ref: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team, player) frames joining the reference leg to native numpy in reference quadrature."""
    ref = _reference(frames) if ref is None else ref
    np_team, np_player, _ = D._run_native(frames, D._REFERENCE_PARAMS, engine="numpy")
    team = D._join_on_keys(ref["team_keys"], {"ref_das": ref["team_das"], "ref_as": ref["team_as"]}, np_team, _K)
    player = D._join_on_keys(
        ref["player_keys"], {"ref_das": ref["player_das"], "ref_as": ref["player_as"]}, np_player, [*_K, "player_id"]
    )
    return team, player


def _assert_reproduces(team: pd.DataFrame, player: pd.DataFrame, *, n_frames: int, n_players: int) -> None:
    tol = D._GOLDEN_TOL_NUMPY
    assert len(team) == n_frames and len(player) == n_players  # every scored row joined: keys agree
    for df, grain in ((team, "team"), (player, "player")):
        for out in ("das", "as"):
            ref, nat = df[f"ref_{out}"].to_numpy(float), df[f"{grain}_{out}"].to_numpy(float)
            assert np.array_equal(np.isfinite(ref), np.isfinite(nat)), (grain, out)
            assert np.allclose(ref, nat, rtol=tol, atol=tol, equal_nan=True), (grain, out, np.nanmax(abs(ref - nat)))
    assert (team["ref_das"] > 0).any(), "non-vacuous: some frame must have positive dangerous space"


def test_the_library_result_shapes_the_reference_leg_relies_on():
    # F5's premise, on the real library: team results for the possession rows only, player results for
    # every row, both on the input rows' own index.
    import accessible_space as asp  # pyright: ignore[reportMissingImports]  # dev-only das-reference extra

    frames = _frames(_frame(10, possession=None) + _frame(11, possession=1))
    lib = R._add_unique_frame_col(R._reference_lib_frames(frames).reset_index(drop=True))
    kept = lib["team_in_possession"].notna().to_numpy()
    common = R._common(infer_direction=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        team = asp.get_dangerous_accessible_space(lib.copy(), **common)
        ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
    assert 0 < kept.sum() < len(lib)  # the scene has rows without possession, or this checks nothing
    assert len(team.das) == kept.sum() and len(ind.player_das) == len(lib)
    if isinstance(ind.player_das, pd.Series):
        assert ind.player_das.index.equals(lib.index)


def test_a_dead_ball_frame_first_does_not_shift_the_player_grain():
    # F5 end to end: a frame without possession comes first; the player values must still match native.
    frames = _frames(_frame(10, possession=None) + _frame(11, possession=1))
    team, player = _parity(frames)
    _assert_reproduces(team, player, n_frames=1, n_players=22)


def test_a_carrier_beyond_the_line_matches_native_only_when_forwarded():
    # F6 end to end, with its negative control: native excludes the carrier from offside. Forwarded, the
    # library agrees exactly; withheld (the pre-fix recipe), it removes the carrier and the frame differs.
    frames = _carrier_scene()
    team, player = _parity(frames)
    _assert_reproduces(team, player, n_frames=1, n_players=22)

    withheld = _reference(frames.drop(columns=["ball_carrier_player_id"]))
    team_w, _ = _parity(frames, ref=withheld)
    gap = float(np.max(np.abs(team_w["ref_das"].to_numpy(float) - team_w["team_das"].to_numpy(float))))
    assert gap > 1e-6, "the scene must exercise carrier offside, or the forwarded check above proves nothing"


def test_string_provider_ids_reproduce_native():
    # F1 end to end: IDSSE-shaped string ids through the real library, with the F5 dead-ball frame first.
    frames = _with_string_ids(_frames(_frame(10, possession=None) + _frame(11, possession=1)))
    team, player = _parity(frames)
    _assert_reproduces(team, player, n_frames=1, n_players=22)
    assert all(str(p).startswith("DFL-OBJ-") for p in player["player_id"])
