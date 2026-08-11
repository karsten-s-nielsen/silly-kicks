"""``snapshot_to_tracking_frames`` id dtypes, across whichever pandas the leg resolves.

ADR-055 dropped a dtype PIN as unimplementable: ``TRACKING_FRAMES_COLUMNS`` declares ``int64`` for
``player_id``/``team_id``, the synthesized ball row is NA in both, and ``int64`` cannot hold NA
(``IntCastingNaNError`` on every snapshot). A ``restore_id_dtype``-based pin was then measured to
change nothing -- with it excised, 0 of the 2 tests written for it went red.

The residual question was whether the behaviour DIFFERS across pandas majors, and `TODO.md`
recorded that it was "only checkable on a pandas-3 environment, which CI does not have". That was
false: three of four CI legs already resolve pandas 3 (ubuntu-3.10 is the sole pandas-2 leg), so
this file gets the differential for free. ``tests/test_ci_pandas_span_wired.py`` is what stops that
silently ceasing to be true.

So this asserts the property consumers actually depend on (ADR-019): ids surviving the snapshot
still compare equal, via ``id_compat``, to the same ids in their source form. That is checkable on
either major, and the resolved ``pd.__version__`` is reported in every failure message so a
divergence names itself instead of being inferred.

**Deliberately NOT a dtype literal.** A test asserting ``dtype == "float64"`` passes or fails on
whatever pandas returns, which is exactly what left this question unverifiable for two cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as T
from silly_kicks.id_compat import ids_match
from silly_kicks.tracking.schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    TRACKING_FRAMES_COLUMNS,
)

#: The case with a recorded 2.3.3 measurement ("the concat yields float64 for the as-built
#: numeric-int fixture"), so it is the one the differential most needs. Built with ``np.array``
#: rather than ``pd.array(..., dtype="int64")``: the latter yields a ``NumpyExtensionArray`` and,
#: while the resulting Series dtype is ``int64`` either way, ``np.array`` removes the question.
_NUMPY_INT = "numpy_int"


def _snapshots(id_dtype: str) -> pd.DataFrame:
    """Two players for one action. Required columns per the port's own docstring:
    ``action_id``, ``team_id``, ``is_goalkeeper``, ``x``, ``y`` (``player_id`` optional -- supplied
    here because it is under test)."""
    if id_dtype == _NUMPY_INT:
        players, teams = np.array([7, 9]), np.array([1, 2])
    else:
        # Built as a Series + `astype` rather than `pd.array(..., dtype=...)`: the parametrization
        # carries dtypes as `str`, and pandas-stubs types `pd.array`'s `dtype` as `None` for a
        # plain list argument, so the typed route is a cast. Semantically identical here.
        declared = pd.api.types.pandas_dtype(id_dtype)
        players = pd.Series([7, 9]).astype(declared)
        teams = pd.Series([1, 2]).astype(declared)
    return pd.DataFrame(
        {
            "action_id": [0, 0],
            "player_id": players,
            "team_id": teams,
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "is_goalkeeper": [True, False],
        }
    )


def _actions() -> pd.DataFrame:
    """The port derives game_id/period_id/time_seconds AND the ball position from here; the ball
    row it synthesizes is where the NA ids come from."""
    return pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [10.0],
            "start_x": [50.0],
            "start_y": [34.0],
        }
    )


@pytest.mark.parametrize("id_dtype", [_NUMPY_INT, "Int64", "object"])
def test_ids_survive_the_snapshot_comparably(id_dtype: str) -> None:
    frames, _links = T.snapshot_to_tracking_frames(_snapshots(id_dtype), _actions())

    for col, probe in (("player_id", 7), ("team_id", 1)):
        assert ids_match(frames[col], probe).any(), (
            f"id {probe!r} in column {col!r} no longer compares equal after the snapshot on "
            f"pandas {pd.__version__} with source dtype {id_dtype!r}; result dtype was "
            f"{frames[col].dtype!r}. This is the ADR-019 property consumers depend on, NOT a dtype "
            f"literal -- if a pandas major changed the concat result, THAT is the finding, and it "
            f"belongs in an xfail(strict=True) carrying both majors' measured behaviour."
        )


def test_the_ball_row_stays_NA_rather_than_becoming_a_sentinel() -> None:
    """A non-NA sentinel bypasses ``pd.isna`` routing and crashes downstream opponent guards
    (ADR-027). The synthesized ball row belongs to no team and holds no player."""
    frames, _links = T.snapshot_to_tracking_frames(_snapshots("Int64"), _actions())
    is_ball = frames["is_ball"].astype("boolean").fillna(False).astype(bool)
    ball = frames[is_ball]

    assert len(ball) == 1, f"expected one synthesized ball row, got {len(ball)}"
    assert ball["team_id"].isna().all(), (
        f"the ball row's team_id is not NA on pandas {pd.__version__} -- an absent team became a "
        f"VALUE, which ADR-027 records as a crash source in downstream opponent guards"
    )
    assert ball["player_id"].isna().all(), (
        f"the ball row's player_id is not NA on pandas {pd.__version__}; the ball is not a player"
    )


def test_output_dtypes_match_the_declared_frames_schema() -> None:
    """The port EMITS the schema it claims to emit -- for every column, not just the ids.

    Until ADR-058 this was unimplementable: the base declared a non-nullable `int64` for
    `player_id`/`team_id` while the synthesized ball row is NA in both, so any cast raised. With the
    base at nullable `Int64` the port can finally satisfy its own declaration, and does.

    The id columns are checked against the TWO declarations ADR-058 established rather than a
    single one, because the port's id domain is the CALLER's: numeric ids are the base
    (`Int64`/`int64`), genuinely-string ids are the kloppy family (`object`). No seventh variant is
    invented for this producer -- that would undo the consolidation ADR-058 just performed.
    """
    for source in (_NUMPY_INT, "Int64", "object"):
        frames, _links = T.snapshot_to_tracking_frames(_snapshots(source), _actions())
        # PER COLUMN, not per frame. Only `player_id`/`team_id` are parameterized here; `game_id`
        # comes from `_actions()` and is numeric in every case, so it stays on the BASE declaration
        # even when the player ids are strings. Asserting the kloppy variant wholesale for the
        # object case would demand `game_id: object` and fail against correct output -- the mistake
        # this comment exists to stop the next reader repeating.
        string_ids = source == "object"
        expected = {
            col: (KLOPPY_TRACKING_FRAMES_COLUMNS[col] if string_ids and col in ("player_id", "team_id") else declared)
            for col, declared in TRACKING_FRAMES_COLUMNS.items()
        }
        mismatched = {
            col: (str(frames[col].dtype), expected[col])
            for col in TRACKING_FRAMES_COLUMNS
            if str(frames[col].dtype) != expected[col]
        }
        assert not mismatched, (
            f"source dtype {source!r} on pandas {pd.__version__}: the emitted frame does not match "
            f"its declared schema. {{column: (actual, declared)}} = {mismatched}. Every column is "
            f"declared, so an unlisted dtype here is the producer disagreeing with the contract, "
            f"not a gap in the contract."
        )


def test_the_id_dtype_no_longer_depends_on_the_pandas_MAJOR() -> None:
    """The measured divergence this file was written to characterise is now CLOSED.

    Recorded before the cast landed: a nullable `Int64` source stayed `Int64` on pandas 2.3.3 but
    was promoted to `Float64` on 3.0.5, and a numpy-int source yielded `float64` on both. Those are
    three different answers to one question, decided by the resolver rather than by this library.

    An explicit cast makes the answer the library's. Asserted as a LITERAL here -- deliberately, and
    in tension with this module's own docstring -- because the property under test IS the dtype:
    once the port casts, "whatever pandas returns" is no longer an acceptable answer. The
    behavioural `id_compat` assertions above remain the contract consumers depend on; this one pins
    that the contract is no longer resolver-dependent.
    """
    numeric = {
        source: str(T.snapshot_to_tracking_frames(_snapshots(source), _actions())[0]["player_id"].dtype)
        for source in (_NUMPY_INT, "Int64")
    }
    assert set(numeric.values()) == {"Int64"}, (
        f"numeric id sources produced {numeric} on pandas {pd.__version__}; both must be 'Int64'. A "
        f"'float64' or 'Float64' here means the cast was lost and float-valued ids are back -- the "
        f"exact shape ADR-019 records as rendering '366.0' against a clean '366'."
    )
