"""Silent-degradation guards on the DFL parse+shape port (see ``parse.py`` LOCAL HARDENING).

Sibling to ``test_parse_port_parity.py``, deliberately NOT folded into it: that file's
stated scope is "the port reproduces GENUINE lakehouse output" (golden parity), whereas
these are behavioural unit guards on the two silly-kicks-local hardening helpers, which by
construction produce NO change against the golden. Keeping them apart keeps the parity gate
readable as a single-purpose contract.

Both hardened paths were **silently degrading**, not failing:

* the ET integrity check keyed on ``period_id``, a column the events shape does not have
  (it has ``period``), so the RuntimeError branch could never fire; and
* ``ball_status`` was assumed string-like (crashes confusingly on a Delta round-trip that
  returns it numeric) and its produced value domain was never validated (the 4.48.1
  failure class -- an out-of-set ``ball_state`` silently zeroes a provider out of a
  downstream domain filter).
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.providers.sportec import (
    derive_idsse_home_team_start_left,
    derive_idsse_home_team_start_left_extratime,
    shape_tracking_to_native,
)
from silly_kicks.providers.sportec.parse import _BALL_STATE_DOMAIN

_HOME = "DFL-CLU-000008"
_AWAY = "DFL-CLU-00000G"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def warnings_as_errors():
    """Promote ``UserWarning`` to an error -- asserts a code path is warning-FREE."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def _events(*, period_col: str | None, periods: list[int], et_kickoff: bool) -> pd.DataFrame:
    """Minimal shaped-events frame.

    ``period_col=None`` omits the period column entirely (neither name present).
    """
    rows = [
        {
            "event_type": "KickOff",
            "kickoff_game_section": "firstHalf",
            "kickoff_team_left": _HOME,
            "_period": 1,
        }
    ]
    if et_kickoff:
        rows.append(
            {
                "event_type": "KickOff",
                "kickoff_game_section": "extraTimeFirstHalf",
                "kickoff_team_left": _AWAY,
                "_period": 3,
            }
        )
    # Filler non-KickOff rows carrying the periods this match actually contains.
    for p in periods:
        rows.append(
            {
                "event_type": "Play",
                "kickoff_game_section": None,
                "kickoff_team_left": None,
                "_period": p,
            }
        )
    df = pd.DataFrame(rows)
    if period_col is None:
        return df.drop(columns=["_period"])
    return df.rename(columns={"_period": period_col})


def _tracking_bronze(ball_status: pd.Series) -> pd.DataFrame:
    """Minimal ``bronze.idsse_tracking`` frame -- 2 player rows sharing 1 frame."""
    n = len(ball_status)
    return pd.DataFrame(
        {
            "match_id": ["J03WMX"] * n,
            "period": [1] * n,
            "frame": list(range(n)),
            "timestamp": [float(i) * 0.04 for i in range(n)],
            "x": [1.0] * n,
            "y": [2.0] * n,
            "s": [3.0] * n,
            "ball_x": [4.0] * n,
            "ball_y": [5.0] * n,
            "ball_z": [0.5] * n,
            "ball_s": [6.0] * n,
            "ball_status": ball_status.reset_index(drop=True),
            "frame_rate": [25] * n,
            "player_id": [f"DFL-OBJ-{i:04d}" for i in range(n)],
            "team_id": [_HOME] * n,
            "is_goalkeeper": [False] * n,
        }
    )


# ---------------------------------------------------------------------------
# Concern 1 -- the ET integrity check that could never fire
# ---------------------------------------------------------------------------
def test_et_guard_fires_on_period_keyed_frame():
    """THE case that was silently skipped: the events shape keys period as ``period``.

    RED by reverting ``_resolve_period_column`` to the lifted
    ``"period_id" in events.columns and ...`` -- ``has_et_periods`` goes False and this
    returns None instead of raising.
    """
    events = _events(period_col="period", periods=[1, 2, 3], et_kickoff=False)
    with pytest.raises(RuntimeError, match="ET periods"):
        derive_idsse_home_team_start_left_extratime(events, _HOME)


def test_et_guard_fires_on_period_id_keyed_frame():
    """No regression: the tracking-frame shape keys period as ``period_id``."""
    events = _events(period_col="period_id", periods=[1, 2, 4], et_kickoff=False)
    with pytest.raises(RuntimeError, match="ET periods"):
        derive_idsse_home_team_start_left_extratime(events, _HOME)


def test_et_guard_raises_when_no_period_column():
    """ "Could not check" must not be indistinguishable from "check passed".

    RED by making ``_resolve_period_column`` return a default / fall through to False.
    """
    events = _events(period_col=None, periods=[], et_kickoff=False)
    with pytest.raises(RuntimeError, match="no period column"):
        derive_idsse_home_team_start_left_extratime(events, _HOME)


@pytest.mark.parametrize("period_col", ["period", "period_id"])
def test_no_et_periods_returns_none(period_col):
    """Non-vacuity for the two guard tests above: a regulation match still returns None.

    Without this, a helper that raised unconditionally would pass them for the wrong reason.
    """
    events = _events(period_col=period_col, periods=[1, 2], et_kickoff=False)
    assert derive_idsse_home_team_start_left_extratime(events, _HOME) is None


@pytest.mark.parametrize("period_col", ["period", "period_id"])
def test_et_value_derived_when_kickoff_present(period_col):
    """Non-vacuity: with ET periods AND ET KickOff metadata, the real value comes back."""
    events = _events(period_col=period_col, periods=[1, 2, 3], et_kickoff=True)
    assert derive_idsse_home_team_start_left_extratime(events, _HOME) is False
    assert derive_idsse_home_team_start_left_extratime(events, _AWAY) is True


def test_first_half_derivation_is_unaffected_by_period_column():
    """The sibling ``derive_idsse_home_team_start_left`` reads no period column at all --
    its guard is unconditional and was never disabled. Pinned so a future "consistency"
    refactor cannot introduce the same latent gate here."""
    for period_col in ("period", "period_id", None):
        events = _events(period_col=period_col, periods=[1, 2], et_kickoff=False)
        assert derive_idsse_home_team_start_left(events, _HOME) is True
        assert derive_idsse_home_team_start_left(events, _AWAY) is False


# ---------------------------------------------------------------------------
# Concern 2A -- ball_status dtype (Delta round-trip returns it numeric)
# ---------------------------------------------------------------------------
def test_ball_status_int64_produces_correct_ball_state():
    """RED by dropping the numeric normalisation: ``.map()`` on a string-keyed dict yields
    all-NA and ``.str.lower()`` raises ``AttributeError``."""
    native = shape_tracking_to_native(_tracking_bronze(pd.Series([1, 0], dtype="int64")))
    for is_ball in (False, True):
        got = native.loc[native["is_ball"] == is_ball].sort_values("frame_id")["ball_state"].tolist()
        assert got == ["alive", "dead"], f"is_ball={is_ball}"


def test_ball_status_float64_with_nulls_produces_correct_ball_state():
    """The float trap: a nulls-bearing numeric column round-trips as float64, so a naive
    ``.astype(str)`` gives ``"0.0"`` -- which does NOT match the ``"0"`` key.

    RED by replacing the ``Int64`` hop with ``.astype(str)``: the mapped values become NA.
    """
    native = shape_tracking_to_native(_tracking_bronze(pd.Series([1.0, np.nan, 0.0], dtype="float64")))
    for is_ball in (False, True):
        got = native.loc[native["is_ball"] == is_ball].sort_values("frame_id")["ball_state"].tolist()
        assert got[0] == "alive", f"is_ball={is_ball}"
        assert pd.isna(got[1]), f"is_ball={is_ball}"
        assert got[2] == "dead", f"is_ball={is_ball}"


def _lifted_reference(bs: pd.Series) -> pd.Series:
    """The PRE-hardening lifted expression, verbatim -- the oracle for 'string in, untouched'.

    Deliberately a copy rather than an import: it is the frozen thing the hardened helper
    must still reproduce on string-like input, so it must not move when the helper does.
    """
    m = {"0": "dead", "1": "alive"}
    # The ignore mirrors the one on the expression this oracle copies (``parse.py``):
    # ``other=None`` is a valid None->NA fill at runtime, and pandas-stubs over-narrows
    # ``other``. Suppressed rather than rewritten because the copy must stay VERBATIM.
    return bs.map(m).fillna(bs.str.lower()).where(bs.notna(), other=None)  # type: ignore[arg-type]


def test_string_ball_status_is_untouched():
    """Parity anchor: genuinely string-like input takes the original code path verbatim.

    This is what keeps ``test_parse_port_parity.py`` green. Asserted against an oracle that
    mirrors the pre-hardening expression -- values AND dtype -- rather than against a
    hard-coded ``object``: the produced dtype is pandas-version-dependent (pandas 3's
    ``.map()`` infers ``str`` where pandas 2 yields ``object``), and that movement is
    pandas', not ours. Both sides of the oracle move together, so the test pins the real
    invariant on every leg.

    This matters because the parity gate alone would NOT catch a dtype move here -- it
    passes ``check_dtype=False`` and canonicalises string-likes to ``object``.

    RED by normalising unconditionally, or by breaking the legacy ``Alive``/``Dead``
    lowercase fallback.
    """
    raw = pd.Series(["1", "0", "Alive", "Dead"], dtype=object)
    native = shape_tracking_to_native(_tracking_bronze(raw))
    got = native.loc[~native["is_ball"]].sort_values("frame_id")["ball_state"].reset_index(drop=True)
    assert got.tolist() == ["alive", "dead", "alive", "dead"]
    pd.testing.assert_series_equal(got, _lifted_reference(raw), check_names=False)


# ---------------------------------------------------------------------------
# Concern 2B -- unvalidated output domain (the 4.48.1 failure class)
# ---------------------------------------------------------------------------
def test_out_of_domain_ball_status_is_surfaced():
    """An unmapped token passes through as ``ball_state``; it must not do so silently.

    RED by deleting the ``_warn_unexpected_ball_state`` call.
    """
    raw = pd.Series(["1", "unknown_token"], dtype=object)
    with pytest.warns(UserWarning, match="unexpected ball_status token"):
        native = shape_tracking_to_native(_tracking_bronze(raw))
    # Documented behaviour: WARN + pass through (the value is named in the message so an
    # operator can extend the map; coercing to NA would be a second silent transformation).
    assert "unknown_token" in set(native["ball_state"].dropna())


def test_out_of_domain_warning_names_the_token_and_the_row_class():
    """The warning must be actionable, not a bare 'something is wrong'.

    RED by dropping the token or the ``context`` from the message.
    """
    raw = pd.Series(["weird"], dtype=object)
    with pytest.warns(UserWarning) as rec:
        shape_tracking_to_native(_tracking_bronze(raw))
    messages = [str(w.message) for w in rec]
    assert any("weird" in m for m in messages)
    assert any("player rows" in m for m in messages)
    assert any("synthetic ball rows" in m for m in messages)


def test_na_ball_status_is_not_a_domain_violation():
    """Nulls are legitimate (a frame with no BallStatus attribute) -- NA must yield NA and
    must NOT be reported as an out-of-domain value.

    RED by validating without ``.dropna()`` (NA would be stringified to ``"<NA>"``/``"nan"``
    and flagged).
    """
    raw = pd.Series(["1", None], dtype=object)
    with warnings_as_errors():
        native = shape_tracking_to_native(_tracking_bronze(raw))
    players = native.loc[~native["is_ball"]].sort_values("frame_id")
    assert players["ball_state"].tolist()[0] == "alive"
    assert pd.isna(players["ball_state"].tolist()[1])


def test_in_domain_values_never_warn():
    """Non-vacuity for the two warn tests: the happy path must be warning-free, otherwise
    ``pytest.warns`` above could be satisfied by an always-on warning."""
    with warnings_as_errors():
        shape_tracking_to_native(_tracking_bronze(pd.Series(["1", "0", "Alive", "Dead"], dtype=object)))
        shape_tracking_to_native(_tracking_bronze(pd.Series([1, 0], dtype="int64")))


def test_ball_state_domain_matches_tracking_schema():
    """Drift pin: ``parse.py`` duplicates the value set rather than importing it (the parse
    port takes no ``silly_kicks`` runtime import). Pin the copy to the authority.

    RED by editing either side of the pair.
    """
    from silly_kicks.tracking.schema import TRACKING_CATEGORICAL_DOMAINS

    assert _BALL_STATE_DOMAIN == TRACKING_CATEGORICAL_DOMAINS["ball_state"]
