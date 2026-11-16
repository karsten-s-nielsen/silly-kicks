"""Every converter applies the one delivery rule and keeps its native label (ADR-133, spec §6.1-§6.2).

Oracle: the delivery rule re-applied to a converter's own finalized output (synthetic dribbles removed, so the
rule's precondition holds) must reproduce the converter's types exactly. A converter that applied the rule
before ``to_spadl_ltr``, or not at all, fails here. The kloppy fixture is in the absolute frame with both teams,
so it also guards direction handling.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import CROSS_START_X, classify_deliveries
from tests.spadl._converter_cases import CONVERTER_CASES

_ID = spadlconfig.actiontype_id
_DELIVERY_CLASS = {
    _ID[t] for t in ("pass", "cross", "corner_crossed", "corner_short", "freekick_crossed", "freekick_short")
}
_CROSSED = {_ID[t] for t in ("cross", "corner_crossed", "freekick_crossed")}
#: Runs with no provider cross label: SkillCorner has none; the kloppy run is Sportec data, whose kloppy
#: deserializer emits no PassType qualifier (test_converter_deliveries pins both sides of that rule).
_NO_NATIVE_LABEL = {"skillcorner", "kloppy"}


def _statsbomb_real() -> pd.DataFrame:
    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl import statsbomb

    raw = json.loads(pathlib.Path("tests/datasets/statsbomb/raw/events/7584.json").read_text(encoding="utf-8"))
    events = flatten_events(raw, 7584)
    home = events["team_id"].dropna().iloc[0]
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=home)
    return actions


def _gradientsports_synthetic() -> pd.DataFrame:
    from silly_kicks.spadl import gradientsports
    from tests.spadl.test_gradientsports import _load_synthetic_events

    actions, _ = gradientsports.convert_to_actions(
        _load_synthetic_events(), home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
    )
    return actions


def _runs() -> dict[str, tuple[str, Callable[[], pd.DataFrame]]]:
    runs: dict[str, tuple[str, Callable[[], pd.DataFrame]]] = {
        name: (name, lambda c=case: c.run(c.build_input())) for name, case in CONVERTER_CASES.items()
    }
    runs["statsbomb_real_7584"] = ("statsbomb", _statsbomb_real)
    runs["gradientsports_synthetic_match"] = ("gradientsports", _gradientsports_synthetic)
    return runs


_RUNS = _runs()


def _output(run: str) -> tuple[str, pd.DataFrame]:
    provider, fn = _RUNS[run]
    if provider == "kloppy":
        pytest.importorskip("kloppy")
    return provider, fn().reset_index(drop=True)


def _synthetic_dribble(provider: str, actions: pd.DataFrame) -> pd.Series:
    is_dribble = actions["type_id"] == _ID["dribble"]
    if provider == "skillcorner":
        return is_dribble & (actions["action_provenance"] == "derived")
    if provider == "gradientsports":
        return pd.Series(False, index=actions.index)  # GS never calls _add_dribbles; its carries are native
    return is_dribble & actions["original_event_id"].isna()


@pytest.mark.parametrize("run", sorted(_RUNS))
def test_output_matches_the_rule(run):
    provider, out = _output(run)
    base = out[~_synthetic_dribble(provider, out)].reset_index(drop=True)
    expected = classify_deliveries(base, cross_start_x=CROSS_START_X)["type_id"]
    mismatch = base.index[expected.to_numpy() != base["type_id"].to_numpy()]
    assert len(mismatch) == 0, base.loc[mismatch, ["period_id", "team_id", "type_id", "start_x", "start_y"]]


@pytest.mark.parametrize("run", sorted(_RUNS))
def test_cross_native_structure(run):
    provider, out = _output(run)
    assert str(out["cross_native"].dtype) == "boolean"
    if provider in _NO_NATIVE_LABEL:
        assert out["cross_native"].isna().all()
        return
    real = ~out.get("is_synthetic", pd.Series(False, index=out.index)).astype(bool)
    real &= ~_synthetic_dribble(provider, out)
    real &= ~out["original_event_id"].astype(str).str.contains("_synth_", regex=False)  # synthesized GK rows
    delivery = out["type_id"].isin(_DELIVERY_CLASS)
    labelled = delivery
    if provider == "metrica":  # Metrica set pieces are positional: no native crossed/short split
        labelled = out["type_id"].isin([_ID["pass"], _ID["cross"]])
    assert out.loc[real & labelled, "cross_native"].notna().all()
    assert out.loc[~labelled, "cross_native"].isna().all()


@pytest.mark.parametrize("run", sorted(_RUNS))
def test_cross_blocked_keyed_to_cross_native(run):
    _, out = _output(run)
    applicable = out["cross_blocked"].notna()
    assert out.loc[applicable, "cross_native"].fillna(False).astype(bool).all()


def test_real_statsbomb_is_non_vacuous():
    """The real match exercises both directions of re-typing and both native labels."""
    _, out = _output("statsbomb_real_7584")
    open_play = out["type_id"].isin([_ID["pass"], _ID["cross"]])
    native_cross = out["cross_native"] == True  # noqa: E712
    is_cross = out["type_id"] == _ID["cross"]
    assert (open_play & native_cross & ~is_cross).any(), "a native cross re-typed as pass"
    assert (open_play & native_cross & is_cross).any(), "a native cross kept as cross"
    assert (open_play & (out["cross_native"] == False) & ~is_cross).any(), "a native pass kept as pass"  # noqa: E712
