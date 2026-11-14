"""Per-converter delivery tests (spec 2026-11-13-cross-definition §7.1, plan Task 3 Step 2, ADR-133).

Every converter runs the same delivery scene, written in its own native format (``_delivery_scenes``). Each test
fails for a converter that drops the delivery rule, applies it before ``to_spadl_ltr`` (the away and period-2
rows), or loses the provider's own label.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from tests.spadl._delivery_scenes import (
    DELIVERY_CASES,
    EXPECTED_TYPE,
    SCENE,
    expected_cross_native,
    scene_rows,
)

_NAME = dict(enumerate(spadlconfig.actiontypes))
_CONVERTERS = sorted(DELIVERY_CASES)


def _run(conv: str, *, home_starts_left: bool = True, reverse: bool = False) -> pd.DataFrame:
    if conv == "kloppy":
        pytest.importorskip("kloppy")
    case = DELIVERY_CASES[conv]
    native = case.build(home_starts_left)
    if reverse:
        native = case.reverse(native)
    return scene_rows(case.run(native, home_starts_left))


def _types(rows: pd.DataFrame) -> dict[int, str]:
    return {int(str(eid)): _NAME[int(t)] for eid, t in rows["type_id"].items()}


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_scene_reaches_the_rule_in_the_acting_teams_frame(conv):
    """Builder sanity: every scene event survives conversion at its scene position (SPADL LTR, within 1.5 m)."""
    rows = _run(conv)
    assert set(rows.index) == {eid for eid, *_ in SCENE}
    xs = dict(zip(rows.index, rows["start_x"].to_numpy(dtype=float), strict=True))
    ys = dict(zip(rows.index, rows["start_y"].to_numpy(dtype=float), strict=True))
    for eid, _p, _t, _team, kind, x, y in SCENE:
        if kind in ("goalkick", "corner_crossed"):  # providers snap restarts to their spot
            continue
        assert abs(xs[eid] - x) < 1.5 and abs(ys[eid] - y) < 1.5, (eid, conv)


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_native_cross_short_delivery_becomes_pass(conv):
    assert _types(_run(conv))[1] == "pass"


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_native_pass_into_box_from_wide_becomes_cross(conv):
    assert _types(_run(conv))[3] == "cross"


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_restart_ended_set_piece_is_short(conv):
    types = _types(_run(conv))
    assert types[5] == "corner_short"
    assert types[6] == "goalkick"


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_direction_invariance(conv):
    """The away team (event 7) and the home team in period 2 (event 9) attack the other way in the native frame
    of the absolute-frame providers; both crosses must survive. A per-period provider gives the same typing when
    the home team starts on the right."""
    types = _types(_run(conv))
    assert types == EXPECTED_TYPE
    if DELIVERY_CASES[conv].per_period:
        assert _types(_run(conv, home_starts_left=False)) == EXPECTED_TYPE


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_order_insensitive(conv):
    """The native input in reverse file order gives the same re-typing (ADR-065)."""
    assert _types(_run(conv, reverse=True)) == EXPECTED_TYPE


def _synth_rows(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions["original_event_id"].astype(str).str.contains("_synth_")]


def test_synthesized_goalkeeper_pass_has_no_native_label():
    """A converter-synthesized GK distribution row carries no provider label, so cross_native is <NA> (spec 6.2;
    the same treatment GS gives its synthesized rows)."""
    from silly_kicks.spadl import metrica, sportec
    from tests.spadl.test_metrica import _df_metrica_pass_by_gk
    from tests.spadl.test_sportec import _df_play_gk

    m, _ = metrica.convert_to_actions(
        _df_metrica_pass_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
    )
    s, _ = sportec.convert_to_actions(_df_play_gk("throwOut"), home_team_id="T-HOME", home_team_start_left=True)
    for name, out in (("metrica", m), ("sportec", s)):
        synth = _synth_rows(out)
        assert len(synth) >= 1, name  # precondition: the converter did synthesize a distribution row
        assert synth["cross_native"].isna().all(), (name, synth["cross_native"].tolist())


def test_kloppy_dataset_without_pass_types_has_no_native_label():
    """kloppy carries the provider label as PassType qualifiers. A dataset whose deserializer emits none (the
    committed Sportec fixture: 9 passes, 0 PassType qualifiers) has no label, so cross_native is <NA> on every
    row rather than a fabricated False; the scene, which carries PassType.CROSS, stays labelled."""
    pytest.importorskip("kloppy")
    from tests.spadl._converter_cases import CONVERTER_CASES

    case = CONVERTER_CASES["kloppy"]
    out = case.run(case.build_input())
    assert out["type_id"].isin([spadlconfig.actiontype_id["pass"]]).any()  # precondition: passes exist
    assert out["cross_native"].isna().all()
    assert _run("kloppy")["cross_native"].notna().any()


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_cross_native_values(conv):
    rows = _run(conv)
    assert str(rows["cross_native"].dtype) == "boolean"
    expected = expected_cross_native(conv)
    got = {int(str(eid)): v for eid, v in rows["cross_native"].items()}
    for eid, want in expected.items():
        if want is pd.NA:
            assert pd.isna(got[eid]), (conv, eid, got[eid])
        else:
            assert got[eid] is not pd.NA and bool(got[eid]) is want, (conv, eid, got[eid])
