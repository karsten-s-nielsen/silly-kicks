"""Per-converter delivery tests (ADR-133).

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
    SCENE,
    expected_cross_native,
    expected_type,
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
def test_scene_typing_and_direction_invariance(conv):
    """Every scene event gets its expected type: a native cross delivered short is a ``pass`` (1), a native pass
    delivered from wide is a ``cross`` (3), a corner keeps the converter's type (5). The away team (7) and
    the home team in period 2 (9) attack the other way in the native frame of the absolute-frame providers; both
    crosses must survive. A per-period provider gives the same typing when the home team starts on the right."""
    assert _types(_run(conv)) == expected_type(conv)
    if DELIVERY_CASES[conv].per_period:
        assert _types(_run(conv, home_starts_left=False)) == expected_type(conv)


@pytest.mark.parametrize("conv", _CONVERTERS)
def test_order_insensitive(conv):
    """The native input in reverse file order gives the same re-typing (ADR-065)."""
    assert _types(_run(conv, reverse=True)) == expected_type(conv)


#: A wide pass that ends in the box (95, 30), then a same-team pass from outside it (80, 30): the gap is a carry that
#: ``_add_dribbles`` fills, and that carry starts in the box.
_CARRY_SCENE = (
    (1, 1, 10.0, "H", "pass", 90.0, 5.0),
    (2, 1, 13.0, "H", "pass", 80.0, 30.0),
    (3, 1, 20.0, "H", "shot", 95.0, 34.0),
)


@pytest.mark.parametrize("conv", ["kloppy", "skillcorner"])
def test_the_rule_runs_before_add_dribbles(conv, monkeypatch):
    """The two converters that moved ``to_spadl_ltr`` up: the next on-ball action of pass 1 is the native pass 2
    (outside the box), so pass 1 stays a ``pass``. Run after ``_add_dribbles``, the rule would see the carry that
    starts in the box and type pass 1 a ``cross``."""
    import tests.spadl._delivery_scenes as ds

    if conv == "kloppy":
        pytest.importorskip("kloppy")
    monkeypatch.setattr(ds, "SCENE", _CARRY_SCENE)
    monkeypatch.setattr(ds, "_ends", lambda: {1: (95.0, 30.0), 2: (95.0, 34.0), 3: (105.0, 34.0)})
    case = DELIVERY_CASES[conv]
    out = case.run(case.build(True), True)
    carries = out[out["type_id"] == spadlconfig.actiontype_id["dribble"]]
    assert ((carries["start_x"] - 95.0).abs() < 1.5).any(), "precondition: a carry from the pass's end exists"
    assert _types(scene_rows(out)) == {1: "pass", 2: "pass", 3: "shot"}


def _synth_rows(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions["original_event_id"].astype(str).str.contains("_synth_")]


def test_synthesized_goalkeeper_pass_has_no_native_label():
    """A converter-synthesized GK distribution row carries no provider label, so cross_native is <NA> (the same
    treatment GS gives its synthesized rows)."""
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
