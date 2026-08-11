"""Positive control for the velocity-fixture discriminator.

An instrument that reports "nothing found" is indistinguishable from a broken one, and this
particular instrument WAS broken in a way only a control could show. Its first revision contrasted
a stationary frame set against a moving one -- velocity MAGNITUDE -- while the defect it hunts is
velocity PRESENCE: declared available, columns absent, extractor yields NaN, fitted model routes it
down a learned missing-value branch. Under the magnitude contrast ``add_ghost_gk`` moved by exactly
0.0 and was filed velocity-BLIND, so the very consumer ADR-053/4.76.0 was about would have CLEARED
every fixture reaching it.

The control is deliberately SPLIT, and not written the way the plan first sketched it.

The sketch asserted that a reconstructed pre-4.76.0 ghost fixture yields ``value_changed is True``.
That couples the gate to a defect CONTINUING TO EXIST: 4.76.0 repaired the ghost path, so its value
no longer changes -- it refuses. This repo has already shipped that mistake once, when
``test_at_least_one_column_was_adjudicated_a_fabrication`` broke precisely because the fabrication
it asserted the existence of had been repaired. So:

* a PLANTED case proves the engine can still convict (against a consumer measurably sensitive
  TODAY), and
* a companion proves the historical fixtures are SURFACED rather than cleared, without asserting
  which side of the sensitive/refuses line they fall on.

The regression guard that would have failed the old implementation is
``test_the_ghost_consumer_is_not_classified_velocity_blind``.
"""

from __future__ import annotations

import pathlib

import pytest

import scripts.audit_velocity_fixtures as mod

# Reconstructed from the 4.76.0 repair commit `c080c94`, which shows the exact pre-fix shape:
# `speed=0.0, speed_source="native"` with NO vx/vy. Reconstructed rather than referenced because
# both files have since been FIXED -- the shape under test no longer exists in the tree.
_PRE_4760_GHOST_FIXTURE = '''
"""Reconstruction of tests/tracking/test_ghost_gk_orientation.py as it stood before 4.76.0."""
import pandas as pd
from silly_kicks.tracking import add_ghost_gk


def _frames_two_shots():
    return pd.DataFrame([
        dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=7, team_id=1, is_ball=False, is_goalkeeper=True, x=5.0, y=34.0, z=0.0,
            speed=0.0, speed_source="native", ball_state="alive",
            team_attacking_direction="ltr", confidence=None, visibility=None,
            source_provider="test",
        )
    ])


def test_ghost_orientation():
    actions = pd.DataFrame({"game_id": [1], "period_id": [1], "action_id": [0]})
    out = add_ghost_gk(actions, _frames_two_shots(), home_team_id=1)
    assert out is not None
'''

# The mirror fixture called five aggregators, not one -- the breadth is the point: a file reaching
# ANY sensitive consumer must surface, and this one reaches several.
_PRE_4760_MIRROR_FIXTURE = '''
"""Reconstruction of tests/tracking/test_action_ltr_mirror_invariance.py as of pre-4.76.0."""
import pandas as pd
from silly_kicks.tracking import (
    add_defensive_line, add_ghost_gk, add_obso, add_pre_shot_gk_context, add_team_shape,
)


def _frames():
    return pd.DataFrame([
        dict(
            game_id=1, period_id=1, frame_id=1, time_seconds=1.0, frame_rate=25.0,
            player_id=7, team_id=1, is_ball=False, is_goalkeeper=True, x=5.0, y=34.0, z=0.0,
            speed=0.0, speed_source="native", ball_state="alive",
            team_attacking_direction="ltr", confidence=None, visibility=None,
            source_provider="test",
        )
    ])


def test_mirror():
    actions = pd.DataFrame({"game_id": [1], "period_id": [1], "action_id": [0]})
    frames = _frames()
    add_defensive_line(actions, frames)
    add_ghost_gk(actions, frames, home_team_id=1)
    add_obso(actions, frames)
    add_pre_shot_gk_context(actions, frames)
    add_team_shape(actions, frames)
'''

# The NEGATIVE control: identical in every respect except that it supplies real kinematics. Without
# it, a classifier that convicted unconditionally would pass every assertion above.
_REMEDIED_FIXTURE = _PRE_4760_GHOST_FIXTURE.replace(
    'speed=0.0, speed_source="native"',
    'speed=4.27, vx=4.0, vy=1.5, speed_source="native"',
)


@pytest.fixture(scope="module")
def consumers() -> dict[str, dict]:
    """One measured sweep shared by the module -- it runs every tracking ``add_*`` twice."""
    return mod.velocity_consumers()


def _plant(tmp_path: pathlib.Path, name: str, source: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_the_engine_can_still_convict(tmp_path, consumers) -> None:
    """PLANTED case against the conviction engine, using a consumer sensitive TODAY.

    Deliberately not one of the historical fixtures: their consumer was repaired, so asserting a
    conviction through it would couple this gate to a defect's continued existence.
    """
    sensitive = sorted(n for n, d in consumers.items() if d["verdict"] == mod.SENSITIVE)
    assert sensitive, (
        "no consumer measured SENSITIVE at all -- the probe failed to perturb anything, and every "
        "'cleared' verdict this instrument produces is worthless"
    )
    aggregator = sensitive[0]
    source = (
        f"import pandas as pd\n"
        f"from silly_kicks.tracking import {aggregator}\n"
        f"def test_planted():\n"
        f'    frames = pd.DataFrame([dict(speed=0.0, speed_source="native")])\n'
        f"    {aggregator}(pd.DataFrame(), frames)\n"
    )
    verdict = mod.classify(_plant(tmp_path, "test_planted.py", source), consumers)

    assert verdict["claims"] is True
    assert verdict["reaches_consumer"] is True
    assert verdict["value_changed"] is True
    assert verdict["convicted"] is True, (
        f"the engine failed to convict a planted fixture that declares velocity, supplies none, "
        f"and calls {sensitive[0]!r} -- measured sensitive with delta "
        f"{consumers[sensitive[0]]['delta']!r}. Any 'no fixtures affected' conclusion from this "
        f"instrument is worthless."
    )


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("test_pre_4760_ghost.py", _PRE_4760_GHOST_FIXTURE),
        ("test_pre_4760_mirror.py", _PRE_4760_MIRROR_FIXTURE),
    ],
)
def test_the_two_known_ADR053_fixtures_are_surfaced(tmp_path, consumers, name, source) -> None:
    """The measured instances of the defect this instrument hunts must not CLEAR.

    Asserted as "surfaced", not as "convicted": 4.76.0 turned the ghost path's silent fabrication
    into a refusal, so the honest verdict for it today is ``refuses``. Which side of that line it
    falls on is a property of the LIBRARY and may change again; that it is not silently cleared is
    a property of the INSTRUMENT, and that is what this pins.
    """
    verdict = mod.classify(_plant(tmp_path, name, source), consumers)

    assert verdict["claims"] is True, (
        "the source classifier no longer recognises the pre-4.76.0 shape (declares "
        "speed_source='native', supplies no vx/vy) -- every candidate count downstream is wrong"
    )
    assert verdict["convicted"] or verdict["surfaced_refusing"], (
        f"{name} CLEARED. It is a reconstruction of a fixture ADR-053/4.76.0 measured as reaching "
        f"a scored model on 5-of-26 imputed features. calls={verdict['calls']} "
        f"sensitive={verdict['sensitive_calls']} refusing={verdict['refusing_calls']}. An "
        f"instrument that clears the known positives cannot support a 'nothing found' conclusion."
    )


def test_the_ghost_consumer_is_not_classified_velocity_blind(consumers) -> None:
    """The specific regression that made the first revision of this instrument useless.

    ``add_ghost_gk`` is THE consumer of ADR-053/4.76.0. Under the old magnitude contrast it scored
    a 0.0 delta twice over -- the sb360 leg-A fixture declares ``speed_source="unavailable"``, so
    the by-design marker (an ALL-rows predicate) returned NaN on both arms, and an all-NaN column
    contributes no numeric delta. It was filed BLIND, which clears every fixture reaching it.

    This assertion fails against that implementation and passes against the absent-vs-present one.
    """
    verdict = consumers.get("add_ghost_gk", {}).get("verdict")
    assert verdict is not None, "add_ghost_gk is not in the measured consumer map at all"
    assert verdict != mod.BLIND, (
        f"add_ghost_gk classified {verdict!r}: velocity-blind. It is the consumer whose declared-"
        f"but-absent velocity ADR-053 measured as scoring on 5-of-26 imputed features, so 'blind' "
        f"means the probe is not exercising it -- check that both arms declare a speed_source "
        f"CONSISTENT with the velocity they supply, rather than inheriting the fixture's "
        f"'unavailable' marker, which suppresses this consumer by design."
    )


def test_a_fixture_that_supplies_velocity_is_not_convicted(tmp_path, consumers) -> None:
    """Negative control: without it, a classifier that convicts unconditionally passes everything."""
    verdict = mod.classify(_plant(tmp_path, "test_remedied.py", _REMEDIED_FIXTURE), consumers)

    assert verdict["claims"] is False, (
        "a fixture supplying real vx/vy was still classified as claiming-without-velocity -- the "
        "candidate filter is convicting the remedy it is supposed to accept"
    )
    assert verdict["convicted"] is False


def test_the_two_probe_arms_differ_ONLY_in_the_vector() -> None:
    """The contrast must isolate ``vx``/``vy`` presence, or its deltas measure the probe.

    This is the FALSE-POSITIVE guard, and it is not redundant with the positive controls above: a
    contaminated probe convicts innocent fixtures while still surfacing every known positive, so
    every other assertion in this module passes against it.

    It is not hypothetical. An intermediate revision let the PRESENT arm overwrite ``speed`` as
    well as adding the vector. Measured, ``add_action_context`` is UNCHANGED by vx/vy at fixed
    speed -- it reads only the scalar -- so that revision convicted
    ``tests/tracking/test_add_action_context.py`` and its atomic mirror on a delta that was
    entirely the probe moving ``speed`` underneath them. With the arms isolated, both clear.
    """
    from tests.sb360 import _fixture as F

    _actions, frames, _links = F.build_leg_a()
    absent, present = mod._velocity_absent(frames), mod._velocity_present(frames)

    assert set(present.columns) - set(absent.columns) == {"vx", "vy"}, (
        f"the arms differ in columns beyond the vector: {sorted(set(present.columns) ^ set(absent.columns))}"
    )
    for col in absent.columns:
        a, b = absent[col], present[col]
        assert a.equals(b), (
            f"probe arms disagree on {col!r}, which is NOT the column under test -- every delta "
            f"this instrument reports is contaminated by that difference, and consumers reading "
            f"{col!r} will be convicted for the probe's behaviour rather than their own"
        )


def test_no_test_fixture_claims_velocity_and_reaches_a_sensitive_consumer(consumers) -> None:
    """STANDING GATE: the audit's conclusion, pinned so it cannot silently stop being true.

    The sweep found 24 files declaring `speed_source="native"`/`"derived"` while supplying no
    `vx`/`vy`, and **zero** of them reach a consumer whose output moves when the vector is supplied.
    That is a finding worth keeping, not a one-off report: a fixture added tomorrow with the same
    shape, calling one of the seven sensitive aggregators, would be asserting on values the
    extractor could not compute -- the ADR-053/4.76.0 defect, re-created.

    Deliberately NOT a locked count of candidates. Pinning "24" would fail on every unrelated test
    file that mentions `speed_source`, training a reader to bump the number without thinking. The
    property that matters is the INTERSECTION being empty, and that is what is asserted.
    """
    # Anchored on the module's own location, never on CWD: `classify_sources` returns paths
    # relative to the repo root, and a bare `Path(rel)` would resolve against wherever pytest
    # happens to be invoked from -- silently scanning nothing, which this gate would then report as
    # a clean result.
    repo = pathlib.Path(mod.__file__).resolve().parents[1]
    counts = mod.classify_sources(repo / "tests")
    assert counts["claiming_without_vx"], (
        "the candidate scan found NO files declaring velocity without vx/vy. That is not a clean "
        "bill of health -- this repo has ~24 -- it means the scan resolved the wrong root and this "
        "gate is passing vacuously."
    )

    convicted = []
    for rel in counts["claiming_without_vx"]:
        verdict = mod.classify(repo / rel, consumers)
        if verdict["convicted"]:
            convicted.append((verdict["file"], verdict["sensitive_calls"]))

    assert not convicted, (
        f"{len(convicted)} test fixture(s) declare velocity, supply no vx/vy, and call an "
        f"aggregator measured to CHANGE its output when the vector is supplied:\n"
        + "\n".join(f"  {f} -> {calls}" for f, calls in convicted)
        + "\nEach is asserting on values the extractor could not compute. Fix the FIXTURE: supply "
        "real vx/vy consistent with the declared speed, or declare speed_source 'unavailable' if "
        "the source genuinely has no temporal history. Re-run "
        "`python scripts/audit_velocity_fixtures.py` for the full report."
    )


def test_every_consumer_carries_an_auditable_reason(consumers) -> None:
    """A ``blind`` verdict is only trustworthy if its delta is recorded beside it.

    Mirrors ``applicability_deltas`` in ``tests/sb360/_registry.py``: a zero-movement
    classification must be VISIBLE, because a zero delta from a probe that silently failed to
    perturb is indistinguishable from a genuinely velocity-blind consumer.
    """
    for name, detail in consumers.items():
        assert detail["verdict"] in (mod.SENSITIVE, mod.REFUSES, mod.BLIND), name
        if detail["verdict"] == mod.REFUSES:
            assert detail["note"], f"{name} refuses with no recorded exception"
        else:
            assert detail["delta"] is not None, f"{name} has no recorded delta"
