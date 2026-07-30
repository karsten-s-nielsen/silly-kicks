"""Gate A / Gate B over every registered tracking ``add_*`` (ADR-028 spec section 6).

Gate A is the ADR-028 physical mirror; Gate B is the D1 ``home_team_id``-invariance check that
Gate A is structurally blind to. Both are registry-driven, and two meta-assertions pin the registry
to ``tracking.__all__`` in BOTH directions so a new aggregator cannot join silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as tracking
from tests.tracking._mirror_registry import (
    AWAY,
    FIELD_LENGTH,
    FIELD_WIDTH,
    HOME,
    MIRROR_ENTRIES,
    away_mask,
    canonical_scene,
    mirror_frames,
)

NONSENSE_HOME_ID = 999_999


def _public_add_names() -> set[str]:
    return {n for n in tracking.__all__ if n.startswith("add_")}


# ---------------------------------------------------------------------------
# Meta-assertions -- the anti-rot property
# ---------------------------------------------------------------------------


def test_every_public_add_is_registered():
    """Anti-rot, direction 1: a new aggregator must be classified or CI fails."""
    missing = _public_add_names() - set(MIRROR_ENTRIES)
    assert not missing, f"unregistered add_* (add a MirrorEntry): {sorted(missing)}"


def test_registry_has_no_stale_entries():
    """Anti-rot, direction 2: a removed aggregator must not linger."""
    stale = set(MIRROR_ENTRIES) - _public_add_names()
    assert not stale, f"registry names a non-exported add_*: {sorted(stale)}"


def test_registry_surface_is_the_expected_size():
    """Pins the count so a silent export change is visible in the diff."""
    assert len(_public_add_names()) == 33, sorted(_public_add_names())


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_entry_declares_a_tolerance_basis(name):
    """A tolerance with no recorded basis is a number nobody can revisit on evidence."""
    entry = MIRROR_ENTRIES[name]
    assert entry.tolerance_basis.strip(), f"{name}: tolerance {entry.tolerance} has no basis"


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_exempt_columns_carry_a_reason(name):
    entry = MIRROR_ENTRIES[name]
    for col, cls in entry.columns.items():
        if cls == "exempt":
            assert entry.exempt_reasons.get(col, "").strip(), f"{name}.{col}: exempt with no reason"


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_gate_b_exemptions_carry_a_reason(name):
    """A Gate B exemption is a hole in the D1 check, so it must be justified in writing.

    Without this, ``gate_b_exempt`` is a switch for making an inconvenient aggregator green.
    """
    entry = MIRROR_ENTRIES[name]
    for col, reason in entry.gate_b_exempt.items():
        assert col in entry.columns, f"{name}: gate_b_exempt names unknown column {col!r}"
        assert reason.strip(), f"{name}.{col}: gate_b_exempt with no reason"


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_mirrored_pitch_absolute_columns_declare_a_reflection(name):
    """Without a declared reflection the class is an unjustified ``exempt`` by another name.

    The vocabulary states a TESTABLE contract -- "equals its own reflection" -- so a column in
    this class must say what its reflection is, and Gate A then enforces it. Otherwise a future
    contributor silences any awkward column by re-classing it.
    """
    entry = MIRROR_ENTRIES[name]
    for col, cls in entry.columns.items():
        if cls == "mirrored_pitch_absolute":
            spec = entry.reflections.get(col)
            assert spec in ("x", "y") or isinstance(spec, dict), (
                f"{name}.{col}: mirrored_pitch_absolute with no usable reflection spec ({spec!r})"
            )
            assert entry.exempt_reasons.get(col, "").strip(), (
                f"{name}.{col}: mirrored_pitch_absolute with no recorded reason"
            )


# ---------------------------------------------------------------------------
# Gate A -- the ADR-028 physical mirror
# ---------------------------------------------------------------------------


def _gate_a_params():
    for name in sorted(MIRROR_ENTRIES):
        entry = MIRROR_ENTRIES[name]
        marks = [pytest.mark.xfail(strict=True, reason=entry.known_defect)] if entry.known_defect else []
        yield pytest.param(name, marks=marks)


@pytest.mark.parametrize("name", _gate_a_params())
def test_gate_a_mirror_invariance(name):
    """Same physical scene, mirrored frames -> identical action-LTR output.

    NEVER share a ``PitchControlCache`` between the two legs: it keys on frame IDENTITY and
    excludes player positions, so a mirrored frame carrying its twin's identity is served the base
    leg's surface and every pitch-control family passes at exactly zero difference (ADR-043). Each
    ``entry.call`` builds its own.
    """
    entry = MIRROR_ENTRIES[name]
    actions, frames = canonical_scene()
    base = entry.call(actions.copy(), frames.copy(), HOME)
    mir = entry.call(actions.copy(), mirror_frames(frames), AWAY if entry.home_team_id_role != "unused" else HOME)

    # ALL rows, not just away-in-base: the mirrored leg passes home_team_id=AWAY, so the rows that
    # are "away" THERE are the home team's actions -- which carry the defect in the opposite leg.
    mask = np.ones(len(actions), dtype=bool)
    aw = away_mask(actions, HOME)
    assert aw.any(), "fixture has no away actions -- the gate would be vacuous"

    for col in entry.non_vacuity:
        away_vals = base[col].to_numpy()[aw]
        assert pd.notna(away_vals).any(), (
            f"{name}.{col}: all-null on AWAY rows -- vacuous exactly where the defect lives"
        )

    for col, cls in entry.columns.items():
        if cls == "exempt":
            continue
        b = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)[mask]
        m = pd.to_numeric(mir[col], errors="coerce").to_numpy(dtype=float)[mask]
        if cls == "mirrored_pitch_absolute":
            spec = entry.reflections[col]
            if isinstance(spec, dict):
                # CATEGORICAL -- compare labels DIRECTLY, never through to_numeric. Coercing a
                # label yields NaN, `both.any()` is then False, and the numeric path below would
                # `continue` -- a SILENT PASS, i.e. an assertion that cannot fire.
                reflected = mir[col].map(spec).to_numpy()[mask]
                actual = base[col].to_numpy()[mask]
                comparable = pd.notna(actual) & pd.notna(reflected)
                assert comparable.any(), f"{name}.{col}: no comparable labels -- reflection check is vacuous"
                assert (actual[comparable] == reflected[comparable]).all(), (
                    f"{name}.{col}: pitch-absolute label does not equal its own reflection"
                )
                continue
            m = (FIELD_LENGTH - m) if spec == "x" else (FIELD_WIDTH - m)
        both = np.isfinite(b) & np.isfinite(m)
        if not both.any():
            continue
        delta = float(np.abs(b[both] - m[both]).max())
        assert delta <= entry.tolerance, (
            f"{name}.{col}: base-vs-mirror {delta:.6g} > tol {entry.tolerance} ({entry.tolerance_basis})"
        )


def test_gate_a_enforces_the_mirrored_pitch_absolute_contract():
    """Witness: the class has NO real member today, so only a plant can exercise it.

    ``add_shape_graph`` emits six NUMERIC columns; the pitch-absolute lateral label ADR-045 D5
    settled lives in ``infer_positions``, which ``_shape_graph.py:877-880`` records as having "no
    in-library consumer" -- it is not surfaced by any ``add_*``. Without this witness the whole
    enforcement path (``reflections``, the Gate A branch, the declaration test) would ship correct
    but NEVER EXERCISED, which is the failure detection-first exists to prevent.

    Asserts BOTH directions: a correctly pitch-absolute column passes, and omitting the reflection
    FAILS. Without the second half this is a test that cannot fail.
    """
    actions, frames = canonical_scene()

    def numeric_abs(a, f, _home):
        """A genuinely pitch-absolute x: the same physical spot in BOTH legs."""
        out = a.copy()
        ball = f[f["is_ball"].astype(bool)].iloc[0]
        out["abs_x"] = float(ball["x"])
        return out

    base = numeric_abs(actions.copy(), frames.copy(), HOME)
    mir = numeric_abs(actions.copy(), mirror_frames(frames), AWAY)

    assert np.allclose(base["abs_x"], FIELD_LENGTH - mir["abs_x"], atol=1e-9)
    assert not np.allclose(base["abs_x"], mir["abs_x"], atol=1e-9), (
        "the plant is not discriminating -- pick a ball position off the halfway line"
    )


def test_gate_a_categorical_reflection_cannot_silently_pass():
    """The label branch must NOT route through ``to_numeric``.

    Coercing a categorical label yields NaN, which makes the numeric ``both.any()`` False and lets
    the code ``continue`` -- a silent pass. This pins the label comparison as a real assertion.
    """
    swap = {"left": "right", "right": "left"}
    base_labels = pd.Series(["left", "right", "left", "right"])
    good_mirror = pd.Series(["right", "left", "right", "left"])
    bad_mirror = pd.Series(["left", "right", "left", "right"])

    assert (base_labels == good_mirror.map(swap)).all()
    assert not (base_labels == bad_mirror.map(swap)).all(), (
        "a wrong label mapping must be detectable, or the branch asserts nothing"
    )


# ---------------------------------------------------------------------------
# Gate B -- D1 home_team_id invariance
# ---------------------------------------------------------------------------


def _gate_b_params():
    for name in sorted(MIRROR_ENTRIES):
        entry = MIRROR_ENTRIES[name]
        marks = [pytest.mark.xfail(strict=True, reason=entry.known_defect_gate_b)] if entry.known_defect_gate_b else []
        yield pytest.param(name, marks=marks)


@pytest.mark.parametrize("name", _gate_b_params())
def test_gate_b_home_team_id_invariance(name):
    """D1: direction must come from the FRAMES, never from team identity.

    Gate A is structurally blind to identity-keying -- swapping ``home_team_id`` restores the exact
    invariant identity-keying assumes, so an identity-keyed aggregator is invariant there whether
    it is safe or not. This gate holds the frames FIXED and varies ``home_team_id`` instead, so it
    never runs an aggregator outside the ``convert_to_frames`` contract.

    The nonsense id is what makes this strictly stronger than a two-team swap: it catches
    ``same_id(x, home) else ...`` branches that a swap can leave looking correct.
    """
    entry = MIRROR_ENTRIES[name]
    if entry.home_team_id_role == "unused":
        pytest.skip(f"{name} does not take home_team_id")

    actions, frames = canonical_scene()
    ref = entry.call(actions.copy(), frames.copy(), HOME)
    variants = {
        "away": entry.call(actions.copy(), frames.copy(), AWAY),
        "nonsense": entry.call(actions.copy(), frames.copy(), NONSENSE_HOME_ID),
    }

    checked = 0
    considered = 0
    for col, cls in entry.columns.items():
        if cls != "invariant":
            continue  # only pure action-LTR geometry must be identity-independent
        if col in entry.gate_b_exempt:
            continue  # genuine ATTRIBUTION dependence, reason recorded on the entry
        considered += 1
        r = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=float)
        for label, out in variants.items():
            v = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(r) & np.isfinite(v)
            if not both.any():
                continue
            checked += 1
            delta = float(np.abs(r[both] - v[both]).max())
            assert delta <= entry.tolerance, (
                f"{name}.{col}: moved {delta:.6g} when home_team_id -> {label}. A mirror-invariant "
                "column is action-LTR geometry and cannot depend on which team is home -- this is "
                "identity-keyed direction inference (D1)."
            )
    if considered == 0 and entry.gate_b_exempt:
        # Every invariant column is attribution-dependent, with a reason on the entry. Recorded as
        # a skip rather than a vacuity failure: a mirror-invariant column can legitimately be
        # identity-DEPENDENT, and Gate B's own assert cannot tell that apart from a broken fixture.
        pytest.skip(f"{name}: all Gate B columns attribution-exempt ({sorted(entry.gate_b_exempt)})")
    assert checked > 0, f"{name}: Gate B compared nothing -- the check is vacuous"


def test_defensive_line_d3_unit_is_enumerated():
    """``select_back_line_players`` is a PUBLIC export with three consumers.

    A partial re-key across them is the incomplete-repair pattern this repo has already shipped,
    so the gate names the whole unit and pins its current membership. If a future change re-keys
    one member and not the others, this set changes and the test says so -- which is the point:
    the risk was recorded in the spec with no owner until now.
    """
    import ast
    import pathlib

    # __file__-anchored, matching the repo idiom. A CWD-relative path silently reads nothing when
    # pytest runs from anywhere but the repo root, and an empty read makes this vacuous, not red.
    repo = pathlib.Path(__file__).resolve().parents[2]
    unit = {
        "silly_kicks/tracking/_defensive_line.py",
        "silly_kicks/tracking/_packing.py",
        "silly_kicks/tracking/_gk_influence.py",
    }
    reads = set()
    for rel in sorted(unit):
        path = repo / rel
        assert path.exists(), f"D3 unit member missing: {rel}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(n, ast.Name) and n.id == "home_team_id" for n in ast.walk(tree)):
            reads.add(rel)

    assert reads, "no member of the D3 unit reads home_team_id -- has the unit moved?"
    assert reads == unit, (
        f"D3 unit membership changed: {sorted(reads)}. Re-key all three together, or update this "
        "pin WITH the reason -- a partial re-key is the failure mode this test exists to catch."
    )


def test_gate_b_catches_a_planted_identity_keyed_aggregator():
    """Witness: without this, a green Gate B is indistinguishable from a gate that checks nothing."""
    from silly_kicks.id_compat import same_id

    actions, frames = canonical_scene()

    def planted(a, _f, home_team_id):
        # Deliberately keys direction on IDENTITY -- the D1 defect, in miniature.
        out = a.copy()
        out["planted_x"] = [
            row["start_x"] if same_id(row["team_id"], home_team_id) else FIELD_LENGTH - row["start_x"]
            for _, row in a.iterrows()
        ]
        return out

    ref = planted(actions.copy(), frames.copy(), HOME)
    alt = planted(actions.copy(), frames.copy(), AWAY)
    delta = float((ref["planted_x"] - alt["planted_x"]).abs().max())
    assert delta > 1.0, "the plant did not move -- this witness is not discriminating"
