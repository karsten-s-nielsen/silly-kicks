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
    """Pins the count so a silent export change is visible in the diff.

    Dropped 35 -> 34 when the keeper-identity resolver was promoted to the public
    ``silly_kicks.keeper_identity`` module (breaking move, no shim): ``add_defending_gk_player_id``
    left ``tracking.__all__`` with it.
    """
    assert len(_public_add_names()) == 34, sorted(_public_add_names())


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
    actions, frames = (entry.scene or canonical_scene)()
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

    actions, frames = (entry.scene or canonical_scene)()
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


# ---------------------------------------------------------------------------
# Gate C -- goal_map dependence (ADR-055)
# ---------------------------------------------------------------------------


def _flip_map(gm):
    """Same map with both ends swapped -- still COHERENT, so it is a rival hypothesis.

    Swapping both teams is not the same as corrupting the map: the result still says the two
    teams defend opposite ends, so ``attacked_goal`` resolves and the degeneracy guard does not
    fire. An aggregator that reads the map must therefore produce a genuinely different answer,
    and one that ignores it produces exactly the same one.
    """
    from types import MappingProxyType

    from silly_kicks.tracking import GoalMap

    def _swap(pool):
        return MappingProxyType({k: (FIELD_LENGTH if v == 0.0 else 0.0) for k, v in pool.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_gate_c_goal_map_is_the_direction_source(name):
    """D1, one variable further out than Gate B.

    Gate B varied ``home_team_id``. Once direction comes from the map that parameter carries
    nothing, so Gate B goes vacuous -- it SKIPS on ``role="unused"``. This gate holds the FRAMES
    fixed and varies the MAP: the declared columns must MOVE. If they do not, the aggregator is
    not reading the map and the re-key is cosmetic.

    Named columns rather than a bare ``moved > 0``: "something moved" is satisfied by a PARTIAL
    re-key, and ``add_gk_influence`` reads the map down two independent paths.

    What this does NOT prove: that the right ACCESSOR was chosen. ``get`` and ``attacked_goal``
    both move when the map is swapped, so a moved column shows the map is consulted, not that the
    own end and the attacked end were not transposed. That half is ``test_goal_map_consumers.py``.
    """
    from silly_kicks.tracking import resolve_defended_goals

    entry = MIRROR_ENTRIES[name]
    if entry.call_with_map is None:
        pytest.skip(f"{name} does not consume a goal map")

    assert entry.gate_c_must_move, (
        f"{name}: call_with_map is set but gate_c_must_move is empty -- the gate would assert "
        "nothing about which paths read the map"
    )

    actions, frames = (entry.scene or canonical_scene)()
    true_map = resolve_defended_goals(frames)
    assert true_map.n_resolved > 0, "the fixture resolves no ends -- Gate C would be vacuous"
    flipped = _flip_map(true_map)
    assert dict(flipped.resolved) != dict(true_map.resolved), "the flip is a no-op"

    ref = entry.call_with_map(actions.copy(), frames.copy(), true_map)
    alt = entry.call_with_map(actions.copy(), frames.copy(), flipped)

    stayed = []
    for col in entry.gate_c_must_move:
        assert entry.columns.get(col) == "invariant", (
            f"{name}: gate_c_must_move names {col!r}, which is not an invariant column"
        )
        r = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(alt[col], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(r) & np.isfinite(v)
        # A column that goes all-NaN in ONE leg has also responded to the map -- but silently,
        # and a comparison over an empty overlap cannot say so. Treat it as "did not move".
        moved = bool(both.any()) and float(np.abs(r[both] - v[both]).max()) > 1e-12
        if not moved:
            stayed.append(f"{col} (comparable rows: {int(both.sum())})")

    assert not stayed, (
        f"{name}: swapping the goal map did NOT move {stayed}. Either that code path does not "
        "read the map -- a partial re-key, which is the failure this gate exists to catch -- or "
        "the gate is vacuous. Both are failures."
    )


def test_gate_c_catches_an_aggregator_that_ignores_the_map():
    """Witness: without this, a green Gate C is indistinguishable from a gate that checks nothing.

    The plant takes a ``goal_map`` and never reads it -- exactly the cosmetic re-key Gate C is
    built to detect -- so its output must be IDENTICAL across the two maps.
    """
    from silly_kicks.tracking import resolve_defended_goals

    actions, frames = canonical_scene()
    true_map = resolve_defended_goals(frames)

    def planted(a, _f, _goal_map):
        out = a.copy()
        out["planted"] = a["start_x"].to_numpy(dtype=float)  # ignores the map entirely
        return out

    ref = planted(actions.copy(), frames.copy(), true_map)
    alt = planted(actions.copy(), frames.copy(), _flip_map(true_map))
    delta = float((ref["planted"] - alt["planted"]).abs().max())
    assert delta == 0.0, "the plant moved -- this witness is not discriminating"


def test_no_module_infers_direction_from_team_identity():
    """D12: NO module in the direction family computes ``same_id(..., home_team_id)``.

    Renamed from ``test_defensive_line_d3_unit_is_enumerated``, and the rename is load-bearing:
    the old name described a PENDING UNIT to be worked through, and that unit is now empty. A
    name that outlives its predicate is how a gate quietly stops meaning what it says.

    **EMPTY IS THE CORRECT STEADY STATE.** A future reader meeting an empty expectation must not
    "fix" it by repopulating the set -- the whole point of the D3 arc is that this set stays
    empty forever. A non-empty result means identity-keyed direction has been reintroduced.

    THE PREDICATE IS D12: a CALL to ``same_id``/``ids_match`` with ``home_team_id``. Its
    predecessor D9 -- "a same_id result guarding a pitch-constant subtraction or a reversing
    slice" -- was implemented and RUN, and it missed 3 of 8 sites including
    ``_defensive_line.py``'s own, because that site decides direction by sorting from the other
    end (``argsort(xs)`` vs ``argsort(-xs)``) without ever reflecting a coordinate, and its
    ``-xs`` is the unary negation D9 nominated as its EXCLUSION criterion for score sites. A
    module-population pin under D9 would have reported ``_defensive_line.py`` already clean
    before any re-key. Matching the CALL SOURCE instead is recall-complete by construction: no
    downstream shape can evade it, and no future author can invent a seventh way to branch on
    the boolean.

    It is also blind to a bare PARAMETER in a signature, which matters: ``_off_ball_runs.py:98``
    still DECLARES a dead ``home_team_id`` (ADR-042 re-keyed its goalward test onto
    ``acting_team_attacks_rtl``), and ``add_off_ball_runs``'s Gate B green IS the measurement
    that the parameter is unread. A name-mention predicate would go red on it and the obvious
    "fix" would delete that evidence.

    SCOPE vs EXEMPTIONS. The file set below is a list of files to SCAN, NOT a list of
    exemptions, and the difference is why this can assert emptiness honestly: narrowing the scan
    never makes a violation INSIDE it invisible -- a new one in a scanned file is still caught.
    Only an exemption list could wave a real violation through.
    """
    import ast
    import pathlib

    # __file__-anchored, matching the repo idiom. A CWD-relative path silently reads nothing when
    # pytest runs from anywhere but the repo root, and an empty read makes this vacuous, not red.
    repo = pathlib.Path(__file__).resolve().parents[2]
    family = {
        "silly_kicks/tracking/_defensive_line.py",
        "silly_kicks/tracking/_packing.py",
        "silly_kicks/tracking/_off_ball_runs.py",
        "silly_kicks/tracking/_structural_pass.py",
        "silly_kicks/tracking/_line_breaking.py",
        "silly_kicks/tracking/_player_influence.py",
        # Re-keyed by ADR-055; kept in scope so a REINTRODUCTION there is caught too.
        "silly_kicks/tracking/_gk_influence.py",
        "silly_kicks/tracking/_cover_shadows.py",
    }

    def _identity_direction_calls(tree) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name not in {"same_id", "ids_match"}:
                continue
            if any(isinstance(n, ast.Name) and n.id == "home_team_id" for n in ast.walk(node)):
                return True
        return False

    offenders = set()
    for rel in sorted(family):
        path = repo / rel
        assert path.exists(), f"direction-family member missing: {rel} -- update this scan set"
        if _identity_direction_calls(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.add(rel)

    assert offenders == set(), (
        f"identity-keyed direction reintroduced in {sorted(offenders)}. Direction comes from the "
        f"GoalMap (both-team sites) or `acting_team_attacks_rtl` (one-team sites) -- never from "
        f"which team is labelled home. See ADR-051 D3 and ADR-055."
    )


def test_the_D12_predicate_would_CATCH_a_reintroduced_identity_key():
    """Non-vacuity: the assertion above passes trivially if the predicate matches nothing.

    An empty expectation is exactly the shape that can rot into a gate which tests nothing -- a
    typo'd function name, an AST walk over the wrong node type, and it still reports success. So
    the predicate is exercised against source that DOES contain the defect.
    """
    import ast

    planted = ast.parse(
        """
def f(team_id, home_team_id):
    defends_x0 = same_id(team_id, home_team_id)
    return defends_x0
"""
    )
    clean = ast.parse(
        """
def f(team_id, goal_map, game_id, period_id):
    return goal_map.get(game_id, period_id, team_id) == 0.0
"""
    )
    dead_param = ast.parse(
        """
def f(actions, frames, *, home_team_id=None):
    return actions
"""
    )

    def _has(tree):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in {"same_id", "ids_match"} and any(
                isinstance(n, ast.Name) and n.id == "home_team_id" for n in ast.walk(node)
            ):
                return True
        return False

    assert _has(planted), "the predicate must CATCH a reintroduced identity key"
    assert not _has(clean), "the predicate must not fire on a goal-map lookup"
    assert not _has(dead_param), (
        "the predicate must be blind to a bare dead PARAMETER -- _off_ball_runs.py:98 declares "
        "one deliberately, and firing on it would destroy the ADR-042 evidence"
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


# ---------------------------------------------------------------------------
# C0 / D7 -- the FIXTURE's own validity.
#
# A gate is only as good as the rows it scores, and this scene feeds every entry in the
# registry. The pre-C0 version declared `vx=0.8, vy=-0.5, speed=1.0` on players whose
# positions were IDENTICAL in all three frames: two contradictory answers to "how fast is
# this player moving", and which one an aggregator saw depended on whether it read the
# columns or differenced the frames. `packing_secured` was all-<NA> and off-ball runs were
# undetectable, so three gates scored nothing while reporting green.
#
# These assertions are the fixture's contract. They are deliberately about the SCENE, not
# about any aggregator: an aggregator that stops moving is a finding, but a scene that stops
# being a physical scene makes every finding meaningless.
# ---------------------------------------------------------------------------


def test_fixture_velocity_columns_agree_with_observed_displacement():
    """The declared vx/vy ARE the trajectory, not a second contradictory fact about it."""
    _actions, frames = canonical_scene()
    players = frames[~frames["is_ball"].astype(bool)]
    checked = 0
    for _pid, grp in players.groupby("player_id"):
        g = grp.sort_values("time_seconds")
        if len(g) < 2:
            continue
        dt = float(g["time_seconds"].diff().dropna().iloc[0])
        implied_vx = float(g["x"].diff().dropna().iloc[0] / dt)
        implied_vy = float(g["y"].diff().dropna().iloc[0] / dt)
        assert implied_vx == pytest.approx(float(g["vx"].iloc[0]), abs=1e-9), f"player {_pid}: vx"
        assert implied_vy == pytest.approx(float(g["vy"].iloc[0]), abs=1e-9), f"player {_pid}: vy"
        checked += 1
    assert checked >= 20, f"only {checked} players checked -- the fixture shrank, this is now weak"


def test_fixture_has_detectable_off_ball_displacement():
    """Off-ball run detection needs motion. The pre-C0 scene had exactly none."""
    _actions, frames = canonical_scene()
    players = frames[~frames["is_ball"].astype(bool)]
    spans = players.groupby("player_id").agg(dx=("x", lambda s: s.max() - s.min()))
    assert (spans["dx"] > 1.0).sum() >= 10, (
        f"only {(spans['dx'] > 1.0).sum()} players move more than 1 m across the scene"
    )


def test_fixture_yields_a_DECIDED_packing_secured_both_ways():
    """`packing_secured` must be non-NA with BOTH values.

    Two non-NA rows carrying the SAME value would satisfy a "non-vacuous" check while
    proving nothing: the column has three states and a gate that never sees False cannot
    tell a working label from one wired to True.
    """
    from silly_kicks.tracking.features import add_packing

    actions, frames = canonical_scene()
    out = add_packing(actions.copy(), frames.copy())
    secured = out["packing_secured"].dropna()
    assert len(secured) >= 2, f"packing_secured decided on only {len(secured)} row(s)"
    assert set(secured.astype(bool)) == {True, False}, (
        f"packing_secured took only {set(secured.astype(bool))} -- one mechanism is unexercised"
    )


def _observed_map_movers(entry, actions, frames):
    """Invariant columns that ACTUALLY move when the goal map is swapped."""
    from silly_kicks.tracking import resolve_defended_goals

    true_map = resolve_defended_goals(frames)
    ref = entry.call_with_map(actions.copy(), frames.copy(), true_map)
    alt = entry.call_with_map(actions.copy(), frames.copy(), _flip_map(true_map))
    movers = []
    for col, kind in entry.columns.items():
        if kind != "invariant" or col not in ref.columns:
            continue
        r = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(alt[col], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(r) & np.isfinite(v)
        if both.any() and float(np.abs(r[both] - v[both]).max()) > 1e-12:
            movers.append(col)
    return set(movers)


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_gate_c_must_move_is_COMPLETE_not_a_hand_picked_subset(name):
    """Every column that witnesses the map must be DECLARED -- set equality, both directions.

    Gate C asserts the declared columns DO move. That is satisfiable by a hand-picked
    SUBSET, and a subset is how a partial re-key reads as success: declare the columns
    that witness site A, leave site B unwitnessed, ship green. ``_packing.py`` has exactly
    that shape -- ``:145`` (the away mirror) and ``:173`` (which end the defending team's
    back line is taken from) are separate sites, and only ``packing_goal_threat``
    witnesses ``:173``.

    THE INSTRUMENT MATTERS, and the wrong one was used first. Screening columns by
    "constant on the base leg" classifies ``packing_goal_threat`` (constant 0 -- because 0
    is the CORRECT answer here: the bypassed players are not in the back line) identically
    to ``back_n_count`` (constant because n=4 is satisfied at both ends). Measured, they
    need OPPOSITE verdicts: flipping the end moves goal_threat ``0 -> [4, 1, 1, 1]`` while
    back_n_count does not move at all. A detector's liveness is not "does it vary across
    rows" but "does it move when the thing it detects changes", so the screen must be the
    SWAP, never the base leg.

    ADR-056's idiom: derive the population, assert it EXACTLY.
    """
    entry = MIRROR_ENTRIES[name]
    if entry.call_with_map is None:
        pytest.skip(f"{name} does not consume a goal map")

    actions, frames = (entry.scene or canonical_scene)()
    observed = _observed_map_movers(entry, actions, frames)
    declared = set(entry.gate_c_must_move)

    assert observed - declared == set(), (
        f"{name}: UNDECLARED witnesses {sorted(observed - declared)}. These columns move "
        f"when the map is swapped but are not in gate_c_must_move, so the gate would stay "
        f"green if the site they witness were left un-re-keyed. Declare them, or record why "
        f"the column is not a witness."
    )
    assert declared - observed == set(), (
        f"{name}: declared {sorted(declared - observed)} do NOT move under the map swap -- "
        f"either the aggregator stopped reading the map, or the fixture stopped exercising "
        f"that path. Both are findings."
    )


def test_the_completeness_gate_would_CATCH_an_under_declared_list():
    """Non-vacuity: the gate above passes trivially if `observed` is always a subset.

    Plant an entry whose declared list omits a real witness and assert the gate rejects it.
    Without this, a bug making `_observed_map_movers` return nothing would leave every
    entry passing while nothing was checked.
    """
    import dataclasses

    candidates = [e for e in MIRROR_ENTRIES.values() if e.call_with_map is not None]
    assert candidates, "no map-consuming entry to plant against -- this gate is unexercised"
    victim = candidates[0]

    actions, frames = (victim.scene or canonical_scene)()
    observed = _observed_map_movers(victim, actions, frames)
    assert observed, f"{victim.name}: no observed movers, so the plant proves nothing"

    under_declared = dataclasses.replace(victim, gate_c_must_move=())
    assert set(under_declared.gate_c_must_move) != observed, (
        "the planted entry must actually under-declare, or this test is vacuous"
    )
    missed = observed - set(under_declared.gate_c_must_move)
    assert missed, "planting removed nothing -- the gate would not have been exercised"
