"""Defensive-line and line-break ``MirrorEntry`` registrations (ADR-028 section 6).

Group: ``defensive_line_and_breaks`` -- ``add_defensive_line``, ``add_line_break``,
``add_off_ball_runs``, ``add_off_ball_context``.

All four are pure back-line / pre-window geometry: no pitch control, no trained model, no EPV
grid, so every Gate A comparison is exact and the tolerance is the float-noise floor rather than
a sized allowance.

Gate A is green for all four at a MEASURED delta of exactly ``0`` -- and that is the expected
reading, not a clean bill of health. Gate A is structurally BLIND to identity-keyed direction:
it swaps ``home_team_id`` along with the mirror, which restores the very invariant such code
assumes.

**HISTORICAL, resolved in 4.80.0 (ADR-051 D3).** Three of these entries used to derive attacking
direction from ``same_id(team, home_team_id)`` and carried a ``defect_b`` marker for it. They no
longer do: ``compute_defensive_line`` takes a ``GoalMap``, and ``_line_break_kernel`` /
``_off_ball_runs`` take a bool resolved from ``acting_team_attacks_rtl``. The markers were
deleted WITH the fix -- they were strict xfails, so an XPASS fails the build and they could not
have survived it.

Gate B is consequently retired for those entries (``home_team_id`` is gone, so it skips on
``role="unused"``) and **Gate C replaces its detection**: hold the frames fixed, swap the MAP,
require the declared columns to move. ``add_off_ball_runs`` is the exception and keeps its Gate B
entry deliberately -- its ``home_team_id`` reaches ``_off_ball_runs_kernel``'s signature and is
never read (ADR-042 re-keyed the goalward test), so that entry's GREEN is the measurement that
the parameter is dead. Declaring it ``"unused"`` would make Gate B skip and throw that evidence
away.
"""

from __future__ import annotations

#: The four linkage columns ``add_defensive_line`` merges. Identifiers and linkage bookkeeping,
#: not geometry -- they are exempt in every group that emits them.
_PROVENANCE = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
_PROVENANCE_REASON = "linkage provenance, not geometry"

_D3 = "D3 re-key pending: identity-keyed direction (spec 4.3)"

#: Every one of these is exact arithmetic on frame coordinates -- a mean, a min/max, a span and
#: two counts -- so the two legs agree bit-for-bit. Measured on ``canonical_scene()``: max
#: base-vs-mirror delta ``0`` (exactly) on all six defensive-line columns, both line-break
#: columns and all four off-ball-run columns. 1e-9 is the float-noise floor, not a sized
#: allowance; if a future change needs more than this, the change is doing something other than
#: reflecting coordinates.
_EXACT_BASIS = (
    "pure geometry; exact under a point reflection -- no pitch control, no trained model, no EPV "
    "grid. Measured base-vs-mirror delta on canonical_scene(): exactly 0 for every column here."
)


def register() -> None:
    from silly_kicks.tracking.features import (
        add_defensive_line,
        add_line_break,
        add_off_ball_context,
        add_off_ball_runs,
    )
    from tests.tracking._mirror_registry import _entry

    # ------------------------------------------------------------------
    # add_defensive_line -- 6 back-line columns + 4 provenance columns
    # ------------------------------------------------------------------
    # Gate B measured on canonical_scene(): defensive_line_x moves 23.75 m, back_line_high_x
    # 11.0, max_lateral_gap 6.0, compactness_x 3.0, lateral_width 4.0 when home_team_id -> AWAY.
    # back_n_count does NOT move (n=4 is satisfied at both ends), which is why it is a poor
    # non-vacuity anchor and a poorer defect witness -- the moving columns carry the signal.
    _entry(
        "add_defensive_line",
        lambda a, f, h: add_defensive_line(a, f),
        {
            "defensive_line_x": "invariant",
            "back_line_high_x": "invariant",
            "compactness_x": "invariant",
            "lateral_width": "invariant",
            "max_lateral_gap": "invariant",
            "back_n_count": "invariant",
            **dict.fromkeys(_PROVENANCE, "exempt"),
        },
        tol=1e-9,
        basis=_EXACT_BASIS,
        role="direction_only",
        non_vacuity=("defensive_line_x", "back_line_high_x", "lateral_width"),
        exempt=dict.fromkeys(_PROVENANCE, _PROVENANCE_REASON),
        # Gate B's variable is GONE (the D3 re-key removed home_team_id), so Gate B now
        # SKIPS on role="unused" and cannot witness the fix -- only the defect it used to
        # catch. Gate C is the same question one variable further out: hold the frames FIXED,
        # swap the MAP, require these columns to move. Sets are MEASURED, not guessed, and the
        # completeness gate asserts set EQUALITY both ways -- an undeclared witness would let a
        # partial re-key ship green.
        call_with_map=lambda a, f, gm: add_defensive_line(a, f, goal_map=gm),
        gate_c_must_move=(
            "defensive_line_x",
            "back_line_high_x",
            "compactness_x",
            "lateral_width",
            "max_lateral_gap",
        ),
        # `back_n_count` is deliberately absent: n=4 is satisfied at BOTH ends, so it cannot
        # move under the swap. Absent because measured dead, not because unexamined.
    )

    # ------------------------------------------------------------------
    # add_line_break -- threshold method (the signature default)
    # ------------------------------------------------------------------
    # method="ward" emits a DISJOINT column set (line_break__ward / lines_broken__ward /
    # line_breaking_type__ward) and is NOT covered here: MIRROR_ENTRIES is keyed by aggregator
    # name and the meta-assertions pin those keys to tracking.__all__ in both directions, so a
    # second "add_line_break[ward]" key would register as stale. The default path is what the
    # public surface serves unqualified; the ward path needs its own gate if it is to be gated.
    #
    # Gate B measured: n_attackers_behind_line 0 -> 8/9 at home_team_id=AWAY (delta 9) and 0 ->
    # 10/1 at the nonsense id (delta 10); line_break itself flips only under the nonsense id
    # (delta 1) -- exactly the `same_id(x, home) else ...` branch a two-team swap leaves looking
    # correct, which is why the nonsense leg is the strictly stronger one.
    _entry(
        "add_line_break",
        lambda a, f, h: add_line_break(a, f),
        {
            "line_break": "invariant",
            "n_attackers_behind_line": "invariant",
        },
        tol=1e-9,
        basis=_EXACT_BASIS,
        role="direction_only",
        non_vacuity=("line_break", "n_attackers_behind_line"),
        # Gate B's variable is GONE (the D3 re-key removed home_team_id), so Gate B now
        # SKIPS on role="unused" and cannot witness the fix -- only the defect it used to
        # catch. Gate C is the same question one variable further out: hold the frames FIXED,
        # swap the MAP, require these columns to move. Sets are MEASURED, not guessed, and the
        # completeness gate asserts set EQUALITY both ways -- an undeclared witness would let a
        # partial re-key ship green.
        call_with_map=lambda a, f, gm: add_line_break(a, f, goal_map=gm),
        # `line_break` DOES move here, which the pre-re-key proxy could not show: it fails
        # through BRANCH CANCELLATION under a two-team home_team_id swap, not through fixture
        # degeneracy, so that proxy never transferred to a map swap. Measured on the re-keyed
        # code, both columns move.
        gate_c_must_move=("line_break", "n_attackers_behind_line"),
    )

    # ------------------------------------------------------------------
    # add_off_ball_runs -- TF-4 pre-window runs
    # ------------------------------------------------------------------
    # role="direction_only" is a DELIBERATE choice, and the honest reading is subtler than the
    # label: `home_team_id` reaches `_off_ball_runs_kernel`'s signature (:98) and is never read
    # in its body -- ADR-042 re-keyed the goalward test onto `acting_team_attacks_rtl` (:176,
    # :212). Declaring "unused" would make Gate B SKIP this entry, which would throw away the
    # one piece of evidence that matters: that the parameter is dead. D3 is specified as
    # "nothing READS home_team_id for direction", and a skipped gate proves nothing about
    # reading. So it stays declared as the direction parameter it historically was, Gate B runs,
    # and its green is the measurement. ("unused" is reserved for a signature that has no
    # `home_team_id` at all.)
    #
    # LIMITATION, measured -- this entry's green is WEAK. canonical_scene() holds player
    # positions constant across its three frames by design, so no displacement exists and no
    # off-ball run can be detected: n_off_ball_runners_pre_window and
    # n_off_ball_runners_toward_goal_pre_window are 0 on every row, and
    # max_off_ball_run_displacement_pre_window / mean_off_ball_run_speed_pre_window are all-NaN
    # (Gate A skips them via `both.any()`). The two count columns are non-NULL so the
    # non-vacuity guard is satisfied, but 0-vs-0 has no discriminating power. Probed
    # `min_displacement_m=0.0` as a rescue: counts come alive (9 per action) but displacement
    # and speed are then exactly 0.0 and toward_goal stays 0, because `dx > 0` is False at zero
    # displacement -- so it buys column liveness and no discriminating power, and the default
    # (which is what production runs) is kept instead. Making this entry sharp needs a fixture
    # with real motion, not a parameter tweak.
    _entry(
        "add_off_ball_runs",
        lambda a, f, h: add_off_ball_runs(a, f, home_team_id=h),
        {
            "n_off_ball_runners_pre_window": "invariant",
            "max_off_ball_run_displacement_pre_window": "invariant",
            "mean_off_ball_run_speed_pre_window": "invariant",
            "n_off_ball_runners_toward_goal_pre_window": "invariant",
        },
        tol=1e-9,
        basis=_EXACT_BASIS,
        role="direction_only",
        non_vacuity=("n_off_ball_runners_pre_window", "n_off_ball_runners_toward_goal_pre_window"),
    )

    # ------------------------------------------------------------------
    # add_off_ball_context -- the TF-4 umbrella (runs + threshold line-break)
    # ------------------------------------------------------------------
    # Emits the union of the two above and inherits `_line_break_kernel`'s identity-keyed
    # direction, so it is a D3 target while `add_off_ball_runs` alone is not. Gate B measured
    # the same n_attackers_behind_line deltas (9 at AWAY, 10 at the nonsense id) -- the
    # off-ball-run half contributes nothing to the failure, which is the correct attribution.
    _entry(
        "add_off_ball_context",
        lambda a, f, h: add_off_ball_context(a, f),
        {
            "n_off_ball_runners_pre_window": "invariant",
            "max_off_ball_run_displacement_pre_window": "invariant",
            "mean_off_ball_run_speed_pre_window": "invariant",
            "n_off_ball_runners_toward_goal_pre_window": "invariant",
            "line_break": "invariant",
            "n_attackers_behind_line": "invariant",
        },
        tol=1e-9,
        basis=_EXACT_BASIS,
        role="direction_only",
        non_vacuity=("line_break", "n_attackers_behind_line", "n_off_ball_runners_pre_window"),
        # Gate B's variable is GONE (the D3 re-key removed home_team_id), so Gate B now
        # SKIPS on role="unused" and cannot witness the fix -- only the defect it used to
        # catch. Gate C is the same question one variable further out: hold the frames FIXED,
        # swap the MAP, require these columns to move. Sets are MEASURED, not guessed, and the
        # completeness gate asserts set EQUALITY both ways -- an undeclared witness would let a
        # partial re-key ship green.
        call_with_map=lambda a, f, gm: add_off_ball_context(a, f, goal_map=gm),
        # The four off-ball columns do NOT move: they come from `_off_ball_runs_kernel`, whose
        # direction was re-keyed onto `acting_team_attacks_rtl` by ADR-042 and never reads the
        # map. Only the line-break half is map-dependent.
        gate_c_must_move=("line_break", "n_attackers_behind_line"),
    )
