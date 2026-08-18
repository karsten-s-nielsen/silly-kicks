"""Fill the generated TODO adjudications with human judgement.

The OBSERVATIONS were measured; these ADJUDICATIONS and rationales are written by hand and
applied consistently by mechanism. Each rule states the reasoning it encodes, so a reviewer
argues with the rule rather than with 90 individual strings.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import textwrap

# --- provenance / source columns -------------------------------------------------------
# Columns whose JOB is to report which path produced the value. Differing between a
# freeze-frame leg and a tracking leg is the CORRECT behaviour -- ADR-043 designed `das_source`
# to do exactly this. Recording these as fabrication would invert their purpose.
PROVENANCE_LIKE = {
    ("add_das", "das_source"),
    # ghost_gk_source reports WHICH path produced the ghost position. On a freeze-frame leg it
    # reads "velocity_unavailable" and on a tracking leg "computed" -- differing is the whole
    # point. Registered here rather than left to FITTED_MODEL below, which would label the
    # provenance column itself a fabrication and invert its purpose. (The POSITIONS are the
    # fitted-model columns, and they now read all_nan -> honest_nan.)
    ("add_ghost_gk", "ghost_gk_source"),
    ("add_press_commitment", "press_commitment_source"),
    ("add_shot_goalmouth", "shot_crossing_source"),
    ("add_shot_goalmouth", "shot_crossing_confidence"),
    ("add_elastic_sync", "elastic_frame_id"),
    ("add_elastic_sync", "elastic_error_seconds"),
}

# --- pitch-control family --------------------------------------------------------------
# Pitch control evaluated at zero velocity is a well-defined POSITIONAL model: weaker than the
# velocity-informed one, but a coherent quantity rather than a fabricated one. The spec names
# this case explicitly as the reason `differs_by_design` exists.
#
# ADR-063 added the four velocity-requiring pitch-control aggregators below to the zero-velocity
# lift; their model-relative (Tier-1) columns now compute the positional model on a declared-
# velocity-less freeze-frame and legitimately DIFFER from the velocity-informed leg -- exactly
# the `positional_pc` case. (Their physical-quantity Tier-2 columns -- reachable area, closing
# time -- are suppressed to NaN and read `all_nan` -> `honest_nan` via AUTO, never reaching here.)
PITCH_CONTROL_DERIVED = {
    "add_obso",
    "add_pitch_control",
    "add_space_creation",
    "add_pausa",
    "add_gk_influence",
    "add_cover_shadows",
    "add_player_influence",
}

# --- fitted models ---------------------------------------------------------------------
# A TRAINED model silently IMPUTING features it was fitted on is receiving out-of-distribution
# input. The output is a plausible number with no basis -- the fabrication this audit exists to
# find. Note the imputation is NOT a zero-fill: `extract_ghost_gk_features` yields NaN, and the
# HGBR reconstruction in `predict_mean` routes NaN down each split's LEARNED missing-value
# direction, which is a different prediction from zero-fill. Measured 2026-08-05.
FITTED_MODEL = {"add_ghost_gk"}

GK_DOMAIN = {
    "add_gk_influence",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_ghost_gk",
}

R = {
    "gk_ablated": (
        "not_exercised",
        "By construction: the gk_absent roster removes the keeper, so this GK feature has "
        "nothing to measure in EITHER leg. Recorded as unexercised because the vocabulary "
        "admits nothing else from no_signal -- but the collapse IS the visibility finding: "
        "the column hard-depends on the keeper being in the freeze-frame, which for SB360 "
        "means in the broadcast camera's view.",
    ),
    "fixture_domain": (
        "not_exercised",
        "The fixture does not produce this column's domain on either leg (no pressing "
        "sequence, shot-occurrence context, or blocking defender to score). A fixture "
        "inadequacy, not a library property -- widening the fixture would move it.",
    ),
    "provenance": (
        "differs_by_design",
        "A provenance column: its job is to report WHICH path produced the value, so "
        "reporting a different path on a freeze-frame leg than on a tracking leg is correct "
        "behaviour. ADR-043 designed das_source to do exactly this.",
    ),
    "window": (
        "differs_by_design",
        "Cause isolated as frame_count, not velocity: the feature needs a temporal window and "
        "a single freeze-frame legitimately yields a different, single-sample answer. Nothing "
        "is fabricated from absent kinematics.",
    ),
    "positional_pc": (
        "differs_by_design",
        "Pitch control evaluated at zero velocity is a well-defined POSITIONAL model -- weaker "
        "than the velocity-informed one, but a coherent quantity rather than an invented one. "
        "A consumer should know the value is positional-only; it is not a fabrication.",
    ),
    # The honest FALLBACK. Reached when the isolation probe could not attribute the change to a
    # single cause and the module is not one whose semantics a specific rule above describes.
    #
    # It exists because the previous fallback was `positional_pc` -- so any module that was not
    # pitch-control-derived and whose cause was not exactly `frame_count` had PITCH CONTROL
    # semantics asserted about it. Measured: `add_elastic_sync.elastic_confidence` was adjudicated
    # with a rationale about "pitch control evaluated at zero velocity", a quantity that module does
    # not compute. A rationale is the part of a verdict a human is asked to ARGUE WITH, so one that
    # describes the wrong mechanism is worse than a vaguer one that is true.
    "mixed_cause": (
        "differs_by_design",
        "Both legs compute this from inputs they actually hold -- nothing is imputed -- but they "
        "differ in BOTH velocity availability and temporal support, so the isolation probe could "
        "not attribute the change to one of them. The value is honest on each leg; what a consumer "
        "must not do is compare a freeze-frame number against a tracking number as though they "
        "were the same measurement.",
    ),
    "ood_model": (
        "silent_degrade",
        "A FITTED model silently IMPUTING the five velocity features it was trained on "
        "(ball_vx/ball_vy/ball_speed/defensive_line_speed/defending_centroid_vx). The extractor "
        "yields NaN, and predict_mean's HGBR reconstruction routes NaN down each split's LEARNED "
        "missing-value direction -- fitted where NaN meant an occasional dropped measurement, "
        "applied where 5 of 26 features are absent on 100% of rows. NOT a zero-fill: measured "
        "NaN -> [6.795, 33.522] vs zero -> [6.888, 33.362]. The output is a plausible coordinate "
        "with no basis, indistinguishable downstream from a velocity-informed prediction. This "
        "is the fabrication the audit exists to find.",
    ),
    "partial_window": (
        "differs_by_design",
        "Cause isolated as frame_count. On a freeze-frame the pre-window contains a single "
        "sample, so the metric is defined for some actions and not others; the NaNs are honest "
        "absences rather than fabricated values.",
    ),
}


def classify(fn: str, col: str, roster: str, obs: str, cause: str) -> tuple[str, str]:
    if obs == "no_signal":
        if roster == "gk_absent" and fn in GK_DOMAIN:
            return R["gk_ablated"]
        return R["fixture_domain"]
    if obs == "partial_nan":
        return R["partial_window"]
    # obs == "differs"
    if (fn, col) in PROVENANCE_LIKE:
        return R["provenance"]
    if fn in FITTED_MODEL:
        return R["ood_model"]
    if cause == "frame_count":
        return R["window"]
    if fn in PITCH_CONTROL_DERIVED:
        return R["positional_pc"]
    # NOT `positional_pc`. Reaching here means the module is not pitch-control-derived, so claiming
    # "pitch control at zero velocity" about it states a mechanism it does not have.
    return R["mixed_cause"]


LINE = re.compile(
    r'^(?P<pad>\s*)"(?P<col>[^"]+)": AxisVerdict\("(?P<obs>[^"]+)", "TODO", '
    r'rationale="TODO cause=(?P<cause>[^"]*)"\),\s*$'
)
ENTRY = re.compile(r"^_entry\($")
NAME = re.compile(r'^\s*"(add_[a-z_]+)",\s*$')

#: A `rationale="..."` line that ruff-format has already put on its own line but cannot shorten.
RATIONALE_LINE = re.compile(r'^(?P<pad>\s*)rationale="(?P<rat>.*)",\s*$')

#: ruff's line limit for this repo.
_MAX_LINE = 120


def _ruff(*args: str) -> None:
    """Run a ruff subcommand over the generated entries. Best-effort: missing ruff is not fatal."""
    # S603: the argv is entirely literal -- `sys.executable` plus subcommands this module's own
    # call sites pass ("check --fix" / "format") and a fixed path. No caller-supplied input
    # reaches it.
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", *args, "tests/sb360/_entries/"],
        check=False,
        capture_output=True,
    )


def _ruff_fix() -> None:
    """Strip lint the generator cannot avoid emitting.

    The generator writes one header for every family, including the ``ADAPTERS`` import -- but a
    family whose entries all use the generic adapter never references it, so ruff removes it as
    unused. Without this step the pipeline's output differs from the committed registry by
    exactly that import, which a round-trip check caught.
    """
    _ruff("check", "--fix")


def _ruff_format() -> None:
    _ruff("format")


def _wrap_long_rationales(path: pathlib.Path) -> int:
    """Split over-long ``rationale=`` strings into implicit concatenation.

    Part of the pipeline, not a manual step: ``ruff format`` cannot break a string LITERAL, so
    the rationales this tool writes exceed the line limit and 90 E501s remain. Leaving this to a
    hand-run command would mean the documented two-script pipeline did not actually reproduce
    the committed registry -- which is exactly what a round-trip check caught.
    """
    lines, changed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        m = RATIONALE_LINE.match(line)
        if m and len(line) > _MAX_LINE:
            pad = m.group("pad")
            inner = pad + "    "
            chunks = textwrap.wrap(m.group("rat"), width=_MAX_LINE - 8 - len(inner))
            lines.append(f"{pad}rationale=(")
            for i, c in enumerate(chunks):
                sep = "" if i == len(chunks) - 1 else " "
                lines.append(f'{inner}"{c}{sep}"')
            lines.append(f"{pad}),")
            changed += 1
        else:
            lines.append(line)
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    total = 0
    for path in sorted(pathlib.Path("tests/sb360/_entries").glob("_*.py")):
        out, fn, roster = [], None, "full"
        for line in path.read_text(encoding="utf-8").splitlines():
            if ENTRY.match(line):
                fn = None
            elif fn is None and (name_match := NAME.match(line)) is not None:
                fn = name_match.group(1)
            if '"gk_absent": {' in line:
                roster = "gk_absent"
            elif '"defender_absent": {' in line:
                roster = "defender_absent"
            elif '"gk_one_end": {' in line:
                # This branch was MISSING when the roster was added, so `roster` simply CARRIED
                # OVER as "defender_absent" through the whole gk_one_end block.
                #
                # Behaviour-neutral today, because `classify` branches only on
                # `roster == "gk_absent"` and neither the stale value nor the correct one equals
                # it. VERIFIED BY EXECUTION, not by reading: a full regenerate + adjudicate with
                # this branch in place re-classified all 127 verdicts and reproduced the entries
                # BYTE-IDENTICALLY. (Re-running `_adjudicate` alone proves nothing here -- it
                # rewrites only lines still marked "TODO", so on an already-adjudicated tree it
                # fills 0 verdicts and never reaches this code at all.)
                #
                # Fixed anyway: the next roster-sensitive rule would have misfired silently, and a
                # parser that tracks two of three rosters is a trap, not a shortcut.
                roster = "gk_one_end"
            elif line.strip() == "velocity={":
                roster = "full"

            m = LINE.match(line)
            if m and fn:
                adj, rat = classify(fn, m.group("col"), roster, m.group("obs"), m.group("cause"))
                cause = m.group("cause") or "n/a"
                rat_full = f"{rat} [measured cause={cause}]"
                out.append(
                    f'{m.group("pad")}"{m.group("col")}": AxisVerdict('
                    f'"{m.group("obs")}", "{adj}", rationale="{rat_full}"),'
                )
                total += 1
            else:
                out.append(line)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print("adjudicated", path)

    # ruff-format FIRST -- it re-lays the AxisVerdict calls and puts `rationale=` on its own
    # line -- THEN wrap, because the wrapper matches that post-format shape. Both steps belong
    # here: leaving the wrap to a hand-run command meant the documented pipeline did not
    # actually reproduce the committed registry, which a round-trip check caught.
    entries = sorted(pathlib.Path("tests/sb360/_entries").glob("_*.py"))
    _ruff_fix()
    _ruff_format()
    wrapped = sum(_wrap_long_rationales(p) for p in entries)
    _ruff_format()
    print(f"\nfilled {total} verdicts, wrapped {wrapped} long rationales")


if __name__ == "__main__":
    main()
