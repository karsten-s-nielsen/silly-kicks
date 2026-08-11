"""Regenerate ``tests/sb360/_entries/*.py`` from MEASURED observations.

COMMITTED deliberately. The observation lock is designed to FAIL on any fixture change and
force re-recording -- so without this tool the first person to touch the fixture would have to
re-derive 90 verdicts by hand. The rationales survive in the generated source; the RULES that
produced them live in ``_adjudicate.py``, and losing those would mean re-inventing the
judgement rather than re-applying it.

Run from the repo root::

    python tests/sb360/_regenerate.py          # measure + write entries with TODO adjudications
    python tests/sb360/_adjudicate.py          # fill the TODOs by rule
    python -m ruff format tests/sb360/

Observations and applicability classes are transcribed from EXECUTION. Adjudications are
auto-assigned only where the vocabulary admits exactly one choice needing no rationale, and
left as a loud TODO otherwise.

NOTE: this measures under pytest's warning policy (see ``main``). Measuring under a different
policy than the gate re-derives under records an observation the gate never reproduces.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict

sys.path.insert(0, os.getcwd())

import silly_kicks.tracking as T
from tests.sb360 import _calls as C
from tests.sb360 import _fixture as F
from tests.sb360._harness import diagnose_cause, run_axis
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import ADAPTERS, Sb360Entry, feature_columns

AXES = (
    ("velocity", "full"),
    ("visibility", "gk_absent"),
    ("visibility", "defender_absent"),
    ("visibility", "gk_one_end"),
)

FAMILY = {
    "context": [
        "add_action_context",
        "add_actor_pre_window",
        "add_pressure_on_actor",
        "add_elastic_sync",
        "add_sync_score",
        "add_gradientsports_player_ids",
    ],
    "gk": [
        "add_gk_influence",
        "add_gk_completion",
        "add_ghost_gk",
        "add_xt_gk",
        "add_pre_shot_gk_position",
        "add_pre_shot_gk_angle",
        "add_shot_goalmouth",
        "add_xshot_occurrence",
        "add_xcross_attempt",
    ],
    "space": [
        "add_pitch_control",
        "add_obso",
        "add_pausa",
        "add_space_creation",
        "add_cover_shadows",
        "add_player_influence",
        "add_das",
    ],
    "shape": [
        "add_team_shape",
        "add_defensive_line",
        "add_shape_graph",
        "add_line_break",
        "add_structural_pass",
        "add_packing",
    ],
    "offball": [
        "add_off_ball_runs",
        "add_off_ball_context",
        "add_off_ball_run_values",
        "add_press_commitment",
        "add_defensive_credit",
    ],
}

AUTO = {"identical": "works", "all_nan": "honest_nan", "raises_a": "raises"}
NEEDS_HUMAN = {"differs", "partial_nan", "no_signal"}

NL = "\n"


def q(s) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


def main() -> None:
    # MATCH pytest's policy (pyproject.toml:257-261). Measuring under `ignore` while the lock
    # re-derives under `error` records an observation taken in conditions the gate never
    # reproduces -- exactly how add_obso was recorded `differs` and then observed `raises_a`.
    warnings.simplefilter("ignore")
    from silly_kicks.tracking import (
        IgnoredSurfaceInputsWarning,
        MissingFeatureContractWarning,
        SyntheticEPVWarning,
    )

    for cat in (SyntheticEPVWarning, IgnoredSurfaceInputsWarning, MissingFeatureContractWarning):
        warnings.simplefilter("error", cat)
    of_family = {n: f for f, names in FAMILY.items() for n in names}
    out_by_family: dict[str, list[str]] = defaultdict(list)
    todo: list[tuple] = []
    # A probe failure means the aggregator contributes ZERO columns, so every roster block for it
    # regenerates EMPTY -- and because the committed verdicts stay correct until someone
    # regenerates, CI never sees it. That shipped: `visible_area_coverage` was defined in
    # `_calls.py` but never registered in `ADAPTERS`, so the `C.generic` fallback raised TypeError
    # and this handler turned it into "this aggregator emits nothing".
    #
    # The `except` stays BROAD deliberately -- an aggregator that legitimately refuses this fixture
    # must not abort a 35-entry regeneration -- but it is no longer SILENT. Failures are collected
    # and reported at the end, so an unregistered adapter announces itself on the run that would
    # otherwise have quietly emptied its block.
    probe_failures: list[tuple[str, str]] = []

    for name in sorted(n for n in T.__all__ if n.startswith("add_")):
        fn = getattr(T, name)
        adapter = ADAPTERS.get(name, C.generic)
        a, fr, links = F.build_leg_a()
        try:
            probe = adapter(fn)(a, fr, links, F.HOME_TEAM_ID)
            cols = feature_columns([c for c in probe.columns if c not in a.columns])
        except Exception as exc:
            cols = ()
            probe_failures.append((name, f"{type(exc).__name__}: {exc}"))
        entry = Sb360Entry(name=name, call=adapter(fn), columns=tuple(cols))

        axis_obs: dict[tuple[str, str], dict[str, str]] = {}
        for axis, roster in AXES:
            try:
                axis_obs[(axis, roster)] = {
                    c: v.observation for c, v in run_axis(entry, axis=axis, roster=roster).items()
                }
            except Exception as exc:
                axis_obs[(axis, roster)] = {"__ERROR__": f"{type(exc).__name__}: {exc}"}

        appl = {}
        for c in cols:
            try:
                appl[c] = derive_applicability(entry, c)
            except Exception:
                appl[c] = ("no_support", {"extreme": 0.0, "near": 0.0})

        causes = {}
        for c in cols:
            if axis_obs[("velocity", "full")].get(c) in NEEDS_HUMAN:
                try:
                    causes[c] = diagnose_cause(entry, c)
                except Exception as exc:
                    causes[c] = f"UNDIAGNOSED-{type(exc).__name__}"

        def rows(obs, indent, axis, roster, *, cols=cols, causes=causes, name=name):
            body = []
            for c in cols:
                o = obs.get(c, "__MISSING__")
                if o in AUTO:
                    body.append(f"{indent}{q(c)}: AxisVerdict({q(o)}, {q(AUTO[o])}),")
                else:
                    cause = causes.get(c, "")
                    body.append(
                        f'{indent}{q(c)}: AxisVerdict({q(o)}, "TODO", rationale={q("TODO cause=" + str(cause))}),'
                    )
                    todo.append((name, axis, roster, c, o, cause))
            return body

        lines = [
            f"{NL}_entry({NL}    {q(name)},{NL}    ADAPTERS[{q(name)}](T.{name})"
            if name in ADAPTERS
            else f"{NL}_entry({NL}    {q(name)},{NL}    C.generic(T.{name})"
        ]
        lines[0] += ","
        lines.append("    columns=(")
        for c in cols:
            lines.append(f"        {q(c)},")
        lines.append("    ),")
        lines.append("    velocity={")
        lines.extend(rows(axis_obs[("velocity", "full")], "        ", "velocity", "full"))
        lines.append("    },")
        lines.append("    visibility={")
        for axis, roster in AXES[1:]:
            lines.append(f"        {q(roster)}: {{")
            lines.extend(rows(axis_obs[(axis, roster)], "            ", axis, roster))
            lines.append("        },")
        lines.append("    },")
        lines.append("    applicability={")
        for c in cols:
            lines.append(f"        {q(c)}: {q(appl[c][0])},")
        lines.append("    },")
        lines.append("    applicability_deltas={")
        for c in cols:
            lines.append(f'        {q(c)}: {{"extreme": {appl[c][1]["extreme"]!r}, "near": {appl[c][1]["near"]!r}}},')
        lines.append("    },")
        lines.append(")")
        out_by_family[of_family.get(name, "misc")].append(NL.join(lines))

    for fam, blocks in out_by_family.items():
        path = f"tests/sb360/_entries/_{fam}.py"
        header = (
            f'"""SB360 verdicts -- {fam} family.{NL}{NL}'
            f"Observations and applicability classes are TRANSCRIBED FROM EXECUTION; only a{NL}"
            f"human writes an adjudication or a rationale.{NL}"
            f'"""{NL}{NL}from __future__ import annotations{NL}{NL}'
            f"import silly_kicks.tracking as T{NL}{NL}"
            f"from tests.sb360 import _calls as C{NL}"
            f"from tests.sb360._registry import ADAPTERS, AxisVerdict, _entry{NL}"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + NL.join(blocks) + NL)
        print("wrote", path, f"({len(blocks)} entries)")

    if probe_failures:
        print(
            f"{NL}!! PROBE FAILED for {len(probe_failures)} aggregator(s) -- each regenerated with"
            f" ZERO columns, so every roster block for it is now EMPTY.{NL}"
            f"   If that is not expected, the usual cause is an adapter defined in _calls.py but"
            f" never registered in ADAPTERS (the C.generic fallback then raises TypeError)."
        )
        for name, err in probe_failures:
            print(f"    {name}: {err}")

    print(f"{NL}HUMAN ADJUDICATION REQUIRED: {len(todo)}")
    for t in todo:
        print("   ", " | ".join(str(x) for x in t))

    if probe_failures:
        # NON-ZERO EXIT, not just a message. The entries are still written -- you need to see them
        # to diagnose -- but the run is NOT clean, and a printed warning scrolls past while an exit
        # code does not. The `except` above stays broad so one refusing aggregator cannot abort a
        # 35-entry regeneration; this is where that resilience stops being silence.
        sys.exit(
            f"regeneration produced {len(probe_failures)} aggregator(s) with ZERO columns: "
            f"{[name for name, _ in probe_failures]}. Their verdict blocks are now EMPTY. Fix the "
            f"probe failure and re-run before committing -- `test_every_aggregator_emits_at_least_"
            f"one_column` will fail on this registry."
        )


if __name__ == "__main__":
    main()
