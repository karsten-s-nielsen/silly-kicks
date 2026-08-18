"""Which test fixtures CLAIM velocity, reach a consumer that CARES, and would score differently?

ADR-053/4.76.0 found two fixtures declaring ``speed_source="native"`` with no ``vx``/``vy``, so a
fitted model scored on 5-of-26 IMPUTED features while asserting a geometric property of that
imputed output. The ghost path REFUSED that input at the shared serving seam from 4.76.0; **ADR-063
(4.85.0) extended the same fail-fast to every pitch-control velocity consumer** (gk_influence,
cover_shadows, player_influence, space_creation, obso, pausa, pitch_control), so the silent-
fabrication class is now EMPTY -- every velocity consumer either REFUSES declared-but-absent
velocity or is genuinely velocity-BLIND. A fixture with the defective shape therefore RED-fails at
test-time (it reaches a ``refuses`` consumer) rather than scoring silently, and this instrument's
job is to SURFACE it -- as ``surfaced_refusing`` now, not ``convicted``.

A grep finds ~24 candidates. **That is not a defect count**, and this script exists so nobody
treats it as one: a fixture that never reaches a velocity-sensitive consumer is correct as written,
and "fixing" two dozen files would churn the suite and bury the real cases.

The question decomposes into two measurable halves:

1. **Which consumers are SENSITIVE?** Measured, not assumed: run each velocity-consuming ``add_*``
   on one fixture with and without ``vx``/``vy`` and see whether its output moves. This is
   library-side and fully executable.
2. **Which candidate fixtures call a sensitive consumer?** Greppable per file.

A fixture is CONVICTED only when both hold. Everything else is reported with its reason, because
"we checked and it did not matter" is a finding, not an absence.

**The probe contrast is ABSENT-vs-PRESENT, which is the 4.76.0 defect signature.** An earlier
revision of this script contrasted a stationary frame set (``vx=vy=0``) against a moving one
(``vx=4``), i.e. it measured sensitivity to velocity MAGNITUDE. That is the wrong quantity: the
defect is that velocity is DECLARED and ABSENT, so the extractor yields NaN and a fitted model
routes it down a learned missing-value branch. 4.76.0 measured exactly that -- ``NaN -> [6.795,
33.522]`` vs zero-fill ``-> [6.888, 33.362]`` -- which is a presence effect, not a magnitude one.

Measured, the two contrasts DISAGREE on three consumers. Under magnitude, ``add_ghost_gk``,
``add_press_commitment`` and ``add_xcross_attempt`` all moved by exactly 0.0 and were filed
**velocity-blind**, so any fixture reaching them would have been CLEARED -- including the very
fixtures ADR-053 convicted. ``add_ghost_gk`` scored 0.0 twice over: the sb360 leg-A fixture
declares ``speed_source="unavailable"``, so the by-design marker (an ALL-rows predicate) made it
return NaN on BOTH arms, and an all-NaN column contributes no delta. The instrument was blind to
the one consumer the audit was about.

**The probe must PERTURB, and the arms must differ ONLY in the vector.** Both declare
``speed_source="native"`` -- leaving the leg-A ``"unavailable"`` marker in place suppresses
precisely the consumers under test -- and both pin the same realistic scalar ``speed``, so the
contrast isolates the VECTOR's presence. A second revision let the PRESENT arm overwrite ``speed``
too, and that alone convicted ``add_action_context`` and its atomic mirror: measured, it is
unchanged by ``vx``/``vy`` at fixed speed and reads only the scalar, so the delta was the probe
moving ``speed`` underneath it. The vector is non-zero because this repo already names a
``vx=vy=0`` fixture as a defect rather than a convenience.

**A column flipping between all-NaN and populated is a CHANGE the numeric diff cannot see** --
``NaN - x`` is NaN, so it never enters a max-abs-difference. It is reported as ``nan_flipped``
rather than folded into the delta, so the reason for a verdict stays legible.

Verdicts are THREE-way, not two. A consumer that RAISES on declared-but-absent velocity cannot
silently fabricate, so it is neither ``sensitive`` nor ``blind``; folding it into ``blind`` is what
produced the false clear above. It is reported as ``refuses`` with the exception recorded.

Reads committed test sources and runs library code on a synthetic frame; no corpus pass, no
external data, nothing to misattribute -- so it deliberately takes NO ``require_clean_tree`` guard
(the ``render_sb360_matrix.py`` exemption reasoning).
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_CLAIM = re.compile(r"""speed_source["']?\s*[:=]\s*["'](native|derived)["']""")
_HAS_VX = re.compile(r"""["']vx["']|\bvx\s*=""")

#: A consumer whose output MOVES when declared velocity is actually supplied. A fixture reaching it
#: with the defective shape asserts on imputed values, silently -- the 4.76.0 signature.
SENSITIVE = "sensitive"
#: A consumer that RAISES on declared-but-absent velocity. It cannot fabricate, so a fixture
#: reaching it would be RED rather than quietly wrong -- actionable, but loud.
REFUSES = "refuses"
#: A consumer whose output is unchanged. A fixture reaching it is correct as written.
BLIND = "blind"


def classify_sources(tests_root: pathlib.Path) -> dict:
    """A/B/C counts over the test tree, with the per-file detail behind them."""
    referencing, claiming, no_velocity = [], [], []
    for p in sorted(tests_root.rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "speed_source" not in src:
            continue
        rel = p.relative_to(tests_root.parent).as_posix()
        referencing.append(rel)
        if not _CLAIM.search(src):
            continue
        claiming.append(rel)
        if not _HAS_VX.search(src):
            no_velocity.append(rel)
    return {"referencing": referencing, "claiming": claiming, "claiming_without_vx": no_velocity}


def aggregators_called(path: pathlib.Path) -> list[str]:
    """``add_*`` names this file calls, by AST -- a substring scan would match prose and imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if isinstance(name, str) and name.startswith("add_"):
            names.add(name)
    return sorted(names)


#: The probe vector. 4.0 / 1.5 m/s is a realistic run, inside every consumer's trained domain and
#: large enough to move a TTI-based surface -- unlike the zero vector this repo records as a
#: fixture defect rather than a convenience.
_PROBE_VX, _PROBE_VY = 4.0, 1.5


def _with_speed(frames):
    """Both arms, identically: declare velocity available and pin a realistic scalar ``speed``.

    ``speed`` is held CONSTANT across the two arms so the contrast isolates the VECTOR's presence.
    An earlier revision let the PRESENT arm overwrite ``speed`` as well, which convicted
    ``add_action_context`` -- measured, it is unchanged by vx/vy at fixed speed and reads only the
    scalar, so the delta was the probe moving ``speed``, not the consumer imputing anything.
    """
    import numpy as np
    import pandas as pd

    out = frames.copy()
    out["speed_source"] = "native"
    is_ball = out["is_ball"] if "is_ball" in out.columns else pd.Series(False, index=out.index)
    moving = ~is_ball.fillna(False).astype(bool)
    out["speed"] = np.where(moving, float(np.hypot(_PROBE_VX, _PROBE_VY)), 0.0)
    return out, moving


def _velocity_absent(frames):
    """The DEFECTIVE fixture shape: velocity declared available, the VECTOR not supplied."""
    out, _moving = _with_speed(frames)
    return out.drop(columns=[c for c in ("vx", "vy") if c in out.columns])


def _velocity_present(frames):
    """The REMEDY shape: the same declared speed, with a consistent vector actually supplied."""
    import numpy as np

    out, moving = _with_speed(frames)
    out["vx"] = np.where(moving, _PROBE_VX, 0.0)
    out["vy"] = np.where(moving, _PROBE_VY, 0.0)
    return out


def _difference(baseline, probe, actions) -> tuple[float, list[str]]:
    """``(max_abs_numeric_delta, columns_that_flipped_all_NaN)`` over the ADDED columns."""
    import pandas as pd

    delta = 0.0
    flipped: list[str] = []
    for col in (c for c in baseline.columns if c not in actions.columns):
        if col not in probe.columns:
            flipped.append(col)
            continue
        a, b = baseline[col], probe[col]
        if a.isna().all() != b.isna().all():
            flipped.append(col)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = (a.astype(float) - b.astype(float)).abs()
            if d.notna().any():
                delta = max(delta, float(d.max()))
    return delta, flipped


def velocity_consumers() -> dict[str, dict]:
    """Every tracking ``add_*``, classified ``sensitive`` / ``refuses`` / ``blind``. MEASURED.

    Returns ``name -> {"verdict", "delta", "nan_flipped", "note"}``. The delta is recorded so a
    ``blind`` verdict is auditable rather than asserted: a zero delta from a probe that failed to
    perturb is indistinguishable from a genuinely velocity-blind consumer.
    """
    from silly_kicks import tracking
    from tests.sb360 import _calls, _fixture
    from tests.sb360._registry import ADAPTERS, _init_adapters

    _init_adapters()
    actions, frames, links = _fixture.build_leg_a()
    absent, present = _velocity_absent(frames), _velocity_present(frames)

    out: dict[str, dict] = {}
    for name in sorted(n for n in tracking.__all__ if n.startswith("add_")):
        call = ADAPTERS.get(name, _calls.generic)(getattr(tracking, name))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = call(actions, absent, links, _fixture.HOME_TEAM_ID)
        except Exception as exc:
            out[name] = {
                "verdict": REFUSES,
                "delta": None,
                "nan_flipped": [],
                "note": f"{type(exc).__name__}: {exc}"[:120],
            }
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = call(actions, present, links, _fixture.HOME_TEAM_ID)
        except Exception as exc:
            out[name] = {
                "verdict": REFUSES,
                "delta": None,
                "nan_flipped": [],
                "note": f"raised only WITH velocity -- {type(exc).__name__}: {exc}"[:120],
            }
            continue
        delta, flipped = _difference(a, b, actions)
        changed = delta > 0.0 or bool(flipped)
        out[name] = {
            "verdict": SENSITIVE if changed else BLIND,
            "delta": delta,
            "nan_flipped": flipped,
            "note": f"{len([c for c in a.columns if c not in actions.columns])} column(s)",
        }
    return out


def classify(path: pathlib.Path, consumers: dict[str, dict], *, tests_root: str = "tests") -> dict:
    """Verdict for ONE candidate file, against a measured ``consumers`` map.

    Split out of ``main`` so the positive control exercises the REAL conviction logic rather than a
    re-implementation of it: a control that re-derives the rule it is checking is a tautology.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    claims = bool(_CLAIM.search(src)) and not _HAS_VX.search(src)
    called = aggregators_called(path)
    sensitive = sorted(n for n in called if consumers.get(n, {}).get("verdict") == SENSITIVE)
    refusing = sorted(n for n in called if consumers.get(n, {}).get("verdict") == REFUSES)
    return {
        "file": path.as_posix(),
        "claims": claims,
        "calls": called,
        "reaches_consumer": bool([n for n in called if n in consumers]),
        "sensitive_calls": sensitive,
        "refusing_calls": refusing,
        "value_changed": bool(sensitive),
        "deltas": {n: consumers[n]["delta"] for n in sensitive + refusing if n in consumers},
        "convicted": claims and bool(sensitive),
        # A fixture reaching a refusing consumer would be RED, not silently wrong -- so it is
        # surfaced separately rather than cleared. In a green tree this should be empty; if it is
        # not, the file supplies velocity somewhere this file-level regex cannot see.
        "surfaced_refusing": claims and bool(refusing),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--tests-root", type=pathlib.Path, default=pathlib.Path("tests"))
    args = ap.parse_args()

    counts = classify_sources(args.tests_root)
    consumers = velocity_consumers()

    convicted, refusing, cleared = [], [], []
    for rel in counts["claiming_without_vx"]:
        verdict = classify(pathlib.Path(rel), consumers)
        if verdict["convicted"]:
            convicted.append(verdict)
        elif verdict["surfaced_refusing"]:
            refusing.append(verdict)
        else:
            cleared.append(verdict)

    by_verdict = {
        v: {k: d for k, d in sorted(consumers.items()) if d["verdict"] == v} for v in (SENSITIVE, REFUSES, BLIND)
    }
    report = {
        "counts": {
            "A_referencing_speed_source": len(counts["referencing"]),
            "B_claiming_native_or_derived": len(counts["claiming"]),
            "C_claiming_without_vx": len(counts["claiming_without_vx"]),
            "sensitive_consumers": len(by_verdict[SENSITIVE]),
            "refusing_consumers": len(by_verdict[REFUSES]),
            "convicted": len(convicted),
            "reaches_a_refusing_consumer": len(refusing),
            "cleared": len(cleared),
        },
        "consumers": by_verdict,
        "convicted": convicted,
        "reaches_a_refusing_consumer": refusing,
        "cleared": cleared,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text if not args.out else json.dumps(report["counts"], indent=2))
    print(f"\nCONVICTED ({len(convicted)}): {[c['file'] for c in convicted]}", file=sys.stderr)


if __name__ == "__main__":
    main()
