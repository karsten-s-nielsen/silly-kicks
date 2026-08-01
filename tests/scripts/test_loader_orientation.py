"""RC4: the pining loader must ship ORIENTED SkillCorner frames (ADR-051, spec §3.4 / §11).

`build_skillcorner_frames` forced ``output_convention="absolute_frame"``, which leaves
``team_attacking_direction`` NULL on every row. ``acting_team_attacks_rtl`` then resolves nothing,
returns an all-False flip, and the ENTIRE ADR-028 per-action re-projection layer silently no-ops — so
every away-team action in the research corpus carried mixed-convention geometry while looking healthy.

**Measured on match 1886347, both sides** (``docs/research/adr028_rc4_orientation/``):

===========  ========  =========
metric       pre-fix   post-fix
===========  ========  =========
unlabelled   1.0000    0.0000
flip         0.0000    0.2398
warnings     1         0
===========  ========  =========

**IDSSE is the CONTROL and is deliberately NOT changed.** `sportec.py` calls
``finalize_orientation`` unconditionally, before its own convention branch, so its frames are already
labelled: measured byte-identical across the fix (unlabelled 0.0000, flip 0.31548055759354365, zero
warnings, both runs). A previous cycle changed it anyway on an assumed — never measured — premise;
spec §11.1 records that, and this control is what would have caught it.

WHY THIS GUARD IS SHAPED THE WAY IT IS (spec §11.9). The first version matched the *keyword*
``output_convention="absolute_frame"``. That had two holes:

1. `_build_gradientsports` **omits the kwarg entirely**, so a keyword matcher cannot see it at all.
2. Its non-vacuity partner re-implemented a weaker matcher inline, so it passed even when the guard
   was dead.

So this guard resolves the **effective convention per builder** — including "omitted, therefore the
converter default" — and pins the whole mapping. A new builder, or a changed one, fails here until
someone updates the expectation, which forces a measurement rather than a silent flip. The
non-vacuity test calls :func:`resolved_conventions` itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_LOADER = Path(__file__).resolve().parents[2] / "scripts" / "_loader_pining.py"

#: builder -> the convention its ``convert_to_frames`` call resolves to.
#: ``None`` means the kwarg is omitted, so the converter's own default applies.
#: NOTE the SkillCorner call lives in the FRAME builder ``build_skillcorner_frames`` (:446), not in
#: the match builder ``_build_skillcorner`` (:498) that calls it. The first draft of this file
#: assumed the latter; the guard caught it on its first run — which is the behaviour wanted, since the
#: mapping is then read from source rather than asserted from memory.
EXPECTED_CONVENTIONS = {
    "build_skillcorner_frames": "ltr",
    "_build_idsse": "absolute_frame",
    "_build_gradientsports": None,
}

_REASONS = {
    "build_skillcorner_frames": (
        "RC4: forcing absolute_frame leaves team_attacking_direction NULL on 100% of rows, so the "
        "ADR-028 re-projection silently no-ops (measured pre-fix: unlabelled 1.0000, flip 0.0000)."
    ),
    "_build_idsse": (
        "NOT a defect and NOT to be 'fixed': sportec calls finalize_orientation unconditionally "
        "before its convention branch, so these frames are already labelled. Measured byte-identical "
        "under both conventions (unlabelled 0.0000, flip 0.31548055759354365). Changing it is a "
        "no-op that reads as a repair — spec §11.1."
    ),
    "_build_gradientsports": (
        "Omits the kwarg, so the converter default applies. Benign for the same reason as IDSSE, but "
        "note a keyword-matching guard is BLIND to this call — which is why this gate resolves the "
        "effective convention instead."
    ),
}


def resolved_conventions(source: str) -> dict[str, str | None]:
    """Map each builder function to the ``output_convention`` its ``convert_to_frames`` call resolves to.

    ``None`` = the kwarg is absent (converter default). A builder with no such call is omitted from
    the result entirely, which is itself detectable by the caller.
    """
    tree = ast.parse(source)
    out: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "convert_to_frames":
                continue
            found: str | None = None
            for kw in inner.keywords:
                if kw.arg == "output_convention":
                    found = kw.value.value if isinstance(kw.value, ast.Constant) else "<non-literal>"
            out[node.name] = found
    return out


def test_every_builder_resolves_its_expected_convention():
    """The whole mapping is pinned, so a silent flip on ANY builder fails here."""
    actual = resolved_conventions(_LOADER.read_text(encoding="utf-8"))
    assert actual == EXPECTED_CONVENTIONS, (
        f"loader convention mapping changed.\n  expected: {EXPECTED_CONVENTIONS}\n  actual:   {actual}\n"
        + "\n".join(f"  {k}: {v}" for k, v in _REASONS.items())
        + "\nIf a change here is intended, MEASURE both sides first (spec §11.1) and update "
        "EXPECTED_CONVENTIONS with the numbers."
    )


def test_skillcorner_is_not_absolute_frame():
    """The RC4 assertion on its own, so a failure names the defect rather than a dict diff."""
    actual = resolved_conventions(_LOADER.read_text(encoding="utf-8"))
    assert actual.get("build_skillcorner_frames") != "absolute_frame", _REASONS["build_skillcorner_frames"]


def test_the_guard_detects_the_PRE_FIX_pattern():
    """Non-vacuity — and it calls :func:`resolved_conventions`, not a re-implementation.

    The previous version of this test re-implemented a weaker matcher inline (dropping the
    ``convert_to_frames`` name filter), so it passed even when the guard it claimed to protect was
    dead. Here the real function is handed the real pre-fix shape.
    """
    pre_fix = (
        "def build_skillcorner_frames(paths, match_id, tracking_limit):\n"
        "    frames, report = tracking_sk.convert_to_frames(\n"
        '        bronze, home_team_id=home_team_id, output_convention="absolute_frame"\n'
        "    )\n"
    )
    assert resolved_conventions(pre_fix) == {"build_skillcorner_frames": "absolute_frame"}


def test_the_guard_sees_an_OMITTED_kwarg():
    """The hole a keyword matcher had: `_build_gradientsports` passes no ``output_convention``."""
    omitted = (
        "def _build_gradientsports(paths, tracking_limit=None):\n"
        "    frames, _r = convert_to_frames(resolved, home_team_id=home_team_id)\n"
    )
    assert resolved_conventions(omitted) == {"_build_gradientsports": None}


def test_the_guard_distinguishes_convert_to_frames_from_other_calls():
    """A same-named kwarg on a DIFFERENT call must not be picked up."""
    decoy = (
        "def _build_x(p):\n"
        '    other_function(thing, output_convention="absolute_frame")\n'
        '    frames = convert_to_frames(b, output_convention="ltr")\n'
    )
    assert resolved_conventions(decoy) == {"_build_x": "ltr"}


@pytest.mark.e2e
def test_loaded_skillcorner_frames_are_oriented():
    """Real load: the measurement the AST guard stands in for on an ordinary CI run."""
    import warnings

    from scripts._loader_pining import load_matches
    from silly_kicks.tracking import OrientationUnresolvedWarning
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

    loaded = next(iter(load_matches(providers=["skillcorner"], max_per_provider=1)), None)
    if loaded is None:
        pytest.skip("no skillcorner match available from pining")
    actions, frames = loaded[-3], loaded[-2]

    players = frames[~frames["is_ball"].astype(bool)]
    assert float(players["team_attacking_direction"].isna().mean()) == 0.0
    assert set(players["team_attacking_direction"].dropna().unique()) == {"ltr", "rtl"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flip = acting_team_attacks_rtl(actions, frames)
    assert not [w for w in caught if issubclass(w.category, OrientationUnresolvedWarning)]
    # Non-vacuity: labelled frames with an all-False flip would make every downstream mirror
    # assertion vacuous, which is the pre-fix state wearing a post-fix mask.
    assert bool(flip.any())
