"""RC4: the pining loader must ship ORIENTED SkillCorner frames (ADR-051, spec §3.4 / §11).

`build_skillcorner_frames` forced ``output_convention="absolute_frame"``, which leaves
``team_attacking_direction`` NULL on every row. ``acting_team_attacks_rtl`` then resolves nothing,
returns an all-False flip, and the ENTIRE ADR-028 per-action re-projection layer silently no-ops — so
every away-team action in the research corpus carried mixed-convention geometry while looking healthy.

**Measured on match 1886347 at FULL frame depth, both sides**
(``docs/research/adr028_rc4_orientation/``):

===========  ========  =========
metric       pre-fix   post-fix
===========  ========  =========
unlabelled   1.0000    0.0000
flip         0.0000    0.4728
warnings     1         0
===========  ========  =========

**IDSSE is the CONTROL and is deliberately NOT changed.** `sportec.py` calls
``finalize_orientation`` unconditionally, before its own convention branch, so its frames are already
labelled: measured byte-identical across the fix (unlabelled 0.0000, 718 of 1363 actions flipped =
0.5267791636096845, zero warnings, both runs). A previous cycle changed it anyway on an assumed —
never measured — premise; spec §11.1 records that, and this control is what would have caught it.
The 718 also confirms spec §2.2's independent "718/718" with a second instrument.

Those figures REPLACE a first measurement taken at ``tracking_limit=3000`` that recorded no such cap
(SkillCorner post-fix 0.2398, IDSSE 0.3155). A truncated frame set leaves ``(game_id, period_id,
team_id)`` keys out of the orientation lookup and those actions default to no-flip **silently**, so
both were lower bounds. ``unlabelled_fraction`` was unaffected — a cap cannot make labels appear —
so the defect itself never depended on it.

WHY THIS GUARD IS SHAPED THE WAY IT IS (spec §11.9). The first version matched the *keyword*
``output_convention="absolute_frame"``. That had two holes:

1. `_build_gradientsports` **omits the kwarg entirely**, so a keyword matcher cannot see it at all.
2. Its non-vacuity partner re-implemented a weaker matcher inline, so it passed even when the guard
   was dead.

So this guard resolves the **call-site argument per builder** — a string literal, or the sentinel
``None`` for "omitted" — and pins the whole mapping. It deliberately does NOT resolve the *effective*
convention: it never reads any converter's default, and those defaults differ (skillcorner ``"ltr"``;
gradientsports/sportec resolve ``None`` to ``"absolute_frame"`` via ADR-006). A new builder, or a
changed one, fails here until someone updates the expectation, which forces a measurement rather
than a silent flip. The
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
        "under both conventions (unlabelled 0.0000, 718/1363 flipped = 0.5267791636096845). Changing "
        "it is a no-op that reads as a repair — spec §11.1."
    ),
    "_build_gradientsports": (
        "Omits the kwarg. NOTE the gradientsports converter resolves None to 'absolute_frame' (via "
        "sportec._resolve_output_convention, ADR-006) — NOT 'ltr'; only skillcorner defaults to ltr. "
        "Benign anyway, for the same reason as IDSSE: gradientsports.py:129 calls finalize_orientation "
        "UNCONDITIONALLY before its convention branch at :187, so the frames are labelled either way. "
        "A keyword-matching guard is BLIND to this call, which is why this gate resolves the "
        "call-site ARGUMENT (including its absence) rather than matching a keyword."
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
                if kw.arg is None:
                    # `**opts` splat: `kw.arg` is None and the convention is unknowable from the AST.
                    # Skipping it would leave `found` as None, which this function DEFINES as "the
                    # kwarg is absent" -- the wrong answer, not merely a missing one. Same family as
                    # RC4 itself: a call whose real convention the reader cannot see.
                    found = "<non-literal>"
                    continue
                if kw.arg != "output_convention":
                    continue
                # A STRING constant is the only thing that resolves to a convention. Everything else
                # -- a name, an f-string, or the literal ``None`` -- becomes the sentinel rather than
                # its own value. That matters for `None` specifically: returning it verbatim would be
                # indistinguishable from "the kwarg is absent", which is the very distinction this
                # function exists to draw, and `_build_gradientsports` depends on it.
                node_v = kw.value
                found = (
                    node_v.value
                    if isinstance(node_v, ast.Constant) and isinstance(node_v.value, str)
                    else "<non-literal>"
                )
            if node.name in out and out[node.name] != found:
                # More than one `convert_to_frames` in one function with DIFFERENT conventions: the
                # last in AST order used to win silently. Collapse to the sentinel so the mapping
                # cannot claim a single convention the function does not have.
                out[node.name] = "<non-literal>"
            else:
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
    # Explicit unpack, not negative indexing: `load_matches` yields a 5-tuple whose arity is a
    # private contract, and `loaded[-3]` silently re-points if a field is appended.
    _provider, _match_id, actions, frames, _home_team_id = loaded

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


def test_an_explicit_None_is_NOT_confused_with_an_omitted_kwarg():
    """`output_convention=None` must resolve to the sentinel, never to ``None``.

    Found by pyright (`_ConstantValue` is not assignable to `str | None`), and the type complaint was
    pointing at a real semantic hole rather than a typing nit: ``ast.Constant`` covers the literal
    ``None``, so returning ``kw.value.value`` verbatim made an EXPLICIT None indistinguishable from
    an ABSENT kwarg. Those mean opposite things here -- absent means "the converter default applies",
    which is exactly what `_build_gradientsports` relies on -- so the collision would have made the
    whole mapping unreadable at the one builder it matters for.
    """
    explicit_none = "def _build_x(p):\n    frames = convert_to_frames(b, output_convention=None)\n"
    omitted = "def _build_x(p):\n    frames = convert_to_frames(b)\n"

    assert resolved_conventions(explicit_none) == {"_build_x": "<non-literal>"}
    assert resolved_conventions(omitted) == {"_build_x": None}
    assert resolved_conventions(explicit_none) != resolved_conventions(omitted)


def test_a_kwargs_SPLAT_is_not_mistaken_for_an_omitted_kwarg():
    """`**opts` gives the WRONG answer, not merely an unknown one, unless handled.

    `kw.arg` is `None` for a splat, so a plain "skip anything that isn't output_convention" leaves
    `found` at `None` — which this module DEFINES as "the kwarg is absent, converter default
    applies". A call that might be passing `absolute_frame` would be recorded as taking the default.
    Same family as RC4: a call site whose real convention the reader cannot see.
    """
    splat = "def _build_x(p):\n    frames = convert_to_frames(b, **opts)\n"
    assert resolved_conventions(splat) == {"_build_x": "<non-literal>"}
    # and it must be distinguishable from a genuinely omitted kwarg
    assert resolved_conventions(splat) != resolved_conventions("def _build_x(p):\n    frames = convert_to_frames(b)\n")


def test_TWO_conflicting_calls_in_one_function_collapse_to_the_sentinel():
    """Last-in-AST-order used to win silently, claiming a single convention the function lacks."""
    two = (
        "def _build_x(p):\n"
        '    a = convert_to_frames(b, output_convention="ltr")\n'
        '    c = convert_to_frames(d, output_convention="absolute_frame")\n'
    )
    assert resolved_conventions(two) == {"_build_x": "<non-literal>"}
    # Two calls that AGREE are not ambiguous and must keep the real answer.
    agree = (
        "def _build_x(p):\n"
        '    a = convert_to_frames(b, output_convention="ltr")\n'
        '    c = convert_to_frames(d, output_convention="ltr")\n'
    )
    assert resolved_conventions(agree) == {"_build_x": "ltr"}
