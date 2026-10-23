from __future__ import annotations

import inspect

import silly_kicks.tracking as T

#: Frame-consuming add_* whose signature legitimately does not fit the canonical (actions, frames, ...) shape.
_CALL_SHAPE_EXEMPT = {
    "add_sync_score": "link-consumer family (TF-6): add_sync_score(actions, links, *, ...), no frames",
    "add_visible_area_coverage": "takes no frames and REQUIRES visible_area",
    "add_gradientsports_player_ids": "jersey/roster helper over different inputs, returns frames",
}


def _public_add_names() -> set[str]:
    return {n for n in T.__all__ if n.startswith("add_")}


def _takes_frames(fn) -> bool:
    return "frames" in inspect.signature(fn).parameters


def _frame_consumers() -> dict[str, inspect.Signature]:
    out = {}
    for name in _public_add_names():
        fn = getattr(T, name)
        sig = inspect.signature(fn)
        if "frames" in sig.parameters:
            out[name] = sig
    return out


def test_frames_is_never_keyword_only():
    offenders = {}
    for name, sig in _frame_consumers().items():
        if name in _CALL_SHAPE_EXEMPT:
            continue
        p = sig.parameters["frames"]
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            offenders[name] = str(sig)
    assert not offenders, f"frames must be positional-or-keyword (canonical shape): {offenders}"


def test_optional_params_after_frames_are_keyword_only():
    offenders = {}
    for name, sig in _frame_consumers().items():
        if name in _CALL_SHAPE_EXEMPT:
            continue
        params = list(sig.parameters.values())
        # allow: actions, frames, and a single required positional model (e.g. xt)
        for p in params[2:]:
            positional = p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            optional = p.default is not inspect.Parameter.empty
            if positional and optional:
                offenders.setdefault(name, []).append(p.name)
    assert not offenders, f"optional params after frames must be keyword-only: {offenders}"


def test_exemptions_carry_a_reason():
    for name, reason in _CALL_SHAPE_EXEMPT.items():
        assert reason.strip(), f"{name} exemption needs a reason"


def test_exemptions_are_all_real_exports():
    stale = set(_CALL_SHAPE_EXEMPT) - _public_add_names()
    assert not stale, f"exemption names a non-exported add_*: {sorted(stale)}"


def test_tf46_restart_stages_follow_the_frame_consumer_call_shape():
    """Task 2.14 (ADR-078): the TF-46 Arm 1 Part 2 restart stages are frame consumers but not ``add_*`` (so the
    enumeration above cannot see them). Each takes ``(actions, frames)`` positional-or-keyword and EVERY other
    parameter keyword-only."""
    from silly_kicks.spadl import refine_restart_types
    from silly_kicks.tracking._kick_resolver import resolve_kicks
    from silly_kicks.tracking._penalty_detector import detect_penalties

    for fn in (resolve_kicks, detect_penalties, refine_restart_types):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params[:2]] == ["actions", "frames"], fn.__name__
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:2]), fn.__name__
        rest = [p.name for p in params[2:] if p.kind != inspect.Parameter.KEYWORD_ONLY]
        assert not rest, f"{fn.__name__}: options must be keyword-only, got positional {rest}"
