"""Deterministic code generator for ``silly_kicks/coordination/_provider_params_generated.py`` (A3).

The generated module is the ONE place the coordination Tier-B numbers live: the interim R4 table at commit 1
(``BASE_SOURCE = "interim"``), replaced by D1's pooled derivation at commit 2 (``BASE_SOURCE = "derivation"``).
``render_generated_params`` is pure and deterministic (sorted keys, ``repr(float(...))`` floats), so the
committed file is exactly ``render_generated_params(None, None)`` and ``test_codegen_empty_reproduces_commit_1_file``
pins it byte-for-byte. ``INTERIM_BASE`` is the single written copy of the R4 numbers, each with its rule.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

GENERATED_PATH = pathlib.Path("silly_kicks/coordination/_provider_params_generated.py")

# The eight COORD_METHOD_FAMILIES (asserted against the vocabulary in tests/coordination/test_config.py).
_FAMILIES = (
    "relative_phase",
    "cross_correlation",
    "vector_coding",
    "coherence",
    "spectral",
    "cluster",
    "team_sync",
    "rsi",
)
# The twelve TEAM_SIGNALS.
_TEAM_SIGNALS = (
    "centroid_x",
    "centroid_y",
    "team_length",
    "team_width",
    "stretch_index",
    "stretch_x",
    "stretch_y",
    "spread",
    "convex_hull_area",
    "defensive_line_x",
    "compactness_x",
    "back_line_high_x",
)
# min_shift_s covers the team signals plus the per-player signals and the cluster amplitude series.
_SHIFT_SIGNALS = (*_TEAM_SIGNALS, "player_x", "player_y", "cluster_amplitude")

#: The R4 interim table -- the ONLY place these numbers are written (A3). Scalars carry their rule inline.
INTERIM_BASE: Mapping[str, Any] = {
    "butterworth_cutoff_hz": 0.4,  # R4: Winter residual-analysis interim cutoff (Hz)
    "max_detection_gap_s": 0.5,  # R4: interim bridge for a detection gap (s)
    "band_low_cpm": 0.22,  # R4: analysis band low edge (cycles/min)
    "band_high_cpm": 0.83,  # R4: analysis band high edge (cycles/min)
    "welch_segment_s": 400.0,  # R4: Welch segment length (s)
    "possession_gap_s": 2.0,  # R4: same-team NA-gap bridge for frames possessions (s)
    "min_observed_fraction": {fam: 0.5 for fam in _FAMILIES},  # R4: per-family detection-coverage floor
    "vc_epsilon": {sig: 0.0 for sig in _TEAM_SIGNALS},  # R4: vector-coding stationarity epsilon (interim 0 = off)
    "min_shift_s": {sig: 60.0 for sig in _SHIFT_SIGNALS},  # R4: surrogate minimum time shift (s)
}

_DOCSTRING = '''"""GENERATED -- coordination Tier-B base + per-provider parameter values (A3 single source).

Commit 1 renders the interim R4 table with ``BASE_SOURCE = "interim"``; Task 19's codegen reproduces this
file for empty inputs, and commit 2 regenerates it with ``BASE_SOURCE = "derivation"`` from D1's pooled
values. This module imports NOTHING from the package (no import cycle) -- the key lists are literal, and
``tests/coordination/test_config.py`` asserts they match the canonical vocabularies.

Do not hand-edit: regenerate via ``scripts/derive_coordination_params.py`` (Task 20)."""'''


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):  # guard: bool is an int subclass
        return repr(value)
    if isinstance(value, float):
        return repr(float(value))
    return repr(value)


def _fmt_value(value: Any, indent: int) -> str:
    """Render a value ruff-canonically: scalars inline; Mappings as a multi-line, sorted, trailing-comma dict
    (the magic trailing comma keeps ruff from collapsing it, so the output is byte-stable under ``ruff format``)."""
    if isinstance(value, Mapping):
        pad, inner = " " * indent, " " * (indent + 4)
        rows = "".join(f'{inner}"{k}": {_fmt_value(value[k], indent + 4)},\n' for k in sorted(value))
        return "{\n" + rows + pad + "}"
    return _fmt_scalar(value)


def _render_params_dict(name: str, annotation: str, params: Mapping[str, Any]) -> str:
    lines = [f"{name}: {annotation} = {{"]
    for key in sorted(params):
        lines.append(f'    "{key}": {_fmt_value(params[key], 4)},')
    lines.append("}")
    return "\n".join(lines)


def render_generated_params(derivation: Mapping[str, Any] | None, calibration: Mapping[str, Any] | None) -> str:
    """Render the generated module text. ``(None, None)`` reproduces the committed commit-1 file.

    ``derivation`` (D1) supplies ``{"pooled": <base params>, "providers": {provider: <params>}}``; when ``None``
    the interim R4 table is the base and there are no per-provider overrides. ``calibration`` (D2) supplies MOVED
    selections applied identically to the pooled base and every provider block.
    """
    base_source, base, providers = generated_maps(derivation, calibration)
    parts = [
        _DOCSTRING,
        "",
        "from __future__ import annotations",
        "",
        f'BASE_SOURCE: str = "{base_source}"',
        "",
        _render_params_dict("BASE_COORDINATION_PARAMS", "dict[str, object]", base),
        "",
        _render_params_dict_providers(providers),
        "",
    ]
    return "\n".join(parts)


def generated_maps(
    derivation: Mapping[str, Any] | None, calibration: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """``(base_source, base, providers)`` -- the values the generated module holds for these inputs.

    Single-sourced: :func:`render_generated_params` writes exactly these, and :func:`params_from_artifacts` builds
    params from exactly these, so a driver computing from the artifacts sees what the committed module will hold.
    """
    base_source = "interim" if derivation is None else "derivation"
    base = dict(INTERIM_BASE if derivation is None else derivation["pooled"])
    providers: dict[str, Any] = {} if derivation is None else dict(derivation["providers"])
    if calibration:
        base = _apply_calibration(base, calibration)
        providers = {p: _apply_calibration(dict(v), calibration) for p, v in providers.items()}
    return base_source, base, providers


def calibration_moves(calibration: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The D2 moves the generated params carry: ``moved_multipliers`` when the confirm gate CLEARED, else none (the
    Tier-B values stand, spec 8.4). Shared by D2's confirm (which renders the module) and D3's artifact handoff."""
    if not calibration or not calibration.get("confirmation", {}).get("gate_cleared"):
        return None
    return calibration.get("moved_multipliers") or None


def params_from_artifacts(provider: str, derivation: Mapping[str, Any] | None, calibration: Mapping[str, Any] | None):
    """``provider``'s :class:`CoordinationParams` from the D1/D2 ARTIFACTS (owner ruling M-5, 2026-10-02: the DGX chain
    hands ``derivation.json`` / ``calibration.json`` forward instead of rewriting the in-package module, so every
    pass runs on the clean commit-1 tree). Identical to ``CoordinationParams.for_provider(provider)`` once commit 2
    commits ``render_generated_params(derivation, calibration_moves(calibration))`` -- the same maps, the same merge.
    """
    import dataclasses

    from silly_kicks.coordination._config import CoordinationParams, _merge_override

    if derivation is None:
        raise SystemExit("the artifact handoff needs D1's derivation.json (--derivation): run D1 --pass reduce first")
    _source, base, providers = generated_maps(derivation, calibration_moves(calibration))
    kwargs = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in base.items()}
    return _merge_override(dataclasses.replace(CoordinationParams(), **kwargs), providers.get(provider, {}))


def _render_params_dict_providers(providers: Mapping[str, Any]) -> str:
    if not providers:
        return "PROVIDER_COORDINATION_PARAMS: dict[str, dict[str, object]] = {}"
    lines = ["PROVIDER_COORDINATION_PARAMS: dict[str, dict[str, object]] = {"]
    for provider in sorted(providers):
        lines.append(f'    "{provider}": {{')
        for key in sorted(providers[provider]):
            lines.append(f'        "{key}": {_fmt_value(providers[provider][key], 8)},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


#: The Tier-B keys whose values are fractions: a move never takes them outside [0, 1].
FRACTION_KEYS = frozenset({"min_observed_fraction"})


def apply_move(key: str, value: Any, *, multiplier: float = 1.0, offset: float = 0.0) -> Any:
    """One D2 move applied to one Tier-B value -- a scalar, or a per-signal / per-family map.

    The ONE implementation D2's sweep levels (``apply_level``) and the codegen / artifact handoff share, so a level
    D2 scored is exactly the value the generated module renders (spec 8.3 M-5; review A-15: D2 clipped a
    ``min_observed_fraction`` level and the codegen did not, rendering 1.1, which ``CoordinationParams`` refuses).
    A fraction key is clipped to [0, 1]."""

    def one(v: Any) -> float:
        moved = float(v) * multiplier + offset
        return min(1.0, max(0.0, moved)) if key in FRACTION_KEYS else moved

    if isinstance(value, Mapping):
        return {k: one(v) for k, v in value.items()}
    return one(value)


def _apply_calibration(params: dict[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Apply D2 MOVED selections (``{key: {"multiplier": m, "offset": o}}``) to a params block."""
    out = dict(params)
    for key, move in calibration.items():
        if key in out:
            out[key] = apply_move(
                key, out[key], multiplier=float(move.get("multiplier", 1.0)), offset=float(move.get("offset", 0.0))
            )
    return out


def write_generated_params(text: str) -> None:
    """Atomically replace the generated module."""
    tmp = GENERATED_PATH.with_suffix(".py.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(GENERATED_PATH)
