"""Coordinate reflection with per-column transform semantics (ADR-045).

A 180-degree point reflection about the pitch centre acts differently on different
KINDS of quantity:

===================  ==========================  ================================
kind                 example columns             transform
===================  ==========================  ================================
``point_x``          ``x``, ``start_x``          ``x -> FIELD_LENGTH - x``
``point_y``          ``y``, ``end_y``            ``y -> FIELD_WIDTH - y``
``vector_x``         ``vx``                      negated
``vector_y``         ``vy``                      negated
``magnitude``        ``speed``                   unchanged
``direction_label``  ``team_attacking_direction``  ``"ltr" <-> "rtl"``
``invariant``        ids, timestamps, ``z``      unchanged
===================  ==========================  ================================

Before this module each reflection site enumerated an x/y column list by hand and
had **no way to express** that a column was a vector. Any column not on the list --
``vx``/``vy`` and ``x_smoothed``/``y_smoothed``, neither of which is in
``TRACKING_FRAMES_COLUMNS`` -- rode through untransformed and silently wrong.

See ADR-045 and NOTICE.
"""

from __future__ import annotations

import re
import types
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "ATOMIC_SPADL_REFLECTION_KINDS",
    "GEOMETRIC_NAME",
    "SPADL_REFLECTION_KINDS",
    "TRACKING_REFLECTION_KINDS",
    "ReflectionKind",
    "UndeclaredGeometricColumnWarning",
    "reflect",
    "reflect_columns",
]


class UndeclaredGeometricColumnWarning(UserWarning):
    """A column with a geometry-shaped name reached a reflection without a declared kind.

    Its own category so a consumer can escalate JUST this to an error (ADR-045 section 4.5):

        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)

    That escalation is how a consumer who fully controls its column universe -- the lakehouse
    does -- gets fail-closed behaviour, without the library imposing it on callers whose
    universe is unbounded by construction (``preserve_native``). Follows the ADR-041
    precedent in ``tracking/_warnings.py``: separate categories so silencing a routine
    notice cannot silence genuine misuse.

    Examples
    --------
    Escalate this category to an error where the caller controls its column universe::

        import warnings
        from silly_kicks.reflection import UndeclaredGeometricColumnWarning
        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)
        # a subsequent reflect(...) with an undeclared geometry-named column now raises
    """


# NOTE: pitch dimensions are resolved from silly_kicks.spadl.config at CALL time, not imported
# at module scope. `from silly_kicks.spadl.config import field_length` would snapshot the value
# at first import, making this module the one reader that cannot see a reassignment -- every
# other site in the repo reads them late-bound by attribute access (spadl/utils.py:1546,
# atomic/spadl/utils.py:1130, atomic/vaep/features.py:166). A silent producer/consumer
# divergence introduced by the module whose purpose is to eliminate silent producer/consumer
# divergence would be a poor start. Resolving inside the call also keeps `reflection` free of
# any package-level dependency, matching the id_compat precedent that justifies its placement.


def _pitch_dims(field_length: float | None, field_width: float | None) -> tuple[float, float]:
    from silly_kicks.spadl import config as spadlconfig

    return (
        spadlconfig.field_length if field_length is None else field_length,
        spadlconfig.field_width if field_width is None else field_width,
    )


ReflectionKind = Literal[
    "point_x",
    "point_y",
    "vector_x",
    "vector_y",
    "magnitude",
    "direction_label",
    "invariant",
]

_DIRECTION_SWAP = {"ltr": "rtl", "rtl": "ltr"}

# Registries are exposed as read-only mappings: a plain dict would let a consumer do
# TRACKING_REFLECTION_KINDS["x"] = "invariant" and silently disable reflection process-wide.
# The Mapping[str, ReflectionKind] parameter type already accepts MappingProxyType.

_TRACKING_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry ---
    "x": "point_x",
    "y": "point_y",
    "z": "invariant",
    # --- preprocess-added (NOT in TRACKING_FRAMES_COLUMNS -- the original blind spot) ---
    "vx": "vector_x",
    "vy": "vector_y",
    "x_smoothed": "point_x",
    "y_smoothed": "point_y",
    # invariant BY DECISION, not oversight: the tag names the preprocess config that
    # produced x_smoothed/y_smoothed, and after a reflection those outputs are mirrored
    # while the tag still claims the original config. D3b establishes that the natural
    # mitigation (re-run smoothing) silently no-ops on the tag check
    # (preprocess/_smoothing.py:100-103), so the staleness is recorded here rather than
    # papered over. Reflecting a config tag would be meaningless; renaming it is out of scope.
    "_preprocessed_with": "invariant",
    # --- magnitudes ---
    "speed": "magnitude",
    # --- labels / identity / provenance ---
    "team_attacking_direction": "direction_label",
    "game_id": "invariant",
    "period_id": "invariant",
    "frame_id": "invariant",
    "time_seconds": "invariant",
    "frame_rate": "invariant",
    "player_id": "invariant",
    "team_id": "invariant",
    "is_ball": "invariant",
    "is_goalkeeper": "invariant",
    "speed_source": "invariant",
    "ball_state": "invariant",
    "confidence": "invariant",
    "visibility": "invariant",
    "source_provider": "invariant",
    "is_goalkeeper_source": "invariant",
}
"""Transform kind per tracking-frame column. Covers TRACKING_FRAMES_COLUMNS plus the
preprocess-added columns. Completeness is CI-gated (tests/test_reflection.py)."""

_SPADL_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry ---
    "start_x": "point_x",
    "start_y": "point_y",
    "end_x": "point_x",
    "end_y": "point_y",
    # ADR-025 restart-coordinate enrichment -- these would ride through unmirrored today.
    "enriched_start_x": "point_x",
    "enriched_start_y": "point_y",
    "enriched_end_x": "point_x",
    "enriched_end_y": "point_y",
    "start_coord_source": "invariant",
    "end_coord_source": "invariant",
    "start_coord_confidence": "invariant",
    "end_coord_confidence": "invariant",
    # --- canonical identity / typing (SPADL_COLUMNS) ---
    "game_id": "invariant",
    "original_event_id": "invariant",
    "action_id": "invariant",
    "period_id": "invariant",
    "time_seconds": "invariant",
    "team_id": "invariant",
    "player_id": "invariant",
    "type_id": "invariant",
    "result_id": "invariant",
    "bodypart_id": "invariant",
    # --- add_names() output. add_names attaches these routinely, and a DECLARED column is
    #     the only one whose kind is guaranteed right, so they belong here. ---
    "type_name": "invariant",
    "result_name": "invariant",
    "bodypart_name": "invariant",
    # --- provider-variant columns, union over the four *_SPADL_COLUMNS dicts ---
    "action_provenance": "invariant",  # kloppy family
    "is_synthetic": "invariant",  # gradientsports (ADR-018 synthesized rows)
    "result_source": "invariant",  # skillcorner (ADR-024 native/inferred/stopgap)
    "tackle_winner_player_id": "invariant",  # sportec (ADR-001 qualifier-derived)
    "tackle_winner_team_id": "invariant",
    "tackle_loser_player_id": "invariant",
    "tackle_loser_team_id": "invariant",
}
"""Transform kind per SPADL action column.

32 columns: the 14 canonical, the 3 ``add_names`` outputs, the 7 provider-variant columns
(union over ``*_SPADL_COLUMNS``), and the 8 ADR-025 enrichment columns. Completeness is a CI
contract (tests/test_reflection.py), NOT a runtime one -- ``preserve_native``
(``spadl/utils.py:1651``) lets a caller attach arbitrarily-named provider fields, so the SPADL
column universe is unbounded by construction and no registry can enumerate it at runtime.
See ADR-045 section 4.5. Verified by union over ``silly_kicks.spadl.schema`` on 2026-07-19.

Every provider-variant column is an identifier or provenance token; none is geometric.
"""

_ATOMIC_SPADL_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry: atomic-SPADL carries a POINT plus a DISPLACEMENT VECTOR ---
    "x": "point_x",
    "y": "point_y",
    "dx": "vector_x",
    "dy": "vector_y",
    # --- identity / typing (ATOMIC_SPADL_COLUMNS + ATOMIC_SPADL_NAME_COLUMNS) ---
    "game_id": "invariant",
    "original_event_id": "invariant",
    "action_id": "invariant",
    "period_id": "invariant",
    "time_seconds": "invariant",
    "team_id": "invariant",
    "player_id": "invariant",
    "type_id": "invariant",
    "bodypart_id": "invariant",
    "type_name": "invariant",
    "bodypart_name": "invariant",
}
"""Transform kind per atomic-SPADL column (15: 13 canonical + 2 name columns).

``dx``/``dy`` are the clearest vector columns in the codebase, and
``atomic/spadl/utils.py:1129-1133`` ALREADY negates them correctly. That site is not being
fixed -- it is being migrated, so the contract lives in ONE place instead of eleven.
"""

# Freeze the three registries: a mutable export would let any consumer do
# TRACKING_REFLECTION_KINDS["x"] = "invariant" and silently disable reflection process-wide.
#
# PRIVATE dict, PUBLIC proxy, one declared type per name. Rebinding a name declared
# `dict[str, ReflectionKind]` to a MappingProxyType does NOT type-check --- measured with the
# repo's pyright: `Type "MappingProxyType[str, K]" is not assignable to declared type
# "dict[str, K]"`. A name has one declared type, so the dict literals above must be the
# PRIVATE `_`-prefixed names and these are the public ones.
TRACKING_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(_TRACKING_REFLECTION_KINDS)
SPADL_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(_SPADL_REFLECTION_KINDS)
ATOMIC_SPADL_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(_ATOMIC_SPADL_REFLECTION_KINDS)

# The one geometric-name pattern every ADR-045 guard shares (see tests/test_reflection.py).
# Published so the conformance guards cannot drift from a private copy.
GEOMETRIC_NAME = re.compile(r"^([vd]?[xy]|[xy]_.*|.*_[xy]|.*_smoothed)$")
"""Fully-anchored, .match()-safe. Covers bare (``x``, ``vx``, ``dx``), prefix
(``x_centered``, ``x_smoothed``) and suffix (``defensive_line_x``, ``enriched_start_x``)
forms.

MEASURED LIMITS -- do NOT restate an earlier "zero misses, zero false positives" claim, which
was false against real repo columns::

    team_shape_centroid_x_attacking            False   (infix axis token)
    defending_centroid_vx                      False   (infix axis token)
    team_shape_defensive_line_height_attacking False   (an x-position with NO axis token)

Tolerable only because this pattern never DECIDES anything (ADR-045 section 4.5): library-owned
columns are covered by the registries, complete by enumeration, and this pattern only reports
on passenger columns and drives the conformance guards. Widening it to catch infix forms trades
false negatives for false positives (``max_x_velocity``), and ADR-043's lesson is that a name
heuristic must not be the enforcement mechanism."""


def _as_mask(mask: pd.Series | np.ndarray, index: pd.Index) -> np.ndarray:
    """Align a boolean mask to ``index``.

    A duplicated index does NOT raise on ``reindex`` (measured, pandas 2.3.3) -- a
    same-shaped duplicate aligns positionally and a SUBSET source silently broadcasts a
    wrong mask. Both are worse than an error, so check explicitly rather than relying on
    reindex to complain.
    """
    if isinstance(mask, pd.Series):
        if not index.is_unique and not mask.index.equals(index):
            raise ValueError(
                "reflect: cannot align a mask to a non-unique index unless the mask carries "
                "exactly that index; reindex would silently broadcast. Pass a positional "
                "ndarray mask, or de-duplicate the index."
            )
        return mask.reindex(index, fill_value=False).to_numpy(dtype=bool)
    return np.asarray(mask, dtype=bool)


def reflect_columns(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    point_x: Sequence[str] = (),
    point_y: Sequence[str] = (),
    vector_x: Sequence[str] = (),
    vector_y: Sequence[str] = (),
    direction_label: Sequence[str] = (),
    field_length: float | None = None,
    field_width: float | None = None,
) -> pd.DataFrame:
    """Point-reflect the masked rows, transforming each column by its stated KIND.

    The explicit sibling of :func:`reflect`, for tables with no declared schema
    (computed feature outputs). The caller states what each column IS; unlisted
    columns are left alone.

    Pure: returns a new frame and never mutates ``df`` (ADR-033).

    Examples
    --------
    Reflect a position and its velocity together::

        import numpy as np, pandas as pd
        from silly_kicks.reflection import reflect_columns
        df = pd.DataFrame({"x": [10.0], "vx": [3.0]})
        out = reflect_columns(df, np.array([True]), point_x=["x"], vector_x=["vx"])
        print(float(out.loc[0, "x"]), float(out.loc[0, "vx"]))
        # 95.0 -3.0
    """
    out = df.copy()
    m = _as_mask(mask, out.index)
    if not m.any():
        return out
    fl, fw = _pitch_dims(field_length, field_width)

    for col in point_x:
        if col in out.columns:
            out.loc[m, col] = fl - out[col].to_numpy(dtype="float64")[m]
    for col in point_y:
        if col in out.columns:
            out.loc[m, col] = fw - out[col].to_numpy(dtype="float64")[m]
    for col in (*vector_x, *vector_y):
        if col in out.columns:
            out.loc[m, col] = -out[col].to_numpy(dtype="float64")[m]
    for col in direction_label:
        if col in out.columns:
            # Swap ltr<->rtl on a full object array, then assign the WHOLE column. A masked
            # `out.loc[m, col] = ...` setitem coerces None->NaN for the masked rows on SOME
            # pandas versions only, turning a flipped null token into NaN while a non-flipped
            # one stays None -- a version-dependent None/NaN mix that breaks ADR-029
            # mirror-invariance (orient(F, flag) == orient(mirror(F), not flag)) on the 3.11
            # CI leg. The identity `get(v, v)` leaves every null token EXACTLY as-is, uniformly
            # for flipped and non-flipped rows, and the full-column object assign never coerces.
            mask_arr = np.asarray(m, dtype=bool)
            vals = out[col].to_numpy(dtype=object).copy()
            vals[mask_arr] = [_DIRECTION_SWAP.get(v, v) for v in vals[mask_arr]]
            out[col] = vals
    return out


def reflect(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    kinds: Mapping[str, ReflectionKind],
    extra_kinds: Mapping[str, ReflectionKind] | None = None,
    on_unknown: Literal["warn", "raise", "ignore"] = "warn",
    field_length: float | None = None,
    field_width: float | None = None,
) -> pd.DataFrame:
    """Point-reflect the masked rows of a schema-bearing table, by declared kind.

    An undeclared column is treated as ``invariant`` -- the correct treatment for a
    caller-attached passenger column -- and WARNS
    (:class:`UndeclaredGeometricColumnWarning`) only if its name is geometry-shaped
    (:data:`GEOMETRIC_NAME`). Supply ``extra_kinds`` to declare one properly.

    Fail-closed lives in the CI registry-completeness meta-assertion, not here
    (ADR-045 section 4.5). Three reasons, and the first is decisive: ``to_spadl_ltr``
    calls this from INSIDE nine converters on a frame already carrying the caller's
    ``preserve_native`` columns, with no reachable ``extra_kinds``, so raising there
    has no remedy. Second, ALL EIGHT catalogued ADR-045 defects were library-owned
    columns that the meta-assertion catches and a runtime raise adds nothing to.
    Third, a per-site policy split would recreate D3 -- two same-named orienters with
    divergent contracts -- one layer up.

    A consumer that fully controls its column universe gets fail-closed with::

        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)

    ``on_unknown="raise"`` is retained as a greppable per-call opt-in (it raises on
    ANY undeclared column, geometric or not); ``"ignore"`` silences the check
    entirely. No silly-kicks code path passes either.

    Pure: returns a new frame and never mutates ``df`` (ADR-033).

    Examples
    --------
    Reflect tracking frames, velocities included::

        import numpy as np, pandas as pd
        from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, reflect
        df = pd.DataFrame({"x": [10.0], "vx": [3.0]})
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
        print(float(out.loc[0, "x"]), float(out.loc[0, "vx"]))
        # 95.0 -3.0
    """
    # extra_kinds is ADD-ONLY: it declares columns the registry does not know. Silently
    # overriding a registry declaration would let a call site redefine semantics locally,
    # which is how divergent conventions (ADR-045 D3) start.
    # `{} if None else dict(...)`, NOT `dict(extra_kinds or {})`: the `or {}` widens the value
    # type to str (the empty dict has no ReflectionKind), so the intermediate `extra` would be
    # `dict[str, str]` and `resolved.update(extra)` below fails pyright. Measured 2026-07-21.
    extra: dict[str, ReflectionKind] = {} if extra_kinds is None else dict(extra_kinds)
    collisions = sorted(set(extra) & set(kinds))
    if collisions:
        raise ValueError(
            f"reflect: extra_kinds may not override registry declarations; collision(s) "
            f"{collisions}. extra_kinds is for columns the registry does not know."
        )
    # `dict(kinds)` + `.update`, NOT `{**kinds, **extra}`. Measured under the repo's pyright
    # (1.1.409, 2026-07-20): the `{**mapping}` spread WIDENS the value type
    # `ReflectionKind` (a Literal) to `str`, so the annotated assignment raises
    # `reportAssignmentType` ("dict[str, str] is not assignable to dict[str, ReflectionKind]").
    # `dict()` of a `Mapping[str, ReflectionKind]` preserves the value type, and `.update` of a
    # dict does not re-widen the declared type -> zero pyright errors, no cast, no ignore.
    # (`extra` is already `dict[str, ReflectionKind]` from the line above; collisions have
    # raised, so update order is irrelevant.)
    resolved: dict[str, ReflectionKind] = dict(kinds)
    resolved.update(extra)

    undeclared = [c for c in df.columns if c not in resolved]
    if undeclared and on_unknown == "raise":
        raise ValueError(
            f"reflect: undeclared column(s) {sorted(undeclared)}. Every column must declare a "
            f"reflection kind so it cannot be silently left untransformed (ADR-045). Add it to "
            f"the registry, or pass extra_kinds={{'<col>': '<kind>'}}."
        )
    if undeclared and on_unknown == "warn":
        # Gate on the NAME, not on mere absence. A passenger column is legitimately
        # undeclared and `invariant` is the right answer for it, so warning on every
        # unknown would be noise on `preserve_native`'s supported output -- and noise is
        # how a real signal gets filtered away. Only a geometry-shaped name is suspicious.
        suspicious = sorted(c for c in undeclared if GEOMETRIC_NAME.match(c))
        if suspicious:
            warnings.warn(
                f"reflect: undeclared column(s) {suspicious} have geometry-shaped names but no "
                f"declared reflection kind, so they were left UNTRANSFORMED. If any is a "
                f"coordinate or a vector this is the ADR-045 defect class. Declare it in the "
                f"registry, or pass extra_kinds={{'<col>': '<kind>'}}. To make this an error: "
                f"warnings.filterwarnings('error', "
                f"category=UndeclaredGeometricColumnWarning).",
                UndeclaredGeometricColumnWarning,
                stacklevel=2,
            )

    buckets: dict[str, list[str]] = {
        "point_x": [],
        "point_y": [],
        "vector_x": [],
        "vector_y": [],
        "direction_label": [],
    }
    for col in df.columns:
        kind = resolved.get(col)
        # `kind is not None` is required: `kind in buckets` does NOT narrow the Optional for
        # pyright, and `buckets[kind]` then errors on "None not assignable to str".
        if kind is not None and kind in buckets:
            buckets[kind].append(col)

    return reflect_columns(
        df,
        mask,
        point_x=buckets["point_x"],
        point_y=buckets["point_y"],
        vector_x=buckets["vector_x"],
        vector_y=buckets["vector_y"],
        direction_label=buckets["direction_label"],
        field_length=field_length,
        field_width=field_width,
    )
