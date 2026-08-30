"""Union of derived columns emitted by every default-config producer (run-and-diff), base-normalised.

Five legs, each running its producers at default config on a real fixture and diffing the columns
they ADD (or, for the ``*_xfns`` / vaep legs whose transformers return a feature-only frame, the
columns they PRODUCE):

1. ``_tracking_add_star_columns`` -- every tracking ``add_*`` aggregator, via the liveness gate's
   ``ENTRIES`` runners (each returns ``(input_df, out)``; added = ``out.columns - input_df.columns``).
2. ``_xfns_columns`` -- every module-level ``*_default_xfns`` list in ``silly_kicks.tracking.features``,
   run as ``(gamestates, frames)`` frame-aware transformers on the liveness scene.
3. ``_spadl_enricher_columns`` -- the public ``add_*`` enrichers in ``silly_kicks.spadl`` (on the
   liveness SPADL scene, with frames for the enrichers that accept them) + their
   ``silly_kicks.atomic.spadl`` mirrors (on a small atomic scene).
4. ``_vaep_columns`` -- ``vaep.base.xfns_default`` + ``vaep.hybrid.hybrid_xfns_default`` on a gamestates
   fixture.
5. ``_restdefense_columns`` -- ``restdefense.compute_rest_defense`` (TF-60) on the restdefense fixture;
   a ``compute_*`` the name-shape discovery misses, so it is run explicitly here.

``emitted_columns`` is the base-normalised union of all five legs (the gamestate-slot marker
``_a{i}`` is stripped so the glossary is keyed on the base/semantic name).

COMPLETENESS CEILING (honest): the coverage gate is only as complete as this harness. The per-leg
non-vacuity anchors (``test_each_leg_is_non_vacuous``) catch a fully-STUBBED leg, but NOT a PARTIAL
one -- a leg that collects some of its real columns and drops the rest passes, and the dropped
columns silently never get required. Catching that would need a second independent enumeration (out
of scope). So every leg MUST genuinely run all its default-config producers and diff, not just enough
to clear the anchor.
"""

from __future__ import annotations

import inspect
import re

# Gamestate-slot marker emitted by lift_to_states (``f"{name}_a{i}"``) and the vaep feature
# functions (``f"{col}_a{i}"``). No base/semantic column name ends in ``_a<digit>``, so the strip
# is loss-free.
_SLOT_SUFFIX = re.compile(r"_a\d+$")


def _base(col: str) -> str:
    """Strip the trailing gamestate-slot marker (``_a0`` / ``_a1`` / ...) from a column name."""
    return _SLOT_SUFFIX.sub("", col)


def _tracking_add_star_columns() -> set[str]:
    """Columns added by every tracking ``add_*`` aggregator (liveness gate ``ENTRIES`` runners)."""
    from tests.tracking.test_aggregator_column_liveness import ENTRIES

    cols: set[str] = set()
    for runner in ENTRIES.values():
        input_df, out = runner()  # type: ignore[operator]
        cols |= set(out.columns) - set(input_df.columns)
    return cols


def _xfns_columns() -> set[str]:
    """Columns produced by every module-level ``*_default_xfns`` list on the liveness scene."""
    from silly_kicks.tracking import features as tracking_features
    from silly_kicks.vaep.feature_framework import is_frame_aware
    from tests.tracking.test_aggregator_column_liveness import _actions, _frames

    actions, frames = _actions(), _frames()
    gamestates = [actions]
    cols: set[str] = set()
    for name in dir(tracking_features):
        if not name.endswith("_default_xfns"):
            continue
        xfns = getattr(tracking_features, name)
        if not isinstance(xfns, (list, tuple)):
            continue
        for fn in xfns:
            out = fn(gamestates, frames) if is_frame_aware(fn) else fn(gamestates)
            cols |= set(out.columns)
    return cols


def _public_add_enrichers(module) -> list:
    """Public ``add_*`` functions defined on ``module`` (deduped by name)."""
    found = {}
    for name in dir(module):
        if not name.startswith("add_"):
            continue
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            found[name] = obj
    return list(found.values())


def _run_enricher(fn, actions, frames) -> set[str]:
    """Run one enricher and return the columns it ADDED (frames + require_gk_role wired by signature)."""
    params = inspect.signature(fn).parameters
    kwargs: dict[str, object] = {}
    if frames is not None and "frames" in params:
        kwargs["frames"] = frames
    if "require_gk_role" in params:
        # Emit gk_role itself rather than requiring a pre-chained add_gk_role.
        kwargs["require_gk_role"] = False
    out = fn(actions, **kwargs)
    return set(out.columns) - set(actions.columns)


def _spadl_enricher_columns() -> set[str]:
    """Columns added by the public spadl + atomic.spadl ``add_*`` enrichers."""
    import silly_kicks.atomic.spadl as atomic_spadl
    import silly_kicks.spadl as spadl
    from tests.atomic._atomic_test_fixtures import (
        _df,
        _make_atomic_gk_action,
        _make_atomic_pass_action,
        _make_atomic_receival,
        _make_atomic_shot_action,
    )
    from tests.tracking.test_aggregator_column_liveness import _actions, _frames

    cols: set[str] = set()

    # Regular SPADL enrichers on the liveness scene (frames threaded to those that accept them).
    actions, frames = _actions(), _frames()
    for fn in _public_add_enrichers(spadl):
        cols |= _run_enricher(fn, actions, frames)

    # Atomic-SPADL mirror enrichers on a small atomic scene (frame-free: the frame-dependent atomic
    # position columns share base names with the regular leg). The atomic scene lacks type/bodypart
    # names + defending_gk_player_id, so add_names / add_pre_shot_gk_context surface those here.
    atomic_actions = _df(
        [
            _make_atomic_gk_action(action_id=0, keeper_action="keeper_save", time_seconds=1.0),
            _make_atomic_pass_action(action_id=1, pass_type="goalkick", player_id=999, time_seconds=2.0),
            _make_atomic_receival(action_id=2, time_seconds=2.5),
            _make_atomic_shot_action(action_id=3, time_seconds=10.0),
        ]
    )
    for fn in _public_add_enrichers(atomic_spadl):
        cols |= _run_enricher(fn, atomic_actions, None)
    return cols


def _vaep_columns() -> set[str]:
    """Columns produced by the default vaep + hybrid-vaep feature lists on a gamestates fixture."""
    from silly_kicks.spadl.utils import add_names
    from silly_kicks.vaep import features as vaep_features
    from silly_kicks.vaep.base import xfns_default
    from silly_kicks.vaep.hybrid import hybrid_xfns_default
    from tests.tracking.test_aggregator_column_liveness import _actions

    # The prev-only hybrid features need >= 1 previous action; use the VAEP default depth (3).
    gamestates = vaep_features.gamestates(add_names(_actions()), 3)
    cols: set[str] = set()
    for fn in list(xfns_default) + list(hybrid_xfns_default):
        cols |= set(fn(gamestates).columns)
    return cols


def _restdefense_columns() -> set[str]:
    """Derived Layer-1 rest-defense columns emitted by compute_rest_defense (TF-60).

    compute_rest_defense is a ``compute_*`` (not an ``add_*``/``*_xfns``), so the name-shape discovery
    misses it; this leg runs it on the restdefense fixture and returns the DERIVED metric columns (the
    sample keys are subtracted by ``_base_schema_and_provenance``; possession_id / is_possession_loss /
    rd_geometry_source are structural/provenance, not features). A NEW emitted metric appears here and
    fails the coverage gate until documented (the run-and-diff anti-rot property)."""
    from silly_kicks.restdefense._compute import compute_rest_defense
    from tests.restdefense._fixtures import make_rest_defense_fixture

    actions, frames = make_rest_defense_fixture()
    samples, _report = compute_rest_defense(actions, frames)
    structural = {"possession_id", "is_possession_loss", "rd_geometry_source"}
    return set(samples.columns) - structural


def _base_schema_and_provenance() -> set[str]:
    """Base schema + linkage-provenance column names -- EXCLUDED per spec Non-goal 1 (not derived features).

    The glossary documents DERIVED columns only. Some liveness runners take a non-actions ``input_df``
    (``add_sync_score`` -> links, ``add_gradientsports_player_ids`` -> jersey frames) and the vaep
    location/movement/time features re-emit schema-named columns, so the raw union leaks the base
    schema. Subtract the canonical schema constants + the name-form of schema + the idempotent
    linkage-provenance set (documented as provenance, not features, in CLAUDE.md).
    """
    from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS
    from silly_kicks.spadl.schema import SPADL_COLUMNS
    from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS

    schema = set(SPADL_COLUMNS) | set(ATOMIC_SPADL_COLUMNS) | set(TRACKING_FRAMES_COLUMNS)
    schema |= {"type_name", "result_name", "bodypart_name"}  # name-form of schema (add_names)
    schema |= {"frame_id", "link_quality_score", "n_candidate_frames", "time_offset_seconds"}  # linkage provenance
    return schema


def emitted_columns() -> set[str]:
    """Base-normalised union of DERIVED columns emitted by every default-config producer."""
    raw = (
        _tracking_add_star_columns()
        | _xfns_columns()
        | _spadl_enricher_columns()
        | _vaep_columns()
        | _restdefense_columns()
    )
    return {_base(c) for c in raw} - _base_schema_and_provenance()
