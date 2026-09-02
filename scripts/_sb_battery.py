"""The SB360 per-aggregator call convention, single-sourced (ADR-053 / round-4 review).

The ADR-053 audit (``tests/sb360``) and the licensed-corpus validation driver
(``scripts/validate_sb360_licensed_corpus.py``) both need to run every registered ``add_*``
aggregator on freeze-frame input with the RIGHT call convention -- which is not uniform: six need a
fitted ``ExpectedThreat``, one needs an ``xg_column`` silly-kicks does not ship, several take
``frames`` keyword-only, one takes ``links`` positionally and no frames, and one is a jersey helper
over different inputs entirely. That per-aggregator adapter layer is exactly what silently
empty-blocked once (``add_visible_area_coverage`` unregistered -> ``generic`` ``TypeError`` swallowed
to ``cols=()``), so a second copy in the driver would fork the already-bitten machinery.

This module is the ONE copy. The adapter bodies + ``ADAPTER_MAP`` were MOVED here from
``tests/sb360/_calls.py`` + ``tests/sb360/_registry._adapters()``; ``tests/sb360/_calls.py`` is now a
re-export shim so the committed ``_entries`` round-trip stays byte-identical, and
``_registry._adapters()`` returns ``ADAPTER_MAP``. Layering is ``tests -> scripts``: nothing here
imports ``tests``, which is the single property that keeps this a clean leaf (pinned by
``tests/scripts/test_sb_battery.py``).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd

import silly_kicks.tracking as tracking
from silly_kicks.tracking._run_features import FAMILY_MODEL_REQUIREMENTS


@functools.cache
def audit_xt():
    """A non-degenerate fitted ``ExpectedThreat`` for the six aggregators that require one.

    Deliberately y-ASYMMETRIC, mirroring ``tests/tracking/_mirror_registry.gate_xt``: a
    y-symmetric grid cannot distinguish a correct point reflection from an x-only mirror.
    Reusing that shape keeps this audit consistent with the gate that already pins orientation.
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    x_ramp = np.linspace(0.02, 0.9, 16)[None, :]
    y_tilt = np.linspace(0.6, 1.4, 12)[:, None]
    xt.xT = x_ramp * y_tilt
    return xt


def generic(fn):
    """Forward ``links``/``home_team_id`` only where the signature accepts them."""
    params = inspect.signature(fn).parameters

    def call(actions, frames, links, home_team_id):
        kwargs = {}
        if "links" in params:
            kwargs["links"] = links
        if "home_team_id" in params:
            kwargs["home_team_id"] = home_team_id
        return fn(actions, frames, **kwargs)

    return call


def with_xt(fn):
    """For ``fn(actions, frames, xt, *, links, home_team_id)``."""
    params = inspect.signature(fn).parameters

    def call(actions, frames, links, home_team_id):
        kwargs = {"links": links}
        if "home_team_id" in params:
            kwargs["home_team_id"] = home_team_id
        return fn(actions, frames, audit_xt(), **kwargs)

    return call


def with_xt_keyword(fn):
    """For ``fn(actions, frames, *, xt=None, links, ...)`` -- xt is KEYWORD-only with a default.

    Passing it is not optional dressing. Left at ``None`` these aggregators fall back to a
    SYNTHETIC EPV surface and emit ``SyntheticEPVWarning``, which CI escalates to an error
    (ADR-041) -- so the audit would record ``raises_a`` for a function that works perfectly well
    when handed the xT a real consumer supplies. Measured: the generator ran under
    ``simplefilter("ignore")`` and saw ``differs``; pytest escalated the warning and saw
    ``raises_a``. A verdict that flips with the warning filter is a verdict about the harness.
    """
    params = inspect.signature(fn).parameters

    def call(actions, frames, links, home_team_id):
        kwargs = {"links": links, "xt": audit_xt()}
        if "home_team_id" in params:
            kwargs["home_team_id"] = home_team_id
        return fn(actions, frames, **kwargs)

    return call


def _with_defending_gk(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Add ``defending_gk_player_id``, a documented prerequisite of both pre-shot GK features.

    Resolved through the library's own ``defending_gk_from_frames`` rather than invented, so
    the audit exercises the real chain a consumer would run. Note it takes no ``links``.
    """
    enriched = actions.copy()
    enriched["defending_gk_player_id"] = tracking.defending_gk_from_frames(enriched, frames)
    return enriched


def pre_shot_gk_angle(fn):
    """``add_pre_shot_gk_angle(actions, *, frames, links)`` -- frames is KEYWORD-only here,
    while its sibling takes it positionally. Both need the ``defending_gk_player_id``
    prerequisite."""

    def call(actions, frames, links, home_team_id):
        return fn(_with_defending_gk(actions, frames), frames=frames, links=links)

    return call


def defensive_credit(fn):
    """``add_defensive_credit`` needs an ``xg_column`` -- silly-kicks ships no xG model.

    The audit supplies a synthetic per-shot xG so the function is EXERCISED rather than
    recorded as unexercised. The column is constant across legs, so it cannot itself cause a
    ``differs``; what varies between legs is only what the function does with the frames.
    """

    def call(actions, frames, links, home_team_id):
        enriched = actions.copy()
        enriched["audit_xg"] = 0.12
        return fn(
            enriched,
            frames,
            xg_column="audit_xg",
            xt=audit_xt(),
            links=links,
        )

    return call


def sync_score(fn):
    """``add_sync_score(actions, links, *, high_quality_threshold)`` -- no frames at all."""

    def call(actions, frames, links, home_team_id):
        return fn(actions, links)

    return call


def pre_shot_gk_position(fn):
    """``add_pre_shot_gk_position(actions, frames, *, links)`` plus the same GK prerequisite."""

    def call(actions, frames, links, home_team_id):
        return fn(_with_defending_gk(actions, frames), frames, links=links)

    return call


def gradientsports_player_ids(fn):
    """A jersey/roster helper over DIFFERENT inputs, returning a tuple.

    It is in ``tracking.__all__`` so the surface gate requires a verdict, but it is not an
    action-coupled aggregator: it maps jersey numbers to player ids. The adapter builds the
    inputs it actually takes and returns the frame half of its tuple, so the audit records a
    real observation rather than an adapter error.
    """

    def call(actions, frames, links, home_team_id):
        jersey = frames[~frames["is_ball"].astype(bool)].copy()
        teams = jersey["team_id"].drop_duplicates().tolist()
        away = next(t for t in teams if str(t) != str(home_team_id))

        # It consumes GS-shaped tracking: `team_side` ("home"/"away") plus `jersey_number`,
        # NOT the SPADL id columns. Jersey numbers are synthesised from player_id.
        jersey["team_side"] = ["home" if str(t) == str(home_team_id) else "away" for t in jersey["team_id"]]
        jersey["jersey_number"] = [str(int(p) % 100) if pd.notna(p) else None for p in jersey["player_id"]]
        # The roster's jersey column is `shirt_number`, NOT `jersey_number` -- the two sides of
        # this join deliberately use different names.
        roster = pd.DataFrame(
            {
                "team_id": [home_team_id] * 11 + [away] * 11,
                "shirt_number": [str(i) for i in range(10, 21)] * 2,
                "player_id": list(range(10, 21)) + list(range(20, 31)),
            }
        )
        out, _report = fn(jersey, roster, home_team_id=home_team_id, away_team_id=away)

        # It returns FRAMES, not actions, so the per-action comparison is structurally
        # inapplicable -- and a raw resolved COUNT would differ trivially between a 132-row
        # Leg A and an 8000-row Leg B. The audited quantity is therefore the resolution RATE,
        # which is leg-size-invariant and measures the property that matters: does
        # jersey -> player_id resolution still work when the input is a freeze-frame?
        rate = float(out["player_id"].notna().mean()) if len(out) else float("nan")
        return actions.assign(gs_jersey_resolution_rate=rate)

    return call


def visible_area_coverage(fn):
    """``add_visible_area_coverage`` takes NO frames and REQUIRES ``visible_area``.

    So it cannot use :func:`generic`, which forwards ``(actions, frames, ...)``. The polygon is
    synthesized here as a fixed half-pitch rather than taken from the fixture, because the SB360
    fixture carries no ``visible_area`` payload -- and a fixed polygon is the honest input for
    this audit: the aggregator reads no frame, so neither the velocity axis nor a roster ablation
    can reach it, and the audit's job is to RECORD that rather than manufacture a difference.
    """
    half = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])

    def call(actions, frames, links, home_team_id):
        visible = pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [half] * len(actions)})
        return fn(actions, visible_area=visible, links=links)

    return call


def with_pressure_methods(fn):
    """``add_pressure_on_actor`` audits BOTH the default ``andrienko_oval`` AND the velocity-derived
    ``bekkers_pi`` -- the latter is opt-in (not in the default ``methods`` tuple), so ``generic``
    would never emit its column. Passing it explicitly makes the ADR-063 honest-NaN tier (spec Part 4)
    audit-observable: the one part of that cycle the two-leg fixture CAN prove."""
    params = inspect.signature(fn).parameters

    def call(actions, frames, links, home_team_id):
        kwargs: dict = {"methods": ("andrienko_oval", "bekkers_pi")}
        if "links" in params:
            kwargs["links"] = links
        return fn(actions, frames, **kwargs)

    return call


#: Model-injection routing is SINGLE-SOURCED from the library
#: (:data:`silly_kicks.tracking._run_features.FAMILY_MODEL_REQUIREMENTS`): the production producer
#: (``run_tracking_features``) and this audit read ONE map of which family needs which injected model,
#: so the two can never disagree about it. Only the audit-synthesized INPUTS (:func:`audit_xt`,
#: ``audit_xg=0.12``, the fixed ``visible_area`` polygon) stay LOCAL -- an audit concern, not the
#: library's. A library requirement code maps to the audit adapter that supplies that model:
_MODEL_ROUTING_ADAPTERS: dict[str, Callable] = {
    "xt_positional": with_xt,  # a fitted ExpectedThreat as the 3rd positional arg
    "xt_keyword": with_xt_keyword,  # xt keyword-only; unset -> SyntheticEPVWarning (CI-escalated)
    "xg": defensive_credit,  # an xg_column silly-kicks does not ship, plus xt
    "link": sync_score,  # add_sync_score(actions, links), no frames
    "visible_area": visible_area_coverage,  # no frames, REQUIRES visible_area
}

#: The ``gk_prereq`` families keep their two distinct call-shape adapters: both supply the same
#: ``defending_gk_player_id`` prerequisite, but they are kept SEPARATE (not unified) so
#: :mod:`tests.sb360._calls` can re-export each by name and the committed ``_entries`` stay
#: byte-identical. The library classifies BOTH as ``gk_prereq``; ``test_sb_battery`` pins that agreement.
_GK_PREREQ_ADAPTERS: dict[str, Callable] = {
    "add_pre_shot_gk_position": pre_shot_gk_position,
    "add_pre_shot_gk_angle": pre_shot_gk_angle,
}

#: Call-shape adapters the library map does NOT classify (they take different inputs / audit a method
#: sweep, not a model injection): the jersey/roster helper and the pressure two-method sweep.
#: ``add_defending_gk_player_id`` was here until the keeper-identity resolver was promoted to the
#: public ``silly_kicks.keeper_identity`` module (breaking move, no shim): it left
#: ``tracking.__all__``, so it is no longer in the SB360 audit's ``registered_add_star_aggregators``
#: surface and needs no call-shape adapter.
_CALL_SHAPE_ADAPTERS: dict[str, Callable] = {
    "add_gradientsports_player_ids": gradientsports_player_ids,
    "add_pressure_on_actor": with_pressure_methods,
}

#: The per-aggregator adapter map, ASSEMBLED from the single-sourced library routing (never a second
#: hand-maintained copy of the groupings). Each library-classified family takes its model-routing
#: adapter or (for ``gk_prereq``) its call-shape adapter; the call-shape specials are added on top.
ADAPTER_MAP: dict[str, Callable] = {
    **{
        name: _MODEL_ROUTING_ADAPTERS[req]
        for name, req in FAMILY_MODEL_REQUIREMENTS.items()
        if req in _MODEL_ROUTING_ADAPTERS
    },
    **_GK_PREREQ_ADAPTERS,
    **_CALL_SHAPE_ADAPTERS,
}


def registered_add_star_aggregators() -> tuple[str, ...]:
    """Every ``add_*`` exported from ``silly_kicks.tracking.__all__``, sorted.

    The SAME predicate as ``tests/sb360/_registry.public_add_star()`` -- single-sourced so the
    audit and the driver can never disagree about which aggregators the battery covers.
    """
    return tuple(sorted(n for n in tracking.__all__ if n.startswith("add_")))


def call_aggregator(name, actions, frames, links, home_team_id):
    """Invoke one registered aggregator with its correct per-aggregator call convention.

    ``ADAPTER_MAP.get(name, generic)(getattr(tracking, name))(actions, frames, links, home_team_id)`` --
    byte-identical to the resolution ``tests/sb360/_regenerate.py`` uses, so the audit and the
    driver run each aggregator EXACTLY the same way.
    """
    return ADAPTER_MAP.get(name, generic)(getattr(tracking, name))(actions, frames, links, home_team_id)


def run_add_star_battery(actions, frames, *, links=None, home_team_id) -> dict[str, pd.DataFrame | str]:
    """Run the whole registered ``add_*`` battery on one match's frames.

    Returns ``{name: added_columns_frame}`` for each aggregator that ran, or ``{name: "raises: ..."}``
    when an aggregator refuses the freeze-frame input -- a real-data raise is a RESULT to record, not
    a crash to abort on (mirroring ``_regenerate.py``'s ``probe_failures``). Threads NO uniform
    ``goal_map``: the per-aggregator adapters resolve whatever each function needs from ``frames``.
    """
    out: dict[str, pd.DataFrame | str] = {}
    for name in registered_add_star_aggregators():
        try:
            result = call_aggregator(name, actions, frames, links, home_team_id)
            added = [c for c in result.columns if c not in actions.columns]
            out[name] = result[added].copy() if added else result.iloc[:, :0].copy()
        except Exception as exc:  # a refusal on real freeze-frames is a recorded result
            out[name] = f"raises: {type(exc).__name__}: {exc}"
    return out
