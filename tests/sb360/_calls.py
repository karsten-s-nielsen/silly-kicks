"""Per-aggregator call adapters.

Every registered entry is invoked as ``call(actions, frames, links, home_team_id)``. Most
aggregators fit a generic adapter that forwards ``links``/``home_team_id`` only where the
signature accepts them. The rest are written OUT BY HAND here.

They are hand-written deliberately. A generic adapter that guesses -- supplying a default for
a required argument, or swallowing a signature mismatch -- turns a wrong call into a recorded
verdict about the library. The measured failure modes it must not paper over:

* six aggregators require a fitted ``ExpectedThreat``;
* ``add_defensive_credit`` also requires an ``xg_column`` that silly-kicks does not ship;
* ``add_pre_shot_gk_angle`` takes ``frames`` KEYWORD-only while its sibling takes it
  positionally;
* ``add_sync_score`` takes ``links`` as its second POSITIONAL argument and no frames at all;
* ``add_gradientsports_player_ids`` is a jersey/roster helper over different inputs entirely,
  and returns a tuple.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd

import silly_kicks.tracking as T


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
    enriched["defending_gk_player_id"] = T.defending_gk_from_frames(enriched, frames)
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
    import numpy as np
    import pandas as pd

    half = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])

    def call(actions, frames, links, home_team_id):
        visible = pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [half] * len(actions)})
        return fn(actions, visible_area=visible, links=links)

    return call
