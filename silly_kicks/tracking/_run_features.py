"""Provider-agnostic tracking-feature producer (the SB360 first-class-provider cycle, Component 4).

:func:`run_tracking_features` is the canonical way to run the whole frame-consuming ``add_*`` family
and return the ENRICHED ACTIONS (action-grain feature columns -- the family is action-coupled, this is
not a tracking-frame producer) plus a structured report. It:

1. resolves keeper identity ONCE (:func:`resolve_keeper_identities`, ADR-055) and bridges it onto BOTH
   grains -- stamping ``defending_gk_player_id`` on the actions and, on the SB360 roster path, the R1
   identity->frame bridge onto the frames' ``is_goalkeeper`` rows so ``add_pre_shot_gk_*`` can locate
   the keeper the anonymous freeze-frame does not name;
2. pre-links ONCE (:func:`link_actions_to_frames`) and shares one :class:`PitchControlCache`, threading
   both into every family that accepts them (the lakehouse performance pattern, now library-owned);
3. injects the caller-supplied models per family via :data:`FAMILY_MODEL_REQUIREMENTS` -- a fitted
   ``ExpectedThreat`` (``xt``) or an ``xg_column``; a family whose required model is ABSENT is SKIPPED
   with an honest reason and its columns are simply not added (never fabricated -- ADR-009/054/063);
4. runs each family under a per-family guard so a refusal on the given frames is RECORDED, not crashed
   (the ``run_add_star_battery`` precedent); and
5. returns the enriched actions plus a conserving :class:`TrackingFeaturesReport`
   (``n_families_run + n_families_skipped == n_families_in``, the ADR-052 idiom).

The library consumes INJECTED artifacts (a fitted ``ExpectedThreat``, an ``xg_column``, a
``{team_id: gk_id}`` roster) -- it never fetches or fits them; ``providers/statsbomb`` stays
pure-shaping (ADR-054). Velocity-constitutive families stay honest-NaN on velocity-less SB360 frames
whether they self-degrade (``add_das`` -> ``das_source="unscoreable_frame"``) or are skipped: naming
the keeper does not make a velocity metric score (ADR-063).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Literal

import numpy as np
import pandas as pd

import silly_kicks.tracking as tracking
from silly_kicks.keeper_identity import (
    KeeperIdentityReport,
    add_defending_gk_player_id,
    apply_keeper_identities_to_frames,
    resolve_keeper_identities,
)
from silly_kicks.spadl.schema import is_time_unmeasured

from .pitch_control import PitchControlCache
from .utils import link_actions_to_frames

#: Per-family model-injection routing, SINGLE-SOURCED so the audit (``scripts/_sb_battery.py``) and the
#: producer can never disagree about which family needs which injected model. The value is a
#: requirement code:
#:
#: - ``"xt_positional"`` -- a fitted ``ExpectedThreat`` as the 3rd POSITIONAL argument.
#: - ``"xt_keyword"`` -- a fitted ``ExpectedThreat`` as a KEYWORD ``xt=`` (a ``None`` default takes the
#:   synthetic-EPV path and emits ``SyntheticEPVWarning``, so it is passed explicitly).
#: - ``"xg"`` -- an injected ``xg_column`` (silly-kicks ships no xG model) plus a fitted ``ExpectedThreat``.
#: - ``"gk_prereq"`` -- needs the ``defending_gk_player_id`` action stamp (supplied by the keeper bridge).
#: - ``"link"`` -- the link-consumer family (TF-6): ``add_sync_score(actions, links)``, no frames.
#: - ``"visible_area"`` -- needs an injected ``visible_area`` polygon table; takes no frames.
#:
#: A family absent from this map is ``generic`` (``frames`` plus whatever optional kwargs its signature
#: accepts). ``scripts/_sb_battery.py`` references this map so its audit adapters do not re-encode it.
FAMILY_MODEL_REQUIREMENTS: dict[str, str] = {
    "add_cover_shadows": "xt_positional",
    "add_gk_influence": "xt_positional",
    "add_off_ball_run_values": "xt_positional",
    "add_player_influence": "xt_positional",
    "add_obso": "xt_keyword",
    "add_pausa": "xt_keyword",
    "add_space_creation": "xt_keyword",
    "add_defensive_credit": "xg",
    "add_pre_shot_gk_position": "gk_prereq",
    "add_pre_shot_gk_angle": "gk_prereq",
    "add_sync_score": "link",
    "add_visible_area_coverage": "visible_area",
}

#: ``add_*`` exports that are NOT part of the frame-consuming production family: the keeper-identity
#: placement helper (applied by the producer itself as the stamp step, not run as a family) and the
#: jersey/roster helper (consumes different inputs the producer does not hold).
FAMILIES_EXCLUDED: frozenset[str] = frozenset({"add_defending_gk_player_id", "add_gradientsports_player_ids"})


@dataclasses.dataclass(frozen=True)
class TrackingFeaturesReport:
    """Run-level audit of a :func:`run_tracking_features` pass.

    Conserves (ADR-052): ``n_families_run + n_families_skipped == n_families_in``. ``family_status``
    maps each selected family name to ``"ran"`` or ``"skipped: <reason>"`` (an absent model, or the
    exception type + message when the family refused the given frames). ``keeper_report`` is the
    :class:`KeeperIdentityReport` from the single keeper-identity resolution the producer threads.

    Examples
    --------
    >>> from silly_kicks.tracking import TrackingFeaturesReport
    >>> report = TrackingFeaturesReport(
    ...     n_families_in=2,
    ...     n_families_run=1,
    ...     n_families_skipped=1,
    ...     family_status={"add_team_shape": "ran", "add_obso": "skipped: xt not supplied"},
    ...     keeper_report=None,
    ... )
    >>> report.n_families_run + report.n_families_skipped == report.n_families_in
    True
    """

    n_families_in: int
    n_families_run: int
    n_families_skipped: int
    family_status: dict[str, str]
    keeper_report: KeeperIdentityReport | None


def _default_families() -> tuple[str, ...]:
    """The frame-consuming ``add_*`` family the producer runs by default (sorted).

    Derived from ``tracking.__all__`` minus :data:`FAMILIES_EXCLUDED`, so a newly-exported aggregator
    joins the default run automatically.
    """
    return tuple(sorted(n for n in tracking.__all__ if n.startswith("add_") and n not in FAMILIES_EXCLUDED))


def _drop_pointers(links: pd.DataFrame, action_ids: pd.Series) -> pd.DataFrame:
    """A copy of ``links`` whose rows for ``action_ids`` point to no frame.

    ``frame_id`` becomes ``<NA>`` (nullable ``Int64``, the link contract's unlinked value), ``time_offset_seconds``
    and ``link_quality_score`` NaN, and ``n_candidate_frames`` 0: no candidate frame was offered.
    """
    out = links.copy()
    hit = out["action_id"].isin(set(action_ids.tolist())).to_numpy()
    if not hit.any():
        return out
    out["frame_id"] = out["frame_id"].astype("Int64")
    out.loc[hit, "frame_id"] = pd.NA
    for col in ("time_offset_seconds", "link_quality_score"):
        if col in out.columns:
            out[col] = out[col].astype("float64")
            out.loc[hit, col] = float("nan")
    if "n_candidate_frames" in out.columns:
        out.loc[hit, "n_candidate_frames"] = 0
    return out


def _blank_unmeasured(enriched: pd.DataFrame, columns: list[str], unmeasured: np.ndarray) -> pd.DataFrame:
    """Write the missing value of each column's dtype on the unmeasured rows (spec section 5.3).

    The link-provenance columns are left as the dropped pointer made them. A family column whose dtype has no
    missing value (numpy bool or integer) is refused: blanking it would need a dtype change (Hyrum), which the
    spec makes a STOP for the plan.
    """
    out = enriched.copy()
    for col in columns:
        if col in _LINK_PROVENANCE_COLUMNS:
            continue
        dtype = out[col].dtype
        if dtype.kind in "biu" and not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            raise TypeError(
                f"run_tracking_features: family column {col!r} has dtype {dtype} with no missing value, so the "
                "blank step for unmeasured rows cannot write one (action-time repair spec section 5.3)."
            )
        out.loc[unmeasured, col] = _missing_value(dtype)
    return out


def _missing_value(dtype: object) -> Any:
    """The missing value of a column dtype: ``None`` (object), ``pd.NA`` (nullable extension), else NaN."""
    if isinstance(dtype, pd.CategoricalDtype):
        return np.nan
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return pd.NA
    if getattr(dtype, "kind", "") == "O":
        return None
    return np.nan


_LINK_PROVENANCE_COLUMNS = frozenset({"frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"})


def _missing_model_reason(
    req: str,
    *,
    xt: object,
    xg_column: str | None,
    visible_area: pd.DataFrame | None,
) -> str | None:
    """The honest-absence skip reason for a family whose required injected model is not supplied."""
    if req in ("xt_positional", "xt_keyword") and xt is None:
        return "xt not supplied"
    if req == "xg":
        if xg_column is None:
            return "xg_column not supplied"
        if xt is None:
            return "xt not supplied"
    if req == "visible_area" and visible_area is None:
        return "visible_area not supplied"
    return None


def _call_family(
    name: str,
    fn,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None,
    xt: object,
    xg_column: str | None,
    visible_area: pd.DataFrame | None,
    home_team_id: object,
    pitch_control_cache: PitchControlCache | None,
) -> pd.DataFrame:
    """Invoke one family with its correct canonical-convention call shape + injected models.

    Optional kwargs (``links`` / ``pitch_control_cache`` / ``visible_area`` / ``home_team_id``) are
    forwarded ONLY where the signature accepts them (the ``scripts/_sb_battery.generic`` idiom), so a
    family that does not take a shared cache is never handed one.
    """
    req = FAMILY_MODEL_REQUIREMENTS.get(name, "none")
    params = inspect.signature(fn).parameters

    def _opt(kwargs: dict, key: str, val: object) -> None:
        if key in params and val is not None:
            kwargs[key] = val

    if req == "link":
        # add_sync_score(actions, links, *, ...) -- link-consumer, no frames.
        return fn(actions, links)
    if req == "visible_area":
        # add_visible_area_coverage(actions, *, visible_area, links=None) -- no frames.
        kwargs: dict = {"visible_area": visible_area}
        _opt(kwargs, "links", links)
        return fn(actions, **kwargs)

    kwargs = {}
    _opt(kwargs, "links", links)
    _opt(kwargs, "pitch_control_cache", pitch_control_cache)
    _opt(kwargs, "visible_area", visible_area)
    _opt(kwargs, "home_team_id", home_team_id)
    if req == "xt_positional":
        return fn(actions, frames, xt, **kwargs)
    if req == "xt_keyword":
        kwargs["xt"] = xt
        return fn(actions, frames, **kwargs)
    if req == "xg":
        kwargs["xg_column"] = xg_column
        if "xt" in params:
            kwargs["xt"] = xt
        return fn(actions, frames, **kwargs)
    # "gk_prereq" (actions already carry the defending_gk_player_id stamp) and "none" (generic).
    return fn(actions, frames, **kwargs)


def run_tracking_features(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    xt: object = None,
    xg_column: str | None = None,
    roster: dict | None = None,
    identity: Literal["native", "roster"] = "native",
    visible_area: pd.DataFrame | None = None,
    home_team_id: object = None,
    families: list[str] | tuple[str, ...] | None = None,
    pitch_control_cache: PitchControlCache | None = None,
) -> tuple[pd.DataFrame, TrackingFeaturesReport]:
    """Run the frame-consuming ``add_*`` family and return the enriched ACTIONS + a report.

    Parameters
    ----------
    actions, frames
        SPADL actions and long-form tracking frames for ONE match. Neither is mutated.
    links
        Optional pre-computed action->frame pointers (:func:`link_actions_to_frames` output). When
        omitted the producer pre-links once and threads the result into every family.
    xt
        A fitted ``ExpectedThreat`` (duck-typed). Families that need it are SKIPPED when it is ``None``.
    xg_column
        Name of a per-shot xG column on ``actions`` (silly-kicks ships no xG model). ``add_defensive_credit``
        is skipped when it is ``None``.
    roster
        The injected ``{team_id: gk_id}`` map for ``identity="roster"`` (SB360). Ignored on the native path.
    identity
        ``"native"`` (trust the real keeper ``player_id`` the frames already carry) or ``"roster"`` (SB360's
        injected-roster path). The R1 identity->frame bridge is applied ONLY on the roster path.
    visible_area
        Optional per-action ``visible_area`` polygon table (SB360 FOV companions / ``add_visible_area_coverage``).
    home_team_id, families, pitch_control_cache
        Forwarded where accepted; a family subset (default: the full frame-consuming family); a shared pitch-control
        cache (built fresh when omitted).

    Returns
    -------
    tuple[pandas.DataFrame, TrackingFeaturesReport]
        The enriched actions (the caller's actions plus ``defending_gk_player_id`` and every emitted
        feature column) and the conserving report.

    Notes
    -----
    **The producer NEVER crashes, and a family that RAISES is RECORDED, not fabricated -- so the report
    is the authority on what actually ran.** Each family runs under a per-family ``try/except`` (the
    ``run_add_star_battery`` precedent): a family whose injected model is absent, or which raises on the
    given frames, is recorded in ``report.family_status`` as ``"skipped: <reason>"`` (an absent model, or
    ``"skipped: <ExcType>: <msg>"`` for an exception) and adds NO columns -- the enriched actions simply
    lack that family's columns. A consumer that reads only the enriched actions must therefore INSPECT
    ``report.family_status`` to distinguish a family that ran from one that was skipped: even an
    unexpected family bug is recorded, not raised, so it is visible in the report but shows up in the
    output only as silently-absent columns. This is the honest-degradation contract (ADR-063/054) -- a
    missing column is never a fabricated value -- but it puts the onus on the caller to check the report.

    Examples
    --------
    Run the GK family on SB360 freeze-frames with an injected roster -- the producer resolves keeper
    identity, bridges it onto both grains, and unlocks ``add_pre_shot_gk_position``::

        from silly_kicks.tracking import run_tracking_features

        enriched, report = run_tracking_features(
            actions,
            frames,
            identity="roster",
            roster={home_team_id: home_gk_id, away_team_id: away_gk_id},
            families=["add_pre_shot_gk_position"],
        )
        enriched["pre_shot_gk_x"]           # a REAL keeper position, not NaN
        report.family_status                # -> {"add_pre_shot_gk_position": "ran"}

    See NOTICE for full bibliographic citations.
    """
    # 1. Resolve keeper identity ONCE, then bridge onto BOTH grains. The resolver is pure; the two
    #    placement helpers return COPIES, so the caller's actions/frames are never mutated. The frame
    #    bridge is applied ONLY on the roster path -- native frames already carry real keeper ids and
    #    stamping the per-period consensus would clobber a mid-period substitution.
    keeper_map, keeper_report = resolve_keeper_identities(actions, frames, identity=identity, roster=roster)
    work_actions = add_defending_gk_player_id(actions, keeper_map)
    work_frames = apply_keeper_identities_to_frames(frames, keeper_map) if identity == "roster" else frames

    # 2. Pre-link ONCE + share one pitch-control cache (both threaded into every family that accepts them).
    if links is None:
        links, _link_report = link_actions_to_frames(work_actions, work_frames)
    if pitch_control_cache is None:
        pitch_control_cache = PitchControlCache()
    # 2b. Action-time repair spec section 5.3: a "placeholder" or "order_only" row has no measured time,
    #     so it takes no frame. Its pointer is dropped here; the blank step after the families covers the families
    #     that slice frames by the action time instead of the pointer.
    unmeasured = is_time_unmeasured(work_actions)
    if unmeasured.any():
        links = _drop_pointers(links, work_actions.loc[unmeasured, "action_id"])

    # 3/5. Resolve the family selection. De-dup a caller-supplied list (order-preserving) so a
    # duplicated family name cannot break the report's conservation invariant -- `family_status` is
    # keyed by name, so `n_families_in` must count DISTINCT families. `_default_families()` is already
    # dup-free.
    selected: tuple[str, ...] = tuple(dict.fromkeys(families)) if families is not None else _default_families()

    # 4. Run each family under a per-family guard -- the producer never crashes. New columns accumulate
    #    onto the stamped actions; linkage-provenance columns are merged once (idempotent).
    enriched = work_actions.copy()
    family_status: dict[str, str] = {}
    for name in selected:
        fn = getattr(tracking, name, None)
        if fn is None:
            family_status[name] = "skipped: not a tracking export"
            continue
        req = FAMILY_MODEL_REQUIREMENTS.get(name, "none")
        missing = _missing_model_reason(req, xt=xt, xg_column=xg_column, visible_area=visible_area)
        if missing is not None:
            family_status[name] = f"skipped: {missing}"
            continue
        try:
            result = _call_family(
                name,
                fn,
                work_actions,
                work_frames,
                links=links,
                xt=xt,
                xg_column=xg_column,
                visible_area=visible_area,
                home_team_id=home_team_id,
                pitch_control_cache=pitch_control_cache,
            )
            for col in result.columns:
                if col not in enriched.columns:
                    # Index-aligned Series assignment (result shares work_actions' index) preserves the
                    # emitted column's dtype -- a positional ndarray assign would coerce nullable ids.
                    enriched[col] = result[col]
            family_status[name] = "ran"
        except Exception as exc:  # a refusal on the given frames is a RECORDED result, not a crash
            family_status[name] = f"skipped: {type(exc).__name__}: {exc}"

    # 4b. The blank step (spec section 5.3): every column a family added gets the missing value of its dtype on each
    #     unmeasured row. The link-provenance columns already carry the dropped pointer (2b).
    if unmeasured.any():
        enriched = _blank_unmeasured(
            enriched, [c for c in enriched.columns if c not in work_actions.columns], unmeasured
        )

    n_run = sum(1 for status in family_status.values() if status == "ran")
    report = TrackingFeaturesReport(
        n_families_in=len(selected),
        n_families_run=n_run,
        n_families_skipped=len(family_status) - n_run,
        family_status=family_status,
        keeper_report=keeper_report,
    )
    return enriched, report
