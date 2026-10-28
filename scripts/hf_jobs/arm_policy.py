"""Launcher-side (OFF-HF) public/owner arm policy (ADR-123 Amendment B).

The ONLY place the HF path touches the public/owner taxonomy. It reads the in-repo redistributability
registry and emits a NEUTRAL ``--arms`` payload keyed by the DATASET's own IN-JOB match keys. The
converter->identity reconciliation (the Bundesliga matches are SPADL-converted under one provider name
but the registry keys them under their licensing identity) is applied INTERNALLY for the registry lookup
only; the EMITTED keys are always the in-job dataset keys (reviewer HVB-SPEC-01 -- else the in-job
``np.isin`` misses those rows and the public arm is empty: the fixed bug, relocated).

The emitted key must EQUAL the in-job ``arms.match_key(source_provider, game_dir.name)``. For the
owner_pergame trainers (xshot/xcross) the job reads each flat unit as a ``{provider}-{game}`` per-game unit
(no ``__`` -- the driver key separator), so the in-job key is ``{source_provider}__{provider}-{game}``,
derived here from each LOCKED unit: its stem ``{provider}__{game}`` AND its recorded ``source_provider``. The
two providers can differ: the DFL matches are staged as ``idsse__<game>`` with ``source_provider`` ``sportec``
(HFU R4-10, 2026-10-30: a key built from the stem alone named no row, and every arm lost those matches).

This module is imported ONLY on the local launcher; it never runs in-job (the job receives
``/tmp/arms.json`` as pre-written data). It is deliberately NOT in any trainer's import closure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scripts._corpus import PUBLIC_CORPUS

# Single-sourced in the neutral provenance module (the in-job trainers normalise provider_list with the
# same map; _corpus_identity must stay I1-clean, so the map lives there, not here).
from scripts._corpus_identity import _CONVERTER_TO_IDENTITY
from scripts.hf_jobs.arms import match_key
from scripts.hf_jobs.datasets import pergame_name


def _is_public(provider: str, game: str) -> bool:
    identity = _CONVERTER_TO_IDENTITY.get(provider, provider)
    return game in PUBLIC_CORPUS.get(identity, frozenset())


def _injob_key(source_provider: str, provider: str, game: str) -> str:
    """The key the owner_pergame trainer computes for a staged unit, single-sourced with the job's own functions:
    ``match_key(source_provider, pergame_name(provider, game))``."""
    return match_key(source_provider, pergame_name(provider, game))


def compute_arms(units: Sequence[Mapping[str, str]]) -> dict:
    """The ``--arms`` JSON payload for a staged flat corpus, built OFF-HF from ``PUBLIC_CORPUS``.

    ``units`` are the staged dataset's locked units (`datasets.staged_units`): each has its shard filename stem
    ``{provider}__{game}`` and its ``source_provider``. The emitted arm memberships use the IN-JOB match keys
    (``{source_provider}__{provider}-{game}``); the registry lookup uses the stem's provider. An arm is included
    only when non-empty and different from the one it extends, so an all-public corpus yields a single ``public``
    arm.
    """
    public, extended, full = [], [], []
    for unit in units:
        provider, _, game = unit["stem"].partition("__")
        key = _injob_key(unit["source_provider"], provider, game)
        full.append(key)
        if _is_public(provider, game):
            public.append(key)
            extended.append(key)
        elif provider == "skillcorner":  # owner-tier SkillCorner -> sc_extended (not public, not full)
            extended.append(key)
    arms: dict[str, list[str]] = {"public": public}
    if set(extended) != set(public):
        arms["sc_extended"] = extended
    if set(full) != set(extended):
        arms["full"] = full
    arms = {name: keys for name, keys in arms.items() if keys}
    ship_order = [name for name in ("public", "sc_extended", "full") if name in arms]
    return {"arms": arms, "ship_order": ship_order}


#: The arms a campaign tier ships at most (spec r4 D18 (c)). Tiers that read one dataset share its store and its
#: extraction, so the launcher hands each tier only these arms; every in-job reader of the shared extract output
#: then keeps only the rows of its own arms (`scripts.hf_jobs.arms.restrict_to_arms`).
TIER_ARMS: dict[str, tuple[str, ...]] = {"public": ("public",), "sc_extended": ("public", "sc_extended")}


def arms_for_tier(payload: dict, tier: str) -> dict:
    """``payload`` (a `compute_arms` result) cut to the arms ``tier`` ships at most, in ship order. A tier without a
    policy refuses, named: a visibility-gated trainer must never run with arms its tier does not ship."""
    allowed = TIER_ARMS.get(tier)
    if allowed is None:
        raise ValueError(f"no arm policy for the campaign tier {tier!r} (known tiers: {sorted(TIER_ARMS)})")
    arms = {name: keys for name, keys in payload["arms"].items() if name in allowed}
    return {"arms": arms, "ship_order": [name for name in payload["ship_order"] if name in arms]}
