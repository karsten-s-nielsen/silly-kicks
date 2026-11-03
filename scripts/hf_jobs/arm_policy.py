"""Launcher-side (OFF-HF) public/owner arm policy (ADR-123 Amendment B).

The ONLY place the HF path touches the public/owner taxonomy. It reads the in-repo redistributability
registry and emits a NEUTRAL ``--arms`` payload keyed by the DATASET's own IN-JOB match keys. The
converter->identity reconciliation (the Bundesliga matches are SPADL-converted under one provider name
but the registry keys them under their licensing identity) is applied INTERNALLY for the registry lookup
only; the EMITTED keys are always the in-job dataset keys (reviewer HVB-SPEC-01 -- else the in-job
``np.isin`` misses those rows and the public arm is empty: the fixed bug, relocated).

The emitted key must EQUAL the in-job key of the owner_pergame trainers (xshot/xcross):
``arms.match_key(source_provider, datasets.pergame_name(provider, game))``, built here with the same two
functions from each staged unit (``datasets.staged_units``: ``stem`` = ``{provider}__{game}``, ``game``,
``source_provider``). The stem names the licensing identity (``idsse``); ``source_provider`` names the SPADL
converter (``sportec``), so the two differ for the Bundesliga matches (HFU spec 16.5, F29).

This module is imported ONLY on the local launcher; it never runs in-job (the job receives
``/tmp/arms.json`` as pre-written data). It is deliberately NOT in any trainer's import closure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scripts._corpus import PUBLIC_CORPUS

# Single-sourced in the neutral provenance module (the in-job trainers normalise provider_list with the
# same map; _corpus_identity must stay I1-clean, so the map lives there, not here).
from scripts._corpus_identity import _CONVERTER_TO_IDENTITY


def _is_public(provider: str, game: str) -> bool:
    identity = _CONVERTER_TO_IDENTITY.get(provider, provider)
    return game in PUBLIC_CORPUS.get(identity, frozenset())


def _injob_key(unit: Mapping[str, str]) -> str:
    """The key the owner_pergame trainer computes from a staged unit (``_extract_refs``)."""
    from scripts.hf_jobs.arms import match_key
    from scripts.hf_jobs.datasets import pergame_name

    provider = unit["stem"].partition("__")[0]
    return match_key(unit["source_provider"], pergame_name(provider, unit["game"]))


def compute_arms(units: Sequence[Mapping[str, str]]) -> dict:
    """The ``--arms`` JSON payload for a staged flat corpus, built OFF-HF from ``PUBLIC_CORPUS``.

    ``units`` are the staged dataset's units (``datasets.staged_units``). The emitted arm memberships use the
    IN-JOB match keys (``_injob_key``). An arm is included only
    when non-empty and different from the one it extends, so an all-public corpus yields a single
    ``public`` arm.
    """
    public, extended, full = [], [], []
    for unit in units:
        provider, _, game = unit["stem"].partition("__")
        key = _injob_key(unit)
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
