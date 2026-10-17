"""Launcher-side (OFF-HF) public/owner arm policy (ADR-123 Amendment B).

The ONLY place the HF path touches the public/owner taxonomy. It reads the in-repo redistributability
registry and emits a NEUTRAL ``--arms`` payload keyed by the DATASET's own IN-JOB match keys. The
converter->identity reconciliation (the Bundesliga matches are SPADL-converted under one provider name
but the registry keys them under their licensing identity) is applied INTERNALLY for the registry lookup
only; the EMITTED keys are always the in-job dataset keys (reviewer HVB-SPEC-01 -- else the in-job
``np.isin`` misses those rows and the public arm is empty: the fixed bug, relocated).

The emitted key must EQUAL the in-job ``arms.match_key(source_provider, game_dir.name)``. For the
owner_pergame trainers (xshot/xcross) the job ``repackage``s the flat corpus into ``{provider}-{game}``
per-game dirs (no ``__`` -- the driver key separator), so the in-job key is
``{provider}__{provider}-{game}``, derived here from each flat shard stem ``{provider}__{game}``.

This module is imported ONLY on the local launcher; it never runs in-job (the job receives
``/tmp/arms.json`` as pre-written data). It is deliberately NOT in any trainer's import closure.
"""

from __future__ import annotations

from collections.abc import Sequence

from scripts._corpus import PUBLIC_CORPUS

# Single-sourced in the neutral provenance module (the in-job trainers normalise provider_list with the
# same map; _corpus_identity must stay I1-clean, so the map lives there, not here).
from scripts._corpus_identity import _CONVERTER_TO_IDENTITY


def _is_public(provider: str, game: str) -> bool:
    identity = _CONVERTER_TO_IDENTITY.get(provider, provider)
    return game in PUBLIC_CORPUS.get(identity, frozenset())


def _injob_key(provider: str, game: str) -> str:
    """The key the owner_pergame trainer computes: ``match_key(source_provider, "{provider}-{game}")``."""
    return f"{provider}__{provider}-{game}"


def compute_arms(flat_stems: Sequence[str]) -> dict:
    """The ``--arms`` JSON payload for a staged flat corpus, built OFF-HF from ``PUBLIC_CORPUS``.

    ``flat_stems`` are the staged dataset's shard filename stems ``{provider}__{game}``. The emitted arm
    memberships use the IN-JOB match keys (``{provider}__{provider}-{game}``). An arm is included only
    when non-empty and different from the one it extends, so an all-public corpus yields a single
    ``public`` arm.
    """
    public, extended, full = [], [], []
    for stem in flat_stems:
        provider, _, game = stem.partition("__")
        key = _injob_key(provider, game)
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
