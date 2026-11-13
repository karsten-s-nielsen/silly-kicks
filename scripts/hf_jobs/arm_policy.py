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

from scripts._corpus import HELD_OUT_PROBE, PUBLICLY_OBTAINABLE, in_registry, unit_identity

#: The arms a visibility-gated trainer can receive, in ship order (ADR-132: `full` is retired for new artifacts).
ARM_ORDER = ("public", "sc_extended")


def _injob_key(unit: Mapping[str, str]) -> str:
    """The key the owner_pergame trainer computes from a staged unit (``_extract_refs``)."""
    from scripts.hf_jobs.arms import match_key
    from scripts.hf_jobs.datasets import pergame_name

    provider = unit["stem"].partition("__")[0]
    return match_key(unit["source_provider"], pergame_name(provider, unit["game"]))


def _is_probe(unit: Mapping[str, str]) -> bool:
    return in_registry(HELD_OUT_PROBE, *unit_identity(unit))


def compute_arms(units: Sequence[Mapping[str, str]]) -> dict:
    """The ``--arms`` JSON payload for a staged flat corpus, built OFF-HF from the ``_corpus`` registries (ADR-132).

    ``units`` are the staged dataset's units (``datasets.staged_units``); the emitted memberships use the IN-JOB
    match keys (``_injob_key``). ``public`` = the publicly obtainable units minus the held-out probe;
    ``sc_extended`` = ``public`` + the licensed SkillCorner units. A probe unit is in no arm. A non-probe unit that
    is neither obtainable nor SkillCorner has no arm and is REFUSED (fail loud, never dropped silently). An arm is
    included only when non-empty and different from the one it extends, so an all-obtainable corpus yields a single
    ``public`` arm.
    """
    public, extended, orphans = [], [], []
    for unit in units:
        if _is_probe(unit):
            continue
        key = _injob_key(unit)
        provider, game = unit_identity(unit)
        if in_registry(PUBLICLY_OBTAINABLE, provider, game):
            public.append(key)
            extended.append(key)
        elif provider == "skillcorner":  # licensed SkillCorner -> sc_extended only
            extended.append(key)
        else:
            orphans.append(unit["stem"])
    if orphans:
        raise ValueError(
            f"unit(s) with no arm: {sorted(orphans)} are neither publicly obtainable nor licensed SkillCorner. "
            "Register them in scripts/_corpus.py (a licensing decision) or remove them from the dataset."
        )
    arms: dict[str, list[str]] = {"public": public}
    if set(extended) != set(public):
        arms["sc_extended"] = extended
    arms = {name: keys for name, keys in arms.items() if keys}
    return {"arms": arms, "ship_order": [name for name in ARM_ORDER if name in arms]}


#: The staged non-tracking sources, which keep their declared provenance: ``snapshot`` = the SB360 freeze-frame
#: shards (``sb360``). A provider in neither this set nor the tracking registries cannot be classified (fail closed).
_NON_TRACKING_PROVIDERS = frozenset({"snapshot"})


def _is_tracking(name: str, units: Sequence[Mapping[str, str]]) -> bool:
    """True for a tracking dataset (its providers are in the registries); False for a known non-tracking one (the
    StatsBomb corpora keep their declared provenance). A dataset that mixes both, or that has a provider in neither
    set, refuses (ADR-132, spec 3: fail closed; review B CW-IB-03)."""
    providers = {unit_identity(u)[0] for u in units}
    unknown = providers - set(PUBLICLY_OBTAINABLE) - _NON_TRACKING_PROVIDERS
    if unknown:
        raise ValueError(
            f"{name}: cannot classify provider(s) {sorted(unknown)} as tracking or non-tracking. Register them in "
            "scripts/_corpus.py (a licensing decision) or in arm_policy._NON_TRACKING_PROVIDERS."
        )
    tracking = providers & set(PUBLICLY_OBTAINABLE)
    if tracking and tracking != providers:
        raise ValueError(f"{name}: mixes tracking and non-tracking providers {sorted(providers)}")
    return bool(tracking)


def check_provenance(name: str, declared, units: Sequence[Mapping[str, str]]) -> None:
    """`datasets record`'s provenance check (ADR-132, spec 6.2): a tracking dataset's ``provenance`` is ``public`` iff
    every staged unit is publicly obtainable (held-out probe units included), else ``extended``; a mismatch refuses."""
    if not _is_tracking(name, units):
        return
    all_obtainable = all(in_registry(PUBLICLY_OBTAINABLE, *unit_identity(u)) for u in units)
    want = "public" if all_obtainable else "extended"
    if declared.provenance != want:
        why = "every unit is publicly obtainable" if all_obtainable else "some units are licensed"
        raise ValueError(
            f"{name}: declared provenance = {declared.provenance!r}, but its staged units make it {want!r} ({why})"
        )


def public_ids(units: Sequence[Mapping[str, str]]) -> list[list[str]] | None:
    """A dataset's ``public_ids`` (spec 6.2, INV-9): the ``[identity, game]`` pairs of the units `compute_arms` puts in
    the ``public`` arm, sorted; None for a non-tracking dataset or one with no public-arm unit."""
    if not _is_tracking("dataset", units):
        return None
    public = set(compute_arms(units)["arms"].get("public", []))
    pairs = sorted([list(unit_identity(u)) for u in units if _injob_key(u) in public])
    return pairs or None


def restrict_arms(payload: Mapping, arms: Sequence[str], units: Sequence[Mapping[str, str]]) -> dict:
    """``payload`` (``compute_arms``) restricted to a campaign job's declared ``arms`` (spec 5.5).

    Refuses (``ValueError``) an empty declaration, an arm the payload lacks, and ONE declared arm that is a strict
    subset of the dataset's units, held-out probe units included: a single-arm job studies and gates on every dataset
    row while it fits on its arm's rows, so its arm must cover them all (a probe unit in that study would break the
    probe's held-out status, spec 5.4). ``ship_order`` keeps the payload order of the kept arms.
    """
    declared = list(dict.fromkeys(arms))
    if not declared:
        raise ValueError("a visibility-gated job must declare a non-empty `arms` list")
    missing = [a for a in declared if a not in payload["arms"]]
    if missing:
        raise ValueError(f"arm(s) {missing} not produced by the dataset (it produces {sorted(payload['arms'])})")
    if len(declared) == 1:
        keys = set(payload["arms"][declared[0]])
        every = {_injob_key(u) for u in units}
        if keys != every:
            raise ValueError(
                f"one-arm job on {declared[0]!r} covers {len(keys)} of the dataset's {len(every)} units: a strict "
                "subset (spec 5.5). Declare both arms (paired) or use a dataset the arm covers fully."
            )
    return {
        "arms": {a: list(payload["arms"][a]) for a in payload["arms"] if a in declared},
        "ship_order": [a for a in payload["ship_order"] if a in declared],
    }
