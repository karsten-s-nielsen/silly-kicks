"""Policy-free corpus provenance + ship-label helpers (ADR-123 Amendment B).

These helpers carry NO public/private rule: they take already-decided inputs (an ``all_public`` flag, a
``shipped`` arm name) and emit provenance. They were split out of the visibility-taxonomy module so the
HF in-job trainers can import them WITHOUT pulling that module -- which defines the redistributability
registry and its row-mask / assertion / label helpers -- into their import closure (I1). The taxonomy
stays launcher/off-HF only. This module must NEVER import the taxonomy module, and its source carries
none of the taxonomy symbol names (the I1 gate is substring-based).
"""

from __future__ import annotations

import hashlib
import json

#: Dataset ``source_provider`` (the SPADL converter name) -> the licensing/dataset IDENTITY recorded in an
#: artifact's ``provider_list`` and used in the model-card vocabulary. The Bundesliga/DFL matches are
#: converted by ``sportec_spadl`` so their frames carry ``source_provider="sportec"``, but the dataset
#: identity every consumer keys on is ``idsse``. This is a NAMING normalization -- NOT a public/private
#: rule -- single-sourced here so the launcher arm policy and the in-job trainers share one vocabulary
#: (I1-safe: no redistributability symbol, so it does not pull the taxonomy into the trainer closure).
_CONVERTER_TO_IDENTITY = {"sportec": "idsse"}


def canonical_providers(providers) -> list[str]:
    """Sorted unique provider IDENTITIES for an artifact's ``provider_list`` (converter names normalised,
    e.g. ``sportec`` -> ``idsse``) -- the stable vocabulary the metadata test, model cards and consumers
    expect. The raw ``source_provider`` is unchanged elsewhere (arm masking still keys on it)."""
    return sorted({_CONVERTER_TO_IDENTITY.get(str(p), str(p)) for p in providers})


def match_id_pairs(providers, match_ids) -> list[list[str]]:
    """Sorted unique ``[provider, match_id]`` pairs of a per-row corpus."""
    return [list(p) for p in sorted({(str(p), str(m)) for p, m in zip(providers, match_ids, strict=True)})]


def canonical_match_id(provider: str, match_id: str) -> str:
    """The canonical id of a match the pipeline names by its staged per-game dir (HFU spec 18.6):
    ``pergame_name`` gives ``f"{stem provider}-{game}"``. A leading ``f"{q}-"`` is stripped, ``q`` the pair's own
    provider or its identity (``sportec`` -> ``idsse``); any other id is returned unchanged, so an already-canonical
    id stays (a dashed ``DFL-MAT-...`` included)."""
    mid, prov = str(match_id), str(provider)
    for q in dict.fromkeys((prov, _CONVERTER_TO_IDENTITY.get(prov, prov))):
        if mid.startswith(f"{q}-"):
            return mid[len(q) + 1 :]
    return mid


def canonical_probe_ids(metrics: dict) -> dict:
    """Off-HF finalize of an xCross ``metrics.json`` (HFU spec 18.6): a NEW dict whose ``probe_sample_matches``
    pairs carry canonical ids (:func:`canonical_match_id`) and whose ``probe_sample_in_training_folds`` keys are
    mapped through those pairs (a key no pair names stays). A metrics without the fields is returned unchanged."""
    out = dict(metrics)
    rename: dict[str, str] = {}
    pairs = metrics.get("probe_sample_matches")
    if isinstance(pairs, list):
        new = []
        for prov, mid in pairs:
            rename[str(mid)] = canonical_match_id(prov, mid)
            new.append([prov, rename[str(mid)]])
        out["probe_sample_matches"] = new
    folds = metrics.get("probe_sample_in_training_folds")
    if isinstance(folds, dict):
        out["probe_sample_in_training_folds"] = {rename.get(str(k), k): v for k, v in folds.items()}
    return out


def check_expected_variant(expected: str | None, *, all_public: bool) -> None:
    """G1 launch preflight, run BEFORE extraction: an expected ``public`` artifact needs an all-public corpus.

    A ``public`` bundle can only come from a corpus in which every requested match is public: any
    owner-tier match makes the paired gate eligible to ship a restricted variant. ``sc_extended`` and
    ``full`` expectations are enforced at ship time only (:func:`check_shipped_variant`).
    """
    if expected == "public" and not all_public:
        raise SystemExit(
            "--expect-variant public, but the requested corpus contains non-public matches (is the "
            "OWNER pining token set?). A public bundle must be trained on public matches only. "
            "Refusing before extraction."
        )


def check_shipped_variant(expected: str | None, shipped: str) -> None:
    """G1 ship-time check: never write an artifact whose shipped variant differs from the expectation."""
    if expected is not None and shipped != expected:
        raise SystemExit(
            f"--expect-variant {expected}, but this run would ship {shipped!r}. Refusing to write the artifact."
        )


def corpus_identity(providers, match_ids, *, all_public: bool) -> dict:
    """The corpus identity an artifact may record: the exact ids for an all-public corpus, else a digest.

    Hub publishes copy ``metrics.json``, so an owner/NDA id must never appear in one; the digest still
    lets two artifacts be proven to share (or not share) a corpus. The in-job trainers always pass
    ``all_public=False`` (HFU spec 17.2, owner 2026-11-09: digest only; gated by
    ``test_in_job_trainers_record_the_corpus_digest``): the canonical public ids come only from
    ``_corpus.rehydrate_public_corpus_ids`` at adoption. Do not re-wire the ``all_public=True`` branch into a trainer.
    """
    pairs = match_id_pairs(providers, match_ids)
    if all_public and pairs:
        return {"corpus_match_ids": pairs}
    digest = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"corpus_match_ids_sha256": digest, "corpus_n_matches": len(pairs)}


def open_data_identity(match_ids) -> dict:
    """The corpus identity of a StatsBomb open-data corpus: the digest over its ``("statsbomb", match_id)``
    pairs. ONE function for the lock (``sb_open.identity.open``) and every event-only trainer, so verify
    compares two strings (HFU spec 14.3)."""
    ids = [str(m) for m in match_ids]
    return corpus_identity(["statsbomb"] * len(ids), ids, all_public=False)


def reproducibility(shipped: str, providers, *, training_commit: str | None = None) -> dict:
    """The ADR-067 M4 caveat, emitted by the TRAINER -- never hand-added to a driver-stamped artifact."""
    if shipped == "public":
        return {"reproducibility": "public"}
    provs = ", ".join(sorted({str(p) for p in providers}))
    at = f" at training_commit {training_commit}" if training_commit else ""
    return {
        "reproducibility": "restricted",
        "reproducibility_note": (
            f"Restricted: trained{at} on a corpus ({provs}) that includes matches which cannot be "
            f"redistributed or carries no public-visibility proof (variant {shipped!r}), so the weights "
            "cannot be reproduced from public data (ADR-067 M4)."
        ),
    }
