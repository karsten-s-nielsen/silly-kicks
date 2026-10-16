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


def match_id_pairs(providers, match_ids) -> list[list[str]]:
    """Sorted unique ``[provider, match_id]`` pairs of a per-row corpus."""
    return [list(p) for p in sorted({(str(p), str(m)) for p, m in zip(providers, match_ids, strict=True)})]


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
    lets two artifacts be proven to share (or not share) a corpus.
    """
    pairs = match_id_pairs(providers, match_ids)
    if all_public and pairs:
        return {"corpus_match_ids": pairs}
    digest = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"corpus_match_ids_sha256": digest, "corpus_n_matches": len(pairs)}


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
