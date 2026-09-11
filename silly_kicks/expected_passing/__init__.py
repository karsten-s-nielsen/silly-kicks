"""silly-kicks expected-passing model (TF-54b).

Event-only ``P(complete | origin -> target geometry)``: a reusable "expected passing"
pass-completion model injected by the territorial-dominance counterfactual metric. Follows the
house bundled-trained-artifact discipline -- pickle-free JSON + SHA256SUMS, a feature contract, a
chirality probe, fail-closed load -- and inference imports NO sklearn (sklearn is imported
function-locally inside ``PassCompletionModel.fit`` only).

Hexagonal / event-only: imports ``silly_kicks.spadl`` + ``silly_kicks.id_compat`` + numpy ONLY;
NEVER ``silly_kicks.tracking`` (pinned by ``tests/expected_passing/test_import_allowlist.py``).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._model import PassCompletionIntegrityError, PassCompletionModel

__all__ = [
    "PassCompletionIntegrityError",
    "PassCompletionModel",
]
