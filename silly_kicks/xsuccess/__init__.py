"""silly-kicks xSuccess (TF-61): event-only, END-BLIND action-completion ``P(success | context)``
over all on-ball action types.

Hexagonal / event-only: imports ``silly_kicks.spadl`` + ``silly_kicks.id_compat`` + numpy/pandas
ONLY (``xgboost``/``sklearn`` are training-only, function-local); NEVER ``silly_kicks.tracking``
(pinned by ``tests/xsuccess/test_import_allowlist.py``). Nothing imports ``xsuccess`` except ``vaep``.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._model import XSuccessIntegrityError, XSuccessModel

__all__ = ["XSuccessIntegrityError", "XSuccessModel"]
