"""Gradient Sports provider port (TF-59 PR1).

Public surface:

* ``extract_keeper_appearances`` --- the keeper-appearance-interval extractor (spec §5.5), producing
  the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port from a GS raw ``events``
  list + ``roster`` list.

Deliberately LIGHT: it imports ONLY ``.appearances`` (which is tracking-free -- ``keeper_identity`` +
``id_compat`` + pandas/stdlib), so ``import silly_kicks.providers.gradientsports`` never pulls the
heavy tracking / converter chain. The GS raw shaping (tracking + SPADL) still lives in
``spadl/gradientsports.py`` / ``tracking/gradientsports.py`` / ``scripts/_loader_pining.py`` and is
NOT re-exported here.
"""

from __future__ import annotations

from .appearances import extract_keeper_appearances

__all__ = ["extract_keeper_appearances"]
