"""SkillCorner provider port (TF-59 PR1).

Public surface:

* ``extract_keeper_appearances`` --- the keeper-appearance-interval extractor (spec §5.5), producing
  the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port from a parsed SkillCorner
  ``match.json`` dict (``players[].playing_time.by_period[]`` + ``match_periods``).

Deliberately LIGHT: it imports ONLY ``.appearances`` (which is tracking-free --
``keeper_identity`` + pandas/stdlib), so ``import silly_kicks.providers.skillcorner`` never pulls the
heavy tracking / converter chain. The SkillCorner raw shaping (tracking + SPADL) still lives in
``spadl/skillcorner.py`` / ``tracking/skillcorner.py`` / ``scripts/_loader_pining.py`` and is NOT
re-exported here.
"""

from __future__ import annotations

from .appearances import extract_keeper_appearances

__all__ = ["extract_keeper_appearances"]
