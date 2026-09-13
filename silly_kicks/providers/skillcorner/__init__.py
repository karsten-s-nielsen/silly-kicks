"""SkillCorner provider port.

Public surface:

* ``extract_keeper_appearances`` (TF-59 PR1) --- the keeper-appearance-interval extractor (spec §5.5),
  producing the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port from a parsed
  SkillCorner ``match.json`` dict (``players[].playing_time.by_period[]`` + ``match_periods``).
* ``parse_passing_options`` (TF-62) --- shapes the Game-Intelligence ``passing_option`` rows into the
  canonical option-set rows the GK-decision metric's native tier consumes.

Deliberately LIGHT: it imports ONLY ``.appearances`` + ``.gi`` (both tracking-free --
``keeper_identity`` + pandas/stdlib), so ``import silly_kicks.providers.skillcorner`` never pulls the
heavy tracking / converter chain. The SkillCorner tracking + SPADL raw shaping still lives in
``spadl/skillcorner.py`` / ``tracking/skillcorner.py`` / ``scripts/_loader_pining.py``.
"""

from __future__ import annotations

from .appearances import extract_keeper_appearances
from .gi import parse_passing_options

__all__ = ["extract_keeper_appearances", "parse_passing_options"]
