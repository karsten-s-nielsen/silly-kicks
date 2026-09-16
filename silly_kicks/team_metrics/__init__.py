"""silly-kicks event-only team-match KPI module (TF-52).

Pure-pandas-over-SPADL, per-``(game_id, team_id)`` team-match KPIs: the Twelve match-report glossary
(PPDA, field tilt, pass tempo, line heights, recoveries, conversion chain, ...) plus the event-based
practitioner set (counter-press windows, post-regain security, build-up taxonomy, breakout-by-channel,
switch-conditioned press). Rests on a single possession-foundation layer that reuses
``spadl.add_possessions``.

Hexagonal / event-only: imports ``silly_kicks.spadl`` + ``silly_kicks.id_compat`` +
``silly_kicks.reflection`` + numpy/pandas ONLY; NEVER ``silly_kicks.tracking`` (pinned by
``tests/team_metrics/test_import_allowlist.py``). NOTHING imports ``team_metrics``. A ``compute_*``,
not an ``add_*`` -- no ``*_xfns``, in no default xfn list. Additive -- no VAEP/tracking retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import TEAM_KPI_COLUMNS, TEAM_KPI_KEYS, TEAM_KPI_METRIC_COLUMNS
from ._compute import compute_team_kpis
from ._config import COUNTERPRESS_PRESETS, CounterpressWindow, TeamKpiParams
from ._report import TeamKpiReport

__all__ = [
    "COUNTERPRESS_PRESETS",
    "TEAM_KPI_COLUMNS",
    "TEAM_KPI_KEYS",
    "TEAM_KPI_METRIC_COLUMNS",
    "CounterpressWindow",
    "TeamKpiParams",
    "TeamKpiReport",
    "compute_team_kpis",
]
