"""Restart-validation gate contract (TF-46 Arm 1 Part 2, spec sections 5.1 and 15).

- ``DataProfile`` -- the (provider, tracking release, GI release, competition-season) key a gate cell is scored
  under. It has NO pairing field: GI / tracking pairing is decided per MATCH, measured (spec section 6.1).
- ``RESTART_GATE_TABLE`` / ``gate`` -- the gate table and its fail-closed lookup. The table SHIPS EMPTY
  (``MappingProxyType({})``): every gated tracking-derived output is withheld. The joint release derives the
  real table from the section 16 pre-flight aggregate (ADR-056).

Pure; no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

# ---------------------------------------------------------------------------
# Data profile (spec section 5.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataProfile:
    """The key a gate cell is scored under. No pairing field (spec section 6.1, C-11 ruling): GI / tracking
    pairing is per MATCH and measured, never a profile-level flag."""

    provider: str
    tracking_release: str
    gi_release: str | None
    competition_season: str


# ---------------------------------------------------------------------------
# Gate entry and key (spec section 15)
# ---------------------------------------------------------------------------


#: The bases a cell can pass on (spec section 5.3).
GateBase = Literal["idsse", "gradientsports", "combined", "native"]


@dataclass(frozen=True)
class GateEntry:
    """One ratified cell: the passing tier and base, the shipped metric, the Wilson and cluster-bootstrap lower
    bounds, the sample size n, the match count m, and how the cell resolved (``"cell"`` or ``"pooled_<level>"``)."""

    tier: Literal["reference", "transfer"]
    base: GateBase
    metric: float
    wilson_lb: float
    bootstrap_lb: float
    n: int
    m: int
    cell_resolution: str
    #: ``kick_time`` only (spec section 6.5 / 16.1, amended 2026-10-22): the pre-registered agreement winner, the
    #: candidate (``tf79_contact`` / ``ball_onset``) with the smaller median absolute error against K in run (2) for
    #: the data profile x restart type; an exact tie (or no paired error) gives the documented default
    #: ``tf79_contact``. ``None`` on every other signal.
    agreement_winner: str | None = None


#: ``(data_profile, restart_type, signal, cell_state)``. An absent key is withheld and counted.
GateKey = tuple[DataProfile, str, str, str]

#: The shipped table. EMPTY until the joint release derives it (spec section 15): every lookup is withheld,
#: fail-closed.
RESTART_GATE_TABLE: Mapping[GateKey, GateEntry] = MappingProxyType({})

#: A module counter: every withheld (absent-key) ``gate`` lookup increments it. Visibility only; never decisive.
withheld_lookups: int = 0


def gate(
    profile: DataProfile,
    restart_type: str,
    signal: str,
    cell_state: str,
    *,
    table: Mapping[GateKey, GateEntry] | None = None,
) -> GateEntry | None:
    """Look a cell up in ``table`` (the shipped ``RESTART_GATE_TABLE`` by default). Return its ``GateEntry`` or,
    on an absent key, ``None`` after incrementing ``withheld_lookups`` (fail-closed). Profile-level signals
    (``z_overhead``, ``footprint``) use restart_type ``"*"`` and cell_state ``"-"``."""
    global withheld_lookups
    tbl = RESTART_GATE_TABLE if table is None else table
    entry = tbl.get((profile, restart_type, signal, cell_state))
    if entry is None:
        withheld_lookups += 1
    return entry


__all__ = ["RESTART_GATE_TABLE", "DataProfile", "GateEntry", "GateKey", "gate"]
