"""Aggregate provenance QA for add_restart_coordinates output (mirrors XtGkReport / LinkReport)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RestartCoordinateReport:
    """Counts per origin/destination source for a restart-coordinate-enriched frame. A convenience
    over a downstream ``GROUP BY start_coord_source`` (not load-bearing). By construction the counts
    equal the columns' ``value_counts``."""

    n_rows: int
    start_source_counts: dict[str, int]
    end_source_counts: dict[str, int]
    n_tripwire_reversions: int  # rows the tripwire reverted (start_coord_source == "tripwire_reverted")

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> RestartCoordinateReport:
        """Build the report from an ``add_restart_coordinates`` / ``resolve_restart_geometry`` frame.
        ``n_tripwire_reversions`` preserves the QA distinction between never-resolvable
        (``unresolved``) and resolved-then-reverted (``tripwire_reverted``) rows (spec section 6).

        Examples
        --------
        Summarize restart-coordinate provenance from an enriched frame::

            rep = RestartCoordinateReport.from_frame(enriched)
            rep.start_source_counts["restart_prior"]  # e.g. 12
        """
        ssc = {str(k): int(v) for k, v in df["start_coord_source"].value_counts(dropna=True).items()}
        esc = {str(k): int(v) for k, v in df["end_coord_source"].value_counts(dropna=True).items()}
        return cls(
            n_rows=len(df),
            start_source_counts=ssc,
            end_source_counts=esc,
            n_tripwire_reversions=int(ssc.get("tripwire_reverted", 0)),
        )
