"""Tracking output schema --- plain Python constants + dataclasses.

Mirrors silly_kicks.spadl.schema. See ADR-004 for the namespace charter
and docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md s 4.2.
"""

import dataclasses

TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "period_id": "int64",
    "frame_id": "int64",
    "time_seconds": "float64",
    "frame_rate": "float64",
    "player_id": "int64",
    "team_id": "int64",
    "is_ball": "bool",
    "is_goalkeeper": "bool",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "speed": "float64",
    "speed_source": "object",
    "ball_state": "object",
    "team_attacking_direction": "object",
    "confidence": "object",
    "visibility": "object",
    "source_provider": "object",
}

KLOPPY_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "game_id": "object",
    "player_id": "object",
    "team_id": "object",
}
"""Kloppy gateway output: object identifiers (kloppy domain types are strings)."""

SPORTEC_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Sportec native output: same shape as kloppy variant --- DFL TeamId / PersonId
are string identifiers."""

PFF_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "player_id": "Int64",
    "team_id": "Int64",
}
"""PFF native output: nullable Int64 identifiers (matches PFF_SPADL_COLUMNS
convention from PR-S18; allows NaN on ball rows). game_id stays int64."""

TRACKING_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id": (1, 5),
    "time_seconds": (0, float("inf")),
    "frame_rate": (1, 60),
    "frame_id": (0, float("inf")),
    "x": (0, 105.0),
    "y": (0, 68.0),
    "z": (0, 10.0),
    "speed": (0, 50.0),
}

TRACKING_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "ball_state": frozenset({"alive", "dead"}),
    "team_attacking_direction": frozenset({"ltr", "rtl"}),
    "speed_source": frozenset({"native", "derived"}),
    "source_provider": frozenset({"pff", "sportec", "metrica", "skillcorner"}),
}


@dataclasses.dataclass(frozen=True)
class TrackingConversionReport:
    """Audit trail for tracking convert_to_frames(). Frame-shaped audit.

    Attributes:
        provider: Provider name, lowercase ("pff" | "sportec" | "metrica" | "skillcorner").
        total_input_frames: Frames in the raw input DataFrame.
        total_output_rows: Long-form expanded row count (frames x players + ball rows).
        n_periods: Number of distinct period_ids.
        frame_coverage_per_period: period_id -> fraction of expected frames present
            (1.0 = no missing frames, given inferred frame_rate).
        ball_out_seconds_per_period: period_id -> total seconds with ball_state="dead".
        nan_rate_per_column: column name -> fraction of NaN rows in output.
        derived_speed_rows: Rows where speed_source="derived".
        unrecognized_player_ids: IDs in input not resolvable via roster.

    Examples
    --------
    Inspect the audit after converting a Sportec match::

        from silly_kicks.tracking import sportec
        frames, report = sportec.convert_to_frames(
            raw, home_team_id="DFL-CLU-A", home_team_start_left=True,
        )
        if report.has_unrecognized:
            print("Unrecognized player IDs:", report.unrecognized_player_ids)
    """

    provider: str
    total_input_frames: int
    total_output_rows: int
    n_periods: int
    frame_coverage_per_period: dict[int, float]
    ball_out_seconds_per_period: dict[int, float]
    nan_rate_per_column: dict[str, float]
    derived_speed_rows: int
    unrecognized_player_ids: set

    @property
    def has_unrecognized(self) -> bool:
        return len(self.unrecognized_player_ids) > 0


@dataclasses.dataclass(frozen=True)
class LinkReport:
    """Audit trail for link_actions_to_frames().

    Attributes:
        n_actions_in: Input action count.
        n_actions_linked: Actions with a frame_id (within tolerance).
        n_actions_unlinked: Actions with NaN frame_id (no frame within tolerance).
        n_actions_multi_candidate: Actions with >1 candidate frame within tolerance
            (closest one returned).
        per_provider_link_rate: source_provider -> linked / in. Single-provider
            in practice, multi-provider supported for forward-compat.
        max_time_offset_seconds: max |Dt| among linked rows; 0.0 if none linked.
        tolerance_seconds: Echoes the call argument.

    Examples
    --------
    Use the audit to validate cross-provider link quality::

        from silly_kicks.tracking.utils import link_actions_to_frames
        pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
        assert report.link_rate >= 0.95
    """

    n_actions_in: int
    n_actions_linked: int
    n_actions_unlinked: int
    n_actions_multi_candidate: int
    per_provider_link_rate: dict[str, float]
    max_time_offset_seconds: float
    tolerance_seconds: float

    @property
    def link_rate(self) -> float:
        return self.n_actions_linked / max(self.n_actions_in, 1)
