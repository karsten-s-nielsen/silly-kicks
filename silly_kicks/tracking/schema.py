"""Tracking output schema --- plain Python constants + dataclasses.

Mirrors silly_kicks.spadl.schema. See ADR-004 for the namespace charter
and docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md s 4.2.
"""

import dataclasses

TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "period_id": "int64",
    "frame_id": "int64",
    # PERIOD-RELATIVE: seconds since the start of the period, resets to 0 each period (ADR-017)
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
    "is_goalkeeper_source": "object",
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

GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "player_id": "Int64",
    "team_id": "Int64",
}
"""Gradient Sports (formerly PFF FC) native output: nullable Int64 identifiers
(matches GRADIENTSPORTS_SPADL_COLUMNS convention from PR-S18; allows NaN on
ball rows). game_id stays int64."""

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
    "source_provider": frozenset({"gradientsports", "sportec", "metrica", "skillcorner", "snapshot"}),
    "is_goalkeeper_source": frozenset({"native", "derived"}),
}


@dataclasses.dataclass(frozen=True)
class TrackingConversionReport:
    """Audit trail for tracking convert_to_frames(). Frame-shaped audit.

    Attributes:
        provider: Provider name, lowercase ("gradientsports" | "sportec" | "metrica" | "skillcorner").
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
    n_teams_gk_derived: int = 0
    """Count of (game_id, team_id) pairs where the positional fallback
    fired (kloppy's native is_goalkeeper count was != 1). 0 means kloppy's
    native flagging was reliable across the whole input. ADR-007."""

    derived_gk_picks: dict[tuple[str, str], list[str]] = dataclasses.field(default_factory=dict)
    """For each (game_id, team_id) where the positional fallback fired,
    the list of player_ids the algorithm flagged as GK. Single-element
    list in normal matches; 2+ in substitution scenarios. Empty dict when
    no fallback fired. Useful for downstream auditing — consumers can
    spot-check 'for matches where source=derived, who did we pick?'.
    ADR-007."""

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

    per_period_link_rate: dict[int, float] = dataclasses.field(default_factory=dict)
    """period_id -> linked / actions-in-that-period. Computed from the internal
    per-period merge (NOT the returned pointers, which drop period_id), so a
    catastrophically-unlinked period is never laundered behind a healthy one.
    Empty for an empty-actions call. See ADR-017."""

    @property
    def link_rate(self) -> float:
        return self.n_actions_linked / max(self.n_actions_in, 1)

    def sync_scores(self, links, *, high_quality_threshold: float = 0.85):
        """Per-action ``sync_score`` DataFrame for the supplied link batch.

        The LinkReport summary holds counts; the link rows themselves are needed
        to compute per-action aggregations -- pass them in.

        Examples
        --------
        >>> # See tests/test_sync_score.py::test_link_report_sync_scores_method
        """
        from .utils import sync_score  # local import to avoid utils -> schema cycle

        return sync_score(links, high_quality_threshold=high_quality_threshold)


@dataclasses.dataclass(frozen=True)
class TimeBaseDiagnosis:
    """Per-period action-vs-frame time-range diagnosis (time-base mismatch hypothesis).

    Produced by ``silly_kicks.tracking.utils._diagnose_time_base`` and surfaced by
    ``validate_time_base`` and the ``link_actions_to_frames`` low-coverage guard.
    A *cause hypothesis* for low link coverage, distinct from the *symptom*
    (low link rate). See ADR-017.

    Attributes:
        per_period_action_range: period_id -> (min, max) action time_seconds.
        per_period_frame_range: period_id -> (min, max) frame time_seconds
            (absent for a period that has actions but no frames).
        per_period_overlap_fraction: period_id -> fraction of the action span
            covered by the frame span (1.0 = frames fully span; 0.0 = disjoint).
        suspected_mismatch_periods: periods with overlap < MISMATCH_OVERLAP_FLOOR,
            ordered worst-first (lowest overlap first).
        message: human-readable summary enumerating suspected periods worst-first.
    """

    per_period_action_range: dict[int, tuple[float, float]]
    per_period_frame_range: dict[int, tuple[float, float]]
    per_period_overlap_fraction: dict[int, float]
    suspected_mismatch_periods: tuple[int, ...]
    message: str

    @property
    def has_suspected_mismatch(self) -> bool:
        return len(self.suspected_mismatch_periods) > 0


@dataclasses.dataclass(frozen=True)
class IdDtypeDiagnosis:
    """Action-vs-frame id-dtype compatibility diagnosis (ADR-019).

    Produced by ``silly_kicks.tracking.utils._diagnose_id_dtypes`` and surfaced by
    ``validate_id_dtypes``. The tracking-feature seams coerce id dtypes transparently;
    this is the opt-in loud guard for a dtype-sensitive consumer (e.g. the lakehouse).

    Attributes:
        per_column: id col -> (action_dtype_str, frame_dtype_str).
        coercion_required_columns: cols whose action/frame numpy kinds differ
            (would silently mis-compare / raise on merge without coercion).
        home_team_id_dtype: dtype/kind of the scalar arg, if supplied (else None).
        home_team_id_requires_coercion: scalar kind vs frame team_id kind differ.
        message: human-readable summary.
    """

    per_column: dict[str, tuple[str, str]]
    coercion_required_columns: tuple[str, ...]
    home_team_id_dtype: str | None
    home_team_id_requires_coercion: bool
    message: str

    @property
    def has_mismatch(self) -> bool:
        return len(self.coercion_required_columns) > 0 or self.home_team_id_requires_coercion
