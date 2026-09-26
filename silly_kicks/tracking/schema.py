"""Tracking output schema --- plain Python constants + dataclasses.

Mirrors silly_kicks.spadl.schema. See ADR-004 for the namespace charter
and docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md s 4.2.
"""

import dataclasses
from typing import Any

TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "period_id": "int64",
    "frame_id": "int64",
    # PERIOD-RELATIVE: seconds since the start of the period, resets to 0 each period (ADR-017)
    "time_seconds": "float64",
    "frame_rate": "float64",
    # player_id NULLABLE by necessity (ball row is NA BY CONSTRUCTION; numpy int64 cannot hold NA --
    # ADR-055/058). Stays Int64 (object in the kloppy family) and NOT `category`, because it is
    # MUTATED post-build (`_das.py` `.loc[ball_mask,"player_id"]="ball"` masked setitem; `_run_values`
    # Int64-reassign; the keeper/actor identity bridges), and `category` is not transparent to a new
    # category at setitem (the ADR-103 dynamic-column rule). team_id IS `category` (F1b / ADR-106
    # option A): STATIC/set-once + very low cardinality (~2/match) -> a large frame-memory win; its
    # only post-build touch is a masked `=None` (-> NaN, allowed on a categorical). `id_compat._decat`
    # unwraps the category to its underlying Int64/object at every comparison, so all id logic is
    # value-neutral. player_id -> category is reconsiderable once DAS is reimplemented natively (that
    # removes the `_das` `"ball"` sentinel blocker).
    "player_id": "Int64",
    "team_id": "category",
    "is_ball": "bool",
    "is_goalkeeper": "bool",
    # F1b (ADR-106): coordinate + kinematic storage is float32. STORAGE only -- every compute path
    # upcasts the coord slice to float64 at its kernel boundary (so the numba/ADR-076 bit-identity
    # contract is re-anchored on float64 inputs and the ONLY numeric drift is the deterministic
    # storage-rounding, ~1e-5 m, far below tracking-sensor precision). Halves the dominant frame
    # column memory. float32 holds NaN, so NA-on-ball-row (z) / positional-only (speed) are preserved.
    # `vx`/`vy`/`x_smoothed`/`y_smoothed` are added by preprocess (not declared here) and are cast to
    # float32 at the end of derive_velocities/smooth_frames.
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "speed": "float32",
    # ADR-103 F1a: only STATIC, set-once low-cardinality columns are `category` (int codes + tiny dict,
    # ~30-50x smaller than object over a match's rows) -- value-transparent for ==/.dropna()/pd.unique/
    # presence. DYNAMIC columns that are mutated post-build stay `object`, because `category` is NOT
    # transparent to setitem OR `.fillna(new_value)` (assigning a non-existing category raises): so
    # `team_attacking_direction` (orientation .loc + reflect swap), `speed_source` (velocity .loc
    # "derived"), and `visibility` (`_truthy_bool` fillna("")) stay object. Forcing them to category
    # would need full-domain CategoricalDtype + a category-preserving orient/velocity/reflect chain --
    # broad + permanently fragile for ~23% more memory; declined. `confidence` is DROPPED (all-null on
    # every provider). `value_counts` on a category enumerates zero-count categories, so the
    # source_provider site (tracking/utils.py:479) casts to object first. Provenance stays per-row (not
    # off-frame): source_provider is read per-row for xt_gk variant selection + keeper linking,
    # is_goalkeeper_source is per-keeper -- off-frame is F1b (retrain). See
    # feedback_category_dtype_only_for_static_columns.
    # Kinematics provenance: "native" | "derived" | SPEED_SOURCE_UNAVAILABLE (see below).
    "speed_source": "object",
    "ball_state": "category",
    "team_attacking_direction": "object",
    "visibility": "object",
    "source_provider": "category",
    "is_goalkeeper_source": "category",
}

KLOPPY_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "game_id": "object",
    "player_id": "object",
}
"""Kloppy gateway output: object ``game_id``/``player_id`` (kloppy domain types are strings). ``team_id``
inherits the base ``category`` (F1b / ADR-106): the underlying is object strings, so it is a
category-of-object here vs a category-of-Int64 on the numeric-id providers -- ``id_compat._decat`` unwraps
either. ``player_id`` stays object (dynamic post-build; see the base schema note)."""

SPORTEC_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Sportec native output: same shape as kloppy variant --- DFL TeamId / PersonId
are string identifiers."""

SKILLCORNER_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""SkillCorner native bronze->frame output: object identifiers (SkillCorner numeric
ids are stringified to match the SPADL ``player_id_native`` convention and the
kloppy-gateway oracle). Same shape as the kloppy variant."""

METRICA_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Metrica native bronze->frame output: object identifiers (``"Home"``/``"Away"`` team
labels + roster-mapped player ids). Same shape as the kloppy variant."""

GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS: dict[str, str] = TRACKING_FRAMES_COLUMNS
"""Gradient Sports (formerly PFF FC) native output: nullable Int64 identifiers
(matches GRADIENTSPORTS_SPADL_COLUMNS convention from PR-S18; allows NaN on
ball rows). game_id stays int64.

Now an ALIAS of the base rather than an override of it. It carried the only honest id declaration
in this file, and the base adopted it; re-stating the two columns would leave a redundant literal
that a future edit could silently diverge from. Kept as a NAME rather than deleted: it is exported
in ``silly_kicks.tracking.__all__``, and aliasing is already this file's idiom for "this provider's
schema equals another's" -- SPORTEC, SKILLCORNER and METRICA all alias KLOPPY the same way."""

SPEED_SOURCE_UNAVAILABLE = "unavailable"
"""``speed_source`` token: this frame SOURCE structurally cannot carry kinematics.

The frame builder declares, per row, that ``speed`` -- and the ``vx``/``vy`` that
:func:`silly_kicks.tracking.preprocess.derive_velocities` produces from the same
positional history -- can NEVER exist for these frames, because the source has no
per-player temporal sequence to differentiate. The canonical case is
:func:`silly_kicks.tracking.snapshot_to_tracking_frames`, which synthesises exactly ONE
frame per action from a per-event freeze-frame (StatsBomb 360): there is no second
sample to take a derivative against, at any sampling rate, ever.

This is deliberately DISTINCT from a NULL ``speed_source``, which means only "not
derived YET" -- the normal state of a metrica / skillcorner frame before
``smooth_frames`` + ``derive_velocities`` run. Without the distinction, a velocity
consumer cannot tell "this data structurally has no velocity" from "the caller forgot
to call ``derive_velocities()``", and the two demand opposite responses: the first is
an honest degrade, the second is a caller bug that must fail loud.

Velocity consumers read it accordingly --- :func:`silly_kicks.tracking.add_das` degrades
to NaN with ``das_source="unscoreable_frame"`` (warned) when every row is marked, and
still raises on unmarked frames that are merely missing ``vx``/``vy``.

THIRD-PARTY frame builders may and should set this deliberately::

    from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE

    frames["speed_source"] = SPEED_SOURCE_UNAVAILABLE  # freeze-frame source: no history

Scope note: the token asserts BOTH ``speed`` and ``vx``/``vy`` are structurally absent,
because both come from the same positional history. A hypothetical source carrying a
NATIVE instantaneous ``speed`` but no differentiable positional history could not be
expressed with this single token and would need its own marker.
"""

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
    "speed_source": frozenset({"native", "derived", SPEED_SOURCE_UNAVAILABLE}),
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

    n_gross_off_pitch: int = 0
    """Count of player/ball rows whose post-transform coords fell GROSS off-pitch beyond tolerance
    (CR 2026-06-30 S1). A correct centre-origin -> SPADL transform keeps bodies within the pitch
    except a tolerance (keepers behind the goal line; out-of-play ball); a non-zero count is a
    coordinate-transform / ingestion data-quality signal. Warned-and-counted, NEVER clamped."""

    n_implausible_gk_teams: int = 0
    """Count of (game_id, team_id) whose resolved is_goalkeeper count is implausible (>2 or 0)
    (CR 2026-06-30 S2). A reliable per-team GK is ~1 (2 with a sub); a higher count means whole-squad
    contamination (positional derivation on a small window) and 0 means a missing roster flag. Warned
    and counted; a machine-observable signal so squad-wide GK contamination cannot recur silently."""

    geometry_excluded: bool = False
    """True when the per-match SYSTEMATIC geometry rate-gate (spec 4.4) excludes this match: a
    catastrophic sign/origin coordinate break puts a SYSTEMATIC fraction of rows off-pitch (vs a
    handful of legitimately off-pitch bodies). Machine-observable so a broken match is DROPPED, not
    silently averaged into a calibration corpus (the per-row n_gross_off_pitch warn is invisible in a
    batch log). SkillCorner native builder only; default False (other providers run no native gate)."""

    geometry_reason: str = ""
    """Human-readable reason a match was ``geometry_excluded`` (which threshold(s) it breached with the
    measured rate); empty when not excluded. Printed to stderr by the loader on exclusion."""

    player_off_pitch_rate: float = 0.0
    """Fraction of PLAYER rows whose post-transform coords fall >3 m off the pitch (spec 4.4 gate
    input). ~0 on clean data; a systematic value signals a coordinate-transform break. Default 0.0."""

    ball_off_pitch_rate: float = 0.0
    """Fraction of BALL rows whose post-transform coords fall >10 m off the pitch (spec 4.4 gate
    input). ~0 on clean data (the largest real ball excursion measured is 9.0 m). Default 0.0."""

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


#: Regimes reported by ``validate_velocity_regime``. Exported for the same reason
#: ``DAS_SOURCE_VALUES`` is: a regime string that can RAISE by default is a consumer-facing
#: contract, and consumers must pin an enum to this set rather than to string literals.
VELOCITY_INFORMED = "velocity_informed"
POSITIONAL_ONLY = "positional_only"
#: SOME rows declare kinematics structurally unavailable and others do not. The case fail-loud
#: exists for: such a frame set cannot be scored coherently.
MIXED = "mixed"
#: No row declares kinematics unavailable, but ``vx``/``vy`` are absent -- the
#: "forgot ``derive_velocities()``" case. NOT a variant of MIXED: nothing is structurally missing
#: and the remedy is a single call, so labelling it MIXED would attach a message ("some rows can
#: carry velocity and others structurally cannot") that is FALSE for these frames -- and it is the
#: case a user is most likely to hit.
VELOCITY_MISSING = "velocity_missing"
#: A zero-row frame set is not a velocity problem. Reported rather than smuggled into another
#: regime, and it does NOT raise -- which FOLLOWS the siblings rather than departing from them:
#: measured on a schema-shaped zero-row frame, ``validate_time_base`` and ``validate_id_dtypes``
#: both return a diagnosis.
EMPTY = "empty"
VELOCITY_REGIME_VALUES: tuple[str, ...] = (
    VELOCITY_INFORMED,
    POSITIONAL_ONLY,
    MIXED,
    VELOCITY_MISSING,
    EMPTY,
)


@dataclasses.dataclass(frozen=True)
class VelocityRegimeDiagnosis:
    """Whether a frame set carries usable kinematics, reported before anything is computed.

    Third member of the ``validate_time_base`` / ``validate_id_dtypes`` family (ADR-017, ADR-019),
    and produced by ``silly_kicks.tracking.utils.validate_velocity_regime``.

    Seventeen of the registered ``add_*`` aggregators produce output that moves with velocity. Two
    read the availability marker themselves; ``add_ghost_gk`` refuses on it. The rest produce an
    HONEST, usable value whose INTERPRETATION changes -- pitch control at zero velocity is a
    well-defined positional model, not a fabrication. What a consumer cannot otherwise tell is that
    the value is positional-only, and that is a property of the whole frame set rather than of any
    row, which is why this is a diagnostic rather than a per-row provenance column that would carry
    a constant.

    Attributes:
        regime: one of :data:`VELOCITY_REGIME_VALUES`.
        speed_source_counts: ``speed_source`` value -> row count (empty if the column is absent).
        has_velocity_columns: whether both ``vx`` and ``vy`` are present.
        message: human-readable summary, and the text of the raise when one occurs.
    """

    regime: str
    speed_source_counts: dict[str, int]
    has_velocity_columns: bool
    message: str


#: Every action's ``visible_area`` polygon covers the pitch (share >= ``full_coverage_floor``): a
#: full-tracking or whole-pitch-observed set. Nothing to warn about.
FOV_REGIME_FULL = "full_coverage"
#: No action reaches full coverage: the provider ships a cropped field of view (a broadcast crop, a
#: partial 360 polygon) on every observed action.
FOV_REGIME_CROPPED = "fov_cropped"
#: No action carries an observed polygon at all -- every row is ``no_polygon`` / ``degenerate`` /
#: ``unlinked``. Nothing was published about what any camera saw.
FOV_REGIME_ABSENT = "absent"
#: Full-coverage actions coexist with cropped/absent ones. Scoring the set under one FOV assumption
#: is incoherent, so this is the fail-loud case (like ``VELOCITY_REGIME``'s ``MIXED``).
FOV_REGIME_MIXED = "mixed"
#: A zero-row visible_area table. Reported rather than smuggled into another regime, and it does NOT
#: raise -- FOLLOWING the siblings (``validate_velocity_regime`` / ``validate_time_base`` return a
#: diagnosis on empty input rather than departing from them).
FOV_REGIME_EMPTY = "empty"
FOV_REGIME_VALUES: tuple[str, ...] = (
    FOV_REGIME_FULL,
    FOV_REGIME_CROPPED,
    FOV_REGIME_ABSENT,
    FOV_REGIME_MIXED,
    FOV_REGIME_EMPTY,
)


@dataclasses.dataclass(frozen=True)
class FovDiagnosis:
    """Whether a frame set's per-action visible_area is full / cropped / absent, before scoring.

    Fourth member of the ``validate_time_base`` / ``validate_velocity_regime`` / ``validate_id_dtypes``
    family (ADR-017, ADR-019), and produced by ``silly_kicks.tracking.utils.validate_fov``. A
    freeze-frame provider does not see the whole pitch, and whether it ships a full, cropped or absent
    field of view is a property of the WHOLE action set -- like ``VelocityRegimeDiagnosis`` -- so it is
    a diagnostic rather than a per-row column that would carry a constant. The per-action observed
    fractions that DO vary row-to-row are the observability companions (Task 4), not this.

    Attributes:
        regime: one of :data:`FOV_REGIME_VALUES`.
        observed_pitch_fraction: ``action_id`` -> observed pitch fraction, only for ``observed``
            actions (the ``visible_area_source`` tokens live in :data:`VISIBLE_AREA_SOURCE_VALUES`).
        source_counts: ``visible_area_source`` token -> action count.
        n_actions: number of actions considered.
        message: human-readable summary, and the text of the raise when one occurs.
    """

    regime: str
    observed_pitch_fraction: dict[Any, float]
    source_counts: dict[str, int]
    n_actions: int
    message: str


@dataclasses.dataclass(frozen=True)
class GkClampDiagnosis:
    """Whether a provider clamps the goalkeeper's tracked position to a hard max distance from goal.

    A data-quality diagnostic, produced by ``silly_kicks.tracking.utils.validate_gk_position_clamp``.
    Some broadcast-tracking providers constrain the keeper to a fixed "goalkeeper zone": measured,
    Gradient Sports pins every keeper at exactly 27.5 m from its own goal and never beyond, so any
    GK-depth / sweeper / ghost-GK analysis on that provider is invalid past the ceiling. The signature
    is a hard ceiling on the keeper's goal-relative x with an anomalous PILEUP at that ceiling (a
    natural keeper has ~0 mass at its max). Whether a provider clamps is a property of the WHOLE frame
    set -- like ``VelocityRegimeDiagnosis`` / ``FovDiagnosis`` -- so it is a diagnostic rather than a
    per-row column that would carry a constant. Oriented via ``resolve_defended_goals`` (ADR-055;
    never team identity).

    Attributes:
        clamped: True iff any examined ``(game_id, team_id)`` keeper shows the clamp signature.
        clamped_units: the canonical ``(game_id, team_id)`` keys whose keeper is clamped.
        ceiling_by_unit: canonical ``(game_id, team_id)`` -> detected ceiling (m from own goal).
        pileup_by_unit: canonical ``(game_id, team_id)`` -> fraction of keeper frames at the ceiling.
        n_units: number of ``(game_id, team_id)`` keeper units examined (>= ``min_keeper_frames``).
        message: human-readable summary, and the text of the warning when one is emitted.
    """

    clamped: bool
    clamped_units: tuple[tuple[str, str], ...]
    ceiling_by_unit: dict[tuple[str, str], float]
    pileup_by_unit: dict[tuple[str, str], float]
    n_units: int
    message: str


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
