# ADR-006: Direction-of-play handling per converter (input-convention dispatch + validator)

| Field | Value |
|---|---|
| **Date** | 2026-05-02 |
| **Status** | Accepted (silly-kicks 3.0.0) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks's SPADL canonical convention -- inherited from upstream socceraction
-- is *"all teams attack left-to-right"*: every team's actions oriented at
high-x in their own frame. The convention is documented in the Sportec converter
docstring, the kloppy gateway comments, and the PFF converter docstring.

The pre-3.0.0 implementation was internally inconsistent. Two architecturally
identical mirror operations were applied sequentially during VAEP fitting:

1. ``silly_kicks.spadl.base._fix_direction_of_play(actions, home_team_id)`` --
   private, called inside every native converter.
2. ``silly_kicks.vaep.features.play_left_to_right(gamestates, home_team_id)``
   -- public, called inside ``vaep.base.VAEP.compute_features``.

Both flipped away-team rows by ``(105 - x, 68 - y)``. The semantic intent --
*what input convention each call assumes* -- was undocumented and incoherent
across providers. The two mirrors cancelled correctly for *one* family of
providers at a time, but no provider had **both** a correct SPADL output AND
correct VAEP gamestates simultaneously:

| Provider input convention | Converter output | VAEP gamestates |
|---|---|---|
| Possession-perspective (StatsBomb / Wyscout) | ✗ Away mirrored to wrong end | ✓ Correct (double-mirror canceled by accident) |
| Absolute-frame (Sportec / Metrica / kloppy / Opta) | ✓ Correct LTR | ✗ Away mirrored to wrong end by the second mirror |
| Per-period absolute (PFF) | ✓ Correct LTR (per-period path independent) | ✗ Away mirrored to wrong end |

The bug had been present in the v0.1.0 fork (verified `git show 0b29178`):
StatsBomb / Wyscout / Opta converters and `vaep/base.py` all called the mirror
functions verbatim from upstream socceraction. Whether any specific consumer
artifact was actually wrong depended on which converter path that consumer's
data went through.

The bug survived 9 minor releases because no test asserted the cross-layer
geometric invariant -- contract tests verified each layer's per-helper
behaviour, but nothing checked that the composition produced physically-correct
output (shots cluster near opponent goal, GK actions at defended goal, etc.).

## Decision

**SPADL canonical convention is "all teams attack left-to-right".** Every
silly-kicks SPADL converter outputs this convention directly; downstream
framework helpers (VAEP, xT, polar features) consume it verbatim and never
re-mirror.

### 1. Single canonical normalizer

A new module `silly_kicks/spadl/orientation.py` introduces:

- `InputConvention` enum: `POSSESSION_PERSPECTIVE`, `ABSOLUTE_FRAME_HOME_RIGHT`,
  `PER_PERIOD_ABSOLUTE`.
- `to_spadl_ltr(actions, *, input_convention, home_team_id, home_attacks_right_per_period=None)`
  -- single canonical normalizer; each converter calls it exactly once,
  declaring its input convention.

Implementation:
- `POSSESSION_PERSPECTIVE` -> no-op (data already LTR per-team).
- `ABSOLUTE_FRAME_HOME_RIGHT` -> mirror away-team rows.
- `PER_PERIOD_ABSOLUTE` -> consult `home_attacks_right_per_period[period]`,
  mirror per (team, period). Implementation re-uses
  `silly_kicks.tracking._direction.home_attacks_right_per_period`.

The dispatcher's no-op branch for possession-perspective providers is kept
explicit (rather than removing the call) so every converter has the same
single-line audit hook for tests/invariants.

### 2. Per-converter input convention assignment

| Converter | `input_convention` | Validator-detected |
|---|---|---|
| `spadl/statsbomb.py` | `POSSESSION_PERSPECTIVE` | yes (real fixtures) |
| `spadl/wyscout.py` | `POSSESSION_PERSPECTIVE` | yes (synthetic 2-team) |
| `spadl/opta.py` | `ABSOLUTE_FRAME_HOME_RIGHT` (NO per-period switch) | yes (synthetic absolute-no-switch); docstring contract added |
| `spadl/sportec.py` | `ABSOLUTE_FRAME_HOME_RIGHT` | yes (IDSSE bronze) |
| `spadl/metrica.py` | `ABSOLUTE_FRAME_HOME_RIGHT` | yes (Metrica bronze) |
| `spadl/kloppy.py` | `ABSOLUTE_FRAME_HOME_RIGHT` (post kloppy `Orientation.HOME_AWAY` transform) | n/a (kloppy already normalised) |
| `spadl/pff.py` | `PER_PERIOD_ABSOLUTE` (via `homeTeamStartLeft`) | n/a (per-period coords need separate validator pattern -- TF-22 follow-up) |

### 3. Input-convention auto-validator

`detect_input_convention(events, *, match_col, x_max, ...)` and
`validate_input_convention(events, declared, *, on_mismatch)` in the same
`orientation.py` module. The detector reads per-(match, team, period) shot
distribution and returns one of the three conventions or `None` (ambiguous).
Tiered confidence: `high` requires ≥10 shots/group, `medium` 5-9, `<5`
defers to declared.

The validator surfaces declared-vs-detected mismatches via:
- `on_mismatch="warn"` (default) -- production pipelines see warnings in logs.
- `on_mismatch="raise"` -- promotes to ValueError.
- `None` (default) -- resolves to `"raise"` under
  `SILLY_KICKS_ASSERT_INVARIANTS=1` env-var (CI mode), else `"warn"`.

**Pure validation, never auto-routing.** The declared `input_convention` is
the load-bearing contract; the validator never overrides it. Rationale: single
source of truth, loud failures over silent fixes, no `auto_route` opt-in
escape hatch (YAGNI).

### 4. VAEP `play_left_to_right` removed from internal call sites

`vaep/base.py:166` (the second mirror) is removed. Converter output is already
canonical SPADL LTR, so the second mirror inverted away-team rows for
absolute-frame providers and accidentally compensated for the converter bug
in possession-perspective providers. Same for atomic VAEP.

The public functions ``silly_kicks.spadl.play_left_to_right``,
``silly_kicks.vaep.features.play_left_to_right``,
``silly_kicks.atomic.spadl.play_left_to_right``,
``silly_kicks.atomic.vaep.features.play_left_to_right`` are retained as public
boundary helpers (absolute-frame -> SPADL LTR) for callers who load actions
from outside silly-kicks. Internal call sites no longer use them.

### 5. Tracking adapters: explicit `output_convention` opt-in (no default flip)

Tracking adapters (`tracking/sportec.py`, `tracking/pff.py`, `tracking/kloppy.py`)
gain an `output_convention: Literal["absolute_frame", "ltr"] | None = None`
kwarg. `None` (legacy unspecified) emits a `DeprecationWarning` recommending
the caller pick explicitly, then falls back to `"absolute_frame"` (the
historical default). Default is **not** flipped to LTR.

Rationale: silently flipping the default would be a breaking change for any
current consumer using absolute-frame tracking output (visualizations,
between-team distance, opponent-relative-speed features). The explicit kwarg +
warning gives callers a forcing function to be explicit without breaking
existing code.

VAEP `compute_features` accepts `frames_convention="absolute_frame"` (default)
and applies the LTR normalisation internally before frame-aware xfns run, so
callers can keep tracking adapters in their natural absolute-frame default.

### 6. Test-layer-hierarchy promotion

The bug existed because silly-kicks tests had only one layer (contract). PR-S22
formalises three layers:

- **Contract tests** (existing): per-helper input/output shape, dispatch logic.
- **Physical-invariant tests** (NEW, `tests/invariants/`): properties any
  correctly-converted football data must satisfy regardless of provider --
  shots cluster near opponent goal, GK actions cluster at defended goal, xT
  goal-monotonic, VAEP shot dist < 50m. Parametrised across all providers.
- **End-to-end pipeline tests** (NEW): full converter -> features -> model
  path on real fixtures, asserting numerical sanity not just non-error
  completion.

Plus a `_finalize_output` debug-mode assertion gated on
`SILLY_KICKS_ASSERT_INVARIANTS=1` -- the strongest in-converter guard.

## Consequences

### Positive

- Bug fixed: every silly-kicks SPADL converter produces canonical SPADL LTR
  directly; downstream consumers no longer need to know the provider's input
  convention.
- Loader regressions are surfaced automatically via the validator (warning
  default, raise in CI).
- `tests/invariants/` directory establishes a permanent third test layer that
  would have caught the bug on day one of v0.1.0.
- The architectural intent (one mirror per row, located at the converter
  boundary) is now documented and enforced by tests.

### Negative / Breaking

- **silly-kicks 3.0.0 is a breaking correctness change.** Every consumer
  artifact derived from native StatsBomb / Wyscout / Opta SPADL must be
  re-derived. Trained VAEP / HybridVAEP / xT models trained on absolute-frame
  providers (Sportec / Metrica / kloppy / PFF / Opta) must be re-trained.
- Tracking adapters now emit a `DeprecationWarning` for callers that don't
  pass `output_convention=` explicitly. Behaviour preserved; warning is the
  forcing function.

### Migration

Per-consumer migration is the consumer's responsibility. The new validator
(at `SILLY_KICKS_ASSERT_INVARIANTS=1`) catches input-convention mismatches
automatically; consumers should set this env-var in CI. The CHANGELOG entry
enumerates the categorical impact rather than specific consumer artifacts.

## Alternatives considered

- **(a) Silently fix only StatsBomb/Wyscout** -- rejected: leaves Sportec /
  Metrica / kloppy / PFF VAEP broken, and the dual-mirror inversion remains
  in the codebase as a hidden trap.
- **(b) Keep dual mirror, document it** -- rejected: the inversion is per-provider,
  unfixable without breaking each provider in turn, and the documentation
  burden grows with every new provider.
- **(c) Delete `_fix_direction_of_play` and `play_left_to_right` entirely**
  -- rejected: public API breakage with no replacement for callers who legitimately
  have absolute-frame data they didn't get through silly-kicks (e.g. raw
  socceraction output, custom loaders).
- **(d) Make detector source of truth, declared convention fallback only**
  -- rejected: single source of truth + loud failures > self-healing silent
  corrections; YAGNI on auto-routing escape hatch (no current consumer needs
  it; if needed in 2028 it's a one-line addition then).
- **(e) Flip tracking adapter default to LTR** -- rejected per lakehouse-session
  feedback: silent breaking change for any current consumer using absolute-frame
  tracking output. Explicit opt-in via `output_convention="ltr"` +
  `DeprecationWarning` recommending callers be explicit is preferred.

## References

- PR-S22 — silly-kicks 3.0.0 implementation
- Pre-fix probe scripts (local-only, not committed):
  `scripts/probe_direction_of_play.py`, `scripts/probe_vaep_dual_mirror.py`,
  `scripts/probe_opta_convention.py`, `scripts/probe_convention_detector.py`.
- Lakehouse session second-opinion review folded into final design (notably
  the test-layer-hierarchy promotion and the no-default-flip tracking decision).

## Erratum (silly-kicks 3.0.1, 2026-05-02)

Two rows of the per-converter input convention assignment table (§ 2)
were incorrect in 3.0.0:

| Converter | declared (3.0.0 — incorrect) | declared (3.0.1 — corrected) |
|---|---|---|
| `spadl/sportec.py` | `ABSOLUTE_FRAME_HOME_RIGHT` | `PER_PERIOD_ABSOLUTE` |
| `spadl/metrica.py` | `ABSOLUTE_FRAME_HOME_RIGHT` | `PER_PERIOD_ABSOLUTE` |

Native Sportec and Metrica bronze events ship per-period-absolute (teams
switch ends after halftime — empirically verified against lakehouse
production fixtures via the SK3-MIG migration session). The kloppy-gateway
path (`spadl/kloppy.py`) remains `ABSOLUTE_FRAME_HOME_RIGHT` — kloppy
normalises upstream via `Orientation.HOME_AWAY`.

The 3.0.1 fix mirrors the existing PFF events-side and tracking-side
Sportec API exactly: callers pass `home_team_start_left: bool` (or the
escape-hatch `home_attacks_right_per_period` mapping), the converter
derives flips and dispatches to `to_spadl_ltr` with `PER_PERIOD_ABSOLUTE`.
See CHANGELOG 3.0.1 for the migration snippet.

The detector heuristic (`detect_input_convention`) was simultaneously
hardened (TF-22): when no team has reliable shots in ≥ 2 distinct
periods, the detector returns `convention=None, confidence="low"` rather
than false-positiving `ABSOLUTE_FRAME_HOME_RIGHT` on sparse-shot
per-period-absolute matches. Validator re-enabled at sportec, metrica,
and pff converter call sites with `declared=PER_PERIOD_ABSOLUTE`.

## Lessons learned (silly-kicks 3.0.1)

PR-S22 reported "1707 passed, 12 skipped, 0 failed" with strict-mode
invariants enabled, yet the Sportec + Metrica per-period bug shipped.
Root cause: `tests/datasets/idsse/sample_match.parquet` (2 shots total)
and `tests/datasets/metrica/sample_match.parquet` (period 1 only) had
insufficient per-period shot density to physically exercise the
per-period orientation invariant. The invariant test layer was present
but blind on these two providers.

Lesson: a physical-invariant test is only load-bearing if the fixture
density allows the invariant to actually be evaluated. Future
provider-coverage PRs should verify the invariant suite's fixture
density on a per-provider basis. PR-S23 adds explicit per-period
fixtures (`tests/datasets/{idsse,metrica}/per_period_match.parquet`)
with documented shot density requirements (≥ 5 shots per (team,
period) group is the validator's medium-confidence threshold; the new
fixtures meet this).
