"""Enforce: every public function / class / method docstring includes an Examples section.

Closes D-8 (PR-S13). Backstops the discipline by failing CI when a future PR adds a public
symbol without an Examples section.

WHY THE REGISTRY IS PINNED TO REALITY (ADR-043, 4.53.0)
-------------------------------------------------------
``_PUBLIC_MODULE_FILES`` used to be a hand-maintained list with **nothing pinning it to the
actual public surface**, so a newly-added public module was silently MISSED rather than
caught -- the file simply never appeared in the parametrization. That is the same
incomplete-by-heuristic failure this release eliminated in
``tests/tracking/test_id_compat_lint.py`` (a NAME heuristic over one package's glob, deleted)
and in the ``gkdv`` import allowlist (a non-recursive glob).

So the surface is now DERIVED (:func:`_discover_public_modules`) and the registry is pinned
to it by meta-assertions, exactly as ``tests/invariants/test_public_id_scalar_registry.py``
pins its own. A module that is neither enforced nor explicitly excluded fails CI.

WHAT "PUBLIC MODULE" MEANS HERE
-------------------------------
A module file that contributes a symbol to the importable surface of ``silly_kicks``. Derived
mechanically as the UNION of two rules, because the package reaches users both ways:

**P1 -- re-export.** The module DEFINES (by ``obj.__module__``) at least one symbol listed in
some public package's ``__all__``. This is what makes private-BY-NAME modules public in
practice: users import ``tracking.compute_ghost_gk``, and ``_ghost_gk.py`` is where the
contract actually lives. The pre-existing registry already relied on this (it lists
``xthreat/_model.py``, ``calibration/_xt.py``, ``tracking/_warnings.py``) while its own
comment claimed to "exclude underscore-prefixed modules" -- the comment was stale, which is
precisely the kind of drift a derivation cannot suffer from.

**P2 -- dotted path.** The module has no underscore-prefixed path component and defines at
least one non-underscore top-level symbol, i.e. it is reachable as
``silly_kicks.spadl.statsbomb.convert_to_actions``. These modules are public without any
``__all__`` re-export, so P1 alone would miss them.

Deliberately NOT public, and therefore absent from both buckets: a module that is
underscore-named AND re-exports nothing (``silly_kicks/_group_metrics.py``, whose own
docstring declares it private with no downstream consumer and states that promoting it to
this registry is a deliberate, requested step). Its symbols carry Examples anyway -- good
documentation is not conditional on a gate -- but the gate does not claim it as public API.

WHAT COUNTS AS AN EXAMPLE (the ``+SKIP`` and import-only tightenings, 4.53.0)
-----------------------------------------------------------------------------
The original check accepted ANY ``>>>`` line -- and, separately, a bare ``Examples`` header
with nothing under it. So ``>>> f(x)  # doctest: +SKIP`` satisfied the gate while
demonstrating nothing a reader can run or a maintainer can check: box-ticking of exactly the
kind the registry rewrite above eliminated one level up. 16 of the 284 enforced symbols
passed that way.

The ``+SKIP`` rule alone was still walk-past-able, because it judged only whether SOME line
was unskipped, never whether that line SHOWED anything. Put ``>>> from x import f`` on the
first line and every line demonstrating the call could stay behind ``+SKIP``; a
``>>> # see tests/... for a runnable example`` comment did the same job. An import is setup
and a pointer to an example is not an example, so neither now makes a section real on its
own (:func:`_demonstrates_something`, judged by PARSING the reconstructed statement, not by
matching text). That camouflage hid **74** further symbols -- more than four times the 16 the
``+SKIP`` rule caught head-on -- which is the measure of how much a rule that inspects only
the FORM of an example misses about its CONTENT. An import FOLLOWED by a real call is
untouched: the import was never the problem.

:func:`_has_real_example` now requires a REAL example -- see ``_REAL_EXAMPLE_RULE``, which the
failure message quotes verbatim so whoever trips it is told what to WRITE. Two forms qualify:
a runnable doctest, or the indented illustrative literal block that is this package's
canonical style (most entry points need a real ``actions`` frame no docstring can conjure, so
non-doctest examples are the norm and must keep passing). The rule targets EMPTY gestures, not
non-doctest ones.

The illustrative arm is SCOPED to the Examples section, which is load-bearing: a NumPy
``Parameters`` block is indented too, so an unscoped version rescues essentially every
docstring in the repo. That mistake silently shrank the offender set from 16 to 13 while this
rule was being drafted, and ``test_skip_only_rule_is_scoped_to_the_examples_section`` exists
because of it.

THE BUCKETS
-----------
``_PUBLIC_MODULE_FILES``
    ENFORCED. Every public module, and every public symbol in it must carry a real Examples
    section unless that SYMBOL is individually excused below.
``_EXAMPLES_DEBT``
    Individual undocumented public SYMBOLS, each with a written note on what it is. Keyed
    ``"<file>::<qualified_name>"``. Self-burning-down: a meta-assertion requires every entry
    to still BE undocumented, so the moment a symbol gets an example CI goes red and tells
    you to delete its entry. It can shrink, never silently grow -- a new symbol is enforced
    by default, and a new module is in neither bucket and fails.

Currently 354 of 579 public symbols across 118 modules are enforced, with 225 tracked as
debt. A ``@overload`` stub is not among either: its body is ``...``, so demanding an example
of one demands something that cannot exist -- and an entry that can never burn down defeats
the whole property above. :func:`_is_overload_stub` skips them, keyed on the decorator each
definition carries rather than on its name, so the IMPLEMENTATION consumers actually call
stays judged.

WHY THE DEBT IS PER-SYMBOL (this release)
-----------------------------------------
It used to be per-MODULE, and that was a net LOSS of coverage hiding inside a change that
tightened the gate. When the ``+SKIP`` tightening demoted 12 symbols, four whole modules left
enforcement with them -- so ``calibration/_xt.py``'s one documented symbol, and every
documented symbol in the other three, silently lost its guard as collateral. The same
arithmetic applied to every pre-existing module entry: ``tracking/features.py`` was excused
for 5 undocumented symbols and took its other 79 DOCUMENTED ones out of enforcement with
them, free to regress unnoticed.

An exemption should cost exactly what it excuses. So the unit is now the symbol: a module
with one gap keeps enforcing everything else in it. The four modules whose ``+SKIP`` filler
started this are documented rather than bucketed, and the bucket they briefly occupied is now
expressed as the individual symbols that were always the real debt: 154 keys across 305
public symbols in those 35 modules, which pulls the other 150 -- already documented, and
until now unguarded -- back under the gate.

A third, much smaller registry ``_EXTRA_COVERAGE`` records enforced modules the derivation
does not classify as public (over-coverage). It exists so that
``test_registered_modules_are_still_public`` can be strict: that assertion is what turns a
partially-blind derivation -- a package that failed to import, taking all of its re-exported
underscore modules out of the surface with it -- into a red build instead of quiet
under-enforcement.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pathlib
import pkgutil
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "silly_kicks"
_ROOT_MODULE = "silly_kicks"

# Public-API modules whose Examples coverage is ENFORCED. Pinned to the derived surface by
# `test_derived_surface_is_fully_accounted_for` -- this list can no longer silently rot.
_PUBLIC_MODULE_FILES = (
    "silly_kicks/reflection.py",
    "silly_kicks/tracking/_visibility.py",
    "silly_kicks/reporting.py",
    "silly_kicks/feature_glossary.py",
    "silly_kicks/spadl/utils.py",
    "silly_kicks/spadl/orientation.py",
    "silly_kicks/spadl/schema.py",
    "silly_kicks/spadl/statsbomb.py",
    "silly_kicks/spadl/opta.py",
    "silly_kicks/spadl/wyscout.py",
    "silly_kicks/spadl/sportec.py",
    "silly_kicks/spadl/metrica.py",
    "silly_kicks/spadl/kloppy.py",
    "silly_kicks/spadl/gradientsports.py",
    "silly_kicks/spadl/skillcorner.py",
    "silly_kicks/atomic/spadl/utils.py",
    "silly_kicks/atomic/spadl/base.py",
    "silly_kicks/vaep/base.py",
    "silly_kicks/vaep/hybrid.py",
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat/_model.py",
    "silly_kicks/xthreat/_params.py",
    "silly_kicks/xthreat/_transitions.py",
    "silly_kicks/xthreat/_value_iteration.py",
    "silly_kicks/xthreat/_eval.py",
    "silly_kicks/xthreat/_physical.py",
    "silly_kicks/tracking/_warnings.py",
    "silly_kicks/tracking/_run_values.py",
    "silly_kicks/tracking/defensive_credit/_params.py",
    "silly_kicks/tracking/defensive_credit/_orchestration.py",
    "silly_kicks/tracking/defensive_credit/_bravery.py",
    "silly_kicks/vaep/labels.py",
    "silly_kicks/vaep/formula.py",
    "silly_kicks/vaep/feature_framework.py",
    "silly_kicks/atomic/vaep/features.py",
    "silly_kicks/atomic/vaep/labels.py",
    "silly_kicks/atomic/vaep/formula.py",
    "silly_kicks/atomic/tracking/features.py",
    "silly_kicks/tracking/gradientsports.py",
    "silly_kicks/tracking/pitch_control/_params.py",
    "silly_kicks/tracking/pitch_control/_dispatch.py",
    "silly_kicks/tracking/pitch_control/_surface.py",
    "silly_kicks/tracking/pitch_control/_spearman.py",
    "silly_kicks/tracking/pitch_control/_fernandez_bornn.py",
    "silly_kicks/tracking/pitch_control/_voronoi.py",
    "silly_kicks/vaep/features/core.py",
    "silly_kicks/vaep/features/actiontype.py",
    "silly_kicks/vaep/features/result.py",
    "silly_kicks/vaep/features/bodypart.py",
    "silly_kicks/vaep/features/spatial.py",
    "silly_kicks/vaep/features/temporal.py",
    "silly_kicks/vaep/features/context.py",
    "silly_kicks/vaep/features/specialty.py",
    "silly_kicks/vaep/features/expected_threat.py",
    "silly_kicks/calibration/_cv.py",
    "silly_kicks/calibration/_features.py",
    "silly_kicks/calibration/_spaces.py",
    "silly_kicks/calibration/_carrier_objective.py",
    "silly_kicks/calibration/_diagnostics.py",
    "silly_kicks/calibration/_selection.py",
    "silly_kicks/calibration/_xt_bandwidth_objective.py",
    "silly_kicks/causal/matching.py",
    "silly_kicks/causal/opportunities.py",
    "silly_kicks/causal/power.py",
    # gkdv (TF-19 PR-3): the four modules that DEFINE its public surface. The registry
    # previously named `gkdv/__init__.py`, which has zero top-level defs -- a vacuous entry
    # that read as coverage while all four of these were unchecked. `test_no_registered_entry_
    # is_vacuous` now blocks that shape.
    "silly_kicks/gkdv/_arms.py",
    "silly_kicks/gkdv/_engine.py",
    "silly_kicks/gkdv/_metric.py",
    "silly_kicks/gkdv/_validate.py",
    # tracking: per-feature primitive + result modules reached via `tracking.__all__`.
    "silly_kicks/tracking/_ball_carrier.py",
    "silly_kicks/tracking/_defensive_line.py",
    "silly_kicks/tracking/_elastic_sync.py",
    "silly_kicks/tracking/_line_breaking.py",
    "silly_kicks/tracking/_obso.py",
    "silly_kicks/tracking/_packing.py",
    "silly_kicks/tracking/_pausa.py",
    "silly_kicks/tracking/_player_influence.py",
    "silly_kicks/tracking/_snapshot.py",
    "silly_kicks/tracking/_space_creation.py",
    "silly_kicks/tracking/_team_shape.py",
    "silly_kicks/tracking/feature_framework.py",
    "silly_kicks/tracking/kloppy.py",
    "silly_kicks/tracking/metrica.py",
    "silly_kicks/tracking/sportec.py",
    "silly_kicks/tracking/utils.py",
    "silly_kicks/tracking/pitch_control/_cache.py",
    "silly_kicks/tracking/preprocess/_config.py",
    "silly_kicks/tracking/preprocess/_config_dataclass.py",
    "silly_kicks/tracking/preprocess/_interpolation.py",
    "silly_kicks/tracking/preprocess/_smoothing.py",
    "silly_kicks/tracking/preprocess/_velocity.py",
    "silly_kicks/xtgk/_resolved_geometry.py",
    # ADR-019 id-identity seam, promoted from `tracking/_id_compat.py` to a repo-wide
    # public module this release. Documented on promotion rather than deferred: a
    # brand-new public module should not ship already in debt.
    "silly_kicks/id_compat.py",
    # --- modules that were WHOLLY unenforced under the module-level debt bucket ---
    # Every one of them is now enforced; the individual symbols that lack an example are
    # excused one at a time in `_EXAMPLES_DEBT`. Between them they hold 150 already-
    # documented public symbols that a module-level exemption was taking off the gate.
    "silly_kicks/atomic/spadl/config.py",
    "silly_kicks/calibration/_gates.py",
    "silly_kicks/calibration/_vaep_brier_objective.py",
    "silly_kicks/calibration/_xt.py",
    "silly_kicks/providers/sportec/parse.py",
    "silly_kicks/providers/statsbomb/parse.py",
    "silly_kicks/spadl/config.py",
    "silly_kicks/tracking/_cover_shadows.py",
    "silly_kicks/tracking/_das.py",
    "silly_kicks/tracking/_ghost_gk.py",
    "silly_kicks/tracking/_gk_completion.py",
    "silly_kicks/tracking/_gk_geometry.py",
    "silly_kicks/tracking/_gk_resolve.py",
    "silly_kicks/tracking/_restart_report.py",
    "silly_kicks/tracking/_shape_graph.py",
    "silly_kicks/tracking/_shot_goalmouth.py",
    "silly_kicks/tracking/_structural_pass.py",
    "silly_kicks/tracking/_xcross_attempt.py",
    "silly_kicks/tracking/_xshot_occurrence.py",
    "silly_kicks/tracking/_xt_gk.py",
    "silly_kicks/tracking/direction.py",
    "silly_kicks/tracking/features.py",
    "silly_kicks/tracking/pressure.py",
    "silly_kicks/tracking/schema.py",
    "silly_kicks/tracking/skillcorner.py",
    "silly_kicks/xtgk/_diagnostics.py",
    "silly_kicks/xtgk/_empirical.py",
    "silly_kicks/xtgk/_markov.py",
    "silly_kicks/xtgk/_metric.py",
    "silly_kicks/xtgk/_possession_value.py",
    "silly_kicks/xtgk/_pressure_levels.py",
    "silly_kicks/xtgk/_retention.py",
    "silly_kicks/xtgk/_retention_features.py",
    "silly_kicks/xtgk/_retention_labels.py",
    "silly_kicks/xtgk/_turnover.py",
    "silly_kicks/xtgk/_validate.py",
)


#: Individual public SYMBOLS whose Examples coverage is not yet enforced, keyed
#: ``"<file>::<qualified_name>"``, each with a note on what it is.
#:
#: PER-SYMBOL on purpose. A module-level bucket makes an exemption cost far more than it
#: excuses -- excusing `tracking/features.py` for 5 gaps took its other 79 DOCUMENTED symbols
#: off the gate too, free to regress unnoticed. Here a module with one gap keeps enforcing
#: everything else in it.
#:
#: Every entry must still BE undocumented (`test_debt_entries_are_really_undocumented`) and
#: must still name a real public symbol in an enforced module
#: (`test_debt_entries_name_real_public_symbols`) -- so writing an example makes CI tell you
#: to delete the entry, and renaming a symbol makes CI tell you the entry is fiction. The
#: bucket shrinks monotonically; it cannot silently absorb a new symbol, because a new symbol
#: is enforced by default and a new MODULE is in neither bucket and fails.
#:
#: An overloaded function reaches this bucket as its IMPLEMENTATION only -- `@overload` stubs
#: are skipped by the walker, because a stub whose body is `...` cannot carry an example and
#: so could never let its own entry burn down.
_EXAMPLES_DEBT: dict[str, str] = {
    # --- atomic-SPADL configuration ---
    "silly_kicks/atomic/spadl/config.py::actiontypes_df": (
        "Atomic-SPADL action-type lookup table -- the functools-cached accessor behind the config constants."
    ),
    # --- atomic-SPADL tracking mirrors: 14 of 36 public symbols. Each mirrors an entry in
    # `tracking/features.py` that is tracked there for the same reason -- a real match's frames
    # plus a linked atomic action. Their standard-SPADL originals are the ones to document
    # first; these follow the original, which is why they are listed rather than rewritten.
    "silly_kicks/atomic/tracking/features.py::actor_arc_length_pre_window": (
        "Atomic mirror: geometric arc length of the actor's path across the pre-action window."
    ),
    "silly_kicks/atomic/tracking/features.py::actor_displacement_pre_window": (
        "Atomic mirror: net Euclidean displacement of the actor across the pre-action window."
    ),
    "silly_kicks/atomic/tracking/features.py::add_actor_pre_window": (
        "Atomic mirror of the TF-3 aggregator emitting the two actor-movement columns."
    ),
    "silly_kicks/atomic/tracking/features.py::pressure_on_actor": (
        "Atomic mirror: multi-flavour pressure on the actor at the linked frame."
    ),
    "silly_kicks/atomic/tracking/features.py::add_pressure_on_actor": (
        "Atomic mirror of the TF-2 umbrella emitting one column per pressure flavour."
    ),
    "silly_kicks/atomic/tracking/features.py::pre_shot_gk_angle_to_shot_trajectory": (
        "Atomic mirror: signed GK-to-shot angle against the goal-centre reference at the linked frame."
    ),
    "silly_kicks/atomic/tracking/features.py::pre_shot_gk_angle_off_goal_line": (
        "Atomic mirror: signed keeper angle off the goal-line normal at the goal-mouth centre."
    ),
    "silly_kicks/atomic/tracking/features.py::add_pre_shot_gk_angle": (
        "Atomic mirror of the aggregator emitting the two GK-angle columns per shot."
    ),
    "silly_kicks/atomic/tracking/features.py::add_shot_goalmouth": (
        "Atomic mirror of the TF-48 aggregator; performs NO coordinate synthesis of its own."
    ),
    "silly_kicks/atomic/tracking/features.py::shot_crossing_y": (
        "Atomic mirror: goal-plane crossing y in the canonical attacked-goal-at-x=105 frame."
    ),
    "silly_kicks/atomic/tracking/features.py::shot_crossing_z": (
        "Atomic mirror: goal-plane crossing height, NaN wherever the provider carries no ball z."
    ),
    "silly_kicks/atomic/tracking/features.py::shot_speed": (
        "Atomic mirror: fitted initial ball speed at contact, on the contact sub-segment."
    ),
    "silly_kicks/atomic/tracking/features.py::shot_time_to_goal_line": (
        "Atomic mirror: elapsed seconds from the refined contact to the goal-plane crossing."
    ),
    "silly_kicks/atomic/tracking/features.py::shot_on_target_derived": (
        "Atomic mirror: nullable on-target flag with the ball-radius tolerance folded in."
    ),
    # --- standard-SPADL configuration ---
    "silly_kicks/spadl/config.py::actiontypes_df": (
        "Standard-SPADL action-type lookup table, the functools-cached accessor used across the converters."
    ),
    "silly_kicks/spadl/config.py::results_df": (
        "Standard-SPADL result lookup table -- the id/name mapping behind result_id, cached per process."
    ),
    "silly_kicks/spadl/config.py::bodyparts_df": (
        "Standard-SPADL bodypart lookup table -- the id/name mapping behind bodypart_id, cached per process."
    ),
    # --- providers/sportec: the DFL/IDSSE parse port behind the [parse-dfl] extra. Every
    # entry needs a real DFL XML export to demonstrate, which is why none carries one yet.
    "silly_kicks/providers/sportec/parse.py::idsse_native_match_id": (
        "Derives the DFL-native match id from parsed match info -- the key the cross-repo bronze contract joins on."
    ),
    "silly_kicks/providers/sportec/parse.py::finalize_bronze_df": (
        "Normalises a raw DFL bronze frame (dtypes and column order) before it reaches a native shaper."
    ),
    "silly_kicks/providers/sportec/parse.py::adapt_idsse_events_for_silly_kicks": (
        "Adapts parsed IDSSE events into the shape the sportec SPADL converter expects as input."
    ),
    "silly_kicks/providers/sportec/parse.py::derive_idsse_home_team_start_left": (
        "Derives the periods 1-2 kickoff-orientation flag from parsed DFL match info."
    ),
    "silly_kicks/providers/sportec/parse.py::derive_idsse_home_team_start_left_extratime": (
        "The extra-time sibling of the kickoff-orientation derivation, feeding the ADR-035 geometric backstop."
    ),
    "silly_kicks/providers/sportec/parse.py::MatchInfo": (
        "Typed return of parse_dfl_match_info -- silly-kicks' own domain name for the versioned bronze contract."
    ),
    "silly_kicks/providers/sportec/parse.py::parse_dfl_match_info": (
        "Parses a DFL match-information XML into MatchInfo (teams, periods, pitch, kickoff sides)."
    ),
    "silly_kicks/providers/sportec/parse.py::parse_dfl_tracking": (
        "Parses a DFL positions XML into RAW tracking bronze -- smoothing and velocities stay consumer-side."
    ),
    "silly_kicks/providers/sportec/parse.py::parse_dfl_events": (
        "Parses a DFL events XML into raw event bronze, ahead of the native-shaping step."
    ),
    "silly_kicks/providers/sportec/parse.py::shape_tracking_to_native": (
        "Shapes tracking bronze into the tracking.sportec EXPECTED_INPUT_COLUMNS converter input."
    ),
    "silly_kicks/providers/sportec/parse.py::shape_events_to_native": (
        "Shapes event bronze into the spadl.sportec EXPECTED_INPUT_COLUMNS converter input."
    ),
    # --- TF-30 cover shadows (the module's compute entry points are already documented) ---
    "silly_kicks/tracking/_cover_shadows.py::CoverShadowParams.k_drag": (
        "Derived drag coefficient of the Spearman ball-drag model, computed from the dataclass fields."
    ),
    "silly_kicks/tracking/_cover_shadows.py::BlockingScoreResult.blocked_threat_fraction": (
        "Share of the counterfactual threat a blocker removes -- the derived accessor on the result type."
    ),
    "silly_kicks/tracking/_cover_shadows.py::compute_threat_pc": (
        "Threat-weighted pitch-control primitive behind the blocking score; needs a real frame to demonstrate."
    ),
    "silly_kicks/tracking/_cover_shadows.py::lane_control": (
        "Per-(passer, receiver) lane blocking probability over the discretized corridor; needs a real frame."
    ),
    "silly_kicks/tracking/_cover_shadows.py::compute_blocking_score": (
        "Voronoi counterfactual threat reduction from removing one defender; needs a real frame and a fitted xT."
    ),
    # --- TF-28 DAS adapter ---
    "silly_kicks/tracking/_das.py::DasUnscoreableError": (
        "Raised when the optional accessible-space dependency cannot score a frame at all."
    ),
    # --- TF-18 ghost GK (GhostClampWarning in the same block IS documented) ---
    "silly_kicks/tracking/_ghost_gk.py::IntegrityError": (
        "Artifact-integrity error category raised by GhostGkModel.load on a tampered or pre-chirality artifact."
    ),
    "silly_kicks/tracking/_ghost_gk.py::keeper_detection_mask": (
        "Per-frame mask marking frames whose keeper is genuinely detected rather than interpolated."
    ),
    "silly_kicks/tracking/_ghost_gk.py::serve_ghost_gk_positions": (
        "Serving helper returning the model's ghost keeper positions for a set of frames."
    ),
    # --- TF-24 calibration harness (ADR-009). The class XtBandwidthObjective needs a real
    # multi-match corpus AND a fitted frozen-xT artifact plus an Optuna trial to demonstrate;
    # the other calibration objective/feature symbols were graduated to illustrative
    # literal-block call sketches (their demonstrations already lived in the docstrings,
    # commented out behind a malformed +SKIP; the sketch form is this package's canonical style).
    "silly_kicks/calibration/_xt_bandwidth_objective.py::XtBandwidthObjective": (
        "SK-xT-3 objective minimizing K-fold held-out xT transition NLL over the bandwidth sweep."
    ),
    # --- ADR-015 causal harness ---
    "silly_kicks/causal/opportunities.py::build_opportunities": (
        "Spell state machine emitting one crosser-anchored row per in-domain possession spell."
    ),
    # --- tracking.utils pre-flight guards and sync scoring ---
    "silly_kicks/tracking/utils.py::validate_time_base": (
        "The ADR-017 pre-link guard that actions and frames share a per-period time base."
    ),
    "silly_kicks/tracking/utils.py::validate_id_dtypes": (
        "The ADR-019 opt-in guard that action and frame id dtypes compare rather than silently miss."
    ),
    "silly_kicks/tracking/utils.py::add_sync_score": (
        "Merges the three sync_score_* columns onto actions from a link batch (TF-6)."
    ),
    # --- tracking.preprocess: the PreprocessConfig pipeline, each step needing real frames ---
    "silly_kicks/tracking/preprocess/_interpolation.py::interpolate_frames": (
        "Linear fill of NaN positional gaps up to max_gap_seconds, leaving longer gaps honest."
    ),
    "silly_kicks/tracking/preprocess/_smoothing.py::smooth_frames": (
        "Savitzky-Golay or EMA position smoothing, emitting additive x_smoothed / y_smoothed columns."
    ),
    "silly_kicks/tracking/preprocess/_velocity.py::derive_velocities": (
        "Adds vx / vy / speed from smoothed positions via the SG derivative; needs a real frame series."
    ),
    # --- ADR-024 GK completion model: the whole fit / predict / persist / variant lifecycle ---
    "silly_kicks/tracking/_gk_completion.py::serve_mode_from_lcb": (
        "Pure per-type serve-mode gate (model vs base_rate) decided from a held-out AUC lower bound."
    ),
    "silly_kicks/tracking/_gk_completion.py::extract_gk_completion_features": (
        "The shared train==serve feature extractor for GK distributions -- both producers route through it."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel": (
        "The ADR-024 logistic completion model, served pure-numpy from a pickle-free JSON+SHA256 artifact."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.fit": (
        "Fits the logistic coefficients on native-outcome rows only (an inferred label is positive-only)."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.predict_proba": (
        "Pure scorer -- sigmoid(X beta) with no serve-mode switching; the gate lives at the caller."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.serve_mode_for_types": (
        "Per-action-type serve mode baked into the artifact, fail-opening on a pre-gate 4.21.0 model."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.base_rate_for_types": (
        "Per-type calibrated base rates served in place of geometry when the type fails its AUC gate."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.to_dict": (
        "JSON-serialisable artifact payload -- coefficients, gate metrics and the version tag."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.from_dict": (
        "Rebuilds a model from its artifact payload, without touching the filesystem."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.save": (
        "Writes the pickle-free artifact (model.json plus its SHA256SUMS entry) to a directory."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.load": (
        "Loads and checksum-verifies an artifact directory, fail-opening on a pre-gate version tag."
    ),
    "silly_kicks/tracking/_gk_completion.py::GkCompletionModel.from_variant": (
        "Loads a bundled variant by key, aliasing the gs KEY onto the bundled default directory."
    ),
    "silly_kicks/tracking/_gk_completion.py::variant_key_for_provider": (
        "Pure provider-to-variant-key mapping: skillcorner gets its own weights, everything else gs."
    ),
    "silly_kicks/tracking/_gk_completion.py::compute_gk_completion": (
        "Standalone per-action completion probability, falling back to a per-type base rate when unscoreable."
    ),
    "silly_kicks/tracking/_gk_completion.py::prepare_gk_completion_training_data": (
        "Training-data preparer sharing the domain predicate and geometry resolution with the serve path."
    ),
    # --- ADR-024 / ADR-025 restart and goal-kick geometry ---
    "silly_kicks/tracking/_gk_geometry.py::native_origin_is_trusted": (
        "The fail-safe provider allowlist deciding whether a native restart origin can be believed."
    ),
    "silly_kicks/tracking/_gk_geometry.py::resolve_gk_geometry": (
        "Goal-kick origin and destination resolution cascade, emitting per-row provenance and confidence."
    ),
    "silly_kicks/tracking/_gk_geometry.py::resolve_restart_geometry": (
        "The single ADR-025 restart-geometry engine that add_restart_coordinates and the GK shim both use."
    ),
    "silly_kicks/tracking/_gk_geometry.py::apply_restart_tripwire": (
        "Geometry tripwire reverting an imputed restart origin that lands outside its lawful region."
    ),
    "silly_kicks/tracking/_gk_geometry.py::flag_native_goalkick_out_of_region": (
        "S4 provenance flag for a NATIVE goal-kick origin outside the plausible region (warn, never revert)."
    ),
    # --- GK resolution seam (defending_gk_from_frames, its sibling, IS documented) ---
    "silly_kicks/tracking/_gk_resolve.py::acting_gk_from_frames": (
        "Acting-team keeper resolution from linked frames, with the roster-identity and GK-sub fallbacks."
    ),
    # --- ADR-025 restart provenance QA ---
    "silly_kicks/tracking/_restart_report.py::RestartCoordinateReport": (
        "Aggregate QA dataclass tallying restart-coordinate provenance sources and confidences."
    ),
    # --- TF-39 shape graph result types ---
    "silly_kicks/tracking/_shape_graph.py::PositionLabel": (
        "The 5x5 face-centre position label enum that infer_positions assigns to each player."
    ),
    "silly_kicks/tracking/_shape_graph.py::ShapeGraph": (
        "Container for the iterative-Delaunay shape graph returned by compute_shape_graph."
    ),
    # --- TF-45 structural pass ---
    "silly_kicks/tracking/_structural_pass.py::StructuralPassParams": (
        "Frozen parameter dataclass whose sigma was empirically tuned on real WC2022 passes."
    ),
    "silly_kicks/tracking/_structural_pass.py::compute_structural_pass_metrics": (
        "Per-frame LBS / SGM / SDI primitive; needs a real frame and a real pass to demonstrate."
    ),
    # --- TF-17 xCross attempt model: extractor, labels, and the full model lifecycle ---
    "silly_kicks/tracking/_xcross_attempt.py::extract_xcross_features": (
        "The 7 paper confounders plus the novel GK block, in goal-relative coordinates."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::build_xcross_labels": (
        "Time-windowed cross-attempt labels built on the shared occurrence-label core."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::prepare_xcross_training_data": (
        "The shared train==serve entry point, with the alive-ball and wide-area domain filter."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel": (
        "Deterministic-XGBoost cross-attempt propensity classifier, served from a pickle-free booster JSON."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.fit": (
        "Fits the booster on prepared training data; needs a real multi-match corpus."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.predict_proba": (
        "Per-frame cross-attempt probability for the in-possession team."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.save": (
        "Writes booster JSON, metadata and SHA256SUMS, including the ADR-040 chirality fingerprint."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.load": (
        "Loads an artifact with fail-closed chirality verification and the base_score compatibility guard."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.from_variant": (
        "Loads a named variant, routing sc_extended to the Hub rather than the wheel."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::compute_xcross_attempt": (
        "The ADR-005 per-frame compute surface over a fitted model."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::add_xcross_attempt": (
        "The ADR-005 action-coupled aggregator emitting the xCross columns."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::xcross_attempt_xfns": (
        "VAEP xfns factory wiring xCross into pre_shot_gk_full_default_xfns."
    ),
    "silly_kicks/tracking/_xcross_attempt.py::XCrossAttemptModel.from_hub": (
        "Downloads published weights from the HuggingFace Hub; a network round trip no example can stand in for."
    ),
    # --- TF-16 xS: the extractor, the labels, and the whole model lifecycle. Every entry
    # needs a real match's frames, and the model ones a fitted booster on top of that.
    "silly_kicks/tracking/_xshot_occurrence.py::IntegrityError": (
        "Artifact-integrity error category raised by the xS loader on a checksum or chirality failure."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::load_xgb_booster_base_score_safe": (
        "The 4.51.0 xgboost 2.x/3.x base_score compatibility loader that strips a bracketed intercept."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::extract_xshot_features": (
        "The paper-faithful 27-feature extractor in goal-relative coordinates; needs one real frame."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::build_xshot_labels": (
        "Time-windowed shot-occurrence labels, positive iff a same-team shot lands in the horizon."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::prepare_xshot_training_data": (
        "The shared train==serve entry point, with the alive-ball and attacking-third domain filter."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::subsample_negatives": (
        "Train-only deterministic negative subsampling, kept out of the shared prepare path on purpose."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel": (
        "Deterministic-XGBoost shot-occurrence classifier, served from a pickle-free booster JSON."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.fit": (
        "Fits the booster and records carrier_params into the model (R3); needs a real multi-match corpus."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.predict_proba": (
        "Per-frame P(shot) for the in-possession team, over a fitted booster."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.save": (
        "Writes booster JSON, metadata and SHA256SUMS, including the ADR-040 chirality fingerprint."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.load": (
        "Loads an artifact with fail-closed chirality verification and the base_score compatibility guard."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.from_variant": (
        "Loads a bundled variant by name, routing sc_extended to the Hub rather than the wheel."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::XShotOccurrenceModel.from_hub": (
        "Downloads published weights from the HuggingFace Hub; a network round trip no example can stand in for."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::compute_xshot_occurrence": (
        "The ADR-005 per-frame compute surface over a fitted model."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::add_xshot_occurrence": (
        "The ADR-005 action-coupled aggregator emitting the xshot_occurrence column."
    ),
    "silly_kicks/tracking/_xshot_occurrence.py::xshot_occurrence_xfns": (
        "VAEP xfns factory emitting xshot_occurrence per gamestate slot."
    ),
    # --- xT-GK v1, FROZEN pending the lakehouse migration to xtgk/ v2 ---
    "silly_kicks/tracking/_xt_gk.py::XtGkReport": (
        "Aggregate QA report for an xt_gk run -- provenance counts, variants and coverage."
    ),
    "silly_kicks/tracking/_xt_gk.py::XtGkReport.from_frame": (
        "Builds the report from an xt_gk output frame's provenance columns."
    ),
    "silly_kicks/tracking/_xt_gk.py::XtGkParams": (
        "The v1 normative parameter surface (gamma / delta / phi / eta), intent-set and never calibrated."
    ),
    "silly_kicks/tracking/_xt_gk.py::XtGkParams.for_philosophy": (
        "The five provisional philosophy presets over the normative parameter surface."
    ),
    "silly_kicks/tracking/_xt_gk.py::compute_xt_gk": (
        "The frozen v1 compute entry point; needs frames, a fitted xT grid and a GK-distribution domain."
    ),
    # --- frame orientation (finalize_orientation, its sibling in the same module, IS documented) ---
    "silly_kicks/tracking/direction.py::compute_attacking_direction": (
        "Labels each frame with its team_attacking_direction from the kickoff-side flags."
    ),
    # --- tracking.features: 30 of 84 public symbols (5 before the import-only tightening,
    # which is where the other 25 came from). The remaining 54 stay enforced -- exactly the
    # coverage a module-level exemption used to throw away.
    #
    # Every entry below is an action-coupled aggregator or a per-Series accessor over one, so
    # each needs a real match's frames AND a linked action to demonstrate. That is what the
    # illustrative-sketch style exists for; writing 25 of them mechanically would be the same
    # box-ticking the +SKIP rule rejects, so they are tracked here instead.
    "silly_kicks/tracking/features.py::add_structural_pass": (
        "TF-45 aggregator emitting raw structural_lbs / structural_sgm / structural_sdi per action."
    ),
    "silly_kicks/tracking/features.py::structural_pass_xfns": (
        "VAEP xfns factory for the three structural-pass primitives."
    ),
    "silly_kicks/tracking/features.py::add_xt_gk": (
        "The v1 xT-GK aggregator, emitting the composite plus its component columns per GK distribution."
    ),
    "silly_kicks/tracking/features.py::xt_gk_xfns": (
        "VAEP xfns factory for xT-GK v1; in no default list, so opting it in is a self-triggered retrain."
    ),
    "silly_kicks/tracking/features.py::add_gk_completion": (
        "The lakehouse wide-table aggregator emitting gk_completion per in-scope GK distribution."
    ),
    # TF-3 actor pre-window movement
    "silly_kicks/tracking/features.py::actor_displacement_pre_window": (
        "Net Euclidean displacement of the actor across the pre-action window, as a per-action Series."
    ),
    "silly_kicks/tracking/features.py::add_actor_pre_window": (
        "TF-3 aggregator emitting the 2 movement columns plus the 4 linkage-provenance columns."
    ),
    # TF-2 pressure
    "silly_kicks/tracking/features.py::pressure_on_actor": (
        "Multi-flavour pressure on the actor at the linked frame; the method picks the published model."
    ),
    "silly_kicks/tracking/features.py::add_pressure_on_actor": (
        "Umbrella aggregator emitting one pressure_on_actor__<method> column per requested flavour."
    ),
    # TF-24 pre-shot GK angles
    "silly_kicks/tracking/features.py::pre_shot_gk_angle_to_shot_trajectory": (
        "Signed angle between goal-centre-to-anchor and GK-to-anchor at the linked pre-shot frame."
    ),
    "silly_kicks/tracking/features.py::pre_shot_gk_angle_off_goal_line": (
        "Signed angle of the keeper off the goal-line normal at the goal-mouth centre."
    ),
    "silly_kicks/tracking/features.py::add_pre_shot_gk_angle": (
        "Aggregator emitting the 2 GK-angle columns at the linked frame for each shot action."
    ),
    # TF-14 defensive line
    "silly_kicks/tracking/features.py::defensive_line_x": (
        "Mean x of the defending team's back line at the linked frame, as a per-action Series."
    ),
    "silly_kicks/tracking/features.py::back_line_high_x": (
        "x of the most advanced back-line player on the defending team at the linked frame."
    ),
    "silly_kicks/tracking/features.py::compactness_x": (
        "x-spread (max minus min) of the defending team's back line at the linked frame."
    ),
    "silly_kicks/tracking/features.py::lateral_width": (
        "y-spread (max minus min) of the defending team's back line at the linked frame."
    ),
    "silly_kicks/tracking/features.py::max_lateral_gap": (
        "Largest y-gap between adjacent y-sorted back-line players -- the seam a runner attacks."
    ),
    "silly_kicks/tracking/features.py::back_n_count": (
        "How many players form the defending team's back line (3, 4 or 5) at the linked frame."
    ),
    "silly_kicks/tracking/features.py::add_defensive_line": (
        "TF-14 aggregator emitting the 6 back-line geometry columns plus linkage provenance."
    ),
    # TF-4 off-ball runs and line breaks
    "silly_kicks/tracking/features.py::add_off_ball_runs": (
        "TF-4 aggregator emitting the 4 off-ball-run columns per action."
    ),
    "silly_kicks/tracking/features.py::add_line_break": (
        "Line-break aggregator; the method kwarg dispatches between the TF-4 and TF-32 Ward detectors."
    ),
    "silly_kicks/tracking/features.py::add_off_ball_context": (
        "Umbrella emitting all 6 off-ball-run plus line-break columns in one linked pass."
    ),
    # TF-31 / TF-39 team structure
    "silly_kicks/tracking/features.py::add_team_shape": (
        "TF-31 aggregator emitting 20 team-shape columns (10 metrics for each of the two teams)."
    ),
    "silly_kicks/tracking/features.py::add_shape_graph": (
        "TF-39 aggregator emitting 6 Delaunay shape-graph columns (3 metrics for each team)."
    ),
    # TF-48 post-shot goalmouth geometry (ADR-030); each needs a real shot's ball track
    "silly_kicks/tracking/features.py::add_shot_goalmouth": (
        "TF-48 aggregator fitting the post-contact ball trajectory to the attacked goal plane."
    ),
    "silly_kicks/tracking/features.py::shot_crossing_y": (
        "Goal-plane crossing y in the canonical attacked-goal-at-x=105 frame, NaN out of scope."
    ),
    "silly_kicks/tracking/features.py::shot_crossing_z": (
        "Goal-plane crossing height, NaN out of scope or wherever the provider carries no ball z."
    ),
    "silly_kicks/tracking/features.py::shot_speed": (
        "Fitted initial ball speed at contact -- always the contact sub-segment (ADR-030 M-1)."
    ),
    "silly_kicks/tracking/features.py::shot_time_to_goal_line": (
        "Elapsed seconds from the refined contact to the goal-plane crossing, the keeper's reaction budget."
    ),
    "silly_kicks/tracking/features.py::shot_on_target_derived": (
        "Nullable on-target flag: crossing inside posts and bar expanded by the ball-radius tolerance."
    ),
    # --- TF-2 pressure parameter dataclasses, one per method flavour ---
    "silly_kicks/tracking/pressure.py::AndrienkoParams": (
        "Frozen parameters for the andrienko_oval pressure flavour (the default method)."
    ),
    "silly_kicks/tracking/pressure.py::LinkParams": ("Frozen parameters for the link_zones pressure flavour."),
    "silly_kicks/tracking/pressure.py::BekkersParams": (
        "Frozen parameters for the bekkers_pi pressure flavour, the measure xT-GK v2 pins."
    ),
    # --- tracking report / diagnosis types and their computed properties ---
    "silly_kicks/tracking/schema.py::TrackingConversionReport.has_unrecognized": (
        "True when a conversion met provider rows it could not classify -- the loud-not-silent tell."
    ),
    "silly_kicks/tracking/schema.py::LinkReport.link_rate": (
        "Overall action-to-frame link rate; the per-period breakdown is what the ADR-017 guard reads."
    ),
    "silly_kicks/tracking/schema.py::LinkReport.sync_scores": (
        "Per-action sync-quality frame for a link batch; needs a real linked match to demonstrate."
    ),
    "silly_kicks/tracking/schema.py::TimeBaseDiagnosis": (
        "The ADR-017 pre-link diagnosis type returned by validate_time_base."
    ),
    "silly_kicks/tracking/schema.py::TimeBaseDiagnosis.has_suspected_mismatch": (
        "Flags a suspected event/frame time-base mismatch, decoupled from the link-rate symptom."
    ),
    "silly_kicks/tracking/schema.py::IdDtypeDiagnosis": (
        "The ADR-019 diagnosis type returned by validate_id_dtypes, the opt-in loud guard."
    ),
    "silly_kicks/tracking/schema.py::VelocityRegimeDiagnosis": (
        "The diagnosis type returned by validate_velocity_regime, third member of the "
        "validate_time_base / validate_id_dtypes family; demonstrated on the validator."
    ),
    "silly_kicks/tracking/schema.py::IdDtypeDiagnosis.has_mismatch": (
        "True when action and frame id dtypes would silently mis-resolve against each other."
    ),
    # --- ADR-038 SkillCorner geometry rate gate ---
    "silly_kicks/tracking/skillcorner.py::GeometryGateReport": (
        "Return type of the per-match geometry rate gate, carrying the player and ball excursion fractions."
    ),
    "silly_kicks/tracking/skillcorner.py::geometry_rate_gate": (
        "The S1 per-match rate gate excluding a match whose off-pitch excursion rate is implausible."
    ),
    # --- xT-GK v2 deep-zone gate ---
    "silly_kicks/xtgk/_diagnostics.py::GateConfig": (
        "Pre-registered gate configuration -- effect floors, n_min and the expected direction."
    ),
    "silly_kicks/xtgk/_diagnostics.py::DeepZoneGateReport": (
        "Gate verdict plus the per-cell evidence it was reached from."
    ),
    "silly_kicks/xtgk/_diagnostics.py::run_deep_zone_gate": (
        "The make-or-break occupied-cell deep-zone pressure gate; needs a fitted possession value."
    ),
    "silly_kicks/xtgk/_diagnostics.py::run_gate_with_ladder": (
        "The pre-registered three-rung ladder (global, then zone-conditional, then STOP)."
    ),
    "silly_kicks/xtgk/_diagnostics.py::run_gate_both_orientations": (
        "Runs the gate under mirror_y equivariance and mirror_x rejection as a sanity pair."
    ),
    "silly_kicks/xtgk/_diagnostics.py::ood_rate_by_source": (
        "Per-source out-of-distribution rate diagnostic over a scored cohort."
    ),
    "silly_kicks/xtgk/_diagnostics.py::frame_present_null_pressure_count": (
        "Counts frame-present rows whose pressure is null -- the tracking-gap tell."
    ),
    # --- xT-GK v2 model-free cross-check adapter ---
    "silly_kicks/xtgk/_empirical.py::EmpiricalPossessionValue": (
        "Model-free per-action first-shot surface, independent of the Markov estimator by design."
    ),
    "silly_kicks/xtgk/_empirical.py::EmpiricalPossessionValue.fit": (
        "Builds the empirical surface by reverse scan; graded on build-up cells, never the deep ones."
    ),
    "silly_kicks/xtgk/_empirical.py::EmpiricalPossessionValue.surface": (
        "The fitted per-zone, per-pressure value surface."
    ),
    "silly_kicks/xtgk/_empirical.py::EmpiricalPossessionValue.value": (
        "Point lookup on the empirical surface, satisfying the PossessionValue Protocol."
    ),
    # --- xT-GK v2 production possession value ---
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue": (
        "The production pressure-stratified value-iteration surface, reusing xthreat.value_iteration verbatim."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.fit": (
        "Fits V(z,p) from an xG-calibrated first-shot reward over the goal-kick-inclusive move set."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.surface": ("The fitted per-zone, per-pressure value grid."),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.value": (
        "Point lookup on the fitted surface; the fit-path NaN contract does NOT hold here."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.support": (
        "Per-cell observation counts backing the surface, so a thin estimate is visible."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.delta_v": (
        "The two-factor Shapley split, where delta_pressure plus delta_position equals delta_v."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.save": (
        "Pickle-free npz plus JSON plus SHA256 persistence of the fitted surface."
    ),
    "silly_kicks/xtgk/_markov.py::MarkovPossessionValue.load": (
        "Restores a fitted surface, including the pressure-level metadata round trip."
    ),
    # --- xT-GK v2 metric assembler ---
    "silly_kicks/xtgk/_metric.py::compute_xt_gk_v2": (
        "Composes the three injected ports into the additive decomposition columns; needs resolved "
        "GK geometry applied first (ADR-036 amendment) and a fitted possession value."
    ),
    # --- xT-GK v2 possession-value port and zone binning ---
    "silly_kicks/xtgk/_possession_value.py::State": (
        "The (zone, pressure) state the possession-value port is keyed on."
    ),
    "silly_kicks/xtgk/_possession_value.py::DeltaV": (
        "The two-factor decomposition returned by delta_v (pressure plus position)."
    ),
    "silly_kicks/xtgk/_possession_value.py::PossessionValue": (
        "The Protocol both the Markov and empirical adapters satisfy -- the hexagonal seam."
    ),
    "silly_kicks/xtgk/_possession_value.py::PossessionValue.value": (
        "Protocol method: the value of a single (zone, pressure) state, implemented by both adapters."
    ),
    "silly_kicks/xtgk/_possession_value.py::PossessionValue.surface": (
        "Protocol method: the whole fitted surface, which the diagnostics read rather than point lookups."
    ),
    "silly_kicks/xtgk/_possession_value.py::PossessionValue.delta_v": (
        "Protocol method: the two-factor split between two states."
    ),
    "silly_kicks/xtgk/_possession_value.py::zone_of": ("Maps one coordinate pair to its grid zone index."),
    "silly_kicks/xtgk/_possession_value.py::flat_zones": (
        "Vectorized coordinate-to-zone binning. Its NaN-to-zone-176 behaviour is a FIT-PATH-ONLY "
        "contract (ADR-036 / 4.46.0) -- an example must not read as a licence to score on it."
    ),
    "silly_kicks/xtgk/_possession_value.py::mirror_zone": (
        "180-degree reflection of a zone index, used by the mirrored turnover proxy."
    ),
    # --- xT-GK v2 pressure stratification ---
    "silly_kicks/xtgk/_pressure_levels.py::band_of_zone": (
        "Maps a zone to its pitch band, the unit the zone-conditional mode cuts within."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::coalesce_frame_present_null_pressure": (
        "Coalesces a frame-present null pressure rather than dropping the row silently."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels": (
        "Pressure tercile stratification in global or zone-conditional mode."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.fit": (
        "Learns the tercile cutpoints, globally or per band depending on mode."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.from_cutpoints": (
        "Builds global levels from caller-supplied cutpoints, bypassing the fit."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.from_band_cutpoints": (
        "Builds zone-conditional levels from per-band cutpoints, bypassing the fit."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.apply": (
        "Assigns a tercile index to each row, honouring the fitted mode."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.occupancy": (
        "Per-tercile row counts, the support the gate's n_min is checked against."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.to_meta": (
        "On-disk metadata form, kept byte-identical for a global fit so old artifacts still load."
    ),
    "silly_kicks/xtgk/_pressure_levels.py::PressureLevels.from_meta": (
        "Restores levels from metadata; an absent pressure_mode means the global mode."
    ),
    # --- xT-GK v2 retention port and the rho model ---
    "silly_kicks/xtgk/_retention.py::RetentionModel": ("The retention Protocol the metric assembler depends on."),
    "silly_kicks/xtgk/_retention.py::RetentionModel.predict_proba": (
        "Protocol method: per-action retention probability."
    ),
    "silly_kicks/xtgk/_retention.py::variant_key_for_provider": (
        "Provider-to-variant mapping for rho; skillcorner has its own weights, everything else falls back."
    ),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel": (
        "The bundled logistic rho model, served pure-numpy from a JSON plus SHA256 artifact."
    ),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.fit": (
        "Fits rho on marts-native retention features over the is_gk_distribution domain."
    ),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.predict_proba": (
        "Pure sigmoid scorer; the caller is responsible for not scoring non-finite geometry."
    ),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.to_dict": ("JSON-serialisable coefficients plus provenance."),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.from_dict": ("Rebuilds a model from its artifact payload."),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.save": ("Writes the pickle-free artifact plus its checksum."),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.load": ("Loads and checksum-verifies an artifact directory."),
    "silly_kicks/xtgk/_retention.py::GkRetentionModel.from_variant": (
        "Loads a bundled variant by provider key, falling back to default when none is bundled."
    ),
    # --- xT-GK v2 retention features and labels ---
    "silly_kicks/xtgk/_retention_features.py::extract_retention_features": (
        "The 8 marts-native rho features, sourced from the gold action marts rather than tracking frames."
    ),
    "silly_kicks/xtgk/_retention_labels.py::retains": (
        "The rho label. Its truncated-window NaN, foul-skip and canonical scan-order rules are "
        "load-bearing for training-label parity (ADR-036 amendment), so an example must teach them."
    ),
    # --- xT-GK v2 turnover cost ---
    "silly_kicks/xtgk/_turnover.py::TurnoverCost": ("The turnover-cost Protocol the metric assembler depends on."),
    "silly_kicks/xtgk/_turnover.py::TurnoverCost.value": (
        "Protocol method: opponent value after a turnover at a state."
    ),
    "silly_kicks/xtgk/_turnover.py::TurnoverCost.surface": ("Protocol method: the whole turnover-cost surface."),
    "silly_kicks/xtgk/_turnover.py::TurnoverCost.support": ("Protocol method: per-cell support behind the surface."),
    "silly_kicks/xtgk/_turnover.py::MirroredTurnoverCost": (
        "The mirror proxy V(mirror_zone(z)); superseded as the default because it overstated deep "
        "opponent threat 10-50x at real support, and retained for comparison."
    ),
    "silly_kicks/xtgk/_turnover.py::MirroredTurnoverCost.value": ("Mirrored point lookup on the fitted surface."),
    "silly_kicks/xtgk/_turnover.py::MirroredTurnoverCost.surface": ("The mirrored surface derived from the fitted V."),
    "silly_kicks/xtgk/_turnover.py::MirroredTurnoverCost.support": ("Support of the underlying V, mirrored zone-wise."),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue": (
        "The faithful possession-bound turnover cost with hierarchical bin widening (4.45.0)."
    ),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue.fit": (
        "Scans to the possession or match boundary; a game_id guard fails loud rather than scanning across matches."
    ),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue.surface": (
        "The fitted opponent-value surface, at the native cell resolution before any widening."
    ),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue.value": (
        "Point lookup, widening the bin when a cell lacks min_support."
    ),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue.support": ("Per-cell turnover counts backing the surface."),
    "silly_kicks/xtgk/_turnover.py::EmpiricalTurnoverValue.resolution_level": (
        "Which widening level answered a lookup -- native cell, coarse block, or global."
    ),
    "silly_kicks/xtgk/_turnover.py::surface_divergence": (
        "Module-level audit comparing two turnover surfaces, the mirror-vs-faithful diagnostic."
    ),
    # --- xT-GK v2 input validation ---
    "silly_kicks/xtgk/_validate.py::PossessionValueInputDiagnosis": (
        "What validate_possession_value_input returns -- orientation and column findings."
    ),
    "silly_kicks/xtgk/_validate.py::validate_possession_value_input": (
        "Attack-orientation and required-column guard run before a possession value is fitted."
    ),
}


#: Registered modules that the derivation does NOT classify as public. Enforcing them anyway is
#: harmless OVER-coverage, but each needs a reason -- otherwise this set becomes the escape
#: hatch that lets `test_registered_modules_are_still_public` be silenced instead of heeded.
_EXTRA_COVERAGE: dict[str, str] = {
    "silly_kicks/tracking/pitch_control/_fernandez_bornn.py": (
        "Pitch-control flavour reached only through the `compute_pitch_control(method=...)` "
        "dispatcher, so it is not itself re-exported. Documented and enforced regardless: the "
        "three flavours are a set, and one silently undocumented backend is the odd one out."
    ),
    "silly_kicks/tracking/pitch_control/_voronoi.py": (
        "Pitch-control flavour reached only through the `compute_pitch_control(method=...)` "
        "dispatcher, so it is not itself re-exported. Enforced alongside its two sibling "
        "backends for the same reason."
    ),
}


# Pure-type symbols that don't fit the illustrative-example pattern.
# Adding a new entry here is a deliberate documentation-policy decision —
# the additive-only nature is a forcing function.
_SKIP_SYMBOLS = frozenset(
    {
        "BoundaryMetrics",  # TypedDict — fields are the documentation
        "CoverageMetrics",  # TypedDict
        "ConversionReport",  # TypedDict
        "CausalEstimate",  # frozen dataclass — fields are the documentation (DetectionResult precedent)
        "DetectionResult",  # frozen dataclass — fields are the documentation
        "InputConvention",  # str-Enum — members are the documentation
        "OpportunityConfig",  # frozen dataclass — fields are the documentation (DetectionResult precedent)
        "PitchControlParams",  # type alias union — components have examples
        "PointScore",  # frozen dataclass — plain per-fold score container, fields are the documentation
        "Selection",  # frozen dataclass — plain selection-result container, fields are the documentation
    }
)


#: Doctest bodies that demonstrate nothing, so they do not make an example "real".
_PLACEHOLDER_CODE = frozenset({"", "...", "pass"})

#: Stated once, and quoted verbatim by the failure message so the rule is legible to whoever
#: trips it. Deliberately phrased as what to WRITE, not as what the parser rejects.
_REAL_EXAMPLE_RULE = (
    "Write an example that SHOWS THE CALL. Two forms count: (a) a doctest that actually runs "
    "-- not annotated '# doctest: +SKIP', not a bare placeholder ('...', 'pass'), and doing "
    "more than importing the symbol or pointing at a test file; or (b) an indented "
    "illustrative code block inside the Examples section (the canonical 'usage sketch' style "
    "used across this package, and the right choice when the call needs a real match's data). "
    "What does NOT count is an Examples section that gestures at an example instead of being "
    "one: only '+SKIP' doctests, only placeholders, only '>>> from x import y', only a "
    "'>>> # see tests/... for a runnable example' comment, or no code at all. An import is "
    "setup, and a pointer to an example is not an example -- but an import FOLLOWED by a real "
    "call is fine, because the import was never the problem."
)


def _examples_section(docstring: str) -> list[str] | None:
    """The body lines of the NumPy ``Examples`` section, or None when there is no header.

    SCOPED deliberately. The illustrative-block arm asks "is there an indented code line?",
    and a NumPy ``Parameters`` block is indented too -- so an unscoped version rescues
    essentially every docstring in the repo regardless of whether an example was ever
    written. Pinned by ``test_skip_only_rule_is_scoped_to_the_examples_section``.
    """
    lines = docstring.splitlines()
    start: int | None = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Examples" and set(lines[i + 1].strip()) == {"-"}:
            start = i + 2
            break
    if start is None:
        return None
    body: list[str] = []
    for j in range(start, len(lines)):
        nxt = lines[j + 1].strip() if j + 1 < len(lines) else ""
        if lines[j].strip() and nxt and set(nxt) == {"-"}:
            break  # the underline of the NEXT NumPy section header
        body.append(lines[j])
    return body


def _doctest_examples(lines: list[str]) -> list[list[str]]:
    """Group doctest input lines into examples: each ``>>>`` plus its ``...`` continuations.

    Grouped rather than judged line-by-line because a ``# doctest: +SKIP`` anywhere in a
    multi-line statement skips the WHOLE statement, which is how doctest itself reads it.
    """
    groups: list[list[str]] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(">>>"):
            groups.append([stripped])
        elif stripped.startswith("...") and groups:
            groups[-1].append(stripped)
    return groups


def _doctest_source(group: list[str]) -> str:
    """The Python source of one grouped doctest example, prompts stripped.

    Strips the three prompt characters plus the single separating space doctest requires,
    which preserves the RELATIVE indentation of a ``...`` continuation so the reconstructed
    statement still parses.
    """
    body: list[str] = []
    for line in group:
        rest = line[3:]  # drop the ">>>" / "..." prompt
        body.append(rest[1:] if rest.startswith(" ") else rest)
    return "\n".join(body)


def _demonstrates_something(group: list[str]) -> bool:
    """True when the example's code does more than name an import or talk about itself.

    An import is SETUP. ``>>> from silly_kicks.tracking import add_team_shape`` tells a reader
    nothing they could not read off the signature, and a comment (``>>> # see tests/... for a
    runnable example``) is a pointer to an example rather than one. Neither is a demonstration,
    so neither makes a section real on its own -- which is precisely how the ``+SKIP`` rule was
    being walked past: put an import on the first line and every line that actually showed the
    call could stay skipped.

    Judged by PARSING rather than by matching text, so ``from x import (a, b)`` across a
    continuation is recognised as the import it is. An unparsable body is judged a
    demonstration: this gate rules on emptiness, not on syntax.
    """
    try:
        tree = ast.parse(_doctest_source(group))
    except SyntaxError:  # pragma: no cover - a fragment doctest never meant to stand alone
        return True
    if not tree.body:
        return False  # comments and blank lines only
    return any(not isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def _has_runnable_doctest(lines: list[str]) -> bool:
    """True when at least one doctest example would actually execute AND show something."""
    for group in _doctest_examples(lines):
        if any("+SKIP" in line for line in group):
            continue
        if group[0][3:].strip() in _PLACEHOLDER_CODE:
            continue
        if not _demonstrates_something(group):
            continue
        return True
    return False


def _has_illustrative_block(lines: list[str]) -> bool:
    """True when the section carries indented non-doctest code (the usage-sketch style).

    This package's canonical example style is an RST literal block, not a doctest -- most
    entry points need a real ``actions`` frame no docstring can conjure. Those are genuine
    examples and must keep passing; the rule targets EMPTY gestures, not non-doctest ones.
    """
    doctest_idx = {i for i, line in enumerate(lines) if line.strip().startswith((">>>", "..."))}
    return any(
        line.strip() and i not in doctest_idx and (line.startswith("    ") or line.startswith("\t"))
        for i, line in enumerate(lines)
    )


def _has_real_example(docstring: str | None) -> bool:
    """True when the docstring demonstrates something a reader could run. See ``_REAL_EXAMPLE_RULE``.

    The predecessor (``_has_examples_section``) accepted any ``>>>`` line, or even a bare
    ``Examples`` header with nothing under it -- so ``>>> f(x)  # doctest: +SKIP`` satisfied
    the gate while demonstrating nothing. 16 of the 284 enforced symbols passed that way.
    """
    if not docstring:
        return False
    section = _examples_section(docstring)
    if section is None:
        # No ``Examples`` header at all: only a genuinely runnable doctest counts, since
        # there is no section to scope the illustrative-block arm to.
        return _has_runnable_doctest(docstring.splitlines())
    return _has_runnable_doctest(section) or _has_illustrative_block(section)


_DocstringEligibleNode = ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef


def _is_overload_stub(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    """True when *node* carries ``@overload`` -- a typing artifact, not a public definition.

    An overload stub's body is ``...`` by construction, so demanding an example of one demands
    something that cannot exist: its debt entry could never burn down, no matter who documented
    the implementation. Consumers call the implementation, and that is what stays judged.

    Keyed on the decorator THIS definition carries, never on its name, so the implementation
    sharing the name is unaffected. Both spellings count -- bare ``@overload`` and the dotted
    ``@typing.overload`` -- because they are the same artifact.
    """
    return any(
        (isinstance(dec, ast.Name) and dec.id == "overload")
        or (isinstance(dec, ast.Attribute) and dec.attr == "overload")
        for dec in node.decorator_list
    )


def _walk_public_definitions(tree: ast.AST) -> list[tuple[str, int, str, _DocstringEligibleNode]]:
    """Yield (kind, lineno, qualified_name, node) for top-level public defs + public methods.

    ``@overload`` stubs are skipped (see :func:`_is_overload_stub`); an overloaded function
    therefore reaches the gate exactly once, as its implementation.
    """
    out: list[tuple[str, int, str, _DocstringEligibleNode]] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS or _is_overload_stub(node):
                continue
            out.append(("function", node.lineno, node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("class", node.lineno, node.name, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") or child.name in _SKIP_SYMBOLS or _is_overload_stub(child):
                        continue
                    out.append(("method", child.lineno, f"{node.name}.{child.name}", child))
    return out


def _debt_key(file_path: str, name: str) -> str:
    """The ``_EXAMPLES_DEBT`` key for one symbol. One spelling, used by every consumer."""
    return f"{file_path}::{name}"


def _split_debt_key(key: str) -> tuple[str, str]:
    """``"<file>::<qualified_name>"`` -> ``(file, qualified_name)``.

    Split on the FIRST ``::`` so a method key (``Model.fit``) round-trips unchanged; a key
    without a separator is a module-level entry, which this bucket no longer expresses.
    """
    file_path, sep, name = key.partition("::")
    assert sep, (
        f"_EXAMPLES_DEBT key {key!r} is not '<file>::<qualified_name>'. The bucket is "
        "per-SYMBOL: a whole-module exemption would take that module's DOCUMENTED symbols "
        "off the gate as collateral, which is exactly what this replaced."
    )
    return file_path, name


def _public_symbol_names(file_path: str) -> set[str]:
    """Qualified names of every public symbol the gate judges in *file_path*."""
    tree = ast.parse((REPO_ROOT / file_path).read_text(encoding="utf-8"))
    return {name for _kind, _lineno, name, _node in _walk_public_definitions(tree)}


def _undocumented_symbols(file_path: str) -> list[tuple[str, int, str]]:
    """``[(kind, lineno, name), ...]`` for every public symbol in *file_path* lacking Examples.

    Single-sourced so the enforcing test and the debt meta-assertions judge documentation by
    the IDENTICAL rule -- a debt entry can never be "undocumented here, documented there".
    An overloaded function appears ONCE, as its implementation: ``@overload`` stubs are
    skipped by :func:`_walk_public_definitions`.
    """
    tree = ast.parse((REPO_ROOT / file_path).read_text(encoding="utf-8"))
    return [
        (kind, lineno, name)
        for kind, lineno, name, node in _walk_public_definitions(tree)
        if not _has_real_example(ast.get_docstring(node))
    ]


def _undocumented_names(file_path: str) -> set[str]:
    """The undocumented public symbol NAMES in *file_path* (the debt-key half of the pair)."""
    return {name for _kind, _lineno, name in _undocumented_symbols(file_path)}


def _excused_names(file_path: str) -> set[str]:
    """The symbol names *file_path* has an ``_EXAMPLES_DEBT`` entry for."""
    prefix = f"{file_path}::"
    return {key[len(prefix) :] for key in _EXAMPLES_DEBT if key.startswith(prefix)}


def _missing_examples(file_path: str) -> list[str]:
    """``["<file>:<lineno>  <kind>  <name>", ...]`` for every ENFORCED symbol lacking Examples.

    Individually-excused symbols are subtracted here, and that subtraction is the whole
    per-symbol point: a module with one gap keeps enforcing everything else in it.
    """
    excused = _excused_names(file_path)
    return [
        f"  {file_path}:{lineno}  {kind}  {name}"
        for kind, lineno, name in _undocumented_symbols(file_path)
        if name not in excused
    ]


@pytest.mark.parametrize("file_path", _PUBLIC_MODULE_FILES)
def test_public_definitions_have_examples_section(file_path: str):
    """Every public function / class / method in *file_path* has an Examples section.

    Except the ones individually excused in ``_EXAMPLES_DEBT`` -- an exemption costs exactly
    the symbol it names, so the rest of the module stays enforced.

    See ``silly_kicks.spadl.add_possessions`` and ``silly_kicks.spadl.boundary_metrics``
    for canonical illustrative-style examples. Add a 3-7 line example showing typical
    usage; no doctest verification is required.
    """
    assert (REPO_ROOT / file_path).exists(), f"public-API module file does not exist: {file_path}"

    missing = _missing_examples(file_path)
    assert not missing, (
        f"Public symbols in {file_path} lack a REAL Examples section:\n"
        + "\n".join(missing)
        + f"\n\n{_REAL_EXAMPLE_RULE}\n\n"
        "Add a 3-7 line example. See `silly_kicks.spadl.add_possessions` or "
        "`silly_kicks.spadl.boundary_metrics` for the canonical illustrative style, or "
        "`silly_kicks.id_compat.ids_match` for a runnable-doctest one. If the symbol "
        "genuinely cannot be demonstrated without a real match's data, write the "
        "illustrative sketch — a '+SKIP' one-liner is what this rule exists to reject, "
        "so silencing it that way will not work. Pure-type symbols (TypedDict / "
        "dataclass) that don't fit the example pattern can be added to `_SKIP_SYMBOLS` "
        "in this test file — but only with a clear documentation-policy justification. "
        "Deferring one is a per-SYMBOL `_EXAMPLES_DEBT` entry keyed "
        "'<file>::<qualified_name>' with a written note; deferring a whole module is "
        "deliberately not expressible, because that is how documented symbols used to "
        "lose their guard as collateral."
    )


# --------------------------------------------------------------------------------------
# Derivation of the public-module surface
# --------------------------------------------------------------------------------------


def _relative(path: str | None) -> str | None:
    """Repo-relative POSIX path, or None if *path* is outside the package."""
    if not path:
        return None
    resolved = pathlib.Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # pragma: no cover - a symbol re-exported from site-packages
        return None


def _public_top_level_names(file_path: pathlib.Path) -> list[str]:
    """Non-underscore top-level function / class names defined in *file_path*."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]


def _walk_public_packages() -> list[object]:
    """Import ``silly_kicks`` and every non-underscore submodule beneath it.

    Underscore modules are NOT imported directly: their public symbols reach users only via a
    package re-export, which this walk already sees through the package's ``__all__``. An
    import failure is WARNED, never swallowed -- a module invisible to the walk would be
    invisible to the meta-assertion too, which is the exact silent-gap class this gate closes.
    """
    root = importlib.import_module(_ROOT_MODULE)
    modules: list[object] = [root]
    for info in pkgutil.walk_packages(root.__path__, _ROOT_MODULE + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or "._" in info.name:
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except Exception as exc:  # pragma: no cover - an unimportable optional-extra module
            warnings.warn(
                f"public-module discovery skipped {info.name}: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
    return modules


@functools.cache
def _discover_public_modules() -> frozenset[str]:
    """The repo-relative files that contribute a symbol to the public surface (P1 UNION P2).

    See the module docstring for what each rule catches and why one alone is not enough.
    """
    found: set[str] = set()

    # P1 -- modules DEFINING an `__all__`-exported symbol.
    for module in _walk_public_packages():
        for name in getattr(module, "__all__", None) or []:
            obj = getattr(module, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if not getattr(obj, "__module__", "").startswith(_ROOT_MODULE):
                continue
            try:
                source = inspect.getsourcefile(obj)
            except TypeError:  # pragma: no cover - builtins / C types
                continue
            relative = _relative(source)
            if relative:
                found.add(relative)

    # P2 -- modules reachable by dotted path that define a public symbol.
    for file_path in PACKAGE_ROOT.rglob("*.py"):
        relative = _relative(str(file_path))
        if relative is None:
            continue  # pragma: no cover
        if any(part.startswith("_") for part in relative.split("/")[1:]):
            continue
        if _public_top_level_names(file_path):
            found.add(relative)

    return frozenset(found)


# --------------------------------------------------------------------------------------
# Meta-assertions: the registry must track the derived surface
# --------------------------------------------------------------------------------------


def test_derived_surface_is_fully_accounted_for():
    """THE anti-rot assertion.

    The hand-list had nothing pinning it to reality, so a new public module was silently
    MISSED rather than caught -- ``silly_kicks/id_compat.py`` and ``silly_kicks/gkdv/`` both
    reached the surface this release and only a human noticing kept the registry near-honest.
    Deriving the surface at run time means a newly-exported module fails CI until it is
    enforced or explicitly, justifiably deferred.
    """
    discovered = _discover_public_modules()

    unaccounted = sorted(discovered - set(_PUBLIC_MODULE_FILES))
    assert not unaccounted, (
        "public module(s) are not accounted for by this gate. Add each to "
        "_PUBLIC_MODULE_FILES; if some of its symbols cannot be documented yet, defer those "
        "SYMBOLS individually in _EXAMPLES_DEBT with a written note. Deferring a whole "
        f"module is not expressible, on purpose: {unaccounted}"
    )


def test_every_public_symbol_is_documented_or_excused():
    """FULL ACCOUNTING, at the granularity the exemptions are now written in.

    The parametrized test says this one module at a time; stating it once over the whole
    registered surface is what makes "nothing falls through" a single readable assertion
    rather than an emergent property of ~130 parametrized cases. Deleting a debt entry
    without writing the example turns this red too, so the two directions meet here.
    """
    unaccounted: list[str] = []
    for file_path in _PUBLIC_MODULE_FILES:
        excused = _excused_names(file_path)
        unaccounted += sorted(_debt_key(file_path, n) for n in _undocumented_names(file_path) - excused)
    assert not unaccounted, (
        "public symbol(s) are neither documented nor excused. Write the example, or add the "
        f"symbol to _EXAMPLES_DEBT with a written note: {unaccounted}"
    )


def test_debt_entries_are_really_undocumented():
    """The debt bucket burns itself down, one SYMBOL at a time.

    An entry whose symbol now HAS an example is excusing nothing, and leaving it in place
    would silently un-enforce that symbol again the moment someone deleted the example.
    Failing here converts "someone documented this" into a CI instruction to drop the entry.

    This is also where per-symbol DISJOINTNESS lives: a documented symbol is enforced, so an
    entry naming one would have it enforced and excused at once. That cannot survive here.
    """
    documented = sorted(
        key
        for key in _EXAMPLES_DEBT
        for file_path, name in [_split_debt_key(key)]
        if (REPO_ROOT / file_path).exists() and name in _public_symbol_names(file_path) - _undocumented_names(file_path)
    )
    assert not documented, (
        "these _EXAMPLES_DEBT symbols now carry a real Examples section, so their entry "
        "excuses nothing -- DELETE each entry, which is what puts the symbol back under "
        f"enforcement and keeps it there: {documented}"
    )


def test_debt_entries_name_real_public_symbols():
    """The mirror direction: an entry that names nothing is dead weight reading as a tracked gap.

    Three ways to become fiction, all caught here: the file is not enforced (so the entry
    excuses nothing), the file left the derived public surface (renamed, unexported), or the
    SYMBOL was renamed or deleted out from under a still-valid file. The last one is the case
    a module-level bucket structurally could not see.
    """
    debt_files = {_split_debt_key(key)[0] for key in _EXAMPLES_DEBT}

    unregistered = sorted(debt_files - set(_PUBLIC_MODULE_FILES))
    assert not unregistered, (
        "_EXAMPLES_DEBT names symbol(s) in module(s) that are not enforced at all, so the "
        f"exemption excuses nothing: {unregistered}"
    )

    not_public = sorted(debt_files - _discover_public_modules() - set(_EXTRA_COVERAGE))
    assert not not_public, (
        f"_EXAMPLES_DEBT names module(s) that are no longer public (renamed or deleted?): {not_public}"
    )

    ghosts = sorted(
        key
        for key in _EXAMPLES_DEBT
        for file_path, name in [_split_debt_key(key)]
        if name not in _public_symbol_names(file_path)
    )
    assert not ghosts, (
        "_EXAMPLES_DEBT names symbol(s) that no longer exist (renamed, deleted, or made "
        f"private). Drop or re-key each entry: {ghosts}"
    )


def test_registered_modules_are_still_public():
    """The mirror of the anti-rot assertion -- and the DERIVATION's own health check.

    An enforced entry that the derivation no longer sees is either genuinely unexported
    (rename, deletion) or, far more dangerously, a sign that the derivation itself went
    partially blind: P1 depends on IMPORTING each package, so if ``silly_kicks.xtgk`` ever
    failed to import, every underscore module it re-exports would drop out of the surface
    silently. ``test_derived_surface_is_fully_accounted_for`` cannot catch that -- a SHRINKING
    surface is always a subset of what is accounted for -- so this direction is what makes the
    import warning in :func:`_walk_public_packages` impossible to ignore.
    """
    orphaned = sorted(set(_PUBLIC_MODULE_FILES) - _discover_public_modules() - set(_EXTRA_COVERAGE))
    assert not orphaned, (
        "registered module(s) are no longer part of the derived public surface. Either they "
        "were unexported/renamed (drop or update the entry), or a package failed to import and "
        "the derivation has gone partially blind -- check the discovery warnings before adding "
        f"anything to _EXTRA_COVERAGE: {orphaned}"
    )


def test_every_exemption_carries_a_written_note():
    """A bare key in either bucket is an unenforced symbol wearing a tracked badge.

    The note is per ENTRY, so making the bucket per-symbol made the notes per-symbol too:
    "this module has some gaps" was exactly the vagueness that let four modules leave
    enforcement without anyone counting what went with them.
    """
    for bucket, label in ((_EXAMPLES_DEBT, "_EXAMPLES_DEBT"), (_EXTRA_COVERAGE, "_EXTRA_COVERAGE")):
        for key, note in bucket.items():
            assert isinstance(note, str) and len(note.strip()) >= 40, (
                f"{label}[{key!r}] needs a real written note, got {note!r}"
            )


def test_no_registered_entry_is_vacuous():
    """An enforced entry with no public top-level definition always passes trivially.

    ``gkdv/__init__.py`` was registered exactly that way -- a pure re-export file with zero
    defs -- so the parametrized test reported PASS for ``gkdv`` while all four modules that
    actually define its public surface (``_arms``/``_engine``/``_metric``/``_validate``) were
    unchecked. A vacuous entry is worse than no entry: it reads as coverage.
    """
    vacuous = sorted(f for f in _PUBLIC_MODULE_FILES if not _public_top_level_names(REPO_ROOT / f))
    assert not vacuous, (
        "registered module(s) define no public top-level symbol, so their entry asserts "
        "nothing. Register the modules that DEFINE the package's public symbols instead: "
        f"{vacuous}"
    )


def test_registry_entries_are_unique_and_disjoint():
    """No entry may be counted twice, and no exemption may explain nothing.

    Symbol-level disjointness -- excused implies NOT enforced -- is asserted by
    ``test_debt_entries_are_really_undocumented``, which is where a documented (therefore
    enforced) symbol carrying an exemption fails. What is left for here is the structural
    half: a module registered twice, and an ``_EXTRA_COVERAGE`` entry for a module nobody
    enforces, which would mask a real orphan instead of explaining one.
    """
    duplicates = sorted({f for f in _PUBLIC_MODULE_FILES if _PUBLIC_MODULE_FILES.count(f) > 1})
    assert not duplicates, f"duplicate _PUBLIC_MODULE_FILES entries: {duplicates}"

    dangling = sorted(set(_EXTRA_COVERAGE) - set(_PUBLIC_MODULE_FILES))
    assert not dangling, f"_EXTRA_COVERAGE names module(s) that are not registered: {dangling}"


def test_every_registered_file_exists():
    """A registry naming a deleted file reads as coverage of something that is gone."""
    debt_files = {_split_debt_key(key)[0] for key in _EXAMPLES_DEBT}
    absent = sorted({f for f in (*_PUBLIC_MODULE_FILES, *debt_files) if not (REPO_ROOT / f).exists()})
    assert not absent, f"registry names non-existent file(s): {absent}"


_SKIP_ONLY_DOCSTRING = """Do a thing.

Examples
--------
>>> do_a_thing(actions, frames)  # doctest: +SKIP
-1.87
"""

_IMPORT_ONLY_DOCSTRING = """Do a thing.

Examples
--------
>>> from silly_kicks.thing import do_a_thing
>>> do_a_thing(actions, frames)  # doctest: +SKIP
-1.87
"""

_COMMENT_ONLY_DOCSTRING = """Do a thing.

Examples
--------
>>> # See tests/test_thing.py for runnable examples.
"""

_IMPORT_THEN_REAL_DOCSTRING = """Do a thing.

Examples
--------
>>> from silly_kicks.thing import do_a_thing
>>> do_a_thing(2, 3)
5
"""

_RUNNABLE_DOCSTRING = """Do a thing.

Examples
--------
>>> do_a_thing(2, 3)
5
"""

_ILLUSTRATIVE_DOCSTRING = """Do a thing.

Examples
--------
Wire it into a conversion, keeping the provider's native possession id::

    actions, report = do_a_thing(events, home_team_id=100)
"""

_PLACEHOLDER_DOCSTRING = """Do a thing.

Examples
--------
>>> ...
"""

_EMPTY_SECTION_DOCSTRING = """Do a thing.

Examples
--------

Parameters
----------
x : int
    The thing.
"""


def test_skip_only_examples_do_not_count():
    """A section of nothing but ``+SKIP`` examples is filler, and must not satisfy the gate.

    ``_has_examples_section`` accepted ANY ``>>>`` line, so ``>>> anything  # doctest: +SKIP``
    ticked the box while demonstrating nothing a reader can run or a maintainer can check.
    That is the same box-ticking weakness the registry rewrite eliminated, one level down:
    the registry stopped trusting a hand-list, and this stops trusting a bare prompt.

    Both directions are pinned, so the rule cannot be satisfied by simply rejecting more:
    the skipped and placeholder forms fail, and BOTH accepted styles -- a runnable doctest
    and the illustrative literal block this gate has always treated as canonical -- pass.
    """
    assert not _has_real_example(_SKIP_ONLY_DOCSTRING), "a +SKIP-only section still counts"
    assert not _has_real_example(_PLACEHOLDER_DOCSTRING), "a bare `>>> ...` placeholder still counts"
    assert not _has_real_example(_EMPTY_SECTION_DOCSTRING), "an empty Examples header still counts"

    assert _has_real_example(_RUNNABLE_DOCSTRING), "a runnable doctest must still count"
    assert _has_real_example(_ILLUSTRATIVE_DOCSTRING), "an illustrative literal block must still count"


def test_import_or_comment_only_examples_do_not_count():
    """Setup is not a demonstration: an example that only imports or only comments is filler.

    ``_has_runnable_doctest`` accepted ANY unskipped ``>>>`` line, so a section whose only
    runnable line is ``>>> from x import y`` passed while every line that actually showed the
    call sat behind ``+SKIP``. That is the SAME filler shape the ``+SKIP`` rule exists to
    reject -- an import line is simply camouflage the rule could not see through, and it hid
    74 symbols behind it, five times the 16 the ``+SKIP`` rule caught head-on.

    Both directions are pinned, so the tightening cannot be satisfied by rejecting more:
    an import FOLLOWED by a real call still counts, because the import was never the problem.
    """
    assert not _has_real_example(_IMPORT_ONLY_DOCSTRING), "an import-only section still counts"
    assert not _has_real_example(_COMMENT_ONLY_DOCSTRING), "a comment-only section still counts"

    assert _has_real_example(_IMPORT_THEN_REAL_DOCSTRING), "an import plus a real call must still count"
    assert _has_real_example(_RUNNABLE_DOCSTRING), "a bare runnable doctest must still count"


def test_overload_stubs_are_skipped_but_the_implementation_is_not():
    """An ``@overload`` stub is a typing artifact, not a distinct public symbol.

    Its body is ``...``, so it can never carry a meaningful example -- which made its debt
    entry UNCLEARABLE, defeating the bucket's self-burning-down property: writing the example
    on the implementation could not retire an entry the stubs kept alive.

    The skip keys on the decorator carried by EACH definition, never on the name, so an
    implementation that shares a name with its stubs is still judged. That direction is the
    one worth pinning: a name-keyed skip would silently exempt the real function too.
    """
    source = """
from typing import overload

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

def f(x):
    '''The implementation, which IS judged.'''

@typing.overload
def g(x: int) -> int: ...

def g(x):
    '''Dotted-decorator spelling is the same artifact.'''
"""
    walked = _walk_public_definitions(ast.parse(source))
    assert [name for _kind, _lineno, name, _node in walked] == ["f", "g"], (
        "expected exactly the two implementations, with every @overload stub skipped"
    )
    for _kind, _lineno, name, node in walked:
        assert ast.get_docstring(node) is not None, f"the skip removed the IMPLEMENTATION of {name}, not its stubs"


def test_the_only_real_overload_set_resolves_to_its_implementation():
    """Anchor the synthetic test above to the one real overload set in the public surface.

    ``prepare_ghost_gk_training_data`` is it (two stubs plus the implementation). If a second
    one ever appears, this stays green -- the point here is that the skip picks the DOCUMENTED
    definition rather than eliminating the symbol from the gate altogether.
    """
    path = REPO_ROOT / "silly_kicks/tracking/_ghost_gk.py"
    walked = _walk_public_definitions(ast.parse(path.read_text(encoding="utf-8")))
    hits = [node for _kind, _lineno, name, node in walked if name == "prepare_ghost_gk_training_data"]
    assert len(hits) == 1, f"expected the implementation alone, got {len(hits)} definitions"
    assert _has_real_example(ast.get_docstring(hits[0])), "the surviving definition should be the documented one"


def test_skip_only_rule_is_scoped_to_the_examples_section():
    """The literal-block arm reads the Examples SECTION, not the whole docstring.

    A NumPy ``Parameters`` block is indented too, so an unscoped "is there an indented
    code line?" test rescues essentially every docstring in the repo -- it silently
    reduced a 16-symbol offender set to 13 while drafting this rule. The section bound is
    what makes the literal-block arm mean "an example was written" rather than "a
    docstring was written".
    """
    with_params = """Do a thing.

Parameters
----------
x : int
    An indented line that is NOT an example.

Examples
--------
>>> do_a_thing(x)  # doctest: +SKIP
"""
    assert not _has_real_example(with_params), "Parameters indentation rescued a +SKIP-only section"


def test_derivation_sees_both_rules():
    """Non-vacuity for the derivation itself.

    P1 and P2 catch structurally different modules, and a derivation that silently collapsed
    to one rule would re-open half the hole. Pin one known member of each: ``_ghost_gk.py`` is
    underscore-named and reachable ONLY as a re-export (P1), while ``spadl/statsbomb.py`` has
    no ``__all__`` re-export and is reachable only by dotted path (P2).
    """
    discovered = _discover_public_modules()
    assert "silly_kicks/tracking/_ghost_gk.py" in discovered, "P1 (re-export) rule found nothing"
    assert "silly_kicks/spadl/statsbomb.py" in discovered, "P2 (dotted-path) rule found nothing"
