"""Every artifact-writing driver must be wired to the fail-closed provenance guard.

CLAUDE.md states the rule: any `scripts/` driver that writes a registered artifact calls
`require_clean_tree(git_provenance(), ...)` FIRST, before paying for any corpus work, and stamps
`run_commit` + `run_tree_dirty` into its output.

That rule was enforced by MEMORY until now, and memory had already failed twice: `validate_xshot_causal.py`
wrote the S3.3 entanglement artifact with no provenance at all, and `validate_xs_probe.py` stamped a
bare `git rev-parse HEAD` -- which returns the same SHA whether or not the tree is modified, i.e.
the exact false-provenance pattern `scripts/_provenance.py` exists to eliminate. Both produced
CITED research artifacts. A hand-run audit found them; this gate is what stops the third one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.scripts._script_population import SCRIPTS, called_names, iter_scripts, string_literals

_SCRIPTS = SCRIPTS  # single-sourced with the shared population seam (Cycle B)

# Drivers that write a registered artifact (metrics.json / parquet / report.md under --out).
# Listed rather than inferred: "writes an artifact" is a semantic property, and a heuristic over
# `write_text` would sweep in dev utilities whose output nobody cites.
ARTIFACT_DRIVERS = (
    # Enrolled with the SB360 coverage audit. Measures real StatsBomb 360 freeze-frame
    # coverage, and its keeper-visibility numbers go in front of a club -- exactly the
    # "cited, uncheckable" shape the rule below exists to prevent.
    #
    # Its sibling `render_sb360_matrix` is deliberately NOT enrolled; the reason now lives in
    # `_NOT_A_DRIVER` below, where the completeness gate can check it, rather than in a comment
    # here that nothing reads (Cycle B).
    "build_sb360_coverage",
    # The ADR-050 §6 ghost box-constant closure. Both consume the pining corpus (external data) and
    # write documents whose numbers are cited: `materialize_tc3_frames` produces the corpus cache
    # the re-fit trains on, and `measure_box_constant_delta` produces the flip count that decides
    # the release's ship claim ("unification, measured no-op" vs a weights comparison).
    "materialize_tc3_frames",
    "measure_box_constant_delta",
    # The cover-shadow RQ1 + pass-risk validation cycle (2026-08-19). `build_rq_pass_scores` is the
    # expensive corpus pass (per-pass GS scores -> gitignored pass_scores.parquet); the two consumers
    # read that persisted table and write the cited docs/research/ artifacts.
    "build_rq_pass_scores",
    "validate_cover_shadow_rq1",
    "validate_pass_risk_calibration",
    # The cover-shadow sigma/lambda re-tune + expected-receiver cycle (2026-08-20). `train_receiver_model`
    # trains the SB360/GS receiver bundle (writes model + metrics.json); `apply_cover_shadow_retune` writes
    # the gated per-provider apply decision -- both consume external data / cite a number, so provenance.
    "train_receiver_model",
    "apply_cover_shadow_retune",
    # TF-24 Stage-1 confirmation. Writes `docs/research/tf24_stage1_confirmation/metrics.json`,
    # whose verdict decides whether the recorded carrier optimum stands or a full Stage-1
    # sweep is owed -- a cited number, so the tree it ran on has to be recorded.
    "check_stage1_argmax",
    "build_gkdv_arm_values",
    "calibrate_xt_bandwidth",
    "measure_cover_shadow_argmax_agreement",
    "build_layer2_spells",
    "derive_opengoal_range",
    "run_signoff_power",
    # Enrolled 4.74.0 (PR 5). It wrote docs/research/xcross_causal/metrics.json with NO
    # run_commit and NO run_tree_dirty -- the third instance of the class this gate exists for,
    # and one the gate could not see because ARTIFACT_DRIVERS is hand-maintained and its only
    # anti-rot assertion is a FLOOR (>= 6 against 14 entries). Cycle B replaces that floor.
    "validate_xcross_causal",
    # Enrolled 4.74.0 (PR 5). Two NEW drivers, both written in commit 3 and both run in
    # commit 4 -- a driver cannot run in the same commit that introduces it, because writing
    # it is what makes the tree dirty.
    "measure_covariate_invariance",
    "measure_platform_probe",
    # Enrolled 4.73.0 (PR-S141). Its predecessor was an ad-hoc pass that shipped a
    # `tracking_limit=3000` cap recorded NOWHERE, halving a published headline -- so the artifact was
    # cited, uncheckable, and wrong. A committed driver with real provenance is the fix.
    "measure_rc4_orientation",
    # --- The five weight TRAINERS, enrolled together in 4.72.0 (ADR-052) ---
    # `train_ghost_gk` stamped `training_commit` into the SHIPPED metadata.json from a bare
    # `git rev-parse HEAD`, which reads identically on a modified tree: a bundled weights file
    # carrying a verifiable-looking claim about code that may never have existed at that commit.
    # The other four made no false claim -- they recorded NOTHING -- which is a different failure
    # and not a lesser one: an artifact nobody can trace back to a commit cannot be reproduced or
    # audited. Enrolled in ONE go deliberately. A partial roll-out is exactly how the prose version
    # of this rule failed twice, and it is why this gate exists at all.
    "train_ghost_gk",
    "train_gk_completion",
    "train_gk_retention",
    "train_xcross_attempt",
    "train_xshot_occurrence",
    "validate_xs_probe",
    "validate_xshot_causal",
    # --- Enrolled by Cycle B (version assigned at commit-prep). All three were found by item 10's
    # FIRST run: each consumes data from outside the repository, writes an artifact, and had no
    # provenance guard at all -- the same class as `validate_xcross_causal` above, and invisible
    # for exactly the same reason (the old anti-rot assertion was a FLOOR).
    #
    # Databricks read-only gold marts (`_loader_databricks.load_xtgk_cohort`).
    "validate_xtgk_possession_value",
    # StatsBomb open data via statsbombpy, plus the pining corpus (`load_matches`).
    "validate_shot_goalmouth_sb",
    # The pining / DGX tracking corpus (`_loader_pining`).
    "calibrate_tracking_defaults",
    # Item 23 step 2a. Owner-tier Gradient Sports via the pining loader; emits COUNTS only, never
    # coordinates. Enrolled the moment it was written -- this gate flagged it on its first run,
    # which is the self-test the spec named: the cycle's own completeness gate catching the
    # cycle's own new driver before a human noticed.
    "measure_gs_shot_distribution",
    # Keeper-box geometry & detection-quality cycle. Validates the shipped SkillCorner keeper-origin
    # resolver on the pining corpus (`load_matches`) and writes `manifest_all.json` whose
    # offpitch_rate / out_of_region_goalkick_rate become the ADR-024 CI rate-gate baselines -- cited
    # numbers, so the tree they ran on has to be recorded.
    "validate_skillcorner_keeper_origin",
    # The SB360 licensed-corpus validation driver. Consumes the licensed StatsBomb 360 corpus via
    # the pining loader (`load_statsbomb_matches`) and writes cited coverage/verdict numbers to
    # docs/research/sb360_licensed_coverage/ -- exactly the "external data -> cited artifact" class
    # this registry enrolls. Licensed per-match shards go to a gitignored root, never the aggregate.
    "validate_sb360_licensed_corpus",
    # The position-only-variants cycle. Consumes the two trainer metrics.json (faithful vs
    # position_only) and writes the velocity-vs-position_only skill-delta artifact to
    # docs/research/position_only_variants/ -- a CITED number (the ADR's "reported comparability
    # cost"), so the tree it ran on is recorded. Flagged by this gate on its first run.
    "compare_position_only_variants",
    # The TF-19 A+2 physics-arm instrument-validity + responsiveness cycle. Walks the pining
    # corpus (`load_matches`) and writes docs/research verdicts (Layer-0/1 verdicts, the §6.1
    # gate_eligible census, the §6.2 named-keeper sign table) whose numbers are cited -- so the
    # tree it ran on is recorded. Reported-not-gated, but a cited artifact all the same.
    "build_tf19_instrument_responsiveness",
)


#: Matched the derivation rule but is correctly NOT an artifact driver; reason required.
#:
#: The discriminator is NOT "consumes external data" -- `build_worldcup_fixture` downloads
#: StatsBomb open data and is still correctly here. It is: **does the output carry numbers
#: someone cites?** A test fixture does not; it is verified by being committed and by the tests
#: that read it, and its generator must stay runnable on a DIRTY tree, because a fixture is
#: regenerated exactly when the code consuming it is being changed. That second prong is
#: `render_sb360_matrix`'s recorded reason, and it generalises to this whole class.
_NOT_A_DRIVER: dict[str, str] = {
    "render_sb360_matrix": (
        "reads a COMMITTED registry and writes a document. It does no corpus work and consumes no "
        "external data, so the guard would add nothing and would make the report unrenderable "
        "during the session that produces it -- a guarded driver cannot run on the dirty tree that "
        "produces its own inputs. The script's own docstring line 3 says the same."
    ),
    "build_worldcup_fixture": (
        "writes the committed WC2018 test FIXTURE (spadl-WorldCup-2018.h5), not a cited artifact. "
        "It does download StatsBomb open data, so it is not exempt for lack of an external source "
        "-- it is exempt because a fixture regenerated during a conversion change must run on the "
        "dirty tree carrying that change."
    ),
    "make_xcross_directional_fixture": (
        "builds the committed frozen directional feature-vector fixture for the xCross CI gates "
        "from SYNTHETIC states. Output is a test input, not a cited number, and it is rebuilt "
        "precisely when the extractor it feeds is being changed."
    ),
    "regenerate_gs_et_native_gk": (
        "regenerates the committed gs_et ET tracking fixture from the local pining CACHE (no "
        "network, no token). Output is a test input, not a cited number."
    ),
    "audit_velocity_fixtures": (
        "reads COMMITTED test sources and runs library code on a synthetic frame -- no corpus "
        "pass, no external data, nothing whose provenance could be misattributed. Its `--out` JSON "
        "is a development report, not a cited number: every figure in it is re-derived on demand "
        "by `tests/scripts/test_audit_velocity_fixtures.py`, which is the standing gate, so a "
        "stale copy on disk cannot be mistaken for evidence. Same reasoning as "
        "`render_sb360_matrix` above, and the script's own docstring states it."
    ),
    "stamp_feature_contracts": (
        "rewrites bundled metadata ONLY, deliberately never calling any model's save(). It "
        "consumes nothing external and derives its contract from the current library, so it must "
        "be re-run after a change to a declared constant -- i.e. on the dirty tree carrying that "
        "change. Its OUTPUT is still policed, by the Cycle B artifact-provenance gate; the "
        "source-side guard and the output-side gate answer different questions and only the "
        "second applies here."
    ),
    "render_sb360_licensed_coverage": (
        "reads the COMMITTED licensed coverage parquet + manifest and writes coverage.md. It does "
        "no corpus work and consumes no external data, so a provenance guard would add nothing and "
        "would make the report unrenderable during the session that produces it. Same class as "
        "`render_sb360_matrix`; provenance travels by reference to the manifest it stamps."
    ),
}

#: Genuinely a driver, enrolled, but invisible to the rule. MUST be empty on landing -- see
#: test_UNDERIVABLE_is_empty. The reason must say WHY it is invisible, so the entry can be
#: retired if the rule improves.
_UNDERIVABLE: dict[str, str] = {}

#: Calls that persist something to disk. Keyed on the CALL, never on a filename literal.
_WRITE_CALLS = frozenset(
    {
        "write_text",
        "write_bytes",
        "to_parquet",
        "to_csv",
        "to_json",
        "savez",
        "savez_compressed",
        "write_table_atomically",
        "dump",
    }
)


def _declares_an_out_flag(tree: ast.AST) -> bool:
    """Any `--*out*` flag, not just an `--out` prefix.

    Measured: the prefix rule missed `--report-out` (calibrate_xt_bandwidth) and `--output-dir`.
    """
    return any(s.startswith("--") and "out" in s for s in string_literals(tree))


def _writes_a_document(tree: ast.AST) -> bool:
    """Detect the WRITE, not the filename.

    An earlier draft matched a `.json`/`.md` suffix literal. Measured, that fails in BOTH
    directions: it MISSES `measure_cover_shadow_argmax_agreement`, which composes its path
    entirely from `args.out` and carries no suffix literal anywhere, and -- worse -- it misses
    `render_sb360_matrix`, the one script the spec names as the counter-example that MUST be a
    candidate so it can be excluded with a reason.
    """
    return bool(called_names(tree) & _WRITE_CALLS)


def _writes_bundled_weights(tree: ast.AST) -> bool:
    """The trainers name a bundled weights path instead of an out-flag. Without this clause the
    three trainers are underivable and the central assertion cannot hold."""
    return any("_weights" in s for s in string_literals(tree))


def _is_artifact_driver(tree: ast.AST) -> bool:
    return (_declares_an_out_flag(tree) and _writes_a_document(tree)) or _writes_bundled_weights(tree)


def _candidates() -> set[str]:
    return {n for n, tree in iter_scripts().items() if _is_artifact_driver(tree)}


def test_the_artifact_driver_population_is_EXACT():
    """Replaces `assert len(ARTIFACT_DRIVERS) >= 6`. A floor cannot detect an omission -- it passed
    at 18 entries while `render_sb360_matrix` and `validate_xcross_causal` were both missing."""
    expected = (set(ARTIFACT_DRIVERS) - set(_UNDERIVABLE)) | set(_NOT_A_DRIVER)
    missing = sorted(_candidates() - expected)
    stale = sorted(expected - _candidates() - set(_UNDERIVABLE))
    assert not missing, (
        f"scripts that look like artifact drivers but are enrolled nowhere: {missing}. Add them to "
        f"ARTIFACT_DRIVERS, or to _NOT_A_DRIVER with a reason."
    )
    assert not stale, f"enrolled but no longer derivable: {stale} -- record in _UNDERIVABLE with a reason"


def test_UNDERIVABLE_is_empty():
    """The blind spot this closes: a script that is neither derivable NOR enrolled is absent from
    every set here and the equality above still holds. That is only unreachable while every enrolled
    driver IS derivable. The day it stops being true, this says so."""
    assert not _UNDERIVABLE, (
        f"_UNDERIVABLE is non-empty: {sorted(_UNDERIVABLE)}. A driver invisible to the rule means a "
        f"NEW driver of the same shape would also be invisible -- broaden the rule instead."
    )


@pytest.mark.parametrize("bucket_name", ["_NOT_A_DRIVER", "_UNDERIVABLE"])
def test_exemptions_name_scripts_that_exist(bucket_name):
    """Self-burning-down, the way _UNMODELLED already is in the C4 gate."""
    bucket = {"_NOT_A_DRIVER": _NOT_A_DRIVER, "_UNDERIVABLE": _UNDERIVABLE}[bucket_name]
    stale = sorted(n for n in bucket if not (_SCRIPTS / f"{n}.py").is_file())
    assert not stale, f"{bucket_name} names scripts that no longer exist: {stale}"


def test_both_population_gates_consume_the_SHARED_universe():
    """The reconciliation is structural, not a set relation.

    Asserting "every corpus-walking artifact driver is in ADR-052's population" is TAUTOLOGICAL
    once both gates call the same predicate over the same script set -- it cannot fail. What CAN
    fail is one gate re-growing its own glob, after which the two universes drift and nothing else
    here would notice.
    """
    import tests.scripts.test_corpus_driver_resilience as adr052

    src = pathlib.Path(adr052.__file__).read_text(encoding="utf-8")
    assert "iter_scripts" in src, "ADR-052's gate no longer consumes the shared universe"
    assert ".glob(" not in src, (
        "the corpus-driver gate re-grew its own glob over scripts/ -- route it through "
        "tests/scripts/_script_population.py or the two populations can drift apart silently"
    )


def test_the_population_rule_reads_CODE_not_PROSE():
    """Non-vacuity, against the real case.

    `make_ghost_gk_golden` carries ZERO `_weights` string literals in code -- its only match is a
    module docstring mentioning `test_weights_bundle_golden.py` while explaining why an output
    golden exists. A literal scan that includes docstrings enrols it as a candidate on the strength
    of a sentence of prose, and it would then need a bogus exemption.
    """
    tree = iter_scripts()["make_ghost_gk_golden"]
    assert not any("_weights" in s for s in string_literals(tree)), (
        "string_literals() is admitting docstrings again -- source-text heuristics cannot tell a "
        "described path from a written one"
    )
    raw = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert any("_weights" in s for s in raw), (
        "fixture drifted: this test is only meaningful while the docstring still mentions a "
        "_weights path -- re-point it at another prose-only match or delete it"
    )


def test_the_rule_FLAGS_an_unenrolled_driver():
    """Non-vacuity, against the real case. `validate_xcross_causal` (4.74.0) had `--out`, wrote
    metrics.json, was absent from the tuple, and its artifact carried no provenance at all. If the
    rule cannot see it un-enrolled, it would not have prevented the thing it was built for."""
    tree = iter_scripts()["validate_xcross_causal"]
    assert _is_artifact_driver(tree)
    assert "validate_xcross_causal" in _candidates() - (
        (set(ARTIFACT_DRIVERS) - {"validate_xcross_causal"}) | set(_NOT_A_DRIVER)
    )


def _source(name: str) -> str:
    return (_SCRIPTS / f"{name}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)
def test_driver_imports_the_shared_provenance_helper(name):
    assert "_provenance" in _source(name), (
        f"{name}.py writes a registered artifact but never imports scripts._provenance. "
        "A bare `git rev-parse HEAD` is NOT provenance: it reads clean on a modified tree."
    )


@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)
def test_driver_offers_an_allow_dirty_escape_hatch(name):
    """The hatch must exist (a dev run is legitimate) and its artifact stays marked -- so the
    absence of `--allow-dirty` is itself evidence the guard was never wired."""
    assert "--allow-dirty" in _source(name), f"{name}.py has no --allow-dirty flag"


def _shells_out_to_rev_parse(src: str) -> bool:
    """True only for an actual CALL passing "rev-parse", never for prose mentioning it.

    A plain substring scan flagged this module's own explanatory docstring, which is the standard
    failure of source-text heuristics: they cannot tell a described defect from a committed one.
    """
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "rev-parse" in arg.value:
                return True
    return False


@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)
def test_driver_never_shells_out_to_rev_parse_directly(name):
    """The whole point of the shared helper. A local `git rev-parse HEAD` bypasses the dirty check
    and re-creates the false-provenance bug in a place the guard above cannot see."""
    assert not _shells_out_to_rev_parse(_source(name)), (
        f"{name}.py calls `git rev-parse` directly -- route it through scripts._provenance, "
        "whose git_provenance() reports the dirty flag alongside the SHA."
    )


def test_the_rev_parse_detector_distinguishes_a_CALL_from_PROSE():
    """Non-vacuity: the detector must fire on the real thing and stay silent on a mention, or it is
    either useless or unusable."""
    assert _shells_out_to_rev_parse('subprocess.check_output(["git", "rev-parse", "HEAD"])')
    assert not _shells_out_to_rev_parse('"""We must never call git rev-parse HEAD directly."""')


def test_the_driver_list_is_not_silently_empty_or_stale():
    """Burn-down half: an entry naming a script that no longer exists is stale scaffolding.

    The `assert len(ARTIFACT_DRIVERS) >= 6` that used to open this test is GONE (Cycle B). A floor
    cannot detect an omission -- it passed at 18 entries while three unguarded artifact drivers and
    `render_sb360_matrix` were all missing. `test_the_artifact_driver_population_is_EXACT` replaces
    it and fails in BOTH directions.
    """
    for name in ARTIFACT_DRIVERS:
        assert (_SCRIPTS / f"{name}.py").is_file(), f"{name}.py no longer exists -- update the list"


def _calls_in(fn: ast.FunctionDef, name: str) -> list[int]:
    return [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name]


def _function(src: str, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)
def test_the_ENTRY_POINT_enforces_the_clean_tree(name):
    """`require_clean_tree` must be called from `main()`, the CLI entry point.

    An earlier version of this gate compared LINE NUMBERS of the guard and the corpus walk across
    the whole module -- which measures definition order, not execution order. It reported a driver
    as unguarded purely because `main()` is defined at the bottom of the file, below the `run()` it
    calls. Enforcing "the guard is in main" checks the property that actually matters: no CLI
    invocation can reach expensive work without passing the check first.
    """
    main_fn = _function(_source(name), "main")
    assert main_fn is not None, f"{name}.py has no main()"
    assert _calls_in(main_fn, "require_clean_tree"), (
        f"{name}.py never calls require_clean_tree from main() -- a CLI run could write a "
        "registered artifact from a dirty tree."
    )


@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)
def test_the_guard_precedes_the_corpus_walk_within_main(name):
    """Where BOTH calls live in `main`, ordering is directly checkable and must hold: the 8.7h loss
    happened because expensive work ran before anything validated it."""
    main_fn = _function(_source(name), "main")
    assert main_fn is not None
    guard = _calls_in(main_fn, "require_clean_tree")
    # The walk = a direct corpus-loader call OR the sharded `for_each` itself (the actual expensive pass).
    # train_receiver_model delegates its loader to a helper but drives `for_each` in main(), so a
    # loader-name-only grep would silently SKIP its ordering check (L4); `for_each` is the real invariant.
    walk = (
        _calls_in(main_fn, "load_matches")
        + _calls_in(main_fn, "load_statsbomb_matches")
        + _calls_in(main_fn, "for_each")
    )
    if not walk:
        pytest.skip("corpus walk is delegated out of main(); the entry-point gate covers it")
    assert min(guard) < min(walk), (
        f"{name}.py starts the corpus walk at line {min(walk)} before checking the tree at "
        f"{min(guard)} -- the check must come first or it protects nothing."
    )


# --------------------------------------------------------------------------------------------
# Platform identity: an artifact must record WHERE it was produced, not only WHAT code ran.


def test_git_provenance_records_the_platform():
    """`run_commit` says what code ran; it does not say on what machine.

    That gap became load-bearing when ghost was re-fit on DGX Spark (aarch64) while the feature
    contract's tolerance note still asserted "every fingerprinted artifact is produced on x86". A
    contract mismatch on a cross-platform artifact must be diagnosable as "wrong machine" from the
    artifact itself, not by re-deriving where it came from.

    It lives in `git_provenance` for the ADR-037 reason the commit does: it is the ONE seam every
    artifact driver already calls, so no driver can forget it.
    """
    from scripts._provenance import git_provenance

    prov = git_provenance()
    assert "platform" in prov, "git_provenance must record the platform"
    assert isinstance(prov["platform"], str) and prov["platform"], "platform must be a non-empty string"
    assert "machine" in prov, "the ISA is what actually differs (aarch64 vs x86_64); record it separately"
    assert isinstance(prov["machine"], str) and prov["machine"]


@pytest.mark.parametrize("failing_call", ["rev-parse", "status"])
def test_platform_survives_a_git_failure(monkeypatch, failing_call):
    """Platform identity does not come from git, so a box without git must still report it.

    `git_provenance` has THREE return statements and two of them are git-failure paths -- one when
    `rev-parse` fails (no git, not a repo) and one when `status` fails after `rev-parse` succeeded.
    Both are parametrised here, because a `platform` merged into only the happy path would leave
    exactly the unprovenanced runs (tarball checkouts, CI images) least able to say where they ran,
    and a single-path test would not see it.
    """
    import scripts._provenance as prov_mod

    def _selective_boom(*args, **kwargs):
        if args and args[0] == failing_call:
            raise FileNotFoundError(f"git {failing_call} unavailable")
        return "0" * 40 if args and args[0] == "rev-parse" else ""

    monkeypatch.setattr(prov_mod, "_git", _selective_boom)
    prov = prov_mod.git_provenance()

    assert prov["tree_state"] == "unknown", "a git failure must not report a known tree state"
    assert prov["dirty"] is True, "unknown provenance is never treated as clean"
    # `.get`, not `[...]`: a dropped key raises KeyError BEFORE the assertion, so the message
    # explaining the defect never reaches the reader. Verified by planting the regression.
    assert prov.get("platform"), f"platform dropped on the {failing_call}-failure path"
    assert prov.get("machine"), f"machine dropped on the {failing_call}-failure path"
