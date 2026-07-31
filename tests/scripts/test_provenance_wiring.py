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

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

# Drivers that write a registered artifact (metrics.json / parquet / report.md under --out).
# Listed rather than inferred: "writes an artifact" is a semantic property, and a heuristic over
# `write_text` would sweep in dev utilities whose output nobody cites.
ARTIFACT_DRIVERS = (
    "build_gkdv_arm_values",
    "calibrate_xt_bandwidth",
    "measure_cover_shadow_argmax_agreement",
    "build_layer2_spells",
    "derive_opengoal_range",
    "run_signoff_power",
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
    """Meta-assertion: a parametrised gate over a list that drifted to nothing passes vacuously."""
    assert len(ARTIFACT_DRIVERS) >= 6
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
    guard, walk = _calls_in(main_fn, "require_clean_tree"), _calls_in(main_fn, "load_matches")
    if not walk:
        pytest.skip("corpus walk is delegated out of main(); the entry-point gate covers it")
    assert min(guard) < min(walk), (
        f"{name}.py starts the corpus walk at line {min(walk)} before checking the tree at "
        f"{min(guard)} -- the check must come first or it protects nothing."
    )
