"""T5 registry gate: every ``thread_safe`` fit stage is committed with its lever, its lever is covered by the
thread-identity gate, and the xgb trainers actually wire the runtime budget into ``fit`` (B27).

Fail-closed (ADR-056 derive-and-assert-exactly): the population is DERIVED by scanning the trainer pipelines,
then asserted to equal the committed registry. A new ``thread_safe`` opt-in that is not registered here fails,
so a stage cannot start using its full vCPU share without a committed cross-thread-count byte-identity test.
"""

from __future__ import annotations

import importlib
import pathlib

TRAINERS = (
    "scripts.train_ghost_gk",
    "scripts.train_ghost_outfield",
    "scripts.train_xshot_occurrence",
    "scripts.train_xcross_attempt",
)

#: The committed opt-in set: each thread_safe fit stage -> (its lever, the NAMED identity test in
#: tests/tracking/test_thread_identity.py that proves ITS model byte-identical across thread counts). A stage
#: added without its own named identity test fails the gate below (B42 -- a lever-name-only check passed
#: vacuously while ghost_outfield ran 8 threads unproven).
#: Each entry: (lever, the named identity test, the model class it MUST fit). The model-class check (B56)
#: closes the "right test name, wrong model" gap -- a binding by name alone could pass while the test fits
#: another model.
COMMITTED_THREAD_SAFE = {
    ("scripts.train_ghost_gk", "fit"): (
        "omp",
        "test_ghost_gk_histgb_byte_identical_across_omp_threads",
        "GhostGkModel",
    ),
    ("scripts.train_ghost_outfield", "fit"): (
        "omp",
        "test_ghost_outfield_histgb_byte_identical_across_omp_threads",
        "GhostOutfieldModel",
    ),
    ("scripts.train_xshot_occurrence", "fit"): (
        "n_jobs",
        "test_xshot_xgb_byte_identical_across_runtime_n_jobs",
        "XShotOccurrenceModel",
    ),
    ("scripts.train_xcross_attempt", "fit"): (
        "n_jobs",
        "test_xcross_xgb_byte_identical_across_runtime_n_jobs",
        "XCrossAttemptModel",
    ),
    # HFU T11: the cv stages opt in too -- each fits its model per fold, so each gets its OWN cv-result
    # byte-identity arm (per-stage, fail-closed).
    ("scripts.train_ghost_gk", "cv"): (
        "omp",
        "test_cv_thread_identity_ghost_gk",
        "GhostGkModel",
    ),
    ("scripts.train_ghost_outfield", "cv"): (
        "omp",
        "test_cv_thread_identity_ghost",
        "GhostOutfieldModel",
    ),
    ("scripts.train_xshot_occurrence", "cv"): (
        "n_jobs",
        "test_cv_thread_identity_xgb",
        "XShotOccurrenceModel",
    ),
    ("scripts.train_xcross_attempt", "cv"): (
        "n_jobs",
        "test_cv_thread_identity_xcross",
        "XCrossAttemptModel",
    ),
}


def _declared_thread_safe() -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for mod in TRAINERS:
        pipe = importlib.import_module(mod).PIPELINE
        for st in pipe.stages:
            if getattr(st, "thread_safe", False):
                found[(mod, st.name)] = st.thread_lever
    return found


def test_thread_safe_stages_match_the_committed_registry():
    # exact equality: a new opt-in not registered here (or a registered one removed) fails loud
    assert _declared_thread_safe() == {k: lever for k, (lever, _test, _model) in COMMITTED_THREAD_SAFE.items()}


def test_every_thread_safe_stage_has_its_own_named_identity_test():
    # B42: bind each (trainer, stage) to a SPECIFIC identity test that fits ITS model across thread counts,
    # and fail if that test is missing (B56: and if it does not actually fit that stage's model).
    import inspect

    import tests.tracking.test_thread_identity as ti

    present = set(dir(ti))
    for k, (_lever, test, model) in COMMITTED_THREAD_SAFE.items():
        assert test in present, f"{k}: committed identity test {test!r} is missing"
        src = inspect.getsource(getattr(ti, test))
        assert model in src, f"{k}: identity test {test!r} does not fit {model} (B56: name-only binding is unsafe)"


def test_xgb_trainers_wire_the_runtime_budget_into_fit():
    # B27: without this the xgb fit stays 1-thread while T3's kwarg test passes trivially
    for mod in ("train_xshot_occurrence", "train_xcross_attempt"):
        src = (pathlib.Path("scripts") / f"{mod}.py").read_text(encoding="utf-8")
        assert "runtime_n_jobs_from_env" in src, f"{mod}: trainer does not read the runtime budget"
        assert "runtime_n_jobs=runtime_n_jobs_from_env()" in src, f"{mod}: budget not passed into fit()"


def test_registry_is_per_stage():
    # HFU T11 S4/B-06: the gate tracks each (trainer, stage) INDEPENDENTLY -- a thread_safe cv stage is its own
    # committed entry with its own identity arm, never covered by the final fit's. Exact equality, not a subset.
    declared = _declared_thread_safe()
    committed = {k: lever for k, (lever, _t, _m) in COMMITTED_THREAD_SAFE.items()}
    assert declared == committed  # every declared thread_safe stage is committed, and vice versa
    for mod in TRAINERS:
        assert (mod, "cv") in committed and (mod, "fit") in committed  # cv and fit tracked separately per trainer
    dropped = dict(committed)
    dropped.pop(next(iter(dropped)))  # remove ONE stage's entry
    assert declared != dropped  # a stage missing its own entry fails the exact-equality gate (per-stage, not global)
