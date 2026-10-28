"""Spec r3 D8: the EXACT registry gate on declared serial reasons (the ADR-056 pattern: derive the population, assert
it exactly).

The R1 audit passes a stage below the box's vCPU only through one of the spec's two structural shapes (a final fit
at the full thread budget; a sequential study at its fold width) or a REVIEWED serial reason. A stage declares its
reason as `Stage.serial_reason`; every declaration must equal its entry in `scripts/hf_jobs/serial_reasons.py`
(reviewed with the plan's R3-8 decision table), and no entry may outlive its declaration. So a stage can never
grant itself an R1 floor with an unreviewed reason (A r4 R4-IMPL-01).
"""

from __future__ import annotations

import dataclasses
import importlib

from scripts.hf_jobs.serial_reasons import SERIAL_REASONS
from tests.scripts.test_compute_contract import _compliant


def serial_reasons_of(pipeline) -> dict[tuple[str, str], str]:
    return {(pipeline.module, st.name): st.serial_reason for st in pipeline.stages if st.serial_reason is not None}


def declared_serial_reasons() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for name in _compliant():
        out.update(serial_reasons_of(importlib.import_module(f"scripts.{name}").PIPELINE))
    return out


def test_every_serial_stage_declares_a_reviewed_reason():
    assert declared_serial_reasons() == SERIAL_REASONS


def test_the_gate_fails_an_unreviewed_declaration_and_a_stale_entry():
    """Anti-rot (ADR-051): the comparison the gate makes really fails both ways."""
    from tests.scripts._toy_pipeline import PIPELINE

    gen = PIPELINE.stage("gen")
    declared = dataclasses.replace(
        PIPELINE,
        stages=tuple(
            dataclasses.replace(st, serial_reason="one corpus item at a time") if st is gen else st
            for st in PIPELINE.stages
        ),
    )
    unreviewed = serial_reasons_of(declared)
    assert unreviewed == {("tests.scripts._toy_pipeline", "gen"): "one corpus item at a time"}
    assert unreviewed != SERIAL_REASONS  # a declaration with no reviewed entry fails the gate
    stale = {**SERIAL_REASONS, ("tests.scripts._toy_pipeline", "gen"): "no longer declared"}
    assert serial_reasons_of(PIPELINE) != stale  # an entry with no declaration fails the gate


def test_a_blank_serial_reason_is_rejected_at_validation():
    import pytest

    from tests.scripts._toy_pipeline import PIPELINE

    blank = dataclasses.replace(
        PIPELINE,
        stages=tuple(dataclasses.replace(st, serial_reason="  ") if st.name == "gen" else st for st in PIPELINE.stages),
    )
    with pytest.raises(ValueError, match="serial_reason"):
        blank.validate()


def test_the_plan_reports_each_stage_serial_reason(tmp_path):
    """The campaign's R1 audit reads the reason from the plan JSON (`stage_summary`)."""
    from scripts._stages import LocalStore, make_context, plan
    from tests.scripts._stage_harness import params_from_args
    from tests.scripts._toy_pipeline import PIPELINE

    reasoned = dataclasses.replace(
        PIPELINE,
        stages=tuple(
            dataclasses.replace(st, serial_reason="reviewed") if st.name == "env" else st for st in PIPELINE.stages
        ),
    )
    ctx = make_context(
        reasoned,
        params_from_args(reasoned, []),
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=None,
    )
    reasons = {s["name"]: s["serial_reason"] for s in plan(ctx)["stages"]}
    assert reasons["env"] == "reviewed" and reasons["gen"] is None
