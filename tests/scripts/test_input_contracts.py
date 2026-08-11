"""Declared input contracts: the mechanism, and the detector that reports stale artifacts (Cycle B).

PR 5 changed a geometry transform and three research artifacts silently went stale. The only thing
that found them was someone tracing `causal/opportunities.py` by hand, after seven review rounds had
missed it. This is what that tracing should have been.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import warnings

import pytest

from scripts._input_contract import contract_digest, declare_inputs

# --------------------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------------------


def test_the_digest_moves_when_a_declared_symbol_changes():
    """The whole mechanism. If this does not hold, a stale artifact reads as current."""
    before = declare_inputs(driver="d", covariates={"arm": ("a", "b")}, geometry_version="goal-relative-2")
    after = declare_inputs(driver="d", covariates={"arm": ("a", "b", "c")}, geometry_version="goal-relative-2")
    assert before["digest"] != after["digest"]


def test_the_digest_is_stable_across_key_order_and_set_iteration():
    """A digest that moves on dict ordering reports every artifact stale on every run, and the
    warning becomes noise nobody reads."""
    a = declare_inputs(driver="d", covariates={"x": {"p", "q"}}, geometry_version="goal-relative-2")
    b = declare_inputs(geometry_version="goal-relative-2", covariates={"x": {"q", "p"}}, driver="d")
    assert a["digest"] == b["digest"]


def test_the_digest_excludes_itself():
    parts = declare_inputs(driver="d", covariates={"x": ("a",)})
    assert contract_digest(parts) == parts["digest"]


def test_canonical_survives_mixed_type_dict_keys():
    """`covariates` is caller-supplied; a naive `sorted(dict)` raises TypeError on mixed keys and
    the crash lands at driver-run time, not gate time."""
    assert declare_inputs(driver="d", covariates={1: "a", "b": "c"})["digest"]


# --------------------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------------------

#: Drivers that declare an `input_contract()`. Kept explicit rather than derived: a driver without
#: one is not a defect (most artifacts predate the mechanism), so there is no population to derive.
_DECLARING = (
    "validate_xshot_causal",
    "validate_xcross_causal",
    "measure_covariate_invariance",
    "build_gkdv_arm_values",
    "measure_gs_shot_distribution",
)

_RESEARCH = pathlib.Path(__file__).resolve().parents[2] / "docs" / "research"


class StaleArtifactWarning(UserWarning):
    """An artifact's declared inputs no longer digest to what live code produces."""


def _artifacts_for(driver: str, root: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for p in sorted(root.rglob("metrics.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("input_contract", {}).get("driver") == driver:
            out.append(p)
    return out


def stale_artifacts(driver: str, live: dict, root: pathlib.Path) -> list[tuple[pathlib.Path, str, str]]:
    """THE DETECTOR. A separate callable so a test can exercise it on a controlled artifact.

    Returns ``(path, recorded_digest, live_digest)`` for every artifact whose declared inputs no
    longer match live code. Empty list means everything is current.

    `root` is a parameter, not a module constant, precisely so the non-vacuity tests below can point
    the detector at a `tmp_path`. Without it the detector could only ever read the real
    `docs/research` tree, which is what made an earlier draft structurally untestable.
    """
    out = []
    for path in _artifacts_for(driver, root):
        recorded = json.loads(path.read_text(encoding="utf-8"))["input_contract"]
        if recorded.get("digest") != live["digest"]:
            out.append((path, recorded.get("digest", "?"), live["digest"]))
    return out


@pytest.mark.parametrize("driver", _DECLARING)
def test_committed_artifacts_still_match_their_declared_inputs(driver):
    """WARN, do not raise -- spec S1.2. An artifact is not a serving path, so a mismatch is not a
    load failure; it must surface at PR time rather than at read time.

    This test therefore PASSES on the failure it reports. ALL of its evidential weight sits on the
    three tests below, which are the only things here that can go red if the comparison breaks.
    """
    live = importlib.import_module(f"scripts.{driver}").input_contract()
    for path, recorded, current in stale_artifacts(driver, live, _RESEARCH):
        warnings.warn(
            f"{path} was produced under a different input contract (recorded {recorded[:12]}, "
            f"live {current[:12]}). Its numbers may be stale.",
            StaleArtifactWarning,
            stacklevel=2,
        )


def test_the_detector_FIRES_on_a_planted_stale_artifact(tmp_path):
    """Non-vacuity, against a REAL artifact this test writes.

    Red if the comparison is inverted, if the digest stops being compared, or if `stale_artifacts`
    returns nothing. Deliberately independent of the artifacts having been regenerated yet, so the
    gate is meaningful in the window between the driver commit and the run.
    """
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    stale = declare_inputs(driver="d", covariates={"arm": ("a",)})
    assert stale["digest"] != live["digest"]  # precondition

    art = tmp_path / "planted" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"run_commit": "0" * 40, "input_contract": stale}), encoding="utf-8")

    assert [p for p, _, _ in stale_artifacts("d", live, tmp_path)] == [art]


def test_the_detector_is_SILENT_on_a_current_artifact(tmp_path):
    """The other side. A detector that flags everything is as useless as one that flags nothing --
    and it would make the warning noise nobody reads."""
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    art = tmp_path / "current" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"run_commit": "0" * 40, "input_contract": live}), encoding="utf-8")

    assert stale_artifacts("d", live, tmp_path) == []


def test_the_detector_ignores_another_drivers_artifact(tmp_path):
    """Keying is by `driver`, so one driver's drift must not be reported against another's."""
    live = declare_inputs(driver="mine", covariates={"arm": ("a", "b")})
    other = declare_inputs(driver="theirs", covariates={"arm": ("z",)})
    art = tmp_path / "other" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"input_contract": other}), encoding="utf-8")

    assert stale_artifacts("mine", live, tmp_path) == []


def test_the_warning_actually_reaches_the_caller(tmp_path):
    """Pins the wiring between the detector and `warnings.warn` -- the one seam the tests above do
    not cross, and the seam the parametrised gate depends on entirely."""
    live = declare_inputs(driver="d", covariates={"arm": ("a", "b")})
    stale = declare_inputs(driver="d", covariates={"arm": ("a",)})
    art = tmp_path / "planted" / "metrics.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"input_contract": stale}), encoding="utf-8")

    with pytest.warns(StaleArtifactWarning):
        for path, recorded, current in stale_artifacts("d", live, tmp_path):
            warnings.warn(f"{path} {recorded[:12]} {current[:12]}", StaleArtifactWarning, stacklevel=2)


@pytest.mark.parametrize("driver", _DECLARING)
def test_every_declaring_driver_WRITES_its_contract_into_the_artifact(driver):
    """Defining `input_contract()` is half the job; the artifact must actually carry it.

    This gate exists because the other half shipped broken. Four of the five drivers defined the
    function and never called it -- so a regenerated artifact would carry no `input_contract`,
    `_artifacts_for` would return `[]` forever, and the warn-only detector above would report
    nothing while appearing healthy. Every gate in the cycle passed on that tree.

    Checking "the function exists and returns a valid dict" is exactly the wrong half: it is a
    property of the DECLARATION, and the mechanism's value is entirely in the WRITE.

    AST-matched on a CALL, not a substring, for the reason `_shells_out_to_rev_parse` records: a
    source scan cannot tell a mention from a use, and this module's own prose names the symbol
    repeatedly.
    """
    import ast

    from tests.scripts._script_population import iter_scripts

    tree = iter_scripts()[driver]
    called = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "input_contract"]
    assert called, (
        f"scripts/{driver}.py defines input_contract() but never CALLS it, so the artifact it "
        f"writes carries no contract and the staleness detector can never see it. Stamp it beside "
        f"run_commit."
    )


@pytest.mark.parametrize("driver", _DECLARING)
def test_every_declaring_driver_exposes_input_contract(driver):
    """Meta-assertion: the parametrised gate above imports `input_contract()` by name. A driver that
    loses it would make that gate error rather than silently pass, but this says so directly."""
    fn = getattr(importlib.import_module(f"scripts.{driver}"), "input_contract", None)
    assert callable(fn), f"scripts/{driver}.py has no input_contract()"
    contract = fn()
    # A real assertion that also narrows: `getattr(..., None)` types fn's return as `object`, and
    # `declare_inputs` returning anything but a mapping would break every consumer downstream.
    assert isinstance(contract, dict), f"{driver}.input_contract() returned {type(contract).__name__}"
    assert contract["driver"] == driver, f"{driver} declares driver={contract.get('driver')!r}"
    assert contract["digest"]
