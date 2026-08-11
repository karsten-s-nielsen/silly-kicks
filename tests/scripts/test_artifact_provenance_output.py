"""The existing provenance gate reads driver SOURCE; this one reads artifact OUTPUT (Cycle B).

A driver can satisfy every assertion in `test_provenance_wiring.py` -- imports the helper, offers
`--allow-dirty`, calls `require_clean_tree` from `main()` -- and still emit an artifact nobody can
trace. K9 exists because only the source half was ever built.

WHICH FIELD carries provenance had FOUR answers on the tree when this was written, which is the
finding rather than a detail:

    run_commit (top-level)       11 docs/research artifacts   <- the ADR-037 standard, canonical here
    _provenance.commit (nested)   2 adr028_rc4_orientation measurements
    training_commit               _ghost_gk_weights/metadata.json
    nothing at all                _xshot/_xcross metadata.json, comparability_report, gate.json

This gate picks ONE canonical shape per surface -- `run_commit` for research artifacts,
`training_commit` for bundled weights -- and applies it uniformly. Divergent shapes are recorded in
`_UNPROVENANCED` WITH the location of their real provenance, so the divergence is documented rather
than silently coalesced. An absent key is worse than a null one: a null is something a reader can
notice.

The glob is `**/*.json`, NOT `**/metrics.json`. Measured: the narrow form matches 7 of 17 research
artifacts and would have ignored every one of the five that carry no provenance -- including
`gate.json`, the output of a driver this very cycle enrolled source-side. A gate that polices 7 of
17 while reporting success is the failure mode this cycle exists to remove.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RESEARCH = _ROOT / "docs" / "research"


def _rel(p: pathlib.Path) -> str:
    return str(p.relative_to(_ROOT)).replace("\\", "/")


#: Committed artifacts deliberately without canonical top-level provenance, each with a reason.
#: An entry is a decision on the record; an omission is an untraceable number.
_UNPROVENANCED: dict[str, str] = {
    "docs/research/tf19_signoff_power/invalidation.json": (
        "an ANNOTATION about another artifact, not a driver output. It records the annotated "
        "artifact's commit in `artifact_run_commit`; it has no run of its own to stamp."
    ),
    "docs/research/skillcorner_corpus/manifest_skillcorner_full.json": (
        "a JSON LIST of corpus ids, not an object -- there is no mapping to stamp. Structural, not an oversight."
    ),
    "docs/research/adr028_rc4_orientation/prefix_measurement.json": (
        "DOES carry provenance, under the nested `_provenance.commit` rather than top-level "
        "`run_commit` -- written by `measure_rc4_orientation.py`, which is enrolled and guarded. "
        "Recorded as a CONVENTION divergence rather than a missing stamp; normalising it edits a "
        "committed research artifact and is a deliberate follow-up, not a drive-by."
    ),
    "docs/research/adr028_rc4_orientation/postfix_measurement.json": (
        "same as its prefix sibling: provenance lives at `_provenance.commit`."
    ),
    "docs/research/xtgk_comparability/comparability_report.json": (
        "written by `scripts/_xtgk_comparability.py`, a PRIVATE module -- underscore-prefixed, so "
        "it is outside the artifact-driver population by construction and no source-side gate can "
        "reach it. A genuine gap, recorded here so it is visible; closing it means giving the "
        "comparability pass a public driver."
    ),
    "docs/research/xtgk_possession_value/gate.json": (
        "produced by `validate_xtgk_possession_value`, which Cycle B wired to the provenance guard "
        "in this same cycle. The artifact predates that wiring, so it will carry `run_commit` from "
        "its next owner-run (Databricks gold marts, owner-tier). Burn this entry down then."
    ),
}


def _research_artifacts() -> list[pathlib.Path]:
    return sorted(_RESEARCH.rglob("*.json"))


def _bundled_metadata() -> list[pathlib.Path]:
    return sorted(_ROOT.glob("silly_kicks/**/_*_weights/*/metadata.json"))


@pytest.mark.parametrize("path", _research_artifacts(), ids=_rel)
def test_research_artifacts_carry_run_provenance(path):
    key = _rel(path)
    if key in _UNPROVENANCED:
        pytest.skip(_UNPROVENANCED[key])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{key} is not a JSON object -- exempt it structurally"
    assert data.get("run_commit"), (
        f"{key} carries no run_commit -- its numbers cannot be traced to the code that produced them"
    )
    assert data.get("run_tree_dirty") is False, (
        f"{key} was produced from a dirty tree (run_tree_dirty={data.get('run_tree_dirty')!r}), so "
        f"its run_commit does not describe the code that ran"
    )


@pytest.mark.parametrize("path", _bundled_metadata(), ids=_rel)
def test_bundled_weights_carry_training_provenance(path):
    key = _rel(path)
    if key in _UNPROVENANCED:
        pytest.skip(_UNPROVENANCED[key])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("training_commit"), (
        f"{key} carries no training_commit. A bundled artifact nobody can trace back to a commit "
        f"cannot be reproduced or audited -- and an ABSENT key is worse than a null one, because a "
        f"null is something a reader can notice."
    )


def test_the_artifact_populations_are_not_silently_empty():
    """Meta-assertion: a parametrised gate over an empty glob passes vacuously."""
    assert len(_research_artifacts()) >= 12, "research-artifact discovery looks broken"
    assert len(_bundled_metadata()) >= 3, "bundled-metadata discovery looks broken"


def test_unprovenanced_exemptions_name_files_that_exist():
    """Self-burning-down: an exemption for a file that no longer exists is stale scaffolding."""
    stale = sorted(k for k in _UNPROVENANCED if not (_ROOT / k).is_file())
    assert not stale, f"_UNPROVENANCED names files that do not exist: {stale}"


def test_the_gate_reads_the_WIDE_glob_not_just_metrics_json():
    """Non-vacuity on the glob itself.

    The narrow `**/metrics.json` form matches 7 files; the wide form matches 17. If someone
    re-narrows it, every artifact that is not named `metrics.json` silently stops being policed --
    which is how `gate.json`, `comparability_report.json` and the rc4 pair went unchecked.
    """
    found = _research_artifacts()
    named_metrics = [p for p in found if p.name == "metrics.json"]
    assert len(found) > len(named_metrics), (
        "the research-artifact glob no longer sees anything but metrics.json -- re-widen it"
    )
