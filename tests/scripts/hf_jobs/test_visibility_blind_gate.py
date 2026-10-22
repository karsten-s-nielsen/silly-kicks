"""I1 gate (ADR-123 Amendment B): no module REACHABLE FROM AN HF JOB may carry a public/private concept.

Walks each visibility-gated trainer's transitive import closure (subpackage-aware) and fails if any module
imports or names a banned visibility symbol. A planted-leak meta-assertion proves the walk TRAVERSES a
subpackage hop (HFVB-PLAN-01), so a future indirect leak via any scripts/hf_jobs/ module is caught.
"""

import ast
from pathlib import Path

import pytest

BANNED = {"match_visibility", "is_public_row", "assert_public_corpus", "artifact_label", "PUBLIC_CORPUS"}
SCRIPTS = Path("scripts")
REPO = Path(".")

# No infra-leaf exemption (ADR-123 Amendment B gold-standard severance): the one pining-reaching helper
# (``list_match_ids``) was relocated out of ``_partition`` into ``_partition_pining``, so the trainer's
# static closure through the ``_driver`` -> ``_partition`` seam no longer reaches ``_loader_pining`` (which
# defines the banned ``match_visibility``). The gate therefore traverses EVERYTHING -- a pure static closure,
# with no module trusted on runtime-reachability grounds.


def _resolve(mod: str, scripts_dir: Path = SCRIPTS, repo: Path = REPO) -> Path | None:
    """Resolve a dotted or bare module name to a real file (subpackage-aware, HFVB-PLAN-01)."""
    if mod.startswith("scripts."):
        p = repo / Path(mod.replace(".", "/") + ".py")
        if p.exists():
            return p
        pkg = repo / Path(mod.replace(".", "/")) / "__init__.py"
        return pkg if pkg.exists() else None
    p = scripts_dir / f"{mod.split('.')[-1]}.py"  # bare sibling, how the trainers import _corpus_identity etc.
    return p if p.exists() else None


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)  # keep the FULL dotted path (resolve subpackages)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    return mods


def _closure(entry: str, scripts_dir: Path = SCRIPTS, repo: Path = REPO) -> set[Path]:
    seen: set[Path] = set()
    stack = [scripts_dir / f"{entry}.py"]
    while stack:
        p = stack.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        for mod in _imported_modules(p):
            nxt = _resolve(mod, scripts_dir, repo)
            if nxt is not None:
                stack.append(nxt)
    return seen


def _banned_hits(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    return (names & BANNED) | {b for b in BANNED if b in src}


def _hf_job_entries() -> list[str]:
    """Every module an HF campaign launches: each registry trainer's pipeline module (or its module, for
    an adapter-launched trainer: the adapter runs its CLI in-job) and every driver (spec 5.8 item 4)."""
    from scripts.hf_jobs.registry import DRIVERS, REGISTRY

    mods = {s.pipeline or s.module for s in REGISTRY.values()} | {d.module for d in DRIVERS.values()}
    return sorted(m.removeprefix("scripts.") for m in mods)


@pytest.mark.parametrize("entry", _hf_job_entries())
def test_every_hf_job_closure_is_visibility_blind(entry):
    hits = {p.name: _banned_hits(p) for p in _closure(entry) if _banned_hits(p)}
    assert not hits, f"visibility concept reachable from the HF job {entry}: {hits}"


def test_the_population_is_derived_and_non_empty():
    entries = set(_hf_job_entries())
    want = {"train_xshot_occurrence", "train_xcross_attempt", "train_ghost_gk", "build_territory_ranking_census"}
    assert want <= entries


def test_gate_traverses_a_subpackage_hop(tmp_path):
    # anti-rot (HFVB-PLAN-01): prove the TRAVERSAL, not just per-file detection. entry -> imports
    # scripts.hf_jobs.leaky (a SUBPACKAGE hop) -> leaky imports match_visibility. _closure MUST reach leaky.py.
    (tmp_path / "scripts" / "hf_jobs").mkdir(parents=True)
    (tmp_path / "scripts" / "entry.py").write_text("from scripts.hf_jobs.leaky import f\n")
    leaky = tmp_path / "scripts" / "hf_jobs" / "leaky.py"
    leaky.write_text("from _loader_pining import match_visibility\n\n\ndef f():\n    return match_visibility\n")
    closure = _closure("entry", scripts_dir=tmp_path / "scripts", repo=tmp_path)
    assert leaky in closure, "closure did not traverse the subpackage hop"
    assert any(_banned_hits(p) for p in closure), "gate missed the indirect leak behind the hop"


def test_adr052_seam_is_traversed_not_exempted():
    """Gold-standard severance anti-rot (HFVB-IMPL-01): the ``_driver`` -> ``_partition`` seam must be
    TRAVERSED by the real trainer closure (no ``_INFRA_LEAVES`` exemption) and carry no banned symbol --
    proving ``list_match_ids`` was relocated to ``_partition_pining`` so the seam no longer reaches
    ``_loader_pining``. Re-introducing an exemption, or an eager pining import in the seam, fails here."""
    closure = _closure("train_xshot_occurrence")
    names = {p.name for p in closure}
    assert {"_driver.py", "_partition.py"} <= names, f"ADR-052 seam not traversed (exemption re-added?): {names}"
    assert not _banned_hits(SCRIPTS / "_partition.py"), "_partition reaches/names a banned visibility symbol"
    assert "_partition_pining.py" not in names, "the pining helper leaked back into the in-job closure"


def test_the_pass_completion_pining_cross_check_is_severed_not_exempted():
    """Severance anti-rot (ADR-123 Amendment B, ADR-128): ``train_pass_completion`` runs on HF via the adapter,
    so its owner-tier ``--source pining`` cross-check lives in ``_pass_completion_pining``, which injects its
    source. The HF closure must not reach that module or the pining client. The owner closure DOES reach the
    pining client, so the cross-check keeps its source and a re-merge into the trainer fails the gate above."""
    hf = {p.name for p in _closure("train_pass_completion")}
    assert {"_pass_completion_pining.py", "_loader_pining.py", "_events_admission.py"}.isdisjoint(hf), hf
    owner = _closure("_pass_completion_pining")
    assert "_loader_pining.py" in {p.name for p in owner}
    assert any(_banned_hits(p) for p in owner), "the owner closure no longer reaches a banned symbol (vacuous)"
