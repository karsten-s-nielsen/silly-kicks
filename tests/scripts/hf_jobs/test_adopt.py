"""`adopt` (ADR-128 spec 5.7): the five expectations, the ADR-037 guards, the allowlisted diff, rehydration
of a public bundle, idempotence. Fakes only: every side effect is injected or confined to a temp git repo."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts._stages import _dataset_identity
from scripts.hf_jobs import adopt as A
from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import expected_identity, locked
from scripts.hf_jobs.identity import artifact_identity

C = "c" * 40
MAN = M.load()


def _record(dataset: str, revision: str | None = None) -> dict:
    """A job record whose dataset identity is the form a job writes (`_stages._dataset_identity`)."""
    ident = _dataset_identity({"dataset": dataset}, None)
    assert ident is not None
    if revision:
        ident["revision"] = revision
    return {"token_inputs": {"extract": {"dataset": ident}, "fit": {}}}


def _public_xshot(d: Path, *, commit: str = C, tier: str = "public", digest: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    meta = {"training_commit": commit, "shipped_variant": tier, "provider_list": ["idsse", "skillcorner"]}
    metrics = {
        "shipped_variant": tier,
        "reproducibility": "public",
        **(digest or expected_identity("xshot17", "pergame")),
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d


# ---- step 4: exactly the five expectations ----------------------------------------------------------


def test_a_compliant_public_artifact_has_no_violation(tmp_path):
    assert (
        A.expectation_violations(MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "f"), _record("xshot17"), C)
        == []
    )


def test_refuses_a_training_commit_other_than_the_campaign_commit(tmp_path):
    bad = A.expectation_violations(
        MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "f", commit="d" * 40), _record("xshot17"), C
    )
    assert any("training_commit" in v for v in bad)


def test_refuses_a_dataset_revision_other_than_the_pinned_one(tmp_path):
    bad = A.expectation_violations(
        MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "f"), _record("xshot17", "0" * 40), C
    )
    assert any("read dataset xshot17" in v for v in bad)


def test_refuses_a_corpus_digest_or_n_other_than_the_staging_record(tmp_path):
    wrong = {"corpus_match_ids_sha256": "0" * 64, "corpus_n_matches": 17}
    bad = A.expectation_violations(
        MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "f", digest=wrong), _record("xshot17"), C
    )
    assert any("corpus" in v for v in bad)
    wrong_n = {**expected_identity("xshot17", "pergame"), "corpus_n_matches": 16}
    bad = A.expectation_violations(
        MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "g", digest=wrong_n), _record("xshot17"), C
    )
    assert any("corpus" in v for v in bad)


def test_refuses_a_shipped_tier_other_than_the_expected_tier(tmp_path):
    bad = A.expectation_violations(
        MAN.artifacts["xshot_default"], _public_xshot(tmp_path / "f", tier="sc_extended"), _record("xshot17"), C
    )
    assert any("expected tier" in v for v in bad)


def test_an_owner_artifact_is_checked_on_the_digest(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({"training_commit": C, "shipped_variant": "sc_extended"}), encoding="utf-8"
    )
    m = {"shipped_variant": "sc_extended", **expected_identity("xshot115", "pergame")}
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    a = MAN.artifacts["hub_xshot_sc_extended"]
    assert A.expectation_violations(a, d, _record("xshot115"), C) == []
    m["corpus_n_matches"] = 114
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    assert any("corpus" in v for v in A.expectation_violations(a, d, _record("xshot115"), C))


# ---- step 5: a public bundle is rehydrated AFTER its digest is verified -----------------------------


def test_a_public_bundle_gets_the_locked_public_ids(tmp_path):
    a = MAN.artifacts["xshot_default"]
    fresh = _public_xshot(tmp_path / "f")
    (fresh / "model.json").write_bytes(b"{}")
    A._bundle(a, fresh, tmp_path / "repo")
    m = json.loads((tmp_path / "repo" / a.path / "metrics.json").read_text(encoding="utf-8"))
    assert m["corpus_match_ids"] == locked("xshot17")["public_ids"]
    assert "corpus_match_ids_sha256" not in m
    owner = MAN.artifacts["ghost_default"]
    (fresh / "metrics.json").write_text(
        json.dumps({"corpus_match_ids_sha256": "x", "corpus_n_matches": 179}), encoding="utf-8"
    )
    A._bundle(owner, fresh, tmp_path / "repo")
    assert "corpus_match_ids" not in json.loads(
        (tmp_path / "repo" / owner.path / "metrics.json").read_text(encoding="utf-8")
    )


# ---- guards + the allowlisted diff + idempotence (a temp git repo) ----------------------------------

_TOY_MANIFEST = """# toy manifest
[events.e1]
version = "9.9.9"
adr = "ADR-999"
title = "Toy re-fit"
basis = "on toys"
cite = ""

[[artifacts]]
id = "toy_wheel"
destination = "wheel:silly_kicks/_toy_weights/default"
job = "ghost_gk"
variant = "default"
tier = "owner"
certification = "owner"
training_commit = "{commit}"
commit_field = "metadata.json:training_commit"
refit_event = "e1"

[[artifacts]]
id = "toy_hub"
destination = "hub:silly-kicks/toy-v1"
job = "ghost_gk"
variant = "default"
tier = "owner"
certification = "owner"
training_commit = "{commit}"
commit_field = "metadata.json:training_commit"
refit_event = "e1"
card = "docs/huggingface/model-cards/toy-v1-model-card.md"
wheel_sibling = "toy_wheel"
hub_class = "GhostGkModel"
hub_role = "hf_only"
"""
_INST = "toy-cccccc-ghost_gk-owner/default"


_GIT = shutil.which("git") or "git"


def _git(root: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 -- fixed argv
        [_GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True, capture_output=True
    )


_NOCOMMIT = """
[[artifacts]]
id = "toy_nocommit"
destination = "wheel:silly_kicks/_toy_weights/nocommit"
job = "ghost_gk"
variant = "default"
tier = "owner"
certification = "owner"
"""


def _toy_repo(tmp_path: Path, *, nocommit: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "docs" / "artifacts").mkdir(parents=True)
    text = _TOY_MANIFEST.format(commit="a" * 40) + (_NOCOMMIT if nocommit else "")
    (root / "docs" / "artifacts" / "manifest.toml").write_text(text, encoding="utf-8")
    (root / "docs" / "artifacts" / "observed.json").write_text("{}\n", encoding="utf-8")
    w = root / "silly_kicks" / "_toy_weights" / "default"
    w.mkdir(parents=True)
    (w / "model.npz").write_bytes(b"old")
    (w / "metadata.json").write_text(json.dumps({"training_commit": "a" * 40}), encoding="utf-8")
    if nocommit:
        n = root / "silly_kicks" / "_toy_weights" / "nocommit"
        n.mkdir(parents=True)
        (n / "model.npz").write_bytes(b"old")
    cards = root / "docs" / "huggingface" / "model-cards"
    cards.mkdir(parents=True)
    note = "In silly-kicks 9.9.9 the wheel's bundled `default` was re-fit on toys (`training_commit=aaaaaaa`)."
    refit = "This Hub artifact was last re-fit in silly-kicks 9.9.9 (ADR-999, `training_commit=aaaaaaa`)."
    (cards / "toy-v1-model-card.md").write_text(
        f"# toy\n\n<!-- generated:hub_note -->{note}<!-- /generated:hub_note --> "
        f"<!-- generated:hub_refit -->{refit}<!-- /generated:hub_refit --> Prose.\n",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "toy")
    staging = tmp_path / "staging"
    fresh = staging / _INST
    fresh.mkdir(parents=True)
    (fresh / "model.npz").write_bytes(b"new")
    (fresh / "metadata.json").write_text(json.dumps({"training_commit": C}), encoding="utf-8")
    (fresh / "metrics.json").write_text(json.dumps({"cv_mae_euclidean_mean": 1.1}), encoding="utf-8")
    verify = {
        "commit": C,
        "campaign": "toy",
        "instances": {
            _INST: {
                "ok": True,
                "certification": {"ok": True, "basis": "cv_parity"},
                "artifacts": {
                    "toy_wheel": {"violations": []},
                    "toy_hub": {"violations": []},
                    **({"toy_nocommit": {"violations": []}} if nocommit else {}),
                },
            }
        },
    }
    (staging / "verify.json").write_text(json.dumps(verify), encoding="utf-8")
    return root, staging


class _Campaign:
    name = "toy"


def _clean() -> dict:
    return {"commit": C, "dirty": False, "tree_state": "clean", "dirty_files": []}


def test_adopt_refuses_a_dirty_tree_and_a_foreign_head(tmp_path):
    root, staging = _toy_repo(tmp_path)
    dirty = {**_clean(), "dirty": True, "tree_state": "dirty", "dirty_files": ["x.py"]}
    with pytest.raises(A.AdoptionRefusedError, match="clean tree"):
        A.adopt(_Campaign(), C, staging=staging, git=lambda: dirty, root=root, gate=False)  # type: ignore[arg-type]
    with pytest.raises(A.AdoptionRefusedError, match="not the campaign commit"):
        A.adopt(_Campaign(), C, staging=staging, git=lambda: {**_clean(), "commit": "d" * 40}, root=root, gate=False)  # type: ignore[arg-type]


def test_adopt_refuses_an_instance_that_did_not_verify(tmp_path):
    root, staging = _toy_repo(tmp_path)
    v = json.loads((staging / "verify.json").read_text(encoding="utf-8"))
    v["instances"][_INST]["ok"] = False
    (staging / "verify.json").write_text(json.dumps(v), encoding="utf-8")
    with pytest.raises(A.AdoptionRefusedError, match="did not verify"):
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False)  # type: ignore[arg-type]


def test_adopt_refuses_a_staging_dir_inside_the_repo(tmp_path):
    root, _staging = _toy_repo(tmp_path)
    with pytest.raises(A.AdoptionRefusedError, match="inside the repo"):
        A.adopt(_Campaign(), C, staging=root / "staging", git=_clean, root=root, gate=False)  # type: ignore[arg-type]


def test_adopt_writes_only_allowlisted_paths_and_is_idempotent(tmp_path):
    root, staging = _toy_repo(tmp_path)
    out = A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False, now="2026-10-18T00:00:00+00:00")  # type: ignore[arg-type]
    status = subprocess.run(  # noqa: S603 -- fixed argv
        [_GIT, "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    changed = {line[3:].strip() for line in status.splitlines()}
    assert changed == {
        "silly_kicks/_toy_weights/default/model.npz",
        "silly_kicks/_toy_weights/default/metadata.json",
        "silly_kicks/_toy_weights/default/metrics.json",
        "docs/artifacts/manifest.toml",
        "docs/artifacts/observed.json",
        "docs/huggingface/model-cards/toy-v1-model-card.md",
    }
    assert sorted(out["adopted"]) == ["toy_hub", "toy_wheel"]
    assert out["next"] == "commit gate: owner approval required"
    # spec INV-9 / 5.7 step 6: training_commit is copied from the campaign record AFTER verification; the
    # human-written manifest keeps every other byte
    man = M.load(root / "docs" / "artifacts" / "manifest.toml")
    assert man.artifacts["toy_wheel"].training_commit == C == man.artifacts["toy_hub"].training_commit
    text = (root / "docs" / "artifacts" / "manifest.toml").read_text(encoding="utf-8")
    assert text == _TOY_MANIFEST.format(commit=C)
    card = (root / "docs/huggingface/model-cards/toy-v1-model-card.md").read_text(encoding="utf-8")
    assert f"`training_commit={C[:7]}`" in card and "aaaaaaa" not in card and card.endswith(" Prose.\n")
    observed = json.loads((root / "docs/artifacts/observed.json").read_text(encoding="utf-8"))
    assert set(observed) == {"toy_wheel", "toy_hub"}
    assert observed["toy_wheel"]["campaign"] == "toy@ccccccc"
    assert set(observed["toy_wheel"]) == {"sha256", "metrics_summary", "env", "campaign", "adopted_at"}
    assert (staging / "CHANGELOG-draft.md").is_file()
    # HFU spec 18.3: a re-run over the UNCOMMITTED adoption refuses at the root status and writes nothing
    with pytest.raises(A.AdoptionRefusedError, match="root tree is not clean"):
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False)  # type: ignore[arg-type]
    assert _porcelain(root) == status
    # over the committed adoption the write phase is a fixed point (each adopted_at keeps its date)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "adopted")
    again = A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False, now="2026-10-19T00:00:00+00:00")  # type: ignore[arg-type]
    assert again["written"] == []


def test_adopt_fails_on_a_path_outside_the_allowlist(tmp_path, monkeypatch):
    """Anti-vacuity for the ADR-037 diff guard: a stray write is reported."""
    root, staging = _toy_repo(tmp_path)
    real = M.render_card

    def leaky(path, man=None, *, card_id=None):
        (root / "scripts").mkdir(exist_ok=True)
        (root / "scripts" / "stray.py").write_text("x = 1\n", encoding="utf-8")
        return real(path, man, card_id=card_id)

    monkeypatch.setattr(M, "render_card", leaky)
    with pytest.raises(A.AdoptionRefusedError, match="outside the allowlist"):
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False)  # type: ignore[arg-type]
    assert _porcelain(root) == ""  # HFU spec 18.3: the refusal rolled every write back, the stray file included


def _porcelain(root: Path) -> str:
    return subprocess.run(  # noqa: S603 -- fixed argv
        [_GIT, "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True, check=True
    ).stdout


# ---- HFU spec 18.2/18.3: no manifest commit without a commit_field; a refusal leaves the tree as found ----------


def test_adopt_writes_no_manifest_commit_for_an_artifact_without_a_commit_field(tmp_path):
    root, staging = _toy_repo(tmp_path, nocommit=True)
    out = A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False, now="2026-10-18T00:00:00+00:00")  # type: ignore[arg-type]
    assert "toy_nocommit" in out["adopted"]
    text = (root / "docs" / "artifacts" / "manifest.toml").read_text(encoding="utf-8")
    assert text == _TOY_MANIFEST.format(commit=C) + _NOCOMMIT  # its entry unchanged byte for byte
    observed = json.loads((root / "docs/artifacts/observed.json").read_text(encoding="utf-8"))
    assert observed["toy_nocommit"]["campaign"] == "toy@ccccccc"


def _failing_bundle(exc: BaseException):
    def bundle(a, fresh, root):
        d = root / a.path
        (d / "model.npz").write_bytes(b"partial")  # a tracked file changed
        (d / "metadata.json").unlink()  # a tracked file deleted
        (d / "new.bin").write_bytes(b"x")  # a new file
        raise exc

    return bundle


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()], ids=["error", "keyboard interrupt"])
def test_a_failure_in_the_write_phase_restores_the_tree(tmp_path, monkeypatch, exc):
    root, staging = _toy_repo(tmp_path)
    monkeypatch.setattr(A, "_bundle", _failing_bundle(exc))
    with pytest.raises(type(exc)) as info:
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False)  # type: ignore[arg-type]
    assert info.value is exc  # the original exception, unchanged
    assert _porcelain(root) == ""
    w = root / "silly_kicks" / "_toy_weights" / "default"
    assert (w / "model.npz").read_bytes() == b"old" and (w / "metadata.json").is_file() and not (w / "new.bin").exists()


def _pre_existing(root: Path) -> dict[str, bytes]:
    (root / "docs" / "artifacts" / "observed.json").write_text(
        '{"mine": 1}' + "\n", encoding="utf-8"
    )  # uncommitted edit
    (root / "notes.txt").write_text("my work" + "\n", encoding="utf-8")  # an untracked file
    return {p: (root / p).read_bytes() for p in ("docs/artifacts/observed.json", "notes.txt")}


def _dirty() -> dict:
    return {**_clean(), "dirty": True, "tree_state": "dirty", "dirty_files": ["notes.txt"]}


@pytest.mark.parametrize(
    ("git", "match"),
    [(_clean, "root tree is not clean"), (_dirty, "clean tree")],
    ids=["provenance clean, root dirty", "provenance dirty"],
)
def test_an_entry_refusal_keeps_uncommitted_work(tmp_path, git, match):
    root, staging = _toy_repo(tmp_path)
    mine = _pre_existing(root)
    with pytest.raises(A.AdoptionRefusedError, match=match):
        A.adopt(_Campaign(), C, staging=staging, git=git, root=root, gate=False)  # type: ignore[arg-type]
    assert {p: (root / p).read_bytes() for p in mine} == mine


def test_a_failed_restore_names_the_path(tmp_path, monkeypatch):
    root, staging = _toy_repo(tmp_path)
    monkeypatch.setattr(A, "_bundle", _failing_bundle(RuntimeError("boom")))

    def run(argv, **kw):
        if argv[:2] == ["git", "checkout"]:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.run(argv, **kw)  # noqa: S603 -- the caller's argv

    with pytest.raises(A.AdoptionRollbackError, match=r"model\.npz") as info:
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False, run=run)  # type: ignore[arg-type]
    assert isinstance(info.value, A.AdoptionRefusedError) and str(info.value.__cause__) == "boom"
    assert not (root / "silly_kicks" / "_toy_weights" / "default" / "new.bin").exists()  # every path was attempted


def test_an_interrupted_gate_keeps_the_adoption(tmp_path, monkeypatch):
    root, staging = _toy_repo(tmp_path)

    def gate(run, root):
        raise KeyboardInterrupt

    monkeypatch.setattr(A, "_local_gate", gate)
    with pytest.raises(KeyboardInterrupt):
        A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=True)  # type: ignore[arg-type]
    assert "silly_kicks/_toy_weights/default/model.npz" in _porcelain(root)


def test_every_registry_trainer_has_a_loader():
    from scripts.hf_jobs.registry import REGISTRY

    assert set(REGISTRY) == set(A.LOADERS)


# ---- bootstrap_observed (plan Task 20): the observed fields of the artifacts that ship today -------


class _FakeHub:
    """``hf_hub_download`` over an in-memory repo; a missing file raises like the Hub does."""

    def __init__(self, tmp: Path, files: dict[str, bytes]):
        self.tmp, self.files, self.asked = tmp, files, []

    def hf_hub_download(self, *, repo_id: str, filename: str, repo_type: str) -> str:
        from huggingface_hub.errors import EntryNotFoundError

        self.asked.append((repo_id, filename, repo_type))
        if filename not in self.files:
            raise EntryNotFoundError(f"{repo_id}/{filename}")
        p = self.tmp / "hub" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename])
        return str(p)


def _bootstrap_world(tmp_path: Path, monkeypatch, *, hub_commit: str = "a" * 40) -> tuple[Path, _FakeHub]:
    """The toy repo as the shipped state: its manifest, wheel bundle and observed.json replace the real ones."""
    root, _staging = _toy_repo(tmp_path)
    w = root / "silly_kicks" / "_toy_weights" / "default"
    metrics = {"cv_mae_euclidean_mean": 1.0, "env": {"python": "3.12"}}
    (w / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    toy_manifest = root / "docs" / "artifacts" / "manifest.toml"
    real_load = M.load
    monkeypatch.setattr(M, "load", lambda path=toy_manifest: real_load(path))
    monkeypatch.setattr(M, "REPO_ROOT", root)
    monkeypatch.setattr(M, "OBSERVED", root / "docs" / "artifacts" / "observed.json")
    monkeypatch.setattr(A, "REPO_ROOT", root)
    files = {"metadata.json": json.dumps({"training_commit": hub_commit}).encode(), "SHA256SUMS": b"00  model.json\n"}
    return root, _FakeHub(tmp_path, files)


def test_bootstrap_observed_records_every_shipped_artifact(tmp_path, monkeypatch):
    root, hub = _bootstrap_world(tmp_path, monkeypatch)
    out = A.bootstrap_observed(api=hub, now="2026-10-19T00:00:00+00:00")
    assert set(out) == {"toy_wheel", "toy_hub"}
    wheel = root / "silly_kicks" / "_toy_weights" / "default"
    assert set(out["toy_wheel"]["sha256"]) == {"model.npz", "metadata.json", "metrics.json"}
    assert out["toy_wheel"]["sha256"] == artifact_identity(wheel)["files"]
    assert out["toy_wheel"]["env"] == {"python": "3.12"}
    # the Hub repo is only read; its missing metrics.json is recorded absent, not fatal
    assert sorted(f for _r, f, _t in hub.asked) == ["SHA256SUMS", "metadata.json", "metrics.json"]
    assert all(r == "silly-kicks/toy-v1" and t == "model" for r, _f, t in hub.asked)
    assert out["toy_hub"]["sha256"] == {k: hashlib.sha256(v).hexdigest() for k, v in sorted(hub.files.items())}
    assert out["toy_hub"]["metrics_summary"] == {} and out["toy_hub"]["env"] == {}
    assert {v["campaign"] for v in out.values()} == {"bootstrap (shipped before ADR-128)"}
    assert {v["adopted_at"] for v in out.values()} == {"2026-10-19T00:00:00+00:00"}
    assert json.loads((root / "docs/artifacts/observed.json").read_text(encoding="utf-8")) == out


def test_bootstrap_observed_refuses_a_shipped_artifact_that_misses_its_expectations(tmp_path, monkeypatch):
    root, hub = _bootstrap_world(tmp_path, monkeypatch, hub_commit="d" * 40)
    with pytest.raises(A.AdoptionRefusedError, match=r"toy_hub: metadata\.json:training_commit"):
        A.bootstrap_observed(api=hub, now="2026-10-19T00:00:00+00:00")
    assert (root / "docs/artifacts/observed.json").read_text(encoding="utf-8") == "{}\n"  # nothing written


def test_bootstrap_observed_without_write_leaves_the_file_alone(tmp_path, monkeypatch):
    root, hub = _bootstrap_world(tmp_path, monkeypatch)
    assert set(A.bootstrap_observed(api=hub, write=False)) == {"toy_wheel", "toy_hub"}
    assert (root / "docs/artifacts/observed.json").read_text(encoding="utf-8") == "{}\n"


# ---- HFU spec 14.4: the committed side of public CV parity; the pinned public builders --------------------


def test_committed_metrics_without_reference_raises(tmp_path, monkeypatch):
    """A wheel bundle with neither a ``metrics.json`` nor a ``cv_reference`` has no committed CV record."""
    import dataclasses

    (tmp_path / "w").mkdir()
    monkeypatch.setattr(A, "REPO_ROOT", tmp_path)
    a = dataclasses.replace(MAN.artifacts["expected_passing"], destination="wheel:w", cv_reference=None)
    with pytest.raises(A.AdoptionRefusedError, match="expected_passing"):
        A._committed_metrics(a, {})


def test_cv_reference_bare_path_and_key(tmp_path, monkeypatch):
    import dataclasses

    (tmp_path / "ref.json").write_text(json.dumps({"completion": {"ece": 0.5}, "x": 1}), encoding="utf-8")
    (tmp_path / "w").mkdir()
    monkeypatch.setattr(A, "REPO_ROOT", tmp_path)
    base = dataclasses.replace(MAN.artifacts["expected_passing"], destination="wheel:w")
    whole = A._committed_metrics(dataclasses.replace(base, cv_reference="ref.json"), {})
    assert whole == {"completion": {"ece": 0.5}, "x": 1}
    assert A._committed_metrics(dataclasses.replace(base, cv_reference="ref.json:completion"), {}) == {"ece": 0.5}
    (tmp_path / "w" / "metrics.json").write_text(json.dumps({"ece": 0.25}), encoding="utf-8")
    assert A._committed_metrics(dataclasses.replace(base, cv_reference="ref.json"), {}) == {"ece": 0.25}


def test_a_bundle_metrics_is_its_committed_cv_record():
    """HFU spec 18.7: the three artifacts that once named a ``cv_reference`` ship their own ``metrics.json``, which is
    their committed CV record; the shipped prose still matches it."""
    import re

    for aid in ("expected_passing", "match_outcome", "xsuccess"):
        a = MAN.artifacts[aid]
        own = json.loads((Path(a.path) / "metrics.json").read_text(encoding="utf-8"))
        assert A._committed_metrics(a, {}) == own, aid
    ep = A._committed_metrics(MAN.artifacts["expected_passing"], {})
    assert (round(ep["auc"], 3), round(ep["ece"], 3), round(ep["brier"], 3)) == (0.794, 0.006, 0.129)
    card = Path("silly_kicks/xsuccess/weights/MODEL_CARD.md").read_text(encoding="utf-8").splitlines()[17]
    m = re.search(r"AUC (\d\.\d+), Brier (\d\.\d+)", card)
    assert m is not None
    auc, brier = float(m.group(1)), float(m.group(2))
    xs = A._committed_metrics(MAN.artifacts["xsuccess"], {})
    assert (round(xs["auc"], 3), round(xs["brier"], 3)) == (auc, brier) == (0.895, 0.084)
    model = json.loads(Path("silly_kicks/match_outcome/weights/model.json").read_text(encoding="utf-8"))
    assert A._committed_metrics(MAN.artifacts["match_outcome"], {})["rho"] == model["rho"]


def test_bootstrap_observed_unchanged(tmp_path, monkeypatch):
    """bootstrap_observed never reads a ``cv_reference``: a wheel without metrics.json records ``{}``."""
    root, hub = _bootstrap_world(tmp_path, monkeypatch)
    man = root / "docs" / "artifacts" / "manifest.toml"
    text = man.read_text(encoding="utf-8")
    assert text.count('id = "toy_wheel"\n') == 1
    man.write_text(
        text.replace('id = "toy_wheel"\n', 'id = "toy_wheel"\ncv_reference = "ref.json"\n'), encoding="utf-8"
    )
    (root / "ref.json").write_text(json.dumps({"cv_mae_euclidean_mean": 9.0}), encoding="utf-8")
    (root / "silly_kicks" / "_toy_weights" / "default" / "metrics.json").unlink()
    out = A.bootstrap_observed(api=hub, now="2026-10-19T00:00:00+00:00", write=False)
    assert out["toy_wheel"]["metrics_summary"] == {}


def test_coverage_note_for_sb_open():
    rec = locked("sb_open")
    n, k = rec["identity"]["open"]["corpus_n_matches"], len(rec["competitions"])
    head = (
        f"committed artifact records no match ids; faithful-migration evidence: corpus_n_matches {n}, {k} competitions"
    )
    assert A.coverage_note(cv_ok=True) == f"{head}, CV parity ok"
    assert A.coverage_note(cv_ok=False) == f"{head}, CV parity FAIL"
    assert (n, k) == (3961, 80)


def _cycle1_instance(job: str, tier: str = "public", variant: str = "default"):
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    return next(i for i in Camp.instances(c, C) if (i.job, i.tier, i.variant) == (job, tier, variant))


class _Run:
    """A fake ``subprocess.run`` for the head-to-head: records argv and env, writes the pair file."""

    def __init__(self, pair: dict):
        self.pair, self.calls = pair, []

    def __call__(self, argv, **kw):
        import types

        self.calls.append((list(argv), kw.get("env")))
        if "--metrics-json" in argv:  # the head-to-head; a load check only needs rc 0
            out = Path(argv[argv.index("--metrics-json") + 1])
            out.write_text(json.dumps(self.pair), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")


def test_public_builders_run_pinned(tmp_path):
    from scripts._sb_open_data import OPEN_DATA_REVISION_ENV
    from scripts.hf_jobs.registry import get

    inst = _cycle1_instance("win_probability")
    fresh = tmp_path / "s" / inst.id
    fresh.mkdir(parents=True)
    committed = A._committed_metrics(MAN.artifacts["win_probability"], {})
    run = _Run({"committed_ece": 0.5, "fresh_ece": 0.5})
    res = A._certify(get("win_probability"), inst, MAN.artifacts["win_probability"], fresh, committed, {}, run=run)
    argv, env = run.calls[0]
    assert env[OPEN_DATA_REVISION_ENV] == locked("sb_open")["revision"]
    assert argv[argv.index("--certification") + 1] == "public" and "--eval-rows" not in argv
    assert res["ok"] and res["cv_parity"]["ok"] and res["in_sample_parity"]["basis"] == "in_sample_parity"


def test_flat_tc3_public_certify_passes_the_eval_rows(tmp_path):
    from scripts.hf_jobs.registry import get

    inst = _cycle1_instance("xshot_occurrence")
    staging = tmp_path / "s"
    fresh = staging / inst.id
    fresh.mkdir(parents=True)
    rows = A._eval_rows_path(staging, inst.id)
    rows.write_bytes(b"x")
    ref = MAN.artifacts["xshot_default"]
    committed = A._committed_metrics(ref, {})
    run = _Run({"committed_pr_auc": 0.5, "fresh_pr_auc": 0.5})
    A._certify(get("xshot_occurrence"), inst, ref, fresh, committed, {}, run=run)
    argv, _env = run.calls[0]
    assert argv[argv.index("--eval-rows") + 1] == str(rows) and "--corpus-dir" not in argv


def test_retrieve_saves_eval_rows_for_flat_tc3_public(tmp_path):
    import pandas as pd

    from scripts._stages import FileSet, LocalStore, write_result
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    target = _cycle1_instance("xshot_occurrence")
    store = LocalStore(tmp_path / "bucket")
    rows = pd.DataFrame({"f": [1.0, 2.0], "_y": [0, 1], "_group": ["a", "b"]})
    write_result(store, "assemble/g1/_output", FileSet({"metrics.json": b"{}"}))
    write_result(store, "extract/g0/_output", rows)

    def plan_fn(inst, *, bucket):
        reduced = inst.id == target.id
        return {
            "output": "assemble",
            "stages": [
                {"name": "extract", "generation": "extract/g0", "reduced": reduced, "units": []},
                {"name": "assemble", "generation": "assemble/g1", "reduced": reduced},
            ],
        }

    out = A.retrieve(c, C, staging=tmp_path / "staging", plan_fn=plan_fn, store_fn=lambda _uri: store)
    assert set(out) == {target.id}
    saved = A._eval_rows_path(tmp_path / "staging", target.id)
    pd.testing.assert_frame_equal(pd.read_parquet(saved), rows)
    assert not (out[target.id] / saved.name).exists()  # beside the artifact, never inside it


# ---- HFU spec 14.9: retrieve and verify report progress; counters read in parallel ---------------------------


class _CountingStore:
    """A LocalStore that counts its reads and listings."""

    def __init__(self, root: Path):
        from scripts._stages import LocalStore

        self.inner, self.reads, self.listings = LocalStore(root), [], 0

    def read_bytes(self, rel):
        self.reads.append(rel)
        return self.inner.read_bytes(rel)

    def names(self, rel):
        self.listings += 1
        return self.inner.names(rel)

    def write_bytes(self, rel, data):
        self.inner.write_bytes(rel, data)


def _counter_fixture(tmp_path: Path):
    """12 units: 7 with counters, 2 excluded, 3 with neither (unrecorded)."""
    store = _CountingStore(tmp_path / "s")
    keys = [f"k{i:02d}" for i in range(12)]
    for i, k in enumerate(keys):
        if i < 7:
            store.write_bytes(f"g/{k}.counters.json", json.dumps({"n_events": i + 1, "n_frames": 10 * i}).encode())
        elif i < 9:
            store.write_bytes(f"g/{k}.excluded.json", json.dumps({"reason": "no_data"}).encode())
    return store, {"generation": "g", "units": keys}


def _serial_counters(store, stage) -> dict:
    """The pre-14.9 serial `_counters` (two reads per counter-less unit), the reference the parallel one keeps."""
    from scripts._driver import _merge_counters

    totals: dict = {}
    missing = 0
    gen = stage["generation"]
    for key in sorted(stage.get("units") or []):
        raw = store.read_bytes(f"{gen}/{key}.counters.json")
        if raw is None:
            if store.read_bytes(f"{gen}/{key}.excluded.json") is None:
                missing += 1
            continue
        _merge_counters(totals, json.loads(raw), key)
    return {"totals": totals, "n_counters_unrecorded": missing}


def test_parallel_counters_equal_serial(tmp_path):
    store, stage = _counter_fixture(tmp_path)
    got = A._counters(store, stage, workers=4)
    assert {k: got[k] for k in ("totals", "n_counters_unrecorded")} == _serial_counters(store, stage)
    assert got["n_read"] == 7
    assert got["n_counters_unrecorded"] == 3 and got["totals"]["n_events"] == sum(range(1, 8))


def test_counters_read_count(tmp_path):
    store, stage = _counter_fixture(tmp_path)
    A._counters(store, stage)
    assert store.listings == 1
    assert sorted(store.reads) == [f"g/k{i:02d}.counters.json" for i in range(7)]


def test_retrieve_lines(tmp_path, capsys):
    import pandas as pd

    from scripts._stages import FileSet, LocalStore, write_result
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    target = _cycle1_instance("xshot_occurrence")
    store = LocalStore(tmp_path / "bucket")
    write_result(store, "assemble/g1/_output", FileSet({"metrics.json": b"{}", "model.json": b"x" * 1000}))
    write_result(store, "extract/g0/_output", pd.DataFrame({"f": [1.0], "_y": [0], "_group": ["a"]}))
    store.write_bytes("extract/g0/u1.counters.json", json.dumps({"n_events": 3}).encode())

    def plan_fn(inst, *, bucket):
        reduced = inst.id == target.id
        return {
            "output": "assemble",
            "stages": [
                {"name": "extract", "generation": "extract/g0", "reduced": reduced, "units": ["u1", "u2"]},
                {"name": "assemble", "generation": "assemble/g1", "reduced": reduced},
            ],
        }

    A.retrieve(c, C, staging=tmp_path / "staging", plan_fn=plan_fn, store_fn=lambda _uri: store)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[retrieve] ")]
    n = len(Camp.instances(c, C))
    i = [x.id for x in Camp.instances(c, C)].index(target.id) + 1
    assert len(lines) == 1, lines
    pat = rf"\[retrieve\] {i}/{n} {re.escape(target.id)}: 2 files, \d+\.\d MB, counters 1/2 in \d+\.\d s"
    assert re.fullmatch(pat, lines[0]), lines[0]


def _verify_world(tmp_path, brier: float):
    """One retrieved xsuccess instance (the others have no record): a fresh bundle, its record, a fake run."""
    inst = _cycle1_instance("xsuccess")
    staging = tmp_path / "staging"
    fresh = staging / inst.id
    fresh.mkdir(parents=True)
    (fresh / "metadata.json").write_text(json.dumps({"training_commit": C}), encoding="utf-8")
    metrics = {"brier": brier, "hpo_trials": 30, **expected_identity("sb_open", "open")}
    (fresh / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    A._record_path(staging, inst.id).write_text(json.dumps(_record("sb_open")), encoding="utf-8")
    run = _Run({"committed_ece": 0.5, "fresh_ece": 0.5})
    return inst, staging, run


@pytest.mark.parametrize(("brier", "outcome"), [(0.084, "ok"), (0.2, "FAIL (certification)")])
def test_verify_lines(tmp_path, capsys, brier, outcome):
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    inst, staging, run = _verify_world(tmp_path, brier)
    A.verify(c, C, staging=staging, run=run)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[verify] ")]
    ids = [x.id for x in Camp.instances(c, C)]
    assert len(lines) == 1, lines
    head = rf"\[verify\] {ids.index(inst.id) + 1}/{len(ids)} {re.escape(inst.id)} {re.escape(outcome)} in \d+\.\d s"
    assert re.fullmatch(rf"{head} note=committed artifact records no match ids; .+", lines[0]), lines[0]


def test_certify_passes_the_builds_in_sample_gate(tmp_path):
    """HFU spec 17.3: the xCross default build certifies on CV parity alone (launch-6 figures); xShot keeps the gate."""
    from scripts.hf_jobs.registry import get

    for job, ref_id, want in (("xcross_attempt", "xcross_default", True), ("xshot_occurrence", "xshot_default", False)):
        inst = _cycle1_instance(job)
        fresh = tmp_path / job / inst.id
        fresh.mkdir(parents=True)
        A._eval_rows_path(tmp_path / job, inst.id).write_bytes(b"x")
        ref = MAN.artifacts[ref_id]
        committed = A._committed_metrics(ref, {})
        run = _Run({"committed_pr_auc": 0.1753, "fresh_pr_auc": 0.1570})  # in-sample regressed by > delta
        res = A._certify(get(job), inst, ref, fresh, committed, {}, run=run)
        assert res["ok"] is want, (job, res)


# ---- HFU spec 17.5: retrieve and verify run 4 instances at once; the results stay in instance order ----------


def _reverse_chain(ids: list[str]):
    """A per-instance hook: instance ``ids[k]`` ends only after ``ids[k + 1]`` ended, so the chain ends in reverse
    (it needs ``len(ids)`` concurrent workers; a serial run fails the wait instead of hanging)."""
    import threading

    events = {x: threading.Event() for x in ids}
    ended: list[str] = []

    def turn(inst_id: str) -> bool:
        if inst_id not in events:
            return False
        k = ids.index(inst_id)
        if k + 1 < len(ids):
            assert events[ids[k + 1]].wait(10), f"{inst_id}: the instances did not run concurrently"
        ended.append(inst_id)
        events[inst_id].set()
        return True

    return turn, ended


def _chain_ids(c) -> list[str]:
    from scripts.hf_jobs import campaign as Camp

    ids = [x.id for x in Camp.instances(c, C)]
    return [ids[k] for k in (1, 6, 11, 16)]  # 4 = the worker count


def test_retrieve_keeps_instance_order_when_instances_end_in_reverse(tmp_path, monkeypatch):
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    chain = _chain_ids(c)
    turn, ended = _reverse_chain(chain)
    staging = tmp_path / "staging"
    monkeypatch.setattr(A, "_retrieve_one", lambda inst, i, n, **_: staging / inst.id if turn(inst.id) else None)
    out = A.retrieve(c, C, staging=staging, plan_fn=lambda *a, **k: {}, store_fn=lambda _uri: None)
    assert ended == chain[::-1]
    assert list(out) == chain and out[chain[0]] == staging / chain[0]


def test_verify_keeps_instance_order_when_instances_end_in_reverse(tmp_path, monkeypatch):
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    chain = _chain_ids(c)
    turn, ended = _reverse_chain(chain)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(A, "_verify_one", lambda inst, i, n, **_: {"ok": True, "n": i} if turn(inst.id) else None)
    report = A.verify(c, C, staging=staging, run=_Run({}))
    assert ended == chain[::-1]
    assert list(report["instances"]) == chain
    assert sorted(json.loads((staging / "verify.json").read_text(encoding="utf-8"))["instances"]) == sorted(chain)


def test_each_open_data_head_to_head_gets_its_own_cache_folder(tmp_path):
    from scripts.hf_jobs.registry import get

    staging = tmp_path / "staging"
    dirs = []
    for job, ref_id in (("win_probability", "win_probability"), ("pass_completion", "expected_passing")):
        inst = _cycle1_instance(job)
        fresh = A._artifact_dir(staging, inst.id)
        fresh.mkdir(parents=True)
        ref = MAN.artifacts[ref_id]
        run = _Run({})
        A._certify(get(job), inst, ref, fresh, A._committed_metrics(ref, {}), {}, run=run)
        dirs.append(run.calls[0][1]["SILLY_KICKS_CORPUS_CACHE_DIR"])
        assert dirs[-1] == str(staging / "_open_data" / inst.id)
    assert dirs[0] != dirs[1]


def test_adopt_finalizes_the_probe_ids_for_the_wheel_and_the_hub_entry(tmp_path):
    """HFU spec 18.6: ONE finalized copy (<staging>/_final/<instance>) feeds the wheel bundle and the Hub observed
    entry, so both carry canonical probe ids and the observed metrics.json hash is the finalized file's."""
    import hashlib

    root, staging = _toy_repo(tmp_path)
    fresh = staging / _INST
    m = {
        "cv_mae_euclidean_mean": 1.1,
        "probe_sample_matches": [["gradientsports", "gradientsports-10502"]],
        "probe_sample_in_training_folds": {"gradientsports-10502": False},
    }
    (fresh / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    A.adopt(_Campaign(), C, staging=staging, git=_clean, root=root, gate=False, now="2026-10-18T00:00:00+00:00")  # type: ignore[arg-type]
    wheel = json.loads((root / "silly_kicks/_toy_weights/default/metrics.json").read_text(encoding="utf-8"))
    assert wheel["probe_sample_matches"] == [["gradientsports", "10502"]]
    assert wheel["probe_sample_in_training_folds"] == {"10502": False}
    observed = json.loads((root / "docs/artifacts/observed.json").read_text(encoding="utf-8"))
    assert observed["toy_hub"]["metrics_summary"]["probe_sample_matches"] == [["gradientsports", "10502"]]
    final = staging / "_final" / _INST / "metrics.json"
    assert observed["toy_hub"]["sha256"]["metrics.json"] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert json.loads((fresh / "metrics.json").read_text(encoding="utf-8")) == m  # the staging record is untouched


def test_finalize_rebuilds_the_final_copy(tmp_path):
    """HFU spec 18.6: `_finalize` rebuilds `<staging>/_final/<instance>` on every call (a stray file is gone)."""
    _root, staging = _toy_repo(tmp_path)
    stray = staging / "_final" / _INST / "stray.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}", encoding="utf-8")
    final = A._finalize(staging, _INST)
    assert not stray.exists() and sorted(p.name for p in final.iterdir()) == sorted(
        p.name for p in (staging / _INST).iterdir()
    )


# ---- compare_staging (HFU fit-units spec 6 step 3) ---------------------------------------------------

_CA, _CB = "a" * 40, "b" * 40


def _compare_campaign():
    from types import SimpleNamespace

    from scripts.hf_jobs.campaign import CampaignJob

    return SimpleNamespace(
        name="cycle1",
        jobs=(
            CampaignJob("xsuccess", "public", "sb_open", ("default",), None, None),
            CampaignJob("match_outcome_dependence", "public", "sb_open", ("default",), None, None),
            CampaignJob("build_territory_ranking_census", "public", "sb_open", ("default",), None, None),
        ),
    )


def _compare_staging_dir(root: Path, commit: str, *, weights: bytes = b"W", census_ok: bool = True) -> Path:
    """A toy retrieved staging folder: an xsuccess artifact, a match_outcome artifact (the commit inside
    model.json) and a verify.json with the census row."""
    c7 = commit[:7]
    x = root / f"cycle1-{c7}-xsuccess-public" / "default"
    x.mkdir(parents=True)
    (x / "model.json").write_bytes(weights)
    (x / "metadata.json").write_text(json.dumps({"training_commit": commit, "feature_names": ["a"]}))
    (x / "SHA256SUMS").write_text(f"{hashlib.sha256(weights).hexdigest()}  model.json\n")
    m = root / f"cycle1-{c7}-match_outcome_dependence-public" / "default"
    m.mkdir(parents=True)
    (m / "model.json").write_text(json.dumps({"rho": -0.1, "training_commit": commit}))
    (m / "SHA256SUMS").write_text(f"{hashlib.sha256((m / 'model.json').read_bytes()).hexdigest()}  model.json\n")
    census = f"cycle1-{c7}-build_territory_ranking_census-public/default"
    (root / census).mkdir(parents=True)
    (root / "verify.json").write_text(
        json.dumps(
            {"commit": commit, "instances": {census: {"ok": census_ok, "certification": {"basis": "rebaseline"}}}}
        )
    )
    return root


def test_compare_staging_reports_each_instance(tmp_path):
    """Identical under the rule: provenance differs, and match_outcome holds its commit inside model.json."""
    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    got = A.compare_staging(_compare_campaign(), staging=new, baseline=old)
    same = {"identical": True, "differences": [], "baseline": "retrieved"}
    assert got["xsuccess-public/default"] == same
    assert got["match_outcome_dependence-public/default"] == same
    assert got["build_territory_ranking_census-public/default"]["census_ok"] is True
    assert list(got) == [
        "xsuccess-public/default",
        "match_outcome_dependence-public/default",
        "build_territory_ranking_census-public/default",
    ]


def test_compare_staging_reports_a_payload_difference(tmp_path):
    new = _compare_staging_dir(tmp_path / "new", _CB, weights=b"X")
    old = _compare_staging_dir(tmp_path / "old", _CA)
    row = A.compare_staging(_compare_campaign(), staging=new, baseline=old)["xsuccess-public/default"]
    assert row["identical"] is False and any(d.startswith("model.json") for d in row["differences"])


def _adopted_copy(root: Path, commit: str, *, feature_names: list[str]) -> Path:
    """The adopted copy of the toy xsuccess artifact (``<root>/_final/<run_key>/default/``, what `_finalize`
    writes): the retrieved files, with ``feature_names`` in ``metadata.json`` set to the given value."""
    import shutil

    inst = f"cycle1-{commit[:7]}-xsuccess-public/default"
    dst = root / "_final" / inst
    shutil.copytree(root / inst, dst)
    (dst / "metadata.json").write_text(json.dumps({"training_commit": commit, "feature_names": feature_names}))
    return dst


def test_compare_staging_reads_the_adopted_copy_of_the_baseline(tmp_path):
    """The baseline of a parity proof is the SHIPPED form (HFU fit-units spec 6, launch 8): the retrieved files
    of the baseline differ from the new launch, its adopted copy does not. An instance without an adopted copy
    is compared with its retrieved files, and each row names the form."""
    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    raw = old / f"cycle1-{_CA[:7]}-xsuccess-public" / "default" / "metadata.json"
    raw.write_text(json.dumps({"training_commit": _CA, "feature_names": ["staged-a"]}))
    _adopted_copy(old, _CA, feature_names=["a"])
    got = A.compare_staging(_compare_campaign(), staging=new, baseline=old)
    assert got["xsuccess-public/default"] == {"identical": True, "differences": [], "baseline": "adopted"}
    assert got["match_outcome_dependence-public/default"]["baseline"] == "retrieved"


def test_compare_staging_takes_the_adopted_copy_over_equal_retrieved_files(tmp_path):
    """The other side of the rule: the retrieved files of the baseline equal the new launch, its adopted copy
    does not. The adopted copy is the authority, so the instance is DIFFERENT."""
    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    _adopted_copy(old, _CA, feature_names=["shipped"])
    row = A.compare_staging(_compare_campaign(), staging=new, baseline=old)["xsuccess-public/default"]
    assert row["identical"] is False and row["baseline"] == "adopted"
    assert any(d.startswith("metadata.json") for d in row["differences"])


def test_compare_staging_never_reads_the_adopted_copy_of_the_new_side(tmp_path):
    """A job must write the shipped form itself: the new side is always its retrieved files, even when an
    adopted copy of the new launch equals the baseline."""
    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    raw = new / f"cycle1-{_CB[:7]}-xsuccess-public" / "default" / "metadata.json"
    raw.write_text(json.dumps({"training_commit": _CB, "feature_names": ["staged-a"]}))
    _adopted_copy(new, _CB, feature_names=["a"])
    row = A.compare_staging(_compare_campaign(), staging=new, baseline=old)["xsuccess-public/default"]
    assert row["identical"] is False and row["baseline"] == "retrieved"


def test_finalize_writes_the_folder_that_compare_reads(tmp_path):
    """`_finalize` and `compare_staging` agree on where the adopted copy is (one path helper)."""
    old = _compare_staging_dir(tmp_path / "old", _CA)
    inst = f"cycle1-{_CA[:7]}-xsuccess-public/default"
    assert A._finalize(old, inst) == A._final_dir(old, inst) == old / "_final" / inst
    new = _compare_staging_dir(tmp_path / "new", _CB)
    row = A.compare_staging(_compare_campaign(), staging=new, baseline=old)["xsuccess-public/default"]
    assert row == {"identical": True, "differences": [], "baseline": "adopted"}


def test_compare_staging_reports_an_instance_in_only_one_folder(tmp_path):
    import shutil

    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    shutil.rmtree(old / f"cycle1-{_CA[:7]}-xsuccess-public")
    got = A.compare_staging(_compare_campaign(), staging=new, baseline=old)
    assert got["xsuccess-public/default"] == {
        "identical": False,
        "differences": ["present in only one staging folder"],
    }


def test_compare_staging_refuses_two_run_keys_on_one_side(tmp_path):
    new = _compare_staging_dir(tmp_path / "new", _CB)
    old = _compare_staging_dir(tmp_path / "old", _CA)
    (old / f"cycle1-{'c' * 7}-xsuccess-public" / "default").mkdir(parents=True)
    with pytest.raises(A.AdoptionRefusedError, match="xsuccess-public"):
        A.compare_staging(_compare_campaign(), staging=new, baseline=old)


@pytest.mark.parametrize("fault", ["no_verify_json", "another_commit", "not_ok"])
def test_compare_staging_census_is_not_ok(tmp_path, fault):
    new = _compare_staging_dir(tmp_path / "new", _CB, census_ok=fault != "not_ok")
    old = _compare_staging_dir(tmp_path / "old", _CA)
    if fault == "no_verify_json":
        (new / "verify.json").unlink()
    elif fault == "another_commit":
        doc = json.loads((new / "verify.json").read_text())
        (new / "verify.json").write_text(json.dumps({**doc, "commit": "d" * 40}))
    got = A.compare_staging(_compare_campaign(), staging=new, baseline=old)
    assert got["build_territory_ranking_census-public/default"]["census_ok"] is False


# ---- one read bound for the whole retrieve (HFU fit-units spec C1) ------------------------------------


class _SlowStore:
    """A LocalStore whose reads and listings take 5 ms and record how many are in flight at once."""

    def __init__(self, root: Path):
        import threading

        from scripts._stages import LocalStore

        self.inner, self.lock, self.in_flight, self.peak = LocalStore(root), threading.Lock(), 0, 0

    def _held(self, fn, rel):
        import time

        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(0.005)
            return fn(rel)
        finally:
            with self.lock:
                self.in_flight -= 1

    def read_bytes(self, rel):
        return self._held(self.inner.read_bytes, rel)

    def names(self, rel):
        return self._held(self.inner.names, rel)

    def write_bytes(self, rel, data):
        self.inner.write_bytes(rel, data)


def _bound_world(tmp_path: Path):
    import pandas as pd

    from scripts._stages import FileSet, write_result
    from scripts.hf_jobs import campaign as Camp

    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    store = _SlowStore(tmp_path / "bucket")
    units = [f"u{i:02d}" for i in range(40)]
    write_result(store.inner, "assemble/g1/_output", FileSet({"metrics.json": b"{}"}))
    write_result(store.inner, "extract/g0/_output", pd.DataFrame({"f": [1.0], "_y": [0], "_group": ["a"]}))
    for i, u in enumerate(units):
        store.write_bytes(f"extract/g0/{u}.counters.json", json.dumps({"n_events": i + 1}).encode())

    def plan_fn(inst, *, bucket):
        return {
            "output": "assemble",
            "stages": [
                {"name": "extract", "generation": "extract/g0", "reduced": True, "units": units},
                {"name": "assemble", "generation": "assemble/g1", "reduced": True},
            ],
        }

    return c, store, plan_fn


def _records(staging: Path) -> dict:
    return {
        p.relative_to(staging).as_posix(): json.loads(p.read_text())["counters"]
        for p in sorted(staging.rglob("*.record.json"))
    }


def test_retrieve_runs_the_plan_reads_under_the_bound(tmp_path):
    """A `--plan` subprocess reads the store too (spec C1: EVERY store read): it holds one slot."""
    import threading
    import time

    c, store, plan_fn = _bound_world(tmp_path)
    lock, state = threading.Lock(), {"in_flight": 0, "peak": 0}

    def counted_plan(inst, *, bucket):
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        try:
            time.sleep(0.02)
            return plan_fn(inst, bucket=bucket)
        finally:
            with lock:
                state["in_flight"] -= 1

    A.retrieve(c, C, staging=tmp_path / "s", plan_fn=counted_plan, store_fn=lambda _uri: store, read_workers=2)
    assert 1 <= state["peak"] <= 2, state  # 4 instance workers ran 4 plans at once before
    assert store.peak <= 2


def test_retrieve_never_reads_more_than_eight_at_once(tmp_path):
    c, store, plan_fn = _bound_world(tmp_path)
    A.retrieve(c, C, staging=tmp_path / "s8", plan_fn=plan_fn, store_fn=lambda _uri: store)
    assert 2 <= store.peak <= 8, store.peak  # 4 instances x 8 counter threads gave up to 32 before the bound
    serial = _SlowStore(tmp_path / "bucket")
    A.retrieve(c, C, staging=tmp_path / "s1", plan_fn=plan_fn, store_fn=lambda _uri: serial, read_workers=1)
    assert serial.peak == 1
    assert _records(tmp_path / "s8") == _records(tmp_path / "s1") and len(_records(tmp_path / "s8")) == 21
