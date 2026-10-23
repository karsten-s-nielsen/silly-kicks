"""`adopt` (ADR-128 spec 5.7): the five expectations, the ADR-037 guards, the allowlisted diff, rehydration
of a public bundle, idempotence. Fakes only: every side effect is injected or confined to a temp git repo."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.hf_jobs import adopt as A
from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import expected_identity, locked
from scripts.hf_jobs.identity import artifact_identity

C = "c" * 40
MAN = M.load()


def _record(dataset: str, revision: str | None = None) -> dict:
    rev = revision or locked(dataset)["revision"]
    return {"token_inputs": {"extract": {"dataset": {"name": dataset, "repo": "r", "revision": rev}}, "fit": {}}}


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


def _toy_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "docs" / "artifacts").mkdir(parents=True)
    (root / "docs" / "artifacts" / "manifest.toml").write_text(_TOY_MANIFEST.format(commit="a" * 40), encoding="utf-8")
    (root / "docs" / "artifacts" / "observed.json").write_text("{}\n", encoding="utf-8")
    w = root / "silly_kicks" / "_toy_weights" / "default"
    w.mkdir(parents=True)
    (w / "model.npz").write_bytes(b"old")
    (w / "metadata.json").write_text(json.dumps({"training_commit": "a" * 40}), encoding="utf-8")
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
                "artifacts": {"toy_wheel": {"violations": []}, "toy_hub": {"violations": []}},
            }
        },
    }
    (staging / "verify.json").write_text(json.dumps(verify), encoding="utf-8")
    return root, staging


class _Campaign:
    name = "toy"
    bucket = "b/x"


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


def test_stamping_writes_every_key_from_its_source_and_regenerates_the_sums(tmp_path):
    import json

    from scripts.hf_jobs import adopt, identity

    d = tmp_path / "art"
    d.mkdir()
    (d / "metadata.json").write_text('{"variant":"default"}')
    (d / "metrics.json").write_text('{"cv":1}')
    att = {"commit": "a" * 40, "main": {"env": {"platform": "Linux-x"}, "tree_state": "clean"}}
    adopt.stamp_provenance(d, attestation=att, params={"training_platform": None}, generation="tok")
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["training_commit"] == "a" * 40 and meta["run_tree_dirty"] is False
    assert meta["generation"] == "tok"
    assert identity.verify_sums(d)  # SHA256SUMS lists the final bytes


class _MemStore:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def read_bytes(self, rel):
        return self.files.get(rel)

    def upload_bytes(self, rel, data):
        self.files[rel] = data

    def names(self, rel_dir):
        base = rel_dir.rstrip("/") + "/"
        return {k[len(base) :] for k in self.files if k.startswith(base) and "/" not in k[len(base) :]}

    def list_tree(self, rel_dir):
        base = rel_dir.rstrip("/") + "/"
        return {k[len(base) :] for k in self.files if k.startswith(base)}


def _wire_adopt(monkeypatch, store, *, ancestor=True):
    from scripts.hf_jobs import adopt as AA
    from scripts.hf_jobs import campaign as CC
    from scripts.hf_jobs import store as SS

    inst = CC.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), "ghost_gk-owner")
    monkeypatch.setattr(CC, "instances", lambda c, commit, *, replica=None, canary=False: [inst])
    monkeypatch.setattr(
        CC,
        "read_plan",
        lambda inst, bucket: {
            "pipeline": "ghost_gk",
            "output": "fit",
            "stages": [{"name": "fit", "generation": "gen_f", "token": "t_f"}],
        },
    )
    monkeypatch.setattr(SS, "BucketStore", lambda bucket, rk, api=None: store)
    monkeypatch.setattr(AA, "is_ancestor", lambda c_, head, **k: ancestor)


def test_assert_adoptable_api_path_passes_with_an_ancestor_attestation_and_canary(tmp_path, monkeypatch):
    from scripts.hf_jobs import adopt as A

    store = _MemStore({"_attest/fit/t_f/anc.json": json.dumps({"commit": "anc"}).encode()})
    _wire_adopt(monkeypatch, store)
    rep = {"canary": {"ghost_gk-owner/default": {"fit": {"ok": True}}}}
    A._assert_adoptable(_Campaign(), "camp", "h" * 40, rep, api=object(), run=None)  # type: ignore[arg-type]


def test_assert_adoptable_api_path_refuses_without_an_attestation(monkeypatch):
    from scripts.hf_jobs import adopt as A

    _wire_adopt(monkeypatch, _MemStore())
    with pytest.raises(A.AdoptionRefusedError, match="attestation"):
        A._assert_adoptable(_Campaign(), "camp", "h" * 40, {}, api=object(), run=None)  # type: ignore[arg-type]


def test_expectation_violations_tolerate_a_commit_free_output(tmp_path):
    # B#30 regression: a commit-free retrieved output (no training_commit; adopt stamps it) must not be flagged
    # "training_commit missing".
    d = tmp_path / "f"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"shipped_variant": "public"}), encoding="utf-8")
    metrics = {"shipped_variant": "public", "reproducibility": "public", **expected_identity("xshot17", "pergame")}
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    bad = A.expectation_violations(MAN.artifacts["xshot_default"], d, _record("xshot17"), C)
    assert not any("training_commit" in v for v in bad), bad


def test_assert_adoptable_detects_a_quarantine_in_a_utc_subdir(monkeypatch):
    # B#36 regression: quarantine writes `_quarantine/{stage}/{token}/{utc}/...`; the check must recurse.
    store = _MemStore(
        {
            "_attest/fit/t_f/anc.json": json.dumps({"commit": "anc"}).encode(),
            "_quarantine/fit/t_f/20261022T000000Z/record.json": b"{}",
        }
    )
    _wire_adopt(monkeypatch, store)
    rep = {"canary": {"ghost_gk-owner/default": {"fit": {"ok": True}}}}
    with pytest.raises(A.AdoptionRefusedError, match="quarantine"):
        A._assert_adoptable(_Campaign(), "camp", "h" * 40, rep, api=object(), run=None)  # type: ignore[arg-type]


def test_stamp_uses_the_real_shipped_variant_in_the_restricted_note(tmp_path):
    # B#37 regression: the restricted note names the real variant, not the reproducibility class.
    d = tmp_path / "art"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({"reproducibility": "restricted", "shipped_variant": "sc_extended", "providers": ["idsse"]}),
        encoding="utf-8",
    )
    att = {"commit": "a" * 40, "main": {"env": {"platform": "Linux"}, "tree_state": "clean"}}
    A.stamp_provenance(d, attestation=att, params={"training_platform": None})
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert "sc_extended" in meta["reproducibility_note"] and "restricted" == meta["reproducibility"]


def test_adopt_reuse_stamps_commit_free_output_and_records_the_provenance(tmp_path, monkeypatch):
    from scripts.hf_jobs import adopt as A

    root, staging = _toy_repo(tmp_path)
    fresh = staging / _INST
    (fresh / "metadata.json").write_text(json.dumps({"variant": "default"}), encoding="utf-8")  # commit-free
    (fresh / "metrics.json").write_text(json.dumps({"cv_mae_euclidean_mean": 1.1}), encoding="utf-8")
    anc = "e" * 40
    head = "h" * 40
    monkeypatch.setattr(A, "_assert_adoptable", lambda *a, **k: None)

    def fake_stamp(c, commit, h, inst_id, fr, *, api, run):
        A.stamp_provenance(
            fr,
            attestation={"commit": anc, "main": {"env": {"platform": "Linux"}, "tree_state": "clean"}},
            params={"training_platform": None},
            generation="tok",
        )
        return anc

    monkeypatch.setattr(A, "_stamp_fresh", fake_stamp)
    out = A.adopt(
        _Campaign(),  # type: ignore[arg-type]
        C,
        staging=staging,
        api=object(),
        git=lambda: {**_clean(), "commit": head},
        root=root,
        gate=False,
    )
    assert sorted(out["adopted"]) == ["toy_hub", "toy_wheel"]
    man = M.load(root / "docs" / "artifacts" / "manifest.toml")
    assert man.artifacts["toy_wheel"].training_commit == anc  # stamped attester, not HEAD
    observed = json.loads((root / "docs/artifacts/observed.json").read_text(encoding="utf-8"))
    assert observed["toy_wheel"]["adopted_at_commit"] == head and observed["toy_wheel"]["attested_by"] == anc


def test_assert_adoptable_is_order_independent_across_instances_sharing_a_stage(monkeypatch):
    # B#31 regression: two instances share stage name "fit" (one attested, one not). The verdict must not depend
    # on instance iteration order -- bare-stage keys collided (last wins) and fail-opened on one order.
    from scripts.hf_jobs import adopt as AA
    from scripts.hf_jobs import campaign as CC
    from scripts.hf_jobs import store as SS

    att = _MemStore({"_attest/fit/t_f/anc.json": json.dumps({"commit": "anc"}).encode()})
    noatt = _MemStore()
    stores = {"rk-att": att, "rk-noatt": noatt}
    insts = [
        CC.Instance("ghost_gk", "owner", "a", "m", (), (), "rk-att"),
        CC.Instance("win", "owner", "b", "m", (), (), "rk-noatt"),
    ]
    monkeypatch.setattr(
        CC,
        "read_plan",
        lambda inst, bucket: {
            "pipeline": "p",
            "output": "fit",
            "stages": [{"name": "fit", "generation": "g", "token": "t_f"}],
        },
    )
    monkeypatch.setattr(SS, "BucketStore", lambda bucket, rk, api=None: stores[rk])
    monkeypatch.setattr(AA, "is_ancestor", lambda c_, h, **k: True)
    rep = {"canary": {"rk-att/a": {"fit": {"ok": True}}, "rk-noatt/b": {"fit": {"ok": True}}}}
    for order in (insts, list(reversed(insts))):
        monkeypatch.setattr(CC, "instances", lambda c, commit, _o=order: list(_o))
        with pytest.raises(AA.AdoptionRefusedError, match="attestation"):  # rk-noatt/fit unattested, either order
            AA._assert_adoptable(_Campaign(), "camp", "h" * 40, rep, api=object(), run=None)  # type: ignore[arg-type]


def test_adoption_refusals_cover_every_rule_from_both_sides():
    from scripts.hf_jobs import adopt

    anc = {"old", "mid"}.__contains__
    base: dict[str, Any] = dict(
        head_tokens={"fit": "T"},
        adopted={"fit": "T"},
        attestations={"fit": ["old", "mid"]},
        quarantined=set(),
        canary_ok={"fit": True},
        is_ancestor=anc,
    )
    assert adopt.adoption_refusals(**base) == []
    assert "token" in adopt.adoption_refusals(**{**base, "adopted": {"fit": "OTHER"}})[0]
    assert "attestation" in adopt.adoption_refusals(**{**base, "attestations": {"fit": []}})[0]
    assert "ancestor" in adopt.adoption_refusals(**{**base, "attestations": {"fit": ["branch"]}})[0]
    assert "quarantine" in adopt.adoption_refusals(**{**base, "quarantined": {"T"}})[0]
    assert "canary" in adopt.adoption_refusals(**{**base, "canary_ok": {"fit": False}})[0]
    assert adopt.oldest_ancestor_attester(["mid", "old"], order=["new", "mid", "old"]) == "old"
