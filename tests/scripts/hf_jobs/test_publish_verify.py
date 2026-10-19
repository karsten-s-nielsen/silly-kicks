"""`publish` (ADR-128 spec 5.7): the sandbox repo is private BEFORE the ADR-088 seam runs; a public target
needs an owner approval per repo; the post-verification catches every mismatch. Fakes only (no network)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from scripts._hub_publish import card_bytes
from scripts.hf_jobs import adopt as A
from scripts.hf_jobs import manifest as M

AID = "hub_ghost_sweeper"
INST = "cycle1-ccccccc-ghost_gk-owner/sweeper"


class _Api:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.calls: list[tuple] = []
        self.remote: dict[str, dict[str, bytes]] = {}

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, private))
        self.remote.setdefault(repo_id, {})

    def model_info(self, repo_id):
        return types.SimpleNamespace(
            siblings=[types.SimpleNamespace(rfilename=f) for f in sorted(self.remote[repo_id])]
        )

    def hf_hub_download(self, *, repo_id, filename, repo_type, force_download=False):
        p = self.tmp / "dl" / repo_id.replace("/", "__") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.remote[repo_id][filename])
        return str(p)


def _publisher(api: _Api):
    def publish(api_, artifact_dir, repo_id, *, model_card):
        api.calls.append(("publish", repo_id))
        files = {p.name: p.read_bytes() for p in Path(artifact_dir).iterdir()}
        files["README.md"] = card_bytes(model_card)
        api.remote.setdefault(repo_id, {}).update(files)

    return publish


def _staging(tmp: Path) -> Path:
    s = tmp / "staging"
    d = s / INST
    d.mkdir(parents=True)
    (d / "rfcde_weights.npz").write_bytes(b"w")
    (d / "metadata.json").write_text("{}", encoding="utf-8")
    (d / "SHA256SUMS").write_text("abc  rfcde_weights.npz\n", encoding="utf-8")
    (s / "verify.json").write_text(
        json.dumps({"commit": "c", "instances": {INST: {"ok": True, "artifacts": {AID: {}}}}})
    )
    return s


def _ok_run(*_a, **_k):
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_sandbox_repo_is_created_private_before_the_publish_seam(tmp_path):
    api = _Api(tmp_path)
    out = A.publish([AID], sandbox=True, api=api, staging=_staging(tmp_path), run=_ok_run, publisher=_publisher(api))
    repo = "silly-kicks/sandbox-ghost-gk-sweeper-v1"
    assert api.calls[:2] == [("create_repo", repo, True), ("publish", repo)]
    assert out[AID]["repo"] == repo and out[AID]["ok"], out


def test_a_public_target_needs_an_approval_for_every_repo(tmp_path):
    api = _Api(tmp_path)
    with pytest.raises(A.AdoptionRefusedError, match="owner-approval"):
        A.publish([AID], sandbox=False, api=api, staging=_staging(tmp_path), run=_ok_run, publisher=_publisher(api))
    assert api.calls == []  # nothing was attempted
    out = A.publish(
        [AID], sandbox=False, api=api, staging=_staging(tmp_path / "b"), run=_ok_run, publisher=_publisher(api),
        approved=["silly-kicks/ghost-gk-sweeper-v1"],
    )  # fmt: skip
    assert out[AID]["repo"] == "silly-kicks/ghost-gk-sweeper-v1"


def _verified(tmp_path, mutate) -> dict:
    api = _Api(tmp_path)
    staging = _staging(tmp_path)
    a = M.load().artifacts[AID]
    repo = "silly-kicks/sandbox-ghost-gk-sweeper-v1"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    _publisher(api)(api, str(staging / INST), repo, model_card=str(M.REPO_ROOT / (a.card or "")))
    run = mutate(api, repo) or _ok_run
    from scripts._hub_publish import MODEL_ONLY_ALLOWLIST

    return A.post_verify(
        api,
        repo,
        a,
        staging / INST,
        run=run,
        allowlist=MODEL_ONLY_ALLOWLIST,
        card=card_bytes(M.REPO_ROOT / (a.card or "")),
    )


def test_post_verify_passes_on_a_faithful_publish(tmp_path):
    assert _verified(tmp_path, lambda api, repo: None)["ok"]


def test_post_verify_catches_a_nested_path(tmp_path):
    def nested(api, repo):
        api.remote[repo]["_feature_cache/x.parquet"] = b"raw"

    assert any("model-only" in p for p in _verified(tmp_path, nested)["problems"])


def test_post_verify_catches_a_different_remote_sha256sums(tmp_path):
    def sums(api, repo):
        api.remote[repo]["SHA256SUMS"] = b"zzz  rfcde_weights.npz\n"

    assert any("SHA256SUMS" in p for p in _verified(tmp_path, sums)["problems"])


def test_post_verify_catches_a_readme_other_than_the_card(tmp_path):
    def readme(api, repo):
        api.remote[repo]["README.md"] = b"stale card"

    assert any("README" in p for p in _verified(tmp_path, readme)["problems"])


def test_post_verify_catches_a_failed_from_hub_load(tmp_path):
    def broken(api, repo):
        return lambda *_a, **_k: types.SimpleNamespace(returncode=1, stdout="", stderr="IntegrityError: chirality")

    assert any("from_hub" in p for p in _verified(tmp_path, broken)["problems"])
