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

    def snapshot_download(self, *, repo_id, repo_type, local_dir):
        self.calls.append(("snapshot_download", repo_id, str(local_dir)))
        for name, data in self.remote[repo_id].items():
            q = Path(local_dir) / name
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
        return str(local_dir)


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
        staging=staging,
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


def test_publish_uploads_the_finalized_copy(tmp_path):
    """HFU spec 18.6: the Hub copy carries canonical probe ids (the finalized staging copy is what is uploaded)."""
    api = _Api(tmp_path)
    s = _staging(tmp_path)
    m = {"probe_sample_matches": [["gradientsports", "gradientsports-10502"]], "probe_sample_in_training_folds": {}}
    (s / INST / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    A.publish([AID], sandbox=True, api=api, staging=s, run=_ok_run, publisher=_publisher(api))
    remote = json.loads(api.remote["silly-kicks/sandbox-ghost-gk-sweeper-v1"]["metrics.json"])
    assert remote["probe_sample_matches"] == [["gradientsports", "10502"]]


# ---- an artifact without a Hub class round-trips with the job loader (HFU fit-units spec C2) ---------

XAID = "hub_xsuccess"
XINST = "cycle1-ccccccc-xsuccess-public/default"
XREPO = "silly-kicks/sandbox-xsuccess-v1"


def _event_only(tmp_path, rc: int = 0, *, stale: dict[str, bytes] | None = None, lose: str | None = None):
    """`post_verify` of ``hub_xsuccess`` (``hub_class = ""``) over a faithful fake publish; the run is recorded."""
    from scripts._hub_publish import MODEL_ONLY_ALLOWLIST

    api, staging, runs = _Api(tmp_path), tmp_path / "staging", []
    d = staging / XINST
    d.mkdir(parents=True)
    (d / "model.json").write_bytes(b"{}")
    (d / "metadata.json").write_text("{}", encoding="utf-8")
    (d / "SHA256SUMS").write_text("abc  model.json\n", encoding="utf-8")
    a = M.load().artifacts[XAID]
    assert a.hub_class == ""
    api.create_repo(XREPO, repo_type="model", private=True, exist_ok=True)
    _publisher(api)(api, str(d), XREPO, model_card=str(M.REPO_ROOT / (a.card or "")))

    dest = staging / "_post_verify" / XREPO.partition("/")[2]
    for name, data in (stale or {}).items():  # files that an EARLIER post_verify left in the folder
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(data)
    if lose is not None:  # the Hub no longer holds this file
        del api.remote[XREPO][lose]

    def run(argv, **_k):
        runs.append([str(x) for x in argv])
        missing = lose is not None and not (dest / lose).exists()
        code = 1 if (rc or missing) else 0
        return types.SimpleNamespace(returncode=code, stdout="", stderr="XSuccessIntegrityError: sha")

    out = A.post_verify(
        api,
        XREPO,
        a,
        d,
        run=run,
        allowlist=MODEL_ONLY_ALLOWLIST,
        card=card_bytes(M.REPO_ROOT / (a.card or "")),
        staging=staging,
    )
    return api, out, runs, staging


def test_post_verify_uses_the_job_loader_without_a_hub_class(tmp_path):
    api, out, runs, staging = _event_only(tmp_path)
    assert out["ok"], out
    dest = staging / "_post_verify" / "sandbox-xsuccess-v1"
    assert ("snapshot_download", XREPO, str(dest)) in api.calls
    assert (dest / "model.json").read_bytes() == b"{}"  # the PUBLISHED files are what is loaded
    assert len(runs) == 1 and str(dest) in runs[0]
    assert any(".load(" in x for x in runs[0]) and not any("from_hub" in x for x in runs[0])


def test_post_verify_does_not_load_a_stale_local_file(tmp_path):
    """A file that an earlier run left in the folder, and that the Hub no longer holds, is not loaded."""
    _api, out, _runs, staging = _event_only(tmp_path, stale={"model.json": b"{}", "old.bin": b"x"}, lose="model.json")
    dest = staging / "_post_verify" / "sandbox-xsuccess-v1"
    assert not (dest / "model.json").exists() and not (dest / "old.bin").exists()
    assert not out["ok"] and any(p.startswith("load failed") for p in out["problems"])


def test_post_verify_reports_a_failed_job_loader(tmp_path):
    _api, out, _runs, _staging_dir = _event_only(tmp_path, rc=1)
    assert not out["ok"] and any(p.startswith("load failed") for p in out["problems"])


def test_post_verify_uses_from_hub_with_a_hub_class(tmp_path):
    runs: list = []

    def recorder(api, repo):
        def run(argv, **_k):
            runs.append([str(x) for x in argv])
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        return run

    assert _verified(tmp_path, recorder)["ok"]
    assert len(runs) == 1 and any("from_hub" in x for x in runs[0])
