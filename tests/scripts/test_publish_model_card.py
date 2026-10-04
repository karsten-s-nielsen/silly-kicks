"""The card-only Hub push seam (combined-cycle spec section 9, D9; ADR-088 amendment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._hub_publish import CARD_SOURCE, card_bytes, publish_card_only

_REPO = "silly-kicks/xshot-occurrence-v1"


class _Sibling:
    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


class _Info:
    def __init__(self, files) -> None:
        self.siblings = [_Sibling(f) for f in files]


class _FakeHub:
    """An in-memory Hub repo: model_info lists files, hf_hub_download serves README bytes,
    upload_file replaces them. `corrupt` makes the read-back differ from the upload."""

    def __init__(self, tmp_path: Path, readme: bytes | None, *, exists: bool = True, corrupt: bool = False) -> None:
        self.tmp, self.readme, self.exists, self.corrupt = tmp_path, readme, exists, corrupt
        self.uploads: list[dict] = []
        self.reads: list[str] = []  # every Hub read, so "refused before any network" is assertable
        self.created: list[dict] = []

    def model_info(self, repo_id: str):
        self.reads.append(f"model_info:{repo_id}")
        if not self.exists:
            raise LookupError(f"404 {repo_id}")
        return _Info(["model.json"] + (["README.md"] if self.readme is not None else []))

    def hf_hub_download(self, *, repo_id, filename, repo_type, force_download):
        self.reads.append(f"download:{repo_id}/{filename}")
        assert filename == "README.md" and repo_type == "model" and force_download is True
        assert self.readme is not None
        path = self.tmp / f"dl_{len(self.reads)}.md"
        path.write_bytes(self.readme)
        return str(path)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        self.readme = kwargs["path_or_fileobj"] + (b"X" if self.corrupt else b"")

    def create_repo(self, **kwargs):
        self.created.append(kwargs)  # a card-only push must never call this


def _root(tmp_path: Path, body: bytes) -> Path:
    card = tmp_path / "repo" / CARD_SOURCE[_REPO]
    card.parent.mkdir(parents=True)
    card.write_bytes(body)
    return tmp_path / "repo"


def test_card_source_names_every_org_repo_and_every_card_exists():
    assert len(CARD_SOURCE) == 10
    for repo, rel in CARD_SOURCE.items():
        assert repo.startswith("silly-kicks/") and Path(rel).is_file(), (repo, rel)


def test_card_bytes_normalizes_crlf_and_refuses_a_missing_card(tmp_path):
    p = tmp_path / "c.md"
    p.write_bytes(b"---\r\nlicense: mit\r\n---\r\n# c\r\n")
    assert card_bytes(p) == b"---\nlicense: mit\n---\n# c\n"
    with pytest.raises(SystemExit, match="does not exist"):
        card_bytes(tmp_path / "nope.md")


def test_unregistered_repo_is_refused_before_any_network(tmp_path):
    hub = _FakeHub(tmp_path, b"x")
    with pytest.raises(SystemExit, match="not a registered"):
        publish_card_only(hub, "silly-kicks/not-a-repo", root=tmp_path)
    assert hub.reads == [] and hub.uploads == []  # B r4 CCC-PLAN-34: no Hub read at all


def test_a_missing_registered_card_is_refused_before_any_network(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    with pytest.raises(SystemExit, match="does not exist"):
        publish_card_only(hub, _REPO, root=tmp_path / "empty-root")
    assert hub.reads == [] and hub.uploads == []


def test_hub_sha256_before_is_the_hash_of_the_hub_bytes(tmp_path):
    import hashlib

    hub = _FakeHub(tmp_path, b"# old hub readme\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"), verify_only=True)
    assert out["hub_sha256_before"] == hashlib.sha256(b"# old hub readme\n").hexdigest()
    assert out["card_sha256"] == hashlib.sha256(b"# new\n").hexdigest()


def test_a_push_never_creates_a_repo(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))
    assert hub.created == []


def test_cli_verify_only_reads_and_never_uploads(tmp_path, monkeypatch, capsys):
    import json

    import huggingface_hub

    from scripts import publish_model_card as P

    hub = _FakeHub(tmp_path, b"# a different hub readme\n")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: hub)
    out = P.main(["--repo-id", _REPO, "--verify-only"])  # the real registered card under the repo root
    assert out["changed"] is True and out["uploaded"] is False and hub.uploads == []
    assert json.loads(capsys.readouterr().out)["repo_id"] == _REPO


def test_missing_repo_is_refused_before_any_upload(tmp_path):
    hub = _FakeHub(tmp_path, None, exists=False)
    with pytest.raises(LookupError):
        publish_card_only(hub, _REPO, root=_root(tmp_path, b"# card\n"))
    assert hub.uploads == []


def test_unchanged_card_is_not_reuploaded(tmp_path):
    hub = _FakeHub(tmp_path, b"# card\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# card\r\n"))
    assert out["changed"] is False and out["uploaded"] is False and hub.uploads == []


def test_changed_card_is_uploaded_as_lf_readme_and_read_back(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\r\n"))
    assert out["uploaded"] is True and out["hub_sha256_after"] == out["card_sha256"]
    (up,) = hub.uploads
    assert up["path_in_repo"] == "README.md" and up["repo_id"] == _REPO and up["repo_type"] == "model"
    assert up["path_or_fileobj"] == b"# new\n"


def test_a_repo_without_a_readme_gets_one(tmp_path):
    hub = _FakeHub(tmp_path, None)
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))
    assert out["hub_sha256_before"] is None and out["uploaded"] is True


def test_read_back_mismatch_fails_loud(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n", corrupt=True)
    with pytest.raises(SystemExit, match="READ-BACK MISMATCH"):
        publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))


def test_verify_only_never_uploads(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"), verify_only=True)
    assert out["changed"] is True and out["uploaded"] is False and hub.uploads == []


def test_cli_offers_only_registered_repos():
    from scripts import publish_model_card as P

    with pytest.raises(SystemExit):
        P.main(["--repo-id", "silly-kicks/not-a-repo", "--verify-only"])
