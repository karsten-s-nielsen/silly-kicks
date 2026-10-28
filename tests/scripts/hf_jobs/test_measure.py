"""Read-only HF measurement jobs (spec r4.4; plan R3-8 S0b): `scripts/hf_jobs/measure.py`.

A measurement job runs unpushed code at production shape. The working-tree patch travels inside the job request
(never staged in a bucket), its sha256 is checked before `git apply`, every mount is read-only and the job gets no
secrets: so a measurement job cannot write a campaign store or produce an artifact.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts._dataset_reads import READ_CAP_ENV
from scripts.hf_jobs import measure as M
from scripts.hf_jobs.launch import BUCKET_MOUNT, DATA_MOUNT, FLAVOR, IMAGE, DatasetMount

BASE = "2474b73f65a76490586ddf33d41e9dc0acb9bef0"
_GIT = shutil.which("git") or "git"
_BASH = shutil.which("bash")
TC3 = DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "a" * 40, "mount", 4)
XSHOT115 = DatasetMount("xshot115", "silly-kicks/hf-refit-xshot115", "b" * 40, "mount", 2)
SB_OPEN = DatasetMount("sb_open", "statsbomb/open-data", "c" * 40, "open")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 -- fixed argv
        [_GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "kept.txt").write_text("one\n", encoding="utf-8")
    (repo / "changed.txt").write_text("before\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def _edit_worktree(repo: Path) -> None:
    (repo / "changed.txt").write_text("after\n", encoding="utf-8")
    (repo / "new.txt").write_text("untracked\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(bytes(range(256)) * 4)
    (repo / "ignored.txt").write_text("never carried\n", encoding="utf-8")


def test_patch_env_round_trips_in_chunks_under_the_argument_cap():
    patch = os.urandom(300_000)  # incompressible, so the payload spans several chunks
    env, digest = M.patch_env(patch)
    assert digest == hashlib.sha256(patch).hexdigest()
    assert len(env) > 1 and all(len(v) <= M.PATCH_CHUNK < 128 * 1024 for v in env.values())
    payload = "".join(env[f"{M.PATCH_ENV_PREFIX}{k}"] for k in range(len(env)))
    assert gzip.decompress(base64.b64decode(payload)) == patch
    assert M.patch_env(patch) == (env, digest)  # deterministic (gzip mtime 0)


def test_the_command_checks_out_the_base_and_checks_the_patch_before_applying_it():
    script = M.measure_command(base=BASE, probe="print('probe')", n_chunks=3, digest="d" * 64)[2]
    lines = script.split("\n")
    assert lines[0] == "set -euo pipefail"
    fetch = next(i for i, line in enumerate(lines) if line.startswith("git fetch"))
    verify = next(i for i, line in enumerate(lines) if "git rev-parse HEAD" in line)
    rebuild = next(i for i, line in enumerate(lines) if "base64 -d | gunzip" in line)
    check = next(i for i, line in enumerate(lines) if "sha256sum" in line)
    apply = next(i for i, line in enumerate(lines) if line.startswith("git apply"))
    assert fetch < verify < rebuild < check < apply
    assert BASE in lines[fetch] and BASE in lines[verify] and "d" * 64 in lines[check]
    assert all(f'"${{{M.PATCH_ENV_PREFIX}{k}}}"' in lines[rebuild] for k in range(3))
    assert BUCKET_MOUNT not in script  # the patch never comes from a bucket


def test_a_pushed_commit_runs_without_a_patch():
    script = M.measure_command(base=BASE, probe="print('probe')", n_chunks=0, digest=None)[2]
    assert "git apply" not in script and M.PATCH_ENV_PREFIX not in script


def test_a_probe_holding_the_heredoc_delimiter_is_refused():
    with pytest.raises(ValueError, match="heredoc delimiter"):
        M.measure_command(base=BASE, probe=f"x = 1\n{M.PROBE_EOF}\n", n_chunks=0, digest=None)


def test_every_mount_is_read_only():
    vols = M.measure_volumes(bucket="silly-kicks/hf-refit-scratch", datasets=[TC3, XSHOT115, SB_OPEN])
    assert [v.read_only for v in vols] == [True, True, True]
    assert vols[0].mount_path == BUCKET_MOUNT
    assert [(v.mount_path, v.revision) for v in vols[1:]] == [
        (f"{DATA_MOUNT}/tc3", "a" * 40),
        (f"{DATA_MOUNT}/xshot115", "b" * 40),
    ]


def test_open_data_rides_the_env_and_mounts_nothing():
    """`sb_open` is StatsBomb open data fetched in-job at its pinned revision (the campaign's `open` access): no
    volume, the revision in the environment, as a campaign job gets it."""
    env = M.measure_env([SB_OPEN, TC3, XSHOT115])
    assert env["SILLY_KICKS_OPEN_DATA_REVISION"] == "c" * 40
    assert env[READ_CAP_ENV] == "2"  # the job-wide read gate: the smallest mount cap
    assert env["SILLY_KICKS_IN_HF_JOB"] == "1"
    vols = M.measure_volumes(bucket=None, datasets=[SB_OPEN])
    assert vols == []


class _Api:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_job(self, **kwargs):
        self.calls.append(kwargs)
        return type("Info", (), {"id": "job-1"})()


def test_a_measurement_job_gets_no_secrets_and_carries_its_patch():
    api = _Api()
    patch = b"diff --git a/x b/x\n"
    job = M.launch_measurement(
        api,
        run_id="r1",
        base=BASE,
        probe="print('probe')",
        patch=patch,
        bucket="silly-kicks/hf-refit-scratch",
        datasets=[TC3],
        timeout_s=600,
    )
    assert job == "job-1"
    (call,) = api.calls
    assert "secrets" not in call  # no token: the job cannot authenticate to write
    assert call["image"] == IMAGE and call["flavor"] == FLAVOR and call["timeout"] == 600
    assert all(v.read_only for v in call["volumes"])
    digest = hashlib.sha256(patch).hexdigest()
    assert call["labels"] == {"campaign": "measure", "commit": BASE, "run_id": "r1", "patch": digest[:16]}
    assert f"{M.PATCH_ENV_PREFIX}0" in call["env"] and digest in call["command"][2]


def test_the_worktree_patch_carries_untracked_files_and_leaves_the_index_alone(tmp_path):
    repo, base = _repo(tmp_path)
    _edit_worktree(repo)
    status = _git(repo, "status", "--porcelain")
    patch = M.worktree_patch(repo, base)
    assert _git(repo, "status", "--porcelain") == status  # the repository's own index is untouched
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(repo), str(clone))
    _git(clone, "config", "core.autocrlf", "false")
    (tmp_path / "tree.patch").write_bytes(patch)
    _git(clone, "apply", "--binary", str(tmp_path / "tree.patch"))
    for name in ("changed.txt", "new.txt", "blob.bin", "kept.txt"):
        assert (clone / name).read_bytes() == (repo / name).read_bytes(), name
    assert not (clone / "ignored.txt").exists()


def test_the_worktree_patch_can_carry_only_named_paths(tmp_path):
    """The HF Jobs API refuses a large request (413), so a job carries only the paths it runs."""
    repo, base = _repo(tmp_path)
    _edit_worktree(repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    patch = M.worktree_patch(repo, base, paths=("pkg", "changed.txt"))
    assert b"pkg/mod.py" in patch and b"changed.txt" in patch
    assert b"new.txt" not in patch and b"blob.bin" not in patch


@pytest.mark.skipif(_BASH is None, reason="the job's shell lines need bash")
@pytest.mark.parametrize("tamper", [False, True])
def test_the_job_shell_rebuilds_the_patch_and_refuses_a_wrong_one(tmp_path, tamper):
    """The job's own shell lines (base64, gunzip, sha256sum, git apply), run on a clone of the base: a patch whose
    sha256 differs from the label is refused before `git apply`."""
    repo, base = _repo(tmp_path)
    _edit_worktree(repo)
    patch = M.worktree_patch(repo, base)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(repo), str(clone))
    _git(clone, "config", "core.autocrlf", "false")
    env, digest = M.patch_env(patch)
    if tamper:
        digest = hashlib.sha256(patch + b"x").hexdigest()
    work = tmp_path / "jobtmp"
    work.mkdir()
    script = "set -euo pipefail\n" + "\n".join(M.patch_lines(n_chunks=len(env), digest=digest, tmp=work.as_posix()))
    assert _BASH is not None
    done = subprocess.run(  # noqa: S603 -- fixed argv
        [_BASH, "-c", script], cwd=clone, env={**os.environ, **env}, capture_output=True
    )
    if tamper:
        assert done.returncode != 0
        assert (clone / "changed.txt").read_text(encoding="utf-8") == "before\n"  # nothing applied
    else:
        assert done.returncode == 0, done.stderr
        assert (clone / "new.txt").read_bytes() == (repo / "new.txt").read_bytes()
        assert (clone / "blob.bin").read_bytes() == (repo / "blob.bin").read_bytes()


# ---- HFU plan R3-8 S3 item 13: the canary mode --------------------------------------------------------------------


def test_the_canary_mode_mounts_exactly_the_canary_bucket_read_write():
    vols = M.measure_volumes(bucket="silly-kicks/hf-refit-scratch", datasets=[TC3], canary=True)
    writable = [(v.source, v.mount_path) for v in vols if not v.read_only]
    assert writable == [(M.CANARY_BUCKET, M.CANARY_MOUNT)]
    assert [v.read_only for v in vols if v.source != M.CANARY_BUCKET] == [True, True]  # the campaign bucket stays RO
    assert all(v.read_only for v in M.measure_volumes(bucket="silly-kicks/hf-refit-scratch", datasets=[TC3]))


def test_a_writable_campaign_bucket_is_refused_where_the_mounts_are_built():
    from huggingface_hub import Volume

    campaign = Volume(type="bucket", source="silly-kicks/hf-refit-scratch", mount_path=M.CANARY_MOUNT, read_only=False)
    with pytest.raises(AssertionError, match="writes only the canary bucket"):
        M.check_mounts([campaign])
    misplaced = Volume(type="bucket", source=M.CANARY_BUCKET, mount_path=BUCKET_MOUNT, read_only=False)
    with pytest.raises(AssertionError, match="writes only the canary bucket"):
        M.check_mounts([misplaced])  # the canary over the campaign mount path: refused too


def test_a_canary_job_gets_no_secrets_and_is_labelled():
    api = _Api()
    M.launch_measurement(
        api,
        run_id="r3",
        base=BASE,
        probe="print('probe')",
        patch=None,
        bucket="silly-kicks/hf-refit-scratch",
        datasets=[],
        timeout_s=600,
        canary=True,
    )
    (call,) = api.calls
    assert "secrets" not in call and call["labels"]["canary"] == "1"
    assert [(v.source, v.read_only) for v in call["volumes"]] == [
        ("silly-kicks/hf-refit-scratch", True),
        (M.CANARY_BUCKET, False),
    ]
