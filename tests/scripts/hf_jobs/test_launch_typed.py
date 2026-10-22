"""Typed HF Jobs launcher (ADR-128): one image, one flavor, labels, pinned env, exact-commit checkout."""

from __future__ import annotations

import ast
import pathlib
import shlex
import shutil
import subprocess
from typing import Any

import pytest

pytest.importorskip("huggingface_hub", minversion="2.2")

from scripts.hf_jobs import launch as L

SHA = "4" * 40


def _spec(**kw):
    base: dict[str, Any] = dict(
        campaign="cycle1",
        run_key="cycle1-4444444-ghost_gk-owner",
        job="ghost_gk",
        variant="default",
        stages=("extract",),
        shard=(1, 4),
        module="scripts.train_ghost_gk",
        args=("--feature-set", "faithful", "--dataset", "tc3"),
        commit=SHA,
        bucket="silly-kicks/hf-refit-scratch",
        workers=3,
        inner_workers=1,
        cap_bytes=8 * 2**30,
        peak_rss_bytes=6 * 2**30,
        timeout_s=3600,
        attempts=3,
        datasets=(L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "mount", 4),),
        extras=(),
    )
    base.update(kw)
    return L.JobSpec(**base)


class _FakeApi:
    def __init__(self):
        self.calls = []

    def run_job(self, **kw):
        self.calls.append(kw)

        class _Info:
            id = "job123"

        return _Info()


def test_launch_uses_the_typed_api_with_the_one_flavor_and_image():
    api = _FakeApi()
    assert L.launch(_spec(), api=api, token="hf_x") == "job123"
    kw = api.calls[0]
    assert kw["flavor"] == "cpu-upgrade" == L.FLAVOR
    assert kw["image"] == L.IMAGE and "@sha256:" in L.IMAGE
    assert kw["secrets"] == {"HF_TOKEN": "hf_x"}
    assert kw["timeout"] == 3600 and kw["attempts"] == 3
    assert kw["labels"] == {
        "campaign": "cycle1",
        "run_key": "cycle1-4444444-ghost_gk-owner",
        "job": "ghost_gk",
        "variant": "default",
        "stage": "extract",
        "shard": "1of4",
        "commit": SHA,
    }
    assert kw["name"] == L.job_name(_spec()) == "cycle1-4444444-ghost_gk-owner--default--extract--1of4"


def test_the_job_name_is_explicit_bounded_and_unique():
    """Real HF rejected the first Cycle 1 launch: `400 Bad Request: Too big: expected string to have <=100
    characters at labels.name`. `run_job` without `name=` lets the server derive one from the long bash
    command; the `hf jobs run` CLI always sets it. The launcher names every job, within the 100 limit."""
    long = dict(run_key="cycle1-4444444-" + "x" * 60 + "-owner", stages=tuple(f"stage_{i}" for i in range(8)))
    a, b = L.job_name(_spec(**long)), L.job_name(_spec(**long, shard=(2, 4)))
    assert len(a) <= L.LABEL_MAX == 100 and len(b) <= 100 and a != b
    assert L.job_name(_spec(**long)) == a  # deterministic
    # real HF (second Cycle 1 launch): `Invalid string: must match pattern /^[a-zA-Z0-9_-]*$/ at labels.name`
    for name in (a, b, L.job_name(_spec())):
        assert L.LABEL_PATTERN.fullmatch(name), name


def test_a_label_value_outside_the_hf_alphabet_refuses_before_any_spend():
    api = _FakeApi()
    with pytest.raises(ValueError, match="pattern"):
        L.launch(_spec(variant="v1.2"), api=api, token="hf_x")
    assert api.calls == []


def test_a_label_value_over_the_hf_limit_refuses_before_any_spend():
    api = _FakeApi()
    with pytest.raises(ValueError, match="100"):
        L.launch(_spec(stages=tuple(f"a_rather_long_stage_name_{i}" for i in range(6))), api=api, token="hf_x")
    assert api.calls == []


def test_no_other_flavor_is_reachable():
    """INV-4: the only flavor literal in the launcher package is the constant."""
    for path in pathlib.Path("scripts/hf_jobs").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        flavors = {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("cpu-")
        }
        assert flavors <= {"cpu-upgrade"}, (path, flavors)


def test_job_env_is_deterministic_and_carries_identity():
    env = L.job_env(_spec())
    assert env["PYTHONHASHSEED"] == "0" and env["NUMBA_NUM_THREADS"] == "1"
    assert env["SILLY_KICKS_IMAGE"] == L.IMAGE
    assert env["SILLY_KICKS_RUN_KEY"] == "cycle1-4444444-ghost_gk-owner"
    assert env["SILLY_KICKS_JOB_LABEL"] == "ghost_gk.default.extract.1of4"
    assert env["SILLY_KICKS_IN_HF_JOB"] == "1"


def test_the_in_job_label_is_rebuilt_from_the_hf_labels():
    """The ledger joins a unit record to its job by this label (the startup measure), so the label the
    job's env carries must equal the one rebuilt from that job's HF labels."""
    s = _spec()
    assert L.label_of(L.labels_for(s)) == L.job_env(s)["SILLY_KICKS_JOB_LABEL"]


def test_job_command_checks_out_the_exact_commit_and_syncs_frozen():
    cmd = L.job_command(_spec())
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert f"git fetch -q --depth 1 origin {SHA}" in script
    assert f'test "$(git rev-parse HEAD)" = "{SHA}"' in script
    assert "--require-hashes" in script and L.UV_REQUIREMENT in script
    assert "uv sync --frozen" in script and "--group hfjobs" in script and "--extra train" in script
    run = script.splitlines()[-1]
    argv = shlex.split(run)
    assert argv[:3] == [".venv/bin/python", "-m", "scripts.train_ghost_gk"]
    assert argv[argv.index("--stages") + 1] == "extract" and argv[argv.index("--shard") + 1] == "1/4"
    assert argv[argv.index("--store") + 1] == "/scratch/cycle1-4444444-ghost_gk-owner"
    assert argv[argv.index("--data-root") + 1] == "/data"
    # Outputs persist in the rw bucket mount (`--store`); the job never uploads (the controller writes
    # only its own records, test_bucket_store.py::test_controller_uploads_only_its_own_records).
    assert "upload" not in script and "HfApi" not in script


def test_volumes_mount_the_bucket_rw_and_the_dataset_ro_at_a_pinned_revision():
    v = L.volumes_for(_spec())
    bucket, ds = v
    assert (bucket.type, bucket.source, bucket.mount_path, bucket.read_only) == (
        "bucket",
        "silly-kicks/hf-refit-scratch",
        "/scratch",
        False,
    )
    assert (ds.type, ds.source, ds.mount_path, ds.revision, ds.read_only) == (
        "dataset",
        "silly-kicks/hf-refit-tc3",
        "/data/tc3",
        "rev1",
        True,
    )


def test_download_access_fetches_only_this_jobs_units():
    s = _spec(datasets=(L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "download"),))
    script = L.job_command(s)[2]
    assert "-m scripts.hf_jobs.datasets fetch tc3" in script and "--shard 1/4" in script
    assert all(vol.type != "dataset" for vol in L.volumes_for(s))


def _fetch_argv(s) -> list[str]:
    line = next(ln for ln in L.job_command(s)[2].splitlines() if "scripts.hf_jobs.datasets fetch" in ln)
    return shlex.split(line)


def test_a_download_fetch_names_every_stage_of_the_job_and_the_profile_sample():
    """A grouped job fetches for each stage it runs; a profile job fetches only its sample (spec R3)."""
    ds = (L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "download"),)
    argv = _fetch_argv(_spec(datasets=ds, stages=("validate", "extract"), shard=(0, 1)))
    assert argv[argv.index("--stages") + 1] == "validate,extract" and "--profile" not in argv
    argv = _fetch_argv(_spec(datasets=ds, shard=(0, 1), profile=16))
    assert argv[argv.index("--profile") + 1] == "16"


def test_a_mounted_dataset_sets_the_jobs_read_cap_to_the_smallest_declared():
    from scripts._dataset_reads import READ_CAP_ENV

    two = (
        L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "mount", 4),
        L.DatasetMount("probe", "silly-kicks/hf-refit-probe", "rev2", "mount", 2),
    )
    assert L.job_env(_spec(datasets=two))[READ_CAP_ENV] == "2"
    assert L.job_env(_spec())[READ_CAP_ENV] == "4"
    dl = (L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "download"),)
    assert READ_CAP_ENV not in L.job_env(_spec(datasets=dl))  # local disk: no gate


@pytest.mark.parametrize(("access", "cap"), [("mount", None), ("mount", 0), ("download", 4), ("open", 4)])
def test_read_concurrency_is_required_for_a_mount_and_refused_otherwise(access, cap):
    with pytest.raises(ValueError, match="read_concurrency"):
        L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", access, cap)


def test_open_data_jobs_pin_the_revision():
    rev = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
    s = _spec(datasets=(L.DatasetMount("sb_open", "statsbomb/open-data", rev, "open"),))
    assert L.job_env(s)["SILLY_KICKS_OPEN_DATA_REVISION"] == rev


def test_the_job_script_is_valid_bash():
    """The job runs ``bash -c <script>``: a syntax error fails every job at start, after the image pull.
    Checked over every dataset branch (mounted, downloaded per shard, open data)."""
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - CI (ubuntu) always has bash
        pytest.skip("bash not available")
    s = _spec(
        datasets=(
            L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "mount", 4),
            L.DatasetMount("probe", "silly-kicks/hf-refit-probe", "rev2", "download"),
            L.DatasetMount("sb_open", "statsbomb/open-data", "4b73468", "open"),
        ),
        extras=("xgboost",),
    )
    r = subprocess.run([bash, "-n", "-c", L.job_command(s)[2]], capture_output=True, text=True)  # noqa: S603
    assert r.returncode == 0, r.stderr


def test_an_lpt_part_runs_its_keys_without_a_round_robin_shard():
    s = _spec(shard_keys_path="/scratch/rk/_partitions/ghost_gk/extract/tok/1of4.json")
    argv = shlex.split(L.job_command(s)[2].splitlines()[-1])
    assert argv[argv.index("--shard-keys") + 1].endswith("1of4.json") and "--shard" not in argv
    assert L.labels_for(s)["shard"] == "1of4"
