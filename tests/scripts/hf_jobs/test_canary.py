"""HFU plan R3-8 S3 item 13: the committed rung-3 canary (`scripts/hf_jobs/canary.py`), run locally.

The writer runs the toy pipeline through the production executor and store into a local stand-in for the canary
mount, writes the probe files and one large output; the mount reader and the bucket-API reader (over a fake
`HfApi` bucket holding the same files) both verify everything. The HF runs are rung 3 (an owner gate).
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

from scripts.hf_jobs import canary as K
from tests.scripts.hf_jobs.test_step_snapshot import Bucket

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
N = 24


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("canary")
    real = tmp / "campaign" / "_output.parquet"
    real.parent.mkdir()
    real.write_bytes(os.urandom(300_000))
    rows = K.writer(
        "t1", root=tmp / "canary", python=sys.executable, work=REPO_ROOT, n_items=N, real_output=real, probe_gap_s=0.0
    )
    return tmp, rows


def _bucket_files(base: pathlib.Path) -> dict[str, bytes]:
    return {f"rung3-t1/{p.relative_to(base).as_posix()}": p.read_bytes() for p in base.rglob("*") if p.is_file()}


def test_the_writer_verifies_every_unit_and_writes_the_probes_and_the_large_output(written):
    tmp, rows = written
    assert rows["gen"]["ok"] == N and rows["gen"]["output_present"] and rows["gen"]["tmp_files"] == 0
    assert rows["combined"]["ok"] == 1 and rows["combined"]["output_present"]
    base = tmp / "canary" / "rung3-t1"
    assert len(list((base / "mtime").glob("*.bin"))) == K.N_MTIME
    assert (base / "real" / "_output.parquet").stat().st_size == 300_000


def test_the_mount_reader_sees_everything_with_mtimes_inside_the_write_windows(written):
    tmp, _rows = written
    rows = K.reader_mount("t1", root=tmp / "canary", n_items=N)
    assert rows["gen"]["ok"] == N and rows["combined"]["ok"] == 1
    assert rows["probes"]["bad"] == [] and rows["probes"]["files"] == K.N_MTIME and "delta_median" in rows["probes"]
    assert rows["real"] is not None and rows["real"]["ok"]


def test_the_api_reader_reads_through_the_bucket_api_not_the_mount(written):
    tmp, _rows = written
    files = _bucket_files(tmp / "canary" / "rung3-t1")
    rows = K.reader_api("t1", bucket="ns/b", n_items=N, api=Bucket(files))
    assert rows["gen"]["ok"] == N and rows["combined"]["ok"] == 1
    assert rows["probes"]["bad"] == [] and "delta_median" not in rows["probes"]  # no mtime through the API
    assert rows["real"] is not None and rows["real"]["ok"]


def test_the_api_reader_reports_a_corrupt_unit_and_a_corrupt_large_output(written):
    tmp, _rows = written
    files = _bucket_files(tmp / "canary" / "rung3-t1")
    side = next(k for k in files if k.endswith("toy__0003.sha256"))
    files[side] = b"0" * 64 + b"\n"
    big = next(k for k in files if k.endswith("real/_output.parquet"))
    files[big] = files[big][:-1] + b"x"
    rows = K.reader_api("t1", bucket="ns/b", n_items=N, api=Bucket(files))
    assert rows["gen"]["bad_sha"] == 1 and rows["gen"]["ok"] == N - 1
    assert rows["real"] is not None and not rows["real"]["ok"]


def test_a_reader_refuses_a_store_with_two_generations_of_a_stage():
    with pytest.raises(SystemExit, match="expected ONE gen generation"):
        K.writer_generations({"toy/gen/a/x.json", "toy/gen/b/x.json", "toy/combined/c/only.json"})
