"""HFU R3-10 (D12): each migrated trainer's pipeline reproduces its legacy CLI byte for byte.

The legacy CLI is the verbatim oracle in ``tests/scripts/_legacy_trainers``. The numbers vary across platforms
and library versions, so the oracle and the pipeline run in the SAME environment here, on the committed offline
open-data tree (``_toy_corpora.build_open_data`` served by ``_open_data_server``). The test needs statsbombpy and
no network: it runs in the ``.venv-hfjobs`` legs and skips in CI, like the other open-data parity tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("statsbombpy")

from scripts._toy_corpora import build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import run_serial

REPO = Path(__file__).resolve().parents[2]

#: trainer -> (legacy oracle module, its CLI args, pipeline module, its args): the campaign's own build args
#: (each pipeline keeps its legacy options and defaults), on the toy tree's one competition (43/106).
CASES = {
    "match_outcome_dependence": (
        "tests.scripts._legacy_trainers.match_outcome_dependence",
        [],
        "scripts.train_match_outcome_dependence",
        [],
    ),
    "pass_completion": (
        "tests.scripts._legacy_trainers.pass_completion",
        ["--source", "open-data"],
        "scripts.train_pass_completion",
        ["--source", "open-data"],
    ),
    "xsuccess": ("tests.scripts._legacy_trainers.xsuccess", [], "scripts.train_xsuccess", []),
}

#: metrics.json keys that differ BY DESIGN: the extract's generation token (the legacy ``for_each`` shard
#: generation against the stage token; the ``win_probability`` migration precedent).
BY_DESIGN = {"generation"}

#: The ONE byte change an owner ruling allows (2026-10-30): pass_completion's card names the pipeline command
#: in its open-data "Reproduce with" line, because the legacy CLI is gone. Applied to the legacy card first.
CARD_COMMAND = {
    "pass_completion": (
        b"python scripts/train_pass_completion.py --out <DIR>",
        b"python -m scripts.train_pass_completion --store <DIR> --run-all --out <DIR>",
    )
}


def _bundle(out: Path) -> dict[str, bytes]:
    """Every artifact file under ``out`` by POSIX path, without the legacy shard tree and stage files."""
    files = {}
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and "shards" not in p.relative_to(out).parts and not rel.startswith("_"):
            files[rel] = p.read_bytes()
    return files


def _lf(data: bytes) -> bytes:
    """A text file as LF: the legacy CLIs write text with the platform newline, the pipelines write bytes."""
    return data.replace(b"\r\n", b"\n")


def _assert_sums_consistent(files: dict[str, bytes], name: str) -> None:
    """A SHA256SUMS file lists the digest of each sibling file, as written or with CRLF folded to LF: a model's
    ``save()`` hashes its LF payload and, on Windows, writes the file in text mode (CRLF), on both sides alike."""
    folder = name.rpartition("/")[0]
    for line in files[name].decode("ascii").splitlines():
        digest, _, listed = line.partition("  ")
        data = files[f"{folder}/{listed}" if folder else listed]
        assert digest in {hashlib.sha256(data).hexdigest(), hashlib.sha256(_lf(data)).hexdigest()}, f"{name}: {listed}"


def _run_legacy(module: str, args: list[str], out: Path, tmp: Path, env: dict) -> None:
    """The legacy CLI exactly as the campaign's one-unit adapter ran it: under ``deterministic_env`` (one BLAS/OMP
    thread), the environment the pipeline's pinned workers get too. xgboost's ``hist`` sums depend on the thread
    count, so an oracle on default threads would not be the campaign's baseline."""
    from scripts._thread_pin import deterministic_env

    cmd = [sys.executable, "-m", module, *args, "--out", str(out), "--allow-dirty", "--cache-dir", str(tmp / "cache")]
    if module.endswith("match_outcome_dependence"):
        cmd += ["--shard-root", str(tmp / "legacy-shards")]
    r = subprocess.run(cmd, cwd=REPO, env=deterministic_env(env), capture_output=True, text=True, check=False)  # noqa: S603
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.slow  # slow: three full trainer runs per case on the toy corpus (ADR-023)
@pytest.mark.parametrize("trainer", sorted(CASES))
def test_pipeline_artifact_is_byte_identical_to_the_legacy_cli(trainer, tmp_path):
    legacy_mod, legacy_args, module, args = CASES[trainer]
    root = build_open_data(tmp_path / "od")
    with serve_open_data(root) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        _run_legacy(legacy_mod, legacy_args, tmp_path / "legacy", tmp_path, env)
        run_serial(module, tmp_path / "store", tmp_path / "data", args, tmp_path / "pipeline", env=env)
    # The legacy run dir holds `metrics.json` + `weights/…`; the pipeline's bundle is FLAT (adopt refuses a nested
    # path) and ships `metrics.json` beside the weights (owner ruling 2026-10-30), so the names compare flattened.
    legacy = {name.removeprefix("weights/"): data for name, data in _bundle(tmp_path / "legacy").items()}
    pipeline = _bundle(tmp_path / "pipeline")
    assert sorted(pipeline) == sorted(legacy)
    for name in sorted(legacy):
        if name.endswith("SHA256SUMS"):
            _assert_sums_consistent(legacy, name)
            _assert_sums_consistent(pipeline, name)
        elif name == "metrics.json":  # canonical JSON text: a NaN metric compares equal to itself
            want = {k: v for k, v in json.loads(legacy[name]).items() if k not in BY_DESIGN}
            got = {k: v for k, v in json.loads(pipeline[name]).items() if k not in BY_DESIGN}
            assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
        elif name.endswith("MODEL_CARD.md") and trainer in CARD_COMMAND:
            old, new = CARD_COMMAND[trainer]
            assert _lf(legacy[name]).count(old) == 1, "the legacy card names the legacy command exactly once"
            assert _lf(pipeline[name]) == _lf(legacy[name]).replace(old, new), name
        elif name.endswith((".json", ".md")):
            assert _lf(pipeline[name]) == _lf(legacy[name]), name
        else:
            assert pipeline[name] == legacy[name], name
