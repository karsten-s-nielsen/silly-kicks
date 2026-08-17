"""Tests for the SB360 licensed-corpus validation driver.

Runs against the committed open-360 golden slice via ``--fixture-only`` (no network). The licensed
30-match run is owner-run. The load-bearing guard is leak safety: a licensed per-match shard must
never land under ``docs/research/`` -- only the reconciled aggregate does.
"""

from __future__ import annotations

import inspect
import json

import scripts.validate_sb360_licensed_corpus as drv


def test_shards_go_to_gitignored_root_not_docs_research(tmp_path):
    shard_root = tmp_path / "gitignored_shards"
    out = tmp_path / "docs" / "research" / "sb360_licensed_coverage"
    # --allow-dirty because the dev/CI tree is modified; the run still records dirty:true.
    drv.main(["--shard-root", str(shard_root), "--out", str(out), "--fixture-only", "--allow-dirty"])

    # The per-match shard (keyed by match id) lives under the gitignored shard root...
    assert list(shard_root.rglob("3893795.parquet")), "per-match shard missing under --shard-root"
    # ...and NOT under docs/research: only the reconciled aggregate + manifest live there.
    assert not list((tmp_path / "docs").rglob("3893795.parquet")), "per-match shard LEAKED under docs/research"
    assert (out / "coverage.parquet").exists(), "reconciled aggregate not written"
    assert (out / "manifest_all.json").exists(), "manifest not written"


def test_manifest_carries_provenance(tmp_path):
    out = tmp_path / "out"
    drv.main(["--shard-root", str(tmp_path / "s"), "--out", str(out), "--fixture-only", "--allow-dirty"])
    manifest = json.loads((out / "manifest_all.json").read_text())
    # git_provenance() keys are stamped into the artifact (commit + the dirty boolean).
    assert "commit" in manifest
    assert manifest.get("dirty") is True  # --allow-dirty run records the fact, never launders it


def test_measure_match_tidy_schema_and_kinds():
    item = next(drv._fixture_items())
    frame = drv.measure_match(item)
    assert set(frame.columns) == set(drv._EMITTED_SHARD_COLUMNS)
    assert list(frame.columns) == list(drv._EMITTED_SHARD_COLUMNS)  # order is the shard schema
    kinds = set(frame["kind"])
    assert {
        "battery_column",
        "companion_source",
        "companion_fraction",
        "pitch_coverage",
        "frame_coverage",
        "roster",
    } <= kinds


def test_honest_degradation_vocabulary_is_exercised_on_real_data():
    """On the real WWC2023 slice the companions must genuinely degrade, not classify everything
    ``observed`` -- otherwise the whole coverage feature is a rubber stamp."""
    item = next(drv._fixture_items())
    frame = drv.measure_match(item)
    src = frame[frame["kind"] == "companion_source"]
    exercised = {s.split(".")[-1] for s, v in zip(src["subject"], src["value"], strict=False) if v and v > 0}
    assert {"no_polygon", "unlinked"} & exercised, f"no degradation tokens on real data: {exercised}"


def test_driver_has_argparse_so_help_is_safe():
    """The parserless-scripts trap: 16 scripts run main() on --help. This one has argparse."""
    src = inspect.getsource(drv)
    assert "add_argument" in src
    assert "def main(argv=None)" in src  # argv-injectable for the hermetic tests above
