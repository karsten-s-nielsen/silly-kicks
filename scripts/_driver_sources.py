"""The NEUTRAL loader port for stage-contract drivers and trainers (ADR-052 D14, ADR-128 spec 5.8 item 4).

A pipeline that runs on HF reads STAGED NEUTRAL datasets through a data path (INV-5, ADR-123 Amendment B):
it never reaches the pining client and never carries a public/private concept. This module is that data
path: cheap per-unit references (``StagedRef``) over the two staged layouts, plus the two neutral helpers
the open-data adapter shares with the pining adapter (``LoadedMatch``, ``resolve_cache_dir``).

The three adapters of the port are: pining (``scripts._loader_pining.pining_source``, owner, DGX only),
staged-neutral (this module, HF), and StatsBomb open data (``scripts._sb_open_data.open_data_source``,
public). The I1 import-closure gate (``tests/scripts/hf_jobs/test_visibility_blind_gate.py``) asserts that
no module a campaign launches reaches a visibility symbol; this module imports neither ``_loader_pining``
nor ``_corpus``.

Unit references read the staging lock (``scripts/hf_jobs/datasets.lock.json``) when the dataset is
recorded there (an HF job, where the dataset may be mounted lazily), else the local tree (a dev or toy
corpus). Dot-directories are never units (the ``hf download`` ``.cache/`` crash of 4.138.0).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd

#: The one raw-artifact cache env var, in the repo's ``SILLY_KICKS_*`` convention (spec §4.1).
CORPUS_CACHE_ENV = "SILLY_KICKS_CORPUS_CACHE_DIR"


def resolve_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """The raw-artifact cache root: the ARGUMENT, else ``$SILLY_KICKS_CORPUS_CACHE_DIR``, else ``None``.

    ``None`` is today's behaviour -- a temp dir per match, nothing persisted. Pining artifacts live
    under ``<root>/<provider>/<match_id>/``, the layout ``cache_dir`` always had, so an existing cache
    directory works unchanged when pointed at. A blank env value reads as unset: it is a typo, and
    treating it as the working directory would scatter artifact trees wherever a driver ran.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(CORPUS_CACHE_ENV, "").strip()
    return Path(env) if env else None


class LoadedMatch(NamedTuple):
    """One loaded match. ``frames`` is ``None`` iff ``events_only`` (pining); an empty DataFrame for
    open data. ``visible_area`` is the SB360 per-action polygon table (statsbomb full loads only), and
    ``report`` is the SkillCorner ``TrackingConversionReport`` on a full load (it carries the S1 rates
    Task 0 records), else ``None`` (interp 2)."""

    provider: str
    match_id: str
    actions: pd.DataFrame
    frames: pd.DataFrame | None
    home_team_id: object
    visible_area: pd.DataFrame | None
    report: object | None


@dataclasses.dataclass(frozen=True)
class StagedRef:
    """A cheap reference to ONE unit of a staged neutral dataset (ADR-123 Amendment B, ADR-128).

    ``files`` are the dataset-relative paths the unit reads, so a job fetches only its own units
    (resume-before-download, spec 5.3 item 3). ``source_provider`` comes from the staging lock (HF) or
    from the shard's first row group (a local dev/toy tree)."""

    provider: str
    game: str
    stem: str
    source_provider: str
    files: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.game)


def dataset_dir(data_root, dataset: str | None) -> Path:
    root = Path(data_root)
    return root / dataset if dataset else root


#: Dataset-level files every unit of a staged dataset may read (an ItemStage's declared files are its
#: unit refs' ``files`` plus these; `datasets.stage_files`).
DATASET_METADATA: tuple[str, ...] = ("home_teams.json",)


# ---- reads (ADR-128, the dataset read gate) ---------------------------------------------------------
# Every read of a staged-dataset file goes through these: ONE whole-file sequential read under the job's
# read gate (`scripts._dataset_reads`), parsed from memory. Never pyarrow directly on a mounted path: on
# real HF that pattern stalled for minutes per file and failed with EIO at 8 workers. The read-audit gate
# (`tests/scripts/test_dataset_read_audit.py`) runs every pipeline and refuses a direct read.


def read_dataset_parquet(path: str | os.PathLike) -> pd.DataFrame:
    """A staged parquet file as a DataFrame: read whole under the read gate, parsed from memory."""
    import io

    from scripts._dataset_reads import read_bytes

    return pd.read_parquet(io.BytesIO(read_bytes(path)))


def read_dataset_json(path: str | os.PathLike):
    """A staged JSON file (e.g. ``home_teams.json``), read whole under the read gate."""
    from scripts._dataset_reads import read_bytes

    return json.loads(read_bytes(path))


def read_dataset_schema(path: str | os.PathLike):
    """A staged parquet file's arrow schema: a footer read under the read gate (no data pages)."""
    import pyarrow.parquet as pq

    from scripts._dataset_reads import read_slot

    with read_slot():
        return pq.read_schema(path)


def _lock_units(dataset: str | None) -> dict | None:
    if not dataset:
        return None
    try:
        from scripts.hf_jobs.datasets import locked

        return locked(dataset)
    except KeyError:
        if os.environ.get("SILLY_KICKS_IN_HF_JOB"):  # in-job: an unrecorded dataset is a hard error
            raise
        return None


def _first_provider(path: Path) -> str:
    import pyarrow.parquet as pq

    from scripts._dataset_reads import read_slot

    with read_slot():
        pf = pq.ParquetFile(path)
        if "source_provider" not in pf.schema_arrow.names or pf.num_row_groups == 0:
            return "unknown"
        col = pf.read_row_group(0, columns=["source_provider"])["source_provider"]
    return str(col[0].as_py()) if len(col) else "unknown"


def staged_flat_refs(data_root, dataset: str | None) -> list[StagedRef]:
    """Units of a ``flat_tc3`` dataset: ``shards/<gen>/<provider>__<game>.parquet`` + ``_actions/``."""
    rec = _lock_units(dataset)
    if rec is not None:
        gen = rec["generation"]
        units = [(u["stem"], u["source_provider"]) for u in rec["units"]]
    else:
        root = dataset_dir(data_root, dataset)
        gens = sorted(p.name for p in (root / "shards").iterdir() if p.is_dir() and not p.name.startswith("."))
        if len(gens) != 1:
            raise ValueError(f"{root}: expected ONE shards/<generation>/ directory, found {gens}")
        gen = gens[0]
        units = [(p.stem, _first_provider(p)) for p in sorted((root / "shards" / gen).glob("*.parquet"))]
    refs = []
    for stem, src in sorted(units):
        provider, _, game = stem.partition("__")
        files = (f"shards/{gen}/{stem}.parquet", f"_actions/{stem}.parquet")
        refs.append(StagedRef(provider, game, stem, src, files))
    return refs


def staged_pergame_refs(data_root, dataset: str | None) -> list[StagedRef]:
    """Units of a ``pergame`` dataset: ``<game>/frames.parquet`` (+ ``actions.parquet``/``shots.parquet``).
    Dot-directories are never units (the ``hf download`` ``.cache/`` crash of 4.138.0)."""
    rec = _lock_units(dataset)
    if rec is not None:
        units = [(u["stem"], u["source_provider"]) for u in rec["units"]]
    else:
        root = dataset_dir(data_root, dataset)
        units = [
            (p.name, _first_provider(p / "frames.parquet"))
            for p in sorted(root.iterdir())
            if p.is_dir() and not p.name.startswith(".") and (p / "frames.parquet").is_file()
        ]
    return [
        StagedRef(src, stem, stem, src, (f"{stem}/frames.parquet", f"{stem}/actions.parquet", f"{stem}/shots.parquet"))
        for stem, src in sorted(units)
        if not stem.startswith(".")
    ]


def home_teams(data_root, dataset: str | None) -> dict:
    p = dataset_dir(data_root, dataset) / "home_teams.json"
    return read_dataset_json(p) if p.is_file() else {}
