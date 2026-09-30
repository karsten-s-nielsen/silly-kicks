"""Persist the trial-invariant F1b study inputs so studies can run as parallel workers (spec 6, 5c).

The ~15 nested-HPO studies (3 candidates x k folds) all consume the same ``(X, y, groups)`` plus the
same ``providers`` / ``match_ids`` / ``is_public`` that build the candidate masks and the public-fold
split. This module writes those once into a shard root and reads them back, so:

- a worker (`run_one_study`) reconstructs exactly the rows a serial ``_paired_data_effect`` would feed
  its ``_hpo_once`` for one ``(candidate, fold)``, and
- the reduce (`assemble_studies`) rebuilds the identical masks + folds to run the cheap paired deltas,
  ship decision, and final fit over the cached study params.

``X`` rides the zero-copy corpus mmap (``_corpus_mmap``); the small arrays and the scalar config are
plain ``.npy`` / JSON. Everything is pickle-free (house style): object arrays become fixed-width
unicode, which ``np.save`` writes without pickle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._corpus_mmap import load_design_matrix, persist_design_matrix


@dataclass
class StudyInputs:
    X: pd.DataFrame  # mmap-backed when uniform dtype
    y: np.ndarray
    groups: np.ndarray
    providers: np.ndarray
    match_ids: np.ndarray
    is_public: np.ndarray
    config: dict


_CONFIG = "study_config.json"


def _save_str_array(path: Path, arr) -> None:
    a = np.asarray(arr)
    np.save(path, a.astype(str) if a.dtype == object else a)


def persist_study_inputs(
    shard_root: str | Path,
    *,
    X,
    y,
    groups,
    providers,
    match_ids,
    is_public,
    config: dict,
) -> Path:
    """Write ``(X, y, groups, providers, match_ids, is_public)`` + ``config`` under ``shard_root``."""
    root = Path(shard_root)
    root.mkdir(parents=True, exist_ok=True)
    persist_design_matrix(X, y, groups, root / "corpus")
    _save_str_array(root / "providers.npy", providers)
    _save_str_array(root / "match_ids.npy", match_ids)
    np.save(root / "is_public.npy", np.asarray(is_public, dtype=bool))
    (root / _CONFIG).write_text(json.dumps(config), encoding="utf-8")
    return root


def load_study_inputs(shard_root: str | Path) -> StudyInputs:
    """Reconstruct the study inputs written by :func:`persist_study_inputs`."""
    root = Path(shard_root)
    X, y, groups = load_design_matrix(root / "corpus")
    return StudyInputs(
        X=X,
        y=y,
        groups=groups,
        providers=np.load(root / "providers.npy", allow_pickle=False),
        match_ids=np.load(root / "match_ids.npy", allow_pickle=False),
        is_public=np.load(root / "is_public.npy", allow_pickle=False),
        config=json.loads((root / _CONFIG).read_text(encoding="utf-8")),
    )
