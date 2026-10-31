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
from dataclasses import dataclass, field
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
    #: ADR-123 Amendment B: candidate-arm row masks {name -> bool mask}, supplied via the --arms job
    #: parameter and persisted so fan-out study workers rebuild byte-identical masks with NO provider
    #: logic. Empty for a pre-Amendment-B inputs dir (back-compat).
    arms: dict[str, np.ndarray] = field(default_factory=dict)
    #: TF-46: the per-ROW stoppage source of each training row (``len == len(X)``, aligned like
    #: ``match_ids``/``providers``), so the trainer stamps the shipped bundle's stoppage provenance from
    #: exactly the shipped-arm rows (Task 4). Empty for a pre-TF-46 inputs dir (back-compat): the combiner
    #: then gets NO parts and FAILS LOUD, never a synthesized ``ball_state`` (SPEC-20).
    stoppage_sources: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))


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
    arms: dict[str, np.ndarray] | None = None,
    stoppage_sources: np.ndarray | None = None,
) -> Path:
    """Write ``(X, y, groups, providers, match_ids, is_public)`` + ``config`` under ``shard_root``.

    ``arms`` (ADR-123 Amendment B) are the candidate-arm row masks; each is written to
    ``arms/<name>.npy`` so fan-out study workers rebuild byte-identical masks from the parameter.

    ``stoppage_sources`` (TF-46) is the per-ROW stoppage source, aligned to ``X`` (``len ==
    len(X)``); written to ``stoppage_sources.npy``. A length mismatch raises (fail loud); ``None``
    writes nothing (a pre-TF-46 inputs dir).
    """
    root = Path(shard_root)
    root.mkdir(parents=True, exist_ok=True)
    persist_design_matrix(X, y, groups, root / "corpus")
    _save_str_array(root / "providers.npy", providers)
    _save_str_array(root / "match_ids.npy", match_ids)
    np.save(root / "is_public.npy", np.asarray(is_public, dtype=bool))
    if arms:
        arms_dir = root / "arms"
        arms_dir.mkdir(exist_ok=True)
        for name, mask in arms.items():
            np.save(arms_dir / f"{name}.npy", np.asarray(mask, dtype=bool))
    if stoppage_sources is not None:
        arr = np.asarray(stoppage_sources)
        if arr.shape[0] != len(X):
            raise ValueError(
                f"persist_study_inputs: stoppage_sources has {arr.shape[0]} rows, X has {len(X)} "
                "(the stoppage source is per-ROW, like match_ids/providers)"
            )
        _save_str_array(root / "stoppage_sources.npy", arr)
    (root / _CONFIG).write_text(json.dumps(config), encoding="utf-8")
    return root


def _study_shard_path(study_shard_dir: str | Path, tag: str) -> Path:
    return Path(study_shard_dir) / f"{tag}.study.json"


def read_study_shard(study_shard_dir: str | Path, tag: str, *, objective_id: str, n_trials: int) -> dict | None:
    """The cached frozen params for study ``tag`` -- or None when there is no shard to TRUST.

    A shard is the result of one HPO store, so it is reused under exactly the store's resume rule (D21):
    only when it was written under the same ``objective_id`` (objective class + commit + declared inputs
    + per-fold tag; a dirty tree's id carries a per-call nonce, so it never matches) and the same
    ``n_trials`` (a larger budget resumes the store and can move the best params). Anything else -- a
    shard from another run's code / corpus / trial budget, or an identity-less one -- is recomputed,
    never served stale.
    """
    shard = _study_shard_path(study_shard_dir, tag)
    if not shard.exists():
        return None
    cached = json.loads(shard.read_text(encoding="utf-8"))
    if cached.get("objective_id") != objective_id or cached.get("n_trials") != n_trials:
        return None
    return dict(cached["params"])


def write_study_shard(study_shard_dir: str | Path, tag: str, params: dict, *, objective_id: str, n_trials: int) -> Path:
    """Write study ``tag``'s frozen params with the identity :func:`read_study_shard` checks."""
    shard = _study_shard_path(study_shard_dir, tag)
    shard.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tag": tag, "objective_id": objective_id, "n_trials": n_trials, "params": params}
    shard.write_text(json.dumps(payload), encoding="utf-8")
    return shard


def load_study_inputs(shard_root: str | Path) -> StudyInputs:
    """Reconstruct the study inputs written by :func:`persist_study_inputs`."""
    root = Path(shard_root)
    X, y, groups = load_design_matrix(root / "corpus")
    arms_dir = root / "arms"
    arms = {p.stem: np.load(p, allow_pickle=False) for p in sorted(arms_dir.glob("*.npy"))} if arms_dir.exists() else {}
    stoppage_path = root / "stoppage_sources.npy"
    # Absence tolerated (arms-style back-compat): an empty array, so the combiner gets NO parts and fails
    # loud, never a synthesized "ball_state" (SPEC-20).
    stoppage_sources = (
        np.load(stoppage_path, allow_pickle=False) if stoppage_path.exists() else np.empty(0, dtype=object)
    )
    return StudyInputs(
        X=X,
        y=y,
        groups=groups,
        providers=np.load(root / "providers.npy", allow_pickle=False),
        match_ids=np.load(root / "match_ids.npy", allow_pickle=False),
        is_public=np.load(root / "is_public.npy", allow_pickle=False),
        config=json.loads((root / _CONFIG).read_text(encoding="utf-8")),
        arms=arms,
        stoppage_sources=stoppage_sources,
    )


def provenance_from_sources(stoppage_sources, match_ids, mask) -> dict:
    """The shipped bundle's stoppage provenance, combined from the per-ROW sources of the SELECTED rows
    (TF-46, Task 4). One part per contributing match (``match_ids``), unioned by
    :func:`~silly_kicks.tracking.combine_stoppage_provenance`.

    ``mask`` is the shipped-arm row mask over the full matrix. The default / pre-TF-46 inputs dir carries
    an empty ``stoppage_sources`` whose length does not match ``mask``: the combiner then gets NO parts and
    RAISES ("no parts"), never synthesizing ``"ball_state"`` -- a refit without stoppage sources must FAIL
    LOUD (SPEC-20). Each part is validated (rule version + known source) by ``combine_stoppage_provenance``.
    """
    from silly_kicks.tracking._stoppages import STOPPAGE_RULE_VERSION, combine_stoppage_provenance

    srcs = np.asarray(stoppage_sources)
    mids = np.asarray(match_ids)
    sel = np.asarray(mask, dtype=bool)
    parts: list[dict] = []
    if srcs.shape[0] == sel.shape[0] and mids.shape[0] == sel.shape[0]:
        sel_srcs, sel_mids = srcs[sel], mids[sel]
        for m in pd.unique(sel_mids):
            part = sorted({str(s) for s in sel_srcs[sel_mids == m]})
            parts.append({"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": part})
    return combine_stoppage_provenance(parts)


def match_stoppage_source(frames) -> str:
    """The single stoppage source of one match's frames (TF-46 Task 4): an event-only provider attaches one
    source (``events_v{N}``) to every frame; an observed provider without the ``stoppage_state`` column reports
    ``ball_state``. One provider per match => one source, broadcast per training row. A match that somehow mixes
    sources FAILS LOUD (a per-row scalar cannot carry two). The token equals what
    :func:`~silly_kicks.tracking.stoppage_provenance` records, so the combined provenance validates.
    """
    from silly_kicks.tracking._stoppages import stoppage_provenance

    sources = stoppage_provenance(frames, where="train extract")["stoppage_sources"]
    if len(sources) != 1:
        raise ValueError(f"match_stoppage_source: expected one stoppage source per match, got {sources}")
    return str(sources[0])


def contributing_units_by_provider(providers, match_ids, mask) -> dict[str, int]:
    """The POST-DROP per-provider contributing-unit (distinct match) count over the SELECTED rows (TF-46 Task 4):
    the ACTUAL that ``adopt`` compares against the INTENDED shipped-arm membership (A PLAN-03; the two come from
    DIFFERENT sources or the guard is circular). Keyed by the raw ``source_provider`` (the arm-key provider), so it
    lines up with the off-HF arm-policy intended counts.
    """
    prov = np.asarray(providers)[np.asarray(mask, dtype=bool)]
    mids = np.asarray(match_ids)[np.asarray(mask, dtype=bool)]
    prov_str = np.array([str(v) for v in prov])
    out: dict[str, int] = {}
    for p in sorted(set(prov_str.tolist())):
        out[p] = len({str(m) for m in mids[prov_str == p]})
    return out
