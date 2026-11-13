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
import math
import sys
import time
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


_CONFIG = "study_config.json"

#: The tag suffix of a nested gate-fold study (ADR-131): ``{arm}_gf{k}`` tunes on the arm's rows outside gate fold k.
#: It never collides with the paired path's ``{name}_f{fold}`` tags, which use the public folds.
GATE_FOLD_SEP = "_gf"


def study_tags(arm_names, n_public_folds: int, cv_arms: dict[str, int], *, run_paired: bool) -> list[str]:
    """The ``study`` stage's tags: today's set (the paired ``{name}_f{fold}`` studies and the full-rows studies, or
    ``single``), then one nested gate-fold study per cv arm and gate fold (ADR-131), in sorted-arm, fold order.

    An arm name that contains ``_gf`` is refused: ``x_gf1`` would collide with arm ``x``'s gate-fold study tag, and
    `gate_fold_rows` would resolve it to arm ``x``'s rows."""
    clash = sorted(n for n in [*arm_names, *cv_arms] if GATE_FOLD_SEP in n)
    if clash:
        raise ValueError(f"arm names must not contain {GATE_FOLD_SEP!r} (the gate-fold study suffix): {clash}")
    if run_paired:
        names = sorted(arm_names)
        tags = [f"{name}_f{fold}" for fold in range(n_public_folds) for name in names] + names
    else:
        tags = ["single"]
    return tags + [f"{arm}{GATE_FOLD_SEP}{k}" for arm in sorted(cv_arms) for k in range(int(cv_arms[arm]))]


def gate_fold_rows(tag: str, arm_rows, plan, y: np.ndarray, groups: np.ndarray) -> np.ndarray | None:
    """The rows a nested gate-fold study ``{arm}_gf{k}`` tunes on: the arm's rows (``arm_rows(arm)``, ascending)
    outside gate fold ``k`` of the trainer's gate fold ``plan(y, groups)``, ascending. ``None`` if ``tag`` is not a
    gate-fold tag. An unknown arm raises (from ``arm_rows``)."""
    name, sep, k = tag.rpartition(GATE_FOLD_SEP)
    if not (sep and name and k.isdigit()):
        return None
    rows = arm_rows(name)
    _i, tr, _te = plan(y[rows], groups[rows])[int(k)]
    return rows[np.sort(tr)]


class TrialPrinter:
    """The ``OptunaStrategy.run(observer=...)`` sink of a study (HFU spec 14.8): one line per finished trial,
    ``[progress] study <tag> trial <k>/<n> best=<v> in <s> s``. ruthless documents the observer as a neutral
    sink that never changes the search. ``k`` counts THIS call's trials (a resumed study continues its global
    trial numbers); ``best`` is the lowest ``metric`` of this call (the xshot/xcross studies minimise it)."""

    def __init__(self, tag: str, n_trials: int, *, metric: str) -> None:
        self.tag, self.n_trials, self.metric = tag, n_trials, metric
        self.k, self.best, self.t0 = 0, math.inf, time.monotonic()

    def __call__(self, event) -> None:
        self._record(float(event.metrics[self.metric]))

    def optuna_callback(self, study, trial) -> None:
        """The same line from an ``optuna`` ``study.optimize(callbacks=[...])`` hook (the xsuccess HPO, launch-4
        finding F21). optuna calls it after each finished trial; it reads the trial and never changes the
        sampler. A failed trial has no value: it is counted and leaves ``best`` unchanged."""
        self._record(None if trial.value is None else float(trial.value))

    def _record(self, value: float | None) -> None:
        self.k += 1
        if value is not None:
            self.best = min(self.best, value)
        s = time.monotonic() - self.t0
        line = f"[progress] study {self.tag} trial {self.k}/{self.n_trials} best={self.best:.6g} in {s:.1f} s"
        print(line, file=sys.stderr, flush=True)


def print_fold(trainer: str, k: int, n: int, t0: float) -> None:
    """``[progress] <trainer> fold <k>/<n> in <s> s`` after a CV fold (HFU spec 15.3); ``t0`` is the loop start."""
    print(f"[progress] {trainer} fold {k}/{n} in {time.monotonic() - t0:.1f} s", file=sys.stderr, flush=True)


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
) -> Path:
    """Write ``(X, y, groups, providers, match_ids, is_public)`` + ``config`` under ``shard_root``.

    ``arms`` (ADR-123 Amendment B) are the candidate-arm row masks; each is written to
    ``arms/<name>.npy`` so fan-out study workers rebuild byte-identical masks from the parameter.
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
    return StudyInputs(
        X=X,
        y=y,
        groups=groups,
        providers=np.load(root / "providers.npy", allow_pickle=False),
        match_ids=np.load(root / "match_ids.npy", allow_pickle=False),
        is_public=np.load(root / "is_public.npy", allow_pickle=False),
        config=json.loads((root / _CONFIG).read_text(encoding="utf-8")),
        arms=arms,
    )


#: The extract side columns that carry a match's stoppage counts (ADR-130), one per `STOPPAGE_SUMMARY_KEYS` entry.
STOPPAGE_SIDE_PREFIX = "_st_"


def stoppage_side_columns(frames: pd.DataFrame, *, horizon_seconds: float, label_times: pd.DataFrame) -> dict:
    """The per-match stoppage counts as extract side columns (``_st_<key>``), broadcast on every row of the match."""
    from silly_kicks.tracking._stoppages import stoppage_summary

    counts = stoppage_summary(frames, horizon_seconds=horizon_seconds, label_times=label_times)
    return {f"{STOPPAGE_SIDE_PREFIX}{k}": int(v) for k, v in counts.items()}


def stoppage_block(combined: pd.DataFrame) -> dict:
    """The trained model's stoppage record (``metrics.json`` and ``metadata.json``; ADR-130): the rule version and,
    per provider, the summed per-match counts of the extract side columns (one row per match, so a count is not
    multiplied by the match's row count)."""
    from silly_kicks.tracking._stoppages import STOPPAGE_RULE_VERSION, STOPPAGE_SUMMARY_KEYS

    cols = [f"{STOPPAGE_SIDE_PREFIX}{k}" for k in STOPPAGE_SUMMARY_KEYS]
    per_match = combined.drop_duplicates(["_provider", "_match_id"])
    providers = (
        {
            str(p): {k: int(g[f"{STOPPAGE_SIDE_PREFIX}{k}"].sum()) for k in STOPPAGE_SUMMARY_KEYS}
            for p, g in per_match.groupby("_provider", sort=True)
        }
        if set(cols) <= set(combined.columns)
        else {}
    )
    return {"rule_version": STOPPAGE_RULE_VERSION, "providers": providers}


# --- The cv stage's nested gate (ADR-131): fold-unit payloads and the pooled skill --------------------------------


def gate_fold_params(ctx, arm: str, k: int) -> dict:
    """The params of gate fold ``k`` of ``arm``: its nested study ``{arm}_gf{k}`` (ADR-131)."""
    return ctx.upstream("study")[f"{arm}{GATE_FOLD_SEP}{k}"]["params"]


#: The per-model size gate ``artifact_size_lt_5mb`` on the xShot/xCross bundles (ADR-132; corpus widening spec 5.6;
#: the value TF-16 pre-registered for these boosters). A bundle of this many bytes or more fails closed.
ARTIFACT_SIZE_LIMIT_BYTES = 5_000_000


def artifact_size_gate(files) -> tuple[int, bool]:
    """``(total bytes, artifact_size_lt_5mb)`` of a bundle FileSet's files (the ``artifact_size_bytes`` total)."""
    n = sum(len(b) for b in files.values())
    return n, n < ARTIFACT_SIZE_LIMIT_BYTES


def _npy(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _from_npy(data: bytes) -> np.ndarray:
    import io

    return np.load(io.BytesIO(data), allow_pickle=False)


def cv_unit(out, positive_rate: float, tr, te, y, groups, providers) -> dict:
    """One cv fold unit's payload. ``out`` is the fold function's ``return_oof=True`` result ``(metrics_tuple, p,
    train_rate)``, or None for a skipped fold (no rows). ``tr`` / ``te`` are the training / held-out positions within
    the arm's rows; ``y``, ``groups`` and ``providers`` are the arm's labels, match labels and row-aligned providers.

    ADR-132 (corpus widening spec 5.7): the unit also carries the held-out rows' provider IDENTITIES (``sportec`` ->
    ``idsse``) and each provider's training-fold true rate from ``tr`` (the rows behind ``train_rate``; NaN for a
    held-out provider with no training rows). An empty provider refuses the unit. ``providers=None`` (an in-process
    composition without providers) records none, so ``cv_skill`` emits no ``by_provider`` and the per-provider gates
    fail closed."""
    if out is None:
        return {
            "fold": None,
            "positive_rate": positive_rate,
            "train_rate": None,
            "train_rate_by_provider": {},
            "te": np.zeros(0, dtype=np.int64),
            "p": np.zeros(0, dtype=np.float64),
            "y": np.zeros(0, dtype=np.int8),
            "groups": np.zeros(0, dtype="<U1"),
            "providers": np.zeros(0, dtype="<U1"),
        }
    res, p, train_rate = out
    te = np.asarray(te, dtype=np.int64)
    unit = {
        "fold": list(res),
        "positive_rate": positive_rate,
        "train_rate": train_rate,
        "te": te,
        "p": np.asarray(p, dtype=np.float64),
        "y": np.asarray(y)[te].astype(np.int8),
        "groups": np.asarray(groups)[te].astype(str),
    }
    if providers is not None:
        from scripts._corpus_identity import provider_identity

        ident = np.array([provider_identity(v) for v in np.asarray(providers).tolist()], dtype=str)
        tr = np.asarray(tr, dtype=np.int64)
        y_tr, id_tr = np.asarray(y)[tr], ident[tr]
        rates = {}
        for prov in sorted(set(id_tr.tolist()) | set(ident[te].tolist())):
            sel = id_tr == prov
            rates[prov] = float(y_tr[sel].mean()) if sel.any() else float("nan")
        unit["train_rate_by_provider"] = rates
        unit["providers"] = ident[te]
    return unit


def cv_unit_fileset(unit: dict):
    """The unit as a FileSet: ``fold.json`` (the metrics tuple, the arm's rate, the training rate) and the held-out
    arrays ``te.npy``, ``p.npy``, ``y.npy``, ``groups.npy`` (pickle-free)."""
    from scripts._stages import FileSet

    head = {k: unit[k] for k in ("fold", "positive_rate", "train_rate", "train_rate_by_provider") if k in unit}
    files = {"fold.json": json.dumps(head, allow_nan=True).encode("utf-8")}
    files.update({f"{k}.npy": _npy(unit[k]) for k in ("te", "p", "y", "groups", "providers") if k in unit})
    return FileSet(files)


def read_cv_unit(fs) -> dict:
    """The inverse of `cv_unit_fileset`."""
    unit = json.loads(fs.files["fold.json"])
    arrays = ("te", "p", "y", "groups", "providers")
    unit.update({k: _from_npy(fs.files[f"{k}.npy"]) for k in arrays if f"{k}.npy" in fs.files})
    return unit


def cv_skill(units: list[dict], *, nested: bool) -> dict | None:
    """The ADR-131 ``skill`` dict of one arm from its cv units in FOLD order (the list position is the gate fold).
    The usable folds' held-out rows are pooled in ascending arm-row order, so the result does not depend on the order
    the units completed in. None when no fold was usable (the skill gates then fail closed). ADR-132: plus
    ``skill["by_provider"]`` (``provider_skill`` against each row's own provider training-fold rate) when every
    usable unit carries its providers."""
    from silly_kicks.tracking._artifact_integrity import forecast_skill, provider_skill

    usable = [(k, u) for k, u in enumerate(units) if u["fold"] is not None]
    if not usable:
        return None
    te = np.concatenate([u["te"] for _, u in usable])
    order = np.argsort(te, kind="stable")
    y = np.concatenate([u["y"] for _, u in usable])[order]
    p = np.concatenate([u["p"] for _, u in usable])[order]
    r = np.concatenate([np.full(len(u["te"]), u["train_rate"], dtype=np.float64) for _, u in usable])[order]
    fold = np.concatenate([np.full(len(u["te"]), k) for k, u in usable])[order]
    groups = np.concatenate([u["groups"] for _, u in usable])[order]
    skill = forecast_skill(y, p, r, fold, groups, nested=nested)
    if all("providers" in u for _, u in usable):
        prov = np.concatenate([u["providers"] for _, u in usable])[order]
        r_prov = np.concatenate(
            [np.array([u["train_rate_by_provider"][v] for v in u["providers"].tolist()]) for _, u in usable]
        ).astype(np.float64)[order]
        skill["by_provider"] = provider_skill(y, p, r_prov, prov, groups, fold)
    return skill
