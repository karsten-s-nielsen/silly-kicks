"""build_contact_scorer: the TF-79 contact scorer's development and held-out runs (scorer spec sections 3 and 4).

A born-ready ADR-128 corpus driver (``DRIVERS["build_contact_scorer"]``) over the staged ``gs_contact_truth``
dataset: per GS match, the frames as the loader serves them and PFF's on-ball sequences. It is visibility-blind
(ADR-123 Amendment B): it maps a unit to its PFF game id with ``canonical_match_id`` and imports no corpus
registry.

* ``--split dev`` reads only the development units. Per match, the scored option rows of the threshold-free merged
  peaks with their control features (``_contact_scorer.option_table``) and labels (2.4), and the sequences with the
  bar's support flag (``track_supported``; scorer spec r7 section 3); 6 grouped folds (the 24 ids sorted by the
  sha256 hex of the id, fold = position mod 6) give out-of-fold credits (with the first-touch decoding) on the
  19-point threshold grid, scored on all pairs and on the supported pairs; the selection on the supported pairs
  (feasible, near-best, then the larger threshold); the refit on all 24 matches is the frozen scorer
  (``model.json``, ``metadata.json``, ``metrics.json``, ``SHA256SUMS``), written beside the report.
* ``--split holdout --dev-report <file>`` reads only the held-out units and scores the frozen scorer the report
  names, once: a clean tree, the development run's commit, and the scorer's ``model.json`` sha256 as recorded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts._corpus_identity import canonical_match_id
from scripts._gs_truth import (
    BAR_FALSE_FIRST,
    BAR_RIGHT,
    BAR_TIMING_S,
    TRUTH_TAU_S,
    meets_bar,
    score_next_touch,
    select_point,
    shares,
    split_of,
    track_supported,
)
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main
from silly_kicks.id_compat import canonical_id

#: The threshold grid (scorer spec section 3): positives are rare, so true touches can score low.
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
N_FOLDS = 6
_COUNTS = ("n", "right", "false_first", "wrong_player", "missed", "n_block", "block_right", "block_missed")
_SUP_COUNTS = ("sup_n", "sup_right", "sup_false_first", "sup_wrong_player", "sup_missed")
_COUNTS_ALL = (*_COUNTS, "n_contacts", "n_unresolved", *_SUP_COUNTS)
#: The bar's population (scorer spec r7 section 3): "right" and "false first" on the track-supported pairs.
BAR_POPULATION = "supported"

_PARAMS = (
    Param("dataset", str, required=True, help="the staged gs_contact_truth dataset (scripts/hf_jobs/datasets.toml)"),
    Param("split", str, required=True, choices=("dev", "holdout"), help="the half of the GS matches to read"),
    Param("dev_report", str, default=None, help="--split holdout: the development report (its frozen scorer)"),
)


def _game(ref) -> str:
    return canonical_match_id("gradientsports", ref.stem)


def _detector_params():
    from silly_kicks.tracking import BallContactParams

    return BallContactParams(min_velocity_change_ms=0.0)  # threshold-free peaks (scorer spec 2.1)


def fold_of(games) -> dict[str, int]:
    """The grouped folds: the ids sorted by the sha256 hex of ``str(int(id))``, fold = position mod ``N_FOLDS``."""
    order = sorted((str(int(g)) for g in games), key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest())
    return {g: i % N_FOLDS for i, g in enumerate(order)}


def label_options(opts: pd.DataFrame, seq: pd.DataFrame, *, tau: float = TRUTH_TAU_S) -> np.ndarray:
    """Scorer spec 2.4: near an edge (within ``tau`` of a sequence's start or end; the nearer edge) the edge's player
    is 1 and the other options 0; in flight (between sequences, farther than ``tau`` from every edge) every option is
    0; inside a sequence away from its edges, NaN (not used)."""
    out = np.full(len(opts), np.nan)
    if len(opts) == 0:
        return out
    t = opts["time_seconds"].to_numpy(dtype=float)
    players = np.array([str(canonical_id(v)) for v in opts["player_id"]], dtype=object)
    period = opts["period_id"].to_numpy()
    for p, s in seq.groupby("period_id"):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        edge_t = np.r_[s["t0"].to_numpy(dtype=float), s["t1"].to_numpy(dtype=float)]
        edge_p = np.r_[s["player"].astype(str).to_numpy(), s["player"].astype(str).to_numpy()]
        t0s, t1s = s["t0"].to_numpy(dtype=float), s["t1"].to_numpy(dtype=float)
        for i in rows:
            d = np.abs(edge_t - t[i])
            j = int(np.argmin(d))
            if d[j] <= tau:
                out[i] = float(players[i] == edge_p[j])
            elif (t1s < t[i]).any() and (t0s > t[i]).any() and not ((t0s <= t[i]) & (t[i] <= t1s)).any():
                out[i] = 0.0  # in flight: between two sequences
    return out


# ---------------------------------------------------------------------------------------------------- the units
def _refs(ctx):
    from scripts._driver_sources import staged_pergame_refs

    split = ctx.params["split"]
    return [
        dataclasses.replace(r, files=(f"{r.stem}/frames.parquet", f"{r.stem}/sequences.parquet"))  # the declared reads
        for r in staged_pergame_refs(ctx.data_root, ctx.params["dataset"])
        if split_of(_game(r)) == split
    ]


def _read(ref, ctx, name: str) -> pd.DataFrame:
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    return read_dataset_parquet(dataset_dir(ctx.data_root, ctx.params["dataset"]) / ref.stem / name)


def _options_work(item, ctx) -> pd.DataFrame:
    from silly_kicks.tracking._contact_scorer import option_table

    game, frames, seq = item
    opts = option_table(frames, _detector_params())
    opts = opts.assign(game=game, label=label_options(opts, seq))
    opts["player_id"] = [str(canonical_id(v)) for v in opts["player_id"]]
    return opts


def _sequences_work(item, ctx) -> pd.DataFrame:
    """The match's PFF sequences with the bar's support flag (scorer spec r7 section 3), read from its frames."""
    game, frames, seq = item
    return seq.assign(supported=track_supported(seq, frames), game=game, player=seq["player"].astype(str))


def _concat(results, ctx) -> pd.DataFrame:
    frames = [r for r in results.values() if len(r)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------------------------------- scoring
def _score(opts: pd.DataFrame, seqs: pd.DataFrame, scorer, threshold: float) -> tuple[dict, list[float]]:
    """Summed section 6 counts and pooled release offsets of ``scorer`` at ``threshold`` over the games in ``opts``."""
    from silly_kicks.tracking import BallContactParams
    from silly_kicks.tracking._contact_scorer import credit_options

    params = BallContactParams(min_velocity_change_ms=0.0, min_contact_probability=threshold)
    counts = dict.fromkeys(_COUNTS_ALL, 0)
    offsets: list[float] = []
    for game in sorted(set(seqs["game"])):
        o = opts[opts["game"] == game]
        contacts = (
            credit_options(o, scorer.predict_proba(o), params, decode_window_s=float(scorer.decode_window_s))
            if len(o)
            else _no_contacts()
        )
        s = score_next_touch(seqs[seqs["game"] == game], contacts)
        offsets += s.pop("release_offsets")
        for k in _COUNTS_ALL:
            counts[k] += int(s[k])
    return counts, offsets


def _no_contacts() -> pd.DataFrame:
    return pd.DataFrame({"period_id": [], "time_seconds": [], "player_id": [], "contact_source": []})


def _fit(opts: pd.DataFrame):
    from silly_kicks.tracking._contact_scorer import ContactScorer

    train = opts[opts["label"].notna()]
    return ContactScorer().fit(train, train["label"].astype(int).to_numpy())


def _fold_units(ctx) -> list[str]:
    return [str(f) for f in range(N_FOLDS)] if ctx.params["split"] == "dev" else []


def _fold_work(key, ctx) -> pd.DataFrame:
    opts, seqs = ctx.upstream("options"), ctx.upstream("sequences")
    folds = fold_of(sorted(set(seqs["game"])))
    held = sorted(g for g, f in folds.items() if f == int(key))
    rows = []
    if held:
        scorer = _fit(opts[~opts["game"].isin(held)])
        o, s = opts[opts["game"].isin(held)], seqs[seqs["game"].isin(held)]
        for th in GRID:
            counts, offsets = _score(o, s, scorer, th)
            rows.append({"fold": int(key), "threshold": th, **counts, "release_offsets": json.dumps(offsets)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- the report
def _revision(dataset: str) -> str | None:
    """The locked Hub revision of the dataset the run read (spec section 4); None for an unrecorded local tree."""
    from scripts._driver_sources import _lock_units

    rec = _lock_units(dataset)
    return None if rec is None else rec.get("revision")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(grid: dict) -> tuple[float | None, bool]:
    """The section 3 selection: the feasible thresholds, the near-best window, then the larger threshold (the stricter
    detector); and whether the choice lies on the grid's edge."""
    picked = select_point(grid, prefer=lambda th: -th, population=BAR_POPULATION)
    chosen = None if picked is None else float(cast("float", picked))
    return chosen, chosen is not None and chosen in (GRID[0], GRID[-1])


def _scorer_files(scorer, report: dict) -> dict[str, bytes]:
    """The frozen scorer's files, flat (ADR-088): ``model.json``, ``metadata.json`` and ``metrics.json`` written from
    this run, and ``SHA256SUMS`` over the three (scorer spec 2.3; commit 5 bundles them byte for byte)."""
    from silly_kicks.tracking import BallContactParams
    from silly_kicks.tracking._contact_scorer import checksums

    with tempfile.TemporaryDirectory() as d:
        scorer.save(d, params=BallContactParams(min_velocity_change_ms=0.0))
        model = (Path(d) / "model.json").read_bytes()
    chosen = report["chosen"]["min_contact_probability"] if report["chosen"] else None
    metadata = {
        "training_commit": report["run_commit"],
        "frames_dataset": report["dataset"],
        "corpus_provenance": {
            "providers": ["gradientsports"],
            "n_games": len(report["matches"]),
            "game_ids": report["matches"],
        },
        "stores_training_data": False,
        "min_contact_probability": chosen,
        "sklearn_version": getattr(scorer, "sklearn_version", None),
    }
    metrics = {
        "run_commit": report["run_commit"],
        "run_tree_dirty": report["run_tree_dirty"],
        "oof_at_chosen": next((g for g in report["grid"] if g["threshold"] == chosen), None),
        "refit_dev_at_chosen": report.get("refit_dev_at_chosen"),
    }
    files = {
        "model.json": model,
        "metadata.json": json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True, default=str).encode("utf-8"),
    }
    return {**files, "SHA256SUMS": checksums(files)}


def _dev_report(ctx) -> FileSet:
    opts, seqs = ctx.upstream("options"), ctx.upstream("sequences")
    oof = ctx.upstream("folds")
    grid = {}
    for th, g in oof.groupby("threshold", sort=True):
        offsets = [o for js in g["release_offsets"] for o in json.loads(js)]
        grid[float(th)] = shares(g[list(_COUNTS_ALL)].sum().to_dict(), offsets)
    chosen, on_edge = _select(grid)
    refit = _fit(opts)
    report: dict = {
        "run_commit": ctx.prov.get("commit"),
        "run_tree_dirty": ctx.prov.get("tree_state") != "clean",
        "split": "dev",
        "tau_s": TRUTH_TAU_S,
        "bar": {"right": BAR_RIGHT, "false_first": BAR_FALSE_FIRST, "timing_s": BAR_TIMING_S},
        "bar_population": BAR_POPULATION,
        "dataset": ctx.params["dataset"],
        "dataset_revision": _revision(ctx.params["dataset"]),
        "matches": sorted(set(seqs["game"])),
        "folds": fold_of(sorted(set(seqs["game"]))),
        "unresolved_rate_scope": "merged peaks with at least one player option",
        "grid": [{"threshold": th, **m} for th, m in grid.items()],
        "chosen": None if chosen is None else {"min_contact_probability": float(chosen)},
        "chosen_on_grid_edge": on_edge,
    }
    if chosen is not None:
        counts, offsets = _score(opts, seqs, refit, chosen)
        report["refit_dev_at_chosen"] = shares(counts, offsets)
    files = _scorer_files(refit, report)
    report["scorer_model_sha256"] = hashlib.sha256(files["model.json"]).hexdigest()
    return FileSet({"contact_scorer_dev.json": json.dumps(report, indent=2, default=str).encode("utf-8"), **files})


def _holdout_report(ctx) -> FileSet:
    from silly_kicks.tracking._contact_scorer import ContactScorer

    path = ctx.params.get("dev_report")
    if not path:
        raise ValueError("the held-out run scores the development report's frozen scorer; pass --dev-report")
    dev = json.loads(Path(path).read_text(encoding="utf-8"))
    if dev.get("split") != "dev" or not dev.get("chosen"):
        raise ValueError(f"{path} is not a development report with a feasible point (detector spec 6)")
    if ctx.prov.get("tree_state") != "clean" or dev.get("run_tree_dirty") is not False:
        raise ValueError("both looks read a clean committed tree (ADR-037); --allow-dirty is refused here")
    if dev.get("run_commit") != ctx.prov.get("commit"):
        raise ValueError(f"the development report is from commit {dev.get('run_commit')}, not {ctx.prov.get('commit')}")
    scorer_dir = Path(path).parent  # the development output: the report beside the frozen scorer
    if _sha(scorer_dir / "model.json") != dev.get("scorer_model_sha256"):
        raise ValueError(f"{scorer_dir / 'model.json'} differs from the scorer the development report recorded")
    scorer = ContactScorer.load(scorer_dir)
    opts, seqs = ctx.upstream("options"), ctx.upstream("sequences")
    th = float(dev["chosen"]["min_contact_probability"])
    counts, offsets = _score(opts, seqs, scorer, th)
    m = shares(counts, offsets)
    report = {
        "run_commit": ctx.prov.get("commit"),
        "run_tree_dirty": ctx.prov.get("tree_state") != "clean",
        "split": "holdout",
        "bar_population": BAR_POPULATION,
        "dataset": ctx.params["dataset"],
        "dataset_revision": _revision(ctx.params["dataset"]),
        "matches": sorted(set(seqs["game"])),
        "unresolved_rate_scope": "merged peaks with at least one player option",
        "min_contact_probability": th,
        "scorer_model_sha256": dev["scorer_model_sha256"],
        "dev_report": str(path),
        **m,
        "meets_bar": bool(meets_bar(m, population=BAR_POPULATION)),
    }
    return FileSet({"contact_scorer_holdout.json": json.dumps(report, indent=2, default=str).encode("utf-8")})


def _report_work(key, ctx) -> FileSet:
    return _dev_report(ctx) if ctx.params["split"] == "dev" else _holdout_report(ctx)


def _declare_scorer(params) -> dict[str, object]:
    """The held-out run's token reads the frozen scorer's content, so a refit at the same path is a new generation."""
    path = params.get("dev_report")
    if params.get("split") != "holdout" or not path:
        return {}
    model = Path(path).parent / "model.json"
    return {"scorer_model_sha256": _sha(model) if model.is_file() else "missing"}


PIPELINE = Pipeline(
    name="build_contact_scorer",
    module="scripts.build_contact_scorer",
    params=_PARAMS,
    stages=(
        ItemStage(
            name="options",
            refs=_refs,
            key=lambda ref: ref.key,
            load=lambda ref, ctx: (_game(ref), _read(ref, ctx, "frames.parquet"), _read(ref, ctx, "sequences.parquet")),
            work=_options_work,
            reduce=_concat,
            params=("split",),
            uses_dataset=True,
        ),
        ItemStage(
            name="sequences",
            refs=_refs,
            key=lambda ref: ref.key,
            load=lambda ref, ctx: (_game(ref), _read(ref, ctx, "frames.parquet"), _read(ref, ctx, "sequences.parquet")),
            work=_sequences_work,
            reduce=_concat,
            params=("split",),
            uses_dataset=True,
        ),
        Stage(
            name="folds",
            units=_fold_units,
            work=_fold_work,
            needs=("options", "sequences"),
            reduce=_concat,
            params=("split",),
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("options", "sequences", "folds"),
            reduce=lambda r, ctx: r["only"],
            params=("split", "dev_report"),
            declare=_declare_scorer,
        ),
    ),
    output="report",
)


if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
