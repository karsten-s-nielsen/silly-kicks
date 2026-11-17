"""build_cross_definition: the open-play cross-definition measurement (ADR-133).

A born-ready ADR-128 driver (scaffolded by `python -m scripts.hf_jobs scaffold`). It reads only staged datasets:

* ``--sb-dataset`` (the staged ``sb_open`` open-data revision): StatsBomb events converted with the current
  converter. Its ``cross_native`` (StatsBomb ``pass.cross``) is a vendor benchmark for the start line.
* ``--dataset`` (a staged ``flat_tc3`` dataset, e.g. ``xshot187``): the staged ``_actions`` RE-TYPED by the
  delivery rule at each candidate line. Stage it from the release commit: the actions then carry native event
  timing and their own ``cross_native``, and the synthetic dribbles (a ``dribble`` row without an
  ``original_event_id``) are dropped before re-typing (the rule runs before ``_add_dribbles``). The per-match stages
  REFUSE a stage without ``cross_native`` (a stage from the converters before ADR-133: SkillCorner rows at the
  reception, not the kick).

The report applies the PRE-DECLARED rules: the start line (the highest equal-weight mean F1 against the native
label over ``F1_VENDORS``; a tie within ``F1_TIE`` goes to the larger line) and the ratio bar (the per-provider
median crosses per match agree within ``RATIO_BAR``) at the reporting line (``--cross-start-x``, else the selected
line).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main

CANDIDATE_LINES: tuple[float, ...] = (52.5, 60.0, 65.0, 70.0, 75.0, 80.0)
F1_VENDORS: tuple[str, ...] = ("statsbomb", "gradientsports")  # the vendors with a full native cross label
F1_TIE: float = 0.005
RATIO_BAR: float = 1.5

_PARAMS = (
    Param("dataset", str, required=True, help="the staged flat dataset (scripts/hf_jobs/datasets.toml)"),
    Param("sb_dataset", str, None, help="the staged StatsBomb open-data dataset (sb_open); omit to skip it"),
    Param(
        "competitions_json",
        str,
        None,
        help="JSON [[competition_id, season_id], ...] (default: every competition in the open-data snapshot)",
    ),
    Param("max_matches", int, None),
    Param("cache_dir", str, None, help="raw open-data events cache root"),
    Param("cross_start_x", float, None, help="the line the ratio bar is reported at (default: the selected line)"),
)


# ---------------------------------------------------------------------------------------------- pure rules
def f1(rule: np.ndarray, native: np.ndarray) -> float:
    """F1 of the rule's crosses against the vendor's crosses (both boolean arrays over the same rows)."""
    tp = float(np.sum(rule & native))
    fp = float(np.sum(rule & ~native))
    fn = float(np.sum(~rule & native))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def select_start_line(f1_by_vendor: dict[str, dict[float, float]], *, vendors=F1_VENDORS, tie=F1_TIE) -> float:
    """The candidate with the highest mean F1 over ``vendors`` (equal weight); a tie within ``tie`` goes to the
    larger line."""
    lines = sorted(next(iter(f1_by_vendor.values())))
    mean = {x: float(np.mean([f1_by_vendor[v][x] for v in vendors])) for x in lines}
    best = max(mean.values())
    return max(x for x in lines if best - mean[x] <= tie)


def ratio_bar(medians: dict[str, float], *, bar: float = RATIO_BAR) -> dict:
    """Max over min of the per-provider median crosses per match; passes when <= ``bar``."""
    lo = min(medians, key=lambda k: medians[k])
    hi = max(medians, key=lambda k: medians[k])
    ratio = float(medians[hi] / medians[lo]) if medians[lo] > 0 else float("inf")
    return {"ratio": ratio, "bar": bar, "passes": bool(ratio <= bar), "lowest": lo, "highest": hi}


# ---------------------------------------------------------------------------------------------- shared row model
def _delivery_rows(actions: pd.DataFrame, *, provider: str, match: str, native_label: bool) -> pd.DataFrame:
    """The open-play pass-class rows of one match, re-typed at the most permissive line (``CANDIDATE_LINES[0]``).
    The rule needs ``start_x >= line``, so a higher line is read off these rows as ``type_geo == "cross"`` and
    ``start_x >= line``."""
    from silly_kicks.spadl import config as spc
    from silly_kicks.spadl.base import classify_deliveries

    if "cross_native" not in actions.columns:
        raise ValueError(
            f"{provider} match {match}: the actions have no cross_native column, so they come from the converters "
            "before ADR-133 (SkillCorner rows at the reception, not the kick). Restage from the release commit."
        )
    a = actions.reset_index(drop=True)
    if "original_event_id" in a.columns:
        synthetic = (a["type_id"] == spc.actiontype_id["dribble"]) & a["original_event_id"].isna()
        a = a[~synthetic].reset_index(drop=True)
    if native_label:
        native = a["cross_native"]
    else:
        native = pd.Series(pd.array([pd.NA] * len(a), dtype="boolean"))
    geo = classify_deliveries(a, cross_start_x=CANDIDATE_LINES[0])
    keep = geo["type_id"].isin([spc.actiontype_id["pass"], spc.actiontype_id["cross"]]).to_numpy()
    names = dict(enumerate(spc.actiontypes))
    return pd.DataFrame(
        {
            "provider": provider,
            "match": match,
            "type_geo": [names[t] for t in geo.loc[keep, "type_id"]],
            "cross_native": pd.array(np.asarray(native, dtype=object)[keep], dtype="boolean"),
            "start_x": a["start_x"].to_numpy(dtype=float)[keep],
        }
    )


# ---------------------------------------------------------------------------------------------- StatsBomb open data
def _refs_work(key, ctx) -> dict:
    import dataclasses

    if not ctx.params.get("sb_dataset"):
        return {"refs": []}
    from scripts._sb_open_data import all_open_competitions, assert_statsbomb_open_data_mode, list_open_data_refs

    assert_statsbomb_open_data_mode()  # fail-closed: public open data only
    p = ctx.params
    # Owner ruling 2026-11-13: the StatsBomb benchmark is every competition in the pinned open-data snapshot.
    comps = (
        [tuple(c) for c in json.loads(p["competitions_json"])] if p["competitions_json"] else all_open_competitions()
    )
    refs = list_open_data_refs(list(comps), max_matches=p["max_matches"])
    return {"refs": [dataclasses.asdict(r) for r in sorted(refs, key=lambda r: str(r.match_id))]}


def _sb_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _sb_load(ref, ctx):
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()
    return str(ref.match_id), load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _sb_work(item, ctx) -> pd.DataFrame:
    match, loaded = item
    return _delivery_rows(loaded[2], provider="statsbomb", match=match, native_label=True)


# ---------------------------------------------------------------------------------------------- staged flat dataset
def _flat_refs(ctx):
    import dataclasses

    from scripts._driver_sources import staged_flat_refs

    # The declared read is the actions table only (files = (frames, actions)): the rule reads no frame.
    return [dataclasses.replace(r, files=(r.files[1],)) for r in staged_flat_refs(ctx.data_root, ctx.params["dataset"])]


def _flat_load(ref, ctx):
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    return (
        ref.provider,
        ref.stem,
        read_dataset_parquet(dataset_dir(ctx.data_root, ctx.params["dataset"]) / ref.files[0]),
    )


def _flat_work(item, ctx) -> pd.DataFrame:
    provider, stem, actions = item
    return _delivery_rows(actions, provider=provider, match=stem, native_label=provider != "skillcorner")


def _concat(results, ctx) -> pd.DataFrame:
    parts = [results[k] for k in sorted(results)]  # sorted-key reduce (ADR-128)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------------------------- report
def _crosses_at(rows: pd.DataFrame, line: float) -> pd.Series:
    return (rows["type_geo"] == "cross") & (rows["start_x"] >= line)


def _f1_table(rows: pd.DataFrame) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    for provider, g in rows[rows["cross_native"].notna()].groupby("provider"):
        native = g["cross_native"].astype(bool).to_numpy()
        out[str(provider)] = {x: f1(_crosses_at(g, x).to_numpy(), native) for x in CANDIDATE_LINES}
    return out


def _report_work(key, ctx) -> FileSet:
    sb = ctx.upstream("sb")
    flat = ctx.upstream("flat")
    flat_summary = ctx.summary("flat")
    rows = pd.concat([sb, flat], ignore_index=True) if len(sb) else flat
    f1_table = _f1_table(rows)
    selected = select_start_line(f1_table) if all(v in f1_table for v in F1_VENDORS) else None
    line = ctx.params["cross_start_x"] if ctx.params["cross_start_x"] is not None else selected
    medians = None
    if line is not None and len(flat):
        per_match = flat.assign(c=_crosses_at(flat, line)).groupby(["provider", "match"])["c"].sum()
        medians = {str(p): float(v) for p, v in per_match.groupby("provider").median().items()}
    payload = {
        "candidate_lines": list(CANDIDATE_LINES),
        "f1_by_vendor": {v: {str(x): f for x, f in t.items()} for v, t in f1_table.items()},
        "selected_start_line": selected,
        "reporting_line": line,
        "median_crosses_per_match": medians,
        "ratio_bar": ratio_bar(medians) if medians else None,
        "n_matches": {str(p): int(n) for p, n in rows.groupby("provider")["match"].nunique().items()},
        # ADR-037 provenance and the ADR-128 corpus-scoped counters of the per-match flat stage.
        "generation": ctx.tokens["flat"],
        "n_attempted": flat_summary["n_units"],
        "n_failed": flat_summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": flat_summary["n_excluded"],
        "run_commit": ctx.prov["commit"],
        "run_tree_dirty": ctx.prov["dirty"],
        "run_tree_state": ctx.prov.get("tree_state"),
    }
    return FileSet(
        {"cross_definition.json": json.dumps(payload, indent=2, sort_keys=True, default=float).encode("utf-8")}
    )


PIPELINE = Pipeline(
    name="cross_definition",
    module="scripts.build_cross_definition",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("sb_dataset", "competitions_json", "max_matches"),
            uses_dataset=True,
            dataset_param="sb_dataset",
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="sb",
            refs=_sb_refs,
            key=lambda ref: ref.key,
            load=_sb_load,
            work=_sb_work,
            reduce=_concat,
            needs=("refs",),
            cross_commit_version="cross-def-sb-1",
            uses_dataset=True,
            dataset_param="sb_dataset",
        ),
        ItemStage(
            name="flat",
            refs=_flat_refs,
            key=lambda ref: ref.key,
            load=_flat_load,
            work=_flat_work,
            reduce=_concat,
            cross_commit_version="cross-def-flat-1",
            uses_dataset=True,
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("sb", "flat"),
            reduce=lambda r, ctx: r["only"],
            params=("cross_start_x",),
        ),
    ),
    output="report",
)


if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
