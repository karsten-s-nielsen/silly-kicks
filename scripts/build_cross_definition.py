"""build_cross_definition: the cross-definition measurement (ADR-133; cross-definition spec sections 7.2-7.3).

A born-ready ADR-128 driver (scaffolded by `python -m scripts.hf_jobs scaffold`). It reads only staged datasets:

* ``--sb-dataset`` (the staged ``sb_open`` open-data revision): StatsBomb events converted with the current
  converter. Its ``cross_native`` (StatsBomb ``pass.cross``) is a vendor benchmark for the start line.
* ``--dataset`` (a staged ``flat_tc3`` dataset, e.g. ``xshot187``): the staged ``_actions`` RE-TYPED by the
  delivery rule, plus the frames. The staged actions came from the converters before ADR-133. Re-typing them
  reproduces the new converters exactly: the rule runs before ``_add_dribbles``, which never reads the type, so
  dropping the synthetic dribbles (a ``dribble`` row without an ``original_event_id``) and re-typing gives the
  new output. The old type is the provider's native label (SkillCorner has none: ``<NA>``).

The report applies the PRE-DECLARED rules: the start line (spec 7.2: the highest mean F1 over StatsBomb and Gradient
Sports, equal weight, tie within 0.005 -> the larger line), the ``CrossDeliveryParams`` calibration (plan Task 10),
and the spec 7.3 corpus evidence at the reporting line (``--cross-start-x``, else the selected line). Run it once with
the provisional parameters to calibrate, then once more with the final values as overrides.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main

CANDIDATE_LINES: tuple[float, ...] = (52.5, 60.0, 65.0, 70.0)
F1_VENDORS: tuple[str, ...] = ("statsbomb", "gradientsports")  # full native cross labels (spec decision 8)
F1_TIE: float = 0.005
RATIO_BAR: float = 1.5

_TRACKING_PARAMS = ("block_radius_m", "direction_window_s", "release_gap_s", "min_ball_coverage", "release_match_m")
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
    Param("cross_start_x", float, None, help="the line the evidence is reported at (default: the selected line)"),
    *(Param(name, float, None, help=f"CrossDeliveryParams.{name} override") for name in _TRACKING_PARAMS),
)

_DELIVERY_NAMES = ("pass", "cross", "corner_crossed", "corner_short", "freekick_crossed", "freekick_short")


# ---------------------------------------------------------------------------------------------- pure rules
def f1(rule: np.ndarray, native: np.ndarray) -> float:
    """F1 of the rule's crosses against the vendor's crosses (both boolean arrays over the same rows)."""
    tp = float(np.sum(rule & native))
    fp = float(np.sum(rule & ~native))
    fn = float(np.sum(~rule & native))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def select_start_line(f1_by_vendor: dict[str, dict[float, float]], *, vendors=F1_VENDORS, tie=F1_TIE) -> float:
    """Spec 7.2: the candidate with the highest mean F1 over ``vendors`` (equal weight).

    A tie within ``tie`` goes to the larger line.
    """
    lines = sorted(next(iter(f1_by_vendor.values())))
    mean = {x: float(np.mean([f1_by_vendor[v][x] for v in vendors])) for x in lines}
    best = max(mean.values())
    return max(x for x in lines if best - mean[x] <= tie)


def ratio_bar(medians: dict[str, float], *, bar: float = RATIO_BAR) -> dict:
    """Spec 7.3: max over min of the per-provider median crosses per match; passes when <= ``bar``."""
    lo = min(medians, key=lambda k: medians[k])
    hi = max(medians, key=lambda k: medians[k])
    ratio = float(medians[hi] / medians[lo]) if medians[lo] > 0 else float("inf")
    return {"ratio": ratio, "bar": bar, "passes": bool(ratio <= bar), "lowest": lo, "highest": hi}


def weighted_quantile(values_by_group: dict[str, np.ndarray], q: float) -> float:
    """The ``q`` quantile pooled with EQUAL weight per group (plan Task 10: no provider decides by row count)."""
    vals, wts = [], []
    for v in values_by_group.values():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(v)
            wts.append(np.full(len(v), 1.0 / len(v)))
    if not vals:
        return float("nan")
    v, w = np.concatenate(vals), np.concatenate(wts)
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    return float(v[order][min(int(np.searchsorted(cum, q, side="left")), len(v) - 1)])


def calibrate(rows: pd.DataFrame) -> dict:
    """Plan Task 10 Step 4: one pre-declared rule per ``CrossDeliveryParams`` field."""
    applicable = rows[rows["applicable"]]
    el = applicable[applicable["release_source"] == "elastic"]
    by = lambda df, col: {p: g[col].to_numpy(dtype=float) for p, g in df.groupby("provider")}  # noqa: E731
    gs_blocked = applicable[
        (applicable["provider"] == "gradientsports")
        & applicable["cross_native"].fillna(False).astype(bool)
        & applicable["cross_blocked"].fillna(False).astype(bool)
        & applicable["first_contact_is_opponent"].fillna(False).astype(bool)
    ]
    fr = rows.groupby("provider")["frame_rate"].min()
    cov_fail = {
        p: float((g["coverage"].dropna() < 0.8).mean()) if g["coverage"].notna().any() else float("nan")
        for p, g in applicable.groupby("provider")
    }
    return {
        "release_match_m": weighted_quantile(by(el, "release_dist_m"), 0.95),
        "block_radius_m": float(np.nanquantile(gs_blocked["first_contact_dist_m"], 0.95))
        if gs_blocked["first_contact_dist_m"].notna().any()
        else float("nan"),
        "release_gap_s": max(weighted_quantile(by(el, "passer_contact_dt"), 0.95), 1.0 / float(fr.max())),
        "direction_window_s": 2.0 / float(fr.min()),
        "min_ball_coverage": 0.8,
        "path_cap_s": 4.0,
        "coverage_below_0.8_share": cov_fail,
        "coverage_owner_decision_needed": bool(any(v > 0.2 for v in cov_fail.values() if np.isfinite(v))),
        "n_block_samples_gs": len(gs_blocked),
    }


# ---------------------------------------------------------------------------------------------- shared row model
def _delivery_rows(
    actions: pd.DataFrame, *, provider: str, match: str, native_label: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delivery-class rows of one match, re-typed at the most permissive line (``CANDIDATE_LINES[0]``)."""
    from silly_kicks.spadl import config as spc
    from silly_kicks.spadl.base import _delivery_context, _native_cross_flag, classify_deliveries

    a = actions.reset_index(drop=True)
    if "original_event_id" in a.columns:
        synthetic = (a["type_id"] == spc.actiontype_id["dribble"]) & a["original_event_id"].isna()
        a = a[~synthetic].reset_index(drop=True)
    if native_label:
        native = a["cross_native"] if "cross_native" in a.columns else pd.Series(_native_cross_flag(a["type_id"]))
    else:
        native = pd.Series(pd.array([pd.NA] * len(a), dtype="boolean"))
    geo = classify_deliveries(a, cross_start_x=CANDIDATE_LINES[0])
    _delivered, next_restart = _delivery_context(a)
    ids = [spc.actiontype_id[t] for t in _DELIVERY_NAMES]
    keep = geo["type_id"].isin(ids).to_numpy()
    names = dict(enumerate(spc.actiontypes))
    out = pd.DataFrame(
        {
            "provider": provider,
            "match": match,
            "row": np.flatnonzero(keep),
            "type_native": [names[t] for t in a.loc[keep, "type_id"]],
            "type_geo": [names[t] for t in geo.loc[keep, "type_id"]],
            "cross_native": pd.array(np.asarray(native, dtype=object)[keep], dtype="boolean"),
            "cross_blocked": pd.array(
                np.asarray(a["cross_blocked"] if "cross_blocked" in a else [pd.NA] * len(a), dtype=object)[keep],
                dtype="boolean",
            ),
            "start_x": a.loc[keep, "start_x"].to_numpy(dtype=float),
            "start_y": a.loc[keep, "start_y"].to_numpy(dtype=float),
            "next_restart": next_restart[keep],
        }
    )
    return out, geo


def _tracking_params(ctx):
    from silly_kicks.tracking._cross_delivery import CrossDeliveryParams

    p = ctx.params
    over = {k: p[k] for k in _TRACKING_PARAMS if p.get(k) is not None}
    if p.get("cross_start_x") is not None:
        over["cross_start_x"] = p["cross_start_x"]
    return CrossDeliveryParams(**over)


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
    rows, _geo = _delivery_rows(loaded[2], provider="statsbomb", match=match, native_label=True)
    return rows


# ---------------------------------------------------------------------------------------------- staged flat dataset
def _flat_refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _flat_load(ref, ctx):
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    root = dataset_dir(ctx.data_root, ctx.params["dataset"])
    return ref.provider, ref.stem, read_dataset_parquet(root / ref.files[0]), read_dataset_parquet(root / ref.files[1])


def _flat_work(item, ctx) -> pd.DataFrame:
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    provider, stem, frames, actions = item
    rows, geo = _delivery_rows(actions, provider=provider, match=stem, native_label=provider != "skillcorner")
    diag = _cross_delivery_diagnostics(geo, frames, _tracking_params(ctx)).iloc[rows["row"].to_numpy()]
    diag = diag.reset_index(drop=True)
    fr = frames["frame_rate"].dropna() if "frame_rate" in frames else pd.Series(dtype=float)
    return pd.concat([rows.reset_index(drop=True), diag], axis=1).assign(
        frame_rate=float(fr.min()) if len(fr) else float("nan")
    )


def _concat(results, ctx) -> pd.DataFrame:
    parts = [results[k] for k in sorted(results)]  # sorted-key reduce (ADR-128)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------------------------- report
def _true(s: pd.Series) -> pd.Series:
    """A nullable boolean as plain bool, <NA> -> False (counts and shares over known-True rows)."""
    return s.astype("boolean").fillna(False).astype(bool)


def _crosses_at(rows: pd.DataFrame, line: float) -> pd.Series:
    return (rows["type_geo"] == "cross") & (rows["start_x"] >= line)


def _f1_table(rows: pd.DataFrame) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    op = rows[rows["type_native"].isin(["pass", "cross"]) & rows["cross_native"].notna()]
    for provider, g in op.groupby("provider"):
        native = g["cross_native"].astype(bool).to_numpy()
        out[str(provider)] = {x: f1(_crosses_at(g, x).to_numpy(), native) for x in CANDIDATE_LINES}
    return out


def evidence(flat: pd.DataFrame, line: float) -> dict:
    """Spec 7.3 at ``line``: crosses per match and the ratio bar, attempts, set pieces, restarts, alignment shares."""
    per_match = flat.assign(c=_crosses_at(flat, line)).groupby(["provider", "match"])["c"].sum()
    med = {str(p): float(v) for p, v in per_match.groupby("provider").median().items()}
    out: dict = {"line": line, "median_crosses_per_match": med, "ratio_bar": ratio_bar(med) if med else None}
    zone = flat[flat["applicable"].astype(bool)]
    out["cross_attempt"] = {
        p: {
            "n_true": int(_true(g["attempt"]).sum()),
            "n_false": int(_true(~g["attempt"].astype("boolean")).sum()),
            "n_unknown": int(g["attempt"].isna().sum()),
            "blocked_share_of_true": float(g.loc[_true(g["attempt"]), "blocked_in"].astype(bool).mean())
            if _true(g["attempt"]).any()
            else float("nan"),
        }
        for p, g in zone.groupby("provider")
    }
    sp = flat[flat["type_native"].str.startswith(("corner", "freekick"))]
    out["set_piece_medians"] = {
        kind: {
            "before": sp.assign(c=sp["type_native"] == f"{kind}_crossed")
            .groupby(["provider", "match"])["c"]
            .sum()
            .groupby("provider")
            .median()
            .to_dict(),
            "after": sp.assign(c=sp["type_geo"] == f"{kind}_crossed")
            .groupby(["provider", "match"])["c"]
            .sum()
            .groupby("provider")
            .median()
            .to_dict(),
        }
        for kind in ("corner", "freekick")
    }
    restart = zone[zone["next_restart"]]
    out["restart_ended"] = {
        p: {
            "n": len(g),
            "n_unknown": int(g["attempt"].isna().sum()),
            "tracking_attempt_share_of_known": float(_true(g["attempt"])[g["attempt"].notna()].mean())
            if g["attempt"].notna().any()
            else float("nan"),
        }
        for p, g in restart.groupby("provider")
    }

    def _recall(g: pd.DataFrame, crossed: str) -> float:
        hit = _true(g["attempt"])
        return float((g.loc[hit, "type_geo"] == crossed).mean()) if hit.any() else float("nan")

    out["set_piece_event_recall_vs_attempt"] = {
        p: {k: _recall(g[g["type_native"].str.startswith(k)], f"{k}_crossed") for k in ("corner", "freekick")}
        for p, g in sp.groupby("provider")
    }
    corners = flat[flat["type_native"].str.startswith("corner")]
    out["corner_kind_source"] = {
        p: {"tracking": int(g["attempt"].notna().sum()), "event": int(g["attempt"].isna().sum())}
        for p, g in corners.groupby("provider")
    }
    out["alignment"] = alignment_table(flat)
    return out


#: Spec 7.3 alignment classes of a delivery row the tracking layer is asked to judge. They partition the rows:
#: each row has exactly one.
ALIGNMENT_CLASSES: tuple[str, ...] = (
    "aligned_elastic",  # answered; release from the confident ELASTIC frame
    "unaligned_but_matched",  # answered; release at the event time, ball within release_match_m
    "unknown_start",  # <NA>: the action start is not finite, so no release can be matched
    "misaligned",  # <NA>: the ball at the release is farther than release_match_m from the action start
    "no_ball_at_release",  # <NA>: no release time, no ball frame at the release, or an unresolved direction
    "coverage_unanswerable",  # <NA>: matched, but too little ball data along the path
)

#: The spec 7.3 row classes the alignment shares are split by (the corner share qualifies corner_kind_source).
ROW_CLASSES: tuple[str, ...] = ("open_play", "corner", "freekick")


def alignment_class(rows: pd.DataFrame) -> pd.Series:
    """One spec 7.3 class per row (``ALIGNMENT_CLASSES``); needs ``release_match_cut``."""
    answered = rows["attempt"].notna().to_numpy()
    elastic = (rows["release_source"] == "elastic").to_numpy()
    known_start = np.isfinite(rows["start_x"].to_numpy(dtype=float)) & np.isfinite(
        rows["start_y"].to_numpy(dtype=float)
    )
    dist = rows["release_dist_m"].to_numpy(dtype=float)
    cut = rows["release_match_cut"].to_numpy(dtype=float)
    cls = np.select(
        [answered & elastic, answered, ~known_start, ~np.isfinite(dist), dist > cut],
        ["aligned_elastic", "unaligned_but_matched", "unknown_start", "no_ball_at_release", "misaligned"],
        default="coverage_unanswerable",
    )
    return pd.Series(cls, index=rows.index)


def _row_class(type_geo: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [type_geo.isin(["pass", "cross"]), type_geo.str.startswith("corner"), type_geo.str.startswith("freekick")],
            list(ROW_CLASSES),
            default="other",
        ),
        index=type_geo.index,
    )


def alignment_table(flat: pd.DataFrame) -> dict:
    """Spec 7.3 alignment shares per provider x row class.

    The population is every delivery row the tracking layer is asked to judge: the applicable rows plus the
    open-play rows whose start is unknown (not known to lie outside the zone, so not ``False``). Each cell holds
    its row count ``n`` and one share per ``ALIGNMENT_CLASSES`` entry; the shares of a cell sum to 1.
    """
    row_class = _row_class(flat["type_geo"])
    unknown_start = ~(
        np.isfinite(flat["start_x"].to_numpy(dtype=float)) & np.isfinite(flat["start_y"].to_numpy(dtype=float))
    )
    judged = flat["applicable"].astype(bool).to_numpy() | ((row_class == "open_play").to_numpy() & unknown_start)
    rows = flat.loc[judged].assign(_row_class=row_class[judged])
    out: dict = {}
    for (provider, rc), g in rows.groupby(["provider", "_row_class"], sort=True):
        cls = alignment_class(g)
        out.setdefault(str(provider), {})[str(rc)] = {
            "n": len(g),
            **{c: float((cls == c).mean()) for c in ALIGNMENT_CLASSES},
        }
    return out


def _report_work(key, ctx) -> FileSet:
    sb = ctx.upstream("sb")
    flat = ctx.upstream("flat")
    params = _tracking_params(ctx)
    rows = pd.concat([sb, flat], ignore_index=True) if len(sb) else flat
    f1_table = _f1_table(rows)
    selectable = all(v in f1_table for v in F1_VENDORS)
    selected = select_start_line(f1_table) if selectable else None
    line = ctx.params["cross_start_x"] if ctx.params["cross_start_x"] is not None else selected
    flat = flat.assign(release_match_cut=params.release_match_m)
    payload = {
        "candidate_lines": list(CANDIDATE_LINES),
        "f1_by_vendor": {v: {str(x): f for x, f in t.items()} for v, t in f1_table.items()},
        "selected_start_line": selected,
        "reporting_line": line,
        "tracking_params": {k: getattr(params, k) for k in (*_TRACKING_PARAMS, "path_cap_s", "cross_start_x")},
        "calibration": calibrate(flat),
        "evidence": evidence(flat, line) if line is not None and len(flat) else None,
        "n_matches": {str(p): int(n) for p, n in rows.groupby("provider")["match"].nunique().items()},
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
            params=(*_TRACKING_PARAMS, "cross_start_x"),
            cross_commit_version="cross-def-flat-1",
            uses_dataset=True,
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("sb", "flat"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="report",
)


if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
