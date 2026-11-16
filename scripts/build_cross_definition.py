"""build_cross_definition: the cross-definition measurement (ADR-133; cross-definition spec sections 7.2-7.3).

A born-ready ADR-128 driver (scaffolded by `python -m scripts.hf_jobs scaffold`). It reads only staged datasets:

* ``--sb-dataset`` (the staged ``sb_open`` open-data revision): StatsBomb events converted with the current
  converter. Its ``cross_native`` (StatsBomb ``pass.cross``) is a vendor benchmark for the start line.
* ``--dataset`` (a staged ``flat_tc3`` dataset, e.g. ``xshot187``): the staged ``_actions`` RE-TYPED by the
  delivery rule, plus the frames. The staged actions came from the converters before ADR-133. Re-typing them
  reproduces the new converters exactly: the rule runs before ``_add_dribbles``, which never reads the type, so
  dropping the synthetic dribbles (a ``dribble`` row without an ``original_event_id``) and re-typing gives the
  new output. The old type is the provider's native label (SkillCorner has none: ``<NA>``).

The report applies the PRE-DECLARED rules: the start line (spec 7.2), the ``CrossDeliveryParams`` calibration and
its run sequence (cross spec section 7.4), and the spec 7.3 corpus evidence at the reporting line
(``--cross-start-x``, else the selected line).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main

CANDIDATE_LINES: tuple[float, ...] = (52.5, 60.0, 65.0, 70.0, 75.0, 80.0)  # cross spec section 7.2
F1_VENDORS: tuple[str, ...] = ("statsbomb", "gradientsports")  # full native cross labels (spec decision 8)
F1_TIE: float = 0.005
RATIO_BAR: float = 1.5
RELEASE_REFERENCE: str = "idsse"  # the clean-sync provider for release_match_m (cross spec section 7.4)
MIN_RELEASE_SAMPLES: int = 200  # owner trigger: fewer IDSSE ELASTIC-confident rows (cross spec section 7.4)
MAX_UNKNOWN_SHARE: float = 0.2  # owner trigger: the share a release match or a coverage floor may turn <NA>
# Owner trigger on the native releases (cross spec section 7.4): a provider's native rows need a median passer-touch
# offset inside +-NATIVE_DT_BOUND_S and a passer-touch share of at least NATIVE_MIN_TOUCH_SHARE. Chosen after the
# native preview (GS -0.23 s / 50.4 %, SkillCorner -0.20 s / 78.4 %); disclosed in the spec.
NATIVE_DT_BOUND_S: float = 0.30
NATIVE_MIN_TOUCH_SHARE: float = 0.40

_TRACKING_PARAMS = ("direction_window_s", "release_gap_s", "min_ball_coverage", "release_match_m")
#: The contact scorer overrides (scorer spec section 4), passed through to ``CrossDeliveryParams.contact_params``: the
#: cross runs read the clean code commit with the development run's frozen scorer and its threshold.
_SCORER_PARAMS = ("contact_scorer", "min_contact_probability")
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
    Param("contact_scorer", str, None, help="BallContactParams.contact_scorer override: the frozen scorer directory"),
    Param("min_contact_probability", float, None, help="BallContactParams.min_contact_probability override"),
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


_CALIBRATED = ("release_match_m", "release_gap_s", "direction_window_s")


def fixed_point(cal: dict, params: dict, *, selected_line: float | None) -> dict:
    """Cross spec section 7.4 run 3: does the calibration reproduce each parameter the run used? (``all`` = every one).

    ``cross_start_x`` checks that the run's diagnostics used the selected start line (an omitted
    ``--cross-start-x`` leaves them at the provisional line). An unknown selection is not a fixed point.
    """
    out = {k: bool(np.isclose(cal[k], params[k], rtol=1e-9, atol=1e-12)) for k in _CALIBRATED}
    out["cross_start_x"] = selected_line is not None and bool(np.isclose(selected_line, params["cross_start_x"]))
    return {**out, "all": all(out.values())}


def _native_stats(passer_contact_dt: pd.Series) -> dict:
    """The native-release check of cross spec section 7.4: row count, passer-touch share, touch-offset quantiles."""
    dt = passer_contact_dt.to_numpy(dtype=float)
    found = dt[np.isfinite(dt)]
    q = np.quantile(found, [0.05, 0.5, 0.95]) if len(found) else np.full(3, np.nan)
    return {
        "n": len(dt),
        "touch_share": float(len(found) / len(dt)) if len(dt) else float("nan"),
        "dt_q05": float(q[0]),
        "dt_q50": float(q[1]),
        "dt_q95": float(q[2]),
    }


def _p95(values: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, 0.95)) if len(v) else float("nan")


def cut_out_evidence(g: pd.DataFrame) -> dict:
    """Cross spec sections 7.3/7.4: the cut-out rows of ``g`` and their share of the rows whose attempt is decided
    or cut out (an unanswerable row is in neither), plus the cut rows' direction counts."""
    cut = g["cut_out"].astype(bool)
    answered = g["attempt"].notna() | cut
    return {
        "n_cut_out": int(cut.sum()),
        "cut_out_share": float(cut.sum() / answered.sum()) if answered.any() else float("nan"),
        "n_cut_out_direction_cut": int((cut & g["direction_cut"].astype(bool)).sum()),
        "n_direction_unresolved": int(g["direction_unresolved"].astype(bool).sum()),
    }


def block_call(rows: pd.DataFrame, *, release_gap_s: float) -> dict:
    """Cross spec section 7.3 run 3 / decision 12: the GS block call (``cut_out``) against PFF code B
    (``cross_blocked``) on the open-play native crosses whose attempt is decided or cut out, and the code-B crosses
    whose block (the next opponent action, ``next_opponent_t``) comes inside ``release_gap_s`` after the release
    (``release_t``).
    """
    g = rows[(rows["provider"] == "gradientsports") & _true(rows["cross_native"]) & rows["cross_blocked"].notna()]
    g = g[g["attempt"].notna() | g["cut_out"].astype(bool)]
    called = g["cut_out"].astype(bool)
    code_b = _true(g["cross_blocked"])
    return {
        "n": len(g),
        "n_called": int(called.sum()),
        "n_code_b": int(code_b.sum()),
        "precision": float((called & code_b).sum() / called.sum()) if called.any() else float("nan"),
        "recall": float((called & code_b).sum() / code_b.sum()) if code_b.any() else float("nan"),
        "n_block_inside_gap": int((code_b & ((g["next_opponent_t"] - g["release_t"]) < release_gap_s)).sum()),
    }


def calibrate(rows: pd.DataFrame) -> dict:
    """Cross spec section 7.4: one pre-declared rule per ``CrossDeliveryParams`` field, on clean samples.

    ``release_match_m`` from the IDSSE reference only; ``release_gap_s`` over the rows that pass it. r10 removes
    ``block_radius_m`` and its sample; the cut-out share and its owner trigger are read here. The section 7.4 table
    gives the reasons and the triggers.
    """
    from silly_kicks.tracking._cross_delivery import CrossDeliveryParams

    applicable = rows[rows["applicable"]]
    el = applicable[applicable["release_source"] == "elastic"]
    # r8: a native release (spec 6.3 step 0) counts wherever an ELASTIC-confident one does, except for the IDSSE
    # release-match reference, whose rows are event-timed and so ELASTIC-aligned.
    aligned = applicable[applicable["release_source"].isin(["native", "elastic"])]
    by = lambda df, col: {p: g[col].to_numpy(dtype=float) for p, g in df.groupby("provider")}  # noqa: E731
    release_match = _p95(el.loc[el["provider"] == RELEASE_REFERENCE, "release_dist_m"])
    # The percentile's own sample: finite IDSSE distances (a NaN distance never enters ``_p95``).
    n_release = int(el.loc[el["provider"] == RELEASE_REFERENCE, "release_dist_m"].notna().sum())
    matched = aligned[aligned["release_dist_m"] <= release_match]
    # Over finite distances only: a NaN distance (unknown start, no ball frame) is <NA> whatever the match is.
    fail_share = {
        str(p): float((d > release_match).mean()) if len(d := g["release_dist_m"].dropna()) else float("nan")
        for p, g in aligned.groupby("provider")
    }
    native = applicable[applicable["release_source"] == "native"]
    native_release = {str(p): _native_stats(g["passer_contact_dt"]) for p, g in native.groupby("provider")}
    fr = rows.groupby("provider")["frame_rate"].min()
    gap = weighted_quantile(by(matched, "passer_contact_dt"), 0.95)
    cut_share = {str(p): cut_out_evidence(g)["cut_out_share"] for p, g in applicable.groupby("provider")}
    cov_fail = {
        p: float((g["coverage"].dropna() < 0.8).mean()) if g["coverage"].notna().any() else float("nan")
        for p, g in applicable.groupby("provider")
    }
    return {
        "release_match_m": release_match,
        "n_release_match_samples": n_release,
        "release_dist_p95_by_provider": {str(p): _p95(g["release_dist_m"]) for p, g in aligned.groupby("provider")},
        "native_release": native_release,
        "native_release_owner_decision_needed": bool(
            any(
                s["touch_share"] < NATIVE_MIN_TOUCH_SHARE
                or not (np.isfinite(s["dt_q50"]) and abs(s["dt_q50"]) <= NATIVE_DT_BOUND_S)
                for s in native_release.values()
            )
        ),
        "release_match_fail_share": fail_share,
        "release_match_owner_decision_needed": bool(
            n_release < MIN_RELEASE_SAMPLES or any(v > MAX_UNKNOWN_SHARE for v in fail_share.values())
        ),
        "release_gap_s": max(gap, 1.0 / float(fr.max())) if np.isfinite(gap) else float("nan"),
        "direction_window_s": 2.0 / float(fr.min()),
        "min_ball_coverage": 0.8,
        "cut_out_window_s": CrossDeliveryParams().cut_out_window_s,
        "path_cap_s": 4.0,
        "coverage_below_0.8_share": cov_fail,
        "coverage_owner_decision_needed": bool(any(v > MAX_UNKNOWN_SHARE for v in cov_fail.values() if np.isfinite(v))),
        "cut_out_share": cut_share,
        "cut_out_owner_decision_needed": bool(any(v > MAX_UNKNOWN_SHARE for v in cut_share.values() if np.isfinite(v))),
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
            "next_opponent_t": _next_opponent_t(a)[keep],
        }
    )
    return out, geo


def _next_opponent_t(a: pd.DataFrame) -> np.ndarray:
    """Per row, the time of the next action by another team in the same period (NaN when none): the PFF block time
    for a GS code-B cross, whose blocker's ``RE`` event the converter keeps (cross spec 7.3)."""
    from silly_kicks.id_compat import same_id

    t = a["time_seconds"].to_numpy(dtype=float)
    per = a["period_id"].to_numpy()
    team = a["team_id"].to_numpy(dtype=object)
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if per[j] != per[i]:
                break
            if not pd.isna(team[j]) and not pd.isna(team[i]) and not same_id(team[j], team[i]):
                out[i] = t[j]
                break
    return out


def _tracking_params(ctx):
    from silly_kicks.tracking._ball_contact import BallContactParams
    from silly_kicks.tracking._cross_delivery import CrossDeliveryParams

    p = ctx.params
    over = {k: p[k] for k in _TRACKING_PARAMS if p.get(k) is not None}
    if p.get("cross_start_x") is not None:
        over["cross_start_x"] = p["cross_start_x"]
    contact = {k: p[k] for k in _SCORER_PARAMS if p.get(k) is not None}
    if "contact_scorer" in contact:
        contact["min_velocity_change_ms"] = 0.0  # the scorer reads threshold-free peaks (scorer spec 2.5)
    return CrossDeliveryParams(**over, contact_params=BallContactParams(**contact))


def _declare_scorer(params) -> dict[str, object]:
    """The frozen scorer's content enters the flat stage's token, so a refit at the same path is a new generation."""
    import hashlib

    path = params.get("contact_scorer")
    if not path or path == "default":
        return {}
    model = Path(path) / "model.json"
    return {
        "contact_scorer_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else "missing"
    }


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


def evidence(flat: pd.DataFrame, line: float, *, release_gap_s: float) -> dict:
    """Spec 7.3 at ``line``: crosses per match and the ratio bar, attempts (with the r10/r12 cut-out counts), set
    pieces, restarts, alignment shares, and the GS block call (r12 7.4)."""
    per_match = flat.assign(c=_crosses_at(flat, line)).groupby(["provider", "match"])["c"].sum()
    med = {str(p): float(v) for p, v in per_match.groupby("provider").median().items()}
    out: dict = {"line": line, "median_crosses_per_match": med, "ratio_bar": ratio_bar(med) if med else None}
    zone = flat[flat["applicable"].astype(bool)]
    out["cross_attempt"] = {
        p: {
            "n_true": int(_true(g["attempt"]).sum()),
            "n_false": int(_true(~g["attempt"].astype("boolean")).sum()),
            "n_unknown": int(g["attempt"].isna().sum()),
            **cut_out_evidence(g),
        }
        for p, g in zone.groupby("provider")
    }
    sp = flat[flat["type_native"].str.startswith(("corner", "freekick"))]

    def _median_per_match(hit: pd.Series) -> dict:
        return sp.assign(c=hit).groupby(["provider", "match"])["c"].sum().groupby("provider").median().to_dict()

    out["set_piece_medians"] = {
        kind: {
            "before": _median_per_match(sp["type_native"] == f"{kind}_crossed"),
            "after": _median_per_match(sp["type_geo"] == f"{kind}_crossed"),
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
    out["block_call"] = block_call(zone, release_gap_s=release_gap_s)
    return out


#: Spec 7.3 alignment classes of a delivery row the tracking layer is asked to judge. They partition the rows:
#: each row has exactly one.
ALIGNMENT_CLASSES: tuple[str, ...] = (
    "native",  # answered; release at the provider's own tracking-clock event time (cross spec 6.3 step 0)
    "aligned_elastic",  # answered; release from the confident ELASTIC frame
    "unaligned_but_matched",  # answered; release at the event time, ball within release_match_m
    "unknown_start",  # <NA>: the action start is not finite, so no release can be matched
    "misaligned",  # <NA>: the ball at the release is farther than release_match_m from the action start
    "no_ball_at_release",  # <NA>: no release time, no ball frame at the release, or an unresolved direction
    "coverage_unanswerable",  # <NA>: matched, but too little ball data along the path
    "cut_out",  # <NA>: matched and answerable, but cut out by an opponent near the release (cross spec 6.3)
)

#: The spec 7.3 row classes the alignment shares are split by (the corner share qualifies corner_kind_source).
ROW_CLASSES: tuple[str, ...] = ("open_play", "corner", "freekick")


def alignment_class(rows: pd.DataFrame) -> pd.Series:
    """One spec 7.3 class per row (``ALIGNMENT_CLASSES``); needs ``release_match_cut``."""
    answered = rows["attempt"].notna().to_numpy()
    native = (rows["release_source"] == "native").to_numpy()
    elastic = (rows["release_source"] == "elastic").to_numpy()
    known_start = np.isfinite(rows["start_x"].to_numpy(dtype=float)) & np.isfinite(
        rows["start_y"].to_numpy(dtype=float)
    )
    dist = rows["release_dist_m"].to_numpy(dtype=float)
    cut = rows["release_match_cut"].to_numpy(dtype=float)
    cut_out = rows["cut_out"].astype(bool).to_numpy()
    cls = np.select(
        [answered & native, answered & elastic, answered, cut_out, ~known_start, ~np.isfinite(dist), dist > cut],
        [
            "native",
            "aligned_elastic",
            "unaligned_but_matched",
            "cut_out",
            "unknown_start",
            "no_ball_at_release",
            "misaligned",
        ],
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
    used = {k: getattr(params, k) for k in (*_TRACKING_PARAMS, "cut_out_window_s", "path_cap_s", "cross_start_x")}
    used |= {k: getattr(params.contact_params, k) for k in _SCORER_PARAMS}
    used |= _declare_scorer({"contact_scorer": params.contact_params.contact_scorer})
    cal = calibrate(flat)
    payload = {
        "candidate_lines": list(CANDIDATE_LINES),
        "f1_by_vendor": {v: {str(x): f for x, f in t.items()} for v, t in f1_table.items()},
        "selected_start_line": selected,
        "reporting_line": line,
        "tracking_params": used,
        "calibration": cal,
        "fixed_point": fixed_point(cal, used, selected_line=selected),
        "evidence": evidence(flat, line, release_gap_s=params.release_gap_s)
        if line is not None and len(flat)
        else None,
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
            params=(*_TRACKING_PARAMS, *_SCORER_PARAMS, "cross_start_x"),
            declare=_declare_scorer,
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
