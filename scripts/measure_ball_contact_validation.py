"""TF-79 ball-contact detection validation (reported-not-gated).

Characterises detect_ball_contacts against EVENT-recorded touches (the SPADL actions) per provider.
It does NOT reduce the detector to a single tight-window precision/recall: on real data the SPADL
event timestamp LAGS the physical ball contact (~0.4 s median on IDSSE), so a tight symmetric window
measures event-feed misalignment, not detector quality. Instead it reports, loss-free:

- contact rate (contacts / min) and the honest UNRESOLVED rate,
- the SIGNED event->nearest-same-player-contact offset distribution (the misalignment TF-79 exposes),
- agreement-vs-tolerance: fraction of events with a same-player contact within tau (tau in 0.2/0.5/1.0 s),
- confidence calibration: match rate (<=0.5 s to an event) by confidence bucket (does higher rank better?),
- the velocity-change |dv| distribution (touch-vs-flight separation signal).

Dense-optical providers first (idsse = public Sportec/IDSSE, gradientsports); SkillCorner broadcast
contributes the unresolved rate. No accuracy threshold -- numbers are REPORTED to a memo, never a gate
(spec sec. 8, amended). Loads via the pining loader; requires PINING_FOR_THE_DATA_TOKEN for the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts._gs_truth import (
    BAR_FALSE_FIRST,
    BAR_RIGHT,
    BAR_TIMING_S,
    SUPPORT_RADIUS_M,
    TRUTH_TAU_S,
    otb_sequences,
    score_next_touch,
    select_point,
    shares,
    split_of,
    track_supported,
)
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import BallContactParams, detect_ball_contacts

#: Tolerances (s) at which event<->contact agreement is reported. FIXED / pre-registered: these are
#: REPORTING grids, never selected to flatter a number (spec sec. 14 no-leakage rule).
TOLERANCES_S = (0.2, 0.5, 1.0)
_MATCH_TOL_S = 0.5  # the tolerance used for the per-contact "matched an event" calibration flag

_DENSE_OPTICAL = ["idsse", "gradientsports"]  # idsse = the public Sportec/IDSSE optical dataset
_PROVIDERS = [*_DENSE_OPTICAL, "skillcorner"]


def _match_minutes(frames: pd.DataFrame) -> float:
    total = 0.0
    ts = frames["time_seconds"]
    period = frames["period_id"]
    for p in period.unique():
        vals = ts[period == p].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            total += float(vals.max())
    return total / 60.0


def _by_player_times(df: pd.DataFrame) -> dict:
    """{(period, canonical_player) -> sorted time_seconds ndarray}."""
    out: dict = {}
    for (per, pid), g in df.groupby(["period_id", df["player_id"].map(canonical_id)], sort=False):
        out[(per, pid)] = np.sort(g["time_seconds"].to_numpy(dtype=np.float64))
    return out


def _nearest(times: np.ndarray, t: float) -> float:
    """Signed nearest (times - t); NaN if empty."""
    if times.size == 0:
        return float("nan")
    i = int(np.searchsorted(times, t))
    cands = []
    if i < times.size:
        cands.append(times[i] - t)
    if i > 0:
        cands.append(times[i - 1] - t)
    return float(min(cands, key=abs))


def _evaluate_match(actions: pd.DataFrame, frames: pd.DataFrame, *, params: BallContactParams | None = None):
    """Return (per-contact + per-event long rows, per-match integer tallies) for one match."""
    contacts = detect_ball_contacts(frames, params=params or BallContactParams())
    src = contacts["contact_source"].astype(object)
    n_all = len(contacts)
    n_unresolved = int((src == "unresolved").sum())
    res = contacts[(src != "unresolved") & contacts["player_id"].notna()]
    truth = actions.loc[actions["player_id"].notna(), ["period_id", "player_id", "time_seconds"]]

    ev_times = _by_player_times(truth.rename(columns={"time_seconds": "time_seconds"}))
    ct_times = _by_player_times(res[["period_id", "player_id", "time_seconds"]])

    rows = []
    # events: signed nearest same-player contact offset; matched within _MATCH_TOL_S
    for _, ev in truth.iterrows():
        key = (ev["period_id"], canonical_id(ev["player_id"]))
        off = _nearest(ct_times.get(key, np.array([])), float(ev["time_seconds"]))
        rows.append(("event", off, np.nan, bool(np.isfinite(off) and abs(off) <= _MATCH_TOL_S)))
    # contacts: confidence + dv; matched to an event within _MATCH_TOL_S (calibration)
    for _, c in res.iterrows():
        key = (c["period_id"], canonical_id(c["player_id"]))
        d = _nearest(ev_times.get(key, np.array([])), float(c["time_seconds"]))
        rows.append(
            (
                "contact",
                float(c["confidence"]),
                float(c["velocity_change_ms"]),
                bool(np.isfinite(d) and abs(d) <= _MATCH_TOL_S),
            )
        )

    tally = {
        "n_all_contacts": n_all,
        "n_unresolved": n_unresolved,
        "n_events": len(truth),
        "minutes_centis": round(_match_minutes(frames) * 100),
    }
    return rows, tally


def _pct(a: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    a = a[np.isfinite(a)]
    return {f"p{int(q * 100)}": (float(np.quantile(a, q)) if a.size else float("nan")) for q in qs}


def _block(long: pd.DataFrame, counters: dict, prefix: str) -> dict:
    def g(k: str) -> int:
        return int(counters.get(f"{k}__{prefix}", 0)) if prefix else int(counters.get(k, 0))

    n_all, n_unres, n_ev = g("n_all_contacts"), g("n_unresolved"), g("n_events")
    minutes = g("minutes_centis") / 100.0
    ev = long[long["kind"] == "event"]
    ct = long[long["kind"] == "contact"]
    offsets = ev["value"].to_numpy(dtype=float)
    agree = {
        f"agree_within_{t}s": (float(np.mean(np.abs(offsets) <= t)) if offsets.size else float("nan"))
        for t in TOLERANCES_S
    }
    conf = ct["value"].to_numpy(dtype=float)
    dv = ct["dv"].to_numpy(dtype=float)
    # confidence calibration: match rate by bucket
    calib = {}
    if len(ct):
        for lo, hi, name in ((0.0, 0.33, "low"), (0.33, 0.66, "mid"), (0.66, 1.01, "high")):
            b = ct[(ct["value"] >= lo) & (ct["value"] < hi)]
            calib[name] = {"n": len(b), "event_match_rate": (float(b["matched"].mean()) if len(b) else float("nan"))}
    return {
        "n_contacts": n_all,
        "n_resolved": len(ct),
        "n_events": n_ev,
        "contact_rate_per_min": (n_all / minutes) if minutes > 0 else float("nan"),
        "unresolved_rate": (n_unres / n_all) if n_all else float("nan"),
        "signed_offset_s": _pct(offsets),
        "agreement_vs_tolerance": agree,
        "confidence_pct": _pct(conf),
        "confidence_calibration": calib,
        "velocity_change_ms_pct": _pct(dv),
        "match_tol_s_for_calibration": _MATCH_TOL_S,
    }


def _aggregate(long: pd.DataFrame, counters: dict, *, n_matches: int, providers, provenance) -> dict:
    present = [p for p in (providers or []) if f"n_all_contacts__{p}" in counters]
    prov = provenance if isinstance(provenance, dict) else {}
    # Top-level `run_commit` / `run_tree_dirty` are the ADR-037 canonical provenance shape that the
    # committed-artifact gate (`tests/scripts/test_artifact_provenance_output.py`) enforces -- NOT a
    # nested `provenance.commit`, which that gate treats as an untraceable number.
    return {
        "run_commit": prov.get("commit"),
        "run_tree_dirty": prov.get("dirty"),
        "overall": _block(long, counters, ""),
        "per_provider": {p: _block(long[long["provider"] == p], counters, p) for p in sorted(present)},
        "corpus_bound": {"n_matches": int(n_matches), "providers": sorted(providers or [])},
    }


# ---------------------------------------------------------------------------------------------------------------
# TF-79 v2 GS truth mode (detector spec 2026-11-15 r2 sections 2, 6 and 7; ADR-134). The truth is PFF's on-ball
# sequences (OTB game events): a player, a team, and [startTime, endTime] on the event clock, period-relative through
# the period's kickoff startTime. The measure is what the cross enricher and TF-46 read: after a release, the first
# contact by another player. Thresholds are chosen on the development half and scored once on the held-out half.
# ---------------------------------------------------------------------------------------------------------------
#: The 20-point rule-mode development grid (detector spec section 6): the reach and the change threshold.
GRID_RADII = (0.8, 1.0, 1.2, 1.5)
GRID_DV = (1.0, 2.0, 3.0, 4.0, 5.0)
#: The other-provider owner trigger (absolute change against the commit-3 v1 reference).
OTHER_PROVIDER_DELTA = 0.10


def _scorer_sha(params: BallContactParams | None) -> str | None:
    """The frozen scorer's ``model.json`` sha256 for the token (None without a scorer)."""
    import hashlib

    if params is None or not params.contact_scorer:
        return None
    model = Path(params.contact_scorer) / "model.json"
    return hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else "missing"


def _token_inputs(params: BallContactParams | None) -> dict:
    """The shard token: a change to any of these values starts a new generation."""
    return {
        "tolerances_s": list(TOLERANCES_S),
        "detector": "v2",
        "params": repr(params or BallContactParams()),
        "contact_scorer_model_sha256": _scorer_sha(params),  # a refit at the same path is a new generation
        "geometry": "frame_ltr",
    }


def scorer_params(contact_scorer: str | None, min_contact_probability: float | None) -> BallContactParams | None:
    """The detector the other-provider reports score: the development run's frozen scorer and its threshold, on
    threshold-free peaks (scorer spec section 3); ``None`` (the library defaults) when no scorer is given."""
    if not contact_scorer:
        return None
    if min_contact_probability is None:
        raise ValueError("--contact-scorer needs --min-contact-probability (the development run's chosen threshold)")
    return BallContactParams(
        contact_scorer=contact_scorer,
        min_contact_probability=float(min_contact_probability),
        min_velocity_change_ms=0.0,
    )


def other_provider_trigger(reference: dict, current: dict) -> dict:
    """The detector spec section 6 owner trigger: an IDSSE or SkillCorner agreement share (same-player contact
    within 0.5 s) that falls by more than ``OTHER_PROVIDER_DELTA`` from the commit-3 v1 reference, or a SkillCorner
    unresolved rate that rises by more than it."""
    details = {}
    fires = False
    for p in ("idsse", "skillcorner"):
        ref = (reference.get("per_provider") or {}).get(p)
        cur = (current.get("per_provider") or {}).get(p)
        if not ref or not cur:
            continue
        fall = ref["agreement_vs_tolerance"]["agree_within_0.5s"] - cur["agreement_vs_tolerance"]["agree_within_0.5s"]
        rise = cur["unresolved_rate"] - ref["unresolved_rate"]
        hit = fall > OTHER_PROVIDER_DELTA or (p == "skillcorner" and rise > OTHER_PROVIDER_DELTA)
        details[p] = {"agreement_fall": fall, "unresolved_rise": rise, "fires": bool(hit)}
        fires = fires or hit
    return {"fires": bool(fires), "details": details}


def run_gs_truth(out, *, match_ids=None, provenance=None, cache_dir=None, source=None) -> dict:
    """Score the RULE detector (the reach gate) against PFF truth on the development half (detector spec section 6):
    the 20-point grid and its selection. It is the model-free baseline of the contact scorer (scorer spec section 3);
    the scorer's own development and held-out runs are ``scripts/build_contact_scorer.py``. ``source=(refs, load)``
    injects a loader for tests, where ``load(ref)`` returns ``(loaded_match, events_json)``.
    """
    from scripts._driver import for_each, shard_path
    from scripts._provenance import git_provenance
    from silly_kicks.tracking._ball_contact import _candidate_peaks, _finalize_contacts

    prov_run = provenance or git_provenance()
    points = [BallContactParams(contact_radius_m=r, min_velocity_change_ms=v) for r in GRID_RADII for v in GRID_DV]
    if source is None:
        from _loader_pining import gradientsports_events_json, pining_source, resolve_cache_dir

        cd = resolve_cache_dir(cache_dir)
        refs, load_full = pining_source(["gradientsports"], match_ids=match_ids, token=None, cache_dir=cd)

        def load(ref):
            return load_full(ref), gradientsports_events_json(ref, cache_dir=cd)
    else:
        refs, load = source
    refs = [r for r in refs if split_of(r.match_id) == "dev"]

    def _work(item):
        lm, events_json = item
        frames = lm[3]
        seq = otb_sequences(events_json)
        seq = seq.assign(supported=track_supported(seq, frames))  # the bar's population (scorer spec r7 section 3)
        peaks = _candidate_peaks(frames, BallContactParams())
        rows = []
        for p in points:
            s = score_next_touch(seq, _finalize_contacts(peaks, p, pid_dtype=frames["player_id"].dtype))
            offsets = s.pop("release_offsets")
            rows.append(
                {
                    "contact_radius_m": p.contact_radius_m,
                    "min_velocity_change_ms": p.min_velocity_change_ms,
                    **s,
                    "release_offsets": json.dumps(offsets),
                }
            )
        return pd.DataFrame(rows)

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        counters=lambda _item, _frame: {"n_matches": 1},
        shard_root=Path(out) / "shards",
        token_inputs={
            "mode": "gs_truth",
            "split": "dev",
            "tau_s": TRUTH_TAU_S,
            "points": [[p.contact_radius_m, p.min_velocity_change_ms] for p in points],
            "detector": "v2",
            "support_radius_m": SUPPORT_RADIUS_M,  # the shards carry the supported counts (scorer spec r7)
        },
        tag="ball_contact_truth",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"ball-contact truth failures: {res.failures}")
    long = pd.concat([pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys], ignore_index=True)
    count_cols = [
        "n",
        "right",
        "false_first",
        "wrong_player",
        "missed",
        "n_block",
        "block_right",
        "block_missed",
        "n_contacts",
        "n_unresolved",
        "sup_n",
        "sup_right",
        "sup_false_first",
        "sup_wrong_player",
        "sup_missed",
    ]
    grid: dict[tuple[float, float], dict] = {}
    for key, g in long.groupby(["contact_radius_m", "min_velocity_change_ms"], sort=True):
        r, v = cast("tuple[float, float]", key)
        offsets = [o for js in g["release_offsets"] for o in json.loads(js)]
        grid[(float(r), float(v))] = shares(g[count_cols].sum().to_dict(), offsets)
    report: dict = {
        "run_commit": prov_run.get("commit"),
        "run_tree_dirty": prov_run.get("dirty"),
        "split": "dev",
        "tau_s": TRUTH_TAU_S,
        "bar": {"right": BAR_RIGHT, "false_first": BAR_FALSE_FIRST, "timing_s": BAR_TIMING_S},
        "matches": sorted(str(r.match_id) for r in refs),
        "grid": [{"contact_radius_m": r, "min_velocity_change_ms": v, **m} for (r, v), m in grid.items()],
    }
    picked = select_point(
        grid, prefer=lambda k: (-k[1], k[0]), population="supported"
    )  # the larger threshold, then the smaller radius
    chosen = cast("tuple[float, float] | None", picked)
    report["chosen"] = {"contact_radius_m": chosen[0], "min_velocity_change_ms": chosen[1]} if chosen else None
    report["chosen_on_grid_edge"] = chosen is not None and (
        chosen[0] in (GRID_RADII[0], GRID_RADII[-1]) or chosen[1] in (GRID_DV[0], GRID_DV[-1])
    )
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "truth_dev.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run(out, *, providers=None, match_ids=None, provenance=None, cache_dir=None, source=None, params=None) -> dict:
    """Validate ball-contact detection and write the memo. ``source=(refs, load)`` injects a loader
    for tests (no network). ``params`` scores a chosen point (the TF-79 v2 other-provider report). ``run`` is called
    directly by tests; enforcement lives in ``main``."""
    from scripts._driver import for_each, shard_path
    from scripts._provenance import git_provenance

    prov_run = provenance or git_provenance()
    provs = list(providers) if providers else list(_PROVIDERS)

    if source is None:
        from _loader_pining import pining_source, resolve_cache_dir

        refs, load = pining_source(provs, match_ids=match_ids, token=None, cache_dir=resolve_cache_dir(cache_dir))
    else:
        refs, load = source

    _meta: dict = {}

    def _work(item):
        provider, _match_id, actions, frames, *_ = item
        rows, tally = _evaluate_match(actions, frames, params=params)
        _meta.clear()
        _meta.update({"n_matches": 1})
        _meta.update({f"{k}__{provider}": v for k, v in tally.items()})
        _meta.update(dict(tally))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["kind", "value", "dv", "matched"])
        df.insert(0, "provider", provider)
        return df

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        counters=lambda _item, _frame: dict(_meta),
        shard_root=Path(out) / "shards",
        token_inputs=_token_inputs(params),
        tag="ball_contact",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"ball-contact validation failures: {res.failures}")

    parts = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys]
    long = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["provider", "kind", "value", "dv", "matched"])
    )
    metrics = _aggregate(
        long, dict(res.counters), n_matches=int(res.counters.get("n_matches", 0)), providers=provs, provenance=prov_run
    )
    _write(Path(out), metrics)
    return metrics


def _fmt_block(name: str, b: dict) -> list[str]:
    """Render one provider's (or the pooled) characterisation block as markdown lines."""
    return [
        f"### {name}",
        f"- contacts: {b['n_contacts']} (resolved {b['n_resolved']}, events {b['n_events']})",
        f"- contacts/min: {b['contact_rate_per_min']}",
        f"- unresolved rate: {b['unresolved_rate']}",
        f"- signed event->contact offset (s): {b['signed_offset_s']}",
        f"- agreement vs tolerance: {b['agreement_vs_tolerance']}",
        f"- confidence calibration (event-match rate by bucket): {b['confidence_calibration']}",
        f"- velocity-change |dv| (m/s): {b['velocity_change_ms_pct']}",
        "",
    ]


def _write(out: Path, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    per = metrics.get("per_provider", {})
    # SkillCorner is the primary production provider -> report it FIRST; others follow alphabetically.
    order = (["skillcorner"] if "skillcorner" in per else []) + sorted(p for p in per if p != "skillcorner")
    cb = metrics["corpus_bound"]
    lines = [
        "# TF-79 ball-contact detection validation",
        "",
        f"Corpus: {cb['n_matches']} matches, providers {cb['providers']}.",
        "Reported, not gated (no accuracy threshold; spec sec. 8, amended).",
        "",
        "SPADL event timestamps LAG the physical ball contact, so a tight symmetric window measures",
        "event-feed misalignment, not detector quality. Hence the characterisation below, not one number.",
        "",
        "## Per provider",
        "",
        "Authoritative read: each provider is scored ONLY on its own matches, so these blocks are not",
        "diluted by the provider mix. SkillCorner (the primary production provider) is listed first.",
        "",
    ]
    for p in order:
        lines += _fmt_block(p, per[p])
    lines += [
        "## Pooled across all providers",
        "",
        "_Provider-mix dependent: dominated by whichever provider contributes the most matches "
        "(here SkillCorner). This is NOT a detector-quality score -- read the per-provider blocks above._",
        "",
    ]
    lines += _fmt_block("overall (pooled)", metrics["overall"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-79 ball-contact detection validation (reported-not-gated).")
    parser.add_argument("--out", type=Path, required=True, help="output dir, e.g. docs/research/ball_contact_detection")
    parser.add_argument("--providers", nargs="+", default=None, help=f"providers (default {_PROVIDERS})")
    parser.add_argument("--match-ids-json", type=Path, default=None, help="optional JSON pin of provider -> [match_id]")
    parser.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (provenance stamps it)")
    parser.add_argument(
        "--gs-truth", action="store_true", help="the rule detector's development grid against PFF truth (GS only)"
    )
    parser.add_argument("--reference", type=Path, default=None, help="v1 metrics.json for the other-provider trigger")
    parser.add_argument("--contact-scorer", default=None, help="the frozen scorer directory (scorer spec section 3)")
    parser.add_argument("--min-contact-probability", type=float, default=None, help="the frozen scorer's threshold")
    args = parser.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    match_ids = json.loads(args.match_ids_json.read_text(encoding="utf-8")) if args.match_ids_json else None
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    if args.gs_truth:
        rep = run_gs_truth(args.out, match_ids=match_ids, provenance=prov)
        print(f"ball-contact truth (rule, dev): chosen={rep.get('chosen')}")
        return
    params = scorer_params(args.contact_scorer, args.min_contact_probability)
    metrics = run(args.out, providers=args.providers, match_ids=match_ids, provenance=prov, params=params)
    if args.reference:
        trig = other_provider_trigger(json.loads(args.reference.read_text(encoding="utf-8")), metrics)
        (args.out / "other_provider_trigger.json").write_text(json.dumps(trig, indent=2), encoding="utf-8")
        print(f"other-provider trigger fires={trig['fires']}")
    o = metrics["overall"]
    print(
        f"ball-contact: contacts/min={o['contact_rate_per_min']:.1f} "
        f"unresolved={o['unresolved_rate']} offset_p50={o['signed_offset_s'].get('p50')}"
    )


if __name__ == "__main__":
    main()
