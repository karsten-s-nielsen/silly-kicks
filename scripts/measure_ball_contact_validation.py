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

import numpy as np
import pandas as pd

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


def run(out, *, providers=None, match_ids=None, provenance=None, cache_dir=None, source=None) -> dict:
    """Validate ball-contact detection and write the memo. ``source=(refs, load)`` injects a loader
    for tests (no network). ``run`` is called directly by tests; enforcement lives in ``main``."""
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
        rows, tally = _evaluate_match(actions, frames)
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
        token_inputs={"tolerances_s": list(TOLERANCES_S), "detector": "rdp+velocity_break+dv", "geometry": "frame_ltr"},
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
    args = parser.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    match_ids = json.loads(args.match_ids_json.read_text(encoding="utf-8")) if args.match_ids_json else None
    metrics = run(
        args.out,
        providers=args.providers,
        match_ids=match_ids,
        provenance=require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty),
    )
    o = metrics["overall"]
    print(
        f"ball-contact: contacts/min={o['contact_rate_per_min']:.1f} "
        f"unresolved={o['unresolved_rate']} offset_p50={o['signed_offset_s'].get('p50')}"
    )


if __name__ == "__main__":
    main()
