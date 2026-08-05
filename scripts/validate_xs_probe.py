"""TF-19 PR-3b: run the xS-arm GK-substitution probe end-to-end on GradientSports matches, for the
v1 (random-outfielder) and/or v2 (model-relevant-defender, ADR-037 amendment) placebo variants.

Reported-not-gated harness (mirrors scripts/validate_xcross_causal.py). Loads GS matches via the
pining loader, and PER MATCH: builds ghost frames + targets via the gkdv engine and computes the
xS substitution deltas ONCE PER VARIANT (each variant's placebo_out population in its own frame so
evaluate_xs_probe never sees both; spec S5). Pools the tidy DELTAS (not raw frames -- memory),
evaluates each variant, computes the spec 3.5 re-gate verdict + a targets->used->band
reconciliation + the v2 non-gating attacker diagnostic, and writes {metrics.json, report.md}.

GS-only by construction (the GS-only GKDV measurement rule); the public xS model is GS-free
(skillcorner+idsse), so every GS match is held-out. Requires PINING_FOR_THE_DATA_TOKEN.

Run (v2 deliverable, records the blindness lock-commit hash):
  python scripts/validate_xs_probe.py --out docs/research/tf19_pr3b_xs_v2 --variant both --lock-commit <sha>
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from silly_kicks.gkdv import build_ghost_frames, provenance_to_targets
from silly_kicks.tracking._ghost_gk import GhostGkModel
from silly_kicks.tracking._model_eval import (
    PROBE_WRAPPERS,
    evaluate_xs_probe,
    regate_verdict,
    substitution_deltas,
)
from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

_PROVIDERS = ["gradientsports"]  # GS-only GKDV measurement rule (NOT a CLI knob)


def re_gate(probe_verdict: str, entanglement: str) -> str:
    """Map the xS probe verdict to the spec 3.5 re-gate outcome (arm='shot')."""
    return regate_verdict(arm="shot", probe_verdict=probe_verdict, entanglement=entanglement)


def _n_unique_gk_frames(deltas: pd.DataFrame) -> int:
    gk = deltas[deltas["actor_role"] == "gk"]
    return len(gk[["game_id", "period_id", "frame_id"]].drop_duplicates())


def _attacker_diag_p95(deltas: pd.DataFrame) -> float:
    """Non-gating attacker-diagnostic p95 (95th pct of per-replicate medians); NaN when absent (v1)."""
    sub = deltas[deltas["actor_role"] == "attacker_diag"]
    rep_med = sub.groupby("replicate")["delta_p"].median()
    return float(np.percentile(rep_med, 95.0)) if len(rep_med) else float("nan")


_VARIANT_PLACEBO = {"v1": "random", "v2": "model_relevant_def"}
_VARIANT_WRAPPER = {"v1": "xs", "v2": "xs_v2"}


def _fmt(v):  # absent prongs on the unmeasurable early-return branch read oddly as "None"
    return "n/a (unmeasurable)" if v is None else v


#: Joins a match id to a drop reason inside a FLAT counter dict. `_driver._merge_counters` sums ints
#: and merges `{str: int}` dicts; it cannot carry a dict OF dicts, so the per-match breakdown is
#: encoded on a composite key and split back out on read.
_BY_MATCH_SEP = "::"


def _flatten_by_match(match_id, reasons) -> dict[str, int]:
    """`{reason: n}` for one match -> `{"<match_id>::<reason>": n}`, refusing an ambiguous component.

    Validated rather than trusted, for the same reason `_driver.join_key` validates its components:
    a component containing the separator makes the split ambiguous, and the resulting mis-attributed
    counts would be silent -- a plausible per-match breakdown assembled from the wrong rows.
    """
    out: dict[str, int] = {}
    for reason, n in dict(reasons).items():
        if _BY_MATCH_SEP in str(match_id) or _BY_MATCH_SEP in str(reason):
            raise ValueError(
                f"{_BY_MATCH_SEP!r} appears in match_id={match_id!r} or reason={reason!r}; the "
                f"per-match counter key would split ambiguously and mis-attribute drop reasons"
            )
        out[f"{match_id}{_BY_MATCH_SEP}{reason}"] = int(n)
    return out


def _per_match_records(counters: dict) -> list[dict]:
    """Rebuild `metrics["per_match"]` from the pass's counters.

    Sourced from the counters rather than from the combined frame because a match that produced NO
    deltas still has a record -- `n_targets == 0` is exactly what `n_contributing` counts -- and an
    empty shard carries no columns to recover it from.
    """
    by_match: dict[str, dict] = {}
    for key, n in counters.get("drop_reasons_by_match", {}).items():
        mid, _, reason = key.partition(_BY_MATCH_SEP)
        by_match.setdefault(mid, {})[reason] = n
    targets = counters.get("n_targets_by_match", {})
    frames_in = counters.get("n_frames_in_by_match", {})
    frames_scored = counters.get("n_frames_scored_by_match", {})
    return [
        {
            "match_id": mid,
            "n_frames_in": frames_in.get(mid),
            "n_frames_scored": frames_scored.get(mid),
            "drop_reasons": by_match.get(mid, {}),
            "n_targets": targets[mid],
        }
        # Sorted so the artifact is stable across a resume, where the shard order is the KEY order
        # rather than the order the loader happened to yield matches in on the first pass.
        for mid in sorted(targets)
    ]


def run(
    out,
    *,
    variant="both",
    match_ids=None,
    tracking_limit=None,
    entanglement="inside_band",
    seed=42,
    token=None,
    lock_commit=None,
    provenance=None,
):
    # ENFORCEMENT lives in main(); this records the truth. `run` is called directly by tests, and a
    # work function that refuses to execute on a dirty checkout cannot be tested without mocking
    # git -- so the CLI refuses and this always stamps what actually ran.
    from scripts._provenance import git_provenance

    prov_run = provenance or git_provenance()

    from _loader_pining import load_matches  # scripts/ on sys.path at runtime (mirrors the trainer)

    variants = ["v1", "v2"] if variant == "both" else [variant]
    ghost_model = GhostGkModel.from_variant("default")  # INSTANCE so build_ghost_frames honors carrier_params
    xs_model = XShotOccurrenceModel.from_variant("default")  # GS-free public weights

    from scripts._driver import for_each, shard_path

    # Per-match metadata that the tidy delta frame CANNOT carry: it comes from `build_ghost_frames`'s
    # report and from `targets`, neither of which is a column. `for_each` calls `work(item)` and then
    # `counters(item, frame)` for the SAME item, in that order, so stashing it here is well-defined.
    # Since 4.72.0 `for_each` persists each item's counters beside its shard and replays them on a
    # skip, so this metadata survives a resume -- without that, a resumed pass would report a corpus
    # of zero matches into a cited artifact.
    _meta: dict = {}

    def _work(item):
        _provider, match_id, _actions, frames, home_team_id = item
        htid = cast("int | str", home_team_id)  # loader yields `object`; the engine wants int | str
        _cf, prov, report = build_ghost_frames(frames, model=ghost_model, home_team_id=htid)
        targets = provenance_to_targets(prov, frames=frames, home_team_id=htid)
        # One full substitution_deltas per variant per match (spec S5): recomputing the pool-independent
        # gk/nearest_def rows is ~13% redundant but keeps each variant's placebo_out population in its OWN
        # frame, so evaluate_xs_probe never sees v1's random AND v2's defender placebo together. Keep only
        # the TIDY deltas -> peak memory is one match.
        parts = [
            substitution_deltas(
                xs_model,
                frames,
                arm="xs",
                mode="targets",
                targets=targets,
                seed=seed,
                placebo=_VARIANT_PLACEBO[v],
            ).assign(variant=v)
            for v in variants
        ]
        _meta.clear()
        _meta.update(
            {
                "n_matches": 1,
                # `_merge_counters` sums ints and merges FLAT {str: int} dicts -- it cannot carry a
                # dict of dicts, so the per-match breakdown is keyed on a composite. Preserved rather
                # than aggregated corpus-wide because `metrics["per_match"]` is a published field of
                # a cited artifact, and quietly coarsening it is a schema break wearing a refactor's
                # clothes.
                "n_frames_in_by_match": {str(match_id): report.n_frames_in},
                "n_frames_scored_by_match": {str(match_id): report.n_frames_scored},
                "n_targets_by_match": {str(match_id): len(targets)},
                "drop_reasons_by_match": _flatten_by_match(match_id, report.drop_reasons),
            }
        )
        return pd.concat(parts, ignore_index=True) if parts else None

    res = for_each(
        load_matches(providers=_PROVIDERS, match_ids=match_ids, token=token, tracking_limit=tracking_limit),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=lambda _item, _frame: dict(_meta),
        shard_root=Path(out) / "shards",
        # What determines a shard's CONTENT: the two models whose substitution produces the deltas,
        # the placebo pool each variant draws, the seed pinning those draws, and the frame cap the
        # engine sees. `--match-ids-json` only chooses WHICH matches are walked (the key separates
        # them), and `--entanglement` is consumed downstream by the verdict, not by the deltas.
        token_inputs={
            "ghost_model": "default",
            "xs_model": "default",
            "variants": list(variants),
            "placebos": [_VARIANT_PLACEBO[v] for v in variants],
            "seed": seed,
            "tracking_limit": tracking_limit,
        },
        tag="xs_probe",
        label="match",
    )
    if res.failures:
        # Loud, and AFTER every successful shard is on disk. This is the 14-hour driver: a re-run
        # retries only the failures, so refusing costs minutes rather than the corpus.
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    per_match = _per_match_records(res.counters)
    if not per_match:
        raise SystemExit("no GS matches loaded -- check PINING_FOR_THE_DATA_TOKEN / --match-ids-json")

    # Combined from THIS PASS'S keys, not `_driver.reconcile`: that helper's whole-generation read
    # requires a partition surface, and `--match-ids-json` here is a reproducibility PIN, not a
    # worker slice (no `providers_for_slice`, no `worker_tag`). A whole-generation read would fold
    # in matches from a wider earlier run over the same --out.
    shards = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys]
    combined = pd.concat([f for f in shards if len(f)], ignore_index=True)
    per_variant_deltas = {
        v: [combined[combined["variant"] == v].drop(columns="variant").reset_index(drop=True)] for v in variants
    }

    # Distinct game_id per match is a LOAD-BEARING premise (dose-response groups by game_id + needs
    # MIN_GAMES=8; the evaluator's duplicate-key guard raises on shared (game_id,period_id,frame_id)).
    # GS ids are 1:1 with native_match_id, so this is insurance -- guard the premise, don't assert it.
    n_games = int(pd.concat(per_variant_deltas[variants[0]], ignore_index=True)["game_id"].nunique())
    n_contributing = sum(1 for m in per_match if m["n_targets"] > 0)
    if n_games < n_contributing:
        warnings.warn(
            f"game_id collision: {n_games} distinct games from {n_contributing} contributing matches "
            "-- dose-response will undercount games and the duplicate-key guard may raise",
            stacklevel=2,
        )

    results = {}
    for v in variants:
        pooled = pd.concat(per_variant_deltas[v], ignore_index=True)
        res = evaluate_xs_probe(pooled)  # the exact evaluator the wrappers wrap
        res["n_frames_used"] = _n_unique_gk_frames(pooled)  # re-add what the wrapper computes
        if v == "v2":
            res["rule"] = "xs-dose-banded-v2"
            res["placebo_pool"] = "model_relevant_def"
            res["attacker_diag_p95"] = _attacker_diag_p95(pooled)
        results[v] = {
            "probe": res,
            "regate_verdict": re_gate(res["verdict"], entanglement),
            "rule_constants": PROBE_WRAPPERS[_VARIANT_WRAPPER[v]]["rule_constants"],
        }

    ref = results[variants[0]]["probe"]  # gk stratum is pool-independent -> n_frames_used identical
    total_targets = sum(m["n_targets"] for m in per_match)
    used = ref["n_frames_used"]
    reconciliation = {  # make silent band-shrink visible; name the distinct-game premise
        "total_targets": total_targets,
        "n_frames_used": used,
        "n_distinct_games": n_games,
        "gated_band_n": ref.get("gated_band_n"),
        # NB: a targets->used drop is non-zero BY CONSTRUCTION -- the ghost engine resolves the ball
        # carrier with the GHOST model's carrier_params while the probe's _eligible_groups uses the XS
        # model's, with no passthrough to align them. Read the >0.5 flag as "larger than that baseline."
        "targets_to_used_drop_frac": (1.0 - used / total_targets) if total_targets else None,
    }

    metrics = {
        "arm": "xs",
        "variants": results,
        # NOT inert: this was written expecting a `fail`, and the v2 probe returned `pass` -- at
        # which point regate_verdict consults it and it DECIDES joins vs joins_with_caveat. The
        # default value is a carry-forward from the cross arm's registration, so a run that does not
        # pass --entanglement is reporting an UNMEASURED input for this arm.
        "entanglement": entanglement,
        "reconciliation": reconciliation,
        "corpus": {
            "providers": _PROVIDERS,
            "n_matches": len(per_match),
            "match_ids": [m["match_id"] for m in per_match],
        },
        "per_match": per_match,
        "seed": seed,
        "tracking_limit": tracking_limit,
        "rng_discipline": "per-match placebo streams (substitution_deltas per match+variant, seed pinned)",
        "lock_commit": lock_commit or prov_run["commit"],
        "run_commit": prov_run["commit"],
        # The lock commit says what the protocol was REGISTERED against; this says whether the tree
        # that actually ran matched it. Without the flag the two are indistinguishable in the record.
        "run_tree_dirty": prov_run["dirty"],
        "run_tree_state": prov_run["tree_state"],
    }
    _write(out, metrics)
    return metrics


def _dose_ladder_line(p: dict) -> str:
    """The dose-response ladder IS the effect -- surface it even when the prongs are omitted."""
    dl = p.get("dose_ladder")
    if not dl:
        return "- dose ladder: n/a (unmeasurable)"
    parts = "   ".join(f"{float(k):.0f} m: {float(v):.4f}" for k, v in dl.items())
    return f"- dose ladder (median |deltaxS| by ghost displacement): {parts}"


def _dose_ratio_line(p: dict) -> str:
    """Effect-vs-control at the 2 m dose -- the number the ratio prong WOULD have used."""
    dl, nd = p.get("dose_ladder"), p.get("nearest_def_median")
    if not dl or not nd:  # `not nd` also guards a zero control (no division)
        return "- effect vs control ratio (2 m / nearest-def): n/a (unmeasurable)"
    d2 = dl.get("2.0") if dl.get("2.0") is not None else next(iter(dl.values()))
    return f"- effect vs control ratio (2 m / nearest-def): {float(d2) / float(nd):.2f}x"


def _variant_block(name: str, entry: dict, rc: dict) -> list[str]:
    p = entry["probe"]
    prongs_omitted = p.get("gated_band_median") is None
    lines = [
        f"### {name}: `{p['verdict']}`   re-gate: `{entry['regate_verdict']}`"
        + (f"   ({p['placebo_pool']} placebo)" if p.get("placebo_pool") else "   (random placebo)"),
        f"- gated_band_n: {p.get('gated_band_n')} (needs >= {rc.get('min_band_n')})   "
        f"frames_used: {p.get('n_frames_used')}",
        _dose_ladder_line(p),
        f"- nearest_def control: {_fmt(p.get('nearest_def_median'))}   "
        f"placebo_p95: {_fmt(p.get('placebo_p95'))}   gated_band_median: {_fmt(p.get('gated_band_median'))}",
        _dose_ratio_line(p),
        f"- dose_response rho / p: {_fmt(p.get('dose_response_rho'))} / {_fmt(p.get('dose_response_p'))}"
        + ("   (prongs omitted -- unmeasurable)" if prongs_omitted else ""),
    ]
    if p.get("attacker_diag_p95") is not None:
        lines.append(f"- attacker diagnostic p95 (non-gating): {_fmt(p.get('attacker_diag_p95'))}")
    return lines


def _render(m: dict) -> str:
    rec = m["reconciliation"]
    variants = m["variants"]
    body = [
        "# TF-19 PR-3b xS-arm probe -- v1 (random) vs v2 (model-relevant defenders)",
        "",
        f"**Entanglement:** {m['entanglement']}   **seed:** {m['seed']}   "
        f"**Matches:** {m['corpus']['n_matches']}   **Games:** {rec.get('n_distinct_games')}",
        f"**Lock commit:** `{m.get('lock_commit')}`   **Run commit:** `{m.get('run_commit')}`   "
        "(blindness: constants locked before the run; verify any intervening diff is inert)",
        "",
    ]
    if "v2" in variants:  # the honest framing is ABOUT v2 -> only print it on a run that has v2
        body += [
            "## The honest framing",
            "- v2 changes EXACTLY ONE thing vs v1: the placebo pool (random outfielder -> ball-nearest "
            "defenders). The defender placebo is a WEAKER control than nearest_def, so it is INERT in the "
            "ratio (`max()` pins to nearest_def); its job is to clear the no_valid_placebo gate with a "
            "principled null, not to move the bar.",
            "- The ratio prong is therefore a 'beat nearest_def by 2x' test, near-certain to pass. v2's REAL "
            "decider is the clustered dose-response permutation, which v1 never reached.",
            "- The attacker diagnostic is reported (non-gating): the nearest attacker is the shooter, so "
            "gating on attackers would answer a model-sensitivity question, not a deterrence one.",
            "",
        ]
    for name in ("v1", "v2"):
        if name in variants:
            body += ["## " + ("v1 (frozen random placebo)" if name == "v1" else "v2 (relevance-matched)"), ""]
            body += _variant_block(name, variants[name], variants[name]["rule_constants"])
            body += [""]
    body += [
        "## Targets -> used -> band reconciliation",
        f"- total targets: {rec['total_targets']}   n_frames_used: {rec['n_frames_used']}   "
        f"distinct games: {rec.get('n_distinct_games')}   gated_band_n: {rec['gated_band_n']}",
        f"- targets->used drop frac: {rec['targets_to_used_drop_frac']} "
        "(a drop is EXPECTED -- ghost vs xs carrier-resolver mismatch; read as 'above that baseline').",
        "",
    ]
    return "\n".join(body) + "\n"


def _write(out, metrics) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(_render(metrics), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-19 PR-3b: run the xS-arm GK-substitution probe on GS matches.")
    ap.add_argument("--out", type=Path, required=True, help="output dir (convention: docs/research/tf19_pr3b)")
    ap.add_argument(
        "--match-ids-json",
        type=Path,
        default=None,
        help='JSON {"gradientsports": [ids]} to pin the corpus for reproducibility',
    )
    ap.add_argument("--tracking-limit", type=int, default=None, help="per-match frame cap (dev-smoke only; None=full)")
    ap.add_argument(
        "--entanglement",
        default="inside_band",
        help=(
            "shot-arm GK-confounder entanglement, from scripts/validate_xshot_causal.py "
            "(docs/research/tf19_causal/xshot/). NOT inert: regate_verdict consults this whenever "
            "the probe verdict is `pass`, which the xS v2 run returned -- so it DECIDES "
            "joins vs joins_with_caveat. The default is a carry-forward from the cross arm's "
            "ADR-037 registration, not a measurement of this arm."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--variant",
        choices=["v1", "v2", "both"],
        default="both",
        help="which placebo variant(s) to run (default: both, side by side)",
    )
    ap.add_argument(
        "--lock-commit",
        default=None,
        help="the commit that froze the v2 pool+constants (auditable blindness; defaults to HEAD). "
        "Record it so the git DAG shows constants-locked-before-run.",
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    args = ap.parse_args()

    # Provenance FIRST -- before the corpus pass `run` starts, so a dirty tree is refused in
    # seconds rather than surfacing as an unfalsifiable SHA in a cited metrics.json.
    from scripts._provenance import git_provenance, require_clean_tree

    match_ids = json.loads(args.match_ids_json.read_text(encoding="utf-8")) if args.match_ids_json else None
    m = run(
        args.out,
        variant=args.variant,
        match_ids=match_ids,
        tracking_limit=args.tracking_limit,
        entanglement=args.entanglement,
        seed=args.seed,
        lock_commit=args.lock_commit,
        provenance=require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty),
    )
    v2 = m["variants"].get("v2") or next(iter(m["variants"].values()))
    v1 = m["variants"].get("v1", {}).get("probe", {})
    print(
        f"v1={v1.get('verdict')}  v2={v2['probe']['verdict']}  regate_v2={v2['regate_verdict']}  "
        f"matches={m['corpus']['n_matches']}  lock={m.get('lock_commit')}"
    )


if __name__ == "__main__":
    main()
