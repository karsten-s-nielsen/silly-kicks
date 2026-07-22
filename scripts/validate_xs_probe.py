"""TF-19 PR-3b Part A: run the xS-arm GK-substitution probe end-to-end on GradientSports matches.

Reported-not-gated harness (mirrors scripts/validate_xcross_causal.py). Loads GS matches via the
pining loader, and PER MATCH: builds ghost frames + targets via the gkdv engine and computes the
xS substitution deltas. Pools the tidy DELTAS (not raw frames -- memory), evaluates the probe once,
computes the spec 3.5 re-gate verdict + a targets->used->band reconciliation, and writes
docs/research/tf19_pr3b/{metrics.json, report.md}.

GS-only by construction (the GS-only GKDV measurement rule); the public xS model is GS-free
(skillcorner+idsse), so every GS match is held-out. Requires PINING_FOR_THE_DATA_TOKEN.

Run:
  python scripts/validate_xs_probe.py --out docs/research/tf19_pr3b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path
from typing import cast

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


def _baseline_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return "unknown"


def _fmt(v):  # absent prongs on the unmeasurable early-return branch read oddly as "None"
    return "n/a (unmeasurable)" if v is None else v


def run(out, *, match_ids=None, tracking_limit=None, entanglement="inside_band", seed=42, token=None):
    from _loader_pining import load_matches  # scripts/ on sys.path at runtime (mirrors the trainer)

    ghost_model = GhostGkModel.from_variant("default")  # INSTANCE so build_ghost_frames honors carrier_params
    xs_model = XShotOccurrenceModel.from_variant("default")  # GS-free public weights

    all_deltas, per_match = [], []
    for _provider, match_id, _actions, frames, home_team_id in load_matches(
        providers=_PROVIDERS, match_ids=match_ids, token=token, tracking_limit=tracking_limit
    ):
        htid = cast("int | str", home_team_id)  # loader yields `object`; the engine wants int | str
        _cf, prov, report = build_ghost_frames(frames, model=ghost_model, home_team_id=htid)
        targets = provenance_to_targets(prov, frames=frames, home_team_id=htid)
        # Score per match and keep only the TIDY deltas -> peak memory is one match.
        deltas = substitution_deltas(xs_model, frames, arm="xs", mode="targets", targets=targets, seed=seed)
        all_deltas.append(deltas)
        per_match.append(
            {
                "match_id": match_id,
                "n_frames_in": report.n_frames_in,
                "n_frames_scored": report.n_frames_scored,
                "drop_reasons": report.drop_reasons,
                "n_targets": len(targets),
            }
        )

    if not all_deltas:
        raise SystemExit("no GS matches loaded — check PINING_FOR_THE_DATA_TOKEN / --match-ids-json")

    deltas_pooled = pd.concat(all_deltas, ignore_index=True)  # distinct game_id per match -> dose clusters + MIN_GAMES
    # Distinct game_id per match is a LOAD-BEARING premise (dose-response groups by game_id + needs
    # MIN_GAMES=8; the evaluator's duplicate-key guard raises on shared (game_id,period_id,frame_id)).
    # GS ids are 1:1 with native_match_id, so this is insurance -- guard the premise, don't assert it.
    n_games = int(deltas_pooled["game_id"].nunique())
    n_contributing = sum(1 for m in per_match if m["n_targets"] > 0)
    if n_games < n_contributing:
        warnings.warn(
            f"game_id collision: {n_games} distinct games from {n_contributing} contributing matches "
            "— dose-response will undercount games and the duplicate-key guard may raise",
            stacklevel=2,
        )

    result = evaluate_xs_probe(deltas_pooled)  # the exact evaluator xs_substitution_probe wraps
    result["n_frames_used"] = _n_unique_gk_frames(deltas_pooled)  # re-add what the wrapper computes

    total_targets = sum(m["n_targets"] for m in per_match)
    used = result["n_frames_used"]
    reconciliation = {  # make silent band-shrink visible; name the distinct-game premise
        "total_targets": total_targets,
        "n_frames_used": used,
        "n_distinct_games": n_games,
        "gated_band_n": result.get("gated_band_n"),
        # NB: a targets->used drop is non-zero BY CONSTRUCTION -- the ghost engine resolves the ball
        # carrier with the GHOST model's carrier_params while the probe's _eligible_groups uses the XS
        # model's, with no passthrough to align them. Read the >0.5 flag as "larger than that baseline."
        "targets_to_used_drop_frac": (1.0 - used / total_targets) if total_targets else None,
    }

    metrics = {
        "arm": "xs",
        "probe": result,
        "regate_verdict": re_gate(result["verdict"], entanglement),
        "entanglement": entanglement,  # inert unless the probe surprises with `pass`
        "reconciliation": reconciliation,
        "rule_constants": PROBE_WRAPPERS["xs"]["rule_constants"],
        "corpus": {
            "providers": _PROVIDERS,
            "n_matches": len(per_match),
            "match_ids": [m["match_id"] for m in per_match],
        },
        "per_match": per_match,
        "seed": seed,
        "tracking_limit": tracking_limit,
        "rng_discipline": "per-match placebo streams (substitution_deltas per match, seed pinned)",
        "baseline_commit": _baseline_commit(),
    }
    _write(out, metrics)
    return metrics


def _dose_ladder_line(p: dict) -> str:
    """The dose-response ladder IS the effect — surface it even when the prongs are omitted."""
    dl = p.get("dose_ladder")
    if not dl:
        return "- dose ladder: n/a (unmeasurable)"
    parts = "   ".join(f"{float(k):.0f} m: {float(v):.4f}" for k, v in dl.items())
    return f"- dose ladder (median |ΔxS| by ghost displacement): {parts}"


def _dose_ratio_line(p: dict) -> str:
    """Effect-vs-control at the 2 m dose — the number the ratio prong WOULD have used."""
    dl, nd = p.get("dose_ladder"), p.get("nearest_def_median")
    if not dl or not nd:  # `not nd` also guards a zero control (no division)
        return "- effect vs control ratio (2 m / nearest-def): n/a (unmeasurable)"
    d2 = dl.get("2.0") if dl.get("2.0") is not None else next(iter(dl.values()))
    return f"- effect vs control ratio (2 m / nearest-def): {float(d2) / float(nd):.2f}x"


def _render(m: dict) -> str:
    p, rec, rc = m["probe"], m["reconciliation"], m["rule_constants"]
    verdict = p["verdict"]
    drop = rec["targets_to_used_drop_frac"]
    drop_flag = (
        "  <-- larger than the baseline carrier-resolver mismatch; investigate"
        if (drop or 0) > 0.5
        else "  (baseline structural drop; see note)"
    )
    prongs_omitted = p.get("gated_band_median") is None
    return (
        "\n".join(
            [
                "# TF-19 PR-3b — xS-arm GK-substitution probe",
                "",
                f"**Probe verdict:** `{verdict}`",
                f"**Re-gate (arm=shot, entanglement={m['entanglement']}):** `{m['regate_verdict']}`",
                f"**Frames used:** {p.get('n_frames_used')}   **Matches:** {m['corpus']['n_matches']}   "
                f"**Games:** {rec.get('n_distinct_games')}   **seed:** {m['seed']}",
                "",
                "## Which branch? (read first — the ghost-accuracy paradox)",
                f"- gated_band_n: {p.get('gated_band_n')}  (needs >= {rc.get('min_band_n')} to be measurable)",
                "- `unmeasurable_at_dose`: band under-filled (couldn't reach 2 m; NOT a null effect)",
                "- `no_valid_placebo`: band fills but the random-outfielder CONTROL can't be certified "
                "(re-gates to `unmeasurable_at_dose`; see Placebo below; NOT a null effect)",
                "- `gated_clean_fail`: band fills, GK flat (closes the arm; names the GK-feature lever)",
                "",
                "## The effect (does the keeper's position move xS at all?)",
                _dose_ladder_line(p),
                f"- unbanded median |ΔxS|: {_fmt(p.get('unbanded_median'))}   "
                f"nearest-defender control: {_fmt(p.get('nearest_def_median'))}",
                _dose_ratio_line(p),
                "",
                "## Placebo (random-outfielder control — the certification blocker)",
                f"- placebo_p95: {_fmt(p.get('placebo_p95'))}   "
                f"placebo_zero_fraction: {_fmt(p.get('placebo_zero_fraction'))}",
                "- an all-zero placebo_p95 + a high zero-fraction => the aggregate xS features barely respond to "
                "one distant player moving 2 m => the control is degenerate => `no_valid_placebo` (prongs not run).",
                "",
                "## Targets -> used -> band reconciliation",
                f"- total targets: {rec['total_targets']}   n_frames_used: {rec['n_frames_used']}   "
                f"distinct games: {rec.get('n_distinct_games')}   gated_band_n: {rec['gated_band_n']}",
                f"- targets->used drop frac: {drop}{drop_flag}",
                "- note: a drop is EXPECTED — the ghost engine resolves the carrier with the ghost model's "
                "carrier_params, the probe with the xs model's; the flag fires only ABOVE that baseline.",
                "",
                "## Rule prongs" + (f"  (omitted — {verdict})" if prongs_omitted else ""),
                f"- gated_band_median: {_fmt(p.get('gated_band_median'))}",
                f"- nearest_def_median: {_fmt(p.get('nearest_def_median'))}",
                f"- placebo_p95: {_fmt(p.get('placebo_p95'))}",
                f"- ratio rule: gated_band_median >= {rc['ratio']} * max(nearest_def_median, placebo_p95)",
                "- absolute floor (TF19): gated_band_median vs 0.01",
                f"- dose_response rho / p: {_fmt(p.get('dose_response_rho'))} / {_fmt(p.get('dose_response_p'))}",
                "",
            ]
        )
        + "\n"
    )


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
        help="banked shot-arm causal result (docs/research/tf19_causal/xshot/); inert unless probe=pass",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    match_ids = json.loads(args.match_ids_json.read_text(encoding="utf-8")) if args.match_ids_json else None
    m = run(
        args.out,
        match_ids=match_ids,
        tracking_limit=args.tracking_limit,
        entanglement=args.entanglement,
        seed=args.seed,
    )
    print(
        f"verdict={m['probe']['verdict']}  regate={m['regate_verdict']}  "
        f"matches={m['corpus']['n_matches']}  frames_used={m['probe'].get('n_frames_used')}  "
        f"targets_drop={m['reconciliation']['targets_to_used_drop_frac']}"
    )


if __name__ == "__main__":
    main()
