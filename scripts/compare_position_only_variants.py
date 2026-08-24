"""Comparability report: velocity vs position-only variant skill delta (Task 8 / spec D6).

**REPORTED, not gated.** The position-only variant is EXPECTED to be weaker -- it drops informative
velocity features -- so this quantifies and DISCLOSES the cost rather than gating on it. It informs
the Risk-1 public-vs-full bundled-ghost ship decision (commit 2); it is not a CI gate.

Reads each variant's ``metrics.json`` (produced by the trainers) and writes a delta table +
provenance to ``docs/research/position_only_variants/``. Provenance-wired per ADR-037 (clean tree,
``run_commit``); no corpus work.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from scripts._provenance import git_provenance, require_clean_tree

#: Held-out metrics compared per model. xShot/xCross are classifiers (PR-AUC / Brier / log-loss,
#: higher PR-AUC + lower Brier/log-loss = better); ghost is a regressor (MAE, lower = better).
_MODEL_KEYS: dict[str, list[str]] = {
    "xshot": ["pr_auc", "brier", "log_loss"],
    "xcross": ["pr_auc", "brier", "log_loss"],
    "ghost": ["overall_mae", "per_provider_mae_max", "cross_fold_std"],
}


def compute_skill_delta(velocity: dict, position_only: dict, keys: list[str]) -> dict:
    """Per-metric ``{velocity, position_only, delta}`` over ``keys`` (pure, no IO).

    ``delta = velocity - position_only``. A key absent from -- or non-numeric on -- either side yields
    ``delta=None`` (REPORTED, never fabricated as 0 -- an unmeasured delta is not a zero delta).
    """
    out: dict[str, dict] = {}
    for k in keys:
        v = velocity.get(k)
        p = position_only.get(k)
        # bool is an int subtype, so exclude it FIRST; the elif narrows both operands to int|float
        # for the subtraction (a separate `both_numeric` flag does not narrow for the type checker).
        if isinstance(v, bool) or isinstance(p, bool):
            delta = None
        elif isinstance(v, (int, float)) and isinstance(p, (int, float)):
            delta = v - p
        else:
            delta = None
        out[k] = {"velocity": v, "position_only": p, "delta": delta}
    return out


def _load_metrics(path: pathlib.Path) -> dict:
    """Flatten a trainer ``metrics.json`` (the held-out numbers nest under a model-specific key)."""
    m = json.loads(path.read_text(encoding="utf-8"))
    flat = dict(m)
    for sub in ("cv", "acceptance", "metrics", "held_out"):
        if isinstance(m.get(sub), dict):
            flat.update(m[sub])
    return flat


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(_MODEL_KEYS), required=True)
    ap.add_argument("--velocity-metrics", type=pathlib.Path, required=True, help="faithful variant metrics.json")
    ap.add_argument("--position-only-metrics", type=pathlib.Path, required=True, help="position_only metrics.json")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/research/position_only_variants"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    # ADR-037: refuse a dirty tree BEFORE writing an artifact; stamp run_commit/run_tree_dirty.
    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    delta = compute_skill_delta(
        _load_metrics(args.velocity_metrics),
        _load_metrics(args.position_only_metrics),
        _MODEL_KEYS[args.model],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}_velocity_vs_position_only.json"
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "delta": delta,
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "run_tree_state": prov["tree_state"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
