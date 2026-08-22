"""Task 12: the three-conjunct, per-provider cover-shadow sigma/lambda apply gate (b2).

ADR-009 amendment + ADR-060 prefer-incumbent. The harness RECOMMENDS sigma/lambda + a manifest; this
DELIBERATE, gated apply moves sigma/lambda **only if ALL three conjuncts hold**, else keeps the incumbent
(honest null). The apply is per-provider: on ``applied`` the recommended edit is the GS entry in
``_cover_shadows._PROVIDER_COVER_SHADOW_PARAMS`` (a small committed constant change, never a global default;
H3), which is why every null path leaves the library byte-identical.

A NON-decisive gate is recorded WITH its reason (R1): a conjunct-1 miss reads "unvalidatable where the
model would matter", NOT "no value".
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass

from scripts import _cover_shadow_thresholds as thr
from scripts._provenance import git_provenance, require_clean_tree

APPLY_OUTCOMES = ("applied", "null:unvalidatable", "null:biased", "null:within-noise")


@dataclass(frozen=True)
class ApplyOutcome:
    outcome: str  # one of APPLY_OUTCOMES
    reason: str
    sigma: float | None = None
    lambda_ctrl: float | None = None


def decide_apply(
    *,
    coverage: float,
    receiver_margin: float,
    ablation_share: float,
    noise_ok: bool,
    candidate_sigma: float,
    candidate_lambda: float,
) -> ApplyOutcome:
    """Apply the GS sigma/lambda iff receiver-validity (H1) AND bias (H2) AND noise+effect (ADR-060) hold.

    Thresholds are REFERENCED from ``_cover_shadow_thresholds`` (never inline literals) so the bar can't
    move silently (R5).
    """
    # NON-FINITE inputs = a degenerate / unmeasurable corpus (e.g. 0 intercepted failures -> coverage
    # 0/0 = NaN, or an all-NaN sweep -> NaN candidate params + NaN ablation share). A `NaN < 0.30` reject
    # comparison is False and would silently PASS, yielding a spurious `applied` -- the exact inversion of
    # the honest-null. Route every non-finite input to the safe null instead.
    validity_finite = math.isfinite(coverage) and math.isfinite(receiver_margin)
    if not validity_finite or coverage < thr.MIN_COVERAGE or receiver_margin < thr.MIN_RECEIVER_MARGIN:
        return ApplyOutcome(
            "null:unvalidatable",
            f"receiver-validity failed: coverage {coverage} (>= {thr.MIN_COVERAGE}?) / margin "
            f"{receiver_margin} (>= {thr.MIN_RECEIVER_MARGIN}?) -- unmeasurable on the easy tail, "
            "NOT 'no value' (R1)",
        )
    if not math.isfinite(ablation_share) or ablation_share >= thr.MAX_BIAS_SHARE:
        return ApplyOutcome(
            "null:biased",
            f"bias failed: lane-pressure share {ablation_share} >= {thr.MAX_BIAS_SHARE} (H2/R3)",
        )
    if not noise_ok:
        return ApplyOutcome("null:within-noise", "ADR-060 effect-size / noise floor not cleared")
    if not (math.isfinite(candidate_sigma) and math.isfinite(candidate_lambda)):
        return ApplyOutcome(
            "null:unvalidatable", "no admissible (sigma, lambda) -- the sweep found no finite candidate"
        )
    return ApplyOutcome("applied", "cleared receiver-validity + bias + noise", candidate_sigma, candidate_lambda)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decide the GS cover-shadow sigma/lambda apply (three-conjunct gate).")
    ap.add_argument(
        "--inputs",
        type=pathlib.Path,
        required=True,
        help="JSON: coverage, receiver_margin, ablation_share, noise_ok, candidate_sigma, candidate_lambda",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)
    d = json.loads(args.inputs.read_text(encoding="utf-8"))
    out = decide_apply(
        coverage=d["coverage"],
        receiver_margin=d["receiver_margin"],
        ablation_share=d["ablation_share"],
        noise_ok=bool(d["noise_ok"]),
        candidate_sigma=d["candidate_sigma"],
        candidate_lambda=d["candidate_lambda"],
    )
    manifest = {
        "outcome": out.outcome,
        "reason": out.reason,
        "recommended_gs_params": (
            {"sigma": out.sigma, "lambda_ctrl": out.lambda_ctrl} if out.outcome == "applied" else None
        ),
        "note": "on 'applied', add recommended_gs_params to _cover_shadows._PROVIDER_COVER_SHADOW_PARAMS "
        "as the GS entry (a committed per-provider constant change; never a global default).",
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "apply_decision.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
