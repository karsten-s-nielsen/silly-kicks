#!/usr/bin/env python
"""Measure the feature-contract fingerprint on THIS platform, and compare two platforms.

Why this exists
---------------
``_feature_contract._CONTRACT_ATOL = 1e-6`` / ``_CONTRACT_RTOL = 0.0`` is the tolerance on the
fail-closed guard for **every** bundled model (ADR-050), and it was CHOSEN, never measured. It becomes
load-bearing the moment a non-x86 process loads an x86-stamped artifact and recomputes the fingerprint
locally. A hand-run measurement previously covered 27 of the 69 bundled features and carried no
provenance at all -- so it could not be cited by the rule it was supposed to support.

One machine cannot measure a cross-platform delta, so this driver does exactly one platform per
invocation and a separate ``--compare`` joins two self-provenanced legs.

Scope is an AND, not an OR
--------------------------
All three bundled contracts are probed: ghost-GK (26 features), xShot (27), xCross (16). Ghost and
xShot are already empirically aarch64-clean -- ``validate_xs_probe.py`` constructs both, routing
through ``load()`` -> ``verify_feature_contract``, and completed a 64-match DGX run. **Nothing
otherwise loads ``XCrossAttemptModel`` on aarch64**: ``validate_xcross_causal.py`` reads
``metadata.json`` directly and never constructs the model, while ``tracking/features.py`` loads it via
``from_variant("default")`` for ``xcross_attempt_xfns``. It is the one contract-bearing artifact never
loaded on aarch64 and it is reachable from a live public path, so excluding it would leave the only
genuinely unverified surface unmeasured.

A tolerance caveat that ``--compare`` records rather than hides
--------------------------------------------------------------
``atol=1e-6`` cannot transfer to the xCross vector even in principle: ``space_controlled`` is
``cell_count / 805 * 7140``, quantized at **8.8696 m^2 per cell -- about 8.87e6 x atol** -- so its
cross-platform error is exactly ``0.0`` or ``>= 8.87`` and the tolerance degenerates to an equality
test. ``box_off_def_ratio`` is likewise an integer ratio. An argmin flip is unlikely (300 random
22-player scenes gave a minimum relative first-vs-second gap of 6.80e-6, none below 1e-12), which is
precisely why measuring is cheap and worth having on the record.

Usage
-----
    python scripts/measure_platform_probe.py --out "$HOME/pr5_runs/platform_probe"       # each machine
    python scripts/measure_platform_probe.py --out DIR --compare A.json B.json           # once, after

Writes ``<machine>-<system>-py<major.minor>.json`` per leg, then ``metrics.json`` on compare.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
from _provenance import git_provenance, require_clean_tree

#: (label, importable module, n features the bundled contract declares)
_TARGETS = (
    ("ghost_gk", "silly_kicks.tracking._ghost_gk"),
    ("xshot", "silly_kicks.tracking._xshot_occurrence"),
    ("xcross", "silly_kicks.tracking._xcross_attempt"),
)

#: Features whose value is QUANTIZED, so `atol` degenerates to an equality test on them. Recorded per
#: leg rather than inferred at compare time, because the list is a property of the extractor.
_QUANTIZED = {"xcross": ["space_controlled", "box_off_def_ratio"]}


def _leg_filename() -> str:
    v = sys.version_info
    return f"{platform.machine()}-{platform.system()}-py{v.major}.{v.minor}.json".lower()


def measure_leg() -> dict:
    """Recompute every bundled contract block on this interpreter."""
    import importlib

    from silly_kicks.tracking import _geometry as _geo

    blocks: dict[str, dict] = {}
    for label, modname in _TARGETS:
        mod = importlib.import_module(modname)
        block = mod._feature_contract_block()
        blocks[label] = {
            "probe_sha256": block["probe_sha256"],
            "fingerprint": block["fingerprint"],
            "constants": block["constants"],
            "n_features": len(block["fingerprint"]),
            "quantized_features": _QUANTIZED.get(label, []),
        }

    prov = git_provenance()
    return {
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "geometry_version": _geo.GEOMETRY_VERSION,
        "blocks": blocks,
        "n_features_total": sum(b["n_features"] for b in blocks.values()),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }


def compare(a: dict, b: dict) -> dict:
    """Join two legs, REFUSING any pair whose difference could be code rather than platform."""
    refusals: list[str] = []
    for field in ("run_commit", "geometry_version"):
        if a.get(field) != b.get(field):
            refusals.append(
                f"legs disagree on {field}: {a.get(field)!r} vs {b.get(field)!r} -- the delta would "
                f"confound platform with CODE, which is the one thing this artifact exists to separate"
            )
    for leg, name in ((a, "a"), (b, "b")):
        if leg.get("run_tree_dirty") is not False:
            refusals.append(f"leg {name} has run_tree_dirty={leg.get('run_tree_dirty')!r}; refusing")
    for label, _ in _TARGETS:
        pa, pb = a["blocks"].get(label, {}), b["blocks"].get(label, {})
        if pa.get("probe_sha256") != pb.get("probe_sha256"):
            refusals.append(f"{label}: probe identity differs between legs")

    deltas: dict[str, dict] = {}
    if not refusals:
        for label, _ in _TARGETS:
            fa = np.asarray(a["blocks"][label]["fingerprint"], float)
            fb = np.asarray(b["blocks"][label]["fingerprint"], float)
            d = np.abs(fa - fb)
            deltas[label] = {
                "n_features": int(fa.size),
                "max_abs_delta": float(np.max(d)) if d.size else 0.0,
                "n_features_moved": int(np.count_nonzero(d > 0.0)),
                "quantized_features": a["blocks"][label]["quantized_features"],
            }

    worst = max((v["max_abs_delta"] for v in deltas.values()), default=float("nan"))
    from silly_kicks.tracking._feature_contract import _CONTRACT_ATOL, _CONTRACT_RTOL

    return {
        "legs": {"a": a["platform"], "b": b["platform"]},
        "run_commit": a.get("run_commit"),
        "run_tree_dirty": False if not refusals else None,
        "geometry_version": a.get("geometry_version"),
        "contract_atol": _CONTRACT_ATOL,
        "contract_rtol": _CONTRACT_RTOL,
        "per_model": deltas,
        "max_abs_delta_overall": worst,
        "n_features_total": a.get("n_features_total"),
        "refusals": refusals,
        "status": "refused" if refusals else ("ok" if worst <= _CONTRACT_ATOL else "exceeds_atol"),
        "caveats": [
            "The two legs confound ARCHITECTURE with INTERPRETER; no third leg disentangles them. "
            "Read a clean result as 'this pair agrees', not as 'architecture is irrelevant'.",
            "atol cannot transfer to the quantized xCross features: space_controlled is quantized at "
            "~8.8696 m^2 per cell (~8.87e6 x atol), so its error is exactly 0.0 or >= 8.87 and the "
            "tolerance is an equality test on it. box_off_def_ratio is an integer ratio.",
            "This measures the CONTRACT FINGERPRINT only. It says nothing about the ghost-GK re-fit's "
            "own acceptance, which spec 6 makes PR 7's item -- do not read a clean result here as "
            "discharging it.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("LEG_A", "LEG_B"))
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run on a dirty tree; the artifact still records run_tree_dirty=true",
    )
    a = ap.parse_args()
    # ADR-037: the CLI refuses, the writer records the truth.
    require_clean_tree(git_provenance(), allow_dirty=a.allow_dirty)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.compare:
        legs = [json.loads(p.read_text(encoding="utf-8")) for p in a.compare]
        m = compare(*legs)
        (a.out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"status={m['status']}  max_abs_delta={m['max_abs_delta_overall']}")
        for r in m["refusals"]:
            print(f"  REFUSED: {r}")
        if m["status"] != "ok":
            raise SystemExit(1)
        return

    leg = measure_leg()
    path = a.out / _leg_filename()
    path.write_text(json.dumps(leg, indent=2), encoding="utf-8")
    print(f"wrote {path.name}  ({leg['n_features_total']} features across {len(leg['blocks'])} models)")


if __name__ == "__main__":
    main()
