#!/usr/bin/env python
"""Regenerate tests/scripts/_fixtures/d2_calibration_golden.json (the D2-SPEC-05b whole-calibration.json golden).

MUST be run against the PRE-C STACKED tree, else the golden is circular (it would merely mirror the current per-variant
code instead of pinning the pre-change behaviour):

    git stash push -- scripts/calibrate_coordination.py scripts/_coordination_corpus.py
    python tests/scripts/_fixtures/regenerate_d2_calibration_golden.py \
        --out tests/scripts/_fixtures/d2_calibration_golden.json
    git stash pop

The golden was first captured at commit fcca558. It is the byte-identical reference the per-variant (option C) code
must reproduce -- see test_whole_calibration_json_byte_identical_to_golden in
tests/scripts/test_d2_layer_a_share_split_identity.py. This generator runs the SAME plant_and_confirm harness as the
test (driver entrypoints only, so it runs at both fcca558 and HEAD), strips the volatiles, and writes the golden.
Requires ruthless-efficiency 0.7.0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the D2 calibration.json golden (run on the PRE-C tree).")
    ap.add_argument("--out", required=True, help="path to write the golden JSON")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tests" / "scripts"))

    import scripts._loader_pining as lp

    lp.match_visibility = lambda *a, **k: {}  # ADR-038: no network (the autouse pytest fixture does not run here)

    import test_d2_layer_a_share_split_identity as harness

    if harness._per_variant_layout():
        raise SystemExit(
            "refusing: this tree stores the baseline share PER VARIANT (post-C); regenerating here is circular. "
            "git stash the two production scripts to the pre-C stacked tree first (see this file's docstring)."
        )

    with tempfile.TemporaryDirectory() as td:
        calibration = harness.plant_and_confirm(pathlib.Path(td) / "out")
    golden = harness._strip_volatiles(calibration)
    golden["__provenance__"] = (
        "Captured from the PRE-C stacked tree (fcca558) by regenerate_d2_calibration_golden.py. This key is stripped "
        "before the byte-identity comparison. Do NOT regenerate against the post-C tree -- that is circular."
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
