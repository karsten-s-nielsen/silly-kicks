#!/usr/bin/env python
"""Regenerate tests/scripts/_fixtures/d2_calibration_golden.json (the whole-calibration.json golden).

    python tests/scripts/_fixtures/regenerate_d2_calibration_golden.py \
        --out tests/scripts/_fixtures/d2_calibration_golden.json

The golden pins the CURRENT expected whole-calibration.json from the synthetic ``plant_and_confirm`` harness
(test_whole_calibration_json_byte_identical_to_golden in test_d2_layer_a_share_split_identity.py) -- it catches
unintended end-to-end calibration drift.

HISTORY. It was first captured at fcca558 to prove the ADR-113 per-variant split (option C) is byte-identical to
the pre-C stacked layout; that proof is locked by the fe449af CI run AND stays guarded going forward by the LIVE
differential tests in the same file (test_per_variant_read_equals_stacked_filter,
test_confirm_reducers_identical_per_variant_vs_stacked, the fold tests) -- which compare per-variant vs stacked
directly, not via this frozen golden. ADR-114 then deliberately bumped OBJECTIVE_VERSION (2 -> 3), which changes
only objective_version + the three objective_id/input_contract digests that hash it (proven: the reliability
VALUES -- selections/population/joint_point/gate_cleared -- are byte-identical; the synthetic fixture has no NA
team entity, so the NA-entity exclusion is a no-op here). Re-baselining the golden therefore mirrors a version
string, not a laundered value change -- it is not circular. Requires ruthless-efficiency 0.7.0.
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

    with tempfile.TemporaryDirectory() as td:
        calibration = harness.plant_and_confirm(pathlib.Path(td) / "out")
    golden = harness._strip_volatiles(calibration)
    golden["__provenance__"] = (
        "The current expected whole-calibration.json from plant_and_confirm (post ADR-113 per-variant split + "
        "ADR-114 OBJECTIVE_VERSION=3), by regenerate_d2_calibration_golden.py. This key is stripped before the "
        "byte-identity comparison. The per-variant-split identity is proven by the live differential tests, not "
        "this golden; see this file's docstring."
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
