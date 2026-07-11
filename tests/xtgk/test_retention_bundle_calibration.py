"""F1: every BUNDLED rho variant's recorded metrics must clear the canonical calibration gate.

The SHA256SUMS/tamper tests verify a bundle isn't *altered*; this verifies it *meets the bar* -- turning
"bundle only if it passes / no lowered bar" into a CI-enforced invariant that guards every future re-bundle.
The bar is the canonical _ECE_MAX/_SLOPE_TOL imported from the trainer (NOT the recorded fields), and the
recorded thresholds must equal them (so a hand-loosened metrics.json can't self-certify).
"""

import json
from pathlib import Path

from scripts.train_gk_retention import _ECE_MAX, _SLOPE_TOL

_WEIGHTS = Path(__file__).resolve().parents[2] / "silly_kicks" / "xtgk" / "_retention_weights"


def test_every_bundled_variant_clears_the_gate():
    variants = [d for d in _WEIGHTS.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    assert variants, "no bundled rho variants found"
    for d in variants:
        m = json.loads((d / "metrics.json").read_text())
        # bar = canonical constants (NOT the recorded fields) so a loosened metrics.json can't self-certify
        assert m["ece"] <= _ECE_MAX, f"{d.name}: ece {m['ece']} > {_ECE_MAX}"
        assert abs(m["reliability_slope"] - 1.0) <= _SLOPE_TOL, f"{d.name}: slope {m['reliability_slope']}"
        assert m["auc"] >= 0.5, f"{d.name}: auc {m['auc']} < chance"
        # defense-in-depth: recorded thresholds must match canonical (guards manual tampering)
        assert m["ece_max"] == _ECE_MAX and m["slope_tol"] == _SLOPE_TOL, f"{d.name}: tampered thresholds"
