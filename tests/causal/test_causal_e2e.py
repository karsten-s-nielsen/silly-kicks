import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks._causal.opportunities import GK_BLOCK, PAPER_CONFOUNDERS, build_opportunities
from silly_kicks.spadl import config as _c

# Reuse the geometry-correct spell-fixture builders (single source -- tests/causal/_fixtures.py, R3-L2).
from tests.causal._fixtures import META, actions, spell

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_xcross_causal.py"


def _driver():
    """Load the driver script in-process (mirrors the trainer-smoke importlib pattern)."""
    spec = importlib.util.spec_from_file_location("_validate_xcross_causal", _DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _multi_spell_frames(n_spells=80, seed=0):
    """TEST DEVICE (R3-L3): one period per spell to manufacture N independent opportunities -- NOT
    physical (a real match has <=5 periods). Each spell spans [10, 12] s so a cross at 11 s survives
    the possession clamp (R3-M1). Mild per-spell jitter on two confounders (ball-x -> dist_endline;
    nearest defender -> dist_nearest_def) gives non-degenerate covariate spread so analyze()'s
    overlap/SMD path is exercised, not a constant-X degenerate one (R3-L1)."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_spells):
        bx = 12.0 + float(rng.uniform(-3.0, 3.0))
        f = spell(5, 10.0, 12.0, ball=(bx, 6.0), period=k + 1)  # goes through the REAL builder path
        jx, jy = 10.0 + float(rng.uniform(-3.0, 3.0)), 30.0 + float(rng.uniform(-3.0, 3.0))
        f.loc[f["player_id"] == 22, ["x", "y"]] = [jx, jy]  # jitter the nearest defender
        f["game_id"] = 1
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def _synth_actions(frames_df):
    """Deterministic: half the periods get a cross (treated, some with a post-cross shot), the other
    half stay control (some with a shot from entry) -> Z AND Y both vary (status='ok')."""
    cross, shot = _c.actiontype_id["cross"], _c.actiontype_id["shot"]
    rows, aid = [], 0
    for i, per in enumerate(sorted(frames_df["period_id"].unique())):
        if i % 2 == 0:  # treated: a cross at 11 s (within the [10,12] s spell)
            rows.append([1, aid, int(per), 5, 11.0, cross, 1, 20, 8, 14, 6])
            aid += 1
            if i % 4 == 0:  # post-cross shot (within W of t_cross)
                rows.append([1, aid, int(per), 5, 11.5, shot, 1, 14, 6, 0, 34])
                aid += 1
        elif i % 3 == 0:  # control with a shot from entry
            rows.append([1, aid, int(per), 5, 12.0, shot, 1, 14, 6, 0, 34])
            aid += 1
    return actions(rows)


@pytest.mark.e2e
def test_build_analyze_write_chain(tmp_path):
    V = _driver()
    frames = _multi_spell_frames(80)
    opp = build_opportunities(frames, _synth_actions(frames), home_team_id=5, model_metadata=META)
    assert len(opp) >= 2
    assert {"Z", "Y", *PAPER_CONFOUNDERS, *GK_BLOCK} <= set(opp.columns)  # column contract
    assert opp["Z"].nunique() == 2  # both arms present -> analyze runs the full path
    m = V.analyze(opp, seed=0)
    V._write(tmp_path, m)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "report.md").exists()
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["status"] == "ok"
    for k in (
        "att_without_gk",
        "att_with_gk",
        "placebo_band_p95",
        "gk_nan_fraction",
        "base_nan_fraction",
        "ps_overlap_fraction",
    ):
        assert k in m
    assert np.isfinite(m["att_with_gk"]["estimate"])
    assert isinstance(m["gk_clears_placebo_band"], bool)
    assert isinstance(m["causal_claim_supported"], bool)


@pytest.mark.e2e
def test_run_with_fake_loader(tmp_path, monkeypatch):
    # Inject a fake _loader_pining so run()'s `from _loader_pining import load_matches` picks it up
    # (function-local-import mocking) -> exercises run()'s coverage/eligible-pool/write.
    frames = _multi_spell_frames(80)
    acts = _synth_actions(frames)
    fake = types.ModuleType("_loader_pining")
    fake.load_matches = lambda **kw: iter([("skillcorner", "m1", acts, frames, 5)])
    monkeypatch.setitem(sys.modules, "_loader_pining", fake)
    V = _driver()
    metrics = V.run(tmp_path, ["skillcorner"], 0.0, 0)
    assert (tmp_path / "metrics.json").exists()
    assert "coverage" in metrics and metrics["coverage"]["skillcorner"]["n_opp"] >= 2


@pytest.mark.e2e
def test_analyze_positivity_guard():
    V = _driver()
    frames = _multi_spell_frames(10)
    opp = build_opportunities(frames, actions([]), home_team_id=5, model_metadata=META)  # no crosses
    m = V.analyze(opp, seed=0)  # all Z=0 -> guard, never NaN ATT (M5)
    assert m["status"] == "no_variation_in_treatment"
    assert "att_with_gk" not in m
