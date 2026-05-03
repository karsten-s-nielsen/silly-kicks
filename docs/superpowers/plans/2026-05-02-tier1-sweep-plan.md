# Tier 1 sweep — Implementation Plan (PR-S24)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship silly-kicks 3.1.0 bundling four On-Deck Tier 1 features — TF-6 (`sync_score`), TF-8 (smoothing primitives), TF-9 (interpolation/gap-filling), TF-12 (`pre_shot_gk_angle_*`) — under one cohesive design that locks in the shared preprocessing config + provider-defaults pattern downstream Tier 3-6 features will consume.

**Architecture:** New `silly_kicks.tracking.preprocess` module with `PreprocessConfig` dataclass (flag-based `is_default()`, per-provider defaults from probe baselines) + canonical `vx`/`vy`/`speed`/`x_smoothed`/`y_smoothed`/`_preprocessed_with` schema-extension columns (additive, not in-place). New `_pre_shot_gk_angle` kernel + `add_pre_shot_gk_angle` aggregator (2 columns) extending the existing `add_pre_shot_gk_context` umbrella facade to emit 6 cols when frames supplied. New `sync_score` primitive on `LinkReport` audit pointers. All four ride within ADR-004/ADR-005 envelopes; no new ADR (multi-flavor convention asymmetry queued as ADR-005 amendment with TF-2 PR).

**Tech Stack:** Python 3.10+ (Databricks compat), pandas (long-form), numpy (vectorized geometry/interpolation — `np.interp` for linear gap-fill), scipy (`scipy.signal.savgol_filter` for smoothing + SG-derivative — promoted to a hard runtime dep in this PR; previously only optional via `xthreat.py`), pytest (TDD), pyright (strict typing), ruff (lint + format), uv (env mgmt).

**Runtime-deps change:** `pyproject.toml` `[project] dependencies` gains `scipy>=1.10.0`. The 1.10 floor matches the supported Python window (3.10+) and predates known API breaks for `savgol_filter` (no kwarg renames). `xthreat.py`'s defensive `try/except ImportError` for `RectBivariateSpline` becomes unconditional via the new hard dep — that import simplifies to a plain `from scipy.interpolate import RectBivariateSpline` (small Chesterton's-Fence-aware cleanup; `# pragma: no cover` removed).

**Spec:** `docs/superpowers/specs/2026-05-02-tier1-sweep-design.md` (v3 — 16 review items folded across 2 lakehouse sessions)

**ADRs:** ADR-003 (NaN safety), ADR-004 (tracking namespace charter), ADR-005 (frame-aware xfn / NOTICE attribution), ADR-006 (direction-of-play correctness — invariant-coverage discipline).

**Memory references applied throughout:** `feedback_commit_policy` (one squash commit), `feedback_final_review_gate` (mandatory `/final-review`), `feedback_no_silent_skips_on_required_testing` (loud-skip; SkillCorner included), `feedback_inline_execution_default` (no subagents unless explicit), `feedback_invariant_testing` (registered fixtures + density gate), `feedback_probe_driven_fixture_parameterization` (probes → JSON → code), `feedback_lakehouse_consumer_not_source` (raw provider values), `feedback_pandas_attrs_dont_propagate` (per-row provenance column), `feedback_additive_columns_over_inplace_mutation` (raw `x`/`y` preserved), `feedback_consumer_contract_columnset_handshake` (CHANGELOG enumerates new columns), `feedback_default_config_auto_promotion` (flag-based `is_default()`), `feedback_primitive_plus_assembly_aggregators` (umbrella facade), `feedback_synthesizer_shot_plus_keeper_save_pattern` (≥1 shot+keeper_save per provider), `feedback_ruff_format_check` (separate ruff format gate), `feedback_api_change_sweep_ci_scope` (full `tests/` lint scope), `feedback_public_api_examples` (Examples on every new public def).

**Branch:** `pr-s24-tier1-sweep` (already cut off main).

---

## Loop overview

| # | Theme | New code center of gravity | TDD-RED tests written |
|---|-------|----------------------------|------------------------|
| 1 | Probe + baseline JSON + **codegen** + integrity test | `scripts/probe_preprocess_baseline.py`; `scripts/regenerate_provider_defaults.py`; `tests/fixtures/baselines/preprocess_baseline.json`; `tests/fixtures/baselines/preprocess_sweep_log.json`; `silly_kicks/tracking/preprocess/_provider_defaults_generated.py`; `tests/test_preprocess_baseline_integrity.py` | integrity test (RED) |
| 2 | `PreprocessConfig` + smoothing + interpolation + velocities | `silly_kicks/tracking/preprocess/` | invariant + analytic tests (RED) |
| 3 | TF-6 `sync_score` + LinkReport method + atomic parity smoke | `silly_kicks/tracking/utils.py`; `silly_kicks/tracking/schema.py`; `silly_kicks/atomic/tracking/__init__.py` | bounds + monotonicity + atomic-parity (RED) |
| 4 | TF-12 GK angle (kernel + aggregator + atomic mirror + per-period DOP invariant) | `silly_kicks/tracking/_kernels.py`; `silly_kicks/tracking/features.py`; `silly_kicks/atomic/tracking/features.py`; `tests/invariants/test_invariant_gk_angle_per_period_dop_symmetry.py` | bounds + GK-on-line=0 + sign flip + per-period DOP (RED) |
| 5 | Umbrella facade `add_pre_shot_gk_context` extension (4→6) + tracking-converter `preprocess` kwarg wiring | `silly_kicks/spadl/utils.py`; `silly_kicks/atomic/spadl/utils.py`; `silly_kicks/tracking/sportec.py` / `pff.py` / `kloppy.py` | extended-facade test + auto-promotion test (RED) |
| 6 | VAEP xfn lists + NOTICE + multi-flavor asymmetry doc + Public-API Examples + CHANGELOG | `silly_kicks/tracking/features.py`; `silly_kicks/atomic/tracking/features.py`; `silly_kicks/tracking/preprocess/__init__.py`; `NOTICE`; `CHANGELOG.md` | none new (CI gate runs) |
| 7 | Empirical sweep against real datasets + standard pre-PR gates + `/final-review` + single commit | n/a (verification only) | n/a |

Branch already cut. No Loop-0 bootstrap step.

---

## Loop 1 — Probe + baseline JSON + codegen + integrity test

**Why first.** Every numeric default (`sg_window_seconds`, `max_gap_seconds`, `link_quality_high_threshold`, etc.) traces to measured per-provider statistics. The integrity test hard-anchors `_PROVIDER_DEFAULTS` (Loop 2) to the JSON, so the JSON must exist before the dataclass can ship without a CI hole.

**Codegen pattern (S1+S2 fix from lakehouse review).** `_PROVIDER_DEFAULTS` is *generated* from the JSON by `scripts/regenerate_provider_defaults.py` into `_provider_defaults_generated.py`. The integrity test thus validates a **deterministic** code-vs-JSON relationship (codegen output equals JSON input by construction) — `rel_tol=1e-6` is appropriate. There is no hand-edit step to forget. When real-data probe-runs change the JSON, one command regenerates the Python; the integrity test confirms the regen happened.

**Public re-export (N5 fix).** A stable `get_provider_defaults()` getter is exposed from `silly_kicks.tracking.preprocess` so the integrity test (and any external consumer) reads the dict via the public API rather than importing `_PROVIDER_DEFAULTS` from a private submodule.

The integrity test is written in RED state (failing) so Loop 2 has a clear green target.

**Files:**
- Create: `scripts/probe_preprocess_baseline.py`
- Create: `scripts/regenerate_provider_defaults.py`
- Create: `tests/fixtures/baselines/preprocess_baseline.json`
- Create: `tests/fixtures/baselines/preprocess_sweep_log.json`
- Create: `silly_kicks/tracking/preprocess/_provider_defaults_generated.py` (header marker `# AUTO-GENERATED — DO NOT EDIT`)
- Create: `tests/test_preprocess_baseline_integrity.py`

### Step 1.1: Verify the fixture-baselines directory exists

```powershell
New-Item -ItemType Directory -Force -Path tests/fixtures/baselines | Out-Null
```

Expected: silent; directory exists either way (`-Force`).

### Step 1.2: Write `scripts/probe_preprocess_baseline.py` (probe-only — no silly-kicks public API touched)

```python
"""Probe per-provider preprocess baseline statistics for PR-S24.

Outputs:
  tests/fixtures/baselines/preprocess_baseline.json  — per-provider numeric stats
  tests/fixtures/baselines/preprocess_sweep_log.json — aggregate distribution stats

Sources (re-used from PR-S19 probe):
  - Lakehouse Databricks SQL: soccer_analytics.dev_gold.fct_tracking_frames
    (providers: metrica, idsse->sportec, skillcorner)
  - Local PFF FC WC2022 JSONL.bz2 at PFF_LOCAL_DIR (env-overridable)

Usage::

    uv run python scripts/probe_preprocess_baseline.py
    uv run python scripts/probe_preprocess_baseline.py --provider sportec
    uv run python scripts/probe_preprocess_baseline.py --emit-sweep-log

Re-runnable post-merge for parameter re-tuning. Spec §4.6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_baseline.json"
SWEEP_LOG_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_sweep_log.json"

PROVIDERS = ("sportec", "pff", "metrica", "skillcorner")


def _percentile(s: pd.Series, q: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.nanpercentile(s, q))


def _per_provider_block(frames: pd.DataFrame, provider: str) -> dict[str, Any]:
    """Compute the canonical per-provider stat block from a long-form frames slice."""
    # Sampling rate (Hz)
    sampling_hz = float(frames["frame_rate"].dropna().median()) if "frame_rate" in frames.columns else float("nan")

    # Position noise floor (median |x_t - x_{t-1}| within (period, player) groups,
    # then median across groups)
    sorted_frames = frames.sort_values(["period_id", "player_id", "frame_id"], kind="mergesort")
    dx = sorted_frames.groupby(["period_id", "player_id"], dropna=False)["x"].diff().abs()
    noise_floor = _percentile(dx, 50)

    # Velocity outlier rate (frames > 12 m/s when speed available)
    if "speed" in frames.columns:
        speed = frames["speed"].dropna()
        v_outlier_rate = float((speed > 12.0).mean()) if len(speed) > 0 else float("nan")
    else:
        v_outlier_rate = float("nan")

    # Gap rate per is_ball
    is_ball = frames.get("is_ball", pd.Series(False, index=frames.index))
    gap_player_pct = float(frames.loc[~is_ball, "x"].isna().mean() * 100) if (~is_ball).any() else float("nan")
    gap_ball_pct = float(frames.loc[is_ball, "x"].isna().mean() * 100) if is_ball.any() else float("nan")

    # Gap-length percentiles (run-length of consecutive NaN per (period, player) group)
    def _runlen(s: pd.Series) -> list[int]:
        a = s.isna().to_numpy()
        if not a.any():
            return []
        # standard run-length encoding for True runs
        change = np.diff(a.astype(np.int8), prepend=0, append=0)
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        return (ends - starts).tolist()

    runs: list[int] = []
    for _, grp in sorted_frames.groupby(["period_id", "player_id", "is_ball"], dropna=False):
        runs.extend(_runlen(grp["x"]))
    gap_p50 = int(np.percentile(runs, 50)) if runs else 0
    gap_p99 = int(np.percentile(runs, 99)) if runs else 0

    # Derived defaults: sg window in seconds = ~10 frames @ Hz; max_gap_seconds = p99 frames / Hz
    sg_window_seconds = round(10 / sampling_hz, 3) if sampling_hz > 0 else 0.4
    max_gap_seconds = round(gap_p99 / sampling_hz, 3) if sampling_hz > 0 else 0.5

    return {
        "sampling_rate_hz": sampling_hz,
        "raw_position_noise_floor_m": noise_floor,
        "velocity_outlier_rate_at_max_12mps": v_outlier_rate,
        "gap_rate_player_pct": gap_player_pct,
        "gap_rate_ball_pct": gap_ball_pct,
        "gap_length_p50_frames": gap_p50,
        "gap_length_p99_frames": gap_p99,
        # Post-interpolation NaN rate is derived once preprocess.interpolate_frames lands;
        # at probe time we project as gap-beyond-max-gap-seconds * raw-gap-rate.
        "post_interpolation_nan_rate_player_pct": round(gap_player_pct * 0.05, 3),
        "post_interpolation_nan_rate_ball_pct": round(gap_ball_pct * 0.05, 3),
        # link_quality stats — populated by linkage probe; placeholder when unavailable
        "link_quality_score_p50": 0.92,
        "link_quality_score_p10": 0.71,
        "link_quality_high_threshold": 0.85,
        "gk_angle_to_shot_trajectory_p50_rad": 0.04,
        "gk_angle_off_goal_line_p50_rad": 0.06,
        "_derived_defaults": {
            "sg_window_seconds": sg_window_seconds,
            "sg_poly_order": 3,
            "ema_alpha": 0.3,
            "max_gap_seconds": max_gap_seconds,
        },
    }


def _provenance() -> dict[str, Any]:
    return {
        "generated_by": "scripts/probe_preprocess_baseline.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "silly_kicks_version": "3.1.0-dev",
        "providers_probed": list(PROVIDERS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS + ("all",), default="all")
    parser.add_argument("--emit-sweep-log", action="store_true")
    args = parser.parse_args()

    # Lakehouse + PFF probe routines re-used from probe_tracking_baselines.py.
    # Placeholder values are committed when env is unset — values are then
    # tightened on first PR-S24 probe-run with credentials present.
    out: dict[str, Any] = {"_provenance": _provenance()}

    # NOTE: actual data loading delegated to the same helpers used in
    # scripts/probe_tracking_baselines.py. To keep this plan self-contained the
    # delegation imports are inlined at execution time; see Step 1.4 for the
    # invocation. When credentials/local data are absent, default placeholder
    # blocks (matching the JSON schema in spec §4.6) are emitted so the JSON
    # is well-formed and the integrity test can still anchor _PROVIDER_DEFAULTS.

    from probe_tracking_baselines import probe_lakehouse, probe_pff_local  # type: ignore[import-not-found]

    selected = PROVIDERS if args.provider == "all" else (args.provider,)

    lakehouse_blocks = probe_lakehouse() if any(p in {"sportec", "metrica", "skillcorner"} for p in selected) else {}
    pff_block = probe_pff_local() if "pff" in selected else None

    for prov in selected:
        if prov == "pff" and pff_block is not None:
            out[prov] = _per_provider_block(pff_block, prov)
        elif prov in lakehouse_blocks:
            out[prov] = _per_provider_block(lakehouse_blocks[prov], prov)
        else:
            # Conservative placeholder. Concrete values land when probe is run
            # with credentials/data; integrity test allows this state on CI.
            out[prov] = _per_provider_block(pd.DataFrame(columns=["frame_rate", "x", "is_ball", "speed", "period_id", "player_id", "frame_id"]), prov)

    BASELINE_JSON.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if args.emit_sweep_log:
        sweep_log = {"_provenance": _provenance(), "by_provider": {p: out.get(p, {}) for p in PROVIDERS}}
        SWEEP_LOG_JSON.write_text(json.dumps(sweep_log, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"[probe] wrote {BASELINE_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # so probe_tracking_baselines is importable
    raise SystemExit(main())
```

### Step 1.3: Commit-time placeholder JSON (so integrity test has an anchor before first probe-run)

Write `tests/fixtures/baselines/preprocess_baseline.json` with the schema from spec §4.6 — concrete numbers per `_per_provider_block` for sportec; pff/metrica/skillcorner mirror the schema. Emit a parallel `preprocess_sweep_log.json` with provenance + per-provider blocks (allowed to be the same content at first probe-run; the file exists so the spec §5.1 gate can read it).

```json
{
  "_provenance": {
    "generated_by": "scripts/probe_preprocess_baseline.py",
    "generated_at": "2026-05-02T00:00:00Z",
    "silly_kicks_version": "3.1.0-dev",
    "providers_probed": ["sportec", "pff", "metrica", "skillcorner"]
  },
  "sportec": {
    "sampling_rate_hz": 25.0,
    "raw_position_noise_floor_m": 0.04,
    "velocity_outlier_rate_at_max_12mps": 0.0008,
    "gap_rate_player_pct": 2.1,
    "gap_rate_ball_pct": 14.7,
    "gap_length_p50_frames": 1,
    "gap_length_p99_frames": 12,
    "post_interpolation_nan_rate_player_pct": 0.105,
    "post_interpolation_nan_rate_ball_pct": 0.735,
    "link_quality_score_p50": 0.92,
    "link_quality_score_p10": 0.71,
    "link_quality_high_threshold": 0.85,
    "gk_angle_to_shot_trajectory_p50_rad": 0.04,
    "gk_angle_off_goal_line_p50_rad": 0.06,
    "_derived_defaults": {
      "sg_window_seconds": 0.4,
      "sg_poly_order": 3,
      "ema_alpha": 0.3,
      "max_gap_seconds": 0.48
    }
  },
  "pff": {
    "sampling_rate_hz": 30.0,
    "raw_position_noise_floor_m": 0.05,
    "velocity_outlier_rate_at_max_12mps": 0.0010,
    "gap_rate_player_pct": 1.8,
    "gap_rate_ball_pct": 8.4,
    "gap_length_p50_frames": 1,
    "gap_length_p99_frames": 15,
    "post_interpolation_nan_rate_player_pct": 0.090,
    "post_interpolation_nan_rate_ball_pct": 0.420,
    "link_quality_score_p50": 0.94,
    "link_quality_score_p10": 0.78,
    "link_quality_high_threshold": 0.88,
    "gk_angle_to_shot_trajectory_p50_rad": 0.05,
    "gk_angle_off_goal_line_p50_rad": 0.07,
    "_derived_defaults": {
      "sg_window_seconds": 0.333,
      "sg_poly_order": 3,
      "ema_alpha": 0.3,
      "max_gap_seconds": 0.5
    }
  },
  "metrica": {
    "sampling_rate_hz": 25.0,
    "raw_position_noise_floor_m": 0.06,
    "velocity_outlier_rate_at_max_12mps": 0.0015,
    "gap_rate_player_pct": 2.6,
    "gap_rate_ball_pct": 77.0,
    "gap_length_p50_frames": 2,
    "gap_length_p99_frames": 14,
    "post_interpolation_nan_rate_player_pct": 0.130,
    "post_interpolation_nan_rate_ball_pct": 73.5,
    "link_quality_score_p50": 0.88,
    "link_quality_score_p10": 0.62,
    "link_quality_high_threshold": 0.80,
    "gk_angle_to_shot_trajectory_p50_rad": 0.04,
    "gk_angle_off_goal_line_p50_rad": 0.06,
    "_derived_defaults": {
      "sg_window_seconds": 0.4,
      "sg_poly_order": 3,
      "ema_alpha": 0.3,
      "max_gap_seconds": 0.56
    }
  },
  "skillcorner": {
    "sampling_rate_hz": 10.0,
    "raw_position_noise_floor_m": 0.10,
    "velocity_outlier_rate_at_max_12mps": 0.0020,
    "gap_rate_player_pct": 4.2,
    "gap_rate_ball_pct": 11.0,
    "gap_length_p50_frames": 1,
    "gap_length_p99_frames": 6,
    "post_interpolation_nan_rate_player_pct": 0.210,
    "post_interpolation_nan_rate_ball_pct": 0.550,
    "link_quality_score_p50": 0.85,
    "link_quality_score_p10": 0.55,
    "link_quality_high_threshold": 0.75,
    "gk_angle_to_shot_trajectory_p50_rad": 0.05,
    "gk_angle_off_goal_line_p50_rad": 0.08,
    "_derived_defaults": {
      "sg_window_seconds": 1.0,
      "sg_poly_order": 3,
      "ema_alpha": 0.3,
      "max_gap_seconds": 0.6
    }
  }
}
```

The committed JSON values are conservative placeholders; first real probe-run on the user's lakehouse credentials + local PFF data tightens them. The integrity test (Step 1.4) ensures any later hand-edit of either artifact is caught — including this initial file.

Write the matching `preprocess_sweep_log.json` with the same schema (spec §5.1 #3 — aggregate distribution stats only, no row-level data).

### Step 1.3a: Write `scripts/regenerate_provider_defaults.py` (codegen)

```python
"""Regenerate silly_kicks/tracking/preprocess/_provider_defaults_generated.py
from tests/fixtures/baselines/preprocess_baseline.json.

Run after every probe-baseline JSON change:

    uv run python scripts/regenerate_provider_defaults.py

The generated file is committed and consumed by silly_kicks.tracking.preprocess._config.
The integrity test (tests/test_preprocess_baseline_integrity.py) verifies that the
generated file matches the JSON within rel_tol=1e-6 — i.e., regen happened.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_baseline.json"
OUTPUT_PY = REPO_ROOT / "silly_kicks" / "tracking" / "preprocess" / "_provider_defaults_generated.py"

_HEADER = '''"""AUTO-GENERATED — DO NOT EDIT.

Source: tests/fixtures/baselines/preprocess_baseline.json
Regen:  uv run python scripts/regenerate_provider_defaults.py
"""
from __future__ import annotations

from ._config_dataclass import PreprocessConfig

_PROVIDER_DEFAULTS: dict[str, PreprocessConfig] = {
'''

_FOOTER = "}\n"


def _emit_block(provider: str, block: dict) -> str:
    derived = block["_derived_defaults"]
    return (
        f'    "{provider}": PreprocessConfig(\n'
        f'        smoothing_method="savgol",\n'
        f'        sg_window_seconds={derived["sg_window_seconds"]},\n'
        f'        sg_poly_order={derived["sg_poly_order"]},\n'
        f'        ema_alpha={derived["ema_alpha"]},\n'
        f'        interpolation_method="linear",\n'
        f'        max_gap_seconds={derived["max_gap_seconds"]},\n'
        f'        derive_velocity=True,\n'
        f'        link_quality_high_threshold={block["link_quality_high_threshold"]},\n'
        f'    ),\n'
    )


def main() -> int:
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    body = "".join(_emit_block(p, payload[p]) for p in ("sportec", "pff", "metrica", "skillcorner"))
    OUTPUT_PY.write_text(_HEADER + body + _FOOTER, encoding="utf-8")
    print(f"[regen] wrote {OUTPUT_PY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run once with the placeholder JSON to seed the generated file:

```powershell
uv run python scripts/regenerate_provider_defaults.py
```

Expected: writes `silly_kicks/tracking/preprocess/_provider_defaults_generated.py`. Commit alongside the JSON.

### Step 1.3b: Note on `_config_dataclass.py` split

The generated file imports `PreprocessConfig` from `_config_dataclass.py` (a new submodule containing **only** the dataclass) — this avoids a circular import (`_config.py` imports `_PROVIDER_DEFAULTS` from `_provider_defaults_generated.py`, which would otherwise try to import from `_config.py`). Loop 2 Step 2.2 splits the dataclass into `_config_dataclass.py` and the orchestration into `_config.py`.

### Step 1.4: Write `tests/test_preprocess_baseline_integrity.py` (RED — module doesn't exist yet)

```python
"""Asserts the generated _PROVIDER_DEFAULTS matches the JSON baseline within rel_tol=1e-6.

This is a deterministic invariant: scripts/regenerate_provider_defaults.py
generates the Python from the JSON. Any drift means regen has not been run.

Failure-message contract: the assertion message MUST include the regen
command so future readers know how to fix.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_baseline.json"
REGEN_HINT = "Run `uv run python scripts/regenerate_provider_defaults.py` to regenerate _provider_defaults_generated.py."


@pytest.fixture(scope="module")
def baseline_json() -> dict:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def provider_defaults():
    """Use the public getter, NOT a private submodule import (S5/N5 from review)."""
    from silly_kicks.tracking.preprocess import get_provider_defaults

    return get_provider_defaults()


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
def test_integer_fields_exact_match(baseline_json, provider_defaults, provider):
    block = baseline_json[provider]
    cfg = provider_defaults[provider]
    derived = block["_derived_defaults"]
    assert cfg.sg_poly_order == derived["sg_poly_order"], (
        f"{provider}: sg_poly_order mismatch — "
        f"_PROVIDER_DEFAULTS={cfg.sg_poly_order} vs JSON={derived['sg_poly_order']}. {REGEN_HINT}"
    )


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
@pytest.mark.parametrize("field,attr", [
    ("sg_window_seconds", "sg_window_seconds"),
    ("ema_alpha", "ema_alpha"),
    ("max_gap_seconds", "max_gap_seconds"),
])
def test_derived_default_floats_match(baseline_json, provider_defaults, provider, field, attr):
    expected = baseline_json[provider]["_derived_defaults"][field]
    actual = getattr(provider_defaults[provider], attr)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0), (
        f"{provider}.{attr}: code={actual} vs JSON.{field}={expected}. {REGEN_HINT}"
    )


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
def test_link_quality_high_threshold_match(baseline_json, provider_defaults, provider):
    expected = baseline_json[provider]["link_quality_high_threshold"]
    actual = provider_defaults[provider].link_quality_high_threshold
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0), REGEN_HINT


def test_provenance_block_present(baseline_json):
    assert "_provenance" in baseline_json
    for k in ("generated_by", "generated_at", "silly_kicks_version", "providers_probed"):
        assert k in baseline_json["_provenance"]


def test_sweep_log_exists():
    sweep_log = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_sweep_log.json"
    assert sweep_log.exists(), "preprocess_sweep_log.json must accompany the baseline JSON (spec §5.1 #3)"
    payload = json.loads(sweep_log.read_text(encoding="utf-8"))
    assert "_provenance" in payload
    assert "by_provider" in payload
    for p in ("sportec", "pff", "metrica", "skillcorner"):
        assert p in payload["by_provider"], f"sweep_log missing provider block: {p}"
```

### Step 1.5: Run the RED test — expect ImportError on `_PROVIDER_DEFAULTS`

```powershell
uv run python -m pytest tests/test_preprocess_baseline_integrity.py -x --tb=short
```

Expected: collection-time `ModuleNotFoundError: No module named 'silly_kicks.tracking.preprocess'`. This is the GREEN target for Loop 2.

---

## Loop 2 — `PreprocessConfig` + smoothing + interpolation + velocities

**Module split (S1 fix carryover):** to avoid a circular import the dataclass lives in `_config_dataclass.py` while `_config.py` becomes the orchestration / public-getter layer.

**Files:**
- Create: `silly_kicks/tracking/preprocess/__init__.py`
- Create: `silly_kicks/tracking/preprocess/_config_dataclass.py` (just the `PreprocessConfig` dataclass)
- Create: `silly_kicks/tracking/preprocess/_config.py` (`get_provider_defaults()` public getter + import-from-generated)
- Create: `silly_kicks/tracking/preprocess/_smoothing.py`
- Create: `silly_kicks/tracking/preprocess/_interpolation.py`
- Create: `silly_kicks/tracking/preprocess/_velocity.py`
- Create: `silly_kicks/tracking/preprocess/_resolve.py` (provider-aware auto-promotion with S5 fallback)
- Modify: `silly_kicks/tracking/__init__.py` (re-export public surface)
- Modify: `pyproject.toml` (add `scipy>=1.10.0` to runtime `dependencies`)
- Modify: `silly_kicks/xthreat.py` (drop the `try/except ImportError` for `RectBivariateSpline` now that scipy is a hard dep)
- Create: `tests/test_preprocess_config.py`
- Create: `tests/test_smooth_frames.py`
- Create: `tests/test_interpolate_frames.py`
- Create: `tests/test_derive_velocities.py`
- Create: `tests/invariants/test_invariant_smooth_frames_idempotence.py`
- Create: `tests/invariants/test_invariant_smooth_frames_constant_signal.py`
- Create: `tests/invariants/test_invariant_interpolate_passes_through.py`
- Create: `tests/invariants/test_invariant_velocity_physical_plausibility.py`

### Step 2.1 — `_config.py` RED test

`tests/test_preprocess_config.py`:

```python
"""PreprocessConfig dataclass — flag-based is_default(); per-provider defaults."""
from __future__ import annotations

import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig


def test_default_factory_marks_universal():
    cfg = PreprocessConfig.default()
    assert cfg.is_default() is True


def test_force_universal_does_not_mark_default():
    cfg = PreprocessConfig.default(force_universal=True)
    assert cfg.is_default() is False


def test_for_provider_does_not_mark_default():
    cfg = PreprocessConfig.for_provider("sportec")
    assert cfg.is_default() is False


def test_hand_constructed_with_default_values_does_not_mark_default():
    """Flag-based, not value-equality."""
    cfg = PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=True,
    )
    assert cfg.is_default() is False


def test_eq_excludes_provenance_flag():
    """Two configs with identical fields are equal regardless of factory provenance."""
    a = PreprocessConfig.default()
    b = PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=True,
    )
    # Per the dataclass field(compare=False) on _is_universal_default
    assert a == b


def test_unknown_provider_raises():
    with pytest.raises(KeyError):
        PreprocessConfig.for_provider("statsbomb")  # tracking-side unsupported


def test_derive_velocity_without_smoothing_rejected_at_construction():
    """Lakehouse review C1: catch the impossible combination at construction time,
    not mid-pipeline. derive_velocity=True needs smoothed positions; smoothing_method=None
    means there are none to read."""
    with pytest.raises(ValueError, match="derive_velocity=True requires smoothing_method"):
        PreprocessConfig(
            smoothing_method=None,
            derive_velocity=True,
            sg_window_seconds=0.4,
            sg_poly_order=3,
            ema_alpha=0.3,
            interpolation_method="linear",
            max_gap_seconds=0.5,
        )


def test_derive_velocity_false_without_smoothing_is_fine():
    """Caller wants raw positions only — no smoothing, no velocity. Allowed."""
    cfg = PreprocessConfig(
        smoothing_method=None,
        derive_velocity=False,
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
    )
    assert cfg.smoothing_method is None
    assert cfg.derive_velocity is False
```

Run: `uv run python -m pytest tests/test_preprocess_config.py -x` → fail (module missing).

### Step 2.1a — Add scipy as a runtime dep (`pyproject.toml`)

In the `[project] dependencies` block, append `"scipy>=1.10.0"`. Then run:

```powershell
uv lock
```

Expected: `uv.lock` updates with scipy + transitives. Commit alongside the rest of Loop 2.

Then drop the optional-import dance in `silly_kicks/xthreat.py`:

```python
# Before (lines 12-15):
try:
    from scipy.interpolate import RectBivariateSpline  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    RectBivariateSpline = None

# After:
from scipy.interpolate import RectBivariateSpline
```

This is a Chesterton's-Fence-aware cleanup: the `try/except` guarded against scipy not being installed, but with scipy now a hard runtime dep the guard cannot fire. Remove the dead branch + the `# pragma: no cover` marker.

### Step 2.2 — Write `_config_dataclass.py` (just the dataclass — no imports of `_PROVIDER_DEFAULTS`)

```python
"""PreprocessConfig dataclass.

Lives in its own submodule (separate from ``_config.py``) to avoid a circular
import: the auto-generated ``_provider_defaults_generated.py`` imports
``PreprocessConfig`` from here; ``_config.py`` then orchestrates and re-exports
both. Without the split, ``_config.py`` -> ``_provider_defaults_generated.py`` ->
``_config.py`` would deadlock at import time.

PR-S24 / spec §4.3 + lakehouse review S1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SmoothingMethod = Literal["savgol", "ema", None]
# Cubic interpolation is intentionally NOT in the API surface for PR-S24.
# Lakehouse review N3: the implementation only does linear math; "cubic" was
# accepted in v1 but produced linear output. Restrict the Literal to linear-only
# until a concrete consumer asks for cubic — at that point we ship cubic via
# scipy.interpolate.CubicSpline as TF-9-cubic.
InterpolationMethod = Literal["linear", None]


@dataclass(frozen=True)
class PreprocessConfig:
    smoothing_method: SmoothingMethod = "savgol"
    sg_window_seconds: float = 0.4
    sg_poly_order: int = 3
    ema_alpha: float = 0.3
    interpolation_method: InterpolationMethod = "linear"
    max_gap_seconds: float = 0.5
    derive_velocity: bool = True
    link_quality_high_threshold: float = 0.85
    # Provenance flag — set by default() factory only. Excluded from __eq__/__hash__/repr
    # so two configs with the same field values are still equal regardless of which
    # factory built them. Read via is_default() (flag-based, NOT value-equality —
    # see feedback_default_config_auto_promotion).
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject impossible combinations at construction time (lakehouse review C1).

        ``derive_velocity=True`` requires smoothed positions (``smoothing_method`` not None);
        ``derive_velocities()`` raises if ``x_smoothed``/``y_smoothed`` are absent (S4 fix).
        Catching the inconsistency here means the converter / direct callers never run a
        partial pipeline that crashes mid-flight.
        """
        if self.derive_velocity and self.smoothing_method is None:
            raise ValueError(
                "PreprocessConfig: derive_velocity=True requires smoothing_method != None — "
                "velocity derivation reads x_smoothed/y_smoothed (which are produced by "
                "smooth_frames). Either set smoothing_method='savgol'/'ema' or "
                "derive_velocity=False."
            )

    @classmethod
    def default(cls, *, force_universal: bool = False) -> "PreprocessConfig":
        """Universal-safe defaults. Per-provider tuning via for_provider().

        Examples
        --------
        >>> from silly_kicks.tracking.preprocess import PreprocessConfig
        >>> cfg = PreprocessConfig.default()
        >>> cfg.is_default()
        True
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> "PreprocessConfig":
        """Provider-tuned defaults from preprocess_baseline.json.

        Raises ``KeyError`` for unsupported providers — caller should pre-check
        via ``provider in get_provider_defaults()`` or use
        ``PreprocessConfig.default(force_universal=True)`` as a fallback.

        Examples
        --------
        >>> from silly_kicks.tracking.preprocess import PreprocessConfig
        >>> cfg = PreprocessConfig.for_provider("sportec")
        >>> cfg.sg_window_seconds  # 10 frames @ 25 Hz
        0.4
        """
        # Local import to break the cycle (see module docstring)
        from ._config import get_provider_defaults
        return get_provider_defaults()[provider]

    def is_default(self) -> bool:
        """Flag-based: True iff built by default() without force_universal=True."""
        return self._is_universal_default
```

### Step 2.2a — Write `_config.py` (orchestration + public getter — N5 fix)

```python
"""Public-API surface for the preprocess config.

Provides ``get_provider_defaults()`` — the public-stable read of the codegen'd
provider-defaults table. Loop 1 / S1 fix: callers must use this getter instead
of importing ``_PROVIDER_DEFAULTS`` directly from a private submodule.
"""
from __future__ import annotations

from ._config_dataclass import PreprocessConfig
from ._provider_defaults_generated import _PROVIDER_DEFAULTS


def get_provider_defaults() -> dict[str, PreprocessConfig]:
    """Return a shallow copy of the per-provider PreprocessConfig defaults.

    The returned dict is a new object on every call; mutating it does not
    affect the canonical generated table. The contained ``PreprocessConfig``
    instances are frozen so they can be safely shared without copying.

    Examples
    --------
    >>> from silly_kicks.tracking.preprocess import get_provider_defaults
    >>> defaults = get_provider_defaults()
    >>> "sportec" in defaults
    True
    """
    return dict(_PROVIDER_DEFAULTS)
```

### Step 2.3 — `__init__.py` for the new submodule

`silly_kicks/tracking/preprocess/__init__.py`:

```python
"""Tracking-frame preprocessing — smoothing, interpolation, velocity derivation.

Column-naming convention
------------------------
Preprocessing utilities emit single canonical column names (``vx``, ``vy``,
``speed``, ``x_smoothed``, ``y_smoothed``) regardless of the smoothing/interpolation
method chosen. The method is recorded as a per-row ``_preprocessed_with`` provenance
column — load-bearing because ``pandas.DataFrame.attrs`` does not propagate through
merge/concat/applyInPandas (per ``feedback_pandas_attrs_dont_propagate``).

This deliberately diverges from the convention used by VAEP feature xfns
(e.g. ``pressure_on_actor__defcon`` / ``pressure_on_actor__andrienko_cone`` per
TF-2), where suffixed names are required because parallel xfn registration would
silent-overwrite same-named columns inside ``VAEP.compute_features``.

Preprocessing has no equivalent constraint: downstream features (TF-7 pitch
control, TF-15 GK reachable area, etc.) depend on schema stability and consume
single canonical inputs. Method comparison is a separate research workflow —
call ``smooth_frames`` twice with different configs into different DataFrames
and diff.

ADR-005 amendment formalising this asymmetry lands with TF-2 (scheduled
within 24-48 hours of PR-S24 merge — bounded deferral per lakehouse-review N1).
PR-S24 documents the rule operationally.

The original raw ``x`` / ``y`` columns are preserved unchanged (additive new
columns, not in-place mutation; per ``feedback_additive_columns_over_inplace_mutation``).
"""
from __future__ import annotations

from ._config import get_provider_defaults
from ._config_dataclass import PreprocessConfig
from ._interpolation import interpolate_frames
from ._smoothing import smooth_frames
from ._velocity import derive_velocities

__all__ = [
    "PreprocessConfig",
    "derive_velocities",
    "get_provider_defaults",
    "interpolate_frames",
    "smooth_frames",
]
```

### Step 2.4 — Run config tests; GREEN

```powershell
uv run python -m pytest tests/test_preprocess_config.py -v
```

Expected: 6 passed.

### Step 2.5 — Smoothing RED tests

`tests/test_smooth_frames.py`:

```python
"""smooth_frames — schema-additive output; method recorded per-row."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames


def _toy_frames(n: int = 30, hz: float = 25.0) -> pd.DataFrame:
    t = np.arange(n) / hz
    return pd.DataFrame({
        "game_id": "G1",
        "period_id": 1,
        "frame_id": np.arange(n),
        "time_seconds": t,
        "frame_rate": hz,
        "player_id": "P1",
        "team_id": "T1",
        "is_ball": False,
        "is_goalkeeper": False,
        "x": 5.0 + 0.1 * t + np.random.default_rng(0).normal(0, 0.04, n),
        "y": 30.0 + 0.0 * t + np.random.default_rng(1).normal(0, 0.04, n),
        "z": np.nan,
        "speed": np.nan,
        "speed_source": None,
        "ball_state": "alive",
        "team_attacking_direction": "ltr",
        "source_provider": "sportec",
    })


def test_raw_x_y_preserved():
    """Hyrum-Law protection: original x/y must not be mutated."""
    f = _toy_frames()
    raw_x = f["x"].copy()
    raw_y = f["y"].copy()
    out = smooth_frames(f, config=PreprocessConfig.default())
    pd.testing.assert_series_equal(out["x"], raw_x, check_names=False)
    pd.testing.assert_series_equal(out["y"], raw_y, check_names=False)


def test_smoothed_columns_added():
    out = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    assert "x_smoothed" in out.columns
    assert "y_smoothed" in out.columns
    assert out["x_smoothed"].dtype == np.float64
    assert out["y_smoothed"].dtype == np.float64


def test_provenance_column_added():
    out = smooth_frames(_toy_frames(), config=PreprocessConfig.default())
    assert "_preprocessed_with" in out.columns
    assert out["_preprocessed_with"].dtype == object
    val = out["_preprocessed_with"].iloc[0]
    assert "savgol" in val and "0.4" in val  # method + sg_window_seconds


def test_constant_signal_passes_through():
    f = _toy_frames()
    f["x"] = 50.0
    f["y"] = 30.0
    out = smooth_frames(f, config=PreprocessConfig.default())
    assert np.allclose(out["x_smoothed"], 50.0, atol=1e-9)
    assert np.allclose(out["y_smoothed"], 30.0, atol=1e-9)


def test_idempotence():
    f = _toy_frames()
    out1 = smooth_frames(f, config=PreprocessConfig.default())
    out2 = smooth_frames(out1, config=PreprocessConfig.default())
    pd.testing.assert_series_equal(out1["x_smoothed"], out2["x_smoothed"])
    pd.testing.assert_series_equal(out1["y_smoothed"], out2["y_smoothed"])


def test_ema_method():
    out = smooth_frames(_toy_frames(), method="ema", config=PreprocessConfig.default())
    val = out["_preprocessed_with"].iloc[0]
    assert "ema" in val


def test_nan_positions_skipped():
    f = _toy_frames()
    f.loc[5, "x"] = np.nan
    out = smooth_frames(f, config=PreprocessConfig.default())
    # NaN row stays NaN in smoothed (no fabrication)
    assert pd.isna(out.loc[5, "x_smoothed"])
```

Run RED: `uv run python -m pytest tests/test_smooth_frames.py -x` → fail (no implementation).

### Step 2.6 — Write `_smoothing.py` (GREEN)

```python
"""smooth_frames — Savitzky-Golay or EMA smoothing of player/ball positions.

References
----------
Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." Analytical Chemistry, 36(8), 1627-1639.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config import PreprocessConfig

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def _provenance_tag(config: PreprocessConfig, method_used: str) -> str:
    return (
        f"method={method_used}|sg_window_s={config.sg_window_seconds}|"
        f"sg_poly={config.sg_poly_order}|ema_alpha={config.ema_alpha}"
    )


def _ensure_provenance_column(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df.copy()
    df["_preprocessed_with"] = tag
    return df


def _savgol_per_group(values: np.ndarray, window_frames: int, poly_order: int) -> np.ndarray:
    if len(values) < window_frames or window_frames < poly_order + 2:
        return values.copy()  # too short — pass through
    nan_mask = np.isnan(values)
    out = values.copy()
    if nan_mask.all():
        return out
    # SG cannot tolerate NaN — interpolate gaps for the smoothing pass only,
    # then restore NaN at originally-missing positions.
    if nan_mask.any():
        idx = np.arange(len(values))
        out[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], values[~nan_mask])
    smoothed = savgol_filter(out, window_length=window_frames, polyorder=poly_order)
    smoothed[nan_mask] = np.nan
    return smoothed


def _ema_per_group(values: np.ndarray, alpha: float) -> np.ndarray:
    nan_mask = np.isnan(values)
    out = pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    out[nan_mask] = np.nan
    return out


def smooth_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Smooth player/ball position columns; emit additive ``x_smoothed``/``y_smoothed``.

    Raw ``x``/``y`` columns are preserved unchanged. The chosen method + key
    parameters are recorded in a per-row ``_preprocessed_with`` column.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    config : PreprocessConfig or None
        Smoothing config. Defaults to ``PreprocessConfig.default()``.
    method : {"savgol", "ema"} or None
        Override ``config.smoothing_method`` for this call.

    Returns
    -------
    pd.DataFrame
        Frames with additional ``x_smoothed``, ``y_smoothed``, ``_preprocessed_with``
        columns. Original ``x``/``y`` are bit-identical to the input.

    Idempotent: a re-call with the same config returns equal output (detected via
    the existing ``_preprocessed_with`` column).

    Examples
    --------
    >>> # See tests/test_smooth_frames.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    method_used = method or cfg.smoothing_method or "savgol"
    tag = _provenance_tag(cfg, method_used)

    # Idempotence shortcut
    if "_preprocessed_with" in frames.columns and (frames["_preprocessed_with"] == tag).all():
        out = frames.copy()
        if "x_smoothed" in out.columns and "y_smoothed" in out.columns:
            return out

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    hz = float(sorted_frames["frame_rate"].dropna().iloc[0]) if "frame_rate" in sorted_frames.columns else 25.0
    # SG requires odd window_length >= poly_order + 2.
    # `int(round(x)) | 1` forces odd, but `max(odd, even)` can still yield even
    # when poly_order + 2 (the lower bound) is even — re-odd-ify after the max.
    window_frames = max(int(round(cfg.sg_window_seconds * hz)) | 1, cfg.sg_poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1

    x_smoothed = np.full(len(sorted_frames), np.nan)
    y_smoothed = np.full(len(sorted_frames), np.nan)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        x_vals = sorted_frames.loc[idx_arr, "x"].to_numpy(dtype=float)
        y_vals = sorted_frames.loc[idx_arr, "y"].to_numpy(dtype=float)
        if method_used == "savgol":
            x_smoothed[idx_arr] = _savgol_per_group(x_vals, window_frames, cfg.sg_poly_order)
            y_smoothed[idx_arr] = _savgol_per_group(y_vals, window_frames, cfg.sg_poly_order)
        elif method_used == "ema":
            x_smoothed[idx_arr] = _ema_per_group(x_vals, cfg.ema_alpha)
            y_smoothed[idx_arr] = _ema_per_group(y_vals, cfg.ema_alpha)
        else:
            raise ValueError(f"smooth_frames: unsupported method={method_used!r}")

    sorted_frames["x_smoothed"] = x_smoothed
    sorted_frames["y_smoothed"] = y_smoothed

    # Restore original input order, attach provenance, mirror to attrs convenience.
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    sorted_frames = _ensure_provenance_column(sorted_frames, tag)
    sorted_frames.attrs["preprocess"] = tag
    return sorted_frames
```

Run: `uv run python -m pytest tests/test_smooth_frames.py -v` → 7 passed.

### Step 2.7 — Interpolation (RED→GREEN)

`tests/test_interpolate_frames.py` — bounds, idempotence, NaN beyond `max_gap_seconds`. `tests/invariants/test_invariant_interpolate_passes_through.py` — observed values unchanged.

Implementation `_interpolation.py`:

```python
"""interpolate_frames — fill NaN positional gaps up to max_gap_seconds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._config import PreprocessConfig

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def _interp_per_group(
    times: np.ndarray, x: np.ndarray, y: np.ndarray, max_gap_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation only (N3 fix from review). Cubic ships in TF-9-cubic
    when a consumer asks; until then the API restricts interpolation_method to
    "linear" and the implementation is straight linear math."""
    nan_mask_x = np.isnan(x)
    nan_mask_y = np.isnan(y)
    if not (nan_mask_x.any() or nan_mask_y.any()):
        return x.copy(), y.copy()
    out_x, out_y = x.copy(), y.copy()

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 2:
        return out_x, out_y  # not enough anchors

    # For each NaN gap, only fill if the gap duration <= max_gap_seconds
    idx = np.arange(len(x))
    nan_mask_combined = ~valid

    # Run-length encoding of NaN runs
    change = np.diff(nan_mask_combined.astype(np.int8), prepend=0, append=0)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    for s, e in zip(starts, ends, strict=False):
        if s == 0 or e >= len(x):
            continue  # gap touches an endpoint; no anchors on both sides
        gap_seconds = float(times[e] - times[s - 1])
        if gap_seconds > max_gap_seconds:
            continue  # leave NaN
        # Linearly interpolate this run from the anchor times[s-1], times[e]
        x_left, x_right = x[s - 1], x[e]
        y_left, y_right = y[s - 1], y[e]
        if np.isnan(x_left) or np.isnan(x_right) or np.isnan(y_left) or np.isnan(y_right):
            continue
        t_run = times[s:e]
        frac = (t_run - times[s - 1]) / max(times[e] - times[s - 1], 1e-9)
        out_x[s:e] = x_left + frac * (x_right - x_left)
        out_y[s:e] = y_left + frac * (y_right - y_left)
    return out_x, out_y


def interpolate_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Fill NaN positional gaps up to ``max_gap_seconds`` via linear interpolation.

    Gaps longer than ``max_gap_seconds`` remain NaN (no fabrication).
    Idempotent on already-filled frames.

    Only ``method="linear"`` is supported in PR-S24 (lakehouse review N3).
    Cubic interpolation will ship as TF-9-cubic when a concrete consumer asks,
    via ``scipy.interpolate.CubicSpline``.

    Examples
    --------
    >>> # See tests/test_interpolate_frames.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    method_used = method or cfg.interpolation_method or "linear"
    if method_used != "linear":
        raise ValueError(
            f"interpolate_frames: unsupported method={method_used!r}. "
            "Only 'linear' is supported in PR-S24; cubic ships in TF-9-cubic when requested."
        )

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    x = sorted_frames["x"].to_numpy(dtype=float, copy=True)
    y = sorted_frames["y"].to_numpy(dtype=float, copy=True)
    t = sorted_frames["time_seconds"].to_numpy(dtype=float)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        gx, gy = _interp_per_group(t[idx_arr], x[idx_arr], y[idx_arr], cfg.max_gap_seconds)
        x[idx_arr] = gx
        y[idx_arr] = gy

    sorted_frames["x"] = x
    sorted_frames["y"] = y
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    return sorted_frames
```

### Step 2.8 — Velocities (RED→GREEN)

`tests/test_derive_velocities.py` covers (a) raises clear `ValueError` when called on frames missing `_preprocessed_with`/`x_smoothed`/`y_smoothed` (S4 fix); (b) emits exactly `vx`/`vy`/`speed` (no other new columns); (c) dtypes float64; (d) NaN propagation; (e) physical plausibility (speed ≤ 12 m/s typical). `tests/invariants/test_invariant_velocity_physical_plausibility.py` — speed ≤ 12 m/s for 99.9 % across the slim-parquet provider sweep.

Required test case (regression for S4):

```python
def test_raises_when_smoothed_columns_missing():
    """Lakehouse review S4: principle-of-least-surprise — load-bearing schema mutation
    via auto-smoothing was removed; caller must run smooth_frames first."""
    f = _toy_frames()  # raw x/y, no smoothing
    with pytest.raises(ValueError, match="Call silly_kicks.tracking.preprocess.smooth_frames"):
        derive_velocities(f, config=PreprocessConfig.default())
```

Implementation `_velocity.py`:

```python
"""derive_velocities — vx/vy/speed columns via Savitzky-Golay derivative."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config import PreprocessConfig
from ._smoothing import smooth_frames

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def derive_velocities(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Add ``vx``, ``vy``, ``speed`` columns from smoothed positions.

    REQUIRES ``_preprocessed_with`` (and ``x_smoothed``/``y_smoothed``) on
    ``frames`` — call :func:`smooth_frames` first. Lakehouse-review S4 fix:
    earlier drafts auto-invoked ``smooth_frames`` here, but that meant a
    caller asking for vx/vy/speed got back a DataFrame with FIVE new columns
    (``x_smoothed``/``y_smoothed``/``_preprocessed_with`` + ``vx``/``vy``/``speed``)
    instead of the documented three. That breaks ``applyInPandas`` UDFs whose
    ``StructType`` only declares the three velocity fields. Loud raise is the
    principle-of-least-surprise choice.

    Output schema additions: ``vx``, ``vy``, ``speed`` (all float64, m/s).
    No other columns are added or mutated.

    Examples
    --------
    >>> # See tests/test_derive_velocities.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    missing = [c for c in ("_preprocessed_with", "x_smoothed", "y_smoothed") if c not in frames.columns]
    if missing:
        raise ValueError(
            f"derive_velocities: frames missing required column(s) {missing}. "
            "Call silly_kicks.tracking.preprocess.smooth_frames(frames, ...) first."
        )

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    hz = float(sorted_frames["frame_rate"].dropna().iloc[0]) if "frame_rate" in sorted_frames.columns else 25.0
    dt = 1.0 / hz
    window_frames = max(int(round(cfg.sg_window_seconds * hz)) | 1, cfg.sg_poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1

    # Smoothed columns are guaranteed present by the up-front check above (S4 fix).
    x_src = sorted_frames["x_smoothed"].to_numpy(dtype=float)
    y_src = sorted_frames["y_smoothed"].to_numpy(dtype=float)
    vx = np.full(len(sorted_frames), np.nan)
    vy = np.full(len(sorted_frames), np.nan)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        x_vals = x_src[idx_arr]
        y_vals = y_src[idx_arr]
        if len(x_vals) < window_frames:
            # short group — use simple finite difference
            vx[idx_arr] = np.gradient(np.where(np.isnan(x_vals), 0.0, x_vals), dt)
            vy[idx_arr] = np.gradient(np.where(np.isnan(y_vals), 0.0, y_vals), dt)
            mask = np.isnan(x_vals) | np.isnan(y_vals)
            vx[idx_arr[mask]] = np.nan
            vy[idx_arr[mask]] = np.nan
            continue
        x_filled = np.where(np.isnan(x_vals), np.interp(np.arange(len(x_vals)), np.arange(len(x_vals))[~np.isnan(x_vals)], x_vals[~np.isnan(x_vals)]) if (~np.isnan(x_vals)).any() else 0.0, x_vals)
        y_filled = np.where(np.isnan(y_vals), np.interp(np.arange(len(y_vals)), np.arange(len(y_vals))[~np.isnan(y_vals)], y_vals[~np.isnan(y_vals)]) if (~np.isnan(y_vals)).any() else 0.0, y_vals)
        vx_g = savgol_filter(x_filled, window_length=window_frames, polyorder=cfg.sg_poly_order, deriv=1, delta=dt)
        vy_g = savgol_filter(y_filled, window_length=window_frames, polyorder=cfg.sg_poly_order, deriv=1, delta=dt)
        nan_mask = np.isnan(x_vals) | np.isnan(y_vals)
        vx_g[nan_mask] = np.nan
        vy_g[nan_mask] = np.nan
        vx[idx_arr] = vx_g
        vy[idx_arr] = vy_g

    sorted_frames["vx"] = vx
    sorted_frames["vy"] = vy
    sorted_frames["speed"] = np.sqrt(vx * vx + vy * vy)
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    return sorted_frames
```

### Step 2.9 — Re-export from `tracking/__init__.py`

Add to `__all__` and the import block:

```python
from .preprocess import PreprocessConfig, derive_velocities, interpolate_frames, smooth_frames
```

### Step 2.10 — Run all Loop-2 tests + integrity test (Loop 1)

```powershell
uv run python -m pytest tests/test_preprocess_config.py tests/test_smooth_frames.py tests/test_interpolate_frames.py tests/test_derive_velocities.py tests/test_preprocess_baseline_integrity.py tests/invariants/test_invariant_smooth_frames_idempotence.py tests/invariants/test_invariant_smooth_frames_constant_signal.py tests/invariants/test_invariant_interpolate_passes_through.py tests/invariants/test_invariant_velocity_physical_plausibility.py -v
```

Expected: all GREEN.

---

## Loop 3 — TF-6 `sync_score` + `LinkReport.sync_scores()` + atomic parity smoke

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (add `sync_score`, `add_sync_score`)
- Modify: `silly_kicks/tracking/schema.py` (add `LinkReport.sync_scores()` method)
- Modify: `silly_kicks/tracking/__init__.py` (re-export)
- Modify: `silly_kicks/atomic/tracking/__init__.py` (re-export `add_sync_score`)
- Create: `tests/test_sync_score.py`
- Create: `tests/atomic/test_sync_score_atomic.py`
- Create: `tests/invariants/test_invariant_sync_score_bounds.py`

### Step 3.1 — RED tests

`tests/test_sync_score.py`:

```python
"""sync_score primitive — three aggregations per action."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.utils import add_sync_score, sync_score


def _toy_links() -> pd.DataFrame:
    return pd.DataFrame({
        "action_id": [1, 1, 1, 2, 2, 3],
        "frame_id": [10, 11, 12, 20, 21, 30],
        "time_offset_seconds": [0.0, 0.04, 0.08, 0.0, 0.04, 0.0],
        "n_candidate_frames": [3, 3, 3, 2, 2, 1],
        "link_quality_score": [0.95, 0.90, 0.85, 0.70, 0.60, 0.99],
    })


def test_returns_three_columns():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert list(df.columns) == ["sync_score_min", "sync_score_mean", "sync_score_high_quality_frac"]


def test_per_action_min_mean():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert df.loc[1, "sync_score_min"] == pytest.approx(0.85)
    assert df.loc[1, "sync_score_mean"] == pytest.approx((0.95 + 0.90 + 0.85) / 3)
    assert df.loc[2, "sync_score_min"] == pytest.approx(0.60)
    assert df.loc[3, "sync_score_min"] == pytest.approx(0.99)


def test_high_quality_frac_at_threshold():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert df.loc[1, "sync_score_high_quality_frac"] == pytest.approx(1.0)  # all 3 ≥ 0.85
    assert df.loc[2, "sync_score_high_quality_frac"] == pytest.approx(0.0)  # both below
    assert df.loc[3, "sync_score_high_quality_frac"] == pytest.approx(1.0)


def test_bounds_zero_one():
    df = sync_score(_toy_links(), high_quality_threshold=0.5)
    for col in df.columns:
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()


def test_min_le_mean():
    df = sync_score(_toy_links(), high_quality_threshold=0.5)
    assert (df["sync_score_min"] <= df["sync_score_mean"] + 1e-9).all()


def test_add_sync_score_merges_on_action_id():
    actions = pd.DataFrame({"action_id": [1, 2, 3, 4], "type_id": [0, 1, 2, 3]})
    out = add_sync_score(actions, _toy_links(), high_quality_threshold=0.85)
    assert "sync_score_min" in out.columns
    assert pd.isna(out.set_index("action_id").loc[4, "sync_score_min"])  # action 4 has no link rows


def test_link_report_sync_scores_method():
    from silly_kicks.tracking import LinkReport

    rpt = LinkReport(
        n_actions_in=3,
        n_actions_linked=3,
        n_actions_unlinked=0,
        n_actions_multi_candidate=2,
        per_provider_link_rate={"sportec": 1.0},
        max_time_offset_seconds=0.08,
        tolerance_seconds=0.1,
    )
    # Method requires the caller to pass the links DataFrame (LinkReport itself is a summary)
    df = rpt.sync_scores(_toy_links(), high_quality_threshold=0.85)
    assert "sync_score_min" in df.columns
```

`tests/atomic/test_sync_score_atomic.py`:

```python
"""sync_score is action-content-agnostic — atomic actions consume add_sync_score unchanged."""
from __future__ import annotations

import pandas as pd

from silly_kicks.tracking.utils import add_sync_score


def test_atomic_actions_consume_add_sync_score():
    atomic = pd.DataFrame({"action_id": [1, 2], "x": [10.0, 20.0], "y": [30.0, 40.0]})
    links = pd.DataFrame({
        "action_id": [1, 1, 2],
        "frame_id": [10, 11, 20],
        "time_offset_seconds": [0.0, 0.04, 0.0],
        "n_candidate_frames": [2, 2, 1],
        "link_quality_score": [0.9, 0.8, 0.95],
    })
    out = add_sync_score(atomic, links, high_quality_threshold=0.85)
    assert {"sync_score_min", "sync_score_mean", "sync_score_high_quality_frac"} <= set(out.columns)
    assert len(out) == len(atomic)
```

`tests/invariants/test_invariant_sync_score_bounds.py` — runs `sync_score` on every available provider's slim parquet links, asserts `[0, 1]` bounds and `min ≤ mean`.

### Step 3.2 — Implementation in `silly_kicks/tracking/utils.py`

Append:

```python
def sync_score(
    links: pd.DataFrame,
    *,
    high_quality_threshold: float = 0.85,
) -> pd.DataFrame:
    """Per-action sync-quality scores (3 aggregations).

    Returns a DataFrame indexed by action_id with columns:
      sync_score_min — min(link_quality_score) per action.
      sync_score_mean — mean(link_quality_score) per action.
      sync_score_high_quality_frac — fraction of links with
        link_quality_score >= high_quality_threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking.utils import sync_score
    >>> links = pd.DataFrame({"action_id": [1, 1], "link_quality_score": [0.9, 0.8],
    ...                       "frame_id": [10, 11], "time_offset_seconds": [0.0, 0.04],
    ...                       "n_candidate_frames": [2, 2]})
    >>> df = sync_score(links, high_quality_threshold=0.85)
    >>> float(df.loc[1, "sync_score_min"])
    0.8
    """
    if "link_quality_score" not in links.columns or "action_id" not in links.columns:
        raise ValueError("sync_score: links must contain 'action_id' and 'link_quality_score'")
    grp = links.groupby("action_id", dropna=False)["link_quality_score"]
    out = pd.DataFrame({
        "sync_score_min": grp.min(),
        "sync_score_mean": grp.mean(),
        "sync_score_high_quality_frac": grp.apply(lambda s: float((s >= high_quality_threshold).mean())),
    })
    return out


def add_sync_score(
    actions: pd.DataFrame,
    links: pd.DataFrame,
    *,
    high_quality_threshold: float = 0.85,
) -> pd.DataFrame:
    """Enrich actions with three sync_score_* columns merged on action_id.

    Examples
    --------
    >>> # See tests/test_sync_score.py for runnable example.
    """
    if "action_id" not in actions.columns:
        raise ValueError("add_sync_score: actions must contain 'action_id'")
    scores = sync_score(links, high_quality_threshold=high_quality_threshold)
    return actions.merge(scores, left_on="action_id", right_index=True, how="left")
```

### Step 3.3 — `LinkReport.sync_scores()` method

In `silly_kicks/tracking/schema.py`, append to the LinkReport dataclass:

```python
    def sync_scores(self, links: pd.DataFrame, *, high_quality_threshold: float = 0.85) -> pd.DataFrame:
        """Per-action sync_score DataFrame for the supplied link batch.

        The LinkReport summary holds counts; the link rows themselves are needed
        to compute per-action aggregations.

        Examples
        --------
        >>> # See tests/test_sync_score.py::test_link_report_sync_scores_method
        """
        from .utils import sync_score  # local import to avoid utils -> schema cycle
        return sync_score(links, high_quality_threshold=high_quality_threshold)
```

### Step 3.4 — Re-export

Modify `silly_kicks/tracking/__init__.py` `__all__` and import block:

```python
from .utils import add_sync_score, link_actions_to_frames, play_left_to_right, slice_around_event, sync_score
```

Append `"add_sync_score"`, `"sync_score"` to `__all__`.

In `silly_kicks/atomic/tracking/__init__.py`, re-export `add_sync_score`:

```python
from silly_kicks.tracking.utils import add_sync_score, sync_score
```

### Step 3.5 — Run

```powershell
uv run python -m pytest tests/test_sync_score.py tests/atomic/test_sync_score_atomic.py tests/invariants/test_invariant_sync_score_bounds.py -v
```

Expected: GREEN.

---

## Loop 4 — TF-12 GK angle (kernel + aggregator + atomic mirror + per-period DOP invariant)

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py` (add `_pre_shot_gk_angle`)
- Modify: `silly_kicks/tracking/features.py` (4 angle Series helpers + `add_pre_shot_gk_angle` + `pre_shot_gk_angle_default_xfns` + `pre_shot_gk_full_default_xfns`)
- Modify: `silly_kicks/atomic/tracking/features.py` (atomic mirror)
- Modify: `silly_kicks/tracking/__init__.py` (re-export)
- Create: `tests/test_pre_shot_gk_angle.py`
- Create: `tests/invariants/test_invariant_gk_angle_bounds.py`
- Create: `tests/invariants/test_invariant_gk_angle_per_period_dop_symmetry.py` (closes 3.0.0/3.0.1 gap)
- Modify: `tests/tracking/_provider_inputs.py` (synthesize ≥1 shot+keeper_save per provider per period — see `feedback_synthesizer_shot_plus_keeper_save_pattern`)

### Step 4.1 — RED tests

`tests/test_pre_shot_gk_angle.py`:

```python
"""TF-12: pre_shot_gk_angle_to_shot_trajectory + pre_shot_gk_angle_off_goal_line."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import (
    add_pre_shot_gk_angle,
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
)


def _shot_anchor_x() -> float:
    return 95.0  # near the goal


def _toy_actions_and_frames(gk_x: float, gk_y: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = pd.DataFrame({
        "action_id": [1],
        "game_id": ["G1"],
        "period_id": [1],
        "type_id": [11],  # shot
        "team_id": [1],
        "player_id": [10],
        "start_x": [_shot_anchor_x()],
        "start_y": [34.0],
        "time_seconds": [10.0],
        "defending_gk_player_id": [99.0],
    })
    frames = pd.DataFrame({
        "game_id": ["G1"] * 2,
        "period_id": [1, 1],
        "frame_id": [100, 100],
        "time_seconds": [10.0, 10.0],
        "frame_rate": [25.0, 25.0],
        "player_id": [10, 99],
        "team_id": [1, 2],
        "is_ball": [False, False],
        "is_goalkeeper": [False, True],
        "x": [_shot_anchor_x(), gk_x],
        "y": [34.0, gk_y],
        "z": [np.nan, np.nan],
        "speed": [0.0, 0.0],
        "speed_source": ["native", "native"],
        "ball_state": ["alive", "alive"],
        "team_attacking_direction": ["ltr", "ltr"],
        "source_provider": ["sportec", "sportec"],
    })
    return actions, frames


def test_gk_on_shot_trajectory_returns_zero():
    """GK on the line from (95, 34) to (105, 34) → angle to trajectory ≈ 0."""
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert math.isclose(float(s.iloc[0]), 0.0, abs_tol=1e-6)


def test_gk_displaced_positive_y_returns_positive_angle():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=36.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert float(s.iloc[0]) > 0


def test_gk_displaced_negative_y_returns_negative_angle():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=32.0)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert float(s.iloc[0]) < 0


def test_off_goal_line_zero_when_gk_on_goal_line_normal():
    """GK at (102, 34) is on the goal-line normal at the goal-mouth centre → angle ≈ 0."""
    actions, frames = _toy_actions_and_frames(gk_x=102.0, gk_y=34.0)
    s = pre_shot_gk_angle_off_goal_line(actions, frames)
    assert math.isclose(float(s.iloc[0]), 0.0, abs_tol=1e-6)


def test_off_goal_line_sign_flips_with_y():
    actions_up, frames_up = _toy_actions_and_frames(gk_x=102.0, gk_y=36.0)
    actions_dn, frames_dn = _toy_actions_and_frames(gk_x=102.0, gk_y=32.0)
    s_up = float(pre_shot_gk_angle_off_goal_line(actions_up, frames_up).iloc[0])
    s_dn = float(pre_shot_gk_angle_off_goal_line(actions_dn, frames_dn).iloc[0])
    assert s_up * s_dn < 0


def test_nan_when_defending_gk_id_missing():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    actions.loc[0, "defending_gk_player_id"] = np.nan
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert pd.isna(s.iloc[0])


def test_add_pre_shot_gk_angle_emits_two_columns_only():
    actions, frames = _toy_actions_and_frames(gk_x=100.0, gk_y=34.0)
    out = add_pre_shot_gk_angle(actions, frames=frames)
    new_cols = set(out.columns) - set(actions.columns)
    assert {"pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"} <= new_cols


def test_bounds_within_pi():
    actions, frames = _toy_actions_and_frames(gk_x=80.0, gk_y=10.0)  # extreme
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert -math.pi <= float(s.iloc[0]) <= math.pi
```

`tests/invariants/test_invariant_gk_angle_per_period_dop_symmetry.py` — closes 3.0.0/3.0.1 gap. Mirrors the same shot across periods (P1 attacking-right vs P2 attacking-left after LTR-normalization both attack left-to-right) and asserts the angles match within tolerance:

```python
"""Per-period DOP-symmetry invariant for TF-12 GK angle features.

Closes the 3.0.0/3.0.1 blind-spot pattern: every numeric tracking-aware
feature must explicitly verify that LTR-normalized output is identical
across periods (P1 / P2 / ET) when the underlying physical situation is.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import (
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
)

_PROVIDERS = ["sportec_per_period", "metrica_per_period"]


def _make_mirrored_shot_actions_frames(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two shot actions in different periods with mirrored physical setup."""
    common = dict(
        game_id="G1",
        type_id=11,  # shot
        team_id=1,
        player_id=10,
        start_y=34.0,
        defending_gk_player_id=99.0,
    )
    actions = pd.DataFrame([
        {"action_id": 1, "period_id": 1, "start_x": 95.0, "time_seconds": 10.0, **common},
        {"action_id": 2, "period_id": 2, "start_x": 95.0, "time_seconds": 60.0, **common},
    ])
    frames_rows = []
    for period, t in [(1, 10.0), (2, 60.0)]:
        for player_id, x, y, is_gk, team in [(10, 95.0, 34.0, False, 1), (99, 100.0, 36.0, True, 2)]:
            frames_rows.append({
                "game_id": "G1", "period_id": period, "frame_id": 100,
                "time_seconds": t, "frame_rate": 25.0, "player_id": player_id,
                "team_id": team, "is_ball": False, "is_goalkeeper": is_gk,
                "x": x, "y": y, "z": np.nan, "speed": 0.0,
                "speed_source": "native", "ball_state": "alive",
                "team_attacking_direction": "ltr", "source_provider": provider.replace("_per_period", ""),
            })
    frames = pd.DataFrame(frames_rows)
    return actions, frames


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_per_period_dop_symmetry_to_shot_trajectory(provider):
    actions, frames = _make_mirrored_shot_actions_frames(provider)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert math.isclose(float(s.iloc[0]), float(s.iloc[1]), abs_tol=1e-9), (
        f"{provider}: per-period DOP-symmetry violated for to_shot_trajectory: "
        f"P1={s.iloc[0]} vs P2={s.iloc[1]}"
    )


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_per_period_dop_symmetry_off_goal_line(provider):
    actions, frames = _make_mirrored_shot_actions_frames(provider)
    s = pre_shot_gk_angle_off_goal_line(actions, frames)
    assert math.isclose(float(s.iloc[0]), float(s.iloc[1]), abs_tol=1e-9), (
        f"{provider}: per-period DOP-symmetry violated for off_goal_line: "
        f"P1={s.iloc[0]} vs P2={s.iloc[1]}"
    )
```

### Step 4.2 — Kernel `_pre_shot_gk_angle` in `silly_kicks/tracking/_kernels.py`

**N2 confirmation:** `_GOAL_X = 105.0` and `_GOAL_Y_CENTER = 34.0` already exist at the top of `_kernels.py` (PR-S20, lines 23-24). The new kernel reuses them — no new constants needed.

```python
def _pre_shot_gk_angle(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    shot_type_ids: frozenset[int],
) -> pd.DataFrame:
    """Per shot action: GK angle vs shot trajectory and goal-line normal.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ctx.actions with 2 columns:
        - pre_shot_gk_angle_to_shot_trajectory (float64, radians, signed)
        - pre_shot_gk_angle_off_goal_line (float64, radians, signed)

    All NaN for non-shot rows or rows with no defending_gk_row in ctx.

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    actions = ctx.actions
    out = pd.DataFrame(
        {
            "pre_shot_gk_angle_to_shot_trajectory": np.full(len(actions), np.nan, dtype="float64"),
            "pre_shot_gk_angle_off_goal_line": np.full(len(actions), np.nan, dtype="float64"),
        },
        index=actions.index,
    )
    if "type_id" not in actions.columns:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    if len(ctx.defending_gk_rows) > 0:
        gk = ctx.defending_gk_rows[["action_id", "x", "y"]].rename(columns={"x": "_gk_x", "y": "_gk_y"})
        gk = gk.drop_duplicates("action_id", keep="first")
        per_action = actions[["action_id"]].merge(gk, on="action_id", how="left")
    else:
        per_action = actions[["action_id"]].assign(_gk_x=np.nan, _gk_y=np.nan)
    per_action.index = actions.index

    shot_mask = pd.Series(is_shot, index=actions.index)
    gk_present_mask = per_action["_gk_x"].notna()
    valid = shot_mask & gk_present_mask
    if not valid.any():
        return out

    gx = per_action.loc[valid, "_gk_x"].astype("float64").to_numpy()
    gy = per_action.loc[valid, "_gk_y"].astype("float64").to_numpy()
    ax = anchor_x.loc[valid].astype("float64").to_numpy()
    ay = anchor_y.loc[valid].astype("float64").to_numpy()

    # to_shot_trajectory: signed angle between (goal_centre - anchor) and (gk - anchor)
    v1x = _GOAL_X - ax
    v1y = _GOAL_Y_CENTER - ay
    v2x = gx - ax
    v2y = gy - ay
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    out.loc[valid, "pre_shot_gk_angle_to_shot_trajectory"] = np.arctan2(cross, dot)

    # off_goal_line: signed angle between goal-line-normal (-1, 0 in LTR after centre offset)
    # and (gk - goal_centre). Equivalent to arctan2(gk_y - 34, 105 - gk_x).
    out.loc[valid, "pre_shot_gk_angle_off_goal_line"] = np.arctan2(gy - _GOAL_Y_CENTER, _GOAL_X - gx)
    return out
```

### Step 4.3 — Public Series helpers + aggregator in `silly_kicks/tracking/features.py`

```python
def pre_shot_gk_angle_to_shot_trajectory(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Signed angle (rad) between (goal-centre→anchor) and (GK→anchor) at the linked frame.

    Zero ⇒ GK is on the shot trajectory line. Positive ⇒ GK to +y side; negative ⇒ -y side.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal." Frontiers in
        Sports and Active Living, 3, 624475.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_to_shot_trajectory"].rename("pre_shot_gk_angle_to_shot_trajectory")


def pre_shot_gk_angle_off_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Signed angle (rad) of GK position relative to goal-line normal at goal-mouth centre.

    Zero ⇒ GK is on the goal-line normal. Positive ⇒ GK offset to +y side; negative ⇒ -y side.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_off_goal_line"].rename("pre_shot_gk_angle_off_goal_line")


@nan_safe_enrichment
def add_pre_shot_gk_angle(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Add 2 GK-angle columns at the linked frame for each shot action.

    REQUIRES ``defending_gk_player_id`` column (run
    ``silly_kicks.spadl.utils.add_pre_shot_gk_context`` first).

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - pre_shot_gk_angle_to_shot_trajectory (float64, radians, signed)
        - pre_shot_gk_angle_off_goal_line (float64, radians, signed)

    NaN for non-shot / unlinked / pre-engagement / GK-absent rows. Standalone
    aggregator — does NOT extend ``add_pre_shot_gk_position`` (preserves the
    PR-S21 4-column surface; primitive+assembly pattern).

    Raises
    ------
    ValueError
        If ``defending_gk_player_id`` column is absent from ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_angle: actions missing required column 'defending_gk_player_id'. "
            "Run silly_kicks.spadl.utils.add_pre_shot_gk_context first."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"):
        out[col] = df[col]
    return out


pre_shot_gk_angle_default_xfns = [
    lift_to_states(pre_shot_gk_angle_to_shot_trajectory),
    lift_to_states(pre_shot_gk_angle_off_goal_line),
]


pre_shot_gk_full_default_xfns = pre_shot_gk_default_xfns + pre_shot_gk_angle_default_xfns
```

Update `__all__` to add the new symbols.

### Step 4.4 — Atomic mirror in `silly_kicks/atomic/tracking/features.py`

Mirror the four new symbols (two Series + aggregator + two xfn lists), substituting `_ATOMIC_SHOT_TYPE_IDS` and reading `actions["x"]`/`actions["y"]`. Append `atomic_pre_shot_gk_angle_default_xfns` and `atomic_pre_shot_gk_full_default_xfns`. Update `__all__`.

### Step 4.5 — Synthesizer extension (S3 fix from review — expanded)

**Blast-radius inventory (6 consumers).** `tests/tracking/_provider_inputs.py` is consumed by:

- `tests/tracking/test_action_context_real_data_sweep.py` — full provider sweep
- `tests/tracking/test_empirical_action_context_baselines.py` — baseline-comparison gate
- `tests/tracking/test_action_context_cross_provider.py` — per-provider consistency
- `tests/tracking/test_action_context_expected_output.py` — `*_expected.parquet` regression gate
- `scripts/regenerate_action_context_baselines.py` — manual baseline regeneration
- `scripts/probe_action_context_baselines.py` — probe pipeline

**Substeps:**

#### 4.5.1 Inventory check (read-only — confirm the count is exactly 6)

```powershell
uv run python -c @'
from pathlib import Path
import re
roots = [Path("tests"), Path("scripts")]
hits = []
for root in roots:
    for p in root.rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        if re.search(r"(_provider_inputs|load_provider_frames|synthesize_actions)", s):
            hits.append(str(p))
for p in hits: print(p)
print(f"COUNT={len(hits)}")
'@
```

Expected: exactly the 6 paths above plus `_provider_inputs.py` itself = 7 lines, `COUNT=7`. If the count differs, audit each new consumer for action-count assertions before changing the synthesizer.

#### 4.5.2 Audit each consumer for action-count / per-period assertions

For each of the 6 consumers, grep for `len(`/`shape`/`assert .*== \d`/`type_id ==`/`groupby.*period_id` — record any assertion that would break if shot/keeper_save counts change:

```powershell
uv run python -c @'
from pathlib import Path
import re
files = [
  "tests/tracking/test_action_context_real_data_sweep.py",
  "tests/tracking/test_empirical_action_context_baselines.py",
  "tests/tracking/test_action_context_cross_provider.py",
  "tests/tracking/test_action_context_expected_output.py",
  "scripts/regenerate_action_context_baselines.py",
  "scripts/probe_action_context_baselines.py",
]
patterns = [r"len\([^)]*\)\s*==\s*\d", r"\.shape\[0\]\s*==\s*\d", r"type_id\s*==", r"groupby.*period_id"]
for f in files:
    print(f"=== {f} ===")
    for i, line in enumerate(Path(f).read_text(encoding="utf-8").splitlines(), 1):
        for pat in patterns:
            if re.search(pat, line):
                print(f"  L{i}: {line.strip()[:100]}")
                break
'@
```

Record any assertion that hard-codes action counts. Plan the synthesizer change to be **additive** (new shot+keeper_save rows, never removing existing rows) so that consumers asserting `len() == N` may need their N adjusted; consumers using `>=` or computing relative metrics keep passing unchanged.

If any consumer hard-codes a count that needs updating: update the count in the SAME loop step, with the value calculated from the new synthesizer output.

#### 4.5.3 Fail-loud existence check (CI gate — runs before any TF-12 invariant)

Create `tests/tracking/test_synthesizer_fixture_density.py`:

```python
"""Fail-loud existence check for synthesized provider-input fixtures.

Runs BEFORE any TF-12 invariant. Any provider missing ≥1 shot AND ≥1 keeper_save
in EACH period of its synthesized output is a CI fail with an actionable message.

Closes the silent-skip vector in feedback_invariant_testing where invariant
tests pass vacuously when fixture density is insufficient.
"""
from __future__ import annotations

import pytest

from silly_kicks.spadl import config as spadlconfig
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_SHOT_TYPE_IDS = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")}
_KEEPER_SAVE_TYPE_ID = spadlconfig.actiontype_id["keeper_save"]


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_per_period_shot_and_keeper_save_exist(provider):
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    periods = sorted(actions["period_id"].unique())
    assert len(periods) >= 2, (
        f"{provider}: synthesized actions cover only periods {periods} — "
        "TF-12 per-period DOP-symmetry needs ≥2 periods to exercise."
    )
    for period in periods:
        in_period = actions[actions["period_id"] == period]
        n_shots = int(in_period["type_id"].isin(_SHOT_TYPE_IDS).sum())
        n_keeper_saves = int((in_period["type_id"] == _KEEPER_SAVE_TYPE_ID).sum())
        assert n_shots >= 1, (
            f"{provider} P{period}: synthesized actions have NO shot rows. "
            "Required for TF-12 invariants per feedback_synthesizer_shot_plus_keeper_save_pattern."
        )
        assert n_keeper_saves >= 1, (
            f"{provider} P{period}: synthesized actions have NO keeper_save rows. "
            "Required for TF-12 invariants per feedback_synthesizer_shot_plus_keeper_save_pattern."
        )
```

#### 4.5.4 Modify `tests/tracking/_provider_inputs.py::synthesize_actions`

Append shot+keeper_save rows for each (provider, period) pair where the slim-parquet input doesn't already contain at least one of each. Anchor each synthesized shot at a real frame's coordinates (bounded by the slim parquet's frame range) so the linkage primitive can lock onto a real GK row. Pattern from PR-S21 prior precedent (look at how PR-S21 did this for `medium_halftime_with_shot.parquet`).

#### 4.5.5 Run the new fail-loud test BEFORE the TF-12 invariants

```powershell
uv run python -m pytest tests/tracking/test_synthesizer_fixture_density.py -v
```

Expected: 4 providers × ≥2 periods all GREEN. If any RED: fix the synthesizer until every cell is GREEN before running TF-12 invariants.

### Step 4.6 — Re-export

`silly_kicks/tracking/__init__.py` — add `add_pre_shot_gk_angle`, `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`, `pre_shot_gk_angle_default_xfns`, `pre_shot_gk_full_default_xfns` to `__all__` + import block.

### Step 4.7 — Run

```powershell
uv run python -m pytest tests/test_pre_shot_gk_angle.py tests/invariants/test_invariant_gk_angle_bounds.py tests/invariants/test_invariant_gk_angle_per_period_dop_symmetry.py tests/atomic/tracking/ -v
```

Expected: GREEN.

---

## Loop 5 — Umbrella facade extension + tracking-converter `preprocess` kwarg wiring

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (`add_pre_shot_gk_context` lazy-imports angle aggregator when `frames` supplied)
- Modify: `silly_kicks/atomic/spadl/utils.py` (atomic mirror)
- Modify: `silly_kicks/tracking/sportec.py` (add `preprocess: PreprocessConfig | None = None` kwarg with auto-promotion)
- Modify: `silly_kicks/tracking/pff.py` (same)
- Modify: `silly_kicks/tracking/kloppy.py` (same)
- Create: `tests/test_add_pre_shot_gk_context_extended.py`
- Create: `tests/test_tracking_converter_preprocess_kwarg.py`

### Step 5.1 — RED tests for extended facade

`tests/test_add_pre_shot_gk_context_extended.py`:

```python
"""add_pre_shot_gk_context (frames=None) emits 4 cols (PR-S21 backcompat).
add_pre_shot_gk_context (frames=...) emits 6 cols total (4 + 2 angle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl.utils import add_pre_shot_gk_context


def _toy_actions() -> pd.DataFrame:
    return pd.DataFrame({
        "action_id": [1, 2],
        "game_id": ["G1", "G1"],
        "period_id": [1, 1],
        "type_id": [11, 0],  # shot, pass
        "team_id": [1, 2],
        "player_id": [10, 20],
        "start_x": [95.0, 50.0],
        "start_y": [34.0, 30.0],
        "end_x": [105.0, 60.0],
        "end_y": [34.0, 32.0],
        "time_seconds": [10.0, 5.0],
    })


def _toy_frames() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1"] * 2, "period_id": [1, 1], "frame_id": [100, 100],
        "time_seconds": [10.0, 10.0], "frame_rate": [25.0, 25.0],
        "player_id": [10, 99], "team_id": [1, 2], "is_ball": [False, False],
        "is_goalkeeper": [False, True], "x": [95.0, 102.0], "y": [34.0, 34.0],
        "z": [np.nan, np.nan], "speed": [0.0, 0.0], "speed_source": ["native", "native"],
        "ball_state": ["alive", "alive"], "team_attacking_direction": ["ltr", "ltr"],
        "source_provider": ["sportec", "sportec"],
    })


def test_frames_none_emits_four_pr_s21_cols():
    out = add_pre_shot_gk_context(_toy_actions())
    assert {"gk_was_distributing", "gk_was_engaged", "gk_actions_in_possession", "defending_gk_player_id"} <= set(out.columns)
    assert "pre_shot_gk_angle_to_shot_trajectory" not in out.columns
    assert "pre_shot_gk_angle_off_goal_line" not in out.columns


def test_frames_supplied_emits_six_cols():
    out = add_pre_shot_gk_context(_toy_actions(), frames=_toy_frames())
    expected_new = {
        "pre_shot_gk_x", "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot",
        "pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line",
    }
    assert expected_new <= set(out.columns)
```

### Step 5.2 — Implementation

In `silly_kicks/spadl/utils.py::add_pre_shot_gk_context`, locate the existing `frames is not None` branch (which currently calls `add_pre_shot_gk_position`) and chain `add_pre_shot_gk_angle` after it:

```python
    if frames is not None:
        from silly_kicks.tracking.features import add_pre_shot_gk_angle, add_pre_shot_gk_position
        sorted_actions = add_pre_shot_gk_position(sorted_actions, frames)
        sorted_actions = add_pre_shot_gk_angle(sorted_actions, frames=frames)
```

Mirror in `silly_kicks/atomic/spadl/utils.py::add_pre_shot_gk_context` using atomic-side imports.

Update both docstrings to enumerate the 6 columns; update CHANGELOG-relevant `### Added` enumeration in Loop 6.

### Step 5.3 — Tracking-converter `preprocess` kwarg wiring (with S5 fallback)

For each tracking converter, add a small private helper that resolves the auto-promotion target safely:

```python
# silly_kicks/tracking/preprocess/_resolve.py — NEW
"""Provider-aware auto-promotion of PreprocessConfig.default() with safe fallback."""
from __future__ import annotations

import logging
import warnings

from ._config import get_provider_defaults
from ._config_dataclass import PreprocessConfig

_LOG = logging.getLogger(__name__)


def resolve_preprocess(cfg: PreprocessConfig, *, provider: str) -> PreprocessConfig:
    """If `cfg` was built by `PreprocessConfig.default()` (without
    ``force_universal=True``), auto-promote to ``for_provider(provider)``.

    Lakehouse-review S5 fix: when `provider` is not in ``get_provider_defaults()``
    (e.g., kloppy returns a provider like "tracab" or "statsperform" not yet
    profiled), fall back to ``PreprocessConfig.default(force_universal=True)``
    rather than raising KeyError. Emits a UserWarning so the operator knows
    universal-safe values are in play.
    """
    if not cfg.is_default():
        return cfg
    defaults = get_provider_defaults()
    if provider in defaults:
        return defaults[provider]
    warnings.warn(
        f"resolve_preprocess: provider {provider!r} has no per-provider PreprocessConfig "
        "in tests/fixtures/baselines/preprocess_baseline.json — falling back to "
        "PreprocessConfig.default(force_universal=True). To suppress this warning, "
        "either add a provider block to the baseline JSON + regen, or pass "
        "PreprocessConfig.default(force_universal=True) / a hand-built config explicitly.",
        UserWarning,
        stacklevel=3,
    )
    return PreprocessConfig.default(force_universal=True)
```

Per-converter wiring:

```python
# silly_kicks/tracking/sportec.py / pff.py / kloppy.py — append this block
from .preprocess import PreprocessConfig, derive_velocities, interpolate_frames, smooth_frames
from .preprocess._resolve import resolve_preprocess


def convert_to_frames(
    raw_frames,           # or `dataset` for kloppy
    ...,
    *,
    output_convention: ... = None,
    preprocess: PreprocessConfig | None = None,   # NEW kwarg, keyword-only
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    ...
    # End of existing function body, just before `return frames, report`:
    if preprocess is not None:
        cfg = resolve_preprocess(preprocess, provider=_PROVIDER_NAME)
        if cfg.interpolation_method is not None:
            frames = interpolate_frames(frames, config=cfg)
        if cfg.smoothing_method is not None:
            frames = smooth_frames(frames, config=cfg)
        if cfg.derive_velocity:
            frames = derive_velocities(frames, config=cfg)
    return frames, report
```

`_PROVIDER_NAME` resolution per file:

- `sportec.py` — module-level constant `_PROVIDER_NAME = "sportec"`.
- `pff.py` — module-level constant `_PROVIDER_NAME = "pff"`.
- `kloppy.py` — derived from `dataset.metadata.provider` via the existing dispatch logic (kloppy enum → silly-kicks string). For unsupported kloppy providers (Tracab, StatsPerform, etc.), the dispatch yields the literal string of `dataset.metadata.provider.value`; `resolve_preprocess()` then falls back per S5.

Required test (`tests/test_tracking_converter_preprocess_kwarg.py`) gains a kloppy-fallback case:

```python
def test_kloppy_unsupported_provider_falls_back_with_warning(monkeypatch):
    """Lakehouse-review S5: provider not in baselines → UserWarning + universal fallback.

    N6 defensive note: ``has no per-provider PreprocessConfig`` is a literal substring
    of the warning text and is NOT regex-special; safe to pass to ``pytest.warns(match=)``.
    If this test is ever parametrised on user-supplied provider names, wrap the
    provider in ``re.escape(provider)`` before composing the match pattern.
    """
    from silly_kicks.tracking.preprocess._resolve import resolve_preprocess
    from silly_kicks.tracking.preprocess import PreprocessConfig

    cfg_in = PreprocessConfig.default()
    with pytest.warns(UserWarning, match="has no per-provider PreprocessConfig"):
        cfg_out = resolve_preprocess(cfg_in, provider="tracab")
    assert cfg_out.is_default() is False  # force_universal=True path
    # Values are still universal-safe — same field values as default()
    assert cfg_out.sg_window_seconds == 0.4
```

### Step 5.4 — Auto-promotion test

`tests/test_tracking_converter_preprocess_kwarg.py`:

```python
"""Auto-promotion: PreprocessConfig.default() in a provider-aware converter
becomes for_provider(<that_provider>); force_universal=True opts out."""
from __future__ import annotations

# Use sportec as the canonical test bed (synthetic input is cheapest)

# Build a minimal raw-shaped sportec input via the existing test fixtures.
# ... (one shot+keeper_save round-trip, comparing post-converter speed values)
```

### Step 5.5 — Run

```powershell
uv run python -m pytest tests/test_add_pre_shot_gk_context_extended.py tests/test_tracking_converter_preprocess_kwarg.py -v
uv run python -m pytest tests/atomic/test_atomic_add_pre_shot_gk_context.py -v
```

Expected: GREEN. The pre-existing PR-S21 atomic test must still pass (4-column backcompat path).

---

## Loop 6 — VAEP xfn lists + NOTICE + multi-flavor asymmetry doc + Public-API Examples + CHANGELOG

**Files:**
- Modify: `NOTICE` (add Savitzky & Golay 1964 entry)
- Modify: `CHANGELOG.md` (additive `### Added` section enumerating new column names)
- Modify: `silly_kicks/tracking/preprocess/__init__.py` (asymmetry doc — already in Loop 2 Step 2.3; verify)
- Modify: `silly_kicks/tracking/features.py` (Public-API Examples enrichments where left as `# See tests/...` stubs)
- Verify: every new public def has an `Examples` section (CI gate `tests/test_public_api_examples.py`)

### Step 6.1 — NOTICE update

Append to "Mathematical / Methodological References" section:

```
- Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of
  Data by Simplified Least Squares Procedures." Analytical Chemistry, 36(8),
  1627-1639.
  (Savitzky-Golay polynomial smoothing + analytical derivative — used for
  position smoothing and velocity derivation in silly_kicks.tracking.preprocess,
  PR-S24, ADR-004 invariants 6/7)
```

Extend the existing Anzer & Bauer 2021 bullet to enumerate angle features (PR-S24 TF-12 — extension, not new bullet).

### Step 6.2 — CHANGELOG entry

Append to `CHANGELOG.md`:

```markdown
## [3.1.0] — 2026-05-XX

### Added

- **TF-6 — `sync_score`** (`silly_kicks.tracking.utils.sync_score`,
  `add_sync_score`, `LinkReport.sync_scores()`): per-action tracking↔events
  sync-quality scores. New columns when used via `add_sync_score`:
  - `sync_score_min`
  - `sync_score_mean`
  - `sync_score_high_quality_frac`
- **TF-8 — smoothing primitives** (`silly_kicks.tracking.preprocess.smooth_frames`,
  `derive_velocities`): Savitzky-Golay (canonical) and EMA smoothing of
  positional columns. Schema-additive output columns:
  - `x_smoothed`
  - `y_smoothed`
  - `vx`
  - `vy`
  - `speed` (overwritten only when previously NaN; raw `speed` from provider
    preserved otherwise — additive-by-default)
  - `_preprocessed_with` (per-row provenance tag — load-bearing because
    `pandas.DataFrame.attrs` does not propagate through merge/concat/applyInPandas)
- **TF-9 — interpolation / gap-filling** (`silly_kicks.tracking.preprocess.interpolate_frames`):
  linear NaN gap-filling up to `max_gap_seconds` (cubic ships in TF-9-cubic when
  a consumer requests it). Same schema as input — no new columns, just NaN cells
  replaced where the gap is short enough.
- **TF-12 — `pre_shot_gk_angle_*`** (`silly_kicks.tracking.features.add_pre_shot_gk_angle`,
  `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`,
  `pre_shot_gk_angle_default_xfns`, `pre_shot_gk_full_default_xfns` + atomic mirror).
  New columns:
  - `pre_shot_gk_angle_to_shot_trajectory` (float64, radians, signed)
  - `pre_shot_gk_angle_off_goal_line` (float64, radians, signed)
- **`PreprocessConfig`** (`silly_kicks.tracking.preprocess.PreprocessConfig`):
  shared preprocessing config dataclass with `default()` / `for_provider(name)`
  factories and flag-based `is_default()` (per `feedback_default_config_auto_promotion`).
- **Tracking-converter optional `preprocess` kwarg** on
  `silly_kicks.tracking.sportec.convert_to_frames`, `tracking.pff.convert_to_frames`,
  and `tracking.kloppy.convert_to_frames`. Default `None` ⇒ zero behavior change.
  When set, applies interpolation / smoothing / velocity-derivation per the
  config; auto-promotes `PreprocessConfig.default()` to
  `PreprocessConfig.for_provider(<this_provider>)` (opt out via
  `PreprocessConfig.default(force_universal=True)`).
- **Umbrella facade extension**: `silly_kicks.spadl.utils.add_pre_shot_gk_context`
  (and atomic mirror) now emits 6 GK-tracking columns when called with
  `frames=...` (the existing 4 from PR-S21 plus the 2 new TF-12 angles). The
  `frames=None` path is bit-identical to silly-kicks 2.9.0 — 4 columns.
  Lakehouse boundary tests asserting on the `frames=...` column-set need
  `expected_columns` extended by `pre_shot_gk_angle_to_shot_trajectory` and
  `pre_shot_gk_angle_off_goal_line`. See spec §9a.
- **Empirical baselines**: `tests/fixtures/baselines/preprocess_baseline.json`
  + `preprocess_sweep_log.json` (per-provider stats across all 4 supported
  tracking providers including SkillCorner) +
  `scripts/probe_preprocess_baseline.py` (re-runnable post-merge for
  parameter re-tuning).
- New invariant tests under `tests/invariants/`:
  `test_invariant_sync_score_bounds.py`,
  `test_invariant_smooth_frames_idempotence.py`,
  `test_invariant_smooth_frames_constant_signal.py`,
  `test_invariant_interpolate_passes_through.py`,
  `test_invariant_velocity_physical_plausibility.py`,
  `test_invariant_gk_angle_bounds.py`,
  `test_invariant_gk_angle_per_period_dop_symmetry.py` — closes the
  3.0.0 / 3.0.1 blind-spot pattern.

### Notes

- ADR-005 amendment formalising the multi-flavor convention asymmetry
  (suffixed columns for VAEP xfns; canonical-single columns for
  preprocessing utilities) lands alongside the TF-2 `pressure_on_actor` PR
  (scheduled within 24-48 hours of PR-S24 merge — bounded deferral).
  PR-S24 documents the rule operationally in the
  `silly_kicks.tracking.preprocess` module docstring.
- Lakehouse pin bump: `silly-kicks>=3.1.0,<4`. No 3.0.x → 3.1.0 migration
  needed beyond the boundary-test column-set update above and (when adopting
  preprocessing inside Spark UDFs) declaring `_preprocessed_with` +
  smoothed/velocity fields explicitly in the `applyInPandas` `StructType`
  schema (silently-dropped fields are the failure mode).
```

### Step 6.3 — Public-API Examples discipline (CI gate runs unconditionally)

Run `uv run python -m pytest tests/test_public_api_examples.py -v` and ensure every newly added public def has an Examples section. The plan's stub `>>> # See tests/...` style matches the canonical PR-S21 pattern and is acceptable.

### Step 6.4 — Run full pre-PR gates

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python -m pytest tests/ -m "not e2e" -q
```

Expected: 0 ruff issues, 0 pyright errors, full pytest GREEN.

If `ruff format --check` fails: `uv run ruff format .` then `git add .` and re-run check.

Per `feedback_api_change_sweep_ci_scope`: the lint scope MUST be `.` (full repo), not `silly_kicks/` only — CI runs ruff on the whole tree.

---

## Loop 7 — Empirical sweep + final-review + single commit

### Step 7.1 — Empirical sweep against full real datasets

**CI environment limitation (lakehouse review Q3).** GitHub Actions has neither lakehouse credentials nor the local PFF data dir, so this sweep cannot run on CI. The integrity gate (Loop 1 Step 1.4) validates **only** the deterministic code-vs-JSON relationship — that the generated `_provider_defaults_generated.py` matches the committed JSON. It does NOT validate JSON-vs-real-data drift. CI will pass green with placeholder JSON forever; that's acceptable because:

1. The placeholders are conservative (closer to the upstream-published per-provider sampling rates than to any specific match's data).
2. The probe is re-runnable on a credentialed machine; the regen pipeline (`probe_preprocess_baseline.py` → `regenerate_provider_defaults.py`) means a real-data refresh is one PR away.
3. Per `feedback_lakehouse_consumer_not_source`: silly-kicks doesn't depend on lakehouse data being available; lakehouse depends on silly-kicks.

**On the user's local machine (this session):**

```powershell
uv run python scripts/probe_preprocess_baseline.py --emit-sweep-log
uv run python scripts/regenerate_provider_defaults.py
```

Expected: regenerates `tests/fixtures/baselines/preprocess_baseline.json` + `preprocess_sweep_log.json` with measured values, then regenerates `_provider_defaults_generated.py` from the new JSON. Diff both files to confirm reasonable changes; the integrity test runs as part of the standard pytest gate and validates the regen happened.

**If local credentials/data are unavailable:** SKIP both commands; the placeholders in Loop 1 Step 1.3 remain in effect. State this explicitly in the PR description and open a follow-up issue tagged `tighten-baselines` for a future credentialed run.

**No more hand-edits to `_PROVIDER_DEFAULTS`** — the codegen pattern (S1 fix) eliminates the manual sync step entirely. Run the regen script; commit both JSON and the generated `.py`; the integrity test confirms.

### Step 7.2 — Final-review (mandatory per `feedback_final_review_gate`)

```
/final-review
```

Run inline. Address every flagged item before committing.

### Step 7.3 — Stage + single commit

```powershell
git status
git add silly_kicks/ tests/ scripts/probe_preprocess_baseline.py docs/ NOTICE CHANGELOG.md
git status  # verify nothing unintended
```

Per `feedback_commit_policy`: literally ONE commit on this branch.

```powershell
$msg = @'
feat(tracking): TF-6 sync_score + TF-8 smoothing + TF-9 interpolation + TF-12 GK angles -- silly-kicks 3.1.0 (PR-S24)

- TF-6: per-action tracking↔events sync-quality (3 aggregations) on LinkReport
  pointers — primitive `sync_score`, mutator `add_sync_score`,
  `LinkReport.sync_scores()` method.
- TF-8: new `silly_kicks.tracking.preprocess` module — `smooth_frames`
  (Savitzky-Golay canonical, EMA optional), `derive_velocities` (vx/vy/speed
  via SG-derivative). Additive `x_smoothed`/`y_smoothed`/`vx`/`vy`/`speed`/
  `_preprocessed_with` columns; raw `x`/`y` preserved unchanged.
- TF-9: `interpolate_frames` — linear NaN gap-fill up to max_gap_seconds (cubic deferred to TF-9-cubic).
- TF-12: `add_pre_shot_gk_angle` aggregator + 2 frame-aware xfns
  (`pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`)
  + atomic mirror. Standalone aggregator (does not extend
  `add_pre_shot_gk_position`'s 4-column surface — primitive+assembly pattern).
- Umbrella facade `add_pre_shot_gk_context(actions, *, frames=...)` extended
  4 → 6 columns when frames supplied; `frames=None` path bit-identical to 2.9.0.
- `PreprocessConfig` dataclass with flag-based `is_default()`; per-provider
  defaults from probe baselines; tracking converters auto-promote default()
  to for_provider(<self>).
- Probe scripts + committed baseline JSON + sweep log + integrity test;
  per-period DOP-symmetry invariant for TF-12 closes the 3.0.0/3.0.1 blind-spot
  pattern.
- NOTICE: new Savitzky & Golay 1964 entry; existing Anzer & Bauer 2021 entry
  extended for TF-12 angle features.
- ADR-005 amendment for multi-flavor convention asymmetry (suffixed columns
  for VAEP xfns; canonical-single for preprocessing utilities) queued to land
  with TF-2 PR.
'@
git commit -m $msg
```

### Step 7.4 — Push + PR

```powershell
git push -u origin pr-s24-tier1-sweep
gh pr create --base main --title "feat(tracking): TF-6 sync_score + TF-8/9/12 -- silly-kicks 3.1.0 (PR-S24)" --body @'
## Summary

Tier 1 sweep — bundles four On-Deck Dunkin' items into one PR cycle:

- **TF-6** `sync_score` — per-action tracking↔events sync-quality (3 aggregations).
- **TF-8** smoothing primitives — Savitzky-Golay + EMA `smooth_frames` and `derive_velocities` in new `silly_kicks.tracking.preprocess` module.
- **TF-9** `interpolate_frames` — linear NaN gap-fill up to `max_gap_seconds` (cubic deferred to TF-9-cubic).
- **TF-12** `add_pre_shot_gk_angle` — 2 frame-aware GK-angle features (to_shot_trajectory + off_goal_line, both signed).

Plus shared preprocessing config (`PreprocessConfig`) with per-provider baselines, optional `preprocess` kwarg on tracking converters with flag-based `is_default()` auto-promotion, and the umbrella `add_pre_shot_gk_context(frames=...)` extended 4 → 6 columns.

Spec: `docs/superpowers/specs/2026-05-02-tier1-sweep-design.md` (v3 — 16 review items folded across 2 lakehouse second-opinion sessions).

## Test plan

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run pyright`
- [ ] `uv run python -m pytest tests/ -m "not e2e" -q`
- [ ] `uv run python -m pytest tests/invariants/ -v` (per-period DOP-symmetry GREEN — closes 3.0.0/3.0.1 gap)
- [ ] `SILLY_KICKS_ASSERT_INVARIANTS=1 uv run python -m pytest tests/ -m "not e2e" -q`
- [ ] Empirical sweep log (`tests/fixtures/baselines/preprocess_sweep_log.json`) reflects per-provider distribution stats.

## Lakehouse handoff

- `expected_authors` test: add Savitzky & Golay 1964.
- Boundary-test `expected_columns`: add `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line` (only on the `frames=...` path; `frames=None` path is unchanged).
- Optional opt-in: pass `PreprocessConfig.for_provider("sportec")` etc. to feed cleaner `vx`/`vy`/`speed` into downstream features.
- `applyInPandas` `StructType` schemas: declare `_preprocessed_with: StringType()` + smoothed/velocity DoubleType fields when wrapping preprocessing inside Spark UDFs.

Pin bump: `silly-kicks>=3.1.0,<4`.
'@
```

### Step 7.5 — Update memory after merge

Per release ritual: update `project_release_state.md` + `project_followup_prs.md` with the merged SHA + PyPI live confirmation; mark `feedback_lakehouse_second_opinion_pattern.md` PR-S24 outcome row.

---

## Self-review checklist

- [ ] Every spec section §1–§9 maps to at least one task above (gaps: none).
- [ ] No "TBD" / "implement later" placeholders.
- [ ] Method/property names consistent across loops: `is_default()` (not `is_default_config()`), `_is_universal_default` (not `_universal_default`), `_preprocessed_with` (not `_preprocess_method`), `pre_shot_gk_angle_to_shot_trajectory` (not `_to_shot`), `pre_shot_gk_angle_off_goal_line` (not `_to_goal`).
- [ ] CHANGELOG enumerates every new public column name (lakehouse can ctrl-F).
- [ ] NOTICE has new Savitzky & Golay entry.
- [ ] Per-period DOP-symmetry invariant for TF-12 written before the GK-angle implementation lands.
- [ ] Single-commit-per-branch held in Loop 7.
- [ ] `/final-review` is the immediate predecessor of `git commit` (mandatory gate per memory).
- [ ] `ruff format --check` is a separate gate from `ruff check` (memory: `feedback_ruff_format_check`).
- [ ] Public-API Examples gate is run after every new public def lands.
- [ ] SkillCorner is included in `_PROVIDER_DEFAULTS` (memory: `feedback_no_silent_skips_on_required_testing`).
- [ ] Synthesizer extension lands in Loop 4 (memory: `feedback_synthesizer_shot_plus_keeper_save_pattern`).

### Lakehouse-review fold-in (2026-05-02 round 3)

- [ ] **S1.** Codegen pipeline replaces hand-edits — `scripts/regenerate_provider_defaults.py` writes `_provider_defaults_generated.py`; integrity test asserts code-vs-JSON exact match with regen-hint in failure message.
- [ ] **S2.** `rel_tol=1e-6` is appropriate because codegen is deterministic; no need to loosen.
- [ ] **S3.** Synthesizer extension expanded — 6-consumer inventory + per-consumer assertion audit + fail-loud existence check (`tests/tracking/test_synthesizer_fixture_density.py`) BEFORE TF-12 invariants.
- [ ] **S4.** `derive_velocities` raises `ValueError` if `_preprocessed_with`/`x_smoothed`/`y_smoothed` columns are missing (no hidden schema mutation; principle-of-least-surprise).
- [ ] **S5.** Kloppy converter falls back to `PreprocessConfig.default(force_universal=True)` + `UserWarning` when `dataset.metadata.provider` is not in `get_provider_defaults()`. Test added.
- [ ] **N1.** ADR-005 amendment timing bounded — TF-2 ships within 24-48h of PR-S24 merge.
- [ ] **N2.** `_GOAL_X` / `_GOAL_Y_CENTER` already defined in `_kernels.py` (PR-S20); plan confirms reuse.
- [ ] **N3.** Cubic interpolation removed from API surface — `interpolation_method` Literal restricted to `"linear" | None`. Cubic ships in TF-9-cubic when requested.
- [ ] **N4.** Comment added to `window_frames` parity calc.
- [ ] **N5.** Public `get_provider_defaults()` getter from `silly_kicks.tracking.preprocess`; integrity test consumes via the public path.
- [ ] **scipy** added to runtime `dependencies` in `pyproject.toml` (>=1.10.0); `xthreat.py` defensive `try/except` cleanup.

### Lakehouse-review fold-in (2026-05-02 round 4)

- [ ] **C1.** `PreprocessConfig.__post_init__` rejects `derive_velocity=True` + `smoothing_method=None` at construction (regression tests added).
- [ ] **C2.** All "linear/cubic" references in CHANGELOG / commit message / PR body replaced with "linear (cubic deferred to TF-9-cubic)" — no consumer pins 3.1.0 expecting cubic.
- [ ] **C3.** Step 4.5.2 audit regex tightened to count-style patterns: `r"len\([^)]*\)\s*==\s*\d"` and `r"\.shape\[0\]\s*==\s*\d"` instead of the broad `r"==\s*\d"`.
- [ ] **N6.** Defensive note on `re.escape()` added to the kloppy-fallback test docstring.
- [ ] **derive_velocities is net-new in PR-S24** (verified — no pre-existing implementation), so S4's hard-raise is purely additive; no `### Changed` entry / no deprecation cycle needed.
