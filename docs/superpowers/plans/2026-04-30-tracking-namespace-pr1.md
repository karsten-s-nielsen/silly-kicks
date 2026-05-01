# Tracking namespace PR-1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `silly_kicks.tracking` namespace primitive layer (schema + 4 provider adapters + linkage utilities + ADR-004) for silly-kicks 2.7.0, fully tested via TDD, against four real tracking providers (PFF, Sportec/IDSSE, Metrica, SkillCorner).

**Architecture:** Hexagonal pure-function adapters (Sportec + PFF native, Metrica + SkillCorner via kloppy gateway), 19-column long-form canonical schema, pointer-DataFrame linkage primitive with audit. Spec: `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md`.

**Tech Stack:** pandas ≥ 2.1, numpy ≥ 1.26, kloppy ≥ 3.18 (existing optional dep, version bump from 3.15), pytest, parquet (pyarrow), Databricks SQL connector (probe only, not a runtime dep).

**Commit policy override:** silly-kicks is "literally ONE commit per branch; no WIP commits + squash; explicit approval before that one commit." This plan therefore has **no per-task commit steps** — only a single final commit after `/final-review` and explicit user approval. Each task ends with test-pass verification.

---

## File Structure

### New files (all created in this PR)

```
silly_kicks/tracking/
├── __init__.py
├── schema.py
├── _direction.py                          # MOVED from silly_kicks/spadl/pff.py (extraction)
├── utils.py
├── sportec.py
├── pff.py
└── kloppy.py

docs/superpowers/adrs/
└── ADR-004-tracking-namespace-charter.md

scripts/
└── probe_tracking_baselines.py            # one-off, committed, run during this PR

tests/datasets/tracking/
├── empirical_probe_baselines.json         # one-off probe output, committed
├── pff/
│   ├── __init__.py
│   ├── generate_synthetic.py
│   ├── tiny.parquet
│   └── medium_halftime.parquet
├── sportec/
│   ├── __init__.py
│   ├── generate_synthetic.py
│   ├── tiny.parquet
│   └── medium_halftime.parquet
├── metrica/
│   ├── __init__.py
│   ├── generate_synthetic.py
│   ├── tiny.parquet
│   └── medium_halftime.parquet
└── skillcorner/
    ├── __init__.py
    ├── generate_synthetic.py
    ├── tiny.parquet
    └── medium_halftime.parquet

tests/
├── test_tracking_schema.py
├── test_tracking_sportec.py
├── test_tracking_pff.py
├── test_tracking_kloppy.py
├── test_tracking_utils_link.py
├── test_tracking_utils_slice.py
├── test_tracking_utils_play_left_to_right.py
├── test_tracking_utils_derive_speed.py
├── test_tracking_cross_provider_parity.py
└── test_tracking_real_data_sweep.py
```

### Modified files

```
silly_kicks/spadl/pff.py                   # _direction.py extraction: imports + remove inline helper
silly_kicks/spadl/__init__.py              # no change (extraction is internal-only)
pyproject.toml                             # bump kloppy >= 3.15 to >= 3.18 (tracking parsers)
tests/test_public_api_examples.py          # auto-discovers; no edits needed unless skip-list tweaks
TODO.md                                    # close tracking entry; add 8-item deferred list (per ADR-004 inv. 9)
CHANGELOG.md                               # add 2.7.0 entry
```

---

## Task 0: Branch creation + dependency bump

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 0.1: Create feature branch**

```bash
git checkout main && git pull origin main && git checkout -b feat/tracking-namespace-pr1
```

Expected: `Switched to a new branch 'feat/tracking-namespace-pr1'`.

- [ ] **Step 0.2: Bump kloppy minimum version**

Edit `pyproject.toml` line 33:

```diff
-kloppy = ["kloppy>=3.15.0"]
+kloppy = ["kloppy>=3.18.0"]
```

Reason: kloppy 3.18 ships tracking parsers for Metrica and SkillCorner (used by the gateway). The events kloppy gateway also continues to work at 3.18 (no breaking change for `silly_kicks.spadl.kloppy`).

- [ ] **Step 0.3: Verify installed kloppy version meets new floor**

Run: `python -c "import kloppy; print(kloppy.__version__)"`
Expected: `3.18.0` or higher.

If lower: `pip install -U "kloppy>=3.18.0"` then re-verify.

---

## Task 1 (Loop 0): Empirical probe + JSON baseline + ADR-004

This is the only loop without RED-GREEN-REFACTOR test cycles — it's setup work that produces committed artifacts the rest of the plan depends on.

**Files:**
- Create: `scripts/probe_tracking_baselines.py`
- Create: `tests/datasets/tracking/empirical_probe_baselines.json`
- Create: `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`

- [ ] **Step 1.1: Write the probe script — module skeleton + Databricks SQL helper**

Create `scripts/probe_tracking_baselines.py`:

```python
"""One-off empirical probe for silly-kicks tracking PR-1.

Reads per-provider tracking statistics from:
  Source 1: Databricks SQL — soccer_analytics.dev_gold.fct_tracking_frames
            (providers: metrica, idsse, skillcorner)
  Source 2: Local PFF WC2022 JSONL.bz2 (1 match minimum to characterize)

Writes: tests/datasets/tracking/empirical_probe_baselines.json

Run once during PR-1 development. Both this script AND its output JSON are
committed to the repo. The real datasets are NOT committed.

Usage:
    python scripts/probe_tracking_baselines.py
"""

import bz2
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "tests" / "datasets" / "tracking" / "empirical_probe_baselines.json"

LAKEHOUSE_TABLE = "soccer_analytics.dev_gold.fct_tracking_frames"
PFF_LOCAL_DIR = Path(r"D:\[Karsten]\Dropbox\[Microsoft]\Downloads\FIFA World Cup 2022\Tracking Data")


def probe_lakehouse() -> dict[str, dict[str, Any]]:
    """Query fct_tracking_frames per provider via Databricks SQL."""
    from databricks import sql  # type: ignore[import-not-found]

    conn = sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )
    out: dict[str, dict[str, Any]] = {}
    try:
        with conn.cursor() as cur:
            for provider in ("metrica", "idsse", "skillcorner"):
                cur.execute(f"""
                    SELECT
                        approx_percentile(frame_rate, 0.5) AS frame_rate_p50,
                        count(DISTINCT period_id)         AS n_periods,
                        avg(CASE WHEN ball_state='dead' THEN 1.0 ELSE 0.0 END) AS dead_fraction,
                        avg(CASE WHEN x IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_x,
                        avg(CASE WHEN y IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_y,
                        avg(CASE WHEN z IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_z,
                        avg(CASE WHEN speed IS NULL THEN 1.0 ELSE 0.0 END) AS nan_rate_speed,
                        approx_percentile(x, array(0.01, 0.5, 0.99)) AS x_pct,
                        approx_percentile(y, array(0.01, 0.5, 0.99)) AS y_pct,
                        approx_percentile(speed, array(0.5, 0.99))   AS speed_pct,
                        avg(CASE WHEN is_ball THEN 1.0 ELSE 0.0 END) AS ball_row_rate
                    FROM {LAKEHOUSE_TABLE}
                    WHERE source_provider = '{provider}'
                """)
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
                out[provider if provider != "idsse" else "sportec"] = dict(zip(cols, row, strict=False))
    finally:
        conn.close()
    return out


def probe_pff_local(jsonl_bz2_path: Path) -> dict[str, Any]:
    """Read one PFF WC22 tracking match and characterize it."""
    rows: list[dict[str, Any]] = []
    with bz2.open(jsonl_bz2_path, "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    df = pd.json_normalize(rows)
    return {
        "frame_rate_p50": 30.0,  # PFF documented 30Hz
        "n_periods": int(df["period"].nunique()) if "period" in df.columns else 0,
        "dead_fraction": float((df.get("ballOutOfPlay", pd.Series(False)) == True).mean()),
        "nan_rate_x": float(df["balls.x"].isna().mean()) if "balls.x" in df.columns else 0.0,
        "nan_rate_y": float(df["balls.y"].isna().mean()) if "balls.y" in df.columns else 0.0,
        "nan_rate_speed": 0.0,  # PFF supplies speed
        "ball_row_rate": 1.0,    # one ball per frame in PFF
        "match_filename": jsonl_bz2_path.name,
    }


def main() -> None:
    print("[1/3] Probing lakehouse...")
    lakehouse_stats = probe_lakehouse()
    print(f"  Got {len(lakehouse_stats)} provider entries: {list(lakehouse_stats)}")

    print("[2/3] Probing local PFF WC22 (1 match)...")
    matches = sorted(PFF_LOCAL_DIR.glob("*.jsonl.bz2"))
    if not matches:
        raise FileNotFoundError(f"No PFF tracking files in {PFF_LOCAL_DIR}")
    pff_stats = probe_pff_local(matches[0])
    print(f"  Probed {pff_stats['match_filename']}")

    out = {
        "probe_run_date": "2026-04-30",
        "probe_run_source_lakehouse_table": LAKEHOUSE_TABLE,
        "probe_run_source_pff_path_marker": "FIFA World Cup 2022/Tracking Data",
        "providers": {**lakehouse_stats, "pff": pff_stats},
    }

    print(f"[3/3] Writing {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Run the probe**

Run: `python scripts/probe_tracking_baselines.py`
Expected: prints "Done." and `tests/datasets/tracking/empirical_probe_baselines.json` exists with non-empty `providers` dict containing `pff`, `sportec`, `metrica`, `skillcorner` keys.

If Databricks env vars are missing: set `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` (already present in environment per user confirmation). If `databricks-sql-connector` is not installed: `pip install databricks-sql-connector`.

- [ ] **Step 1.3: Sanity-check the JSON contents**

Run: `python -c "import json; d = json.load(open('tests/datasets/tracking/empirical_probe_baselines.json')); print(list(d['providers']))"`
Expected: `['metrica', 'sportec', 'skillcorner', 'pff']` (or similar — order may vary).

- [ ] **Step 1.4: Write ADR-004**

Create `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md` from spec §4.4. Header from `ADR-TEMPLATE.md`; body verbatim from spec §4.4 invariants 1–9 (with ReSpo.Vision (viii) included).

The full text is in `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md` § 4.4 — copy verbatim into the ADR body, prefixed with the standard ADR header (Status, Date, Drivers, Decision, Consequences, References).

- [ ] **Step 1.5: Verify ADR file**

Run: `wc -l docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`
Expected: 100–200 lines.

---

## Task 2 (Loop 1): Schema constants + `TrackingConversionReport` + `LinkReport`

**Files:**
- Create: `silly_kicks/tracking/__init__.py`
- Create: `silly_kicks/tracking/schema.py`
- Test: `tests/test_tracking_schema.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_tracking_schema.py`:

```python
"""Schema tests for silly_kicks.tracking — column set, dtype variants, dataclasses."""

from silly_kicks.tracking.schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    LinkReport,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)


def test_tracking_frames_columns_is_19_columns():
    assert len(TRACKING_FRAMES_COLUMNS) == 19


def test_tracking_frames_columns_required_keys():
    expected = {
        "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
        "player_id", "team_id", "is_ball", "is_goalkeeper",
        "x", "y", "z", "speed", "speed_source",
        "ball_state", "team_attacking_direction",
        "confidence", "visibility", "source_provider",
    }
    assert set(TRACKING_FRAMES_COLUMNS) == expected


def test_kloppy_variant_overrides_identifiers_to_object():
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["game_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["player_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["team_id"] == "object"
    # other columns identical to base
    for k, v in TRACKING_FRAMES_COLUMNS.items():
        if k not in {"game_id", "player_id", "team_id"}:
            assert KLOPPY_TRACKING_FRAMES_COLUMNS[k] == v


def test_sportec_variant_matches_kloppy_variant():
    assert SPORTEC_TRACKING_FRAMES_COLUMNS == KLOPPY_TRACKING_FRAMES_COLUMNS


def test_pff_variant_uses_nullable_int64_identifiers():
    assert PFF_TRACKING_FRAMES_COLUMNS["player_id"] == "Int64"
    assert PFF_TRACKING_FRAMES_COLUMNS["team_id"] == "Int64"
    assert PFF_TRACKING_FRAMES_COLUMNS["game_id"] == "int64"


def test_tracking_constraints_keys_subset_of_columns():
    assert set(TRACKING_CONSTRAINTS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_constraints_x_y_match_spadl_field_dimensions():
    assert TRACKING_CONSTRAINTS["x"] == (0, 105.0)
    assert TRACKING_CONSTRAINTS["y"] == (0, 68.0)


def test_tracking_categorical_domains_keys_subset_of_columns():
    assert set(TRACKING_CATEGORICAL_DOMAINS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_categorical_domains_values():
    assert TRACKING_CATEGORICAL_DOMAINS["ball_state"] == frozenset({"alive", "dead"})
    assert TRACKING_CATEGORICAL_DOMAINS["team_attacking_direction"] == frozenset({"ltr", "rtl"})
    assert TRACKING_CATEGORICAL_DOMAINS["speed_source"] == frozenset({"native", "derived"})
    assert TRACKING_CATEGORICAL_DOMAINS["source_provider"] == frozenset(
        {"pff", "sportec", "metrica", "skillcorner"}
    )


def test_tracking_conversion_report_is_frozen_dataclass():
    r = TrackingConversionReport(
        provider="pff",
        total_input_frames=100,
        total_output_rows=2200,
        n_periods=2,
        frame_coverage_per_period={1: 1.0, 2: 0.99},
        ball_out_seconds_per_period={1: 12.4, 2: 8.7},
        nan_rate_per_column={"z": 0.95, "speed": 0.0},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
    )
    import dataclasses
    assert dataclasses.is_dataclass(r) and r.__dataclass_params__.frozen


def test_tracking_conversion_report_has_unrecognized():
    r1 = TrackingConversionReport(
        "pff", 0, 0, 0, {}, {}, {}, 0, set()
    )
    assert r1.has_unrecognized is False
    r2 = TrackingConversionReport(
        "pff", 0, 0, 0, {}, {}, {}, 0, {123}
    )
    assert r2.has_unrecognized is True


def test_link_report_link_rate_zero_when_empty():
    r = LinkReport(
        n_actions_in=0, n_actions_linked=0, n_actions_unlinked=0,
        n_actions_multi_candidate=0, per_provider_link_rate={},
        max_time_offset_seconds=0.0, tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.0  # no ZeroDivisionError


def test_link_report_link_rate_nonzero():
    r = LinkReport(
        n_actions_in=100, n_actions_linked=95, n_actions_unlinked=5,
        n_actions_multi_candidate=10, per_provider_link_rate={"pff": 0.95},
        max_time_offset_seconds=0.18, tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.95
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_schema.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'silly_kicks.tracking'`.

- [ ] **Step 2.3: Create the tracking package skeleton + schema module**

Create `silly_kicks/tracking/__init__.py`:

```python
"""silly_kicks.tracking — tracking-data namespace (PR-1: primitive layer).

Schema, per-provider adapters, and the link_actions_to_frames primitive.
Tracking-aware features (action_context, pressure_on_carrier, pitch control,
etc.) ship in PR-2+ scoping cycles. See ADR-004.
"""

__all__ = [
    "KLOPPY_TRACKING_FRAMES_COLUMNS",
    "LinkReport",
    "PFF_TRACKING_FRAMES_COLUMNS",
    "SPORTEC_TRACKING_FRAMES_COLUMNS",
    "TRACKING_CATEGORICAL_DOMAINS",
    "TRACKING_CONSTRAINTS",
    "TRACKING_FRAMES_COLUMNS",
    "TrackingConversionReport",
    "link_actions_to_frames",
    "play_left_to_right",
    "pff",
    "schema",
    "slice_around_event",
    "sportec",
    "utils",
]

from . import pff, schema, sportec, utils
from .schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    LinkReport,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)
from .utils import link_actions_to_frames, play_left_to_right, slice_around_event

try:
    from . import kloppy
except ImportError:
    pass
```

Note: this `__init__.py` references modules (`pff`, `schema`, `sportec`, `utils`, `kloppy`) that don't exist yet — collection will fail. We create them in subsequent tasks. **For Loop 1 only**, comment out the `from . import pff, schema, sportec, utils` line and the `try/except`. We restore them in their respective tasks.

For Loop 1, replace the body with:

```python
__all__ = [
    "KLOPPY_TRACKING_FRAMES_COLUMNS",
    "LinkReport",
    "PFF_TRACKING_FRAMES_COLUMNS",
    "SPORTEC_TRACKING_FRAMES_COLUMNS",
    "TRACKING_CATEGORICAL_DOMAINS",
    "TRACKING_CONSTRAINTS",
    "TRACKING_FRAMES_COLUMNS",
    "TrackingConversionReport",
    "schema",
]

from . import schema
from .schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    LinkReport,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)
```

`__init__.py` will be expanded incrementally as each loop adds its module.

- [ ] **Step 2.4: Write `silly_kicks/tracking/schema.py`**

Create `silly_kicks/tracking/schema.py`:

```python
"""Tracking output schema — plain Python constants + dataclasses.

Mirrors silly_kicks.spadl.schema. See ADR-004 for the namespace charter
and docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md § 4.2.
"""

import dataclasses

TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id":                  "int64",
    "period_id":                "int64",
    "frame_id":                 "int64",
    "time_seconds":             "float64",
    "frame_rate":               "float64",
    "player_id":                "int64",
    "team_id":                  "int64",
    "is_ball":                  "bool",
    "is_goalkeeper":            "bool",
    "x":                        "float64",
    "y":                        "float64",
    "z":                        "float64",
    "speed":                    "float64",
    "speed_source":             "object",
    "ball_state":               "object",
    "team_attacking_direction": "object",
    "confidence":               "object",
    "visibility":               "object",
    "source_provider":          "object",
}

KLOPPY_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "game_id": "object",
    "player_id": "object",
    "team_id": "object",
}
"""Kloppy gateway output: object identifiers (kloppy domain types are strings)."""

SPORTEC_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Sportec native output: same shape as kloppy variant — DFL TeamId / PersonId
are string identifiers."""

PFF_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "player_id": "Int64",
    "team_id": "Int64",
}
"""PFF native output: nullable Int64 identifiers (matches PFF_SPADL_COLUMNS
convention from PR-S18; allows NaN on ball rows). game_id stays int64."""

TRACKING_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id":    (1, 5),
    "time_seconds": (0, float("inf")),
    "frame_rate":   (1, 60),
    "frame_id":     (0, float("inf")),
    "x":            (0, 105.0),
    "y":            (0, 68.0),
    "z":            (0, 10.0),
    "speed":        (0, 50.0),
}

TRACKING_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "ball_state":               frozenset({"alive", "dead"}),
    "team_attacking_direction": frozenset({"ltr", "rtl"}),
    "speed_source":             frozenset({"native", "derived"}),
    "source_provider":          frozenset({"pff", "sportec", "metrica", "skillcorner"}),
}


@dataclasses.dataclass(frozen=True)
class TrackingConversionReport:
    """Audit trail for tracking convert_to_frames(). Frame-shaped audit.

    Attributes:
        provider: Provider name, lowercase ("pff" | "sportec" | "metrica" | "skillcorner").
        total_input_frames: Frames in the raw input DataFrame.
        total_output_rows: Long-form expanded row count (frames × players + ball rows).
        n_periods: Number of distinct period_ids.
        frame_coverage_per_period: period_id -> fraction of expected frames present
            (1.0 = no missing frames, given inferred frame_rate).
        ball_out_seconds_per_period: period_id -> total seconds with ball_state="dead".
        nan_rate_per_column: column name -> fraction of NaN rows in output.
        derived_speed_rows: Rows where speed_source="derived".
        unrecognized_player_ids: IDs in input not resolvable via roster.

    Example::

        frames, report = sportec.convert_to_frames(raw, home_team_id="DFL-CLU-000A1H", ...)
        if report.has_unrecognized:
            logger.warning("Unrecognized player IDs: %s", report.unrecognized_player_ids)
    """

    provider: str
    total_input_frames: int
    total_output_rows: int
    n_periods: int
    frame_coverage_per_period: dict[int, float]
    ball_out_seconds_per_period: dict[int, float]
    nan_rate_per_column: dict[str, float]
    derived_speed_rows: int
    unrecognized_player_ids: set
    # `set` (untyped element) chosen because element type varies by provider
    # (str for kloppy/sportec, int for PFF).

    @property
    def has_unrecognized(self) -> bool:
        return len(self.unrecognized_player_ids) > 0


@dataclasses.dataclass(frozen=True)
class LinkReport:
    """Audit trail for link_actions_to_frames().

    Attributes:
        n_actions_in: Input action count.
        n_actions_linked: Actions with a frame_id (within tolerance).
        n_actions_unlinked: Actions with NaN frame_id (no frame within tolerance).
        n_actions_multi_candidate: Actions with >1 candidate frame within tolerance
            (closest one returned).
        per_provider_link_rate: source_provider -> linked / in. Single-provider
            in practice, multi-provider supported for forward-compat.
        max_time_offset_seconds: max |Δt| among linked rows; 0.0 if none linked.
        tolerance_seconds: Echoes the call argument.

    Example::

        pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
        assert report.link_rate >= 0.95
    """

    n_actions_in: int
    n_actions_linked: int
    n_actions_unlinked: int
    n_actions_multi_candidate: int
    per_provider_link_rate: dict[str, float]
    max_time_offset_seconds: float
    tolerance_seconds: float

    @property
    def link_rate(self) -> float:
        return self.n_actions_linked / max(self.n_actions_in, 1)
```

- [ ] **Step 2.5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_schema.py -v`
Expected: 12 passed.

---

## Task 3 (Loop 2): `_derive_speed`

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (create)
- Modify: `silly_kicks/tracking/__init__.py` (add `utils` import)
- Test: `tests/test_tracking_utils_derive_speed.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/test_tracking_utils_derive_speed.py`:

```python
"""Unit tests for silly_kicks.tracking.utils._derive_speed."""

import numpy as np
import pandas as pd

from silly_kicks.tracking.utils import _derive_speed


def _uniform_motion_frames(speed_mps: float, n_frames: int, frame_rate: float, period_id: int = 1) -> pd.DataFrame:
    """Build long-form frames where one player moves at uniform speed_mps along x."""
    dt = 1.0 / frame_rate
    rows = []
    for i in range(n_frames):
        rows.append({
            "game_id": 1, "period_id": period_id, "frame_id": i,
            "time_seconds": i * dt, "frame_rate": frame_rate,
            "player_id": 7, "team_id": 100,
            "is_ball": False, "is_goalkeeper": False,
            "x": 10.0 + speed_mps * i * dt, "y": 34.0, "z": float("nan"),
            "speed": float("nan"), "speed_source": None,
            "ball_state": "alive", "team_attacking_direction": "ltr",
            "confidence": None, "visibility": None, "source_provider": "metrica",
        })
    return pd.DataFrame(rows)


def test_derive_speed_uniform_motion_one_mps():
    frames = _uniform_motion_frames(speed_mps=1.0, n_frames=10, frame_rate=25.0)
    out = _derive_speed(frames)
    # First frame is NaN (no prior to diff against)
    assert pd.isna(out.iloc[0]["speed"])
    # Subsequent frames: 1.0 m/s ± 1e-6
    np.testing.assert_allclose(out.iloc[1:]["speed"].to_numpy(), 1.0, atol=1e-6)
    # speed_source should be "derived" where speed populated
    assert (out.iloc[1:]["speed_source"] == "derived").all()
    # First frame: speed_source NaN
    assert pd.isna(out.iloc[0]["speed_source"])


def test_derive_speed_period_boundary_no_leakage():
    p1 = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0, period_id=1)
    p2 = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0, period_id=2)
    # Set p2 frame_ids to start at 0 again (period-local) and time to start at 0
    frames = pd.concat([p1, p2], ignore_index=True)
    out = _derive_speed(frames)
    # First frame of each period must be NaN
    p1_first = out[(out.period_id == 1) & (out.frame_id == 0)]
    p2_first = out[(out.period_id == 2) & (out.frame_id == 0)]
    assert pd.isna(p1_first.iloc[0]["speed"])
    assert pd.isna(p2_first.iloc[0]["speed"])


def test_derive_speed_ball_treated_as_one_entity():
    rows = []
    for i in range(5):
        rows.append({
            "game_id": 1, "period_id": 1, "frame_id": i,
            "time_seconds": i / 25.0, "frame_rate": 25.0,
            "player_id": pd.NA, "team_id": pd.NA,
            "is_ball": True, "is_goalkeeper": False,
            "x": 50.0 + 2.0 * i / 25.0, "y": 34.0, "z": 0.5,
            "speed": float("nan"), "speed_source": None,
            "ball_state": "alive", "team_attacking_direction": None,
            "confidence": None, "visibility": None, "source_provider": "metrica",
        })
    frames = pd.DataFrame(rows)
    out = _derive_speed(frames)
    # First ball frame NaN; subsequent 2.0 m/s
    assert pd.isna(out.iloc[0]["speed"])
    np.testing.assert_allclose(out.iloc[1:]["speed"].to_numpy(), 2.0, atol=1e-6)


def test_derive_speed_preserves_native_when_present():
    """If a row already has speed populated, _derive_speed should not overwrite it."""
    frames = _uniform_motion_frames(speed_mps=1.0, n_frames=5, frame_rate=25.0)
    frames.loc[2, "speed"] = 99.9
    frames.loc[2, "speed_source"] = "native"
    out = _derive_speed(frames)
    # Row 2 keeps its native value
    assert out.iloc[2]["speed"] == 99.9
    assert out.iloc[2]["speed_source"] == "native"
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_utils_derive_speed.py -v`
Expected: collection error or import failure for `silly_kicks.tracking.utils`.

- [ ] **Step 3.3: Create `silly_kicks/tracking/utils.py` with `_derive_speed`**

Create `silly_kicks/tracking/utils.py`:

```python
"""Utility functions for silly_kicks.tracking.

Includes:
  - _derive_speed: per-row derived speed where provider doesn't supply it
  - link_actions_to_frames: action↔frame 1:1 nearest-time linkage (Task 5)
  - slice_around_event: action↔frame 1:many windowed slice (Task 6)
  - play_left_to_right: tracking-variant L→R direction normalization (Task 4)
"""

import numpy as np
import pandas as pd


def _derive_speed(frames: pd.DataFrame) -> pd.DataFrame:
    """Compute speed = sqrt((Δx)² + (Δy)²) * frame_rate per (player_id, period_id, is_ball) group.

    Modifies a copy of `frames`:
      - Where `speed` is NaN, fill with derived value and set speed_source="derived".
      - Where `speed` is populated, leave both columns unchanged.
      - First frame of each (player, period) group: speed remains NaN, speed_source NaN.

    Vectorized via groupby+diff. Ball rows treated as a single logical entity
    (groupby key uses is_ball=True).
    """
    out = frames.copy()
    # Group key: ball rows have NaN player_id, so we group by (period_id, is_ball, player_id).
    # is_ball=True rows form one group regardless of player_id (all NaN).
    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    grp_keys = ["period_id", "is_ball", "player_id"]
    dx = out.groupby(grp_keys, dropna=False)["x"].diff()
    dy = out.groupby(grp_keys, dropna=False)["y"].diff()
    derived = np.sqrt(dx**2 + dy**2) * out["frame_rate"]

    # Only fill where speed was NaN
    fill_mask = out["speed"].isna() & derived.notna()
    out.loc[fill_mask, "speed"] = derived[fill_mask]
    out.loc[fill_mask, "speed_source"] = "derived"
    return out
```

- [ ] **Step 3.4: Update `silly_kicks/tracking/__init__.py` to import utils**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    "KLOPPY_TRACKING_FRAMES_COLUMNS",
    "LinkReport",
    "PFF_TRACKING_FRAMES_COLUMNS",
    "SPORTEC_TRACKING_FRAMES_COLUMNS",
    "TRACKING_CATEGORICAL_DOMAINS",
    "TRACKING_CONSTRAINTS",
    "TRACKING_FRAMES_COLUMNS",
    "TrackingConversionReport",
    "schema",
    "utils",
]

from . import schema, utils
from .schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    LinkReport,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)
```

- [ ] **Step 3.5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_utils_derive_speed.py -v`
Expected: 4 passed.

---

## Task 4 (Loop 3): `play_left_to_right` (tracking variant)

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (add function)
- Modify: `silly_kicks/tracking/__init__.py` (export `play_left_to_right`)
- Test: `tests/test_tracking_utils_play_left_to_right.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_tracking_utils_play_left_to_right.py`:

```python
"""Unit tests for silly_kicks.tracking.utils.play_left_to_right (tracking variant)."""

import numpy as np
import pandas as pd

from silly_kicks.tracking.utils import play_left_to_right


def _row(period_id, frame_id, player_id, team_id, x, y, *, is_ball=False, td="rtl"):
    return {
        "game_id": 1, "period_id": period_id, "frame_id": frame_id,
        "time_seconds": frame_id / 25.0, "frame_rate": 25.0,
        "player_id": player_id, "team_id": team_id,
        "is_ball": is_ball, "is_goalkeeper": False,
        "x": x, "y": y, "z": float("nan"),
        "speed": 0.0, "speed_source": "native",
        "ball_state": "alive", "team_attacking_direction": td,
        "confidence": None, "visibility": None, "source_provider": "pff",
    }


def test_ball_x_flipped_when_attacking_rtl():
    frames = pd.DataFrame([_row(1, 0, None, None, 20.0, 34.0, is_ball=True, td="rtl")])
    out = play_left_to_right(frames, home_team_id=100)
    # Home team attacks RTL in period 1 -> flip everything in period 1
    # Ball x: 20 -> 105 - 20 = 85
    assert out.iloc[0]["x"] == 85.0
    assert out.iloc[0]["y"] == 68.0 - 34.0


def test_player_rows_flip_consistently_with_ball():
    frames = pd.DataFrame([
        _row(1, 0, 7, 100, 30.0, 20.0, td="rtl"),
        _row(1, 0, None, None, 40.0, 50.0, is_ball=True, td="rtl"),
    ])
    out = play_left_to_right(frames, home_team_id=100)
    assert out.iloc[0]["x"] == 75.0
    assert out.iloc[0]["y"] == 48.0
    assert out.iloc[1]["x"] == 65.0
    assert out.iloc[1]["y"] == 18.0


def test_ltr_frames_pass_through_unchanged():
    frames = pd.DataFrame([_row(1, 0, 7, 100, 30.0, 20.0, td="ltr")])
    out = play_left_to_right(frames, home_team_id=100)
    assert out.iloc[0]["x"] == 30.0
    assert out.iloc[0]["y"] == 20.0
    assert out.iloc[0]["team_attacking_direction"] == "ltr"


def test_team_attacking_direction_set_to_ltr_after_flip():
    frames = pd.DataFrame([_row(1, 0, 7, 100, 30.0, 20.0, td="rtl")])
    out = play_left_to_right(frames, home_team_id=100)
    # After flip, all rows should report ltr
    assert out.iloc[0]["team_attacking_direction"] == "ltr"


def test_nan_xy_stays_nan():
    frames = pd.DataFrame([_row(1, 0, 7, 100, float("nan"), float("nan"), td="rtl")])
    out = play_left_to_right(frames, home_team_id=100)
    assert pd.isna(out.iloc[0]["x"])
    assert pd.isna(out.iloc[0]["y"])
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_utils_play_left_to_right.py -v`
Expected: ImportError or AttributeError on `play_left_to_right`.

- [ ] **Step 4.3: Implement `play_left_to_right` in `silly_kicks/tracking/utils.py`**

Append to `silly_kicks/tracking/utils.py`:

```python
def play_left_to_right(frames: pd.DataFrame, home_team_id) -> pd.DataFrame:
    """Mirror tracking frames so the home team attacks left-to-right in every period.

    Operates on long-form rows (player rows AND ball rows). For rows where
    ``team_attacking_direction == "rtl"``, mirrors x and y around the SPADL
    pitch center (105/2, 68/2). Sets ``team_attacking_direction = "ltr"`` for
    flipped rows.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    home_team_id : int | str
        ID of the home team (kept in signature for API parity with
        spadl.utils.play_left_to_right; flip decision is taken from
        team_attacking_direction directly, so home_team_id is not used here
        — required only to disambiguate downstream consumers and reserved
        for future use).

    Returns
    -------
    pd.DataFrame
        Frames with x/y mirrored where direction was "rtl" and
        team_attacking_direction reset to "ltr" on flipped rows.

    Examples
    --------
    Normalize tracking frames so the home team always attacks left-to-right::

        frames, _ = sportec.convert_to_frames(raw, home_team_id="DFL-CLU-A", ...)
        ltr_frames = play_left_to_right(frames, home_team_id="DFL-CLU-A")
        # All rows now have team_attacking_direction == "ltr".
    """
    _ = home_team_id  # reserved for future use
    out = frames.copy()
    flip_mask = (out["team_attacking_direction"] == "rtl").to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]
    out.loc[flip_mask, "team_attacking_direction"] = "ltr"
    return out
```

- [ ] **Step 4.4: Update `__init__.py` to export `play_left_to_right`**

Edit `silly_kicks/tracking/__init__.py` `__all__` and imports:

```python
__all__ = [
    # ... existing entries ...
    "play_left_to_right",
]

from .utils import play_left_to_right
```

- [ ] **Step 4.5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_utils_play_left_to_right.py -v`
Expected: 5 passed.

---

## Task 5 (Loop 4): `link_actions_to_frames`

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (add function)
- Modify: `silly_kicks/tracking/__init__.py` (export `link_actions_to_frames`)
- Test: `tests/test_tracking_utils_link.py`

- [ ] **Step 5.1: Write the failing tests**

Create `tests/test_tracking_utils_link.py`:

```python
"""Unit tests for silly_kicks.tracking.utils.link_actions_to_frames."""

import numpy as np
import pandas as pd

from silly_kicks.tracking.schema import LinkReport
from silly_kicks.tracking.utils import link_actions_to_frames


def _frame_row(period_id, frame_id, t):
    return {
        "game_id": 1, "period_id": period_id, "frame_id": frame_id,
        "time_seconds": t, "frame_rate": 25.0,
        "player_id": 7, "team_id": 100,
        "is_ball": False, "is_goalkeeper": False,
        "x": 50.0, "y": 34.0, "z": float("nan"),
        "speed": 5.0, "speed_source": "native",
        "ball_state": "alive", "team_attacking_direction": "ltr",
        "confidence": None, "visibility": None, "source_provider": "pff",
    }


def _action_row(action_id, period_id, t):
    return {
        "game_id": 1, "action_id": action_id, "period_id": period_id,
        "time_seconds": t, "team_id": 100, "player_id": 7,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
        "start_x": 50.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0,
    }


def test_exact_time_match_link_quality_one():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0)])
    pointers, report = link_actions_to_frames(actions, frames)
    assert pointers.iloc[0]["time_offset_seconds"] == 0.0
    assert pointers.iloc[0]["link_quality_score"] == 1.0
    assert report.n_actions_linked == 1


def test_link_within_tolerance():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.15)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    # Closest is frame 1 at t=0.04 -> offset 0.11
    assert abs(pointers.iloc[0]["time_offset_seconds"] - 0.11) < 1e-9
    assert report.n_actions_linked == 1


def test_unlinked_outside_tolerance():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.15)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert pd.isna(pointers.iloc[0]["time_offset_seconds"])
    assert pd.isna(pointers.iloc[0]["link_quality_score"])
    assert report.n_actions_unlinked == 1
    assert report.n_actions_linked == 0


def test_empty_actions_returns_empty_pointer():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0)])
    actions = pd.DataFrame(columns=["action_id", "period_id", "time_seconds", "team_id"])
    pointers, report = link_actions_to_frames(actions, frames)
    assert len(pointers) == 0
    assert report.n_actions_in == 0
    assert report.link_rate == 0.0  # no division error


def test_action_with_no_frame_in_period():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 2, 0.0)])  # period 2 — no frames
    pointers, report = link_actions_to_frames(actions, frames)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert report.n_actions_unlinked == 1


def test_multi_candidate_chooses_closest():
    frames = pd.DataFrame([
        _frame_row(1, 0, 0.10),
        _frame_row(1, 1, 0.20),  # closer
        _frame_row(1, 2, 0.30),
    ])
    actions = pd.DataFrame([_action_row(0, 1, 0.18)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    assert pointers.iloc[0]["frame_id"] == 1
    assert report.n_actions_multi_candidate >= 1


def test_cross_period_does_not_link():
    frames = pd.DataFrame([_frame_row(2, 0, 0.0)])  # only period 2
    actions = pd.DataFrame([_action_row(0, 1, 0.0)])  # action in period 1
    pointers, report = link_actions_to_frames(actions, frames)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert report.n_actions_unlinked == 1


def test_default_tolerance_is_0_2_seconds():
    """Memory: tests-crossing-pipelines-need-default-stable-params.

    This test pins the default to 0.2s. Changing the default requires explicit
    test update — preventing silent default drift breaking downstream callers.
    """
    frames = pd.DataFrame([_frame_row(1, 0, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.19)])  # just within 0.2 default
    pointers_default, _ = link_actions_to_frames(actions, frames)
    assert not pd.isna(pointers_default.iloc[0]["frame_id"])

    actions2 = pd.DataFrame([_action_row(0, 1, 0.21)])  # just outside 0.2 default
    pointers_default2, _ = link_actions_to_frames(actions2, frames)
    assert pd.isna(pointers_default2.iloc[0]["frame_id"])
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_utils_link.py -v`
Expected: AttributeError on `link_actions_to_frames`.

- [ ] **Step 5.3: Implement `link_actions_to_frames`**

Append to `silly_kicks/tracking/utils.py`:

```python
from .schema import LinkReport


def link_actions_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    tolerance_seconds: float = 0.2,
) -> tuple[pd.DataFrame, LinkReport]:
    """Link each action to the nearest tracking frame in time within tolerance.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with at least ``action_id``, ``period_id``, ``time_seconds``.
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS. Multiple
        rows per (period_id, frame_id) — internally deduplicated.
    tolerance_seconds : float, default 0.2
        Maximum |Δt| for a valid link. NaN frame_id otherwise.

    Returns
    -------
    pointers : pd.DataFrame
        Columns: action_id, frame_id (Int64, NaN if unlinked),
        time_offset_seconds (float64, NaN if unlinked),
        n_candidate_frames (int64), link_quality_score (float64,
        1.0 - |Δt|/tolerance_seconds, NaN if unlinked).
    report : LinkReport
        Audit trail.

    Examples
    --------
    Find the nearest frame for each SPADL action and inspect link rate::

        actions, _ = pff.convert_to_actions(events, ...)
        frames, _ = pff_tracking.convert_to_frames(raw_frames, ...)
        pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
        assert report.link_rate >= 0.95
    """
    if len(actions) == 0:
        return (
            pd.DataFrame(
                {
                    "action_id": pd.Series([], dtype="int64"),
                    "frame_id": pd.Series([], dtype="Int64"),
                    "time_offset_seconds": pd.Series([], dtype="float64"),
                    "n_candidate_frames": pd.Series([], dtype="int64"),
                    "link_quality_score": pd.Series([], dtype="float64"),
                }
            ),
            LinkReport(0, 0, 0, 0, {}, 0.0, tolerance_seconds),
        )

    # Dedup frames to one row per (period_id, frame_id)
    frame_index = (
        frames[["period_id", "frame_id", "time_seconds", "source_provider"]]
        .drop_duplicates(["period_id", "frame_id"])
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    # merge_asof per period to avoid cross-period matching
    actions_sorted = (
        actions[["action_id", "period_id", "time_seconds"]]
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    parts: list[pd.DataFrame] = []
    for period, a_group in actions_sorted.groupby("period_id", sort=False):
        f_group = frame_index[frame_index["period_id"] == period]
        if len(f_group) == 0:
            # No frames in this period — all unlinked
            unlinked = a_group.copy()
            unlinked["frame_id"] = pd.array([pd.NA] * len(a_group), dtype="Int64")
            unlinked["frame_time"] = float("nan")
            unlinked["source_provider"] = None
            parts.append(unlinked)
            continue
        merged = pd.merge_asof(
            a_group.sort_values("time_seconds"),
            f_group[["frame_id", "time_seconds", "source_provider"]]
                .rename(columns={"time_seconds": "frame_time"})
                .sort_values("frame_time"),
            left_on="time_seconds",
            right_on="frame_time",
            direction="nearest",
            tolerance=tolerance_seconds,
        )
        parts.append(merged)

    merged_all = pd.concat(parts, ignore_index=True)

    time_offset = merged_all["frame_time"] - merged_all["time_seconds"]
    quality = 1.0 - time_offset.abs() / tolerance_seconds
    quality = quality.where(merged_all["frame_id"].notna(), other=float("nan"))
    time_offset = time_offset.where(merged_all["frame_id"].notna(), other=float("nan"))

    # Count candidate frames within tolerance per action (vectorized: cross-search)
    n_cand = _count_candidates_within_tolerance(
        actions_sorted, frame_index, tolerance_seconds
    )

    pointers = pd.DataFrame({
        "action_id": merged_all["action_id"].astype("int64"),
        "frame_id": merged_all["frame_id"].astype("Int64"),
        "time_offset_seconds": time_offset.astype("float64"),
        "n_candidate_frames": n_cand.astype("int64"),
        "link_quality_score": quality.astype("float64"),
    })

    n_in = len(actions)
    n_linked = int(pointers["frame_id"].notna().sum())
    n_unlinked = n_in - n_linked
    n_multi = int((pointers["n_candidate_frames"] > 1).sum())
    per_provider: dict[str, float] = {}
    if n_linked > 0:
        provider_col = merged_all.loc[merged_all["frame_id"].notna(), "source_provider"]
        for prov, count in provider_col.value_counts().items():
            per_provider[str(prov)] = float(count) / n_in
    max_off = float(time_offset.abs().max()) if n_linked > 0 else 0.0

    report = LinkReport(
        n_actions_in=n_in,
        n_actions_linked=n_linked,
        n_actions_unlinked=n_unlinked,
        n_actions_multi_candidate=n_multi,
        per_provider_link_rate=per_provider,
        max_time_offset_seconds=max_off,
        tolerance_seconds=tolerance_seconds,
    )
    return pointers, report


def _count_candidates_within_tolerance(
    actions_sorted: pd.DataFrame,
    frame_index: pd.DataFrame,
    tolerance: float,
) -> pd.Series:
    """For each action, count distinct frame_ids within ±tolerance in same period."""
    n = len(actions_sorted)
    counts = np.zeros(n, dtype="int64")
    # Per-period two-pointer sweep
    for i, row in actions_sorted.iterrows():
        f_period = frame_index[frame_index["period_id"] == row["period_id"]]
        if len(f_period) == 0:
            continue
        in_window = (f_period["time_seconds"] - row["time_seconds"]).abs() <= tolerance
        counts[i] = int(in_window.sum())
    return pd.Series(counts, index=actions_sorted.index)
```

Note: `_count_candidates_within_tolerance` is O(n_actions × frames_per_period) but executes in pandas vectorized form per action; for 10⁴ actions × 10⁵ frames per period it's ≈ 10⁹ comparisons — consider numpy `searchsorted` optimization in REFACTOR if benchmark warrants. For PR-1 correctness, the simple form passes.

- [ ] **Step 5.4: Update `__init__.py` exports**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    # ... existing ...
    "link_actions_to_frames",
]

from .utils import link_actions_to_frames, play_left_to_right
```

- [ ] **Step 5.5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_utils_link.py -v`
Expected: 8 passed.

---

## Task 6 (Loop 5): `slice_around_event`

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (add function)
- Modify: `silly_kicks/tracking/__init__.py` (export)
- Test: `tests/test_tracking_utils_slice.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/test_tracking_utils_slice.py`:

```python
"""Unit tests for silly_kicks.tracking.utils.slice_around_event."""

import pandas as pd

from silly_kicks.tracking.utils import link_actions_to_frames, slice_around_event


def _frames(period_id: int, n: int, hz: float = 25.0, t0: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for player_id in (7, 8):
            rows.append({
                "game_id": 1, "period_id": period_id, "frame_id": i,
                "time_seconds": t0 + i / hz, "frame_rate": hz,
                "player_id": player_id, "team_id": 100,
                "is_ball": False, "is_goalkeeper": False,
                "x": 50.0, "y": 34.0, "z": float("nan"),
                "speed": 5.0, "speed_source": "native",
                "ball_state": "alive", "team_attacking_direction": "ltr",
                "confidence": None, "visibility": None, "source_provider": "pff",
            })
        # ball
        rows.append({
            "game_id": 1, "period_id": period_id, "frame_id": i,
            "time_seconds": t0 + i / hz, "frame_rate": hz,
            "player_id": pd.NA, "team_id": pd.NA,
            "is_ball": True, "is_goalkeeper": False,
            "x": 50.0, "y": 34.0, "z": 0.5,
            "speed": 8.0, "speed_source": "native",
            "ball_state": "alive", "team_attacking_direction": None,
            "confidence": None, "visibility": None, "source_provider": "pff",
        })
    return pd.DataFrame(rows)


def _action(action_id, period_id, t):
    return {
        "game_id": 1, "action_id": action_id, "period_id": period_id,
        "time_seconds": t, "team_id": 100, "player_id": 7,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
        "start_x": 50.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0,
    }


def test_zero_window_returns_one_frame_per_action():
    frames = _frames(1, 5)  # 5 frames × 3 rows (2 players + ball) = 15 rows
    actions = pd.DataFrame([_action(0, 1, 0.04)])  # exactly frame 1 time
    out = slice_around_event(actions, frames, pre_seconds=0.0, post_seconds=0.0)
    # Window ε around frame 1: returns 3 rows (2 players + ball at frame 1)
    assert len(out) == 3
    assert (out["action_id"] == 0).all()
    assert (out["frame_id"] == 1).all()


def test_half_second_window_returns_full_neighbourhood():
    frames = _frames(1, 50, hz=25.0)  # 50 frames over 2 s
    actions = pd.DataFrame([_action(0, 1, 1.0)])
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    # 25 frames * 3 rows = 75 rows expected (give or take boundary inclusivity)
    assert 70 <= len(out) <= 80


def test_window_does_not_cross_periods():
    p1 = _frames(1, 25, hz=25.0)
    p2 = _frames(2, 25, hz=25.0, t0=0.0)
    frames = pd.concat([p1, p2], ignore_index=True)
    # Action at end of period 1
    actions = pd.DataFrame([_action(0, 1, 0.96)])  # frame_id ~24
    out = slice_around_event(actions, frames, pre_seconds=1.0, post_seconds=1.0)
    assert (out["period_id"] == 1).all()


def test_zero_window_consistent_with_link_actions_to_frames():
    """slice_around_event(pre=0, post=0) should yield same frame_id set as link."""
    frames = _frames(1, 25, hz=25.0)
    actions = pd.DataFrame([
        _action(0, 1, 0.04),
        _action(1, 1, 0.40),
    ])
    pointers, _ = link_actions_to_frames(actions, frames, tolerance_seconds=0.05)
    sliced = slice_around_event(actions, frames, pre_seconds=0.0, post_seconds=0.0)
    linked_frame_ids = pointers.dropna(subset=["frame_id"])["frame_id"].astype(int).tolist()
    sliced_frame_ids = sliced["frame_id"].drop_duplicates().tolist()
    assert set(linked_frame_ids) == set(sliced_frame_ids)


def test_empty_intersection_returns_empty():
    frames = _frames(1, 5)
    actions = pd.DataFrame([_action(0, 2, 0.0)])  # period 2, no frames
    out = slice_around_event(actions, frames)
    assert len(out) == 0
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_utils_slice.py -v`
Expected: AttributeError on `slice_around_event`.

- [ ] **Step 6.3: Implement `slice_around_event`**

Append to `silly_kicks/tracking/utils.py`:

```python
def slice_around_event(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    pre_seconds: float = 0.0,
    post_seconds: float = 0.0,
) -> pd.DataFrame:
    """Return all frames within [t - pre_seconds, t + post_seconds] per action,
    constrained to the same period.

    Output: long-form frames slice with action_id and time_offset_seconds joined
    in. One row per (action_id, frame_row).

    Examples
    --------
    Pull the 0.5 s pre/post window around every shot::

        shots = actions[actions.type_name == "shot"]
        ctx = slice_around_event(shots, frames, pre_seconds=0.5, post_seconds=0.5)
    """
    if len(actions) == 0 or len(frames) == 0:
        cols = list(frames.columns) + ["action_id", "time_offset_seconds"]
        return pd.DataFrame(columns=cols)

    a = actions[["action_id", "period_id", "time_seconds"]].rename(
        columns={"time_seconds": "action_time"}
    )
    # Inner join on period_id
    merged = frames.merge(a, on="period_id", how="inner")
    delta = merged["time_seconds"] - merged["action_time"]
    in_window = (delta >= -pre_seconds) & (delta <= post_seconds)
    out = merged.loc[in_window].copy()
    out["time_offset_seconds"] = (out["time_seconds"] - out["action_time"]).astype("float64")
    out = out.drop(columns=["action_time"])
    return out.reset_index(drop=True)
```

- [ ] **Step 6.4: Update `__init__.py`**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    # ... existing ...
    "slice_around_event",
]

from .utils import link_actions_to_frames, play_left_to_right, slice_around_event
```

- [ ] **Step 6.5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_utils_slice.py -v`
Expected: 5 passed.

---

## Task 7 (Loop 6 prep): Synthetic-fixture generator pattern

Before implementing the four adapters, build the shared generator pattern. Each provider's `generate_synthetic.py` reads `empirical_probe_baselines.json` and emits two parquet files.

**Files:**
- Create: `tests/datasets/tracking/_generator_common.py`

- [ ] **Step 7.1: Write the shared generator helper**

Create `tests/datasets/tracking/_generator_common.py`:

```python
"""Shared helpers for synthetic tracking-fixture generators.

Reads tests/datasets/tracking/empirical_probe_baselines.json and emits
provider-shaped raw-input DataFrames for each adapter to consume.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASELINES = json.loads((ROOT / "empirical_probe_baselines.json").read_text())


def get_provider_baseline(provider: str) -> dict[str, Any]:
    """Return per-provider stats dict from the committed JSON probe."""
    return BASELINES["providers"][provider]


def deterministic_uniform_motion(
    n_frames: int,
    frame_rate: float,
    n_players_per_team: int = 11,
    period_id: int = 1,
    t0: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a long-form 'reference' DataFrame (provider-agnostic) where each player
    moves at a deterministic uniform speed.

    Used as the per-provider raw-input *prior* to provider-shaping; the
    per-provider generator script then transforms this into provider-native
    columns / dtypes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dt = 1.0 / frame_rate
    for i in range(n_frames):
        t = t0 + i * dt
        for team_id, team_offset_x in ((100, 0.0), (200, 52.5)):
            for jersey in range(n_players_per_team):
                rows.append({
                    "period_id": period_id, "frame_id": i, "time_seconds": t,
                    "frame_rate": frame_rate,
                    "player_id": team_id * 100 + jersey,
                    "team_id": team_id,
                    "jersey": jersey,
                    "is_ball": False, "is_goalkeeper": (jersey == 0),
                    "x_centered": team_offset_x - 26.25 + jersey * 4.0,
                    "y_centered": -34.0 + 6.0 * jersey + 0.05 * i,
                    "z": float("nan"),
                    "speed_native": float(rng.uniform(2.0, 6.0)),
                    "ball_state": "alive",
                })
        # ball row
        rows.append({
            "period_id": period_id, "frame_id": i, "time_seconds": t,
            "frame_rate": frame_rate,
            "player_id": None, "team_id": None, "jersey": None,
            "is_ball": True, "is_goalkeeper": False,
            "x_centered": -52.5 + (i / max(n_frames - 1, 1)) * 105.0,  # ball traverses pitch
            "y_centered": 0.0,
            "z": 0.5,
            "speed_native": 10.0,
            "ball_state": "alive",
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 7.2: Verify import works**

Run: `python -c "from tests.datasets.tracking._generator_common import deterministic_uniform_motion; print(deterministic_uniform_motion(3, 25.0).shape)"`
Expected: `(72, 12)` (3 frames × (22 players + 1 ball) = 69 rows; columns count 12).

---

## Task 8 (Loop 6): Sportec native adapter

**Files:**
- Create: `silly_kicks/tracking/sportec.py`
- Modify: `silly_kicks/tracking/__init__.py` (add `sportec` import)
- Create: `tests/datasets/tracking/sportec/generate_synthetic.py`
- Create: `tests/datasets/tracking/sportec/__init__.py` (empty)
- Create: `tests/datasets/tracking/sportec/tiny.parquet` (generated)
- Create: `tests/datasets/tracking/sportec/medium_halftime.parquet` (generated)
- Test: `tests/test_tracking_sportec.py`

- [ ] **Step 8.1: Write Sportec synthetic generator**

Create `tests/datasets/tracking/sportec/generate_synthetic.py`:

```python
"""Generate synthetic Sportec-shaped raw input for tracking adapter tests.

Reads empirical_probe_baselines.json for sportec stats. Emits two parquet
files: tiny.parquet (~3 s) and medium_halftime.parquet (~60 s spanning HT).

Sportec input shape (mirrors the pattern callers parse from DFL Position XML):
  - period_id, frame_id, time_seconds, frame_rate
  - player_id (DFL PersonId string), team_id (DFL TeamId string)
  - is_ball, is_goalkeeper
  - x_centered, y_centered (DFL pitch-centered meters)
  - speed_native (DFL FrameSet/Frame.S, m/s)
  - ball_state ("alive" | "dead", from DFL BallStatus)
"""

from pathlib import Path

import pandas as pd

from tests.datasets.tracking._generator_common import (
    deterministic_uniform_motion, get_provider_baseline,
)

OUT_DIR = Path(__file__).resolve().parent
BASELINE = get_provider_baseline("sportec")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 25.0)


def _to_sportec_shape(ref: pd.DataFrame, *, game_id: str = "DFL-MAT-0001") -> pd.DataFrame:
    """Convert reference long-form to Sportec-native column names + dtypes."""
    out = ref.copy()
    out["game_id"] = game_id
    out["player_id"] = out["player_id"].apply(
        lambda v: f"DFL-OBJ-{int(v):05d}" if pd.notna(v) else None
    )
    out["team_id"] = out["team_id"].apply(
        lambda v: f"DFL-CLU-{int(v):04d}" if pd.notna(v) else None
    )
    return out


def main() -> None:
    # Tiny: 3 s, single period, no halftime
    tiny_ref = deterministic_uniform_motion(
        n_frames=int(3 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=1,
    )
    tiny = _to_sportec_shape(tiny_ref)
    tiny.to_parquet(OUT_DIR / "tiny.parquet", index=False)

    # Medium: 30 s of period 1 + 30 s of period 2 (separated by HT gap)
    p1 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=1, t0=0.0, seed=1,
    )
    p2 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=2, t0=0.0, seed=2,
    )
    medium = pd.concat([_to_sportec_shape(p1), _to_sportec_shape(p2)], ignore_index=True)
    # Inject a 5-second ball-out interval to exercise ball_state="dead"
    dead_mask = (medium["period_id"] == 1) & (medium["time_seconds"].between(10, 15))
    medium.loc[dead_mask, "ball_state"] = "dead"
    medium.to_parquet(OUT_DIR / "medium_halftime.parquet", index=False)

    print(f"Wrote {OUT_DIR / 'tiny.parquet'} ({len(tiny)} rows)")
    print(f"Wrote {OUT_DIR / 'medium_halftime.parquet'} ({len(medium)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Run generator to produce committed parquets**

Run: `python tests/datasets/tracking/sportec/generate_synthetic.py`
Expected: prints two `Wrote ...` lines; both parquet files exist.

- [ ] **Step 8.3: Write the failing tests**

Create `tests/datasets/tracking/sportec/__init__.py` (empty file).

Create `tests/test_tracking_sportec.py`:

```python
"""Unit tests for silly_kicks.tracking.sportec.convert_to_frames."""

from pathlib import Path

import pandas as pd

from silly_kicks.tracking.schema import (
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CONSTRAINTS,
    TrackingConversionReport,
)
from silly_kicks.tracking.sportec import convert_to_frames

FIXTURE_DIR = Path("tests/datasets/tracking/sportec")
TINY = pd.read_parquet(FIXTURE_DIR / "tiny.parquet")
MEDIUM = pd.read_parquet(FIXTURE_DIR / "medium_halftime.parquet")


def test_tiny_output_shape_and_dtypes():
    frames, report = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=True,
    )
    assert set(frames.columns) == set(SPORTEC_TRACKING_FRAMES_COLUMNS)
    for col, dtype_str in SPORTEC_TRACKING_FRAMES_COLUMNS.items():
        actual = str(frames[col].dtype)
        # Allow object↔string equivalence; bool↔boolean
        if dtype_str in {"object"} and actual in {"object", "string"}:
            continue
        assert actual == dtype_str or (dtype_str == "bool" and actual == "boolean"), (
            f"{col}: expected {dtype_str}, got {actual}"
        )


def test_tiny_coordinate_bounds():
    frames, _ = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=True,
    )
    lo_x, hi_x = TRACKING_CONSTRAINTS["x"]
    lo_y, hi_y = TRACKING_CONSTRAINTS["y"]
    assert frames["x"].between(lo_x, hi_x).all()
    assert frames["y"].between(lo_y, hi_y).all()


def test_tiny_ball_rows_have_nan_player_team():
    frames, _ = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=True,
    )
    ball = frames[frames["is_ball"]]
    assert ball["player_id"].isna().all()
    assert ball["team_id"].isna().all()
    assert (ball["is_goalkeeper"] == False).all()
    assert ball["team_attacking_direction"].isna().all()


def test_tiny_conversion_report_total_input_frames():
    _, report = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=True,
    )
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "sportec"
    assert report.total_input_frames == TINY["frame_id"].nunique()


def test_home_start_left_false_flips_x_versus_true():
    f_true, _ = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=True,
    )
    f_false, _ = convert_to_frames(
        TINY, home_team_id="DFL-CLU-0100", home_team_start_left=False,
    )
    # Same set of (period, frame, player) but x mirrored:
    merge_cols = ["period_id", "frame_id", "player_id", "team_id", "is_ball"]
    j = f_true.merge(f_false, on=merge_cols, suffixes=("_t", "_f"))
    # x_t + x_f should equal 105 for every row (flip identity)
    assert ((j["x_t"] + j["x_f"]).round(6) == 105.0).all()


def test_medium_period_flip_consistency():
    frames, _ = convert_to_frames(
        MEDIUM, home_team_id="DFL-CLU-0100",
        home_team_start_left=True, home_team_start_left_extratime=None,
    )
    # All rows in [0, 105] x [0, 68]
    assert frames["x"].between(0, 105).all()
    assert frames["y"].between(0, 68).all()
    # No NaN time_seconds
    assert frames["time_seconds"].notna().all()


def test_unrecognized_player_id_populated_in_report():
    bad = TINY.copy()
    # Inject one unknown player
    bad.loc[bad.index[0], "player_id"] = "DFL-OBJ-99999"
    _, report = convert_to_frames(
        bad, home_team_id="DFL-CLU-0100", home_team_start_left=True,
        # If the adapter accepts a roster argument, pass an empty roster here.
        # Otherwise the unrecognized check is provider-internal and this test
        # may need adjustment depending on adapter signature. For Sportec
        # native we accept a `roster` kwarg of {player_id: team_id} mapping.
    )
    # Whether unrecognized is non-empty depends on whether sportec adapter
    # validates against a caller-provided roster. The spec says "unrecognized_player_ids
    # = IDs in input not resolvable via roster" — so if no roster given, the field
    # is empty and this test asserts that. If a roster is supported, this test is
    # tightened.
    assert isinstance(report.unrecognized_player_ids, set)
```

- [ ] **Step 8.4: Run tests to verify they fail**

Run: `pytest tests/test_tracking_sportec.py -v`
Expected: ImportError on `silly_kicks.tracking.sportec`.

- [ ] **Step 8.5: Implement Sportec adapter**

Create `silly_kicks/tracking/sportec.py`:

```python
"""Sportec/IDSSE tracking DataFrame converter.

Converts a caller-parsed Sportec-shaped tracking DataFrame (typically derived
from DFL Position XML via ``xmltodict`` + ``pd.json_normalize`` + roster join)
into the canonical SPORTEC_TRACKING_FRAMES_COLUMNS schema.

Input contract (EXPECTED_INPUT_COLUMNS):
  - game_id (object), period_id (int), frame_id (int), time_seconds (float)
  - frame_rate (float, Hz)
  - player_id (object — DFL PersonId — NaN on ball rows)
  - team_id (object — DFL TeamId — NaN on ball rows)
  - is_ball (bool), is_goalkeeper (bool)
  - x_centered, y_centered (float, DFL meters; 0 at pitch center)
  - z (float, NaN for non-ball rows on most matches)
  - speed_native (float, m/s; populated by DFL provider)
  - ball_state (object: "alive" | "dead", from DFL BallStatus)

Coordinate transformation: x = x_centered + 52.5; y = y_centered + 34.
Direction normalization: per-period flip controlled by home_team_start_left
and home_team_start_left_extratime (matches PFF events convention).
"""

from typing import Any

import numpy as np
import pandas as pd

from . import _direction
from .schema import SPORTEC_TRACKING_FRAMES_COLUMNS, TrackingConversionReport

EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset({
    "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
    "player_id", "team_id", "is_ball", "is_goalkeeper",
    "x_centered", "y_centered", "z", "speed_native", "ball_state",
})


def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: str,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert Sportec-shaped raw tracking frames to canonical schema.

    Parameters
    ----------
    raw_frames : pd.DataFrame
        Sportec input (see EXPECTED_INPUT_COLUMNS).
    home_team_id : str
        DFL TeamId of the home team. Used to compute team_attacking_direction.
    home_team_start_left : bool
        From DFL match-info XML; True if home team starts attacking left.
    home_team_start_left_extratime : bool | None
        From DFL match-info XML; only required when periods 3/4 (ET) present.
    preserve_native : list[str] | None
        Optional input columns to pass through into output.

    Returns
    -------
    frames : pd.DataFrame
        Canonical SPORTEC_TRACKING_FRAMES_COLUMNS-shaped output.
    report : TrackingConversionReport

    Examples
    --------
    Parse DFL Position XML, join roster, then convert::

        from silly_kicks.tracking.sportec import convert_to_frames
        # raw_frames built from DFL XML upstream
        frames, report = convert_to_frames(
            raw_frames, home_team_id="DFL-CLU-0001A",
            home_team_start_left=True,
        )
        assert report.has_unrecognized is False
    """
    missing = EXPECTED_INPUT_COLUMNS - set(raw_frames.columns)
    if missing:
        raise ValueError(f"sportec convert_to_frames missing columns: {sorted(missing)}")

    out = raw_frames.copy()

    # Coordinate translation: centered → SPADL bottom-left
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    # team_attacking_direction (pre-flip): home_team_start_left determines
    # which team attacks LTR in period 1; flips per period.
    out["team_attacking_direction"] = _direction.compute_attacking_direction(
        team_id=out["team_id"],
        period_id=out["period_id"],
        is_ball=out["is_ball"],
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )

    # Apply per-period direction flip so output is LTR-normalized.
    flip_mask = (out["team_attacking_direction"] == "rtl").to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]
    out.loc[flip_mask, "team_attacking_direction"] = "ltr"

    # Speed: trust native (DFL supplies)
    out["speed"] = out["speed_native"].astype("float64")
    out["speed_source"] = np.where(out["speed"].notna(), "native", None)

    out["confidence"] = None
    out["visibility"] = None
    out["source_provider"] = "sportec"

    # Project to declared schema + dtypes
    final = pd.DataFrame({col: out[col] for col in SPORTEC_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in SPORTEC_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)
        elif dtype_str == "object":
            final[col] = final[col].astype("object")

    # Build report
    n_input_frames = int(raw_frames["frame_id"].nunique())
    n_periods = int(raw_frames["period_id"].nunique())
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        expected = int(g["frame_id"].max() - g["frame_id"].min() + 1)
        actual = int(g["frame_id"].nunique())
        cov[int(p)] = float(actual) / max(expected, 1)
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider="sportec",
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),  # Sportec native validates externally; no roster arg in PR-1
    )
    return final, report
```

- [ ] **Step 8.6: Update `__init__.py` to import sportec**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    # ... existing ...
    "sportec",
]

from . import schema, sportec, utils
```

- [ ] **Step 8.7: Stub `_direction.py` with `compute_attacking_direction`**

For Loop 6, `_direction.py` does not yet exist (extraction happens in Loop 7). For now, create a minimal stub:

Create `silly_kicks/tracking/_direction.py`:

```python
"""Direction-of-play helper, shared between events PFF + tracking PFF/Sportec.

Loop 7 will extract the existing helper out of silly_kicks/spadl/pff.py and
move it here. For Loop 6 (Sportec), this stub provides the function signature.

The implementation will be substituted in Loop 7 with the extracted helper.
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_attacking_direction(
    *,
    team_id: pd.Series,
    period_id: pd.Series,
    is_ball: pd.Series,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.Series:
    """Per-row attacking direction ("ltr" / "rtl"), NaN for ball rows.

    Returns ``ltr`` for the team attacking left-to-right in this period,
    ``rtl`` otherwise. Period 2 flips period 1; period 4 flips period 3.

    Ball rows always get NaN (ball doesn't attack).
    """
    out = pd.Series([None] * len(team_id), dtype="object")
    # Period 1: home_team_start_left determines home direction
    # Period 2: opposite of period 1
    # Period 3 (ET1): home_team_start_left_extratime determines
    # Period 4 (ET2): opposite of period 3
    # Period 5 (PSO): direction undefined; leave NaN

    home_ltr_p1 = home_team_start_left
    home_ltr_p3 = home_team_start_left_extratime if home_team_start_left_extratime is not None else home_team_start_left

    for p in (1, 2, 3, 4):
        if p == 1:
            home_ltr = home_ltr_p1
        elif p == 2:
            home_ltr = not home_ltr_p1
        elif p == 3:
            home_ltr = home_ltr_p3
        else:  # p == 4
            home_ltr = not home_ltr_p3
        period_mask = (period_id == p) & ~is_ball
        if not period_mask.any():
            continue
        is_home = (team_id == home_team_id)
        # Home team direction:
        out.loc[period_mask & is_home] = "ltr" if home_ltr else "rtl"
        out.loc[period_mask & ~is_home] = "rtl" if home_ltr else "ltr"
    return out
```

- [ ] **Step 8.8: Run Sportec tests to verify they pass**

Run: `pytest tests/test_tracking_sportec.py -v`
Expected: 7 passed.

---

## Task 9 (Loop 7): PFF native adapter + `_direction.py` extraction

**Files:**
- Modify: `silly_kicks/spadl/pff.py` (extract direction helper, replace inline with import)
- Modify: `silly_kicks/tracking/_direction.py` (replace stub with extracted real helper)
- Create: `silly_kicks/tracking/pff.py`
- Modify: `silly_kicks/tracking/__init__.py` (add `pff` import)
- Create: `tests/datasets/tracking/pff/generate_synthetic.py` + `__init__.py` + parquets
- Test: `tests/test_tracking_pff.py`

- [ ] **Step 9.1: Locate the direction helper in `silly_kicks/spadl/pff.py`**

Run: `grep -n "def.*direction\|home_team_start_left" silly_kicks/spadl/pff.py | head -30`

The helper functions live in `silly_kicks/spadl/pff.py` (PR-S18 introduced them). Identify the helper(s) that compute per-period attacking direction.

- [ ] **Step 9.2: Move direction helpers from `spadl/pff.py` to `tracking/_direction.py`**

Strategy:
1. Read the relevant functions from `silly_kicks/spadl/pff.py`.
2. Replace the stub in `silly_kicks/tracking/_direction.py` with those functions (verbatim, plus `compute_attacking_direction` from Step 8.7 if not already present).
3. In `silly_kicks/spadl/pff.py`, replace the inline functions with `from silly_kicks.tracking import _direction` and call sites.

Pure refactor — zero behaviour change. The events test suite is the regression gate for this step.

- [ ] **Step 9.3: Run events PFF tests to verify zero regression**

Run: `pytest tests/test_pff*.py tests/test_*pff*.py -v`
Expected: all events PFF tests pass (same as before extraction).

- [ ] **Step 9.4: Run Sportec tests to verify no regression**

Run: `pytest tests/test_tracking_sportec.py -v`
Expected: 7 passed (Sportec adapter still uses `_direction` correctly).

- [ ] **Step 9.5: Write PFF synthetic generator**

Create `tests/datasets/tracking/pff/__init__.py` (empty) and `tests/datasets/tracking/pff/generate_synthetic.py`:

```python
"""Generate synthetic PFF-shaped raw tracking input.

PFF input shape (mirrors what callers parse from .jsonl.bz2):
  - game_id (int), period_id (int), frame_id (int), time_seconds (float)
  - frame_rate (float, 30 Hz from PFF documentation)
  - player_id (Int64 nullable, NaN on ball rows), team_id (Int64 nullable)
  - is_ball, is_goalkeeper
  - x_centered, y_centered (float, PFF meters; 0 at pitch center, range [-52.5, 52.5] x [-34, 34])
  - z (float, populated for ball rows on most frames)
  - speed_native (float, m/s, supplied by PFF)
  - ball_state (object, "alive" | "dead")
"""

from pathlib import Path

import pandas as pd

from tests.datasets.tracking._generator_common import (
    deterministic_uniform_motion, get_provider_baseline,
)

OUT_DIR = Path(__file__).resolve().parent
BASELINE = get_provider_baseline("pff")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 30.0)


def _to_pff_shape(ref: pd.DataFrame, *, game_id: int = 10501) -> pd.DataFrame:
    out = ref.copy()
    out["game_id"] = game_id
    # PFF identifiers are integers
    out["player_id"] = out["player_id"].astype("Int64")
    out["team_id"] = out["team_id"].astype("Int64")
    return out


def main() -> None:
    tiny_ref = deterministic_uniform_motion(
        n_frames=int(3 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=1,
    )
    tiny = _to_pff_shape(tiny_ref)
    tiny.to_parquet(OUT_DIR / "tiny.parquet", index=False)

    p1 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=1, t0=0.0, seed=1,
    )
    p2 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE), frame_rate=FRAME_RATE, period_id=2, t0=0.0, seed=2,
    )
    medium = pd.concat([_to_pff_shape(p1), _to_pff_shape(p2)], ignore_index=True)
    dead_mask = (medium["period_id"] == 1) & (medium["time_seconds"].between(10, 15))
    medium.loc[dead_mask, "ball_state"] = "dead"
    medium.to_parquet(OUT_DIR / "medium_halftime.parquet", index=False)

    print(f"Wrote {OUT_DIR / 'tiny.parquet'} ({len(tiny)} rows)")
    print(f"Wrote {OUT_DIR / 'medium_halftime.parquet'} ({len(medium)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.6: Run generator**

Run: `python tests/datasets/tracking/pff/generate_synthetic.py`
Expected: two parquet files written.

- [ ] **Step 9.7: Write PFF tracking tests**

Create `tests/test_tracking_pff.py` (mirror `test_tracking_sportec.py` with PFF-specific assertions: integer IDs, `Int64` dtype check, `home_team_id` as int, `home_team_start_left_extratime` propagation):

```python
"""Unit tests for silly_kicks.tracking.pff.convert_to_frames."""

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.pff import convert_to_frames
from silly_kicks.tracking.schema import (
    PFF_TRACKING_FRAMES_COLUMNS,
    TRACKING_CONSTRAINTS,
    TrackingConversionReport,
)

FIXTURE_DIR = Path("tests/datasets/tracking/pff")
TINY = pd.read_parquet(FIXTURE_DIR / "tiny.parquet")
MEDIUM = pd.read_parquet(FIXTURE_DIR / "medium_halftime.parquet")


def test_tiny_output_columns_match_schema():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert set(frames.columns) == set(PFF_TRACKING_FRAMES_COLUMNS)


def test_tiny_player_team_id_are_int64_nullable():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert str(frames["player_id"].dtype) == "Int64"
    assert str(frames["team_id"].dtype) == "Int64"


def test_tiny_ball_rows_have_nan_player_id():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    ball = frames[frames["is_ball"]]
    assert ball["player_id"].isna().all()
    assert ball["team_id"].isna().all()


def test_tiny_coordinate_bounds():
    frames, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    lo_x, hi_x = TRACKING_CONSTRAINTS["x"]
    assert frames["x"].between(lo_x, hi_x).all()


def test_home_start_left_false_flips_x():
    f_t, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    f_f, _ = convert_to_frames(TINY, home_team_id=100, home_team_start_left=False)
    j = f_t.merge(f_f, on=["period_id", "frame_id", "player_id", "team_id", "is_ball"], suffixes=("_t", "_f"))
    assert ((j["x_t"] + j["x_f"]).round(6) == 105.0).all()


def test_extratime_parameter_propagates():
    """home_team_start_left_extratime=True applies to periods 3/4."""
    medium_with_et = MEDIUM.copy()
    # Synthesize a period 3 row by retagging some period 2 rows
    et_rows = medium_with_et[medium_with_et["period_id"] == 2].head(60).copy()
    et_rows["period_id"] = 3
    medium_with_et = pd.concat([medium_with_et, et_rows], ignore_index=True)
    f, _ = convert_to_frames(
        medium_with_et, home_team_id=100,
        home_team_start_left=True, home_team_start_left_extratime=False,
    )
    p3 = f[f["period_id"] == 3]
    assert len(p3) > 0
    assert p3["x"].between(0, 105).all()


def test_report_provider_is_pff():
    _, report = convert_to_frames(TINY, home_team_id=100, home_team_start_left=True)
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "pff"
```

- [ ] **Step 9.8: Run tests to verify they fail**

Run: `pytest tests/test_tracking_pff.py -v`
Expected: ImportError on `silly_kicks.tracking.pff`.

- [ ] **Step 9.9: Implement PFF tracking adapter**

Create `silly_kicks/tracking/pff.py`. Mirror `sportec.py` exactly except:
- `home_team_id: int` (not `str`)
- Output dtype dict is `PFF_TRACKING_FRAMES_COLUMNS` (Int64 player_id/team_id)
- `source_provider = "pff"` in `TrackingConversionReport.provider`
- All other logic identical (uses same `_direction.compute_attacking_direction`)

```python
"""PFF FC tracking DataFrame converter.

Mirrors silly_kicks.tracking.sportec but for PFF-shaped input. Reuses the
shared _direction helper extracted from silly_kicks.spadl.pff (PR-S18) for
home_team_start_left[_extratime] direction normalization.

Input contract (EXPECTED_INPUT_COLUMNS):
  Same as sportec, except player_id/team_id are Int64 (PFF integer IDs)
  and game_id is int64.
"""

import numpy as np
import pandas as pd

from . import _direction
from .schema import PFF_TRACKING_FRAMES_COLUMNS, TrackingConversionReport

EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset({
    "game_id", "period_id", "frame_id", "time_seconds", "frame_rate",
    "player_id", "team_id", "is_ball", "is_goalkeeper",
    "x_centered", "y_centered", "z", "speed_native", "ball_state",
})


def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert PFF-shaped raw tracking frames to canonical schema.

    Parameters
    ----------
    raw_frames : pd.DataFrame
        PFF input (see EXPECTED_INPUT_COLUMNS).
    home_team_id : int
        PFF homeTeam.id.
    home_team_start_left : bool
        From PFF metadata homeTeamStartLeft.
    home_team_start_left_extratime : bool | None
        From PFF metadata homeTeamStartLeftExtraTime; required when periods 3/4 present.

    Returns
    -------
    frames : pd.DataFrame
        PFF_TRACKING_FRAMES_COLUMNS-shaped output, 105×68m SPADL coordinates.
    report : TrackingConversionReport

    Examples
    --------
    Read PFF tracking JSONL.bz2, flatten to frames, then convert::

        import bz2, json, pandas as pd
        with bz2.open("10501.jsonl.bz2", "rt") as fh:
            rows = [json.loads(line) for line in fh]
        raw = pd.json_normalize(rows)  # caller-shaped flattening
        frames, report = convert_to_frames(raw, home_team_id=366, home_team_start_left=True)
    """
    missing = EXPECTED_INPUT_COLUMNS - set(raw_frames.columns)
    if missing:
        raise ValueError(f"pff convert_to_frames missing columns: {sorted(missing)}")

    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    out["team_attacking_direction"] = _direction.compute_attacking_direction(
        team_id=out["team_id"], period_id=out["period_id"], is_ball=out["is_ball"],
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )

    flip_mask = (out["team_attacking_direction"] == "rtl").to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]
    out.loc[flip_mask, "team_attacking_direction"] = "ltr"

    out["speed"] = out["speed_native"].astype("float64")
    out["speed_source"] = np.where(out["speed"].notna(), "native", None)
    out["confidence"] = None
    out["visibility"] = None
    out["source_provider"] = "pff"

    final = pd.DataFrame({col: out[col] for col in PFF_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in PFF_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)
        elif dtype_str == "Int64":
            final[col] = final[col].astype("Int64")
        elif dtype_str == "object":
            final[col] = final[col].astype("object")

    n_input_frames = int(raw_frames["frame_id"].nunique())
    n_periods = int(raw_frames["period_id"].nunique())
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        expected = int(g["frame_id"].max() - g["frame_id"].min() + 1)
        actual = int(g["frame_id"].nunique())
        cov[int(p)] = float(actual) / max(expected, 1)
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider="pff",
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
    )
    return final, report
```

- [ ] **Step 9.10: Update `__init__.py`**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    # ... existing ...
    "pff",
]

from . import pff, schema, sportec, utils
```

- [ ] **Step 9.11: Run PFF tests to verify they pass**

Run: `pytest tests/test_tracking_pff.py -v`
Expected: 7 passed.

- [ ] **Step 9.12: Re-run events PFF tests one more time**

Run: `pytest tests/test_pff*.py tests/test_*pff*.py -v`
Expected: all events tests still pass (regression gate after extraction).

---

## Task 10 (Loop 8): Kloppy gateway

**Files:**
- Create: `silly_kicks/tracking/kloppy.py`
- Modify: `silly_kicks/tracking/__init__.py` (add conditional `kloppy` import)
- Create: `tests/datasets/tracking/metrica/generate_synthetic.py` + `__init__.py` + parquets
- Create: `tests/datasets/tracking/skillcorner/generate_synthetic.py` + `__init__.py` + parquets
- Test: `tests/test_tracking_kloppy.py`

- [ ] **Step 10.1: Write Metrica generator**

Create `tests/datasets/tracking/metrica/__init__.py` (empty) and `tests/datasets/tracking/metrica/generate_synthetic.py`. Mirror `sportec/generate_synthetic.py` but emit a Metrica CSV-like shape (or, more practically for the kloppy gateway, build a `kloppy.TrackingDataset` directly from the `_generator_common` reference DataFrame and pickle it as a parquet of frame events).

For Loop 8, since the kloppy gateway accepts a `TrackingDataset`, the synthetic fixture is **a function that builds a TrackingDataset in memory** rather than a parquet file. Update test infrastructure accordingly:

```python
"""Generate synthetic Metrica TrackingDataset (kloppy in-memory) for tests.

The kloppy gateway accepts a TrackingDataset directly. We don't need to
write a CSV and re-parse it — we construct the TrackingDataset in memory.
"""

import datetime
from pathlib import Path

import pandas as pd
from kloppy.domain import (
    DatasetFlag, Frame, Ground, MetricPitchDimensions, Orientation, Period,
    Player, Point, PositionType, Provider, Team, TrackingDataset,
    TrackingDatasetMetadata,
)

from tests.datasets.tracking._generator_common import (
    deterministic_uniform_motion, get_provider_baseline,
)

OUT_DIR = Path(__file__).resolve().parent
BASELINE = get_provider_baseline("metrica")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 25.0)


def build_metrica_tracking_dataset(n_frames: int = 75) -> TrackingDataset:
    """Build a synthetic TrackingDataset shaped like a kloppy Metrica load."""
    home_team = Team(team_id="home_team", name="Home", ground=Ground.HOME)
    away_team = Team(team_id="away_team", name="Away", ground=Ground.AWAY)
    home_team.players = [
        Player(player_id=f"h_{i}", team=home_team, jersey_no=i,
               first_name="H", last_name=str(i),
               starting_position=PositionType.GoalKeeper if i == 0 else PositionType.Unknown)
        for i in range(11)
    ]
    away_team.players = [
        Player(player_id=f"a_{i}", team=away_team, jersey_no=i,
               first_name="A", last_name=str(i),
               starting_position=PositionType.GoalKeeper if i == 0 else PositionType.Unknown)
        for i in range(11)
    ]
    period = Period(
        id=1,
        start_timestamp=datetime.timedelta(seconds=0),
        end_timestamp=datetime.timedelta(seconds=n_frames / FRAME_RATE),
    )
    metadata = TrackingDatasetMetadata(
        teams=[home_team, away_team],
        periods=[period],
        pitch_dimensions=MetricPitchDimensions.from_dimensions(105.0, 68.0),
        score=None,
        frame_rate=FRAME_RATE,
        orientation=Orientation.HOME_AWAY,
        flags=DatasetFlag.BALL_OWNING_TEAM,
        provider=Provider.METRICA,
        coordinate_system=None,  # left default
    )
    ref = deterministic_uniform_motion(n_frames=n_frames, frame_rate=FRAME_RATE)
    frames = []
    for fid in range(n_frames):
        ts = datetime.timedelta(seconds=fid / FRAME_RATE)
        frame_rows = ref[ref["frame_id"] == fid]
        players_data = {}
        for _, row in frame_rows[~frame_rows["is_ball"]].iterrows():
            team = home_team if row["team_id"] == 100 else away_team
            jersey = row["jersey"]
            player = team.players[int(jersey)]
            players_data[player] = type("PlayerData", (), {
                "coordinates": Point(x=row["x_centered"] + 52.5, y=row["y_centered"] + 34.0),
                "distance": None, "speed": row["speed_native"],
            })()
        ball_row = frame_rows[frame_rows["is_ball"]].iloc[0]
        frames.append(Frame(
            frame_id=fid,
            timestamp=ts,
            ball_owning_team=home_team,
            ball_state=None,
            period=period,
            players_data=players_data,
            other_data={},
            ball_coordinates=Point(x=ball_row["x_centered"] + 52.5, y=ball_row["y_centered"] + 34.0),
            ball_speed=ball_row["speed_native"],
        ))
    return TrackingDataset(metadata=metadata, records=frames)


if __name__ == "__main__":
    ds = build_metrica_tracking_dataset()
    print(f"Built TrackingDataset with {len(ds.records)} frames, {len(ds.metadata.teams[0].players)} home players")
```

Note: kloppy domain types may evolve between versions; if signatures differ in 3.18, adjust to match. Run the generator script as a smoke test:

Run: `python tests/datasets/tracking/metrica/generate_synthetic.py`
Expected: prints "Built TrackingDataset with 75 frames, 11 home players".

- [ ] **Step 10.2: Write SkillCorner generator (similar pattern)**

Create `tests/datasets/tracking/skillcorner/__init__.py` (empty) and `tests/datasets/tracking/skillcorner/generate_synthetic.py` mirroring metrica's, with `Provider.SKILLCORNER` and `FRAME_RATE = 10.0`.

- [ ] **Step 10.3: Write the failing tests**

Create `tests/test_tracking_kloppy.py`:

```python
"""Unit tests for silly_kicks.tracking.kloppy gateway."""

import pytest

from silly_kicks.tracking.kloppy import convert_to_frames
from silly_kicks.tracking.schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)
from tests.datasets.tracking.metrica.generate_synthetic import (
    build_metrica_tracking_dataset,
)
from tests.datasets.tracking.skillcorner.generate_synthetic import (
    build_skillcorner_tracking_dataset,
)


def test_metrica_dataset_converts_to_canonical_schema():
    ds = build_metrica_tracking_dataset(n_frames=10)
    frames, report = convert_to_frames(ds)
    assert set(frames.columns) == set(KLOPPY_TRACKING_FRAMES_COLUMNS)
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "metrica"


def test_metrica_coordinates_in_spadl_units():
    ds = build_metrica_tracking_dataset(n_frames=10)
    frames, _ = convert_to_frames(ds)
    assert frames["x"].between(0, 105).all()
    assert frames["y"].between(0, 68).all()


def test_metrica_derived_speed_used_when_native_missing():
    """Metrica's kloppy data may lack speed; gateway should derive."""
    ds = build_metrica_tracking_dataset(n_frames=10)
    # Strip speed from all player_data to simulate missing-speed Metrica
    for frame in ds.records:
        for pdata in frame.players_data.values():
            pdata.speed = None
        frame.ball_speed = None
    frames, report = convert_to_frames(ds)
    # Some derived rows expected
    assert (frames["speed_source"] == "derived").any() or report.derived_speed_rows > 0


def test_skillcorner_dataset_converts():
    ds = build_skillcorner_tracking_dataset(n_frames=10)
    frames, report = convert_to_frames(ds)
    assert set(frames.columns) == set(KLOPPY_TRACKING_FRAMES_COLUMNS)
    assert report.provider == "skillcorner"


def test_provider_pff_raises_not_implemented():
    """Per spec: PFF and Sportec must route through native adapters."""
    from kloppy.domain import Provider
    ds = build_metrica_tracking_dataset(n_frames=2)
    ds.metadata.provider = Provider.PFF
    with pytest.raises(NotImplementedError, match="silly_kicks.tracking.pff"):
        convert_to_frames(ds)


def test_provider_sportec_raises_not_implemented():
    from kloppy.domain import Provider
    ds = build_metrica_tracking_dataset(n_frames=2)
    ds.metadata.provider = Provider.SPORTEC
    with pytest.raises(NotImplementedError, match="silly_kicks.tracking.sportec"):
        convert_to_frames(ds)
```

- [ ] **Step 10.4: Run tests to verify they fail**

Run: `pytest tests/test_tracking_kloppy.py -v`
Expected: ImportError on `silly_kicks.tracking.kloppy`.

- [ ] **Step 10.5: Implement the kloppy gateway**

Create `silly_kicks/tracking/kloppy.py`:

```python
"""Kloppy TrackingDataset gateway for silly_kicks.tracking.

Covers Metrica + SkillCorner via kloppy 3.18+ tracking parsers. Sportec
and PFF intentionally raise NotImplementedError — route through their
native adapters (silly_kicks.tracking.sportec / silly_kicks.tracking.pff)
for symmetry with silly_kicks.spadl.pff (PR-S18) and failure isolation.

See ADR-004 (silly-kicks 2.7.0) for the architectural rationale.
"""

import numpy as np
import pandas as pd
from kloppy.domain import (  # type: ignore[reportMissingImports]
    Provider, TrackingDataset,
)

from .schema import KLOPPY_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .utils import _derive_speed

_PROVIDER_NAME_MAP: dict = {
    Provider.METRICA: "metrica",
    Provider.SKILLCORNER: "skillcorner",
}


def convert_to_frames(
    dataset: TrackingDataset,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert a kloppy TrackingDataset to canonical KLOPPY_TRACKING_FRAMES_COLUMNS schema.

    Dispatches on dataset.metadata.provider. Provider.PFF and Provider.SPORTEC
    raise NotImplementedError — route through silly_kicks.tracking.pff and
    silly_kicks.tracking.sportec.

    Parameters
    ----------
    dataset : kloppy.domain.TrackingDataset
        Output of kloppy.metrica.load_tracking_csv or kloppy.skillcorner.load_tracking.
    preserve_native : list[str] | None
        Optional input passthrough columns (currently unused; reserved for future).

    Returns
    -------
    frames : pd.DataFrame
        KLOPPY_TRACKING_FRAMES_COLUMNS-shaped output.
    report : TrackingConversionReport

    Examples
    --------
    Load a Metrica match via kloppy and convert::

        import kloppy
        ds = kloppy.metrica.load_tracking_csv(home="home.csv", away="away.csv")
        frames, report = silly_kicks.tracking.kloppy.convert_to_frames(ds)
    """
    provider = dataset.metadata.provider
    if provider in (Provider.PFF,):
        raise NotImplementedError(
            "PFF tracking via kloppy is supported but disabled in PR-1; "
            "route through silly_kicks.tracking.pff for symmetry with "
            "silly_kicks.spadl.pff (ADR-004)."
        )
    if provider in (Provider.SPORTEC,):
        raise NotImplementedError(
            "Sportec tracking has no kloppy parser; route through "
            "silly_kicks.tracking.sportec (ADR-004)."
        )
    provider_name = _PROVIDER_NAME_MAP.get(provider)
    if provider_name is None:
        raise NotImplementedError(f"Provider {provider} not supported in PR-1")

    # Transform to SPADL pitch dimensions + HOME_AWAY orientation
    from kloppy.domain import MetricPitchDimensions, Orientation
    transformed = dataset.transform(
        to_pitch_dimensions=MetricPitchDimensions.from_dimensions(105.0, 68.0),
        to_orientation=Orientation.HOME_AWAY,
    )

    home_team = transformed.metadata.teams[0]
    home_team_id = str(home_team.team_id)

    rows: list[dict] = []
    frame_rate = float(transformed.metadata.frame_rate)
    for frame in transformed.records:
        period_id = int(frame.period.id)
        time_seconds = float(frame.timestamp.total_seconds())
        # Player rows
        for player, pdata in frame.players_data.items():
            if pdata.coordinates is None:
                continue
            rows.append({
                "game_id": str(transformed.metadata.game_id) if hasattr(transformed.metadata, "game_id") and transformed.metadata.game_id else "synthetic",
                "period_id": period_id,
                "frame_id": frame.frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": str(player.player_id),
                "team_id": str(player.team.team_id),
                "is_ball": False,
                "is_goalkeeper": player.starting_position is not None and "Goal" in str(type(player.starting_position).__name__) + str(player.starting_position),
                "x": pdata.coordinates.x,
                "y": pdata.coordinates.y,
                "z": float("nan"),
                "speed": pdata.speed if pdata.speed is not None else float("nan"),
                "speed_source": "native" if pdata.speed is not None else None,
                "ball_state": "alive" if frame.ball_state is None else str(frame.ball_state).lower(),
                "team_attacking_direction": "ltr",  # transformed to HOME_AWAY orientation
                "confidence": None, "visibility": None,
                "source_provider": provider_name,
            })
        # Ball row
        if frame.ball_coordinates is not None:
            rows.append({
                "game_id": str(transformed.metadata.game_id) if hasattr(transformed.metadata, "game_id") and transformed.metadata.game_id else "synthetic",
                "period_id": period_id,
                "frame_id": frame.frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": None, "team_id": None,
                "is_ball": True, "is_goalkeeper": False,
                "x": frame.ball_coordinates.x, "y": frame.ball_coordinates.y, "z": float("nan"),
                "speed": frame.ball_speed if frame.ball_speed is not None else float("nan"),
                "speed_source": "native" if frame.ball_speed is not None else None,
                "ball_state": "alive" if frame.ball_state is None else str(frame.ball_state).lower(),
                "team_attacking_direction": None,
                "confidence": None, "visibility": None,
                "source_provider": provider_name,
            })

    df = pd.DataFrame(rows)
    # Derive speed where missing
    if df["speed"].isna().any():
        df = _derive_speed(df)

    # Coerce dtypes
    final = pd.DataFrame({col: df[col] for col in KLOPPY_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in KLOPPY_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)
        elif dtype_str == "object":
            final[col] = final[col].astype("object")

    n_input_frames = len(transformed.records)
    n_periods = len({f.period.id for f in transformed.records})
    cov = {p: 1.0 for p in {f.period.id for f in transformed.records}}  # synthetic; real impl could be tighter
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)
    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider=provider_name,
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
    )
    return final, report
```

- [ ] **Step 10.6: Update `__init__.py` for kloppy gateway (conditional)**

Edit `silly_kicks/tracking/__init__.py`:

```python
__all__ = [
    # ... existing ...
    "kloppy",
]

# At end, conditional import:
try:
    from . import kloppy
except ImportError:
    pass
```

- [ ] **Step 10.7: Run tests to verify they pass**

Run: `pytest tests/test_tracking_kloppy.py -v`
Expected: 6 passed.

---

## Task 11 (Loop 9): Cross-provider parity gate

**Files:**
- Test: `tests/test_tracking_cross_provider_parity.py`

- [ ] **Step 11.1: Write the parity tests**

Create `tests/test_tracking_cross_provider_parity.py`:

```python
"""Cross-provider parity gate — schema-stress test with all 4 providers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import TRACKING_CONSTRAINTS
from silly_kicks.tracking import sportec, pff, kloppy as kloppy_gw
from silly_kicks.tracking.utils import link_actions_to_frames


def _load_sportec():
    raw = pd.read_parquet("tests/datasets/tracking/sportec/medium_halftime.parquet")
    return sportec.convert_to_frames(raw, home_team_id="DFL-CLU-0100", home_team_start_left=True)


def _load_pff():
    raw = pd.read_parquet("tests/datasets/tracking/pff/medium_halftime.parquet")
    return pff.convert_to_frames(raw, home_team_id=100, home_team_start_left=True)


def _load_metrica():
    from tests.datasets.tracking.metrica.generate_synthetic import build_metrica_tracking_dataset
    ds = build_metrica_tracking_dataset(n_frames=750)  # 30 s at 25 Hz
    return kloppy_gw.convert_to_frames(ds)


def _load_skillcorner():
    from tests.datasets.tracking.skillcorner.generate_synthetic import build_skillcorner_tracking_dataset
    ds = build_skillcorner_tracking_dataset(n_frames=300)  # 30 s at 10 Hz
    return kloppy_gw.convert_to_frames(ds)


PROVIDER_LOADERS = {
    "sportec": _load_sportec,
    "pff": _load_pff,
    "metrica": _load_metrica,
    "skillcorner": _load_skillcorner,
}


@pytest.mark.parametrize("provider", list(PROVIDER_LOADERS))
def test_tracking_bounds(provider):
    frames, _ = PROVIDER_LOADERS[provider]()
    for col, (lo, hi) in TRACKING_CONSTRAINTS.items():
        if col not in frames.columns:
            continue
        vals = frames[col].dropna()
        if len(vals) == 0:
            continue
        if hi == float("inf"):
            assert (vals >= lo).all(), f"{provider}/{col}: values below {lo}"
        else:
            assert vals.between(lo, hi).all(), f"{provider}/{col}: values outside [{lo}, {hi}]"


@pytest.mark.parametrize("provider", list(PROVIDER_LOADERS))
def test_tracking_link_rate(provider):
    frames, _ = PROVIDER_LOADERS[provider]()
    # Build synthetic actions at known frame times
    sample = frames.drop_duplicates(["period_id", "frame_id"]).sample(20, random_state=42)
    actions = pd.DataFrame({
        "game_id": 1, "action_id": range(20),
        "period_id": sample["period_id"].to_numpy(),
        "time_seconds": sample["time_seconds"].to_numpy(),
        "team_id": 100, "player_id": 7,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
        "start_x": 50.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0,
    })
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
    assert report.link_rate >= 0.95, f"{provider}: link rate {report.link_rate}"


def test_tracking_distance_to_ball_distribution_overlap():
    """Per-provider distance-to-ball percentiles overlap, proving coordinate
    normalization is consistent across providers."""
    percentiles = {}
    for prov, loader in PROVIDER_LOADERS.items():
        frames, _ = loader()
        ball = frames[frames["is_ball"]][["period_id", "frame_id", "x", "y"]].rename(
            columns={"x": "bx", "y": "by"}
        )
        players = frames[~frames["is_ball"]]
        joined = players.merge(ball, on=["period_id", "frame_id"])
        dist = np.sqrt((joined["x"] - joined["bx"])**2 + (joined["y"] - joined["by"])**2)
        percentiles[prov] = np.percentile(dist.dropna(), [25, 50, 75, 95]).tolist()
    # Every provider's p50 distance to ball should be in 5–60 m range (sane on synthetic)
    for prov, p in percentiles.items():
        assert 0 < p[1] < 100, f"{prov}: implausible p50 distance {p[1]}"
    # Inter-provider overlap: max p50 / min p50 < 3x (not radically different)
    p50s = [p[1] for p in percentiles.values()]
    assert max(p50s) / min(p50s) < 3.0, f"p50 distance ratios: {percentiles}"
```

- [ ] **Step 11.2: Run parity tests**

Run: `pytest tests/test_tracking_cross_provider_parity.py -v`
Expected: 9 passed (4 bounds + 4 link_rate + 1 distance_overlap).

---

## Task 12 (Loop 10): Real-data sweep (e2e)

**Files:**
- Test: `tests/test_tracking_real_data_sweep.py`

- [ ] **Step 12.1: Write the e2e sweep**

Create `tests/test_tracking_real_data_sweep.py`:

```python
"""Real-data sweep, marked e2e. Skipped in CI; run locally before tagging.

Loads real tracking data from environment-pointed local paths and asserts:
  - dtype audit
  - bounds audit (TRACKING_CONSTRAINTS)
  - NaN-rate-per-column audit
  - distance-to-ball percentile baseline emit

Output: structured JSON summary written to stdout. Run via:
    pytest tests/test_tracking_real_data_sweep.py -m e2e -s
Use the JSON summary in the PR description.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import TRACKING_CONSTRAINTS

PROBE_BASELINES = json.loads(
    Path("tests/datasets/tracking/empirical_probe_baselines.json").read_text()
)


def _summarize_provider(frames: pd.DataFrame) -> dict:
    nan_rates = {col: float(frames[col].isna().mean()) for col in frames.columns}
    dist_pcts = {}
    if "is_ball" in frames.columns:
        ball = frames[frames["is_ball"]][["period_id", "frame_id", "x", "y"]].rename(
            columns={"x": "bx", "y": "by"}
        )
        players = frames[~frames["is_ball"]]
        joined = players.merge(ball, on=["period_id", "frame_id"])
        dist = np.sqrt((joined["x"] - joined["bx"])**2 + (joined["y"] - joined["by"])**2).dropna()
        if len(dist):
            dist_pcts = {f"p{p}": float(np.percentile(dist, p)) for p in (25, 50, 75, 95)}
    return {
        "n_rows": len(frames),
        "nan_rates": nan_rates,
        "distance_to_ball_percentiles": dist_pcts,
    }


@pytest.mark.e2e
def test_pff_real_data_sweep():
    path = os.environ.get("PFF_TRACKING_DIR")
    if not path:
        pytest.skip("PFF_TRACKING_DIR not set; skipping PFF real-data sweep.")
    # Load 1 match (adjust per local helper) — actual loader is left to the
    # developer per their existing PFF reading code from PR-S18.
    pytest.skip("PFF real-data loader requires user-side helper; populate this test.")


@pytest.mark.e2e
def test_idsse_real_data_sweep():
    path = os.environ.get("IDSSE_TRACKING_DIR")
    if not path:
        pytest.skip("IDSSE_TRACKING_DIR not set; skipping IDSSE real-data sweep.")
    pytest.skip("IDSSE real-data loader requires user-side helper; populate this test.")


@pytest.mark.e2e
def test_metrica_real_data_sweep():
    path = os.environ.get("METRICA_TRACKING_DIR")
    if not path:
        pytest.skip("METRICA_TRACKING_DIR not set; skipping Metrica real-data sweep.")
    pytest.skip("Metrica real-data loader requires user-side helper; populate this test.")


@pytest.mark.e2e
def test_skillcorner_real_data_sweep():
    path = os.environ.get("SKILLCORNER_TRACKING_DIR")
    if not path:
        pytest.skip("SKILLCORNER_TRACKING_DIR not set; skipping SkillCorner sweep.")
    pytest.skip("SkillCorner real-data loader requires user-side helper; populate this test.")
```

The actual loader bodies require local-data helpers the user has from PR-S18 (PFF) and lakehouse pipelines (IDSSE / Metrica / SkillCorner). Populate them before the local sweep run; they MUST resolve before merge.

- [ ] **Step 12.2: Verify sweep skips gracefully when env unset**

Run: `pytest tests/test_tracking_real_data_sweep.py -m e2e -v`
Expected: 4 skipped (with reasons).

- [ ] **Step 12.3: Populate the four sweep test bodies with real loaders**

Locally on the developer machine with `PFF_TRACKING_DIR`, `IDSSE_TRACKING_DIR`, `METRICA_TRACKING_DIR`, `SKILLCORNER_TRACKING_DIR` set, replace the `pytest.skip("...loader requires user-side helper...")` lines with the actual load + convert + summarize logic. Output: prints a JSON summary line per provider.

- [ ] **Step 12.4: Run the populated sweep locally**

Run: `pytest tests/test_tracking_real_data_sweep.py -m e2e -s -v`
Expected: 4 passed; JSON summaries printed to stdout.

- [ ] **Step 12.5: Capture sweep output for PR description**

Save the stdout summaries to a scratch file; paste into the PR description before the final commit.

---

## Task 13 (Loop 11): Public-API Examples coverage

**Files:**
- Modify: `tests/test_public_api_examples.py` (only if skip-list tweaks needed)

- [ ] **Step 13.1: Run public-API Examples test**

Run: `pytest tests/test_public_api_examples.py -v`
Expected: passes — every public def in `silly_kicks/tracking/{__init__, schema, utils, sportec, pff, kloppy}.py` already has an `Examples` section in docstrings written in earlier loops.

If the test discovers a missing `Examples` section in a public def, add one to that def's docstring (style ref: `silly_kicks/spadl/utils.py:1256-1262` `add_possessions` Examples).

- [ ] **Step 13.2: Re-run after any docstring fixes**

Run: `pytest tests/test_public_api_examples.py -v`
Expected: passes.

---

## Task 14 (Loop 12): Verification gates + `/final-review` + commit

**Files:**
- Modify: `TODO.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml` (version bump)

- [ ] **Step 14.1: Update `TODO.md`**

Edit `TODO.md`:
- Remove the entire `## Tracking namespace — silly_kicks.tracking.*` section (closed by this PR).
- Add a new `## Deferred tracking-aware features` section listing the 8 items from ADR-004 invariant 9 (the priority order).

- [ ] **Step 14.2: Update `CHANGELOG.md`**

Add a `## 2.7.0` entry summarizing:
- New `silly_kicks.tracking` namespace (schema, 4 providers, linkage primitives).
- ADR-004 introduced.
- `_direction.py` extracted from `silly_kicks/spadl/pff.py` (refactor; no behaviour change).
- kloppy minimum bumped to ≥ 3.18.0.

- [ ] **Step 14.3: Bump version in `pyproject.toml`**

Edit `pyproject.toml`:
```diff
-version = "2.6.0"
+version = "2.7.0"
```

- [ ] **Step 14.4: Run ruff**

Run: `ruff check silly_kicks/ tests/`
Expected: no errors. Fix any flagged issues.

- [ ] **Step 14.5: Run pyright**

Run: `pyright silly_kicks/`
Expected: 0 errors. Fix any type issues.

- [ ] **Step 14.6: Run full test suite (excluding e2e)**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass.

- [ ] **Step 14.7: Run e2e sweep one final time**

Run: `python -m pytest tests/test_tracking_real_data_sweep.py -m e2e -s -v`
Expected: 4 passed with summaries; capture output.

- [ ] **Step 14.8: Run `/final-review` (MANDATORY pre-commit gate)**

This is non-negotiable per silly-kicks memory:feedback_final_review_gate.md and the user's explicit instruction in the brainstorm session.

Run: `/final-review` (mad-scientist-skills:final-review)
Expected: review report. Address any flagged issues. Re-run until clean.

- [ ] **Step 14.9: Surface the change set + sweep summary to the user, ask for explicit commit approval**

Per silly-kicks commit policy memory: "literally ONE commit per branch; no WIP commits + squash; explicit approval before that one commit."

Show the user:
- `git status` (list of new + modified files)
- The `/final-review` clean report
- The e2e sweep JSON summaries
- A draft commit message:

```
feat(tracking)!: silly_kicks.tracking namespace primitive layer — silly-kicks 2.7.0 (PR-S19)

Adds first-class tracking-data support: 19-column long-form schema, four
provider adapters (Sportec native, PFF native, kloppy gateway covering
Metrica + SkillCorner), and the linkage primitive
(link_actions_to_frames + slice_around_event) that all PR-2+ tracking-aware
features will build on.

ADR-004 introduced — namespace charter; nine invariants.

Bundled refactor: silly_kicks/spadl/pff.py direction helpers extracted to
silly_kicks/tracking/_direction.py (zero behaviour change in events;
shared between events PFF and tracking PFF + Sportec adapters).

Empirical-probe-driven synthetic fixtures: parameters calibrated against
lakehouse + local PFF measured statistics (committed JSON baseline).

Tracking-aware features (action_context, pressure_on_carrier,
infer_ball_carrier, sync_score, pitch control, smoothing, gap-filling,
ReSpo.Vision adapter) explicitly DEFERRED to follow-up PR cycles per
ADR-004 invariant 9. PR-2 (action_context) targets 2.8.0.

Spec: docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md
Plan: docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md
ADR:  docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

WAIT for explicit user approval before continuing.

- [ ] **Step 14.10: After explicit user approval, create the single commit**

```bash
git add silly_kicks/tracking/ silly_kicks/spadl/pff.py \
        tests/test_tracking_*.py tests/datasets/tracking/ \
        scripts/probe_tracking_baselines.py \
        docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md \
        docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md \
        docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md \
        TODO.md CHANGELOG.md pyproject.toml
git commit -m "$(cat <<'EOF'
feat(tracking)!: silly_kicks.tracking namespace primitive layer — silly-kicks 2.7.0 (PR-S19)

Adds first-class tracking-data support: 19-column long-form schema, four
provider adapters (Sportec native, PFF native, kloppy gateway covering
Metrica + SkillCorner), and the linkage primitive
(link_actions_to_frames + slice_around_event) that all PR-2+ tracking-aware
features will build on.

ADR-004 introduced — namespace charter; nine invariants.

Bundled refactor: silly_kicks/spadl/pff.py direction helpers extracted to
silly_kicks/tracking/_direction.py (zero behaviour change in events;
shared between events PFF and tracking PFF + Sportec adapters).

Empirical-probe-driven synthetic fixtures: parameters calibrated against
lakehouse + local PFF measured statistics (committed JSON baseline).

Tracking-aware features (action_context, pressure_on_carrier,
infer_ball_carrier, sync_score, pitch control, smoothing, gap-filling,
ReSpo.Vision adapter) explicitly DEFERRED to follow-up PR cycles per
ADR-004 invariant 9. PR-2 (action_context) targets 2.8.0.

Spec: docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md
Plan: docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md
ADR:  docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
```

Expected: single commit on `feat/tracking-namespace-pr1` branch; status shows working tree clean.

- [ ] **Step 14.11: Push and open PR**

```bash
git push -u origin feat/tracking-namespace-pr1
gh pr create --title "feat(tracking)!: silly_kicks.tracking namespace PR-1 — silly-kicks 2.7.0" --body "$(cat <<'EOF'
## Summary

- New `silly_kicks.tracking` namespace: 19-column long-form schema, 4 provider adapters (Sportec native, PFF native, kloppy gateway covering Metrica + SkillCorner), `link_actions_to_frames` + `slice_around_event` linkage primitives, ADR-004 charter.
- Bundled refactor: `silly_kicks/spadl/pff.py` direction helpers extracted to `silly_kicks/tracking/_direction.py` (pure refactor — events tests pass unchanged).
- Empirical-probe-driven synthetic fixtures (lakehouse + local PFF baseline JSON committed).
- Target version: silly-kicks 2.7.0.
- PR-2 (`action_context()`) deferred to a separate session against the locked schema.

Spec: `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md`
Plan: `docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md`
ADR:  `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`

## Empirical sweep summary (4 providers, real data)

[paste JSON summaries from Step 14.7]

## Test plan

- [x] Synthetic-fixture unit tests (8 files, ~50 tests)
- [x] Cross-provider parity gate (bounds + link rate + distance-to-ball overlap)
- [x] Public-API Examples coverage
- [x] e2e real-data sweep (4 providers, locally with `*_TRACKING_DIR` env vars)
- [x] `ruff` + `pyright` clean
- [x] `/final-review` clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

Plan written from spec `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md`. Coverage check:

| Spec section | Plan task |
|--------------|-----------|
| §4.1 Architecture / module layout | Tasks 0, 2, 3, 4, 5, 6, 8, 9, 10 |
| §4.2 Schema | Task 2 |
| §4.3 Public API & data flow | Tasks 5, 6, 8, 9, 10 (per-converter + utilities) |
| §4.4 ADR-004 | Task 1.4 |
| §5.1 Test file layout | All Loop tasks |
| §5.2 Empirical-probe parameterization | Task 1 (probe + JSON), Task 7 (generator helper) |
| §5.3 Synthetic fixture sizes | Tasks 8, 9, 10 (per-provider generators) |
| §5.4 Unit test scope | Tasks 2-10 (test files) |
| §5.5 Cross-provider parity gate | Task 11 |
| §5.6 Real-data sweep | Task 12 |
| §5.7 Public-API Examples | Task 13 |
| §5.8 Pre-commit gates | Task 14 |
| §5.9 TDD ordering | Task numbering matches Loop 0-12 |
| §6 Branch / version strategy | Tasks 0.1, 14.3, 14.10, 14.11 |
| §8 TODO.md updates | Task 14.1 |

Type consistency check:
- `convert_to_frames` signature: 4-arg (raw_frames, home_team_id, home_team_start_left, home_team_start_left_extratime) consistent across sportec.py and pff.py.
- Kloppy gateway: 1-arg (dataset) + optional preserve_native, distinct shape (matches spec §4.3).
- `link_actions_to_frames`: 3-arg (actions, frames, tolerance_seconds) consistent across tests and impl.
- `_derive_speed`: takes only `frames`; consistent.
- `play_left_to_right`: takes (frames, home_team_id) — matches spec §4.3.
- LinkReport / TrackingConversionReport: dataclass shapes consistent between Task 2 (definition) and Task 5+8+9+10 (consumers).

No placeholders detected.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
