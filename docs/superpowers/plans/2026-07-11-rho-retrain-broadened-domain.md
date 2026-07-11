# ρ retrain on broadened domain + loader collapse + resolver hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Retrain the ρ retention model on the now-live broadened `is_gk_distribution` domain (re-bundle GS `default`; re-attempt SkillCorner, bundle iff it clears the calibration gate), collapse the transitional loader probe, harden the `acting_gk_from_frames` team-join dtype seam, and CI-enforce the calibration bar on bundled weights.

**Architecture:** Three coupled parts + a CI guard. Part C (resolver hardening) and Part B (loader collapse) are pure-code/CI. Part A (retrain) is owner-run local training against the live Databricks gold marts, with a conditional SkillCorner branch. No public API signature change.

**Tech Stack:** Python, pandas/numpy, scikit-learn (logistic), pytest, Databricks SQL (read-only). No new deps.

**Spec:** `docs/superpowers/specs/2026-07-11-rho-retrain-broadened-domain-design.md`

**Commit policy:** ONE squashed commit per branch, only on explicit user approval (Task 5). No per-task commits. Branch `pr-s111-rho-retrain-broadened-domain` already exists (holds the spec + this plan, uncommitted).

**Order rationale:** pure-code parts first (C, B — CI-verifiable), then the owner-run retrain (A), then docs. The two easy-to-write-shallow guards flagged by review — the **F1 CI calibration test** (Task 3) and the **non-vacuous Part C fixture** (Task 1) — are called out as their own steps.

---

## File Structure

- **Modify** `silly_kicks/tracking/_gk_resolve.py` — Part C one-line dtype hardening + comment.
- **Create** `tests/tracking/test_gk_resolve_dtype_hardening.py` — non-vacuous byte-identity + mismatched-dtype + Int64-NA tests.
- **Modify** `scripts/_loader_databricks.py` — Part B: remove probe helpers, unconditional SQL, fail-loud docstring.
- **Modify** `tests/xtgk/test_retention_loader_domain.py` — drop the removed-helper unit tests; add the unconditional-SQL guard.
- **Create** `tests/xtgk/test_retention_bundle_calibration.py` — Part A CI gate-enforcement (F1).
- **Modify** `scripts/train_gk_retention.py` — MODEL_CARD template `andrienko_oval`→`bekkers_pi` fix.
- **Modify** `silly_kicks/xtgk/_retention_weights/default/{model.json,metrics.json,MODEL_CARD.md,SHA256SUMS}` — re-bundled (owner-run).
- **Conditionally create** `silly_kicks/xtgk/_retention_weights/skillcorner/*` + **modify** `silly_kicks/xtgk/_retention.py` (`_PROVIDER_VARIANT`) — only if SC clears the gate.
- **Modify** `tests/xtgk/test_retention.py` — add SC `from_variant` + `variant_key_for_provider` assertions (both branches).
- **Modify** docs: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `docs/superpowers/adrs/ADR-036-*.md`, `uv.lock`.

---

## Task 1: Part C — `acting_gk_from_frames` team-join dtype hardening

**Files:**
- Modify: `silly_kicks/tracking/_gk_resolve.py:165`
- Test: `tests/tracking/test_gk_resolve_dtype_hardening.py`

- [ ] **Step 1: Write the tests (byte-identity NON-VACUOUS + mismatched + Int64-NA)**

Create `tests/tracking/test_gk_resolve_dtype_hardening.py`:

```python
"""Part C: acting_gk_from_frames team-join dtype hardening — byte-identity + mismatched + NA."""

import numpy as np
import pandas as pd

from silly_kicks.tracking import acting_gk_from_frames, defending_gk_from_frames

_PASS = 0


def _frow(pid, team, gk, t, *, x=50.0):
    return dict(
        game_id=1, period_id=1, frame_id=round(t * 25), time_seconds=t, frame_rate=25.0,
        player_id=pid, team_id=team, is_ball=False, is_goalkeeper=gk, x=float(x), y=34.0,
        z=0.0, speed=1.0, vx=0.0, vy=0.0, speed_source="native", ball_state="alive",
        team_attacking_direction="ltr", confidence=None, visibility=None,
        source_provider="gradientsports", is_goalkeeper_source="native",
    )


def _frames(rows, *, team_dtype="Int64", player_dtype="Int64"):
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype(player_dtype)
    f["team_id"] = f["team_id"].astype(team_dtype)
    return f


# GK team 5 = player 1, GK team 6 = player 2, both detected at t~10 and t~20.
def _two_keeper_frames(**kw):
    rows = []
    for t in (9.9, 10.0, 19.9, 20.0):
        rows += [_frow(1, 5, True, t), _frow(2, 6, True, t, x=100.0)]
    return _frames(rows, **kw)


def _action(team_id, t, *, team_dtype=None):
    df = pd.DataFrame([dict(game_id=1, action_id=0, period_id=1, time_seconds=t,
                            team_id=team_id, player_id=99, type_id=_PASS, result_id=1)])
    if team_dtype:
        df["team_id"] = df["team_id"].astype(team_dtype)
    return df


# --- (i) BYTE-IDENTITY, NON-VACUOUS: a normal matched row AND a float-NaN team row on the
#         defending (opposing) path — the ~match_team/NaN branch that IS the hardening. Golden
#         values pin CURRENT behavior; Step 2 (run before the fix) confirms them.
def test_byte_identity_matched_and_nan_team_defending():
    f = _two_keeper_frames()
    # matched-dtype action (team 5, Int64-comparable python int) -> defending GK = opposing = player 2
    a_matched = _action(5, 10.0)
    # float-NaN team action -> exercises ~match_team on an all-NaN compare (the non-vacuous case)
    a_nan = _action(np.nan, 20.0, team_dtype="float64")

    d_matched = defending_gk_from_frames(a_matched, f)
    d_nan = defending_gk_from_frames(a_nan, f)
    ac_matched = acting_gk_from_frames(a_matched, f)  # team 5 -> acting GK = player 1

    # GOLDEN (confirm in Step 2 against current code; must be byte-identical after the fix):
    assert d_matched.tolist() == [2]          # opposing keeper
    assert ac_matched.tolist() == [1]         # acting keeper
    # NaN-team defending: current code picks lowest-player_id among all "opposing" (all match ~NaN);
    # pin whatever Step 2 shows — it MUST be unchanged post-fix.
    assert d_nan.tolist() == [1]              # <-- CONFIRM in Step 2; the byte-identity anchor


# --- (ii) MISMATCHED DTYPE now resolves (was NaN): string action team vs Int64 frame team.
def test_mismatched_dtype_now_resolves():
    f = _two_keeper_frames()  # Int64 team ids
    a = _action("5", 10.0, team_dtype="object")  # string action team_id
    ac = acting_gk_from_frames(a, f)
    assert ac.tolist() == [1]  # acting GK (team 5) resolved despite str-vs-Int64 (was NaN pre-fix)


# --- (iii) nullable-Int64 NA team: raw == raises on masking; fixed -> deterministic, no raise.
def test_int64_na_team_does_not_raise():
    f = _two_keeper_frames()
    a = _action(pd.NA, 10.0, team_dtype="Int64")
    ac = acting_gk_from_frames(a, f)  # must NOT raise
    assert ac.isna().all() or ac.notna().all()  # deterministic (no boolean-masking crash)
```

- [ ] **Step 2: Run to confirm the byte-identity golden + see (ii)/(iii) fail**

Run: `python -m pytest tests/tracking/test_gk_resolve_dtype_hardening.py -v`
Expected: `test_byte_identity_matched_and_nan_team_defending` PASSES against current code (this **confirms the golden** — if `d_nan` ≠ `[1]`, edit the golden to the actual current value, since byte-identity means "unchanged", not a specific value). `test_mismatched_dtype_now_resolves` FAILS (returns `[NaN]`). `test_int64_na_team_does_not_raise` FAILS (raises on boolean masking). If (i) itself errors on the Int64 path, that's the raw-`==` fragility — fine, it's covered by (iii).

- [ ] **Step 3: Apply the one-line hardening**

In `silly_kicks/tracking/_gk_resolve.py`, at line 165:

```python
    # Team predicate: acting team (==) vs opposing team (!=). dtype-safe (ADR-019) — ids_equal is
    # POSITIONAL and returns a non-nullable np.bool_ array; .to_numpy() masks gk_in_frame positionally
    # (gk_in_frame is a fresh inner-merge -> RangeIndex, so this coincides with index-aligned == on
    # matched dtypes). Do NOT reindex gk_in_frame above without revisiting this. NaN action team_id ->
    # comparison False -> dropped (acting) / included (~, opposing) -- byte-identical to the raw path.
    match_team = ids_equal(gk_in_frame["gk_team_id"], gk_in_frame["team_id"]).to_numpy()
    picked = gk_in_frame[match_team if same_team else ~match_team]
```

(`ids_equal` is already imported. Line 166 `picked = ...` is unchanged in logic — the `~` now operates on a plain bool array.)

- [ ] **Step 4: Run — all Part C tests pass + existing gates stay green**

Run: `python -m pytest tests/tracking/test_gk_resolve_dtype_hardening.py tests/tracking/test_acting_gk_from_frames.py tests/tracking/test_gk_resolve.py tests/invariants/test_invariant_gk_resolve.py tests/spadl/test_gk_fallback_integration.py -q`
Expected: PASS. Byte-identity test STILL passes (proves `defending_gk_from_frames` untouched); (ii)/(iii) now pass; the four existing gates unchanged. If any existing gate moves → the fix is not byte-identical, STOP and diagnose.

---

## Task 2: Part B — collapse the transitional loader probe

**Files:**
- Modify: `scripts/_loader_databricks.py`
- Test: `tests/xtgk/test_retention_loader_domain.py`

- [ ] **Step 1: Write the unconditional-SQL guard, drop the removed-helper tests**

In `tests/xtgk/test_retention_loader_domain.py`, **delete** `test_should_select_is_gk_distribution_present_absent` and `test_build_retention_sql_conditionally_includes_column` (the helpers are being removed), and add:

```python
def test_retention_sql_is_unconditional_and_probe_helpers_gone():
    import scripts._loader_databricks as L

    assert "c.is_gk_distribution" in L._RETENTION_SQL           # unconditional select
    assert "{is_gk_distribution_select}" not in L._RETENTION_SQL  # no template hole
    assert not hasattr(L, "should_select_is_gk_distribution")
    assert not hasattr(L, "_build_retention_sql")
    assert not hasattr(L, "_IS_GK_DISTRIBUTION_PROBE")
```

(The `test_domain_present_nonnull/null/absent` + `test_dropped_column_is_not_a_feature` tests STAY — the trainer domain logic is unchanged.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py::test_retention_sql_is_unconditional_and_probe_helpers_gone -q`
Expected: FAIL — `_RETENTION_SQL` doesn't exist yet (it's `_RETENTION_SQL_TEMPLATE`), helpers still present.

- [ ] **Step 3: Collapse the loader**

In `scripts/_loader_databricks.py`, replace `_RETENTION_SQL_TEMPLATE` + `_IS_GK_DISTRIBUTION_PROBE` + `should_select_is_gk_distribution` + `_build_retention_sql` with a single unconditional constant:

```python
# --- xT-GK v2 retention (rho) cohort loader (ADR-036 §Part 3, marts-native) -----------------------
# Tracking-frames deprecated: features come from the gold action marts. fct_action_values supplies
# the base SPADL (geometry/type/result/possession); fct_action_context supplies pressure AND the
# GK-distribution domain flag (is_gk_distribution = tracking.gk_distribution_mask, resolve_gk="robust").
# Keyed on (match_key, action_id). is_gk_distribution is a HARD dependency (materialized by lakehouse
# F1 as of silly-kicks 4.44.0); a missing column surfaces as a column-named Databricks error.
# pressure = bekkers_pi, pinned in PR-S109.
_RETENTION_SQL = """
WITH v AS (SELECT * FROM soccer_analytics.dev_gold.fct_action_values WHERE data_source = %(ds)s)
SELECT
  v.match_key AS game_id, v.period AS period_id, v.action_id, v.time_seconds,
  v.team_id, v.player_id, v.start_x, v.start_y, v.end_x, v.end_y,
  v.action_type, v.action_result, v.possession_id, v.data_source,
  c.pressure_on_actor__bekkers_pi AS pressure,
  c.is_gk_distribution
FROM v
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c
  ON c.match_key = v.match_key AND c.action_id = v.action_id
ORDER BY v.match_key, v.period, v.time_seconds, v.action_id
"""
```

And simplify `load_retention_cohort`'s try-block + docstring:

```python
def load_retention_cohort(data_source: str) -> pd.DataFrame:
    """Full attack-LTR action stream for the rho retention trainer (marts-native; NO tracking frames).

    Requires ``fct_action_context.is_gk_distribution`` (lakehouse F1; silly-kicks >= 4.44.0) — a HARD
    dependency, unconditionally selected. Maps gold ``action_type``/``action_result`` to SPADL ids,
    carries ``pressure`` (bekkers_pi) + the ``is_gk_distribution`` GK-distribution domain flag (NULLs
    coalesced to False). Sorted by (game_id, period_id, time_seconds, action_id).
    """
    import silly_kicks.spadl.config as spadlconfig

    if data_source not in _ALLOWED_PROVIDERS:
        raise ValueError(f"data_source {data_source!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    conn = _connect()
    try:
        df = _query_param(conn.cursor(), _RETENTION_SQL, {"ds": data_source})
    finally:
        conn.close()
    df["type_id"] = df["action_type"].map(spadlconfig.actiontype_id).fillna(-1).astype("int64")
    df["result_id"] = df["action_result"].map(spadlconfig.result_id).fillna(-1).astype("int64")
    for col in ("start_x", "start_y", "end_x", "end_y", "pressure", "time_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_gk_distribution"] = df["is_gk_distribution"].fillna(False).astype(bool)
    df = df[df["time_seconds"].notna()].copy()
    return df.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run the loader-domain tests**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -q`
Expected: PASS (new guard green; domain present/null/absent + dropped-column-guard stay green).

---

## Task 3: Part A — CI gate-enforcement (F1) + MODEL_CARD fix + owner-run retrain

**Files:**
- Create: `tests/xtgk/test_retention_bundle_calibration.py`
- Modify: `scripts/train_gk_retention.py` (MODEL_CARD template)
- Modify (owner-run): `silly_kicks/xtgk/_retention_weights/default/*`; conditionally `skillcorner/*` + `_retention.py`
- Modify: `tests/xtgk/test_retention.py`

- [ ] **Step 1: Write the CI calibration-enforcement test (F1)**

Create `tests/xtgk/test_retention_bundle_calibration.py`:

```python
"""F1: every BUNDLED rho variant's recorded metrics must clear the canonical calibration gate."""

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
```

- [ ] **Step 2: Run against the CURRENT bundle (guards the existing default)**

Run: `python -m pytest tests/xtgk/test_retention_bundle_calibration.py -q`
Expected: PASS — the current `default` (PR-S109) already cleared the gate, so its `metrics.json` satisfies the bar. (If it FAILS, the current bundle is out of spec — STOP and report before retraining.)

- [ ] **Step 3: Fix the MODEL_CARD pressure doc-bug**

In `scripts/train_gk_retention.py`, in the `MODEL_CARD.md` template, change `pressure_on_actor__andrienko_oval` → `pressure_on_actor__bekkers_pi` (the loader feeds bekkers_pi as `pressure`; the re-bundle regenerates the card). Verify: `grep -n "andrienko_oval" scripts/train_gk_retention.py` → no matches after the edit.

- [ ] **Step 4: Owner-run retrain — GS `default` (broadened domain)**

Run (local, live Databricks):
`python -m scripts.train_gk_retention --provider gradientsports --variant default`
Expected: prints `loaded ~88958 actions (64 matches)`, then `CORPUS rows≈1500–2500` — the domain is 3,873 GK-distribution actions, but the geometry + truncated-window filters keep only a fraction (the old goal-kicks-only default kept 396 of 1,001 ≈ 40%), so a count well below 3,873 is EXPECTED, not a bug. Then `OOF AUC=… ECE=… slope=… GATE=PASS|FAIL`, and on pass `bundled -> …/default`. **A `GATE=FAIL` (exit 3) is a legitimate, planned-for outcome, NOT a defect** — the current default sits at ECE 0.090 vs the 0.100 bar (0.010 headroom), and the broadened domain adds open-play GK passes with different retention dynamics, so a fail is plausible. **If `GATE=FAIL`: STOP** — do not bundle; the old default stays; report the metrics + escalate (do not lower the bar). On PASS, `default/{model.json,metrics.json,MODEL_CARD.md,SHA256SUMS}` are regenerated on the broadened domain.

- [ ] **Step 5: Owner-run retrain — SkillCorner variant (conditional)**

Run: `python -m scripts.train_gk_retention --provider skillcorner --variant skillcorner`
Expected: `CORPUS rows≈5487 (108 matches)`, gate PASS or FAIL.

- **If GATE=PASS** → keep `_retention_weights/skillcorner/`; set the variant registry in `silly_kicks/xtgk/_retention.py:28`:
  ```python
  _PROVIDER_VARIANT: dict[str, str] = {"skillcorner": "skillcorner"}
  ```
  and add to `tests/xtgk/test_retention.py`:
  ```python
  def test_skillcorner_variant_bundled_and_routed():
      from silly_kicks.xtgk._retention import GkRetentionModel, variant_key_for_provider
      assert variant_key_for_provider("skillcorner") == "skillcorner"
      m = GkRetentionModel.from_variant("skillcorner")
      import pandas as pd
      from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES
      p = m.predict_proba(pd.DataFrame({c: [0.0] for c in RETENTION_FEATURE_NAMES}))
      assert 0.0 <= float(p[0]) <= 1.0
  ```
- **If GATE=FAIL** → delete the `_retention_weights/skillcorner/` dir (do NOT bundle), leave `_PROVIDER_VARIANT={}`, and add:
  ```python
  def test_skillcorner_falls_back_to_default():
      from silly_kicks.xtgk._retention import _PROVIDER_VARIANT, variant_key_for_provider
      assert _PROVIDER_VARIANT == {}
      assert variant_key_for_provider("skillcorner") == "gs"  # -> default
  ```
  Record the failing SC metrics (AUC/ECE/slope) in the CHANGELOG + ADR (Task 4).

- [ ] **Step 6: Re-run the F1 enforcement + retention tests against the NEW bundle**

Run: `python -m pytest tests/xtgk/test_retention_bundle_calibration.py tests/xtgk/test_retention.py -q`
Expected: PASS. Every present variant clears the gate; `from_variant` loads; the SC branch test (whichever) is green. If F1 fails on a freshly-bundled variant, the trainer bundled something out of spec → STOP.

---

## Task 4: Version bump + docs + ADR + final-review

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `docs/superpowers/adrs/ADR-036-*.md`

- [ ] **Step 1: Bump 4.43.0 → 4.44.0 (lockstep)**

Edit `pyproject.toml` (`version = "4.44.0"`) + `silly_kicks/__init__.py` (`__version__ = "4.44.0"`) + `uv.lock` (the `name = "silly-kicks"` block → `version = "4.44.0"`). Verify: `grep -rn "4.43.0" pyproject.toml silly_kicks/__init__.py` → none. Then `uv lock --check` → exit 0.

- [ ] **Step 2: CHANGELOG (record the metrics manifest + SC decision)**

Prepend a `## [4.44.0] — 2026-07-11` entry: the ρ retrain on the broadened `is_gk_distribution` domain (GS `default`: n_rows, AUC/ECE/slope PASS); the **SkillCorner decision** (bundled with its metrics, OR not-bundled with its failing metrics + `_PROVIDER_VARIANT={}`); Part B loader collapse; Part C dtype hardening (byte-identity preserved); the F1 CI calibration guard; the MODEL_CARD pressure fix. Flag the `compute_xt_gk_v2` **serve-output change** (xT-GK v2 retrain trigger; opt-in, not a forced VAEP retrain) for the lakehouse.

- [ ] **Step 3: TODO — remove the two done follow-ups, keep the heads-ups**

In `TODO.md`: delete the "ρ retrain on the broadened domain + collapse the transitional loader probe" item (done) and update the "Current release" line to 4.44.0/PR-S111. Keep tracked: the lakehouse nullable-column relay + event-only half-b + the owner re-run of `validate_xtgk_v2.py` construct-validity (§2.1/m3). The `acting_gk_from_frames` dtype-fragility TODO is now **done** — remove it.

- [ ] **Step 4: CLAUDE.md**

Update the retention narrative: ρ `default` retrained on the broadened domain (n_rows), SC variant bundled-or-not (per gate), Part B unconditional loader read, Part C dtype hardening. Terse, matching surrounding density.

- [ ] **Step 5: ADR-036 amendment**

Append an amendment (`ls docs/superpowers/adrs/ADR-036-*.md`): the broadened-domain retrain (auditable metrics manifest, ADR-009), the SC bundle/no-bundle decision, the F1 CI enforcement, Part B collapse, Part C hardening + byte-identity guarantee, and the §2.1 gate-scope note (ρ moves metric construct-validity, NOT the deep-zone gate; Q4 locked).

- [ ] **Step 6: /final-review + full gates**

Invoke `final-review` (C4 count stays 28 — no new aggregator/container; a re-bundled model + hardened compare + loader cleanup are not architectural; regenerate/verify `architecture.{dsl,html}`).
Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Run: `ruff check . && ruff format --check .` · `pyright`
Expected: all green. Fix any issue before Task 5.

---

## Task 5: Single commit (explicit approval) + push + PR

- [ ] **Step 1: Present `git status` + `git diff --stat`; REQUEST APPROVAL** (repo policy: one commit, only on explicit approval). Note the SC branch outcome in the summary (bundled vs fell-back).
- [ ] **Step 2 (after approval): Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(xtgk): retrain rho on broadened is_gk_distribution domain + loader collapse + resolver hardening -- silly-kicks 4.44.0 (ADR-036, PR-S111)

Retrain the rho retention default (+ SkillCorner variant per the calibration gate) on the now-live
broadened GK-distribution domain; collapse the transitional loader probe to an unconditional read;
harden the acting_gk_from_frames team-join dtype seam (byte-identity preserved); CI-enforce the
calibration bar on bundled weights. No public API change; compute_xt_gk_v2 serve output changes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011yincSJTWHYjAXZAQpBQRy
EOF
)"
```

- [ ] **Step 3: Push + PR**

```bash
git push -u origin pr-s111-rho-retrain-broadened-domain
gh pr create --title "feat(xtgk): retrain rho on broadened is_gk_distribution domain -- silly-kicks 4.44.0 (ADR-036, PR-S111)" --body "<summary + spec/plan links + SC decision + metrics>"
```

Watch CI only if asked; otherwise report the PR URL and stop.

---

## Self-Review Notes

- **Spec coverage:** Part A retrain (Task 3), Part B collapse (Task 2), Part C hardening (Task 1), F1 CI gate (Task 3.1), m1 non-vacuous fixture (Task 1.1), m2 comment (Task 1.3), m4 fail-loud docstring (Task 2.3), MODEL_CARD fix (Task 3.3), version/docs/ADR (Task 4) — all mapped.
- **Conditional SC branch** is written both ways (Task 3.5) — the PR shape depends on the live gate result; neither path lowers the bar.
- **Byte-identity golden** (Task 1) is confirmed against current code in Step 2 before the fix (refactor-TDD), so the `d_nan` value is pinned, not guessed.
- **No per-task commits** — single commit in Task 5 behind approval.
- **Owner-run steps** (Task 3.4/3.5) run live against Databricks locally (pining #1 / DGX-not-needed for a logistic).
