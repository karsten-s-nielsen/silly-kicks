# xT-GK v2 construct-validity with the real ρ (SP5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire a runnable real-ρ path into `scripts/validate_xtgk_v2.py` — parameterize `construct_validity_scores` with an injected retention model, restrict evaluation to the GK-distribution domain, read v1 from the stored `fct_action_context.xt_gk` column, add a runnable `main()`, and run it locally (GS + SC) to certify the production ρ's metric lift over baselines.

**Architecture:** Scripts + tests only (+ two additive `load_xtgk_cohort` columns), no wheel/library behaviour change. `construct_validity_scores` gains an injected `retention` (default `_ConstRho` stub → CI unchanged), restricts the metric eval to `is_gk_distribution` test rows (ρ is GK-domain-trained; `compute_xt_gk_v2` doesn't self-gate), and reports v2's AUC lift over `raw_completion` / `destination_xt` / `v1_stored` (read from `c.xt_gk`, no frames).

**Tech Stack:** Python, pandas/numpy, sklearn (AUC), pytest, Databricks SQL (read-only). No new deps.

**Spec:** `docs/superpowers/specs/2026-07-11-xtgk-v2-construct-validity-real-rho-design.md`

**Commit policy:** ONE squashed commit per branch, only on explicit user approval (Task 6). No per-task commits. Branch `pr-s112-xtgk-v2-construct-validity` already exists (holds the spec + this plan, uncommitted).

---

## File Structure

- **Modify** `scripts/_loader_databricks.py` — add `c.is_gk_distribution` + `c.xt_gk` to `_XTGK_ACTIONS_SQL` + coerce.
- **Modify** `scripts/validate_xtgk_v2.py` — parameterize `construct_validity_scores` (inject retention, GK-domain restriction, `v1_stored` baseline, drop `_v1_composite`/`frames` plumbing); add runnable `main()` + report writer.
- **Modify** `tests/xtgk/test_validate_v2_smoke.py` — update the stub-path assertions (`v1_composite` → `v1_stored`, add `lift`).
- **Create** `tests/xtgk/test_validate_v2_real_rho.py` — the non-vacuous real-features branch + the domain-fallback test.
- **Create (owner-run output)** `docs/research/xtgk_v2_construct_validity/{gradientsports,skillcorner}.md` — the real result reports.
- **Modify** docs: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-036-*.md`, `uv.lock`.

---

## Task 1: Loader — add `is_gk_distribution` + `xt_gk` to `load_xtgk_cohort`

**Files:**
- Modify: `scripts/_loader_databricks.py`
- Test: `tests/xtgk/test_retention_loader_domain.py` (extend — it already imports the module)

- [ ] **Step 1: Write the SQL-shape guard test**

Append to `tests/xtgk/test_retention_loader_domain.py`:

```python
def test_xtgk_cohort_sql_selects_is_gk_distribution_and_xt_gk():
    # PR-S112: the construct-validity harness reads the GK-distribution domain + the stored v1 (c.xt_gk).
    import scripts._loader_databricks as L

    assert "c.is_gk_distribution" in L._XTGK_ACTIONS_SQL
    assert "c.xt_gk" in L._XTGK_ACTIONS_SQL
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py::test_xtgk_cohort_sql_selects_is_gk_distribution_and_xt_gk -q`
Expected: FAIL — neither column is in `_XTGK_ACTIONS_SQL` yet.

- [ ] **Step 3: Add the columns + coercion**

In `scripts/_loader_databricks.py`, in `_XTGK_ACTIONS_SQL`, extend the `fct_action_context c` selection. Change:
```python
  c.pressure_on_actor__bekkers_pi, c.pressure_on_actor__andrienko_oval,
```
to:
```python
  c.pressure_on_actor__bekkers_pi, c.pressure_on_actor__andrienko_oval,
  c.is_gk_distribution, c.xt_gk,
```

In `load_xtgk_cohort`, add `"xt_gk"` to the `numeric` coercion tuple (float, NaN preserved where v1 is null), and add a boolean coercion for `is_gk_distribution` after the `type_id`/`result_id` loop:
```python
    actions["is_gk_distribution"] = actions["is_gk_distribution"].astype("boolean").fillna(False).astype(bool)
```
Update the loader docstring to note the two additive columns (GK-distribution domain marker + stored v1 composite; the deep-zone gate ignores them).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -q`
Expected: PASS (the new guard + the existing domain/loader tests).

---

## Task 2: Parameterize `construct_validity_scores` (inject ρ, GK-domain, v1_stored)

**Files:**
- Modify: `scripts/validate_xtgk_v2.py`
- Test: `tests/xtgk/test_validate_v2_real_rho.py` (new), `tests/xtgk/test_validate_v2_smoke.py` (update)

- [ ] **Step 1: Write the new tests**

Create `tests/xtgk/test_validate_v2_real_rho.py`:

```python
"""PR-S112: construct_validity_scores real-ρ path -- GK-domain restriction + injected retention + v1_stored."""

import numpy as np
import pandas as pd

from scripts.validate_xtgk_v2 import construct_validity_scores


class _FakeRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.6)


def _cohort(*, with_domain: bool, with_v1: bool):
    # possession-parity split needs both even+odd possession_ids; a mix of GK-dist True/False rows.
    from tests.xtgk.conftest import mixed_shot_and_shotless_cohort

    a = mixed_shot_and_shotless_cohort().reset_index(drop=True)
    test_rows = a.index[(a["possession_id"] % 2 == 1)]  # the eval (odd-possession) split
    assert len(test_rows) >= 2, "fixture must have >=2 test-split rows"
    if with_domain:
        a["is_gk_distribution"] = (np.arange(len(a)) % 3 != 0)
        # GUARANTEE both a True and a False row IN THE TEST split -> non-vacuous 0 < n_test_gk < n_test
        a.loc[test_rows[0], "is_gk_distribution"] = True
        a.loc[test_rows[1], "is_gk_distribution"] = False
    if with_v1:
        a["xt_gk"] = np.linspace(-0.02, 0.05, len(a))
        if with_domain:
            # NULL xt_gk on an odd-possession + is_gk_distribution=True row (test_rows[0]) so the null lands
            # IN test_gk and actually drops from the v1 denominator (the coverage path). NOT index[:2] (train).
            a.loc[test_rows[0], "xt_gk"] = np.nan
    return a


def test_real_rho_gk_domain_restriction_and_v1_stored():
    a = _cohort(with_domain=True, with_v1=True)
    s = construct_validity_scores(a, xg_column="xg", pressure_column="pressure", retention=_FakeRho())
    n_test = int((a["possession_id"] % 2 == 1).sum())
    assert 0 < s["n_test_gk"] < n_test          # restriction applied (non-vacuous)
    assert np.isfinite(s["xt_gk_v2"]["auc"])     # real features built + scored, no crash
    assert s["v1_stored"]["n"] < s["n_test_gk"]  # STRICT: the null xt_gk row is dropped from v1's denominator
    assert "v2_on_v1_rows" in s                  # apples-to-apples v2-vs-v1 number reported
    assert "v1_composite" not in s               # old key gone
    assert "lift" in s                           # lift over max(raw, dest, v1_stored)


def test_domain_fallback_when_column_absent():
    a = _cohort(with_domain=False, with_v1=False)
    s = construct_validity_scores(a, xg_column="xg", pressure_column="pressure")  # stub path, no domain col
    assert s["n_test_gk"] == int((a["possession_id"] % 2 == 1).sum())  # falls back to all-test
    for k in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "lift"):
        assert k in s
```

Update `tests/xtgk/test_validate_v2_smoke.py` — replace the `v1_composite` key with `v1_stored` and add `lift`:
```python
    for key in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "lift"):
        assert key in scores, f"missing baseline {key}"
    assert np.isfinite(scores["xt_gk_v2"]["auc"])
    assert np.isfinite(scores["destination_xt"]["auc"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/xtgk/test_validate_v2_real_rho.py tests/xtgk/test_validate_v2_smoke.py -q`
Expected: FAIL — `v1_stored`/`lift`/`n_test_gk` keys don't exist; `construct_validity_scores` still has the old `frames` signature + `v1_composite`.

- [ ] **Step 3: Rewrite `construct_validity_scores` + drop `_v1_composite`**

In `scripts/validate_xtgk_v2.py`, **delete** the `_v1_composite` helper (lines 67-74) and replace `construct_validity_scores` (lines 77-110) with:

```python
def construct_validity_scores(
    actions: pd.DataFrame, *, xg_column: str, pressure_column: str, retention=None
) -> dict:
    a = actions.reset_index(drop=True)
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    train_mask = (a["possession_id"] % 2 == 0).to_numpy()  # out-of-sample by possession parity
    train, test = a[train_mask].copy(), a[~train_mask].copy()
    pl = PressureLevels().fit(train[pressure_column])
    v = MarkovPossessionValue().fit(train, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    tc = MirroredTurnoverCost(v)

    # target on the FULL test (the forward scan needs the intact possession sequence), then restrict to the
    # GK-distribution domain (where rho / v2 are defined; compute_xt_gk_v2 does not self-gate).
    y_full = _possession_reaches_shot(test)
    if "is_gk_distribution" in test.columns:
        gk = test["is_gk_distribution"].fillna(False).to_numpy(dtype=bool)
    else:
        gk = np.ones(len(test), dtype=bool)  # synthetic CI fixtures: no domain column -> all-test
    test_gk = test[gk].reset_index(drop=True)
    y = y_full[gk]

    if retention is None:
        retention = _ConstRho()
        feats = pd.DataFrame(index=test_gk.index)  # _ConstRho ignores content
    else:
        from silly_kicks.xtgk._retention_features import extract_retention_features

        feats = extract_retention_features(test_gk, pressure_column=pressure_column)

    v2 = compute_xt_gk_v2(
        test_gk, possession_value=v, retention=retention, turnover_cost=tc,
        pressure_column=pressure_column, pressure_levels=pl, retention_features=feats,
    )
    raw_completion = (test_gk["result_id"] == spadlconfig.result_id["success"]).astype(int).to_numpy()
    dest = _destination_only_v(test_gk, v, pl, pressure_column)
    v1 = pd.to_numeric(test_gk["xt_gk"], errors="coerce").to_numpy() if "xt_gk" in test_gk.columns else np.full(len(test_gk), np.nan)

    v2_arr = v2["xt_gk_v2"].to_numpy()
    v2_auc, raw_auc, dest_auc = _auc(y, v2_arr), _auc(y, raw_completion), _auc(y, dest)
    v1_ok = np.isfinite(v1)
    v1_auc = _auc(y, v1)  # _auc filters non-finite scores (v1 nulls dropped from its denominator)
    # apples-to-apples v2-vs-v1: v2 restricted to the v1-covered rows (v1 covers ~89% GS / 100% SC), so the
    # "does v2 beat v1" number is on a matched denominator (the lift below uses full-test-gk baselines).
    v2_on_v1_rows = _auc(y[v1_ok], v2_arr[v1_ok])
    baselines = [b for b in (raw_auc, dest_auc, v1_auc) if np.isfinite(b)]
    lift = float(v2_auc - max(baselines)) if (np.isfinite(v2_auc) and baselines) else float("nan")

    return {
        "xt_gk_v2": {"auc": v2_auc},
        "raw_completion": {"auc": raw_auc},
        "destination_xt": {"auc": dest_auc},
        "v1_stored": {"auc": v1_auc, "n": int(v1_ok.sum())},
        "v2_on_v1_rows": {"auc": v2_on_v1_rows},
        "lift": lift,
        "n_test_gk": int(len(test_gk)),
        "_note": (
            "GK-distribution-domain eval (is_gk_distribution); V out-of-sample (possession-parity split), "
            "rho IN-SAMPLE (the production model serves its training population); V is ~expected first-shot "
            "xG so absolute AUC vs possession-reaches-shot is partly circular -- read LIFT over max(baselines). "
            "v1_stored read from fct_action_context.xt_gk (no frames)."
        ),
    }
```

Remove the now-unused `frames`-related imports if any (`add_xt_gk` was imported inside `_v1_composite` only — confirm none remain: `grep -n "add_xt_gk" scripts/validate_xtgk_v2.py`).

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/xtgk/test_validate_v2_real_rho.py tests/xtgk/test_validate_v2_smoke.py -q`
Expected: PASS. The real-features branch runs `extract_retention_features` on the GK subset; the stub/smoke path is unchanged behaviourally (bar the renamed key).

---

## Task 3: Runnable `main()` + report writer

**Files:**
- Modify: `scripts/validate_xtgk_v2.py`

- [ ] **Step 1: Add `main()` + `_write_report`**

Append to `scripts/validate_xtgk_v2.py`:

```python
def _write_report(provider: str, variant: str, scores: dict) -> str:
    from pathlib import Path

    lines = [
        f"# xT-GK v2 construct-validity — {provider}\n",
        f"- ρ variant: `{variant}` · GK-distribution test rows: **{scores['n_test_gk']}**\n",
        "\n| metric | AUC | n |\n|---|---|---|\n",
        f"| **xt_gk_v2** | {scores['xt_gk_v2']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| raw_completion | {scores['raw_completion']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| destination_xt | {scores['destination_xt']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| v1_stored (c.xt_gk) | {scores['v1_stored']['auc']:.4f} | {scores['v1_stored']['n']} |\n",
        f"| xt_gk_v2 (on v1-covered rows) | {scores['v2_on_v1_rows']['auc']:.4f} | {scores['v1_stored']['n']} |\n",
        f"\n**LIFT** (v2 − max baseline, full GK-test): **{scores['lift']:+.4f}**\n",
        f"\n**v2 vs v1 (matched rows):** v2 {scores['v2_on_v1_rows']['auc']:.4f} vs v1 {scores['v1_stored']['auc']:.4f} "
        f"(Δ {scores['v2_on_v1_rows']['auc'] - scores['v1_stored']['auc']:+.4f})\n",
        f"\n> {scores['_note']}\n",
    ]
    out = Path(__file__).resolve().parent.parent / "docs" / "research" / "xtgk_v2_construct_validity" / f"{provider}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    return str(out)


def main() -> int:
    import argparse

    # Parse args BEFORE the heavy/connection imports so `--help` short-circuits connection-free.
    ap = argparse.ArgumentParser(description="xT-GK v2 construct-validity with the real bundled rho (SP5).")
    ap.add_argument("--provider", default="gradientsports")
    a = ap.parse_args()

    from _loader_databricks import load_xtgk_cohort  # type: ignore[import-not-found]
    from validate_xtgk_possession_value import _FRAME_PRESENT_COLUMN, _PRESSURE_COLUMN, _XG_COLUMN, prepare_cohort  # type: ignore[import-not-found]

    from silly_kicks.xtgk._retention import GkRetentionModel, variant_key_for_provider

    raw, _ = load_xtgk_cohort(a.provider)
    actions = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
    variant = variant_key_for_provider(a.provider)
    rho = GkRetentionModel.from_variant(variant)
    scores = construct_validity_scores(actions, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, retention=rho)
    print(f"provider={a.provider} variant={variant} n_test_gk={scores['n_test_gk']}")
    for k in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "v2_on_v1_rows"):
        print(f"  {k}: AUC={scores[k]['auc']:.4f}" + (f" (n={scores[k]['n']})" if "n" in scores[k] else ""))
    print(f"  LIFT (v2 - max baseline) = {scores['lift']:+.4f}")
    print("wrote", _write_report(a.provider, variant, scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports + is argparse-runnable (no Databricks)**

Run: `python -c "import scripts.validate_xtgk_v2 as m; assert hasattr(m, 'main') and hasattr(m, '_write_report')"`
Run: `python scripts/validate_xtgk_v2.py --help 2>&1 | head -3`
Expected: import OK; `--help` prints the argparse usage connection-free (`parse_args` runs before the heavy `_loader_databricks`/`validate_xtgk_possession_value` imports, so `--help` exits before any Databricks import).

---

## Task 4: Owner-run — GS + SC, commit the reports

**Files:**
- Create (generated): `docs/research/xtgk_v2_construct_validity/{gradientsports,skillcorner}.md`

- [ ] **Step 1: Run GS**

Run (local, live Databricks): `python scripts/validate_xtgk_v2.py --provider gradientsports`
Expected: prints `n_test_gk` (~1500-1900), the AUC table, the LIFT, and `wrote docs/research/.../gradientsports.md`. **Record the lift honestly** — a null/negative lift is a reportable finding (§9.1), not a bug. v1_stored coverage ~89% (n < n_test_gk).

- [ ] **Step 2: Run SC**

Run: `python scripts/validate_xtgk_v2.py --provider skillcorner`
Expected: same shape; SC v1 coverage 100% (n == n_test_gk). Uses the bundled `skillcorner` ρ variant.

- [ ] **Step 3: Sanity-read both reports**

Run: `cat docs/research/xtgk_v2_construct_validity/gradientsports.md docs/research/xtgk_v2_construct_validity/skillcorner.md`
Confirm: AUC table populated, v1_stored `n` reflects coverage (GS < n_test_gk, SC == n_test_gk), lift present, note carries the V-out-of-sample / ρ-in-sample caveat. These reports are committed as the deliverable.

---

## Task 5: Version bump + docs + ADR + final-review

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-036-*.md`

- [ ] **Step 1: Patch bump 4.44.0 → 4.44.1 (lockstep)**

Edit `pyproject.toml` (`version = "4.44.1"`) + `silly_kicks/__init__.py` (`__version__ = "4.44.1"`) + `uv.lock` (silly-kicks block). Verify: `grep -rn "4.44.0" pyproject.toml silly_kicks/__init__.py` → none. Then `uv lock --check` → exit 0.

- [ ] **Step 2: CHANGELOG (with the real lift numbers)**

Prepend a `## [4.44.1] — 2026-07-11` entry: SP5 real-ρ construct-validity wired + run; the per-provider result (GS/SC AUC table + lift over max(raw, dest, v1_stored)); v1 read from the stored `c.xt_gk` (frames-free), NOT recomputed; GK-distribution-domain eval; two additive `load_xtgk_cohort` columns; scripts+tests-only (no wheel behaviour change).

- [ ] **Step 3: TODO — mark the SP5 follow-up done**

In `TODO.md`: remove/close the "wire + run the real-ρ construct-validity" deferred item (done); update the "Current release" line to 4.44.1/PR-S112. Keep the lakehouse nullable-column + xt_gk_v2-re-materialize relays.

- [ ] **Step 4: ADR-036 amendment**

Append an amendment (`ls docs/superpowers/adrs/ADR-036-*.md`): SP5 real-ρ construct-validity run (per-provider lift, the result); the injected-retention seam; the GK-domain restriction (ρ is GK-trained, `compute_xt_gk_v2` doesn't self-gate); v1 kept as a **stored-column** baseline (not frames-recompute) + why (integrity — it's the strongest baseline); V-out-of-sample/ρ-in-sample caveat.

- [ ] **Step 5: /final-review + full gates**

Invoke `final-review` (C4 count stays 28 — a script harness + report + two additive loader columns are not architectural; verify `architecture.dsl` unchanged).
Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Run: `ruff check . && ruff format --check .` · `pyright`
Expected: all green. Fix any issue before Task 6.

---

## Task 6: Single commit (explicit approval) + push + PR

- [ ] **Step 1: Present `git status` + `git diff --stat`; REQUEST APPROVAL** (repo policy). Include the headline result (per-provider lift) in the summary.
- [ ] **Step 2 (after approval): Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(xtgk): SP5 real-rho construct-validity for xT-GK v2 -- silly-kicks 4.44.1 (ADR-036, PR-S112)

Wire a runnable real-rho path into validate_xtgk_v2.py: inject the bundled GkRetentionModel, restrict
evaluation to the GK-distribution domain (rho is GK-trained; compute_xt_gk_v2 does not self-gate), read
v1 from the stored fct_action_context.xt_gk column (frames-free), and report v2's AUC lift over
raw_completion / destination_xt / v1_stored. Run locally for GS + SC; reports committed. Scripts+tests
only (+ two additive loader columns); no wheel behaviour change.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011yincSJTWHYjAXZAQpBQRy
EOF
)"
```

- [ ] **Step 3: Push + PR**

```bash
git push -u origin pr-s112-xtgk-v2-construct-validity
gh pr create --title "test(xtgk): SP5 real-rho construct-validity for xT-GK v2 -- silly-kicks 4.44.1 (ADR-036, PR-S112)" --body "<summary + spec/plan links + the per-provider lift result>"
```

Watch CI only if asked; otherwise report the PR URL and stop.

---

## Self-Review Notes

- **Spec coverage:** loader columns (Task 1), parameterize + GK-domain + v1_stored (Task 2), main + report (Task 3), owner-run GS+SC (Task 4), version/docs/ADR (Task 5). All mapped.
- **Non-vacuous fixture** (Task 2 Step 1) — the fixture forces both a True and a False `is_gk_distribution` row INTO THE TEST split (asserts `0 < n_test_gk < n_test`), and nulls `xt_gk` on an **odd-possession + True** row (which lands in `test_gk`) so the v1-coverage drop is actually exercised (**strict** `v1_stored["n"] < n_test_gk`). Not `index[:2]` (those are even-possession = train, so the null would never reach v1's denominator — the caught vacuity).
- **Domain fallback** (Task 2) — the absent-column branch is tested; the CI smoke (no domain column) still works.
- **§9.2/9.3 gates** — `prepare_cohort` passes both new columns through (verified: it `.copy()`s + row-filters, doesn't drop columns); `extract_retention_features` reads `pressure_column="pressure_on_actor__bekkers_pi"` (verified: reads `actions[pressure_column]`).
- **The lift is not pre-decidable** — Task 4 records it honestly; a null/negative lift is a finding, not a bug.
- **No per-task commits** — single commit in Task 6 behind approval.
