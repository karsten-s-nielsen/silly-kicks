# xT-GK v2 FINISH release — faithful `V_opp` + full validation (single PR) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Finish xT-GK v2 in one release — replace the geometric-proxy `V_opp` (`V(mirror(zone))`) with Jeff §2.3's faithful observed-post-turnover estimate (with mandatory deep-cell sparsity handling), fix the loader's keeper-id column, and run the three program analyses (construct-validity, component decomposition, keeper discrimination) + a secondary faithfulness audit **on the faithful metric**, committing the final artifacts.

**Architecture:** The `V_opp` swap is injection-seam-only — `compute_xt_gk_v2(..., turnover_cost=...)` consumes `turnover_cost.value(zone,p)`, so **no `_metric.py` change**; only which `TurnoverCost` adapter is production. `EmpiricalTurnoverValue` becomes production (gaining bin-widening/support-gate); `MirroredTurnoverCost` demotes to cross-check.

**Tech Stack:** Python, pandas/numpy, sklearn, scipy, pytest, Databricks (read-only). No new deps.

**Handoff (spec):** `docs/superpowers/specs/2026-07-11-xtgk-v2-faithful-vopp-diagnostics-handoff.md` (W1–W6 + §3 honest-reporting guardrail + §4 acceptance checklist).

**Mechanics (handoff §0):** ONE branch (`pr-s112-xtgk-v2-construct-validity`, keep it) · ONE commit · ONE PR · **4.45.0 minor** (library behaviour change). Base is 4.44.0 (verified; no 4.44.1). The proxy GS/SC reports currently under `docs/research/` are **discarded** — only the faithful reports are committed. Commit only on explicit user approval (Task 8).

**§3 guardrail (non-negotiable):** W4/W5/W6 outcomes are NOT pre-decidable. Do NOT retune the deep-zone gate, pick κ, drop baselines, or filter keepers to force a pass. A null/negative result on the faithful metric is the clean input for the Jeff conversation — report it.

**A-priori parameters (§3 / R3) — FIXED before any W4/W5 number is looked at; never swept against outcomes:**
- **`V_opp` window scope = possession-bound** (`window_seconds=None`) — **scope-symmetric with V** (V is the possessing team's first-shot xG over its possession; V_opp is the opponent's first-shot xG over their won possession). Chosen a priori on the faithfulness/symmetry argument, NOT on outcomes (R1). The **10s-capped** variant is computed and **reported only as a sensitivity** — never the headline.
- **`min_support = 30`** — anchored to the pre-registered deep-zone gate `n_min=30` (clean, non-outcome rationale).
- **`coarsen`** — justified a priori by support geometry (a block must plausibly reach `min_support` where a native deep cell can't); default stated + rationale, not tuned to a result.
- **`κ = 1`** (default) for the headline W4/W5. The W6 `κ∈[1,2]` sweep is **reported as evidence for the Jeff κ/turnover-weighting question ONLY — never used to pick the headline.**

---

## File Structure
- **Modify** `silly_kicks/xtgk/_turnover.py` — W2: promote `EmpiricalTurnoverValue` to production (bin-widening + support-gate + per-zone n + resolution-level tracking); update the "not shipped" docstring; keep `MirroredTurnoverCost` as cross-check. (No `_metric.py` change.)
- **Modify** `scripts/_loader_databricks.py` — W1: `c.player_key` in `load_xtgk_cohort` (done); confirm.
- **Modify** `scripts/validate_xtgk_v2.py` — W4: fit + inject the faithful `EmpiricalTurnoverValue` in `main()`; add the component-decomposition + mirror-vs-empirical-divergence to the committed report; W5 keeper-discrimination; W6 audit.
- **Create** `tests/xtgk/test_turnover_faithful.py` — W2 TDD (sparse-deep non-vacuous fixture, support-gate, port, divergence).
- **Create** `tests/xtgk/test_player_key_contract.py` — W3 data-contract (`player_key ≥99%`; owner-gated `@e2e` + a pure fixture guard).
- **Create (owner-run FINAL reports)** `docs/research/xtgk_v2_construct_validity/{gradientsports,skillcorner}.md` (regenerated, faithful) + `.../decomposition.md` + `.../keeper_discrimination.md` + `.../faithfulness_audit.md`.
- **Modify** docs: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-036-*.md`, `uv.lock`, `CLAUDE.md`.

---

## Task 1 — W1: loader reads the resolved keeper (`player_key`)

**Files:** `scripts/_loader_databricks.py` · `tests/xtgk/test_retention_loader_domain.py`

- [ ] **Step 1: Guard test** — append to `tests/xtgk/test_retention_loader_domain.py`:
```python
def test_xtgk_cohort_sql_selects_player_key():
    import scripts._loader_databricks as L
    assert "c.player_key" in L._XTGK_ACTIONS_SQL  # RESOLVED keeper (player_id is null for goal-kicks)
```
- [ ] **Step 2: Run** → PASS (the `c.player_key` column was already added). If it fails, add `c.player_key` to `_XTGK_ACTIONS_SQL`'s `fct_action_context c` select. `player_key` is a surrogate id — do NOT coerce numeric; leave as returned. Confirm `prepare_cohort` carries it (it `.copy()`s + row-filters — no column drop).
- [ ] **Step 3:** `python -m pytest tests/xtgk/test_retention_loader_domain.py -q` → PASS.

---

## Task 2 — W2: faithful `V_opp` (observed-post-turnover + deep-cell bin-widening)

**Files:** `silly_kicks/xtgk/_turnover.py` · `tests/xtgk/test_turnover_faithful.py`

**Design (the key engineering — flagged for review):** `EmpiricalTurnoverValue.fit` currently sets sparse cells to `0.0` (verified). Add a **support-gated hierarchical fallback** so every zone resolves to the finest estimate with enough support:
- Level 0 = native (zone,p) cell (`num/den`, current logic).
- Level 1 = **coarse block** — pool the l×w grid into blocks of `coarsen×coarsen` native cells; aggregate turnovers into the block; block estimate = block_num/block_den.
- Level 2 = **global per pressure** (all turnovers at that tercile).
- `value(zone,p)` returns the finest level whose support ≥ `min_support`: native if `den(cell)≥min_support`, else the containing coarse block if `block_den≥min_support`, else the global-per-p mean. Store the **resolved surface** + native `support` (honest per-cell n, exposed via the port) + a per-cell **resolution level** for the report.
- New params: `min_support: int = 30` (= the gate's pre-registered `n_min`) + `coarsen: int` (a-priori-justified). Report per-zone n and the resolution-level map.
- **Window scope (R1 — §3 integrity):** support `window_seconds=None` → **possession-bound** (drop only the `(t[j]−t[i]) > window_seconds` break in `_opp_first_shot_after_turnover`; the `team[j]==team[i]` ball-back break already bounds it — the "free" scope-symmetric variant). **Production V_opp is possession-bound** (scope-symmetric with V; set a priori). Keep the 10s-capped path for the reported sensitivity. **This matters because a 10s cap mechanically shrinks V_opp everywhere — so a dzv-dominance drop under a 10s-capped faithful V_opp could be a window artifact, NOT "the mirror over-stated deep threat." Those are opposite findings; §3 forbids banking the artifact.**
- **R5 caveat (document):** under `zone_conditional` terciles, native cells pooled into one coarse block may have had their p-level assigned with different per-band cutpoints (a block's `p=2` cells can mean different absolute pressure). Second-order (a coarse block is a spatial neighborhood, usually one band) — note it in the class docstring + the report.
- **`game_id` guard (R6 — correctness consequence of possession-bound):** with `window_seconds=None`, the ONLY remaining hard bound in `_opp_first_shot_after_turnover` is `game[j] != game[i]`. But `fit` defaults `game` to zeros when `game_id` is absent, and `time_seconds` resets per match — so a missing/null `game_id` on a multi-match cohort lets the uncapped scan cross match boundaries (charge a match-A turnover with a match-B opponent shot; the old 10s cap masked this only buggily — a reset clock makes `t[j]−t[i]` negative so the time break never fired). **Add an input-validation guard in `fit`: `game_id` (or `match_key`) present AND non-null** (fail-loud, per the xtgk input-validator pattern, ADR-017/019). Production cohorts carry `game_id` (via `load_xtgk_cohort`), so this only bites a malformed caller.
- **Docstring:** change from "NOT shipped cross-check" → "faithful production `V_opp` (§2.3, observed post-turnover, possession-bound, bin-widened)." `MirroredTurnoverCost` docstring → "cross-check (was production; over-states deep cells)."

- [ ] **Step 1: Write the tests** — `tests/xtgk/test_turnover_faithful.py`:
```python
import numpy as np, pandas as pd
from silly_kicks.xtgk import EmpiricalTurnoverValue, MirroredTurnoverCost, TurnoverCost
# ... build a synthetic action cohort with:
#  (a) a WELL-SUPPORTED cell -> value == its native empirical mean (level 0);
#  (b) a SPARSE DEEP cell (den < min_support) whose COARSE BLOCK is well-supported
#      -> value falls back to the block estimate (NOT 0.0, NOT global) -- the NON-VACUOUS widening case;
#  (c) a fully-empty region -> value == global-per-p mean (level 2).
# Assert: isinstance(tc, TurnoverCost); support(p) exposes native n; resolution-level map records 0/1/2;
#  a known post-turnover chain yields the expected opp-first-shot xG at its loss zone.
def test_faithful_vopp_bin_widening_non_vacuous(): ...
def test_faithful_vopp_satisfies_turnovercost_port(): ...
def test_mirror_vs_empirical_divergence_reported(): ...  # divergence helper returns per-zone |emp-mirror|
def test_possession_bound_vs_10s_scope(): ...  # R1: a chain whose opp shot is >10s but within the won
#   possession -> CREDITED when window_seconds=None (possession-bound), DROPPED when window_seconds=10.0.
#   Fixture carries game_id for >=2 matches (R6): a match-B opponent shot is NOT charged to a match-A turnover
#   under window_seconds=None (the match boundary bounds the scan, not the removed time cap).
def test_fit_requires_non_null_game_id(): ...  # R6: fit raises fail-loud on absent/null game_id (multi-match
#   possession-bound correctness depends on it, per ADR-017/019 input-validator pattern).
```
The sparse-deep fixture MUST actually trigger the level-1 fallback (assert the resolved value equals the block estimate, ≠ 0 and ≠ global) — else the widening branch is untested (the recurring vacuity trap).
- [ ] **Step 2: Run** → FAIL (bin-widening + params don't exist).
- [ ] **Step 3: Implement** the fallback in `EmpiricalTurnoverValue.fit` (compute native + coarse + global per tercile; build the resolved surface + level map), update `value`/`surface`/`support`, add `min_support`/`coarsen`, update docstrings, and add a `divergence_vs(mirror, p)` helper (per-zone `|resolved − mirror.surface(p)|`).
- [ ] **Step 4: Run** the W2 tests + the xtgk regression + metric tests → PASS. `compute_xt_gk_v2` unchanged (injection seam): `python -m pytest tests/xtgk/ -q`.

---

## Task 3 — W3: best-practice guards

**Files:** `tests/xtgk/test_player_key_contract.py` · `CLAUDE.md`

- [ ] **Step 1:** `tests/xtgk/test_player_key_contract.py` — an owner-gated `@pytest.mark.e2e` test asserting `fct_action_context.player_key ≥ 99%` non-null on `is_gk_distribution` rows (live current 99.9%) — catches a wrong-column read AND a resolver regression; PLUS a CI-runnable pure test that `load_xtgk_cohort`'s SQL sources keeper identity from `fct_action_context` (`c.player_key`), never `bronze.spadl_actions` (assert `s.player_key`/raw-bronze keeper is not the source).
- [ ] **Step 2:** Document the convention in `CLAUDE.md` (one line, near the GK narrative): **GK-domain consumers use `player_key` (resolved), never `player_id` (raw, null for goal-kicks); analysis loaders source resolved fields from gold marts, not raw bronze.**
- [ ] **Step 3:** Relay the canonical-GK-marts note (`fct_gk_tracking_actions`/`fct_gk_tracking_stats`; `dist_xt_gk_v2_mean` per `gk_player_key` on the 4.45.0 re-materialization) to the lakehouse — capture it in the ADR amendment (Task 7) + the CHANGELOG Hyrum flag. Run: `python -m pytest tests/xtgk/test_player_key_contract.py -q -m "not e2e"` → PASS.

---

## Task 4 — W4: construct-validity + component decomposition on the FAITHFUL metric (final reports)

**Files:** `scripts/validate_xtgk_v2.py` · `docs/research/xtgk_v2_construct_validity/*.md`

- [ ] **Step 1:** In `validate_xtgk_v2.py::main()`, fit the faithful `EmpiricalTurnoverValue` (**possession-bound**, `min_support=30`, `coarsen`) **on the train split** and inject it as `turnover_cost` (replacing `MirroredTurnoverCost`). Keep the mirror + a 10s-capped empirical fitted for the reports. Extend `_write_report` with:
  - the component-decomposition table (`pos/pev/ret/dzv` |mean| share + `ρ·ΔV`-alone / partial / full AUC≤5 & Spearman);
  - **R1 deep-cell disentanglement** — per deep cell, side by side: **native empirical value · native n · resolution level (0/1/2) · mirror value · possession-bound value · 10s-capped value**. This lets the reader separate "mirror over-stated deep threat" (production possession-bound ≪ mirror, at real support) from "window shrinks V_opp" (10s ≪ possession-bound). Do NOT report only the aggregate dzv drop;
  - **R4** — state the report is on the **train-fit** V_opp (no leakage into the AUC) and **emit the resolution-level map** for the train fit (deep cells are sparser at half the data → more global fallbacks; a level-2 (global) deep cell must NOT read as a real estimate).
- [ ] **Step 2:** Add a CI test that `main`/`construct_validity_scores` run on a synthetic cohort with a faithful (fittable) turnover cost (extend `tests/xtgk/test_validate_v2_real_rho.py`). Keep the existing stub-path smoke green.
- [ ] **Step 3 (owner-run, local Databricks):** `python scripts/validate_xtgk_v2.py --provider gradientsports` and `--provider skillcorner`. **Record the faithful numbers honestly (§3)** — the lift and the decomposition before/after (did `dzv`'s 89% dominance drop? did `ρ·ΔV` show through?). Commit `{gradientsports,skillcorner}.md` + `decomposition.md` (final, not preliminary).

---

## Task 5 — W5: keeper discrimination + face validity (the real SP5 instrument)

**Files:** `scripts/validate_xtgk_v2.py` (or a sibling `scripts/xtgk_v2_keeper_discrimination.py`) · `docs/research/xtgk_v2_construct_validity/keeper_discrimination.md`

- [ ] **Step 1:** Add a keeper-discrimination routine: compute `compute_xt_gk_v2` (production ρ + **faithful** `V_opp`, fit on the **full** GK-distribution cohort — descriptive spread, R4; state this in the report) over the cohort. Compare v2 vs **v1** (`c.xt_gk`) on the SAME cohort.
  - **ICC — computed on the ACTION-level values grouped by `player_key`, NOT on per-keeper means (R2).** Each keeper's ≥N distributions are the within-keeper replicates; ICC = between-keeper variance ÷ (between + within) partitioned from the **per-action** `xt_gk_v2`. (ICC on collapsed means is degenerate — no within-group term.) v1 the same way on the same cohort. **`≥N`-distribution filter stated a priori** (gives each keeper enough replication for a stable within-keeper term).
  - **Per-keeper mean is ONLY for the ranking**, not the ICC. Report CV secondary WITH the "unstable near zero mean" caveat (the current-metric CV blew up on near-zero-mean partials — 1.5–10). Emit the per-keeper **ranking** for the owner's coaching eye.
- [ ] **Step 2:** CI test on synthetic keepers (pure) — ICC partitions variance from **action-level** values (assert it changes when within-keeper spread changes, i.e. not degenerate-on-means), ranking emits, `≥N` filter applies (non-vacuous: ≥2 keepers each side, each with ≥N actions).
- [ ] **Step 3 (owner-run):** run GS + SC; commit `keeper_discrimination.md` — v2-vs-v1 ICC (+ CV caveat), keeper count, the ranking for the owner's coaching eye. **Report whatever it shows (§3):** if v2 still ≈ v1-flat, that is the finding.

---

## Task 6 — W6: secondary faithfulness audit (committed report section)

**Files:** `docs/research/xtgk_v2_construct_validity/faithfulness_audit.md`

- [ ] **Step 1:** Write `faithfulness_audit.md` covering, each **done-in-PR or deferred-with-written-reason** (no silent scope creep/omission, §W6):
  - **V reward interpretation** — we use `E[first-shot xG]`; Jeff §2.1 says "expected threat over the remainder of the possession." State the interpretation + the evidence (V's weak realized-xG OOS Spearman 0.03–0.06). **Deferred** (do NOT re-implement V here) — flagged for owner/Jeff.
  - **κ sweep `[1,2]` re-examined AFTER the faithful `V_opp`** — run the sweep, report AUC/keeper-ICC per κ. **Do NOT retune κ to pass (§3)** — report the sweep as evidence for the Jeff κ/turnover-weighting question.
  - **PEV dormant** (`p′=p`, receiver-pressure `q` deferred per Jeff §8-step-7) — note the metric carries no pressure-value-added term.
- [ ] **Step 2 (owner-run):** run the κ sweep locally, fill the numbers, commit.

---

## Task 7 — Version 4.45.0 + docs + ADR + final-review

- [ ] **Step 1:** Bump 4.44.0 → **4.45.0** in `pyproject.toml` + `silly_kicks/__init__.py` + `uv.lock`; `grep -rn "4.44.0" pyproject.toml silly_kicks/__init__.py` → none; `uv lock --check` → 0.
- [ ] **Step 2:** CHANGELOG `## [4.45.0]` — the faithful `V_opp` swap (mirror→observed-post-turnover, bin-widened; **`compute_xt_gk_v2` serve-output change → Hyrum-flag the lakehouse to re-materialize `xt_gk_v2_*` on 4.45.0**); the W1 keeper-column fix + W3 guards; the W4/W5/W6 results (summarize the real numbers). This IS a library behaviour change.
- [ ] **Step 3:** TODO — close the SP5 + faithful-`V_opp` items; update Current release to 4.45.0/PR-S112; keep any owner/Jeff-deferred items (V-reward interpretation).
- [ ] **Step 4:** ADR-036 amendment — the `V_opp` faithfulness fix (§2.3) + bin-widening; role reversal (mirror→cross-check); the decomposition before/after; the keeper-discrimination verdict; the `player_key` convention + data-contract; the canonical-GK-marts relay.
- [ ] **Step 5:** `/final-review` (C4 count stays 28 — no new aggregator/container; a turnover-adapter swap + reports are not architectural). Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip` · `ruff check . && ruff format --check .` · `pyright`. All green.

---

## Task 8 — Single commit (explicit approval) + push + PR

- [ ] **Step 1:** Present `git status` + `git diff --stat` + the headline faithful results (lift, decomposition before/after, keeper ICC v2-vs-v1). **REQUEST APPROVAL.**
- [ ] **Step 2 (after approval): Commit** (one squashed commit):
```
feat(xtgk): finish xT-GK v2 -- faithful observed-post-turnover V_opp + full validation -- silly-kicks 4.45.0 (ADR-036, PR-S112)
```
- [ ] **Step 3:** `git push -u origin pr-s112-xtgk-v2-construct-validity` + `gh pr create`. Watch CI only if asked.

---

## Self-Review Notes
- **W1 done** (loader `player_key`); Task 1 is a guard + confirm.
- **W2 is the real engineering** — the bin-widening design (native→coarse-block→global, support-gated) is spelled out; the sparse-deep fixture is required to be **non-vacuous** (asserts the fallback fires).
- **R1 (window-scope confound) folded in** — production `V_opp` is **possession-bound** (scope-symmetric with V, set a priori), 10s reported as a sensitivity, and the W4 report **disentangles over-statement from window-shrinkage per deep cell** (native/n/level/mirror/possession-bound/10s side by side) so a dzv drop can't be banked as validation if it's a window artifact (§3).
- **R2 (ICC correctness) folded in** — ICC is computed on **action-level** values grouped by `player_key` (within-keeper replication), NOT collapsed per-keeper means (degenerate); per-keeper mean only for the ranking.
- **R3 (a-priori params) folded in** — `window`=possession-bound, `min_support=30` (=gate `n_min`), `coarsen` justified, `κ=1` headline; the W6 κ-sweep is reported-for-Jeff only. All FIXED before any W4/W5 number is read (§3).
- **R4/R5 folded in** — each report states its V_opp fit split (W4 train / W5 full) + emits the resolution-level map (a global-fallback deep cell isn't read as real); the zone_conditional × coarse-pool caveat is documented.
- **R6 folded in** — the possession-bound switch makes correctness depend on `game_id` (the only remaining scan bound); `fit` fail-loud-guards `game_id` present+non-null (ADR-017/019), and the scope test carries ≥2 matches to exercise the cross-boundary case.
- **No preliminary reports** — proxy GS/SC reports discarded; only faithful finals committed.
- **Single commit** in Task 8 behind approval; 4.45.0 minor.
