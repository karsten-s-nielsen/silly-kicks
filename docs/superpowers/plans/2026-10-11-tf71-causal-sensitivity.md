# TF-71 Causal Unmeasured-Confounding Sensitivity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure, deterministic unmeasured-confounding sensitivity layer to `silly_kicks.causal` — the E-value (VanderWeele & Ding 2017) and the marginal sensitivity model (Zhao–Small–Bhattacharya 2019), both computed on a Hájek-IPW ATT, reimplementing the published ZSB algorithm (oracle-validated against `bootsens/R/ipw.R`) — and wire it into the two causal harnesses.

**Architecture:** New sibling module `causal/sensitivity.py` beside `matching.py`/`opportunities.py`/`power.py`. The matched ATT stays the headline estimate (unchanged); a Hájek-IPW ATT built from the SAME `fit_propensity` carries the MSM (ZSB's proven estimator + valid percentile bootstrap). The E-value's risk ratio + CI are also computed on the IPW estimand (the matched RR is a reported companion). `bootsens/R/ipw.R` is a committed numeric reference oracle.

**Tech Stack:** Python, numpy, scipy, the existing `silly_kicks.causal.matching.fit_propensity` (sklearn, fit-time only). No new runtime dependency. R (`bootsens`) used ONCE, offline, to generate the committed oracle fixture — never a runtime/CI dep.

**Spec:** `docs/superpowers/specs/2026-10-11-tf71-causal-sensitivity-design.md` (APPROVE ×2 at r4). Executors read both.

## Global Constraints

- **ONE approval-gated final commit** (Karsten's discipline) — spec + plan + code + tests + oracle fixture + CHANGELOG together. **NO per-task commits.** Each task ends with its tests green; the executor does NOT `git add`/`git commit` until the whole plan is done, the full suite is green, and the owner has given an explicit per-commit yes. The skill's "commit after each step" cadence is overridden here.
- **Feature branch** `feat/tf71-causal-sensitivity` (already created off `main` `02258e9`); no worktree.
- **Pure / additive:** `causal/matching.py` and `CausalEstimate` are UNTOUCHED. `Z`/`X`/`Y` are never mutated. Deterministic given `seed`.
- **Faithful port:** `_get_extrema`/`_extrema_md`/`_extrema_os_att`/`_bootsens_att` reproduce `bootsens/R/ipw.R` exactly (handoff at `D:\Development\_handoffs\silly-kicks-tf71\bootsens_ipw.R`). The `ipw.R` numeric oracle (Task 2) is the faithfulness gate.
- **Pinned NO-GO bars (pre-numbers, tighten-only):** `agreement_ok` iff `|matched_att - ipw_att| <= matched_se` (one Abadie–Imbens SE); `overlap_ok` iff `ESS = (Σw)²/Σw² >= 0.10·n_focal` AND `max normalized weight <= 0.10`.
- **Pinned coverage gate:** planted-confounder MSM coverage `>= 0.90` across `>= 500` seed-fixed replications.
- **E-value on the IPW estimand:** `rr = mean(Y[Z==1]) / p0_cf` (`p0_cf` = γ=0 IPW `E[Y(0)|A=1]`); CI = ZSB percentile bootstrap (resample + refit, NO rematch). `matched_rr` is a companion only.
- **γ = log Λ** (log odds ratio); `gammas[0] == 0.0` ⇒ Γ = 1; the γ=0 collapse (`min == max`) is automatic.
- **Surface gates:** register the new module in `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`; `SensitivityResult` in `_SKIP_SYMBOLS`; doctest on `sensitivity_analysis`). `tests/scripts/test_input_contracts.py` stays green (warns, not fails). ruff + pyright at CI scope.
- **No DGX, no corpus run, no memo regeneration** this cycle. The xcross/`tf19_entanglement` memo regen is a tracked owner-run follow-up.

---

## File Structure

- **Create** `silly_kicks/causal/sensitivity.py` — the whole seam: `SensitivityResult`, the ZSB reimplementation (`_get_extrema`/`_extrema_md`/`_extrema_os_att`/`_bootsens_att`), the NO-GO helpers (`_agreement_ok`/`_ess_overlap`), the E-value (`e_value` + `_ev`/`_ev_ci_limit`/`_matched_rr`), `marginal_sensitivity`, `sensitivity_analysis`. One responsibility: sensitivity analysis. ~230 LOC.
- **Create** `tests/causal/test_sensitivity.py` — the gates (Tasks 1–6, 8).
- **Create** `tests/causal/fixtures/ipw_oracle.json` — the committed ipw.R oracle fixture (pinned inputs → R targets + provenance), produced by the Docker generator below.
- **Create** `tools/oracle/ipw/{Dockerfile,gen_ipw_oracle.R,README.md}` — the dev-only pinned R oracle generator (Docker). **New top-level `tools/` dir** (silly-kicks' first Docker; matches the lakehouse's Spark-local Docker pattern). Not in the wheel, not in CI-for-tests. Routine placement — relocatable to `scripts/oracle/ipw/` at build if the owner prefers to avoid a new top-level dir.
- **Modify** `silly_kicks/causal/__init__.py` — export the public names.
- **Modify** `tests/test_public_api_examples.py` — register the module + `_SKIP_SYMBOLS`.
- **Modify** `scripts/validate_xcross_causal.py`, `scripts/validate_xshot_causal.py` — call + emit + `declare_inputs`.
- **Modify** `CHANGELOG.md` + `TODO.md` (the owed corpus-regen follow-up row).
- **Modify** `NOTICE` — the academic-attribution entry (VanderWeele–Ding, Zhao–Small–Bhattacharya, Otsu–Rai, Abadie–Imbens; `bootsens` GPL-3-in-image note).
- **Modify** `silly_kicks/_version.py` — the single-sourced minor bump (ADR-079), number confirmed with the owner at commit-prep.
- **Modify** `AGENTS.md` — add `sensitivity_analysis` to the causal map bullet (budget-neutral; **`tests/test_agents_md_budget.py` must stay green** — the bullet stays ≤ 600 chars and the file ≤ the byte ceiling).
- **Create** `docs/superpowers/adrs/ADR-117-causal-unmeasured-confounding-sensitivity.md` + this spec + plan (the docs travel with the one commit).

---

### Task 1: `_get_extrema` — the closed-form IPW extrema (ipw.R `get.extrema`)

**Files:**
- Create: `silly_kicks/causal/sensitivity.py`
- Create: `tests/causal/fixtures/ipw_oracle.json` (committed oracle targets, from the Docker R run)
- Create: `tools/oracle/ipw/Dockerfile`, `tools/oracle/ipw/gen_ipw_oracle.R`, `tools/oracle/ipw/README.md` (dev-only pinned R oracle generator; not in the wheel/CI-for-tests)
- Test: `tests/causal/test_sensitivity.py`

**Interfaces:**
- Produces: `_get_extrema(A, Y, fitted_prob, gamma, *, estimand_all) -> tuple[float, float]` (min, max).

**Oracle fixture generation — Dockerized, pinned, repeatable (owner-ruled 2026-10-11):** the `<R…>` targets are produced by a dev-only pinned Docker image, NEVER by hand and NEVER by the Python port (a port-filled target makes gate 2 circular/vacuous). `tools/oracle/ipw/Dockerfile` = a pinned `rocker/r-ver:4.4.1` base that `COPY`s the committed `bootsens` `ipw.R` (self-contained base-R + `data.table`; no GitHub/network pull) and installs `data.table` pinned; `tools/oracle/ipw/gen_ipw_oracle.R` defines the pinned fixture inputs (no Y-ties — avoids R-vs-numpy sort-tie ambiguity), `source()`s `ipw.R`, computes `get.extrema` / `extrema.os` for cases a/b/c, and writes `ipw_oracle.json` with provenance (`rocker` image digest + `ipw.R` sha256 + `data.table` version). Regen = `docker run --rm -v <repo>:/work <image> Rscript /work/tools/oracle/ipw/gen_ipw_oracle.R` (README documents it); matches the lakehouse's Docker-for-Spark pattern + the das-reference pinned-oracle discipline. **R is absent on dev hosts; Docker supplies it** — the Task-1 prerequisite, dev-only (never a runtime/CI-for-tests dep; the committed JSON is all the tests read). **If the Docker run has not produced the targets, Task 1 is BLOCKED and so is the commit** — surface to the owner; never fill placeholders with port output. Shape:
```json
{"source": "bootsens/R/ipw.R @ github.com/qingyuanzhao/bootsens",
 "cases": [
   {"name": "get_extrema_all_g0", "fn": "get.extrema", "estimand_all": true, "gamma": 0.0,
    "A": [1,1,0,1,0], "Y": [1.0,0.0,1.0,0.0,1.0], "fitted_prob": [0.6,0.4,0.55,0.7,0.3],
    "expected": [<Rmin>, <Rmax>]},
   {"name": "get_extrema_missing_g1", "fn": "get.extrema", "estimand_all": false, "gamma": 1.0, "...": "..."},
   {"name": "os_att_g1_noadj", "fn": "extrema.os.att", "reg_adjust": false, "gamma": 1.0,
    "A": [...], "X": [[...]], "Y": [...], "fitted_prob_flip": [...], "expected": [<lo>, <hi>]},
   {"name": "os_att_g1_aipw", "fn": "extrema.os.att", "reg_adjust": true, "gamma": 1.0,
    "A": [...], "X": [[...]], "Y": [...], "fitted_prob_flip": [...], "y_fitted": [...], "expected": [<lo>, <hi>]}
 ]}
```
(`fitted_prob_flip` = `glm(1-A ~ X)` fitted probs — the `extrema.os(att)` path refits on `1-A`, so the injected prob is the flipped-treatment propensity; `y_fitted` = the AIPW outcome-regression predictions; both emitted by `gen_ipw_oracle.R` so the Python injection seam is driven by the SAME inputs R used.) The `<R…>` values come ONLY from the Docker R run (independence HARD RULE above) and are committed with their provenance block; a reviewer can re-run the container and diff.

- [ ] **Step 1: Write the failing oracle test for `_get_extrema`**
```python
import json, pathlib
import numpy as np
import pytest
from silly_kicks.causal import sensitivity as S

_FIX = json.loads((pathlib.Path(__file__).parent / "fixtures" / "ipw_oracle.json").read_text())

def _case(name):
    return next(c for c in _FIX["cases"] if c["name"] == name)

def test_get_extrema_matches_ipw_r():
    for name in ("get_extrema_all_g0", "get_extrema_missing_g1"):
        c = _case(name)
        lo, hi = S._get_extrema(np.array(c["A"], float), np.array(c["Y"], float),
                                np.array(c["fitted_prob"], float), c["gamma"],
                                estimand_all=c["estimand_all"])
        assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)
```
- [ ] **Step 2: Run → FAIL** (`pytest tests/causal/test_sensitivity.py::test_get_extrema_matches_ipw_r -v`) — `ModuleNotFoundError`/`AttributeError`.
- [ ] **Step 3: Implement `_get_extrema`** in `silly_kicks/causal/sensitivity.py`
```python
"""Unmeasured-confounding sensitivity for silly_kicks.causal (TF-71, ADR-015 sibling).

E-value (VanderWeele & Ding 2017) + marginal sensitivity model (Zhao-Small-Bhattacharya 2019),
both on a Hajek-IPW ATT, reimplementing the published ZSB algorithm (oracle-validated against bootsens/R/ipw.R). Pure; deterministic given seed.
See docs/superpowers/specs/2026-10-11-tf71-causal-sensitivity-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silly_kicks.causal.matching import estimate_att, fit_propensity, propensity_match

DEFAULT_GAMMAS = (0.0, 0.22, 0.41, 0.69, 1.10, 1.39, 1.79)  # log-OR; Gamma ~ {1,1.25,1.5,2,3,4,6}
DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05
OVERLAP_MIN_ESS_FRACTION = 0.10
OVERLAP_MAX_WEIGHT_SHARE = 0.10


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _get_extrema(A, Y, fitted_prob, gamma, *, estimand_all) -> tuple[float, float]:
    """Reimplements ZSB's get.extrema (oracle-validated against bootsens).
    Returns (min, max) of the IPW estimator over the Gamma ball."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    c = 1.0 if estimand_all else 0.0
    eg = np.exp(-_logit(fitted_prob))
    keep = A != 0
    eg, Yk, Ak = eg[keep], Y[keep], A[keep]
    order = np.argsort(-Yk, kind="stable")  # R order(-Y): Y descending
    eg, Yk, Ak = eg[order], Yk[order], Ak[order]
    lo = Ak * (c + np.exp(-gamma) * eg)
    hi = Ak * (c + np.exp(gamma) * eg)

    def _combine(fwd, rev):  # c(0, cumsum(fwd)) + c(rev(cumsum(rev(rev))), 0)
        a = np.concatenate(([0.0], np.cumsum(fwd)))
        b = np.concatenate((np.cumsum(rev[::-1])[::-1], [0.0]))
        return a + b

    with np.errstate(invalid="ignore", divide="ignore"):
        maximum = np.nanmax(_combine(Yk * hi, Yk * lo) / _combine(hi, lo))
        minimum = np.nanmin(_combine(Yk * lo, Yk * hi) / _combine(lo, hi))
    return float(minimum), float(maximum)
```
- [ ] **Step 4: Run → PASS.** The max uses `up` forward + `low` reverse; the min swaps — exactly ipw.R lines 45–60.

---

### Task 2: `_extrema_md` / `_extrema_os_att` + the injection seam + AIPW (ipw.R `extrema.md`/`extrema.os`)

**Files:** Modify `silly_kicks/causal/sensitivity.py`; Test `tests/causal/test_sensitivity.py`.

**Interfaces:**
- Consumes: `_get_extrema`, `fit_propensity`.
- Produces:
  `_ols_fit_predict(X, Y, mask) -> np.ndarray`;
  `_extrema_md(A, X, Y, gamma, *, estimand_all, reg_adjust, seed, fitted_prob=None, y_fitted=None) -> tuple[float,float]`;
  `_extrema_os_att(A, X, Y, gamma, *, reg_adjust, seed, fitted_prob=None, y_fitted=None) -> tuple[float,float]`.
  `fitted_prob=`/`y_fitted=` are the **test-only injection seam** (bypass `fit_propensity ≠ R-glm`).

- [ ] **Step 1: Write the failing oracle tests (cases b + c)**
```python
def test_extrema_os_att_matches_ipw_r_noadjust():
    c = _case("os_att_g1_noadj")
    lo, hi = S._extrema_os_att(np.array(c["A"], float), np.array(c["X"], float), np.array(c["Y"], float),
                               c["gamma"], reg_adjust=False, seed=0,
                               fitted_prob=np.array(c["fitted_prob_flip"], float))
    assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)

def test_extrema_os_att_matches_ipw_r_aipw():
    c = _case("os_att_g1_aipw")
    lo, hi = S._extrema_os_att(np.array(c["A"], float), np.array(c["X"], float), np.array(c["Y"], float),
                               c["gamma"], reg_adjust=True, seed=0,
                               fitted_prob=np.array(c["fitted_prob_flip"], float),
                               y_fitted=np.array(c["y_fitted"], float))
    assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)
```
- [ ] **Step 2: Run → FAIL** (functions not defined).
- [ ] **Step 3: Implement**
```python
def _ols_fit_predict(X, Y, mask) -> np.ndarray:
    """lm(Y ~ X) fit on rows where mask is True, predicted for ALL rows (ipw.R OR step)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd[mask], Y[mask], rcond=None)
    return Xd @ beta


def _extrema_md(A, X, Y, gamma, *, estimand_all, reg_adjust, seed, fitted_prob=None, y_fitted=None):
    """Reimplements ZSB's extrema.md (oracle-validated against bootsens).
    fitted_prob / y_fitted are the test-only injection seam."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if fitted_prob is None:
        fitted_prob, _ = fit_propensity(X, A.astype(int), seed=seed)
    if y_fitted is None:
        y_fitted = _ols_fit_predict(X, Y, A == 1) if reg_adjust else np.zeros_like(Y)
    lo, hi = _get_extrema(A, Y - y_fitted, fitted_prob, gamma, estimand_all=estimand_all)
    add = float(np.mean(y_fitted)) if estimand_all else float(np.mean(y_fitted[A == 0]))
    return lo + add, hi + add


def _extrema_os_att(A, X, Y, gamma, *, reg_adjust, seed, fitted_prob=None, y_fitted=None):
    """Reimplements ZSB's extrema.os(estimand='att') = mean(Y[A==1]) - rev(extrema.md(1-A,...,'missing'))."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    lo, hi = _extrema_md(1.0 - A, X, Y, gamma, estimand_all=False, reg_adjust=reg_adjust,
                         seed=seed, fitted_prob=fitted_prob, y_fitted=y_fitted)
    treated_mean = float(np.mean(Y[A == 1]))
    return treated_mean - hi, treated_mean - lo  # mean - rev([lo,hi])
```
- [ ] **Step 4: Run → PASS** (a/b/c now all oracle-verified; mis-ported add-back, `rev`, or `1-A` would fail).

---

### Task 3: `_bootsens_att` — ZSB percentile bootstrap (ipw.R `bootsens.os`)

**Files:** Modify `silly_kicks/causal/sensitivity.py`; Test `tests/causal/test_sensitivity.py`.

**Interfaces:**
- Consumes: `_extrema_os_att`.
- Produces: `_bootsens_att(A, X, Y, *, gamma, alpha, n_boot, reg_adjust, seed) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing determinism test**
```python
def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Z = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    Y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * Z + X[:, 0])))).astype(float)
    return Z, X, Y

def test_bootsens_att_deterministic():
    Z, X, Y = _toy()
    a = S._bootsens_att(Z.astype(float), X, Y, gamma=0.5, alpha=0.05, n_boot=50, reg_adjust=False, seed=7)
    b = S._bootsens_att(Z.astype(float), X, Y, gamma=0.5, alpha=0.05, n_boot=50, reg_adjust=False, seed=7)
    assert a == b                       # same seed -> identical
    assert a[0] <= a[1]
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
def _bootsens_att(A, X, Y, *, gamma, alpha, n_boot, reg_adjust, seed):
    """Reimplements ZSB's bootsens.os (oracle-validated against bootsens): resample n with replacement
    + REFIT propensity inside each resample (ZSB's validity condition; no rematch), percentile CI.
    Deterministic via one default_rng(seed)."""
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(A)
    rng = np.random.default_rng(seed)
    los = np.empty(n_boot)
    his = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        los[b], his[b] = _extrema_os_att(A[s], X[s], Y[s], gamma, reg_adjust=reg_adjust, seed=seed)
    return float(np.quantile(los, alpha / 2)), float(np.quantile(his, 1 - alpha / 2))
```
- [ ] **Step 4: Run → PASS.** (`np.quantile` default = R type-7 linear; matches `quantile()`.)

---

### Task 4: `SensitivityResult` + `marginal_sensitivity` + the γ=1 collapse gate

**Files:** Modify `silly_kicks/causal/sensitivity.py`; Test `tests/causal/test_sensitivity.py`.

**Interfaces:**
- Consumes: `_extrema_os_att`, `_bootsens_att`, `estimate_att`, `abadie_imbens_se`, `fit_propensity`.
- Produces: the `SensitivityResult` dataclass (exact fields per spec §SensitivityResult) + `_ess`/`_overlap_ok` + `marginal_sensitivity(Z, X, Y, *, gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, reg_adjust=False, seed) -> SensitivityResult`.

- [ ] **Step 1: Write the failing γ=1 collapse + monotone-grid test**
```python
def test_msm_gamma1_collapses_to_point():
    Z, X, Y = _toy()
    r = S.marginal_sensitivity(Z, X, Y, gammas=(0.0, 0.5, 1.0), n_boot=20, seed=1)
    assert r.gammas[0] == 0.0
    assert r.msm_lo[0] == r.msm_hi[0] == pytest.approx(r.ipw_att, abs=1e-12)  # automatic collapse
    assert all(r.msm_lo[i] <= r.msm_hi[i] for i in range(len(r.gammas)))
    assert r.msm_lo[2] <= r.msm_lo[0] and r.msm_hi[2] >= r.msm_hi[0]           # band widens with Gamma
    assert r.e_value_point is None                                            # MSM-only entry point
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement the dataclass + helpers + `marginal_sensitivity`**
```python
@dataclass(frozen=True)
class SensitivityResult:
    rr: float | None
    rr_ci: tuple[float, float] | None
    matched_rr: float | None
    e_value_point: float | None
    e_value_ci: float | None
    target: str
    ipw_att: float
    matched_att: float
    agreement_gap: float
    agreement_ok: bool
    overlap_ok: bool
    msm_go: bool
    reg_adjust: bool
    gammas: tuple[float, ...]
    msm_lo: tuple[float, ...]
    msm_hi: tuple[float, ...]
    msm_ci_lo: tuple[float, ...]
    msm_ci_hi: tuple[float, ...]
    gamma_star: float
    n_boot: int
    alpha: float
    seed: int
    outcome_is_binary: bool
    notes: tuple[str, ...]


def _agreement_ok(matched_att: float, ipw_att: float, matched_se: float) -> bool:
    """NO-GO bar, factored out so BOTH branches are unit-testable (gate 6): the matched vs IPW
    ATT point estimates agree within one Abadie-Imbens SE (self-scaling, no arbitrary threshold)."""
    return bool(abs(matched_att - ipw_att) <= matched_se)


def _ess_overlap(Z, X, seed) -> tuple[bool, float]:
    """Hajek-weight overlap guard for the ATT counterfactual. The weights are the ATT control-side
    odds-weights ps/(1-ps) (a PROXY for the Hajek 'missing'-arm weights the ipw.R port forms on 1-A);
    they share the same near-0/1-propensity blow-up, so this ESS/max-share guard is a faithful overlap
    tripwire even though it is not the identical per-replicate weight vector."""
    ps, _ = fit_propensity(X, np.asarray(Z, dtype=int), seed=seed)
    ctrl = np.asarray(Z) == 0
    w = ps[ctrl] / (1.0 - ps[ctrl])          # ATT control weights ~ odds (proxy; see docstring)
    if len(w) == 0 or w.sum() <= 0:
        return False, 0.0
    ess = (w.sum() ** 2) / np.sum(w ** 2)
    max_share = float(np.max(w) / w.sum())
    n_focal = int((np.asarray(Z) == 1).sum())
    ok = (ess >= OVERLAP_MIN_ESS_FRACTION * n_focal) and (max_share <= OVERLAP_MAX_WEIGHT_SHARE)
    return bool(ok), float(ess)


def marginal_sensitivity(Z, X, Y, *, gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT,
                         alpha=DEFAULT_ALPHA, reg_adjust=False, seed):
    Z = np.asarray(Z)
    if not (len(gammas) and gammas[0] == 0.0 and all(gammas[i] <= gammas[i + 1] for i in range(len(gammas) - 1)) and gammas[0] >= 0.0):
        raise ValueError("gammas must be ascending, non-negative, and start at 0.0")
    A = Z.astype(float)
    lo = []; hi = []; ci_lo = []; ci_hi = []
    for g in gammas:
        el, eh = _extrema_os_att(A, X, Y, g, reg_adjust=reg_adjust, seed=seed)
        cl, ch = _bootsens_att(A, X, Y, gamma=g, alpha=alpha, n_boot=n_boot, reg_adjust=reg_adjust, seed=seed)
        lo.append(el); hi.append(eh); ci_lo.append(cl); ci_hi.append(ch)
    ipw_att = lo[0]  # min == max at gamma = 0
    ps, _ = fit_propensity(X, Z.astype(int), seed=seed)
    ce = estimate_att(Y, Z, ps, X)
    matched_att, matched_se = float(ce.estimate), float(ce.se)
    gap = abs(matched_att - ipw_att)
    agreement_ok = _agreement_ok(matched_att, ipw_att, matched_se)
    overlap_ok, _ess = _ess_overlap(Z, X, seed)
    gstar = next((g for g, cl, ch in zip(gammas, ci_lo, ci_hi) if cl <= 0.0 <= ch), float("inf"))
    notes = ("MSM on the Hajek-IPW ATT (reimplemented ZSB, oracle-validated against bootsens/ipw.R); "
             "gamma_star is a property of the IPW-ATT, not a certificate for the matched ATT "
             "(point agreement is necessary, not sufficient). "
             "Does NOT establish causality or detect model misspecification.",)
    return SensitivityResult(
        rr=None, rr_ci=None, matched_rr=None, e_value_point=None, e_value_ci=None,
        target="att", ipw_att=ipw_att, matched_att=matched_att, agreement_gap=gap,
        agreement_ok=agreement_ok, overlap_ok=overlap_ok, msm_go=(agreement_ok and overlap_ok),
        reg_adjust=reg_adjust, gammas=tuple(gammas), msm_lo=tuple(lo), msm_hi=tuple(hi),
        msm_ci_lo=tuple(ci_lo), msm_ci_hi=tuple(ci_hi), gamma_star=float(gstar),
        n_boot=n_boot, alpha=alpha, seed=seed, outcome_is_binary=bool(np.all(np.isin(np.unique(Y), (0.0, 1.0)))),
        notes=notes,
    )
```
- [ ] **Step 4: Run → PASS.** `ipw_att = lo[0]` IS the γ=0 extremum (min==max), so the collapse assertion is exact.

---

### Task 5: `e_value` (IPW RR + CI) + NO-GO gate + binary guard + VanderWeele–Ding worked examples

**Files:** Modify `silly_kicks/causal/sensitivity.py`; Test `tests/causal/test_sensitivity.py`.

**Interfaces:**
- Consumes: `_extrema_md`, `propensity_match`, `fit_propensity`.
- Produces: `_ev(rr)`, `_ev_ci_limit(ci)`, `_matched_rr(Z, X, Y, *, seed)`, `e_value(Z, X, Y, *, target, n_boot, alpha, seed) -> tuple[float, float, float, tuple[float,float], float]`.

- [ ] **Step 1: Write the failing VW&D worked-example + binary-guard + NO-GO tests**
```python
def test_evalue_worked_examples():
    assert S._ev(3.9) == pytest.approx(7.263, abs=1e-3)   # VanderWeele & Ding 2017
    assert S._ev(2.0) == pytest.approx(3.414, abs=1e-3)   # 2nd non-trivial pair (2 + sqrt(2)): pins the curve
    assert S._ev(1.0) == pytest.approx(1.0, abs=1e-12)
    assert S._ev(0.5) == S._ev(2.0)                        # RR<1 reciprocal symmetry
    assert S._ev_ci_limit((0.8, 1.3)) == 1.0               # CI crosses 1 -> no confounding needed

def test_evalue_requires_binary_outcome():
    Z, X, _ = _toy()
    with pytest.raises(ValueError):
        S.e_value(Z, X, np.linspace(0, 5, len(Z)), target="att", n_boot=10, alpha=0.05, seed=0)

def test_agreement_predicate_both_sides():   # gate 6: BOTH agreement_ok branches, deterministic
    assert S._agreement_ok(0.10, 0.11, 0.05) is True      # gap 0.01 <= one A&I SE (0.05)
    assert S._agreement_ok(0.10, 0.30, 0.05) is False     # gap 0.20 >  one A&I SE -> NO-GO

def test_nogo_integration_all_sides():       # gate 6: msm_go=True side AND overlap_ok=False side
    Z, X, Y = _toy()                                      # good overlap -> matching ~ IPW agree
    r = S.marginal_sensitivity(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert r.overlap_ok is True and r.agreement_ok is True and r.msm_go is True
    rng = np.random.default_rng(3)
    Xp = rng.normal(size=(200, 1)) * 6.0                  # extreme logits -> near-0/1 propensities
    Zp = (rng.uniform(size=200) < 1 / (1 + np.exp(-Xp[:, 0]))).astype(int)
    Yp = (rng.uniform(size=200) < 0.3).astype(float)
    rp = S.marginal_sensitivity(Zp, Xp, Yp, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert rp.overlap_ok is False and rp.msm_go is False
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
def _ev(rr: float) -> float:
    rr = float(rr)
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def _ev_ci_limit(ci: tuple[float, float]) -> float:
    lo, hi = ci
    if lo <= 1.0 <= hi:
        return 1.0
    return _ev(lo if lo > 1.0 else hi)  # limit nearest 1


def _matched_rr(Z, X, Y, *, seed) -> float | None:
    ps, _ = fit_propensity(X, np.asarray(Z, dtype=int), seed=seed)
    matched = propensity_match(ps, np.asarray(Z, dtype=int), target="att")
    if not matched:
        return None
    Y = np.asarray(Y, dtype=float)
    focal = np.array(sorted(matched), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    p_focal, p_comp = float(np.mean(Y[focal])), float(np.mean(Y[comp]))
    return p_focal / p_comp if p_comp > 0 else None


def e_value(Z, X, Y, *, target="att", n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, seed):
    Z = np.asarray(Z)
    Y = np.asarray(Y, dtype=float)
    if not np.all(np.isin(np.unique(Y), (0.0, 1.0))):
        raise ValueError("e_value requires a binary outcome Y in {0, 1}")
    n = len(Z)

    def _rr(zz, xx, yy):
        p1 = float(np.mean(yy[zz == 1]))
        lo, _ = _extrema_md(1.0 - zz.astype(float), xx, yy, 0.0, estimand_all=False, reg_adjust=False, seed=seed)
        return p1 / lo if lo > 0 else np.nan

    rr = _rr(Z, np.asarray(X, float), Y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    Xf = np.asarray(X, dtype=float)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        boots[b] = _rr(Z[s], Xf[s], Y[s])
    rr_ci = (float(np.nanquantile(boots, alpha / 2)), float(np.nanquantile(boots, 1 - alpha / 2)))
    return _ev(rr), _ev_ci_limit(rr_ci), float(rr), rr_ci, _matched_rr(Z, X, Y, seed=seed)
```
- [ ] **Step 4: Run → PASS.** `_ev(3.9) = 3.9 + sqrt(3.9·2.9) = 7.263`.

---

### Task 6: `sensitivity_analysis` entry point + planted-confounder coverage + its precondition + purity

**Files:** Modify `silly_kicks/causal/sensitivity.py`; Test `tests/causal/test_sensitivity.py`.

**Interfaces:**
- Consumes: `marginal_sensitivity`, `e_value`.
- Produces: `sensitivity_analysis(Z, X, Y, *, target="att", gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, reg_adjust=False, seed) -> SensitivityResult` (the doctested public entry point).

- [ ] **Step 1: Write the failing coverage + precondition + purity tests**
```python
_TRUE_LAMBDA = 0.69   # known confounder log-OR -> true Gamma = e^0.69 ~ 2.0; MUST be a gamma grid point

def _planted(seed, *, lam=_TRUE_LAMBDA, n=600):
    """A BINARY unmeasured confounder U (log-OR = lam) drives both Z and Y via X + lam*U; Y has NO Z
    term, so the true unconfounded ATT is 0. U is BINARY so the induced sensitivity Gamma = e^lam is
    DEFINED and finite (a Gaussian U has unbounded odds-ratio -> 'true Gamma' is ill-defined). The
    propensity is fit on X only (U omitted) -> confounded. Returns Z, X, Y, U."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    U = (rng.uniform(size=n) < 0.5).astype(float)                # BINARY -> bounded odds-ratio
    eta = X[:, 0] + lam * U
    p = 1 / (1 + np.exp(-eta))
    Z = (rng.uniform(size=n) < p).astype(int)
    Y = (rng.uniform(size=n) < p).astype(float)                  # NO Z term -> true ATT = 0
    return Z, X, Y, U

def test_planted_confounder_fixture_precondition():   # gate 5 (ADR-056): fixture validity before gate 4 scores it
    Z, X, Y, U = _planted(0)
    assert set(np.unique(U)) <= {0.0, 1.0}                       # BINARY -> true Gamma = e^lam is defined
    assert 0.2 < Z.mean() < 0.8                                  # prevalence off the boundary (in-domain)
    ps, _ = S.fit_propensity(X, Z, seed=0)
    naive = S.estimate_att(Y, Z, ps, X).estimate
    assert abs(naive) > 0.02                                     # the omitted U visibly biases the naive ATT

@pytest.mark.slow   # ADR-023: expensive + invariant -> the dedicated slow job, not inline
def test_planted_confounder_coverage():
    # Pinned gate (spec): across >= 500 reps, the MSM bootstrap CI at the TRUE Gamma (gamma = lam) covers
    # the truth (ATT = 0) in >= 90%. CI band (msm_ci_lo/hi), NOT the point band; at the true Gamma, NOT the grid max.
    reps, covered = 500, 0
    lam = _TRUE_LAMBDA
    gammas = (0.0, lam)                                          # grid INCLUDES the true Gamma at index -1
    for seed in range(reps):
        Z, X, Y, _ = _planted(seed, lam=lam)
        r = S.sensitivity_analysis(Z, X, Y, gammas=gammas, n_boot=200, seed=seed)
        if r.msm_ci_lo[-1] <= 0.0 <= r.msm_ci_hi[-1]:            # CI band at the TRUE Gamma covers ATT=0
            covered += 1
    assert covered / reps >= 0.90

def test_result_determinism_seed():   # gate 7 BOTH sides: same seed identical; diff seed -> CI moves, points stable
    Z, X, Y, _ = _planted(0)
    a = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=1)
    b = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=1)
    c = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=2)
    assert a == b                                               # same seed -> byte-identical
    assert a.msm_lo == c.msm_lo and a.msm_hi == c.msm_hi and a.ipw_att == c.ipw_att  # point estimates seed-free
    assert (a.msm_ci_lo, a.msm_ci_hi) != (c.msm_ci_lo, c.msm_ci_hi)                   # CI depends on the bootstrap seed

def test_aipw_reduces_to_hajek_when_or_constant():   # gate 9: constant OR -> AIPW add-back cancels -> Hajek path
    Z, X, Y, _ = _planted(0)
    lo_h, hi_h = S._extrema_os_att(Z.astype(float), X, Y, 0.5, reg_adjust=False, seed=0)
    lo_a, hi_a = S._extrema_os_att(Z.astype(float), X, Y, 0.5, reg_adjust=True, seed=0,
                                   y_fitted=np.full(len(Z), 0.3))   # CONSTANT OR via the injection seam
    assert (lo_a, hi_a) == pytest.approx((lo_h, hi_h), abs=1e-12)

def test_purity_and_binary_fold():
    Z, X, Y, _ = _planted(0)
    Z0, X0, Y0 = Z.copy(), X.copy(), Y.copy()
    r = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert np.array_equal(Z, Z0) and np.array_equal(X, X0) and np.array_equal(Y, Y0)   # unmutated
    assert r.outcome_is_binary and r.e_value_point is not None and r.matched_rr is not None
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `sensitivity_analysis`**
```python
def sensitivity_analysis(Z, X, Y, *, target="att", gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT,
                         alpha=DEFAULT_ALPHA, reg_adjust=False, seed):
    base = marginal_sensitivity(Z, X, Y, gammas=gammas, n_boot=n_boot, alpha=alpha,
                                reg_adjust=reg_adjust, seed=seed)
    Y = np.asarray(Y, dtype=float)
    if not np.all(np.isin(np.unique(Y), (0.0, 1.0))):
        return base  # non-binary: MSM only, E-value fields stay None
    ep, eci, rr, rr_ci, mrr = e_value(Z, X, Y, target=target, n_boot=n_boot, alpha=alpha, seed=seed)
    return dataclasses.replace(base, rr=rr, rr_ci=rr_ci, matched_rr=mrr, e_value_point=ep, e_value_ci=eci)
```
*(add `import dataclasses` at the top.)*
- [ ] **Step 4: Run → PASS** (coverage gate in the slow job). Add a doctest to `sensitivity_analysis`:
```python
    """...
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> Z = (rng.uniform(size=200) < 1/(1+np.exp(-X[:,0]))).astype(int)
    >>> Y = (rng.uniform(size=200) < 0.4).astype(float)
    >>> r = sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    >>> r.gammas[0] == 0.0 and r.msm_lo[0] == r.msm_hi[0]
    True
    """
```

---

### Task 7: Exports + public-API registration (surface gate)

**Files:** Modify `silly_kicks/causal/__init__.py`; Modify `tests/test_public_api_examples.py`.

**Interfaces:** Consumes the Task 1–6 public names.

- [ ] **Step 1:** add to `causal/__init__.py` `__all__` + imports: `e_value`, `marginal_sensitivity`, `sensitivity_analysis`, `SensitivityResult` (from `silly_kicks.causal.sensitivity`).
- [ ] **Step 2:** in `tests/test_public_api_examples.py`: add `"silly_kicks/causal/sensitivity.py"` to `_PUBLIC_MODULE_FILES`; add `"SensitivityResult"` to `_SKIP_SYMBOLS` (frozen result dataclass — fields are the documentation, `DetectionResult`/`OpportunityConfig` precedent). `e_value`/`marginal_sensitivity`/`sensitivity_analysis` are documented by the `sensitivity_analysis` doctest + the `_ev`/helpers are private; if the gate still asks for an example on `e_value`/`marginal_sensitivity`, add a one-line doctest to each (prefer that over a debt entry).
- [ ] **Step 3: Run the surface gate** `.venv/Scripts/python.exe -m pytest tests/test_public_api_examples.py -q` → PASS (`test_derived_surface_is_fully_accounted_for` green; no unaccounted symbol).

---

### Task 8: Harness wiring (xcross + xshot) + the wiring test

**Files:** Modify `scripts/validate_xcross_causal.py`, `scripts/validate_xshot_causal.py`; Test `tests/causal/test_sensitivity.py` (or `tests/scripts/`).

**Interfaces:** Consumes `sensitivity_analysis`.

- [ ] **Step 1: Write the failing wiring test** (synthetic, no corpus): construct the small `(Z, X, Y)` the harness builds, call the harness's new `_sensitivity_block(Z, X, Y, seed)` helper, assert it returns a JSON-safe dict carrying `ipw_att`/`gamma_star`/`msm_go`/`e_value_point`/`agreement_gap` and that `json.dumps` succeeds.
```python
def test_sensitivity_block_is_json_safe():
    import json
    Z, X, Y, _ = _planted(0)
    from importlib import import_module
    blk = import_module("validate_xcross_causal")._sensitivity_block(Z, X, Y, seed=0)
    json.dumps(blk)
    assert {"ipw_att", "gamma_star", "msm_go", "agreement_gap"} <= set(blk)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** in each harness:
  (a) add `_sensitivity_block(Z, X, Y, *, seed) -> dict` that calls `sensitivity_analysis(...)` and `dataclasses.asdict(...)` → JSON-safe (tuples→lists, `inf`→`"inf"` string for `gamma_star`), plus the §What-it-does-NOT-measure caveat text under a `"caveat"` key;
  (b) call it after the existing `estimate_att`/`estimate_atnt` (`validate_xcross_causal.py:101-114`; the xshot twin) on the SAME `(Z, X_full, Y)`, and add the block to the `metrics` dict before `json.dumps` (`:247` / `validate_xshot_causal.py:487`);
  (c) extend `declare_inputs(...)` (`validate_xcross_causal.py:34`, `validate_xshot_causal.py:60`) to declare the sensitivity inputs — `gammas`, `n_boot`, `alpha`, `reg_adjust`, `seed` (so ADR-056 sees the dependency; this intentionally makes the committed memos read stale → `StaleArtifactWarning`, non-failing).
- [ ] **Step 4: Run → PASS.** Then `.venv/Scripts/python.exe -m pytest tests/scripts/test_input_contracts.py -q` → PASS (warns, does not fail).

---

### Task 9: Docs + the owed follow-up + full gate + STOP at the commit

**Files:** Modify `CHANGELOG.md`, `TODO.md`.

- [ ] **Step 1:** CHANGELOG `[Unreleased]` (or the owner-set version at commit-prep) Added entry: `silly_kicks.causal.sensitivity` — E-value + ZSB marginal sensitivity model (reimplements the published ZSB algorithm, oracle-validated against `bootsens/ipw.R`, on a Hájek-IPW ATT), matched-vs-IPW agreement NO-GO, AIPW variant, `ipw.R` oracle; harnesses emit the fields (committed memos pending an owner-run regen).
- [ ] **Step 2:** `TODO.md` — add the owner-approved owed follow-up row: an owner-run corpus regen of `docs/research/xcross_causal/metrics.json` + `docs/research/tf19_entanglement/metrics.json` at clean `main` to populate real numbers + clear the `StaleArtifactWarning`. Keep `tests/test_todo_md_format.py` green.
- [ ] **Step 3: Full local gate** — `.venv`/fresh-uv env: `pytest tests/ -m "not e2e" --benchmark-skip -q` (+ the `slow` job for the coverage gate), surface gates (`test_public_api_examples`, `test_input_contracts`, `test_todo_md_format`, c4, metric_contracts), ruff check + format-check at CI scope (`silly_kicks/ tests/ scripts/`), pyright on the touched files. All green (the xcross/xshot `StaleArtifactWarning` is expected + non-failing).
- [ ] **Step 4: STOP at the commit gate.** Present the diff + file list. Do NOT commit. Hand to independent `/review-impl` (×2, separate sessions; author does not review own). On dual APPROVE + explicit owner per-commit yes → ONE commit (code + tests + oracle fixture + `tools/oracle/ipw/` generator + spec + plan + CHANGELOG + TODO), then push, PR, merge after CI green. Surface at commit-prep: the minor version number; SENS-SPEC-04 (`AGENTS.md:126` ADR-005) fix-or-note (RESOLVED-by-reading: cite defensible, no change); confirm `test_agents_md_budget` green after the AGENTS edit.

---

## Self-Review

**Spec coverage — the 11 pre-registered gates, each mapped to a test (corrected after plan-review r1):** gate 1 VW&D worked examples → T5 `test_evalue_worked_examples` (RR=3.9→7.263 AND RR=2→3.414 — two non-trivial pairs + RR=1 + CI-crosses-1) ; gate 2 `ipw.R` oracle (a `get.extrema`, b `extrema.os`, c AIPW) via the injection seam → T1 + T2 ; gate 3 γ=1 collapse → T4 `test_msm_gamma1_collapses_to_point` ; gate 4 planted-confounder coverage → T6 `test_planted_confounder_coverage` (CI band `msm_ci_lo/hi`, BOUNDED binary-U with known Λ, at the TRUE Γ, ≥0.90/≥500, `@slow`) ; gate 5 fixture precondition → T6 `test_planted_confounder_fixture_precondition` (binary U, prevalence off-boundary, naive bias present) ; gate 6 agreement+overlap NO-GO ALL THREE SIDES → T5 `test_agreement_predicate_both_sides` (both `agreement_ok` branches, via the factored `_agreement_ok`) + `test_nogo_integration_all_sides` (`msm_go=True` side + `overlap_ok=False` side) ; gate 7 determinism BOTH sides → T6 `test_result_determinism_seed` (same seed identical; different seed → CI moves, point estimates stable) ; gate 8 binary guard → T5 `test_evalue_requires_binary_outcome` ; gate 9 AIPW → T6 `test_aipw_reduces_to_hajek_when_or_constant` (constant OR cancels) ; gate 10 purity → T6 `test_purity_and_binary_fold` ; gate 11 harness wiring → T8. Surface/`declare_inputs`/stale-memo → T7/T8 ; scope/owed follow-up → T9 ; workflow (one commit, reviews) → Global Constraints + T9 Step 4. **No gap, and no gate narrowed below the spec's pinned bar.**

**Placeholder scan:** the only unfilled literals are the `<R…>` oracle TARGETS in `ipw_oracle.json` — produced by the pinned Docker R run (Task 1) and committed with provenance, NEVER hand- or port-filled; every algorithm/test body is real code. No TBD/handwave.

**Type consistency:** `_get_extrema`/`_extrema_md`/`_extrema_os_att` all return `tuple[float,float]` (min,max); `_bootsens_att` returns `(ci_lo,ci_hi)`; `e_value` returns the 5-tuple `(e_point,e_ci,rr,rr_ci,matched_rr)` consumed by `sensitivity_analysis` via `dataclasses.replace`; `_agreement_ok(matched_att, ipw_att, matched_se) -> bool` is used by `marginal_sensitivity` and unit-tested directly; `SensitivityResult` field names match the spec and the constructors. `fit_propensity(X, Z_int, seed=)`/`estimate_att(Y, Z, ps, X)`/`propensity_match(ps, Z_int, target=)` signatures match `matching.py` as read. `gamma` is log-OR throughout; `estimand_all` bool threads T1→T2.
