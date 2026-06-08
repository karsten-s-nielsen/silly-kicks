# TF-17 PR-C: xCross Causal Validation Harness — Implementation Plan (rev. 2, post-review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the paper-faithful causal validation harness for xCrossAttempt — a private numpy/sklearn propensity-matching port + crosser-anchored opportunity builder + thin driver + known-truth tests + ADR-015 — answering "is GK position a real backdoor confounder of cross→shot?"

**Architecture:** Two pure modules under a private `silly_kicks/_causal/` package (`matching.py` estimator port, `opportunities.py` spell-based opportunity-row builder), a driver `scripts/validate_xcross_causal.py` split into a pure `analyze()` + an I/O `run()`, regular-suite known-truth unit tests + an e2e that drives the integration seam. No public API, no new dependency (numpy + sklearn only). The harness output is a *reported* research artifact, never a ship/CI gate.

**Tech Stack:** Python, numpy, scikit-learn (`LogisticRegression`, `NearestNeighbors`), pandas, pytest. Windows host: `uv run pyright`, `uv run ruff check` / `ruff format --check`, `uv run python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-06-tf17-xcross-pr-c-design.md`

**Revision note (rev. 2):** folds in the d32 round-1 review — H1–H4, M1–M7, L1–L3 (see git history).

**Revision note (rev. 3):** folds in the d32 round-2 review — **R2-H1** (Task-5 fixture geometry was inverted: a
team's GK at *low* x means it *attacks high* x, so the low-x balls were never "advanced" → zero spells → the
fixtures couldn't go green; GK x-positions are swapped so team 5 attacks x=0 and the low-x wide balls are in
domain), **R2-H2** (the e2e now genuinely chains `build_opportunities` → `analyze` → `_write` on real frames +
monkeypatches `iter_matches` so `run()` executes — it no longer fakes opportunity rows in numpy), **R2-H3**
(the rev.2 variable spell-window `Z` reintroduced spell-length confounding → switched to a **fixed post-entry
exposure window** `T`; the spell machine is now dedup-only), **R2-M1** (outcome anchored strictly *after* the
cross for treated → no reverse-direction leakage), **R2-M2** (`PAPER_CONFOUNDERS` imported from
`_xcross_attempt._CONFOUNDERS`, not re-literal'd), **R2-M3** (report `base_nan_fraction` too), **R2-M4** (`run()`
e2e coverage + restore `test_one_frame_domain_blip`), **R2-L1** (drop the dead score-None guard), **R2-L2**
(`same_id` for the scalar team continuation), **R2-L3** (note the permutation-null is slightly conservative).

---

## ⚠️ Commit policy for this plan (owner instruction)

**No commits until TF-17 is completely done (PR-B + PR-C).** The standard TDD "Commit" step is replaced
throughout by a **Stage** step (`git add <paths>`). All staged work rides the single end-of-feature commit
(Task 10), whose structure (one combined 4.16.0 vs two releases) is an explicit ship-time decision with the
owner. Run `/final-review` before that single commit.

## Reuse map (DRY — do NOT re-derive)

| Need | Reuse from |
|---|---|
| Carrier inference | `_ball_carrier.infer_ball_carrier(frames, **carrier_params)` |
| Possession team per frame | `_ball_carrier.derive_team_in_possession(frames, carrier)` |
| Defended goal_x per (game,period,team) | `_xcross_attempt._build_goal_map(frames)` |
| Wide-area domain predicate | `_xcross_attempt._in_wide_area(bx, by, goal_x, advance_m)` + `_ADVANCE_M` |
| Confounder row `X` (incl GK block) | `_xcross_attempt.extract_xcross_features(grp, *, gk_team_id, goal_x, carrier_player_id, score_differential=…)` |
| **The 7 paper confounders (single source — R2-M2)** | `_xcross_attempt._CONFOUNDERS` (verified line 45 = the 7; `XCROSS_FEATURE_NAMES_FAITHFUL = _BALL_FEATURES + _CONFOUNDERS + XCROSS_GK_BLOCK`) — import as `PAPER_CONFOUNDERS`, do **not** re-literal |
| Feature/GK column names | `_xcross_attempt.XCROSS_FEATURE_NAMES_FAITHFUL` (GK = the `gk_*` tail) |
| Dtype-safe scalar team compare | `tracking._id_compat.same_id(a, b)` (spell continuation — R2-L2) |
| **score_differential lookup (M1)** | `_ghost_gk._build_score_lookup(actions, home_team_id)` + `_xcross_attempt._has_results(actions)` |
| Dtype-safe action↔frame id compare | `tracking._id_compat.ids_match(series, scalar)` (ADR-019) |
| Shot/cross action type ids | `silly_kicks.spadl.config.actiontype_id[...]` |
| Pining corpus loader | `scripts/_loader_pining.py` (confirm the iterator name/signature — Open item) |
| Model domain constants | bundled `_xcross_weights/default/metadata.json` (`cross_types`, `carrier_params`) |

---

## Task 0: Package + test scaffolding

**Files:** Create `silly_kicks/_causal/__init__.py`, `tests/causal/__init__.py`

- [ ] **Step 1: Private package init (exports nothing public)**

```python
# silly_kicks/_causal/__init__.py
"""Private causal-validation port (ADR-015). Pure numpy/sklearn matching estimators +
opportunity-row builder for the TF-17 xCross causal harness. NOT imported by
silly_kicks/__init__; promote to a public silly_kicks/causal/ only when a 2nd consumer
(TF-19) lands. No public API is exported here by design."""
```

- [ ] **Step 2: Test package init** — create empty `tests/causal/__init__.py`.
- [ ] **Step 3: Verify** — `uv run python -c "import silly_kicks._causal"` → exit 0.
- [ ] **Step 4: Stage** — `git add silly_kicks/_causal/__init__.py tests/causal/__init__.py`

---

## Task 1: `matching.py` — `fit_propensity` (standardized) + `propensity_match`

**Files:** Create `silly_kicks/_causal/matching.py`; Test `tests/causal/test_matching.py`

- [ ] **Step 1: Failing tests**

```python
# tests/causal/test_matching.py
import numpy as np

from silly_kicks._causal import matching as M


def _toy(n=200, seed=0):
    """Confounder x drives both treatment and outcome; known +0.5 effect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    p = 1 / (1 + np.exp(-0.8 * x[:, 0]))
    z = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 2.0 * x[:, 0] + 0.5 * z + rng.normal(scale=0.1, size=n)
    return x, z, y


def test_fit_propensity_in_unit_interval_and_standardizes():
    # Multi-scale columns (metres vs tiny angles): the fit must standardize internally.
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(50, 10, 300), rng.normal(0, 0.01, 300)])
    z = (rng.uniform(size=300) < 0.5).astype(int)
    ps, coefs = M.fit_propensity(x, z, seed=42)
    assert ps.shape == (300,)
    assert np.all((ps > 0) & (ps < 1))
    assert np.isfinite(coefs).all()


def test_fit_propensity_deterministic():
    x, z, _ = _toy()
    a, _ = M.fit_propensity(x, z, seed=42)
    b, _ = M.fit_propensity(x, z, seed=42)
    np.testing.assert_array_equal(a, b)


def test_propensity_match_with_replacement_reuses_controls():
    ps = np.array([0.9, 0.9, 0.9, 0.1, 0.88])
    z = np.array([1, 1, 1, 0, 0])
    matched = M.propensity_match(ps, z, target="att")
    treated = np.where(z == 1)[0]
    assert set(matched.keys()) == set(treated.tolist())
    used = list(matched.values())
    assert all(z[c] == 0 for c in used)
    assert len(used) != len(set(used))  # a control reused (with replacement)


def test_no_caliper_keeps_all_treated():
    x, z, _ = _toy()
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    assert set(matched.keys()) == set(np.where(z == 1)[0].tolist())
```

- [ ] **Step 2: Run → FAIL** — `uv run python -m pytest tests/causal/test_matching.py -q` (module missing).

- [ ] **Step 3: Implement**

```python
# silly_kicks/_causal/matching.py
"""Pure propensity-score matching estimators (ADR-015), numpy + sklearn, no R, no new dep.

1:1 nearest-neighbor matching on the propensity score, WITH REPLACEMENT, ties allowed,
NO caliper (paper-faithful: Cao et al. 2025, arXiv:2505.11841). Variance via the
Abadie-Imbens (2006) matching estimator (Imbens & Rubin 2015, Ch. 19) -- naive/bootstrap
SEs are biased under matching-with-replacement. Determinism via explicit seeds.

See NOTICE for the citation + the state-vs-sender faithfulness caveat; ADR-015 for the two
named approximations (J=1 within-group sigma^2; estimated-PS variance is conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

Target = Literal["att", "atnt"]

# Pre-registered "non-trivial" GK-ablation shift floor + the placebo band percentile (named +
# asserted so they cannot silently drift). Reported, never a CI gate.
GK_ABLATION_MIN_SHIFT = 0.01
PLACEBO_BAND_PERCENTILE = 95.0


@dataclass(frozen=True)
class CausalEstimate:
    estimate: float
    se: float
    balance: pd.DataFrame   # covariate / smd_pre / smd_post
    n_focal: int
    matched: dict[int, int]


def fit_propensity(X: np.ndarray, Z: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Logistic propensity e(x)=P(Z=1|X) on STANDARDIZED covariates (M6: raw confounders are
    multi-scale -- metres/radians/counts -- and lbfgs is not scale-robust). Returns (scores in
    (0,1), coefficients on the standardized scale).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> Z = np.array([0, 0, 1, 1])
    >>> ps, _ = fit_propensity(X, Z, seed=0)
    >>> bool(np.all((ps > 0) & (ps < 1)))
    True
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=int)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    clf = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(Xs, Z)
    ps = np.clip(clf.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    return ps, clf.coef_.ravel()


def propensity_match(ps: np.ndarray, Z: np.ndarray, *, target: Target) -> dict[int, int]:
    """1:1 NN match on the propensity score, with replacement, ties allowed, no caliper.

    `target="att"`: each TREATED -> nearest CONTROL ({treated_idx: control_idx}).
    `target="atnt"`: each CONTROL -> nearest TREATED. No unit dropped. NN tie-break is
    index-order deterministic, so no seed is needed (L1).
    """
    ps = np.asarray(ps, dtype=float).reshape(-1, 1)
    Z = np.asarray(Z, dtype=int)
    treated, control = np.where(Z == 1)[0], np.where(Z == 0)[0]
    focal, pool = (treated, control) if target == "att" else (control, treated)
    if len(pool) == 0 or len(focal) == 0:
        return {}
    nn = NearestNeighbors(n_neighbors=1).fit(ps[pool])
    _, idx = nn.kneighbors(ps[focal])
    return {int(focal[i]): int(pool[idx[i, 0]]) for i in range(len(focal))}
```

- [ ] **Step 4: Run → PASS** — `uv run python -m pytest tests/causal/test_matching.py -q`
- [ ] **Step 5: Stage** — `git add silly_kicks/_causal/matching.py tests/causal/test_matching.py`

---

## Task 2: `smd_balance` + `abadie_imbens_se` (leaf functions, independently green — L2)

**Files:** Modify `silly_kicks/_causal/matching.py`; Test `tests/causal/test_matching.py`

- [ ] **Step 1: Failing tests**

```python
# append to tests/causal/test_matching.py
def test_smd_balance_table_shape():
    x, z, _ = _toy(n=400, seed=5)
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    bal = M.smd_balance(x, z, matched, target="att")
    assert list(bal.columns) == ["covariate", "smd_pre", "smd_post"]
    assert bal.shape[0] == x.shape[1]
    assert abs(bal["smd_post"].iloc[0]) < abs(bal["smd_pre"].iloc[0])  # matching improves balance


def test_abadie_imbens_se_positive_finite():
    x, z, y = _toy(n=600, seed=11)
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    se = M.abadie_imbens_se(y, z, matched, ps, target="att")
    assert se > 0 and np.isfinite(se)


def test_ai_se_differs_from_naive_under_reuse():
    rng = np.random.default_rng(0)
    n_t, n_c = 50, 5  # few controls -> heavy reuse -> AI correction strictly positive
    ps = np.concatenate([np.full(n_t, 0.7), np.linspace(0.65, 0.75, n_c)])
    z = np.concatenate([np.ones(n_t, int), np.zeros(n_c, int)])
    y = np.concatenate([rng.normal(1.0, 1.0, n_t), rng.normal(0.0, 1.0, n_c)])
    matched = M.propensity_match(ps, z, target="att")
    ai = M.abadie_imbens_se(y, z, matched, ps, target="att")
    focal = np.array(sorted(matched.keys()))
    comp = np.array([matched[int(i)] for i in focal])
    naive = (y[focal] - y[comp]).std(ddof=1) / np.sqrt(len(focal))
    assert ai > naive
```

- [ ] **Step 2: Run → FAIL** — `uv run python -m pytest tests/causal/test_matching.py -k "smd or abadie or ai_se" -q`

- [ ] **Step 3: Implement**

```python
# add to silly_kicks/_causal/matching.py

def smd_balance(X: np.ndarray, Z: np.ndarray, matched: dict[int, int], *, target: Target) -> pd.DataFrame:
    """Standardized mean differences per covariate, before vs after matching. SMD =
    (mean_focal - mean_comp) / pooled_sd. Post-match uses the matched comparison set (with
    with-replacement multiplicity). Lower |SMD| == better balance."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=int)
    focal_val, comp_val = (1, 0) if target == "att" else (0, 1)
    focal = np.where(Z == focal_val)[0]
    comp = np.where(Z == comp_val)[0]
    matched_comp = np.array([matched[int(i)] for i in focal], dtype=int)
    rows = []
    for k in range(X.shape[1]):
        xf, xc_all, xc_m = X[focal, k], X[comp, k], X[matched_comp, k]
        sd = np.sqrt((xf.var(ddof=1) + xc_all.var(ddof=1)) / 2.0) or 1.0
        rows.append({"covariate": k, "smd_pre": (xf.mean() - xc_all.mean()) / sd,
                     "smd_post": (xf.mean() - xc_m.mean()) / sd})
    return pd.DataFrame(rows, columns=["covariate", "smd_pre", "smd_post"])


def _within_group_sigma2(Y: np.ndarray, Z: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Conditional outcome variance sigma^2(X_i) via the within-treatment-group nearest neighbor
    on the propensity score (J=1): sigma2_i = (1/2)(Y_i - Y_h(i))^2 (ADR-015)."""
    ps = np.asarray(ps, dtype=float).reshape(-1, 1)
    Y = np.asarray(Y, dtype=float)
    sigma2 = np.zeros(len(Y))
    for val in (0, 1):
        idx = np.where(Z == val)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=2).fit(ps[idx])
        _, nbr = nn.kneighbors(ps[idx])
        for r, i in enumerate(idx):
            h = idx[nbr[r, 1]]  # column 0 is self
            sigma2[i] = 0.5 * (Y[i] - Y[h]) ** 2
    return sigma2


def abadie_imbens_se(Y, Z, matched: dict[int, int], ps, *, target: Target) -> float:
    """Abadie-Imbens (2006) matching SE for ATT/ATNT, 1:1 with replacement (Imbens & Rubin
    2015, Ch. 19): V = (between + reuse) / N1^2, where between = sum (tau_i - tau_bar)^2 and
    reuse = sum_j K_j(K_j-1) sigma2_j over comparison units used K_j times. Deterministic."""
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=int)
    focal = np.array(sorted(matched.keys()), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    n_focal = len(focal)
    if n_focal == 0:
        return float("nan")
    tau_i = (Y[focal] - Y[comp]) if target == "att" else (Y[comp] - Y[focal])
    between = float(np.sum((tau_i - tau_i.mean()) ** 2))
    sigma2 = _within_group_sigma2(Y, Z, ps)
    counts = np.bincount(comp, minlength=len(Y))  # K_j
    reuse = float(np.sum(counts * np.maximum(counts - 1, 0) * sigma2))
    return float(np.sqrt(max((between + reuse) / (n_focal**2), 0.0)))
```

- [ ] **Step 4: Run → PASS** — `uv run python -m pytest tests/causal/test_matching.py -k "smd or abadie or ai_se" -q`
- [ ] **Step 5: Stage** — `git add silly_kicks/_causal/matching.py tests/causal/test_matching.py`

---

## Task 3: `estimate_att` / `estimate_atnt` (deterministic, no seed — L1/L2)

**Files:** Modify `silly_kicks/_causal/matching.py`; Test `tests/causal/test_matching.py`

- [ ] **Step 1: Failing tests**

```python
# append to tests/causal/test_matching.py
def test_recovers_known_ate():
    x, z, y = _toy(n=1500, seed=3)
    ps, _ = M.fit_propensity(x, z, seed=42)
    att = M.estimate_att(y, z, ps, x)
    assert abs(att.estimate - 0.5) < 0.15  # true effect 0.5
    assert att.se > 0 and np.isfinite(att.se)
    assert att.n_focal == int(z.sum())


def test_atnt_runs():
    x, z, y = _toy(n=800, seed=7)
    ps, _ = M.fit_propensity(x, z, seed=42)
    atnt = M.estimate_atnt(y, z, ps, x)
    assert np.isfinite(atnt.estimate) and atnt.se > 0


def test_estimate_deterministic():
    x, z, y = _toy(n=500, seed=9)
    ps, _ = M.fit_propensity(x, z, seed=42)
    assert M.estimate_att(y, z, ps, x).estimate == M.estimate_att(y, z, ps, x).estimate
```

- [ ] **Step 2: Run → FAIL** — `uv run python -m pytest tests/causal/test_matching.py -k "known_ate or atnt or estimate_deterministic" -q`

- [ ] **Step 3: Implement**

```python
# add to silly_kicks/_causal/matching.py

def _estimate(Y, Z, ps, X, *, target: Target) -> CausalEstimate:
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=int)
    matched = propensity_match(ps, Z, target=target)
    focal = np.array(sorted(matched.keys()), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    if len(focal) == 0:
        return CausalEstimate(float("nan"), float("nan"),
                              pd.DataFrame(columns=["covariate", "smd_pre", "smd_post"]), 0, {})
    tau_i = (Y[focal] - Y[comp]) if target == "att" else (Y[comp] - Y[focal])
    se = abadie_imbens_se(Y, Z, matched, ps, target=target)
    balance = smd_balance(X, Z, matched, target=target)
    return CausalEstimate(float(tau_i.mean()), se, balance, int(len(focal)), matched)


def estimate_att(Y, Z, ps, X) -> CausalEstimate:
    """ATT = mean over treated of (Y_treated - Y_matched_control). With-replacement matching;
    Abadie-Imbens SE. Deterministic given ps."""
    return _estimate(Y, Z, ps, X, target="att")


def estimate_atnt(Y, Z, ps, X) -> CausalEstimate:
    """ATNT = effect on the untreated = mean over controls of (Y_matched_treated - Y_control)."""
    return _estimate(Y, Z, ps, X, target="atnt")
```

- [ ] **Step 4: Run → PASS** — `uv run python -m pytest tests/causal/test_matching.py -q` (all matching tests)
- [ ] **Step 5: Stage** — `git add silly_kicks/_causal/matching.py tests/causal/test_matching.py`

---

## Task 4: `placebo_shift` — row-permuted-GK null band (H3/L3)

**Files:** Modify `silly_kicks/_causal/matching.py`; Test `tests/causal/test_matching.py`

- [ ] **Step 1: Failing tests**

```python
# append to tests/causal/test_matching.py
def test_placebo_permuted_gk_zero_shift():
    # GK block with NO real association -> permuted-row null centered ~0.
    rng = np.random.default_rng(13)
    n = 1200
    x_base = rng.normal(size=(n, 3))
    x_gk = rng.normal(size=(n, 4))           # independent of (z, y)
    z = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.6 * x_base[:, 0]))).astype(int)
    y = 1.0 + x_base[:, 0] + 0.5 * z + rng.normal(scale=0.1, size=n)
    out = M.placebo_shift(x_base, x_gk, y, z, n_seeds=20, rng_seed=0)
    assert out["shifts"].shape == (20,)
    assert abs(np.median(out["shifts"])) < 0.1
    assert np.isfinite(out["band_p95"])


def test_placebo_deterministic():
    rng = np.random.default_rng(2)
    n = 400
    x_base, x_gk = rng.normal(size=(n, 3)), rng.normal(size=(n, 4))
    z = (rng.uniform(size=n) < 0.5).astype(int)
    y = rng.normal(size=n)
    a = M.placebo_shift(x_base, x_gk, y, z, n_seeds=10, rng_seed=7)["shifts"]
    b = M.placebo_shift(x_base, x_gk, y, z, n_seeds=10, rng_seed=7)["shifts"]
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run → FAIL** — `uv run python -m pytest tests/causal/test_matching.py -k placebo -q`

- [ ] **Step 3: Implement**

```python
# add to silly_kicks/_causal/matching.py

def placebo_shift(X_base, X_gk, Y, Z, *, n_seeds: int, rng_seed: int) -> dict:
    """Null distribution of the ATT shift from adding a ROW-PERMUTED GK block (H3). Permuting
    the rows of X_gk preserves its marginals + within-block correlation and destroys only its
    alignment with (Z, Y) -- isolating "GK carries Z/Y signal" from "GK columns aren't Gaussian".
    The real GK shift is "real" only if it clears band_p95. Deterministic given rng_seed.

    Returns {"shifts": (n_seeds,), "band_p95": float, "base_att": float}.
    """
    X_base = np.asarray(X_base, dtype=float)
    X_gk = np.asarray(X_gk, dtype=float)
    Y, Z = np.asarray(Y, dtype=float), np.asarray(Z, dtype=int)
    n = len(Z)
    ps_base, _ = fit_propensity(X_base, Z, seed=rng_seed)
    base = estimate_att(Y, Z, ps_base, X_base).estimate
    rng = np.random.default_rng(rng_seed)
    shifts = np.empty(n_seeds)
    for s in range(n_seeds):
        perm = rng.permutation(n)
        x_full = np.hstack([X_base, X_gk[perm]])
        ps, _ = fit_propensity(x_full, Z, seed=rng_seed + 1 + s)
        shifts[s] = estimate_att(Y, Z, ps, x_full).estimate - base
    return {"shifts": shifts, "band_p95": float(np.percentile(np.abs(shifts), PLACEBO_BAND_PERCENTILE)),
            "base_att": float(base)}
```

- [ ] **Step 4: Run → PASS** — `uv run python -m pytest tests/causal/test_matching.py -q`
- [ ] **Step 5: pyright + ruff** — `uv run pyright silly_kicks/_causal/ ; uv run ruff check silly_kicks/_causal/ tests/causal/ ; uv run ruff format --check silly_kicks/_causal/ tests/causal/`
- [ ] **Step 6: Stage** — `git add silly_kicks/_causal/matching.py tests/causal/test_matching.py`

---

## Task 5: `opportunities.py` — spell state-machine builder (H1, M1, M3) + real dedup tests (H4)

**Files:** Create `silly_kicks/_causal/opportunities.py`; Test `tests/causal/test_opportunities.py`

The builder is a per-(game,period) spell state-machine. It tracks spell **entry** and **end**; the end serves
both as the dedup boundary (one row per continuous wide-area spell; `MAX_SPELL_SECONDS` caps a never-closing
run) **and as the ceiling on the treatment window (R3-M1)**: `Z=1` iff a possessing-team cross falls in
`(entry, min(entry + EXPOSURE_WINDOW_SECONDS, spell_end)]`. The fixed `T` cap keeps Z-exposure bounded (no
spell-length confounding — and since `Y`'s window is already fixed, clamping the Z-window to `spell_end`
introduces no new duration→Y path), while the `spell_end` cap prevents misattributing a cross from a *later*
re-possession phase to this opportunity. `Y` = a possessing-team shot in `(anchor, anchor+W]`
(`OUTCOME_WINDOW_SECONDS`), `anchor = t_cross` for treated (strictly post-cross — R2-M1) / `entry` for controls;
`Y` is **not** possession-clamped (a documented modeling choice — treated/control Y-windows are time-shifted).
Features are anchored at entry; `score_differential` is wired (M1); `X` = the 7 paper confounders (imported from
`_CONFOUNDERS` — R2-M2) + 6 GK columns, no ball features (M3).

- [ ] **Step 1: Create the shared fixtures module (R3-L2), then write the failing tests** (each dedup branch
  is a distinct, non-trivial fixture — H4)

```python
# tests/causal/_fixtures.py
"""Shared geometry-correct spell-fixture builders for the causal tests (single source -- R3-L2).

GEOMETRY (R2-H1): team 5 ATTACKS x=0 -> team-5 GK at HIGH x (101, defends 105); team-6 GK at LOW x
(4, defends 0). _build_goal_map then yields goal_x=0 for team-5 possession, so a low-x wide ball
(x<=35 advanced, y<14 or y>54) is in the advanced wide area.
"""
import pandas as pd

META = {"cross_types": ["cross"], "carrier_params": {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}}
WIDE = (12.0, 6.0)       # advanced (x<=35 from goal 0) + wide (y=6<14)
CENTRAL = (12.0, 34.0)   # advanced but central (14<y<54) -> NOT wide area
NEAR0 = (8.0, 6.0)


def frow(pid, team, gk, x, y, t, *, is_ball=False, period=1):
    return dict(game_id=1, period_id=period, frame_id=round(t * 25), time_seconds=round(t, 3),
                frame_rate=25.0, player_id=pid, team_id=team, is_ball=is_ball, is_goalkeeper=gk,
                x=float(x), y=float(y), z=0.0, speed=2.0, vx=2.0, vy=0.0, speed_source="native",
                ball_state="alive", team_attacking_direction="ltr", source_provider="test")


def frames(possession_by_time, ball_xy_by_time, *, period=1):
    rows = []
    for t, pt in possession_by_time.items():
        bx, by = ball_xy_by_time[t]
        rows.append(frow(10 if pt == 5 else 20, pt, False, bx, by, t, period=period))  # carrier on ball
        rows += [frow(11, 5, False, 18.0, 40.0, t, period=period), frow(12, 5, False, 15.0, 30.0, t, period=period),
                 frow(21, 6, False, 8.0, 40.0, t, period=period), frow(22, 6, False, 10.0, 30.0, t, period=period),
                 frow(1, 5, True, 101.0, 34.0, t, period=period),   # team-5 GK high x -> attacks 0
                 frow(2, 6, True, 4.0, 34.0, t, period=period)]      # team-6 GK low x -> defends 0
        rows.append(frow(pd.NA, pd.NA, False, bx, by, t, is_ball=True, period=period))
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


def spell(team=5, t0=10.0, t1=10.4, dt=0.2, ball=WIDE, *, period=1):
    """A continuous in-domain possession spell for `team` over [t0, t1] (inclusive)."""
    poss, xy, t = {}, {}, t0
    while t <= t1 + 1e-9:
        key = round(t, 3)
        poss[key], xy[key] = team, ball
        t += dt
    return frames(poss, xy, period=period)


def actions(rows):
    return pd.DataFrame(rows, columns=["game_id", "action_id", "period_id", "team_id",
                                       "time_seconds", "type_id", "result_id",
                                       "start_x", "start_y", "end_x", "end_y"])
```

```python
# tests/causal/test_opportunities.py
import numpy as np
import pandas as pd

from silly_kicks._causal import opportunities as O
from silly_kicks.spadl import config as _c
from tests.causal._fixtures import CENTRAL, META, NEAR0, WIDE, actions, frames, spell


def test_single_spell_one_row():
    f = frames({10.0: 5, 10.2: 5, 10.4: 5}, {10.0: WIDE, 10.2: WIDE, 10.4: NEAR0})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 1 and opp["possessing_team"].iloc[0] == 5
    assert opp["spell_duration_seconds"].iloc[0] >= 0


def test_reentry_after_turnover_is_new_spell():
    # team 6's low-x ball is NOT advanced toward team-6's attacked goal (105) -> out-of-domain;
    # so the turnover closes team 5's spell and team 5 re-entering opens a second.
    f = frames({10.0: 5, 10.2: 5, 10.4: 6, 10.6: 5, 10.8: 5}, {t: WIDE for t in (10.0, 10.2, 10.4, 10.6, 10.8)})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert (opp["possessing_team"] == 5).sum() == 2


def test_reentry_after_domain_exit_is_new_spell():
    f = frames({10.0: 5, 10.2: 5, 10.4: 5, 10.6: 5}, {10.0: WIDE, 10.2: CENTRAL, 10.4: WIDE, 10.6: WIDE})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 2


def test_one_frame_domain_blip():  # R2-M4: single out-of-domain frame closes+reopens
    f = frames({10.0: 5, 10.2: 5, 10.4: 5}, {10.0: WIDE, 10.2: CENTRAL, 10.4: WIDE})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 2


def test_period_boundary_splits_spells():
    f1 = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE}, period=1)
    f2 = frames({1.0: 5, 1.2: 5}, {1.0: WIDE, 1.2: WIDE}, period=2)  # period-relative time reset
    opp = O.build_opportunities(pd.concat([f1, f2], ignore_index=True), actions([]),
                                home_team_id=5, model_metadata=META)
    assert len(opp) == 2  # never merged across the period boundary


def test_treatment_capped_by_window_T():  # R3-M1/R2-H3: the fixed T cap, on a LONG continuous spell
    T = O.EXPOSURE_WINDOW_SECONDS
    f = spell(5, 10.0, 10.0 + T + 4.0)  # spell extends well past entry+T (and < MAX_SPELL_SECONDS)
    cross = _c.actiontype_id["cross"]
    a_in = actions([[1, 0, 1, 5, 10.0 + T - 1.0, cross, 1, 20, 8, 14, 6]])
    assert int(O.build_opportunities(f, a_in, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 1
    a_out = actions([[1, 0, 1, 5, 10.0 + T + 1.0, cross, 1, 20, 8, 14, 6]])  # past T, still within the spell
    assert int(O.build_opportunities(f, a_out, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0


def test_treatment_capped_by_possession_end():  # R3-M1: clamp to spell_end kills cross-phase misattribution
    f = spell(5, 10.0, 10.4)  # short spell ends ~10.4, well within T
    cross = _c.actiontype_id["cross"]
    a_after = actions([[1, 0, 1, 5, 12.0, cross, 1, 20, 8, 14, 6]])  # cross AFTER possession ended
    assert int(O.build_opportunities(f, a_after, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0
    a_in = actions([[1, 0, 1, 5, 10.3, cross, 1, 20, 8, 14, 6]])  # cross within the spell
    assert int(O.build_opportunities(f, a_in, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 1


def test_opponent_cross_is_negative():
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 6, 10.3, _c.actiontype_id["cross"], 1, 20, 8, 14, 6]])
    assert int(O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0


def test_outcome_strictly_post_cross():  # R2-M1
    f = spell(5, 10.0, 11.5)  # spell spans the cross at 11.0
    cross, shot = _c.actiontype_id["cross"], _c.actiontype_id["shot"]
    pre = actions([[1, 0, 1, 5, 11.0, cross, 1, 20, 8, 14, 6],
                   [1, 1, 1, 5, 10.5, shot, 1, 14, 6, 0, 34]])  # shot precedes the cross
    o1 = O.build_opportunities(f, pre, home_team_id=5, model_metadata=META)
    assert int(o1["Z"].iloc[0]) == 1 and int(o1["Y"].iloc[0]) == 0
    post = actions([[1, 0, 1, 5, 11.0, cross, 1, 20, 8, 14, 6],
                    [1, 1, 1, 5, 11.5, shot, 1, 14, 6, 0, 34]])  # shot after the cross, within W
    assert int(O.build_opportunities(f, post, home_team_id=5, model_metadata=META)["Y"].iloc[0]) == 1


def test_control_outcome_from_entry():
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 5, 11.0, _c.actiontype_id["shot"], 1, 14, 6, 0, 34]])  # no cross -> control
    opp = O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)
    assert int(opp["Z"].iloc[0]) == 0 and int(opp["Y"].iloc[0]) == 1  # control Y from entry over W


def test_score_differential_populated():  # M1
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 5, 1.0, _c.actiontype_id["shot"], _c.result_id["success"], 14, 6, 0, 34]])
    opp = O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)  # team 5 (home) scored at t=1
    assert not np.isnan(opp["score_differential"].iloc[0])


def test_confounder_set_is_seven_no_ball_features():  # M3 + R2-M2
    from silly_kicks.tracking._xcross_attempt import _CONFOUNDERS
    assert O.PAPER_CONFOUNDERS == list(_CONFOUNDERS)  # single source of truth, not re-literal'd
    f = spell(5, 10.0, 10.4)
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    for c in O.PAPER_CONFOUNDERS + O.GK_BLOCK:
        assert c in opp.columns
    for ball in ("ball_r", "ball_theta", "ball_speed"):
        assert ball not in O.PAPER_CONFOUNDERS


def test_carrier_handoff_midspell_stays_one_row():  # H4: genuine carrier flip, one continuous spell
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    f = frames({10.0: 5, 10.2: 5}, {10.0: (12.0, 6.0), 10.2: (30.0, 6.0)})  # ball stays advanced+wide
    m2 = f["time_seconds"] == 10.2
    f.loc[m2 & (f["player_id"] == 10), ["x", "y"]] = [12.0, 6.0]   # p10 left behind
    f.loc[m2 & (f["player_id"] == 11), ["x", "y"]] = [30.0, 6.0]   # p11 now on the ball
    car = infer_ball_carrier(f, **META["carrier_params"])
    assert car["ball_carrier_player_id"].dropna().nunique() >= 2  # fixture genuinely flips the carrier
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 1


def test_carrier_coverage_reported():
    f = spell(5, 10.0, 10.2)
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert "carrier_resolved" in opp.columns
```

- [ ] **Step 2: Run → FAIL** — `uv run python -m pytest tests/causal/test_opportunities.py -q`

- [ ] **Step 3: Implement**

```python
# silly_kicks/_causal/opportunities.py
"""Crosser-anchored opportunity-row builder for the xCross causal harness (ADR-015).

A per-(game,period) spell state-machine: one row per continuous wide-area possession-spell, anchored
at entry (the paper's sender-level unit). The spell end serves as the dedup boundary AND the ceiling
on the treatment window (R3-M1):
  Z = 1 iff a possessing-team cross occurs in (entry, min(entry + EXPOSURE_WINDOW_SECONDS, spell_end)];
      the fixed T cap keeps Z-exposure bounded (no spell-length confounding -- Y's window is already
      fixed, so clamping to spell_end adds no duration->Y path), and the spell_end cap prevents
      misattributing a cross from a LATER re-possession phase to this opportunity.
  Y = a possessing-team shot in (anchor, anchor + OUTCOME_WINDOW_SECONDS], anchor = t_cross for
      treated (strictly post-cross -> no reverse-direction leakage, R2-M1) else entry for controls;
      Y is NOT possession-clamped (documented modeling choice -- treated/control windows time-shifted).
X = the 7 paper confounders (imported from _xcross_attempt._CONFOUNDERS -- single source, R2-M2) + 6
GK columns; ball-geometry features are excluded (surface-model inputs, not paper confounders). Pure;
no I/O. Reuses the shipped xCross domain/carrier/feature helpers so the matched corpus is the model's
training domain by construction. Dedup R2-M1: a new spell starts only on a possession break or a
wide-area domain exit; a mid-spell carrier hand-off stays one row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as _spc
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
from silly_kicks.tracking._id_compat import ids_match, same_id
from silly_kicks.tracking._xcross_attempt import (
    _ADVANCE_M,
    _CONFOUNDERS,
    XCROSS_FEATURE_NAMES_FAITHFUL,
    _build_goal_map,
    _has_results,
    _in_wide_area,
    extract_xcross_features,
)

# X split (M3 + R2-M2): the 7 paper confounders are the SINGLE-SOURCE _CONFOUNDERS (not re-literal'd);
# ball_r/theta/speed are surface-model inputs, NOT paper confounders, and are excluded from the causal X.
PAPER_CONFOUNDERS = list(_CONFOUNDERS)
GK_BLOCK = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL if c.startswith("gk_")]

# Pre-registered windows (named + asserted). The treatment/outcome windows are FIXED (R2-H3), so they
# are NOT a function of spell length; MAX_SPELL_SECONDS only bounds the dedup state-machine.
MAX_SPELL_SECONDS = 30.0       # dedup cap: split a pathological never-closing in-domain run
EXPOSURE_WINDOW_SECONDS = 8.0  # T: Z = cross in (entry, entry+T]
OUTCOME_WINDOW_SECONDS = 6.0   # W: Y = shot in (anchor, anchor+W]

_PROV_COLS = ["game_id", "period_id", "entry_frame_id", "entry_time", "end_time",
              "spell_duration_seconds", "possessing_team", "carrier_resolved"]


def build_opportunities(frames, actions, *, home_team_id, model_metadata, advance_m=_ADVANCE_M) -> pd.DataFrame:
    cross_types = tuple(model_metadata.get("cross_types", ("cross",)))
    carrier_params = dict(model_metadata.get("carrier_params", {}))
    carrier = infer_ball_carrier(frames, **carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)
    score_fn = None
    if _has_results(actions) and home_team_id is not None:
        from silly_kicks.tracking._ghost_gk import _build_score_lookup
        score_fn = _build_score_lookup(actions, home_team_id)

    spells: list[dict] = []
    for (gid, per), g in poss.groupby(["game_id", "period_id"], sort=False):
        g = g.sort_values(["time_seconds", "frame_id"])
        frame_keys = list(dict.fromkeys(zip(g["frame_id"].tolist(), g["time_seconds"].tolist())))
        spell: dict | None = None
        for fid, t in frame_keys:
            grp = g[g["frame_id"] == fid]
            team, goal_x, in_dom = _frame_domain_state(grp, goal_map, gid, per, advance_m)
            continues = (  # R2-L2: same_id is the dtype-safe scalar team compare
                spell is not None and in_dom and same_id(team, spell["team"])
                and (t - spell["entry_time"]) <= MAX_SPELL_SECONDS
            )
            if continues:
                spell["end_time"], spell["end_frame_id"] = float(t), fid
                continue
            if spell is not None:
                spells.append(spell)
                spell = None
            if in_dom:
                spell = dict(gid=gid, per=per, team=team, goal_x=goal_x, grp=grp,
                             entry_frame_id=fid, entry_time=float(t), end_time=float(t), end_frame_id=fid)
        if spell is not None:
            spells.append(spell)

    rows = [_row(sp, actions, cross_types, score_fn, home_team_id) for sp in spells]
    cols = PAPER_CONFOUNDERS + GK_BLOCK + _PROV_COLS + ["Z", "Y"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _frame_domain_state(grp, goal_map, gid, per, advance_m):
    in_poss = grp["team_in_possession"].dropna()
    if in_poss.empty:
        return None, None, False
    poss_team = in_poss.iloc[0]
    ball = grp[grp["is_ball"]]
    if "ball_state" in grp.columns and len(ball) and str(ball["ball_state"].iloc[0]) == "dead":
        return poss_team, None, False
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, poss_team)]
    if not defending:
        return poss_team, None, False
    goal_x = goal_map.get((gid, per, defending[0]))
    if goal_x is None:
        return poss_team, None, False
    bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
    by = float(ball["y"].iloc[0]) if len(ball) else np.nan
    return poss_team, goal_x, _in_wide_area(bx, by, goal_x, advance_m)


def _row(sp, actions, cross_types, score_fn, home_team_id) -> dict:
    grp, gid, per, team, goal_x = sp["grp"], sp["gid"], sp["per"], sp["team"], sp["goal_x"]
    carrier_s = grp["ball_carrier_player_id"].dropna()
    carrier_pid = carrier_s.iloc[0] if not carrier_s.empty else None
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, team)]
    sd = np.nan
    if score_fn is not None:
        # R2-L1: _build_score_lookup returns a _zero callback when no goals -> raw is never None/NaN.
        raw = score_fn(gid, sp["entry_time"])  # home - away
        sd = float(raw) if same_id(team, home_team_id) else -float(raw)
    feats = extract_xcross_features(
        grp, gk_team_id=defending[0], goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
    ).iloc[0]
    row = {c: float(feats[c]) for c in PAPER_CONFOUNDERS + GK_BLOCK}
    entry = sp["entry_time"]
    z, t_cross = _label_treatment(actions, gid, per, team, cross_types, entry, sp["end_time"])
    anchor = t_cross if z else entry
    row.update(
        game_id=gid, period_id=per, entry_frame_id=sp["entry_frame_id"], entry_time=entry,
        end_time=sp["end_time"], spell_duration_seconds=sp["end_time"] - entry,
        possessing_team=team, carrier_resolved=carrier_pid is not None,
        Z=z, Y=_label_outcome(actions, gid, per, team, anchor),
    )
    return row


def _team_period_action_times(actions, gid, per, team, type_names) -> np.ndarray:
    type_ids = {_spc.actiontype_id[n] for n in type_names}
    sel = (  # ids_match: dtype-safe action<->frame team/game id seam (ADR-019)
        ids_match(actions["game_id"], gid)
        & (actions["period_id"] == per)
        & ids_match(actions["team_id"], team)
        & actions["type_id"].isin(type_ids)
    )
    return np.sort(actions.loc[sel, "time_seconds"].to_numpy(dtype=float))


def _label_treatment(actions, gid, per, team, cross_types, entry, end_time) -> tuple[int, float | None]:
    hi = min(entry + EXPOSURE_WINDOW_SECONDS, end_time)  # R3-M1: clamp the Z-window to possession continuity
    ts = _team_period_action_times(actions, gid, per, team, cross_types)
    win = ts[(ts > entry) & (ts <= hi)]
    return (1, float(win[0])) if len(win) else (0, None)


def _label_outcome(actions, gid, per, team, anchor) -> int:
    ts = _team_period_action_times(actions, gid, per, team, ("shot", "shot_freekick", "shot_penalty"))
    return int(bool(((ts > anchor) & (ts <= anchor + OUTCOME_WINDOW_SECONDS)).any()))
```

> **Implementer note:** `home_team_id` is threaded into `_row` for the score sign (`same_id` makes it
> dtype-safe). Verify `_build_score_lookup`'s callback is `(game_id, time) -> home-away` and never
> returns None/NaN (the `_zero` callback when no goals) — if that ever changes, restore a guard.

- [ ] **Step 4: Run → PASS** — `uv run python -m pytest tests/causal/test_opportunities.py -q`
- [ ] **Step 5: pyright + ruff** — `uv run pyright silly_kicks/_causal/ ; uv run ruff check silly_kicks/_causal/ tests/causal/ ; uv run ruff format --check silly_kicks/_causal/ tests/causal/`
- [ ] **Step 6: Stage** — `git add silly_kicks/_causal/opportunities.py tests/causal/_fixtures.py tests/causal/test_opportunities.py`

---

## Task 6: driver — pure `analyze()` + I/O `run()` (M2/M3/M4/M5, H2 seam)

**Files:** Create `scripts/validate_xcross_causal.py`

- [ ] **Step 1: Implement (analyze = pure; run = I/O)**

```python
# scripts/validate_xcross_causal.py
"""Maintainer driver: xCross causal validation harness (TF-17 PR-C, ADR-015).

analyze() is PURE (opportunity frame -> metrics dict) so the e2e can drive it; run() does only
loader I/O + analyze + artifact write. The GK-vs-placebo finding is REPORTED, never asserted: a
null (or an unsupported claim) is a valid result.

Usage:
  python scripts/validate_xcross_causal.py --data-dir <DIR> --out <DIR> \
      [--providers skillcorner,idsse,gradientsports] [--carrier-coverage-min 0.6] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks._causal import matching as M
from silly_kicks._causal.opportunities import GK_BLOCK, PAPER_CONFOUNDERS

_OVERLAP_MIN = 0.5  # min fraction of treated inside the control PS support to claim common support


def _impute_with_indicator(X_gk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Missing-indicator method (M2): a `gk_missing` column (1 if ANY GK col is NaN) + mean
    imputation. The indicator carries the missingness signal so imputation doesn't fabricate the
    confounder-of-interest. Returns (X_gk_imputed_with_indicator, indicator, nan_fraction)."""
    if X_gk.size == 0:
        return X_gk, np.zeros(len(X_gk)), 0.0
    miss = ~np.isfinite(X_gk)
    indicator = miss.any(axis=1).astype(float)
    nan_fraction = float(miss.any(axis=1).mean())
    col_mean = np.where(np.isfinite(np.nanmean(np.where(miss, np.nan, X_gk), axis=0)),
                        np.nanmean(np.where(miss, np.nan, X_gk), axis=0), 0.0)
    imp = X_gk.copy()
    imp[miss] = np.take(col_mean, np.where(miss)[1])
    return np.hstack([imp, indicator.reshape(-1, 1)]), indicator, nan_fraction


def _overlap_fraction(ps: np.ndarray, Z: np.ndarray) -> float:
    """Fraction of treated whose PS lies within [min,max] of the control PS (common-support)."""
    t, c = ps[Z == 1], ps[Z == 0]
    if len(t) == 0 or len(c) == 0:
        return 0.0
    return float(((t >= c.min()) & (t <= c.max())).mean())


def analyze(opp: pd.DataFrame, *, seed: int = 0) -> dict:
    """Pure: opportunity frame -> metrics dict. No I/O."""
    Y, Z = opp["Y"].to_numpy(float), opp["Z"].to_numpy(int)
    # M5 positivity guard -- never silently emit NaN ATT
    if int(Z.sum()) == 0 or int((1 - Z).sum()) == 0:
        return {"status": "no_variation_in_treatment", "n_opportunities": int(len(opp)),
                "n_treated": int(Z.sum())}

    X_base_raw = opp[PAPER_CONFOUNDERS].to_numpy(float)
    base_nan_frac = float((~np.isfinite(X_base_raw)).any(axis=1).mean()) if X_base_raw.size else 0.0  # R2-M3
    X_base = _mean_impute(X_base_raw)
    X_gk_raw = opp[GK_BLOCK].to_numpy(float)
    X_gk, _ind, gk_nan_frac = _impute_with_indicator(X_gk_raw)

    ps_base, _ = M.fit_propensity(X_base, Z, seed=seed)
    att_base = M.estimate_att(Y, Z, ps_base, X_base)
    X_full = np.hstack([X_base, X_gk])
    ps_full, _ = M.fit_propensity(X_full, Z, seed=seed)
    att_full = M.estimate_att(Y, Z, ps_full, X_full)
    atnt_full = M.estimate_atnt(Y, Z, ps_full, X_full)

    gk_shift = att_full.estimate - att_base.estimate
    placebo = M.placebo_shift(X_base, X_gk, Y, Z, n_seeds=200, rng_seed=seed)
    clears = abs(gk_shift) > max(placebo["band_p95"], M.GK_ABLATION_MIN_SHIFT)

    # M4 overlap + SMD-improves gate
    overlap = _overlap_fraction(ps_full, Z)
    smd_improved = bool(att_full.balance["smd_post"].abs().max() < att_full.balance["smd_pre"].abs().max())
    claim_supported = bool(overlap >= _OVERLAP_MIN and smd_improved)

    return {
        "status": "ok",
        "n_opportunities": int(len(opp)),
        "n_treated": int(Z.sum()),
        "base_rate_Y": float(Y.mean()),
        "att_without_gk": {"estimate": att_base.estimate, "se": att_base.se},
        "att_with_gk": {"estimate": att_full.estimate, "se": att_full.se},
        "atnt_with_gk": {"estimate": atnt_full.estimate, "se": atnt_full.se},
        "gk_ablation_shift": gk_shift,
        "placebo_band_p95": placebo["band_p95"],
        "gk_clears_placebo_band": bool(clears),
        "gk_nan_fraction": gk_nan_frac,
        "base_nan_fraction": base_nan_frac,
        "ps_overlap_fraction": overlap,
        "smd_max_pre": float(att_full.balance["smd_pre"].abs().max()),
        "smd_max_post": float(att_full.balance["smd_post"].abs().max()),
        "causal_claim_supported": claim_supported,
        "seed": seed,
        "estimator": "abadie_imbens_2006_with_replacement_J1",
        "caveat": (  # R3-L4: state honesty about what the numbers do and don't mean
            "state-vs-sender + tracking-only opportunity detection; league/era differ from paper. "
            "Common support = treated-within-control-PS-range (no density trimming). Treated/control "
            "Y-windows are time-shifted (treated anchored at t_cross, control at entry). Z is a "
            "same-team cross within T of entry, clamped to possession continuity."
        ),
    }


def _mean_impute(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    miss = ~np.isfinite(X)
    cm = np.nanmean(np.where(miss, np.nan, X), axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    out = X.copy()
    out[miss] = np.take(cm, np.where(miss)[1])
    return out


def run(data_dir: Path, out: Path, providers: list[str], carrier_min: float, seed: int) -> dict:
    from scripts._loader_pining import iter_matches  # confirm name/signature (Open item)
    from silly_kicks._causal.opportunities import build_opportunities

    meta = _load_model_metadata()
    eligible, coverage = [], {}
    for prov in providers:
        opps = []
        for frames, actions, home_team_id in iter_matches(data_dir, prov):
            o = build_opportunities(frames, actions, home_team_id=home_team_id, model_metadata=meta)
            if not o.empty:
                opps.append(o)
        if not opps:
            coverage[prov] = {"n_opp": 0, "carrier_coverage": 0.0, "included": False}
            continue
        df = pd.concat(opps, ignore_index=True)
        cov = float(df["carrier_resolved"].mean())
        coverage[prov] = {"n_opp": int(len(df)), "carrier_coverage": cov, "included": cov >= carrier_min}
        if cov >= carrier_min:
            eligible.append(df[df["carrier_resolved"]])

    if not eligible:
        metrics = {"status": "no_eligible_provider", "coverage": coverage}
    else:
        metrics = analyze(pd.concat(eligible, ignore_index=True), seed=seed)
        metrics["coverage"] = coverage
    _write(out, metrics)
    return metrics


def _load_model_metadata(variant: str = "default") -> dict:
    p = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking" / "_xcross_weights" / variant / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _write(out: Path, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_render(metrics), encoding="utf-8")


def _render(m: dict) -> str:
    if m.get("status") != "ok":
        return f"# xCross causal validation (TF-17 PR-C)\n\nstatus: {m.get('status')}\n"
    return (
        "# xCross causal validation (TF-17 PR-C)\n\n"
        f"- Opportunities: {m['n_opportunities']} ({m['n_treated']} treated; base Y={m['base_rate_Y']:.3f})\n"
        f"- ATT without GK: {m['att_without_gk']['estimate']:.4f} (SE {m['att_without_gk']['se']:.4f})\n"
        f"- ATT with GK:    {m['att_with_gk']['estimate']:.4f} (SE {m['att_with_gk']['se']:.4f})\n"
        f"- GK ablation shift: {m['gk_ablation_shift']:.4f}; placebo band p95: {m['placebo_band_p95']:.4f}\n"
        f"- **GK clears placebo band: {m['gk_clears_placebo_band']}** (reported, not a gate)\n"
        f"- NaN fraction GK/base: {m['gk_nan_fraction']:.3f}/{m['base_nan_fraction']:.3f}; "
        f"PS overlap: {m['ps_overlap_fraction']:.3f}\n"
        f"- SMD max pre/post: {m['smd_max_pre']:.3f} / {m['smd_max_post']:.3f}; "
        f"**claim supported: {m['causal_claim_supported']}**\n"
        f"- Caveat: {m['caveat']}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--providers", default="skillcorner,idsse,gradientsports")
    ap.add_argument("--carrier-coverage-min", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.data_dir, a.out, a.providers.split(","), a.carrier_coverage_min, a.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint** — `uv run ruff check scripts/validate_xcross_causal.py ; uv run ruff format --check scripts/validate_xcross_causal.py ; uv run pyright scripts/validate_xcross_causal.py`
- [ ] **Step 3: Stage** — `git add scripts/validate_xcross_causal.py`

---

## Task 7: e2e — drives the integration seam (H2)

**Files:** Create `tests/causal/test_causal_e2e.py`

- [ ] **Step 1: Write the e2e — genuinely chain `build_opportunities` → `analyze` → `_write` on real
  frames (R2-H2), plus a `run()` pass via a fake loader. NO synthetic-numpy opportunity rows.**

```python
# tests/causal/test_causal_e2e.py
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from silly_kicks._causal.opportunities import GK_BLOCK, PAPER_CONFOUNDERS, build_opportunities
from silly_kicks.spadl import config as _c

# Reuse the geometry-correct spell-fixture builders (single source -- tests/causal/_fixtures.py, R3-L2).
from tests.causal._fixtures import META, actions, spell


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
            rows.append([1, aid, int(per), 5, 11.0, cross, 1, 20, 8, 14, 6]); aid += 1
            if i % 4 == 0:  # post-cross shot (within W of t_cross)
                rows.append([1, aid, int(per), 5, 11.5, shot, 1, 14, 6, 0, 34]); aid += 1
        elif i % 3 == 0:  # control with a shot from entry
            rows.append([1, aid, int(per), 5, 12.0, shot, 1, 14, 6, 0, 34]); aid += 1
    return actions(rows)


@pytest.mark.e2e
def test_build_analyze_write_chain(tmp_path):
    from scripts.validate_xcross_causal import _write, analyze
    frames = _multi_spell_frames(80)
    opp = build_opportunities(frames, _synth_actions(frames), home_team_id=5, model_metadata=META)
    assert len(opp) >= 2
    assert {"Z", "Y", *PAPER_CONFOUNDERS, *GK_BLOCK} <= set(opp.columns)  # column contract
    assert opp["Z"].nunique() == 2  # both arms present -> analyze runs the full path
    m = analyze(opp, seed=0)
    _write(tmp_path, m)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "report.md").exists()
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["status"] == "ok"
    for k in ("att_without_gk", "att_with_gk", "placebo_band_p95", "gk_nan_fraction",
              "base_nan_fraction", "ps_overlap_fraction"):
        assert k in m
    assert np.isfinite(m["att_with_gk"]["estimate"])
    assert isinstance(m["gk_clears_placebo_band"], bool)
    assert isinstance(m["causal_claim_supported"], bool)


@pytest.mark.e2e
def test_run_with_fake_loader(tmp_path, monkeypatch):
    # Inject a fake scripts._loader_pining so run()'s `from scripts._loader_pining import iter_matches`
    # picks it up (function-local-import mocking) -> exercises run()'s coverage/eligible-pool/write.
    frames = _multi_spell_frames(80)
    actions = _synth_actions(frames)
    fake = types.ModuleType("scripts._loader_pining")
    fake.iter_matches = lambda data_dir, provider: iter([(frames, actions, 5)])
    monkeypatch.setitem(sys.modules, "scripts._loader_pining", fake)
    import scripts.validate_xcross_causal as V
    metrics = V.run(tmp_path, tmp_path, ["skillcorner"], carrier_min=0.0, seed=0)
    assert (tmp_path / "metrics.json").exists()
    assert "coverage" in metrics and metrics["coverage"]["skillcorner"]["n_opp"] >= 2


@pytest.mark.e2e
def test_analyze_positivity_guard(tmp_path):
    from scripts.validate_xcross_causal import analyze
    frames = _multi_spell_frames(10)
    opp = build_opportunities(frames, actions([]), home_team_id=5, model_metadata=META)  # no crosses
    m = analyze(opp, seed=0)  # all Z=0 -> guard, never NaN ATT (M5)
    assert m["status"] == "no_variation_in_treatment"
    assert "att_with_gk" not in m
```

> **Implementer note:** importing `scripts.validate_xcross_causal` may need `scripts/__init__.py` or a
> `PYTHONPATH`/`conftest` shim — mirror the `train_xcross_attempt.py` smoke pattern in
> `tests/tracking/test_xcross_attempt_integration.py`. The e2e's 200-seed placebo on ~80 rows is the
> slowest step (~seconds) — acceptable under `@e2e`; reduce `n_spells` if it drags.

- [ ] **Step 2: Run → PASS** — `uv run python -m pytest tests/causal/test_causal_e2e.py -q -m e2e`
- [ ] **Step 3: Stage** — `git add tests/causal/test_causal_e2e.py`

---

## Task 8: ADR-015 + NOTICE

**Files:** Create `docs/superpowers/adrs/ADR-015-causal-validation-port.md`; Modify `NOTICE`

- [ ] **Step 1: Write ADR-015** (include the two named approximations — M7)

```markdown
# ADR-015: Private causal-validation port for trained-model confounder testing

## Status
Accepted (TF-17 PR-C, 2026-06-07)

## Context
TF-17's novel claim is a goalkeeper-position confounder block on cross propensity. PR-B's shipped
validation (GK-block ablation + substitution probe) measures *surface movement*, not a causal
effect. Cao et al. (2025, arXiv:2505.11841) frame crossing causally via propensity-score matching
(R `Matching`). We need a paper-faithful causal test — ATT/ATNT on crosser-anchored opportunity
rows — without adding R or a new Python dependency, and without letting a research finding gate a
runtime feature.

## Decision
- A **private** `silly_kicks/_causal/` package: pure numpy/sklearn matching estimators
  (`matching.py`) + a pure spell-based opportunity-row builder (`opportunities.py`). No public API;
  not imported by `silly_kicks/__init__`. Promote to public `silly_kicks/causal/` only when a second
  consumer (TF-19) lands.
- **1:1 NN propensity matching, with replacement, ties allowed, no caliper** (paper-faithful);
  logistic propensity on **standardized** covariates. **Abadie-Imbens (2006) matching SE**
  (Imbens & Rubin 2015, Ch. 19).
- **Two named approximations** (so a future production consumer knows what to revisit):
  1. `σ²(X)` via the J=1 within-treatment-group nearest neighbor.
  2. Matching is on the **estimated** propensity score; the fixed-matching-variable AI formula is
     **conservative** under estimated-PS matching (Abadie-Imbens 2016, *Econometrica*). Acceptable
     for a *reported* artifact.
- The treatment window is `(entry, min(entry+T, spell_end)]` (R3-M1): a **fixed `T` cap** keeps
  Z-exposure bounded (no spell-length confounding — and since `Y`'s window is already fixed, the
  `spell_end` clamp adds no duration→`Y` path), while the `spell_end` cap prevents misattributing a
  cross from a *later* re-possession phase. NOT the variable spell length (rejected — R2-H3 confounder),
  NOT the surface model's 1 s frame horizon. The outcome is measured strictly **after** treatment
  (`(t_cross, t_cross+W]` treated; `(entry, entry+W]` control) to avoid reverse leakage (R2-M1); `Y` is
  not possession-clamped (treated/control windows are time-shifted — documented). The confounder set is
  the **7 paper confounders** (ball-geometry surface features excluded). GK missingness uses the
  **missing-indicator method**, not mean-fill. No causal claim is made when PS overlap (treated-within-
  control-PS-range; no density trimming) or post-match balance fails (`causal_claim_supported=False`).
- The GK-vs-placebo null is the **row-permuted GK block** (preserves GK marginals + within-block
  correlation). Note (R2-L3): row-permutation also breaks GK↔base-confounder correlation, so the null
  is *slightly conservative* vs a pure `Z`/`Y`-alignment null — standard for permutation nulls. The
  finding is **reported, never a ship/CI gate**; the only CI gates are the known-truth method tests
  (`tests/causal/`). A null causal finding is valid.

## Consequences
- No new runtime dependency; `import silly_kicks` stays light.
- The harness is maintainer-re-runnable on any corpus; its report is bundled, not recomputed in CI.
- If TF-19 needs the estimator, the private port is promoted (one move), not rewritten.
```

- [ ] **Step 2: Extend the NOTICE Cao et al. entry** (after the existing state-vs-sender note, ~line 79)

```
  PR-C adds a paper-faithful causal-validation harness (silly_kicks/_causal/, private; ADR-015):
  propensity-score matching (ATT/ATNT, with replacement, Abadie-Imbens SEs) on crosser-anchored
  opportunity-spell rows, ablating the GK confounder block against a row-permuted-GK placebo null
  band. It reconstructs the paper's sender-level unit; the remaining divergence (tracking-only
  opportunity detection; different league/era corpus) is reported, not hidden. The causal finding
  is a reported research result, never a ship gate.
```

- [ ] **Step 3: Stage** — `git add docs/superpowers/adrs/ADR-015-causal-validation-port.md NOTICE`

---

## Task 9: Maintainer DGX run — bundle metrics.json + report

**Files:** Create `docs/research/xcross_causal/metrics.json` + `report.md` (final path confirmed at run time; small text → git, not Hub)

- [ ] **Step 1: Full non-e2e suite green on Windows first** — `uv run python -m pytest tests/ -m "not e2e" -q`
- [ ] **Step 2: Run on the pining corpus** (DGX, venv `~/sk-s81-venv`, token `~/.pining_env`):

```bash
python scripts/validate_xcross_causal.py --data-dir <pining_pull_dir> --out ~/Development/xcross_causal_out \
  --providers skillcorner,idsse,gradientsports --carrier-coverage-min 0.6 --seed 0
```
Expected: writes `metrics.json` + `report.md`; logs per-provider carrier coverage + the GK-vs-placebo verdict + `causal_claim_supported`. ATT sign vs paper (positive = cross → more shots) logged, not gated.

- [ ] **Step 3: Fetch + bundle** — scp back; `git add docs/research/xcross_causal/metrics.json docs/research/xcross_causal/report.md`

---

## Task 10: Final integration — docs, version, /final-review, single commit (closes TF-17)

**Files:** `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `NOTICE` (T8), version files (per decision below)

- [ ] **Step 1: Resolve the commit/release structure with the owner** — **(A)** one combined 4.16.0
  (PR-C's CHANGELOG into `[4.16.0]`; no further bump) vs **(B)** two releases (PR-B 4.16.0; PR-C
  **4.17.0**, own CHANGELOG section + version bump), both committed back-to-back. Present with a
  recommendation; do not pick unilaterally.
- [ ] **Step 2: PR-C CHANGELOG section** (incl. the real-data headline numbers from Task 9), CLAUDE.md
  one-liner (PR-C closes TF-17; private `_causal/` port; ADR-015), TODO TF-17 row → **CLOSED**.
- [ ] **Step 3: Version hard-gate** (only if Option B): bump `pyproject.toml` + `silly_kicks/__init__.py`
  + `uv.lock`; confirm all match the CHANGELOG.
- [ ] **Step 4: `/final-review`** (mandatory pre-commit gate).
- [ ] **Step 5: Full green gate** — `uv run python -m pytest tests/ -m "not e2e" -q ; uv run pyright silly_kicks/ ; uv run ruff check silly_kicks/ scripts/ tests/ ; uv run ruff format --check silly_kicks/ scripts/ tests/`
- [ ] **Step 6: Build + size-check both artifacts** — `uv build` then confirm sdist AND wheel < 100 MB.
- [ ] **Step 7: Request explicit commit approval → single commit** (`git commit -F <msgfile>` on
  Windows) → push → admin-squash-merge → wait CI green → tag. **Closes TF-17.**

---

## Self-Review (rev. 4 — run after writing; fixed inline)

- **Round-3 coverage (owner chose Option B):** R3-M1 Z clamped to `(entry, min(entry+T, spell_end)]`
  (`_label_treatment` takes `end_time`; two tests `test_treatment_capped_by_window_T` /
  `..._by_possession_end`) ✓; R3-L1 e2e covariate jitter (ball-x + nearest-defender) ✓; R3-L2 shared
  builders in `tests/causal/_fixtures.py` (unit + e2e import from it; no sibling-test-module import) ✓;
  R3-L3 "one period per spell is a test device" comment ✓; R3-L4 report caveat states common-support =
  range-only (no trimming) + Y time-shift ✓. Round-2 items (R2-H1..L3) remain ✓ (verified by the d32
  round-2 pass).
- **Type consistency:** `_label_treatment(actions, gid, per, team, cross_types, entry, end_time) ->
  (int, float|None)` (end_time added) feeds `anchor` in `_row`; `build_opportunities` returns
  `PAPER_CONFOUNDERS + GK_BLOCK + _PROV_COLS + [Z, Y]`, consumed by T6 `analyze` + the T7 column-contract
  assert; `_fixtures.py` exports `META/WIDE/CENTRAL/NEAR0/frow/frames/spell/actions` used by T5 + T7;
  Tasks 1-4 (matching.py) untouched.
- **Placeholder scan:** none.

## Open items the reviewing session should still scrutinize

1. **`iter_matches` loader signature** (T6 `run`) — confirm the actual `scripts/_loader_pining` iterator
   name + return shape `(frames, actions, home_team_id)` against PR-B's code; adjust if it differs. (The
   T7 `run()` e2e injects a fake loader, so the contract is at least pinned by a test.)
2. **Pre-registered constants** — `EXPOSURE_WINDOW_SECONDS=8`, `OUTCOME_WINDOW_SECONDS=6`,
   `MAX_SPELL_SECONDS=30` (dedup cap), `_OVERLAP_MIN=0.5`; confirm values before the real-data run. (The
   rev.2 spell-length confounder and the rev.3 misattribution are both **resolved** by the R3-M1 clamp —
   `(entry, min(entry+T, spell_end)]` — not deferred. The double-count-across-re-entries edge is also
   largely closed: a cross in a re-entry spell B is clamped out of spell A's window by A's `spell_end`.)
3. **`ids_match` on `actions["game_id"]`/`["team_id"]`** (`_team_period_action_times`) — confirm this is the
   right dtype-safe primitive for the action↔frame-team seam (ADR-019 helper; `_causal/` is outside the
   AST-lint glob, so correctness rides on this choice, not the lint).
4. **Bundled artifact path** `docs/research/xcross_causal/` — confirm vs any existing research convention.
5. **`infer_ball_carrier` on short spells** — the 2-frame carrier-handoff unit fixture asserts the carrier
   id actually flips (loud if it doesn't); the e2e uses [10,12] s spells. If hysteresis suppresses carrier
   resolution on a fixture, lengthen it.
