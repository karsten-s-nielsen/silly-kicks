# xT-GK fidelity fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring silly-kicks' xT-GK PEV and DZV terms into fidelity with Eyestone's published framework (Q1–Q3 answered 2026-06-27): PEV measures its forward gain on the GK-revalued surface `V_GK = xT·φ` (not raw xT, which flatlines in keeper zones); DZV becomes the published defensive-zone revaluation multiplier `M(z) = φ(z,d)·[1 − V_GK(z)/max(V_GK)]` applied as the revaluation *increment* on the origin possession value (Option A); Option B (base = −xT*(origin), RAV sole owner of the destination) is preserved unchanged.

**Architecture:** All math is single-sourced in `silly_kicks/tracking/_xt_gk.py` (`add_xt_gk`/`xt_gk_xfns`/atomic mirror only wrap `compute_xt_gk`). A canonical φ(z,d) grid feeds a second convolved surface `V_GK*` alongside the existing raw `xT*`. PEV reads `V_GK*`; DZV reads `V_GK*` + analytic φ; **base and RAV stay on raw `xT*`** (the invariant — φ enters value via PEV and DZV only). The scalar `phi` param remains the preset-modulated overall DZV weight; the canonical shape (α, β) lives in the φ grid.

**Tech Stack:** numpy, scipy.ndimage (gaussian_filter), pandas; pytest.

---

## Key formulas (the contract)

```
φ(z,d) = α·(1 − d/D_max)^(−β)   for d < D_threshold ;  else 1.0      (α=2.1, β=0.8 canonical; D_max=105; D_threshold = defensive_third_boundary = 35)
d      = LTR origin x (team attacks +x), 0..105

V_GK   = xT ⊙ φ_grid                       # element-wise revaluation
V_GK*  = gaussian_filter(V_GK, σ)          # convolved like xT*  (σ = convolution_sigma = 0.8)
xT*    = gaussian_filter(xT, σ)            # raw, UNCHANGED

base     = −xT*(z)                          # raw — UNCHANGED (Option B)
RAV      = p·xT*(z′) − δ·(1−p)·xT*(z′ᶜ)     # raw — UNCHANGED (Option B; RAV sole destination owner)
progress = V_GK*(z′) − V_GK*(z)             # CHANGE 1: revalued surface
PEV      = ρ · max(0, progress)             # kept EXACTLY as built
M(z)     = φ(z,d) · (1 − V_GK*(z)/max(V_GK*))                          # the published multiplier (~2.5)
DZV      = where(d < D_threshold, (M(z) − 1)·V_GK*(z), 0.0)            # CHANGE 2: Option A increment, def-third gated

composite = T·(base + γ·PEV + RAV) + φ_scalar·DZV     # _composite UNCHANGED
```

Invariant: changing α/β (φ shape) must change only `xt_gk_pev` + `xt_gk_dzv`; `xt_gk_base` + `xt_gk_rav` must be byte-identical.

---

## File structure

- Modify: `silly_kicks/tracking/_xt_gk.py` — `XtGkParams` (drop `v_def`, add `dzv_alpha`/`dzv_beta`/`dzv_d_max`); new pure helpers `_phi_of_d`, `_phi_grid`; rewrite `_dzv`; `compute_xt_gk` builds `V_GK*` and routes PEV/DZV to it.
- Modify: `tests/tracking/test_xt_gk.py` — rewrite DZV unit + params-range tests; add φ helpers, revalued-PEV, invariant, scale-anchor tests.
- Modify: `docs/superpowers/adrs/ADR-024-xt-gk.md` — amendment (4.35.0, Q1–Q3 fidelity fix).
- Modify: `NOTICE` — note the φ(z,d) revaluation surface in the xT-GK formulation paragraph.
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` — version 4.34.0 → 4.35.0.

No change to `features.py` / `atomic/tracking/features.py` (they wrap `compute_xt_gk`); the `_OUTPUT_COLS` set is unchanged (same six value columns).

---

## Task 1: XtGkParams — canonical φ params, drop dead v_def

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (the `XtGkParams` dataclass + `_OUTPUT_COLS` region)
- Test: `tests/tracking/test_xt_gk.py::TestXtGkParams`

- [ ] **Step 1: Update the failing test** — replace the `v_def` assertion with the new fields:

```python
def test_default_is_frozen_and_in_range(self):
    p = XtGkParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.gamma = 0.5  # type: ignore[misc]  # frozen
    assert 0.1 <= p.gamma <= 0.4
    assert 0.3 <= p.delta <= 0.8
    assert 0.8 <= p.eta <= 0.9
    assert p.phi > 0.0
    assert p.dzv_alpha == pytest.approx(2.1)   # canonical
    assert p.dzv_beta == pytest.approx(0.8)    # canonical
    assert p.dzv_d_max > 0.0
    assert p.defensive_third_boundary > 0.0    # = D_threshold
    assert p.pressure_scale > 0.0
    assert p.convolution_sigma >= 0.0
    assert p.pressure_method == "andrienko_oval"
```

- [ ] **Step 2: Run** `python -m pytest tests/tracking/test_xt_gk.py::TestXtGkParams::test_default_is_frozen_and_in_range -q` → FAIL (no `dzv_alpha`).

- [ ] **Step 3: Edit `XtGkParams`** — remove `v_def`, add the three φ params (keep `defensive_third_boundary` as D_threshold):

```python
    gamma: float = 0.25  # PEV pressure-escape sensitivity   (range 0.1-0.4)
    delta: float = 0.55  # RAV risk-aversion                 (range 0.3-0.8)
    phi: float = 1.0  # DZV overall weight (preset-modulated; canonical SHAPE is dzv_alpha/beta)
    eta: float = 0.85  # temporal-sequence discount        (range 0.8-0.9)
    # --- DZV canonical revaluation φ(z,d) = α·(1 − d/D_max)^(−β) for d < D_threshold else 1 ---
    dzv_alpha: float = 2.1  # CANONICAL (Eyestone 2026-06-27)
    dzv_beta: float = 0.8  # CANONICAL (Eyestone 2026-06-27)
    dzv_d_max: float = 105.0  # provisional; pitch length
    defensive_third_boundary: float = 35.0  # NORMATIVE: D_threshold = own defensive third end (105/3)
    pressure_scale: float = 50.0  # rho squash scale; intent-set
```

- [ ] **Step 4: Run** the Task-1 test → PASS. Run the whole `TestXtGkParams` class.

## Task 2: φ helpers — `_phi_of_d` and `_phi_grid`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (new helpers near `_convolve_grid`)
- Test: `tests/tracking/test_xt_gk.py::TestPureHelpers`

- [ ] **Step 1: Write failing tests:**

```python
    def test_phi_of_d_canonical_values(self):
        from silly_kicks.tracking._xt_gk import _phi_of_d
        phi = _phi_of_d(np.array([0.0, 5.0, 34.0, 35.0, 60.0]), alpha=2.1, beta=0.8, d_max=105.0, d_threshold=35.0)
        assert phi[0] == pytest.approx(2.1)            # d=0 -> alpha
        assert phi[1] == pytest.approx(2.1 * (1 - 5/105) ** -0.8)
        assert phi[2] == pytest.approx(2.1 * (1 - 34/105) ** -0.8)
        assert phi[2] > 2.8                             # ~2.9 just below threshold
        assert phi[3] == pytest.approx(1.0)            # at threshold -> cliffs to 1
        assert phi[4] == pytest.approx(1.0)            # outside def third -> 1
        assert (phi >= 1.0).all()

    def test_phi_grid_is_row_constant_and_matches_phi_of_d(self):
        from silly_kicks.tracking._xt_gk import _phi_grid, _phi_of_d
        g = _phi_grid((12, 16), alpha=2.1, beta=0.8, d_max=105.0, d_threshold=35.0)
        assert g.shape == (12, 16)
        np.testing.assert_allclose(g[0], g[5])          # depends on x (col) only -> rows identical
        # column-center x for col c = field_length * (c + 0.5)/n_cols
        xc = spadlconfig.field_length * (np.arange(16) + 0.5) / 16
        np.testing.assert_allclose(g[0], _phi_of_d(xc, 2.1, 0.8, 105.0, 35.0))
```

- [ ] **Step 2: Run** `python -m pytest tests/tracking/test_xt_gk.py::TestPureHelpers -q -k "phi"` → FAIL (import error).

- [ ] **Step 3: Implement helpers** (after `_convolve_grid`):

```python
def _phi_of_d(
    d: npt.NDArray[np.float64], alpha: float, beta: float, d_max: float, d_threshold: float
) -> npt.NDArray[np.float64]:
    """Eyestone's defensive-zone revaluation factor φ(z,d) = α·(1 − d/D_max)^(−β) for
    d < D_threshold, else 1.0. d = distance from own goal = LTR origin x (team attacks +x).
    φ ≥ 1, rising with depth toward the threshold, then cliffing to 1 outside the defensive
    third. See NOTICE for full bibliographic citations (Eyestone xT-GK)."""
    d = np.asarray(d, float)
    active = d < d_threshold
    # (1 − d/D_max) is strictly positive for d < D_threshold < D_max -> no negative base
    raised = alpha * np.power(1.0 - np.where(active, d, 0.0) / d_max, -beta)
    return np.where(active, raised, 1.0)


def _phi_grid(
    shape: tuple[int, int], alpha: float, beta: float, d_max: float, d_threshold: float
) -> npt.NDArray[np.float64]:
    """Per-cell φ(z,d) grid matching an xT grid's (n_rows, n_cols). φ depends on x only
    (d = column-center x), so every row is identical. Column c -> x = field_length·(c+0.5)/n_cols
    (cell centre, matching xthreat's cell convention)."""
    n_rows, n_cols = shape
    xc = spadlconfig.field_length * (np.arange(n_cols) + 0.5) / n_cols
    row = _phi_of_d(xc, alpha, beta, d_max, d_threshold)
    return np.tile(row, (n_rows, 1))
```

- [ ] **Step 4: Run** the φ tests → PASS.

## Task 3: Rewrite `_dzv` — Option A increment of the published multiplier

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (`_dzv`)
- Test: `tests/tracking/test_xt_gk.py::TestComponents::test_dzv_*`

- [ ] **Step 1: Replace the DZV unit test:**

```python
    def test_dzv_is_revaluation_increment_in_defensive_third(self):
        # Option A: DZV = (M-1)·V_GK(z), M = φ(z,d)·(1 − V_GK(z)/max V_GK); gated to def third.
        from silly_kicks.tracking._xt_gk import _dzv, _phi_of_d
        vgk_origin = np.array([0.02, 0.02])     # small deep possession value
        start_x = np.array([10.0, 60.0])        # in def third, outside
        vgk_max = 1.0
        out = _dzv(start_x, vgk_origin, vgk_max, alpha=2.1, beta=0.8, d_max=105.0, boundary=35.0)
        phi0 = _phi_of_d(np.array([10.0]), 2.1, 0.8, 105.0, 35.0)[0]
        m0 = phi0 * (1 - 0.02 / 1.0)
        assert out[0] == pytest.approx((m0 - 1.0) * 0.02)   # positive increment in def third
        assert out[0] > 0.0
        assert out[1] == pytest.approx(0.0)                  # outside def third -> 0
```

- [ ] **Step 2: Run** `python -m pytest tests/tracking/test_xt_gk.py::TestComponents -q -k dzv` → FAIL (old signature).

- [ ] **Step 3: Rewrite `_dzv`:**

```python
def _dzv(start_x, vgk_star_origin, vgk_star_max, alpha, beta, d_max, boundary):
    """Eyestone DZV — defensive-zone revaluation, Option A (the increment over raw credit).
    M(z) = φ(z,d)·(1 − V_GK(z)/max V_GK) is the published multiplier (~2.5); the composite
    adds the revaluation GAIN it confers on the origin possession value, (M−1)·V_GK(z), so
    base (which surrenders raw origin threat) and DZV stay orthogonal. Gated to the defensive
    third (φ active region). See NOTICE for full bibliographic citations (Eyestone xT-GK)."""
    start_x = np.asarray(start_x, float)
    vgk = np.asarray(vgk_star_origin, float)
    phi = _phi_of_d(start_x, alpha, beta, d_max, boundary)
    m = phi * (1.0 - vgk / vgk_star_max)
    in_def_third = start_x < boundary
    return np.where(in_def_third, (m - 1.0) * vgk, 0.0)
```

- [ ] **Step 4: Run** the DZV test → PASS.

## Task 4: `compute_xt_gk` — build V_GK*, route PEV + DZV to it; keep base/RAV raw

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (`compute_xt_gk` body, the grid/lookup block ~lines 434–469)
- Test: `tests/tracking/test_xt_gk.py::TestComputeXtGk`

- [ ] **Step 1: Add behavioral tests** (revalued-PEV, invariant, scale anchor) — see Task 5. First make the existing `TestComputeXtGk` suite pass with the new internals.

- [ ] **Step 2: Edit the grid/lookup block.** Replace the single hoisted `dest_star`/`origin_star` block and the DZV call:

```python
    xt_star = _convolve_grid(xt.xT, p.convolution_sigma)              # raw — base + RAV
    phi_grid = _phi_grid(xt.xT.shape, p.dzv_alpha, p.dzv_beta, p.dzv_d_max, p.defensive_third_boundary)
    vgk_star = _convolve_grid(np.asarray(xt.xT, float) * phi_grid, p.convolution_sigma)  # revalued — PEV + DZV
    vgk_max = float(np.nanmax(vgk_star))

    sx = sub_geom["origin_x"].to_numpy(float)
    sy = sub_geom["origin_y"].to_numpy(float)
    ex = sub_geom["dest_x"].to_numpy(float)
    ey = sub_geom["dest_y"].to_numpy(float)

    dest_star = _grid_value(xt_star, ex, ey)        # raw — RAV owns the destination (Option B)
    origin_star = _grid_value(xt_star, sx, sy)      # raw — base (Option B)
    base = _base(origin_star)
    # CHANGE 1: PEV's forward gain is measured on the GK-revalued surface (raw xT flatlines deep).
    dest_vgk = _grid_value(vgk_star, ex, ey)
    origin_vgk = _grid_value(vgk_star, sx, sy)
    progress = _progress(dest_vgk, origin_vgk)
```

Then `pev = _pev(rho, progress)` is unchanged. RAV is unchanged (`_rav(pc, dest_star, _counter_value(xt_star, ex, ey), p.delta)`). Replace the DZV line:

```python
    dzv = _dzv(sx, origin_vgk, vgk_max, p.dzv_alpha, p.dzv_beta, p.dzv_d_max, p.defensive_third_boundary)
```

(Delete the old `_grid_value(xt.xT, sx, sy)` raw-origin DZV lookup.)

- [ ] **Step 3: Run** `python -m pytest tests/tracking/test_xt_gk.py::TestComputeXtGk -q`. Fix the two existing DZV-dependent tests:
  - `test_backpass_penalty_corrected_upward`: keep `assert out.loc[1, "xt_gk_dzv"] > 0.0` (still holds on the realistic grid; back-pass origin x=25 is in the def third, φ>1 → positive increment).
  - `test_uses_injected_grid_not_self_fit`: unaffected (base on raw `xT*`, σ=0).

- [ ] **Step 4: Run** the full `TestComputeXtGk` → PASS.

## Task 5: Fidelity guard tests (CHANGE 1 + invariant + scale anchor)

**Files:**
- Test: `tests/tracking/test_xt_gk.py::TestComputeXtGk` (new methods)

- [ ] **Step 1: Add a deep-flat grid fixture** (raw forward gain ≈ 0 in the defensive third) at module level:

```python
def _deep_flat_xt():
    """Raw xT FLAT (~0.005) across the defensive third, rising only past it. On this grid a
    short deep build-out has ~zero RAW forward gain — the keeper-zone flatline Eyestone fixes.
    Revaluation (V_GK = xT·φ) restores a positive gain. Shape (12, 16)."""
    xt = ExpectedThreat(l=16, w=12)
    ramp = np.concatenate([np.full(6, 0.005), np.linspace(0.005, 1.0, 10)])  # cols 0-5 flat, then rise
    xt.xT = np.tile(ramp, (12, 1))
    return xt
```

- [ ] **Step 2: Add the tests:**

```python
    def test_pev_reads_revalued_surface_not_raw(self):
        # CHANGE 1: a short deep build-out (origin x=5 -> dest x=25, both in the flat def third)
        # has ~zero RAW gain -> PEV ~0 with revaluation OFF; canonical φ lights it up.
        actions = _gk_actions().iloc[[0]].copy()
        actions.loc[0, "start_x"] = 5.0
        actions.loc[0, "end_x"] = 25.0  # stays in the flat defensive third
        frames = _frames_for(actions)
        off = XtGkParams(dzv_alpha=1.0, dzv_beta=0.0)   # φ ≡ 1 -> V_GK ≡ xT (revaluation disabled)
        on = XtGkParams()                                # canonical α=2.1, β=0.8
        pev_off = compute_xt_gk(actions, frames, xt=_deep_flat_xt(), params=off).loc[0, "xt_gk_pev"]
        pev_on = compute_xt_gk(actions, frames, xt=_deep_flat_xt(), params=on).loc[0, "xt_gk_pev"]
        assert pev_off == pytest.approx(0.0, abs=1e-4)   # raw deep gain flatlines
        assert pev_on > pev_off                          # revaluation restores the gain

    def test_phi_shape_changes_only_pev_and_dzv_not_base_or_rav(self):
        # The invariant Eyestone fixed: φ enters value via PEV + DZV ONLY. Base + RAV (raw xT*)
        # must be byte-identical when the φ shape changes.
        actions = _gk_actions()
        frames = _frames_for(actions)
        a = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(dzv_alpha=2.1, dzv_beta=0.8))
        b = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(dzv_alpha=3.5, dzv_beta=1.5))
        np.testing.assert_array_equal(a["xt_gk_base"].to_numpy(), b["xt_gk_base"].to_numpy())
        np.testing.assert_array_equal(a["xt_gk_rav"].to_numpy(), b["xt_gk_rav"].to_numpy())
        assert not np.allclose(  # at least one of PEV/DZV moved (the in-scope, in-def-third rows)
            np.nan_to_num(a[["xt_gk_pev", "xt_gk_dzv"]].to_numpy()),
            np.nan_to_num(b[["xt_gk_pev", "xt_gk_dzv"]].to_numpy()),
        )

    def test_dzv_scale_is_order_hundredth_not_unity(self):
        # Scale anchor (Eyestone): per-action DZV must land O(0.01), not the literal multiplier
        # O(2.5). Back-pass origin x=25 is in the defensive third on the realistic grid.
        actions = _gk_actions().iloc[[1]].copy()
        frames = _frames_for(actions)
        dzv = compute_xt_gk(actions, frames, xt=_gk_realistic_xt()).loc[1, "xt_gk_dzv"]
        assert 0.0 < dzv < 0.5   # positive bar, two-plus orders below the raw ~2.5 multiplier
```

- [ ] **Step 3: Run** `python -m pytest tests/tracking/test_xt_gk.py -q` → all PASS.

## Task 6: Full xT-GK suite + adjacent gates

- [ ] **Step 1:** `python -m pytest tests/tracking/test_xt_gk.py -q` → PASS.
- [ ] **Step 2:** Run the auto-enumerating gates that touch the xt_gk surface:
  `python -m pytest tests/test_enrichment_nan_safety.py tests/test_add_star_purity.py tests/tracking/test_aggregator_column_liveness.py tests/tracking/test_id_compat_lint.py tests/tracking/test_frame_aware_xfns_dup_action_id.py -q` → PASS.
- [ ] **Step 3:** `pyright silly_kicks/` (full package, per user rule) → no new errors.
- [ ] **Step 4:** `ruff format --check . ; ruff check .` → clean.

## Task 7: Docs — ADR-024 amendment + NOTICE + version bump

**Files:** `docs/superpowers/adrs/ADR-024-xt-gk.md`, `NOTICE`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1:** Append an ADR-024 amendment "(4.35.0, 2026-06-27) — PEV/DZV fidelity fix (Eyestone Q1–Q3)": CHANGE 1 (PEV on V_GK*), CHANGE 2 (DZV = published M(z), Option A increment, scale-reconciled), invariant (φ only in PEV+DZV), params (α/β canonical, D_* provisional; `v_def` retired), and the post-run verification checklist (four-term sum, φ-not-in-RAV grep, DZV distribution + by-zone profile vs ~0.009 mean, PEV lights-up-for-short-deep-build-out, d=35 boundary sanity). VAEP/tracking note: opt-in feature (no forced retrain) but an `xt_gk` serve-output change → lakehouse re-materializes `fct_action_context` + re-runs the WC2022 cohort/report.
- [ ] **Step 2:** Update the NOTICE xT-GK formulation paragraph to mention the φ(z,d) defensive-zone revaluation surface `V_GK = xT·φ`.
- [ ] **Step 3:** Bump version 4.34.0 → 4.35.0 in `pyproject.toml` + `silly_kicks/__init__.py`; add a `CHANGELOG.md` 4.35.0 entry; update `TODO.md` Current-release header.
- [ ] **Step 4:** Run `/final-review`.

## Task 8: STOP before commit

- [ ] Per the owner's instruction, do NOT commit. Summarize the diff, the test results, and the realized DZV magnitude observed in the unit fixtures, and hand off for the owner's review + commit approval.

---

## Self-review notes

- Spec coverage: CHANGE 1 (Task 4), CHANGE 2 (Tasks 3–4), φ canonical (Tasks 1–2), invariants (Task 5 invariant test + the raw `xt_star` base/RAV lines), verification checklist (Task 7 ADR amendment; the runtime four-term-sum identity is already pinned by `test_composite_discounts_threat_terms_only_not_dzv`). Scale anchor (Task 5).
- `_OUTPUT_COLS` unchanged → `add_xt_gk`/`xfns`/atomic mirror/`XtGkReport` need no edits.
- `phi` scalar param retained as the overall DZV weight (presets 0.8–1.3 keep modulating); canonical shape in the grid.
