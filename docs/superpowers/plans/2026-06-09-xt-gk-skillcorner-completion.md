# xT-GK SkillCorner completion (gold-standard `result_id` fix + provider-aware variants) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **No git worktrees** — work on a feature branch in the main checkout (`git switch -c feat/xt-gk-skillcorner-completion`).

**Goal:** Make SkillCorner `xt_gk` construct-correct and poolable with Gradient Sports by fixing SkillCorner pass-completion `result_id` to the native outcome, then adding provider-aware completion-model variant selection + pooling-safety gates.

**Architecture:** Three layers, in a load-bearing order (spec §4): (1) fix SkillCorner `result_id` at the converter to a single native-completion construct (`pass_outcome` primary, `received` success-only, flagged residual) so the canonical label is correct for every consumer (VAEP + features + xT-GK); (2) provider-aware variant selection in the completion model (pure `variant_key_for_provider` + `from_variant` + auto-resolution in `compute_xt_gk`); (3) train/gate the SkillCorner variant (GS-transfer re-measurement, common-scale calibration, cross-provider comparability) before the lakehouse pools. Builds on the already-shipped xT-GK feature + GS `default` GK-completion model.

**Tech Stack:** Python 3.10 (uv `.venv`), pandas/numpy, `np.select` vectorized converter dispatch, pure-numpy logistic `GkCompletionModel` (JSON+SHA256, no pickle), pytest (`-m "not e2e"` for CI; owner/public-run gates via `_loader_pining`), ruff + pyright.

**Source of truth:** `docs/superpowers/specs/2026-06-09-xt-gk-multiprovider-completion-design.md` (APPROVED FOR IMPLEMENTATION). Cross-references below cite spec decisions (D-Sn) and review resolutions (N1–N4, §-numbers).

**Scope note:** bundled into the in-flight **4.21.0** branch (xT-GK + GS goal-kick coverage already built). This plan is the SkillCorner extension. It is a **VAEP-retrain trigger** (SkillCorner scores/concedes shift) — accepted (D-S8); the lakehouse re-materializes SkillCorner VAEP + xT-GK together and waits until ready.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `silly_kicks/spadl/skillcorner.py` | SkillCorner → SPADL converter | Fix pass/set-piece `result_id` to native completion; emit `result_source` |
| `silly_kicks/spadl/schema.py` | SPADL schemas | Add `result_source` to `SKILLCORNER_SPADL_COLUMNS` |
| `silly_kicks/tracking/_gk_completion.py` | GK-completion model + variant registry | Add pure `variant_key_for_provider`; extend `from_variant` |
| `silly_kicks/tracking/_xt_gk.py` | xT-GK compute + provenance + report | Auto-resolve variant; emit `xt_gk_completion_variant`/`xt_gk_completion_source`; `XtGkReport.spans_multiple_variants` |
| `scripts/train_gk_completion.py` | Train script (owner-run) | `--variant skillcorner`; GS-transfer re-measure; common-scale calibration; GK-pass AUC + goal-kick LCB |
| `scripts/_xtgk_comparability.py` (new) | D-S9 comparability gate (owner-run) | SC-vs-GS `xt_gk` distribution comparison + affine re-scale recommendation |
| `tests/spadl/test_skillcorner_completion.py` (new) | Converter completion tests (CI) | N1 regression, offside, `result_source` tiers |
| `tests/tracking/test_gk_completion_variants.py` (new) | Variant-selection tests (CI) | pure mapping, auto-select, mixed-provider raise, override |
| `tests/regressions/extratime/golden_skillcorner_*.parquet` | SkillCorner conversion golden(s) | Regenerated (reviewed diff) |
| `docs/.../ADR-024*`, `CHANGELOG.md`, `CLAUDE.md`, model card | Docs | Amend for the `result_id` fix + variant family |

---

## Phase 0 — Measurement & Chesterton's Fence (decides the residual-gap policy)

**This phase produces a decision, not shipped code. It is owner/public-run (SkillCorner pining via `_loader_pining`), not a CI gate.** Its output feeds Task 3's residual-gap branch. Do it first (spec §4 plan-task-1).

### Task 1: Settle why `same_team_next` was chosen + the residual-gap mechanism

**Files:**
- Read: `docs/superpowers/specs/2026-05-14-skillcorner-events-converter-design.md`, `silly_kicks/spadl/skillcorner.py:260-352`
- Create (throwaway, delete after): `xtgk_gap_policy_probe.py` at repo root

- [ ] **Step 1: Chesterton's Fence — read the converter spec + git history for the `same_team_next` result rule.**

Run: `git log -p --follow -S "same_team_next" -- silly_kicks/spadl/skillcorner.py | head -200`
Read the converter design spec's "result" section. Record: which `end_type` rows the `same_team_next` branch is the result for (from `skillcorner.py:334-352` it is the catch-all for pass/cross/set-piece/possession_loss/unknown rows — clearance/foul/shot have explicit results). Confirm `pass_outcome` is absent only for non-pass rows.

- [ ] **Step 2: Measure native-vs-proxy agreement on real SkillCorner (public token).**

```python
# xtgk_gap_policy_probe.py
import sys, tempfile; from pathlib import Path
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from _loader_pining import _list_matches, _resolve_token, _base_url, _artifact_key, _download_to_temp
tok, base = _resolve_token(None), _base_url()
rows = []
for m in _list_matches("skillcorner", tok, base)[:8]:
    k = _artifact_key(m["artifacts"], suffix="_dynamic_events.csv")
    with tempfile.TemporaryDirectory() as t:
        p = _download_to_temp("skillcorner", m["id"], k if k in m["artifacts"] else "events", tok, base, Path(t))
        rows.append(pd.read_csv(p, low_memory=False))
ev = pd.concat(rows, ignore_index=True)
passes = ev[ev["end_type"] == "pass"].copy()
po = passes["pass_outcome"].astype(str)
has_po = po.isin(["successful", "unsuccessful", "offside"])
po_succ = (po == "successful")
nt = passes["team_id"].shift(-1)
stn = (passes["team_id"] == nt)
rec = passes["received"].astype(str).str.lower().isin(["true", "1", "1.0"])
sub = passes[has_po]
print(f"passes={len(passes)} pass_outcome-present={has_po.mean():.1%}")
print(f"same_team_next vs pass_outcome agreement: {(stn[has_po].to_numpy()==po_succ[has_po].to_numpy()).mean():.3f}")
print(f"received        vs pass_outcome agreement: {(rec[has_po].to_numpy()==po_succ[has_po].to_numpy()).mean():.3f}")
print(f"rows lacking pass_outcome but with received=True: {(~has_po & rec).sum()}")
print(f"rows lacking both pass_outcome and received=True: {(~has_po & ~rec).sum()}  ({(~has_po & ~rec).mean():.1%})")
```

Run: `.venv/Scripts/python.exe xtgk_gap_policy_probe.py` (use `run_in_background=true`, poll the output file)

- [ ] **Step 2b: Measure the candidate `inferred` success signal (F2 — define it precisely).** `inferred` is **NOT** `same_team_next` and **NOT** spatial attribution. The only precise, attribution-free candidate is **"the next SPADL action's `player_id == player_targeted_id`"** (the targeted player took the next touch → a clean *success* signal, like `received`). Measure, **split by sub-domain** (goal-kick vs GK-pass): its coverage (needs `player_targeted_id` + a next action) and its agreement with `pass_outcome==successful` on rows where both exist. (Note the asymmetry, like `received`: next-action==targeted → clean success; next-action≠targeted → ambiguous, NOT a fail signal.)

- [ ] **Step 3: Decide the residual policy PER SUB-DOMAIN (F3) and record it.** Goal-kicks (no `received`, `pass_outcome` ~60%, ~40% residual — the xT-GK focus) and GK-passes (`received` 64%, small residual) have structurally different coverage; decide each, don't apply one global threshold.

The tiers are fixed (`native` = `pass_outcome`; `inferred` = clean success signals: `received==True`, plus next-action==`player_targeted_id` **iff** Step 2b shows it agrees with `pass_outcome` ≥ ~0.9 and adds coverage; `stopgap` = `same_team_next` residual). What Phase 0 decides is **only** whether the Step-2b signal is clean enough to *promote into `inferred`* (per sub-domain). Because **training excludes `stopgap` (F1)**, the residual tier's *only* job is best-available VAEP `result_id` coverage — so `same_team_next` is acceptable there regardless (no reconstruction needed; F2 de-risked by F1). Record the chosen `inferred` membership + the measured agreement/coverage (per sub-domain) in the spec §4 + a memory note. **Confirm explicitly that goal-kicks get the cleanest available training label** (native `pass_outcome` ∪ any promoted `inferred`), since that is where the noise bites.

- [ ] **Step 4: Clean up.** Run: `rm -f xtgk_gap_policy_probe.py`

> **✅ MEASURED OUTCOME (2026-06-09, 8 matches / 39,083 events).** Chesterton's Fence: the converter design spec is silent on pass-completion construct → `same_team_next` was a convenient default, no guarded reason; safe to replace, keep as residual stopgap. Per sub-domain:
> - **GK-pass: `pass_outcome` present 100%** (succ 0.807, both classes) → **train fully on `native`; zero residual, no stopgap needed.**
> - **goal-kick: `pass_outcome` 59.5%** (succ 0.699, both classes) → train on the native 59.5%; **40.5% residual → `stopgap` (`same_team_next`, agreement 0.717 vs true completion)** for VAEP `result_id` coverage, **training-excluded (G1)**.
> - **`next==player_targeted_id` NOT promoted into `inferred`** — agreement 0.44–0.55 ≈ chance. `inferred` = `received==True` only (rare on pass-event rows, `rec_rate≈0`; training-excluded anyway, G1).
> - Confirms the VAEP-retrain-trigger is a real correction: old all-`same_team_next` overstated goal-kick success (0.858 vs true 0.699, **+16 pp**).

---

## Phase 1 — SkillCorner `result_id` native-completion fix (foundational)

### Task 2: Native-completion helper (pure, TDD)

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py` (add a module-level helper near `_is_cross`)
- Test: `tests/spadl/test_skillcorner_completion.py` (create)

- [ ] **Step 1: Write the failing test.**

```python
# tests/spadl/test_skillcorner_completion.py
import numpy as np, pandas as pd
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.skillcorner import _native_completion_result

S, F = spadlconfig.result_id["success"], spadlconfig.result_id["fail"]

def test_pass_outcome_primary_and_received_success_only():
    # rows: 0 native success, 1 native unsuccessful, 2 offside, 3 no pass_outcome+received=True,
    # 4 no pass_outcome + received=False but same_team_next=True (N1: must NOT be forced fail),
    # 5 neither native field, same_team_next=False
    pass_outcome = pd.Series(["successful", "unsuccessful", "offside", np.nan, np.nan, np.nan])
    received = pd.Series([np.nan, np.nan, np.nan, True, False, np.nan])
    same_team_next = pd.Series([True, True, False, False, True, False])
    is_passlike = pd.Series([True, True, True, True, True, True])
    rid, src = _native_completion_result(pass_outcome, received, same_team_next, is_passlike)
    assert rid[0] == S and src[0] == "native"          # pass_outcome successful
    assert rid[1] == F and src[1] == "native"          # unsuccessful
    assert rid[2] == F and src[2] == "native"          # offside -> fail
    assert rid[3] == S and src[3] == "inferred"        # received True -> success (clean signal, NOT native pass_outcome)
    # N1 regression: row 4 has received=False but is NOT routed to fail by received;
    # it falls to the residual (same_team_next=True here) -> success, tagged stopgap.
    assert rid[4] == S and src[4] == "stopgap"
    assert rid[5] == F and src[5] == "stopgap"
```

- [ ] **Step 2: Run it to verify it fails.** Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_skillcorner_completion.py -q` → FAIL (`_native_completion_result` undefined).

- [ ] **Step 3: Implement the helper.** Add to `silly_kicks/spadl/skillcorner.py` (residual branch = the Task-1 decision; the skeleton below uses `same_team_next` stopgap — swap to the `player_targeted_id` reconstruction if Task 1 chose it):

```python
def _native_completion_result(
    pass_outcome: pd.Series,
    received: pd.Series,
    same_team_next: pd.Series,
    is_passlike: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-construct pass completion (spec N1). pass_outcome PRIMARY (SPADL 'reached a
    teammate'); received==True success-ONLY (received==False NEVER -> fail); residual ->
    flagged stopgap (same_team_next). Returns (result_id_array, result_source_array) where
    result_source ∈ {native (pass_outcome), inferred (clean success signal: received==True;
    Phase 0 may add next-action==player_targeted_id), stopgap (same_team_next residual)}.
    Only {native, inferred} are clean enough for GK-completion training (F1); stopgap keeps
    the proxy value for VAEP result_id coverage but is excluded from training."""
    n = len(pass_outcome)
    rid = np.full(n, spadlconfig.result_id["fail"], dtype=int)
    src = np.full(n, "stopgap", dtype=object)
    po = pass_outcome.astype("string").str.lower()
    has_po = po.isin(["successful", "unsuccessful", "offside"]).to_numpy()
    po_succ = (po == "successful").to_numpy()
    rec_true = received.astype("string").str.lower().isin(["true", "1", "1.0"]).to_numpy()
    stn = same_team_next.fillna(False).to_numpy()
    passlike = is_passlike.fillna(False).to_numpy()
    # tier 1: native pass_outcome
    rid = np.where(passlike & has_po & po_succ, spadlconfig.result_id["success"], rid)
    rid = np.where(passlike & has_po & ~po_succ, spadlconfig.result_id["fail"], rid)
    src = np.where(passlike & has_po, "native", src)
    # tier 2: received==True (clean success-only augmentation) -> tagged "inferred" (F1: training-eligible,
    # but distinct from native pass_outcome). received==False is NOT a fail signal (N1).
    rec_aug = passlike & ~has_po & rec_true
    rid = np.where(rec_aug, spadlconfig.result_id["success"], rid)
    src = np.where(rec_aug, "inferred", src)
    # tier 3 (residual): flagged stopgap via same_team_next  (received==False is NOT a fail signal)
    resid = passlike & ~has_po & ~rec_true
    rid = np.where(resid & stn, spadlconfig.result_id["success"], rid)
    rid = np.where(resid & ~stn, spadlconfig.result_id["fail"], rid)
    src = np.where(resid, "stopgap", src)
    return rid, src
```

- [ ] **Step 4: Run the test to verify it passes.** Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_skillcorner_completion.py -q` → PASS.

- [ ] **Step 5: Commit.** `git add silly_kicks/spadl/skillcorner.py tests/spadl/test_skillcorner_completion.py` then commit `feat(skillcorner): native-completion result helper (single construct, N1)` — **HOLD for sentinel approval per the commit-sentinel rule.**

### Task 3: Wire the helper into the converter result dispatch

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py:260-352` (result section) + the actions DataFrame build (`:354+`)
- Test: `tests/spadl/test_skillcorner_completion.py`

- [ ] **Step 1: Write the failing integration test** (build a tiny `dynamic_events`-shaped DataFrame through `convert_to_actions`; assert a `pass_outcome=='successful'` pass → `result_id==success` + `result_source=='native'`, and that a `clearance`/`foul`/`shot` row's `result_id` is unchanged from the pre-fix value — Chesterton scope guard). Run → FAIL.

- [ ] **Step 2: Surface the raw native columns** (near `:243-245`, mirroring the optional-column idiom):

```python
pass_outcome_col = pp["pass_outcome"] if "pass_outcome" in pp.columns else pd.Series(np.nan, index=pp.index)
received_col = pp["received"] if "received" in pp.columns else pd.Series(np.nan, index=pp.index)
```

- [ ] **Step 3: Replace the `same_team_next` result branches** (`:334-352`). Keep the explicit clearance/foul/shot/goal branches; compute pass/set-piece completion via the helper:

```python
is_passlike = pd.Series(
    np.isin(type_id_arr, [
        spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"],
        spadlconfig.actiontype_id["goalkick"], spadlconfig.actiontype_id["throw_in"],
        spadlconfig.actiontype_id["corner_short"], spadlconfig.actiontype_id["corner_crossed"],
        spadlconfig.actiontype_id["freekick_short"], spadlconfig.actiontype_id["freekick_crossed"],
    ]),
    index=pp.index,
)
passlike_rid, passlike_src = _native_completion_result(pass_outcome_col, received_col, same_team_next, is_passlike)
is_goal = gi_after == "goal_for"
result_id_arr = np.select(
    [is_clearance, is_foul, is_shot & is_goal, is_shot & ~is_goal, is_passlike.to_numpy()],
    [spadlconfig.result_id["success"], spadlconfig.result_id["success"],
     spadlconfig.result_id["success"], spadlconfig.result_id["fail"], passlike_rid],
    default=spadlconfig.result_id["fail"],
)
# result_source: native/inferred/stopgap for passlike rows; "native" for the explicit deterministic branches
result_source_arr = np.where(is_passlike.to_numpy(), passlike_src, "native")
```

- [ ] **Step 4: Add `result_source` to the actions DataFrame** (`:356+`): `"result_source": result_source_arr,`.

- [ ] **Step 5: Run.** `.venv/Scripts/python.exe -m pytest tests/spadl/test_skillcorner_completion.py -q` → PASS.

- [ ] **Step 6: Commit** `feat(skillcorner): result_id native completion + result_source (D-S8)` — **HOLD for sentinel.**

### Task 4: Schema — add `result_source`

**Files:**
- Modify: `silly_kicks/spadl/schema.py:83` (`SKILLCORNER_SPADL_COLUMNS`) + `_finalize_output` if it enforces the column set
- Test: `tests/spadl/test_skillcorner_completion.py`

- [ ] **Step 1: Failing test** — assert `convert_to_actions(...)` output has a `result_source` column of dtype object with values ⊆ `{"native","inferred","stopgap"}`. Run → FAIL.
- [ ] **Step 2: Add** `"result_source": "object",` to `SKILLCORNER_SPADL_COLUMNS`.
- [ ] **Step 3: Run** → PASS.
- [ ] **Step 4: Commit** `feat(skillcorner): result_source column in SKILLCORNER_SPADL_COLUMNS` — **HOLD for sentinel.**

### Task 5: Regenerate SkillCorner goldens + flag the VAEP-retrain trigger

**Files:**
- Modify: `tests/regressions/extratime/golden_skillcorner_*.parquet` (regenerate), the regen script if one exists
- Modify: `CHANGELOG.md` (retrain-trigger note)

- [ ] **Step 1: Identify the SkillCorner golden(s).** Run: `python -c "import glob; print(glob.glob('tests/regressions/**/golden_skillcorner*', recursive=True))"`
- [ ] **Step 2: Run the golden test to see the diff.** Run the SkillCorner golden/regression test; confirm the **only** changed columns are `result_id` (pass/set-piece rows) + the new `result_source` — non-pass/defensive `result_id` unchanged (Chesterton scope). Eyeball the success-rate shift (expect goal-kick success ≈ 0.7–0.8, not 1.0).
- [ ] **Step 3: Regenerate** via the committed regen mechanism (NOT hand-edited). Verify with `git diff --stat`.
- [ ] **Step 4: Add a CHANGELOG entry** under 4.21.0 marking the **VAEP-retrain trigger** (SkillCorner scores/concedes label distribution shift; lakehouse re-materializes).
- [ ] **Step 5: Commit** `test(skillcorner): regen golden for native-completion result_id (VAEP retrain trigger)` — **HOLD for sentinel.**

---

## Phase 2 — Provider-aware variant selection

### Task 6: Pure `variant_key_for_provider`

**Files:**
- Modify: `silly_kicks/tracking/_gk_completion.py` (near `from_variant`, `:202`)
- Test: `tests/tracking/test_gk_completion_variants.py` (create)

- [ ] **Step 1: Failing test** (exhaustive over the 5 enum values + None/unknown, artifact-free):

```python
# tests/tracking/test_gk_completion_variants.py
import pytest
from silly_kicks.tracking._gk_completion import variant_key_for_provider

@pytest.mark.parametrize("provider,expected", [
    ("skillcorner", "skillcorner"),
    ("gradientsports", "gs"), ("sportec", "gs"), ("snapshot", "gs"),
    ("metrica", "gs"), (None, "gs"), ("unknown_x", "gs"),
])
def test_variant_key_for_provider(provider, expected):
    assert variant_key_for_provider(provider) == expected
```

- [ ] **Step 2: Run** → FAIL (undefined).
- [ ] **Step 3: Implement** (pure, no artifact IO):

```python
_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}  # everything else -> the native-completion gs model

def variant_key_for_provider(source_provider: str | None) -> str:
    """Map a tracking source_provider to a completion-variant key (spec C4 — pure, no IO).
    gs/sportec/snapshot/metrica/unknown/None -> 'gs' (all native-completion construct once
    result_id is fixed); 'skillcorner' -> its own key (weights may equal gs — D-S1 re-measure)."""
    return _PROVIDER_VARIANT.get(str(source_provider).lower() if source_provider is not None else "", "gs")
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gk-completion): pure variant_key_for_provider (C4)` — **HOLD for sentinel.**

### Task 7: Auto-resolve the variant in `compute_xt_gk` / `_completion_p`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py:255-279` (`_completion_p`) + `compute_xt_gk` (`:282`)
- Test: `tests/tracking/test_gk_completion_variants.py`

> **m2 — re-verify line cites before editing.** `_xt_gk.py` cites (`:42` `_PROVENANCE_COLS`, `:45` `XtGkReport`, `:255-279` `_completion_p`, `:274` hardcoded `from_variant("default")`, `:282` `compute_xt_gk`, `:319` end of fitted-guard, `:325-341` provenance block) were accurate at plan time but line numbers drift across the Phase-1/2 edits — `grep`/re-read the symbol before each surgical edit, don't trust the number.

- [ ] **Step 1: Failing tests:** (a) SkillCorner frames + `completion=None` → the `skillcorner` model is resolved (monkeypatch `from_variant` to record the key); (b) GS frames → `"gs"`/`"default"`; (c) an explicit `completion=GkCompletionModel(...)` instance beats auto-selection; (d) frames mixing two real providers → `ValueError`; (e) frames mixing `snapshot` + one real provider → no raise. Run → FAIL.

- [ ] **Step 2: Add provider resolution** at the top of `compute_xt_gk` (after the `xt` fitted-guard, `:319`):

```python
from ._gk_completion import variant_key_for_provider
if not isinstance(completion, GkCompletionModel) and completion is None:
    provs = [p for p in pd.unique(frames["source_provider"].dropna()) if str(p).lower() != "snapshot"]
    if len(provs) > 1:
        raise ValueError(f"xT-GK: frames span multiple real providers {provs}; one call = one match = one provider.")
    completion = _resolve_completion_variant(provs[0] if provs else None)  # see Step 3
```

- [ ] **Step 3: Add the IO seam** in `_gk_completion.py` (thin, keeps Task-6 mapping pure) + fall back to `gs` with a warning where a `skillcorner` artifact is absent (D-S1 may not bundle distinct weights):

```python
def _resolve_completion_variant(source_provider):  # in _xt_gk.py, lazy import
    from ._gk_completion import GkCompletionModel, variant_key_for_provider
    key = variant_key_for_provider(source_provider)
    try:
        return GkCompletionModel.from_variant(key)
    except FileNotFoundError:
        if key != "gs":
            import warnings
            warnings.warn(f"no bundled Gk-completion weights for {key!r}; falling back to 'default' (gs).", stacklevel=2)
        return GkCompletionModel.from_variant("default")
```

Update `_completion_p` (`:274`) to accept an already-resolved model (callers pass the resolved `completion`), so it no longer hardcodes `from_variant("default")` when `compute_xt_gk` resolved a provider model. Keep the `default` fallback for direct `_completion_p` callers.

- [ ] **Step 4: Run** → PASS. Also run the existing xT-GK suite to confirm no regression: `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "xt_gk" -q`.
- [ ] **Step 5: Commit** `feat(xt-gk): provider-aware completion variant auto-selection (D-S2)` — **HOLD for sentinel.**

---

## Phase 3 — Provenance columns

### Task 8: `xt_gk_completion_variant` + `xt_gk_completion_source` + `spans_multiple_variants`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (`_PROVENANCE_COLS` `:42`, `compute_xt_gk` provenance block `:325-341`, `XtGkReport` `:45`), atomic mirror
- Test: `tests/tracking/test_gk_completion_variants.py` + the existing provenance test

- [ ] **Step 1: Failing test:** scored SkillCorner rows carry `xt_gk_completion_variant == "skillcorner"` and `xt_gk_completion_source ∈ {"model","base_rate"}` (m2: `model` for geometry-present rows, `base_rate` only geometry-missing — in the RAV path geometry-missing → NaN, so expect `model` for scored rows); `XtGkReport.from_frame(out).spans_multiple_variants is False` for single-provider, `True` for a manually-concatenated two-variant frame. Run → FAIL.

- [ ] **Step 2: Emit the columns** — add `"xt_gk_completion_variant"`, `"xt_gk_completion_source"` to `_PROVENANCE_COLS`; populate in `compute_xt_gk` (`variant` = the resolved key; `source` = `"model"` for scored rows). Mirror in the atomic xT-GK path.

- [ ] **Step 3: Extend `XtGkReport`** with `spans_multiple_variants: bool` + populate in `from_frame` (`df["xt_gk_completion_variant"].nunique(dropna=True) > 1`).

- [ ] **Step 4: Run** → PASS; run the provenance-skip idempotence + atomic-mirror parity gates.
- [ ] **Step 5: Commit** `feat(xt-gk): completion variant/source provenance + spans_multiple_variants (H1/m-c)` — **HOLD for sentinel.**

---

## Phase 3.5 — Train the completion model on the unbiased native label only (F1 + G1)

### Task 9: `result_source`-aware training filter in `prepare_gk_completion_training_data`

**Why (review F1 + G1):** `prepare_gk_completion_training_data` derives `y = (result_id == success)` uniformly (`_gk_completion.py:283`). Two problems if unfiltered:
- **F1:** the `stopgap` (`same_team_next` proxy) rows — ~40% of goal-kicks — are noise-labeled, depressing the goal-kick AUC the gate reports.
- **G1 (the calibration trap):** the `inferred` tier (`received==True`, and any promoted next-action==`player_targeted_id`) is **structurally positive-only** — these are *confirm-completion* signals (clean successes, no clean fails; `received==False` was ruled out as a fail signal, N1). So every `inferred` row is `y=1`. Mixing them with `native` (`pass_outcome`, which supplies **both** classes from one definition) pushes the training positive rate **above** the true completion rate → the logistic intercept shifts high → **systematically over-predicted `p`** → **mis-calibrated**. Since calibration is the *primary* hard gate (consumed multiplicatively in RAV), a positive-only tier in the training set is precisely the failure the gate exists to catch.

**Fix:** train the GK-completion model on **`result_source == "native"` only** — `pass_outcome` is the one construct that yields both success *and* fail from a single rule, hence unbiased. `result_id` keeps the `inferred`/`stopgap` values for **VAEP coverage** (D-S8 uniform), but **both are excluded from xT-GK completion training**. This costs GK-pass training coverage (`pass_outcome` ~35% there) — acceptable: an unbiased smaller set beats a calibration-biased larger one for a probability scored on scale. Provider-agnostic: GS actions have no `result_source` column → no-op. (This also moots F2's inferred-promotion question *for training* — positive-only signals never train; they only enrich VAEP `result_id`.)

**Files:**
- Modify: `silly_kicks/tracking/_gk_completion.py:252-298` (`prepare_gk_completion_training_data`, slot into the existing `keep` drop at `:289`)
- Test: `tests/tracking/test_gk_completion_variants.py`

- [ ] **Step 1: Write the failing test.**

```python
def test_training_uses_native_label_only():
    import numpy as np, pandas as pd
    from silly_kicks.tracking._gk_completion import prepare_gk_completion_training_data
    from silly_kicks.spadl import config as spadlconfig
    S, F = spadlconfig.result_id["success"], spadlconfig.result_id["fail"]
    # 4 goalkicks, all geometry/id-scoreable: 2 native (1 success, 1 fail), 1 inferred (positive-only),
    # 1 stopgap. Only the 2 native rows may train (G1: inferred is positive-only -> calibration bias).
    actions = _make_goalkick_fixture(  # helper: 4-row goalkick frame w/ start/end coords + game_id
        result_id=[S, F, S, S],
        result_source=["native", "native", "inferred", "stopgap"],
    )
    X, y, groups = prepare_gk_completion_training_data(actions, frames=None)
    assert len(y) == 2                       # inferred + stopgap dropped; only native trains
    assert set(y) == {0, 1}                  # native supplies BOTH classes
    assert "result_source" not in X.columns  # not a feature
```

- [ ] **Step 2: Run** → FAIL (inferred + stopgap currently kept; len==4).

- [ ] **Step 3: Implement the filter** — at `:289`, extend `keep` (after `geom_ok & id_ok`):

```python
keep = geom_ok & id_ok
# F1+G1: train ONLY on the native (pass_outcome) label -- the one construct giving both classes.
# inferred (received==True / next-action==targeted) is positive-only -> would bias the intercept high
# -> mis-calibrate p (the primary hard gate). stopgap is the weak proxy. result_id keeps both values
# for VAEP coverage; the completion model never trains on either. result_source is a SkillCorner-only
# column (absent for GS -> no-op).
if "result_source" in domain.columns:
    keep = keep & (domain["result_source"] == "native").to_numpy()
```

(The `min_class_fraction` degenerate guard at `:293` then runs on the native subset. **m4:** it runs on the *combined* GK-distribution domain (goal-kicks + GK-passes), so goal-kick sparsity alone won't trip it — GK-passes keep the combined label two-class. The goal-kick model-vs-base-rate decision is the **Task 10 LCB-AUC gate**, not a `prepare_*` raise.)

- [ ] **Step 4: Run** → PASS. Also a no-op guard: a fixture WITHOUT a `result_source` column (GS-shaped) keeps all scoreable rows (assert len unchanged) — proves provider-agnostic.

- [ ] **Step 5: Commit** `feat(gk-completion): train completion model on native label only (F1+G1)` — **HOLD for sentinel.**

---

## Phase 4 — Training & gates (owner/public-run — NOT CI)

**These run against the SkillCorner pining corpus via `_loader_pining` (public token). They are the training-time green criteria, not PR gates. CI exercises the synthetic fixtures from Phases 1–3.**

### Task 10: `train_gk_completion.py --variant skillcorner` + GS-transfer re-measurement (D-S1)

**Files:**
- Modify: `scripts/train_gk_completion.py`
- Output: `silly_kicks/tracking/_gk_completion_weights/skillcorner/` (model.json + SHA256SUMS + metrics.json) **iff** GS-transfer fails the floor

- [ ] **Step 1:** Add `--variant skillcorner`: load the SkillCorner pining corpus (post-Task-3 corrected `result_id` + `result_source`), call `prepare_gk_completion_training_data` — which (Task 9, F1) now **excludes `stopgap`-labelled rows** from training, so the model learns only on the clean `{native, inferred}` subset. Report the clean-label count per sub-domain (goal-kicks especially) in metrics.json — a too-small clean goal-kick sample is an honest "base-rate-serve goal-kicks" signal, not a reason to re-admit noise.
- [ ] **Step 2: D-S1 re-measurement (decisive):** report **GS `default` model served on the corrected-label SkillCorner held-out** — AUC + calibration (ECE, reliability-slope), match-grouped, on the GK-pass sub-domain. Compare to a SkillCorner-fit (GroupKFold OOF).
  - If GS-transfer clears the gate (AUC ≥ 0.70 GK-pass **and** calibration within tolerance vs the GS reliability target) → **do not bundle distinct weights**; `from_variant("skillcorner")` resolves to `gs` (register the alias). Record in metrics.json.
  - Else → fit + bundle SkillCorner-specific weights.
- [ ] **Step 3: Gates (common-scale, m3):** calibration ECE + reliability-slope checked against the **GS variant's** reliability target (lakehouse C4); GK-pass held-out AUC ≥ 0.70; goal-kicks scored only if a **lower-confidence-bound** AUC clears the floor (report n). Write all to metrics.json.
- [ ] **Step 4 (smoke, CI-able):** a `@pytest.mark.slow` does-it-run smoke on a tiny synthetic corpus (no network) asserting the `--variant skillcorner` code path executes + writes metrics.json keys. (The real numbers are owner-run.)
- [ ] **Step 5: Commit** `feat(train): --variant skillcorner + GS-transfer re-measurement (D-S1)` (+ bundled weights iff fitted) — **HOLD for sentinel.**

### Task 11: Cross-provider comparability gate (D-S9, review F4)

**Files:**
- Create: `scripts/_xtgk_comparability.py` (owner-run)
- Modify (only in the rare evidence-backed re-scale case, per G2): `silly_kicks/tracking/_xt_gk.py` (per-variant post-composite `xt_gk` affine, clamped)

- [ ] **Step 1:** Implement `compare_xtgk_distributions(sc_xtgk, gs_xtgk, *, bands, min_n)` → per matched distance/zone band: mean/quantile offset + n; bands with `n < min_n` flagged **under-powered** (reported, not silently passed).
- [ ] **Step 2: Contingency (N4 + F4 + G2) — post-calibration, expect `within_tolerance` OR `escalate`, NOT `correctable`.** Task 10 already common-scale-calibrates `p` (reliability vs the GS target). After both variants' `p` are calibrated to the same reliability, a residual `xt_gk` offset is **by elimination the threat-term difference** — i.e. **genuine football** (SC's ~17 m goal-kicks really do face different geometry), which F4 says must **not** be erased. So:
  - `within_tolerance` → pool directly (calibration handled the scale).
  - `escalate` (**the default** for any residual offset) → document the difference, do **not** auto-conform SC to GS; surface to the maintainer.
  - `correctable` → **rare; only with strong documented evidence the offset is a measurement ARTIFACT** (e.g. a known tracking-calibration offset uniform across *all* bands) — not merely "distributions differ."
- [ ] **Step 3: Where a re-scale lives (G2 — corrected from rev-1).** `xt_gk = p·xT★(z′) − δ(1−p)·xT★(counter)` is **nonlinear in `p`** and depends on the threat-grid terms, so an affine on `p` (in `predict_proba`) does **not** produce a known affine on `xt_gk` *and* would silently undo the Task-10 calibration certified on the un-rescaled `p`. Therefore, in the rare `correctable` case, the re-scale is a **post-composite affine on `xt_gk` itself**, applied **per-variant in `_xt_gk.py`** (after the composite), **clamped to a sane `xt_gk` range (m5)** so a fitted `a·x+b` can't push values out of domain into a downstream consumer. It is baked into the silly-kicks output (not emitted for the lakehouse to apply), so the lakehouse still pools raw. **Default path applies no re-scale** (escalate or within_tolerance).
- [ ] **Step 4:** Document the run + verdict in `docs/research/xtgk_comparability/` (REPORTED, owner-run; not a CI gate). The lakehouse consumes the verdict before pooling (sequence step 5).
- [ ] **Step 5: Commit** `feat(xt-gk): cross-provider comparability gate (D-S9, N4/F4/G2; default escalate, no re-scale)` — **HOLD for sentinel.**

---

## Phase 5 — Docs, C4, version, final

### Task 12: Docs + ADR + C4 + version bump

**Files:** `docs/.../adrs/ADR-024*` (amend), `CHANGELOG.md`, `CLAUDE.md`, model card under the weights dir, `docs/c4/architecture.dsl`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `uv.lock`

- [ ] **Step 1: Amend ADR-024** — SkillCorner `result_id` native-completion fix (D-S8, VAEP-retrain trigger), provider-aware variant family (D-S1/D-S2), pooling gates (D-S7/D-S9), no-pool-without-comparability contract (H1). **m1 — qualify the headline honestly:** "completion corrected to native **where native fields exist**; residual rows keep a **flagged `stopgap`** (`same_team_next`) value for VAEP coverage" — do NOT claim "correct for every consumer" unqualified (the `result_source` flag makes the tiers visible; the claim must not outrun it). Note the GK-completion model trains only on `{native, inferred}` (F1).
- [ ] **Step 2: CLAUDE.md** — one line under the SkillCorner converter note: `result_id` now native completion (`pass_outcome`→`received`→stopgap) + `result_source`; **VAEP-retrain trigger**. Note the xT-GK completion variant family.
- [ ] **Step 3: Model card** (`_gk_completion_weights/*/`) — state the label construct (native completion via corrected `result_id`), per-variant coverage, the comparability/calibration gates.
- [ ] **Step 4: C4 check** — the tracking-container enumeration lists trained models + aggregator COUNT. A new completion *variant* is **not** a new aggregator and the GK-completion model already exists → confirm tokens/count unchanged; **skip regen if so** (per the C4 drift rule). The converter change is in `spadl/` (not enumerated). Document the check outcome.
- [ ] **Step 5: Version bump** — confirm 4.21.0 across `pyproject.toml` + `__init__` + CHANGELOG + TODO; `uv lock`. (Per the every-PR-bumps convention; 4.21.0 is the in-flight branch.)
- [ ] **Step 6: Commit** `docs: ADR-024 amendment + CLAUDE/model-card/C4 for SkillCorner completion` — **HOLD for sentinel.**

### Task 13: Full suite + lint + final-review

- [ ] **Step 1: Lint (full CI parity):** `ruff check silly_kicks/ tests/ scripts/`; `ruff format --check silly_kicks/ tests/ scripts/`; `pyright silly_kicks/`.
- [ ] **Step 2: Full suite:** `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q` → all green (report the count; do not narrate from memory).
- [ ] **Step 3: Run `/final-review`** (code/doc/C4 consistency; the result_id-fix is ADR-worthy → confirm ADR-024 amendment present).
- [ ] **Step 4: Final gated commit + PR** — present command + diff, HOLD for sentinel; bare push; `gh pr create` with the VAEP-retrain-trigger + cross-repo-sequencing (N3) called out in the body. Merge/tag/publish per the release checklist after CI passes (user-gated).

---

## Self-Review (against the spec)

- **Spec coverage:** D-S8 (Task 2–5), D-S1 re-measure (Task 10), D-S2 selection (Task 6–7), H1/m-c provenance (Task 8), D-S9/N4/F4 comparability (Task 11), N1 single-construct + `received`-success-only (Task 2 test), N2 (docs Task 12), N3 sequencing (phase order + PR body), m1 offside (Task 2/10).
- **Review-4 (rev-1) fixes:** **F1** training excludes `stopgap` (Task 9, CI-tested), **F2** `inferred` precisely defined = clean success signals, F1 de-risks it, **F3** residual policy per sub-domain (Phase 0), **F4** re-scale requires artifact-evidence else escalate, **m1** ADR headline qualified, **m2** re-verify line cites, **m3** LCB goal-kick gate.
- **Review-4 (rev-2) fixes:** **G1** train on `result_source == "native"` ONLY — `inferred` is positive-only (no clean fail) → would bias the intercept high → mis-calibrate the multiplicatively-consumed `p`; only `pass_outcome` gives both classes (Task 9, CI-tested with a 4-tier fixture; `inferred`/`stopgap` stay in `result_id` for VAEP only). **G2** the comparability re-scale (a) cannot live in `predict_proba` (`xt_gk` is nonlinear in `p` + would undo the certified calibration) → relocated to a per-variant **post-composite affine on `xt_gk`** in `_xt_gk.py`, and (b) is **mostly vestigial after common-scale calibration** → default `escalate`, `correctable` only on documented measurement-artifact evidence (Task 11). **m4** degenerate guard runs on the combined domain → goal-kick fallback is the Task-10 LCB gate, not a `prepare_*` raise (Task 9 wording). **m5** clamp any affine output to a sane range.
- **Sequence (N3):** Phase 1 (result_id) → Phase 3.5 (training filter) → Phase 4 Task 10 (retrain/refit + GS-transfer + gates) → Task 11 comparability → lakehouse re-materialize+pool. CI-testable work (Phases 1–3.5) is separable from owner-run gates (Phase 4).
- **Type consistency:** `variant_key_for_provider(str|None)->str`; `_resolve_completion_variant` returns `GkCompletionModel`; `_native_completion_result(...)->(np.ndarray,np.ndarray)`; `result_source ∈ {native,inferred,stopgap}`; training keeps `{native,inferred}`; `xt_gk_completion_source ∈ {model,base_rate}`. Consistent across tasks.
- **Open input:** Phase 0 decides only whether the next-action-targeted signal is clean enough to promote into `inferred` (per sub-domain) — the residual stays `same_team_next`/`stopgap` regardless (training-excluded), so there is no longer a load-bearing build-time fork.

---

## Execution Handoff

Plan saved. **Per the user's instruction, this goes to the lakehouse session for review before execution.** After review + any revisions, execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.
2. **Inline** — execute in this session with checkpoints.
