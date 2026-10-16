# TF-79 follow-up: ball-trajectory dedup + z-provenance — design

**Status:** DRAFT (awaiting review) — **AMENDED 2026-10-15 (owner-ruled): §11 adds z-provenance + `is_aerial` honest-NaN (the spike's ball-z finding) INTO this cycle; re-review covers §11.**
**Deciders:** Karsten
**ADR (at cycle end):** ADR-125 *(provisional — next free after the frozen part-deux hf-jobs cycle holding 122/123 and TF-79's 124; confirm free at cycle end, do not hard-claim early)*
**Spike that motivates this:** the post-4.136.0 ball-contact provider spike (results summarised in sec. 2).

---

## 1. Problem

The TF-79 966-match validation (4.136.0) reported ball-contact detection **degraded on Gradient Sports** (event->contact agreement @0.5 s ≈ 5 %, vs SkillCorner 79 %, IDSSE 45 %). The 4.136.0 docs attributed this to a **"blunted / pre-smoothed ball."** That attribution is **wrong** and is corrected here.

## 2. Root cause (spike-verified)

- At **real touches** (event-anchored, raw x/y), the GS ball velocity-vector change is **|Δv| p50 ≈ 3.0 m/s — as sharp as IDSSE (3.09)**. The touch signal is present; the ball is not blunted. The "|Δv| ≈ 1" that motivated the blunted-ball claim was measured at *detected contacts* — a biased sample of the detector's own low-|Δv| over-fires, not real touches.
- GS **under-detects**: ≈ 0.65 contacts/event vs IDSSE ≈ 3.7.
- Mechanism: `silly_kicks/tracking/_ball_contact.py::_velocity_breaks` computes per-frame velocity by **raw consecutive-frame differencing** (`vel = np.diff(xy) / dt`). Ball tracking carries **~50 % consecutive-duplicate positions** (IDSSE 42 %, GS 50 %). A duplicate frame yields `speed = 0`; the next real frame yields a jump → a **0↔real velocity sawtooth** that masks the real touch's speed-drop. IDSSE has the same duplicates but slow per-frame motion, so its sawtooth is negligible; GS's faster motion (29.97 fps) makes it fatal. Frame **gaps are negligible** (0.1 %) — this is distinct from the ADR-120 gap problem; it is its **sibling** (frame *duplicates* corrupting velocity, vs frame *gaps*).
- **Fix validated (5 GS + 2 IDSSE):** collapsing consecutive-duplicate ball positions before the break legs recovers GS agreement @0.5 s from **0.036–0.062 to 0.333–0.467 (~8–10×)** — GS lands at IDSSE's level; contact density rises ~900→~4000 (IDSSE-like). **Harmless to IDSSE** (0.437→0.435, −9 contacts). So GS is a **detector-conditioning bug, not a degraded provider.**

## 3. Decision

1. **Provider-agnostic.** Apply the dedup unconditionally — every provider carries duplicate ball frames, the IDSSE control shows it is harmless to the good providers, and a provider branch + a detection-aware guard is avoidable complexity. (Confirmed with owner.)
2. **Placement = a pure, reusable dedup helper called by `detect_ball_contacts`** (not buried in the detector; not folded into the ADR-120 preprocess-velocity path). Rationale, verified in-code:
   - `detect_ball_contacts` reads **only raw positions** (`required = {is_ball, period_id, frame_id, time_seconds, x, y}`; no velocity column in its contract).
   - **RDP is intrinsically position-geometric** — it simplifies the position polyline and cannot consume a derived-velocity column — so the detector *must* read raw positions; the velocity-break and |Δv| legs difference the **same** raw path for coherence.
   - Therefore consuming the ADR-120 derived velocity (option "C") is **ruled out**: RDP demands raw positions, and splitting legs across raw (RDP/|Δv|) and derived (velocity-break) would be incoherent.
   - A single dedup of the raw trajectory, feeding all three legs, is the DRY, coherent fix. The duplicate-frame artifact is a **general, measured** frames property (not speculative surface), so a shared helper — the sibling of ADR-120's gap handling — is justified.

## 4. Design

### 4.1 The helper (pure)

A pure function that returns a **keep-mask** over a per-period ball trajectory:

```
trajectory_dedup_mask(x: np.ndarray, y: np.ndarray, *, tol_m: float = 1e-6) -> np.ndarray  # bool
```

- Keep index 0; keep index `i>0` iff the point moved more than `tol_m` from the **immediately preceding** point: `hypot(x[i]-x[i-1], y[i]-y[i-1]) > tol_m`. This collapses *runs* of consecutive identical positions to one, while leaving a genuine A→B→A oscillation intact.
- `tol_m` default `1e-6` m: coords are **float32 storage** (ADR-106), so true consecutive duplicates differ by exactly 0; a sub-µm tol catches them without touching real motion. (Not a smoothing threshold — it only removes non-moving repeats.)
- **Home:** a private module in the frame-conditioning seam (proposed `silly_kicks/tracking/preprocess/` as the ADR-120 sibling, or `silly_kicks/_frame_index.py`; **PRIVATE — no public `__all__` addition**, so no public-surface gate cascade). Final module pinned in the plan.

### 4.2 Wiring into `detect_ball_contacts`

Per period, immediately after `seg_df = ball_by_period.get(pid).sort_values("frame_id")` and the `fid/tt/bx/by/bz/fr` extraction (currently lines ~370–376):

1. Compute `keep = trajectory_dedup_mask(bx, by)`.
2. Apply `keep` to **all** per-period arrays together (`fid, tt, bx, by, bz`) → one coherent dedup'd trajectory for `_segments`, `_rdp`, `_velocity_breaks`, `_velocity_change_at`, and attribution.
3. The representative `frame_id` on a contact row is therefore a **kept, real** ball frame; `players_by_frame.get(pid, int(fid[gp]))` attribution is unchanged.

### 4.3 Window semantics (must-handle subtlety)

Frame-count windows (`dvw = round(0.12 / frame_period)`, `w = round(0.2 / frame_period)`) count **array steps**. After dedup the array is sparser, so a fixed frame-count spans more wall-clock. The fix must keep these windows spanning their intended **seconds** on the dedup'd sampling — i.e. derive them from the dedup'd local `dt` (time-based), not the original `frame_period`. `frame_period` for the merge window stays from `frame_rate` (cadence is unchanged). The plan pins the exact recomputation; a wrong window here silently changes |Δv| and detection-quality.

### 4.4 Edge cases (documented)

- **All-duplicate segment** (ball never moves): collapses to 1 point → `< 2` points → segment skipped → no contact. Correct.
- **Single-frame period:** unchanged (already skipped).
- **z-only vertical bounce at fixed (x, y):** collapsed to one point; the 2-D velocity/RDP legs do not use z-motion, and attribution uses the kept frame's z — a rare, documented loss.

## 5. Blast radius / Hyrum

- The contact table **schema is unchanged** (same columns) *for the dedup half*. **AMENDED BY §11.4/§11.6:** §11 adds `aerial_source` to the contacts table + `z_source` to the frames schema (+2 `feature_glossary` columns, +1 frame-schema column). Still **no new public `add_*`/aggregator**, so the mirror-registry SIZE and the c4 aggregator-count are unchanged — it is the frame-schema (ADR-058), `feature_glossary`, category-dtype, and SB360-verdict gates that move (§11.6).
- **Values change on every provider** (GS massively; IDSSE/SkillCorner slightly — e.g. IDSSE −9 contacts/match). **Byte-identical is NOT claimed** for any provider. This is the intended correctness improvement.
- **No retrain** — `detect_ball_contacts` is a pure, model-free primitive.
- **Downstream:** TF-46 (set-piece, not yet built) will consume the improved first-contact; the deferred PathCRF item likewise. The committed 966-match memo regenerates (sec. 6). TF-48 untouched (Chesterton's Fence).

## 6. Validation (reported-not-gated, unchanged stance)

- Re-run `scripts/measure_ball_contact_validation.py` over the full owner-tier corpus on the canonical pining compute host; parallelised per the established for_each fan-out. No accuracy threshold — numbers REPORTED to the memo, never gated (matching TF-79).
- **Pre-registered expectation** (from the spike, not a gate): GS agreement @0.5 s rises to ≈ 0.4 (from 0.05); IDSSE (~0.45) and SkillCorner (~0.79) effectively unchanged.
- **Document the per-provider yardstick caveat in the memo + ADR:** SkillCorner's events are a tracking-derived possession model, so validating tracking-detected contacts against them is partly **circular** → its agreement is optimistic; IDSSE's **independent** Sportec feed is the honest yardstick. GS post-fix ≈ IDSSE confirms a normal provider.
- The memo is a committed research artifact: it must carry top-level `run_commit` / `run_tree_dirty: false` (ADR-037 gate) and be generated on a clean tree — hence it is the cycle's **second** commit (sec. 9).

## 7. Documentation corrections

- **`docs/context/tracking-features.md`** (LIVE): correct the GS caveat from "blunted ball" to the duplicate-frame velocity-sawtooth mechanism + the dedup fix.
- **ADR-124** (historical TF-79 record): add a one-line forward note that its GS "blunted ball" caveat is **superseded by ADR-125**; do not rewrite the historical record otherwise.
- **New ADR-125:** the dedup decision (provider-agnostic, helper placement, C ruled out), the corrected GS root cause, and the SkillCorner-circular-yardstick caveat.
- **CHANGELOG:** the new release entry records the fix and states it corrects the 4.136.0 GS characterisation. The shipped 4.136.0 entry is **not** rewritten (history stays honest to what shipped).

## 8. Scope

**In:** the dedup helper + its wiring + window-semantics fix; unit + detector tests; the doc corrections; ADR-125; the re-validation memo.
**Out (not deferred silently — explicitly out of this cycle):** any broader frame-conditioning refactor; changes to the ADR-120 preprocess velocity path; the deferred PathCRF alignment stage; any TF-48 change; any GS-specific branch. A velocity-break consumer other than `detect_ball_contacts` adopting the helper is future work, not this cycle.

## 9. Tests / gates

- **New:** `trajectory_dedup_mask` unit tests — run collapse, A→B→A preserved, float32-exact duplicates, all-dup segment, tol boundary.
- **Detector:** a GS-like duplicate-frame fixture that under-detects before and recovers after (both sides asserted); IDSSE-like control unchanged; existing `test_ball_contact.py` updated where the dedup changes expected values.
- **Purity:** the helper is pure; `detect_ball_contacts` stays in `PURITY_ENTRIES` (still does not mutate input).
- **Window-semantics:** a test pinning that `dvw`/`w` span the intended seconds on a sparse (dedup'd) sampling.
- *Dedup half:* no new public surface → mirror-registry size, c4 counts, feature-glossary, NOTICE unaffected. **§11 adds +2 `feature_glossary` columns + a frame-schema column** (see §11.6) — mirror-registry size, c4 aggregator-count, and NOTICE stay unaffected (no new public `add_*`/aggregator/attributed method). Re-run the full CI-faithful suite (the committed-artifact + full-suite-only gates in particular).

## 10. Commit plan

- **One coherent, fully-tested code commit** — helper + wiring + window fix + tests + doc corrections (context, ADR-124 note) + ADR-125. No micro-commits; no commit-after-every-step. **Explicit owner approval required immediately before this commit** (diff shown, owner says yes).
- **Then** regenerate the validation memo on the clean committed tree (ADR-037 provenance) → a **second** commit (memo + CHANGELOG), also **explicit-owner-approval-gated**.
- **Non-squash** admin merge (preserves the two commits).
- **No version number** is claimed until commit-prep (derive the next-free then); **no ADR number hard-claimed** until cycle end.

---

## 11. Amendment (2026-10-15, owner-ruled): z-provenance + `is_aerial` honest-NaN

**Why (spike finding):** the ball-`z` investigation found z is **present (non-NaN) on all tracking providers** but **MEASURED only on IDSSE/Sportec** (dense optical, floored at 0, continuous); **ESTIMATED on SkillCorner** (broadcast — the loader artifact is literally `<mid>_tracking_extrapolated`; non-physical signature: ball p50 0.20 m up, off-ground 81–86 % of frames); **measured-but-noisy on Gradient Sports** (z reaches −3.2 m). So §4's intent — `is_aerial` honest-NaN where z is **absent OR estimated** — is unmet: the detector reads SkillCorner's estimated z as if measured and emits a boolean. Owner folds the fix into THIS cycle (4.137.0 / ADR-125). This changes `is_aerial` on SkillCorner (boolean → `<NA>`), so more values move; still no retrain.

**11.1 `z_source` frames column (mechanism = provider DECLARES, not inferred).** A new tracking-frame column `z_source`, **closed vocab `{measured, estimated, absent}`**, **category** dtype (static per match — ADR-106 / the static-low-card rule). **Added to BOTH schema bases (GSBC-PLAN-03):** `TRACKING_FRAMES_COLUMNS` (`schema.py:10`, Int64-id base → GS + the Int64 providers) **and** `KLOPPY_TRACKING_FRAMES_COLUMNS` (`schema.py:64`, object-id base) — the `SPORTEC`/`SKILLCORNER`/`METRICA` aliases are `= KLOPPY` (`:74/:78/:83`) and `GRADIENTSPORTS` is `= TRACKING_FRAMES_COLUMNS` (`:87`), so adding to the two bases covers every alias. **SkillCorner (estimated) AND IDSSE/Sportec (measured) build from the KLOPPY base** (`skillcorner.py:357`, `sportec.py`), so omitting KLOPPY would (a) fail the ADR-058 KLOPPY-mirrors-base gate (`test_tracking_schema.py`) and (b) leave SkillCorner frames with no `z_source` → the §11 fix would never reach its target. Plus `_cast_to_declared_schema`. Set at each per-provider frame-build adapter (ADR-054/031 seam) from a single central classification — IDSSE/Sportec `measured`, Gradient Sports `measured`, SkillCorner `estimated`, Metrica `measured` (verify at impl), StatsBomb/SB360 `absent` (no continuous ball z), kloppy per underlying provider; `absent` wherever there is no continuous ball z. Provenance is the provider's ground truth (the adapter knows if its z is extrapolated), never inferred from values.

**11.2 `is_aerial` honest-NaN.** In `detect_ball_contacts`, `is_aerial = <NA>` unless the contact frame's `z_source == "measured"` **and** z is finite; boolean (`z > aerial_z_m`) only then. Realizes §4. (The existing absent-z NaN is subsumed — `absent` ≠ `measured`.) **Fail-safe default (GSBC-PLAN-02):** when the `z_source` column is ABSENT from the input frames (a legacy/synthetic caller pre-dating Z1), default to `"absent"` — NOT `"measured"` — so a missing declaration yields honest-NaN, never a false boolean (fail-closed, matching the §11 discipline). Real provider frames always carry `z_source` (Z1); synthetic test fixtures that assert a boolean `is_aerial` tag `z_source="measured"` explicitly.

**11.3 Gradient Sports stays `measured`.** GS measures z (dense optical); its impossible negatives are **ground-level** noise, **below** the 1.8 m aerial boundary, so aerial calls stay directionally reliable — keeping the signal. The ground-level negative-z noise is a documented provider caveat (beside the GK-clamp / blunted-ball caveats), NOT a demotion to `estimated`.

**11.4 `aerial_source` on the contacts table.** A new `aerial_source` column on the per-contact table (closed vocab `{measured, estimated, absent}` = the `z_source` at the contact frame) and `first_contact_aerial_source` on `add_ball_contacts`, **category** dtype — so a `<NA>` `is_aerial` is self-explaining (estimated vs absent vs measured). Registered in `feature_glossary`, the contact-columns contract, and the SB360 freeze-frame verdict. NOT geometric → no `reflection.py` entry.

**11.5 `validate_z_provenance` guard (the validate-family cross-check).** New `validate_z_provenance` / `ZProvenanceDiagnosis` / `ZProvenanceWarning` (the `validate_gk_position_clamp` / `validate_velocity_regime` family): fails loud when a provider's z **signature** contradicts its **declared** `z_source` — so a mis-declared adapter cannot silently ship. **The flagged signature is PERSISTENT ELEVATION (the extrapolation tell), NOT negatives (GSD-SPEC-02):** a `measured`-declared provider is flagged when its ball z sits persistently off-ground (p50 above a small threshold AND off-ground fraction beyond ~0.7 — the SkillCorner signature: p50 0.20 m, 81–86 % off-ground) OR is all-zero / all-NaN. **Ground-level negatives do NOT flag** — IDSSE (p50 ~0, 25–38 % off-ground) and **Gradient Sports (p50 ~0, negatives to −3.2 m, 39–51 % off-ground) both PASS** as consistent with `measured` (§11.3: GS negatives are ground-level, below the 1.8 m boundary). Thresholds fixed/pre-registered against the spike numbers, never fit to the accuracy corpus. Signature-inference earns its place here as a VALIDATION, never the mechanism.

**11.6 Scope / blast-radius delta.** +1 frames schema column (`z_source`) + per-provider classification + adapter touch; +1 contacts column (`aerial_source`) + `first_contact_aerial_source`; `is_aerial` semantics change (SkillCorner → NaN). No retrain. Gates added: `TRACKING_FRAMES_COLUMNS` schema/alias/cast tests, the category-dtype static-column gate, `feature_glossary` + contact-columns contract + SB360 verdict for `aerial_source`, and `validate_z_provenance` unit tests (declared-vs-signature, both directions). The re-validation memo (commit 2) now also reports per-provider aerial shares keyed to `z_source` (trustworthy only where `measured`).

**§8 scope update:** the items "z-provenance / `is_aerial` honest-NaN on estimated-z providers" and "`aerial_source`" move from out-of-scope to **IN** (this amendment). Still OUT: any broader frame-conditioning refactor, ADR-120 preprocess-velocity changes, PathCRF, TF-48.
