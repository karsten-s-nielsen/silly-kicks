# Spec: the per-player detection primitive (shared observability seam)

**Status:** rev 1 — REVIEWED (Ragnarok/TF-58 session, APPROVE; 1 SHOULD-FIX + 3 CONSIDERs + 1 coordination note, all applied — report `D:\Development\_reviews\2026-09-27-detection-primitive-spec.md`). Ready for the plan (`docs/superpowers/plans/2026-09-27-detection-primitive.md`) → TDD impl on a feature branch. Nothing implemented yet.
**Date:** 2026-09-27.
**Repo / base:** `karstenskyt__silly-kicks`, `main` @ `d505bb7` (native DAS + F1b already landed, code-only).
**Placement:** `docs/superpowers/specs/` (owner-approved repo placement 2026-09-27 — this is a silly-kicks release cycle). ADR: `docs/superpowers/adrs/ADR-109-detection-primitive-parallel-to-fov.md`.
**Splits from:** `spec-gk-detection-gate.md` (rev 0.1). That spec bundled the detection PRIMITIVE (§3.1) with the KEEPER-metric WIRING (§3.2–§3.3, ~13 GK outputs). This spec pulls **only the primitive + its contract** forward — the shared foundation — because an active **TF-58 team-coordination** cycle needs it now. The keeper wiring stays in the follow-up (see §8).
**Why now (not deferred):** TF-58 is in flight. Its collective variables (`compute_team_shape`, `_defensive_line`) consume **outfield** player positions on SkillCorner and do **not** consult the per-player `visibility` flag (verified on `main`: both filter `~is_ball & ~is_goalkeeper` + "valid coordinates", no detection gate). SkillCorner extrapolates undetected players. **Measured on the public corpus (A-League 2024/25 MIT, 20 matches, 17.6 M outfield player-frames):** outfield per-player detection **0.607** (~39 % extrapolated); **mean 6.07 of 10** outfielders detected per team-frame; the full outfield is observed in only **10.6 %** of frames, and **<7 detected in ~46 %** (≥7: 54.5 %). The keeper is worse (17.6 %) but is excluded from team-shape — 0.607 is TF-58's relevant exposure. TF-58 is **temporal** coordination (relative phase / cross-correlation over time), which needs continuous **observed** tracks — so on the 909-match SkillCorner corpus its collective variables are computed substantially from **imputed motion over a frame-to-frame-shifting observed set** (public-corpus probe; numbers above). **TF-58 already gates outfield detection itself** (`coordination/_signals.py`: `_detected_rows` / `observed_fraction`, D1-SPEC-05) — so the risk this PR removes is **duplication/drift, not a broken TF-58**: after it lands, TF-58's `_detected_rows` delegates to `detected_mask`, single-sourcing the doctrine; and it is the foundation for the deferred GK-gate (§8).

---

## 1. Executive summary

The per-player detection logic already exists — `silly_kicks/tracking/_ghost_gk.py::keeper_detection_mask` — but it is (a) named keeper-specific, (b) housed in a ghost-model-private module, and (c) consumed at exactly one site (ghost-GK training). Its **body is player-agnostic** (it masks any `visibility` Series). This spec **extracts it into a general primitive** `detected_mask(...)` in the neutral `_provider_visibility.py` (where its inputs — the provider sets, `validate_provider`, `assert_detection_aware_visibility`, `_detection_discarded_message` — already live), **adds the general opt-out** the workstream's fail-closed doctrine calls for, and leaves `keeper_detection_mask` as a **byte-identical delegating wrapper**.

No new methodology, no new columns, **no behaviour change to any shipped output** (the ghost-GK training path stays byte-identical, parity-tested). It is a layering + naming move that turns a ghost-private keeper function into the shared observability seam TF-58 and the deferred GK-gate both build on.

---

## 2. Scope

**In scope**
- `detected_mask(visibility, *, provider, assume_observed=False)` in `_provider_visibility.py` (§3.1).
- `keeper_detection_mask` refactored to delegate to it, byte-identical (§3.2).
- The D1 ADR (per-player detection primitive **parallel** to the polygon-FOV system) — draft `ADR-D1-detection-primitive-parallel-to-fov.md`.
- Red-first tests (§4).

**Out of scope (explicit)**
- The ~8 keeper choke points + the ~13 GK outputs' honest-NaN/counted-drops, glossary caveats, the §7 open questions, the version/Hyrum/lakehouse re-materialise — **the GK detection-gate follow-up** (§8; the parent spec's §3.2–§3.3/§4/§5/§7).
- **TF-58's own wiring** of `detected_mask` into its collective-variable consumption — TF-58's cycle owns whether/how to gate its outfield inputs. This spec only provides the seam.
- Any change to `compute_team_shape` / `_defensive_line` — TF-58 is editing those; this PR does not touch them (no collision).
- Unifying with the polygon-FOV system (D1: parallel by design).

---

## 3. Design

### 3.1 The primitive (add to `_provider_visibility.py`)

```
detected_mask(visibility: pd.Series, *, provider: str, assume_observed: bool = False) -> np.ndarray
```

Per-row boolean mask (aligned to the input Series), fail-closed — the exact doctrine of the current `keeper_detection_mask`, generalised in name + given the opt-out:

- `validate_provider(provider)` **FIRST — always, even under opt-out** (unknown/typo'd provider → raise; the provider name is fail-closed regardless of `assume_observed`).
- `assume_observed=True` → return all-`True` (the **general opt-out**, bypassing only the detection verdict; logged/flagged by the caller, never the default, never consumer-specific).
- `provider in _FULLY_OBSERVED_PROVIDERS` → all-`True` (no flag exists, none needed).
- `provider in _DETECTION_AWARE_PROVIDERS` → `assert_detection_aware_visibility(visibility, provider=...)` (entirely-null → raise: the flag was discarded), then `visibility.fillna(False).astype(bool).to_numpy()` (detected `True`, extrapolated/null-per-row `False`).

**Rationale for the order (D1-SPEC-04):** rev 0.1 returned on opt-out *before* `validate_provider`, which fails **open** on a typo'd/unclassified provider. Validating the name first (matching the existing `keeper_detection_mask` order) makes the taxonomy fail-closed for every caller; the opt-out is the explicit escape from the *detection* gate, not from provider classification (an opt-out caller on a genuinely new provider must add it to a set — a 1-line change, and the right friction). Byte-identical to `keeper_detection_mask` for `assume_observed=False` (same `validate_provider`→fully-observed→assert→mask order); the only new behaviour is the opt-out branch.

### 3.2 Keeper wrapper (byte-identical)

`keeper_detection_mask(visibility, *, provider)` in `_ghost_gk.py` becomes a one-line delegate:
```
return _pv.detected_mask(visibility, provider=provider)
```
It keeps its name + public surface (one consumer: ghost-GK training, `train_ghost_gk.py`), passes no `assume_observed`, so the ghost-GK path is unchanged. A parity test pins byte-identity (§4.1).

### 3.3 What consumers do with it (informational — not this PR)
- **TF-58 ALREADY gates outfield detection** independently (`coordination/_signals.py`: `_detected_rows` + `observed_fraction`). So this PR's value re TF-58 is **single-sourcing, not enabling**: after it lands, TF-58's `_detected_rows` **delegates** to `detected_mask` so the detection doctrine has one implementation (not two, drifting). D1-SPEC-05.
- **GK detection-gate follow-up** (§8): wire the ~8 keeper choke points through `detected_mask` / a keeper-row resolver.
- **ghost-GK training**: already a consumer via the `keeper_detection_mask` wrapper (§3.2).

### 3.4 Prerequisite — load path (already resolved)
`detected_mask` reads `visibility`, preserved only by the **native** SkillCorner builder; the pining path was re-routed onto it (parent spec §3.4, verified merged: `scripts/_loader_pining.py:802/:867/:958` on `main` @ `d505bb7` — the parent's `:798/:864/:957` drifted +4/+3/+1 with the F1b/DAS landings; D1-SPEC-03). For a detection-aware provider with entirely-null `visibility` the primitive **raises** (via `assert_detection_aware_visibility`) — a residual kloppy-gateway path surfaces loudly, not silently (T7, follow-up).

---

## 4. Registered tests (TDD, red-first)

1. **Keeper parity (load-bearing safety):** `keeper_detection_mask` returns **byte-identical** results before/after the refactor across fully-observed / detected / extrapolated / all-null(raises) cases — the ghost-GK training path is provably unchanged. (Existing `keeper_detection_mask` tests must stay green unmodified.)
2. **General primitive on non-keeper visibility:** `detected_mask` on an arbitrary (outfield) `visibility` Series returns the correct per-row mask.
3. **Fail-closed:** detection-aware + entirely-null → raises (message = `_detection_discarded_message`); unknown provider → raises (via `validate_provider`).
4. **General opt-out:** `assume_observed=True` **on a KNOWN provider** → all-`True`, reproducing the ungated mask exactly (proving the flag is the only escape and the default is safe); **an unknown/unclassified provider RAISES even under opt-out** (`validate_provider` runs first — the opt-out bypasses only the detection verdict, not provider classification; D1-SPEC-04).
5. **Fully-observed no-op:** each of `gradientsports`/`sportec`/`idsse`/`metrica` → all-`True`.
6. **Non-vacuity:** a mixed `visibility` (some `True`, some `False`) → a **non-constant** mask (guard the vacuous-fixture trap).

---

## 5. Versioning / attribution / API surface
- **Additive + byte-identical to every shipped output** → code-only to `main`, **no version bump** (rides the next release, matching the F1b/DAS code-only landings); no CHANGELOG-forcing behaviour change; **no NOTICE** (implements no published method); **no glossary / `metric_contracts` change** (no columns).
- **`detected_mask` is a PRIVATE symbol, not a public-API addition (D1-SPEC-01).** `_provider_visibility` is a single-underscore private module (not in `tracking.__all__`); the public-API-examples gate and the CI doctest job (`--ignore-glob="*/_[!_]*.py"`, `ci.yml:127`) both **skip** it — so an Examples block there is executed/gated by nothing. Frame `detected_mask` exactly like its siblings `validate_provider`/`assert_detection_aware_visibility`: private, in the neutral `_provider_visibility` home. **THIS PR needs no import-allowlist / `docs/PRIVATE_CONSUMERS.md` entry (D1-PLAN-07):** its sole consumer is `keeper_detection_mask` in `_ghost_gk`, an **intra-`tracking/`** call — not a cross-package case. The enforced contract this PR is the **§4 unit tests + the private-module placement**; a docstring/doctest is reader/local-run only — do **not** claim the public-API gate covers it. The cross-package/documented-consumer entries arrive with the **consuming** PRs (TF-58 `_detected_rows` delegation; the GK-gate), each in its own PR.

## 6. Commit discipline
One feature branch off `main`; one coherent, fully-tested commit (primitive + wrapper + tests + ADR together). **No commit/push without Karsten's explicit per-commit approval.** No worktree.

## 7. Coordination with TF-58
Both this PR and TF-58 add capability functions to `_provider_visibility.py` (this: `detected_mask`; TF-58: `dead_ball_observed`) — adjacent, composing additions. Land order is Karsten's call; whichever lands second rebases (a trivial adjacent-function merge). After this lands, TF-58 rebases and wires `detected_mask` into its outfield consumption.

## 8. The follow-up (NOT this PR) — GK detection-gate wiring
A later cycle, after TF-58: wire the ~8 keeper choke points (restdefense `_structure`/`_danger`/`_arms`, `_gk_influence` + `features`, `gk_decision` Tier B, `gkdv/_engine`) through `detected_mask` / a keeper-row resolver → honest-NaN / counted-drops for the ~13 GK outputs on undetected SkillCorner keepers; glossary caveats; the §7 open questions (`rd_num_superiority_gk` semantics; `gk_decision` Tier A caveat); Task-0b consumer sweep; version + Hyrum notice + one lakehouse re-materialise. Tracked as a TODO On-Deck row.

## 9. Recommended next step
Plan (`docs/superpowers/plans/2026-09-27-detection-primitive.md`) → TDD implement on one feature branch → human-gated commit. (Reviewed — see below.)

## 10. Review history
- **Plan review (2026-09-27)** — Ragnarok/TF-58 session review of the plan = **APPROVE**, no BLOCKING (`D:\Development\_reviews\2026-09-27-detection-primitive-plan.md`). Two findings, both applied here (the plan itself was correct):
  - **D1-PLAN-06 (SHOULD FIX):** §4.4 (test 4) still carried the rev-0.1 "opt-out returns before the provider check" text, contradicting §3.1/§4.3/ADR-109 — fixed to "opt-out on a KNOWN provider → all-True; unknown/unclassified RAISES even under opt-out (D1-SPEC-04)".
  - **D1-PLAN-07 (CONSIDER):** §5 reworded — this PR needs NO import-allowlist/`PRIVATE_CONSUMERS.md` entry (its sole consumer `keeper_detection_mask` is intra-`tracking/`, not cross-package); those entries land with the consuming PRs.
- **rev 1 (2026-09-27)** — Ragnarok/TF-58 session review = **APPROVE**, no BLOCKING (`D:\Development\_reviews\2026-09-27-detection-primitive-spec.md`; reviewer conflict disclosed — the named consumer reviewed against the tree, not its plans). Applied:
  - **D1-SPEC-01 (SHOULD FIX):** `detected_mask` is a PRIVATE symbol via `_PRIVATE_IMPORT_ALLOWLIST`, **not** the public-API-examples gate (`_provider_visibility` is single-underscore; the doctest job `--ignore-glob="*/_[!_]*.py"`, `ci.yml:127`, skips it). §5 rewritten.
  - **D1-SPEC-04 (CONSIDER):** `validate_provider` now runs **before** the opt-out (fail-closed on a typo'd/unclassified provider; opt-out bypasses only the detection verdict). §3.1.
  - **D1-SPEC-03 (CONSIDER):** loader anchors re-anchored `:802/:867/:958` on `d505bb7`. §3.4.
  - **D1-SPEC-05 (coordination):** TF-58 **already** gates outfield detection (`coordination/_signals.py`) — reframed the value as **single-sourcing (delegate `_detected_rows` → `detected_mask`), not enabling**; not rework. §Why-now, §3.3.
  - **D1-SPEC-02 (CONSIDER):** `fov_min_observed_fraction=0.7` location corrected in ADR-109 (`gk_decision/_config.py:45`, ADR-077).
  - Could-not-verify (no action): keeper 17.6 % is in the sibling keeper CSV; the target was verified against the ragnarok clone as a `d505bb7` proxy; the primitive/GK-wiring split is the owner's scope call (made).
- **rev 0.1 (2026-09-27)** — initial draft (split from the workstream `spec-gk-detection-gate.md`).
