# Plan: the per-player detection primitive (one PR, TDD)

**Spec:** `docs/superpowers/specs/2026-09-27-detection-primitive-design.md` (rev 1, APPROVE). **ADR:** `docs/superpowers/adrs/ADR-109-detection-primitive-parallel-to-fov.md`. **Base:** `main` @ `d505bb7`. **One feature branch, one coherent commit, human-gated.** No worktree. Spec/ADR are placed (owner-approved 2026-09-27); the commit needs Karsten's explicit per-commit OK.

## Task 0 — baseline
- Branch off `main` (`d505bb7`), clean checkout. Confirm the CI-scope gate is green at baseline (`ruff check silly_kicks/ tests/ scripts/` + `--format --check`, bare `pyright`, `pytest tests/tracking/test_ghost_gk*.py -m "not e2e"` + the existing `keeper_detection_mask` tests).
- Locate the existing `keeper_detection_mask` tests (grep `keeper_detection_mask` under `tests/`); they are the parity anchor — they must stay green **unmodified**.

## Task 1 — RED (write failing tests first)
Add `tests/tracking/test_provider_visibility.py` (or extend the existing detection tests). Per `spec §4`:
1. **Parity** — `keeper_detection_mask` byte-identical across fully-observed / detected / extrapolated / all-null(raises) fixtures (compare against a captured pre-refactor reference array).
2. **General** — `detected_mask` on a non-keeper (outfield) `visibility` Series → correct per-row mask.
3. **Fail-closed** — detection-aware + entirely-null → raises with `_detection_discarded_message`; unknown provider → raises via `validate_provider`.
4. **Opt-out** — `assume_observed=True` on a KNOWN provider → all-`True`, equals the ungated mask; **an unknown/typo'd provider RAISES even under opt-out** (D1-SPEC-04: `validate_provider` runs first; the opt-out bypasses only the detection verdict, not provider classification).
5. **Fully-observed no-op** — gradientsports/sportec/idsse/metrica → all-`True`.
6. **Non-vacuity** — mixed `visibility` → non-constant mask.
Run → **FAIL** (`detected_mask` absent). Record the failure.

## Task 2 — GREEN (implement)
- Add `detected_mask(visibility: pd.Series, *, provider: str, assume_observed: bool = False) -> np.ndarray` to `silly_kicks/tracking/_provider_visibility.py` (spec §3.1): **`validate_provider` FIRST** → opt-out (`assume_observed` → all-True) → fully-observed all-True → `assert_detection_aware_visibility` → `visibility.fillna(False).astype(bool).to_numpy()`. It is a **PRIVATE symbol** in the neutral `_provider_visibility` home. **No import-allowlist / `PRIVATE_CONSUMERS.md` entry is needed THIS PR (D1-PLAN-07):** the sole consumer is `keeper_detection_mask` in `_ghost_gk` — an intra-`tracking/` call, not a cross-package case (don't hunt for an entry to add). Enforced contract this PR = the §4 unit tests + the private-module placement; a docstring/doctest is reader/local-run only, NOT the public-API-examples gate (D1-SPEC-01). Cross-package consumer entries land with the consuming PRs (TF-58, GK-gate).
- Refactor `_ghost_gk.keeper_detection_mask` to `return _pv.detected_mask(visibility, provider=provider)` (spec §3.2). No other change to `_ghost_gk` or the trainer.
- Run Task-1 tests → **PASS**; the untouched `keeper_detection_mask` tests → still green.

## Task 3 — gate
- Full CI scope: `ruff check silly_kicks/ tests/ scripts/` + `--format --check`; bare `pyright`; `pytest tests/ -m "not e2e"` (or the touched-surface subset if the full suite is impractical locally — note the CI matrix is the full gate on push).
- No glossary / `metric_contracts` / NOTICE change (assert none needed). No version bump (code-only; rides the next release).

## Task 4 — ADR
- `docs/superpowers/adrs/ADR-109-detection-primitive-parallel-to-fov.md` — placed (owner-approved 2026-09-27); reference it from the primitive's docstring. Commit it with the code.

## Task 5 — /final-review → STOP
- `/final-review` (skip version bump — code-only, no release). Then **stop at the commit gate**: show `git status --short` + `git diff --stat` + the proposed commit message; **do nothing until Karsten's explicit approval** for this commit + push.
- After it lands: **TF-58 rebases** onto it and wires `detected_mask` into its outfield collective-variable consumption (TF-58's cycle, its policy).

## Not in this PR
The ~8 keeper choke points + 13 GK outputs + glossary caveats + §7 open questions + version/Hyrum/lakehouse re-materialise = the **GK detection-gate follow-up** (a later cycle, after TF-58). Tracked as a TODO On-Deck row.

## Review history
- **2026-09-27 — APPROVE** (Ragnarok/TF-58 session, no BLOCKING; `D:\Development\_reviews\2026-09-27-detection-primitive-plan.md`; reviewer-conflict disclosed — the named consumer, verified vs the tree). D1-PLAN-06 (SHOULD FIX) + D1-PLAN-07 (CONSIDER) were both **spec-side** wording fixes (applied to the spec §4.4 / §5); Task 1.4 + Task 2 here reworded to match. The plan traced cleanly to spec rev 1; commit discipline exemplary. Ready for TDD impl on a feature branch.
