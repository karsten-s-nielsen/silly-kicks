# Adopt-for-refit verify/adopt seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax. Execution is NATIVE (subagents banned, owner rule).

**Goal:** Let an HF campaign declare `mode = "refit"` so `verify`/`adopt` judge a corpus-and-weights
refit as a supersede of the stale committed model, not as a re-verify of it.

**Architecture:** A `mode` field on the `Campaign` dataclass, threaded into the `verify` dispatch; three
mode-routed gate changes (corpus-identity stamp form, skip the `reproduces` integrity gate for a refit,
a public-floor fallback when the committed bundle no longer loads). No controller change: the committed
controller already isolates a halted instance (verified `campaign.py:542`/`:829`), so a sibling's
`assemble` is never stranded; Task 5 is the operational acceptance-proof rerun, not code (r1→r3 log).

**Tech Stack:** Python 3.12, the ADR-128 HF-campaign controller (`scripts/hf_jobs/`), the trainers
(`scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`), pytest. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-11-03-adopt-for-refit-seam-design.md` (dual-approved r2).

## r1 → r2 (plan A/B review resolutions)

- **PLAN-A-01 BLOCKING + PLAN-B-SF2 (Task 5 halt-isolation) — ACCEPTED, code change DROPPED.** Verified against
  committed code: `step()` already continues past a halted instance (`campaign.py:542` `continue`) and the
  `--watch` loop exits only on `settled = all(done or failed)` (`:829`), so one halt never strands a sibling
  with a launchable `assemble`. My premise was false; the guard could not go red. The observed a761710 strand
  was operational (one-shot/kill sequence, not a controller bug) and is moot: Task 1's fresh `tf46_refit`
  rerun re-runs all 4 jobs. Task 5 is now the operational acceptance-proof (no code), which also closes
  PLAN-A-02.
- **PLAN-B-SF1 (Task 4 fallback not gated to mode) — ACCEPTED.** `_certify` is called mode-blind
  (`adopt.py:483`); Task 3 threads `mode` only to the `_integrity` dispatch. Task 4 now also threads `mode`
  into `_certify` and gates the fallback to `mode == "refit"`, with a rematerialize-NEGATIVE test (anti-goal
  §4: a rematerialize public arm must NOT take the fallback).
- **PLAN-A-02 SHOULD FIX (no step owns "six arms verify green, re-run on DGX") — ACCEPTED.** New Task 5 owns it
  (operational, owner-gated).
- **PLAN-A-03 CONSIDER (branch base) — ANSWERED in Global Constraints** (base on `tf-46-arm1-clean`, reason stated).
- **PLAN-B CONSIDER — APPLIED:** ADR number is NOT 130 (in-flight); assigned at commit-prep avoiding collision,
  with an ADR-128 mode-field amendment noted (Task 6); Task 5's operational step does NOT gate the single
  commit; Task 4 names the in-process committed-bundle `load()` in its Interfaces.
- No A/B disagreement this round. Both: Tasks 1/2/3 sound and reachable; one-commit-owner-gated structure good.

**r2 → r3 (plan review r2: A APPROVE, B REQUEST CHANGES).** REFIT-PLAN-06 (B SHOULD FIX) — ACCEPTED: Task 4's
committed-bundle pre-check must use the SUBPROCESS `load_check` (`adopt.py:268`, `HFJOBS_PYTHON`), NOT an
in-process `load()`. The fingerprint-drift crash happens in the `.venv-hfjobs` subprocess (`headtohead.py:244`);
an in-process load in the controller interpreter can give a different verdict, so a drifted bundle would still
crash the head-to-head and the fallback would never fire. Task 4 now calls `load_check(inst.job,
REPO_ROOT/ref.path, run=run)` and falls back on a non-`None` result; the tests stub `load_check`/`run`, not
`load()`. REFIT-PLAN-07 (CONSIDER): Task 3 Step 2 relabelled — only `test_refit_skips_integrity` is red; the
two rematerialize tests are regression guards that pass from the start. REFIT-PLAN-08 (CONSIDER): the Task 4
negative test names its observable checks (`load_check` not called + head-to-head `run` invoked). The plan
`:482`→`:483` call-site line is corrected (A nit). A APPROVED r2; this r3 delta is isolated to Task 4's
pre-check mechanism + two test labels.

## Global Constraints

- **ONE commit per cycle, owner-gated (owner rule — overrides the skill's per-task commits).** No per-task
  commits. Each task ends at "tests green." The whole cycle is ONE commit, made only after explicit owner
  approval at the end (the final task is the commit-prep gate, not a commit).
- One feature branch off the current `tf-46-arm1-clean` HEAD (`a761710`), NOT `main` (PLAN-A-03): the Task 5
  acceptance proof re-runs the refit, which needs a761710's committed stoppage lock/generations + the
  trainers as they stand this cycle, and Commit-1 is the active unmerged branch. If Commit-1 merges to main
  first, rebase this branch onto main. Native execution.
- `mode` default is `"rematerialize"` (today's behaviour); validated against `{"refit", "rematerialize"}`.
- `rematerialize` keeps today's exact behaviour, including the `_integrity`/`reproduces` gate (anti-goal:
  the re-materialize path must keep its teeth).
- The refit floor is never auto-pass: a refit still clears certification (owner cv-parity / public floor)
  and the trainer's own fail-closed acceptance gates ran at fit.
- ADR-128: a campaign runs from a committed `scripts/hf_jobs/campaigns/*.toml`.
- id-dtype rules (ADR-019) and corpus-redistribution rules (ADR-116; public corpus ids stay) unchanged.
- CI-faithful trio green before the commit: `ruff check silly_kicks/ tests/ scripts/` + `ruff format --check`,
  `pyright`, `pytest -m "not e2e"` (CI scope). ruff pin 0.15.7, pyright 1.1.409.

## Review Focus

- A `rematerialize` campaign (no `mode` key, or `mode = "rematerialize"`) must behave EXACTLY as today —
  `_integrity`/`reproduces` still runs and still fails on a real reproduction drift (Task 3 test).
- A refit public arm whose committed bundle STILL loads must use the real head-to-head floor, not the
  fallback (Task 4 test: loadable committed → head-to-head path).
- A refit public arm whose FRESH acceptance gate FAILS (e.g. xcross-public) must FAIL the floor, never
  auto-pass (Task 4 test: drifted committed + failing fresh acceptance → floor fails).
- A `rematerialize` public arm with a drifted committed bundle must NOT take the refit fallback — the
  fallback is gated to `mode == "refit"` (Task 4 rematerialize-negative test; anti-goal §4).
- An unknown `mode` value must be refused at load, not silently treated as refit (Task 1 test).
- The committed controller already isolates a halted instance (`step` `:542` continue; watch `settled`
  exit `:829`), so a sibling's `assemble` is never stranded by a halt — no code task; the Task 5 operational
  rerun re-runs all 4 jobs under fresh keys, so nothing is left stranded.

---

## Task 1: `mode` on `Campaign`, the committed `tf46_refit.toml`, and its gate

**Files:**
- Modify: `scripts/hf_jobs/campaign.py` (the `Campaign` dataclass ~`:51`; `load_campaign` ~`:107`)
- Create: `scripts/hf_jobs/campaigns/tf46_refit.toml`
- Test: `tests/scripts/hf_jobs/test_campaign_tf46_refit.py`

**Interfaces:**
- Produces: `Campaign.mode: str` (new field, default `"rematerialize"`); `load_campaign` reads
  `campaign.mode` with that default and raises `ValueError` on a value outside `{"refit","rematerialize"}`.

- [ ] **Step 1 — failing tests.** In the new test file:
  - `test_mode_defaults_to_rematerialize`: `load_campaign(<a toml with no mode key>).mode == "rematerialize"`.
  - `test_mode_refit_is_read`: a toml with `mode = "refit"` → `.mode == "refit"`.
  - `test_unknown_mode_refused`: `mode = "bogus"` → `pytest.raises(ValueError, match="mode")`.
  - `test_tf46_refit_campaign_is_committed_and_refit`: `load_campaign("scripts/hf_jobs/campaigns/tf46_refit.toml")`
    has `name == "tf46_refit"`, `mode == "refit"`, and exactly the 4 instances
    `{(xshot_occurrence, public), (xshot_occurrence, sc_extended), (xcross_attempt, public), (xcross_attempt, sc_extended)}`,
    each `variants == ("default","position_only")`; the two xcross jobs carry `probe_dataset == "xcross_probe"`.
- [ ] **Step 2 — run, expect FAIL** (`Campaign` has no `mode`; file absent).
  Run: `python -m pytest tests/scripts/hf_jobs/test_campaign_tf46_refit.py -v`
- [ ] **Step 3 — add `mode`.** Add `mode: str = "rematerialize"` to `Campaign` (frozen dataclass; a trailing
  default keeps every positional constructor caller valid). In `load_campaign`, set
  `mode=c.get("mode", "rematerialize")` and raise `ValueError(f"campaign mode {mode!r} not in {{refit, rematerialize}}")`
  when out of set.
- [ ] **Step 4 — write `tf46_refit.toml`.** `[campaign] name="tf46_refit"`, `mode="refit"`,
  `bucket="silly-kicks/hf-refit-scratch"`, `target_minutes=60`, `concurrency_cap=10`, `attempts=3`; the 4
  job blocks copied from the current `~/xprobe_build/cycle1_refit.toml` body (xshot public→xshot17,
  xshot sc_extended→xshot115, xcross public→xshot17+probe xcross_probe, xcross sc_extended→xshot115+probe
  xcross_probe; variants `["default","position_only"]`).
- [ ] **Step 5 — run, expect PASS.**

## Task 2: digest-always corpus-identity stamp

**Files:**
- Modify: `scripts/train_xshot_occurrence.py:937`; the mirror call in `scripts/train_xcross_attempt.py`
- Test: `tests/scripts/test_refit_corpus_identity_stamp.py`

**Interfaces:**
- Consumes: `scripts._corpus_identity.corpus_identity(providers, match_ids, *, all_public)` (unchanged).
- Produces: both trainers call it with `all_public=False`, so `metrics.json` carries
  `corpus_match_ids_sha256` + `corpus_n_matches`, never the exact `corpus_match_ids`.

- [ ] **Step 1 — failing tests.**
  - `test_xshot_stamp_is_digest_even_for_all_public`: build a tiny all-public `StudyInputs`-like metrics path
    (or call the stamping helper directly), assert the emitted dict has `corpus_match_ids_sha256` +
    `corpus_n_matches` and NOT `corpus_match_ids`.
  - `test_xcross_stamp_is_digest_even_for_all_public`: same for the xcross trainer.
  - Guard both sides: assert `corpus_identity([...],[...], all_public=True)` still returns the exact form
    (the library fn is unchanged; only the CALL SITE changes).
- [ ] **Step 2 — run, expect FAIL** (today an all-public fit stamps the exact form).
- [ ] **Step 3 — change the call sites.** Replace `all_public=bool(inp.is_public.all())` with `all_public=False`
    at `train_xshot_occurrence.py:937` and its xcross mirror. Leave a one-line comment: the in-job trainer is
    visibility-blind (ADR-123 Amendment B); `adopt` rehydrates exact public ids at step 5 via
    `rehydrate_public_corpus_ids`.
- [ ] **Step 4 — run, expect PASS.**

## Task 3: verify routes `_integrity` by mode

**Files:**
- Modify: `scripts/hf_jobs/adopt.py` (`verify` dispatch ~`:485`; thread the campaign `mode` to the dispatch)
- Test: `tests/scripts/hf_jobs/test_adopt_refit_integrity_routing.py`

**Interfaces:**
- Consumes: `Campaign.mode` (Task 1); `INTEGRITY_FAMILIES` (`adopt.py:47`); `_integrity` (`adopt.py:289`).
- Produces: the dispatch runs `_integrity` iff `inst.job in INTEGRITY_FAMILIES and mode != "refit"`.

- [ ] **Step 1 — failing tests.** Stub `_certify`→`{"ok": True}` and spy on `_integrity`:
  - `test_refit_skips_integrity`: with `mode="refit"`, verifying an xshot instance does NOT call `_integrity`,
    and the row has no failing `integrity.go`.
  - `test_rematerialize_runs_integrity`: with `mode="rematerialize"`, an xshot instance DOES call `_integrity`
    (keeps its teeth).
  - `test_rematerialize_integrity_still_fails_on_drift`: feed `_integrity` a fresh metrics that does not
    reproduce the shipped (`n_rows` differ) → `verify` marks the instance `ok=False` (re-materialize teeth).
- [ ] **Step 2 — run.** `test_refit_skips_integrity` is RED (verify runs `_integrity` regardless today). The
    two `rematerialize` tests are regression guards and PASS from the start (today's behaviour) — they prove
    the fix does not change the re-materialize path (REFIT-PLAN-07).
- [ ] **Step 3 — thread mode + gate.** Pass the campaign `mode` into `verify`'s per-instance loop (read it
    from the loaded `Campaign`); change the dispatch condition to
    `if inst.job in INTEGRITY_FAMILIES and mode != "refit":`. No change to `_integrity` itself or to the
    `rematerialize` path.
- [ ] **Step 4 — run, expect PASS.**

## Task 4: public-floor fallback when the committed bundle fails its own contract

**Files:**
- Modify: `scripts/hf_jobs/adopt.py` — `_certify` (`:379`, gains a `mode` parameter) + its caller in `verify`
  (`:483`, passes `mode`)
- Test: `tests/scripts/hf_jobs/test_adopt_refit_public_floor.py`

**Interfaces:**
- Consumes: `Campaign.mode` (Task 1); `load_check(job, artifact_dir, *, run) -> str | None` (`adopt.py:268`,
  the EXISTING SUBPROCESS loader — runs the family `load()` in `.venv-hfjobs` via `HFJOBS_PYTHON`, returns
  `None` when clean, an error string otherwise, the SAME env+mechanism the head-to-head uses at
  `headtohead.py:244`); the fresh `metrics.json` `acceptance.*` block.
- Produces: `_certify(..., *, mode: str, run)` — in `mode == "refit"` public path, when
  `load_check(inst.job, REPO_ROOT / ref.path, run=run)` returns non-`None` (the committed bundle fails its
  contract in the subprocess), the floor is the fresh model's `acceptance` verdict (ok iff every gate true);
  on `None` (loads clean), the head-to-head as today. In `mode == "rematerialize"` the fallback is NEVER taken.

- [ ] **Step 1 — failing tests.** Stub `load_check` (NOT an in-process `load()`): make it return a non-`None`
    drift string for the drifted cases, `None` for the loadable case; spy on the head-to-head `run`.
  - `test_drifted_committed_floor_uses_fresh_acceptance_pass`: `mode="refit"`, `load_check`→non-None;
    fresh `acceptance` all-true → `_certify` returns `ok=True`, reason names the drifted/absent baseline,
    head-to-head `run` NOT invoked.
  - `test_drifted_committed_floor_fails_on_bad_fresh`: `mode="refit"`, `load_check`→non-None; fresh
    `acceptance` `brier_lt_base_rate_brier: false` (xcross-public shape) → `ok=False` (never auto-pass).
  - `test_loadable_committed_uses_headtohead`: `mode="refit"`, `load_check`→`None` → the head-to-head `run`
    IS invoked (observable: the `run` spy recorded the `scripts.hf_jobs.headtohead` argv), fallback NOT taken.
  - `test_rematerialize_drifted_committed_does_not_fallback` (anti-goal §4, REMATERIALIZE-NEGATIVE):
    `mode="rematerialize"`, `load_check`→non-None → `_certify` does NOT consult `load_check` for a fallback
    and takes the head-to-head path. Observable: `load_check` is NOT called (spy records zero calls) and the
    head-to-head `run` IS invoked.
- [ ] **Step 2 — run, expect FAIL** (no `mode` on `_certify`; no `load_check` pre-check; the refit tests red).
- [ ] **Step 3 — thread mode + implement the gated fallback.** Add `mode` to `_certify` and pass it at the
    `verify` call site (`:483`). In the public branch, ONLY when `mode == "refit"`, call
    `load_check(inst.job, REPO_ROOT / ref.path, run=run)` first; on a non-`None` result return the floor from
    the fresh `acceptance` block (ok iff every gate true) with a reason naming the drifted/absent baseline. A
    clean `load_check`, and every `rematerialize` arm, take the existing head-to-head path unchanged.
- [ ] **Step 4 — run, expect PASS.**

## Task 5: acceptance proof — rerun `tf46_refit` and verify six arms green (operational, owner-gated)

No code. This task owns spec §2/§8 "the six passing arms verify green under the refit path, re-run on the
DGX" (PLAN-A-02). The committed controller already isolates a halted instance (PLAN-A-01, verified
`campaign.py:542`/`:829`), so there is NO halt-isolation code change; the a761710 stranded assemble is moot
because this fresh rerun re-runs all 4 jobs under the renamed `tf46_refit` run keys.

**Files:** none (operational; recorded in the run ledger `docs/research/hf_campaigns/`).

- [ ] **Step 1 — on the implemented + CI-green seam branch, owner-gated:** on the DGX (`~/sk-tf46`,
    `.venv-hfjobs`), `campaign scripts/hf_jobs/campaigns/tf46_refit.toml preflight` → `launch --watch` →
    `retrieve --staging <outside repo>` → `verify --staging …`.
- [ ] **Step 2 — assert the six arms (xshot public+sc_extended, xcross sc_extended; both variants) verify
    GREEN** under the refit path (certification ok; `_integrity` skipped for refit); the two xcross-public
    arms FAIL only the legitimate acceptance gate (expected; accept-gate + withdraw is separate Commit-2 work).
- [ ] **Step 3 — this operational step does NOT gate the single code commit** (PLAN-B CONSIDER): the commit
    (Task 6) lands the code + tests; the rerun is the empirical acceptance proof after it, launched only on
    owner go (billed HF jobs).

## Task 6: ADR + commit-prep gate (owner-gated)

**Files:**
- Create: `docs/superpowers/adrs/ADR-1NN-adopt-for-refit-mode.md` — number assigned at commit-prep, NOT 130
  (in-flight on the TF-46 Commit-2 work); pick the next free number and verify no collision (PLAN-B CONSIDER).
- Modify: `AGENTS.md` key-convention bullet + `docs/context/<domain>.md` pointer if the seam warrants it; and
  decide whether ADR-128 needs a `mode`-field amendment (the campaign gains a first-class mode) — amend or
  cross-reference from the new ADR (PLAN-B CONSIDER).

- [ ] **Step 1 — ADR.** Record the refit-mode decision: the three mode-routed gates, why `reproduces` is
    re-materialize-only, the digest-always stamp + rehydrate, the committed-campaign `mode` field, the
    public-floor fallback. Cite the spec.
- [ ] **Step 2 — full CI-faithful trio green** (ruff + pyright + `pytest -m "not e2e"` at CI scope).
- [ ] **Step 3 — STOP at the commit gate.** Show the owner the full diff + the green test/lint report. Make
    NO commit without explicit owner approval (owner rule). On approval: one commit for the cycle. CHANGELOG
    + any version bump only if a shipped wheel changes (this seam changes scripts/tests, not the library
    surface — confirm at commit-prep).

---

## Self-review notes

- Spec coverage: §6.1 mode → Task 1; §6.2 B1a / §6.5-i digest stamp → Task 2; §6.2 B1b integrity routing →
  Task 3; §6.2 B2 / §6.4 / §6.5-ii public floor (mode-gated) → Task 4; §6.3 assemble isolation → ALREADY in
  the committed controller (verified `campaign.py:542`/`:829`), no code; the §2/§8 acceptance proof → Task 5
  (operational); §5/§8 committed-campaign gate → Task 1; ADR (§7) → Task 6.
- Review Focus: each line has its owning task test (rematerialize teeth → Task 3; loadable-committed,
  failing-fresh, and rematerialize-no-fallback floors → Task 4; unknown mode → Task 1; halt-isolation is a
  verified property of the committed controller, re-run covers the rest → Task 5).
- Proportion: signatures + test names/assertions, bodies left to the implementer; no per-task commits
  (owner one-commit rule).
- Spec §6.3 wording ("the controller CURRENTLY halts the --watch loop before a sibling's assemble") was shown
  FALSE at review (the committed controller already isolates); the spec is corrected accordingly, and Task 5
  carries no halt-isolation code.
