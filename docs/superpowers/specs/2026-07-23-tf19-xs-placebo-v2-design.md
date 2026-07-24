# TF-19 xS-probe placebo v2 (relevance-matched) — design

**Date:** 2026-07-23
**Target:** silly-kicks (next-free release — confirm with owner at commit-prep) / PR-S`<NN>` / ADR-037 amendment (or new ADR — decide at commit-prep)
**Status:** design, revised after two cross-session review rounds, awaiting owner review
**Source:** TF-19 GKDV cycle (ADR-037); PR-3b Part A run (`docs/research/tf19_pr3b/`, silly-kicks 4.55.4 / PR-S126); the registered xS probe in `silly_kicks/tracking/_model_eval.py`.
**Scope class:** research instrument (in NO xfn list, no VAEP consumer) — **C4-free, no retrain trigger**.

---

## 1. Problem

TF-19 PR-3b Part A ran the registered xS-arm GK-substitution probe on 64 GradientSports matches and
returned **`no_valid_placebo` → re-gate `unmeasurable_at_dose`** — **not** because there is no GK effect,
but because the probe's *secondary null control* was degenerate.

**Diagnosis (the reframe).** The probe runs **two** controls per frame:
- **`nearest_def`** — displace the nearest defender by the GK's substitution vector. **Valid**:
  `nearest_def_median = 0.00499`.
- **`placebo_out`** — displace a **random** outfielder by the same vector. **Degenerate**: all 20
  per-replicate medians are `0.0` (`placebo_p95 = 0.0`), `placebo_zero_fraction = 0.6645` — a random,
  usually *distant* player moving ~2 m rarely changes the aggregate xS.

`evaluate_xs_probe`'s `no_valid_placebo` gate fires on `placebo_p95 <= 0` (`_model_eval.py:555-562`),
**short-circuiting before the clustered dose-response ever runs** (`_dose_response_clustered`, `:576`).
So PR-3b measured the GK effect's *magnitude* (dose-responsive: 2 m `0.0154` / 3 m `0.0200` / 4 m
`0.0222`; ≈3.09× the nearest-defender control; 5.3% gated-band zero-fraction) **but never tested its
significance.** The instrument works; the RANDOM placebo is the uninformative part, and it vetoes the
whole test.

**Goal.** Move the xS arm from `unmeasurable_at_dose` to a **real, citeable verdict** by replacing the
random placebo with a **relevance-matched** null — **pre-registered, not tuned-to-pass** — so the test
actually *runs* and the dose-response is evaluated. The verdict (pass / band-pass-flat / unmeasurable) is
whatever the locked rule returns.

## 2. Resolved design decisions

| # | decision | resolution |
|---|---|---|
| D1 | how to change a frozen pre-registered test | **New pre-registered variant `xs-dose-banded-v2`, run ALONGSIDE the frozen v1.** v1's rule / constants / `evaluate_xs_probe` are untouched; the record reports BOTH v1's `no_valid_placebo` and v2's verdict. Amending v1 in place was rejected — it rewrites a frozen pre-registration after seeing it fail. |
| D2 | what defines the "relevance-matched" placebo pool | **Model-relevant DEFENDERS only.** The xS model conditions on its **5 nearest defenders + 5 nearest attackers to the ball** (`_xshot_occurrence.py:242-243`, `k=5`; `def_xy` already excludes GKs, `:207`). The gated pool = the **5 nearest defenders, minus the `nearest_def`, by `player_id`** (the "minus GK" is a documented no-op — GKs are already out of `def_xy`). **Attackers are EXCLUDED from the gated pool** — the nearest attacker to the ball *is* the shooter/ball-carrier (the threat source); gating on it would inflate the placebo through attacking geometry (`OppDist_*`/`OppAng_*`), not deterrence, most likely yielding a **confidently-wrong `fail`** against §8's semantics. Defender-only keeps the *deterrence* interpretation: "does the GK suppress this shot more than another model-relevant defender's repositioning." Attackers survive only as a **reported, carrier-excluded, non-gating diagnostic** carrying a distinct `actor_role` (§3, §5). Rejected alternatives: all-outfielder / ball-nearest (admits the shooter); distance-to-goal-matched (small, overlaps the last-ditch defenders → the `nearest_def` control). |
| D3 | how much of the frozen rule changes | **Exactly one thing: the placebo pool** (random outfielder → model-relevant defenders). All other constants frozen at v1's values — ratio `2.0×`, TF-19 absolute floor `0.01`, `min_band_n = 100`, `min_stratum_n = 50`, `placebo_replicates = 20`, `placebo_band_pct = 95`, `max_placebo_zero_fraction = 0.95`, the clustered dose-response test + α `0.05`, `min_games = 8`. The `no_valid_placebo` gate stays LIVE (§4). |

**What v2 actually decides — the honest consequence (corrected in review 2).** The gated pool is the
defenders **#2–#5** nearest the ball (`nearest_def` ≈ #1 is excluded), so they sit *farther* from the ball
than #1 and have *less* leverage on the xS features. Its `placebo_p95` is therefore **expected to land
below `nearest_def_median` (0.005)** — the defender placebo is a **weaker** control than `nearest_def`,
not a stronger one. Since the ratio prong reads `2.0 × max(nearest_def, placebo_p95)` (`:595`), the
`max()` **pins to `nearest_def`** and the placebo is **inert in the bar**. With `gk_med = 3.09 ×
nearest_def`, the bar (`2 × nearest_def = 0.010`) is well below `gk_med = 0.0154`, so **the ratio prong is
near-certain to pass** (a `fail` would need the placebo `p95 > 0.0077` — a farther defender moving xS >50%
more than the nearest — implausible on average, at most a sliver on the p95 tail).

So the placebo's real job in v2 is **(1)** to clear the instrument-validity `no_valid_placebo` gate with a
**principled, non-degenerate** control (which v1's random placebo could not), and **(2)** to be a
reportable fair null — **not to move the bar.** With the gate cleared and the ratio near-certain to pass,
**v2's genuine open question is the clustered dose-response permutation** — sign-flip across the 64 games
at α 0.05, which v1 *never reached*. That p-value is **not knowable from artifacts in hand**; the real
suspense is `pass` (dose-response significant) vs `band_pass_flat_dose_response` (not). This is exactly
where v2 earns its keep: it lets the significance test run for the first time.

**Not blind, by disclosure.** v2 is designed after seeing v1's full output. The defender-pool
`placebo_p95` is estimable from PR-3b's stored per-replicate medians + pining, and the ratio outcome is
near-predetermined (above) — so the *only* genuinely unknown quantity is the dose-response p-value.
Blindness is by **discipline**, made **auditable** (§6): the pool + constants land in a lock commit, the
run happens only after it, and that commit hash is recorded in the run's `metrics.json`, so the git DAG
shows constants-locked-before-run.

## 3. The v2 placebo mechanism (`_targets_deltas`)

Everything except the pool from which the gated placebo player is drawn is unchanged from v1:

- Per frame, the gated pool = the **≤5 nearest defenders to the ball** (the extractor's reference and
  `def_xy`, GK-free), **excluding the `nearest_def` by `player_id`**. The exclusion is id-based, so it is
  clean when the carrier ≠ ball (pass in flight / loose ball): `nearest_def` is removed if present, a
  no-op otherwise. **Pool size is 0–5** — 4 when `nearest_def` is among the ball-nearest-5, 5 when it is
  not, fewer when the frame has few defenders back.
- Each of the **20 replicates** draws **one** player from that pool with `default_rng(seed + r)` (v1
  seeding) and displaces it by the **same per-frame paired vector** (GK → ghost target). The registered
  off-pitch policy is unchanged (score, never clamp).
- Downstream is unchanged: the gated defender-placebo rows carry `actor_role="placebo_out"` (so the
  **unchanged** `evaluate_xs_probe` bands them at `:544`); `placebo_p95` = the 95th percentile of the 20
  per-replicate medians.
- **Empty/sparse pool → no fabricated 0.** A frame whose defender pool is empty after the exclusion (a
  fast break with few defenders back) contributes **no** placebo row for that replicate (`:378`). The
  defender pool therefore **can** under-populate → the placebo **can** degenerate on a sparse-defender
  frame set. That is intended: it keeps the `no_valid_placebo` gate a *live* discriminator (§4), an
  empirical expectation of clearing, not a construction.

**Attacker diagnostic (reported, non-gating, distinct role).** For interpretation only, the run also
computes a diagnostic from the **≤5 nearest attackers to the ball with the carrier excluded by id**
(the carrier is the shooter; excluding it by id is cleaner than a positional "2nd-nearest" rule when the
carrier is not the geometrically-nearest attacker), and these rows carry a **distinct
`actor_role="attacker_diag"`** — a role the unchanged
`evaluate_xs_probe` **does not band** (it keys on `"placebo_out"`/`"nearest_def"` only, `:544`), so they
**cannot** enter the gated `placebo_p95`. The driver reports their p95 separately as a model-sensitivity
sanity check ("does the model respond to attacker displacement at all"). It feeds **no** verdict branch;
it exists so the write-up can show *why* attackers were kept out of the gate, with the number in hand.

**Pool derivation — plan detail (§9).** Default: re-derive the ball-nearest-5 defenders geometrically
(matching the extractor's `_nearest_k((bx,by), def_xy)`); or reuse the extractor's exact defender ids if
cheaply exposed. Both yield the same principled pool.

## 4. The rule (unchanged; the gate stays LIVE; the dose-response is the decider)

v2 reuses `evaluate_xs_probe` **verbatim** (placebo-origin-agnostic, `:540-544`) — **no evaluator change.**
The ladder is identical:
1. Band-population gate: `trusted_stratum ≥ 50` and `gated_band_n ≥ 100`, else `unmeasurable_at_dose`.
2. **`no_valid_placebo` gate (LIVE):** `placebo_p95 > 0` **and** `placebo_zero_fraction ≤ 0.95` **and**
   `nearest_def_median > 0`, else `no_valid_placebo`. A populated defender pool moves the surface; a
   sparse-defender frame set can degenerate it, so the gate *can* fire (a real finding). Expected to
   clear on attacking-third shot frames (defenders typically back) — an empirical expectation.
3. Ratio prong: `gk_med ≥ 2.0 × max(nearest_def_median, placebo_p95)`. As shown in §2, the defender
   placebo is inert here (weaker than `nearest_def`), so this is effectively a **"beat the nearest
   defender by 2×"** test, **near-certain to pass** on the in-hand numbers.
4. **Clustered dose-response** (per-game ρ, sign-flip permutation across games; `min_games = 8`, α `0.05`)
   — **the genuine decider for v2**, run for the first time (v1 short-circuited at gate 2). `ok` → `pass`;
   `flat` → `band_pass_flat_dose_response`; `underpowered` (n_games < 8; unlikely at 64) →
   `unmeasurable_at_dose`.
5. `regate_verdict(arm="shot", …)` mapping unchanged; TF-19 absolute floor (`0.01`) unchanged.

## 5. Code structure

Minimal, and the "single change" is visible at the code level:

- **`substitution_deltas(...)` + `_targets_deltas(...)`** gain a keyword `placebo: str = "random"`
  (default = exact v1 behaviour → **no existing caller changes**). `"model_relevant_def"` selects the new
  gated pool; `_targets_deltas` additionally emits the `attacker_diag`-roled rows.
- **New private helpers** `_model_relevant_def_pool(grp, gk_team, cpid, *, k=5)` (ball-nearest-`k`
  defenders minus `nearest_def` by id) and `_attacker_diag_pool(...)` (2nd–`k`th nearest attackers,
  shooter excluded). The latter's rows are tagged `actor_role="attacker_diag"`.
- **New registration** `PROBE_WRAPPERS["xs_v2"]` — a wrapper calling the probe with
  `placebo="model_relevant_def"` and a `rule_constants` block **copied from v1** plus a
  `placebo_pool="model_relevant_def"` field (the registry self-documents the one difference).
- **`evaluate_xs_probe` — UNCHANGED.** `xs_substitution_probe` gains the `placebo=` passthrough (or a thin
  `xs_substitution_probe_v2`); the frozen `xs_substitution_probe` default stays `"random"`.
- **Driver** (`scripts/validate_xs_probe.py`): a `--variant {v1,v2,both}` (default `both`). Structure:
  **one match-load pass, one full `substitution_deltas` delta-compute per variant per match** — the GK +
  `nearest_def` rows are numerically identical across variants (`gk_mask` `:330`, `_nearest_def_mask`
  `:195` are pool-independent), so recomputing them per variant is ~13% redundant but keeps each variant's
  deltas in its **own frame**. That per-variant separation is the point: each `evaluate_xs_probe` call sees
  **exactly one `placebo_out` population** (v1's random OR v2's defender, never both) — the collision the
  shared "compute-once-then-tag-both" alternative would have created is impossible by construction. The
  `attacker_diag` rows live in the v2 frame, reported but never banded. Correctness floor: v1's numbers
  reproduce PR-3b's exactly. (Runtime cost — two computes per match — is the ~1.5–3× budget §7 notes.)

## 6. Testing & pre-registration (CI)

**Pre-registration = the committed code + an auditable blindness lock.** The v2 pool + constants land in a
dedicated **lock commit**; the ~64-match run happens **only after** it and **records that commit's hash in
`metrics.json`** (mirroring PR-3b's `baseline_commit`). The git DAG then *shows* constants-locked-before-run
— blindness is checkable evidence, not just a promise. In writing: **do not compute the defender-pool
`placebo_p95` on real data before the lock commit.**

**CI validates the MECHANISM; the CONSTRUCT is settled in design (this document), before the lock** — a
fixture built so pool members move a planted surface is constructed to pass and cannot surface a "who's in
the pool" error, which is why the attacker/carrier issue was fixed in design. The meta-tests:
- **Non-degeneracy (the fix works):** on a fixture where the model-relevant **defenders** move the planted
  xS surface, v2 yields `placebo_p95 > 0` — **and** the same fixture through the v1 random placebo yields
  `placebo_p95 ≈ 0` (both sides asserted — the non-vacuity discipline).
- **Discrimination preserved:** a planted GK-signal model clears the ratio; a GK-blind model does not.
- **Construct guards (two levels), the "second-producer seam":** a **unit** test that the shooter /
  nearest-attacker `player_id` is never in the gated defender pool; **and** a **frame-level** test that
  every `actor_role=="placebo_out"` row in a v2 deltas frame is **defender-sourced** (no `attacker_diag`
  or random-pool row can carry the gated role) — pinning that the unchanged evaluator provably cannot see
  attacker rows.
- **v1 frozen:** a regression pinning v1's `xs_substitution_probe` / constants / verdict byte-stable
  (`PROBE_WRAPPERS["xs"]` `rule_constants` unchanged).
- **Import allowlist / PRIVATE_CONSUMERS:** v2 lives in `tracking/_model_eval.py` (private) like v1; no new
  cross-package coupling.

## 7. The run & deliverable

- The PR ships the v2 **code + CI tests**; the **lock commit** fixes the pool + constants.
- **In-cycle run after the lock** (mirrors PR-3b): I run the ~64-match GS probe (`--variant both`,
  pining access). Runtime is **~1.5–3× PR-3b's ~11 h** — PR-3b scored ~22 actor-perturbations/frame (GK +
  nearest_def + 20 random-placebo replicates); `--variant both` adds v2's 20 defender-placebo + 20
  attacker-diagnostic replicates per frame (one full `substitution_deltas` per variant, §5), so budget
  ~16–33 h. `--variant v2` (~2× PR-3b) is the shorter path if the owner cites the frozen PR-3b v1 verdict
  by reference instead of re-proving it. Records the verdicts under **`docs/research/tf19_pr3b_xs_v2/`**
  (`metrics.json` + `report.md`) with the **lock-commit hash**, reporting **both** v1's `no_valid_placebo`
  and v2's verdict + the attacker diagnostic side by side.
- GS-only by construction; the public xS model is GS-free, so every GS match is held-out (as in PR-3b).

## 8. Reporting / honest framing

The `report.md` states plainly:
- v2 changed **exactly one thing** (the placebo pool: random → model-relevant **defenders**); attackers
  were excluded because the nearest attacker is the shooter, so gating on them answers a model-sensitivity
  question, not a deterrence one (the attacker diagnostic is reported to show this, non-gating).
- The defender placebo is a **weaker** control than `nearest_def` and **inert in the ratio** — its job is
  to clear instrument-validity with a principled null, not to move the bar. The ratio prong is therefore
  a "beat the nearest defender by 2×" test, **near-certain to pass**; **v2's real decider is the
  clustered dose-response permutation**, run for the first time, and the verdict turns on it: **`pass`**
  (dose-response significant across the 64 games) or **`band_pass_flat_dose_response`** (not). A
  `no_valid_placebo` remains possible only if the defender pool degenerates.
- v2 is **pre-registered but not blind** — blindness held by the §6 lock-commit discipline (auditable via
  the recorded hash), not by ignorance; the ratio outcome is near-predetermined and disclosed as such.
- v1's `no_valid_placebo` is reported alongside; the frozen v1 record is never rewritten.

## 9. Open questions (carry into the plan, none blocking)

- **Pool derivation** (§3): geometric vs the extractor's exact defender selection — plan pick; same pool.
- **One-pass compute + per-variant frames** (§5): the sharing is at the compute layer; each evaluation
  gets its own single-`placebo_out` frame. Correctness floor: v1-reproduces-PR-3b.
- **`k`** fixed at **5** (the xS feature count) — a locked constant.
- **ADR**: amend ADR-037 (register the v2 rule + `placebo=` + the defender-only + `attacker_diag` roles)
  vs a small new ADR — decide at commit-prep.

## 10. Attribution / C4 / retrain

- **NOTICE**: no new methodological reference (a placebo redesign within the existing ADR-037 probe; xS
  attribution arXiv:2512.00203 unchanged).
- **C4**: **C4-free** (research instrument in `tracking/_model_eval.py`, not an aggregator — count
  unchanged).
- **Retrain**: **none** (in no default xfn list; the xS/ghost weights are untouched — v2 only re-reads
  them through a different placebo pool).
