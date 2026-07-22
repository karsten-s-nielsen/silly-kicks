# TF-19 PR-3b (Part A) — xS-arm GK-substitution probe: end-to-end run

**Status:** DESIGN — for review (part-deux session authored; other silly-kicks session to review).
**Date:** 2026-07-21. **Repo baseline:** v4.55.0 (`ed20ac7`).
**Scope:** Part A only (the runnable, unblocked slice of PR-3b). Part B (spec §6.4 world-test harness) is explicitly out of scope and gated on separate owner sign-off.

---

## Executive summary (read this first)

Every primitive needed to measure the xS arm's GK-responsiveness has shipped (PR-3, v4.53.0): the ghost engine `build_ghost_frames`, the `provenance_to_targets` adapter, the `xs_substitution_probe` / `evaluate_xs_probe` chain, the bundled xS model, and the GS pining loader. **What is missing is a single fact: nobody has run them together on real data.** No script, test, or e2e drives real frames → `build_ghost_frames` → `provenance_to_targets` → `xs_substitution_probe` against the real xS model; the probe's own tests use hand-fabricated targets, and the adapter's test stops at contract validation.

This spec adds three files, one docs-register entry, and changes no library code:

1. **`scripts/validate_xs_probe.py`** — a reported-not-gated driver (mirroring `scripts/validate_xcross_causal.py`) that loads GS matches, runs the chain **per match**, pools the tidy **deltas** (not the raw frames — see §3.1), computes the probe verdict + the §3.5 re-gate verdict + a targets→used→band reconciliation, and writes `docs/research/tf19_pr3b/{metrics.json, report.md}`.
2. **`tests/gkdv/test_xs_probe_wiring.py`** — a CI-safe test that inlines the *same* chain on synthetic in-domain frames + a planted model, proving the `build_ghost_frames → provenance_to_targets → xs_substitution_probe` seam runs and is **non-vacuous** (the ghost actually moves the keeper), so a flat real-data reading can never be a *seam* artifact. (It proves the seam, **not** that real `absolute_frame` output lands in the engine's domain — that is a separate impl gate, §5.1/§10.5.)
3. **`tests/scripts/test_validate_xs_probe.py`** — a CI-safe test of the driver's `run()` orchestration via a monkeypatched loader (so the N-match pool/accumulate logic is covered by CI, not first exercised by the expensive owner run).
   Plus a **`docs/PRIVATE_CONSUMERS.md`** entry recording the test/driver's first-party use of the private `tracking._model_eval` probe symbols (the register the codebase already keeps for exactly this — no promotion, no library change; ADR-037 kept the probe private on purpose).

**Two co-equal expected outcomes — pre-register both, do not assume the clean fail (the "ghost-accuracy paradox").** The dose band gates on `displacement_m ≥ XS_PROBE_DOSE_M = 2.0 m`, but the ghost's held-out MAE is ~1.07 m — so most ghost-to-keeper displacements sit *inside* [0, 2) m and fall **outside** the gated band. The two branches, in likely order:
- **`unmeasurable_at_dose`** (arguably the more likely first-order result): fewer than `XS_PROBE_MIN_BAND_N = 100` trusted frames clear 2 m, or fewer than `XS_PROBE_MIN_STRATUM_N = 50` land in the trusted stratum → re-gates to `unmeasurable_at_dose`. This closes **nothing** — the instrument couldn't reach the dose; it is *not* a null effect.
- **`fail`/`band_pass_flat_dose_response`** (the "clean fail" path): the band *does* fill but the GK response is flat/small (only `GK_r`/`GK_theta` of 27 features respond) → re-gates to `gated_clean_fail`, which *does* close the arm and names the GK-feature lever.

The run's **first** question is therefore "does the band even reach N = 100?" and only conditionally "fail vs pass?". Either recorded verdict is the deliverable; the point is to replace an assumption with a reproducible number.

**No library change, no model retrain, no probe-rule-constant change** (the xS constants were locked in PR-1). The one new public-facing artifact is the `docs/research/tf19_pr3b/` report.

---

## 1. Context & the gap

The TF-19 GKDV cycle asks whether keeper *positioning* measurably deters attacks. PR-1 (4.47.0) locked the probe rule; PR-2 (4.51.0) bundled the chirality-corrected weights; PR-3 (4.53.0) shipped the `gkdv/` engine + both physics arms. The xCross arm's frozen probe already returned `gated_clean_fail` (ratio 2.21×, absolute floor missed). The **xS arm's probe has never been run** — the engine that produces its ghost-target input only shipped in PR-3, and no driver was written to connect the two.

TODO.md (`:45`) records this verbatim: *"the xS arm is still unmeasured… `xs_substitution_probe` consumes ghost-substituted `targets` from the `silly_kicks/gkdv/` engine … The probe is therefore unblocked but still never run; that measurement is PR-3b."*

### 1.1 What already exists (verified against `ed20ac7`, do not rebuild)

| Piece | Location | Signature (verified) |
|---|---|---|
| Ghost engine | `gkdv/_engine.py:453` | `build_ghost_frames(frames, *, model=None, home_team_id, carrier=None, params=GkdvParams()) -> (cf, provenance, GkdvReport)` |
| Targets adapter | `gkdv/_engine.py:595` | `provenance_to_targets(provenance, *, frames, home_team_id) -> DataFrame` (exactly `_TARGET_COLUMNS`) |
| xS probe | `tracking/_model_eval.py:609` | `xs_substitution_probe(model, frames, targets, *, seed=42) -> dict` |
| Probe evaluator | `tracking/_model_eval.py:480` | `evaluate_xs_probe(deltas) -> dict` (verdict + diagnostics) |
| Re-gate | `tracking/_model_eval.py:636` | `regate_verdict(*, arm, probe_verdict, entanglement) -> str` (`arm ∈ {"shot","cross"}`) |
| xS model | `tracking/_xshot_occurrence.py:362` | `XShotOccurrenceModel.from_variant("default")` (bundled public weights; `predict_proba` at `:420`) |
| Ghost model | `tracking/_ghost_gk.py` | `GhostGkModel.from_variant("default")` |
| GS loader | `scripts/_loader_pining.py:176` | `load_matches(*, providers, match_ids=None, token=None, …) -> Iterator[(provider, match_id, actions, frames, home_team_id)]` |

The targets contract is pinned identical on both sides (`gkdv/_engine._TARGET_COLUMNS == tracking._model_eval._TARGET_COLUMNS`, asserted by `tests/gkdv/test_provenance_to_targets.py:27`), so `provenance_to_targets` output is directly usable as `xs_substitution_probe`'s `targets` argument.

### 1.2 The seam nobody exercises

`tests/gkdv/test_provenance_to_targets.py` chains `build_ghost_frames → provenance_to_targets` but **stops at `_validate_targets`** — it never feeds the result into the probe. `tests/tracking/test_probe_discriminating_power.py` calls `xs_substitution_probe(planted_model(kind), frames, targets)` end-to-end but builds `targets` **by hand** (perturbing the real GK directly), never touching the `gkdv` engine. So the join — real ghost targets → probe — is untested and unrun.

---

## 2. Approach (chosen: B+; alternative recorded)

**B+ (chosen): `scripts/` driver + inline-chain CI wiring test. No library orchestrator.**
The four primitives are called in sequence by a thin script; the seam is regression-protected by a test that inlines the same sequence on synthetic data. This grows no library surface, needs no ADR-037 change (the probe stays private; scripts and tests may import private modules — they are not governed by the `gkdv` import-allowlist), and matches the established reported-not-gated harness precedent (`validate_xcross_causal.py` = pure `analyze` + I/O `run` + `_write` + `argparse` `main`; method correctness lives in CI method-tests, the real run is owner-triggered).

**A (rejected for now): a library orchestrator `gkdv.run_xs_arm_probe(...)`.** Reusable, and the wiring test would call it directly — but it grows the `gkdv` public API and needs a confined `ALLOW_PRIVATE` exemption for `gkdv → tracking._model_eval` (mirroring `_das_port.py` wrapping `tracking._das`). YAGNI: the chain has exactly one caller today. Recorded as the natural refactor when a second caller appears (e.g. a §6.4 Layer-1 re-use, or an automated per-retrain re-run).

---

## 3. The driver — `scripts/validate_xs_probe.py`

Structure mirrors `validate_xcross_causal.py`: a pure core + an I/O `run` + `_write` + `main`.

### 3.1 Flow

```
ghost_model = GhostGkModel.from_variant("default")     # INSTANCE (not a name) so build_ghost_frames honors its carrier_params
xs_model    = XShotOccurrenceModel.from_variant("default")   # GS-free public weights → every GS match is held-out

all_deltas, per_match = [], []
for provider, match_id, actions, frames, home_team_id in load_matches(
        providers=["gradientsports"], match_ids=match_ids, tracking_limit=tracking_limit):
    _cf, prov, report = build_ghost_frames(frames, model=ghost_model, home_team_id=home_team_id)
    targets = provenance_to_targets(prov, frames=frames, home_team_id=home_team_id)
    deltas  = substitution_deltas(xs_model, frames, arm="xs", mode="targets", targets=targets, seed=seed)
    all_deltas.append(deltas)                            # TIDY deltas (1 GK row/frame + controls) — tiny vs raw frames
    per_match.append({"match_id": match_id, "n_frames_in": report.n_frames_in,
                      "n_frames_scored": report.n_frames_scored, "drop_reasons": report.drop_reasons,
                      "n_targets": len(targets)})

deltas_pooled = pd.concat(all_deltas, ignore_index=True)  # distinct game_id per match ⇒ dose clusters + MIN_GAMES=8 satisfied
result = evaluate_xs_probe(deltas_pooled)                  # the exact evaluator xs_substitution_probe wraps
result["n_frames_used"] = int(len(deltas_pooled[deltas_pooled["actor_role"] == "gk"]
                                  [["game_id","period_id","frame_id"]].drop_duplicates()))  # what the wrapper adds
regate = regate_verdict(arm="shot", probe_verdict=result["verdict"], entanglement=entanglement)
```

- **Pool DELTAS, not raw frames (memory).** `evaluate_xs_probe` needs `XS_PROBE_MIN_GAMES = 8` distinct games and `XS_PROBE_MIN_BAND_N = 100` gated-band rows, so a single match cannot clear them — pooling is required. But pooling *raw frames* and calling one `xs_substitution_probe` materializes a per-frame reset-indexed copy for every eligible frame across all ~64 matches at once (`_model_eval.py:167`); the parallel session hit exactly this wall on the xT-GK gate under pandas-3 and had to go subprocess-per-match. So the driver runs `substitution_deltas` **per match** and pools the *tidy* deltas (one GK row/frame + controls) — peak memory is one match, and `MIN_GAMES`/`MIN_BAND_N` are still satisfied because the deltas carry distinct `game_id`. This is exactly what `xs_substitution_probe` does internally, unrolled across matches; `n_frames_used` is re-added by hand (the wrapper computes it, `_model_eval.py:616`).
- **RNG discipline (pin it, it is not free).** Pooling deltas resets the placebo stream **per match** (`substitution_deltas` seeds `default_rng(seed + r)` per replicate, `_model_eval.py:375`), whereas a single pooled-frames call would draw one stream across all contexts. Both are valid "random outfielder per frame" disciplines; this design pins **per-match placebo streams** and records that in `metrics.json` so the verdict is reproducible.
- **Two models, distinct roles:** the *ghost* model must be a `GhostGkModel` **instance** — `build_ghost_frames`'s docstring (`:487`) warns a variant *name* string cannot resolve the model's `carrier_params`, so the ball-carrier inference would silently fall back to defaults. The *xS* model is the scored model (`predict_proba`).
- **Re-gate + entanglement (inert on the expected paths).** `regate_verdict` uses `arm="shot"` (its vocabulary; the xS arm ↔ the "shot" causal arm), with the shot arm's banked `entanglement="inside_band"` (DGX 2026-07-13/14; `docs/research/tf19_causal/xshot/`), a `--entanglement` CLI flag. **Note it is load-bearing only off the expected outcome:** `regate_verdict` returns `gated_clean_fail` for `fail` and `unmeasurable_at_dose` for `unmeasurable_at_dose` *before* it reads `entanglement` (`_model_eval.py:646,650`); entanglement only bites if the probe *surprises* with `pass` (→ `joins_with_caveat`). Threading it is correct defensiveness, not a driver of the likely result.

### 3.2 CLI (argparse)

| Flag | Default | Purpose |
|---|---|---|
| `--out` | *(required)* `Path` | output dir; convention `docs/research/tf19_pr3b/` |
| `--match-ids-json` | `None` | optional JSON `{"gradientsports": ["…"]}` to pin the corpus (mirrors the trainer's flag) |
| `--tracking-limit` | `None` | per-match frame cap (None = full match; small caps are dev-smoke only) |
| `--entanglement` | `inside_band` | banked shot-arm causal result for the re-gate |
| `--seed` | `42` | probe seed (matches `xs_substitution_probe` default) |

Providers are fixed to `["gradientsports"]` in the driver (the GS-only GKDV measurement rule is a hard constraint here, not a user knob) — documented in the module docstring, not exposed as a flag.

### 3.3 Output (`_write` → `docs/research/tf19_pr3b/`)

- `metrics.json`: the full `result` dict (verdict + every diagnostic: `dose_ladder`, `gated_band_n`, `gated_band_median`, `nearest_def_median`, `placebo_p95`, `dose_response_rho`/`_p`, `gated_band_zero_fraction`, `off_pitch_control_fraction`, `n_frames_used`, …), plus `regate_verdict`, `entanglement`, the pinned rule constants (from `PROBE_WRAPPERS["xs"]["rule_constants"]`), `per_match` provenance, `corpus` (provider/match ids/n), a **`reconciliation`** block (below), and the **reproducibility triple `baseline_commit` / `seed` / `tracking_limit`** (the placebo draws depend on `seed`, so all three are required for a "reproducible verdict").
- **`reconciliation` (the C3 guard against silent band-shrink).** The ghost engine's domain (ball ≤ `domain_ball_to_goal_m = 35 m`, `_engine.py:91`) and the probe's own `_eligible_groups` gate (arm="xs": `|bx − goal_x| ≤ 35` **plus** a probe-side `infer_ball_carrier` resolution + GK-mask, `_model_eval.py:160-167`) can *partially* disagree — `_targets_deltas` raises only when the overlap is **exactly zero** (`_model_eval.py:362`); a partial disagreement silently shrinks the band and can surface as `unmeasurable_at_dose`, which one would then *wrongly* read as "the ghost sits on the keeper." So the driver records `Σ per_match.n_targets → result.n_frames_used → result.gated_band_n` (plus `n_distinct_games`) and flags a targets→used drop in `report.md`. **A drop is non-zero *by construction*:** the ghost engine resolves the ball-carrier with the *ghost* model's `carrier_params` while the probe's `_eligible_groups` uses the *xs* model's (`_model_eval.py:130`), with no passthrough to align them — so the `> 0.5` flag means "larger than that baseline resolver mismatch," not "any drop is a bug." **R1 (distinct-game premise, guarded not asserted):** the pooled `evaluate_xs_probe` groups dose-response by `game_id` (needs `XS_PROBE_MIN_GAMES = 8`) and its duplicate-key guard keys on `(game_id, period_id, frame_id)`, so a `game_id` collision between two matches would either undercount games → `unmeasurable_at_dose` for the wrong reason, or raise cryptically. GS ids are 1:1 with `native_match_id`, so this is insurance — the driver **warns** if `n_distinct_games < n_contributing_matches` rather than assuming it. This makes the "plausible number from a computation that didn't happen" (spec §5) visible instead of merely reconstructable from the JSON.
- `report.md`: a `_render(metrics)` human-readable summary — headline verdict + **both** candidate branches (§6), the re-gate row, the ratio prong (`gated_band_median` vs `2× max(nearest_def_median, placebo_p95)`), the absolute-floor check, dose-response, the reconciliation drop, and the coverage/non-vacuity diagnostics.

---

## 4. The CI wiring test — `tests/gkdv/test_xs_probe_wiring.py`

**Purpose:** close the untested seam (§1.2) with a test that *cannot pass vacuously*. It runs the real chain on synthetic in-domain frames; it does **not** re-test the probe's pass/fail logic (that is already covered by `test_probe_discriminating_power.py` with hand-built targets).

Fixtures (reuse, do not invent): `tests/gkdv/_fixtures.py::multi_frame_in_domain(n_frames=…)` (built on `test_ghost_gk._make_ghost_gk_frames`, repositioned inside the 35 m domain) + `test_ghost_gk._fitted_model()[0]` (a real `GhostGkModel` instance).

Assertions, each with a stated red condition:
1. **Chain executes** — `build_ghost_frames → provenance_to_targets → xs_substitution_probe(planted_model("mixed"), frames, targets)` runs without raising. *Red if:* the seam is broken (schema/orientation mismatch, dtype, missing column).
2. **Contract** — the `provenance_to_targets` output equals `_TARGET_COLUMNS` and passes `_validate_targets`. *Red if:* the adapter drifts from the probe contract.
3. **Non-vacuity (the load-bearing assertion)** — the produced deltas contain at least one `actor_role == "gk"` row with `displacement_m > 0`, i.e. **the ghost actually moved the keeper**. *Red if:* the ghost collapses to the actual position (the exact silent-null this whole design guards against — see §5).
4. **Verdict shape** — the returned dict contains `"verdict"` in the registered value set and `"n_frames_used" >= 1`. *Red if:* the probe surface changes under us.

The test uses `planted_model("mixed")` (from `tests/tracking/_probe_fixtures.py`) as the scored xS model — a real callable with `predict_proba` and GK-dominant sensitivity — so a moved keeper produces non-zero `delta_p`, making assertion 3 discriminating rather than trivially true. It does **not** assert a specific `pass`/`fail` verdict (that would require replicating the in-domain frames to ≥8 games / ≥100 band rows and a specific ghost displacement, which is `test_probe_discriminating_power.py`'s job on hand-built targets).

---

## 5. Coordinate frame & the three silent-null traps (design obligations)

The recurring failure mode in this codebase is a plumbing detail manufacturing a "the GK doesn't matter" null (kloppy y-inversion, the `flat_zones` zone-176 origin, the mirrored StatsBomb Pressure events, the identity-keyed `PitchControlCache`). The design must not add a fifth.

1. **Coordinate frame — handled by construction (one impl-time check).** The loader emits frames in `output_convention="absolute_frame"`; `provenance_to_targets` converts the goal-relative ghost (`ghost_gr_x/y`) to absolute-frame `target_x/y` (the write-back flip), so targets and frames share one absolute frame. The probe derives its move vector as `target − gk_xy` off each frame's own GK row — consistent, and **no orientation code belongs in the driver**. The xS *feature* half (loader frames → `extract_xshot_features`) is the exact path the xS e2e already exercises, and both the ghost engine and the xS features resolve the defended goal from the GK map (`_gk_resolve.defended_goal_x` never reads `team_attacking_direction` — verified), so both are orientation-agnostic. **Not yet verified:** that `build_ghost_frames`'s *domain filter* (in-possession team attacking ∧ ball within 35 m of the attacked goal) behaves correctly on the loader's `absolute_frame` output — the e2e does not run the ghost engine. This is a cheap impl-time confirmation (assert a non-trivial `n_frames_scored` on one real GS match before the full run) and is listed in §10.
2. **`PitchControlCache` identity-keying — not reachable here.** The xS probe path does not build pitch-control surfaces (it re-extracts xS features via `predict_proba`), so the moved-GK/shared-cache trap does not arise on this arm. The driver passes **no** `pitch_control_cache` anywhere. (The trap is a physics-arm concern, out of Part A's scope.)
3. **DAS direction — not on this arm.** Δ-DAS is a physics-arm feature (Part-B/report territory), not part of the xS-probe chain. Not exercised here.

The one live trap for *this* arm is #1, and it is neutralized by the adapter; assertion 3 in §4 is the guard that proves it stayed neutralized.

---

## 6. Expected result & interpretation — two co-equal branches

Two mechanisms pull in opposite directions, so **both outcomes are pre-registered** and the run decides between them; do not assume the clean fail.

- **`unmeasurable_at_dose` (likely first-order — the "ghost-accuracy paradox").** The dose band needs `displacement_m ≥ XS_PROBE_DOSE_M = 2 m`, but the ghost's held-out MAE is ~1.07 m, so the actual keeper usually sits *within* 2 m of the ghost and the frame never enters the gated band. If `< XS_PROBE_MIN_BAND_N = 100` frames clear 2 m (or `< XS_PROBE_MIN_STRATUM_N = 50` in the trusted stratum), `evaluate_xs_probe` returns `unmeasurable_at_dose` (`_model_eval.py:534`), which re-gates to `unmeasurable_at_dose` (`:646`). This **closes nothing** — the instrument could not reach the dose; it is *not* a null effect, and the report must say so.
- **`gated_clean_fail` (the "clean fail").** If the band *does* fill, only `GK_r`/`GK_theta` (2/27 features) respond and `openGoal` excludes the keeper (`_xshot_occurrence.py:124`, `:207`), so the GK response is likely flat/small → `fail`/`band_pass_flat_dose_response` → `gated_clean_fail`. This *does* close the xS arm and names the concrete next lever (GK-aware xS features, ADR-011).
- **`joins`/`joins_with_caveat` (a genuine surprise).** A real GK response would be the finding of the cycle.

The run's **first** diagnostic is the band population (`gated_band_n` vs 100, via the §3.3 reconciliation), and only then the fail-vs-pass rule. Whichever lands, the value is the recorded, reproducible number replacing today's assumption.

---

## 7. Testing strategy

- **CI-gated (two tests, both dataless):** `tests/gkdv/test_xs_probe_wiring.py` (§4 — the `build_ghost_frames → provenance_to_targets → probe` **seam**, non-vacuous) **and** `tests/scripts/test_validate_xs_probe.py` (C4 — the driver's `run()` **orchestration** via a monkeypatched `load_matches` yielding two synthetic matches, so the per-match loop / delta-pool / `per_match` accumulation / empty-corpus `SystemExit` are covered by CI, not first exercised by the expensive owner run; plus the pure `re_gate`/`_render`/`_write`). Both run on every leg, no real data, no network. `[das]` is irrelevant here (the xS arm doesn't touch accessible-space).
- **Reported-not-gated:** the real run (`scripts/validate_xs_probe.py` on GS matches) — owner/local-triggered (needs `PINING_FOR_THE_DATA_TOKEN` + a long wall-clock, no artifact caching on the pining path). Its output (`docs/research/tf19_pr3b/`) is the deliverable, reviewed by a human, exactly like `docs/research/tf19_causal/` and `docs/research/xcross_causal/`.

There is no wall-clock assertion and no new CI dependency.

---

## 8. Non-goals (explicitly out of scope)

- **Part B / spec §6.4** — the model-free "world-test" discrimination harness (Layers 0–3), the 11-row `gkdv_discrimination_verdict`, the additive `causal/OpportunityConfig.outcome_max_distance_m` field, and deriving the placeholder constants (`N_min`, the `openGoal_with_GK` threshold). Gated on separate owner sign-off (the spec's §6.4 header: *"must be signed off before any constant is written into code"*).
- Any change to the xS model, its weights, or the probe rule constants.
- A library orchestrator (approach A) — deferred until a second caller exists.
- The physics arms' Δ-DAS / Δ-cover-shadow readings — those belong to the fuller discrimination story (Part B), not the xS-arm viability measurement.

---

## 9. File manifest

| File | Action | Notes |
|---|---|---|
| `scripts/validate_xs_probe.py` | **create** | driver; NOT a `_loader_*`/`calibrate_*` file, so within the part-deux session's allowed scope; it *reads* `_loader_pining` (read-only use of an isolation-zone module, permitted) |
| `tests/gkdv/test_xs_probe_wiring.py` | **create** | CI wiring test (the seam); reuses `tests/gkdv/_fixtures.py` + `tests/tracking/_probe_fixtures.py` |
| `tests/scripts/test_validate_xs_probe.py` | **create** | CI test of the driver's `run()` orchestration via a monkeypatched `load_matches` (C4) + the pure helpers (`re_gate`/`_render`/`_write`) |
| `docs/PRIVATE_CONSUMERS.md` | **modify** | record the test + driver as first-party consumers of the private `tracking._model_eval` probe symbols, with the exit condition "promote if a *cross-package* consumer appears" (C5 — keeps ADR-037's deliberate privacy; no promotion) |
| `docs/research/tf19_pr3b/{metrics.json, report.md}` | **generated by the run** | produced when the driver is run; committed with the code per the run-then-commit choice (C10) |
| `TODO.md:45` | **update on completion** | flip the "unblocked but never run" line once the run is recorded (rides this PR's single commit) |

No library (`silly_kicks/**`) file is modified. (C5: recording in `docs/PRIVATE_CONSUMERS.md` is a docs entry, not a library change; the probe stays private per ADR-037.)

---

## 10. Open questions — RESOLVED by the other-session review (2026-07-21)

1. **Unit of analysis / pooling.** RESOLVED: pool, but at the **deltas** level (§3.1, C2 — pooling raw frames is memory-fragile), with the targets→used→band reconciliation (§3.3, C3). "All GS matches" is the right unit (the public xS model is entirely GS-held-out).
2. **Corpus pin + reproducibility.** RESOLVED: pin via `--match-ids-json` (recommend the same GS set the other TF-19 runs used) AND record `seed` + `tracking_limit` alongside `baseline_commit` in `metrics.json` (§3.3, C8 — the placebo draws depend on `seed`).
3. **Non-vacuity strength.** RESOLVED: keep `≥ 1` for the *seam* test but raise the fixture to `multi_frame_in_domain(30)` (≈6 scored) so `.any()` is not a 2-sample coin flip (C6). The *fraction* guard belongs to the real run's band-population check (Task 5), a different guard.
4. **Entanglement input.** RESOLVED: consume the banked `inside_band`; do not re-run the causal arm. Noted it is inert on the expected `fail`/`unmeasurable` paths — it only bites on a `pass` surprise (§3.1, C7).
5. **Ghost-engine ↔ `absolute_frame` compatibility.** RESOLVED (the one genuinely unverified item, correctly flagged): the CI wiring test proves the *seam* on synthetic in-domain geometry, NOT that real `absolute_frame` output lands in the engine's domain — the exec-summary overclaim was corrected to "cannot be a *seam* artifact." Kept as the Task-5 impl gate (`report.n_frames_scored > 0` on one real GS match, `tracking.orient_frames_to_ltr` fallback, no library change).

**Applied dispositions:** C1 (pre-register both branches, §1/§6), C2 (pool deltas + pin RNG, §3.1), C3 (reconciliation, §3.3), C4 (monkeypatched `run()` CI test, §7/§9), C5 (keep private + `PRIVATE_CONSUMERS.md`, §2/§9), C6/C7/C8/C9, and the §5.1/exec-summary overclaim fix. C10 = run-then-commit (owner choice).

---

## Attribution
TF-19 GKDV cycle; probe machinery ADR-037 (PR-1); gkdv engine ADR-043 (PR-3). xS-occurrence model arXiv:2512.00203. This spec covers only the Part-A run; §6.4 (Part B) attribution/design lives in `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`.
