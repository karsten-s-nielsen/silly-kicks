# silly-kicks GK detection-gate wiring (TF-64) — design spec (Rev 3)

**Date:** 2026-10-01 · **Status:** DRAFT rev 3, applying the Rev-2 two-reviewer round (reviewer 1 APPROVE; reviewer 2 REQUEST CHANGES — one new SHOULD FIX, TF64-SPEC-07). For the next two-reviewer re-review round. Nothing built; no branch, no commit.
**Target repo:** `silly-kicks`, on `main` `949730e` (verified clean apart from this cycle's own in-flight docs). Every `file:line` is anchored on `main` `949730e` — ⚠ re-verify at plan time.
**Scope anchor:** ADR-109 ("per-player detection is a primitive PARALLEL to polygon-FOV") shipped the `detected_mask` primitive **and** explicitly deferred "the keeper-metric wiring (the ~13 GK outputs) … a separate, later change." **TF-64 is that change — wiring only.**

## Rev-3 change log (Rev-2 two-reviewer round)
- **TF64-SPEC-07 (SHOULD FIX, reviewer 2):** test 8 claimed "a future keeper-position consumer cannot ship ungated" without committing to a MECHANICAL enumeration — a hand-list is a self-certifying guard (the `pre_shot_gk_*` miss proves name-based enumeration fails), a false guarantee that lets the E7 hole recur green. Test 8 now DERIVES its population from code (the ADR-056 "a registry gate DERIVES its population and asserts it EXACTLY" pattern), asserting each keeper-position reader is gated or in an explicit out-of-scope allowlist (§4 test 8).
- **Reviewer-1 sweep floor:** the round-1 reviewer's 4 further candidates (`tracking/_gk_geometry.py`, `shot_stopping/_compute.py`, `positioning/_compute.py`, `tracking/_cover_shadows.py`) are named in §3.2 as the plan-time sweep FLOOR (not the maximum — test 8 derives the full population).
- The three owner scope calls (`pre_shot_gk_*` gate-vs-out; gk_decision native-tier caveat-only; TF-58 no-hard-ordering) are surfaced in §7, not findings. **`pre_shot_gk_*` is now DECIDED (owner, 2026-10-01): GATE** — folded into §2/§3.2/§7/test 8. The 4 reviewer-1 sweep-floor candidates remain plan-time dispositions (each returns to the owner if a genuine scope cut).

## Rev-2 change log (two-reviewer round)
- **B1 (BLOCKING):** inventory was not proven exhaustive + tests keyed on a count. Now an explicit NAMED output list (8 restdefense + Tier-B + gkdv + 4 influence = **14**), enumeration **by keeper-position-read (not by `gk_*` name)**, all 12 `gk_*` glossary entries classified, a glossary-discovery gate, and `pre_shot_gk_*` surfaced as a candidate the original audit did not list (§1/§3.2/§3.3/§4).
- **B2 (BLOCKING):** the wrapper's missing-keeper-row verdict was undefined (fail-OPEN). Now defined fail-closed; callers map **both** `False` and the missing/ambiguous case to honest-NaN/drop; a missing-row boundary test is registered (§3.1/§4).
- **S3:** the wrappers are exported as **public** `tracking` seams (consumers import public), with PRIVATE_CONSUMERS.md + per-package allowlist rows — avoids the cross-package private-import CI gate (§3.1/§5).
- **S4:** `features.py` anchor corrected (`silly_kicks/tracking/features.py`; the real gk sites are the `pre_shot_gk_*` block ~`:610-848` + the influence consumption ~`:3543`/~`:3952`, not `:3460-3464`) (§3.2).
- **S5:** TF-58 ordering softened to "whichever lands second rebases `_provider_visibility.py`"; TF-58 pinned as a parked branch, not on `main` (§5).
- **S6 + companion:** the Task-0b downstream sweep was DONE; §5 now names the real consumers + sizes the re-mat from evidence (not an asserted "one").
- CONSIDERs folded (gkdv `:47-57` + `_mark` near `:326`; `_reconstruct` `:176`/`:184`; native-tier/reconstruction-tier vocab; "byte-identical" = behaviour; minor bump; CI runner named).

---

## 0. What ADR-109 already shipped (TF-64 is wiring-only)
Verified on `main`: the primitive `detected_mask(visibility, *, provider, assume_observed=False)` (`silly_kicks/tracking/_provider_visibility.py:84`; fail-closed; sets `:22`/`:27`; `_detection_discarded_message:54`) **and** the D1 (parallel-not-unified) + fail-closed + general-opt-out decision (ADR-109) **already shipped**; `_ghost_gk.keeper_detection_mask:425` is a byte-identical delegating wrapper. So TF-64 needs **no new primitive and no new ADR** — it adds public keeper-row wrappers over `detected_mask` and wires the choke points.

## 1. Why
SkillCorner broadcast tracking detects the keeper in **17.6%** of live frames (public A-League MIT, 20 matches; ~86.9% when the opponent attacks her box, ~never past midfield). An audit found **14 keeper-position-dependent outputs run UNGATED on SkillCorner**, consuming ~80%-extrapolated keeper positions as if observed: **8 restdefense GK columns** (`rd_gk_line_height`, `rd_gk_to_line_distance`, `rd_danger_behind_line_gk`, `rd_gk_coverage_behind_line`, `rd_gk_reachable_coverage_m2`, `rd_num_superiority_gk`, `rd_gk_deter_threat`, `rd_gk_deter_space`), **`gk_decision` reconstruction-tier**, the **`gkdv` engine**, and the **4 `gk_influence` glossary features**. Polygon-FOV is inert (SB360-only). This is anti-pattern **E7** (metric on unobserved positions). TF-64 closes all 14 in one coherent change.

## 2. Scope
**In:** public keeper-row wrappers over `detected_mask` (§3.1); wiring the choke points for the 14 audited outputs **plus the owner-gated `pre_shot_gk_*` family** (owner decision 2026-10-01, §3.2/§7); glossary caveats + the mechanical keeper-position-reader gate (§3.3/§4 test 8); red-first tests (§4); versioning/CHANGELOG/downstream (§5).
**Out (explicit):** the primitive + parallel-not-unified ADR (shipped, ADR-109); new GK metric families; unifying detection with polygon-FOV; `gk_decision` native-tier's opaque vendor GI feed (§7, caveat-only); GKDV *measurement* policy (stays GradientSports-only).

## 3. Design

### 3.1 Public keeper-row wrappers (extend `tracking/_provider_visibility.py`, export via `tracking.__all__`)
```
resolve_detected_keeper_xy(frame_rows, *, provider, assume_keeper_observed=False) -> (x, y) | (nan, nan)
keeper_detected(frame_rows, gk_id, *, provider, assume_keeper_observed=False) -> bool | None
```
Delegate to `detected_mask`. **Complete, fail-closed verdict set (B2):**
- fully-observed provider → `(x,y)` / `True` (unchanged).
- detection-aware + keeper row `visibility True` → `(x,y)` / `True`.
- detection-aware + `visibility False` (extrapolated) → `(nan,nan)` / `False`.
- detection-aware + **keeper row absent or not exactly one** (the `_resolve_a_keeper_id -> None` case, `_danger.py:50`) → `(nan,nan)` / `None` meaning **not-observed**. **Not a skip-gate.**
- detection-aware + `visibility` entirely-null → **raise** (`_detection_discarded_message`).
- unknown provider → **raise**.
- `assume_keeper_observed=True` → ungated (the general opt-out; logged; never default; never consumer-specific).
**Caller contract:** every choke point maps **BOTH `False` and `None`** (and `(nan,nan)`) to honest-NaN/drop — treating `None` as "observed" is the fail-OPEN E7 hole this forbids. **Public** (`tracking.__all__`) so restdefense/gk_decision/gkdv import a public seam (S3); add PRIVATE_CONSUMERS.md + per-package `test_import_allowlist.py` rows if any consumer must stay on a private import.

### 3.2 Wiring (choke points, re-anchored to `main` `949730e`; re-verify at plan time)
| site | metric(s) | gate shape |
|---|---|---|
| `restdefense/_structure.py:72` (`_gk_x`; `:119`/`:124`) | `rd_gk_line_height`, `rd_gk_to_line_distance` | honest-NaN when undetected |
| `restdefense/_danger.py:50` (`_resolve_a_keeper_id`) | `rd_danger_behind_line_gk`, `rd_gk_coverage_behind_line`, `rd_gk_reachable_coverage_m2` | `None`/undetected → NaN dependents, counted |
| `restdefense/_counting.py` GK mask (via `_structure.py`) | `rd_num_superiority_gk` | **§7: honest-NaN the `_gk` variant; keep the non-GK count** |
| `restdefense/_arms.py` (`_frame_deltas` ~`:258`, actual leg) | `rd_gk_deter_threat`, `rd_gk_deter_space` | drop-and-count the frame |
| `tracking/_gk_influence.py:344` (`compute_gk_influence` gk_mask) + sibling gk_mask `:202`; consumed in `tracking/features.py` (the `add_gk_influence` path ~`:3543`/~`:3952`) | 4 `gk_influence` glossary features | honest-NaN when undetected |
| `gk_decision/_reconstruct.py` (`_drops` `:98`; FOV gate `:176`) | gk_decision reconstruction-tier | add a `keeper_undetected` counted drop parallel to `fov_cropped` (inc site `:184`) |
| `gkdv/_engine.py` (drop cascade `:47-57`; `_mark` near `:326`) | gkdv | add `_DROP_KEEPER_UNDETECTED`; never score Δ on an undetected keeper |
| `tracking/features.py:610-848` (`add_pre_shot_gk_position`/`add_pre_shot_gk_angle`) | `pre_shot_gk_x/y/distance/angle` | **honest-NaN when the pre-shot keeper is undetected (owner decision 2026-10-01 — GATE)** |

All drop/NaN paths feed the conserving Reports (`n_*` still sum to `n_in`).
**⚠ Exhaustiveness (B1/TF64-SPEC-07):** the 14 audited outputs + the gated `pre_shot_gk_*` family are the NAMED set. The enumeration criterion is **"reads a keeper frame position,"** NOT "name starts `gk_*`" — enforced MECHANICALLY by test 8 (§4), not by this hand-list. Sweep-floor status:
- **GATED (owner decision 2026-10-01, §7):** `pre_shot_gk_x/y/distance/angle` + `add_pre_shot_gk_position`/`add_pre_shot_gk_angle` (`tracking/features.py:610-848`) — reads the pre-shot keeper position → E7-exposed; wire it like the 14 (honest-NaN on an undetected keeper). Surfaced by the mechanical sweep; was not in the audited 14.
- **Still to disposition at plan** (reviewer-1 sweep): `tracking/_gk_geometry.py`, `shot_stopping/_compute.py`, `positioning/_compute.py`, `tracking/_cover_shadows.py`. Each is an owner scope call — an off-ball keeper-position read on SkillCorner is E7-exposed → gate; an on-ball / shot-frame read where detection is high may be out-of-scope-with-reason, but the reason must be stated and checkable (and any genuine scope cut returns to the owner). The floor is the minimum test 8 must surface, not the maximum.

### 3.3 Glossary + attribution
The **12** `name="gk_*"` glossary entries split: **4 gated** (`_M_GK_INFLUENCE`, frame-position-dependent) + **8 out-of-scope** (`_M_SPADL_UTILS`, event-derived, read NO keeper frame position: `gk_role`, `gk_was_distributing`, `gk_was_engaged`, `gk_actions_in_possession`, `gk_pass_length_m`, `gk_pass_length_class`, `gk_xt_delta`, `gk_completion`). The 4 gated get a data-quality `definition` note ("honest-NaN when the keeper is not detected on detection-aware providers"); the 8 get no change (documented reason: event-derived). No new columns → no `metric_contracts` change; no new methodology → `attribution=None`, no NOTICE.

### 3.4 Prerequisite (Task-0: RESOLVED) — the gate reads `visibility`, preserved by the native SkillCorner builder; the pining path was re-routed onto it. Re-confirm `scripts/_loader_pining.py` native routing at plan time. Detection-aware + null-visibility → the gate raises (loud).
### 3.5 Thresholds — boolean gate; the SB360 `fov_min_observed_fraction=0.7` does not transfer, untouched.

## 4. Registered tests (TDD, red-first)
1. **Per-metric non-vacuity over the NAMED set — the 14 + the gated `pre_shot_gk_*` family** (not "~13"): a REAL undetected keeper (`visibility=False`, not a missing row) moves each named output to NaN/dropped; a `visibility=True` fixture leaves it numeric. Vacuous-fixture guard.
2. **Missing-keeper-row boundary (B2):** a frame with no keeper row (or >1) → the wrapper returns the not-observed verdict AND each dependent output goes honest-NaN/drop (proves `None` is not read as observed).
3. **Fail-closed:** detection-aware + null `visibility` → raises; unknown provider → raises.
4. **General opt-out:** `assume_keeper_observed=True` reproduces pre-change values byte-for-byte.
5. **Fully-observed unaffected:** sportec/idsse/gradientsports/metrica byte-identical.
6. **Report conservation:** each new drop reason keeps `n_*` sums intact.
7. **Hyrum snapshot** of exactly which SkillCorner rows change.
8. **Keeper-position-reader gate (B1/TF64-SPEC-07) — MECHANICAL, not a hand-list.** A meta-test that **derives its population from code** (the ADR-056 "a registry gate DERIVES its population and asserts it EXACTLY" pattern): an AST/registry sweep of every frame-consuming `add_*`/`compute_*` that reads a keeper row (`is_goalkeeper`, a resolved `defending_gk_player_id`, or `_resolve_a_keeper_id`). It asserts each derived member is **either** detection-gated **or** present in an explicit, enumerated out-of-scope allowlist (each allowlist entry carries its stated reason). It must FAIL on a member that is neither — so a future keeper-position reader cannot ship ungated-and-unlisted. A hand-maintained list is explicitly forbidden here: it is self-certifying (the `pre_shot_gk_*` miss proves name/hand enumeration fails) and would ship a false "cannot ship ungated" guarantee. The derived population MUST include the §3.2 floor (the 14 gated outputs + the now-gated `pre_shot_gk_*` family + the 4 reviewer-1 candidates still to be dispositioned at plan); test 8 cannot go green until every derived member is gated or in the reasoned allowlist.
9. **CI gates:** `metric_contracts`, glossary coverage/NOTICE, SB360 boundary verdicts green. Runner: `ci.yml` runs `pytest tests/`.

## 5. Versioning, CHANGELOG, downstream (D4) + TF-58
- **Task-0b sweep — DONE (this cycle).** The lakehouse (repos `luxury-lakehouse` + its `-d32` mirror) consumes all three families: a **gk_decision mart** (incl. reconstruction-tier), a **gkdv mart** (`delta_das`/`delta_threat_suppression`), the **gk_influence** features, and the **restdefense GK columns**. PRIVATE_CONSUMERS.md already pins the gk_decision module paths + the gkdv-arms SB360 behaviour; the `rd_gk_*` columns are consumed by the restdefense mart (not individually pinned).
- **Re-materialise sizing (corrected from "one"):** the change honest-NaNs the 14 outputs on **undetected SkillCorner keepers only** → the re-mat is the **GK-mart set on SkillCorner-sourced data** (gk_decision + gkdv + restdefense-GK + gk_influence); velocity-bearing/optical providers are byte-identical.
- **Minor version bump** (`silly_kicks/_version.py`, ADR-079) + CHANGELOG `### Changed` naming the 14 columns now honest-NaN behind the fail-closed default, `assume_keeper_observed` opt-out documented. Hyrum: a disclosed behaviour change on SkillCorner GK values.
- **TF-58:** the parked `feat/tf58-team-coordination` branch (NOT on `main`; `_DEAD_BALL_OBSERVED_PROVIDERS` is not on `main`) also edits `_provider_visibility.py`. TF-64 depends only on the shipped primitive, so there is **no hard ordering** — **whichever of TF-58 / TF-64 lands second rebases `_provider_visibility.py`**; reuse TF-58's capability-function pattern + `_detection_discarded_message`; one feature branch per cycle, no worktrees, do not build on TF-58's unmerged work.

## 6. Commit discipline
One feature branch off `main`; one coherent fully-tested commit (wrappers + all wirings + tests + glossary + CHANGELOG — CI enforces atomicity). No commit without explicit per-commit approval. No worktree, no micro-commits.

## 7. Open questions — RESOLVED (⚠ scope reductions need owner sign-off)
- **`rd_num_superiority_gk` grain — DECIDED:** honest-NaN the `_gk` variant on an undetected keeper; keep the non-GK superiority count.
- **`gk_decision` native-tier (opaque vendor GI feed) — DECIDED: caveat-only, no code change (owner signed off).** We cannot audit SkillCorner's internal GI; add a known-limitation caveat to its glossary/report. The native-tier reads the opaque GI feed, not a frame keeper row, so it is NOT in test 8's swept population; a separate assertion covers that the caveat exists.
- **`pre_shot_gk_*` (B1 extension) — DECIDED (owner, 2026-10-01): GATE.** Reads the keeper position → E7-exposed; wire it like the 14 (honest-NaN the pre-shot keeper position on an undetected SkillCorner keeper). `add_pre_shot_gk_position`/`add_pre_shot_gk_angle` join the §3.2 wiring scope; test 8 derives them into the gated set. Low cost: detection is high on box-attack frames, so the honest-NaN rate is small.
- **Plan-time sweep floor (§3.2) — each dispositioned at plan:** `tracking/_gk_geometry.py`, `shot_stopping/_compute.py`, `positioning/_compute.py`, `tracking/_cover_shadows.py`.
- **Task-0b list — swept (§5).**

## 8. Related / overlap
The GK-observability investigation's S4 + S5 audit prompts (pressure phase-pooling; `team_metrics` pressing raw-count/block pooling) were specced separately as **TF-66** (E6/E2/E5). TF-64 = **E7 only**.

## 9. Review instructions
Independent **multi-pass `/review-spec`, two reviewer sources per round** (the GK-observability investigation session + a silly-kicks session), **≥3 passes total, agreement reported, model id + skill version per pass** (RESEARCH-DISCIPLINE Part F). Confirm: the named-14 exhaustiveness + the **mechanical** keeper-position-reader gate (test 8 DERIVES its population from code, ADR-056 pattern — reject any lingering hand-list; floor incl. `pre_shot_gk_*` + the 4 reviewer-1 candidates); the missing-row fail-closed verdict + its test; the public-seam + allowlist resolution; the re-anchored sites (`_arms` ~`:258`, `features.py` ~`:3543`/`:610-848`, `gkdv` `:47-57`, `_reconstruct` `:176`/`:184`); the TF-58 rebase framing; the `gk_decision` native-tier caveat-only scope reduction (owner signed off). Probe omissions (Part F).
