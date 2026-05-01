# silly-kicks — TODO

Quick-reference action items. Architectural decisions live in [docs/superpowers/adrs/](docs/superpowers/adrs/).

**Last updated**: 2026-05-01 (PR-S21 cycle in flight — `feat/pre-shot-gk-plus-baselines`)
**(A) silly-kicks 2.8.0 SHIPPED** (PR-S20 — `action_context()` tracking-aware features + VAEP integration; ADR-005).

---

## On Deck

| Size | What it means |
|------|---------------|
| **Monstah** | Multi-phase epic |
| **Wicked** | Looks small, surprisingly impactful |
| **Dunkin'** | Quick run, keeps things moving |

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-2 | `pressure_on_actor()` — pressure feature with formula choice | Wicked | Andrienko et al. 2017; Link et al. 2016; Bekkers 2024; lakehouse `fct_defcon_pressure` (9,826 rows formula reference) | **Formula-choice spec decision.** **References:** Andrienko, G., Andrienko, N., Budziak, G., Dykes, J., Fuchs, G., von Landesberger, T., & Weber, H. (2017), "Visual analysis of pressure in football." Data Mining and Knowledge Discovery, 31, 1793-1839 (cone-sum 1/d^k). Link, D., Lang, S., & Seidenschwarz, P. (2016), "Real Time Quantification of Dangerousity in Football Using Spatiotemporal Tracking Data." PLOS ONE, 11(12) (exponential decay). Bekkers, J. (2024), DEFCON-style (distance × angle). Inherit whichever formula the lakehouse `fct_defcon_pressure` mart implements (probe at PR-S22 start to confirm which) for cross-pipeline parity. ~150 LOC + possible ADR-006 if formula choice is contentious. |
| TF-3 | `actor_distance_pre_window` — 0.5 s pre-action distance traveled | Dunkin' | Bauer & Anzer 2021 (counterpressing detection); New PR-S20 deferral | **References:** Bauer, P., & Anzer, G. (2021), "Data-driven detection of counterpressing in professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049 — uses pre-action movement as feature. First time-window feature; uses `slice_around_event` (PR-S19). ~80 LOC. |
| TF-4 | `off_ball_runs` — attacking teammate runs in pre-action window | Wicked | Power et al. 2017 (OBSO); Spearman 2018; Decroos & Davis 2020 | **References:** Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017), "Not all passes are created equal." KDD '17 (OBSO — Off-Ball Scoring Opportunity). Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Decroos, T., & Davis, J. (2020), Player-Vectors blog/extension of VAEP. Per-teammate temporal analysis. Heaviest of the deferred bench. ~200-300 LOC + scaling considerations. |
| TF-5 | `infer_ball_carrier(frames, tolerance_m=...)` | Wicked | Lakehouse session (2026-04-30); Bauer & Anzer 2021 (uses ball-carrier identification implicitly); ADR-004 #3 | Heuristic per-frame carrier inference (closest-player-to-ball-with-velocity-toward-ball). No single canonical academic reference — most papers assume the carrier is given. **Pragmatic reference:** Bauer, P., & Anzer, G. (2021), Section 3 — describes carrier-identification heuristic similar to ours. Foundational utility; many downstream features will consume. ~150 LOC. |
| TF-6 | `sync_score` per-action tracking↔events sync-quality score | Dunkin' | ADR-004 #4; novel utility (no canonical academic reference) | QA primitive. Reuses `link_actions_to_frames` pointers + `link_quality_score`. No academic citation required — this is library-internal QA, not a published metric. ~50 LOC. |
| TF-7 | Pitch-control models (Spearman / Voronoi) | Monstah | Spearman 2018; Fernández & Bornn 2018; Spearman et al. 2017; ADR-004 #5 | **References:** Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017), "Physics-Based Modeling of Pass Probabilities in Soccer." MIT Sloan SAC. Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Fernández, J., & Bornn, L. (2018), "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." MIT Sloan SAC. Numba acceleration, broadcast-tracking edge cases, validation harness. Own scoping cycle. |
| TF-8 | Smoothing primitives (Savitzky-Golay, EMA) | Dunkin' | Savitzky & Golay 1964 (canonical); ADR-004 #6 | **References:** Savitzky, A., & Golay, M. J. E. (1964), "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." Analytical Chemistry, 36(8), 1627-1639. Per-provider preprocessor. ~80 LOC. |
| TF-9 | Multi-frame interpolation / gap filling | Dunkin' | Standard numerical methods (no domain-specific paper); ADR-004 #7 | Standard cubic-spline / linear interpolation. No domain-specific citation; standard signal-processing practice. ~100 LOC. |
| TF-10 | Lakehouse boundary adapter for `add_action_context` outputs | Wicked | Lakehouse-side; tracked here for cross-repo visibility | Wires PR-S20's 4 features into `fct_action_values` / new mart. Not in silly-kicks repo; logged for coordination. References inherited from PR-S20 spec §11. |
| TF-12 | `pre_shot_gk_angle_*` (signed angle from goal-line normal, off-line angular displacement, etc.) | Dunkin' | Anzer & Bauer 2021; PR-S21 deferral | Library ships positions + 2 distances in PR-S21; angle conventions deferred. Multiple competing definitions (relative to shot trajectory? to goal-line normal? signed vs unsigned?) — pick one canonical convention with downstream reviewer input before landing. ~30-50 LOC. |
| TF-13 | Frame-based defending-GK identification (fallback when events-based `defending_gk_player_id` is NaN) | Wicked | Bauer & Anzer 2021 (Section 3 carrier-ID heuristic, similar shape); Bekkers 2024 (DEFCON GK identification) | Heuristic: defender closest to own goal at the linked frame, possibly conditional on jersey/role data when supplied. Composes with PR-S21's strict events-only ID (callers opt into fallback). ~80-120 LOC + ADR if chosen heuristic is contentious. |
| TF-14 | Defensive-line features (line height, line compactness, line break detection) | Wicked | Power et al. 2017 (line break in OBSO); Spearman 2018; Anzer & Bauer 2021 | Per-frame defending team's outfield line geometry (median y of back-4, std dev, max gap). Could replace ad-hoc "defenders behind the ball" features in xG. ~150 LOC. |

---

## Active Cycle

PR-S21 — TF-1 (`pre_shot_gk_position_*`) + TF-11 (baselines backfill)
(target silly-kicks 2.9.0).

Branch: `feat/pre-shot-gk-plus-baselines`. Spec + plan: [docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md](docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md), [docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md](docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md).

After ship, this section gets archived; PR-S21 ships within the existing
ADR-005 envelope — no new ADR.

---

## Technical Debt

### Blocked or Deferred

(none currently queued)

---

## Research & Future Work

ReSpo.Vision tracking adapter — licensing-blocked. Track here when licensing clears.
