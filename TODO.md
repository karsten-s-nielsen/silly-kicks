# silly-kicks — TODO

Quick-reference action items. Architectural decisions live in [docs/superpowers/adrs/](docs/superpowers/adrs/).

**Last updated**: 2026-05-03. **Current release**: silly-kicks 3.1.0. Per-version history lives in [CHANGELOG.md](CHANGELOG.md).

---

## On Deck

Items are ranked top-to-bottom by specification completeness. Tier 2 is spec-complete and ready to implement directly; Tier 3–4 require empirical tuning or are heavier engineering; Tier 5–6 contain novel research components and have dependency cliffs (everything in Tier 5+ is blocked by TF-7 pitch control). Within each tier, items are ordered by additional implementation effort needed.

| Size | What it means |
|------|---------------|
| **Monstah** | Multi-phase epic |
| **Wicked** | Looks small, surprisingly impactful |
| **Dunkin'** | Quick run, keeps things moving |

### Tier 2 — Single published reference (ready to ship)

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-3 | `actor_distance_pre_window` — 0.5 s pre-action distance traveled | Dunkin' | Bauer & Anzer 2021 (counterpressing detection); New PR-S20 deferral | **Ready to ship.** Single canonical reference; uses existing `slice_around_event` (PR-S19). First time-window feature. **References:** Bauer, P., & Anzer, G. (2021), "Data-driven detection of counterpressing in professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049 — uses pre-action movement as feature. ~80 LOC. |
| TF-2 | `pressure_on_actor()` — multi-flavor pressure feature | Wicked | Andrienko et al. 2017; Link et al. 2016; Bekkers 2024; lakehouse `fct_defcon_pressure` (parity default); multi-flavor reframe (2026-05-02) | **Ready to ship.** Reframed from "pick one formula" to **ship all three formulas with a `method=` flavor parameter**, defaulting to `"defcon"` for cross-pipeline parity with the lakehouse `fct_defcon_pressure` mart. The three formulas encode genuinely different physical assumptions, not different conventions of the same metric: **`method="defcon"`** (Bekkers 2024, distance × angle — defender's blocking angle relative to attacker→goal line); **`method="andrienko_cone"`** (Andrienko 2017, cone-sum 1/d^k — directionally biased toward goal-cone); **`method="link_exp"`** (Link 2016, exponential decay — omnidirectional smooth falloff). Each method gets its own parameter knob (k for cone-sum, λ for exp decay, weights for DEFCON). **References:** Andrienko, G., Andrienko, N., Budziak, G., Dykes, J., Fuchs, G., von Landesberger, T., & Weber, H. (2017), "Visual analysis of pressure in football." Data Mining and Knowledge Discovery, 31, 1793-1839. Link, D., Lang, S., & Seidenschwarz, P. (2016), "Real Time Quantification of Dangerousity in Football Using Spatiotemporal Tracking Data." PLOS ONE, 11(12). Bekkers, J. (2024), DEFCON-style. Includes parity test against lakehouse mart for the default method. **Implementation note (column-name collision):** the kernel must emit flavor-suffixed column names (e.g., `pressure_on_actor__defcon`, `pressure_on_actor__andrienko_cone`), not a single fixed `pressure_on_actor` column. Consumers comparing flavors register parallel `functools.partial(pressure_on_actor, method="X")` xfns in `VAEP.xfns`; same-named columns would silently overwrite inside `VAEP.compute_features`. Sets the precedent for all future multi-flavor xfns; may warrant ADR-005 amendment alongside this PR if the convention feels cross-cutting enough at spec time. ~250-300 LOC + parity-test scaffolding. |
| TF-10 | Lakehouse boundary adapter for `add_action_context` outputs | Wicked | Lakehouse-side; tracked here for cross-repo visibility | **Ready to ship (lakehouse repo, not silly-kicks).** Wires PR-S20's 4 features into `fct_action_values` / new mart. Mechanical mapping; spec already documented in PR-S20 §11. References inherited from PR-S20 spec §11. |

### Tier 3 — Heuristic with empirical tuning

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-5 | `infer_ball_carrier(frames, tolerance_m=...)` | Wicked | Lakehouse session (2026-04-30); Bauer & Anzer 2021 (uses ball-carrier identification implicitly); ADR-004 #3 | **Heuristic + empirical-tuning.** Heuristic shape from Bauer & Anzer 2021 §3 (closest-player-with-velocity-toward-ball). Distance/velocity tolerances need empirical baseline against linked-events. No single canonical academic reference — most papers assume the carrier is given. **Pragmatic reference:** Bauer, P., & Anzer, G. (2021), Section 3 — describes carrier-identification heuristic similar to ours. Foundational utility; many downstream features will consume. ~150 LOC. |
| TF-13 | Frame-based defending-GK identification (fallback when events-based `defending_gk_player_id` is NaN) | Wicked | Bauer & Anzer 2021 (Section 3 carrier-ID heuristic, similar shape); Bekkers 2024 (DEFCON GK identification) | **Heuristic.** Defender closest to own goal at the linked frame, possibly conditional on jersey/role data when supplied. Composes with PR-S21's strict events-only ID (callers opt into fallback). ~80-120 LOC + ADR if chosen heuristic is contentious. |
| TF-14 | Defensive-line features (line height, line compactness, line break detection) | Wicked | Power et al. 2017 (line break in OBSO); Spearman 2018; Anzer & Bauer 2021 | **Spec choice + heuristic.** Per-frame defending team's outfield line geometry (median y of back-4, std dev, max gap). "Back-4 identification" method (k-means on y? rank-based? role-data-conditional?) needs spec choice. Could replace ad-hoc "defenders behind the ball" features in xG. ~150 LOC. |

### Tier 4 — Multi-paper synthesis (specified but heavier)

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-4 | `off_ball_runs` — attacking teammate runs in pre-action window | Wicked | Power et al. 2017 (OBSO); Spearman 2018; Decroos & Davis 2020 | **Multi-paper synthesis; scaling considerations.** Per-teammate temporal analysis. Heaviest of the deferred bench. **References:** Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017), "Not all passes are created equal." KDD '17 (OBSO — Off-Ball Scoring Opportunity). Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Decroos, T., & Davis, J. (2020), Player-Vectors blog/extension of VAEP. ~200-300 LOC + scaling considerations. |
| TF-7 | Pitch-control models (Spearman / Voronoi) | Monstah | Spearman 2018; Fernández & Bornn 2018; Spearman et al. 2017; ADR-004 #5 | **Foundational dependency for the entire GKDV research program (TF-15..TF-19) — every Layer-1 primitive consumes raw and threat-weighted pitch-control fields.** Three published implementations; numba acceleration, broadcast-tracking edge cases, validation harness. Own scoping cycle. **References:** Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017), "Physics-Based Modeling of Pass Probabilities in Soccer." MIT Sloan SAC. Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Fernández, J., & Bornn, L. (2018), "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." MIT Sloan SAC. |

### Tier 5 — Novel research with published anchor (lift + extend; blocked by TF-7)

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-15 | GK influence primitives — threat-weighted pitch-control share, GK reachable area, GK closing time | Wicked | Spearman 2018; Fernández & Bornn 2018; Get Goalside critique (raw pitch control over-credits the GK); 2026-05-01 deterrent investigation | **GKDV Layer 1.** **Depends on: TF-7, TF-13, TF-14.** Per-frame primitives isolating the GK's spatial influence: (a) `gk_pitch_control_share_weighted` — GK's share of pitch control weighted by threat-at-(x,y) so high control over the centre circle does not dominate; (b) `gk_reachable_area_m2` — area reachable within τ seconds via `r(τ) = max_speed × τ`, minus the defensive line's reachable area; (c) `gk_closing_time_to_zone(zone)` — time-to-intercept toward candidate cross-landing zones (six-yard box swarm). **Spec choice: threat-weighting recipe.** **References:** Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Fernández, J., & Bornn, L. (2018), "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." MIT Sloan SAC. Inherits TF-7 kernels. ~200 LOC. |
| TF-18 | Ghost-GK primitive — `expected_gk_position(state)` regression | Wicked | Le, Yue, Carr & Lucey 2017 (Data-Driven Ghosting); arXiv:2406.17220 (NFL Ghosts conditional density); novel for silly-kicks | **GKDV Layer 2 (counterfactual replacement primitive — most spec-complete of the Layer 2 trio).** **Depends on: TF-7, TF-14.** Given (ball_x, ball_y, defensive_line_x, period, score_state, possession_team), predict the league-average GK (x, y). Independently publishable as "league-average GK positioning given state." Start with regression (ridge / GBM) — well-defined ML formulation; upgrade path to conditional density (NFL Ghosts framework) once baseline lands. **References:** Le, H. M., Yue, Y., Carr, P., & Lucey, P. (2017), "Data-Driven Ghosting Using Deep Imitation Learning." MIT Sloan SAC. "NFL Ghosts: A framework for evaluating defender positioning with conditional density estimation." arXiv:2406.17220 (2024). ~200 LOC. |
| TF-16 | `xShotOccurrence` — P(shot taken in next k frames \| frame state) | Wicked | arXiv:2512.00203v2 (Beyond Expected Goals: Probabilistic Framework for Shot Occurrences); Anzer & Bauer 2021; novel for silly-kicks | **GKDV Layer 2 (paper-lift).** **Depends on: TF-7, TF-15.** Distinct from xG — predicts whether the on-ball player will *attempt* a shot, not whether the shot scores. Trained on per-frame state (ball x/y, defender geometry, GK position, defensive-line geometry) with binary label `did_attempt_shot_within_k_frames`. Shallow GBM; aligns with PR-S20/S21 feature pipeline and ADR-005 `_frame_aware` xfn marker. **Replication risk:** feature availability vs. our schema may diverge from paper; careful replication required. **References:** "Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in Soccer." arXiv:2512.00203v2 (2025). Anzer, G., & Bauer, P. (2021), Frontiers in Sports and Active Living, 3, 624475 (xGOT — GK position as feature). ~250 LOC + model artefact. |
| TF-17 | `xCrossAttempt` — P(cross attempted \| state) with GK-position confounders | Wicked | arXiv:2505.11841 (Framing Causal Questions in Sports Analytics: Crossing in Soccer); literature gap (their confounder set excludes all GK variables) | **GKDV Layer 2 (paper-lift + novel GK-confounder extension — most research-heavy item in Tier 5).** **Depends on: TF-7, TF-13, TF-14, TF-15.** Lifts the Cao et al. 2025 ATT/ATNT framework but **adds GK-position variables to the propensity model** (their headline gap: confounder set excluded all GK variables). Trained on wide-area frames with binary label `did_attempt_cross_within_k_frames`. Use propensity-score matching for ATNT estimation: among plays where a cross was *not* attempted, what would shot count have been if cross was attempted, *conditional on actual GK position*? **References:** "Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer." arXiv:2505.11841 (2025) — R `Matching` package, logistic propensity, 2,225 crosses / 30 matches CSL baseline (ATE 1.6%, ATT 5.0%). ~300 LOC + model artefact + matching harness. |

### Tier 6 — Novel synthesis (true research; blocked by Tier 5)

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-19 | **GKDV — GK Deterrent Value** composition + audit harness | Monstah | Le et al. 2017 (ghosting); arXiv:2505.11841 (causal-crossing); arXiv:2512.10355 (DEFCON-GNN — prevention-focused EPV-reduction); novel synthesis — closes 2026-05-01 deterrent-investigation literature gap | **GKDV Layer 3 — headline metric.** **Depends on: TF-15, TF-16, TF-17, TF-18.** For every frame in possession in the final third, compute `Δ_attempt(action) = P(action \| actual_GK) − P(action \| ghost_GK)` for action ∈ {shot, cross, key_pass}, weight by realized-or-expected outcome value, and sum across the build-up window. Negative GKDV ⇒ deterrent effect detected. **Validation strategy needs design:** GK-coach qualitative annotation set; correlation with conceded xG; expected-sign test on known sweeper-keepers (Alisson, Ter Stegen, Neuer should score strongly negative on aggressive presses) vs. line-keepers. Sample-size analysis required. **References:** Le et al. 2017 (ghosting). "Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer." arXiv:2505.11841 (2025). "Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks." arXiv:2512.10355 (2025) — DEFCON-GNN; compose-able comparator (per-defender EPV-reduction credit). Two PR cycles' worth realistically; spec + plan as own scoping cycle; likely needs its own ADR. ~400 LOC + audit notebook + held-out validation set. |

**GKDV research program (TF-15..TF-19).** TF-15 through TF-19 form a coherent research arc towards the **GK Deterrent Value** metric — a per-frame measure of how much the defending goalkeeper's actual position depresses opponent attempt-probabilities (shot / cross / key pass) relative to a league-average "ghost" GK in the same frame state. Origin: 2026-05-01 deterrent-effect investigation; closes the published-literature gap that no GK-evaluation framework today (StatsBomb four-pillar, Lamberts GVM, Sloan SAC data-driven framework, Anzer & Bauer xGOT, Driblab Goals Prevented) measures positioning-as-deterrent. **Foundation:** TF-7 (pitch control) + TF-13 (frame-based GK ID) + TF-14 (defensive-line geometry). **Layer 1:** TF-15 (threat-weighted GK influence primitives). **Layer 2:** TF-16 / TF-17 / TF-18 (decision-probability surfaces for shot-occurrence and cross-attempt + ghost-GK regression). **Layer 3:** TF-19 (composition + validation harness). **Sequencing:** TF-15 lands first as a self-contained publishable primitive; TF-16/17/18 can be parallelised (TF-18's regression baseline is the most spec-complete of the trio, TF-17's GK-confounder extension is the most research-heavy); TF-19 is two PR cycles and gets its own ADR.

---

## Active Cycle

(none currently active — see On Deck for next candidates)

---

## Technical Debt

### Blocked or Deferred

(none currently queued)

---

## Research & Future Work

ReSpo.Vision tracking adapter — licensing-blocked. Track here when licensing clears.

**TF-23 (Dunkin'/Wicked):** Native Sportec-XML / Metrica-CSV loader inside silly-kicks. **Engineering-ready, priority-deferred.** Currently the architectural intent (per `silly_kicks/spadl/sportec.py` docstring) routes raw-XML / raw-CSV ingestion through the kloppy gateway; native converters take pre-DataFrame'd events. PR-S23 deferred adding native loaders. Pure parser implementation (no metric design); replicable from kloppy's source + provider XSD/CSV schemas. Surface this only if/when consumer demand exists for a kloppy-free ingestion path.
