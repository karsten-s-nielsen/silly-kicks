# silly-kicks — TODO

Quick-reference action items. Architectural decisions live in [docs/superpowers/adrs/](docs/superpowers/adrs/).

**Last updated**: 2026-05-04. **Current release**: silly-kicks 3.2.0. Per-version history lives in [CHANGELOG.md](CHANGELOG.md).

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
| TF-10 | Lakehouse boundary adapter for `add_action_context` outputs | Wicked | Lakehouse-side; tracked here for cross-repo visibility | **Ready to ship (lakehouse repo, not silly-kicks).** Wires PR-S20's 4 features into `fct_action_values` / new mart. Mechanical mapping; spec already documented in PR-S20 §11. References inherited from PR-S20 spec §11. |

### Tier 3 — Heuristic with empirical tuning

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-5 | `infer_ball_carrier(frames, tolerance_m=...)` | Wicked | Lakehouse session (2026-04-30); Bauer & Anzer 2021 (uses ball-carrier identification implicitly); ADR-004 #3 | **Heuristic + empirical-tuning.** Heuristic shape from Bauer & Anzer 2021 §3 (closest-player-with-velocity-toward-ball). Distance/velocity tolerances need empirical baseline against linked-events. No single canonical academic reference — most papers assume the carrier is given. **Pragmatic reference:** Bauer, P., & Anzer, G. (2021), Section 3 — describes carrier-identification heuristic similar to ours. Foundational utility; many downstream features will consume. ~150 LOC. |
| TF-13 | Frame-based defending-GK identification (fallback when events-based `defending_gk_player_id` is NaN) | Wicked | Bauer & Anzer 2021 (Section 3 carrier-ID heuristic, similar shape); Bekkers 2024 (DEFCON GK identification) | **Heuristic.** Defender closest to own goal at the linked frame, possibly conditional on jersey/role data when supplied. Composes with PR-S21's strict events-only ID (callers opt into fallback). ~80-120 LOC + ADR if chosen heuristic is contentious. |
| TF-14 | Defensive-line features (line height, line compactness, line break detection) | Wicked | Power et al. 2017 (line break in OBSO); Spearman 2018; Anzer & Bauer 2021 | **Spec choice + heuristic.** Per-frame defending team's outfield line geometry (median y of back-4, std dev, max gap). "Back-4 identification" method (k-means on y? rank-based? role-data-conditional?) needs spec choice. Could replace ad-hoc "defenders behind the ball" features in xG. ~150 LOC. |
| TF-24 | `LinkParams.k3` Optuna calibration (post-PR-S25) | Wicked | Lakehouse session 2026-05-03 (k3 calibration tooling discussion) | **Post-release calibration.** Optimize `LinkParams.k3` (and optionally the joint 6-scalar Link parameter set: r_hoz/r_lz/r_hz + angle_hoz_lz_deg / angle_lz_hz_deg + k3) via Optuna TPE sweep, 50-100 trials, single CPU node, against held-out VAEP fold from lakehouse `bronze.model_validation_runs`. Single objective: VAEP held-out Brier-score (or per-action calibration NLL). Update `LinkParams` defaults; update spec note from "engineering choice" to "Optuna-calibrated against `<fold_name>` on `<date>`". Same script reusable for k1..k5 if scope expands. **Wrong tool: lakehouse `evolve` framework** — single-scalar Bayesian optimization is Optuna-shaped, not evolve-shaped. Pre-release optimization avoided to (a) prevent circular validation against the training fold, (b) match the Link 2016 paper's own "formula + later empirical calibration" sequencing. ~1-2 days. |
| TF-25 | Structural-form evolution of pressure aggregations | Wicked–Monstah | Lakehouse session 2026-05-03 (k3 calibration tooling discussion); lakehouse `evolve` framework | **Lakehouse-evolve-shaped follow-up to TF-24.** Use lakehouse evolve framework to evolve the aggregation function FORM (not just k3 scalar). Three concrete targets: (1) per-provider saturation forms — different sample rates, position-noise characteristics, and field-coverage assumptions across Sportec/StatsBomb 360/Metrica/Wyscout may demand different aggregations; (2) continuous `r_zo(α)` as alternative to three-zone bucketing; (3) non-linear distance-pressure curves beyond `1 - d/r_zo` (quadratic, sigmoid, learned-curve). **Trigger condition:** only fire if TF-24's Optuna sweep shows k3 itself moves meaningfully across providers — that's the signal that the FORM, not just the scalar, is provider-dependent. Without that signal, this is over-engineering. ~1-2 cycles + L40S budget for eval loop. |

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
