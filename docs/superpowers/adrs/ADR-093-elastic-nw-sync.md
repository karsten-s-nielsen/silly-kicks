# ADR-093: Event↔tracking synchronization via extended Needleman–Wunsch (ELASTIC v2) replaces the greedy TF-43 aligner

| Field | Value |
|---|---|
| **Date** | 2026-09-11 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

TF-43 shipped ELASTIC's MLSA-2025 incarnation: a per-action **greedy** argmax that, for each event independently, scanned a ±1 s frame window and picked the frame maximizing `0.6·ball_accel + 0.4·player-ball-proximity`. Measured against Kim et al.'s CC-BY benchmark (3 re-annotated Sportec Open matches — J03WMX/J03WN1/J03WPY) under our **cross-source** conditions (Kim's events vs our DFL tracking, vote-map anchored): **TF-43 ≈ 8.2 % W2** — the greedy collapses under the ~0.8 s cross-source jitter (the paper's own greedy manages 84.1 % W2 on its same-source data, and its NW reaches 96.5 % W2). The greedy flaw is structural: each event is placed independently, so a mistake cannot self-correct, mistakes cluster into cascades, and it is fragile to timing jitter.

ELASTIC v2 (Kim et al., CIKM 2026, arXiv:2608.30227) reframes synchronization as a **global, order-preserving sequence alignment** (extended Needleman–Wunsch) between the event sequence — enriched with **virtual termination events** so each event's end (reception/out/goal) is found jointly with its start — and a sparse set of physically-plausible ball-touch candidate frames. It uses no neural nets (scipy peak detection + a numpy DP). The reference code is **MPL-2.0**; this repo is MIT.

Forcing function: TF-57 (owner-routed 2026-09-11) to close the gap. The validation oracle was established last session; the algorithm was fully transcribed from the paper this cycle.

## Decision

Replace the greedy MLSA-2025 aligner with a **clean-room NW reimplementation from the paper** (never lifting the MPL code), delivered through two surfaces over one pure engine: the upgraded elastic mart producer (`align_events_to_frames` / `add_elastic_sync` / `elastic_sync_xfns` + atomic mirror, keeping `elastic_frame_id`/`elastic_confidence`/`elastic_error_seconds` — *values move* — and **adding** the reception columns `elastic_receive_frame_id`/`_confidence`/`_error_seconds`), and a new `link_actions_to_frames_elastic` returning the canonical `(pointers, LinkReport)` contract as an alternative strategy usable across the `add_*` family via `links=`. The guarded time-based `link_actions_to_frames` (ADR-004/017) is untouched and remains the default. ELASTIC-NW **requires continuous tracking**: on declared velocity-unavailable-by-design freeze-frames (`validate_velocity_regime == POSITIONAL_ONLY`, e.g. StatsBomb-360) it returns an honest empty alignment (all-NaN) rather than fabricate a sync from disconnected snapshots.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep greedy | no change | 8.2 % exact; cascade-prone; the constant-0.6 ADR-019 bug class | measured ~11× behind SOTA; owner directed the upgrade |
| B. Lift ELASTIC's reference NW + metric | fastest | MPL-2.0 into MIT | licence-incompatible (the PathCRF precedent); reimplemented instead |
| C. Learned ball-touch (PathCRF-style) candidate detector | closer to SOTA | MPL + large scope; a learned model | out of scope; kinematic peaks suffice to close most of the gap |
| D. (chosen) Clean-room NW + two surfaces + freeze-frame refusal | closes the gap on the fair (W2) metric; hexagonal; honest on freeze-frames | breaking `ElasticSyncParams`; mart re-materialize | — |

## Consequences

### Positive

- **Substantially closes the gap on the paper's metric (W2).** The paper (arXiv:2608.30227) adopts **W2 (within-2-frames, 0.08 s) as its PRIMARY metric** and reports no exact-frame headline (Table 2, event start: ELASTIC-NW **96.5 %** W2, ELASTIC-Greedy **84.1 %** W2, on its own same-source data). Our DGX 3-match headline (our DFL tracking + an approximate `home_N→DFL` vote-map — the *same* cross-source conditions as TF-43's 8.2 % W2): **W2 = 0.862** (per-match 0.848 / 0.847 / 0.891; min-fold 0.847; reception W5 mean 0.869), **≈ 11× the greedy** — **above the paper's own greedy (0.841 W2)** and **~10 pts below the paper's same-source NW (0.965)**. That residual is the **cross-source event-time jitter** (~0.8 s Kim-vs-DFL vote-map anchor; median ~20 frames; per-event, not drift), **NOT the alignment algorithm or player identity**: an exhaustive clean-room lever sweep (scoring via OpenEvolve, ball height, velocity-direction, a time-proximity prior, a local time re-anchor) found each tapped or dead. Reached via three cumulative wins over the as-built NW — central-difference acceleration (paper Eq. 3), in-play (`ball_state=="alive"`) episode grouping, and an OpenEvolve-tuned `_score` (owner-approved 2026-09-14). *(This supersedes an earlier record that cited "start exact 29.0 %" against an "88.4 % exact" target — a metric misreading: the paper has no exact-frame headline and 88.4 % is not its NW figure. Corrected 2026-09-14 against the paper as read; the authoritative comparison is on W2.)*
- **Joint start + reception detection** (three additive columns), feeding `slice_around_event` and matching the benchmark's `receive_frame_id`.
- **A second linker under the canonical contract**, so any `add_*` tracking feature can be computed on NW alignment (`links=`) instead of time-nearest.
- **Honest freeze-frame behaviour**: elastic on SB360 now produces nothing (with the continuous-only precondition enforced), where the greedy path emitted silent spurious values.
- A committed CC-BY oracle + a CI regression gate on sync accuracy (would have caught the constant-0.6 bug).

### Negative

- **Breaking:** `ElasticSyncParams` and `add_elastic_sync`'s keyword signature change (greedy weights removed); no shim (fail-loud, per the repo's clean-break precedent).
- **Mart re-materialize:** `elastic_*` values move + 3 new columns → lakehouse re-materializes `elastic_*`; any VAEP consumer that opted the `elastic_sync_xfns` columns in retrains (the xfns are in NO default list, so there is no default-config retrain).
- The oracle is player-mapping-limited (a conservative floor); the headline is the Claude-run DGX report, not the committed gate.
- **The `_score` weights are OpenEvolve-TUNED, not paper constants (owner-approved 2026-09-14).** The paper-faithful uniform-weight path had plateaued; the tuning (`w_ba/w_pbd/w_kd/w_dyn` + a directional slope term) is a small, removable lift (revert-to-uniform ≈ −0.006 W2 / −0.05 oracle-exact, gates stay green). It is a data-fit, **never** an MPL-source read — the algorithm, categories and feature set stay clean-room from the paper. The tuning-run provenance (objective/folds/seeds/config) + a **held-out** (leave-one-match-out) validation land in the paired **provenance commit** (two-commit pattern, §16.9), run clean-tree on the DGX against `e611aa1`: **mean held-out W2 0.860 ≈ in-sample 0.862** — the LOO-refit weights sit in the shipped neighborhood and the shipped vector ranks top-3–8 of 27 per fit-pair, so the ship bar is **not** in-sample-inflated (`docs/research/tf57_elastic_scoring_tuning/`). The committed-oracle floor is in-sample and is the regression guard, not the acceptance number. `test_dataclass_defaults_are_paper_intent_set_constants` is scoped to the dataclass constants, not the tuned scorer weights.

### Neutral

- No new `add_*` aggregator (C4 aggregator count stays 33; the elastic linker is a linker), but the 3 new `elastic_receive_*` glossary columns bump the C4 feature-column count 399→402 (399 is the post-TF-62 base), so `architecture.dsl`/`architecture.html` are re-rendered (Graphviz `dot`).
- Attribution: Kim et al. 2026 (arXiv:2608.30227) + the CC-BY benchmark (Bassek et al. 2025, doi:10.1038/s41597-025-04505-y) in NOTICE.

## References

- Spec: `docs/superpowers/specs/2026-09-11-tf57-elastic-nw-sync-design.md`; plan: `docs/superpowers/plans/2026-09-11-tf57-elastic-nw-sync.md`.
- ADR-004/017 (linkage + LinkReport), ADR-019 (id_compat), ADR-063 (declared-unavailable → honest degrade), ADR-053 (SB360 audit), ADR-005 (attribution/NOTICE), ADR-073 (sub-quadratic guard).
