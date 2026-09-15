# ADR-094: TF-52 event-only team-KPI module (`silly_kicks.team_metrics`)

| Field | Value |
|---|---|
| **Date** | 2026-09-15 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

silly-kicks carries per-player quality metrics (TF-54 territory, TF-55 duels), goalkeeper metrics (the GKDV / xt-gk / shot-stopping cluster) and per-action tracking features, but no **team-level match-report KPIs** — the pressing / progression / build-up numbers a coach or analyst reads off a match report (Twelve.football glossary; MSC / Tigres-Clemson practitioner sets). Every KPI in that set is computable from **SPADL events alone** (possession segmentation + geometry), so it belongs in the event-only `compute_*` tier next to `territory` / `duels` / `shot_stopping`, not in the tracking layer.

Two forcing functions shaped the work. **Scope:** the owner set scope to "(C) v1 Twelve glossary + the events-based v2 practitioner items" — so the tracking-flavoured v2 items (interception-height relative to the opponent line; compactness-recovery-time) and the aerial chaining (SPADL has no aerial type) are out of this module by construction. **Reproducibility:** the owner required the validity evidence to be a **public, reproducible** artifact rather than an owner-tier corpus run — thousands of public event matches exist (StatsBomb open data; the public Wyscout / Pappalardo 2019 set), so the reliability study runs on those.

The possession foundation the KPIs rest on already exists (`spadl.add_possessions`), which dissolved the originally-assumed dependency on a TF-52-owned segmenter — the module consumes it rather than reimplementing it.

## Decision

Ship one new **event-only** package **`silly_kicks/team_metrics/`**. It is a `compute_*` (NOT an `add_*` action-coupled aggregator — the C4 aggregator count stays 33), emits one row per `(game_id, team_id)`, mirrors the frozen-`Params` / `_columns` schema / `_report` census idiom, imports `spadl` + `id_compat` ONLY (never `tracking`; AST import-allowlist gate, and nothing imports it), canonicalises ids (ADR-019), drop-and-counts a non-two-team match honestly (ADR-042), documents every metric column in `feature_glossary` + `NOTICE`, and adds one C4 container. Additive — **no VAEP / tracking retrain, no re-materialize.**

1. **Three KPI families over one possession foundation.** `compute_team_kpis(actions, *, xg_column=None, params)` builds the possession context once (via `spadl.add_possessions`, threaded through `TeamKpiParams`) and assembles: **pressing** (`_pressing.py` — PPDA, defensive intensity, time-to-defensive-action/recovery, recoveries + within-Ns %, counter-press regains), **progression** (`_progression.py` — field tilt, pass tempo, long-ball %, the three line heights, the conversion chain, final-third entries, shots, high-opportunity shots, breakout-by-channel, possessions-retained-after-Ns), and **build-up** (`_buildup.py` — the 6-state build-up outcome taxonomy, post-regain security, switch-conditioned press success). Own-touch KPIs use each team's own action-LTR frame; opponent-frame reconciliation (PPDA denominator) is library-owned via `_orientation.py` (`silly_kicks.reflection`, ADR-028/045).

2. **The within-Ns post-recovery companion is the transition-output block (owner-ratified Option B, commit-gate).** §4.5 specified "the offensive metrics" without enumerating them; the owner ratified the set as the four per-action **counts** `final_third_entries` / `box_touches` / `shots` / `high_opportunity_shots`, re-computed inside `post_recovery_window_seconds` of a recovery. This required adding **two new base KPIs** — a plain `shots` count (the module previously had only the xG-gated `high_opportunity_shots`) and a raw `final_third_entries` count (a per-action ball-crossing of `x = 2·field_length/3`, complementing the possession-rate `poss_to_final_third_pct`). A *breakout* stays **excluded on correctness grounds** — it is one channel per possession, so a per-possession event has no per-action-window restriction — as do possession-rates; these are correctness exclusions, not scope cuts.

3. **`TeamKpiParams` frozen, `for_provider` EMPTY (ADR-009).** Geometry defaults from `spadlconfig` (ADR-050); the counter-press window is a seconds-XOR-passes value object (`CounterpressWindow`, `__post_init__`-enforced) with documented named presets (`COUNTERPRESS_PRESETS`). Per-provider overrides ship empty until a separate gated apply-PR.

4. **The reliability study is a public-corpus, reported-not-gated driver.** `scripts/validate_team_kpi_reliability.py` runs `compute_team_kpis` over StatsBomb open data (defaulting to the **entire open-data manifest** — `all_open_competitions()`, thousands of matches; a single tournament is far too thin for a team-discrimination ICC / split-half) + the public Wyscout / Pappalardo set, and writes an aggregate report with three legs: per-KPI **reliability** (team-discrimination ICC(1) + odd/even split-half + Type-II slope), **possession-foundation ground truth** (`add_possessions` boundary recall / precision / F1 vs the provider's native possession id, threaded via `preserve_native`), and per-provider **comparability** (`--compare`; a KPI is flagged poolable only where both providers report finite same-sign ICC within a tolerance). It is an ADR-052 `for_each` shard driver with ADR-037 clean-tree provenance + an ADR-056 input contract; the pure stat + shaping kernels are CI-tested, the corpus orchestration is owner-run. **Public-only is FAIL-CLOSED, corpus-appropriately (ADR-038) — NOT the pining `assert_public_corpus`, whose 27-match redistribution registry cannot represent an open-data corpus**: the StatsBomb leg refuses to run with credentials configured (`assert_statsbomb_open_data_mode`; no creds ⇒ statsbombpy resolves the open-data manifest only), and the Wyscout leg allowlists the seven public Pappalardo competitions. It **recommends, never applies** (ADR-009).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Put team KPIs in the tracking layer | reuse `team_shape` / defensive-line | forces tracking on an event-only concern; shrinks provider coverage to tracking providers | every v1/events-v2 KPI is event-computable; the tracking-flavoured items were scoped out |
| B. Reimplement possession segmentation inside TF-52 | self-contained | duplicates `spadl.add_possessions`; a second segmenter to keep in sync | consume the existing primitive; the assumed dependency dissolved |
| C. Post-recovery companion = the 2 existing counts (`box_touches`, `high_opportunity_shots`) only | no new base KPIs | the module has no plain shots count and is xG-dependent for its only shot metric; a thin transition-output family | owner ratified Option B — add `shots` + `final_third_entries` for the full counter-attack-output block |
| D. Post-recovery companion = also `breakout_*` (the "original 5") | richer | a breakout is per-possession; restricting "which channel" to a per-action window is incoherent | correctness exclusion, not a scope cut |
| E. Validate on a small owner-tier corpus | quick | not reproducible by third parties | owner required a public, reproducible artifact — StatsBomb open + public Wyscout |
| F. (chosen) event-only `compute_*` package + public-corpus reliability driver | mirrors the sibling idiom; reproducible; additive | a new sibling package trips ~8 repo-wide gates that must be wired | — |

## Consequences

### Positive

- A team-level match-report KPI layer on the `(game_id, team_id)` grain, event-only so it runs on **every** SPADL provider (public or owner-tier); a public, reproducible reliability artifact.
- Adds a plain `shots` count and a raw `final_third_entries` count the module lacked; the post-recovery family is the full transition-output block and xG-independent (only `high_opportunity_shots` needs an injected xG).
- Fully additive: no VAEP / tracking retrain, no re-materialize. The module is validated on synthetic fixtures with known reliability + honest-NaN / conservation / order-insensitivity gates; the reliability driver's pure stat + shaping kernels are CI-tested. The public-corpus artifact itself is the owner-run commit-2 deliverable (`docs/research/tf52_team_kpi_reliability/`), not produced in CI.

### Negative

- A new sibling package trips ~8 repo-wide gates (feature-glossary coverage, C4 count / description cap, NOTICE linkage, `_PUBLIC_MODULE_FILES`, scale-guard registry, provenance + input-contract enrollment, import allowlist) — several fail only in the full suite, so wiring all of them is load-bearing.
- The tracking-flavoured v2 items (interception-height, compactness-recovery-time) and aerial chaining are **not** in this module — the first two by the event-only scope, the third because SPADL has no aerial type. Left to a future decision.
- `feature_glossary.Unit` gained two rate literals (`passes/min`, `actions/min`) for the per-minute KPIs (`pass_tempo`, `defensive_intensity`) — a small closed-vocab extension, additive (no `GLOSSARY_SCHEMA_VERSION` shape change).

### Neutral

- Glossary feature-column count 402 → 446 (+44); the C4 glossary container description + `architecture.html` re-rendered via Graphviz `dot`.
- `final_third_entries` uses a per-action crossing definition (start outside, end inside the final third); a failed action's SPADL `end` is its death location, so this is a ball-progression count, not an intended-target count — the same honest-limit shape as the sibling metrics.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-15-tf52-team-kpi-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-15-tf52-team-kpi.md`
- **ADRs:** sibling event-only `compute_*` packages — ADR-085 (shot_stopping), ADR-086 (territory + duels), ADR-092 (gk_decision); ADR-009 (`for_provider` empty), ADR-019 (id canonicalisation), ADR-028/045 (orientation/reflection), ADR-042 (conservation census), ADR-048 (feature glossary), ADR-050 (geometry constants), ADR-052 (`for_each` driver), ADR-056 (input contract), ADR-037 (clean-tree provenance).
- **External references:** Twelve.football match-report glossary (Soccermatics module 3); MSC Bootcamp practitioner KPIs (Tigres Femenil / Clemson; Coventry academy); the public Wyscout data set (Pappalardo et al. 2019, *Scientific Data* 6:236).
