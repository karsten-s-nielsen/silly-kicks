# TF-59 — GK Shot-Stopping (Goals Prevented / GSAA) + Gold-Standard Keeper-Appearance Resolution — Design

- **Status:** Draft for review — **rev 2** (addresses the 2026-09-01 external review). Not committed; no version / PR-S / ADR numbers assigned — those are set at commit-prep after `git fetch && git merge origin/main`, per house rules.
- **Date:** 2026-09-01
- **Feature:** TF-59 (GK shot-stopping metrics), plus a prerequisite reshape of keeper-identity resolution into shared, gold-standard, appearance-interval-aware infrastructure.
- **Delivery:** a **two-PR arc** — **PR1** (shared infra, §5) lands first, then **PR2** (the metric, §6) branches off the updated `main`. Two sequential feature branches, **no worktrees**; each PR is **one coherent, fully-tested commit** (**no micro-commits**). Each lands only after explicit human approval of that specific diff. (Owner decision 2026-09-01: keep the two-PR split — PR1 is reusable infra that stands alone.)
- **Related:** TF-59 (TODO On-Deck); ADR-078 (single keeper resolver — amended here); ADR-055 (defended-goal single-source / `GoalMap`); ADR-043 (`id_compat` promotion precedent); ADR-062 / ADR-054 (SB360 provider + anonymity); ADR-046 (`shot_blocked`); ADR-018 (own goals by result); ADR-017 (period-relative time); ADR-005 (tracking-aware features; academic-attribution/NOTICE discipline); ADR-011 (trained-artifact/port conventions); ADR-001 (converters never override ids).

---

## 1. Summary

silly-kicks has an extensive goalkeeper stack — xT-GK v1/v2, `GkCompletionModel`, `GkRetentionModel`, GKDV, ghost-GK — but **all of it is positioning/distribution**. There is **no shot-stopping metric**. TF-59 fills that gap with the gold-standard, PSxG-based shot-stopping numbers (**Goals Prevented** and **GSAA**), event-only, per-`(goalkeeper, match)`.

Getting shot-stopping *right* requires knowing **which keeper faced each shot at the moment it was taken**, correctly through substitutions, red cards, and emergency keepers. That is a gold-standard concern the library should own, not push onto every caller. The existing resolver (`tracking.resolve_keeper_identities`, ADR-078) is close but (a) lives inside the heavy `tracking/` package, (b) formally requires `frames`, and (c) explicitly forecloses mid-match substitution resolution. This design promotes it to shared infrastructure, gives it an event-only path, and adds **appearance-interval-granular resolution** fed by a new injected port — validated against real substitution data that four supported providers actually carry.

Delivered as a **two-PR arc** — two sequential feature branches off `main`; **no worktrees**; **one coherent commit per PR, no micro-commits**:

- **PR1 (shared infra):** promote the resolver to a public `silly_kicks/keeper_identity.py`, add an event-only path, add the injected **keeper appearance-interval port** + interval-granular per-action resolution, and ship four provider extractors that populate the port from raw feeds. PR1 stands alone as reusable infra.
- **PR2 (the metric):** a new event-only `silly_kicks/shot_stopping/` package computing Goals Prevented / GSAA, consuming PR1's resolved defending-keeper identity + an injected per-shot PSxG.

Both PRs are **additive to trained models — no VAEP/tracking retrain, no re-materialize.** PR1's promotion is an import-path break (no shim, fail-loud), behaviour-preserving for existing callers.

---

## 2. Goals and non-goals

### Goals

1. **Goals Prevented** and **GSAA** per `(GK, match)`, event-only, from an injected per-shot Post-Shot xG.
2. **Gold-standard keeper-on-pitch attribution**: each shot attributed to the exact keeper on the pitch at its timestamp, correct through substitutions / red cards / emergency keepers, resolved by the library (not the caller).
3. **Reusable, shared keeper-identity infrastructure**: one public resolver usable by any package, event-only-capable, not coupled to `tracking/`.
4. **A provider-neutral appearance-interval port** with four working extractors (StatsBomb/SB360, SkillCorner, Gradient Sports, Sportec/DFL-IDSSE), proving the abstraction across independent native encodings.
5. **No retrain**: everything additive; existing outputs byte-identical.

### Non-goals

- **Clean Sheets Earned (CSE).** Descoped from TF-59 v1 by **explicit owner decision in the 2026-09-01 design session** ("we can leave out CSE"). CSE is a newer, more experimental composite (scores clean-sheet-days only; correlates weakly with difficulty; needs a second VAEP/xT value input); Goals Prevented / GSAA is the gold-standard core. The descope is recorded durably **both here and in `TODO.md`**, where the TF-59 backlog row is split into TF-59 v1 (= GP+GSAA) and a new TF-59b (CSE) follow-up task — so an approved backlog deliverable is not silently dropped; it is deferred *with* recorded approval, not cut without a trace. (This resolves the review's blocking finding: the approval existed in-session but was not durable, and `TODO.md:37` still listed CSE as shipped.)
- **Shipping an xG / PSxG model.** silly-kicks ships none; PSxG is injected (port pattern), exactly like `xg_column` in `vaep/labels.py` and `xtgk/_xg_reward.py`.
- **A per-shot save-difficulty model beyond injected PSxG.** PSxG *is* the difficulty model.
- **Metrica appearance support.** Metrica is anonymized with no roster/identity; excluded by measurement (see §3).
- **GK shot-stopping style/technique, xGOT decomposition, rebound-value chains.** Out of scope.

---

## 3. Data availability (measured, not assumed)

All figures below were **measured live during design (2026-09-01)** via the Hugging Face dataset API (the four `peggy44/*` SkillCorner datasets) and the pining API (GS owner tier; IDSSE + public-SkillCorner public tier). They are load-bearing: the port design and validation plan rest on them. Per house rule (input measurement is a scratch run, not a committed driver), the probes are **not** committed — so the §3.2/§3.3 counts are **not reproducible from the committed tree** (they come from external gated/licensed corpora). They are provenance for the design, re-checkable by re-running the same read-only API queries against those corpora.

### 3.1 Substitution / appearance data by provider

| Provider | Native encoding of keeper-on-pitch | Confirmed on real data | Committable (redistributable)? |
|---|---|---|---|
| **StatsBomb / SB360** | Explicit `Substitution` events (period/min/sec + player-out + `substitution.replacement.id`) + `Starting XI` | ✅ 3 committed open-data fixtures + 30-match licensed corpus (pining) | **Yes — open data** |
| **SkillCorner** | Per-player `playing_time.by_period[]` `start_frame`/`end_frame` + `start_time`/`end_time` + `red_card`; `player_role.acronym=="GK"` | ✅ 899 matches across 4 HF datasets | No — CC-BY-NC |
| **Gradient Sports** | `SUB` gameEvent (`playerOffId`/`playerOnId` + `startGameClock`/`period`); roster `positionGroupType` for GK | ✅ 64 WC2022 matches (owner pining) | No — owner-tier |
| **Sportec / DFL (IDSSE)** | `<Substitution PlayerIn/PlayerOut/Team/PlayingPosition>` (`TW` = keeper) + `other_action_player_becomes_goalkeeper` qualifier; timing on the wrapping event | ✅ 7 public matches (pining) | **Yes — CC-BY 4.0** |
| **Metrica** | None (anonymized; `SUBSTITUTION` excluded; no roster) | ✅ confirmed absent | n/a |

Each classic-event-stream converter already **drops** substitution events at conversion — but at its **own** differently-named exclusion constant, not one shared seam: `_EXCLUDED_EVENT_TYPES` (`statsbomb` / `sportec` / `opta` / `kloppy`), `_EXCLUDED_TYPES` (`metrica`), `_EXCLUDED_WS_TYPE_IDS` (`wyscout`), and a local `excluded_ge_types` set (`gradientsports`) — different names and contents each. SkillCorner has **no** such constant — it carries no substitution *events* at all (its intervals live in the match metadata, not the event stream). The common thread is only that the substitution signal is present in the **raw** feed and discarded before SPADL — so the extractors read the raw feed, not SPADL actions.

### 3.2 SkillCorner GK-change corpus (the interval-resolution validation set)

Measured across the four HF SkillCorner datasets (`meta/<id>.json` for 24/25; `matches/<id>.json` for 25/26 — identical schema, two folder layouts the extractor must handle):

| Dataset | Matches | GK-change matches | GK red cards |
|---|---|---|---|
| PremierLeague24-25 | 380 | 10 | 1 |
| PremierLeague25-26 | 270 | 5 | 1 |
| ChampionsLeague25-26 | 150 | 4 | 0 |
| RealMadrid24-25 | 99 | 3 | 0 |
| **Total** | **899** | **22** | **2** |

~2.4% GK-change rate (the real-football rate); zero parse failures; every match has a clean 2-starting-GK structure. Example red-card case: PL25-26 `2031981`, keeper `34082` red-carded and off at `00:05:44`.

### 3.3 Committable-fixture map

- **Real GK-substitution unit test (committed):** StatsBomb open data (only reliable committable source — thousands of redistributable matches).
- **Happy-path extractor fixtures (committed):** StatsBomb open · SkillCorner public (10 A-League) · IDSSE public (7). All carry the real schema. Measured: 0 GK subs among the 10 public SkillCorner and 0 among the 7 IDSSE (expected at ~2.4% over small corpora), so neither public corpus yields a committable GK-*change* fixture — only happy-path.
- **Real GK-change at scale (e2e, not committed — licensing):** SkillCorner peggy44 (22) · GS owner (64).
- **Emergency-keeper edge:** synthetic committed fixture. **Investigated 2026-09-01:** the 2 IDSSE "outfielder-becomes-keeper" signals were **false positives** — `<OtherPlayerAction … ChangeOfCaptain="true" PlayerBecomesGoalkeeper="false"/>` (captaincy-change events, not keeper changes), so no committable real emergency-keeper case exists in the 7 public IDSSE matches. The DFL **encoding is confirmed**: an emergency keeper is `<OtherPlayerAction PlayerBecomesGoalkeeper="true" Player=… Team=…/>` (always `"false"` across these 7), so the Sportec extractor reads it precisely.

**Key property:** interval *resolution* is **provider-agnostic** — it runs on the normalized port, so it is validated once (StatsBomb real GK-sub fixture + synthetic edges). Only the per-provider *extractors* are validated per provider (each on a committable happy-path fixture; GS on owner e2e).

---

## 4. Architecture overview

```
                 raw provider feed (per provider)
                          │  extract_keeper_appearances()   [PR1, per provider]
                          ▼
        KeeperAppearances  (game, team, keeper_id, period, start/end  — period-relative)
                          │
   actions ──────────────┤  add_defending_gk_player_id(actions, keeper_map, *, appearances)  [PR1]
   (SPADL, event-only)    │     precedence: appearance-interval > goal-kick event > roster > NA
                          ▼
        actions + defending_gk_player_id  (per-action defending keeper)
                          │
   injected PSxG ─────────┤  compute_shot_stopping(...)   [PR2]
                          ▼
        per-(GK, match): Goals Prevented / GSAA  (± in-play penalties)
```

Keeper identity is a **shared** concern (PR1). The metric (PR2) owns only the valuation math and consumes a resolved column. This is the "shared/upstream resolver, thin pure metric" split (rationale in §7).

---

## 5. PR1 — Keeper-identity resolver + appearance-interval port

### 5.1 Promotion to `silly_kicks/keeper_identity.py`

Move `tracking/_keeper_identity.py` → **public `silly_kicks/keeper_identity.py`**, mirroring the ADR-043 `id_compat` promotion: **breaking import-path change, no shim, fail-loud** (a stale `from silly_kicks.tracking import resolve_keeper_identities` fails at import rather than silently degrading).

- Public surface (unchanged semantics): `resolve_keeper_identities`, `add_defending_gk_player_id`, `apply_keeper_identities_to_frames`, `KeeperIdentity`, `KeeperIdentityMap`, `KeeperIdentityReport`, `KEEPER_ID_SOURCE_*`.
- The `native`/frame path keeps delegating to TF-13 `defending_gk_from_frames`/`acting_gk_from_frames` (ADR-055 single-source), but via a **lazy, function-local import** of `silly_kicks.tracking._gk_resolve`, so `import silly_kicks.keeper_identity` stays lightweight (no eager pull-in of the ~30 tracking submodules / numba that even `import silly_kicks.tracking._gk_resolve` triggers, since it runs `tracking/__init__`). **This changes the delegation test seam:** `tests/tracking/test_keeper_identity_native.py:65-66` currently monkeypatches the module attribute `silly_kicks.tracking._keeper_identity.defending_gk_from_frames`; a function-local import means that patch no longer intercepts, so the test migrates to patch the **definition site** `silly_kicks.tracking._gk_resolve.defending_gk_from_frames` (patched at call time — the lazy import picks up the mock). Required test migration (listed in §5.6), **not** a byte-identical no-op at the test layer. (Alternative rejected: a module-top import preserves the attribute-patch seam but re-imports all of `tracking`, defeating the lightness goal that motivates the promotion.)
- `tracking/__init__` **no longer re-exports** these symbols (clean break); `tracking/features.py`, `tracking/_run_features.py` and the `scripts/` consumers import from the new home.

Rationale: TF-59 is the first genuine cross-package *library* consumer of the resolver, and it is event-only — it must not depend on `tracking/`. The promotion follows the measured-demand + heavy-import-avoidance pattern ADR-043 set for `id_compat`.

### 5.2 Event-only enumeration path

Today the `roster` path reads `frames` only to enumerate which `(game, period, team)` exist (it touches no positional columns). Add an **event-only entry point** that enumerates those triples from `actions` instead, so an event-only caller supplies no `frames`. The `native` path still requires `frames` (it reads positions). Concretely: `frames` becomes optional on the roster/appearance path; when absent, `(game, period, team)` are derived from `actions[["game_id","period_id","team_id"]]` (canonicalized via `id_compat`).

### 5.3 Keeper appearance-interval port

A normalized, injected table `KeeperAppearances` (a pandas DataFrame with a documented column schema `KEEPER_APPEARANCE_COLUMNS`, matching the "schemas are plain dicts" house style — no pandera):

| Column | Type | Meaning |
|---|---|---|
| `game_id` | id (nullable) | match |
| `team_id` | id (nullable) | the keeper's team |
| `player_id` | id (nullable) | the keeper |
| `period_id` | int | period |
| `start_time_seconds` | float | **period-relative** on-pitch start (ADR-017) |
| `end_time_seconds` | float | period-relative on-pitch end; `+inf`/NaN ⇒ to period end |
| `source` | str | provenance token (`native_intervals` / `sub_events` / `starting_xi` …) |

- **Period-relative** by construction to match SPADL `time_seconds` (ADR-017); each extractor is responsible for converting its provider's native time (frames / clock / minute) into this base using the provider's period map.
- Keeper rows only (the metric needs keepers); an extractor may compute all-player appearances internally but the port is typed for keepers.
- ids via `id_compat`; nullable dtypes.

### 5.4 Interval-granular resolution + precedence

`add_defending_gk_player_id(actions, keeper_map, *, appearances=None) -> DataFrame` (a new keyword-only `appearances`; **byte-identical when omitted**, so every existing caller is unaffected):

- With `appearances`: for each action, derive the **defending team** (the opponent of the acting team within the action's `(game, period)`, exactly as today), then select the defending team's keeper whose `[start_time_seconds, end_time_seconds]` covers the action's `time_seconds`. Emit that `player_id`.
- Precedence when both are available: **appearance-interval (authoritative) > goal-kick event override > roster starter > NA**. Rationale for interval > goal-kick: an appearance interval is a **direct** record of on-pitch time (from lineup/sub data), whereas the goal-kick-event rung is a sparse **proxy** — it fires only when a keeper happens to take a goal kick, and cannot localize a keeper change that occurs between goal kicks; the interval catches every change and dates it exactly. A shot whose time falls in an appearance gap falls back to the coarse `keeper_map`; a genuinely unresolvable keeper is `pd.NA` (never a fabricated id — ADR-027).
- `resolve_keeper_identities` continues to produce the coarse `(game, period, team) → KeeperIdentity` map (roster + goal-kick + native); the appearance table layers the fine-grained keeper-on-pitch at the stamping step. The two are complementary: appearances best, map fallback + cross-check. The cross-check surfaces as a per-action **`defending_gk_source`** provenance column over `{appearance, map_fallback, appearance_map_conflict, unresolved}` (the ADR-054 source-column pattern) — an interval-vs-map disagreement is `appearance_map_conflict`. `defending_gk_source` is emitted **only** on the appearance path, so the omit-`appearances` output stays byte-identical.

This **amends ADR-078**, which foreclosed substitution resolution. New ADR (PR1) records the amendment.

### 5.5 Provider extractors

Signature per provider: `extract_keeper_appearances(<raw inputs>) -> KeeperAppearances`.

- **StatsBomb / SB360:** `Starting XI` → opening keeper per team; `Substitution` events (period/min/sec + out + `replacement.id`; incoming position identifies a GK sub) → interval boundaries; freeze-frame anonymity is irrelevant (appearances come from the `events`/lineups artifacts, not `freeze_frames`).
- **SkillCorner:** `match.json` / `matches/<id>.json` `players[].playing_time.by_period[]` `start_frame`/`end_frame` → period-relative seconds via `match_periods`; `player_role.acronym=="GK"` for starters; a substitute keeper (`acronym=="SUB"`) is identified by interval-boundary continuity cross-checked against the goal-kick-event rung. Handles both folder layouts (`meta/` and `matches/`).
- **Gradient Sports:** `SUB` gameEvents (`playerOffId`/`playerOnId` + `startGameClock`) joined to roster `positionGroupType` for GK identity.
- **Sportec / DFL (IDSSE):** `<Substitution>` (`PlayingPosition=="TW"` = GK) + event timing; **emergency keepers via `<OtherPlayerAction PlayerBecomesGoalkeeper="true"/>`** (schema verified 2026-09-01; the `other_action_player_becomes_goalkeeper` qualifier).

**Design decision — placement (flagged for review; premise corrected in rev 2).** **All four extractors are new** — `find silly_kicks/providers -name appearances.py` → none today. And only two of the four in-scope providers have a `providers/` package at all: `providers/statsbomb/` and `providers/sportec/` (each currently just `parse.py`). **Gradient Sports and SkillCorner have no `providers/` package** — their raw shaping lives in `spadl/gradientsports.py` / `spadl/skillcorner.py` (converters) plus `scripts/_loader_pining.py` (raw fetch/parse). So the "`providers/` = raw→canonical" premise holds cleanly only for statsbomb/sportec. Options:

- **(i) library `providers/` for all four** — add `appearances.py` to `providers/statsbomb/` + `providers/sportec/`, and **create two new packages** `providers/gradientsports/` + `providers/skillcorner/` for their extractors. Consistent long-term home, reusable + unit-tested; but it introduces two new provider packages for providers whose *other* raw handling still lives in `spadl/`/`scripts/` (a partial, not clean, consolidation) and enlarges PR1 (feeds Open Q#3).
- **(ii) `scripts/`-side** — extractors as loader helpers next to the existing per-provider raw parsing; the library owns only the port schema + resolution. Lighter, no new packages, but the extractors are untested as library API and not reusable.

Recommendation leans **(i)** for the long-term gold-standard home, but the corrected premise (two brand-new packages; GS/SkillCorner raw handling not yet in `providers/`) makes this a genuine **owner/reviewer decision**, not a foregone one.

### 5.6 Blast radius / breaking changes

The promotion is a public import-path break. In-repo consumers to migrate (fail-loud, no shim):

- `silly_kicks/tracking/features.py`, `silly_kicks/tracking/_run_features.py`, `silly_kicks/tracking/__init__.py` (re-exports removed).
- `scripts/build_tf19_instrument_responsiveness.py`, `scripts/_sb_roster.py`, `scripts/_sb_battery.py`.
- Tests importing `tracking.resolve_keeper_identities` / `KeeperIdentity*` — **including `tests/tracking/test_keeper_identity_native.py`, whose delegation monkeypatch migrates from the `_keeper_identity` module attribute to the `_gk_resolve` definition site** (per §5.1), plus the roster/contract test files that reference the old `silly_kicks.tracking._keeper_identity` path.
- `docs/PRIVATE_CONSUMERS.md` updated (the module becomes public; the private-path pin is removed).

No behaviour change for existing callers (the new `appearances`/event-only paths are opt-in; `native`/`roster` outputs byte-identical) → **no retrain, no re-materialize.**

### 5.7 PR1 validation

- **Provider-agnostic resolution** (validated once): committed StatsBomb open-data **real GK-substitution** fixture proves per-action attribution flips at the sub minute; committed **synthetic** red-card / emergency-keeper fixtures prove those edges; the omit-`appearances` path is asserted **byte-identical** to current output (both-sides test).
- **Per-provider extractors:** each on a committable happy-path fixture (StatsBomb open, SkillCorner public, IDSSE public) proving raw → intervals; GS extractor via owner `@e2e`.
- **Large-scale real GK-change** `@e2e` against the SkillCorner peggy44 corpus (22 GK-changes / 2 reds) and GS owner corpus — not committed (licensing).
- Import-allowlist test for any new package boundary; `id_compat` used for every id compare; conservation on the resolver report preserved.

---

## 6. PR2 — TF-59 shot-stopping metric

### 6.1 Package and API

New event-only package `silly_kicks/shot_stopping/` (shape mirrors `restdefense/`: `__init__.py` + `_config.py` + `_compute.py` + `_columns.py`; flat `__all__`; no `add_*` action-coupled aggregator). *(Package name `shot_stopping` proposed; `goalkeeping` considered but rejected as too broad for one metric — reviewer may adjust.)*

Primary entry point:

```
compute_shot_stopping(
    actions: pd.DataFrame,
    *,
    psxg_column: str,
    defending_gk_column: str = "defending_gk_player_id",
) -> pd.DataFrame   # one row per (game_id, defending keeper player_id)
```

- Expects `actions` already carrying `defending_gk_column`, stamped by PR1's `add_defending_gk_player_id` (clean separation: PR1 resolves identity, PR2 aggregates). A thin convenience wrapper that bundles resolution may be added but the core stays pure.
- `psxg_column` follows the `xg_column` idiom: missing column → raise with the canonical "silly-kicks ships no xG/PSxG model" message + ADR pointer.

Output columns (`SHOT_STOPPING_COLUMNS`): `game_id`, `player_id` (keeper), `team_id`, `shots_faced`, `goals_conceded`, `psxg_faced`, `goals_prevented`, and the penalty-split companions `shots_faced_excl_penalties`, `goals_conceded_excl_penalties`, `psxg_faced_excl_penalties`, `goals_prevented_excl_penalties`. `goals_prevented ≡ GSAA` (documented; PSxG is the per-shot expected goal, so `ΣPSxG − GA` is goals-saved-above-expectation — no separate baseline).

### 6.2 Metric definitions

- **On-target shot faced by keeper K** = a row where: `type_name ∈ {shot, shot_penalty, shot_freekick}`, the action's team ≠ K's team (K is the defending keeper via `defending_gk_column`), the shot is **not explicitly blocked**, and the injected `psxg_column` is non-null (PSxG exists only for on-target shots ⇒ its presence *is* the on-target gate). **`shot_blocked` null-handling (ADR-046 nullable `boolean`):** exclude iff `shot_blocked` is literally `True`; treat `pd.NA` as *not blocked* (a non-shot / unknown row is not a block). A raw `shot_blocked != True` returns `pd.NA` on NA rows (silently masking them out), so the predicate is written null-safely as `~shot_blocked.eq(True).fillna(False)` — keep the row unless it is known-blocked.
- **Goal conceded** = such a shot with `result_id == success` (a scored on-target shot). Own goals (`result_id == owngoal` on `bad_touch`, ADR-018) are **excluded from both shots-faced and goals-conceded** — they are not the keeper's shot-stopping.
- **Goals Prevented (= GSAA)** `= Σ psxg_faced − goals_conceded`, per `(game, keeper)`.
- **Penalties:** reported **with and without in-play penalties** (`shot_penalty`); the `_excl_penalties` companions drop `shot_penalty` rows. **Shootout (period 5) excluded entirely.**
- **Grain:** per-`(GK, match)`; season/window aggregation is the caller's (sum the additive columns).

### 6.3 Edge cases

- Keeper `defending_gk_column` NaN on a shot → not attributable to any keeper. It is **not** silently dropped or misattributed; because it has no keeper it cannot be a per-keeper column, so it is surfaced as a **match-level attribution-coverage figure** on a returned report / `attrs` (honest coverage, ADR-042 discipline) — a low attribution rate stays visible instead of reading as clean data.
- No shots faced by a keeper in a match → the keeper still appears with zeros iff they have an appearance; otherwise absent (caller-driven).
- Rebounds: each on-target shot has its own PSxG row; summed — standard.

### 6.4 PR2 validation

- Analytic fixtures: a keeper facing a known set of PSxG values and goals → exact GP/GSAA; penalty-split correctness; own-goal excluded; blocked excluded; shootout excluded; period/keeper attribution through a mid-match GK change (using a PR1 appearance table).
- `feature_glossary` coverage for the emitted columns (an explicit `compute_*` leg, like `restdefense.compute_rest_defense` — the name-shape auto-discovery does not see `compute_shot_stopping`).
- Purity / NaN-safety per the `add_*`/compute conventions where applicable.

---

## 7. Design decisions (with rationale)

1. **Keeper identity is shared infrastructure, not TF-59-local** — one resolver (ADR-055/078), promoted to `silly_kicks/keeper_identity.py`; TF-59 consumes a resolved column. (Owner decision, 2026-09-01 session: promote the resolver upstream rather than keep it TF-59-local.)
2. **True interval-granular (sub/red-card) resolution is built this cycle** — not the coarse-only fallback. (Owner decision, 2026-09-01 session.) Amends ADR-078's foreclosure. Justified by measured multi-provider data (§3).
3. **Appearance data is an injected port** — four providers carry it natively in four encodings; the port normalizes them; the library owns extractors (placement decision in §5.5).
4. **CSE descoped from v1** — GP/GSAA is the gold-standard shot-stopping core; CSE is experimental with a second-model dependency. Owner decision (2026-09-01 session); recorded durably in §2 and in `TODO.md` (CSE → its own TF-59b follow-up task).
5. **PSxG injected** — port pattern; silly-kicks ships no xG.
6. **PSxG-presence is the on-target gate; own goals / blocked / shootout excluded; penalties reported both ways** — standard gold-standard shot-stopping conventions.
7. **No retrain** — everything additive; existing outputs byte-identical; promotion is import-path-only.

---

## 8. Testing strategy

- Mirror `ci.yml` scope: `python -m pytest tests/ -m "not e2e"`, `ruff check/format` on `silly_kicks/ tests/ scripts/`, `pyright` bare. New `@e2e` tests skip without data / token.
- Both-sides discipline (CLAUDE.md): the omit-`appearances` byte-identity test asserts the *unchanged* side; the interval-resolution test asserts a mutation that *should* move the attribution across the sub minute actually does.
- Import-allowlist test for the new metric package (public-seam deps only; nothing imports it; `tracking` never imports it).
- No committed licensed data; e2e-only for peggy44 SkillCorner + GS owner corpora.

---

## 9. Bookkeeping

- **ADRs (two, numbers assigned at commit-prep):** PR1 — keeper-identity resolver promotion + event-only path + appearance-interval port + interval resolution (amends ADR-078); PR2 — TF-59 shot-stopping metric.
- **NOTICE (academic-attribution discipline, ADR-005):** entries for Goals-Prevented / GSAA (standard PSxG−GA) under `shot_stopping`; docstrings cross-link `See NOTICE …`. (ADR-005 is primarily the tracking-aware-features contract; CLAUDE.md attributes the academic-attribution/NOTICE discipline to it.)
- **feature_glossary:** explicit `compute_shot_stopping` leg + `FeatureColumn` records for every emitted column.
- **C4:** PR2 adds one event-only container (`shot_stopping`); PR1 adds the promoted `keeper_identity` module + provider appearance-extractor modules. Re-render `architecture.html` via Graphviz `dot` (never Smetana), per house rule.
- **CHANGELOG / TODO / version:** one entry per PR at commit-prep; single-source the version in `silly_kicks/_version.py` (ADR-079).

---

## 10. Rejected alternatives

- **Per-shot `defending_keeper_id` injected by the caller (no library resolution).** Rejected: pushes sub/red-card correctness onto every caller, unvalidatable; gold standard means the library owns it.
- **Team-grain shot-stopping as the headline.** Rejected: a degradation; per-keeper is the metric (team-grain only as an explicit fallback if no identity is resolvable).
- **Frame-presence derivation of intervals for SkillCorner.** Rejected: SkillCorner is broadcast tracking (measured ~19.6% GK detection) — keeper absence ≠ subbed off. Intervals come from the metadata, not frame presence. (Full-optical providers could derive from presence, but the metadata path is uniform and reliable.)
- **Keeping the resolver in `tracking/`.** Rejected: an event-only metric must not drag in `tracking/`'s heavy imports; the resolver is a cross-cutting concern (id_compat precedent).
- **Metrica support.** Rejected by measurement: anonymized, no roster/identity.

---

## 11. Decisions (resolved during the 2026-09-01 review cycle)

1. **Extractor placement (§5.5) — RESOLVED (owner, 2026-09-01):** library `providers/` (option i) — add `appearances.py` to `providers/statsbomb` + `providers/sportec`, and create new `providers/gradientsports/` + `providers/skillcorner/` packages.
2. **Package name (§6.1) — RESOLVED (owner, 2026-09-01):** `shot_stopping` (the missing GK pillar; distinct from the distribution/deterrence/positioning modules).
3. **PR1 size / structure — RESOLVED (owner, 2026-09-01):** two-PR arc (PR1 infra + PR2 metric), all four extractors in PR1; no worktrees; one coherent commit per PR, no micro-commits.
4. **Emergency-keeper fixture — RESOLVED (2026-09-01):** the 2 IDSSE signals were false positives (captaincy events, `PlayerBecomesGoalkeeper="false"`); no committable real case in the public IDSSE corpus → **synthetic** committed fixture. DFL emergency-keeper encoding confirmed for the Sportec extractor (`OtherPlayerAction/@PlayerBecomesGoalkeeper`).

---

## 12. Delivery / commit discipline

- **Two-PR arc**, two sequential **feature branches** off `main` — **no worktrees**. Each PR is **one coherent, fully-tested commit** — **no micro-commits**. The uncommitted spec + the TODO.md CSE split ride **PR1**.
- **Each commit lands only after explicit human approval of that specific diff.** No version / PR-S / ADR numbers until commit-prep (after `git fetch && git merge origin/main`, re-deriving NEXT-FREE, per PR).
- No `git commit`/`git push` without explicit per-commit approval.
