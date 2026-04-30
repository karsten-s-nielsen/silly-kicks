# PFF FC events-data converter — silly-kicks 2.6.0

**Status:** Approved (design)
**Target release:** silly-kicks 2.6.0
**Author:** Karsten S. Nielsen with Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.5.0 (`docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`)
**Triggers:**
- First-class PFF FC / Gradient Sports support across the silly-kicks event-data converter family.
- FIFA Men's World Cup 2022 PFF release publicly available after signup; full local copy in hand for development-time empirical baselining.
- Tracking-namespace expansion deferred from this PR but captured as the first formal `TODO.md` entry since 2.5.0, informed by verified lakehouse prior art.

---

## 1. Problem

silly-kicks supports five event-data providers natively (StatsBomb, Opta, Wyscout, Sportec/IDSSE, Metrica) plus the Kloppy gateway. PFF FC / Gradient Sports — distributor of the public FIFA Men's World Cup 2022 event + tracking dataset — is a notable gap. PFF's event model is hierarchical (`gameEvents` envelope + `possessionEvents` payload + `fouls[]` array per row, ~140 leaf fields, ~80% null per row because each row represents one event class) and does not fit any existing converter shape.

PFF FC has no kloppy parser as of kloppy 3.x. Adopting PFF therefore requires a native silly-kicks converter, not a gateway.

This PR adds `silly_kicks.spadl.pff` as a first-class converter on par with `sportec.py` and `metrica.py`. Tracking-data support — which PFF also distributes — is **out of scope** for this PR by deliberate choice (Section 6); it would introduce two simultaneously-novel patterns (in-package I/O policy revision + a new `silly_kicks.tracking` namespace) and is better designed in a dedicated scoping cycle alongside ≥3 other tracking providers.

## 2. Goals

1. **Native PFF events → SPADL converter** at `silly_kicks.spadl.pff`, on par with sportec/metrica/wyscout in coverage discipline, output-contract enforcement, and ConversionReport audit trail.
2. **Library-grade abstraction.** No lakehouse-shaped assumptions. The PFF converter must be useful to many downstream consumers (notebooks, alternative warehouses, research pipelines) — not solely the maintainer's luxury-lakehouse.
3. **Synthetic-only test fixtures** committed to the repo. Real WC 2022 data is not committed and not depended on for CI. PFF licensing for redistributable real-data slices remains pending; if confirmed later, real-data fixtures would be additive but never load-bearing.
4. **Atomic-SPADL and VAEP composability** for free — no code changes in `silly_kicks/atomic/` or `silly_kicks/vaep/` (asserted by tests, not enforced by code).
5. **TODO.md entry** capturing tracking-namespace deferred design with verified lakehouse prior art and library-native architectural rules.

## 3. Non-goals

- **No raw PFF JSON loader inside the package.** The hexagonal pure-function contract that the rest of silly-kicks observes (events DataFrame in, SPADL DataFrame + ConversionReport out, zero I/O) applies to PFF without exception. Callers parse the raw JSON themselves; the converter docstring includes a worked-example reference helper.
- **No tracking namespace, no `silly_kicks.tracking.*` modules, no `silly_kicks.providers.pff` package.** All deferred to a separately-scoped release.
- **No new ADRs.** Decisions in this PR follow established ADR-001 (identifier conventions are sacred) and ADR-003 (NaN-safe enrichment, not load-bearing on the converter itself). The deferred ADR-004 for the tracking namespace charter is referenced from `TODO.md` and will be authored at the time tracking is scoped.
- **No `PFF_DATA_DIR` env-var-gated tests.** Repo policy is that every test must work on a fresh clone with no external setup; local-data tests would violate this for any contributor.

## 4. Design

### 4.1 Architecture

Single-file native converter, mirroring `sportec.py`:

- **New module:** `silly_kicks/spadl/pff.py`. Pure function. Hexagonal. Zero I/O.
- **New schema constant:** `PFF_SPADL_COLUMNS` in `silly_kicks/spadl/schema.py`.
- **Updated module exports:** `silly_kicks/spadl/__init__.py` adds `from . import pff` and `"pff"` in `__all__`.
- **No new dependencies.** Pandas + numpy only. No new optional dependencies.
- **No changes to `silly_kicks/atomic/` or `silly_kicks/vaep/`.** These accept any DataFrame matching `SPADL_COLUMNS`; `PFF_SPADL_COLUMNS` is a strict superset.

### 4.2 Public API

```python
from silly_kicks.spadl import pff

actions, report = pff.convert_to_actions(
    events,                                    # PFF-shaped events DataFrame (Section 4.3)
    home_team_id=366,                          # int, from PFF metadata homeTeam.id
    home_team_start_left=True,                 # bool, from PFF metadata homeTeamStartLeft
    home_team_start_left_extratime=None,       # bool | None, from homeTeamStartLeftExtraTime
    preserve_native=None,                      # list[str] | None, optional input column passthrough
)
```

The two new direction-of-play parameters (`home_team_start_left`, `home_team_start_left_extratime`) are a deliberate departure from the sportec signature, justified by PFF's perspective-real coordinate convention (Section 4.7). Both flags come straight from a single PFF metadata read; the caller computes them once per match.

`preserve_native` follows the existing convention: any column listed is passed through into the output unchanged. Validated against `EXPECTED_INPUT_COLUMNS` to prevent silent misspellings.

### 4.3 Input contract — `EXPECTED_INPUT_COLUMNS`

Caller produces a flat DataFrame with one row per PFF event (one row per element of the top-level event JSON list), columns named in silly-kicks snake_case. The contract is a `frozenset` constant at the top of `pff.py`:

```python
EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset({
    # --- Identification & timing ---
    "game_id",                    # int64; PFF gameId
    "event_id",                   # int64; PFF gameEventId (unique per game)
    "possession_event_id",        # Int64 nullable; PFF possessionEventId
    "period_id",                  # int64; gameEvents.period (1, 2, 3=ET1, 4=ET2, 5=PSO)
    "time_seconds",               # float64; PFF startTime (period-relative seconds)
    "team_id",                    # int64; gameEvents.teamId
    "player_id",                  # Int64 nullable (null on END/SUB without on-ball player)
    # --- Event-class dispatch keys ---
    "game_event_type",            # str; OTB / OUT / SUB / FIRSTKICKOFF / SECONDKICKOFF / END
    "possession_event_type",      # str | None; PA / SH / CR / CL / BC / CH / RE / TC / IT / null
    "set_piece_type",             # str | None; K / F / C / T / G / P / O / null
    # --- Ball position (PFF coordinate convention) ---
    "ball_x",                     # float64; PFF centered meters, range ~[-52.5, 52.5]
    "ball_y",                     # float64; PFF centered meters, range ~[-34.0, 34.0]
    # --- Body part / pass / cross qualifiers ---
    "body_type",                  # str | None; L / R / H / O / null
    "ball_height_type",           # str | None
    "pass_outcome_type",          # str | None; C (complete) / F (fail) / etc.
    "pass_type",                  # str | None
    "incompletion_reason_type",   # str | None
    "cross_outcome_type", "cross_type", "cross_zone_type",
    # --- Shot qualifiers ---
    "shot_outcome_type",          # str | None; G / S / B / W / M / O
    "shot_type", "shot_nature_type", "shot_initial_height_type",
    "save_height_type", "save_rebound_type",
    # --- Carry / dribble qualifiers ---
    "carry_type", "ball_carry_outcome", "carry_intent",
    "carry_defender_player_id",   # Int64 nullable
    # --- Challenge / tackle qualifiers ---
    # PFF source JSON carries challenge actor IDs as player-only; team
    # affiliation must be supplied by the caller via a roster join when
    # flattening (see § 4.5). Pushing this upstream keeps the converter
    # purely event-row-local and parallel with sportec's tackle-team contract.
    "challenge_type", "challenge_outcome_type",
    "challenger_player_id", "challenger_team_id",                # Int64 nullable
    "challenge_winner_player_id", "challenge_winner_team_id",    # Int64 nullable
    "tackle_attempt_type",
    # --- Clearance / rebound / GK / touch qualifiers ---
    "clearance_outcome_type", "rebound_outcome_type", "keeper_touch_type",
    "touch_outcome_type", "touch_type",
    # --- Foul (one PFF event row has at most one fouls[0] entry; flatten) ---
    "foul_type", "on_field_offense_type", "final_offense_type",
    "on_field_foul_outcome_type", "final_foul_outcome_type",
})
```

~40 columns. Caller produces the DataFrame via `pd.json_normalize(events_json, max_level=2)` followed by a one-pass column rename (camelCase → snake_case + `gameEvents.` / `possessionEvents.` prefix strip), plus a roster join to populate `challenger_team_id` / `challenge_winner_team_id` (PFF carries these only as player IDs). The converter docstring includes the exact ~15-line reference helper for callers who don't want to write their own.

**Validation discipline:** on entry, `_validate_input_columns(events, EXPECTED_INPUT_COLUMNS)` raises `ValueError` naming the missing columns. Same pattern as every other converter.

The flat-vocabulary input shape (Approach 2 in the brainstorming session, mirroring sportec) was chosen over an `extra: dict` approach (Approach 1, mirroring statsbomb) because:

1. It is the most-recent silly-kicks convention; the next maintainer sees a clean modern reference.
2. PFF qualifiers are materially richer than statsbomb's (~30 relevant qualifiers per `possessionEvents` vs ~5–10 in statsbomb's `extra`). Hiding 30 fields behind `.str.get()` chains hurts readability and `np.select` dispatch performance.
3. A flat, fully-typed input contract lets every consumer (notebooks, lakehouse, research pipelines) validate against a single column list rather than learning a nested-key vocabulary.
4. Tiny synthetic test fixtures read naturally as flat columns; nested-dict fixtures need pre-construction code.

### 4.4 Event mapping

Dispatch is `np.select` over `(game_event_type, possession_event_type, set_piece_type)`:

| `game_event_type` | `possession_event_type` | `set_piece_type` | SPADL action |
|---|---|---|---|
| `OTB` | `PA` | `O` (open play) / `K` (kickoff) | `pass` |
| `OTB` | `PA` | `F` | `freekick_short` |
| `OTB` | `PA` | `C` | `corner_short` |
| `OTB` | `PA` | `T` | `throw_in` |
| `OTB` | `PA` | `G` | `goalkick` |
| `OTB` | `CR` | `O` | `cross` |
| `OTB` | `CR` | `F` | `freekick_crossed` |
| `OTB` | `CR` | `C` | `corner_crossed` |
| `OTB` | `SH` | `O` / `K` | `shot` |
| `OTB` | `SH` | `F` | `shot_freekick` |
| `OTB` | `SH` | `P` | `shot_penalty` |
| `OTB` | `CL` | (any) | `clearance` |
| `OTB` | `BC` | (any) | `dribble` |
| `OTB` | `CH` | (any) | `tackle` (with `tackle_winner_*` / `tackle_loser_*` passthrough — Section 4.5) |
| `OTB` | `RE` | (any) | `keeper_save` (default) / `keeper_pick_up` (when `keeper_touch_type` is catch-class) |
| `OTB` | `TC` | (any) | `bad_touch` |
| `OTB` | `IT` | (any) | _excluded_ — ball receipt (analog of StatsBomb "Ball Receipt*") |
| `OUT` | (any) | (any) | _excluded_ — non-action; SPADL has no out-of-play type |
| `SUB` / `END` / `FIRSTKICKOFF` / `SECONDKICKOFF` | (any) | (any) | _excluded_ — period boundaries / lineup changes |

**Fouls.** PFF surfaces fouls as the `fouls[]` array on the same event row. The converter synthesizes one extra SPADL `foul` row when `foul_type` is non-null. Cards are encoded via the SPADL `result` column following the existing convention (no separate "card" action_type), driven by `final_foul_outcome_type`. Same model statsbomb, sportec, metrica use today.

**Dribbles.** PFF surfaces ball-carries (`BC`) explicitly with start position, defender, intent, and outcome. The converter maps `BC → dribble` directly. The synthetic-dribble pass `_add_dribbles` from `silly_kicks/spadl/base.py` is **skipped** for PFF — using both would double-count. End coordinates of dribble (and pass / cross / clearance) actions come from a generalized version of the existing `_fix_clearances` post-processing: `end_x/y = next_action.start_x/y` within the same possession.

### 4.5 Identifier conventions (ADR-001 alignment) + `PFF_SPADL_COLUMNS`

PFF challenges (`CH`) carry both a `challenger_player_id` and a `challenge_winner_player_id` in source JSON. Team affiliation for these actor IDs is **not** present on the event row — PFF only stores it on the per-match roster. The caller is responsible for the roster join when flattening, surfacing `challenger_team_id` and `challenge_winner_team_id` as input columns. The converter consumes them directly. This keeps the converter purely event-row-local (no roster-aware machinery inside).

Per ADR-001 (silly-kicks 2.0.0, identifier conventions are sacred), `team_id` / `player_id` in the SPADL output mirror the caller's input verbatim — specifically, the `gameEvents.playerId` (the on-the-ball actor in PFF's data model). Tackle winner/loser surface as **dedicated passthrough columns**, exactly the sportec model.

New schema constant in `silly_kicks/spadl/schema.py`:

```python
PFF_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "tackle_winner_player_id": "Int64",   # pandas nullable
    "tackle_winner_team_id":   "Int64",
    "tackle_loser_player_id":  "Int64",
    "tackle_loser_team_id":    "Int64",
}
"""PFF SPADL output schema: SPADL_COLUMNS + 4 nullable Int64 tackle-actor
passthrough columns. NaN on rows where no challenge winner/loser is
identifiable (i.e., everywhere except CH events). See ADR-001 for the
identifier-conventions rationale shared with SPORTEC_SPADL_COLUMNS."""
```

**Input → output mapping for the four passthrough columns:**

```
tackle_winner_player_id = challenge_winner_player_id        # direct PFF qualifier
tackle_winner_team_id   = challenge_winner_team_id          # caller-supplied via roster join
tackle_loser_player_id  = challenger_player_id  if challenge_winner_player_id != challenger_player_id
                          else event row's player_id        # i.e., the on-ball actor lost
tackle_loser_team_id    = challenger_team_id    if challenge_winner_player_id != challenger_player_id
                          else event row's team_id          # i.e., the on-ball team lost
```

The "loser" derivation handles both directions of CH outcome: if the challenger won (took the ball), the on-ball actor is the loser; if the challenger lost, the challenger is the loser themselves. Self-tackles are not a soccer concept, so the case `challenger_player_id == event row's player_id` is treated as data-quality NaN (logged to `unrecognized_counts`).

The dtype choice is `Int64` (pandas nullable) rather than `object` (sportec's choice) because PFF native player/team identifiers are integers, whereas kloppy hands sportec strings. This is a deliberate, documented departure. Long-term unification of the two extended schemas (sportec/PFF) under a common name is a follow-up TODO, **not** in scope of this PR.

`_finalize_output()` in `silly_kicks/spadl/utils.py` is generalized to recognize the `Int64` extension dtype on schema entries — a small surface-area change that is fully backwards-compatible with existing object/int64 dtype handling.

### 4.6 Coordinate normalization

PFF coordinates are pitch-centered meters: origin at center spot, x ∈ ~[-52.5, 52.5], y ∈ ~[-34.0, 34.0]. SPADL is bottom-left-origin meters: origin at corner, x ∈ [0, 105], y ∈ [0, 68]. The converter translates inside `_translate_coordinates`:

```python
actions["start_x"] = events["ball_x"] + 52.5
actions["start_y"] = events["ball_y"] + 34.0
```

PFF events do not carry an explicit end position; the ball moved-to position is implicit in the next event's start. End coordinates are filled by a generalized version of the existing `_fix_clearances` post-processing — applied to passes, crosses, dribbles, and clearances — so `end_x/y = next_action.start_x/y` within the same possession. One shared helper, not per-action-type.

### 4.7 Direction-of-play normalization

PFF coordinates reflect actual on-field direction, which **switches between periods**. Other silly-kicks providers don't need per-period flipping because their input is already perspective-fixed (StatsBomb / Wyscout / Opta record from a fixed perspective; sportec / kloppy hand silly-kicks pre-normalized data via the existing `play_left_to_right` machinery). PFF is the first provider where the converter must apply the per-period flip itself.

This is why the function signature has the two new parameters `home_team_start_left` and `home_team_start_left_extratime`. Both come straight from the PFF metadata JSON (`homeTeamStartLeft`, `homeTeamStartLeftExtraTime`).

Per-period rule: a row needs flipping if its team is attacking left in that period. Implementation:

```python
home_attacks_right_per_period = {
    1: home_team_start_left,
    2: not home_team_start_left,
    3: bool(home_team_start_left_extratime),
    4: not bool(home_team_start_left_extratime),
    5: True,                             # PSO — single-end; flip is moot for SH+P
}
team_attacks_right = (
    actions["team_id"].eq(home_team_id)
    == actions["period_id"].map(home_attacks_right_per_period)
)
flip_idx = ~team_attacks_right
actions.loc[flip_idx, ["start_x", "end_x"]] = 105.0 - actions.loc[flip_idx, ["start_x", "end_x"]].values
actions.loc[flip_idx, ["start_y", "end_y"]] = 68.0  - actions.loc[flip_idx, ["start_y", "end_y"]].values
```

After this, every action satisfies the standard SPADL invariant: the actor's team attacks left-to-right.

**ET fallback policy.** If a match has period_id ∈ {3, 4} rows but `home_team_start_left_extratime` is `None`, the converter raises `ValueError` rather than silently mis-orienting. Documented in the docstring. Caller is forced to be explicit when ET data is present.

### 4.8 Output guarantees

- **Schema:** `PFF_SPADL_COLUMNS`. Enforced by `_finalize_output()` (generalized for `Int64`).
- **Dtypes:** strict per-column. New `Int64` extension dtype on the four tackle-passthrough columns; SPADL_COLUMNS columns unchanged.
- **Empty-input contract:** `convert_to_actions(empty_df, ...)` returns `(empty_df_with_PFF_SPADL_COLUMNS_schema, ConversionReport(provider="PFF", total_events=0, total_actions=0, mapped_counts={}, excluded_counts={}, unrecognized_counts={}))`.
- **Input non-mutation:** original `events` DataFrame is not touched. Asserted in tests.
- **`action_id`:** dense 0-indexed monotonic increasing, applied post-foul/dribble synthesis.

### 4.9 ConversionReport

```python
ConversionReport(
    provider="PFF",
    total_events=len(events),
    total_actions=len(actions),
    mapped_counts={"pass": ..., "cross": ..., "shot": ..., "clearance": ...,
                   "tackle": ..., "dribble": ..., "foul": ...,
                   "freekick_short": ..., "throw_in": ..., "goalkick": ...,
                   "corner_short": ..., "corner_crossed": ..., "freekick_crossed": ...,
                   "shot_freekick": ..., "shot_penalty": ...,
                   "keeper_save": ..., "keeper_pick_up": ..., "bad_touch": ...},
    excluded_counts={"OUT": ..., "SUB": ..., "FIRSTKICKOFF": ..., "SECONDKICKOFF": ...,
                     "END": ..., "OTB+IT": ...},
    unrecognized_counts={...},
)
```

- Keys in `mapped_counts` are SPADL action-type names (consistent with the canonical contract; `coverage_metrics()` keys on the same names).
- Keys in `excluded_counts` are PFF-side identifiers — bare `gameEventType` for whole-event exclusions (`OUT`, `SUB`, period boundaries) and `"OTB+IT"`-style strings for `(gameEventType, possessionEventType)` pair exclusions.
- Keys in `unrecognized_counts` are stringified `(game_event_type, possession_event_type)` pairs — surfaces future PFF vocabulary additions meaningfully (e.g., `"OTB+XX"` for some new possession event type appearing in a non-WC2022 PFF release).

### 4.10 Atomic-SPADL and VAEP composability

No code changes in `silly_kicks/atomic/` or `silly_kicks/vaep/`. They consume any DataFrame conforming to `SPADL_COLUMNS`, and `PFF_SPADL_COLUMNS` is a strict superset. Composability is asserted via tests:

```python
def test_pff_atomic_composability():
    actions, _ = pff.convert_to_actions(events, home_team_id=..., home_team_start_left=...)
    atomic_actions = atomic.spadl.convert_to_atomic_spadl(actions)
    assert len(atomic_actions) > 0

def test_pff_vaep_composability():
    actions, _ = pff.convert_to_actions(events, home_team_id=..., home_team_start_left=...)
    features = vaep.features.gamestates(...)
    labels = vaep.labels.scores(actions)
    assert features.notna().any().all()
```

These run against the synthetic match (Section 5.3) in `tests/spadl/test_pff.py`. The standard SPADL and atomic-SPADL `add_*` enrichment helpers (`add_gk_role`, `add_possessions`, `add_pre_shot_gk_context`, `add_gk_distribution_metrics` — each shipped in both `silly_kicks.spadl.utils` and `silly_kicks.atomic.spadl.utils`) automatically apply to PFF output because they're already decorated with `@nan_safe_enrichment` (per ADR-003) and operate on SPADL-shape DataFrames. The extra `tackle_winner_*` / `tackle_loser_*` columns pass through as inert.

### 4.11 Result, bodypart, and set-piece detail mappings

**Pass / cross result:**

| Source value | SPADL result |
|---|---|
| `pass_outcome_type = "C"` (complete) | `success` |
| `pass_outcome_type = "F"` (fail) | `fail` |
| `pass_outcome_type` other / null | `fail` (defensive default) |
| Same vocabulary applies to `cross_outcome_type`. | |

**Shot result:**

| `shot_outcome_type` | SPADL result |
|---|---|
| `G` (goal) | `success` |
| `O` (own goal) | `owngoal` |
| `S` (saved) / `B` (blocked) / `W` (wide) / `M` (missed) / null / other | `fail` |

**Foul result (cards):**

| `final_foul_outcome_type` | SPADL result |
|---|---|
| yellow-card variants (Y / 2Y / etc.) | `yellow_card` |
| red-card variants (R / SR / etc.) | `red_card` |
| plain foul / null | `success` |

(Exact PFF foul-outcome vocabulary is finalized at implementation time against the synthetic-match generator. Variant codes documented in the converter docstring.)

**Body-part:**

| `body_type` | SPADL bodypart |
|---|---|
| `L` (left foot) | `foot_left` |
| `R` (right foot) | `foot_right` |
| `H` (head) | `head` |
| `O` (other) | `other` |
| null | `foot` (default — same as the other converters) |

If silly-kicks SPADL bodypart vocabulary lacks `foot_left` / `foot_right` distinction in the released config, both fall back to `foot`. Verified at implementation time against `silly_kicks.spadl.config.bodyparts_df()`.

### 4.12 Rebound disambiguation

PFF `RE` events represent goalkeeper rebounds (post-shot ball control). `keeper_touch_type` disambiguates between:

- Catch-class values → `keeper_pick_up`.
- Punch / parry / deflect-class values → `keeper_save`.
- Null / unrecognized → `keeper_save` (defensive default; logged to `unrecognized_counts` if a previously-unseen value appears, rather than silently falling through).

The full set of `keeper_touch_type` codes and their catch-vs-deflect classification is enumerated at implementation time against the synthetic-match generator and the converter docstring. Locking specific code letters (e.g., `C` for catch, `T` for trap) here would predate that enumeration; the implementation plan resolves the exact mapping.

## 5. Testing

### 5.1 Synthetic-only policy

**Tests must work on a fresh clone with no external setup.** No `PFF_DATA_DIR` env-var-gated tests. No real WC 2022 data committed. PFF licensing for redistributable real-data slices remains pending; if confirmed later, real-data fixtures would be additive but never load-bearing.

### 5.2 Test files

- `tests/spadl/test_pff.py` — contract tests + dispatch-path tests using inline mini-DataFrames + e2e against `synthetic_match.json` (asserting action counts, dispatch coverage, `ConversionReport` contents, schema/dtype compliance, no input mutation, Atomic-SPADL composability, VAEP label generation runs cleanly).
- `tests/datasets/pff/synthetic_match.json` — committed, ~150–250 events.
- `tests/datasets/pff/_generate_synthetic_match.py` — committed, deterministic-seed generator.
- `tests/datasets/pff/README.md` — documents the synthetic-only policy, license-pending note, and how to regenerate the JSON.
- `tests/spadl/test_cross_provider_parity.py` gains a parametrize entry for PFF so the new converter participates in all the existing cross-pipeline parity checks (boundary metrics, coverage metrics, `_finalize_output` schema compliance, `add_possessions` cross-validation).

### 5.3 Synthetic match fixture

The synthetic match must cover, with ≥2× redundancy per dispatch path:

- Every row of the Section 4.4 dispatch table.
- Every set-piece composition (kickoff / open play / corner / free kick / throw-in / goal kick / penalty).
- Every result_id mapping (success, fail, owngoal, yellow_card, red_card).
- Every body_type mapping (L / R / H / O / null default).
- `OUT` / `SUB` / `FIRSTKICKOFF` / `SECONDKICKOFF` / `END` events for non-trivial `excluded_counts` exercise.
- Challenges with explicit `challenge_winner_player_id ≠ challenger_player_id` so `tackle_winner_*` / `tackle_loser_*` passthrough has real values to assert.
- Ball-carries with both `R` (retained) and `L` (lost) outcomes.
- Cards (yellow + red) for the foul→result mapping.
- Both periods, with realistic time progression — enough to drive Atomic-SPADL post-processing and VAEP label generation through their full code paths.

Realistic size: ~150–250 events, JSON file ~30–80 KB. Generator script at `tests/datasets/pff/_generate_synthetic_match.py` produces the JSON deterministically with a fixed RNG seed. Generator is committed; regeneration is a manual maintainer action, not a pytest step. The test suite asserts the committed JSON's content stability against the generator output (so accidental edits are caught).

### 5.4 Vocabulary-completeness caveat

Synthetic-only e2e validates **conversion correctness against silly-kicks's code**, not **vocabulary completeness against future real PFF data**. There may be qualifier values or event-type combinations in other PFF datasets (Premier League release, future World Cups, club tier) that the WC 2022 sample does not include and that this PR therefore does not cover.

Mitigation already in place: the `ConversionReport.unrecognized_counts` mechanism — same one statsbomb / wyscout / opta have today. Any future user encountering an unmapped vocabulary value gets a loud signal in the report, not a silent drop. This is the established silly-kicks contract for vocabulary drift over time.

### 5.5 docs/examples notebook

`docs/examples/pff_wc2022_walkthrough.ipynb` (or `.py` markdown-style; format chosen at implementation time per repo convention). Reads from a user-supplied PFF directory path (parameterized at the top of the notebook), demonstrates:

1. JSON parsing into the `EXPECTED_INPUT_COLUMNS` DataFrame.
2. `pff.convert_to_actions(...)` with metadata extraction (home_team_id, home_team_start_left).
3. Atomic-SPADL composition.
4. A simple VAEP fit on a few matches.
5. A `coverage_metrics` summary.

This file is **documentation, not test** — runs against any user's PFF data, never imported by pytest. Lives in `docs/examples/` (new directory). A README in `docs/examples/` documents the convention for any future provider walkthroughs.

## 6. Deferred — Tracking namespace (TODO.md entry)

The full `TODO.md` entry below captures the validated tracking-namespace ideas surfaced during brainstorming, anchored to verified lakehouse prior art and library-native architectural rules. It is the design starting point for whoever scopes the tracking work.

---

```markdown
# Open Items

## Tracking namespace — `silly_kicks.tracking.*`

Scope: bring tracking-data support into silly-kicks as a first-class library
surface, parallel to `silly_kicks.spadl.*`. Deferred from the PFF events-only
PR (silly-kicks 2.6.0) per scoping decision recorded in
`docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md` § 6.

### Why tracking adds separate value

Events and tracking are complementary, not redundant. Events carry semantic
intent (cross vs long pass, set-piece type taxonomy, foul/card/VAR). Tracking
unlocks analytical surfaces events fundamentally cannot:

- Off-ball player positions: pressure measurement, defensive compactness,
  pass-option counts, off-ball runs, defender density between shooter and
  goal.
- Player kinematics: velocity, acceleration, sprint counts, high-intensity
  distance, fatigue proxies.
- Pre/post-event context windows (e.g., defender behaviour 3 s after a shot).
- Pitch-control / Voronoi / Spearman models — require every player's
  position at every frame.
- Tracking-aware VAEP: nearest-defender distance at action start,
  ball-carrier speed, receiver-zone density. Known to materially improve
  VAEP / xG / xPass calibration.
- Refined GK / pre-shot context — `add_pre_shot_gk_context` becomes
  genuinely accurate once GK position is known per-frame, not inferred.
- Forward-compatible QA layer: tracking can detect mistagged event
  locations, supporting silent-drift detection.

### Architectural rules (carried forward from PFF events-only design)

1. **Hexagonal, pure-function.**
   `silly_kicks.tracking.<provider>.convert_to_frames(raw_frames_df, roster_df, ...) -> tuple[pd.DataFrame, ConversionReport]`.
   Same shape as the SPADL converters.
2. **No I/O in the package.** Caller parses provider files
   (JSON, JSONL, CSV, XML+positions) into the provider-shaped input
   DataFrame. Caller-responsible, every consumer.
3. **Provider-agnostic at OUTPUT, provider-shaped at INPUT.** The
   `TRACKING_FRAMES_COLUMNS` schema is canonical; per-provider adapters
   translate native vocabularies into it.
4. **Library-native conventions, not lakehouse-specific.** Coordinates:
   silly-kicks SPADL 105 × 68 meters, NOT the lakehouse's StatsBomb 120 × 80
   units (which exists for legacy event-coordinate alignment). Identifier
   types: per-provider conventions matching the SPADL converter for the
   same provider (object for kloppy/sportec, int64/Int64 for PFF and
   StatsBomb-style providers), NOT synthetic
   `provider_match_side_jersey` FQNs.
5. **Long-form frames DataFrame.** One row per (frame, player). Ball is its
   own row (`is_ball=True`, `player_id=null`, `team_id=null`) — NOT a
   wide-form schema with per-player columns.
6. **One ADR at scoping time** (ADR-004 placeholder): "silly-kicks now
   supports tracking. Hexagonal pure-function contract. Canonical long-form
   frames schema. Per-provider adapter modules. No I/O in the package.
   Smoothing / possession windowing / pressure features deferred to future
   scoped releases."

### Canonical schema starter (informed by lakehouse evidence, not copied from it)

The luxury-lakehouse `fct_tracking_frames` mart already runs production
tracking across 3 providers (Metrica, IDSSE/Sportec, SkillCorner) on 20
matches / ~38M player-frame rows (verified 2026-04-30 via Databricks probe
of `soccer_analytics.dev_gold.fct_tracking_frames`). Its 15-column shared
schema is strong prior art for which fields a tracking schema needs.
silly-kicks should adopt the **field set**, not the dtype/unit choices
(see rule 4 above). Starter draft, subject to revision at scoping:

```python
TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id":         "int64 | object",  # provider-dependent, mirrors SPADL
    "period_id":       "int64",
    "frame_id":        "int64",            # frame within period
    "time_seconds":    "float64",          # period-relative, like SPADL
    "frame_rate":      "float64",          # Hz (10 / 25 / 30 fps real-world)
    "player_id":       "int64 | object",   # null for ball rows
    "team_id":         "int64 | object",   # null for ball rows
    "is_ball":         "bool",
    "is_goalkeeper":   "bool",
    "x":               "float64",          # SPADL meters [0, 105]
    "y":               "float64",          # SPADL meters [0, 68]
    "z":               "float64",          # null for non-ball rows that lack z
    "speed":           "float64",          # m/s; null when provider doesn't supply
    "confidence":      "object",           # provider-specific tier
    "visibility":      "object",
    "source_provider": "object",
}
```

### Minimum viable surface (Y-lite)

- `silly_kicks.tracking.schema` — `TRACKING_FRAMES_COLUMNS`,
  `TRACKING_CONSTRAINTS`. Reuses existing `ConversionReport`.
- `silly_kicks.tracking.<provider>` for ≥3 of {pff, sportec, metrica,
  skillcorner} — one canonical adapter per provider,
  `convert_to_frames(...)`.
- `silly_kicks.tracking.utils` — exactly one cross-pipeline utility on day
  one: `link_actions_to_frames(actions, frames, tolerance_seconds=...)`.
  This is the single operation downstream consumers cannot easily reproduce
  themselves and that everything else (pressure features, possession
  windows, tracking-aware VAEP) eventually requires. Plus
  `play_left_to_right` analog for tracking (mirror of SPADL's existing
  helper).
- One ADR (ADR-004).

### Provider candidates (open data validated as of 2026-04-30)

Verified by inspecting `karstenskyt__luxury-lakehouse` and a read-only
Databricks probe of `soccer_analytics.dev_gold.fct_tracking_frames`:

| Provider          | Open data                                            | Format            | Hz  | Lakehouse evidence (2026-04-30) |
|-------------------|------------------------------------------------------|-------------------|-----|----------------------------------|
| Metrica           | 3 sample games, GitHub `metrica-sports/sample-data`  | CSV               | 25  | 3 matches, 9.5M player-frames    |
| IDSSE / Sportec   | Bassek 2025 *Bundesliga Match Data Open Dataset*     | XML+positions     | 25  | 7 matches, 21.9M player-frames   |
| SkillCorner       | ~9 open broadcast-tracking matches on GitHub         | JSON              | 10  | 10 matches, 6.8M player-frames   |
| PFF FC            | WC 2022 (signup-gated)                               | JSONL.bz2         | 30  | not loaded; primary candidate    |
| TRACAB / SecondSpectrum / Hawkeye / InStat | none public                          | —                 | —   | excluded                          |
| ReSpo.Vision      | partnership-only releases reported                   | broadcast-derived | —   | future contact; flag for licensing inquiry once an open subset becomes available |

Schema-stress test goal: lock the canonical schema only after 4 providers
load cleanly into it (multi-format CSV/JSON/JSONL/XML, multi-Hz 10/25/30,
broadcast-derived + optical + native — heterogeneity prevents PFF-shaped
lock-in).

### Decision points to revisit at scoping time

1. **Native adapter per provider vs kloppy gateway + 1–2 natives.** Kloppy
   already parses tracking from SkillCorner, Sportec, Metrica, TRACAB, and
   SecondSpectrum — a `silly_kicks.tracking.kloppy` gateway (mirroring the
   existing `silly_kicks.spadl.kloppy`) would reach all of them with one
   adapter. Trade-off: depth of native adapters vs breadth of kloppy reach.
   PFF lacks a kloppy parser — must be native regardless.
2. **Long-form schema ball-row encoding.** Confirm `is_ball=True` separate
   row beats wide-form alternatives (e.g., per-frame `ball_x` / `ball_y`
   columns broadcast across player rows, as the lakehouse does). Both work;
   pick after writing the linkage utility against both shapes.
3. **Speed / acceleration computation.** Some providers supply `speed`
   natively (PFF, IDSSE); others don't (Metrica, SkillCorner). Decide:
   trust source where present, derive elsewhere via lag/diff (lakehouse
   pattern), or always derive for consistency. The lakehouse uses
   per-row `frame_rate` to support mixed-Hz unions cleanly — same trick
   applies here.
4. **Test-fixture policy.** Same constraint as the events converter: tests
   cannot depend on local data not in the repo. Synthetic tracking
   fixtures (one per supported provider) become load-bearing. Generator
   scripts in `tests/datasets/<provider>/` mirror the events pattern.
5. **Coordinate-system reaffirmation.** The events converter already runs
   in 105×68 meters; the tracking adapter must produce the same. ADR-004
   explicitly forbids re-importing the lakehouse's StatsBomb-units choice.
6. **Direction-of-play normalization** has the same per-period flip issue
   as PFF events; the same `home_team_start_left[_extratime]` parameters
   apply. Reuse the helper carved out for the events converter.

### Deferred (explicitly NOT in the eventual tracking-namespace PR)

- Smoothing primitives (Savitzky-Golay, exponential moving average).
- Possession-window slicing (`slice_around_event`, etc.).
- Pressure / defender-density features.
- Tracking-aware VAEP / xPass features.
- Pitch-control models (Spearman, Voronoi).
- Multi-frame interpolation / gap filling.
- Streaming / chunked reading for large tracking files.
- ReSpo.Vision adapter (pending licensing).

Each of these gets its own scoping cycle once the canonical schema and
linkage utility are stable and a concrete consumer use case is established.

### Validation plan

- All 4 candidate tracking providers (sportec, metrica, skillcorner, pff)
  must load tracking data cleanly into `TRACKING_FRAMES_COLUMNS` before
  the schema is locked. Note: skillcorner has tracking only (no events
  in silly-kicks today); sportec, metrica, and pff have both.
- `link_actions_to_frames` round-trip test: events ↔ frames join must
  produce non-empty results on every provider's synthetic-match pair.
- `play_left_to_right` for tracking must produce the same orientation
  invariants as SPADL `play_left_to_right`.
- Cross-provider parity test: distance-to-ball distributions per
  source_provider must overlap (proves coordinate normalization is
  consistent across providers).
```

---

## 7. Release & versioning

- **silly-kicks 2.6.0.** Additive feature, no breaking changes, no removed surface.
- CHANGELOG entry under `## [2.6.0]` (date filled at release).
- No version bumps to dependencies. No new optional dependencies.
- `pyproject.toml` only changes via the version bump.

## 8. ADRs

**No new ADR for this PR.** Decisions in this design follow ADR-001 (identifier conventions are sacred — alignment via dedicated `tackle_winner_*` / `tackle_loser_*` columns) and ADR-003 (NaN-safe enrichment — confirmed not load-bearing on a converter, only on `add_*` enrichment helpers).

The deferred ADR-004 (tracking-namespace charter) is referenced from the TODO.md entry above and will be authored at the time tracking is scoped.

## 9. CLAUDE.md

No CLAUDE.md updates needed. The new converter follows existing conventions; nothing about its existence requires durable instruction. The line "See TODO.md for tracked work" remains accurate.

## 10. Implementation plan reference

A separate implementation plan will be authored after spec sign-off via the writing-plans skill, capturing the test-driven sequence: schema + scaffolding → input validation → coordinate translation → direction-of-play → dispatch table → set-piece composition → fouls / cards → tackle passthrough → dribble synthesis → ConversionReport → atomic / VAEP composability tests → cross-provider parity → synthetic-match generator → docs/examples notebook.

Per repo policy, no implementation work begins until both the spec and the implementation plan are approved.
