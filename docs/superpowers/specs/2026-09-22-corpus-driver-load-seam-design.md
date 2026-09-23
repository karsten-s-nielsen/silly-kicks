# Corpus-driver load seam — resume-before-load, recorded load failures, excluded outcome, events-only loads, sharded xT fits, one cache convention

**Status:** APPROVED round 6 + plan-review ratifications 2026-09-23 (§7/§3/§9; see the ratifications block below) · **Version / PR / ADR:** next-free, resolved at commit-prep against `main` — nothing here carries a pre-claimed number · **Branch:** `feat/corpus-driver-load-seam` off `main` @ `4ac26d0` (4.123.0) · **Amends:** ADR-052 (corpus-driver resilience), in place.

**Round-2 revision** addresses review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec.md` (CDLS-SPEC-01 … 14; mapping in §10) and one owner-directed change: D-1 is now resolved by the repo's established per-artifact convention, confirmed by a pre-registered measurement (§8). The owner reviewed that change before directing it to be folded into this round. The reviewer's round-1 "do not change D-1-as-owner-decision" is superseded by that instruction.

**Round-3 revision** addresses re-review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec-r2.md` (CDLS-SPEC-15 … 21; mapping in §10). Rule C now targets loops that *load*, with flow-sensitive name resolution and function-granular exemptions. The D-1 confirmation rule is redesigned with a stated false-alarm rate. Its outcome is now data consumed by an always-built mechanism, not a code branch.

**Round-4 revision** addresses re-review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec-r3.md` (CDLS-SPEC-22 … 26; mapping in §10).
- The verdict artifact records every measured match with a status.
- Unmeasured SkillCorner matches refuse an events-only pass by name unless `--allow-unmeasured`, which records them.
- Task 0 runs the event check on tracking-unloadable matches too.
- CDLS-SPEC-23 is answered with evidence rather than applied: the quoted rule is in CLAUDE.md verbatim, and the citation now gives its line.

**Round-5 revision** addresses re-review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec-r4.md` (CDLS-SPEC-27, 28; mapping in §10).
- Admission moves out of `load_match` into an edge policy layer that the artifact's producer never touches, so Task 0 cannot deadlock against its own output.
- Task 0's reduce refuses on outstanding failures unless `--allow-failed`, which records them.

**Round-6 revision** addresses re-review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec-r5.md` (CDLS-SPEC-29 … 31; mapping in §10). The loader wrappers drop the ported `events_only` keyword, §4.4 is aligned with the admitted loader, and Rule D becomes function-granular.

**Plan-review ratifications (2026-09-23)** — three deviations the plan surfaced, each verified sound by the plan review (`D:\Development\_reviews\2026-09-23-corpus-driver-load-seam-plan.md`) and ratified by the owner, folded into the sections below: (a) §7 gains a THIRD generation-move class (a SkillCorner-including events-only pass carries the admission digest in its token); (b) §3 reclassifies `measure_gs_shot_distribution` S → E (it discards frames, so it loads events-only); (c) §9 is two commits (code, then the measurement artifact), not one.

## 1. Executive summary

Unblocking the TF-56 Commit-2 battery exposed four defects in how `scripts/` corpus drivers load matches. None is TF-56-specific: each lives in the shared seam (`scripts/_driver.for_each` plus the loader modules) or is copied across drivers. An AST audit over `scripts/*.py` (§3) finds them in most corpus drivers:

1. **Resume re-pays the load.** 34 drivers hand `for_each` a stream from `load_matches` / `load_statsbomb_matches` / `load_open_data_matches` (count re-derived by the round-1 reviewer). The loader downloads and parses a match *before* yielding it, and `for_each` checks for an existing shard only *after*. So a resumed pass re-downloads and re-parses every finished match (~44 s per SkillCorner match, measured on the DGX 2026-09-21).
2. **A match that never loads kills the pass.** A load error raises inside the generator, outside `for_each`'s per-item `try`. The pass crashes before writing its manifest, and every resume crashes on the same match. Under a completeness check, the only way out would be to drop the match from the corpus.
3. **Event-only consumers parse tracking.** At least five loops fit xT (or slice actions) through a full load and throw the frames away — the TF-56 fit's OOM and ~12 h parse, repeated elsewhere. Two drivers go further and hold every match's frames in a list at once.
4. **The download cache is optional and inconsistent.** Some drivers thread `--cache-dir`, many don't. Those re-download every artifact on every pass.

The audit also found three gate holes that let this spread:
- **Hand-listed loaders.** The ADR-052 gate names corpus loaders by hand and misses `load_open_data_matches`. So five corpus drivers are outside its population (executed: `_population()` returns 38 drivers; `train_match_outcome_dependence`, `evolve_xsuccess_features`, `train_xsuccess`, `train_win_probability` and `build_sb360_coverage` are absent). Four of those use `for_each`, but nothing would catch a regression. The fifth, `train_match_outcome_dependence`, walks 3,961 matches with no shards at all.
- **Second loops are invisible.** The gate cannot see an un-sharded loop in a driver that also calls `for_each`. `validate_gk_decision`'s reconstruction legs walk ~935 tracking matches that way.
- **Private drivers are skipped.** It skips private-named drivers such as `_xtgk_comparability`.

**Design (§4).**
- The loader splits into a cheap, sized list of match references plus `load_match(ref, …)`.
- `for_each` gains a `load=` hook. It checks for a finished result *before* loading, and a load error becomes a recorded, named failure.
- An `excluded` outcome (the SkillCorner geometry gate) becomes a first-class, persisted, conserved result, instead of an empty shard or a stderr line.
- `load_match` gets an `events_only` mode for every pining provider.
- **xT fits become sharded passes.** Per-match zone-count shards go through the same seam and are reduced with ADR-102's `ExpectedThreat.fit_from_counts`. They are byte-identical to a pooled fit and inherit resume, recorded failures and exclusions.
- Every network loader honours one cache variable.
- Every driver migrates, and a CI gate makes the old shapes unrepresentable.

**What changes outside `scripts/`.** One additive library API. `ExpectedThreat.zone_counts(actions) -> XtZoneCounts` is built from the *same* counting code `fit()` uses (§4.4), so `fit()` stays byte-identical, pinned by the SK-xT-1 oracle and the ADR-102 tests. No feature values change, and nothing needs retraining or re-materializing.

**What does not change.** Shard keys, shard schemas and generation tokens stay the same for every driver whose per-match output is unchanged, so finished shard generations resume across the migration. Three named exceptions (§7): `measure_cover_shadow_argmax_agreement`, `measure_rc4_orientation`, and any SkillCorner-including events-only pass (its token carries the admission digest). `load_matches`, `load_statsbomb_matches` and `load_open_data_matches` keep their signatures and outputs as thin wrappers, so e2e tests and ad-hoc callers keep working.

**D-1 (§8) is resolved by established convention.** A quality check governs only the data it measures. The tracking geometry gate keeps admitting tracking. Events-only consumers admit the 14 of 889 SkillCorner matches it excludes only if their events pass a pre-registered, event-side check with a stated false-alarm rate, and they record each one by name. The check's mechanism is built unconditionally; a registered measurement run before migration supplies its verdicts as data.

## 2. Motivation and evidence

### 2.1 How TF-56 surfaced it

The TF-56 driver fit its within-corpus xT via `list(load_matches(..., tracking_limit=0))`. For every native pining builder (IDSSE, Gradient Sports, SkillCorner), `tracking_limit` caps frames only *after* the whole tracking file has been read and parsed (§3 note; the `load_matches` docstring's "passed to the kloppy parser for SkillCorner" is stale and is corrected in this cycle). `list()` held every frame set, so the DGX fit process died of OOM.

Fixing that locally exposed that the worker loop had the same resume and load-failure defects as every sibling driver. That fix (events-only fit, fit-once manifest, fan-in-checked reduce) is parked in `stash@{0}` on `feat/tf56-positioning`. The owner's ruling (2026-09-22, in session): fix it systemically, gold standard, scope not a constraint.

### 2.2 The seam behaviour, from the code

`scripts/_driver.py::for_each` (lines 568–612 at `4ac26d0`):

```python
for i, item in enumerate(items, start=1):      # the loader has ALREADY downloaded + parsed here
    k = join_key(key(item))
    ...
    if already_done(generation, k):             # the resume check comes second
        skipped += 1; ...; continue
    attempted += 1
    try:
        frame = work(item)                      # only work() is guarded
    except Exception as exc:
        failures[k] = ...
```

`scripts/_loader_pining.py::load_matches` calls `_build_match_with_retry` directly in its loop. After 3 attempts that raises `RuntimeError`, which propagates out of the generator into the `for` statement above, not into the `try`.

ADR-052 D4 documents the first defect and chose streaming anyway: *"invert onto an id list only where `work` is unambiguously trivial next to the load … `items` must be STREAMED and never `list()`ed — materialising ~80 matches' tracking frames would defeat resume and can OOM."* D4's reason is to never hold many frame sets at once. A sized list of cheap references plus a per-item load hook keeps that property, because one match is loaded at a time and only if it is not done. So the D4 trade-off no longer applies, and §4.7 amends it.

## 3. Audit — every corpus driver, derived by AST (`4ac26d0`)

Method: parse every `scripts/*.py` (private-named included), then find:
- each `for_each(...)` item source,
- every call to a loader function,
- every `for` loop over a loader call.

The plan commits this as the gate's population function (§5). Shapes:

- **S** — streams full loads into `for_each` (defects 1+2).
- **M** — materializes every loaded match (frames included) into a list, then `for_each` walks it (defects 1+3 plus the D4 OOM class).
- **E** — an actions-only loop that parses tracking (defect 3).
- **U** — an un-sharded loop over a corpus (no resume at all; the ADR-052 "second loop" hole).
- **I** — already walks a list of ids or paths with the load inside `work`. This is the correct shape; it migrates onto the shared refs/hook anyway, for one idiom and the excluded outcome.
- **X** — deliberate whole-corpus load, exempt with a stated reason.

| Driver | Loader family | Shape | cache today | Notes |
|---|---|---|---|---|
| `build_gkdv_arm_values` | pining | S | no | |
| `build_layer2_spells` | pining | S | no | |
| `build_rq_pass_scores` | pining (GS) | S | `--cache-dir` | |
| `build_tf19_instrument_responsiveness` | pining | S | no | |
| `build_tf60_layer3_arm_values` | pining | M + E | no | `--xt-out` fit loop parses tracking and discards frames; arms pass holds every frame set in `loaded` |
| `derive_opengoal_range` | pining | S | no | |
| `materialize_tc3_frames` | pining | S | `--cache-dir` | also `preflight_reference_parity`, which loops over an injected `load_matches` for one match; becomes a direct single `load_match` call (§4.5) |
| `measure_gs_shot_distribution` | pining | E | `--cache-dir` | events-only: it discards `_frames` (ratified S→E at plan review 2026-09-23); loads via `events_only_loader` |
| `run_signoff_power` | pining | S | no | |
| `train_gk_completion` | pining | S | `--cache-dir` | |
| `train_xcross_attempt` | pining + `--data-dir` parquet | S | `--cache-dir` | |
| `train_xshot_occurrence` | pining + `--data-dir` parquet | S | `--cache-dir` | |
| `tune_structural_pass_sigma` | pining (GS) | S | no | |
| `validate_shot_goalmouth_sb` | pining (GS) | S | `--cache-dir` | |
| `validate_skillcorner_keeper_origin` | pining (SC) | S | no | |
| `validate_xcross_causal` | pining | S | no | |
| `validate_xs_probe` | pining | S | no | the 14-hour driver ADR-052 was born from |
| `validate_xshot_causal` | pining | S | no | |
| `measure_cover_shadow_argmax_agreement` | pining | M + E | `--cache-dir` | xT fit over `all_actions` of the materialized list |
| `calibrate_xt_bandwidth` | pining | I + E | `--cache-dir` | already inverted; re-lists the manifest per match; parses tracking for actions |
| `calibrate_tracking_defaults` | pining / databricks | X + I + E | `--cache-dir` | `_load_fold` is the Optuna objective INPUT (X, documented); `_load_xt_corpus_pining` is inverted but parses tracking (`tracking_limit=50`) for actions |
| `train_receiver_model` | pining / statsbomb | S ×2 | `--cache-dir` | two passes over `_load_corpus` |
| `validate_sb360_licensed_corpus` | statsbomb | S | no | |
| `validate_territorial_defense` | statsbomb | S | `--cache-dir` | its `--list-matches` downloads and builds every SB360 match just to print ids (a comprehension over `load_statsbomb_matches`) |
| `validate_gk_decision` | SC GI mini-loader + pining + statsbomb | S + U | `--cache-dir` | native leg streams `_load_gi_matches` (which silently `continue`s on a missing artifact); **reconstruction legs are un-sharded loops over ~875 SC tracking + ~60 SB360 matches** |
| `measure_rc4_orientation` | pining | provider-grain `for_each`, load inside `work` | `--cache-dir` | items are providers (`key=(provider,)`); `_work` streams `load_matches(max_per_provider=1)`; token already includes `run_commit` |
| `build_sb360_coverage` | statsbombpy | S | no | |
| `evolve_xsuccess_features` | open-data | S | no | own multi-competition walk |
| `train_pass_completion` | open-data / pining (GS) | S | no | own multi-competition walk |
| `train_win_probability` | open-data | S | no | own multi-competition walk |
| `train_xsuccess` | open-data | S | no | own multi-competition walk |
| `validate_territory_counterfactual` | open-data | S | no | own multi-competition walk |
| `validate_team_kpi_reliability` | open-data + wyscout | S + prepass | no | |
| `build_territory_ranking_census` | open-data | U prepass + I | no | prepass loads every match's actions into memory before `for_each` over ids |
| `validate_match_outcome_calibration` | open-data | U prepass + I | no | same shape |
| `train_match_outcome_dependence` | open-data | **U, ungated** | no | 3,961 matches, accumulate-then-fit, no shards; outside the ADR-052 gate population (`load_open_data_matches` missing from `_CORPUS_CALLS`) — as are `evolve_xsuccess_features`, `train_xsuccess`, `train_win_probability` and `build_sb360_coverage`, which do adopt `for_each` |
| `_xtgk_comparability` (private) | pining | U + E | `--cache-dir` | invisible to the gate (private name) |
| `_loader_pining_to_cache` (private) | pining | S (cache warmer) | `--out` (required) is its *materialized* frames/actions cache; raw artifacts are not cached (no `cache_dir` passed) | `select_match_ids` → skip `_cached` → stream `load_matches`; becomes a `for_each` pass (§4.5) |
| `check_stage1_argmax`, `measure_box_constant_delta` | parquet paths | I | n/a | local files; no network loader |
| `train_ghost_gk`, `train_ghost_outfield` | tc3 parquet walk | I / documented exception | n/a | `train_ghost_gk` stays ADR-052 D4's documented exception (game ids only knowable inside the file) |
| `_loader_databricks.load_matches` | databricks | X | n/a | only consumer is `_load_fold`; batched IN-query (ADR-068) — a per-item load would regress it |

**Unaffected, listed so the table is complete.** Five gate-population members are Shape-B cohort-cache drivers with no per-match loader loop, so nothing in this design touches them: `train_gk_retention`, `validate_xtgk_possession_value`, `validate_xtgk_v2`, `xtgk_v2_kappa_sweep`, `xtgk_v2_keeper_discrimination`.

**`tracking_limit` note.** `_build_idsse` parses the whole DFL positions XML, then caps. `_build_gradientsports` decompresses and parses every JSONL line, then caps. `build_skillcorner_frames` reads the whole parquet/JSONL, then `_head_with_player_data` caps. None avoids the parse.

The TF-56 driver (`build_tf56_positioning_validity`, on `feat/tf56-positioning`) is S + E. It adopts this seam when that branch merges `main`.

## 4. Design

### 4.1 Loader layer: references + single-match load (`scripts/_loader_pining.py`)

```python
@dataclass(frozen=True)
class MatchRef:
    provider: str
    match_id: str
    artifacts: Mapping[str, str]   # the manifest entry's artifact map (already fetched by the listing)

def list_match_refs(*, providers, match_ids=None, max_per_provider=None,
                    token=None, base_url=None) -> list[MatchRef]: ...

def load_match(ref: MatchRef, *, events_only: bool, tracking_limit=None,
               cache_dir=None, token=None, base_url=None) -> LoadedMatch: ...

class LoadedMatch(NamedTuple):
    provider: str; match_id: str; actions: pd.DataFrame
    frames: pd.DataFrame | None        # None iff events_only
    home_team_id: object
    visible_area: pd.DataFrame | None  # SB360 only, and only when not events_only

class MatchExcluded(ItemExcluded): ...   # raised by load_match for the SkillCorner S1 gate (tracking loads),
                                         # and by the §4.1 admission layer for events-only SkillCorner loads
```

- **Selection is single-sourced.** `list_match_refs` uses `_wanted_for_provider`, the function `load_matches` and `select_match_ids` already share, and lists each provider's manifest once. `select_match_ids` becomes `[(r.provider, r.match_id) for r in list_match_refs(...)]`.
- **Exclusion raises instead of skipping.** `load_match` wraps the existing `_build_match_with_retry` (download + retry + cache). It applies the S1 geometry gate by **raising `MatchExcluded(reason)`** instead of `continue`. The reason records the gate's measured `player_off_pitch_rate` and `ball_off_pitch_rate`, so the clause that fired is inspectable later.
- **`events_only` is keyword-only with no default.** Every call site states which it wants, and the gate (§5) relies on that.
- **`events_only=True`** never requests the tracking (or SB360 `freeze_frames`) artifact and never builds frames. The actions are byte-identical to the full build, because each provider's actions half is the same function its full builder calls.
  - **Already done:** idsse, SkillCorner and Gradient Sports, ported from the TF-56 WIP (`_artifact_roles`, `_build_match_actions`, `_idsse_actions` / `_skillcorner_actions` / `_gradientsports_actions`, `tests/scripts/test_loader_pining_events_only.py`). Each provider is pinned with the tracking artifact absent from `paths`, and GS across `homeTeamStartLeftExtraTime ∈ {True, False, None}`.
  - **New work: StatsBomb.** The ported code and tests still encode my retracted round-1 claim that StatsBomb has no separable tracking artifact. `freeze_frames` is its own artifact, and `build_statsbomb_match`'s actions (events + metadata + roster identity) never read it. `test_artifact_roles_statsbomb_has_no_events_only_mode` and `test_build_match_actions_refuses_statsbomb_and_unknown_providers` are **inverted**. They become "StatsBomb events-only actions are byte-identical to the full build on the committed SB360 slice with `freeze_frames` absent", plus "an unknown provider raises".
- **Dependency direction.** `ItemExcluded` lives in a new leaf module `scripts/_item_outcome.py` that imports nothing from `scripts/`. Both `_driver` (which recognises it) and the loaders (which raise it) import it. The loader, an adapter, never imports the orchestration seam just for an exception type. A test pins that each of `_item_outcome`, `_driver` and `_loader_pining` imports alone.
- **Wrappers stay byte-identical.** `load_matches` / `load_statsbomb_matches` become wrappers over refs + `load_match`. They keep their `4ac26d0` signature, 5-/6-tuple, the `EXCLUDED …` stderr line and the `excluded n/m` summary. **The wrappers are tracking-only and carry no `events_only` keyword** (round 6, CDLS-SPEC-29): they call `load_match(ref, events_only=False)`. The ported TF-56 work had added `events_only` plus `Literal[True]`/`Literal[False]` overloads to `load_matches`. That would leave a public, un-admitted events-only entry point, and its forwarding call `load_match(ref, events_only=events_only)` would fail Rule D. So the keyword, the overloads, and the ported tests that exercise `load_matches(events_only=...)` are removed. Their coverage moves to direct `load_match(events_only=True)` tests and to the admitted `events_only_loader` (§6). `4ac26d0` never had the keyword, so frozen-copy parity is unaffected. Its only consumer was TF-56's WIP fit, now the §4.4 count pass. Output is identical, pinned by a parity test against a frozen copy of the `4ac26d0` implementation. The one observable difference is timing: every manifest is listed up front rather than interleaved with loads.
- **One cache convention (§4.3).** `cache_dir=None` resolves from `SILLY_KICKS_CORPUS_CACHE_DIR`, the repo's `SILLY_KICKS_*` env-var convention (`SILLY_KICKS_NUMBA_CACHE`, `SILLY_KICKS_ASSERT_INVARIANTS`). Pining artifacts go under `<root>/<provider>/<match_id>/`, which is today's `cache_dir` layout, so an existing cache directory works unchanged when pointed at. Unset means today's behaviour: a temp dir per match, no persistent cache.
- **Events-only SkillCorner admission (§8) — a policy layer at the edge, not inside `load_match`** (round 5, CDLS-SPEC-27). A tracking load keeps the S1 gate exactly as today. An events-only SkillCorner load cannot run S1, so consumers admit it against the committed, provenance-stamped verdict artifact that the §8 registered driver produces.
  - **Why the edge.** Admission is policy, and CLAUDE.md's convention is "policy lives at the edge, never in the shared engine". Round 4 put it inside `load_match`, which deadlocked the artifact's own producer. On a first run there is no artifact; on a re-run the new matches are unmeasured; and a non-`sound` match could never be re-measured. So `load_match(ref, events_only=True)` stays a **pure loader**. It fetches and parses and consults nothing.
  - **The admission layer** lives in `scripts/_events_admission.py`:

    ```python
    class EventsOnlyAdmission:            # loaded from verdicts.json; fail-closed
        digest: str
        def preflight(self, refs, *, allow_unmeasured: bool) -> AdmissionRecord: ...
        def check(self, ref) -> None: ...  # raises MatchExcluded(<status, verdict, reason>) if not admitted

    def events_only_loader(refs, *, allow_unmeasured=False, cache_dir=None
                           ) -> tuple[Callable[[MatchRef], LoadedMatch], AdmissionRecord]: ...
    ```

    `events_only_loader` is the one sanctioned events-only entry for consumers. It runs the preflight once, then returns the per-item `load=` callable, which is `check(ref)` followed by `load_match(ref, events_only=True, ...)`. The `AdmissionRecord` carries `digest` and `unmeasured_admitted` into `XtFitProvenance` and `token_inputs`.
  - **The artifact is required only when a SkillCorner ref is requested.** An events-only pass over idsse, Gradient Sports, StatsBomb or open data never loads it, so a missing SkillCorner artifact cannot block unrelated work.
  - **Admission, per measured match.** `s1_passed` → admitted. `s1_excluded` or `tracking_unloadable` → admitted iff the event verdict is `sound`, else `MatchExcluded(<recorded status, verdict and reason>)`. `events_unloadable` → never admitted (its events cannot load anyway).
  - **Unmeasured matches refuse by default, by name.** The preflight computes `unmeasured = requested SkillCorner refs − measured listing`. If any are unmeasured, the pass refuses and names their keys, with the remedy: re-run the §8 driver, which is resumable and cheap on a warm cache. `--allow-unmeasured` admits them and records their keys in `XtFitProvenance.unmeasured_admitted` — the `--allow-failed` / `--allow-dirty` idiom. Refusal, not silent exclusion, is deliberate: excluding unmeasured matches by default would silently shrink the corpus.
  - The artifact is loaded once per process and fail-closed: a missing, unprovenanced or dirty-tree artifact refuses at the preflight, in seconds rather than after three failed items.
  - The artifact's digest joins the pass's `token_inputs`, so a regenerated artifact starts a new generation.
  - **Enforcement.** Rule D (§5) makes a bare `load_match(..., events_only=<not False>)` a CI failure outside the admission module and the artifact's registered producer. A consumer cannot skip admission by calling the loader directly.
- The stale `load_matches` docstring line about `tracking_limit` and kloppy is corrected (§3 note). So is the S1 code comment calling the exclusion "dormant on the kloppy path"; the native builder has fed the gate since Task 7 retired the kloppy frame path.

### 4.2 Open-data and SB360 loaders (`scripts/_sb_open_data.py`, `scripts/build_sb360_coverage.py`)

- **One multi-competition walk.** `list_open_data_refs(competitions, *, match_ids=None, max_matches=None) -> list[OpenDataRef]` single-sources the walk, with a GLOBAL cap. Nine drivers currently re-implement it by hand, each with its own `seen` counter or `chain.from_iterable`: `build_territory_ranking_census`, `evolve_xsuccess_features`, `train_match_outcome_dependence`, `train_pass_completion`, `train_win_probability`, `train_xsuccess`, `validate_match_outcome_calibration`, `validate_team_kpi_reliability`, `validate_territory_counterfactual`.
- **Single-match load with the shared cache.** `load_open_data_match(ref, *, preserve_native=(), cache_dir=None)` loads one match. With a cache root resolved (the argument, else `SILLY_KICKS_CORPUS_CACHE_DIR`), it caches the fetched raw events JSON under `<root>/statsbomb_open/<match_id>.json`, so resumes and repeat battery runs stop re-fetching from GitHub. **Unset means no cache**, fetching every time (today's behaviour). It is inherently event-only; its `frames` stays the empty DataFrame it is today.
- **Public-only guard unchanged.** `assert_statsbomb_open_data_mode()` still runs before any fetch (fail-closed public-only).
- **Wrapper kept.** `load_open_data_matches` stays as a byte-identical wrapper.
- **SB360 coverage too.** `build_sb360_coverage`'s statsbombpy walk gets the same refs + hook treatment.

### 4.3 The seam: `for_each(..., load=)` (`scripts/_driver.py`) and `ItemExcluded` (`scripts/_item_outcome.py`)

```python
# scripts/_item_outcome.py  (leaf module; imports nothing from scripts/)
class ItemExcluded(Exception):
    """A deterministic admission decision: this item is not in the pass, for `reason`."""
    def __init__(self, reason: str): ...
    reason: str

# scripts/_driver.py
def for_each(items, *, key, work, shard_root, token_inputs, token_reason=None,
             counters=None, tag="all", label="item", max_consecutive_failures=3,
             load=None) -> CorpusPassResult: ...
```

With `load` given, `items` are **references** (cheap; `Sized` recommended, which gives a real `[i/n]`). Each one goes through:

1. `k = join_key(key(ref))`. The key is computed from the **ref**, never from a loaded item. That is what makes resume-before-load possible. It must equal the key the driver used before migration, so existing generations resume (per-driver test, §6).
2. Shard present → skip; counters replayed (unchanged).
3. **Exclusion marker present → skip as excluded**, with the recorded reason replayed into the result.
4. Otherwise attempt `item = load(ref); frame = work(item)` inside **one** `try`:
   - `ItemExcluded(reason)` (from `load` or `work`) → write `<k>.excluded.json` (`{"reason": ...}`, atomically) and count it excluded. An exclusion is a success, so it resets the consecutive-failure run.
   - Any other exception → recorded failure. It counts toward `max_consecutive_failures`, so three unloadable matches in a row still abort; that is a systematic problem such as an expired token.
5. `counters(item, frame)` receives the **loaded** item, as today.

The rules that make this correct:

- **Exclusion is a first-class outcome, never an empty shard and never a failure.** An empty shard means "ran, produced nothing" and a failure means "retry on resume". An excluded item is neither: it is a deterministic admission decision, so it is persisted and skipped on resume. The marker is `.json`, invisible to every `*.parquet` glob, so `reconcile`, `assert_conservation`'s shard count and every driver's combine are unaffected.
- **Only deterministic exclusions may raise `ItemExcluded`** — a pure function of the item's artifact bytes and the code (the S1 gate is one). Time-varying *availability*, such as a manifest that does not list an artifact *yet*, must not persist. It is resolved at **ref-listing** time instead: re-evaluated every run, never written as a marker, and counted and named by the lister (`validate_gk_decision`, §4.6).
- **Marker validity equals shard validity.** A marker lives in the generation directory, exactly like a shard, so it inherits ADR-052's declared token-completeness limit. A change to the code that decides an exclusion needs the same token bump as a change to the code that computes a shard. This is no new hazard: the S1 gate already decides which matches a driver scores.
- **Conservation absorbs exclusions without a new parameter.** `assert_conservation` counts a key as accounted if it has a shard **or** an exclusion marker. The escape-hatch signature stays unchanged, and the "no convenient default for an invisible value" rule (the `counters_unrecorded` lesson) holds, because there is no default to get wrong.
- **Exclusions are reported.** `CorpusPassResult` gains `excluded: int` and `exclusions: dict[key, reason]`, covering this pass's own keys and replayed on resume. `manifest()` gains `n_excluded`, which `_partition.aggregate_manifests` sums like any int. "Excluded" is corpus-scoped (deterministic), so a resumed pass reports it, unlike `n_attempted`.
- **Without `load`, `for_each` behaves exactly as today**, pinned by the existing `test_driver.py` suite plus a byte-identity test of its manifests. A driver on the old shape keeps working until the gate (§5) refuses it.
- **Injectivity is unchanged.** It is checked on ref keys as they stream.

### 4.4 Sharded xT fits: a count pass reduced by `fit_from_counts` (CDLS-SPEC-01)

The round-1 design had a shared helper that streamed events and fit xT in one un-sharded pass. That re-created defect 2: one unloadable match aborts every fit, or silently shrinks the fit corpus. It also escaped the §5 gate. The base commit already ships the primitive that fixes it. ADR-102's `ExpectedThreat.fit_from_counts` builds the four matrices from integer zone counts that are **additive** across partitions. So an xT fit is a `for_each` pass like any other, and inherits resume-before-load, recorded failures, exclusions, conservation and the gate.

**Library (additive, `silly_kicks/xthreat/`):**

```python
@dataclass(frozen=True)
class XtZoneCounts:
    l: int; w: int
    shot_counts: NDArray[np.int64]              # (w, l)
    goal_counts: NDArray[np.int64]              # (w, l)
    move_counts: NDArray[np.int64]              # (w, l)  valid-START moves (feeds _action_prob)
    transition_start_counts: NDArray[np.int64]  # (w, l)  valid START+END moves (Singh denominator)
    transition_counts: NDArray[np.int64]        # (w*l, w*l)  SUCCESSFUL from->to (Singh numerator)
    def __add__(self, other: "XtZoneCounts") -> "XtZoneCounts": ...   # raises on a (l, w) mismatch
    @classmethod
    def zeros(cls, l: int, w: int) -> "XtZoneCounts": ...
    def as_fit_kwargs(self) -> dict[str, NDArray[np.int64]]: ...

class ExpectedThreat:
    def zone_counts(self, actions: pd.DataFrame) -> XtZoneCounts: ...   # uses self.l / self.w
```

- **Single-sourced counting.** ADR-102 already splits each matrix builder into a pure `_*_from_counts` core plus a from-actions wrapper, but the wrappers compute their counts inline. This cycle extracts those inline counting steps into private per-aggregate extractors: shots/goals from `_scoring_prob`; valid-start moves from `_action_prob`; valid start+end and successful transitions from `singh_transition_matrix`. The wrappers **and** `zone_counts` call the same extractors, so `fit()` and the counts path cannot diverge in which rows they count. The extraction is output-preserving: the SK-xT-1 frozen oracle and the ADR-102 tests stay green unchanged.
- **Byte-identical to a pooled fit.** Counts are integers and the reduce is integer summation, so summed per-match counts equal pooled counts exactly. `fit_from_counts(**summed.as_fit_kwargs())` then builds byte-identical matrices, and the same deterministic `value_iteration` gives a byte-identical `xT` (`fit()` and `fit_from_counts()` pass identical arguments to it, `_model.py:150` / `:478`). §6 pins this with `np.array_equal(..., equal_nan=True)`, not a tolerance.
- **The method is the model's own.** `zone_counts` is a method on `ExpectedThreat`, so counts are always binned on the grid the model will fit.
- Singh-only, like `fit_from_counts`. Every xT-fitting driver in §3 uses the default `singh_counts`. `calibrate_xt_bandwidth` needs raw destinations for its KDE sweep and keeps its action shards.
- Surface obligations:
  - `XtZoneCounts` is exported in `xthreat.__all__`.
  - Both new public symbols carry an Examples section (public-API example gate).
  - **ADR-102 is amended**: `zone_counts` extends its counts contract, and the count *extractors* join the single-sourced cores.
  - The CHANGELOG entry and the CLAUDE.md xT bullet are written in the release commit.
  - No retrain, no re-materialize, C4-free (no new container or aggregator).

**Scripts (`scripts/_xt_corpus.py`):**

```python
def xt_count_pass(refs, *, key, load_actions, shard_root, token_inputs, tag="all",
                  l=16, w=12, label="match") -> CorpusPassResult: ...
def fit_xt_from_count_pass(res: CorpusPassResult, *, l=16, w=12,
                           allow_failed=False) -> tuple[ExpectedThreat, XtFitProvenance]: ...

@dataclass(frozen=True)
class XtFitProvenance:
    fit_keys: tuple[str, ...]          # shards that contributed counts
    excluded: Mapping[str, str]        # key -> reason (from the pass)
    failed: Mapping[str, str]          # key -> error  (from the pass)
    l: int; w: int
    counts_digest: str                 # digest of the summed counts
    admission_digest: str | None       # §8 verdict artifact consulted (events-only SkillCorner), else None
    allowed_failed: bool               # True iff the caller passed allow_failed=True
    unmeasured_admitted: tuple[str, ...]  # keys admitted under --allow-unmeasured (§4.1), else ()
```

- **The count pass.** `xt_count_pass` is `for_each(refs, load=load_actions, work=lambda actions: counts_to_frame(ExpectedThreat(l, w).zone_counts(actions)))`. Here `load_actions` wraps the admitted loader from `events_only_loader(refs, ...)` for pining (so SkillCorner admission applies, §4.1), and `load_open_data_match(ref).actions` for open data.
- **Its token** declares `l`, `w`, `events_only: True` and a count-shard schema version (the 4.77.1 pair).
- **Shards are sparse long-form rows** `(aggregate, from_zone, to_zone, n)`: `to_zone = -1` for the four zone aggregates, nonzero entries only. `counts_to_frame` / `counts_from_frames` round-trip exactly (tested).
- **The reduce** sums the shards of `res.keys` and calls `fit_from_counts`. It never drops silently: `XtFitProvenance` names every excluded and failed key.
- **Failed fits refuse by default, uniformly** (CDLS-SPEC-17). `fit_xt_from_count_pass(res, *, allow_failed=False)` raises when `res.failures` is non-empty. That preserves today's complete-or-nothing behaviour for every caller: `build_tf60_layer3_arm_values`, `measure_cover_shadow_argmax_agreement`, `_xtgk_comparability`, and TF-56.
  - A failure is either transient, which a resume retries (a failure writes no shard), or persistent, which is a data defect that needs a deliberate decision.
  - Each caller exposes `--allow-failed`, following the `--allow-dirty` idiom: it permits the fit and **records** the failed keys in `XtFitProvenance` and the artifact, never laundering them.
  - A permanently unloadable match is therefore recorded and refused by default, never a crash loop and never a silent shrink.
- **Callers:**
  - `build_tf60_layer3_arm_values` (`--xt-out` serializes `fit_keys` as its `corpus_ids`),
  - `measure_cover_shadow_argmax_agreement`,
  - `_xtgk_comparability`,
  - TF-56 after merge.
  
  `calibrate_tracking_defaults._load_xt_corpus_pining` keeps its action shards (they feed `fit_frozen_xt`, a separate artifact contract) and switches its load to the admitted `events_only_loader` (§4.1, §4.5), never a bare `load_match(events_only=True)`, which Rule D rejects.

### 4.5 Migrations by shape

- **S → refs + `load=`.** `for_each(list_match_refs(...), key=<unchanged>, load=lambda r: load_match(r, events_only=False, tracking_limit=..., cache_dir=...), work=<unchanged, now receives the LoadedMatch>)`. `work`'s item unpacking is adapted (5-tuple → `LoadedMatch` fields). Shard content is unchanged, so generation tokens are unchanged.
- **M → two passes, no materialized list.** First, `xt_count_pass` + `fit_xt_from_count_pass`. Second, `for_each(refs, load=full load, work=...)` with the fitted xT closed over. The OOM class disappears. Where a token declares the loaded match ids (`measure_cover_shadow_argmax_agreement`), it now declares the **requested** refs plus the fit's `counts_digest`. A ref list precedes the load, so it cannot know S1 exclusions in advance. This moves that driver's generation whatever D-1 resolves to (§7).
- **E → events-only loads via the admitted loader.** Each of these goes through `events_only_loader` (§4.1; Rule D): `build_tf60 --xt-out` (via the count pass); `calibrate_xt_bandwidth._load_one_match`, which also switches to refs and drops its per-match manifest re-listing; `calibrate_tracking_defaults._load_xt_corpus_pining`; and `_xtgk_comparability`'s fit loop (via the count pass).
- **U → sharded `for_each` passes.**
  - `validate_gk_decision`'s two reconstruction legs become their own `for_each` passes with refs + `load=`. Their outputs are per-match sample tables, so they shard naturally, combined from `res.keys`.
  - `train_match_outcome_dependence` becomes a `for_each` over open-data refs whose per-match output is its `_match_tuples` table, fit in a reduce over the shards.
  - The two open-data prepasses (`build_territory_ranking_census`, `validate_match_outcome_calibration`) each become a `for_each` over refs writing per-match action slices, combined before the whole-corpus barrier computation. This follows the `calibrate_tracking_defaults._load_xt_corpus_pining` precedent.
  - `_xtgk_comparability`'s accumulate loop gets the same treatment.
- **I → shared refs + hook.**
  - `calibrate_xt_bandwidth`: same keys. The excluded outcome replaces "empty shard for an S1 drop".
  - `train_ghost_outfield`: its path list is already the ref list; `load=pd.read_parquet`.
- **`measure_rc4_orientation` (CDLS-SPEC-04) → per-match refs.** Today its items are providers and `_work` streams `load_matches(max_per_provider=1)` inside `work`, a loop Rules A and C would refuse. It becomes `for_each(list_match_refs(providers=_PROVIDERS, max_per_provider=1), key=lambda r: (r.provider, r.match_id), load=<full load>, work=<measure the LoadedMatch>)`.
  - The key changes from `(provider,)` to `(provider, match_id)`, a generation move (§7). The cost is nil: its token already includes `run_commit`, so every code change starts a new generation anyway.
  - Its `run()` guard keeps its meaning. A provider with no scored match refuses to publish, and an S1-excluded match now counts as "no scored match" explicitly rather than via an empty generator.
  - The shard rows already carry `provider` and `match_id`, so the published artifact's content is unchanged.
- **`validate_gk_decision._load_gi_matches` (CDLS-SPEC-10).** It stops silently `continue`-ing on a missing GI artifact. A missing artifact is **manifest state**, which changes over time, so it is resolved at ref-listing and never persisted as a marker. The GI ref-lister returns `(refs, unavailable: dict[key, reason])`, and the driver writes `n_gi_unavailable` plus the keys into its manifest and metrics. A later upload is picked up on the next run.
- **`_loader_pining_to_cache` (cache warmer) → a `for_each` pass** (CDLS-SPEC-15).
  - It becomes `for_each(todo_refs, key=lambda r: (r.provider, r.match_id), load=lambda r: load_match(r, events_only=False, tracking_limit=..., cache_dir=...), work=<write_match_cache, return None>)`.
  - `_cached` stays as the resume pre-filter over `list_match_refs` output. It iterates refs without loading, which Rule C allows.
  - It gains recorded failures, exclusions and progress.
  - Its "no vx/vy" SKIP becomes `ItemExcluded("no vx/vy")`: a pure function of the artifact, so a persisted marker is correct.
  - The S1 drop becomes a named marker instead of a silent skip.
  - It now passes `cache_dir`, so raw artifacts are cached as well. `--out` stays required, because the materialized frames/actions cache is a different kind of cache from the raw-artifact cache.
- **`materialize_tc3_frames.preflight_reference_parity`** loads its one match with a direct `load_match(ref, events_only=False)` call; its injected `load_matches` parameter becomes `load_match`. No loop remains, so no exemption is needed.
- **`--list-matches` paths that load in order to list** switch to ref listers. `validate_territorial_defense --list-matches` today downloads and builds every SB360 match (327) through a comprehension over `load_statsbomb_matches` just to print their ids.
- **X stays, recorded.** `calibrate_tracking_defaults._load_fold` is the Optuna input and has its own RAM fail-fast guard. `_loader_databricks.load_matches` is ADR-068-batched and its only consumer is `_load_fold`. Both go in the gate's reasoned exemption bucket.

### 4.6 What the §5 gate closes, and what it does not

The ADR-052 follow-up reads: *"A driver that calls `for_each` over something trivial and separately accumulates over the real corpus writes no shards for that second loop and lists none of its items. Catching that needs a fan-in check at reconcile time; recorded as a follow-up."*

This cycle closes it **for every loading loop the §5 rules can see**. Stream loaders are banned in drivers (Rule A). Every corpus load must go through `load_match` / `load_open_data_match` or an injected `load*` callable, and Rule C flags any loop construct that performs such a load outside the named homes of a loading loop.

**Residual, stated:** a loop over refs that reaches the loader only through indirection spanning functions or modules is not seen statically. Examples: a ref list stored on an object and iterated elsewhere, or a load callable passed through two helpers. The §6 planted cases pin exactly which shapes are covered.

### 4.7 ADR-052 amendment (in place, no new ADR number)

- **D4 revised.** The goal (never hold many frame sets) is kept and the mechanism changes: drivers pass `for_each` a cheap reference list and a `load=` hook, so resume skips the load, and the "invert only where work is trivial" rule is retired. `train_ghost_gk` stays the documented exception, because its game ids are only knowable by reading inside each file.
- **New D13: the excluded outcome** (§4.3), including the determinism rule and marker validity equals shard validity.
- **New D14: loader contract** (§4.1). Refs plus an explicit-`events_only` single-match load; streams are wrappers. The loader stays pure. Admission (events-only SkillCorner) is a policy layer at the edge (`scripts/_events_admission.py`), enforced by Rule D.
- **New D15: xT fits are count passes** (§4.4). ADR-102 is amended alongside (`zone_counts`, the shared count extractors).
- **The follow-up** is closed to the extent §4.6 states, with the residual recorded.

### 4.8 Source factories (reuse; amendment 2026-09-23, owner-ratified)

The §4.5 migration recipe repeats one 3-line source construction — `resolve_cache_dir` +
`list_match_refs` + a `load=lambda ref: load_match(ref, events_only=False, tracking_limit=…,
cache_dir=…)` closure — in every tracking driver. It was hand-written identically 4× during
execution (`_pining_source` in `train_xshot_occurrence`/`train_xcross_attempt`, `_corpus_source` in
`train_receiver_model`, `_pining_source` in `validate_sb360_licensed_corpus`) plus inline in ~12
more. That is the reuse signal, so the construction is extracted into two public factories:

- **`_loader_pining.pining_source(providers, *, match_ids, max_per_provider, tracking_limit,
  cache_dir, token, base_url) -> (refs, load)`** — `load(ref)` yields a full `LoadedMatch`
  (`events_only=False`).
- **`_sb_open_data.open_data_source(competitions, *, match_ids, max_matches, preserve_native,
  cache_dir) -> (refs, load)`** — the open-data analogue.

The events-only mode already has its factory (`_events_admission.events_only_loader`), so the surface
is symmetric: one source factory per load mode. **Decision: extract AND retrofit all** already-migrated
drivers onto the factory (one seam across the whole cycle); the ~17 remaining drivers use it from the
start. Generations stay byte-identical (same refs + load semantics → same shard keys / tokens).
Both factories are public loader-module functions without a `load_`/`list_`/`select_`/`fetch_` prefix,
so each takes a `_LOADER_NON_CORPUS` entry (else the §5.1 population gate fails); both are gate-clean
(they call `list_match_refs`/`load_match(events_only=False)` — not a stream loader, not a loading
loop). Drivers that reshape the item, key by `_source_key`/`ref.match_id`, or dual-source (--data-dir)
keep a thin per-driver wrapper around the factory. Plan Task 8.5 implements it. No new ADR number —
this is a reuse extraction of the D14 loader contract, not a new contract.

## 5. CI gate (extends `tests/scripts/test_corpus_driver_resilience.py`)

This follows the derive-and-assert-exactly pattern from ADR-056, with three buckets: registry, reasoned exemptions, and `_UNDERIVABLE` asserted empty.

1. **Population derived from the loader modules, complete by enumeration.** Every public function defined in `scripts/_loader_pining.py`, `scripts/_sb_open_data.py` and `scripts/_loader_databricks.py` (AST) must be classified. It is either:
   - a **corpus function** (name prefix `load_` / `list_` / `select_` / `fetch_`; these form the corpus-call set that replaces the hand-written `_CORPUS_CALLS`), or
   - in `_LOADER_NON_CORPUS` with a reason. At `4ac26d0` that is `build_statsbomb_match`, `build_skillcorner_frames`, `match_visibility`, `assert_statsbomb_open_data_mode`, `all_open_competitions`, `shape_action_values`, `resolve_retention_model`.
   
   The exemption is asserted exact both ways. A new corpus loader named by the convention joins automatically; any other new public function fails CI until classified. (`fetch_action_values` / `fetch_idsse_events` join the corpus set by prefix; they have no driver callers, so the population does not grow.) A plant pins that the derived set contains the known loaders, so a vacuous derivation fails.
2. **Scan every script, private ones included.** Every `scripts/*.py`, with **no module-level exemptions**: the loaders, `_driver.py` and `_xt_corpus.py` are scanned too, and only named functions are exempt (Rule C). That closes the `_xtgk_comparability` blind spot and stops an un-sharded loop regrowing unseen inside the seam's own modules.
3. **Rule A — no stream loaders in drivers.** `load_matches`, `load_statsbomb_matches` and `load_open_data_matches` may not be called by any driver outside `_STREAM_LOADER_EXEMPT` (initially `calibrate_tracking_defaults` for `_load_fold`, with a reason). The exemption is asserted exact both ways.
4. **Rule B — `events_only` is always explicit.** Every `load_match(...)` call site passes `events_only=` as a keyword. (The signature has no default, so omitting it is a `TypeError`; the static rule catches it before runtime and documents intent.)
4a. **Rule D — events-only loads go through admission** (round 5, CDLS-SPEC-27).
   - A `load_match(...)` call whose `events_only=` value is anything other than the literal `False` — the literal `True`, a variable, or any expression — fails CI unless the call sits inside an allowlisted function.
   - **Function-granular, like Rule C** (round 6, CDLS-SPEC-31), so a second un-admitted path cannot grow unseen inside an allowed module. An entry is a qualified `module.function` naming a **module-level** function. A call counts as inside it if it appears anywhere in its body, closures defined within included, because the per-item callable `events_only_loader` returns is such a closure.
   - `_UNADMITTED_EVENTS_ONLY_ALLOWED`, each entry with a reason, asserted exact both ways. Initially:
     - `_events_admission.events_only_loader` — it *is* the admitted loader;
     - `build_skillcorner_s1_event_validity._events_pass` — the artifact's registered producer, which must measure without the gate its output feeds.
   - Consumers use `events_only_loader`.
   - Loader modules are scanned too. `_loader_pining.load_matches` / `load_statsbomb_matches` call `load_match(ref, events_only=False)` only (CDLS-SPEC-29), so they need no entry.
   - Plants:
     - red: a bare `load_match(r, events_only=True)` in a driver; a variable-valued `events_only=` in a driver; a second un-admitted call in another function of `_events_admission.py`;
     - green: an `events_only_loader(...)` twin; the producer's `_events_pass` call.
5. **Rule C — no un-sharded *loading* loops** (redefined in round 3, CDLS-SPEC-15). Iterating a ref or id list *without loading* is legitimate: metadata, corpus fingerprints, visibility checks, resume pre-filters. So the rule targets the load, not the iterable.
   - A **loop construct** is a `for` statement, a list/set/dict comprehension, a generator expression, or a `map(...)` / `filter(...)` call.
   - It is a **loading loop** iff:
     - **(i) it iterates a load.** Its iterable is a call to a §5.1 corpus function whose name starts with `load_`, or to a parameter of the enclosing function whose name starts with `load`. Or its iterable is a **Name whose reaching binding** is such a call. Resolution is flow-sensitive: the nearest preceding assignment to that name in the same function, by source position. A rebinding to anything else clears it, so `train_gk_completion.main`'s rebound `pairs` is not flagged.
     - **(ii) it performs a load per item.** Its body calls `load_match`, `load_open_data_match`, any §5.1 `load_*` corpus function, or a parameter of the enclosing function whose name starts with `load`. For `map` / `filter`, "body" means the function argument: a Name, or a lambda's body. The injected-callable case covers the shape round 1's shared fit helper had.
   - **Measured, not assumed.** A scratch AST emulation of this definition, run over `scripts/*.py` at `4ac26d0`, flags 23 functions (keyed on the innermost enclosing function). Every one is a §4.5 migration site, plus the exempt `_load_fold`. None of the five green sites named below is flagged. The plan commits the emulation as the gate and re-derives the list.
   - A loading loop fails unless its innermost enclosing function is in `_UNSHARDED_LOOP_EXEMPT`. Entries are qualified `module.function`, each carries a reason, and the set is asserted exact both ways.
   - Initial `_UNSHARDED_LOOP_EXEMPT` — the only legitimate homes of a loading loop:
     - `_driver.for_each` — it *is* the sharded loop (it calls its `load` parameter per item);
     - `_loader_pining.load_matches`, `_loader_pining.load_statsbomb_matches`, `_sb_open_data.load_open_data_matches` — the stream wrappers, which Rule A bans from drivers;
     - `calibrate_tracking_defaults._load_fold` — the documented X item: the Optuna objective's input, held whole by design behind its own RAM fail-fast guard (also in `_STREAM_LOADER_EXEMPT`).
   - **Round-2 sites, disposed explicitly:**
     - *Stay green, asserted by name in a live-population test* (so a rule change that flags them fails loudly): `train_gk_completion._corpus_taxonomy` (comprehensions over `select_match_ids` output), `train_gk_completion.main` (loop over a rebound `pairs`), `train_xcross_attempt._corpus_fingerprint`, `train_xshot_occurrence._corpus_fingerprint`, and `_loader_pining_to_cache.main`'s `_cached` pre-filter. The live-population test runs against the **migrated** tree. At `4ac26d0`, `_loader_pining_to_cache.main` is flagged, because its load loop and its pre-filter share one function; it goes green once the load loop moves into `for_each` (§4.5). The other four are green at `4ac26d0` already.
   - `_loader_databricks.load_matches` is **not** exempt: its loops call `_convert`, never a loader or a `load*` parameter, so it is not a loading loop, and listing it would make the "exact both ways" check fail on day one (CDLS-SPEC-24).
     - *Migrated, so no exemption is needed:* `_loader_pining_to_cache`'s load loop becomes a `for_each` pass, and `materialize_tc3_frames.preflight_reference_parity` becomes a direct single `load_match` call (§4.5).
6. **Non-vacuity plants.**
   - Rule C, red: (i) direct stream-loader iterable; (i) Name bound to a stream loader; (i) iterable that calls a `load*` parameter; (ii) body calls `load_match`; (ii) `map` with a loading lambda; (ii) body calls a `load*` parameter.
   - Rule C, green: rebinding before the loop; an id-only loop over `select_match_ids` output.
   - Rule A and Rule B: one red plant and one migrated green twin each (the existing gate's idiom).
   - The §4.6 residual: a cross-function indirection plant that *is* uncaught, so the stated limit is pinned rather than implied.
7. **The existing adoption / conservation / injectivity cases stay.** `_CORPUS_CALLS` is replaced by the derived set.

Considered and rejected: a runtime guard making `load_match` refuse outside a `for_each` pass. It would fire only when a driver runs, which for most drivers is on the owner's box, not in CI. It would also force an opt-out at every legitimate direct call (tests, the parity preflight). The static rules plus §4.4's removal of the only known indirect loop cover the observed classes.

## 6. Testing (TDD, all local; no DGX before `/review-impl`)

- **Seam** (`tests/scripts/test_driver.py`):
  - The load hook is never called for a key with a shard or a marker (call-count spy, non-vacuous: the spy fires for a fresh key).
  - A load error becomes a recorded failure and the pass continues.
  - Three consecutive load failures abort.
  - `ItemExcluded` from `load` and from `work` writes a marker, is skipped and replayed on resume, is counted by conservation, and is invisible to `reconcile` and parquet globs.
  - An exclusion resets the failure run.
  - Injectivity on ref keys.
  - Sized refs render `[i/n]`.
  - Without `load`, behaviour is byte-identical, pinned against recorded manifests from the current suite.
  - `_item_outcome` imports nothing from `scripts/`.
- **Loaders:**
  - `list_match_refs` agrees with `select_match_ids`.
  - `load_match(events_only=True)` is byte-identical to the full build's actions for idsse, SkillCorner, GS (ported), and StatsBomb (new; the two inverted tests, §4.1).
  - `load_matches` / `load_statsbomb_matches` / `load_open_data_matches` wrappers are byte-identical to frozen copies of the `4ac26d0` implementations over monkeypatched network, including the S1 exclusion print and summary.
  - `SILLY_KICKS_CORPUS_CACHE_DIR` resolution (argument beats env beats unset → today's behaviour), for pining and open data.
  - Open-data cache round-trip: fetch once; the second load reads the file (fetch-spy count 1).
  - `MatchExcluded`'s reason carries both measured rates.
- **xT counts (library, `tests/xthreat/`):**
  - `ExpectedThreat.zone_counts` equals the ADR-102 test's independent `_aggregate` oracle on its fixtures A and B, including the valid-start / NaN-end boundary row. That oracle stays independent, never replaced by the function under test.
  - `fit_from_counts(**zone_counts(a).as_fit_kwargs())` equals `fit(a)`: all four matrices and `xT` `np.array_equal(..., equal_nan=True)`.
  - **Additivity:** for `spadl_actions` split into pseudo-matches, `sum(zone_counts(part))` equals `zone_counts(pooled)`, and the fitted `xT` is `array_equal(..., equal_nan=True)` to the pooled fit.
  - `XtZoneCounts.__add__` raises on a grid mismatch.
  - SK-xT-1 oracle and ADR-102 tests unchanged and green.
- **xT count pass (scripts):**
  - `counts_to_frame` / `counts_from_frames` round-trip exactly.
  - `xt_count_pass` → `fit_xt_from_count_pass` equals a pooled fit on the same fixture (`array_equal(..., equal_nan=True)`).
  - A pass with a failed ref: `fit_xt_from_count_pass` **raises** by default. With `allow_failed=True` it fits, and `XtFitProvenance.failed` and `allowed_failed` record the failure.
  - A failing ref does not abort the count pass itself; `for_each` records it and continues. The reduce then refuses or records per the line below.
  - An excluded ref is named in `excluded`.
  - A resumed count pass loads nothing (spy).
- **Per driver:**
  - Its existing tests stay green. The test files that monkeypatch loaders are swept onto `list_match_refs` / `load_match` fakes. The plan **derives** that list (AST: files whose monkeypatch targets name a loader function) rather than carrying a number; round 1's "19" was method-dependent (13–27).
  - A **key-compatibility** test per migrated driver: `key(ref) == key(old_item)` for the same match, proving finished generations resume. The §7 exceptions are asserted as intended moves instead.
  - Drivers whose output is unchanged keep their `token_inputs` byte-identical (asserted).
  - U-shape conversions get a local reduce test over synthetic shards (the `feedback_test_trainer_locally_before_dgx` gate).
- **Gate:** the §5 rules with plants on both sides, per Rule C shape, plus the residual plant. A live-population test names the five §5 green sites and asserts that none is flagged.
- **Events-only SkillCorner admission (§4.1, §8), both sides of every status:**
  - `s1_passed` → admitted.
  - `s1_excluded` / `tracking_unloadable` with verdict `sound` → admitted; with any other verdict → `MatchExcluded` carrying the recorded reason.
  - `events_unloadable` → not admitted.
  - An unmeasured requested ref → the preflight refuses and names the key.
  - The same ref with `--allow-unmeasured` → admitted, and its key appears in `XtFitProvenance.unmeasured_admitted`.
  - A missing, unprovenanced or dirty-tree artifact refuses at preflight — but only when a SkillCorner ref is requested. A GS-only events-only pass runs with no artifact present.
  - The artifact digest enters `token_inputs` and `XtFitProvenance.admission_digest`.
  - `load_match(ref, events_only=True)` itself consults nothing: it loads a match the artifact marks non-`sound`, since admission is the edge's job.
- **Task 0 driver (local, before the DGX run; the `feedback_test_trainer_locally_before_dgx` gate):**
  - A mirrored synthetic cell (all anchored actions reflected) is `reversed`; its unmirrored twin is not (both sides of the band).
  - A cell with `n < 5` is unscored, and an unscored regulation cell gives `insufficient`.
  - A clamped-origin synthetic match trips `boundary_pileup`; its clean twin does not.
  - The Poisson-binomial null matches a brute-force enumeration on a small cell.
  - The reduce writes `verdicts.json` with provenance and the full measured listing.
  - A match whose tracking load fails but whose events load gets status `tracking_unloadable` **and** an event verdict.
  - A tracking-pass failure writes no shard, so a resume retries it rather than freezing the status.
  - **No self-deadlock (CDLS-SPEC-27), both sides:**
    - Task 0 runs to completion with **no** artifact present.
    - It runs over a **stale** artifact that lacks newly listed matches, and measures them instead of refusing.
    - It **re-measures** a match that the prior artifact marked non-`sound`; a changed statistic changes the verdict.
    - A spy asserts the producer never opens `verdicts.json`.
  - **The reduce refuses on outstanding failures (CDLS-SPEC-28):** it raises and names the keys by default. With `--allow-failed` it writes the artifact, and the affected entries carry `allowed_failed: true` and the error.
- **Whole suite + ruff (CI scope) + pyright** green before hand-off. The local pyright baseline is 16 pre-existing errors (measured 2026-09-22 with this cycle's changes stashed, all in `build_sb360_coverage`, `probe_sb_cross_blocked` and network-gated e2e tests). The plan re-measures it on this branch's base, and the bar is zero new ones.

## 7. Hyrum / consumers / generation moves

- **Additive surfaces.** Driver CLIs gain `--cache-dir` where missing. Manifests gain `n_excluded`; `aggregate_manifests` sums ints, and its `dropped_fields` report stays empty. `xthreat` gains `zone_counts` / `XtZoneCounts` (additive public API; `fit()` byte-identical).
- **Unchanged shapes.** Loader wrappers keep their public shapes, so e2e tests (`tests/tracking/*_e2e.py`, 8 files) and the lakehouse-facing `scripts/` contracts are unchanged. `select_match_ids` is kept.
- **Generation moves (three; the third ratified at plan review 2026-09-23).**
  - `measure_cover_shadow_argmax_agreement`: its token now declares the requested refs plus the fit's `counts_digest`, because a ref list precedes the load.
  - `measure_rc4_orientation`: its key changes from `(provider,)` to `(provider, match_id)`; free, since its token already includes `run_commit`.
  - **A SkillCorner-including events-only pass** (`calibrate_xt_bandwidth`, `calibrate_tracking_defaults._load_xt_corpus_pining`): the admission artifact's digest joins its `token_inputs` (§4.1), so the generation moves. Required for correctness — the fit corpus now depends on the artifact, which admits the `sound` S1-excluded matches the pre-migration streamed path `continue`d past (no shard existed for them). The digest joins the token only when it is not `None`, so a pass with no SkillCorner ref keeps its token byte-identical.
  
  All three are exceptions to §1's "generation tokens stay the same". None has an in-flight run.
- **Fit corpus ≠ scored corpus within one driver (from D-1, deliberate).** An M-shape driver's xT is fit from an events-only count pass. That pass includes S1-excluded SkillCorner matches whose events are `sound`, which the tracking-based scoring pass excludes. Each artifact records both sets: the fit's `XtFitProvenance.fit_keys` and the scoring pass's exclusions.
- **Research artifacts.** Only the new `docs/research/skillcorner_s1_event_validity/` is generated (Task 0); no existing artifact is re-generated. Drivers whose xT fit becomes events-only produce a fit corpus that includes the `sound` S1-excluded matches on their next run. They record it, so the difference is visible.

## 8. D-1 — events-only consumers and the SkillCorner geometry gate (resolved by established convention; confirmed by measurement)

**The question.** The S1 gate (`silly_kicks/tracking/skillcorner.py::geometry_rate_gate`) needs built tracking frames. It excludes a match whose player off-pitch rate (> 3 m) exceeds 0.005, or whose ball off-pitch rate (> 10 m) exceeds 0.0005. `load_matches` then drops the whole match, events included. An events-only load cannot evaluate it. Scale: 14 of 889 cached SkillCorner matches (1.6%, `docs/research/gk_decision_construct_validity/findings.md`).

**The principle.** A quality check governs the data it measures. S1 measures tracking coordinates and decides whether a match's tracking is admissible. It says nothing about the match's events unless the defect is shown to reach them. Today's event exclusion is not a decision anyone made: it is a side effect of `load_matches` dropping the whole match.

**The established convention.** The gk_decision battery already applies this per-artifact split. Its events-based native tier used all 909 SkillCorner GI matches, the 14 included. Its tracking-based reconstruction tier used the 875 S1-passing matches, and `findings.md` gives the reason: "a geometrically-broken match's reconstructed tracking would corrupt the comparison". Round 1 escalated this as an owner decision when the repo had already settled it.

**The evidence on the 14.** `findings.md` attributes them to the **ball** clause. That threshold sits at the noise floor: the clean public-10 worst was 0.00000 (`skillcorner.py` constants and spec 2026-07-14 §4.4). A coordinate-system break shows in the **player** clause instead: a catastrophic sign/origin break measured 0.34139, 68× its threshold. SkillCorner derives its dynamic events from its own tracking. Whether a ball-track defect reaches the event coordinates cannot be established from the tree; the Task 0 measurement below answers exactly that. `findings.md` also gives the attribution in aggregate, not per match, and Task 0 records the clause per match.

**Resolution.**
- Tracking consumers keep S1 exactly as today, now as a persisted, named exclusion (§4.3).
- Events-only consumers admit an S1-excluded match **iff its events pass an event-side check** built for this purpose, and record every such match (§4.4 provenance, §7). S1 rejected the match's tracking. The events need their own positive evidence. CLAUDE.md's ADR-059 bullet (line 124) states the rule verbatim: *"The durable rule: "no counter-evidence" is not evidence — when a FILTER precedes a universal claim, ask what the filter removed."* (The same bullet also says the detector "must require DISCRIMINATING evidence, and DEFER when it has none".)
- **The mechanism is built unconditionally; the measurement only fills its data.** Round 2 made "build an admission check" a branch taken on a bad measurement. Now the check exists regardless (§4.1), and the Task 0 verdict artifact is its data. Whatever the 14 turn out to be, no code path is added or calibrated after the data is seen.

**Rejected: apply the tracking verdict to events.**
- It uses the wrong criterion, applying a tracking verdict to a different artifact.
- Deciding it needs a full tracking parse per match, the cost this cycle removes.
- *Correction of round 1:* a verdict memoized on artifact content hash plus gate version would be reproducible, with the cache affecting only cost. Reproducibility was not the problem.

**The event-side check, pre-registered (round 3, CDLS-SPEC-16).** Round 2's rule compared ~112 per-cell statistics against two-sided `[p1, p99]` bands. With sound events it would have taken the "otherwise" branch about 90% of the time by chance. It also named tackles and interceptions, which the SkillCorner converter never emits (verified: `silly_kicks/spadl/skillcorner.py` emits shot, cross, pass, clearance, dribble, foul, goalkick, corner, throw-in and free-kick types). Replaced by:

- **Anchored actions.** These are the SkillCorner SPADL types whose half is fixed by the laws of the game, read in action-LTR (the acting team attacks x = 105):
  - own-half-anchored `goalkick` and `clearance` are **consistent** iff `start_x < 52.5`. SkillCorner's goal-kick origin is the broadcast ball detection 14–20 m downfield (CLAUDE.md), still well inside the own half.
  - attacking-half-anchored `shot`, `corner_short` and `corner_crossed` are **consistent** iff `start_x > 52.5`.
- **Cell** = one (team, period). `n` = anchored actions in the cell, `k` = consistent ones. A cell is **scored** iff `n ≥ 5`. The empty-cell rule: an unscored cell contributes no evidence either way.
- **Reference rates** `p_t` = the pooled consistency of each anchored type `t` over every scored cell of the S1-passing matches, measured in the same run. A cell's null distribution for `k` is the exact Poisson-binomial over its actions' type rates, computed with the public `silly_kicks.match_outcome.goal_count_pmf` DP.
- **Reversal test** (one-sided; the mirror signature). A scored cell is `reversed` iff **both**:
  - (a) `k / n < 0.5` — the majority of the cell sits on the wrong side;
  - (b) `P_null(K ≤ k) < 0.05 / m`, where `m` = the total number of scored cells across the S1-excluded matches (Bonferroni).
  
  A mirror drives `k / n` toward `1 − p_t`, far below 0.5, so both conjuncts fire decisively. Conjunct (a) guards against overdispersion making the binomial-type null anti-conservative.
- **Boundary pile-up test** (an origin or scale break hidden by the events converter's on-pitch clamp). `b` = the fraction of a match's open-play action starts (every type except goalkick, corners, throw-in and free-kicks) lying within 0.01 m of `x ∈ {0, 105}` or `y ∈ {0, 68}`. The match is flagged iff `b > τ`, where `τ = max(0.05, 3 × q₀.₉₉₉(b over S1-passing matches))`. The clamp turns off-pitch coordinates into boundary mass, and a sound match has almost none.
- **Per-match verdict**, in precedence order:
  1. `reversed` — any reversed cell;
  2. `boundary_pileup`;
  3. `insufficient` — any regulation (team, period) cell, periods 1–2, unscored;
  4. `sound`.
  
  Unscored extra-time cells are allowed and reported. A mirror confined to extra time is a stated residual.
- **False-alarm rate, stated.**
  - Reversal: family-wise ≤ 0.05 across the S1-excluded matches under the Poisson-binomial null (Bonferroni). Conjunct (a) makes it smaller in practice.
  - Boundary: expected false alarms ≤ 14 × 0.001 ≈ 0.014, by construction of `τ` (at least 3× the passing 99.9th percentile).
  - **Empirical calibration, reported:** the identical per-cell threshold (`0.05 / m`) and `τ` are applied to every S1-passing match, and the flagged count is reported. A flagged S1-passing match would be a separate finding (events defective in a tracking-validated match) that also affects tracking consumers, so it is surfaced to the owner rather than decided here.
  - `insufficient` is missing evidence, not a false alarm. Its rate on S1-passing matches is reported too.
- **Clause per match.** The driver records each S1-excluded match's measured player and ball off-pitch rates, so the clause that fired is known per match rather than in aggregate. It is reported as context; the verdict does not depend on it.

**Task 0 — the registered driver** (`scripts/build_skillcorner_s1_event_validity.py`; plan Task 0, before any migration lands). It runs on the DGX over every owner SkillCorner match (full corpus, never subset), with ADR-052 shards, ADR-037 provenance and an ADR-056 input contract. It runs **two `for_each` passes over the same ref list** (round 4, CDLS-SPEC-22):
1. **Events pass** (a bare `load_match(ref, events_only=True)` — the producer is the one consumer Rule D allows to bypass admission): the event statistics for every match. This includes matches whose tracking will not load, since their events still do.
2. **Tracking pass** (`load_match(ref, events_only=False)`): S1's player and ball off-pitch rates for every match. An S1 exclusion is a persisted marker carrying the rates.

**The producer never reads its own artifact** (CDLS-SPEC-27). It does not consult `verdicts.json` in either pass or in the reduce. That avoids three failures:
- a first run cannot be blocked by a missing artifact;
- a re-run after new uploads measures the new matches instead of refusing them;
- a previously non-`sound` match is re-measured.

Verdicts are *recomputed* in every reduce from the per-match statistic shards and that reduce's `p_t` / `τ` / `m`, so no verdict freezes. The statistic shards are deterministic per match, and the verdict is a function of the whole corpus.

A load failure in either pass is a recorded failure with no shard, so a resume retries it. That is why the passes are separate: a transient tracking error must not become a permanent verdict.

**The reduce refuses while either pass has outstanding failures** (CDLS-SPEC-28), naming the failed keys and the pass. A commit-bound `verdicts.json` must not freeze a transient failure into a status. Otherwise `tracking_unloadable` would event-gate a match whose tracking is fine — excluding it if its event verdict is `insufficient` — and `events_unloadable` would never be admitted. `--allow-failed` writes the artifact anyway and records each such match's status with `allowed_failed: true` and its error — the same idiom as §4.4.

The reduce assigns every listed match one **status**:
- `s1_passed`;
- `s1_excluded` (tracking marker);
- `tracking_unloadable` (tracking-pass failure, reachable only under `--allow-failed`; the recorded error is kept);
- `events_unloadable` (events-pass failure, reachable only under `--allow-failed`).

Every match whose events loaded also gets an **event verdict**.

It writes `docs/research/skillcorner_s1_event_validity/verdicts.json` with provenance. The file holds:
- the **full measured listing** — every key with its status, event verdict, reason, per-cell statistics, and S1 rates where measured;
- the listing's digest;
- `p_t`, `τ` and `m`;
- the S1-passing calibration counts.

A human-readable `findings.md` accompanies it.

**Decision, fixed now; the outcome is data.** Events-only SkillCorner admission follows §4.1:
- `s1_passed` is admitted.
- `s1_excluded` and `tracking_unloadable` are admitted iff their event verdict is `sound`.
- `events_unloadable` is never admitted.
- Any other outcome raises `MatchExcluded` with the recorded reason.
- An **unmeasured** requested match refuses the pass by name unless `--allow-unmeasured` admits and records it.

Consequences:
- If all 14 are `sound`, the events-only corpus includes all of them, and every artifact names them.
- If some are not, they are excluded from events-only passes too, by name. No threshold is fitted to the observed separation.
- **Staleness refuses, it does not slip through.** A SkillCorner match uploaded after the Task 0 run is unmeasured, so an events-only pass that requests it stops and names it until the driver is re-run (resumable, cheap on a warm cache) or the owner passes `--allow-unmeasured`, which is recorded. Every events-only fit also records `admission_digest` (§4.4).

Considered and rejected: running the event-side check inline on every events-only load instead of reading an artifact. It would apply an events-only admission to S1-passing matches that tracking loads admit, splitting the two paths' corpora without any measured defect in those matches. Its `insufficient` verdict would also exclude sound S1-passing matches.

## 9. Commit and review plan

One feature branch, **two commits** — code, then the measurement artifact — each behind an explicit approval gate. Ratified at plan review 2026-09-23 (was "one commit"): §8's clean-tree, provenance-stamped Task 0 measurement cannot run from an uncommitted driver, so it lands after the code commit; the established clean-tree two-phase pattern (TF-53/57/61/63). Both commits are fully-tested; neither is a micro-commit.
1. This spec (round 6).
2. Independent `/re-review`.
3. Plan.
4. `/review-plan`.
5. TDD implementation.
6. Whole suite green.
7. Independent `/review-impl`.
8. **Commit 1** (explicit approval): all code — seam, loaders, admission, xT counts, the Task 0 driver, every migration, the gate, docs. Events-only SkillCorner passes refuse at this commit (no artifact yet); tests use fixtures.
9. **Owner-run Task 0** on the DGX from the clean commit-1 SHA (it gates D-1's confirmation).
10. **Commit 2** (explicit approval): `docs/research/skillcorner_s1_event_validity/verdicts.json` + `findings.md`, plus any CHANGELOG/TODO lines citing its numbers.

**TF-56 follow-on:** `feat/tf56-positioning` merges `main` after this lands, and its stashed driver work is rebuilt on the seam:
- its fit-only step becomes the §4.4 count pass;
- its `--reduce-only` census uses `CorpusPassResult.exclusions`, markers and `XtFitProvenance` instead of its own loader-excluded inference;
- its fit follows the §4.4 default: refuse on failures unless `--allow-failed`, which records them.

## 10. Round-2 revision map

| Finding | Change |
|---|---|
| CDLS-SPEC-01 (BLOCKING) | §4.4 rewritten: xT fits are `for_each` count passes reduced by ADR-102 `fit_from_counts`, byte-identical to pooled fits (`array_equal`, §6); failures and exclusions named in `XtFitProvenance`. Adds `ExpectedThreat.zone_counts` / `XtZoneCounts`, single-sourced with `fit()`. (Round 3 makes the failure policy uniform: CDLS-SPEC-17.) |
| CDLS-SPEC-02 | §5 Rule C defined over five shapes (a)–(e) with a plant each; §4.6 states what is closed and the residual, with a residual plant. (Redefined in round 3 over loading loops: CDLS-SPEC-15.) |
| CDLS-SPEC-03 | §4.1 scopes "already done" to idsse / SkillCorner / GS; the two StatsBomb tests are inverted; StatsBomb events-only is new work. |
| CDLS-SPEC-04 | §4.5 gives `measure_rc4_orientation` per-match refs (generation move, free given `run_commit` in its token); §3 row and §7 updated. |
| CDLS-SPEC-05 | §4.6 quotes the ADR-052 text verbatim. |
| CDLS-SPEC-06 | §3 `_loader_pining_to_cache` row: `--out` is its cache dir; defaults to the env var. |
| CDLS-SPEC-07 | §3 lists the five cohort-cache drivers as unaffected. |
| CDLS-SPEC-08 | §2.1 and §3 note: post-parse for all three native builders; stale docstring corrected in-cycle. |
| CDLS-SPEC-09 | Generation move moved to §7 with the other exception; §1 names both. |
| CDLS-SPEC-10 | §4.3 determinism rule; §4.5 GI availability resolved at ref-listing, never a marker. |
| CDLS-SPEC-11 | One env var, `SILLY_KICKS_CORPUS_CACHE_DIR`, for pining and open data; unset behaviour stated for both. |
| CDLS-SPEC-12 | `ItemExcluded` declared in the new leaf module `scripts/_item_outcome.py`; the loader does not import `_driver`. |
| CDLS-SPEC-13 | The test-file sweep list is derived in the plan, not carried as a number. |
| CDLS-SPEC-14 | §7 and §8 state that fit and scored corpora differ within M-shape drivers, and that both sets are recorded. |
| Owner-directed | §8: D-1 resolved by the established per-artifact convention (gk_decision precedent), confirmed by a pre-registered Task 0 measurement with a fixed decision rule; round-1 rejection rationale for (b) corrected. |

### Round-3 revision map

| Finding | Change |
|---|---|
| CDLS-SPEC-15 | §5 Rule C redefined over *loading* loops only: (i) iterating a load, including a flow-sensitively resolved Name; (ii) a per-item load in the body or `map`/`filter` function. Iterating ref/id lists for metadata is allowed. Exemptions are function-granular (`_driver.for_each`, the stream wrappers, the databricks batch loader, `_load_fold`), asserted exact. The five round-2 sites are named green in a live-population test. `_loader_pining_to_cache` becomes a `for_each` pass and the parity preflight a direct `load_match` (§4.5). A scratch emulation at `4ac26d0` flags 23 functions, all migration sites plus `_load_fold`, and none of the green sites. |
| CDLS-SPEC-16 | §8 check redesigned: anchored SkillCorner types that exist (goalkick/clearance vs shot/corners); per (team, period) cell with `n ≥ 5` and an explicit empty-cell rule; one-sided reversal test = majority wrong side AND exact Poisson-binomial `p < 0.05/m` (Bonferroni, family-wise ≤ 0.05); boundary pile-up test with `τ ≥ 3×` the passing 99.9th percentile (≤ 0.014 expected); empirical calibration on S1-passing matches reported. The mechanism is built unconditionally and the verdict artifact is data, so no code path is fitted to the observed separation. |
| CDLS-SPEC-17 | §4.4: failed fits refuse by default for every caller; `--allow-failed` permits and records (the `--allow-dirty` idiom). |
| CDLS-SPEC-18 | §3 row: `--out` (required) is the materialized cache; raw artifacts are not cached today. The env var is post-migration and applies to the raw-artifact cache only. |
| CDLS-SPEC-19 | §4.1: the stale "dormant on the kloppy path" S1 comment is corrected alongside the docstring. |
| CDLS-SPEC-20 | §4.4 and §4.7: ADR-102 is amended (`zone_counts`, shared count extractors). §5.2: no module-level scan exemptions; `_xt_corpus.py` and `_driver.py` are scanned, and only named functions are exempt. |
| CDLS-SPEC-21 | §4.4 and §6: `np.array_equal(..., equal_nan=True)`. |
| Found while revising | `validate_territorial_defense --list-matches` builds every SB360 match to print ids; `--list-matches` paths switch to ref listers (§4.5). My round-2 statistic named tackles/interceptions, which SkillCorner never emits; corrected (§8). |

### Round-4 revision map

| Finding | Change |
|---|---|
| CDLS-SPEC-22 | §4.1 and §8: `verdicts.json` records the full measured listing with a per-match status (`s1_passed` / `s1_excluded` / `tracking_unloadable` / `events_unloadable`) plus an event verdict. Task 0 runs two `for_each` passes (events, tracking), so tracking-unloadable matches still get an event check and a transient tracking error retries instead of freezing a status. The events-only preflight refuses unmeasured requested refs by name. `--allow-unmeasured` admits and records them (`XtFitProvenance.unmeasured_admitted`). Refusal rather than exclusion, so the corpus never shrinks silently. §6 tests both sides of each status. |
| CDLS-SPEC-23 | **Not applied — the claim is incorrect.** CLAUDE.md line 124 (ADR-059 bullet) contains, verbatim: *"The durable rule: "no counter-evidence" is not evidence — when a FILTER precedes a universal claim, ask what the filter removed."* It is also present at `4ac26d0`. §8 now quotes it exactly with its location, alongside the "DISCRIMINATING evidence, and DEFER" line the reviewer suggested. |
| CDLS-SPEC-24 | §5: `_loader_databricks.load_matches` removed from `_UNSHARDED_LOOP_EXEMPT`. Its loops call `_convert`, not a loader, so it is not a loading loop. |
| CDLS-SPEC-25 | §5: the green-site live-population test runs on the migrated tree. `_loader_pining_to_cache.main` is flagged at `4ac26d0` and turns green once its load loop moves into `for_each`; the other four are green already. |
| CDLS-SPEC-26 | §6: the count-pass line now says a failing ref does not abort the *pass*, and the reduce refuses or records. §9 step 1 reads "round 4". |

### Round-5 revision map

| Finding | Change |
|---|---|
| CDLS-SPEC-27 | §4.1: admission is an edge policy layer (`scripts/_events_admission.py`: `EventsOnlyAdmission`, `events_only_loader`), and `load_match(events_only=True)` stays a pure loader — per CLAUDE.md's "policy lives at the edge, never in the shared engine". §5 Rule D: a `load_match` call with `events_only` other than literal `False` fails CI outside the admission module and the registered producer (exact both ways, plants both sides). §8: the producer never reads `verdicts.json` and recomputes every verdict in each reduce. §6: Task 0 runs with no artifact, over a stale artifact, and re-measures a previously non-`sound` match; a spy asserts it never opens the artifact. The artifact is required only when a SkillCorner ref is requested. |
| CDLS-SPEC-28 | §8: Task 0's reduce refuses while either pass has outstanding failures, naming keys and pass. `--allow-failed` writes the artifact with `allowed_failed: true` plus the error per affected match. `tracking_unloadable` / `events_unloadable` are reachable only under `--allow-failed`. §6 tests both sides. |

### Round-6 revision map

| Finding | Change |
|---|---|
| CDLS-SPEC-29 | §4.1: `load_matches` / `load_statsbomb_matches` are tracking-only wrappers with no `events_only` keyword; they call `load_match(ref, events_only=False)`. The ported TF-56 keyword, its `Literal` overloads, and the ported tests exercising `load_matches(events_only=...)` are removed. Coverage moves to direct `load_match(events_only=True)` tests and to `events_only_loader`. Frozen-copy parity is unaffected (`4ac26d0` never had the keyword). |
| CDLS-SPEC-30 | §4.4: `_load_xt_corpus_pining` uses the admitted `events_only_loader`, never a bare `load_match(events_only=True)`. |
| CDLS-SPEC-31 | §5 Rule D: function-granular allowlist (`_events_admission.events_only_loader`, `build_skillcorner_s1_event_validity._events_pass`), keyed on the module-level function with closures included. A plant shows a second un-admitted call in another function of `_events_admission.py` goes red. |
