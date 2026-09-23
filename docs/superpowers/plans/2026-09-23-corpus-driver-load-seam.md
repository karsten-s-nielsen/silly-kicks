# Corpus-driver load seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline (no subagents — banned on this project, owner ruling 2026-09-23; do **not** use superpowers:subagent-driven-development). Commit-1 spans ~40 driver migrations and **will not fit one session** (review CONSIDER-06): each task ends at a green, uncommitted checkpoint, so a compaction or a fresh session resumes cleanly from the last green task; only the Task 17 STOP gates on owner approval. Steps use checkbox (`- [ ]`) syntax for tracking.

> **AMENDMENT 2026-09-23 (owner-ratified): source-factory reuse (extract + retrofit all).** The
> S-migration recipe repeats the same 3-line source construction (`resolve_cache_dir` +
> `list_match_refs` + a `load=lambda ref: load_match(ref, events_only=False, tracking_limit=…,
> cache_dir=…)` closure) in every tracking driver, and it was hand-written identically 4× during
> execution (`_pining_source` in `train_xshot_occurrence`/`train_xcross_attempt`, `_corpus_source`
> in `train_receiver_model`, `_pining_source` in `validate_sb360_licensed_corpus`) plus inline in
> ~12 more. That duplication is extracted into two public factories, **`pining_source`** (in
> `scripts/_loader_pining.py`) and **`open_data_source`** (in `scripts/_sb_open_data.py`), each
> returning `(refs, load)` where `load(ref)` yields a full `LoadedMatch` (`events_only=False`). The
> events-only mode already has its factory (`_events_admission.events_only_loader`), so the surface
> is symmetric: one source factory per load mode. **Decision: extract AND retrofit all** already-migrated
> drivers onto the factory, so the whole cycle lands on one shared seam. Generations stay
> byte-identical (same refs, same load semantics → same shard keys/tokens). Both factories are public
> loader-module functions with no `load_`/`list_`/`select_`/`fetch_` prefix, so each needs a
> `_LOADER_NON_CORPUS` entry with a reason (else the §5.1 population gate fails). The factories are
> gate-clean (they call `list_match_refs`/`load_match(events_only=False)` — not a stream loader, not a
> loading loop). Drivers that RESHAPE the item (`(mid, actions, frames)`; the SB360 6-tuple), key by
> `_source_key`/`ref.match_id`, or dual-source (--data-dir) keep a thin per-driver wrapper around the
> factory. See **Task 8.5**. Spec §4.1/§4.5 amended (revision map).

**Goal:** Every `scripts/` corpus driver loads one match at a time behind `for_each`'s resume check, records load failures and deterministic exclusions instead of crashing, loads events-only when it only needs events (with SkillCorner admission at the edge), fits xT as a sharded count pass, and honours one cache variable. A CI gate makes the old shapes unrepresentable.

**Architecture:** A cheap, sized list of match references (`list_match_refs` / `list_open_data_refs`) plus a single-match loader (`load_match` / `load_open_data_match`) replaces streaming loaders in drivers. `for_each` gains a `load=` hook: it checks for a shard or exclusion marker *before* loading, and routes load errors and `ItemExcluded` through its per-item `try`. xT fits become per-match zone-count shards reduced by ADR-102's `ExpectedThreat.fit_from_counts`, using a new additive library method `ExpectedThreat.zone_counts` that shares `fit()`'s counting code. Events-only SkillCorner admission is an edge policy layer (`scripts/_events_admission.py`) fed by a registered measurement driver.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy (tests only), `ruthless.fingerprint` (via `scripts/_driver.py`), pytest, AST (`ast` stdlib) for the CI gate.

**Spec:** `docs/superpowers/specs/2026-09-22-corpus-driver-load-seam-design.md` (round 6, APPROVED — review `D:\Development\_reviews\2026-09-22-corpus-driver-load-seam-spec-r6.md`). Every task argues from it; read the spec section a task cites before starting.

## Global Constraints

- **Branch:** `feat/corpus-driver-load-seam` off `main` @ `4ac26d0`. No worktrees, no other branch.
- **Commits:** NO per-task commits. Each task ends at "tests green, no commit". The owner approves commits explicitly (see "Commit and run plan" below).
- **TDD, always:** failing test first, watch it fail for the right reason, minimal code, watch it pass. No production code without a failing test.
- **No pre-claimed numbers:** no version, `PR-Snnn` or ADR number is written anywhere until commit-prep.
- **Library change is additive only:** the one `silly_kicks/` change is `ExpectedThreat.zone_counts` + `XtZoneCounts` + private count extractors. `ExpectedThreat.fit()` output stays byte-identical (SK-xT-1 oracle `tests/xthreat_legacy_reference.py` and `tests/xthreat/test_fit_from_counts.py` unchanged and green).
- **Shard keys, shard schemas and generation tokens stay byte-identical** for every migrated driver whose per-match output is unchanged. The two spec-named generation moves are `measure_cover_shadow_argmax_agreement` and `measure_rc4_orientation` (spec §7). In addition, an events-only pass that requests SkillCorner refs moves generation, because the admission digest joins its token (interpretation 8).
- **Never shrink a corpus to make a run feasible.** Refusal (`--allow-*` to proceed, recorded) is the idiom; silent exclusion is forbidden.
- **Cache variable:** exactly one, `SILLY_KICKS_CORPUS_CACHE_DIR`. Argument beats env beats unset. Unset = today's behaviour (temp dir per match / fetch every time).
- **`events_only` has no default** on `load_match`; every call site passes it by keyword.
- **Rule D:** a `load_match(...)` call whose `events_only=` is not the literal `False` is allowed only inside `_events_admission.events_only_loader` and `build_skillcorner_s1_event_validity._events_pass` (module-level functions; closures inside them included).
- **Test commands** (from repo root): `.venv/Scripts/python.exe -m pytest <path> -q -p no:randomly --tb=short`. Anything under `tests/tracking/` also needs `--benchmark-skip` (it hangs without it). Never pip-install into `.venv`.
- **Plan code is content-exact, not layout-exact:** every task runs `python -m ruff format` on the files it touched before `ruff check` / `ruff format --check` — the formatter wraps lines the plan's blocks leave long. Anything `ruff check` still reports after formatting is a real finding.
- **Lint/type at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare). Local pyright baseline on this branch's base: re-measure in Task 17 Step 1; the bar is zero new errors.
- **Parserless scripts:** `--help` executes `main()` on 16 parserless `scripts/*.py`. Never run a script with `--help` without first checking `grep -q add_argument <file>`.
- **Id comparisons** in any new code go through `silly_kicks.id_compat` (ADR-019); `astype(str)` on an id column is a defect — use `canonical_id_series`.
- **All `warnings.warn()` calls pass `stacklevel=2`.**

## Commit and run plan (proposed; owner decides at plan review)

**Ratified at plan review 2026-09-23; spec §9 amended to two commits to match.** The measurement driver needs a clean committed tree to run (ADR-037 provenance; §4.1 admission refuses a dirty-tree artifact), so Task 0 lands after the code commit. No migration depends on the measurement's outcome (the admission mechanism is built unconditionally; the measurement only supplies data, spec §8). Following the repo's established two-phase clean-provenance pattern (TF-53/57/61/63):

1. **Commit 1** (explicit approval): all code — seam, loaders, admission layer, xT counts, the Task 0 driver, every driver migration, the CI gate, docs — with the full suite, ruff and pyright green. Events-only SkillCorner passes refuse at this commit (no artifact yet); tests use fixture artifacts.
2. **Owner-run Task 0 on the DGX** from the clean commit-1 SHA (Task 18 runbook).
3. **Commit 2** (explicit approval): `docs/research/skillcorner_s1_event_validity/verdicts.json` + `findings.md` from that run, plus any CHANGELOG/TODO lines that cite its numbers.

(There is no ordering that runs a provenance-stamped measurement from an uncommitted driver, so one commit was never actually reachable — a single commit would still have to run Task 0 afterward as a separate artifact commit.)

## File Structure

**Create**
| File | Responsibility |
|---|---|
| `scripts/_item_outcome.py` | Leaf module: `ItemExcluded`. Imports nothing from `scripts/`. |
| `scripts/_events_admission.py` | Edge policy: `EventsOnlyAdmission`, `AdmissionRecord`, `AdmissionRefusedError`, `events_only_loader`, `ADMISSION_ARTIFACT`. |
| `scripts/_xt_corpus.py` | xT count pass: `counts_to_frame`, `counts_from_frames`, `xt_count_pass`, `fit_xt_from_count_pass`, `XtFitProvenance`. |
| `scripts/build_skillcorner_s1_event_validity.py` | Registered Task 0 measurement driver (spec §8). |
| `tests/scripts/test_item_outcome.py` | Leaf-module import test. |
| `tests/scripts/test_driver_load_hook.py` | `for_each(load=)` + exclusion markers + conservation. |
| `tests/scripts/test_loader_refs.py` | `MatchRef`, `list_match_refs`, `load_match`, wrappers parity, cache env. |
| `tests/scripts/test_sb_open_data_refs.py` | `list_open_data_refs`, `load_open_data_match`, open-data cache, wrapper parity. |
| `tests/scripts/_frozen_loader_4ac26d0.py` | Frozen copy of the `4ac26d0` `load_matches` / `load_statsbomb_matches` / `load_open_data_matches` bodies, for wrapper parity. |
| `tests/xthreat/test_zone_counts.py` | `XtZoneCounts` / `ExpectedThreat.zone_counts`. |
| `tests/scripts/test_xt_corpus.py` | Count pass + reduce + provenance. |
| `tests/scripts/test_events_admission.py` | Admission statuses, preflight, fail-closed load. |
| `tests/scripts/test_build_skillcorner_s1_event_validity.py` | Task 0 statistics, verdicts, two passes, reduce, no-self-deadlock. |

**Modify**
| File | Change |
|---|---|
| `scripts/_driver.py` | `for_each(load=)`, exclusion markers, conservation counts markers, `CorpusPassResult.excluded/exclusions`, `manifest()`/`manifest_fields` gain `n_excluded`. |
| `scripts/_loader_pining.py` | Refs + `load_match` + `LoadedMatch` + `MatchExcluded` + `resolve_cache_dir`; wrappers over refs (drop ported `events_only` keyword + overloads); StatsBomb events-only; docstring/comment fixes. |
| `scripts/_sb_open_data.py` | `OpenDataRef`, `list_open_data_refs`, `load_open_data_match` (+cache); wrapper. |
| `silly_kicks/xthreat/_grid.py`, `_transitions.py`, `_model.py`, `__init__.py` | Shared count extractors; `XtZoneCounts`; `ExpectedThreat.zone_counts`; export. |
| `tests/scripts/test_loader_pining_events_only.py` | Invert the two StatsBomb tests; remove tests of `load_matches(events_only=...)`; retarget onto `load_match`. |
| `tests/scripts/test_corpus_driver_resilience.py` (+ `tests/scripts/_script_population.py` if a shared helper is needed) | Derived corpus-call set, private-script scan, Rules A–D, plants, live-population green sites. |
| `tests/scripts/test_provenance_wiring.py` | Register the Task 0 driver in `ARTIFACT_DRIVERS`. |
| ~40 `scripts/*.py` drivers + their tests | Migrations (Tasks 9–15). |
| `docs/superpowers/adrs/ADR-052-corpus-driver-resilience.md`, `ADR-102-expectedthreat-fit-from-counts.md`, `CLAUDE.md`, `CHANGELOG.md`, `TODO.md` | Task 16. |

## Shared interfaces (every task uses these names and types exactly)

```python
# scripts/_item_outcome.py
class ItemExcluded(Exception):
    """A deterministic admission decision: this item is not in the pass."""
    def __init__(self, reason: str, *, details: Mapping[str, object] | None = None) -> None: ...
    reason: str
    details: dict[str, object]          # JSON-serializable; persisted in the marker

# scripts/_driver.py
def for_each(items, *, key, work, shard_root, token_inputs, token_reason=None, counters=None,
             tag="all", label="item", max_consecutive_failures=3, load=None) -> CorpusPassResult
def exclusion_path(generation, key) -> pathlib.Path          # <gen>/<join_key(key)>.excluded.json
def already_excluded(generation, key) -> bool
def write_exclusion(generation, key, exc: ItemExcluded, *, tag: str) -> None   # atomic, {"reason":..., "details":{...}}
def read_exclusion(generation, key) -> dict | None           # the marker payload, or None
@dataclasses.dataclass(frozen=True)
class CorpusPassResult:              # existing fields unchanged; two appended with defaults
    ...; excluded: int = 0; exclusions: dict = dataclasses.field(default_factory=dict)   # joined key -> reason
    @property
    def shard_keys(self) -> tuple[str, ...]: ...   # keys minus failures minus exclusions, pass order: every one has a parquet
def manifest_fields(generation, *, attempted, failed, counters_unrecorded, excluded) -> dict   # adds "n_excluded"; no default for excluded

# scripts/_loader_pining.py
CORPUS_CACHE_ENV = "SILLY_KICKS_CORPUS_CACHE_DIR"
def resolve_cache_dir(cache_dir: str | Path | None) -> Path | None
@dataclass(frozen=True)
class MatchRef:
    provider: str
    match_id: str
    artifacts: Mapping[str, str] = field(compare=False, hash=False, repr=False)
    @property
    def key(self) -> tuple[str, str]: return (self.provider, self.match_id)
def list_match_refs(*, providers, match_ids=None, max_per_provider=None, token=None, base_url=None) -> list[MatchRef]
class LoadedMatch(NamedTuple):
    provider: str
    match_id: str
    actions: pd.DataFrame
    frames: pd.DataFrame | None          # None iff events_only (pining); empty DataFrame for open data
    home_team_id: object
    visible_area: pd.DataFrame | None    # SB360 only, and only when not events_only
    report: object | None                # tracking TrackingConversionReport (SkillCorner full loads), else None
class MatchExcluded(ItemExcluded): ...
def load_match(ref: MatchRef, *, events_only: bool, tracking_limit=None, cache_dir=None,
               token=None, base_url=None) -> LoadedMatch
# load_matches / load_statsbomb_matches: 4ac26d0 signatures and outputs, tracking-only, no events_only keyword.

# scripts/_sb_open_data.py
@dataclass(frozen=True)
class OpenDataRef:
    competition_id: int
    season_id: int
    match_id: str
    home_team_id: int
    @property
    def key(self) -> tuple[str, str]: return ("statsbomb", self.match_id)
def list_open_data_refs(competitions, *, match_ids=None, max_matches=None) -> list[OpenDataRef]
def load_open_data_match(ref: OpenDataRef, *, preserve_native=(), cache_dir=None) -> LoadedMatch
# load_open_data_matches: 4ac26d0 signature and output.

# silly_kicks/xthreat
@dataclasses.dataclass(frozen=True, eq=False)
class XtZoneCounts:
    l: int; w: int
    shot_counts: np.ndarray; goal_counts: np.ndarray; move_counts: np.ndarray        # (w, l) int64
    transition_start_counts: np.ndarray                                              # (w, l) int64
    transition_counts: np.ndarray                                                    # (w*l, w*l) int64
    def __add__(self, other: "XtZoneCounts") -> "XtZoneCounts": ...                  # ValueError on (l, w) mismatch
    @classmethod
    def zeros(cls, l: int, w: int) -> "XtZoneCounts": ...
    def as_fit_kwargs(self) -> dict[str, np.ndarray]: ...
class ExpectedThreat:
    def zone_counts(self, actions: pd.DataFrame) -> XtZoneCounts: ...

# scripts/_events_admission.py
ADMISSION_ARTIFACT: Path             # <repo>/docs/research/skillcorner_s1_event_validity/verdicts.json
STATUSES = ("s1_passed", "s1_excluded", "tracking_unloadable", "events_unloadable")
@dataclass(frozen=True)
class AdmissionRecord:
    digest: str | None                   # None when no SkillCorner ref was requested
    unmeasured_admitted: tuple[str, ...] # join_key strings
class EventsOnlyAdmission:
    digest: str
    @classmethod
    def load(cls, path: Path = ADMISSION_ARTIFACT) -> "EventsOnlyAdmission": ...   # fail-closed
    def preflight(self, refs, *, allow_unmeasured: bool) -> AdmissionRecord: ...
    def check(self, ref: MatchRef) -> None: ...      # raises MatchExcluded(<status, verdict, reason>)
def events_only_loader(refs, *, allow_unmeasured=False, cache_dir=None, artifact_path: Path = ADMISSION_ARTIFACT
                       ) -> tuple[Callable[[MatchRef], LoadedMatch], AdmissionRecord]: ...

# scripts/_xt_corpus.py
COUNT_SHARD_COLUMNS = ["aggregate", "from_zone", "to_zone", "n"]
COUNT_SHARD_SCHEMA_VERSION = "xt-counts-1"
def counts_to_frame(counts: XtZoneCounts) -> pd.DataFrame
def counts_from_frames(frames: Iterable[pd.DataFrame], *, l: int, w: int) -> XtZoneCounts
@dataclass(frozen=True)
class XtFitProvenance:
    fit_keys: tuple[str, ...]; excluded: Mapping[str, str]; failed: Mapping[str, str]
    l: int; w: int; counts_digest: str
    admission_digest: str | None; allowed_failed: bool; unmeasured_admitted: tuple[str, ...]
def xt_count_pass(refs, *, key, load_actions, shard_root, token_inputs, tag="all", l=16, w=12,
                  label="match") -> CorpusPassResult
def fit_xt_from_count_pass(res: CorpusPassResult, *, l=16, w=12, allow_failed=False,
                           admission: AdmissionRecord | None = None) -> tuple[ExpectedThreat, XtFitProvenance]

# scripts/build_skillcorner_s1_event_validity.py  (module-level functions; Rule D keys on _events_pass)
def _events_pass(refs, *, shard_root, cache_dir, token_inputs, tag="all") -> CorpusPassResult
def _tracking_pass(refs, *, shard_root, cache_dir, token_inputs, tag="all") -> CorpusPassResult

# tests/scripts/_fake_corpus.py  (Task 2 creates; every later task's tests consume it)
def make_ref(provider: str, match_id: str, artifacts: Mapping[str, str] | None = None) -> MatchRef
def make_loaded(provider: str, match_id: str, *, actions=None, frames=None, home_team_id="H",
                visible_area=None, report=None) -> LoadedMatch
class SpyLoader:
    """Stand-in for load_match: records (ref.key, kwargs) per call; serves LoadedMatch per key."""
    def __init__(self, matches: Mapping[tuple[str, str], LoadedMatch], *,
                 fail: Collection[tuple[str, str]] = (), exclude: Mapping[tuple[str, str], str] | None = None) -> None: ...
    calls: list[tuple[tuple[str, str], dict]]
    def __call__(self, ref: MatchRef, **kwargs) -> LoadedMatch: ...   # fail -> RuntimeError; exclude -> MatchExcluded
def install_fake_corpus(monkeypatch, module, *, refs: list[MatchRef], loader: SpyLoader) -> None
    # patches module.list_match_refs (returns refs, ignoring filters) and module.load_match (the spy)
```

## Plan-level interpretations of the spec (for the plan reviewer)

Each item below refines an interface the spec states more loosely. None changes spec behaviour; each is flagged so the reviewer can reject it.

1. **`ItemExcluded(reason, *, details=None)`.** The spec's constructor takes `reason` only. Task 0 must read each S1-excluded match's measured player/ball rates back out of its marker (§8 "S1 rates where measured"). Parsing floats out of a free-text reason is fragile, so the marker carries a structured `details` map beside `reason`. `reason` stays the human-readable string the spec requires.
2. **`LoadedMatch.report`.** The spec's `LoadedMatch` has no `report`. Task 0's tracking pass must record S1 rates for S1-*passing* matches too, and those rates live on the SkillCorner `TrackingConversionReport` (`player_off_pitch_rate`, `ball_off_pitch_rate`), computed by the native builder on pre-`_preprocess` frames. Recomputing them from the returned, preprocessed frames could differ. So `LoadedMatch` carries the report (`None` for every non-SkillCorner or events-only load).
3. **`n_excluded` is always present in the manifest.** §4.3 says `manifest()` gains `n_excluded`; §6 says no-`load` passes keep byte-identical manifests. A field that appears only on some passes is a Hyrum trap and breaks `aggregate_manifests`'s uniform-schema assumption across mixed workers. So every manifest carries `n_excluded`. The no-`load` identity test asserts every pre-existing field byte-identical, plus `n_excluded == 0`. `manifest_fields` gains a required `excluded` keyword with no default (the `counters_unrecorded` rule). No script calls `manifest_fields` directly (measured: only `CorpusPassResult.manifest()` and `tests/scripts/test_driver.py`).
4. **Exclusion markers work with or without `load`.** §4.3 step 4 names `ItemExcluded` "from `load` or `work`". No existing `work` raises it, so a no-`load` pass is unchanged.
5. **`_STREAM_LOADER_EXEMPT` is function-granular** (`calibrate_tracking_defaults._load_fold`), not module-granular. The same module also streams in `_load_xt_corpus_pining._work` (line 237 at `4ac26d0`), which must migrate; a module-level exemption would hide it.
6. **The gate lands RED before the migrations** (Task 8 before Tasks 9–15). CLAUDE.md (ADR-051): "a gate written after its own repair arrives green and is never observed failing". Task 8 records the exact expected violation list; each migration task removes its own entries; Task 17 asserts the list is empty.
7. **Two commits, not one** — see "Commit and run plan" above.
8. **An events-only pass that requests SkillCorner refs starts a new generation.** §4.1 puts the admission artifact's digest into `token_inputs`; §7 lists only two generation moves. The two statements collide for `calibrate_xt_bandwidth` and `calibrate_tracking_defaults._load_xt_corpus_pining` when SkillCorner is requested. **This is a THIRD generation-move class beyond the original spec §7 two; spec §7 amended to three, ratified at plan review 2026-09-23.** The move is *required* for correctness: the fit's input corpus now depends on the admission artifact, which admits the `sound` S1-excluded SkillCorner matches the pre-migration streamed path silently `continue`d past inside `load_matches` (so no shard existed for them, not an empty one — review CONSIDER-03). Reusing the old generation would fit from a different corpus than the token claims. So the digest joins the token only when it is not `None` (a pass with no SkillCorner ref keeps its token byte-identical), and a SkillCorner-including events-only pass moves generation. The new count passes (`build_tf60 --xt-out`, `measure_cover_shadow`, `_xtgk_comparability`) had no shards before, so nothing moves there.
9. **`n_excluded` is summed but carries no commit-consistency vote.** `_partition.aggregate_manifests` removes a manifest's vote on commit consistency only when it positively declares it built nothing. `n_excluded` is replayed on resume (corpus-scoped, §4.3), so without this rule a fully resumed worker reports `n_excluded > 0`, regains its vote, and re-arms the false alarm `test_a_FULL_RESUME_pass_does_not_re_arm_the_commit_false_alarm` exists to prevent. Task 1 therefore changes `aggregate_manifests` (red test first) so it still sums `n_excluded` but excludes it from the vote.
10. **Exclusion counting.** A *fresh* exclusion counts in `attempted` and `excluded`. A *replayed* one (marker present on resume) counts in `excluded` only — never in `skipped` or `attempted`. The reason: `res.skipped` must keep meaning "a finished shard from a prior pass" for its consumers (it is carried into returned tuples and driver "processed" arithmetic — e.g. `train_xcross_attempt`); folding replayed exclusions into `skipped` would silently inflate it. Any "built" arithmetic in a driver is `attempted - failed - <fresh exclusions>`. (The exact per-consumer effect is verified per migrating task, not asserted here — review CONSIDER-04.)
11. **`CorpusPassResult.shard_keys`.** `res.keys` includes excluded keys, which have a marker and no parquet, so any combiner that reads `shard_path(gen, k)` for each key raises `FileNotFoundError` on the first S1 exclusion. One seam property fixes it at every combine site: combiners and the xT reduce iterate `res.shard_keys` (keys minus failures minus exclusions). Which specific drivers combine by per-key `shard_path` is enumerated and fixed per migrating task, not asserted here (review CONSIDER-05).
12. **Exclusions reach every artifact.** Drivers with a manifest get `n_excluded` from `res.manifest()`. Drivers that write no manifest (e.g. the two trainers, `validate_xcross_causal`, `validate_xs_probe`) add `n_excluded` and `excluded` (`{key: reason}`) to their metrics artifact. The spec's rule is "exclusions are reported"; today those matches vanish silently. Additive keys only.

## Derived populations at `4ac26d0` (re-derived by the committed gate in Task 8)

**Rule C loading loops (23 functions; scratch AST emulation of spec §5 Rule C, keyed on the innermost enclosing function):** `_loader_pining_to_cache.main`, `_xtgk_comparability._collect`, `_xtgk_comparability.main`, `build_territory_ranking_census._corpus_matches`, `build_territory_ranking_census.main`, `build_tf60_layer3_arm_values.main`, `calibrate_tracking_defaults._load_fold` (exempt), `calibrate_tracking_defaults._load_xt_corpus_pining._work`, `calibrate_xt_bandwidth._load_one_match`, `evolve_xsuccess_features.main`, `materialize_tc3_frames.preflight_reference_parity`, `measure_cover_shadow_argmax_agreement.main`, `train_match_outcome_dependence.main`, `train_pass_completion.main._all_matches`, `train_receiver_model._load_corpus`, `train_win_probability.main._matches`, `train_xsuccess.main`, `validate_gk_decision._reconstruction_verdicts`, `validate_match_outcome_calibration._iter_open_matches`, `validate_sb360_licensed_corpus._pining_items`, `validate_team_kpi_reliability._load_statsbomb_open_matches`, `validate_territorial_defense.main`, `validate_territory_counterfactual._corpus_matches`. None of the five §5 green sites is flagged.

**Rule A stream-loader call sites (37 driver modules):** `_loader_pining_to_cache`, `_xtgk_comparability`, `build_gkdv_arm_values`, `build_layer2_spells`, `build_rq_pass_scores`, `build_territory_ranking_census`, `build_tf19_instrument_responsiveness`, `build_tf60_layer3_arm_values`, `calibrate_tracking_defaults` (`_load_fold` exempt; `_load_xt_corpus_pining._work` migrates), `calibrate_xt_bandwidth`, `derive_opengoal_range`, `evolve_xsuccess_features`, `materialize_tc3_frames`, `measure_cover_shadow_argmax_agreement`, `measure_gs_shot_distribution`, `measure_rc4_orientation`, `run_signoff_power`, `train_gk_completion`, `train_match_outcome_dependence`, `train_pass_completion`, `train_receiver_model`, `train_win_probability`, `train_xcross_attempt`, `train_xshot_occurrence`, `train_xsuccess`, `tune_structural_pass_sigma`, `validate_gk_decision`, `validate_match_outcome_calibration`, `validate_sb360_licensed_corpus`, `validate_shot_goalmouth_sb`, `validate_skillcorner_keeper_origin`, `validate_team_kpi_reliability`, `validate_territorial_defense`, `validate_territory_counterfactual`, `validate_xcross_causal`, `validate_xs_probe`, `validate_xshot_causal`. (`build_sb360_coverage` walks statsbombpy directly; Task 11 migrates it onto refs + `load=` although no Rule A/C site names it.)

**Test files that monkeypatch or reassign a loader (sweep list, 20 files):** `tests/calibration/test_calibrate_cli.py`, `tests/calibration/test_calibrate_xt_bandwidth_cli.py`, `tests/calibration/test_loader_pining.py`, `tests/causal/test_causal_e2e.py`, `tests/scripts/test_build_layer2_spells.py`, `tests/scripts/test_build_rq_pass_scores.py`, `tests/scripts/test_build_territory_ranking_census.py`, `tests/scripts/test_calibrate_load_fold_budget.py`, `tests/scripts/test_driver_resume_oracle.py`, `tests/scripts/test_loader_artifacts.py`, `tests/scripts/test_loader_pining_cache_skip.py`, `tests/scripts/test_loader_statsbomb.py`, `tests/scripts/test_match_outcome_train.py`, `tests/scripts/test_measure_rc4_orientation.py`, `tests/scripts/test_tf54b_drivers.py`, `tests/scripts/test_train_pass_completion.py`, `tests/scripts/test_train_receiver_model.py`, `tests/scripts/test_validate_shot_goalmouth_sb_shards.py`, `tests/scripts/test_validate_xs_probe.py`, `tests/scripts/test_xtgk_comparability_cache.py`. Tests that only *call* the wrappers directly (the e2e files, `test_loader_orientation.py`, `test_loader_skillcorner_native.py`, `test_sb_open_data.py`, `test_scale_guards.py`, …) keep working unchanged, because the wrappers keep their signatures.

## Tasks

| # | Task | Spec |
|---|---|---|
| 1 | Seam: `_item_outcome.ItemExcluded` + `for_each(load=)` + exclusion markers | §4.3 |
| 2 | Pining loader: refs, `load_match`, `MatchExcluded`, cache env, wrappers, StatsBomb events-only, `_fake_corpus` test helper | §4.1 |
| 3 | Open-data loader: refs, single-match load, cache, wrapper | §4.2 |
| 4 | Library: `XtZoneCounts` + `ExpectedThreat.zone_counts` (shared extractors) | §4.4 |
| 5 | `scripts/_xt_corpus.py` count pass + reduce | §4.4 |
| 6 | `scripts/_events_admission.py` admission layer | §4.1, §8 |
| 7 | Task 0 driver `build_skillcorner_s1_event_validity.py` | §8 |
| 8 | CI gate: derived population, private scan, Rules A–D, plants — lands RED | §5 |
| 9 | Migrate S-shape pining drivers, batch A (9 drivers) | §4.5 |
| 10 | Migrate S-shape pining drivers, batch B (8 drivers) | §4.5 |
| 11 | Migrate StatsBomb/SB360 drivers (`train_receiver_model`, `validate_sb360_licensed_corpus`, `validate_territorial_defense`, `validate_gk_decision`, `build_sb360_coverage`) | §4.5, §4.6 |
| 12 | Migrate open-data S-shape drivers (6) | §4.2, §4.5 |
| 13 | Migrate open-data U-shape drivers (`build_territory_ranking_census`, `validate_match_outcome_calibration`, `train_match_outcome_dependence`) | §4.5 |
| 14 | Migrate xT-fit M/E/U drivers (`build_tf60_layer3_arm_values`, `measure_cover_shadow_argmax_agreement`, `_xtgk_comparability`) | §4.4, §4.5, §7 |
| 15 | Migrate remaining drivers (`calibrate_xt_bandwidth`, `calibrate_tracking_defaults`, `measure_rc4_orientation`, `train_ghost_outfield`, `_loader_pining_to_cache`) | §4.5, §7 |
| 16 | Docs: ADR-052 amendment, ADR-102 amendment, CLAUDE.md, CHANGELOG, TODO | §4.7, §7 |
| 17 | Whole-branch verification and commit-1 stop | §6, §9 |
| 18 | Owner-run Task 0 on the DGX + commit-2 stop | §8, §9 |

> Each task ends **tests green, no commit**. Where a step shows code, it is the *non-obvious* part (a new signature, the one load-bearing assertion); obvious bodies are described, not pasted. Run the task-scoped tests, then `ruff format` → `ruff check` → `ruff format --check` → `pyright` on the touched files.

### Task 1: Seam — `_item_outcome.ItemExcluded` + `for_each(load=)` + exclusion markers

**Spec:** §4.3. **Interpretations:** 3, 4, 9, 10.

**Files:**
- Create `scripts/_item_outcome.py`, `tests/scripts/test_item_outcome.py`, `tests/scripts/test_driver_load_hook.py`.
- Modify `scripts/_driver.py` (`for_each`, `exclusion_path`/`already_excluded`/`write_exclusion`/`read_exclusion`, `CorpusPassResult`, `manifest_fields`, `assert_conservation`), `scripts/_partition.py` (`aggregate_manifests` — interpretation 9), `tests/scripts/test_driver.py` and `tests/scripts/test_driver_resume_oracle.py` (no-`load` byte-identity + `n_excluded`).

**Interfaces:**
- Produces: `ItemExcluded`, `for_each(..., load=None)`, `CorpusPassResult.excluded`/`.exclusions`/`.shard_keys`, `manifest_fields(..., excluded)`, the marker helpers, `n_excluded` in manifests (all in Shared interfaces).
- Consumes: nothing new.

- [ ] **Step 1 — `_item_outcome` leaf.** Write `tests/scripts/test_item_outcome.py`: (a) `ItemExcluded("r", details={"x": 1}).details == {"x": 1}` and a non-JSON-serializable `details` raises at construction; (b) `import ast; assert no "scripts" import` — parse `_item_outcome.py`, assert no `Import`/`ImportFrom` whose root module is `scripts` or a bare `_`-sibling. Watch fail (module missing). Implement `_item_outcome.py` per Shared interfaces (`# noqa: N818` — the name is `ItemExcluded`, not `…Error`, by spec). Pass.

- [ ] **Step 2 — marker helpers (red).** In `test_driver_load_hook.py`: `write_exclusion` then `read_exclusion` round-trips `{"reason", "details"}`; `exclusion_path` is `<gen>/<join_key(key)>.excluded.json`; `already_excluded` is True after a write; a truncated/invalid marker makes `read_exclusion` return `None` and warn (`stacklevel=2`). Watch fail. Implement the four helpers in `_driver.py` (atomic write via temp+`os.replace`, matching `write_shard`). Pass.

- [ ] **Step 3 — `for_each(load=)` resume-before-load (red).** Test with a `SpyLoader`-like counter: a key whose shard exists is never passed to `load`; a fresh key is (non-vacuous — assert the spy DOES fire for the fresh key). Watch fail (`for_each` has no `load`). Implement: when `load is not None`, the existing resume/marker checks run on `key(ref)` **before** `item = load(ref)`; `load(ref); work(item)` share one `try`. `load is None` path is untouched. Pass.

- [ ] **Step 4 — exclusion outcome (red).** Tests: `ItemExcluded` from `load` **and** from `work` each writes a marker, is skipped+replayed on resume, resets the consecutive-failure run, is counted by `assert_conservation` (shard **or** marker = accounted), and is invisible to `reconcile` + `*.parquet` globs. A non-`ItemExcluded` exception is still a recorded failure and counts toward `max_consecutive_failures` (3 in a row aborts). Watch fail. Implement the `except ItemExcluded`/`except Exception` split in the one `try`; extend `CorpusPassResult` (`excluded`, `exclusions`, `shard_keys` property) and `assert_conservation`. Pass.

- [ ] **Step 5 — manifest `n_excluded` (red, interpretation 3).** Test: `manifest_fields(..., excluded=2)["n_excluded"] == 2`; omitting `excluded` is a `TypeError` (no default); `CorpusPassResult.manifest()` carries `n_excluded`. Watch fail. Implement. Pass.

- [ ] **Step 6 — no-`load` byte-identity (red, interpretation 3).** Test: a `load=None` pass produces a manifest byte-identical to a recorded `4ac26d0` manifest for the same fixture **except** it now also has `n_excluded == 0`; every pre-existing field is unchanged. Watch fail (if `n_excluded` absent) / adjust. Confirm existing `test_driver.py` + `test_driver_resume_oracle.py` still green.

- [ ] **Step 7 — `aggregate_manifests` vote (red, interpretation 9).** Test `test_a_REPLAYED_exclusion_count_does_not_give_a_resume_a_commit_vote`: two manifests, one a full resume (`n_attempted=0`) with `n_excluded>0` at a different commit, still report `commit_consistent is True`. Watch fail. Implement: `aggregate_manifests` sums `n_excluded` but excludes it from the "declares it built nothing" vote (a resume with only replayed exclusions keeps no vote). Confirm `test_a_FULL_RESUME_pass_does_not_re_arm_the_commit_false_alarm` still green. Pass.

- [ ] **Step 8 — verify.** `pytest tests/scripts/test_item_outcome.py tests/scripts/test_driver_load_hook.py tests/scripts/test_driver.py tests/scripts/test_driver_resume_oracle.py tests/scripts/test_partition.py`; then ruff+pyright on the touched files. No commit.

### Task 2: Pining loader — refs, `load_match`, `MatchExcluded`, cache env, wrappers, StatsBomb events-only, `_fake_corpus`

**Spec:** §4.1. **Interpretations:** 1, 2, 5.

**Files:**
- Modify `scripts/_loader_pining.py`; `tests/scripts/test_loader_pining_events_only.py` (invert 2 StatsBomb tests; drop `load_matches(events_only=…)` tests; retarget to `load_match`).
- Create `tests/scripts/test_loader_refs.py`, `tests/scripts/_frozen_loader_4ac26d0.py`, `tests/scripts/_fake_corpus.py`.

**Interfaces:**
- Consumes: `ItemExcluded` (Task 1). Produces: `MatchRef`/`list_match_refs`/`LoadedMatch`/`MatchExcluded`/`load_match`/`resolve_cache_dir`/`CORPUS_CACHE_ENV`, the tracking-only wrappers, and the `_fake_corpus` helpers (all in Shared interfaces). Every later task's tests consume `_fake_corpus`.

- [ ] **Step 1 — `resolve_cache_dir` (red).** Test: argument beats `SILLY_KICKS_CORPUS_CACHE_DIR` beats unset→`None`; a blank env value reads as unset (not cwd). Watch fail. Implement. Pass.

- [ ] **Step 2 — `MatchRef` + `list_match_refs` (red).** Test: `MatchRef(p, m).key == (p, m)`; `artifacts` excluded from equality/hash/repr (`field(compare=False, hash=False, repr=False)`); `[r.key for r in list_match_refs(...)] == select_match_ids(...)` over a monkeypatched `_list_matches`. Watch fail. Implement `list_match_refs` on the shared `_wanted_for_provider`; rewrite `select_match_ids` as `[r.key for r in list_match_refs(...)]`. Pass.

- [ ] **Step 3 — `load_match` full load + S1 raise (red).** Tests (fake provider builders / monkeypatched `_build_match_with_retry`): a full load returns a 7-field `LoadedMatch`; a SkillCorner load whose report has `geometry_excluded=True` raises `MatchExcluded` whose `.details` carries both float rates (`player_off_pitch_rate`, `ball_off_pitch_rate`) — interpretation 1/2. Watch fail. Implement `load_match` wrapping `_build_match_with_retry`; `LoadedMatch` gains `report` (interpretation 2). Pass.

- [ ] **Step 4 — `events_only=True` byte-identity (red).** For idsse/SkillCorner/GS (ported) assert events-only actions equal the full build's actions with the tracking artifact absent from `paths`; GS across `homeTeamStartLeftExtraTime ∈ {True, False, None}`. **Invert** the two StatsBomb tests to: "StatsBomb events-only actions == full-build actions on the committed SB360 slice with `freeze_frames` absent", and "an unknown provider raises". Watch fail (StatsBomb events-only not implemented). Implement StatsBomb in `_artifact_roles`/`_build_match_actions`. Pass.

- [ ] **Step 5 — wrappers parity (red).** `tests/scripts/_frozen_loader_4ac26d0.py` holds verbatim copies of the `4ac26d0` `load_matches`/`load_statsbomb_matches` bodies. Test: over a monkeypatched network the live wrappers are byte-identical to the frozen ones (tuples, the `EXCLUDED …` stderr line, the `excluded n/m` summary). Watch fail. Rewrite the wrappers over `list_match_refs`+`load_match(events_only=False)`; **remove** the ported `events_only` keyword + `Literal` overloads (CDLS-SPEC-29). Pass.

- [ ] **Step 6 — `_fake_corpus` helper.** Write `tests/scripts/_fake_corpus.py` (Shared interfaces): `make_ref`, `make_loaded`, `SpyLoader` (records `(ref.key, kwargs)`; `fail`→`RuntimeError`, `exclude`→`MatchExcluded`), `install_fake_corpus` (patches a driver module's `list_match_refs`+`load_match`, and replaces any `load_matches`/`load_statsbomb_matches` it exposes with a function that raises `AssertionError("… still streams …")` so an unmigrated driver fails loud, not silently network). One self-test that `SpyLoader` records and raises as specified.

- [ ] **Step 7 — import-direction test.** `_item_outcome`, `_driver` and `_loader_pining` each import alone (subprocess or `importlib`); pin `_loader_pining` imports `_item_outcome` but not `_driver`.

- [ ] **Step 8 — verify.** `pytest tests/scripts/test_loader_refs.py tests/scripts/test_loader_pining_events_only.py tests/scripts/test_loader_*` + the e2e-adjacent loader tests that call the wrappers (`test_loader_statsbomb.py`, `test_loader_artifacts.py`, `test_loader_pining_cache_skip.py`); ruff+pyright. No commit.

### Task 3: Open-data loader — refs, single-match load, cache, wrapper

**Spec:** §4.2.

**Files:** Modify `scripts/_sb_open_data.py`; create `tests/scripts/test_sb_open_data_refs.py`. Extend `tests/scripts/_frozen_loader_4ac26d0.py` with the `4ac26d0` `load_open_data_matches` body.

**Interfaces:** Consumes `LoadedMatch`/`resolve_cache_dir` (Task 2). Produces `OpenDataRef`/`list_open_data_refs`/`load_open_data_match` (Shared interfaces); `load_open_data_matches` stays a byte-identical wrapper.

- [ ] **Step 1 — refs walk with a GLOBAL cap (red).** Test: `list_open_data_refs([(c1,s1),(c2,s2)])` lists each manifest once in order; `max_matches` is a global cap spent before the second competition; `match_ids` filters across competitions (manifest order, not request order); `OpenDataRef` compares on its key fields, not `match_date`. Watch fail. Implement. Pass.
- [ ] **Step 2 — `load_open_data_match` event-only + fail-closed (red).** Test: returns a `LoadedMatch` with empty `frames`, `player_name` + `xg` columns attached, `visible_area=None`/`report=None`; `assert_statsbomb_open_data_mode()` refuses when `SB_USERNAME`/`SB_PASSWORD` set. Watch fail. Implement. Pass.
- [ ] **Step 3 — shared cache round-trip (red).** Test: with a cache root the raw events JSON lands at `<root>/statsbomb_open/<id>.json`, the second load reads it (fetch-spy count 1); env var used when no argument; unset → fetch every time, writes nothing. Watch fail. Implement `_cached_json` over the resolved root. Pass.
- [ ] **Step 4 — wrapper parity (red).** `load_open_data_matches` byte-identical to the frozen `4ac26d0` body over monkeypatched network, across `{}`, `match_ids=`, `max_matches=`, `preserve_native=`. Watch fail. Rewrite as a wrapper over refs+`load_open_data_match`. Pass.
- [ ] **Step 5 — verify.** `pytest tests/scripts/test_sb_open_data_refs.py tests/scripts/test_sb_open_data.py`; ruff+pyright. No commit.

### Task 4: Library — `XtZoneCounts` + `ExpectedThreat.zone_counts` (shared count extractors)

**Spec:** §4.4. **Amends ADR-102.**

**Files:** Modify `silly_kicks/xthreat/` (`_grid.py`/`_transitions.py`/`_model.py` per where the count cores live; `__init__.py` export); create `tests/xthreat/test_zone_counts.py`. Register the module in the public-API Examples gate if a new module file is added.

**Interfaces:** Produces `XtZoneCounts`, `ExpectedThreat.zone_counts` (Shared interfaces). Consumed by Task 5.

- [ ] **Step 1 — extract count cores (refactor, green throughout).** ADR-102 already has `_*_from_counts` cores. Extract the inline counting each from-actions wrapper does into private per-aggregate extractors (shots/goals from `_scoring_prob`; valid-start moves from `_action_prob`; valid start+end and successful transitions from `singh_transition_matrix`). Wrappers now call the extractors. **Output-preserving:** `tests/xthreat_legacy_reference.py` (SK-xT-1 oracle) and `tests/xthreat/test_fit_from_counts.py` stay green with **no edit** — run them after the refactor as the gate.

- [ ] **Step 2 — `XtZoneCounts` (red).** Test: `zeros(l,w)` shapes `(w,l)`×4 + `(w*l,w*l)`, int64; `__add__` sums elementwise and **raises `ValueError`** on an `(l,w)` mismatch; `as_fit_kwargs()` returns the dict `fit_from_counts` accepts; frozen + `eq=False`. Add an Examples doctest-or-literal-block (public-API gate). Watch fail. Implement (int64 coercion in `__post_init__`). Pass.

- [ ] **Step 3 — `zone_counts` equals the ADR-102 oracle (red).** Test: `ExpectedThreat(l,w).zone_counts(actions)` equals the ADR-102 test's independent `_aggregate` oracle on its fixtures A and B, **including the valid-start / NaN-end boundary row** — that oracle stays independent, never the function under test. Add an Examples block. Watch fail. Implement `zone_counts` calling the shared extractors on `self.l`/`self.w`. Pass.

- [ ] **Step 4 — round-trip to a fit (red).** Test: `fit_from_counts(**zone_counts(a).as_fit_kwargs())` equals `fit(a)` — all four matrices **and** `xT` via `np.array_equal(..., equal_nan=True)` (not a tolerance). Watch fail (only if extractors diverged). Pass.

- [ ] **Step 5 — additivity (red).** Split `spadl_actions` into pseudo-matches (`np.array_split`, giving each part a distinct pseudo `game_id`); assert `sum(zone_counts(part)) == zone_counts(pooled)` and the fitted `xT` is `array_equal(..., equal_nan=True)` to the pooled fit. Watch fail. Pass.

- [ ] **Step 6 — verify.** `pytest tests/xthreat/ tests/test_public_api_examples.py`; ruff+pyright on `silly_kicks/xthreat/`. No commit.

### Task 5: `scripts/_xt_corpus.py` — count pass + reduce

**Spec:** §4.4.

**Files:** Create `scripts/_xt_corpus.py`, `tests/scripts/test_xt_corpus.py`.

**Interfaces:** Consumes `XtZoneCounts`/`zone_counts` (Task 4), `for_each`/`CorpusPassResult.shard_keys` (Task 1). Produces `counts_to_frame`/`counts_from_frames`/`xt_count_pass`/`fit_xt_from_count_pass`/`XtFitProvenance`/`COUNT_SHARD_*` (Shared interfaces). Consumed by Task 14 + TF-56.

- [ ] **Step 1 — sparse round-trip (red).** Test: `counts_to_frame` emits `(aggregate, from_zone, to_zone, n)` sparse rows (`to_zone=-1` for the four zone aggregates; nonzero only) and `counts_from_frames([...], l, w)` reconstructs the exact `XtZoneCounts`. Watch fail. Implement. Pass.
- [ ] **Step 2 — count pass + reduce equals pooled fit (red).** Test: `xt_count_pass(refs, key=…, load_actions=<serves fixture actions>, shard_root=…, token_inputs=…)` then `fit_xt_from_count_pass(res)` equals a pooled `fit()` on the same fixture (`array_equal(..., equal_nan=True)`). Watch fail. Implement `xt_count_pass` as `for_each(refs, load=load_actions, work=lambda a: counts_to_frame(ExpectedThreat(l,w).zone_counts(a)))`; the reduce sums the shards of `res.shard_keys` (interpretation 11) and calls `fit_from_counts`. Pass.
- [ ] **Step 3 — failure policy (red, CDLS-SPEC-17).** Test: a failing ref does not abort the pass (`for_each` records it); `fit_xt_from_count_pass` **raises** by default when `res.failures` is non-empty; with `allow_failed=True` it fits and `XtFitProvenance.failed`/`allowed_failed` record it; an excluded ref is named in `.excluded`; a resumed count pass loads nothing (spy). Watch fail. Implement. Pass.
- [ ] **Step 4 — provenance digest.** Test: `XtFitProvenance.counts_digest` is stable over the summed counts; `admission_digest`/`unmeasured_admitted` thread from an injected `AdmissionRecord` (a stub here; real one in Task 6). Pass.
- [ ] **Step 5 — verify.** `pytest tests/scripts/test_xt_corpus.py`; ruff+pyright. No commit.

### Task 6: `scripts/_events_admission.py` — admission layer

**Spec:** §4.1, §8.

**Files:** Create `scripts/_events_admission.py`, `tests/scripts/test_events_admission.py`.

**Interfaces:** Consumes `MatchExcluded`/`load_match`/`MatchRef` (Task 2). Produces `EventsOnlyAdmission`/`AdmissionRecord`/`AdmissionRefusedError`/`events_only_loader`/`ADMISSION_ARTIFACT`/`STATUSES` (Shared interfaces). This is the one Rule-D-allowed events-only consumer path.

- [ ] **Step 1 — per-status admission (red).** Over a fixture `verdicts.json`: `s1_passed`→admitted; `s1_excluded`/`tracking_unloadable` with event verdict `sound`→admitted, any other verdict→`check` raises `MatchExcluded` carrying the recorded status+verdict+reason; `events_unloadable`→never admitted. Watch fail. Implement `EventsOnlyAdmission.check`. Pass.
- [ ] **Step 2 — preflight refuses unmeasured (red).** Test: an unmeasured requested SkillCorner ref makes `preflight(refs, allow_unmeasured=False)` raise `AdmissionRefusedError` naming the key; `allow_unmeasured=True` admits and the key appears in `AdmissionRecord.unmeasured_admitted`; a pass requesting **no** SkillCorner ref needs no artifact (`digest is None`). Watch fail. Implement. Pass.
- [ ] **Step 3 — fail-closed load (red).** Test: a missing / unprovenanced / dirty-tree artifact makes `EventsOnlyAdmission.load` refuse **at preflight**, in seconds — but only when a SkillCorner ref is requested. Watch fail. Implement (`# noqa: N818` if `AdmissionRefusedError` needs it — it ends `Error`, so no). Pass.
- [ ] **Step 4 — `events_only_loader` wiring (red).** Test: it runs preflight once, returns `(load, record)` where `load(ref)` is `check(ref)` then `load_match(ref, events_only=True, cache_dir=…)`; the per-item callable is a closure (Rule D keys on the enclosing function). Non-vacuous: a spy shows `check` runs before `load_match`. Watch fail. Implement. Pass.
- [ ] **Step 5 — verify.** `pytest tests/scripts/test_events_admission.py`; ruff+pyright. No commit.

### Task 7: Task 0 driver — `scripts/build_skillcorner_s1_event_validity.py`

**Spec:** §8. **Carry item from spec review:** the events pass's `for_each(..., load=lambda r: load_match(r, events_only=True, ...))` must live inside the module-level function `_events_pass` — the one Rule-D allowlist entry besides `events_only_loader`.

**Files:** Create `scripts/build_skillcorner_s1_event_validity.py`, `tests/scripts/test_build_skillcorner_s1_event_validity.py`. Modify `tests/scripts/test_provenance_wiring.py` (register in `ARTIFACT_DRIVERS`).

**Interfaces:** Consumes `list_match_refs`/`load_match` (Task 2), `for_each`/markers (Task 1), the `goal_count_pmf` DP (`silly_kicks.match_outcome`), `require_clean_tree`/`git_provenance` (`scripts/_provenance`), `declare_inputs` (`scripts/_input_contract`). Produces `_events_pass`/`_tracking_pass` (Shared interfaces) + `reduce_verdicts`/`findings_markdown`.

- [x] **Step 1 — anchored-consistency stat (red).** Pure helper over a synthetic actions frame: own-half-anchored `goalkick`/`clearance` consistent iff `start_x < 52.5`; attacking-half-anchored `shot`/`corner_short`/`corner_crossed` iff `start_x > 52.5`; a cell is `(team, period)`; scored iff `n ≥ 5`. Assert counts on a hand-built cell. ASCII-only source (driver ASCII gate). Watch fail. Implement. Pass.

- [x] **Step 2 — reversal + boundary tests (red, both sides of the band).** (a) A mirrored synthetic cell (all anchored starts reflected across 52.5) is `reversed` — both conjuncts fire (`k/n < 0.5` AND exact Poisson-binomial `P(K≤k) < 0.05/m` via `goal_count_pmf`); its unmirrored twin is not. (b) A clamped-origin match (open-play starts piled within 0.01 m of a boundary) trips `boundary_pileup` at `τ = max(0.05, 3×q99.9)`; its clean twin does not. (c) The Poisson-binomial null matches a brute-force enumeration on a small cell. Watch fail. Implement the per-cell reversal test and the per-match boundary test. Pass.

- [x] **Step 3 — per-match verdict precedence (red).** `reversed` > `boundary_pileup` > `insufficient` (any regulation P1–2 cell unscored) > `sound`; unscored ET cells reported, not `insufficient`. Assert precedence on constructed matches. Watch fail. Implement. Pass.

- [x] **Step 4 — two passes + statuses (red).** `_events_pass` (bare `load_match(events_only=True)` — the Rule-D-allowed producer) and `_tracking_pass` (`events_only=False`; an S1 exclusion is a marker carrying the rates) over the **same** ref list, each a `for_each` pass keyed `(provider, match_id)`. Using `_fake_corpus`: a match whose tracking load fails but whose events load succeeds gets status `tracking_unloadable` **and** an event verdict; a tracking-pass failure writes no shard (resume retries). Watch fail. Implement. Pass.

- [x] **Step 5 — reduce refuses on failures; recomputes verdicts (red, CDLS-SPEC-27/28).** `reduce_verdicts` assigns every listed match a status + (where events loaded) an event verdict, recomputing `p_t`/`τ`/`m` from the shards every run (no frozen verdict); it **raises** naming keys+pass while either pass has outstanding failures, and with `allow_failed=True` writes `verdicts.json` with `allowed_failed: true` + the error per affected match. A spy asserts the producer **never opens `verdicts.json`** (runs with no artifact, over a stale artifact, and re-measures a previously non-`sound` match). Watch fail. Implement `reduce_verdicts` + `findings_markdown`; `main()` does `require_clean_tree` FIRST, `declare_inputs`, `--allow-failed`/`--allow-dirty`/`--match-ids-json`/`--list-matches`. Pass.

- [x] **Step 6 — provenance wiring.** Add the driver to `ARTIFACT_DRIVERS` in `test_provenance_wiring.py`; confirm it imports `require_clean_tree`, offers `--allow-dirty`, never shells `rev-parse`, calls `require_clean_tree` from `main()`.

- [x] **Step 7 — verify.** `pytest tests/scripts/test_build_skillcorner_s1_event_validity.py tests/scripts/test_provenance_wiring.py`; ruff+pyright. No commit. **The DGX run is Task 18**, not now.

### Task 8: CI gate — derived population, private scan, Rules A–D, plants (lands RED)

**Spec:** §5. **Interpretations:** 5, 6.

**Files:** Modify `tests/scripts/test_corpus_driver_resilience.py` (+ create `tests/scripts/_corpus_load_rules.py` and/or `tests/scripts/_script_population.py` if a shared helper is cleaner). This task **lands red on purpose** (interpretation 6): the ledgers below enumerate the exact `4ac26d0` violations; Tasks 9–15 each delete their own entries; Task 17 asserts both ledgers empty.

**Interfaces:** Produces the `_corpus_load_rules` helpers (`all_script_trees`, `public_loader_functions`, `corpus_functions`, `rule_a`/`rule_b`/`rule_c`/`rule_d`, `Violation`, `LOADER_NON_CORPUS`), the ledgers `_RULE_A_PENDING`/`_RULE_C_PENDING`, the exemption sets, `_KEY_EXCEPTIONS`, and the key-pin tests consumed by Tasks 9–15.

- [x] **Step 1 — derived population (red, both ways).** Every public function in `_loader_pining.py`/`_sb_open_data.py`/`_loader_databricks.py` (AST) is either a corpus function (prefix `load_`/`list_`/`select_`/`fetch_`) or in `_LOADER_NON_CORPUS` with a reason; asserted exact both ways. A plant pins the derived set contains the known loaders (non-vacuous). Watch fail (helper missing). Implement `corpus_functions`/`public_loader_functions`; seed `_LOADER_NON_CORPUS` with the seven §5.1 names. Pass.

- [x] **Step 2 — scan every script incl. private (red).** `all_script_trees()` parses every `scripts/*.py`, private included, no module-level exemption. Watch fail. Implement. Pass.

- [x] **Step 3 — Rule A (red, ledgered).** No `load_matches`/`load_statsbomb_matches`/`load_open_data_matches` call in any driver outside `_STREAM_LOADER_EXEMPT` (`calibrate_tracking_defaults._load_fold`, function-granular — interpretation 5). `live == set(_RULE_A_PENDING)` (the 37 modules in "Derived populations"), asserted both ways. Plants: one red, one migrated-green twin. Watch the ledgered assertion pass at the RED tree. 

- [x] **Step 4 — Rule B (red).** Every `load_match(...)` passes `events_only=` as a keyword (the signature has no default; the static rule documents intent + catches it before runtime). Plant both sides.

- [x] **Step 5 — Rule C (red, ledgered).** A **loading loop** (spec §5.5: iterates a load — flow-sensitively-resolved Name included — AND its body/`map`/`filter`-fn performs a per-item load) fails unless its innermost function is in `_UNSHARDED_LOOP_EXEMPT` (`_driver.for_each`; the three stream wrappers; `_load_fold`). Commit the scratch AST emulation as the rule; `live == set(_RULE_C_PENDING)` (the 23 functions minus the exempt, in "Derived populations"), both ways. Plants: (i) direct stream iterable; (i) Name bound to a stream loader; (i) iterable calling a `load*` param; (ii) body calls `load_match`; (ii) `map` with a loading lambda; (ii) body calls a `load*` param; green: rebinding-before-loop; id-only loop over `select_match_ids`. Live-population test names the five green sites (`train_gk_completion._corpus_taxonomy`/`.main`, `train_xcross_attempt._corpus_fingerprint`, `train_xshot_occurrence._corpus_fingerprint`, `_loader_pining_to_cache.main`'s `_cached`) and asserts none flagged **on the migrated tree** (interpretation: `_loader_pining_to_cache.main` is flagged at `4ac26d0`, green after Task 15).

- [x] **Step 6 — Rule D (red).** A `load_match(...)` whose `events_only=` is not literal `False` fails unless inside an `_UNADMITTED_EVENTS_ONLY_ALLOWED` module-level function (`_events_admission.events_only_loader`, `build_skillcorner_s1_event_validity._events_pass`; closures inside included), both ways. Plants: red — bare `events_only=True` in a driver; variable-valued `events_only=` in a driver; a second un-admitted call in another function of `_events_admission.py`. green — an `events_only_loader` twin; the producer's `_events_pass` call.

- [x] **Step 7 — key-pin scaffold + `_KEY_EXCEPTIONS`.** Add `_KEY_EXCEPTIONS` (initially `build_rq_pass_scores`→`ref.match_id`, `measure_gs_shot_distribution`→`f'{ref.provider}_{ref.match_id}'`, `_xt_corpus`→`key`), `test_ref_key_joins_exactly_like_the_old_item_key` (`join_key(MatchRef(p,m).key) == join_key((str(p),str(m)))`), and `test_every_migrated_driver_keeps_its_pre_migration_key` (AST over every `for_each(..., load=...)` `key=`, must be `ref.key` or the driver's `_KEY_EXCEPTIONS` entry). Tasks 9–15 extend `_KEY_EXCEPTIONS` as they migrate.

- [x] **Step 8 — record RED, meta-assert the ledgers.** Run the gate file; **record** the exact failing set (this is the interpretation-6 red landing — the ledgers ARE the recorded set, so the file passes with the ledgers populated and the migrations pending). Add the anti-rot meta-assertions (ledgers ⊆ derived population; `_UNDERIVABLE` asserted empty). `pytest tests/scripts/test_corpus_driver_resilience.py`; ruff+pyright. No commit.

### Task 8.5: Source factories `pining_source` / `open_data_source` + retrofit the migrated drivers

**Spec:** §4.1, §4.5 (amended 2026-09-23). **Owner-ratified: extract AND retrofit all.**

**Rationale:** see the top-of-plan amendment banner. The 3-line tracking-source construction was hand-repeated in every migrated driver (4 as named `_pining_source`/`_corpus_source` helpers, ~12 inline); collapse it into two factories, one per load mode (the events-only mode already has `events_only_loader`).

**Files:** Modify `scripts/_loader_pining.py` (add `pining_source`), `scripts/_sb_open_data.py` (add `open_data_source`), `tests/scripts/_corpus_load_rules.py` (`_LOADER_NON_CORPUS` += both names), and every already-migrated driver + its `_pining_source`/`_corpus_source` copy; create `tests/scripts/test_source_factories.py`.

**Interfaces produced:**
```python
# scripts/_loader_pining.py
def pining_source(providers, *, match_ids=None, max_per_provider=None, tracking_limit=None,
                  cache_dir=None, token=None, base_url=None) -> tuple[list[MatchRef], Callable[[MatchRef], LoadedMatch]]:
    cd = resolve_cache_dir(cache_dir)
    refs = list_match_refs(providers=providers, match_ids=match_ids, max_per_provider=max_per_provider, token=token, base_url=base_url)
    def load(ref): return load_match(ref, events_only=False, tracking_limit=tracking_limit, cache_dir=cd, token=token, base_url=base_url)
    return refs, load
# scripts/_sb_open_data.py
def open_data_source(competitions, *, match_ids=None, max_matches=None, preserve_native=(), cache_dir=None
                     ) -> tuple[list[OpenDataRef], Callable[[OpenDataRef], LoadedMatch]]:
    cd = resolve_cache_dir(cache_dir)
    refs = list_open_data_refs(competitions, match_ids=match_ids, max_matches=max_matches)
    def load(ref): return load_open_data_match(ref, preserve_native=preserve_native, cache_dir=cd)
    return refs, load
```

- [ ] **Step 1 — factories + tests (red-first).** Test: `pining_source(...)` returns `(refs, load)` where `refs == list_match_refs(...)` and `load(ref)` == `load_match(ref, events_only=False, …)` (spy the two primitives; assert the `load` closure passes `events_only=False` literal, resolved `cache_dir`, and threads `token`/`tracking_limit`). Same for `open_data_source` over `list_open_data_refs`/`load_open_data_match`. Implement.
- [ ] **Step 2 — classify.** `_LOADER_NON_CORPUS += {"pining_source": "<reason>", "open_data_source": "<reason>"}`; the §5.1 population test stays exact both ways. Confirm the factories are gate-clean (rule_a/rule_c see no violation in the loader modules).
- [ ] **Step 3 — retrofit the migrated drivers.** Every driver + organic `_pining_source`/`_corpus_source` migrated in Tasks 9–11 uses `pining_source`/`open_data_source`; reshaping/dual-source drivers keep a thin wrapper around it. Generations MUST stay byte-identical (verify a token/shard-key sample is unchanged). Delete the 4 organic copies. `_KEY_EXCEPTIONS` entries unchanged (the key= still keys the same way).
- [ ] **Step 4 — verify.** `pytest tests/scripts/` + the gate; ruff+pyright. No commit.

> **Execution note:** Tasks 12–15 (remaining drivers) use the factory FROM THE START (recipe step 1–2, factory form). Task 8.5's retrofit covers the 20 already migrated; do it once the factory exists so the whole cycle lands on one seam.

### Task 9: Migrate S-shape pining drivers, batch A — reference migration + 8 drivers

**Spec:** §4.5. **Interpretations:** 11, 12.

**Files:** Modify `scripts/build_layer2_spells.py`, `build_gkdv_arm_values.py`, `build_tf19_instrument_responsiveness.py`, `derive_opengoal_range.py`, `run_signoff_power.py`, `build_rq_pass_scores.py`, `materialize_tc3_frames.py`, `measure_gs_shot_distribution.py`, `train_gk_completion.py`, and their tests; `tests/scripts/test_corpus_driver_resilience.py` (delete this task's ledger rows).

**Interfaces:** Consumes `list_match_refs`/`load_match`/`LoadedMatch` (Task 2), `for_each(load=)`/`shard_keys` (Task 1), `events_only_loader` (Task 6), `_fake_corpus` (Task 2), the Task 8 ledgers + `_KEY_EXCEPTIONS`.

**The S-migration recipe** (every S driver, this task and Task 10). **Amended 2026-09-23** to the factory (`pining_source`); the pre-factory two-line form is kept below in brackets as the historical shape the already-migrated drivers used before the Task-8.5 retrofit:
1. Import `pining_source` from `_loader_pining` (was: `list_match_refs, load_match`), same import style the driver uses.
2. `for_each(load_matches(<sel>, tracking_limit=…), key=lambda item:(str(item[0]),str(item[1])), …)` → `refs, load = pining_source(<sel>, tracking_limit=…, cache_dir=args.cache_dir)` then `for_each(refs, key=lambda ref: ref.key, load=load, …)`. Selection stays on the listing; `tracking_limit` moves to the load. [pre-factory: `for_each(list_match_refs(<sel>), key=lambda ref: ref.key, load=lambda ref: load_match(ref, events_only=False, tracking_limit=…, cache_dir=args.cache_dir), …)`.]
3. `work` receives a `LoadedMatch` (7 fields); a 5-name unpack gains `*_`.
4. Add `--cache-dir` where missing (help: `raw-artifact cache root (default: $SILLY_KICKS_CORPUS_CACHE_DIR, else no cache)`).
5. Every combine that reads a shard per key iterates `res.shard_keys` (interpretation 11); a combine that read `res.keys` **without first refusing on failures** now refuses (a failed key has no shard).
6. Token dicts are **not** touched — finished generations resume.

- [ ] **Step 1 — tests first, per driver.** For each driver whose `main()`/`_extract` runs under cheap stubs, add via `_fake_corpus`: a walk-one-match end-to-end test; a resume-loads-nothing test asserting the shard lands in the generation named by the `4ac26d0` token literal and a second run's `SpyLoader.calls` count is unchanged. For the combine-site drivers (`train_gk_completion._extract`, `run_signoff_power`, `derive_opengoal_range`, `materialize_tc3_frames`) add an "an excluded match is left out of the combine, not `FileNotFoundError`" test. `materialize_tc3_frames.preflight_reference_parity` gets a test that it takes injected `list_match_refs`+`load_match` and makes ONE direct load; an excluded/unlisted reference raises `SystemExit`. Watch them fail (the fake corpus raises "still streams", or the preflight signature mismatches) — record the exact failures.

- [ ] **Step 2 — migrate the reference driver.** `build_layer2_spells` first (the recipe's worked example). Watch its tests pass.

- [ ] **Step 3 — migrate the other 8** per the recipe. `measure_gs_shot_distribution` additionally becomes events-only (it never reads frames; load via `events_only_loader`, drop its dead `--tracking-limit`, record `events_only`/admission digest/exclusions in `scope`) — spec §3 amended S→E, ratified at plan review 2026-09-23. `build_rq_pass_scores` keeps `key=lambda ref: ref.match_id` (its pre-migration key), added to `_KEY_EXCEPTIONS`; `measure_gs_shot_distribution` keeps `f'{ref.provider}_{ref.match_id}'`. Watch all pass.

- [ ] **Step 4 — delete this task's ledger rows** from `_RULE_A_PENDING` (10) and `_RULE_C_PENDING` (`materialize_tc3_frames.preflight_reference_parity`). Gate green for these drivers.

- [ ] **Step 5 — verify.** `pytest` the nine drivers' tests + the gate + `test_provenance_wiring.py`; ruff+pyright on the touched files. No commit.

### Task 10: Migrate S-shape pining drivers, batch B — 8 drivers

**Spec:** §4.5. Same recipe as Task 9.

**Files:** Modify `scripts/train_xshot_occurrence.py`, `train_xcross_attempt.py`, `tune_structural_pass_sigma.py`, `validate_shot_goalmouth_sb.py`, `validate_skillcorner_keeper_origin.py`, `validate_xcross_causal.py`, `validate_xs_probe.py`, `validate_xshot_causal.py`, and their tests; the gate ledgers. Create `tests/scripts/test_trainer_load_seam.py`, `test_tune_structural_pass_sigma.py`, `test_validate_xcross_causal.py` (drivers with no corpus-pass test today).

**Interfaces:** As Task 9. Produces in both trainers: `_pining_source(...) -> (refs, load)` (replaces `_iter_matches_from_pining`), a module-level `_source_key(item)` (a `MatchRef`→`.key`, a `--data-dir` tuple→`(str(item[0]),str(item[1]))`), and `_extract(..., load=None)`; both added to `_KEY_EXCEPTIONS` as `_source_key`.

- [ ] **Step 1 — tests first.** Per driver, the Task-9 walk-one / resume-loads-nothing / excluded-left-out tests via `_fake_corpus`. Repair the network dependency this migration INTRODUCES into `test_corpus_taxonomy.py::test_a_restricted_corpus_NEVER_ships_a_public_label` (review SHOULD-FIX-02): at `4ac26d0` the test is offline — it monkeypatches `train_xshot_occurrence._iter_matches_from_pining` + `match_visibility`. This task deletes `_iter_matches_from_pining`, so the patched-out path is gone and the test would reach the live pining manifest via `select_match_ids`/`list_match_refs`. Install the fake corpus on `_loader_pining` so listing, loads and the fingerprint stay offline. Add the trainers' `_source_key`-keys-both-paths test. Watch fail; record the failures (stream-refused / missing `list_match_refs` / `TypeError` on `load=`).

- [ ] **Step 2 — migrate.** Trainers: replace `_iter_matches_from_pining` with `_pining_source`, keep the `--data-dir` local generator, give `_extract` a `load=` param, combine over `res.shard_keys`. `validate_shot_goalmouth_sb`: resolve the cache root ONCE (`resolve_cache_dir`) and share it with the loader and `--z-compare`. `validate_skillcorner_keeper_origin`: SkillCorner tracking, so an S1 exclusion is a counted marker (`n_excluded` in `manifest_all.json`); combine stays `reconcile` (parquet-only). `validate_xshot_causal.build_shards`/`validate_xcross_causal.run`/`validate_xs_probe.run` gain a `cache_dir` param. Watch all pass.

- [ ] **Step 3 — delete this task's 8 Rule-A ledger rows** (`_RULE_C_PENDING` has none for this task). Gate green.

- [ ] **Step 4 — verify.** `pytest` the eight drivers + the three new test files + gate + `test_trainer_cache_and_providers.py`/`test_input_contracts.py`/`test_causal_e2e.py -m "not e2e"`; ruff+pyright. No commit.

### Task 11: Migrate StatsBomb / SB360 / GI drivers

**Spec:** §4.5, §4.6. **Heaviest task, most likely to overrun a session** (review CONSIDER-07: `gk_decision` reconstruction legs + `build_sb360_coverage` raw-360 + the new `fetch_open_360_raw`) — checkpoint green after EACH driver so a resume restarts at the last finished one, not at Step 1. **Found while migrating (verified by the reviewer, SHOULD-FIX confirmed REAL):** the `attempted - skipped - failed` "processed" line in `validate_sb360_licensed_corpus` and `build_sb360_coverage` subtracts skips twice (`attempted` already excludes skips — `_driver.py:596` after `:580`); becomes `len(res.shard_keys) - res.skipped`.

**Files:** Modify `scripts/train_receiver_model.py`, `validate_sb360_licensed_corpus.py`, `validate_territorial_defense.py`, `validate_gk_decision.py`, `build_sb360_coverage.py`, `scripts/_sb_open_data.py` (add `fetch_open_360_raw` + `_open_frames`/`_cached_json` for the raw-360 driver), and their tests; the gate ledgers.

**Interfaces:** Consumes Task 2/3 loaders, Task 1 seam, `_fake_corpus`. Produces `_sb_open_data.fetch_open_360_raw(match_id, *, cache_dir=None) -> (events, frames_raw)` and `validate_gk_decision.list_gi_refs(...) -> (refs, unavailable)` / `load_gi_match(ref, ...)`.

- [ ] **Step 1 — tests first.** `train_receiver_model`: `_corpus_source(provider) -> (refs, load)` routes `statsbomb`→SB360 full-load, any tracking provider→`load_match`; the deployment + extract passes key by `ref.match_id`; resume loads nothing. `validate_sb360_licensed_corpus`: `--fixture-only` unchanged; the pining path loads via refs; the double-subtraction "processed" fixed. `validate_territorial_defense`: `--list-matches` lists refs without loading (today it builds every SB360 match to print ids); the pass loads via refs. `validate_gk_decision`: `list_gi_refs` returns `(refs, unavailable)` from the manifest artifact map (no download); a match without GI artifacts is counted `n_gi_unavailable`, never silently skipped; the two reconstruction legs become sharded `for_each` passes over refs, combined from `res.shard_keys`; native samples read back from the native pass's shards (identical `_measure_match`, never re-run). `build_sb360_coverage`: `load_match_raw(match, *, cache_dir)` fetches events+360 through `fetch_open_360_raw` inside `for_each` after the resume check; `measure_match` takes `(ref, events, frames_raw)`; `_list_match_refs` replaces the streaming `_iter_matches`; `--cache-dir` added; double-subtraction fixed. Watch fail; record.

- [ ] **Step 2 — migrate** the five drivers + `_sb_open_data` per the tests. `validate_gk_decision` reconstruction: two `for_each` passes (fidelity over SkillCorner GI refs; SB360 reachability sweep over statsbomb refs), each with its own generation token. Watch pass.

- [ ] **Step 3 — delete this task's ledger rows** (`train_receiver_model._load_corpus`→`_corpus_source`, `validate_sb360_licensed_corpus._pining_items`, `validate_territorial_defense.main`/`.main._matches`, `validate_gk_decision._reconstruction_verdicts`; `build_sb360_coverage` was never ledgered — record it migrated anyway). Gate green.

- [ ] **Step 4 — verify.** `pytest` the five drivers' tests + `test_sb_open_data_refs.py` + `test_validate_territorial_defense.py`/`test_gk_decision_battery_kernels.py`/`test_build_sb360_coverage.py` + gate + `test_scale_guards.py` (territorial-defense subquadratic guard); ruff+pyright. No commit.

### Task 12: Migrate open-data S-shape drivers (6)

**Spec:** §4.2, §4.5.

**Files:** Modify `scripts/evolve_xsuccess_features.py`, `train_pass_completion.py`, `train_win_probability.py`, `train_xsuccess.py`, `validate_territory_counterfactual.py`, `validate_team_kpi_reliability.py`, and their tests; the gate ledgers.

**Interfaces:** Consumes `list_open_data_refs`/`load_open_data_match` (Task 3), the seam, `_fake_corpus` (add an open-data variant or a `SpyLoader` over `OpenDataRef`).

- [ ] **Step 1 — tests first.** Each driver's own multi-competition walk (its `seen`/`chain.from_iterable`) is replaced by `list_open_data_refs(competitions, match_ids=…, max_matches=…)` (GLOBAL cap) + `for_each(refs, load=lambda r: load_open_data_match(r, preserve_native=…, cache_dir=…), …)`. `train_pass_completion`'s Gradient Sports events-only path goes through `events_only_loader` (Rule D). `validate_team_kpi_reliability` keeps its wyscout prepass. Add per-driver walk-one / resume-loads-nothing tests; assert the global cap and the shared cache. Watch fail; record.

- [ ] **Step 2 — migrate** the six; keys stay `ref.match_id`/existing. Watch pass.

- [ ] **Step 3 — delete this task's ledger rows.** Gate green.

- [ ] **Step 4 — verify.** `pytest` the six + `test_train_pass_completion.py`/`test_team_kpi_reliability.py`/`test_tf54b_drivers.py` + gate; ruff+pyright. No commit.

### Task 13: Migrate open-data U-shape drivers (3)

**Spec:** §4.5.

**Files:** Modify `scripts/build_territory_ranking_census.py`, `validate_match_outcome_calibration.py`, `train_match_outcome_dependence.py`, and their tests; the gate ledgers.

**Interfaces:** As Task 12. Each U-shape becomes a `for_each` over refs writing per-match slices, combined before the whole-corpus barrier (the `_load_xt_corpus_pining` precedent).

- [ ] **Step 1 — tests first (the `feedback_test_trainer_locally_before_dgx` gate).** A **local reduce test over synthetic shards** for each: the prepass writes per-match action/tuple slices, and the barrier computation (ICC census / calibration / Poisson-binomial dependence fit) runs over the combined shards with a byte-identical result to the pre-migration in-memory path on a small fixture. `train_match_outcome_dependence` (3,961 matches, no shards today) gains resume. Remove its `_NOT_YET_MIGRATED` marker if the gate had one. Watch fail; record.

- [ ] **Step 2 — migrate** the three. Watch pass.

- [ ] **Step 3 — delete this task's ledger rows** (`build_territory_ranking_census._corpus_matches`/`.main`, `validate_match_outcome_calibration._iter_open_matches`, `train_match_outcome_dependence.main`). Gate green.

- [ ] **Step 4 — verify.** `pytest` the three + `test_match_outcome_train.py` + `test_build_territory_ranking_census.py` + gate; ruff+pyright. No commit.

### Task 14: Migrate xT-fit M/E/U drivers via the count pass

**Spec:** §4.4, §4.5, §7. **Generation move (interpretation, spec §7):** `measure_cover_shadow_argmax_agreement` — its token now declares the requested refs + the fit's `counts_digest`.

**Files:** Modify `scripts/build_tf60_layer3_arm_values.py`, `measure_cover_shadow_argmax_agreement.py`, `_xtgk_comparability.py`, and their tests; the gate ledgers.

**Interfaces:** Consumes `xt_count_pass`/`fit_xt_from_count_pass`/`XtFitProvenance` (Task 5), `events_only_loader` (Task 6), the seam, `_fake_corpus`.

- [ ] **Step 1 — tests first.** Each M/E/U driver's xT fit becomes: `xt_count_pass(refs, key=…, load_actions=<events_only_loader-backed>, …)` then `fit_xt_from_count_pass(res, allow_failed=args.allow_failed)`. The second (scoring/arms) pass is `for_each(refs, load=full load, work=…)` with the fitted `xT` closed over — no materialized list (the OOM class gone). `build_tf60 --xt-out` serializes `XtFitProvenance.fit_keys` as its `corpus_ids`. Each caller gains `--allow-failed` (refuse-by-default, record). Add: a fit-equals-pooled test on a small fixture; a resume-loads-nothing test; a `--allow-failed` records-not-launders test; for `measure_cover_shadow` the new-generation-token test. Watch fail; record.

- [ ] **Step 2 — migrate** the three. Watch pass.

- [ ] **Step 3 — delete this task's ledger rows** (`build_tf60_layer3_arm_values.main`, `measure_cover_shadow_argmax_agreement.main`, `_xtgk_comparability._collect`/`.main`). Gate green.

- [ ] **Step 4 — verify.** `pytest` the three + `test_xtgk_comparability_cache.py` + `test_measure_cover_shadow*` + gate; ruff+pyright. No commit.

### Task 15: Migrate remaining drivers

**Spec:** §4.5, §7. **Generation move (spec §7):** `measure_rc4_orientation` key `(provider,)`→`(provider, match_id)` (free — its token already has `run_commit`).

**Files:** Modify `scripts/calibrate_xt_bandwidth.py`, `calibrate_tracking_defaults.py`, `measure_rc4_orientation.py`, `train_ghost_outfield.py`, `_loader_pining_to_cache.py`, and their tests; the gate ledgers.

**Interfaces:** Consumes Task 2/5/6, the seam, `_fake_corpus`.

- [ ] **Step 1 — tests first.**
  - `calibrate_xt_bandwidth._load_one_match` → refs + `events_only_loader` (drops its per-match manifest re-listing); keeps its action shards (KDE needs raw destinations); SkillCorner-including → new generation (interpretation 8).
  - `calibrate_tracking_defaults._load_xt_corpus_pining._work` → `events_only_loader` (never a bare `load_match(events_only=True)`, Rule D); `_load_fold` stays the recorded X exemption.
  - `measure_rc4_orientation` → `for_each(list_match_refs(providers=…, max_per_provider=1), key=lambda r:(r.provider,r.match_id), load=full, work=…)`; assert the key move + that the published shard content (already carries `provider`/`match_id`) is unchanged + the `run()` "no scored match" guard still fires (an S1-excluded match counts as no scored match explicitly).
  - `train_ghost_outfield` → its path list IS the ref list; `load=pd.read_parquet` (I-shape, one idiom).
  - `_loader_pining_to_cache.main` → a `for_each` pass; `_cached` stays the resume pre-filter over `list_match_refs` (iterates refs, no load — Rule C allows); "no vx/vy" SKIP → `ItemExcluded("no vx/vy")`; S1 drop → named marker; passes `cache_dir` (raw artifacts now cached); `--out` stays required (materialized cache ≠ raw cache).
  Watch fail; record.
- [ ] **Step 2 — migrate** the five. Watch pass.
- [ ] **Step 3 — delete this task's ledger rows** (`calibrate_xt_bandwidth._load_one_match`, `calibrate_tracking_defaults._load_xt_corpus_pining._work`, `measure_rc4_orientation`, `_loader_pining_to_cache.main`). Gate green; **both ledgers now empty** except any Task-17-verified residual.
- [ ] **Step 4 — verify.** `pytest` the five + `test_calibrate_*` + `test_measure_rc4_orientation.py` + `test_loader_pining_to_cache*` + gate; ruff+pyright. No commit.

### Task 16: Docs — ADR-052 amendment, ADR-102 amendment, CLAUDE.md, CHANGELOG, TODO

**Spec:** §4.7, §7.

**Files:** Modify `docs/superpowers/adrs/ADR-052-corpus-driver-resilience.md`, `ADR-102-*.md`, `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`.

- [ ] **Step 1 — ADR-052 amendment (in place, no new number).** D4 revised (refs + `load=` hook; "invert only where trivial" retired; `train_ghost_gk` stays the documented exception); new D13 (excluded outcome + marker-validity=shard-validity determinism rule); D14 (loader contract + edge admission + Rule D); D15 (xT fits are count passes).
- [ ] **Step 2 — ADR-102 amendment.** `zone_counts` extends the counts contract; the count extractors join the single-sourced cores.
- [ ] **Step 3 — CLAUDE.md.** The xT bullet gains `zone_counts`/`XtZoneCounts`; a durable line for the corpus-driver seam (resume-before-load, `ItemExcluded`, `events_only`+admission, count-pass fits, one cache var). No pre-claimed numbers.
- [ ] **Step 4 — CHANGELOG + TODO.** CHANGELOG entry (number resolved at commit-prep, not now). TODO: mark the cycle, and the TF-56 follow-on (spec §9).
- [ ] **Step 5 — verify.** No test change; confirm nothing else references a removed symbol. No commit.

### Task 17: Whole-branch verification and commit-1 stop

**Spec:** §6, §9.

- [ ] **Step 1 — pyright baseline.** Measure `python -m pyright` (bare) on the branch base; record the pre-existing error count (spec: 16, all in `build_sb360_coverage`/`probe_sb_cross_blocked`/network-gated e2e). The bar is **zero new**.
- [ ] **Step 2 — ledgers empty.** `_RULE_A_PENDING` and `_RULE_C_PENDING` are empty (or hold only a Task-8 recorded residual with a reason); `_UNDERIVABLE` empty; the §4.6 residual plant still pins the stated limit.
- [ ] **Step 3 — whole suite.** `python -m pytest tests/ -m "not e2e"` green (add `--benchmark-skip` for any `tests/tracking/` selection). Capture every `FAILED` (never `tail`).
- [ ] **Step 4 — lint/type at CI scope.** `python -m ruff check silly_kicks/ tests/ scripts/`; `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright` — zero new.
- [ ] **Step 5 — orphan audit.** Every new symbol has a consumer; every doc/CHANGELOG/TODO line the cycle needs is written; no `--help`-executing driver was left parserless (`grep -q add_argument`); the Task 0 driver's `verdicts.json` is NOT committed (that is Task 18).
- [ ] **Step 6 — STOP.** Present the full diff / file list for explicit commit-1 approval. **Do not commit.** Hand off for independent `/review-impl`, freezing the tree during the review.

### Task 18: Owner-run Task 0 on the DGX + commit-2 stop

**Spec:** §8, §9. **Owner-run; not executed by the implementing session.**

- [ ] **Step 1 — runbook.** From the clean commit-1 SHA, on the DGX (`ssh karsten@192.168.68.73`, owner token in `~/.pining_owner.env`), run `scripts/build_skillcorner_s1_event_validity.py` over the full owner SkillCorner corpus (never a subset — `feedback_do_not_shrink_the_validation_corpus`), resumable, warm cache. It writes `docs/research/skillcorner_s1_event_validity/verdicts.json` + `findings.md` with provenance.
- [ ] **Step 2 — inspect.** Report the 14 S1-excluded matches' event verdicts and the S1-passing calibration counts. Whatever they are, no code path is added or calibrated after the data is seen (spec §8).
- [ ] **Step 3 — commit-2 stop.** Present `verdicts.json` + `findings.md` (+ any CHANGELOG/TODO lines citing its numbers) for explicit commit-2 approval. **Do not commit.**


