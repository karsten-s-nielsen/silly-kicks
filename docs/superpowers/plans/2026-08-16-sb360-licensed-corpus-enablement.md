# SB360 licensed-corpus enablement + visibility-aware count features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the 30-match licensed StatsBomb 360 corpus (now served by `pining-for-the-data`) through the library, correct fidelity + roster identity, add opt-in visibility-aware companion columns to the three count features, and ship a leak-safe validation artifact — all additively.

**Architecture:** A scripts-side raw-JSON flattener (`_sb_raw.py`) feeds the existing `providers/statsbomb` shaping port; the pining loader gains a `statsbomb` path by widening the INTERNAL `_build_match` (public `load_matches` untouched at its 5-tuple). Visibility rides in companion columns on `add_action_context` only (per-Series functions and `tracking_default_xfns` untouched → additive by construction), computed by one `classify_region_observation` helper. A `_driver.py` validation pass runs the ADR-053 `works` battery on the corpus with shards to a gitignored root. Design spec: `docs/superpowers/specs/2026-08-16-sb360-licensed-corpus-enablement-design.md`.

**Tech Stack:** Python, pandas/numpy, pytest, `statsbombpy` (scripts-only, `importorskip`-guarded — golden de-fork only), the pining-for-the-data HTTP API, `scripts/_driver.py`/`_provenance.py`.

## Global Constraints

- **Additive, no behaviour change.** `load_matches`'s public 5-tuple (`_loader_pining.py:258`) and every unpack site (`build_gkdv_arm_values.py:187`, `tests/causal/test_causal_e2e.py:99`) are untouched; the three count Series functions and `tracking_default_xfns` (features.py:489-494) are untouched; the calibration path never computes a companion. A CI gate proves this per task.
- **No new runtime dependency.** `statsbombpy` stays a `scripts/`-only, `importorskip`-guarded dep (ADR-054). The `providers/statsbomb` port stays pure-shaping (no raw-JSON parsing inside it).
- **Licensed data is never committed.** Open-data 360 matches (redistributable) back all CI fixtures; the licensed 30-match run is owner-run. Validation-driver shards go to a **gitignored top-level root**; only the reconciled aggregate + provenance lands under `docs/research/`. Never-commit rule is **ADR-009:11**; fail-closed private default is **ADR-038:58-59**.
- **Drivers adopt `scripts/_driver.py`** (`for_each`) and stamp provenance via `scripts/_provenance.py` — `require_clean_tree(git_provenance(), allow_dirty=…)` in `main()`; `git_provenance()` returns keys `commit`/`dirty`/`tree_state`; the driver stamps `run_commit`/`run_tree_dirty` INTO the artifact (as `build_sb360_coverage.py:399-403` does). Enroll the driver in ADR-056 `ARTIFACT_DRIVERS` (`test_provenance_wiring.py`).
- **New gates land red-first** — observed failing before the fix.
- **Verification is shown, not claimed** — before any commit is requested, paste the exit-coded output of `python -m pytest tests/ -m "not e2e" --benchmark-skip`, `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`; run on both `.venv` (3.10) and `.venv312` (3.12). Bare `pyright` covers the new `scripts/_sb_raw.py` and the modified drivers: `[tool.pyright]` in `pyproject.toml` sets `include = ["silly_kicks", "tests", "scripts"]` + `extraPaths = ["scripts"]` (verified), so scripts-side type errors fail the same gate.
- **No commit steps** appear in this plan; the user commits once, on explicit approval. Version/PR/ADR numbers are assigned at commit-prep after merging `origin/main`.
- **The three modules** (do not misroute imports): `convert_to_actions` is `silly_kicks.spadl.statsbomb`; `shape_snapshots` is `silly_kicks.providers.statsbomb`; `snapshot_to_tracking_frames` is `silly_kicks.tracking`.

---

### Task 1: EXTRACT-and-unify the raw-JSON flattener (`scripts/_sb_raw.py`) — a real de-fork

> **EXECUTION NOTES (2026-08-17):** (1) the body was found **SIX** times, not three — the plan's
> three below plus `tests/test_xthreat_statsbomb_e2e.py`, `tests/invariants/_loaders.py`, and
> `tests/spadl/test_cross_provider_parity.py`. All six de-forked; `tests/spadl/test_end_coord_integration.py`
> was LEFT (a genuinely different body — larger key set + nested `possession_team` decomposition;
> Chesterton's Fence). (2) The frozen characterization oracle was **INLINED** in
> `tests/scripts/test_sb_raw.py` rather than a separate `tests/scripts/_legacy_adapt.py` module,
> because `scripts` is a namespace package on the test path and a sibling module would merge into it.

**The flattener already exists as duplicated inline copies** (plan-time survey P1 found three;
implementation found **six** — the survey was incomplete): `scripts/build_sb360_coverage.py:178
_adapt_events`, `scripts/build_worldcup_fixture.py:145 _adapt_events_to_silly_kicks_input` (same body +
a possession passthrough surfaced via `surface_native`, kept distinct from the downstream
`convert_to_actions(preserve_native=…)` per the spec's don't-overload rule), the adapter helper at
`tests/spadl/test_add_possessions.py:815`, and inline adapters in
`tests/spadl/test_cross_provider_parity.py`, `tests/invariants/_loaders.py` and
`tests/test_xthreat_statsbomb_e2e.py`.
All emit the `EXPECTED_INPUT_COLUMNS` set `{game_id, event_id, period_id, timestamp, team_id, player_id,
type_name, location, extra}` (statsbomb.py:20) keyed off `_TOP_LEVEL_KEYS`, from raw StatsBomb event
dicts (`sb.events(fmt="dict")` in production, build_sb360_coverage.py:214-215) — precisely the shape
`events.json.gz` carries. So this is **extraction + unification, not new authorship**, and the reference
is `_adapt_events`, **NOT** `sb.events(fmt="dataframe")` (which flattens away `type_name`/`extra` and
would `KeyError`; `convert_to_actions` reads `type_name` + the `extra` dict — statsbomb.py:27/29,73-107).

**Files:**
- Create: `scripts/_sb_raw.py`
- Modify (re-point onto the extracted flattener): `scripts/build_sb360_coverage.py`, `scripts/build_worldcup_fixture.py`, `tests/spadl/test_add_possessions.py`, `tests/spadl/test_cross_provider_parity.py`, `tests/invariants/_loaders.py`, `tests/test_xthreat_statsbomb_e2e.py`
- Create: `tests/scripts/test_sb_raw.py`
- Reference (read): `build_sb360_coverage.py:178-214`, `build_worldcup_fixture.py:145` (the possession passthrough), `silly_kicks/spadl/statsbomb.py:20` (`EXPECTED_INPUT_COLUMNS`), `:73-107` (`_flatten_extra` — what `extra` must carry).

**Interfaces:**
- Produces: `flatten_events(raw: list[dict], match_id, *, surface_native=()) -> pd.DataFrame` (the `_adapt_events` body + `_TOP_LEVEL_KEYS`, with the optional possession passthrough via `surface_native` — which then feeds `convert_to_actions(preserve_native=["possession"])` — so `build_worldcup_fixture` doesn't regress; named `surface_native` NOT `preserve_native`, per the spec's don't-overload rule); `parse_freeze_frames(raw) -> list[dict]`; `parse_metadata(raw) -> {home_team_id, xy_fidelity_version, shot_fidelity_version}`; `parse_roster(raw) -> dict[int, dict]` (`.get()`-tolerant).

- [ ] **Step 1: Write the characterization test (the load-bearing de-fork guarantee — committed, no network).** On a small raw event fixture, assert the extracted `flatten_events` equals the pre-extraction `_adapt_events` output.

```python
def test_flatten_events_equals_pre_extraction_adapter(raw_events_fixture):
    import scripts._sb_raw as raw
    from tests.scripts._legacy_adapt import adapt_events_legacy  # a frozen copy of the old body
    got = raw.flatten_events(raw_events_fixture, match_id=1)
    pd.testing.assert_frame_equal(got, adapt_events_legacy(raw_events_fixture, 1))

def test_flatten_events_schema_contract(raw_events_fixture):
    from silly_kicks.spadl.statsbomb import EXPECTED_INPUT_COLUMNS
    cols = set(raw.flatten_events(raw_events_fixture, match_id=1).columns)
    assert EXPECTED_INPUT_COLUMNS <= cols

def test_round_trip_through_convert_to_actions(raw_events_fixture):
    from silly_kicks.spadl.statsbomb import convert_to_actions
    actions, _ = convert_to_actions(raw.flatten_events(raw_events_fixture, match_id=1), home_team_id=…)
    assert len(actions) > 0 and set(["type_id", "start_x", "start_y"]).issubset(actions.columns)
```

- [ ] **Step 2: Run, expect FAIL** (module absent). Run: `python -m pytest tests/scripts/test_sb_raw.py -m "not e2e" -v`.

- [ ] **Step 3: Extract the flattener + the three parsers.** Move the `_adapt_events` body (+ `_TOP_LEVEL_KEYS`) into `scripts/_sb_raw.py::flatten_events`, keeping the `build_worldcup_fixture` possession passthrough as an optional `surface_native` kwarg (distinct from the downstream `convert_to_actions(preserve_native=…)`). Add `parse_freeze_frames`/`parse_metadata`/`parse_roster` (pure, `.get()`-tolerant). Freeze a copy of the old body at `tests/scripts/_legacy_adapt.py` for the characterization test. **`_legacy_adapt.py` is a PERMANENT characterization oracle, not scaffolding** — its module docstring says so: it is the frozen pre-extraction `_adapt_events` body, retained so the characterization test pins that `flatten_events` never silently drifts from the behaviour it lifted; if `flatten_events` must change behaviour, this file and the assertion change together, deliberately (round-4 minor note 2 — so a later reader meets the intent, not a puzzle).

- [ ] **Step 4: Re-point all six call sites** — `build_sb360_coverage.py:178/214`, `build_worldcup_fixture.py:145`, the `test_add_possessions.py:815` helper, and the inline adapters in `tests/spadl/test_cross_provider_parity.py`, `tests/invariants/_loaders.py` and `tests/test_xthreat_statsbomb_e2e.py` — at `_sb_raw.flatten_events`. This is the real de-fork (six copies → one; the plan-time P1 survey found only the first three).

- [ ] **Step 5: Run the committed tests + the two re-pointed suites, expect PASS.** Run: `python -m pytest tests/scripts/test_sb_raw.py tests/spadl/test_add_possessions.py -m "not e2e" -v` (and confirm `build_worldcup_fixture`/`build_sb360_coverage` still import + run their fixture tests).

- [ ] **Step 6: (optional, owner-run) an `@e2e` whole-pipeline coverage sanity check** against `statsbombpy` on an open match — a *coverage-number* check on the full pipeline, NOT a schema equivalence (the committed characterization test is the load-bearing guarantee). Mark `@pytest.mark.e2e` + `importorskip("statsbombpy")`.

---

### Task 2: The statsbomb loader path (non-breaking; fidelity + roster)

**Files:**
- Modify: `scripts/_loader_pining.py` (`_download_artifacts`, `_build_match` widen, `_build_statsbomb`, `load_statsbomb_matches`)
- Create: `tests/scripts/test_loader_statsbomb.py`
- Reference (read): `_loader_pining.py:246` (the internal 4-tuple unpack), `:262` (`_build_match_with_retry`), `:347` (`_build_match`), `:770-779` (GS roster `.get()` pattern); the three shaping fns (Global Constraints).

**Interfaces:**
- Produces: `build_statsbomb_match(paths, match_id) -> (actions, frames, home_team_id, visible_area, report)`; `load_statsbomb_matches(match_ids=None, *, token=None, base_url=None, cache_dir=None) -> Iterator[(provider, match_id, actions, frames, home, visible_area)]`.
- `_build_match` internally returns a 5-tuple `(actions, frames, home, visible_area, report)` for ALL providers (`visible_area=None` for non-snapshot ones).

- [ ] **Step 1: Red-first `load_matches`-unchanged gate.** Assert the public generator still yields a 5-tuple after the internal widening — this is the additive backstop for the calibration path.

```python
def test_load_matches_public_arity_is_unchanged(monkeypatch):
    import scripts._loader_pining as lp
    # stub _build_match_with_retry to a widened 5-tuple; load_matches must still yield exactly 5.
    monkeypatch.setattr(lp, "_build_match_with_retry", lambda *a, **k: (_ACT, _FR, 5, None, None))
    monkeypatch.setattr(lp, "_list_matches", lambda *a, **k: [{"id": "m1", "artifacts": {}}])
    (rec,) = list(lp.load_matches(providers=["skillcorner"], match_ids={"skillcorner": ["m1"]}, token="t"))
    assert len(rec) == 5  # (provider, match_id, actions, frames, home)
```

- [ ] **Step 2: Run, expect FAIL** (`_build_match` still 4-tuple; the stub's 5-tuple unpacks wrong). Run: `python -m pytest tests/scripts/test_loader_statsbomb.py -k public_arity -v`.

- [ ] **Step 3: Widen `_build_match` internally** to return `(actions, frames, home, visible_area, report)` for every branch (`visible_area=None` for idsse/gs/skc), update the `:246` unpack and `_build_match_with_retry`'s pass-through, and make `load_matches` drop `visible_area` when yielding its 5-tuple. Run Step-1 test → PASS. **Confirm the idsse/gs/skc build outputs are byte-identical** (a snapshot test of one skillcorner build before/after).

- [ ] **Step 4: Implement `build_statsbomb_match`** — `_download_artifacts` statsbomb branch (`{events, freeze_frames, metadata, roster}`, no `tracking`); parse metadata (fidelity + home_team_id); `flatten_events` → `convert_to_actions(events, home_team_id, xy_fidelity_version=…, shot_fidelity_version=…)`; `parse_freeze_frames` → `shape_snapshots(frames_raw, actions, fidelity_version=xy_fidelity_version)` → `(snapshots, visible_area, join_report)`; `snapshot_to_tracking_frames(snapshots, actions)`; `parse_roster` → identity columns on `actions` keyed by SPADL `player_id`. **Do NOT run `_preprocess`.** **Wire it into `_build_match`'s provider switch (`_loader_pining.py:347-363`)** — a `statsbomb` branch calling `build_statsbomb_match(paths, match_id)` (it ignores `tracking_limit`; freeze-frames have none) and returning the widened 5-tuple — so `_build_match_with_retry` (Step 6) can reach it.

- [ ] **Step 5: Committed loader tests on an OPEN-360 slim fixture** (redistributable): `build_statsbomb_match` returns the 5-tuple with the right schema; `frames["speed_source"]` is `"unavailable"`; `visible_area` is populated per action; `actions` carry roster identity columns; and fidelity — a metadata `xy_fidelity_version=2` yields different scaled coords than 1 (assert against a fidelity-1 recompute). Run: `python -m pytest tests/scripts/test_loader_statsbomb.py -m "not e2e" -v` → PASS.

- [ ] **Step 6: Implement `load_statsbomb_matches`** as a thin wrapper reusing `_build_match_with_retry` (download/retry/cache), yielding the 6-tuple. Add a shape test (monkeypatched build) that it yields `(provider, match_id, actions, frames, home, visible_area)`.

- [ ] **Step 7: (owner-run) real-data pining probe** on 1–2 licensed matches: `python -c "from scripts._loader_pining import load_statsbomb_matches; …"` (or a tiny probe script), confirm well-formed actions/frames/visible_area/roster. Paste counts. `PINING_FOR_THE_DATA_TOKEN` required.

---

### Task 3: The geometry classifier + feature-level source constant (`_visibility.py`)

**Files:**
- Modify: `silly_kicks/tracking/_visibility.py` (add `classify_region_observation` + `REGION_OBSERVATION_SOURCE_VALUES`)
- Modify: `tests/tracking/test_visibility.py`
- Reference (read): `_visibility.py:53-58` (`VISIBLE_AREA_SOURCE_VALUES`, do NOT widen), `:105-148` (`region_observed_fraction`: NaN on absent/degenerate/zero-area, RAISES on non-convex), `:235-241` (presence/degeneracy pre-check pattern).

**Interfaces:**
- Produces: `REGION_OBSERVATION_SOURCE_VALUES = (observed, no_polygon, degenerate_polygon, degenerate_region)` (a FEATURE-LEVEL superset — reuses the polygon tokens, adds `degenerate_region`; **not** a widening of the pinned/exported `VISIBLE_AREA_SOURCE_VALUES`). `unlinked` is NOT here (it is an action↔frame property overlaid by the caller). `classify_region_observation(polygon, region) -> tuple[float, str]` returning `(fraction, source)`.

- [ ] **Step 1: Write the classifier test (both sides + every source token).**

```python
import numpy as np, pytest
from silly_kicks.tracking._visibility import classify_region_observation, REGION_OBSERVATION_SOURCE_VALUES

_TRI = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0]])  # convex region

def test_fully_observed():
    f, s = classify_region_observation(_TRI, _TRI)      # region ⊆ polygon
    assert f == 1.0 and s == "observed"
def test_partial():
    left = np.array([[0.,0.],[52.5,0.],[52.5,68.],[0.,68.]])
    f, s = classify_region_observation(left, _TRI)
    assert 0.0 < f < 1.0 and s == "observed"            # measured; provenance is 'observed'
def test_no_polygon():
    f, s = classify_region_observation(None, _TRI)
    assert np.isnan(f) and s == "no_polygon"
def test_degenerate_region_never_raises():
    zero = np.array([[10.,10.],[10.,10.],[10.,10.]])    # zero-area region
    f, s = classify_region_observation(_TRI, zero)
    assert np.isnan(f) and s == "degenerate_region"     # NOT a ValueError
def test_all_sources_are_in_the_closed_set():
    assert set(REGION_OBSERVATION_SOURCE_VALUES) == {"observed","no_polygon","degenerate_polygon","degenerate_region"}
```

- [ ] **Step 2: Run, expect FAIL.** Run: `python -m pytest tests/tracking/test_visibility.py -k classify -v`.

- [ ] **Step 3: Implement `classify_region_observation`.** Presence/degeneracy pre-check first (absent/`len(poly) < MIN_VERTICES` → `no_polygon`/`degenerate_polygon`, mirroring `:235-241`); guard the REGION (zero-area → `degenerate_region`, NaN, never call `region_observed_fraction` which would still NaN but is clearer to pre-guard); else `fraction = region_observed_fraction(polygon, region)`, `source = "observed"`. Note (R2-F): the region is always convex from the callers, so `region_observed_fraction`'s non-convex raise is unreachable — the guard exists for zero-area.

- [ ] **Step 4: Run, expect PASS.**

---

### Task 4: `add_action_context` visibility companions (inscribed disks; the additive home)

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_action_context` gains `visible_area=None`; append 6 companion columns)
- Modify: `silly_kicks/tracking/_kernels.py` (a shared `_inscribed_disk(cx, cy, r, n) -> (N,2)` helper + the triangle region already exists)
- Modify: `tests/tracking/test_action_context.py` (or the file that tests `add_action_context`)
- Reference (read): `features.py:442-486` (`add_action_context`: builds `ctx = _resolve_action_frame_context`, emits the three counts + `frame_id`/…; sets `frame_id=NaN` for unlinked at `:459`), `_kernels.py:140-234` (regions), `_visibility.py:218-221` (the ADR-019 `canonical_id` join to mirror).

**Interfaces:**
- Consumes: `classify_region_observation` (Task 3); `visible_area` DataFrame (`action_id → polygon`) from Task 2.
- Produces (only when `visible_area` supplied): `nearest_defender_distance_observed_fraction/_source`, `receiver_zone_density_observed_fraction/_source`, `defenders_in_triangle_to_goal_observed_fraction/_source` on the returned DataFrame.

- [ ] **Step 1: Red-first ADDITIVE gate — the load-bearing guarantee.** Two assertions: the three primary columns are byte-identical with and without `visible_area`; and the six companions appear iff `visible_area` is supplied.

```python
def test_add_action_context_primary_columns_unchanged_by_visible_area(sb360_open_match):
    actions, frames, visible_area = sb360_open_match
    base = add_action_context(actions, frames)
    withva = add_action_context(actions, frames, visible_area=visible_area)
    for c in ["nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"]:
        pd.testing.assert_series_equal(base[c], withva[c])           # primary unchanged
    assert "nearest_defender_distance_observed_fraction" in withva and c_absent(base)  # companions only when supplied
```

- [ ] **Step 2: Run, expect FAIL** (`visible_area` kwarg absent / companions absent). Run: `python -m pytest tests/tracking/test_action_context.py -k visible_area -v`.

- [ ] **Step 3: Add `_inscribed_disk` to `_kernels.py`** (fixed `n=20` vertices; inscribed so coverage under-reports — the honesty invariant) + a unit test that its area is `< π r²` (inscribed) and `assert_allclose` on a known disk.

- [ ] **Step 4: Implement the companions in `add_action_context`.** Add `visible_area: pd.DataFrame | None = None`. When supplied: get the per-action polygon lookup from the **shared `_polygons_by_action(visible_area) -> dict` helper** (extract it in `_visibility.py` from `add_visible_area_coverage`'s current inline ADR-019 `canonical_id` join at `:218-221`, and re-point `add_visible_area_coverage` at it — one join, not a second copy); for each action compute the region (`triangle` for triangle-feature; `_inscribed_disk(end_x,end_y,radius,20)` for receiver-zone; `_inscribed_disk(start_x,start_y,nearest_dist,20)` for nearest-defender — **special-case NaN distance: emit `_source="degenerate_region"`, `_fraction=NaN`, and do NOT call the classifier** (the disk radius is NaN, so the region-of-interest can't be constructed — it stays in the closed set)); else call `classify_region_observation(polygon, region)`; **overlay `unlinked`** from `ctx.pointers` (where the action didn't link, set `_source="unlinked"`, `_fraction=NaN`, exactly as `frame_id=NaN` at `:459`). Emit the six columns. Leave the per-Series functions + `tracking_default_xfns` untouched.

- [ ] **Step 5: Run the additive gate + companion-behaviour tests, expect PASS.** Include: partial region → `fraction ∈ (0,1)`; no polygon → `NaN`/`no_polygon`; degenerate region → `degenerate_region` no raise; NaN-distance nearest-defender → NaN propagated; unlinked action → `unlinked`. Assert `_source` values ⊆ `REGION_OBSERVATION_SOURCE_VALUES ∪ {unlinked}`.

- [ ] **Step 6: Per-Series-untouched gate.** Assert `nearest_defender_distance(actions, frames)` etc. return a bare `pd.Series` (unchanged signature/type) and `tracking_default_xfns` is unchanged — the calibration path additive proof.

- [ ] **Step 7: Pin the orientation invariant (R2-G)** in a docstring + a comment at the region-construction site: polygon and region are both raw-SPADL sharing one frame because the kernels don't re-orient (fixed goal x=105); a re-oriented provider must re-orient the polygon too.

---

### Task 5: The licensed-corpus validation driver (leak-safe) + single-sourced call convention

**The battery's CALL CONVENTION is single-sourced, not just its name set (P2, round-4).** Spec §8.1 (H4)
commits this driver to running the `add_*` battery on real freeze-frames and reporting the
`honest_nan`/`silent_degrade` distribution as the finding. The drift-prone part is NOT the name enumeration
(`_registry.public_add_star()` already single-sources that) but the **per-aggregator call convention** —
`tests/sb360/_calls.py`'s adapters + `_registry._adapters()`'s `ADAPTER_MAP` — which is exactly the layer
that already silent-emptied once (`add_visible_area_coverage` unregistered → `generic` `TypeError` swallowed
to `cols=()` at the real swallow `_regenerate.py:137-138`; the incident is *narrated* at `_registry.py:73-79`
but that is the comment, not the swallow). A driver that re-implements that layer forks the already-bitten
machinery; a `scripts/`→`tests/` import to reuse it is backwards layering. **Resolution: MOVE the adapter
bodies + `ADAPTER_MAP` into `scripts/_sb_battery.py` (verified fixture-independent — `_calls.py` imports only
`functools`/`inspect`/`numpy`/`pandas`/`silly_kicks.tracking` + `xthreat` lazily, no `_fixture`/axis
machinery), leave `tests/sb360/_calls.py` a re-export shim so the committed `_entries/*.py` round-trip stays
BYTE-IDENTICAL, and expose ONE `call_aggregator(name, …)`** that both the audit (`_registry._adapters()`
returns `ADAPTER_MAP`) and the driver resolve. Layering is correct (`tests`→`scripts`; `scripts` imports no
`tests`). The adapter-coverage anti-rot ALREADY EXISTS (`tests/sb360/test_registry_surface.py:174-192`) and now
covers the driver too, since it shares the map. Reviewer's dropped-`goal_map` note is honoured by construction:
`generic` forwards `links`/`home_team_id` only where `inspect.signature` accepts them and threads NO uniform
`goal_map`.

**Two invariants that keep `scripts/_sb_battery.py` a clean leaf (round-5, Claim 2):**
1. **`scripts/_sb_battery.py` imports ZERO `tests.sb360` modules** — that single property keeps it a leaf and the layering one-directional. Add it as an explicit constraint AND an import-graph assertion (`test_sb_battery.py`: parse the module's AST, assert no `tests` import).
2. **Leave the `_registry.py:83-89` lazy-init untouched** (the `ADAPTERS: dict = {}` + deferred `_init_adapters()`). Its cycle comment ("`_calls` imports this module") reads stale, but the real trigger is the `_entries/*.py → _registry._adapters() → _calls`/`_sb_battery` path; the move neither needs nor should touch it (Chesterton's Fence). `_adapters()` changes ONLY what it returns.

**Files:**
- Create: `scripts/_sb_battery.py` (adapter bodies + `ADAPTER_MAP` MOVED from `tests/sb360/`; `registered_add_star_aggregators`, `call_aggregator`, `run_add_star_battery`)
- Modify: `tests/sb360/_calls.py` (→ re-export shim), `tests/sb360/_registry.py` (`_adapters()` returns `scripts._sb_battery.ADAPTER_MAP`)
- Create: `scripts/validate_sb360_licensed_corpus.py`
- Create: `tests/scripts/test_sb_battery.py`, `tests/scripts/test_validate_sb360_licensed_corpus.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (enroll the driver in `ARTIFACT_DRIVERS`)
- Reference (read): `scripts/build_sb360_coverage.py:87/390/399-403` (gitignored `DEFAULT_SHARD_ROOT`, provenance stamping), `scripts/_driver.py` (`for_each`, `reconcile`), `tests/sb360/_calls.py` (the adapters to move), `tests/sb360/_registry.py:43-89` (`_adapters()`/`ADAPTERS`), `:293-295` (`public_add_star`), `tests/sb360/test_registry_surface.py:174-192` (the existing adapter-coverage guard).

**Interfaces:**
- Produces (in `_sb_battery.py`): `registered_add_star_aggregators() -> tuple[str, ...]` (`tuple(sorted(n for n in tracking.__all__ if n.startswith("add_")))` — the SAME predicate as `_registry.public_add_star()`); `ADAPTER_MAP: dict[str, Callable]`; `call_aggregator(name, actions, frames, links, home_team_id) -> pd.DataFrame` (`ADAPTER_MAP.get(name, generic)(getattr(T, name))(actions, frames, links, home_team_id)` — the EXACT `_regenerate.py:132-135` resolution); `run_add_star_battery(actions, frames, *, links=None, home_team_id) -> dict[str, pd.DataFrame | str]` — loops `call_aggregator` over `registered_add_star_aggregators()`, returning each aggregator's ADDED columns OR a `"raises: <exc>"` marker (a real-freeze-frame raise is a RESULT, mirroring `_regenerate.py`'s `probe_failures`; NO uniform `goal_map`).
- `measure_match(match_id, actions, frames, visible_area, home) -> pd.DataFrame` — three SEGREGATED measurements (see Step 4): the battery's per-column VERDICT distribution (not values), the three count features' real-`visible_area` `observed_source`/`fraction`, and the explicit real-`visible_area` pitch-coverage (`add_visible_area_coverage`) distribution.
- `main(argv=None)` with `for_each`, an injectable `--shard-root` (default the gitignored `DEFAULT_SHARD_ROOT` mirror), aggregate to `docs/research/sb360_licensed_coverage/`, provenance stamped.

- [ ] **Step 1: Write the shared-call-convention test (both consumers resolve ONE map).** Assert (a) `set(registered_add_star_aggregators()) == _registry.public_add_star()`; (b) `_registry._adapters() == scripts._sb_battery.ADAPTER_MAP` (content-equal; `_adapters()` returns a fresh copy so the audit can't mutate the shared map) AND every value is a function object FROM `scripts._sb_battery` (`fn.__module__ == "scripts._sb_battery"`) — the de-fork proof that the audit resolves the moved adapters, not a stale local copy; (c) `call_aggregator("add_action_context", …)` on the open-360 snapshot fixture returns a non-empty added-column set; (d) **adapter coverage:** every `registered_add_star_aggregators()` name resolves through `call_aggregator` on the fixture WITHOUT a swallowed `TypeError` (emits ≥1 column or a recorded `raises:` marker — never a silent `cols=()`), the driver-side mirror of `test_registry_surface.py:174-192`. Run: `python -m pytest tests/scripts/test_sb_battery.py -v` → FAIL (module absent), implement the MOVE + shim, re-run the FULL sb360 suite to prove the round-trip is byte-identical, then PASS.

- [ ] **Step 2: Red-first leak test — no per-match licensed row under `docs/research/`** — via the injectable `--shard-root` so it is hermetic (no monkeypatching a module global).

```python
def test_shards_go_to_gitignored_root_not_docs_research(tmp_path):
    import scripts.validate_sb360_licensed_corpus as drv
    shard_root = tmp_path / "gitignored_shards"
    out = tmp_path / "docs" / "research" / "sb360_licensed_coverage"
    drv.main(["--shard-root", str(shard_root), "--out", str(out), "--fixture-only"])
    leaked = list((tmp_path / "docs").rglob("*.parquet")) + list((tmp_path / "docs").rglob("*shard*"))
    assert not leaked, f"per-match rows under docs/research/: {leaked}"
    assert list(shard_root.rglob("*.parquet")), "shards must land under the injected gitignored root"
```

- [ ] **Step 3: Run, expect FAIL** (driver absent). Run: `python -m pytest tests/scripts/test_validate_sb360_licensed_corpus.py -k gitignored -v`.

- [ ] **Step 4: Implement `measure_match` + `main()`.** THREE segregated measurements, kept apart on purpose (round-5, Claim 3):
  1. **Battery → VERDICTS ONLY, never values.** `run_add_star_battery(actions, frames, links=links, home_team_id=home)` uses the shared adapters' known-safe EXERCISE inputs (synthetic xt/xg, and a fixed half-pitch for `add_visible_area_coverage` — `_calls.py:213`), so its per-column NUMBERS are synthetic-input hybrids, NOT corpus measurements. `measure_match` records only the structural verdict per column — `works`/`honest_nan`/`silent_degrade`/`raises` (the ADR-053 vocabulary) — comparable to the synthetic-fixture audit. It does **not** tabulate battery values: a fed-the-half-pitch `visible_area_fraction ≈ 0.5` on every action reads as "SB360 observes half the pitch," the exact coverage-denominator-as-signal trap (ADR-042).
  2. **Count-feature companions → REAL `visible_area`.** `add_action_context(actions, frames, visible_area=<real>)` → the three features' `observed_source`/`fraction` distributions over the triangle/disks (spec §8.1).
  3. **Pitch-level coverage → REAL `visible_area`, explicit and segregated.** `add_visible_area_coverage(actions, visible_area=<real>, links=links)` → the real `visible_area_fraction`/`observed_pitch_fraction`/`visible_area_source` distribution over the n=30 corpus. This is the honest counterpart to (1) and closes spec §8.1's pitch-observed-fraction deliverable, which NEITHER the synthetic battery nor the three count features measure — it would otherwise fall through the gap.

  Also emit the roster/per-keeper resolution rate. `main()` flags: `--shard-root` (default the gitignored `DEFAULT_SHARD_ROOT` mirror), `--out` (aggregate dir), `--allow-dirty` (dev), and `--fixture-only` (run the committed open-360 slim fixture, no pining/network — the CI-reachable path the leak test drives). `for_each` writes shards to `--shard-root`; `reconcile` writes ONLY the aggregate under `--out`; `require_clean_tree`+`git_provenance` in `main()`, stamp `run_commit`/`run_tree_dirty`.

- [ ] **Step 5: Add the ASCII + argparse guard tests + the driver-shape test on the OPEN fixture.** Run: `python -m pytest tests/scripts/test_validate_sb360_licensed_corpus.py -m "not e2e" -v` → PASS.

- [ ] **Step 6: Enroll in `ARTIFACT_DRIVERS`** (`test_provenance_wiring.py`) and run it → PASS.

- [ ] **Step 7: (owner-run, clean tree, DGX/local) full 30-match run** producing `docs/research/sb360_licensed_coverage/` (aggregate only). Paste the coverage + degrade-distribution summary. Confirm no per-match shard is staged for commit (`git status`).

---

### Task 6: ADR-053 observed-region audit axis + re-adjudicate the three features

> **EXECUTION DECISION (2026-08-17): the observed-region axis was NOT built — it is VACUOUS, and a
> documented scope note replaces it.** Measured: the SB360 companions depend on the polygon + action
> geometry, not on kinematics or roster, so on the full-coverage audit fixture BOTH legs
> (`build_leg_a` vs `build_leg_b`) come out byte-identical → every verdict would be `identical → works`.
> A two-leg axis that records `works` without ever exercising partial visibility is the "coverage
> denominator masquerading as a signal" / "a gate that certifies the failure it catches is worse than
> none" trap the codebase names elsewhere. The companions ARE verified, from both sides, where it is
> meaningful — `test_visibility.py`, `test_add_action_context.py`, and the licensed-corpus driver
> (all five degradation tokens observed on live matches). The scope note lives in
> `tests/sb360/_registry.py::audited_surface` (a maintainer tempted to add the axis meets the
> reasoning first). This REINTERPRETS §9 of the spec, so it is recorded in ADR-062 and
> routed back to review. The original mechanism design below is preserved as the rejected approach.

**The call-convention de-fork already landed in Task 5** (adapters + `ADAPTER_MAP` moved to `scripts/_sb_battery.py`, `_calls.py` a shim, `_registry._adapters()` returns the shared map). Task 6 is the ADR-053 audit-axis deliverable ONLY. The `_regenerate.py:130` inline enumeration and `_registry.public_add_star()` are a pre-existing byte-equal internal pair guarded by `test_registry_surface.py` — Chesterton's Fence, left untouched (not this cycle's fork).

**The axis MECHANISM is what preserves byte-identity, and it is load-bearing (round-5, Claim 1).** The observed-region axis must be a **new optional `Sb360Entry` field** (`observed_region: dict[str, AxisVerdict] = field(default_factory=dict)`, mirroring `visibility` at `_registry.py:128`), emitted by `_regenerate` through a **dedicated conditional block** (write `observed_region={…}` ONLY when non-empty, i.e. only for the three count features), and yielded by `iter_verdicts` only when present. It must **NOT** be added to `AXES` (`_regenerate.py:39-44`): `AXES` drives both the compute loop (`:143`) and the emitted `visibility={}` block (`:194`, over `AXES[1:]`), so an `AXES` entry would grow EVERY entry's block → all 486 verdicts regenerate → byte-identity breaks wholesale, AND it would mis-key the new axis under `visibility` (the R2-D collision). Scope the observation to a loop over the three count features only. With this, the other ~483 verdict blocks emit byte-identically (no new kwarg appears in their `_entry(…)` calls); only the three count features' blocks grow. **Reading cannot close this — the Step-3 round-trip diff is the gate.**

**Files:**
- Modify: the ADR-053 audit harness (`tests/sb360/` — `_registry.py` (new field + `iter_verdicts` + `_entry`), `_regenerate.py` (conditional emission), `_adjudicate.py`, the three count features' `_entries/_context.py` block)
- Reference (read): ADR-053:59-67 (the two-axis adjudication; the existing axis named "visibility" = roster ablation), `tests/sb360/_registry.py:106-138` (`Sb360Entry`/`AxisVerdict`), `:225-249` (`_entry`), `:252-263` (`iter_verdicts` — THE single verdict-iteration seam a new axis must route through), `_regenerate.py:39-44` (`AXES` — do NOT extend), `:190-207` (the emitted block).

- [ ] **Step 1: Add the observed-region axis as a new optional field** (NOT an `AXES` entry; NOT "visibility" — that name is roster ablation, R2-D). Add `observed_region` to `Sb360Entry` + `_entry(…)`; teach `iter_verdicts` to yield `("observed_region", <region>, col, v)` only when the field is non-empty; add the conditional emission block to `_regenerate` (write it only for entries whose `observed_region` is non-empty). Compute the observations by a scoped loop over the three count features under a supplied `visible_area` — not the global `for axis, roster in AXES` sweep.
- [ ] **Step 2: Re-derive the machine observation** for the three features' new columns under the observed-region axis and **write the human adjudication + rationale** per column (the ADR-053 discipline). **Back up `tests/sb360/_entries/*.py` first** (regenerate is NOT idempotent).
- [ ] **Step 3: Run the SB360 registry tests, expect PASS — and treat the round-trip diff as the byte-identity GATE.** Run `_regenerate` on the backed-up tree and diff against the pre-regenerate `_entries/*.py`: assert the ONLY change is the `add_action_context` entry in `_context.py` gaining an `observed_region={…}` block (the companions are emitted by that one aggregator, R2-A) and every other entry — in `_context.py` and every other family file — is byte-identical. A diff anywhere else means the axis leaked into the global path (an `AXES` touch) — the failure this mechanism exists to prevent.

---

### Task 7: Full open-match E2E (the committed integration backstop)

**Files:**
- Create/extend: `tests/scripts/test_sb360_open_e2e.py` (`@pytest.mark.e2e`)

- [ ] **Step 1: Write the full-chain E2E** (`@e2e`, `importorskip("statsbombpy")` or a committed open-360 fixture): `load_statsbomb_matches`(open match) → `add_action_context(…, visible_area=…)`, asserting: real primary counts; `observed_fraction ∈ [0,1] ∪ {NaN}`; `observed_source` ⊆ the closed set ∪ `unlinked`; at least one partially-observed action (`fraction ∈ (0,1)`); **the honest-degradation vocabulary is genuinely exercised — at least one row carries a token from `{no_polygon, unlinked, degenerate_region}`** (E2E-2: a run whose every source is `observed` would pass a subset check vacuously while hiding a classifier that never degrades; goalkick frame coverage at 32.6% (the repo's authoritative artifact, `docs/research/sb360_coverage/coverage.md:54` = 84/258; the `build_sb360_coverage.py:231` inline "23.3%" is the older 9-match measurement) and per-frame `visible_area` gaps make `unlinked`/`no_polygon` near-certain on a real match); the primary columns match a `visible_area`-absent run. This is the only committed backstop for the wiring (licensed matches can't be in CI).
- [ ] **Step 2: (owner-run) run it, paste the result.**

---

### Task 8: Final CI-faithful verification

- [ ] **Step 1: Full non-e2e suite on `.venv` (3.10)** — `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`. Paste summary.
- [ ] **Step 2: Full non-e2e suite on `.venv312` (3.12)** — same.
- [ ] **Step 3: Lint + types at CI scope** — `ruff check` / `ruff format --check` on `silly_kicks/ tests/ scripts/`; `pyright`. Paste exit-coded output.
- [ ] **Step 4: Confirm C4 count unchanged** (no new action-coupled aggregator; `add_action_context` gains a kwarg, not a new aggregator) — coordinate the SB360-in-pining C4 prose with the keeper-box session (§10) rather than duplicating it.

---

## Self-review — spec coverage

- §4 loader path (non-breaking `_build_match` widen + explicit `statsbomb` dispatch, `build_statsbomb_match`, `load_statsbomb_matches`, fidelity, roster) → Task 2. ✅
- §4 flattener as a real 3-way EXTRACT-and-unify de-fork (H5/R2-E; P1: reference is `_adapt_events`, not `sb.events(fmt="dataframe")`) → Task 1. ✅
- §7 classifier + feature-level source constant (H2/R2-B/R2-C) → Task 3. ✅
- §7 companions on `add_action_context` only, shared `_polygons_by_action` join, inscribed disks, unlinked overlay, NaN-distance → `degenerate_region`, orientation (H1/R2-A/R2-B/R2-F/R2-G; P3) → Task 4. ✅
- §8 validation driver, leak-safe shards via injectable `--shard-root`, single-sourced CALL CONVENTION (adapters + `ADAPTER_MAP` moved to `scripts/_sb_battery.py`, `_calls.py` a shim, leaf-invariant asserted; battery = VERDICTS ONLY, real-`visible_area` pitch coverage measured explicitly — round-5 Claim 3), provenance (H4/B3/M3) → Task 5. ✅
- §9 ADR-053 observed-region axis as a NEW OPTIONAL FIELD with conditional emission (byte-identity mechanism, NOT an `AXES` entry — round-5 Claim 1) + `iter_verdicts` keying + round-trip diff gate (H4/R2-D) → Task 6. ✅
- §9 open-match E2E with the honest-degradation vocabulary exercised (L1; E2E-2) → Task 7. ✅
- Global constraints (additive gates, no runtime dep, licensed-data discipline, dual-major, pyright scripts/ scope) → Global Constraints + per-task red-first gates. ✅
