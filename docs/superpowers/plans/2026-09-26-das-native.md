# Native DAS (drop `accessible-space`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task, **inline in one session** (no subagents). Steps use checkbox (`- [ ]`) syntax.
> **Model routing (owner rule): this plan must NOT be executed by Opus 5.5.** It runs on the owner's chosen
> implementation model.

**Goal:** Replace the `accessible-space` 2.0.15 adapter behind TF-28 with a native, parity-proven,
materially faster DAS engine (numpy port + fused numba kernel), fix the proven reference defects as
documented divergences, remove the `player_id = "ball"` sentinel, and drop the dependency from every runtime
and CI path.

**Architecture:** hexagonal — `_das_pack` (pandas → ragged float64 arrays, input port) → `_das_engine`
(reference-order numpy engine + dispatch) / `_das_numba` (fused `@njit` adapter) → `_das` (public facade,
output port). Direction comes from `GoalMap` (ADR-055). Periodic angular quadrature is the default
(ADR-108); the reference quadrature survives only as an internal parity-gate mode. Two commits, each behind
an explicit owner-approval gate; the DGX artifact runs happen between them on the committed tree.

**Tech Stack:** numpy, pandas (2 and 3), scipy (`cumulative_trapezoid`), numba (optional `[numba]`),
pytest (+ `pytest-benchmark`), the repo seams `group_rows`, `id_compat`, `resolve_defended_goals`,
`scripts/_driver.py`, `_provenance`.

**Spec:** `docs/superpowers/specs/2026-09-26-das-native-design.md` (APPROVED r2). Reviews:
`D:\Development\_reviews\2026-09-26-das-native-spec.md`, `...-spec-r2.md`. Read the spec in full before
Task 1; section references below (§x.y) point into it.

## Global Constraints

- **Gold standard; scope/breaking are not constraints** (owner). Fix at the correct seam; no workarounds.
  Nothing is deferred, dropped or simplified without explicit owner approval — surface and ask.
- **Parity numerics (spec §6.4):** float64 everywhere; no `fastmath`; the numpy engine follows the
  reference's floating-point order exactly (Appendix A). Gates: numpy vs golden `rtol = atol = 1e-12`;
  numba vs golden and vs numpy `rtol = atol = 1e-10`; identical finite masks; values asserted **finite AND
  close**, never "did not crash".
- **Float32 boundary (ADR-106):** `_das_pack` upcasts `x`, `y`, `vx`, `vy` with
  `np.asarray(..., dtype=np.float64)`; the numba kernels declare explicit float64 signatures. DAS never
  writes back onto a coordinate column; it only adds new float64 columns.
- **No `player_id` write anywhere in DAS.** The ball is found by the truthy `is_ball` mask.
- **`DasUnscoreableError` is the ONLY degradable DAS exception; `DAS_SOURCE_VALUES` stays five tokens**
  (ADR-043). Do not widen either.
- **Direction never from team identity** (ADR-051); unresolved direction is a NaN value.
- **All DAS `@njit` code lives in `silly_kicks/tracking/_das_numba.py`** (covered by the existing CI
  numba cache key, spec §6.1).
- **Commits: exactly two, each with an explicit owner-approval gate immediately before it.** No
  micro-commits, no per-task commits, no push/tag/publish without explicit approval. Present the exact diff
  / file list and wait for a yes.
- **No version number until commit-prep** (single-sourced `silly_kicks/_version.py`). ADRs: ADR-107, ADR-108.
- **Artifact drivers refuse a dirty tree** (`require_clean_tree(git_provenance())`); untracked files count
  as dirty (`_provenance`). DGX runs happen only on the committed commit-1 tree, owner-run.
- **Never `pip install` into `.venv`.** `.venv` already carries `accessible-space` 2.0.15 for golden
  generation; verify the version, do not reinstall. Build a separate pandas-3 venv (py3.13 + pandas 3.x) for
  the pandas-3 leg.
- **Lint at CI scope** (`silly_kicks/ tests/ scripts/`); bare `pyright`; mirror ci.yml's pytest invocation;
  `tests/tracking/` runs need `--benchmark-skip`.
- Every new `warnings.warn(..., stacklevel=2)`. Dtype-safe id comparisons through `silly_kicks.id_compat`
  only. `group_rows` callers register in `tests/_scale_guarded.py::SCALE_GUARDED`.
- **Branch:** `feat/das-native` (no worktree). The rebase onto `main` happens after F1b merges, with owner
  approval for the force-push.

## Spec clarifications (owner-ratified 2026-09-26 after plan review DAS-PLAN-01/02; synced into spec §3, §6.6, §6.7)

- **SC-1 (spec §6.6, paired-leg moved set).** The moved rows are **derived**, not caller-declared: a row is
  "moved" iff any of its `x`, `y`, `vx`, `vy` differs bitwise (NaN-aware) between the legs. Every other
  column (`game_id`, `period_id`, `frame_id`, `player_id`, `team_id`, `is_ball`, `team_in_possession`, the
  carrier column) and the whole ball row must agree exactly on every row, else `ValueError`. Sharing is
  therefore applied only to bit-identical rows, which makes the ADR-043 landmine structurally impossible
  rather than guarded, and needs no API change to the public `gkdv.delta_das_batch`.
- **SC-2 (spec §6.7, carrier column).** The carrier column (`ball_carrier_player_id` by default) must be
  constant within a frame (NaN-aware); a frame where it varies raises `ValueError`, the same rule as
  D-POSSVAR (the reference silently took the first row in input order, the D-PASSER defect class).

---

## PHASE A — COMMIT 1 (native engine, migration, docs)

### Task 1: Golden reference fixture + generator (the oracle, RED baseline)

**Files:**
- Create: `tests/tracking/_fixtures/das_golden/_generate.py`
- Create (generated, committed): `tests/tracking/_fixtures/das_golden/{scenes_frames.csv, scenes_passes.csv,
  reference_das_team.csv, reference_das_player.csv, reference_xc.csv, reference_errors.json, metadata.json,
  SHA256SUMS}`
- Create: `tests/tracking/_fixtures/das_golden/.gitattributes` (`* binary`)
- Create: `tests/tracking/_das_golden.py` (loader module; importable as `tests.tracking._das_golden`)
- Create: `tests/tracking/test_das_golden_fixture.py`
- Modify: `pyproject.toml` (add dev-only extra `das-reference = ["accessible-space==2.0.15"]`)

**Interfaces:**
- Produces, in `tests/tracking/_das_golden.py`: `GOLDEN_DIR: Path`, `GOLDEN_FILES: tuple[str, ...]`,
  `NORMAL_SCENES: tuple[str, ...]` (S01–S12c), and `load_golden() -> GoldenFixture` where `GoldenFixture`
  exposes `frames_for(scene) -> pd.DataFrame` and `passes_for(scene) -> pd.DataFrame` (dtypes re-applied from
  `metadata.json`), `reference_for(scene) -> dict[str, np.ndarray]` (`team_as`, `team_das`, `player_as`,
  `player_das`, ordered by frame key then player id), `das_team: pd.DataFrame`, `xc: pd.DataFrame`,
  `errors: dict`, and `metadata: dict`.

- [ ] **Step 1: Confirm the oracle.** `.venv/Scripts/python -c "import accessible_space, importlib.metadata as m; print(m.version('accessible_space'))"` must print `2.0.15`. Do not install anything into `.venv`.
- [ ] **Step 2: Write the generator.** It builds deterministic scenes (fixed RNG seeds) and calls the
  library **directly** (`accessible_space.get_dangerous_accessible_space` for team AS/DAS,
  `get_individual_dangerous_accessible_space` for per-player, `get_expected_pass_completion` for xC). It
  must not import `silly_kicks.tracking._das` (that module is rewritten later). It prepares the library's
  input itself: float64 coordinates shifted to the centred pitch (`x − 52.5`, `y − 34.0`, computed in
  float64), ids cast to numpy `object`, the library's `ball_player_id` convention in its own copy, and the
  scene's known direction passed as a numeric `attacking_direction_col` (no inference). **Normal scenes are
  simulated one `(game_id, period_id)` at a time** so the library's `frame_id`-only keying cannot conflate
  them. Scenes (each with 3–6 frames, 11 v 11 + ball unless stated):

  | Scene | Content |
  |---|---|
  | S01 | home in possession attacking +x, carrier onside |
  | S02 | away in possession attacking −x |
  | S03 | two periods, disjoint frame ids, direction flips at half-time |
  | S04 | two games, disjoint frame ids |
  | S05 | active offside: an attacker beyond the second-last defender, ahead of the ball, in the opponent half |
  | S06 | carrier (passer) beyond the offside line (exclusion matters) |
  | S07 | ball off the pitch (`y = −0.5`) |
  | S08 | one player with NaN `x`/`y` |
  | S09 | one player with NaN `vx`/`vy`, one frame inside and one outside the 5 m tolerance of the trajectory |
  | S10 | ragged player sets with substitutes (union of 26 players, 20–22 per frame) |
  | S11 | possession switches between frames |
  | S12a/b/c | S01 geometry with `team_id`/`player_id` as Int64, object, category |
  | M01–M03 | point reflections (`x → 105 − x`, `y → 68 − y`, `v → −v`, direction flipped) of S01, S05, S10 |
  | X01 | 12 xC passes (varied angles/lengths, both teams) on S01/S02/S10 frames |

  **Divergence scenes** are run through the library in the defect-exhibiting way and their outputs (or
  exception type + message) recorded: V-KEY (same `frame_id` in two periods, one call), V-BALLNAN, V-OFF
  (one defender), V-PASSER (rows reverse-ordered, passer beyond the line), V-POSSABSENT, V-DUP (duplicate
  player row), V-MULTIBALL (two ball rows), V-DIRVAL (direction 0.5), V-POSSVAR, V-XC-FRAME (pass frame
  absent), V-XC-TEAM (pass team absent). A library exception is recorded in `reference_errors.json` as
  `{scene_id: {"type": ..., "message": ...}}`.
- [ ] **Step 3: Deterministic serialisation.** CSV with `float_format="%.17g"` (exact float64 round-trip),
  `lineterminator="\n"`, rows sorted by their key, UTF-8 without BOM. `metadata.json` (sorted keys, indent
  2) records: accessible-space version; SHA256 of `accessible_space/{core,interface,utility,motion_models}.py`;
  every module-level `_DEFAULT_*` constant of `core.py` and `interface.py`; numpy/scipy/pandas versions; the
  per-column dtype map; the scene catalogue; the generator's own SHA256. `SHA256SUMS` covers every file.
- [ ] **Step 4: Run the generator** into the fixture directory with `.venv/Scripts/python
  tests/tracking/_fixtures/das_golden/_generate.py --out tests/tracking/_fixtures/das_golden`.
- [ ] **Step 5: Write `test_das_golden_fixture.py`:**

```python
def test_golden_checksums_match():
    root = GOLDEN_DIR
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

def test_golden_regenerates_byte_for_byte(tmp_path):
    pytest.importorskip("accessible_space")
    if importlib.metadata.version("accessible_space") != "2.0.15":
        pytest.skip("golden is pinned to accessible-space 2.0.15")
    subprocess.run([sys.executable, str(GOLDEN_DIR / "_generate.py"), "--out", str(tmp_path)], check=True)
    for name in GOLDEN_FILES:
        assert (tmp_path / name).read_bytes() == (GOLDEN_DIR / name).read_bytes(), name

def test_golden_is_non_vacuous():
    g = load_golden()
    assert np.isfinite(g.das_team["DAS"]).sum() >= 60          # normal scenes carry real values
    assert (g.das_team.loc[g.das_team.scene_id == "V-BALLNAN", "DAS"] == 0.0).any()   # defect recorded
```

- [ ] **Step 6: Run** `python -m pytest tests/tracking/test_das_golden_fixture.py -v` — PASS (this task's
  deliverable is the oracle itself).

### Task 2: `PassSimParams`, profiles and simulation grids

**Files:**
- Create: `silly_kicks/tracking/_das_params.py`
- Test: `tests/tracking/test_das_params.py`

**Interfaces:**
- Produces:

```python
Quadrature = Literal["periodic", "reference"]

@dataclass(frozen=True)
class PassSimParams:
    n_angles: int; phi_offset: float; n_v0: int; v0_min: float; v0_max: float
    radial_gridsize: float; pass_start_location_offset: float; time_offset_ball: float
    b0: float; b1: float; player_velocity: float; inertial_seconds: float; tol_distance: float
    use_max: bool; v_max: float; a_max: float; factor: float; factor2: float
    normalize: bool; respect_offside: bool; exclude_passer: bool; danger_weight: float
    quadrature: Quadrature = "periodic"

DAS_PARAMS: PassSimParams   # spec §6.3 DAS profile
XC_PARAMS: PassSimParams    # spec §6.3 xC profile (n_angles unused: one angle per pass)

@dataclass(frozen=True)
class SimGrids:
    phi: np.ndarray; cos_phi: np.ndarray; sin_phi: np.ndarray        # (Φ,)
    v0: np.ndarray                                                   # (V,)
    d: np.ndarray                                                    # (T,) radial distances
    t_ball: np.ndarray                                               # (V, T)
    dt: np.ndarray                                                   # (V, T-1) = np.diff(t_ball, axis=-1)
    dt0: np.ndarray                                                  # (V,)  = t_ball[:, 1] - t_ball[:, 0]
    rate_divisor: np.ndarray                                         # (V,)  = factor * v0 ** (-factor2)
    dr: np.ndarray                                                   # (T,)
    d_area: np.ndarray                                               # (Φ, T) quadrature-dependent dA

def simulation_grids(params: PassSimParams) -> SimGrids   # functools.lru_cache(maxsize=16)
```

  Model-fixed choices (both reference profiles use them) are **not** parameters: approximate two-point TTA,
  `keep_inertial_velocity=True`, the efficient sigmoid, possibility (not probability) densities.

- [ ] **Step 1: Failing tests** — profiles equal the golden-recorded constants; validation from both sides;
  grid shapes; quadrature weights.

```python
def test_das_profile_matches_reference_constants():
    meta = load_golden().metadata["accessible_space_defaults"]
    assert DAS_PARAMS.b1 == meta["interface._DEFAULT_B1_FOR_DAS"] == -2000
    assert DAS_PARAMS.inertial_seconds == meta["interface._DEFAULT_INERTIAL_SECONDS_FOR_DAS"]
    assert DAS_PARAMS.factor == meta["core._DEFAULT_FACTOR"]
    # ... every field of both profiles, one assertion each, mapped in a table in the test

@pytest.mark.parametrize("field,bad", [("n_angles", 0), ("v0_min", 0.0), ("radial_gridsize", -1.0),
                                       ("danger_weight", 0.0), ("quadrature", "trapezoid")])
def test_invalid_params_raise(field, bad):
    with pytest.raises(ValueError):
        dataclasses.replace(DAS_PARAMS, **{field: bad})

def test_das_radial_grid_has_46_points():
    assert simulation_grids(DAS_PARAMS).d.shape == (46,)

def test_quadrature_weights_differ_only_at_the_two_end_rays():
    per = simulation_grids(DAS_PARAMS).d_area
    ref = simulation_grids(dataclasses.replace(DAS_PARAMS, quadrature="reference")).d_area
    diff = per != ref
    assert diff[0].all() and diff[-1].all() and not diff[1:-1].any()
    assert np.all(per[[0, -1]] > ref[[0, -1]])            # periodic restores the missing half wedges
```

- [ ] **Step 2: Run — FAIL** (module absent).
- [ ] **Step 3: Implement** per Appendix A.1 (grid construction verbatim). Periodic end rays use the
  wrap-around midpoints `phi_lower[0] = (phi[-1] − 2π + phi[0]) / 2`, `phi_upper[-1] = (phi[-1] + phi[0] +
  2π) / 2`; interior bounds are the reference's midpoints, so interior `d_area` is bitwise identical across
  modes. Keep `dr` and `d_area` **separate** (the reference multiplies `(field · dr) · dA`; a premultiplied
  weight would change the association).
- [ ] **Step 4: Run — PASS.**

### Task 3: `_das_pack` — input port, contract, keys, direction, reason codes

**Files:**
- Create: `silly_kicks/tracking/_das_pack.py`
- Modify: `tests/_scale_guarded.py` (register `silly_kicks.tracking._das_pack`)
- Create: `tests/tracking/_das_helpers.py` (shared test helpers, grown by later tasks — see below)
- Test: `tests/tracking/test_das_pack.py`

**Test helpers (`tests/tracking/_das_helpers.py`), defined once and reused by Tasks 3–12:**
- Task 3 adds pytest fixtures built from golden scene S01 via `load_golden().frames_for("S01")`:
  `frames` (canonical dtypes, direction column `dir`), `frames_float32` (coords/velocities cast to float32),
  `frames_two_periods_same_frame_id` (V-KEY input), and `frames_by_case` (a dict mapping each reason-code case
  name to a minimally mutated copy of `frames`).
- Task 4 adds `pack_golden(scene) -> PackedFrames` (packs a golden scene with its `dir` column) and
  `run_engine(frames, params, *, engine) -> dict[str, np.ndarray]` (packs with `attacking_direction_col="dir"`,
  calls `compute_das`, returns `team_as`, `team_das`, `player_as`, `player_das` aligned to the golden
  reference ordering by frame key and player id).
- Task 5 adds `kernel_args(dtype=np.float64) -> tuple` (the positional argument tuple for
  `das_frames_serial` built from `pack_golden("S01")`, with float arrays cast to `dtype`).
- Task 7 adds `ghost_pair_from_golden(scene, *, keeper_shift_m=3.0) -> tuple[pd.DataFrame, pd.DataFrame]`
  (actual frames and a copy with the defending keeper's `x` moved by `keeper_shift_m` towards midfield,
  every other column untouched).

**Interfaces:**
- Consumes: `resolve_defended_goals`, `GoalMap.attacked_goal`, `group_rows`, `id_compat.ids_equal` /
  `canonical_id_series`, `velocity_unavailable_by_design`.
- Produces:

```python
class Reason(IntEnum):
    OK = 0; NO_POSSESSION = 1; NO_BALL = 2; NO_PLAYERS = 3; BALL_NAN = 4
    POSSESSION_TEAM_ABSENT = 5; DIRECTION_UNRESOLVED = 6

@dataclass(frozen=True)
class PackedFrames:
    keys: pd.DataFrame            # one row per frame: game_id, period_id, frame_id (sorted)
    offsets: np.ndarray           # int64 (n_frames + 1,), player rows of frame f = [offsets[f], offsets[f+1])
    px: np.ndarray; py: np.ndarray; pvx: np.ndarray; pvy: np.ndarray   # float64, centred coords
    p_attacking: np.ndarray       # bool, player's team == frame's team in possession (id_compat)
    p_is_passer: np.ndarray       # bool, player id == frame's carrier id (id_compat)
    p_input_pos: np.ndarray       # int64 positions into the input frame
    ball_xy: np.ndarray           # float64 (n_frames, 2), centred
    direction: np.ndarray         # float64 (n_frames,), +1 / -1 / NaN
    reason: np.ndarray            # uint8 (n_frames,) Reason codes
    ball_input_pos: np.ndarray    # int64 (n_frames,), -1 when no ball row

def pack_frames(frames: pd.DataFrame, *, goal_map: GoalMap | None, attacking_direction_col: str | None,
                player_in_possession_col: str | None) -> PackedFrames
```

- [ ] **Step 1: Failing tests** — one per row of spec §6.7 plus keys/offsets/dtypes/purity/sentinel:

```python
def test_duplicate_player_rows_raise(frames):
    dup = pd.concat([frames, frames.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        pack_frames(dup, goal_map=None, attacking_direction_col=None, player_in_possession_col=None)

def test_goal_map_and_direction_col_are_mutually_exclusive(frames): ...
def test_direction_values_must_be_plus_minus_one(frames): ...        # 0.5 -> ValueError
def test_possession_varying_within_frame_raises(frames): ...
def test_carrier_varying_within_frame_raises(frames): ...            # SC-2
def test_two_ball_rows_raise(frames): ...

def test_keys_include_game_and_period(frames_two_periods_same_frame_id):
    p = pack_frames(frames_two_periods_same_frame_id, goal_map=None, attacking_direction_col=None,
                    player_in_possession_col=None)
    assert len(p.keys) == 2                                          # D-KEY: no conflation

def test_packed_kinematics_are_float64_even_from_float32_storage(frames_float32):
    p = pack_frames(frames_float32, goal_map=None, attacking_direction_col=None, player_in_possession_col=None)
    assert p.px.dtype == p.py.dtype == p.pvx.dtype == p.pvy.dtype == np.float64   # vx/vy not covered by the AST gate

def test_input_unmodified_and_player_id_never_written(frames):
    before = frames.copy(deep=True)
    pack_frames(frames, goal_map=None, attacking_direction_col=None, player_in_possession_col=None)
    pd.testing.assert_frame_equal(frames, before)

def test_category_player_id_packs(frames):
    f = frames.assign(player_id=frames["player_id"].astype("category"))
    pack_frames(f, goal_map=None, attacking_direction_col=None, player_in_possession_col=None)  # no raise

@pytest.mark.parametrize("case,reason", [("nan_possession", Reason.NO_POSSESSION), ("no_ball", Reason.NO_BALL),
    ("ball_nan", Reason.BALL_NAN), ("possession_team_absent", Reason.POSSESSION_TEAM_ABSENT),
    ("direction_unresolved", Reason.DIRECTION_UNRESOLVED)])
def test_reason_codes(case, reason, frames_by_case): ...
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** Validation order: required columns (existing messages kept verbatim from
  `_validate_das_inputs`, including the `speed_source` degradable branch raising
  `DasUnscoreableError(das_source=DAS_SOURCE_UNSCOREABLE_FRAME)`), then the new fail-loud checks, then the
  all-NaN possession `DasUnscoreableError` (existing message). Sort with `group_rows` over
  `(game_id, period_id, frame_id)`; within a frame order players by `pd.factorize(<player_id values, category
  unwrapped>, sort=True)` codes (the reference's pivot column order, Appendix A.2). Direction:
  `attacking_direction_col` verbatim when given; else `goal_map` (built with `resolve_defended_goals(frames)`
  when `None`, which requires `is_goalkeeper`) → `+1.0` if `attacked_goal(..., allow_guess=True) == 105.0`,
  `−1.0` if `0.0`, NaN otherwise. Reason priority when several apply: NO_POSSESSION, NO_BALL, NO_PLAYERS,
  BALL_NAN, POSSESSION_TEAM_ABSENT, DIRECTION_UNRESOLVED.
- [ ] **Step 4: Run — PASS.** Also run `tests/test_scale_guard_registry.py` and
  `tests/tracking/test_frame_coord_upcast_gate.py` — PASS.

### Task 4: numpy engine (reference order) + dispatch + `DasResult`

**Files:**
- Create: `silly_kicks/tracking/_das_engine.py`
- Test: `tests/tracking/test_das_engine_parity.py`, `tests/tracking/test_das_invariance.py`

**Interfaces:**
- Consumes: `PackedFrames`, `PassSimParams`, `simulation_grids`.
- Produces:

```python
@dataclass(frozen=True)
class DasResult:
    team_as: np.ndarray; team_das: np.ndarray          # float64 (n_frames,) — in-possession team density
    player_as: np.ndarray; player_das: np.ndarray      # float64 (n_player_rows,) aligned with PackedFrames rows
    reason: np.ndarray                                 # uint8 (n_frames,), copied from PackedFrames

def compute_das(packed: PackedFrames, params: PassSimParams, *, chunk_size: int | None = None,
                n_threads: int | None = None, engine: Literal["auto", "numpy", "numba"] = "auto") -> DasResult
def _numpy_integrand(packed: PackedFrames, params: PassSimParams) -> tuple[np.ndarray, np.ndarray]
    # test-only diagnostic: per-(frame, angle, radial) integrands g_as, g_das BEFORE the area weights
```

  `engine="auto"` → numba when importable and `SILLY_KICKS_DAS_FORCE_NUMPY != "1"`, else numpy. Frames whose
  reason is not `OK` get NaN outputs without being simulated.

- [ ] **Step 1: Failing parity tests** (numpy only in this task):

```python
REF = dataclasses.replace(DAS_PARAMS, quadrature="reference")

@pytest.mark.parametrize("scene", NORMAL_SCENES)
def test_numpy_engine_reproduces_reference(scene):
    g = load_golden()
    got = run_engine(g.frames_for(scene), REF, engine="numpy")       # packs with the scene's direction column
    exp = g.reference_for(scene)
    for col in ("team_as", "team_das", "player_as", "player_das"):
        assert np.array_equal(np.isfinite(got[col]), np.isfinite(exp[col])), col
        np.testing.assert_allclose(got[col], exp[col], rtol=1e-12, atol=1e-12, err_msg=col)

def test_parity_gate_discriminates():
    g = load_golden()
    bumped = dataclasses.replace(REF, b1=REF.b1 * (1 + 1e-9))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(run_engine(g.frames_for("S05"), bumped, engine="numpy")["team_das"],
                                   g.reference_for("S05")["team_das"], rtol=1e-12, atol=1e-12)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(run_engine(g.frames_for("S05"), DAS_PARAMS, engine="numpy")["team_das"],
                                   g.reference_for("S05")["team_das"], rtol=1e-12, atol=1e-12)
```

  Invariance tests (`test_das_invariance.py`): `chunk_size ∈ {None, 1, 7, n_frames}` → `np.array_equal`;
  input rows shuffled → identical outputs keyed by frame/player; games concatenated → equal to per-game calls.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the numpy engine per Appendix A (every expression shape is prescribed there).
  Vectorise over a chunk of frames padded to the chunk's maximum player count with NaN kinematics; loop over
  speeds; call `scipy.integrate.cumulative_trapezoid(y, x=t_ball_row, initial=0, axis=-1)`; reduce with the
  same `np.nansum(np.where(...))` / `np.max` / `np.nanmax` / `np.sum(axis=(-2, -1))` calls as the reference.
  Default `chunk_size` for numpy: see Task 5 Step 5 (measured).
- [ ] **Step 4: Run — PASS** at `max |Δ| = 0.0` expected; if any difference survives, find the
  association that differs from Appendix A before considering anything else. A non-zero residue that cannot
  be eliminated is surfaced to the owner (spec §13), never absorbed by loosening the tolerance.

### Task 5: numba adapter (serial + `prange`), explicit signatures, dispatch

**Files:**
- Create: `silly_kicks/tracking/_das_numba.py`
- Modify: `silly_kicks/tracking/_das_engine.py` (dispatch to numba)
- Test: `tests/tracking/test_das_kernel.py`; extend `test_das_engine_parity.py`, `test_das_invariance.py`
- Create: `tests/tracking/test_das_benchmark.py`

**Interfaces:**
- Produces (module-private, bound lazily from `_das_engine`):
  `das_frames_serial(offsets, px, py, pvx, pvy, p_att, p_pass, ball_xy, direction, frame_ok, <grid arrays>,
  <scalar params>, out_team_as, out_team_das, out_player_as, out_player_das)` and `das_frames_parallel(...)`
  with the identical signature (`@njit(parallel=True)`, `prange` over frames). Explicit float64/int64/bool
  signatures via `numba.types`; `cache=_NUMBA_CACHE` (reuse the pitch-control convention).

- [ ] **Step 1: Failing tests:**

```python
def test_float32_arrays_rejected_by_kernel():
    pytest.importorskip("numba")
    from silly_kicks.tracking import _das_numba
    args = kernel_args(dtype=np.float32)
    with pytest.raises(TypeError):
        _das_numba.das_frames_serial(*args)

def test_das_modules_do_not_eagerly_import_the_kernel():
    code = "import sys, silly_kicks.tracking._das, silly_kicks.tracking._das_engine; " \
           "assert 'silly_kicks.tracking._das_numba' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

@pytest.mark.parametrize("scene", NORMAL_SCENES)
def test_numba_engine_reproduces_reference(scene): ...     # rtol = atol = 1e-10, identical finite masks

def test_serial_and_parallel_kernels_are_byte_identical(): ...
def test_n_threads_restores_numba_thread_count(): ...
def test_force_numpy_env_selects_numpy(monkeypatch): ...
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the fused kernel per spec §6.5 and Appendix A: per frame → offside mask →
  danger/on-pitch per (angle, radial) → per angle: TTA[P, T] → per speed stream along t (compute `a` for
  every player into a length-P scratch, sequential `S_att`/`S_def`, trapezoid update
  `I += (dt * (S_t + S_prev)) / 2.0`, two `exp`, running `max` into `rho[P, T]`) → normalise → accumulate
  team and player sums in the same `(field · dr) · dA` shape. Scratch arrays allocated once per frame
  (O(P·T)). Division where the reference divides.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Measure and set the default `chunk_size` per engine.** With `test_das_benchmark.py`
  (`pytest-benchmark`, `--benchmark-only`, local) time numpy chunks {16, 32, 64, 128} and numba chunks {512,
  1024, 4096, 8192} on 20k synthetic frames; pick the fastest whose tracemalloc working peak stays under
  256 MB; record the table in ADR-107 (Task 13). Record serial and `prange` ms/frame (1, 2, 4, 8, 16 threads)
  for the commit-1 `estimate_das_cost` constants (Task 9).

### Task 6: Quadrature exact relation + mirror (ADR-108)

**Files:**
- Test: `tests/tracking/test_das_quadrature.py`

- [ ] **Step 1: Write the tests:**

```python
def test_das_difference_is_exactly_the_end_ray_weight_difference():
    packed = pack_golden("S01")
    g_as, g_das = _numpy_integrand(packed, DAS_PARAMS)                    # (F, Φ, T), quadrature-independent
    per = compute_das(packed, DAS_PARAMS, engine="numpy")
    ref = compute_das(packed, dataclasses.replace(DAS_PARAMS, quadrature="reference"), engine="numpy")
    grids_p = simulation_grids(DAS_PARAMS)
    grids_r = simulation_grids(dataclasses.replace(DAS_PARAMS, quadrature="reference"))
    delta_w = grids_p.dr * grids_p.d_area - grids_r.dr * grids_r.d_area   # nonzero only on rays 0, n-1
    expected = np.sum(g_das * delta_w, axis=(1, 2))
    np.testing.assert_allclose(per.team_das - ref.team_das, expected, rtol=1e-12, atol=1e-12)

def test_relation_discriminates_an_interior_weight_error(monkeypatch): ...
    # perturb d_area[5, :] by (1 + 1e-9) in a patched grid -> the relation above must fail

@pytest.mark.parametrize("scene,mirror", [("S01", "M01"), ("S05", "M02"), ("S10", "M03")])
def test_periodic_das_is_mirror_invariant(scene, mirror):
    a = run_engine(load_golden().frames_for(scene), DAS_PARAMS, engine="numpy")
    b = run_engine(load_golden().frames_for(mirror), DAS_PARAMS, engine="numpy")
    # F1 (§6.4): RELATIVE rtol=1e-6 -- the arccos opening-angle floor is ~1e-7 rel (S10), so rtol=0
    # would fail there; atol=1e-9 covers the near-zero rows. Matches test_das_quadrature.py.
    np.testing.assert_allclose(a["team_das"], b["team_das"], rtol=1e-6, atol=1e-9)

def test_reference_quadrature_is_not_mirror_invariant():                # the defect, recorded
    g = load_golden()
    assert np.max(np.abs(g.reference_for("S01")["team_das"] - g.reference_for("M01")["team_das"])) > 1.0
```

- [ ] **Step 2: Run — PASS** (Tasks 2 and 4 already implement the quadrature; this task pins it).

### Task 7: Paired-leg kernel (SC-1)

**Files:**
- Modify: `silly_kicks/tracking/_das_pack.py` (`pack_paired`), `_das_engine.py` (`compute_das_paired`),
  `_das_numba.py` (paired serial/parallel kernels)
- Test: `tests/tracking/test_das_paired.py`

**Interfaces:**
- Produces:

```python
def pack_paired(actual: pd.DataFrame, counterfactual: pd.DataFrame, *, goal_map: GoalMap | None,
                attacking_direction_col: str | None, player_in_possession_col: str | None
                ) -> tuple[PackedFrames, PackedFrames, np.ndarray]   # third: bool moved mask per packed row

def compute_das_paired(actual: PackedFrames, counterfactual: PackedFrames, moved: np.ndarray,
                       params: PassSimParams, *, chunk_size=None, n_threads=None, engine="auto"
                       ) -> tuple[DasResult, DasResult]
```

- [ ] **Step 1: Failing tests:**

```python
def test_paired_equals_two_independent_calls_bitwise(engine):
    actual, ghost = ghost_pair_from_golden("S01")          # keeper moved 3 m
    a, b = compute_das_paired(*pack_paired(actual, ghost, goal_map=None, attacking_direction_col="dir",
                                           player_in_possession_col=None), DAS_PARAMS, engine=engine)
    for leg, frames in ((a, actual), (b, ghost)):
        solo = compute_das(pack_frames(frames, goal_map=None, attacking_direction_col="dir",
                                       player_in_possession_col=None), DAS_PARAMS, engine=engine)
        for f in ("team_as", "team_das", "player_as", "player_das"):
            assert np.array_equal(getattr(leg, f), getattr(solo, f), equal_nan=True), f

def test_moved_keeper_changes_das():                       # non-vacuity
    a, b = ...; assert np.nanmax(np.abs(a.team_das - b.team_das)) > 0

@pytest.mark.parametrize("mutation", ["ball_moved", "possession_changed", "carrier_changed",
                                      "row_order_changed", "player_set_changed"])
def test_leg_contract_violations_raise(mutation): ...     # ValueError
```

- [ ] **Step 2: Run — FAIL.** **Step 3: Implement (F2, §6.6; ratified 2026-09-26)** — `pack_paired`
  validates SC-1 and derives `moved` (still used by consumers to name the counterfactual rows), and
  `compute_das_paired` ships as **two calls of the single-leg path** in canonical row order, on BOTH
  engines — correct, validated, offside-correct, and bit-identical to two independent legs by
  construction. The offside-aware sharing kernel (recompute offside per leg, share only status-matching
  rows) is **DROPPED this cycle**: the SC-1 "share every unmoved row" rationale is unsound under the
  offside cross-dependency (a moved row can flip an unmoved row's offside status). **Step 4: Run — PASS.**

### Task 8: Native `get_xc`

**Files:**
- Modify: `_das_pack.py` (`pack_passes`), `_das_engine.py` (`compute_xc`), `_das_numba.py` (xC kernel)
- Test: `tests/tracking/test_xc_native.py`

**Interfaces:**
- Produces: `pack_passes(passes, frames, *, player_in_possession_col=None) -> PackedPasses` (one simulated
  frame per pass: the pass's tracking frame keyed by `(game_id, period_id, frame_id)`, ball = event start
  where finite else the tracking ball, `phi = arctan2(end_y_c − start_y_c, end_x_c − start_x_c)` on centred
  coordinates, attacking = players of the event team, passer = event player; reason codes add
  `PASS_FRAME_MISSING`, `PASS_TEAM_ABSENT`) and `compute_xc(packed, params=XC_PARAMS, *, chunk_size=None,
  n_threads=None, engine="auto") -> np.ndarray` (float64 per pass, NaN where the reason is not OK).

- [ ] **Step 1: Failing tests** — xC golden parity (numpy 1e-12, numba 1e-10) on X01; D-XC-FRAME and
  D-XC-TEAM give NaN plus exactly one aggregated `UserWarning` naming the counts, while
  `reference_errors.json` records the library's `ValueError`; passer absent from its frame → computed.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement** per Appendix A.6. **Step 4: Run — PASS.**

### Task 9: Public facade `_das.py` rewrite

**Files:**
- Modify: `silly_kicks/tracking/_das.py` (rewrite), `silly_kicks/tracking/__init__.py` (exports unchanged;
  verify)
- Modify: `tests/tracking/test_das_cost_guardrail.py`
- Test: `tests/tracking/test_das.py` (retargeted; see Task 12 for the disposition list)

**Interfaces:**
- Produces the spec §6.2 signatures exactly, plus the confined private
  `individual_das_paired(actual, counterfactual, *, goal_map=None, attacking_direction_col=None,
  player_in_possession_col=_DEFAULT_PLAYER_IN_POSSESSION_COL, params=None, chunk_size=None, n_threads=None)
  -> tuple[pd.DataFrame, pd.DataFrame]` (each leg = its input copy + `AS`, `DAS` per player).

- [ ] **Step 1: Failing tests** — output contract: `get_das` adds `AS`/`DAS` (team value broadcast to every
  row of the frame incl. the ball row; NaN on non-OK frames), `get_individual_das` adds per-player values
  (ball row NaN), input unmodified, `player_id` dtype preserved; `**kwargs` / `use_progress_bar` →
  `TypeError`; `params` with `quadrature="reference"` → `ValueError`; `goal_map` + direction column →
  `ValueError`; `warn_cost` on both functions; `DasCostWarning` fires iff `estimate_das_cost(frames) >
  _DAS_COST_WARN_SECONDS` (100.0 s, the wall-time the old 5000-frame × 0.02 s threshold encoded); the
  existing all-unscoreable `UserWarning` fires once with the rewritten message.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** Delete `_to_das_coords`, `_prepare_frames`, `_call_simulation`,
  `_check_das_output_alignment`, `_import_accessible_space`, `_has_simulatable_frame`,
  `_frames_with_ball_and_players`, `_nan_das_result`, `_pin_attacking_direction`, `_COLUMN_MAP`,
  `_suppress_offside_warning` (Chesterton check done in spec §6.7: each existed only for the library seam).
  Keep `DAS_SOURCE_*`, `DasUnscoreableError`, `_resolve_player_in_possession_col`,
  `_warn_no_carrier_once`. `estimate_das_cost(frames, *, n_threads=None)` = scored frames ×
  `_DAS_SECONDS_PER_FRAME[engine]` / effective threads (`n_threads × _PRANGE_EFFICIENCY` when > 1), constants
  from Task 5 Step 5 with a provenance comment. Rewrite the module docstring (no library references; cite
  ADR-107/108/043).
- [ ] **Step 4: Run — PASS.**

### Task 10: `features.py` migration (link restriction, `GoalMap`, vectorised mapping) + registries

**Files:**
- Modify: `silly_kicks/tracking/features.py:2905-3362`
- Modify: `tests/tracking/_mirror_entries/trained_and_das.py` (relative `rtol=1e-6` re-derived, F1;
  `relative_tolerance=True`, `call_with_map` + `gate_c_must_move` for `add_das`) and
  `tests/tracking/_mirror_registry.py` (the `relative_tolerance` field + Gate A rtol branch),
  `tests/test_add_star_purity.py`, `tests/test_enrichment_nan_safety.py`,
  `tests/tracking/conftest_id_dtype.py`, `tests/tracking/test_aggregator_column_liveness.py`,
  `tests/invariants/test_public_id_scalar_registry.py`, `tests/tracking/_xfn_default_lists.py`,
  `tests/tracking/test_frame_aware_xfns_dup_action_id.py`

**Interfaces:**
- Consumes: `pack_frames`, `compute_das`, `_kernels.resolve_frame_ids_by_position`,
  `resolve_defended_goals`.
- Produces: `add_das` / `das_at_action` spec §6.2 signatures; `das_xfns` unchanged protocol;
  `_das_lookup(frames, frame_keys, *, goal_map, attacking_direction_col, chunk_size, n_threads, params) ->
  pd.DataFrame` (columns `game_id, period_id, frame_id, team_id, das_sum`) replacing
  `_precompute_das_lookup`; `_map_das_to_actions` vectorised (merge on canonical keys via
  `id_compat.align_join_keys`, no `iterrows`).

- [ ] **Step 1: Failing tests** — `das_at_action` and `das_xfns` simulate only linked frames (a spy on
  `compute_das` sees `len(packed.keys) == n_linked_frames`); `das_source` mapping per spec §6.7 (every token
  reachable, incl. `team_unresolved`); `add_das(goal_map=…)` threads the map (spy); the per-team sum equals
  the sum of individual player DAS; mirror entry passes at the RELATIVE `rtol=1e-6` (F1, §6.4 — corrects
  the aspirational 1e-9; the periodic engine is mirror-invariant to the `arccos` float floor, ~1e-7 rel);
  Gate C moves the DAS columns when the map is varied; `add_das` passes `warn_cost=False` internally.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** Build `GoalMap` with `resolve_defended_goals(frames)` on the **full** frames
  (unless supplied); resolve frame ids positionally (ADR-020) for every action set (for `das_xfns`, the union
  over the three gamestate slots); pack only those frame keys; aggregate per `(game, period, frame, team)`
  over player rows; map back with a keyed merge. Keep the `DasUnscoreableError` catch sites exactly as
  today (three entry points, same warnings, same `exc.das_source` stamping).
- [ ] **Step 4: Run** the migrated registries and `tests/tracking/test_das*.py` — PASS. Re-derive
  `_DAS_MIRROR_TOL` as a RELATIVE `rtol=1e-6` (F1, §6.4; `relative_tolerance=True` on the entry), measure
  the residual on `canonical_scene()` (~1e-15 rel there; the ~1e-7 floor is scene S10), and rewrite the
  `basis` string with the measured residual + the `arccos`-floor / periodic-quadrature cause (ADR-108).

### Task 11: Consumers — gkdv port, gkdv arms, positioning, calibration, restdefense

**Files:**
- Modify: `silly_kicks/gkdv/_das_port.py`, `silly_kicks/gkdv/_arms.py` (`delta_das_batch`),
  `silly_kicks/gkdv/__init__.py` (docstring), `silly_kicks/positioning/_objectives.py` (`DasObjective`)
- Modify: `tests/gkdv/test_arms.py`, `test_arms_batch.py`, `test_arm_direction_key.py`,
  `test_import_allowlist.py`, `tests/positioning/test_objectives.py`
- Verify unchanged: `silly_kicks/calibration/_features.py`, `silly_kicks/restdefense/_arms.py`,
  `silly_kicks/tracking/_run_features.py` and their suites

- [ ] **Step 1: Failing tests** — `delta_das_batch` builds ONE `GoalMap` from the actual legs and threads
  it into both legs (adapted `test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs`); `team_das_by_frame`
  goes through `individual_das_paired` (spy) and equals the independent-leg result bitwise; the allowlist
  exemption is `_das.individual_das_paired` (count 1) and `_pin_attacking_direction` is gone;
  `DasObjective(goal_map=gm).score(trial)` does not change direction when the trial moves players across
  midfield.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement** (`pin_direction` returns the `GoalMap`; the port's
  docstring states it is the one seam onto native DAS). **Step 4: Run — PASS**, including
  `tests/restdefense/`, `tests/calibration/`, `tests/tracking/test_run_tracking_features.py`.

### Task 12: Legacy suite migration, scale/memory gates, dark-test recovery

**Files:**
- Modify: `tests/tracking/test_das.py`, `test_das_e2e.py`, `test_das_offside.py`,
  `tests/invariants/test_das_invariants.py`
- Create: `tests/tracking/test_das_divergences.py`, `tests/tracking/test_das_scale_memory.py`

- [ ] **Step 1: Record the baseline** before editing: `python -m pytest tests/tracking/test_das.py
  tests/tracking/test_das_e2e.py tests/tracking/test_das_offside.py tests/invariants/test_das_invariants.py
  --collect-only -q` and the `-m "not e2e"` collected count; paste both counts into the Task 15 report.
- [ ] **Step 2: Disposition of `test_das.py` classes** — delete with reason: `TestTodasCoords`
  (`_to_das_coords` deleted), `TestImportGuard` (no optional extra), `TestPrepareFramesNumericPlayerId`
  (replaced by `test_das_pack` id axes), `TestDasOutputAlignmentGuard` (alignment by construction),
  `TestPinAttackingDirection` (replaced by `GoalMap` tests in Task 10/11), the `_call_simulation` cases of
  `TestDasExceptionGracefulDegradation` (library seam gone; keep the `DasUnscoreableError` degrade cases).
  Retarget: `TestInputValidation`, `TestDasTeamAsymmetry`, `TestChunkSizePassthrough` (now asserts
  `chunk_size` reaches `compute_das`), `TestDasXfns`, `TestDasLinkedFrameRestriction`,
  `TestAttackingDirectionCol*`, `TestZeroFrameSubsetDegradesToNaN`, `TestXcZeroFrameSubsetDegradesToNaN`.
- [ ] **Step 3:** remove the `e2e` markers from `test_das_e2e.py`, `test_das_invariants.py` and
  `TestGetDasShapeAlignment` (their reason — library crashes — is gone); port `test_das_offside.py` to the
  native path (keep its onside/offside-carrier semantics).
- [ ] **Step 4:** `test_das_divergences.py` — one test per D-* row of spec §6.8 asserting the native
  behaviour **and** that the golden recorded the reference defect (value or `reference_errors.json` entry).
- [ ] **Step 5:** `test_das_scale_memory.py` — `assert_subquadratic_growth` (from
  `tests/_perf_structural.py`) over `pack_frames` + `compute_das` with a scoped `rows_scanned_counter`;
  tracemalloc: at fixed `chunk_size`, working peak (peak minus output arrays) for 20k frames ≤ 1.1 × the
  peak for 2k frames.
- [ ] **Step 6: Run** all DAS suites — PASS; record the new collected counts (the formerly dark tests now
  run under `-m "not e2e"`).

### Task 13: Packaging, CI, ADRs, docs, C4, NOTICE

**Files:**
- Modify: `pyproject.toml` (delete `das`; remove `accessible-space` from `test`; keep `das-reference`),
  `.github/workflows/ci.yml` (three install lines, the `das` comment block, the numba-key comment at
  `:82-83` → five files incl. `tracking/_das_numba.py`)
- Create: `docs/superpowers/adrs/ADR-107-native-das-engine.md`, `ADR-108-periodic-das-quadrature.md`
- Modify: ADR-043, ADR-106, ADR-012 (amendment sections), `AGENTS.md`, `docs/context/tracking-features.md`,
  `docs/context/conventions-core.md`, `NOTICE`, `README.md`, `silly_kicks/feature_glossary.py`,
  `docs/PRIVATE_CONSUMERS.md`, comments in `silly_kicks/tracking/schema.py:19-26`, `_snapshot.py`,
  `gradientsports.py:363`
- Modify: `docs/c4/architecture.dsl`, regenerate `docs/c4/architecture.html`
- Regenerate: `.test_durations`

- [ ] **Step 1:** packaging + CI edits; verify the unknown-extra behaviour: in a scratch venv under the
  session scratchpad, `uv pip install --dry-run "silly-kicks[das] @ <path to a wheel built with uv build>"`
  and record the exact output (expected: a warning, not an error) in ADR-107.
- [ ] **Step 2:** ADR-107 (engine, packing, dispatch, paired kernel, SC-1/SC-2, chunk-size table from
  Task 5, divergence table) and ADR-108 (quadrature defect evidence, periodic definition, parity mode, the
  corpus shift quoted in commit 2); amendments; `AGENTS.md` (architecture line, DAS bullet, Dependencies
  line) and `python -m pytest tests/test_agents_md_budget.py` — PASS.
- [ ] **Step 3:** NOTICE — DAS entries state a native reimplementation of Bischofberger & Baca (2026) and
  reproduce the accessible-space MIT licence text with "Copyright (c) 2024 Jonas Bischofberger".
- [ ] **Step 4:** C4 — remove `accessibleSpace` and its three relations, state DAS native in `tracking`;
  render with the pinned pipeline (`structurizr.war export` → `c4_assemble.py --inject-wrap-width` →
  `java -jar plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg` →
  `c4_assemble.py --svg-dir`), Windows paths for Java tools; a clean assemble proves `dot` ran.
- [ ] **Step 5:** regenerate `.test_durations` per ADR-074 (the repo's documented command) and run
  `tests/test_ci_shard_wiring.py` — PASS.

### Task 14: Corpus parity + performance driver (runs at commit 1, owner)

**Files:**
- Create: `scripts/validate_das_native_parity.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS`)
- Test: `tests/scripts/test_das_native_parity_driver.py`

**Interfaces:**
- CLI: `--out <DIR> --shard-root <DIR> [--providers ...] [--list-matches] [--allow-dirty]`, `for_each`
  over `list_match_refs` (resume before load), `require_clean_tree`, `declare_inputs`, aggregates only.
- Per match: link actions to frames; build `GoalMap` from full frames; pack linked frames; legs = reference
  (`accessible-space` per `(game, period)` with the native direction column, float64 inputs; library input
  prepared exactly as the golden generator does — share that helper by moving it into the driver module and
  importing it from `_generate.py`), native numpy (`reference`), native numba (`reference`), native
  production (`periodic`); timings per leg; paired vs independent on the gkdv ghost pair.
- Output: `docs/research/das_native_parity/metrics.json` with the spec §7.2 fields, per provider, plus
  provenance (`run_commit`, `run_tree_dirty`, versions, platform) and a `population` block: matches listed
  per provider by `_loader_pining._list_matches`, matches scored, and matches excluded with reason — including
  the SB360 count excluded as structurally unscoreable (velocity-less), per spec §7.2.

- [ ] **Step 1: Failing test** — the driver's **full reduce path** runs locally on an injected two-match
  synthetic corpus (no network) with the reference leg stubbed by the golden reference outputs, producing a
  schema-complete `metrics.json` whose counts reconcile (`assert_conservation`). This is mandatory before any
  DGX run (the trainer-locally rule).
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS**; `--list-matches` works
  offline-safe (clear error without a token).

### Task 15: Full verification, final review, commit-1 gate

- [ ] **Step 1:** `python -m ruff check silly_kicks/ tests/ scripts/`; `python -m ruff format --check
  silly_kicks/ tests/ scripts/`; bare `pyright` — all clean.
- [ ] **Step 2:** `python -m pytest tests/ -m "not e2e" -v --tb=short --benchmark-skip` on `.venv`
  (pandas 2) and on the pandas-3 venv; capture every `FAILED`; report pass arithmetic (collected / passed /
  skipped / deselected) for both legs, and the Task 12 before/after counts.
- [ ] **Step 3:** run `/final-review` (pre-commit quality gate + C4 check).
- [ ] **Step 4: Freeze the tree** and hand off for the external implementation review (owner-coordinated).
  No edits during the review; apply findings only after it returns.
- [ ] **Step 5: HUMAN-APPROVAL GATE — commit 1.** Present `git status`, `git diff --stat` and the file list;
  commit message draft (`feat(tracking): native DAS engine, drop accessible-space (ADR-107/108)`, trailers
  per the session's attribution rule, no `Claude-Session:` trailer). **Wait for an explicit yes.** Push only
  with explicit approval.

---

## PHASE B — COMMIT 2 (artifacts + release)

### Task 16: Owner-run DGX runbook (at the commit-1 SHA)

- [ ] **Step 1:** prepare the DGX checkout at the commit-1 SHA (`git fetch origin feat/das-native && git
  checkout -B das-native FETCH_HEAD`), venv with `.[numba,das-reference,test]` plus `pyarrow` and
  `ruthless-efficiency>=0.4.0`; verify `silly_kicks.__version__`, `accessible_space` 2.0.15, numba import.
- [ ] **Step 2 (owner runs):** `scripts/validate_das_native_parity.py --out docs/research/das_native_parity
  --shard-root <shards> --providers <all velocity-bearing providers>` (nohup + log; poll for completion with
  an explicit LIVE/DEAD token).
- [ ] **Step 3 (owner runs):** regenerate ΔDAS arm values (`scripts/build_gkdv_arm_values.py --out <ARMDIR>
  --arm das` with the providers/slice recorded in the current `tf19_signoff_power` upstream provenance);
  rebuild spells (`scripts/build_layer2_spells.py --out <SPELLDIR> --match-ids-json <same slice>`); re-run
  `scripts/build_tf19_instrument_responsiveness.py --out docs/research/tf19_instrument_responsiveness` with
  the arguments recorded in its current artifact; re-run `scripts/run_signoff_power.py --out
  docs/research/tf19_signoff_power --spells <SPELLDIR>/layer2_spells.parquet --arm-values
  <ARMDIR>/arm_values_delta_das.parquet --seed 0`.
- [ ] **Step 4:** copy artifacts back (scp, verify checksums both sides); confirm every run ended before
  copying.

### Task 17: Artifact gate, annotations, constants, release docs, commit-2 gate

**Files:**
- Create: `tests/tracking/test_das_parity_artifact.py`,
  `docs/research/tf24_stage2_refresh/invalidation.json`
- Modify/replace: `docs/research/tf19_signoff_power/invalidation.json` (updated or retired to match the
  fresh run), `silly_kicks/tracking/_das.py` (`_DAS_SECONDS_PER_FRAME`, `_PRANGE_EFFICIENCY` from the
  artifact), `silly_kicks/_version.py`, `CHANGELOG.md`, `TODO.md`, ADR-108 (corpus shift figures)

- [ ] **Step 1:** artifact gate — parity bounds, zero unexplained finite-mask mismatches, the spec §4.2
  targets, `run_tree_dirty is False`, `run_commit` present. If a target is missed or any new fail-loud
  condition fired on the owner corpus, **stop and surface to the owner** before continuing.
- [ ] **Step 2:** tf24 `invalidation.json` classifying `das_team`, `das_opponent`, `das_diff` (calibration
  features) and `das_degraded` with the measured corpus shift, citing ADR-107/108 and the commit-1 SHA.
- [ ] **Step 3:** constants from the artifact; CHANGELOG entry with the Hyrum block (spec §11) and the
  corpus median/p90/max; version bump at commit-prep with owner agreement on the number; TODO grooming.
- [ ] **Step 4:** full verification as Task 15 Steps 1–2.
- [ ] **Step 5: HUMAN-APPROVAL GATE — commit 2.** Present the diff; wait for an explicit yes. The lakehouse
  downstream notice (spec §11.2) is handed to the owner for relay.

---

## Appendix A — Reference-replication rules (numpy engine; numba follows the same formulas)

**A.1 Grids** (`simulation_grids`): pitch bounds `x ∈ [−52.5, 52.5]`, `y ∈ [−34, 34]`.
`max_pass_length = sqrt((x_max − x_min) ** 2 + (y_max − y_min) ** 2) + radial_gridsize * 3`;
`d = np.arange(offset, max_pass_length + offset + radial_gridsize, radial_gridsize)`;
`phi = np.linspace(phi_offset, 2 * np.pi + phi_offset, n_angles, endpoint=False)`;
`v0 = np.linspace(v0_min, v0_max, n_v0)`; `t_ball = (d[None, :] − d[0]) / v0[:, None]` then
`t_ball += time_offset_ball`; `rate_divisor = factor * (v0 ** (−factor2))`.
Radial bounds: `r_lo[1:] = (r[:-1] + r[1:]) / 2; r_lo[0] = r[0]; r_hi[:-1] = (r[:-1] + r[1:]) / 2;
r_hi[-1] = r[-1]; dr = r_hi − r_lo`. Angular bounds identical in form (reference); periodic replaces only
`phi_lo[0]` and `phi_hi[-1]` (Task 2). `d_area = dphi[:, None] / (2 * np.pi) * (np.pi * r_hi[None, :] ** 2)
− dphi[:, None] / (2 * np.pi) * (np.pi * r_lo[None, :] ** 2)`.

**A.2 Player order:** within a frame, players in ascending order of their `player_id` value (category
unwrapped), i.e. the order `pivot` gives the reference. Padding rows go after real rows (they contribute
exactly +0.0 / 0).

**A.3 TTA:** `x_mid = x + vx * inertial_seconds`, `y_mid = y + vy * inertial_seconds`;
`remaining = player_velocity` if not `use_max`, else `np.minimum(np.sqrt(vx ** 2 + vy ** 2) + a_max *
inertial_seconds, v_max)` (the reference's `np.linalg.norm` over the two-row stack);
`tta = np.sqrt((x_mid − xt) ** 2 + (y_mid − yt) ** 2) / remaining + inertial_seconds`;
`tol_mask = np.sqrt((x − xt) ** 2 + (y − yt) ** 2) < tol_distance`;
`tta[tol_mask] = (np.hypot(xt − x, yt − y) / player_velocity)[tol_mask]`; then passer → `inf` (xC), and
`np.nan_to_num(tta, nan=np.inf)`. Trajectory points: `xt = bx + cos_phi * d`, `yt = by + sin_phi * d`.

**A.4 Rates and densities:** `tmp = tta − t_ball[k]`; `tmp = b0 + b1 * tmp` (overflow ignored);
`tmp = 0.5 * (tmp / (1 + np.abs(tmp)) + 1)` (invalid ignored); `tmp = np.nan_to_num(tmp, nan=0)`;
`ar = tmp / rate_divisor[k]`. `s_att = np.nansum(np.where(att, ar, 0), axis=P)`, `s_def` likewise with
`~att`. `int = cumulative_trapezoid(s, x=t_ball[k], initial=0, axis=-1)`; `p0 = np.exp(−int)`.
`opp = np.where(att, p0_def, p0_att)`; `rho_k = opp * ar`; `rho_k = rho_k * dt0[k] / radial_gridsize`;
`rho = max over k`. If `normalize`: `num_max = np.max(rho * radial_gridsize, axis=(P, T))`;
`rho = rho / num_max` (invalid ignored). `att_rho = np.nanmax(np.where(att, rho, 0), axis=P)`.

**A.5 Danger and integration:** `x_n = x_grid * direction`, `y_n = y_grid * direction`;
`semi = 7.32 / 2 + 0.06`; `y_goal = np.clip(y_n, −semi, semi)`;
`dist = np.sqrt((x_n − 52.5) ** 2 + (y_n − y_goal) ** 2)`; opening angle: `u = (52.5 − x_n, semi − y_n)`,
`v = (52.5 − x_n, −semi − y_n)`, `div = norm(u) * norm(v)` (`div == 0 → inf`),
`angle = np.abs(np.arccos((u0 * v0 + u1 * v1) / div))`; `logit = −0.52156283 + −0.14447723 * dist +
0.40579492 * angle`; `danger = 1 / (1 + np.exp(−logit))`. Dangerous densities `danger ** (1 / danger_weight)
* density` before clipping; `on = (x_grid >= −52.5) & (x_grid <= 52.5) & (y_grid >= −34) & (y_grid <= 34)`;
clipped `np.where(on, field, 0)`; `AS = np.sum(field * dr * d_area, axis=(Φ, T))` in exactly that
multiplication order (team) and `np.sum(field * dr * d_area, axis=(Φ, T))` per player.

**A.6 xC:** single angle per pass; ball at event start (tracking ball where the event coordinate is NaN);
`respect_offside=False`; passer TTA → `inf`; `normalize=False`; the per-pass `v0` estimate the reference
computes (`get_pass_velocity`) is **not** computed — with the fixed `v0` grid it never reaches the simulation
(spec §6.12); `cum = np.maximum.accumulate(att_rho,
axis=T) * radial_gridsize`; clip: for each `t`, `cum[t] = cum[t] if on[t] else (cum_clipped[t−1] if t > 0
else 0)`; `xc = cum_clipped[-1]` (the last finite value); `np.clip(xc, 0, 1)`.

**A.7 Offside (DAS):** `norm_x = x_c * direction`; defenders = non-attacking players with finite `norm_x`;
if fewer than two → no offside (D-OFF); else `second_last` = second-largest defender `norm_x`;
`line = max(second_last, ball_norm_x)`; offside iff attacking & finite & `norm_x > line` & `norm_x > 0` &
not passer → the player's kinematics become NaN before TTA.

---

## Self-review (plan author)

- **Spec coverage:** §6.1 → T2–T5, T9; §6.2 → T9–T11; §6.3/§6.4 → T2, T4, T5, Appendix A; §6.5 → T4, T5;
  §6.6 → T7 (SC-1); §6.7 → T3, T9 (SC-2); §6.8 → T12 Step 4; §6.9 → T3, T10, T11; §6.10 → Appendix A.7;
  §6.11 → T3; §6.12 → T8; §6.13 → T10, T11; §6.14 → T5, T9, T17; §7.1 → T1, T4–T8; §7.2 → T14, T16, T17;
  §7.3 → T16, T17; §8 → T1–T12; §9 → T13; §10 → T13, T17; §11 → T17; §12 → T15, T17.
- **Placeholders:** none; measured values (chunk defaults, cost constants) have a prescribed measurement
  procedure and destination.
- **Type consistency:** `PassSimParams`, `SimGrids`, `PackedFrames`, `Reason`, `DasResult`,
  `compute_das`, `compute_das_paired`, `pack_frames`, `pack_paired`, `pack_passes`, `compute_xc`,
  `individual_das_paired` are used with the same names and fields throughout.

## Execution handoff

Plan → external plan review (owner-coordinated) → execution **inline** in one session on the owner's
implementation model (not Opus 5.5), task by task, using superpowers:executing-plans, with the two commit
gates above.
