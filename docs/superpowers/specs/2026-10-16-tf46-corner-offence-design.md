# TF-46 — Corner identification + offence routine inference (Arm 1)

> **Status:** design (uncommitted through the cycle, per the no-doc-commits-until-cycle-end rule).
> **Cycle scope:** corner identification (shared, amendment (a)) **plus Arm 1 (offence, label-free)**.
> Arm 2 (defence coupling) and Arm 3 (role-conditioned valuation) remain **separate later cycles**
> (owner ruling 2026-10-11 (1); reconfirmed 2026-10-16). Arm 3 cannot land here — it needs Arm 2.
> **ADR:** number assigned at commit-prep (next free; part-deux PR #284 holds reserved numbers below
> ADR-124). Do not pin an ADR or version number before commit — they change.
> **Release:** held until part-deux (#284) lands; this cycle specs/plans/implements up to, not
> including, release. Version bump happens only at commit-prep.
> **Reference:** Shaw & Gopaladesikan, "Routine Inspection: A Playbook for Corner Kicks" (42 Analytics;
> deployed at SL Benfica + Aalborg; 1,723 corners / 234 matches). This spec follows that deployed
> pipeline; its published numbers (83.4% supervised-defence accuracy, 70% logistic floor) are **Arm-2
> reference benchmarks, not gates here**.

## 1. Why

A corner is a rehearsed, strategy-dominated restart (~5 per team per game, ~1.5–2% direct conversion).
Tracking data makes the attacking team's synchronised runs legible: a small vocabulary of
initial→target runs, a handful of co-occurring run combinations ("features"), and a team's distinct
repeated routines. Shaw & Gopaladesikan published and deployed an unsupervised, label-free pipeline
that recovers exactly this. TF-46 Arm 1 brings that pipeline into silly-kicks as **raw primitives**
(ADR-110): the library ships the encoder, the fitted zone/routine models and the cluster structure;
archetype naming (jellyfish / train / overload) and which-routine-beats-which ranking stay
consumer-side.

This cycle also builds the **shared scaffolding** that Arm 2 reuses: corner identification, the
reflection-to-canonical-side normalisation, the per-corner window definitions, and the corpus driver.

**Owner steer (2026-10-16): gold standard, best practice, most value; scope is not a limiter.** The
fork resolutions in §3 reflect that: fit on our corpus with principled model selection (not Shaw's
literal counts), ship the full offence pipeline through routine clustering, support every provider,
and add a descriptive per-routine chance-quality primitive (Shaw's explicitly-deferred "which routines
are most effective", offence side, no counterfactual).

## 2. What this cycle ships (and what it does not)

**Ships:**

1. **Corner identification** (shared) — a provider-aware `identify_corners` that anchors a corner and
   its attacking team from SPADL + SkillCorner GI tokens, excludes short/unclear corners, and records
   the kick frame and corner side.
2. **Corner windows** (shared) — per-corner `t_kick`, `t_init`, `t_target`, `t_outcome`, winner, and
   the first-contact/first-possession timing diagnostics, consuming shipped TF-79.
3. **Reflection-to-canonical-side** (shared) — a Y-reflection of all corners to one flag through the
   existing `reflection.py` registry.
4. **Offence run encoder** — the probabilistic (initial-zone × target-zone) 42-ish-run encoder.
5. **Fitted zone model** — `RunZoneModel` (initial + target GMMs, active-zone selection), params-only
   npz behind the ADR-011 seam.
6. **Routine features** — `CornerRoutineModel` (NMF basis `W`) + the per-corner coefficient transform.
7. **Routine clustering primitive** — agglomerative clustering of the coefficient vectors → per-corner
   cluster assignment + linkage (raw structure only).
8. **Per-routine chance quality** — a descriptive primitive: shots / goals / xG inside `t_outcome`,
   per corner and aggregated by feature expression. No ghost counterfactual → **not Arm 3**.
9. **Corpus driver(s)** + a **construct-validity memo** for the chance-quality effectiveness claim.

**Does not ship (explicit, so a reviewer does not read these as silent cuts):**

- Arm 2 defence coupling, corner-GK primitives, zonal-config-effectiveness — **next cycle**.
- Arm 3 role-conditioned ghosts / per-defender credit — **deferred; needs Arm 2**.
- Consumer-side archetype naming and routine ranking (ADR-110 — stays out of the library, by design).
- No GNN / TacticAI line (owner ruling; outside the no-NN idiom).

## 3. Resolved design forks (owner rulings 2026-10-16)

| Fork | Resolution | Rationale |
|---|---|---|
| **TF-79 dependency** | Target-snapshot timing **and** winner from TF-79 `first_contact_in_window`; `t_target = min(t_firstcontact + 1 s, t_kick + 2 s)`; `t_kick+2 s` branch is the fallback when TF-79 resolves no contact. Emit a `first_possession_time` (GI) diagnostic alongside. | A "first on-ball event" is a touch, so TF-79 contact is more faithful than GI possession (pilot agreement 55%); one call serves timing + the owner-required winner. Arm 1 **consumes** shipped TF-79 but is **not gated** by its coverage. The GI diagnostic feeds a convergent-agreement check. |
| **Zone definitions** | Fit the initial + target GMMs on the owner-tier corpus; **select component counts by BIC / log-likelihood elbow** (Shaw's own criterion), not hard-coded 15/7/6. Keep the geometric "active = inside the penalty area" rule. Freeze selected-k + means/weights to npz (ADR-011). Report selected counts vs Shaw's 6×7 as convergent validation. | Our 909-match multi-league corpus deserves its own fit; Shaw's k came from a single 234-match league. Encoder dimension is therefore `n_initial × n_target` (nominally ≈42), data-driven. |
| **Arm-1 v1 depth** | Full in-library pipeline: encoder + fitted GMMs + NMF basis/transform + agglomerative clustering primitive. Naming + ranking stay consumer-side (ADR-110). | Routine features are the headline output and the natural reusable primitive; the library ships raw cluster structure, not team-specific labels. |
| **Provider scope** | SkillCorner (primary; the corpus + GI id) + SB360 (two-freeze-frame encoder) + optical tracking (Sportec/IDSSE, Metrica) via the generic SPADL-corner anchor. Optical full-observability providers power the cross-provider stability gate. | Gold standard = every provider; optical obs validates the broadcast fit. |
| **Effectiveness value-add** | Per-routine chance-quality **descriptive** primitive (shots/goals/xG in the 6 s outcome window, by feature expression). | Shaw's deferred effectiveness question, offence side; uses existing shot/xG; descriptive, so not Arm 3. |
| **Arm sequencing** | Arms stay sequenced: offence now, defence next, valuation last. | Owner ruling 2026-10-11 (1), reconfirmed 2026-10-16. |

## 4. Package layout

A new sibling family package (trips ~8 repo-wide gates — §10: metric-contracts, feature-glossary,
C4-count, purity, mirror-registry, SB360-verdict, id-dtype and public-api-examples; **plus a new
`corners.__all__`-pinned liveness gate, C2**). The tracking liveness + call-convention registries are
correctly scoped OUT (§10.3). Home: **`silly_kicks/corners/`** (corner-specific; Arm 2 defence +
corner-GK extend the same package next cycle).

```
corners/__init__.py        # PUBLIC surface; __all__ exports the compute/add fns + *_METRIC_COLUMNS + *_KEYS
corners/_config.py         # CornerWindowParams, RunEncodingParams, RoutineParams, ChanceQualityParams
corners/_identify.py       # identify_corners (provider-aware anchor + short-corner exclusion + corner side)
corners/_windows.py        # corner_windows (t_init/t_target/t_outcome, winner, timing diagnostics; calls TF-79)
corners/_reflect.py        # reflect_corners_to_canonical_side + the corner reflection-kinds map
corners/_zones.py          # RunZoneModel (initial + target GMM fit/serve; active-zone selection); encode_runs
corners/_routines.py       # CornerRoutineModel (NMF W); transform; cluster_corner_routines (agglomerative)
corners/_encode.py         # compute_corner_run_encoding (per-corner run vector = the NMF term matrix column)
corners/_effectiveness.py  # compute_corner_chance_quality + aggregate_chance_quality_by_feature
corners/_columns.py        # CORNER_*_METRIC_COLUMNS + CORNER_*_KEYS (grain constants)
corners/_report.py         # describe / summary rendering
```

Tests mirror the path: `tests/corners/test_*.py`. Fitted-model fixtures under `tests/corners/`
conftest; shared freeze-frame / tracking fixtures reuse `tests/fixtures/`.

## 5. Public surface

Shapes follow the authoring runbook (`compute_*` = fresh table at its own grain; `add_*` = pure
additive enricher of a passed frame). Corner analysis is per-corner aggregation, so most of the surface
is `compute_*`; the per-corner-table enrichers are `add_*`.

```python
# --- shared: identification + windows + reflection -------------------------------------------
def identify_corners(actions: pd.DataFrame, *, provider: str,
                     params: CornerWindowParams = ...) -> pd.DataFrame:
    """Per-corner table. Grain key: corner_id. Columns in §6.1.
    Excludes short/unclear corners (kept as rows with corner_kind, not dropped silently)."""

def add_corner_windows(corners: pd.DataFrame, frames: pd.DataFrame, contacts: pd.DataFrame, *,
                       params: CornerWindowParams = ...) -> pd.DataFrame:
    """Additive (pure; purity-registered): t_init, t_target, t_outcome, winner_team_id/player_id,
    first_contact_time, first_possession_time, target_snapshot_source. Consumes TF-79
    first_contact_in_window."""

def reflect_corners_to_canonical_side(df: pd.DataFrame, corners: pd.DataFrame, *,
                                      kinds: Mapping[str, ReflectionKind]) -> pd.DataFrame:
    """Y-reflect every corner to the canonical flag via reflection.reflect_columns, keyed on
    corner_side. Pure; returns a new frame."""

# --- offence: encoder + fitted models + routines ----------------------------------------------
class RunZoneModel:                      # ADR-011 params-only (npz + JSON meta + SHA256SUMS)
    def fit(self, init_xy, target_xy) -> "RunZoneModel": ...
    def encode_runs(self, init_xy, target_xy, active_mask) -> np.ndarray:  # probabilistic (n_i*n_t)-vec
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "RunZoneModel": ...   # runs verify_chirality + _feature_contract

def compute_corner_run_encoding(corners, frames, *, model: RunZoneModel,
                                params: RunEncodingParams = ...) -> pd.DataFrame:
    """Per-corner (n_i*n_t)-element run vector (the NMF term-matrix column). Grain: corner_id."""

class CornerRoutineModel:                # ADR-011 params-only (NMF basis W)
    def fit(self, term_matrix, *, rank=None) -> "CornerRoutineModel": ...   # rank: elbow-selected if None
    def transform(self, term_matrix) -> np.ndarray:                          # per-corner r-dim H coeffs
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "CornerRoutineModel": ...

def compute_corner_routine_features(encoding, *, model: CornerRoutineModel) -> pd.DataFrame:
    """Per-corner r-dim routine-feature coefficients (H columns). Grain: corner_id."""

def cluster_corner_routines(features, *, params: RoutineParams = ...) -> pd.DataFrame:
    """Agglomerative clustering of the coefficient vectors. Per-corner cluster_id + the linkage
    structure. RAW structure only — no archetype names, no ranking (ADR-110)."""

# --- descriptive effectiveness (value-add; not Arm 3) -----------------------------------------
def compute_corner_chance_quality(corners, actions, *, xg_column: str | None = None,
                                  params: ChanceQualityParams = ...) -> pd.DataFrame:
    """Per-corner shots / goals / xG inside [t_kick, t_outcome]. Grain: corner_id. Descriptive.
    n_shots/n_goals are model-free (always computed); sum_xg is honest-NaN when xg_column is omitted
    (never a fabricated 0.0); a NAMED-but-absent xg_column raises (§469 amendment)."""

def aggregate_chance_quality_by_feature(features, chance_quality) -> pd.DataFrame:
    """Soft-weighted (by H coefficient) mean chance quality per routine feature. Descriptive
    aggregation — NOT a ranking (which-beats-which stays consumer-side)."""
```

No speculative aliases (speculative API is debt). Frame-consuming `add_*` follow the ADR-078 one-call-
shape (`frames` positional-or-keyword, optional kwargs keyword-only). NaN-tolerant enrichers decorate
`@nan_safe_enrichment` (ADR-003).

## 6. Data contracts

### 6.1 The corners table (`identify_corners`)

> **Amendment 2026-10-16 (owner-approved):** the grain key is **`game_id`** (SPADL-native — every
> metric family keys on `game_id`, and the loaded SPADL frame carries `game_id`, not `match_id`), NOT
> the `match_id` this spec's earlier prose used. Supersedes `match_id` throughout (incl. §13's join
> key). The pining driver's `match_id` ref maps to the loaded match whose actions carry `game_id`.

| column | dtype | meaning |
|---|---|---|
| `game_id` | `object` | grain key (SPADL-native), with `corner_id` |
| `corner_id` | `int64` | grain key: fresh per-game chronological positional index; **NOT yet a cross-table FK** (no consumers). Join-key-stability note in §13 (ADR-065 binds only if it becomes a cross-table key). |
| `period_id` | `int64` | period |
| `t_kick` | `float64` | period-relative kick time (s) |
| `attacking_team_id` | id (provider-native, via `id_compat`) | corner-taking team |
| `corner_side` | `category` `{left, right}` | which flag (in the frame-LTR frame) — drives reflection |
| `kick_x`, `kick_y` | `float64` | ball position at the kick, frame-LTR |
| `corner_kind` | `category` `{direct, short, unclear}` | short/unclear kept, not dropped (ADR-042) |
| `corner_source` | `category` | `{skillcorner_gi, spadl_corner, sb360_event}` provenance |

`corner_kind ∈ {short, unclear}` rows are excluded from the offence encoder but **kept** in the table
with the reason recorded (no silent drop).

### 6.2 Window columns (`corner_windows`, additive onto the corners table)

| column | dtype | meaning |
|---|---|---|
| `t_init` | `float64` | `t_kick − 2 s` (speed-rise onset; tracking). On SB360 = kick freeze-frame time (train/serve divergence measured, §7 / §12.7) |
| `t_target` | `float64` | `min(t_firstcontact + 1 s, t_kick + 2 s)` |
| `t_outcome` | `float64` | `t_kick + 6 s` (shot-within-6 s outcome window; paper §5.2) |
| `first_contact_time` | `float64` | TF-79 first contact after the kick; NaN when unresolved |
| `first_possession_time` | `float64` | GI first-possession time (diagnostic; convergent-agreement gate) |
| `winner_team_id` / `winner_player_id` | id (nullable) | first-contact winner from TF-79; `<NA>` when unresolved — **no GI fallback** |
| `target_snapshot_source` | `category` `{first_contact, kick_plus_2s, sb360_event}` | which `t_target` branch fired (`sb360_event` = first-on-ball-event freeze frame) |

### 6.3 Run-encoding + routine columns

- `compute_corner_run_encoding` → one row per corner, `(n_initial × n_target)` float columns
  `run_{i}{t}` (probabilistic weight = Σ_players P(init zone i) · P(target zone t)).
- `compute_corner_routine_features` → one row per corner, `r` float columns `routine_feat_{k}`.
- `cluster_corner_routines` → `cluster_id` (`int64`) per corner + a linkage artifact (returned, not a
  column).

### 6.4 Chance-quality columns

| column | dtype | meaning |
|---|---|---|
| `n_shots` | `int64` | shots inside `[t_kick, t_outcome]` |
| `n_goals` | `int64` | goals inside the window |
| `sum_xg` | `float64` | Σ xG inside the window (from `xg_column`) |
| `chance_quality_source` | `category` | provenance of the xG column |

NaN-row discipline (ADR-042): a corner with no resolvable outcome window keeps its row with `0`/NaN and
a reason, never a dropped row.

## 7. Provider sourcing (per-provider snapshot rules)

| provider | corner id | `t_init` snapshot | `t_target` snapshot | winner / timing |
|---|---|---|---|---|
| **SkillCorner** | GI: first `player_possession` after restart with `start_type ∈ {corner_reception, corner_interception}`; `game_interruption_before=="corner_for"` + other start type ⇒ `unclear`; never the ball-at-flag | continuous frames at `t_kick − 2 s` | continuous frames at `t_target` | TF-79 on the kick→+6 s window |
| **Optical** (Sportec/IDSSE, Metrica) | SPADL `corner_crossed` / `corner_short` action; kick frame via ball-motion onset (ball-in-play flag + acceleration, paper §2) | continuous frames at `t_kick − 2 s` | continuous frames at `t_target` | TF-79 on the window |
| **SB360** | SPADL corner event (freeze-frame provider) | **kick freeze frame** (no continuous −2 s; divergence measured, §12.7) | first-on-ball-event freeze frame | **no continuous tracking ⇒ no TF-79 contacts ⇒ winner/first_contact_time = `<NA>`**; `t_target` from the first-on-ball-event freeze frame, `target_snapshot_source = sb360_event` |

The SB360 `t_init`-vs-`−2 s` semantic gap is a **train/serve mismatch** — a `RunZoneModel` fit on
tracking `−2 s` initial positions is served on SB360 kick-frame initial positions, which can
mis-assign initial zones. Owner blessed the two-freeze-frame path (amendment), but per the C5 ruling
(2026-10-16) the divergence is **measured and bounded (§12.7)**, not merely caveated. SkillCorner
`_skillcorner_inference.py:27-29` currently maps
every `*_interception` generically and `skillcorner.py:354` emits only the corner kick — `identify_corners`
reads the specific corner tokens; it does **not** change the existing converter output (additive).

## 8. Orientation / reflection

Two orientations compose and must not be conflated:

1. **Frame-LTR** (attacking left→right) is already handled upstream (`orient_frames_to_ltr`, ADR-029;
   defended goal via `resolve_defended_goals` → `GoalMap`, ADR-055). TF-46 consumes LTR frames; it does
   not re-derive attacking direction.
2. **Corner-side canonicalisation** is a **Y-reflection** (the two corner flags sit at opposite `y`
   within the LTR frame). `reflect_corners_to_canonical_side` builds a per-corner mask from
   `corner_side` and calls `reflection.reflect_columns` with the corner reflection-kinds map
   (`point_y` / `vector_y` flips; zone labels and probabilities are `invariant`). All corners are
   normalised to one flag (Shaw convention).

The corner geometry columns (`kick_y`, snapshot `y`, any emitted run coordinate) register in
`silly_kicks/reflection.py` with the right `ReflectionKind`, so the `test_mirror_registry` Gate A/B
check covers them and a left-corner / right-corner scene of identical geometry yields mirror-equal
encodings (§10 invariance gate). The two-snapshot run encoding is translation-invariant (TF-38-agnostic),
but the GMMs are fit in the canonical frame, so reflection runs **before** zone assignment.

## 9. Fitted models (ADR-011 params-only seam)

Two trained artifacts, both following the ReceiverModel / XSuccessModel lifecycle (sklearn/xgboost at
FIT only; numpy at serve; pickle-free):

- **`RunZoneModel`** — initial GMM (means, covariances, weights) + target GMM + the active-zone index
  set (target components inside the penalty area, via `in_penalty_area_absolute`, ADR-050) + zone
  labels. Serialised as npz + JSON metadata + `SHA256SUMS`. `load()` runs `verify_chirality` and the
  `_feature_contract` (declared constants compared first → `IntegrityError`).
- **`CornerRoutineModel`** — the NMF basis matrix `W` (`(n_i*n_t) × r`), the selected rank `r`, and the
  α regularisation. npz + metadata + checksums; same load guards.

Model **selection** (fit-time, recorded in the artifact metadata): GMM component counts by BIC / log-
likelihood elbow; NMF rank `r` at the reconstruction-error (`‖X − WH‖_F`) elbow (Shaw selected `r=30`;
we report our curve). Chirality + feature-contract guards per ADR-011; serve path is pure/deterministic
→ a `PURITY_ENTRIES` entry for every `add_*`.

**Publication** (ADR-088) via `publish_model_with_card` (requires `--model-card`,
`create_repo(exist_ok=True)`, card→README, allowlist) happens **at release** — held with the part-deux
hold; the owner publishes. In-cycle, the fit driver writes local npz artifacts.

## 10. Repo-wide gates this family must wire (new-sibling-package checklist)

From the authoring runbook — each reds CI if skipped; several fail **only in the full suite**:

1. **`metric_contracts` (ADR-098):** export `CORNER_*_METRIC_COLUMNS` + `CORNER_*_KEYS` in
   `__all__`; add `MetricContract` entries to `METRIC_CONTRACTS` (hardcoded literals, round-tripped by
   `tests/test_metric_contracts.py`; `_UNDERIVABLE` stays empty).
2. **Purity (ADR-033):** every public `add_*` in `PURITY_ENTRIES` (`"corners:<name>"`); ≥2 variants for
   any conditional-column branch; `@nan_safe_enrichment`.
3. **Family shape = `compute_*` aggregator in `corners/`** (analogous to `shot_stopping/` / `territory/`
   / `duels/`), **not** a `tracking/_<family>.py` xfn. Therefore the **tracking-specific** liveness
   (`test_aggregator_column_liveness.py`, pinned to `tracking.__all__`) and call-convention
   (`test_call_convention_registry.py`) registries do **not** apply — this family is not in
   `tracking.__all__`. The one additive enricher (`add_corner_windows`;
   `compute_corner_chance_quality` is a fresh-table `compute_*`) follows the compute-family gates
   (metric-contracts + purity + glossary) and the ADR-078 one-call-shape as a convention, but is not
   registered in the tracking registries. **If** any corner enricher is
   later exposed as a tracking xfn, surface it — do not silently route it into `tracking.__all__`.
   **Corners-own liveness gate (C2, owner-ruled 2026-10-16):** because the tracking liveness registry is
   scoped out, the family ships its OWN `corners.__all__`-pinned liveness gate —
   `tests/corners/test_corner_column_liveness.py` — asserting no emitted metric column is 100 %-null on
   an in-domain fixture and every float metric column with ≥2 observations carries >1 distinct value; a
   meta-assertion pins the gate population to `corners.__all__` (a new corner `add_*`/`compute_*` that
   wires no liveness entry reds CI), with its own fixture-precondition test (ADR-032 idiom). This honours
   the repo mandate "no 100 %-null column for any aggregator, no exception set" that the precedent
   families (`shot_stopping`/`territory`/`duels`) do not yet carry. By-design-dark SB360 continuous-only
   columns are covered by the SB360 verdict budget (item 7), never a silent liveness exclusion.
4. **`*_xfns` leak guard:** TF-46 reads post-kick outcomes (chance quality reads `result_id` / shots) —
   it ships **no `*_xfns` factory** and stays out of every default xfn list (ADR-030/047/049). The
   absence is asserted by the auto-discovering guard.
5. **`feature_glossary` + `NOTICE` + C4:** a `FeatureColumn` record per emitted column (`emitting_module`
   = the home compute module); a `NOTICE` entry for Shaw & Gopaladesikan + the GMM (Dempster et al.)
   and NMF (Lee & Seung; Kuang et al.; Hoyer; Févotte & Idier) methods (ADR-005); C4 aggregator-count
   bump in `docs/c4/architecture.{dsl,html}` + the `AGENTS.md` TF-table count (ADR-048), regenerated
   via the `c4` skill with Graphviz `dot`.
6. **Mirror registry (ADR-045):** corner geometry columns register their `ReflectionKind`; Gate A/B.
7. **SB360 freeze-frame verdict (ADR-053):** every `add_*` carries a freeze-frame verdict;
   `UNAUDITABLE_BOUNDARY` stays empty; a both-NaN column on freeze frames is `no_signal → not_exercised`
   (bump `NOT_EXERCISED_BUDGET` with a recorded reason — continuous-only primitives are legit-dark on
   freeze frames; do not fabricate coverage).
8. **`PUBLIC_ID_SCALAR` / id-dtype gates (ADR-019):** any public fn with an entity-`*_id` param wires
   `PUBLIC_ID_SCALAR_ENTRIES` + `ids_match`; route every id through `id_compat`.
9. **public-api-examples + doctests:** the new public surface gets Examples (full-suite gate).

The spec execution runs the **CI-faithful** invocation (`pytest -m "not e2e"`, the lint at CI scope,
`pyright` whole-package) and the SB360 cascade **before** declaring green — a targeted-local run misses
items 3/5/6/7/9 (cost a red CI round on the TF-79 cycle).

## 11. Corpus drivers (on `scripts/_driver.py`)

Two drivers, both adopting the `for_each` seam (resume-before-load, generation-dir fingerprint, one
shard per item, `require_clean_tree(git_provenance())` first, `declare_inputs` digest):

- **`scripts/build_corner_routines.py`** — fits `RunZoneModel` + `CornerRoutineModel` on the owner-tier
  corpus (SkillCorner primary; optical for cross-provider validation). Refs via `list_match_refs` /
  `list_gi_refs` + `load=`; `events_only=` stated; `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION`
  pinned; aggregate-only artifacts under the reversibility policy (no reversible ids in committed
  outputs). The fit itself is an aggregate over corner run vectors → shardable per match, combined from
  `res.shard_keys`.
- **`scripts/validate_corners.py`** — the construct-validity harness (§12). Lands
  `docs/research/corner_routines/{README.md,metrics.json}` with `run_commit`, `run_tree_dirty: false`,
  and the `input_contract` block.

Both enrol in `ARTIFACT_DRIVERS` (population derived + asserted exactly; content predicates read
`string_literals`). A `tests/scripts/test_*` guard per driver (CI-safe slice; real-corpus run is
`e2e`/opt-in). ASCII-only driver source (the driver ASCII gate; ruff does not catch `§`).

## 12. Validation (in-cycle, pre-registered — reported, not deferred to the owner)

Arm 1's encoder / routine outputs are **descriptive / unsupervised** → **no accuracy claim, no accuracy
gate** (consistent with TF-46's "benchmarks not gates" stance and the name-by-mechanism discipline).
Reported structural + reliability checks:

1. **Non-vacuity** — the run encoder populates more than one zone-pair across the corpus (not a
   degenerate all-mass-in-one-run map). Asserted, two-sided.
2. **Orientation invariance** — a left-corner and the mirrored right-corner of identical geometry
   produce mirror-equal encodings (Gate A/B + a dedicated corner invariance test; non-vacuity: an
   off-centre probe that *must* move under reflection).
3. **Synthetic recovery** — simulated corners with planted zone assignments and planted routine basis:
   the GMM recovers the zones and NMF recovers the basis (mechanics only, not real-world accuracy).
4. **NMF reconstruction curve** — report `‖X − WH‖_F` vs `r`; the selected rank sits at the elbow.
5. **Cross-provider stability** — zone / feature distributions on SkillCorner (broadcast) vs optical
   Sportec/IDSSE (full observability) agree within a pre-registered margin.
6. **Convergent timing agreement** — TF-79 first-contact vs GI first-possession time
   (`first_possession_time` diagnostic); reported, not gated.
7. **SB360 init-snapshot divergence (C5, owner-ruled 2026-10-16)** — measure and bound the train/serve
   mismatch from fitting `RunZoneModel` on tracking `−2 s` initial positions but serving on SB360
   kick-frame initial positions. On a provider that has **both** continuous tracking **and** freeze
   frames (Sportec/IDSSE), derive the matched ground truth: for the same corners, report (a) the
   per-player `−2 s`→kick-frame positional-delta distribution and (b) the resulting **initial-zone
   reassignment rate** (fraction of players whose argmax initial zone changes between the two
   snapshots). This bounds the SB360 serve error directly. Reported in the §11 memo; the cross-provider
   stability check (item 5) is additionally extended to include SB360 against the matched-provider
   baseline so SB360 is not left unmeasured.

**Chance-quality effectiveness** carries a full **construct-validity memo** (the three-gate shape,
`docs/howto/construct-validity.md`): responsiveness (per-feature chance quality moves off its baseline),
discrimination (does a feature's chance quality separate teams beyond a permutation null, net of team
strength), and transfer (season-to-season sign agreement). The memo leads with the caveat of **what the
number does not measure**: per-routine chance quality is an *outcome-association*, not a causal routine
valuation (that is Arm 3); it does not control for defensive setup or finishing skill. A power/validation
harness takes an `InjectionSpec`, never an outcome vector (the observed outcome is unrepresentable),
and counts inestimable replicates (`n_degenerate_by_size`).

## 13. Risks / open implementation questions (for the plan, not blocking the spec)

- **Fit provenance** — pooled-corpus vs per-provider `RunZoneModel` npz. Leaning pooled with a
  per-provider cross-check (item 5); decide in the plan from the cross-provider spread.
- **Kick-frame onset on optical** — the ball-motion-onset detector (ball-in-play + acceleration) is a
  small primitive; verify against the SPADL corner action time and record the offset distribution.
- **Active-zone penalty-area rule** — `in_penalty_area_absolute` is frame-explicit; confirm it keys off
  the attacked goal (`GoalMap`) after frame-LTR + corner-side reflection.
- **NMF determinism** — seed the multiplicative-update init; the fitted `W` is frozen to npz, so serve
  is deterministic, but the fit must be reproducible for the artifact digest.
- **`corner_id` join-key stability (C4/B-02)** — `corner_id` is a per-match chronological positional
  `int64`, so re-classifying or removing a corner renumbers every subsequent id. With no cross-table
  consumers today this is low-risk, but a persisted per-corner mart that joins on `corner_id` would see
  a Hyrum break on re-materialise. The plan either (a) keeps `corner_id` a within-table grain key only
  (documented as unstable across re-runs, join on a stable natural key — `(game_id, period_id,
  t_kick)` — for any cross-table use), or (b) if a cross-table FK is later needed, promotes it under the
  ADR-065 atomic-migration discipline. Decide in the plan; default (a).

## 14. Release / versioning

Additive family (new package, new columns, no existing byte changes) → a purely additive release, no
retrain of existing models, no lakehouse re-materialise beyond the new marts. **Version bump and the
ADR number are pinned only at commit-prep**, after CI is green and part-deux has landed — not now. The
spec, plan, and ADR stay uncommitted on `tf-46-corner-offence` through the cycle; reviewers read the
uncommitted files.

**Commit gate (C3 — explicit, binding on the plan).** No `git commit` or `git push` happens without
the owner's explicit approval for that specific commit: the session stops at the point of committing,
shows exactly what would go in (diff / file list), and waits for an explicit yes. None of "tests are
green", "the plan says commit", or approval of an earlier commit counts as approval. **No micro-commits
and no per-step/per-task commit cadence** — whatever an upstream skill template suggests. A commit is
one coherent, fully-tested state (the feature + its tests green), however many steps it took; the whole
cycle lands as a small number of such commits, each individually owner-approved. The plan inherits this
gate verbatim — every task's terminal state is a tested deliverable **accumulated on the branch**, not
a commit; commits are a separate owner-gated step at cycle end.

## 15. Amendment 2026-10-17 (owner-approved) — corpus-run enablement fixes

The Task-14 corpus run (the first time the pipeline met real provider data) surfaced two crashes, a
pre-existing converter defect, and a large avoidable cost. The owner approved fixing all as part of the
cycle before the clean deliverable run. Evidence is measured, not inferred.

> **Delta since the last A/B spec-approval (`1f483f0f`), for the focused re-review.** The spec sections
> changed since that approval are: this §15 (A–D below), the §6.1 grain amendment (`game_id`),
> §5's `compute_corner_chance_quality` signature (`xg_column: str` → `str | None`), §449 `--providers`
> (drops `sportec`), and §469 (Tier-1 model-free `n_goals`/`n_shots` + Tier-2 per-provider native xG).
> Reviewers need only assess this delta, not the whole (already-approved) spec. The (C) converter change
> is a separate base-code concern — see `docs/superpowers/adrs/ADR-127-sportec-corner-crossed-discriminator.md`.

- **(A) NA-safe id compare in `corners/_windows._team_of`.** The tracking ball row carries a `<NA>`
  `player_id` (ADR-058), so the raw `==` against the frame's `player_id` built an NA-bearing mask that
  crashed under boolean indexing (`boolean value of NA is ambiguous`) on every SkillCorner match.
  Corrected to `id_compat.ids_match` (ADR-019). Regression: `tests/corners/test_nan_safe_frames.py`.
- **(B) DEFENSIVE NA-safe id hardening in `corners/_encode._nearest_frame_xy`.** The same raw-`==`
  anti-pattern on the ball row's `<NA>` `team_id`, hardened to `ids_match` + positional numpy masks
  (ADR-019; avoiding the fresh-RangeIndex `&` trap). Unlike (A), this path did NOT reproduce a crash:
  measured on object / category / Int64 `team_id`, `canonical_id_series(...) == scalar` resolves to a
  plain-`bool` mask (NA→False) and the ball row is correctly excluded, so it is a defensive hardening,
  not a bug fix. The regression file's `_encode` case therefore asserts correct ball-row-tolerant
  encoding (a guard), not a red-green crash; `_windows` (A) is the red-green crash-pin.
- **(C) Sportec/IDSSE SPADL corner-crossed discriminator (base-code; `spadl/sportec.py`).** The crossed
  vs short test used `corner_target_area ∈ {box, penaltyBox, sixYardBox}` / `corner_placing ∈ {aerial,
  high}` — tokens that appear in NO real DFL data (they came from a self-referential unit fixture). So
  every DFL corner was mistyped `corner_short` (measured: 57 corners across all 7 IDSSE matches, 0
  crossed), and IDSSE contributed zero direct corners. The real DFL discriminator is `play_height`,
  which on CornerKick events takes `{high, flat}`: measured clean partition `high` ⟺ crossed (53),
  `flat`/`corner_target_area=="short"` ⟺ short (4). Fixed to key on `play_height=="high"`, keeping the
  old tokens in the union for backward-compatibility with any pre-normalized bronze input. This is
  **value-changing** (not additive): ~93% of DFL corners flip `corner_short`→`corner_crossed`, SemVer
  **minor**. Per owner ruling 2026-10-17 it lands as **its own base-code commit** (separate from the
  additive corners-feature commit) with its **own ADR + own independent A/B review**
  (`docs/superpowers/adrs/ADR-127-sportec-corner-crossed-discriminator.md`). Measured consumer blast
  radius (DFL/IDSSE only; SkillCorner/StatsBomb type corners natively): **no consumer breaks** —
  group-both consumers (`vaep/formula`, `coordination`, `tracking/_elastic_sync`/`_ghost_gk`/`_packing`,
  `team_metrics/_possession`, `atomic/_simplify`) unaffected; distinguishing consumers
  (`tracking/defensive_credit/_bravery`, xSuccess/VAEP `type_corner_crossed`/`type_corner_short` one-hots)
  shift toward correctness, **no retrain required** (schemas unchanged). Materialized IDSSE/DFL marts
  carrying trained-model predictions are stale-until-re-materialized; re-materialization is owner-ruled
  OUT OF SCOPE this cycle (lakehouse not a concern, 2026-10-18). Tests in `tests/spadl/test_sportec.py`.
- **(Optimization) Velocity-free corner loading.** The corner pipeline reads x/y only
  (`detect_ball_contacts` derives its own ball velocity), yet the loader force-ran `derive_velocities`
  on every full match — measured **32.2 s of the 36.4 s** per SkillCorner match. The loader gained an
  opt-in `with_velocity` (default `True`, backward-compatible); the corner drivers pass
  `with_velocity=False`, which keeps position smoothing (~1 s) and skips velocity. Measured per-match
  load 36 s → ~5 s. The 20 h serial estimate was the artifact of this plus running build+validate
  serially; velocity-free + the planned shard parallelisation brings the full corpus to well under an
  hour.
- **(Providers) `--providers` default.** Dropped `sportec` — it is the same data as `idsse` at the
  pining layer and 404s the listing. Default is now `[skillcorner, idsse, statsbomb]`.

**Verified (2026-10-17):** local targeted tests green (`tests/spadl/test_sportec.py`,
`tests/corners/`, `tests/calibration/test_loader_pining.py`, `tests/scripts/test_loader_*`); `ruff`
CI-scope clean; `pyright` 0 errors on the changed modules; a 47-match DGX pilot (`--allow-dirty`
scratch) confirmed IDSSE now yields direct corners (was 0), velocity-free SkillCorner load ~5–12 s, and
both models fit (`RunZoneModel`, `CornerRoutineModel`). The deliverable fit/validate run is on a CLEAN
clone after the commit-1 amend lands (`run_tree_dirty:false`), per §14.

**Open validity items for the full-corpus §12 memo (owner-directed, not commit-1 blockers):**

1. **Active target zones.** The 110-corner pilot fit yielded `n_active_target=2` (Shaw's playbook has
   ~7). This is plausibly a small-sample effect; the full corpus (thousands of corners) is the valid
   test. If the full corpus still yields a low active-zone count, that is a finding to surface, not to
   accept silently. (The constant NMF reconstruction across ranks in the pilot is a direct consequence
   of the 2-dimensional run vector and resolves once the encoding is richer.)
2. **Chance-quality `xg` source — RESOLVED 2026-10-17 (owner: Tier 1 + Tier 2).** silly-kicks ships no
   xG model by design (ADR-110 raw primitives; ADR-085 injected PSxG; `shot_stopping/_compute.py`), so
   no model is built. Implemented:
   - **Tier 1 (model-free backbone).** `compute_corner_chance_quality` makes `xg_column` optional:
     `n_shots`/`n_goals` are always computed (fully observed, provider-uniform); `sum_xg` is honest-NaN
     when no xG is injected (never a fabricated 0.0); a *named* missing `xg_column` still raises.
     `validate_corners.reduce_samples` runs responsiveness / discrimination / cross-provider on each of
     `n_goals`, `n_shots`, `sum_xg`; discrimination groups by a **provider-qualified canonical team key**
     (ADR-019 — pooled `team_id` is mixed-dtype across providers and per-provider namespaced).
   - **Tier 2 (per-provider native xG companion).** The loader surfaces each provider's own shot xG as a
     uniform `xg` via the sanctioned `preserve_native` passthrough (`_normalize_native_xg`): sportec/IDSSE
     `shot_x_g`, statsbomb nested `extra.shot.statsbomb_xg` (`_sb_shot_xg_column`). SkillCorner has no
     clean shot xG → honest-NaN `sum_xg`. Per provider, never pooled across different xG models.
   Verified on the 47-match pilot: all providers now carry `n_goals`/`n_shots`; sportec/statsbomb
   populate `sum_xg`, SkillCorner `sum_xg` honest-NaN.

## 16. Amendment 2026-10-18 (owner-approved) — target-zone construction + routine-NMF regularization

The first full-corpus fit (2702 corners / 26732 player-snapshots) exposed two degeneracies in the
model construction. Both are measured, calibrated on the full corpus, and owner-approved; they change
`RunZoneModel.fit`, the encoder, and the `CornerRoutineModel` NMF default, so the fitted weights are
re-fit and this amendment carries its own independent A/B review. Supersedes the §9/§6 construction.

- **(D) Target-zone region = goal-centre radius ≤ `goal_radius_m` (default 25 m), NOT the literal
  penalty area.** The old active filter (`in_penalty_area_absolute` on the target-GMM component means)
  collapsed to **2** active zones on real data: the target GMM was fit over ALL attacking players,
  ~59% of whom sit at midfield (stay-back / short-safety), so only 2 of 16 component means landed in
  the 18-yard box. A literal box also drops the dynamics the metric exists to capture — **late runners
  arriving from deep** and **second-ball occupants just outside the box** (punches / clearances /
  pull-backs). Fix: fit the target GMM on the targets **within `goal_radius_m` of the goal centre**
  (box + top-of-box + second-ball/edge); every fitted component is then an attacking target zone
  (`active_target_idx` = all). Calibration (full corpus): a ≤25 m radius → **BIC k = 7** zones
  (near/far post, 6-yard, penalty-spot, top-of-box, and two second-ball/edge zones) — matching Shaw's
  ~7; ≤16.5 m → 4, ≤22 m → 5. Init zones stay unrestricted (init radial median ≈ 36 m — deep late-runner
  origins). `goal_radius_m` is a `RunZoneModel` param, serialized in the bundle and declared in the
  ADR-050 feature contract (a radius change forces a re-fit).
- **(E) Encoding rule = OCCUPANCY, not displacement.** A player counts iff their TARGET is in the
  region; the `min_run_distance_m` displacement threshold is removed (`RunEncodingParams` loses that
  field). Measured: **31.8 %** of second-ball-band occupants (goal-radius 16.5–25 m) move < 2 m, so a
  displacement gate would have dropped the stationary second-ball/edge holder. Deep stay-back players
  (target at midfield) are excluded by the region.
- **(F) Routine NMF regularization: default `alpha` 1.0 → 0.0.** With `alpha_W = alpha_H = 1.0` and
  `l1_ratio = 1.0`, the L1 penalty on an occupancy term matrix (values ~0–1) drove W and H to ≈ 0, so
  `‖X − W·H‖_F` equalled `‖X‖_F` **constant across every rank** (elbow stuck at 2 — a dead basis). The
  liveness toy passed `alpha = 0.0` and never saw it; the fit driver used the default. Plain Frobenius
  NMF (`alpha = 0.0`, still tunable) restores a monotone-decreasing reconstruction curve
  (52.9 at k=2 → 24.4 at k=7 → 1.9 at k=15 → plateau 0.8 at k≥16), elbow-selected **rank 15**.

**Verified (2026-10-18, full corpus 2702 corners):** `n_active_target = 7`, `n_runs = 42` (6×7),
routine reconstruction monotone-decreasing, selected rank 15; local `tests/corners/` + validate/build
driver tests + repo-wide gates green, `ruff` + `pyright` clean. The deliverable re-fit (clean tree) is
run after this amendment's A/B review + owner-gated commit; the degenerate first-run weights are
discarded.
