# TF-79 — Ball-contact detection primitive (tracking)

**Status:** IMPLEMENTED in 4.136.0 — brainstormed + cross-session-reviewed (A+B APPROVE) 2026-10-14; amended 2026-10-15 from real-corpus E2E (see Amendment below).
**Owner:** Karsten
**Cycle:** single feature branch `tf-79-ball-contact-detection`
**ADR:** **ADR-124** (RESOLVED, owner ruling 2026-10-15) — the next free number after the frozen, reviewed part-deux hf-jobs cycle (`origin/feat/hf-jobs-training-harness @ 6f6508a`, PR#284), which holds the two below it; TF-79 took the next free number rather than disturb that frozen commit. Release versions are independent: TF-79 = ADR-124 / v4.136.0; part-deux releases after at v4.137.0.
**Depends on:** TF-43 `tracking/_elastic_sync` (shipped), TF-48 `tracking/_shot_goalmouth` (prior art, untouched)
**Consumed by:** TF-46 Arm 1 (open question — target snapshot may need first-contact time), TF-46 Arm 2(c) corner-GK (kick→contact displacement, GK-at-contact), the deferred PathCRF "missing ball-touch" item (candidate list → Needleman–Wunsch alignment)

---

## Amendment 2026-10-15 (empirical E2E findings — gold-standard revision)

Real-corpus E2E (idsse / gradientsports / skillcorner, owner-directed before commit) changed three things from the approved design. All additive; owner-approved (gold standard, minimal data loss):

1. **Candidates are RANKED, never hard-dropped (output-contract change).** A dropped real touch is unrecoverable downstream (NW alignment can't realign what isn't there), so recall is paramount at this candidate stage. The contacts table gains a **`velocity_change_ms`** column (|Δv| at the contact — the physical touch-vs-flight-noise signal; measured touch |Δv| p50 ≈ 4.85 m/s vs flight-noise p90 ≈ 3.25 m/s on IDSSE), and `confidence` is **recalibrated to be monotone in |Δv|** (+ source agreement + proximity + detection quality) so flight-curvature candidates rank near 0 and real touches near 1. `add_ball_contacts` surfaces `first_contact_velocity_change_ms`. A new `BallContactParams.min_velocity_change_ms` is an **opt-in** hard floor **defaulting to 0.0 (no loss)**; consumers threshold on `velocity_change_ms`/`confidence`. (§3, §4, §6 below read with this change.)
2. **§8 validation reframed from tight-window precision/recall to CHARACTERISATION.** SPADL event timestamps LAG the physical contact (median ≈ −0.4 s on IDSSE, IQR ~1.4 s) — the exact event-feed misalignment TF-79/PathCRF exists to expose — so a single ±0.2 s precision/recall measures misalignment, not the detector. The memo now reports: contacts/min, honest unresolved rate, the **signed event→contact offset distribution**, **agreement-vs-tolerance** (τ ∈ {0.2, 0.5, 1.0}s), **confidence calibration** (event-match rate by confidence bucket), and the **|Δv| touch-vs-flight separation**. Still reported-not-gated. (Supersedes §8.2's precision/recall framing below.)
3. **Provider caveat — Gradient Sports.** GS delivers a pre-smoothed ("blunted") ball trajectory (contact |Δv| p50 ≈ 0.97 m/s vs IDSSE ≈ 4.85), so ball-contact detection is **degraded on GS** (under-fires; low agreement). Confirmed to be the provider's delivered data, not silly-kicks preprocess (`smooth_frames` is additive — it emits `x_smoothed`, never overwrites `x`/`y`; detect reads raw `x`/`y`). Documented like the SkillCorner-`z` / velocity-regime caveats. Validated on IDSSE (optical) + SkillCorner (broadcast); GS reported with this caveat.

---

## 1. Motivation

Several downstream features need to know **when the ball was first touched and by whom**, derived from tracking rather than taken from a provider's event feed:

- TF-46 corner analysis needs the first contact after a corner restart (winner, time) — SkillCorner GI first-contact labels are *possession-based*, not contact-based, and agreed with a tracking contact in only 55% of a 29-corner pilot sample.
- TF-46 Arm 2(c) needs the GK position at first contact and the kick→contact displacement.
- The deferred PathCRF item needs a raw list of candidate touches to align against the event stream and insert dropped touches.

Owner ruling (2026-10-13): build **one shared primitive** that these consumers call, not a detector private to TF-46 (the single-implementation rule, cf. ADR-055). TF-79 is the **candidate-touch stage** of that work; the Needleman–Wunsch event alignment and inserted-touch-event stage stay in the deferred PathCRF item.

A feasibility pilot (external, SkillCorner opendata, A-League, 10 matches, MIT; not reproduced in-repo) found a tracking contact in only 49% of broadcast corners using a single kinematic rule (ball speed ≥ 8 m/s sustained; first contact = first turn > 35° or speed drop > 40% with a player within 2 m). On broadcast, the dominant limiter is ball-undetected frames, not the rule. On dense optical the rule choice drives recall/precision — which is why validation runs on dense optical first.

## 2. Scope and non-goals

**In scope:** a pure, tracking-derived function that turns a ball trajectory (with `z`) plus player positions into a table of candidate ball contacts, and an additive action-grain enricher for the first contact in an action window.

**Non-goals (explicit):**
- **No Needleman–Wunsch event alignment, no inserted touch events.** That is the deferred PathCRF item. TF-79 emits candidates only.
- **TF-48 `_shot_goalmouth` is not touched.** It *verifies* a known shot's contact; it does not *discover* touches. Re-pointing TF-48 at this primitive is value-changing and needs its own ruling (Chesterton's Fence).
- **No PathCRF code is lifted.** PathCRF is MPL-2.0; silly-kicks is MIT. RDP (Ramer–Douglas–Peucker, 1972–1973) is a classical, unencumbered algorithm implemented fresh here.
- **No `*_xfns`.** This is a producer primitive (like TF-5 `infer_ball_carrier`, TF-48), not a VAEP feature function.
- **No retrain.** Additive, pure; no trained artifact.

## 3. Public surface — `silly_kicks/tracking/_ball_contact.py`

```python
@dataclass(frozen=True)
class BallContactParams:
    rdp_epsilon_m: float = 0.3            # 2-3x optical ball positional noise (~0.1 m): above noise, below a real deflection
    speed_drop_frac: float = 0.40         # velocity-break: fractional speed drop (pilot/lit)
    heading_turn_deg: float = 35.0        # velocity-break: heading change, degrees (pilot/lit)
    contact_radius_m: float = 2.0         # attribution radius (TF-48 prior: _CONTACT_RADIUS_M)
    contact_exist_radius_m: float = 5.0   # looser existence bar (TF-48 prior: _CONTACT_EXIST_RADIUS_M)
    contact_max_z_m: float = 2.6          # playable ball height (TF-48 prior: _CONTACT_MAX_Z_M)
    aerial_z_m: float = 1.8               # chest/head boundary; meaningful only where z is MEASURED (§8.1 spike)
    merge_window_s: float = 0.12          # applied as max(merge_window_s, 2 frame-periods); below min inter-touch interval, above one frame
    min_candidate_speed_ms: float = 2.0   # ignore near-dead-ball jitter

def detect_ball_contacts(frames, *, params: BallContactParams = BallContactParams()) -> pd.DataFrame:
    """The primitive. Per-contact grain. One row per candidate contact."""

def add_ball_contacts(actions, frames, *, params: BallContactParams = BallContactParams()) -> pd.DataFrame:
    """Additive action-grain enricher: first contact within each action window.
    NaN-safe (ADR-003), pure (PURITY_ENTRIES / ADR-033). `frames` positional-or-keyword,
    optional kwargs keyword-only (ADR-078 call convention)."""

def first_contact_in_window(contacts, period_id: int, t0: float, t1: float) -> pd.Series | None:
    """Low-level window query the TF-46 corner logic calls directly."""
```

The surface is intentionally minimal (no speculative `ball_contact_at_action` alias — speculative API is debt). `BallContactParams` is a frozen dataclass; `__post_init__` validation rejects invalid combinations (e.g. `contact_radius_m > contact_exist_radius_m`).

## 4. Data contract — the contacts table

`detect_ball_contacts` returns one row per candidate contact:

| column | dtype | meaning |
|---|---|---|
| `period_id` | `int64` | period |
| `frame_id` | `int64` | frame id of the contact (matches `TRACKING_FRAMES_COLUMNS["frame_id"]`) |
| `time_seconds` | `float64` | period-relative time (s) (matches `TRACKING_FRAMES_COLUMNS["time_seconds"]`) |
| `player_id` | nullable (`Int64`/object, provider-native) | attributed toucher; **`<NA>` when unresolved** |
| `contact_x` | `float64` | ball x at contact, **frame-LTR** |
| `contact_y` | `float64` | ball y at contact, **frame-LTR** |
| `velocity_change_ms` | `float64` | |Δv| of the ball velocity vector at the contact (touch-vs-flight signal; Amendment 2026-10-15) |
| `confidence` | `float64` | `[0, 1]`, monotone in |Δv| (Amendment 2026-10-15); **NaN when unresolved** |
| `is_aerial` | nullable `boolean` | ball above `aerial_z_m`; **`<NA>` when `z` NaN or z-provenance "estimated"** (§6, §8) |
| `contact_source` | `category` | `{rdp, velocity_break, both, unresolved}` |

Grain key: `(period_id, frame_id)`. "Unresolved" is a kept row (ADR-042) with `contact_source="unresolved"`, `player_id=<NA>`, `confidence=NaN` — never a guess (E7).

`contact_x`/`contact_y` are emitted in the frame-LTR frame and **registered in `silly_kicks/reflection.py`** with the appropriate `ReflectionKind` (ADR-045), so a reflection pass transforms them correctly; `contact_source`/`is_aerial`/`confidence`/`velocity_change_ms`/`player_id`/`time_seconds` are declared non-geometric.

The action-grain `add_ball_contacts` output adds, to each action: `first_contact_time`, `first_contact_player_id`, `first_contact_x`, `first_contact_y`, `first_contact_velocity_change_ms`, `first_contact_confidence`, `first_contact_is_aerial`, `first_contact_source` (all NaN/`<NA>` when no contact resolves in the window).

## 5. Detection algorithm (pure numpy, per period, per ball track)

1. **Build the ball trajectory** from `is_ball` rows for the period, ordered by `frame_id`. `z` is read where present (`TRACKING_FRAMES_COLUMNS["z"]`, float32 storage → upcast to float64 at the kernel boundary per ADR-106).
2. **Gap handling (E7):** runs of NaN ball coordinates split the trajectory into segments. No interpolation across a detection gap; candidates are detected within segments only. The gap boundaries themselves are not contacts.
3. **Structural breaks (RDP):** `rdp(ball_xy, rdp_epsilon_m)` → interior vertices = noise-robust direction-change points.
4. **Kinematic breaks (velocity-break):** frames where the fractional speed drop exceeds `speed_drop_frac` **or** the heading change exceeds `heading_turn_deg`, with thresholds **scaled to the frame rate** (per-frame deltas normalised to per-second). Catches speed-only contacts (trap/cushion) that bend the path little.
5. **Union + dedup:** candidate frames = RDP ∪ velocity-break; collapse candidates within `max(merge_window_s, 2 frame-periods)`. A candidate carried by both signals gets `contact_source="both"`. **Representative frame of a merged cluster (pre-registered — the timing-error metric §8.2 reads it):** the frame of **maximum velocity-break magnitude** in the cluster (sharpest kinematic signal ≈ best contact-instant estimate); ties broken by the **earliest** frame. A cluster carried by RDP only (no velocity-break frame) takes the RDP vertex frame. The representative's `time_seconds`/`frame_id`/`contact_x`/`contact_y` are the emitted contact's; its `contact_source` is `both` if any member came from each signal, else the single source. This rule is unit-tested (§10).
6. **Attribution (z-aware 3D proximity):** for each candidate, find the nearest player within the attribution radius using the TF-48 playability rule (`playable = z NaN | z ≤ contact_max_z_m`): a ball passing overhead is not a contact. Prefer `contact_radius_m`; fall back to `contact_exist_radius_m` as the looser existence bar. No player in range → unresolved.
7. **Confidence** (§6) and **`is_aerial`** (§6) computed per attributed contact.

RDP, velocity-break, and attribution are each separately unit-tested (§10). No existing RDP implementation is in the library (the atomic-SPADL `_simplify` is unrelated action compression), so RDP is implemented and tested fresh.

## 6. Confidence, aerial, provenance semantics

- **`confidence ∈ [0, 1]`**, a documented monotone fusion of: break magnitude (larger speed-drop / sharper turn → higher), proximity closeness (nearer player → higher), and local detection quality (fraction of the surrounding window with the ball detected → higher). Monotonicity in each input is asserted in tests.
- **`is_aerial`** (nullable `boolean`): `ball z > aerial_z_m` at the contact frame. **Honest-NaN** where `z` is NaN, or where the provider's `z` is **estimated rather than measured** (§8 spike). The pilot's aerial shares (56–69%) are not trusted until the z-provenance question is answered.
- **`contact_source`** is a closed-vocabulary `category` column (`{rdp, velocity_break, both, unresolved}`) on the contacts table (and `first_contact_source` on actions), validated by a dedicated vocab test in `tests/tracking/test_ball_contact.py`.
  - **Amendment 2026-10-14 (RATIFIED — owner-approved in the implementation plan + the 2026-10-15 amendment):** the original text here said `contact_source` is "added to `TRACKING_CATEGORICAL_DOMAINS` in `tracking/schema.py`." That is INFEASIBLE — `test_tracking_schema.py::test_tracking_categorical_domains_keys_subset_of_columns` asserts every key of that dict is a column of the closed 19-col `TRACKING_FRAMES_COLUMNS` **frame** schema; `contact_source` is a contacts-table / actions column, not a frame column, so registering it there turns the full suite RED. Surfaced by TF-79 plan-review r1 (A `TF79-PLAN-01` / B `TF79-SPEC-03`); both spec reviews (r1/r2) checked the gate's name, not its assertion body, and missed it. Corrected home: a `category` dtype on the contacts table/actions with the fixed vocab above, not the frame-column domain registry.

## 7. Orientation / reflection

Contacts carry `contact_x`/`contact_y` in the frame-LTR frame (consistent with `orient_frames_to_ltr`). TF-46 reflect-normalises corners to one side downstream; TF-79 does not reflect on its own. The two coordinate columns are registered in `reflection.py` so that any reflection pass (and the `test_mirror_registry` Gate A/B check) handles them, and so a left-corner and right-corner scene of the same geometry produce mirror-equal outputs (tested, §10).

## 8. Validation (in-cycle, pre-registered — not deferred to the owner)

Validation is part of this cycle, not a follow-up.

### 8.1 z-provenance spike (run during spec execution, before the aerial design is finalised)
Determine whether each tracking provider's ball `z` is **measured or estimated**:
- Inspect kloppy's SkillCorner `z` handling and the SkillCorner tracking schema.
- Measure `z` variance / quantisation in an owner-tier pining sample (via the media-PC delegation path for heavy runs).

**Decision fork this gates:** if a provider's `z` is estimated (or constant/absent), `is_aerial` is honest-NaN for that provider and the aerial shares are reported as "unverified". Recorded in the TF-79 ADR (ADR-124) and in the validation memo.

### 8.2 characterisation driver

> **SUPERSEDED by the Amendment 2026-10-15 (see top).** The headline is NO LONGER a tight-window precision/recall vs event time — that measures the ~0.4 s event-feed lag, not the detector. The driver (`scripts/measure_ball_contact_validation.py`) now reports, loss-free: contacts/min, unresolved rate, the signed event→contact offset distribution, agreement-vs-tolerance (τ ∈ {0.2, 0.5, 1.0}s), confidence calibration (event-match rate by confidence bucket), and the |Δv| touch-vs-flight separation. The original precision/recall text is kept below for provenance.

A driver on the `scripts/_driver.py` seam (`for_each`, resume-before-load over cheap match refs, shard generation-dir + schema token) with a `require_clean_tree(git_provenance())` provenance stamp (both from `scripts/_provenance.py`, not `_driver.py`; `require_clean_tree` takes a keyword-only `allow_dirty`). It computes, vs event-recorded touches:
- *(superseded)* **precision**, **recall**, **timing error** (median |Δt| of matched contacts). A tracking contact matches an event touch iff it is the nearest event touch **within a pre-registered match tolerance `match_tol_s`** (§14): the SAME player (where both carry one) and |Δt| ≤ `match_tol_s`. `match_tol_s` is **fixed and pre-registered** (default `0.20 s`, derived from provider event-timestamp jitter, not fit to the accuracy numbers — see §14).
- Run on **dense optical first**: Sportec/IDSSE (public optical, precise event timing) and Gradient Sports (owner-tier).
- Then the **honest unresolved-rate** on SkillCorner broadcast (fraction of touches where the ball is undetected in the window).

### 8.3 reporting, not gating
No accuracy threshold gate (matching TF-46's "reference benchmarks, not shipped gates" stance). The precision/recall/timing numbers are **reported** in a memo at `docs/research/ball_contact_detection/README.md` (construct-validity memo shape per `docs/howto/construct-validity.md`), with the **corpus bound recorded inside the artifact** and aggregate-only figures under the reversibility policy. Private corpus locations are owner-supplied variables / env (the pining token), never literal paths in the committed memo.

What **does** block in CI: the non-vacuity, liveness, purity, reflection-invariance, and contract gates in §9.

## 9. CI surface a new tracking-feature module trips (full-suite-only in several cases)

TF-79 is a new **module inside the existing `tracking/` package**, and `add_ball_contacts` is an action-grain enricher, not a `compute_*` aggregator mart. So it does **not** trip `test_metric_contracts` (that registry — `METRIC_CONTRACTS` — holds only aggregator mart families such as `shot_stopping`/`territory`/`duels`; no tracking frame/action enricher is listed there) and it does **not** trip the new-sibling-*package* gates (import allowlist, etc.). The gates it does trip are the tracking-feature gates:

1. `PURITY_ENTRIES` / purity CI gate — `add_ball_contacts` and `detect_ball_contacts` input-mutation-free (ADR-033).
2. `test_call_convention_registry` — `add_ball_contacts(actions, frames, *, params)` registered (`frames` positional-or-keyword; `params` keyword-only) (ADR-078).
3. `test_aggregator_column_liveness` — the enricher's output non-NaN and non-constant on an in-domain fixture; the fixture gets its **own precondition test** (in-domain, no constant feature — the PR-S118 degeneracy, ADR-056).
4. `feature_glossary` — `FeatureColumn` entries + `describe_level` for every emitted column (count gate).
5. `NOTICE` — an entry naming RDP (Ramer–Douglas–Peucker) and citing PathCRF as inspiration-not-lifted (ADR-005).
6. per-`add_*` SB360 freeze-frame verdict (`verdict_provenance`, `UNAUDITABLE_BOUNDARY` empty, ADR-053). **Boundary note:** TF-79 is trajectory-based; a single SB360 freeze frame has no trajectory, so `add_ball_contacts` resolves **no contact** on SB360 (honest-NaN). The verdict records exactly this — SB360 is an unobservable input for this primitive, not an unaudited one.
7. `test_mirror_registry` Gate A/B + `reflection.py` registration for `contact_x`/`contact_y` (ADR-028/045); the c4 model (`docs/c4`) feature-column / aggregator-count pin.
8. `contact_source` closed vocabulary (`{rdp, velocity_break, both, unresolved}`) — a `category` dtype on the contacts table, validated by a dedicated vocab test in `tests/tracking/test_ball_contact.py`. **NOT** `TRACKING_CATEGORICAL_DOMAINS` / `test_tracking_categorical_domains_values` (that registry is frame-columns only; see §6 Amendment).

(Several of these fail **only in the full suite**; run `python -m pytest tests/ -m "not e2e"` before declaring done — Shift Left.)

## 10. Testing strategy (TDD — tests first)

- **RDP:** collinear points removed; a known polyline simplifies to its true vertices at a given epsilon; epsilon monotonicity.
- **velocity-break:** a planted speed-drop and a planted heading-turn each fire; a smooth trajectory fires nothing; frame-rate scaling (same physical event at 10 fps and 25 fps → same contacts).
- **merge / representative (§5.5):** two candidates within the merge window collapse to one contact; the representative frame is the max-velocity-break-magnitude frame (earliest on tie); an RDP-only cluster takes the vertex frame; two true touches spaced beyond the window stay two contacts.
- **attribution / z-aware:** a ball passing overhead (`z > contact_max_z_m`) is not attributed; nearest-player tie-breaks; no-player-in-range → unresolved row kept.
- **confidence:** monotone in each of the three inputs; bounded `[0, 1]`.
- **E7 / gaps:** NaN ball runs split segments; no contact synthesised across a gap; unresolved rows kept with the right provenance.
- **reflection invariance:** left-corner and right-corner mirror scenes give mirror-equal contacts (both sides tested).
- **non-vacuity:** on a synthetic match with planted contacts, contacts populate more than one `contact_source` value and more than one player.
- **synthetic recovery:** a hand-built trajectory with known planted touches is recovered (precision/recall on the fixture), and the fixture generator reproduces its committed fixture byte-for-byte first.
- **purity:** inputs unmutated.

## 11. File inventory

New:
- `silly_kicks/tracking/_ball_contact.py`
- `tests/tracking/test_ball_contact.py`
- `scripts/measure_ball_contact_validation.py` (driver; argparse — never a parserless script)
- `docs/research/ball_contact_detection/README.md` (validation memo, landed at cycle end)
- `docs/superpowers/adrs/ADR-124-ball-contact-detection.md`

Edited:
- `silly_kicks/tracking/__init__.py` (exports)
- `silly_kicks/tracking/schema.py` — **NOT edited** (contact_source is NOT a frame column; §6 Amendment). Left here struck for provenance.
- `silly_kicks/reflection.py` (`ReflectionKind` for `contact_x`/`contact_y`)
- the `feature_glossary` / reporting module (`FeatureColumn` entries)
- `NOTICE`
- `AGENTS.md` (TF table: TF-79 row — module + public surface)
- `docs/context/tracking-features.md` (narrative)
- `docs/c4` (model pin, via the Graphviz `dot` render pipeline)
- `TODO.md` (move TF-79 off On Deck on ship)

## 12. ADR-124 — decisions recorded

1. One shared candidate-touch primitive (not private to TF-46); single-implementation rule.
2. Output = per-contact table (shape A), additive action enricher, no `*_xfns`; not a `METRIC_CONTRACTS` family (enricher, not aggregator mart — a later contacts-based mart would register, this primitive does not).
3. Hybrid detection: RDP ∪ velocity-break, z-aware attribution, TF-48 radius/height priors reused.
4. `is_aerial` honest-NaN where `z` is absent or estimated; z-provenance per provider recorded.
5. Candidate-touch-only boundary; PathCRF alignment and TF-48 re-pointing explicitly out.
6. Validation in-cycle, reported (no accuracy gate), dense-optical-first.

## 13. Workflow and commit gate

- One feature branch `tf-79-ball-contact-detection` off `main`; no worktrees.
- This spec, the ADR, and the research memo stay **uncommitted on the branch until cycle end** (no doc commits until cycle end); the reviewer reads the uncommitted file.
- One coherent, fully-tested commit for the implementation (not micro-commits). Lint (`ruff check silly_kicks/ tests/ scripts/` + `--format --check`) and `pyright` and `pytest -m "not e2e"` green before done (Shift Left).
- **No `git commit` / `git push` without Karsten's explicit per-commit approval.** The session stops at commit-prep, shows the diff/file list, and waits.

## 14. Open spec-time items (decided defaults recorded; one live investigation)

- `is_aerial` on providers with no `z`: nullable `boolean`, honest-NaN (keep the column). **[decided]**
- Validation providers this cycle: all three (Sportec/IDSSE + Gradient Sports + SkillCorner unresolved-rate). **[decided]**
- Validation = reported memo + liveness/non-vacuity CI gates; no accuracy threshold. **[decided]**
- `metric_contracts`: TF-79 is **not** registered (enricher, not aggregator mart — §9); a later contacts-based *aggregator* mart would register, this primitive does not (the TF-79 ADR records the boundary). **[decided]**
- Parameter defaults are **fixed and pre-registered**, derived from physics/geometry + the TF-48 priors (see `BallContactParams` in §3). **[decided]**
- **z-provenance per provider (§8.1) — the one live investigation**, gating the `is_aerial` reliability claim. Run during spec execution.

### Parameter selection — no-leakage rule
Parameters are **never selected against the real optical accuracy corpus** — fitting thresholds to maximise precision/recall on the same data the accuracy is then reported on inflates the number. The only ground-truth surface parameters may be checked against is the **synthetic-recovery fixture** (planted contacts, known answers, no leakage). Accuracy (§8.2) is reported on the real corpus with the **fixed** defaults above. A real-corpus sensitivity analysis, if wanted, is a **pre-registered sweep reported separately** — never a "pick the best-performing value" step.

This rule extends to the **§8.2 evaluation parameters**, not just the detection parameters (per the Amendment 2026-10-15 the headline is now the characterisation grid — agreement tolerances `{0.2, 0.5, 1.0} s` and the `0.5 s` calibration-match rule — **fixed and pre-registered** reporting grids, never selected to flatter a number; the old single `match_tol_s=0.20 s` is superseded). A post-hoc tolerance — widening the window until more contacts "match" — inflates precision and recall, so it is forbidden; `match_tol_s` is derived from provider event-timestamp jitter and frozen. The merged-cluster representative rule (§5.5) is likewise pre-registered, since the timing-error metric reads the representative frame.
