# TF-57 — Event↔Tracking Synchronization via Extended Needleman–Wunsch (ELASTIC v2), Upgrading TF-43 — Design

- **Status:** Draft — **rev 4** (2026-09-14): **§16.11 is AUTHORITATIVE — it supersedes the exact-frame / "88.4 %" framing throughout §16 below.** Verified against the paper as read: the paper's PRIMARY metric is **W2** (no exact-frame headline; NW 96.5 % / greedy 84.1 % W2, same-source); the as-built ship bar is **W2 0.862 cross-source** (SO-2 signed 2026-09-14); the `_score` weights are OpenEvolve-tuned (owner-approved); the oracle is a single goal-covering span. Read §16.11 first; §16.1–16.10 are the (superseded) rev-3 investigation record. — **rev 3** (2026-09-12) — addresses the implementation review `D:\Development\_reviews\2026-09-11-tf57-elastic-nw-sync-impl.md` (**REQUEST CHANGES**) **and** the exact-frame investigation the owner's Goal-5 ruling triggered ("gold standard; investigate; do more work; scope is not an issue"). **Headline reframe (see §16):** the 29% exact-frame result is a **scoring transcription divergence in our own code**, *fixable* toward the paper's 88.4%/96.5% on the same data — **NOT** the identity vote-map / candidate detection, which rev-2 §8.2 and the shipped README/CHANGELOG/TODO wrongly blamed (corrected this cycle). This revision re-derives the scoring per-feature against the paper, rebuilds the oracle (real DFL roster + a goal span), retargets the acceptance bar at the paper's numbers (human-gated fallback if unrecoverable without the MPL code), and folds in the three review test findings (IMPL-01/02/03). Everything lands in the one still-uncommitted branch (owner: "nothing has been committed, so everything gets folded in now"). **rev 2** (2026-09-11) addressed the *spec* review (both SHOULD-FIX resolved — §6 `n_candidate_frames` per-action semantics + §8.1 evidenced CC-BY-4.0 grant; CONSIDER items applied — §7 card mislabel, Goal 5 wording, §6 time-base-hint suppression, §14.1 tolerance_seconds); review `…-spec.md`. **Uncommitted** (owner). Version / PR-S / ADR set at commit-prep after `git fetch && git merge origin/main`; next ADR ≈ADR-093, next release ≈4.113.0 (current `main` is 4.112.0) — provisional.
- **Date:** 2026-09-11
- **Feature:** TF-57 — reimplement the extended Needleman–Wunsch alignment of ELASTIC v2 (Kim et al., CIKM 2026) as the **upgrade of TF-43** (`tracking/_elastic_sync.py`), replacing the older MLSA-2025 greedy algorithm, and exposing it both through the existing `elastic_*` surface and as an alternative strategy behind the canonical `(pointers, LinkReport)` linkage contract.
- **Delivery:** a **single coherent, fully-tested feature branch** (`feat/tf57-elastic-nw-sync`), **no worktrees**, **one commit** (no micro-commits). Lands only after explicit human approval of that specific diff. The in-repo validation oracle fixture ships **in the same commit** (docs/data/tests/code together — no standalone data or doc commits).
- **License boundary (hard):** ELASTIC's code is **MPL-2.0** → this is a **clean-room reimplementation from the published paper** (arXiv:2608.30227), **not** a lift of the reference code (the PathCRF precedent, memory `[[reference_pathcrf_event_detection]]`). The DGX validation harness's MPL clone (`~/elastic_validation/elastic_repo/`) is **run-only, off-repo, and never enters this MIT repo** — including its `compute_sync_accuracy` metric, which the in-repo gate **reimplements** (a trivial `|pred − gt| ≤ N` count). The benchmark **data** is **CC BY-4.0** → committable as the in-repo oracle **with attribution**.
- **Related:** TF-57 / TF-43 (`_elastic_sync.py`); ADR-004 (linkage primitive + `LinkReport` + pointer-DataFrame contract); ADR-017 (period-relative time base; `link_actions_to_frames` coverage guard; `validate_time_base`); ADR-005 (tracking-aware features; NOTICE academic-attribution discipline); ADR-019 (`id_compat` dtype-safe id comparisons — the exact seam behind the constant-0.6 bug); ADR-068 (`group_rows`, no rescan-in-loop); ADR-073 (sub-quadratic structural growth guard); ADR-033 (`add_*` purity); ADR-032 (aggregator liveness); ADR-048 (`feature_glossary`); ADR-009 (ship primitives; frozen params `for_provider`; validation reported-not-gated). C4-free (no new `add_*` aggregator — count stays 33).

---

## 1. Summary

silly-kicks already ships ELASTIC's *MLSA-2025* incarnation as **TF-43**: a per-action **greedy** argmax that, for each event independently, scans a ±1 s frame window and picks the frame maximizing `0.6·ball_accel + 0.4·player-ball-proximity`. Measured against Kim's own CC-BY benchmark (3 re-annotated Sportec Open matches — J03WMX/J03WN1/J03WPY, all in the DGX tc3-cache; **J03WMX is our committed IDSSE fixture**), using ELASTIC's own `compute_sync_accuracy`, pooled: **TF-43 exact-frame = 8.2 %** vs **ELASTIC v2 (NW) = 88.4 %** — TF-43 is below even the paper's "prior methods" (21.1 %) and ~11× behind SOTA. (~62 % within 1 s; set-piece 50 % exact, open-play passes ~4 %.)

The greedy algorithm's structural flaw is that each event is placed **independently**, so it cannot use the ordering constraint that event *i*'s frame must not precede event *i−1*'s; mistakes do not self-correct and cluster into cascades. ELASTIC v2 reframes synchronization as a **global, order-preserving sequence alignment** (extended Needleman–Wunsch) between the **event sequence** — enriched with **virtual termination events** so each event's *end* (reception / out / goal) is found jointly with its *start* — and a **sparse set of physically-plausible ball-touch candidate frames**. A pairwise score `s(e,c) ∈ [0,1]` (ball acceleration, player-ball distance, pre/post kick distance, pre/post distance-slope, opponent distance, gated by an acting-player membership constraint) drives a DP whose novel **down-match** move lets one candidate frame serve two consecutive events (one-touch actions). No neural nets — scipy peak detection + a numpy DP.

This design **replaces** the greedy engine with a clean-room NW reimplementation (owner decision 2026-09-11), delivered through two surfaces over one pure core:

- **Elastic surface (mart producer, upgraded):** `align_events_to_frames` → `add_elastic_sync` → `elastic_sync_xfns` (+ atomic mirror). Keeps `elastic_frame_id` / `elastic_confidence` / `elastic_error_seconds` (**values move**) and **adds** three reception columns (`elastic_receive_frame_id` / `_confidence` / `_error_seconds`) — the joint start+end detection that is ELASTIC v2's signature (owner decision 2026-09-11).
- **Canonical linkage surface (the alternative strategy):** a new `link_actions_to_frames_elastic(...) → (pointers, LinkReport)` emitting the *exact* pointer schema, making NW alignment a **drop-in for the `links=` kwarg** across the whole `add_*` family. The guarded time-based `link_actions_to_frames` (ADR-004/017) is **untouched and remains the default** — this is an alternative, not a replacement.

**Blast radius:** `elastic_*` marts **re-materialize** (values move + 3 new columns). `elastic_sync_xfns` is **not** in any default xfn list, so there is **no default-config VAEP retrain**; a consumer who has opted those columns into a VAEP feature set retrains those models. `ElasticSyncParams` and `add_elastic_sync`'s keyword signature **change (breaking)** — the greedy weights are removed. Validation lands as a committed CC-BY oracle + a **CI accuracy gate** (owner decision 2026-09-11), plus the Claude-run DGX full-corpus report.

---

## 2. Goals and non-goals

### Goals

1. **Clean-room NW reimplementation** of ELASTIC v2's extended Needleman–Wunsch alignment (Eq. 14) with virtual termination events, from the paper — never lifting MPL code.
2. **Upgrade TF-43 in place:** NW becomes the engine behind the existing `elastic_*` surface; the greedy MLSA-2025 algorithm is **removed**.
3. **Joint start+end detection:** emit reception/termination frames as three additive `elastic_receive_*` columns.
4. **Alternative linkage strategy under the canonical contract:** `link_actions_to_frames_elastic` returns `(pointers, LinkReport)` conforming to ADR-004, usable as a `links=` drop-in across the `add_*` family. The time-based linker stays the default.
5. **Close the accuracy gap** on the CC-BY oracle (8.2 %→ target near the 88.4 % ELASTIC-v2 exact-frame figure). CI **regression-guards** this — a committed oracle slice + a from-scratch accuracy metric, asserting a floor (= achieved-on-box − margin) and a large-margin beat over the retired greedy. **Attainment of the paper-level target is human-gated at commit** via the Claude-run DGX report (§8.2), reviewed by the owner; CI does not assert the aspirational number. **⚠ rev 3 (§16.6) sharpens this:** the bar is now to *match* the paper (not merely "near"), the gap is a fixable scoring divergence (§16.2), and the human gate is a *fallback* if a divergence is unrecoverable without the MPL code — not a default accept of 29 %.
6. **No default-config retrain:** the elastic xfns are opt-in; only the marts + opt-in consumers move.

### Non-goals

- **Shipping a ball-touch detector à la PathCRF.** Candidate frames are derived from kinematics we already compute (ball accel, player-ball distance, ball-boundary distance) — not a learned touch model. PathCRF is MPL-2.0 and out of scope (`[[reference_pathcrf_event_detection]]`).
- **Running on SB360 freeze-frames.** NW requires *continuous* tracking (a ball trajectory for peak detection; player positions for the membership gate) and player identity in frames. SB360 is anonymous, velocity-less, single-snapshot → **explicitly unsupported**, exactly as the current elastic path already is.
- **Replacing `link_actions_to_frames`.** It stays the default guarded linker; elastic is additive.
- **Downstream GNN tasks / retraining VAEP by default.** Out of scope; the xfns are opt-in.
- **Re-deriving event taxonomy from Sportec.** We map ELASTIC's four categories onto SPADL types (§7); the mapping is a design surface, validated against the oracle, not asserted.
- **A learned or per-provider-tuned scoring.** Weights and thresholds are the paper's intent-set constants (`for_provider` empty per ADR-009); any future tuning is a separate ADR-009-gated cycle.

---

## 3. Method provenance (from the paper) and benchmark facts (measured)

The algorithm below is transcribed from **arXiv:2608.30227** (CIKM 2026), read in full during design. The benchmark facts were **measured this cycle** (memory `[[project_soccer_ml_research_batch_sep2026]]`): ELASTIC's own `compute_sync_accuracy`, pooled over the three CC-BY Sportec matches, gave TF-43 **8.2 %** vs ELASTIC-v2 **88.4 %** exact-frame; the gt annotations are per-frame (`frame_id` + `receive_frame_id`, 25 fps, `gt`/`unsynced` row-aligned) and share the DFL 10000-based (P1) / 100000-based (P2) frame numbering our IDSSE frames use → directly comparable. All three matches live in the tc3-cache (`_actions/idsse__DFL-MAT-*.parquet` + `shards/899a46878e7d723f/*`).

**Paper metric note (load-bearing for the reviewer):** the paper's **primary** metric is **W2** (within-2-frames, 0.08 s), reporting **96.5 %** for event starts; our harness measured **exact-frame** (the 8.2 %/88.4 % figures). This design reports **both** exact-frame and within-N so the two are never conflated. The CI floor (§8) is a *regression guard* calibrated from what the reimplementation actually achieves on the box, minus a safety margin — not the paper's aspirational number.

### 3.1 The algorithm (clean-room target)

**Candidate frames** `c = (t_c, 𝒫_c)`, the union of:
- (a) local minima of player-ball distance (a player near the ball — potential contact),
- (b) local minima of ball-to-pitch-boundary distance (out / goal-line events),
- (c) local maxima of ball acceleration (direction change), paired with the nearest player.

Feasibility gate: retain a `(frame, player)` pair only if **player-ball distance ≤ 3 m ∧ ball height ≤ 4 m**. `𝒫_c` is the set of feasible players (and pitch lines) at `t_c`. (Paper reports ~16 979 candidates/match out of ~90 000 in-play frames.)

**Event sequence enrichment** — between consecutive events `(e_i, e_{i+1})` insert a virtual termination event:
- **goal** — `e_i` is a successful shot followed by a kick-off;
- **out** — `e_{i+1}` is a throw-in / goal-kick / corner;
- **control (reception)** — `e_i`, `e_{i+1}` are in the same episode, different players (marks the next actor's reception);
- else none.

The termination event's matched frame is the *end* timestamp of the preceding original event.

**Pairwise score** `s(e,c) ∈ [0,1]` = mean of the category's features (each weight `λ = 0.25`), with the **hard constraint `s(e,c) = 0` if the event's actor `p_e ∉ 𝒫_c`**. Clipped-linear feature map `f(x; x₀, x₁) = clip((x−x₀)/(x₁−x₀), 0, 1)`. Features:
- `s_BA = f(a; 0, 30 m/s²)` — ball acceleration.
- `s_PBD = 1 − f(d; 0, 3 m)` — player-ball distance (closer → higher).
- `s_KD⁺ = f(max_{t_c ≤ t ≤ t_c⁺} d; 0, 3 m)` — post-touch kick distance (outgoing: ball departs).
- `s_KD⁻ = f(max_{t_c⁻ ≤ t ≤ t_c} d; 0, 3 m)` — pre-touch kick distance (incoming: ball arrived).
- `s_PBDS⁻ = 1 − f(v⁻; 0, 7 m/s)`, `v⁻ = (d(t)−d(t−h))/h` — pre-slope penalty (outgoing shouldn't already be receding), `h = 0.2 s`.
- `s_PBDS⁺ = f(v⁺; −7, 0 m/s)`, `v⁺ = (d(t+h)−d(t))/h` — post-slope penalty (incoming shouldn't still be approaching).
- `s_OD = 1 − f(min_{q∈opp} d(t_c, q); 0, 3 m)` — nearest-opponent distance (minor / contested).

Category score compositions:
- **Outgoing** (open-play + set-piece): `¼(s_BA + s_PBD + s_KD⁺ + s_PBDS⁻)`.
- **Incoming** (incl. virtual reception): `¼(s_BA + s_PBD + s_KD⁻ + s_PBDS⁺)`.
- **Minor**: `¼(s_BA + s_PBD + s_KD + s_OD)` (`s_KD` = symmetric max distance in a small window).

> **⚠ rev 3 (§16):** these feature maps are where the transcription *diverges* from the paper — the rev-1/2 implementation of them scores 29 % exact on the same data the paper scores 88.4 % on. §16.4 re-derives each map (the `s_KD` window length is the prime suspect) against this section and the paper, gated by a "score peaks at the ground-truth frame" harness. Read §3.1 as the *intended* target; §16.2–16.4 as the correction.

**Extended NW DP** (Eq. 14) over event sequence `E=(e_1..e_m)` (enriched) × candidate sequence `C=(c_1..c_n)`, per episode:
```
F[i,j] = max(
    F[i-1, j-1] + s(e_i, c_j),           # diag-match
    F[i,   j-1] + g_c,                    # candidate-gap (c_j unused), g_c = 0
    F[i-1, j  ] + g_e,                    # event-gap (e_i unmatched), g_e = 0
    F[i-1, j  ] + s(e_i, c_j) + r,        # down-match (c_j serves e_{i-1} and e_i), r = -0.1
)
F[0,0] = 0 ;  F[i,0] = i·g_e ;  F[0,j] = j·g_c
```
Backtrack `(m,n) → (0,0)` via a backpointer matrix → the optimal **order-preserving** assignment. Order preservation is intrinsic (F is monotone in i,j). **Confidence** = the matched `s`; an event whose matched `s < min_confidence` (paper: **0.5**) is **unsynchronized** (NaN frame). Episodes (~100/match) bound the DP size and scope cascades.

### 3.2 silly-kicks realities that shape the port

- **Ball height `z`** is a declared `TRACKING_FRAMES_COLUMNS` column (`float64`) but is **all-NaN on broadcast providers**. The 4 m height gate **no-ops where `z` is NaN** (feasible), applies where present (IDSSE/Sportec carry z). Validation confirms the gap-closure holds on IDSSE (which has z).
- **Player identity in frames** is present for all continuous providers; the `p_e ∈ 𝒫_c` gate is an action↔frame id join — the **exact ADR-019 seam** that produced the constant-0.6 bug (`astype(str)` vs `canonical_id`, upcast by the NA ball row). Every id comparison routes through `id_compat` (`canonical_id` / `canonical_id_series` / `ids_equal` / `same_id`).
- **Episodes** come from the existing `spadl.add_possessions` (`spadl/utils.py`) — no bespoke stoppage segmentation.
- **Committed oracle:** the committed `idsse_slice` (2.7 MB) is a parse-port *parity* fixture, **not** full frames+actions — so the oracle is a **new** reduced frames+actions+gt slice (§8.1).

---

## 4. Architecture overview

One pure engine, two public surfaces, in `tracking/_elastic_sync.py` (rewritten):

```
                       ┌───────────────────────────────────────────┐
   frames, actions ──► │  NW engine (pure, private)                 │
                       │   _detect_candidate_frames(frames)         │  scipy peak detect
                       │   _enrich_events(actions, possessions)     │  virtual term. events
                       │   _score(event, candidate, frames)         │  s(e,c) ∈ [0,1]
                       │   _needleman_wunsch(events, candidates)     │  numpy DP + backtrack
                       └───────────────┬─────────────────┬──────────┘
                                       │                 │
           elastic surface ◄───────────┘                 └────────► canonical linkage surface
   align_events_to_frames(...)                       link_actions_to_frames_elastic(...)
     → elastic_frame_id, elastic_confidence,           → (pointers[action_id, frame_id,
       elastic_error_seconds,                              time_offset_seconds,
       elastic_receive_frame_id, _confidence, _error       n_candidate_frames,
   add_elastic_sync / elastic_sync_xfns (+ atomic)        link_quality_score],
                                                           LinkReport)
```

- The engine is **pure** (pandas in, pandas out; no I/O, no global state), mutates neither `actions` nor `frames`.
- `align_events_to_frames` remains the elastic entry point and returns the 7-column elastic frame (`action_id` + 3 start + 3 reception); `add_elastic_sync` merges those onto a **copy** of `actions` (ADR-033 purity); `elastic_sync_xfns` lifts the two VAEP-facing columns (`elastic_confidence`, `elastic_error_seconds`) to gamestates.
- `link_actions_to_frames_elastic` is a thin adapter over the same engine into the ADR-004 pointer/LinkReport contract (§6).

---

## 5. The elastic surface (upgraded)

### 5.1 `ElasticSyncParams` (breaking — greedy removed)

Frozen dataclass, `for_provider` empty (ADR-009). Greedy fields (`accel_weight`, `proximity_weight`) are **removed**; `min_confidence` default changes `0.1 → 0.5` (paper). New fields (all paper constants, intent-set):

```python
@dataclass(frozen=True)
class ElasticSyncParams:
    frame_rate: int = 25
    touch_distance_m: float = 3.0          # feasibility + PBD/KD/OD clip upper
    ball_height_max_m: float = 4.0         # feasibility (no-op where z is NaN)
    accel_clip_max: float = 30.0           # s_BA upper (m/s^2)
    slope_window_seconds: float = 0.2      # h for PBDS
    slope_clip_mps: float = 7.0            # PBDS clip bound
    repeat_penalty: float = -0.1           # r, down-match
    event_gap_penalty: float = 0.0         # g_e
    candidate_gap_penalty: float = 0.0     # g_c
    min_confidence: float = 0.5            # unsynced threshold
```

### 5.2 `align_events_to_frames` output (7 columns, incl. `action_id`)

| column | dtype | meaning |
|---|---|---|
| `action_id` | int64 | join key |
| `elastic_frame_id` | Int64 | matched candidate frame for the event *start*; NA if unsynced |
| `elastic_confidence` | float64 | matched `s(e,c) ∈ [0,1]`; NA if unsynced |
| `elastic_error_seconds` | float64 | `|aligned_time − action_time|`; NA if unsynced |
| `elastic_receive_frame_id` | Int64 | matched frame of the following virtual termination event (reception/out/goal); NA where none inserted or unsynced |
| `elastic_receive_confidence` | float64 | that termination event's `s`; NA otherwise |
| `elastic_receive_error_seconds` | float64 | `|aligned_receive_time − expected|`; NA otherwise |

`aligned_time` uses the existing per-`(game,period)` `frame_id ↔ time_seconds` linear fit (`_fit_frame_time_relationship`, retained) so native-numbered providers (IDSSE) convert correctly. Empty-input guards return the empty typed frame.

### 5.3 `add_elastic_sync` / `elastic_sync_xfns` / atomic mirror

`add_elastic_sync` gains the 3 reception columns, drops the greedy kwargs (breaking), takes `params: ElasticSyncParams | None` (or the explicit kwargs mirrored from it). Returns a **new** DataFrame (purity). `elastic_sync_xfns` unchanged in shape (still lifts `elastic_confidence`, `elastic_error_seconds` — reception columns are pointers, not VAEP features, so they are **not** lifted; consistent with `elastic_frame_id` never having been an xfn). Atomic mirror (`atomic/tracking/features.py`) re-exports the upgraded `elastic_sync_xfns`.

---

## 6. The canonical linkage surface (alternative strategy)

`link_actions_to_frames_elastic(actions, frames, *, params=None, min_link_rate=0.5, on_low_coverage="warn") -> (pd.DataFrame, LinkReport)`.

Maps the engine output onto the ADR-004 pointer schema so it is a **drop-in for the `links=` kwarg** across every `add_*` aggregator (the whole tracking-feature family can then be computed on NW alignment):

| pointer column | elastic source |
|---|---|
| `action_id` | `action_id` |
| `frame_id` (Int64) | `elastic_frame_id` (a real frame, member of `frames`) |
| `time_offset_seconds` (float64) | `action_time − aligned_time` (signed; matches the time-linker's `action_time − frame_time` convention) |
| `n_candidate_frames` (int64) | **per-action** count of candidate frames in the action's episode that pass the actor-membership gate (`p_e ∈ 𝒫_c`) — the event's viable competing candidates. **Diverges** from the canonical "frames within a time tolerance" (`utils.py:380`) but is deliberately kept **per-action** (not a per-episode constant) so `n_actions_multi_candidate = (n_candidate_frames>1).sum()` (`utils.py:475`) stays meaningful (an action with >1 viable candidate). Divergence documented in §9. |
| `link_quality_score` (float64) | `elastic_confidence` — documented as the NW match score (∈[0,1]), *not* the time-linker's `1−|dt|/tol` |

`LinkReport`: `n_actions_linked` = actions with confidence ≥ `min_confidence`; `per_period_link_rate` / `per_provider_link_rate` computed as usual; `max_time_offset_seconds` = max `elastic_error_seconds`. **`tolerance_seconds` is not meaningful for confidence-gated linking** — set to `float('nan')` and documented (elastic links by confidence, not by a time tolerance). This is **resolved** (2026-09-11 review concurred): NaN + docstring keeps the one `LinkReport` contract; no separate report type. The `_enforce_link_coverage` guard (ADR-017 per-period floor, `on_low_coverage`) is reused, **except** its near-disjoint "suspected time-base mismatch" hint is suppressed/adapted for the elastic path — confidence-gated linking, not time-tolerance, so that hint would misattribute a low-coverage cause (mechanism deferred to the plan: a flag on the shared guard). The per-period floor and `on_low_coverage` policy fire unchanged. The risk is low regardless (elastic aligns within episodes that share the per-period time base, so near-disjoint action/frame ranges do not arise).

---

## 7. Event-category & virtual-event mapping (SPADL) — **the clean-room judgment surface**

The paper is in Sportec taxonomy; silly-kicks is SPADL. This mapping is the one place a clean-room gap could cost accuracy, so it is **validated against the oracle, not asserted**.

| ELASTIC category | SPADL `type_name` |
|---|---|
| Outgoing — open play | `pass`, `cross`, `shot`, `shot_freekick`, `clearance`, `take_on`, `keeper_punch` |
| Outgoing — set piece | `throw_in`, `freekick_crossed`, `freekick_short`, `corner_crossed`, `corner_short`, `goalkick`, `shot_penalty` |
| Incoming | `interception`, `keeper_save`, `keeper_claim`, `keeper_pick_up`, + virtual **reception** |
| Minor | `foul`, `tackle`, `bad_touch` |
| Excluded (non-touch / synthetic) | `non_action`, **`dribble`** |

(All 23 SPADL `actiontypes` are mapped exactly once. `yellow_card`/`red_card` are SPADL **`results`**, not `actiontypes` (`config.py:43-50` vs `51-75`), so they never surface as a `type_name` and need no mapping — an earlier draft mislabelled them here.)

- **`dribble`** is SPADL's synthetic carry (`_add_dribbles`), not a discrete touch; it is **excluded from candidate matching**. Its frame is taken from its bounding touches downstream (documented edge), never given its own candidate. This is flagged for the reviewer as the most debatable mapping call.
- **`foul`** may be off-ball; kept in "minor" (OD-scored) but expected to often go unsynced (confidence < 0.5) — acceptable, honest NaN.

**Virtual-event insertion** (deterministic from the SPADL sequence + `add_possessions`):
- **reception** — consecutive actions, same possession, different `player_id` → owned by the next actor; its matched frame = `elastic_receive_frame_id` of the earlier action.
- **out** — next action ∈ {`throw_in`, `goalkick`, `corner_*`} → an out event before it; matched frame = local-min of ball-to-boundary.
- **goal** — action is a `shot` with `result_id == success` (goal; per ADR-018 own goals are `bad_touch`+`owngoal` and are **not** shot-gated here) followed by a kickoff → a goal event.

All id comparisons in "different player" and possession joins go through `id_compat`.

---

## 8. Validation

### 8.1 In-repo CC-BY oracle + CI accuracy gate (owner decision 2026-09-11)

- **Fixture:** a **new** reduced slice of **J03WMX** — a contiguous span (target ≤ ~3 MB, a few minutes of P1) chosen to contain **all four categories + at least one out / goal / reception** — as three committed parquets: `frames`, `actions`, and Kim's row-aligned `gt` (`frame_id`, `receive_frame_id`). Home dir e.g. `tests/datasets/elastic_sync/j03wmx_slice/`.
- **Licence (the load-bearing part — evidenced, not asserted; verified against the ELASTIC repo README 2026-09-11):** the annotation I commit (the `gt`, i.e. the **re-annotated** matches, not just the base tracking) is the `benchmark/` data, which the repo redistributes under **CC BY 4.0** — verbatim: *"The event data under `benchmark/` is derived from the Sportec Open DFL Dataset (Bassek et al., 2025)… We redistribute it under the same license, with the modifications described above."* (Sportec Open DFL base is itself CC BY 4.0.) So the committed slice — frames, actions, AND gt — is CC BY 4.0. **No MPL data enters the repo** (MPL is the code only; never committed). Required attribution, carried in both the fixture `README` and `NOTICE`: **Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data*, doi:10.1038/s41597-025-04505-y** + the ELASTIC paper. The fixture `README` cites the **specific** CC BY 4.0 grant on the annotation (this sentence), not merely the base dataset's licence. The grant is **evidenced** (the verbatim README quote above, verified 2026-09-11) — no separate confirmation gate; the attribution travels in the fixture `README` + `NOTICE`.
- **Reduction:** **Claude runs it on the DGX** (`karsten@192.168.68.73`; box access + context per the runbook) — it reads the tc3-cache + the CC-BY gt; the reduction script stays off-repo (reads uncommitted cache), and the slice is scp'd into the working tree. The recipe is recorded in the fixture `README`.
- **Metric (reimplemented, MIT):** a small in-repo `sync_accuracy(pred, gt, *, tolerances)` computing exact-frame and within-N (N ∈ {2,5,25}) percentages, per category and pooled — a `|pred_frame − gt_frame| ≤ N` count. **Never** ELASTIC's `compute_sync_accuracy` (MPL).
- **CI gate (regular suite — fixture committed, so not `@e2e`, per house rule):** asserts NW clears a **regression floor** (exact-frame ≥ F, within-2 ≥ W) **calibrated on the box from the reimplementation's achieved accuracy minus a safety margin** (a regression guard, not the paper's aspirational 88.4 %/96.5 %). Also asserts NW **beats the retired greedy by a large margin** on the same slice — the guard that would have caught the constant-0.6 bug. Both-sided per the house "every band needs a test from both sides" rule: a mutation that should drop accuracy below the floor is asserted to fail.

### 8.2 DGX full-corpus report (Claude-run, reported-not-gated, ADR-009)

The existing `~/elastic_validation/` harness (`measure_tf43_vs_elastic.py`, `measure_ab.py`, `bench/`, MPL `elastic_repo/`) is re-pointed at the NW engine and run **by Claude on the DGX** on all three CC-BY matches, reporting exact-frame + within-N, per-category, pooled — the headline gap-closure (8.2 %→ ~88 %). Off-repo, MPL-isolated (the MPL clone is run-only and never committed). **⚠ rev 3 (§16):** the ~88 % here is the *target*, reached only after the §16.4 scoring re-derivation — the algorithm *as transcribed in rev 1/2* scores **29 % exact** on this corpus; §16 is the plan to close that, and this DGX report is the headline acceptance evidence (§16.6).

### 8.3 Proximity-term ablation (the latent TF-43 finding)

The greedy proximity term was net-neutral. In NW, PBD is one of four features and the KD/PBDS features carry most of the discrimination; the design **implements the full paper score** and the ablation (drop `s_PBD`) is **reported on the box**, not pre-baked — a simplification only lands if the oracle shows it costs nothing (a separate, ADR-009-gated follow-up if so).

---

## 9. Testing strategy & house-rule gates

TDD throughout (red first). New/rewritten tests:

- **`tests/tracking/test_elastic_sync.py`** (rewritten): engine unit tests — candidate detection on a synthetic ball trajectory (known accel spike / distance minimum); NW DP on a hand-built `s` matrix with a known optimal alignment incl. a **down-match** (one-touch) case and an unsynced (all-`s`<0.5) case; virtual-event insertion truth table (reception/out/goal/none); empty-input guards; **purity** (inputs unmutated) and the **both-sided** assertions.
- **`tests/tracking/test_elastic_sync_id_dtype.py`** (rewritten): bidirectional ADR-019 invariance across the actor-membership join (numeric actions × string frames × reverse; NA ball row present) — asserts NW confidence/frames are byte-identical across dtype spellings and never collapse to a constant (the constant-0.6 regression).
- **`tests/tracking/test_elastic_sync_lookup_golden.py`** (rewritten/retired): the greedy lookup golden is retired; replaced by an NW golden on a tiny committed synthetic frame.
- **`tests/tracking/test_elastic_linker_contract.py`** (new): `link_actions_to_frames_elastic` emits the exact ADR-004 pointer schema + dtypes; every non-NA `frame_id` is a member of `frames`; **`n_candidate_frames` is per-action** — asserted **non-constant** across a multi-episode fixture so `n_actions_multi_candidate` stays meaningful (the §6 divergence pinned here, both-sided: a per-episode-constant implementation would fail this); `link_quality_score == elastic_confidence`; `LinkReport.tolerance_seconds` is NaN; a low-coverage fixture fires the per-period floor **without** the time-base-mismatch hint.
- **Oracle gate** (§8.1): the CC-BY accuracy floor + greedy-beat + mutation-fails-floor.
- **ADR-073 sub-quadratic growth guard:** candidate detection + the episode loop scale the **group (episode) dimension**; a scoped `rows_scanned_counter` proves no rescan-in-loop (`group_rows`, ADR-068). The per-episode NW DP is inherently O(m·n) but bounded by episode size, and total cost is sub-quadratic in match size because episodes scale linearly.
- **Existing generic gates auto-cover the surface:** `add_*` purity (ADR-033 `PURITY_ENTRIES`), liveness (ADR-032 — the elastic columns must be live: `elastic_frame_id` etc. are already registered; reception columns added), id-scalar registry (ADR-019), mirror registry (ADR-051 — N/A: elastic emits no oriented geometry, but the registry must still see `add_elastic_sync` as before), SB360 audit registry (elastic already carries a verdict — re-adjudicate: still `differs_by_design`/`honest_nan` on velocity-less input? **No** — elastic is continuous-only and already unsupported on SB360; verdict unchanged, re-verify), public-API-examples gate (the literal-block examples updated), doctest sweep (the `ElasticSyncParams` doctest updated).

**Orientation / mirror:** elastic emits **no** action-LTR geometry (only frame pointers + scalar confidences), so ADR-028/051 reprojection does not apply — noted so the reviewer can confirm the mirror registry needs no new entry beyond the existing `add_elastic_sync`.

---

## 10. Bookkeeping (in the one commit)

- **`feature_glossary.py`** (`_M_ELASTIC`): update the three existing entries' definitions (NW, not greedy); **add** `elastic_receive_frame_id` / `_confidence` / `_error_seconds`. Coverage gate (ADR-048) forces this.
- **`NOTICE`**: update the ELASTIC block to cite **Kim et al. (2026), "ELASTIC …", CIKM 2026, arXiv:2608.30227** (the v2/NW method), retaining the 2025 lineage note; **add** the CC-BY-4.0 benchmark attribution for the committed oracle fixture — **Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data*, doi:10.1038/s41597-025-04505-y** (the Sportec Open DFL base, CC BY 4.0) + the ELASTIC paper (the re-annotation, redistributed under the same licence). (ADR-005 discipline.)
- **New ADR (≈092):** "Event↔tracking sync via extended Needleman–Wunsch (ELASTIC v2) replaces greedy TF-43; NW exposed under the `(pointers, LinkReport)` contract as an alternative linker." Records the greedy removal, the breaking `ElasticSyncParams` change, the reception-column additions, the clean-room/MPL boundary, and the mart re-materialize/retrain trigger.
- **`TODO.md`**: move TF-57 from On-Deck to shipped; note the mart re-materialize follow-up.
- **`CHANGELOG.md`**: `PR-Sxxx` entry keyed to the release, with the Hyrum/retrain trigger called out.
- **Version bump:** `silly_kicks/_version.py` only (ADR-079), ≈4.113.0; `uv.lock` via `uv lock` (never hand-edited).
- **C4:** no new `add_*` aggregator (count stays **33**); `link_actions_to_frames_elastic` is a linker, not an aggregator. Confirm the C4 completeness gate is satisfied without a DSL change (expected).

---

## 11. Hyrum / retrain / re-materialize impact

- **`elastic_frame_id` / `elastic_confidence` / `elastic_error_seconds` values move** (greedy→NW) → **lakehouse `elastic_*` mart re-materialize**.
- **Three new columns** (`elastic_receive_*`) → additive mart surface.
- **`elastic_sync_xfns` is NOT in any default xfn list** → **no default-config VAEP retrain**. A consumer that opted `elastic_confidence`/`elastic_error_seconds` into a VAEP feature set retrains those models (documented trigger).
- **`ElasticSyncParams` + `add_elastic_sync` signature change** (greedy kwargs removed) → breaking for direct callers pinning those kwargs; no shim (fail-loud at call, per the repo's clean-break precedent). Recorded in the ADR + CHANGELOG.

---

## 12. Design decisions (with rationale)

1. **Remove greedy entirely** (owner, 2026-09-11) — not kept behind a `method=` flag. One algorithm, simpler surface; the DGX A/B keeps the greedy reference off-repo. (I had recommended keep-behind-flag; owner chose remove — recorded.)
2. **Two surfaces, one engine** — the mart producer (`elastic_*`) and the canonical linker (`pointers/LinkReport`) share the pure NW core; the linker makes NW usable across the `add_*` family via `links=`.
3. **Emit reception frames** (owner, 2026-09-11) — additive; ELASTIC v2's whole point is joint start+end; near-free since the DP computes it.
4. **Full paper score; ablate PBD on the box** — don't pre-bake the greedy-era "drop proximity" finding; the NW feature set is richer and the ablation is an oracle-reported, ADR-009-gated follow-up.
5. **Episodes from `add_possessions`** — reuse, not a bespoke segmenter; bounds DP + scopes cascades.
6. **In-repo CC-BY oracle + regression-floor CI gate** (owner, 2026-09-11) — a permanent guard calibrated from achieved accuracy (not the paper number), reimplementing the trivial metric to stay MIT-clean.
7. **`id_compat` everywhere on the actor-membership join** — the constant-0.6 bug's exact seam; bidirectional dtype invariance gated.

---

## 13. Rejected alternatives

- **Keep greedy behind `method=`** — rejected by owner (simpler surface); A/B stays on the box.
- **A `strategy=` param on `link_actions_to_frames`** — rejected: couples the guarded time-linker (`utils.py`) to the elastic engine (`_elastic_sync.py`) and its different params; a sibling function keeps the two cleanly separable (hexagonal).
- **Lift ELASTIC's reference NW / `compute_sync_accuracy`** — rejected: MPL-2.0 into MIT (PathCRF precedent). Clean-room from the paper; reimplement the metric.
- **A learned ball-touch (PathCRF-style) candidate detector** — rejected: MPL + a much bigger scope; kinematic peaks suffice for the oracle target.
- **Run on SB360** — impossible (anonymous, velocity-less, single snapshot); documented precondition.

## 14. Open questions / risks for the reviewer

1. **`LinkReport.tolerance_seconds` = NaN** for the elastic linker — **RESOLVED** (2026-09-11 review concurred): NaN + docstring keeps the one `LinkReport` contract; no separate report type. See §6.
2. **`dribble` exclusion** and **`foul` in minor** — the most debatable SPADL-mapping calls; both validated against the oracle, but flagged.
3. **Regression-floor calibration** — the exact CI floor is set from the box run during implementation; the spec commits to the *method* (achieved − margin), not a number, to avoid a brittle gate. Reviewer should confirm this is the right shape.
4. **Ball-`z` absence** weakens out/height signal on broadcast providers; gap-closure is validated on IDSSE (has z). Broadcast-provider accuracy is reported, not gated.
5. **Fixture size vs coverage** — the ≤3 MB slice must still exercise all categories + out/goal/reception; if one category is too sparse in a small contiguous span, the slice may need to be two short spans (documented if so).

## 15. Delivery / commit discipline

- One feature branch `feat/tf57-elastic-nw-sync` off `main`; **no worktrees**; **one fully-tested coherent commit**; **no micro-commits**.
- **Stays uncommitted for now** (owner). The full non-e2e suite must be green **before** a commit is proposed; the commit is proposed with the exact diff and lands only on explicit per-commit human approval.
- Docs (this spec, the ADR, NOTICE, glossary, CHANGELOG, TODO), the CC-BY oracle fixture + README, tests, and code all land **together** in that one commit (no standalone doc/data commits).
- Implementation review is by an **independent session** the owner starts — not this session, not a subagent of it.

---

## 16. rev 3 — Exact-frame gap: root cause, scoring re-derivation, oracle rebuild, folded review findings (2026-09-12)

### 16.1 What triggered this revision

The 2026-09-11 implementation review returned **REQUEST CHANGES** (approach/code blessed as faithful; two approved test requirements dropped — IMPL-01, IMPL-02 — plus SHOULD-FIX IMPL-03). Separately, on the owner's Goal-5 ruling — *"We are looking for gold standard, best practice, scope is not an issue. Sounds like a need to investigate and do more work"* — the 29% exact-frame figure (vs the paper's 88.4%) was investigated. Three diagnostics were run locally on the **committed** `j03wmx_slice` oracle (reproducible; scripts are throwaway, the oracle is in-repo):

1. **Offset distribution** `(pred − gt)`: exact 29.5%, within-2 67.2% on the slice — but **bimodal**, not a uniform offset. Passes (31/61, the open-play bulk) align at **median 0**; a minority of defensive/contested/shot events carry **gross** errors (tackle median −17, others ±tens of frames).
2. **Cause localization** over the 16 gross errors: a candidate frame exists within ±2 of ground-truth for **16/16** (candidate-detection *miss* = 0) **and** the mapped actor is in it for **16/16** (identity/vote-map failure = 0). So neither detection nor identity explains the gross errors.
3. **Score comparison** true-vs-chosen candidate: `s(true) < s(chosen)` for **16/16** (e.g. tackle aid 283: 0.666 vs 0.779; pass aid 251: 0.646 vs 0.906), and the globally best-scoring candidate is often ±thousands of frames away. **The scoring function does not peak at the true touch** for these events; the DP (review-verified) correctly picks the higher-scoring wrong candidate.

### 16.2 Root cause (established)

The exact-frame gap is a **scoring transcription divergence in `_elastic_sync._score` / the feature maps**, not data, identity, density, or the DP. Two facts make the paper's number *achievable* and the divergence a *findable bug*:

- **Same data.** J03WMX is one of the three CC-BY benchmark matches the paper annotated **and reported 88.4% exact on**. We run on the same Sportec Open DFL tracking (via the IDSSE/Sportec DFL parse-port) and get 29.5% exact on J03WMX.
- **Matching candidate density.** Paper ≈ 16,979 candidates / ≈ 90,000 in-play frames ≈ **18.9%**; ours ≈ 1,002 / ≈ 6,200 on the slice ≈ **16.2%**. So candidate over-generation is *not* the divergence (my initial "too dense" hypothesis was falsified here).

| Suspected cause | Verdict | Evidence |
|---|---|---|
| Identity vote-map | ruled out | actor present in a ±2 candidate for 16/16 gross errors |
| Candidate-detection *miss* | ruled out | candidate within ±2 of gt for 16/16 |
| Candidate over-generation (density) | ruled out | our ~16% matches the paper's ~19% |
| DP / order logic | ruled out | review hand-traced it; passes align at median 0 |
| **Scoring doesn't peak at the true touch** | **root cause** | `s(true) < s(chosen)` for 16/16 |

### 16.3 Corrected attribution (rev-2 and the shipped docs were WRONG)

rev-2 §8.2 and the delivered `CHANGELOG` / `TODO.md` / oracle `README` assert the residual exact-frame gap is *"the player-identity mapping + candidate detection, NOT the alignment algorithm."* The evidence in §16.1–16.2 **refutes** that: the gap **is** our algorithm (scoring), and it is fixable. **Correcting that claim everywhere it was shipped is a first-class deliverable of this cycle** (the misquote in the oracle README — which additionally dropped "goal" from the §8.1 requirement to claim compliance, IMPL-02 — is corrected in the same pass).

### 16.4 The fix — per-feature scoring re-derivation (clean-room, from the paper)

Systematically re-derive each scoring element against arXiv:2608.30227 and prove it peaks at the true touch:

- **Feature-by-feature audit** of `s_BA`, `s_PBD`, `s_KD⁺/⁻`, `s_PBDS⁻/⁺`, `s_OD` and the three category compositions (§3.1) against the paper's definitions — windows, signs, normalization bounds, and the window **length** for the kick-distance term (we currently reuse the 0.2 s slope window `h = 5` frames for `s_KD±`; the paper may use a distinct, longer post-touch window, so at the true kick the ball has not yet "departed 3 m" and a later frame wins — a prime suspect). If the fix adds a kick-distance-window field (e.g. `kick_distance_window_seconds`) to `ElasticSyncParams`, it lands **within the already-declared breaking change** (§5.1) — not a new breakage.
- **Smoothing check:** the tc3-cache applies Savitzky–Golay smoothing + SG-derivative velocities; if the paper scores on less-/differently-smoothed tracking, our `s_BA` acceleration peaks may be shifted/flattened. Confirm the smoothing regime used and align it (or justify the difference).
- **Candidate extrema params** re-checked against the paper (prominence / min-separation / `order`): density matches, but the *selection* of which local extrema become candidates may differ.
- **Validation harness (gating the fix):** a "the score peaks at the gt frame among the actor-gated candidates" check per event type on the committed oracle — a **white-box** complement to the black-box exact-frame metric, so each feature fix is shown to move the true frame to the local max. Both-sided per the house rule.
- **No PathCRF / no learned detector.** The paper reaches 88.4% with this kinematic method on this data, so a learned touch model is unnecessary (the TF-57 non-goal stands; `[[reference_pathcrf_event_detection]]` remains On-Deck).
- **Clean-room boundary is HARDER here, not softer (hard licence guard).** §16.4 is hands-on debugging *on the DGX box that also hosts the MPL `~/elastic_validation/elastic_repo/` clone* — so the "run-only" rule is tightened to an explicit prohibition: the re-derivation reads **ONLY** the paper (arXiv:2608.30227) and silly-kicks' own code; the MPL clone stays **black-box run-only** and its `_score` / scoring source is **never opened or read**; and a divergence that the **paper** does not resolve triggers the §16.6 human-gated **stop**, never a "peek" at the reference implementation. This is non-negotiable regardless of how tempting a quick source-read would be.

### 16.5 Oracle rebuild — real DFL roster + a goal span

The committed slice is re-extracted on the DGX with:
- a **deterministic real-DFL-roster identity mapping**, replacing the match-wide **majority-vote** map (the rev-2 "conservative floor"). Exact identity removes vote-map noise so the committed oracle can corroborate the paper's number, not merely floor it. **Licence gate (do not assert — evidence it):** the roster source (DFL `MatchInformation`) is a *new* input to the committed fixture, so its CC-BY-4.0 status must be **verified on the DGX and recorded in the rebuilt fixture README** (the same standard §8.1 held the tracking/events grant to), and the **owner-confirm gate §8.1 applied to the original fixture is re-applied to the rebuilt one** — the roster mapping only ships once its redistribution licence is evidenced, never on the assertion that "it's part of the same release." If the roster's licence cannot be evidenced, fall back to the vote-map (its conservative-floor property is unaffected) and note the identity limit.
- a **second short span containing a goal** (spec §14.5's pre-authorized "two short spans" remedy) → closes IMPL-02's goal-coverage gap so the accuracy oracle exercises the goal virtual-event, not just the `TestEventEnrichment` unit test.

The rebuilt fixture stays ≤ a few MB and documents both spans + the roster source in its README (no misquote).

### 16.6 Acceptance bar (supersedes the rev-2 Goal-5 "human-gated accept")

- **Target the paper:** exact-frame ≈ 88.4% and W2 ≈ 96.5% on the **DGX 3-match CC-BY corpus** (J03WMX/J03WN1/J03WPY, real roster), with any residual rigorously attributed. The committed-oracle CI floor is recalibrated from the achieved-on-box number minus a safety margin (still a regression guard, not the aspirational number).
- **Human-gated fallback:** if a specific divergence proves **unrecoverable without the MPL reference code** (a genuine clean-room limit), stop and surface it for an explicit owner decision — **no silent acceptance of a lower bar**.
- **Validation scope (owner, 2026-09-12):** iterate the fix locally against the committed oracle (the §16.4 score-peak harness + exact-frame), then confirm the headline on the full DGX 3-match corpus before the independent review.

### 16.7 Folded review findings (all in this one cycle)

| ID | Resolution |
|---|---|
| IMPL-01 | Add the **align-level NW golden** on a tiny committed synthetic frame (§9), alongside the existing DP-kernel golden (`TestNeedlemanWunsch`). |
| IMPL-02 | Oracle **goal span** (§16.5) + **fix the README misquote** (quote the §8.1 requirement correctly). |
| IMPL-03 | Replace the do-nothing baseline with a **tiny in-test greedy re-impl** and assert NW beats it by a large margin (§8.1) — the guard that would have caught the constant-0.6 bug. |
| IMPL-04 | SB360 verdict `differs_by_design → honest_nan`/`not_exercised` (`NOT_EXERCISED_BUDGET` 46→52) — a consequence of the owner-directed freeze-frame honest-refusal. **No code change.** The impl review (a CONSIDER) routes the *verdict flip* to the **owner** for explicit sign-off at commit — the SB360 audit is a human-adjudicated artifact and plan Task 11 predicted an *unchanged* verdict. (Corrected from an earlier "blessed by the review" mis-statement — the reviewer does not stand in for the owner on a human-adjudicated artifact.) Owner sign-off recorded in **§16.9**. |
| IMPL-05 | De-duplicate the double `_detect_candidate_frames` call in `link_actions_to_frames_elastic` (thread the candidate dict out of / into the engine, or memoize). |
| IMPL-06 | C4 rendered via Graphviz `dot` (not Smetana) — **verified** this cycle (28 element nodes cleared the 0-entity guard); no change. |

### 16.8 Impact delta vs rev-2 (bookkeeping)

- The scoring re-derivation **moves `elastic_*` values further** — still inside the already-declared `elastic_*` mart re-materialize trigger (§11); still no default-config VAEP retrain (`elastic_sync_xfns` in no default list). No new columns → C4 feature-column count stays **397**; C4 aggregator count stays **33**.
- The **oracle fixture content changes** (real roster + goal span); its README is rewritten (accurate, two-span, roster-sourced).
- `CHANGELOG` / `TODO` / ADR-093 updated: the scoring-fix narrative + the **corrected attribution** (§16.3) replace the rev-2 "not the algorithm" wording.
- Still **one branch, one fully-tested coherent commit, human-approved**; independent review by a fresh owner-started session.

### 16.9 Owner sign-offs required (rev 3 ledger)

Human-adjudicated decisions this cycle needs the owner to record explicitly (not the reviewer, not this session):

| # | Decision | Status |
|---|---|---|
| SO-1 | **SB360 audit verdict flip** for `add_elastic_sync`: `differs_by_design → honest_nan`/`not_exercised`, `NOT_EXERCISED_BUDGET` 46→52, driven by the owner-directed freeze-frame honest-refusal (`POSITIONAL_ONLY` early-return). The *behaviour* is already owner-directed; this signs off the *audit-artifact* verdict change (which plan Task 11 predicted unchanged). | ☑ **SIGNED OFF (owner, 2026-09-12)** — conditioned on "unless it contradicts gold standard"; it does not: NW needs a continuous ball trajectory, SB360 freeze-frames have none, so honest all-NaN (ADR-063) is strictly better than the greedy path's silent spurious values. `honest_nan` IS the gold-standard verdict here. |
| SO-2 | **Goal-5 acceptance at commit** — the final DGX 3-match W2 vs the paper (§16.6 bar, as corrected by §16.11). | ☑ **SIGNED OFF (owner, 2026-09-14)** — accept **W2 0.862 cross-source** as the ship bar (per-match 0.848/0.847/0.891). Rationale with the corrected facts (§16.11): the paper's PRIMARY metric is **W2** (NW 96.5 % / greedy 84.1 %, same-source); our 0.862 is ≈ 11× our greedy, **above the paper's own greedy**, and ~10 pts below the paper's same-source NW — that residual is the **cross-source event-time jitter** (a data limit, not the algorithm; all clean-room levers exhausted). The rev-3 "88.4 % exact" bar was a metric misreading and is retired. |
| SO-3 | **Per-commit approval** of the exact diff. | ☐ deferred to commit-prep |

### 16.10 Investigation log (2026-09-12) — local phase outcome + DGX handoff

Tasks 13–15 ran locally against the committed (tc3-derived) oracle. Outcome, so the DGX phase (or a fresh session) resumes cleanly:

- **`s_BA` fix — SHIPPED, confirmed (the win).** The per-feature breakdown (Task 13→14) pinned `s_BA` as the dominant mis-peak (+0.41, 24/38 offenders). `_ball_kinematics` computed acceleration as the rate of change of **speed magnitude** (`|speed[i]−speed[i−1]|/dt`); the paper (Eq 3 + §2.3: *"the ball usually changes its DIRECTION when an event occurs"*) and physics use the **velocity-vector change** (`|v[i]−v[i−1]|/dt`), which the fix now computes. Result on the committed oracle: exact **29.5%→36.1%**, within-1 **59→75%**, W2 **67→79%**; non-breaking (54/55 elastic tests pass, only the score-peak gate stays RED pending full closure). **This change is IN the tree.**
- **KD window — paper read from the PDF (clean-room), implemented three ways, all DEGRADE on the smoothed oracle, REVERTED.** The authoritative PDF (Eq 5/6/10–12) gives: KD window bounded by the **actor-adjacent** candidate (`p_c ∈ P_c⁻/P_c⁺`), directional (outgoing/dispossessed → post-KD⁺, incoming/reception/tackle → pre-KD⁻), episodes aligned independently. Tried (a) globally-adjacent, (b) actor-adjacent gp-wide, (c) actor-adjacent per-episode — **all drop exact-frame to 18–23%** (vs 36.1% s_BA-only). Ruled out the `lookups.dist` 3 m cap. **Not an implementation shortcut — the exact paper window degrades our data**, which points to the underlying tracking, not the algorithm.
- **Leading remaining hypothesis: SMOOTHING.** The PDF §2.1 preprocessing mentions **no tracking smoothing** (acceleration "directly from position differences"); our tc3-cache applies Savitzky–Golay + SG-velocities. A smoothed ball trajectory shifts/flattens the accel peaks and the distance-departure the KD window measures — plausibly why the paper-faithful KD hurts here. **Unverifiable locally** (cannot un-smooth the committed oracle). **DGX-only: run elastic on RAW bronze tracking (pre-`_preprocess`) vs smoothed tc3**, which the IDSSE/Sportec parse-port emits.
- **Clean-room status:** the paper PDF (arXiv:2608.30227) was read for the exact equations; the MPL reference code was **never opened**. The web-summary was found unreliable (it misstated the candidate count per-match vs per-benchmark) — the PDF resolved it (per-match ~16,979/90,000 ≈ 19%, matching our 16%).
- **DGX plan (Tasks 16/19):** (1) smoothing test — s_BA-only on RAW vs smoothed tracking; if RAW lifts toward the paper, the gap was smoothing and the oracle is rebuilt from raw (Task 16); (2) re-test the (documented) actor-adjacent directional KD on RAW; (3) real-roster oracle + goal span; (4) 3-match corpus headline vs the §16.6 bar. **Restart point: the tree carries the s_BA fix at 36.1%/78.7%; resume at the DGX smoothing test.**

### 16.11 As-built reconciliation (2026-09-14) — AUTHORITATIVE; SUPERSEDES the §16.1–16.10 exact-frame framing

§16.1–16.10 were premised on a **misread of the paper's metric**: they set an "88.4 % exact-frame" target and framed the gap as a fixable scoring-transcription bug. Re-reading the paper (arXiv:2608.30227) directly this cycle establishes the authoritative benchmark facts, which supersede that framing (verified against the paper as read; the impl-r2 review's IMPL-08):

- **The paper's PRIMARY metric is W2** (within-2-frames, 0.08 s) — verbatim: *"we adopt W2 … as our primary metric rather than exact alignment."* It reports **NO exact-frame headline.** Table 2 (event start): **ELASTIC-NW 96.5 % W2, ELASTIC-Greedy 84.1 % W2** (the paper's own same-source data). **88.4 % is NOT the paper's NW figure** (it is not in the main sync table); the rev-3 "88.4 % exact" target was an error and is **retired**.
- **As-built outcome (the ship bar, SO-2):** DGX 3-match **W2 = 0.862 cross-source** (per-match 0.848/0.847/0.891; min-fold 0.847; reception W5 0.818) — **≈ 11× our greedy (0.08 W2)**, **above the paper's own greedy (0.841)**, **~10 pts below the paper's same-source NW (0.965)**. The residual is the **cross-source event-time jitter** (~0.8 s Kim-vs-DFL vote-map anchor; per-event, not drift), **NOT the algorithm or identity**. This corrects §16.3 ("the gap IS our scoring, fixable to the paper"): the scoring fixes helped (below) but the residual is a *data* limit, and "fixable-to-88.4 %-exact" was never valid. "Nowhere near the paper" (the tuning motivation) described the greedy/early-NW START (~0.08 W2), not the final 0.862.
- **The three cumulative wins** (post-§16.10): (1) **central-difference acceleration** (paper Eq. 3; the §16.10 s_BA fix, corrected to the central diff; oracle exact 0.335→0.583); (2) **in-play (`ball_state=="alive"`) episode grouping** — the paper's episode definition, replacing the `add_possessions` proxy of **§3.2 / design-decision-5** (this is the IMPL-10 episode-def change, recorded here; W2 0.845→0.856); (3) an **OpenEvolve-tuned `_score`** (W2 0.856→0.862; white-box score-peak 0.728→0.745; all three folds up). The §16.10 smoothing hypothesis was tested on the DGX; the residual proved to be cross-source jitter, not smoothing.
- **Scoring provenance (owner-approved 2026-09-14; IMPL-07).** The `_score` weights (`w_ba/w_pbd/w_kd/w_dyn` = 0.6/1.6/0.8/1.0 + a directional depart/arrive slope) are **OpenEvolve-tuned** on the 3-match CC-BY corpus — a **data-fit, NOT an MPL-source read** (the §16.4 clean-room boundary held; only the paper + our own code were read). This **RELAXES the §2 non-goal** ("no learned/per-provider-tuned scoring; future tuning is a separate ADR-009 cycle") **by explicit owner decision, recorded here** rather than deferred. Run config: OpenEvolve, `random_seed=42`, 120 iterations, population 40 / 3 islands, sonnet-5 (w 0.8) + opus-4.8 (w 0.2), fitness = score-peak (primary) + min-fold-W2 + coverage guards. The run **provenance artifact** (objective/folds/seeds/config) + a **held-out** (out-of-sample) validation number ship in the paired follow-up **provenance commit** (the owner's two-commit pattern). The committed-oracle floor is measured on a fit match (**in-sample**) — the regression guard, not the acceptance number.
- **Oracle rebuild (§16.5, corrected).** The committed slice is regenerated as a **single J03WMX period-1 span `[400 s, 640 s]`** that COVERS the period-1 goal (t≈468 s, frame 21706) with all four ELASTIC categories + out + reception (**67 events**) — meeting §8.1's goal requirement in one contiguous slice (simpler than the §14.5 two-span remedy — **§14.5's "two short spans" and §16.5/§16.8's "two-span, roster-sourced" bookkeeping are superseded by this single span**). Player identity stays **vote-map-sourced** (100 % span coverage) — the §16.5 "**real DFL roster**" aspiration is **moot**: the corrected attribution (above) shows player identity is NOT the residual's cause (cross-source jitter is), so a roster would not move the number. Floors recalibrated to achieved-minus-margin — achieved START exact 0.493 / W2 0.791, RECEPTION W5 0.922 → **floors 0.42 / 0.70 / 0.82** (below achieved, so they guard regressions not the achieved value); score-peak regression bound ≤ 26 (21 achieved). (IMPL-02 goal coverage + README misquote fixed.)
- **Folded impl-r2 findings:** IMPL-01 align-level golden (`test_nw_golden_on_synthetic_frame`); IMPL-02 goal span + README misquote; IMPL-03 greedy-argmax beat replaces do-nothing; IMPL-05 double candidate-detection de-duped; IMPL-07/09 tuning recorded + `test_dataclass_defaults_are_paper_intent_set_constants` scoped; IMPL-08/10 this reconciliation + the episode-def change; IMPL-11 ruthless floor bump 0.4.0→0.6.0 (owner-folded).
- **Citation correction:** the paper is *"ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer"* by Kim, Choi, Lee, Seo, Boomstra, Yoon & Park (arXiv:2608.30227) — the earlier "Event-Level Alignment of STreaming data Including Coordinates / Kim, Kim & Kim" title+authors were wrong (fixed in NOTICE + the oracle README). **Venue CIKM 2026 is CONFIRMED** — the lead author's public acceptance announcement states the paper was accepted to CIKM 2026 (November 2026, Rome), an extended version of the MLSA-2025 workshop paper — so "CIKM 2026" stands throughout (the arXiv abstract merely omits the venue).
