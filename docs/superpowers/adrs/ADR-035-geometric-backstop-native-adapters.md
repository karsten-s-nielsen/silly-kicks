# ADR-035: Geometric frame-LTR backstop on the native tracking adapters

| Field | Value |
|---|---|
| **Date** | 2026-06-19 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (TF-23b; cross-session spec+plan review) |
| **References** | ADR-034 (parent, TF-23), ADR-031 (Gate D), ADR-029 (supersedes its no-refactor fence for the native adapters), ADR-019 (id-dtype), ADR-010 (ET guard) |

## Context

The native tracking adapters `silly_kicks.tracking.{gradientsports,sportec}.convert_to_frames`
orient each period from a **caller-supplied flag** via `direction.home_attacks_right_per_period`.
The per-period flip is only as correct as that flag. The live GS-ET incident (2026-06-13) was a
consumer passing a wrong `home_team_start_left_extratime` placeholder → periods 3/4 reversed →
every geometry-derived ET tracking feature mis-oriented.

ADR-034 (TF-23, 4.33.0) shipped the flag-free, idempotent geometric orienter
`tracking.orient_frames_to_ltr_by_geometry` (per-period home-GK-median-x anchor → point-reflect any
period whose home GK sits on the attacking half) and explicitly parked "the geometric net on the GS
native adapter" as **TF-23b**, the gate that lets the luxury-lakehouse retire its own
`correct_frames_to_home_ltr` backstop entirely. ADR-031 separately lists "IDSSE-ET handedness
(Gate D)" as open; the sportec adapter shares the identical flag-flip pattern, so the same backstop
closes Gate D.

## Decision

The native tracking adapters apply `orient_frames_to_ltr_by_geometry` as an always-on **idempotent
backstop** after the flag-flip, via a shared `direction.finalize_orientation(...)`. **Backstop, not
replacement** (the flag-flip stays; the net only self-corrects). Scope: **GS + sportec**.

- **Shared `direction.finalize_orientation` (supersedes the ADR-029 no-refactor fence — native
  adapters ONLY).** ADR-029:55-56 fenced "sportec/GS adapters **and kloppy gateway** are NOT
  refactored through the helper" on a "zero gain (primitives already shared)" rationale. That premise
  is now dead for the native adapters: TF-23b would otherwise add a *third* inline copy of the
  orientation tail (ET guard → per-period flag flip → period-gated `team_attacking_direction` label →
  backstop), and the 2026-06-09 dtype-safe `is_home` fix already had to be applied twice in lockstep.
  `finalize_orientation` makes it one function with one insertion point; the refactor is golden-safe
  (the backstop is a byte-identical no-op on the correct-flag RT goldens; the flip+label ops are
  unchanged). The supersession is **scoped to the two native adapters**; the **kloppy gateway stays
  intentionally un-routed** (it is already oriented), so a future reader does not re-open that closed
  decision.
- **Policy injection, not a re-implemented guard.** The net `raises` on a zero-home-match by default
  (ADR-019: mis-orienting is worse than failing). It gains
  `on_missing_home: Literal["raise","warn"] = "raise"`; the adapters pass `"warn"` so the net emits
  the warning and returns the frame un-oriented (the flag-flip result stands) — preserving the
  adapters' established warn-don't-raise contract with one definition of "can't anchor". Default
  `"raise"` keeps every direct/lakehouse caller byte-identical.
- **Additive `copy: bool = True` knob.** `finalize_orientation` already owns a copy-at-entry (clean
  value semantics, input never mutated), so it passes `copy=False` to the net — a `convert_to_frames`
  call does **two** full-frame copies, not three, on tracking-scale data (~1M+ rows/match). Default
  `True` preserves the input-never-mutated contract for direct callers. Both `on_missing_home` and
  `copy` are additive, default-preserving public-net API additions (recorded in the CHANGELOG).
- **The net never orients PSO (period 5) — public-net behavior change.** Orienting a penalty-shootout
  frame by GK geometry is meaningless (both teams attack one end), so the net's flip loop is
  restricted to `_LTR_KNOWN_PERIODS=(1,2,3,4)` — for **all** callers, including the TF-23
  SkillCorner/Metrica builders that call the net. We do not preserve the garbage flip behind a
  default. PSO frames are excluded from all geometric feature analysis, so practical impact is nil.

## Consequences

- Closes ADR-031 **Gate D** (IDSSE-ET handedness) — gated on the real-IDSSE-ET self-correction
  validation (G3, below). Realizes ADR-034's TF-23b gate, so the lakehouse can now retire
  `correct_frames_to_home_ltr` **entirely** for all four tracking providers (metrica/skillcorner via
  the TF-23 builders; idsse/GS via this backstop). During the migration window the lakehouse may run
  both nets; double-application is safe because **both are idempotent**, so it can retire its backstop
  lazily after adopting ≥4.34.0, not atomically. (Its re-materialize trigger, not silly-kicks'.)
- C4-free (no new action-coupled aggregator; count stays 28). No atomic mirror (converter, not
  aggregator). No default-xfn change. No NOTICE change (reuses TF-23's ADR-053 promotion credit).
- **VAEP/tracking retrain trigger** for the matches whose ET flag was wrong — see the enumerated
  scope below.
- **Anchor-context asymmetry.** The native adapters convert the **full match**, so the median anchor
  runs over a whole period (~tens of thousands of frames) and a sweeper-keeper/corner transient
  cannot move the median out of the GK's half — no per-batch guard is needed here. The opposite of the
  luxury-lakehouse TF-23 path, which feeds the builders ~250-frame windows where a transient *can*
  flip the deepest player and therefore needs a cross-batch GK/orientation-consistency guard. Same
  anchor, different consumption pattern → guard downstream, not here.
- **Chesterton's Fence (period 5).** `git log -S` confirms the all-period flip loop was introduced in
  `cf7b29b` (TF-23, 4.33.0) with no period-5 test — an unscoped side effect of a PSO-free-providers
  PR, not an intentional fence. Recorded so it is not re-added.

### Enumerated retrain scope

The **exact** changed-match set (the G1 non-no-op list, 2026-06-19 DGX run — authoritative):

- **`gradientsports` 10506** and **`gradientsports` 10517** — the only 2 of 71 corpus matches whose
  4.34.0 output differs from 4.33.0. Both differ in **periods 3–4 (ET) ONLY** (reflection-consistent
  deltas; |Δ| exceeds the pitch bounds only on out-of-play ball rows under the point reflection).
  These are the GS WC2022 ET-tracking matches whose constant `homeTeamStartLeftExtraTime` placeholder
  was geometrically wrong; the backstop self-corrects them. (Of the 3 GS ET-tracking matches, the 3rd
  had a correct flag → no-op.) **Downstream consumers + the lakehouse re-materialize the ET frames of
  these 2 GS matches only.** Estimate ≤3 confirmed (events=5/tracking=3 distinction held).

### Period-5 (PSO) match list

**None.** The 2026-06-19 G1 run found **0** matches with period-5 frames across GS×64 + IDSSE×7. The
net's period-5 flip-loop restriction (§ never-orient-PSO) is a forward-looking safety net that did not
fire on the current corpus; it remains correct for any future shootout-tracking feed and for the
lakehouse SC/Metrica builders (which self-assess PSO presence).

### G1 / G2 / G3 empirical validation (hard ship gate — PASSED 2026-06-19)

DGX run (clean origin/main 4.33.0 baseline vs the 4.34.0 working tree, same pining inputs, per-match
`assert_frame_equal(check_dtype=True)`):

- **G1 — PASS.** 71 matches compared (GS×64 + IDSSE×7): **69 byte-identical**, **2 changed**
  (`gradientsports` 10506 + 10517, ET-only — the intended self-corrections above). No correct-flag
  match is a non-no-op → the no-retrain claim holds for 69/71; the 2 changed are the enumerated scope.
- **G2 — PASS.** GS self-correction on **real native GK** validated on match 10517 (home GK raw
  x≈90 in P3 under the placeholder flag → corrected to x≈15 by the backstop); committed as a permanent
  CI gate via `test_real_et_roundtrip.py::test_gs_real_et_native_gk_geometric_self_correction` on the
  regenerated native-GK fixture.
- **G3 — re-scoped (no real IDSSE-ET data).** All 7 IDSSE pining matches are regulation-only
  (periods [1,2]; 0 ET), so real IDSSE-ET self-correction cannot be demonstrated on this corpus. Gate D
  (IDSSE-ET handedness) closure therefore rests on: (a) the **synthetic** sportec ET self-correction
  gates (`test_adapter_extra_time_orientation.py` [sportec] + `test_et_guard_parity.py`
  tracking-sportec — committed, all CI legs), and (b) the **provider-agnostic** mechanism: the sportec
  adapter shares the identical `finalize_orientation` flip+backstop path proven on GS real data (G1/G2).
  No real-IDSSE-ET claim is made beyond this; revisit if cup/ET IDSSE tracking is later ingested.
