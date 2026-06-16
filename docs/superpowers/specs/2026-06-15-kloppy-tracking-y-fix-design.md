# Spec — Fix the kloppy-tracking-y inversion (ADR-031)

**Status:** REVIEWED (lakehouse rev 2 + rev 3 + rev 4) + Gate-D measured per-period on the DGX (2026-06-16)
**Date:** 2026-06-15 (updated 2026-06-16 with the Gate-D verdict + rev-3)
**Author:** main session (Karsten / Claude)
**Delivery:** ONE spec, sequenced PRs — **PR-S94 (T1)** → **PR-S95 (T3 parse port)**; **T2 is a measured
no-op** (Gate D = native clean), documented in ADR-031 (optional thin PR-S96, see §3)
**Base:** `main` @ `7734ed4` (v4.28.0, clean)
**Decision record:** ADR-031 (new; silly-kicks numbering)
**Retrain trigger:** scoped per PR — **PR-S94 fixes the CALIBRATION/pining path + external kloppy-gateway
consumers, NOT lakehouse production** (Gate C, rev-4: the lakehouse builds SkillCorner/Metrica via its
OWN `convert.py` builders, not the silly-kicks gateway). So PR-S94 retrains the calibration
recommendations, not lakehouse VAEP/tracking. PR-S95: IDSSE calibration consumers + lakehouse IDSSE
re-materialize (after port adoption). **T2/native: NO retrain (Gate D clean).** Lakehouse-prod
SkillCorner/Metrica y-correctness is a SEPARATE lakehouse fix (Gate-C handoff, §5).

---

## 1. Problem

kloppy-derived tracking frames carry a y-axis **inverted** relative to the SPADL action
y-axis: for the same physical point, `action_y == 68 − frame_y`. Confirmed for **SkillCorner**
and **IDSSE** (kloppy path), and present **by the same code path** for **Metrica** (untested, see Gate A).

- It is a **single-axis y mirror**, NOT an orientation problem. ADR-028 (per-action LTR
  reprojection) and ADR-029 (`orient_frames_to_ltr`) and `play_left_to_right` are ALL 180°
  point reflections or identity — a single-axis (y-only) flip is **outside** the orientation
  family, so re-orientation is the wrong lever and does not rescue it.
- It is NOT a clock/linkage problem: action↔shooter time gaps at the measured shots are ~0.00–0.02 s.
- The error magnitude is `|68 − 2y|`: **zero at y = 34** (pitch centre), growing to ~full pitch
  width at the touchlines. This is why it stayed hidden — the aggregator liveness gate is
  non-null-only, the GK-roster e2e is x-based, and the synthetic mirror-invariance fixtures are
  self-consistent (they assume a clean 180° relationship and cannot manufacture a y-only mirror).

**Severity:** HIGH for the kloppy tracking path (SkillCorner, Metrica, and the IDSSE *dev harness*).
**NOT** present on the native sportec path (Gate D, §4.3) — so **production IDSSE is correct**.

The authoritative bug report (committed on `main`) is
`docs/research/bug_kloppy_tracking_y_inverted.md`. This spec is the design to fix it.

---

## 2. Root cause (verified at the anchors, this tree)

The two kloppy gateways normalize coordinates **differently**:

- **Event gateway — canonical (correct).** `silly_kicks/spadl/kloppy.py:196-202` transforms with
  `to_orientation=Orientation.HOME_AWAY` **and**
  `to_coordinate_system=_SoccerActionCoordinateSystem(...)`. That coordinate system
  (`spadl/kloppy.py:293-326`) pins `origin=Origin.BOTTOM_LEFT` and
  `vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP`, so kloppy applies each provider's
  native→canonical vertical flip. Events come out in the SPADL convention.

- **Tracking gateway — NOT pinned (the bug).** `silly_kicks/tracking/kloppy.py:104-113`
  transforms with `to_pitch_dimensions=MetricPitchDimensions(0..105, 0..68, standardized=False)`
  **and** `to_orientation=Orientation.HOME_AWAY`, but **never** `to_coordinate_system`.
  `to_pitch_dimensions` only rescales axes; it does **not** normalize vertical orientation. So the
  frames retain each provider's kloppy-native vertical → y inverted vs the canonical events.

- **Dev loader — same omission.** `scripts/_loader_pining.py::_kloppy_tracking_to_frames`
  (`:318-359`, used for IDSSE in the harness) makes the identical un-pinned `transform()` call.

`grep` confirms `VerticalOrientation` / `to_coordinate_system` / `BOTTOM_TO_TOP` / `Origin` appear
in `spadl/kloppy.py` but **nowhere** in `silly_kicks/tracking/`.

**Gate-D corollary (2026-06-16):** the **native** sportec adapter (`tracking/sportec.py`,
`y = y_centered + 34` on the parser's *raw* DFL Y) produces the **canonical** y — kloppy's *tracking*
transform is the lone inverter, not the raw DFL data. So the native path was right all along (§4.3).

**Chesterton's Fence:** the un-pinned tracking transform originated in **PR-S19 (v2.7.0, #24)** —
the `silly_kicks.tracking` namespace's first commit — not PR-S26/3.3.0 as an earlier note said.
The CS-pin was added to the event gateway but never mirrored to the tracking gateway: an
**asymmetry / oversight**, not a deliberate guard. (Plan re-verifies via `git log -p` on the
introducing commit.)

---

## 3. Scope — one spec, sequenced PRs (owner-decided 2026-06-15; Gate-D-adjusted 2026-06-16)

Nothing is deferred; the work is **sequenced** into independently-revertible PRs so a confirmed prod
fix is not bundled behind an untested hypothesis or a cross-repo refactor, and each retrain trigger is
scoped to its own change (lakehouse rev-2/rev-3 review, owner adopted).

**Single-source on the native path — at the PARSER layer, not just the converter.** The rev-2/rev-3
review established (against the lakehouse repo) that **production IDSSE never touches kloppy**: it
parses DFL with a hand-rolled, stdlib-`xml.etree`, kloppy-free parser (`src/ingestion/idsse.py`) and
Savitzky–Golay-smooths at ingest. So single-sourcing only the *converter* would leave the *parser +
smoothing + velocity* divergent — the skew moves up a layer (§4.4 four-layer table). The drift-free
design upstreams the lakehouse's pure parse functions into silly-kicks as a reusable **parse port**,
consumed by both the dev harness and the lakehouse.

**Gate D verdict (DGX, 2026-06-16): the native sportec adapter is y-CORRECT, confirmed PER PERIOD.**
Native↔kloppy comparison over ~6M player-frames on 2 IDSSE matches, broken out by period (C2): at the
correctly-oriented period (`|nx−kx| med=max=0.00`; the correct `home_team_start_left` alternates P1/P2
as teams switch ends), `native_y == 68 − kloppy_y` exactly (`|ny−(68−ky)| med=max=0.00`) in **both P1
and P2**; the harness check gives `kloppy_y == 68 − canonical_y`, so `native_y == canonical_y`.
**Production IDSSE is not broken; T2 has no fix to ship and no retrain.** ET (P3/P4) unverified (no ET
in the IDSSE set). (Repro: `scripts/_tf48_gate_d.py`.) This **dissolves B1** — the native-IDSSE golden
is green in PR-S95 with no prior fix.

| PR | Thread | Status | Content |
|----|--------|--------|---------|
| **PR-S94** | **T1** — kloppy-tracking-y inversion in the **gateway** (SkillCorner + Metrica) | Confirmed (2 sessions) | Shared `_kloppy_coordinates` extraction + CS-pin the gateway + the silly-kicks-side cross-provider y-identity golden (§7) + gateway parity contract test. **Fixes the calibration/pining path + external gateway consumers — NOT lakehouse prod** (Gate C). Confirmed high-severity for those consumers; ships first, revertibly. |
| **PR-S95** | **T3** — single-source the IDSSE/Sportec **parser** via a reusable **parse+shape port** | Confirmed drift (parse/smooth/velocity) | New `silly_kicks/providers/sportec/parse.py` behind a `[parse-dfl]` extra, seeded by upstreaming the lakehouse's pure DFL parse fns (typed returns); re-route `_loader_pining.py` IDSSE to port-parse + native-convert; retire `_kloppy_tracking_to_frames` + the kloppy IDSSE events path; **native-IDSSE y-identity golden (green — Gate D)**; numeric single-source gate (§4.5); lakehouse adopts the port + deletes its copy (cross-repo). |
| **T2 / (opt) PR-S96** | native Sportec y-handedness | **Gate D = CLEAN (done)** | No code fix, no retrain. Documented in ADR-031; the native-IDSSE golden in PR-S95 is its durable guard. *Optionally* a thin standalone PR-S96 carrying just a native-sportec-handedness invariant test if you want it isolated — otherwise absorbed into PR-S95. |

**Separately tracked (NOT folded into T3):** the IDSSE **ball-at-wrong-end** smells like a distinct
ball-data bug; re-checked on the native path (PR-S95 Gate T3b), and either fixed or
documented-with-evidence — never allowed to silently widen T3.

**Truly out of scope** (clean / event-only — confirmed by report + code): Gradient Sports native
tracking adapter; StatsBomb / Wyscout / Opta (event-only, no tracking).

---

## 4. Fix design

### 4.1 Shared coordinate-system extraction (PR-S94)

Extract `_SoccerActionCoordinateSystem` out of `spadl/kloppy.py` into a **neutral shared module**
`silly_kicks/spadl/_kloppy_coordinates.py`; both gateways import it from there.

- Rationale: DRY (events and frames must not drift); `tracking/` already imports from `spadl/` in
  13 places (`spadl.config`, `spadl.utils`), so there is **no** parallel-namespace violation; the
  new module isolates the kloppy-domain coordinate dependency in one place.
- The module imports the kloppy-domain symbols the class needs (`CoordinateSystem`, `Origin`,
  `VerticalOrientation`, `MetricPitchDimensions`, `Dimension`, `PitchDimensions`, `Provider`) plus
  `spadl.config` for `field_length` / `field_width`.
- `spadl/kloppy.py` keeps a re-export so the event path is **byte-equivalent** — guarded by the
  WC2018 golden (§7.4).

A thin helper may live alongside the class:

```python
def socceraction_coordinate_system(metadata) -> _SoccerActionCoordinateSystem:
    cs = metadata.coordinate_system
    return _SoccerActionCoordinateSystem(pitch_length=cs.pitch_length, pitch_width=cs.pitch_width)
```

(Final shape decided in the plan; the requirement is single-source + byte-equivalent event path.)

### 4.2 T1 — CS-pin the tracking gateway (SkillCorner + Metrica only)

The dev loader's `_kloppy_tracking_to_frames` retires under T3, so T1 touches only the shipped gateway
`tracking/kloppy.py` — which serves the kloppy-only providers SkillCorner + Metrica. Add
`to_coordinate_system=` to the existing `dataset.transform(...)` call (`:104-113`):

```python
transformed = dataset.transform(
    to_pitch_dimensions=MetricPitchDimensions(0..105, 0..68, standardized=False, 105, 68),
    to_orientation=Orientation.HOME_AWAY,
    to_coordinate_system=socceraction_coordinate_system(dataset.metadata),  # NEW
)
```

- kloppy then applies each provider's native→canonical vertical flip **correctly**, and is a
  **no-op for an already-canonical provider** — exactly why a blanket `frames["y"]=68−y` is WRONG
  (it would double-invert a clean provider; Metrica is untested and must not be blanket-flipped).
- The proven equivalence (max dev 0.000 over 8800 SkillCorner pairs: output == `(x, 68−y)` of current
  frames; x unchanged) shows the CS-pin preserves the 0..105 / 0..68 scaling and only corrects the
  vertical. It also corrects `derive_velocities` vy and composes with `play_left_to_right`.

**Gate 0 RESULT (DGX, 2026-06-16, SkillCorner ~220k player-frames):** the correct signature is
**CS-ONLY** — `to_orientation=HOME_AWAY` + `to_coordinate_system=...`, **dropping `to_pitch_dimensions`**
(candidate B): `max|x−cur.x|=0` and `max|y−(68−cur.y)|=0` exactly. Candidate A (keep
`to_pitch_dimensions` AND add the CS) is a **silent non-fix** — `to_pitch_dimensions` overrides the CS's
vertical, leaving `y` inverted (`max|y−cur.y|=0`). So Task 3.1 removes `to_pitch_dimensions` and relies
on the CS's own `standardized` 0–105/0–68 dimensions — **the exact form the event gateway already
uses** (`spadl/kloppy.py:194-197`); x is byte-identical. Repro: `scripts/_tf48_cspin_equiv.py`.

### 4.3 T2 — native sportec DFL path: Gate D MEASURED → CLEAN (no fix)

`tracking/sportec.py` is a **native** adapter (no kloppy): `:131-132` does `x = x_centered + 52.5`,
`y = y_centered + 34.0` on the parser's *raw* DFL X/Y, with a per-period direction flip (`:144-146`).

**Gate D (DGX, 2026-06-16) — measured, faithful to production** (`scripts/_tf48_gate_d.py`): parsed
real IDSSE DFL with the lakehouse parser's exact logic (raw DFL X/Y, NO flip — confirmed by reading
`_parse_positions_xml`: `px=float(X); py=float(Y)`), built the native-input shape, ran the real
`tracking.sportec.convert_to_frames`, and compared **in bulk** (~5.2M player-frames, 2 matches) to the
correctly-oriented (kloppy) reference and the canonical CS-pinned events:

```
(A) action ↔ kloppy-frame shooter, HOME off-centre: dx≈0.3–2.3, d_yflip≈1–6 ≪ d_identity≈27–32
    → kloppy frames correctly ORIENTED but y-INVERTED vs canonical (the reference). Harness valid.
(B) native ↔ kloppy, PER PERIOD (C2 — full match, median AND max). The correct orientation alternates
    by period (P1: hsl=False, P2: hsl=True — teams switch ends at half). At the correctly-oriented
    period in each run (|nx−kx| med=max=0.00):
       P1 and P2 BOTH:  |ny−(68−ky)| med=0.00 / MAX=0.00   (and |ny−ky| ≈ 18–23, i.e. NOT equal)
    → native_y == 68 − kloppy_y == canonical_y, exactly, in every covered period. No masked
      per-period sign error (the bulk-median P1-only earlier result would have hidden one).
```

**Verdict: the native sportec adapter is y-correct, confirmed per period (P1 + P2, exact, max=0.00) on
2 IDSSE matches (IDSSE *is* DFL, so this validates the adapter for all Sportec/DFL data). No y-map
change, no retrain.** Refutes the prior "native shares kloppy's inversion" hypothesis — kloppy's
*tracking* transform is the inverter; the raw DFL Y the native adapter consumes is already SPADL-up.
**Production IDSSE events and tracking agree.** **Caveat (C2): ET (P3/P4) is unverified** — the parser
maps only `firstHalf`/`secondHalf` and both test matches are P1/P2-only (no ET in the IDSSE set); the
native ET per-period flip uses `home_team_start_left_extratime`, untested here. Recorded in ADR-031;
the durable guard is the native-IDSSE entry in the §7.1 golden (PR-S95).

### 4.4 T3 (PR-S95) — single-source the IDSSE/Sportec PARSER via a parse+shape port

**The drift is FOUR-layered** (rev-2/rev-3 review, verified against the lakehouse repo):

| Layer | Dev harness (`_loader_pining.py`) | Production (lakehouse `src/ingestion/idsse.py`) |
|-------|-----------------------------------|--------------------------------------------------|
| Parse | kloppy `sportec.load_event`/`load_tracking` | hand-rolled stdlib `xml.etree.ElementTree`, kloppy-free |
| Smooth | silly-kicks `smooth_frames` (`_preprocess`, `:253-257`) | lakehouse `analytics.smoothing.smooth_positions` (`idsse.py:992`) |
| Velocity | silly-kicks `derive_velocities` (`_preprocess`) | lakehouse `_derive_velocities_savgol` (`tracking_context.py:835`, "matches silly-kicks `_velocity.py:84-124`" *by comment*) |
| Convert | kloppy `spadl.kloppy` + `_kloppy_tracking_to_frames` | native `spadl.sportec` / `tracking.sportec.convert_to_*` |

A converter-only single-source (the rejected kloppy-shim) leaves parse + smooth + velocity divergent —
parity-by-comment is exactly the drift class this spec exists to kill. So the fix single-sources the
**parser** and folds smoothing+velocity into one explicit "shared canonical callable" decision (§4.5).

**Fix — a per-provider PARSE+SHAPE PORT, owned in silly-kicks, consumed by both** (hexagonal:
silly-kicks owns the DFL→native parse it already converts from; the lakehouse is a consumer).
Generalizable shape (`parse_<provider>_*`, pure, behind an optional extra, data-quality stays
consumer-side); IDSSE is the proof-of-concept because the lakehouse's parse fns are already pure.

1. **New module** `silly_kicks/providers/sportec/parse.py`, behind a new optional extra
   `silly-kicks[parse-dfl]` (keeps core install thin; the port is stdlib-`xml.etree`, Spark-free).
   **Typed returns** (output shape is an API under Hyrum's Law — dataclass/TypedDict, field renames
   are breaking):
   - `parse_dfl_match_info(info_xml) -> MatchInfo` (seeds `_parse_teams`, `_parse_match_metadata`,
     `derive_idsse_home_team_start_left` incl. the `extraTimeFirstHalf` ET variant)
   - `parse_dfl_events(events_xml, *, player_team_map, ...) -> list[EventRow]` (seeds
     `_parse_events_xml` / `_build_event_row`)
   - `parse_dfl_tracking(positions_xml, *, player_team_map, ...) -> TrackingFrames` (seeds
     `_parse_positions_xml`, two-pass ball-then-players), emitting the native-input shape
     `tracking.sportec.EXPECTED_INPUT_COLUMNS`.

   **Honest naming (S4): this is a parse+SHAPE port, not a pure parse port.** `parse_dfl_tracking`
   emits the downstream converter's input contract (`EXPECTED_INPUT_COLUMNS`), not a
   provider-faithful domain model — pragmatically right (avoids a mapping layer), but it means a
   future provider whose native converter wants a different shape reuses only the parse half. Called
   out so the "generalizable" claim isn't oversold (R7).

   **Data-quality is NOT a port parameter (S1 — composition, not a DI hook).** The port has **no**
   `post_process` callable. Consumers compose explicitly: **`velocity(smooth(parse_dfl_tracking(...)))`**
   — smooth-then-velocity (C1: the canonical order — smooth positions first, then differentiate;
   matches `_preprocess` `:253-257` and the lakehouse). Same drift-safety as a hook, simpler contract,
   no Hyrum surface on an optional callable, and tests pure functions instead of parse×hook
   combinations. silly-kicks exposes its canonical smoother/velocity as standalone callables so a
   consumer that wants single-sourced data-quality composes *those* (§4.5).
2. **Re-route `_loader_pining.py` IDSSE** (`:300-315`) to the port → native
   `spadl.sportec.convert_to_actions` + `tracking.sportec.convert_to_frames`. **No kloppy parser in
   the loop.** `home_team_start_left` from `parse_dfl_match_info` (DFL `<KickOff TeamLeft>`), NOT the
   `_loader_databricks.py:174` placeholder.
3. **Retire** `_kloppy_tracking_to_frames` (`:318-359`, single caller `:306`) + the kloppy IDSSE
   events path (`:304`) — National Park / Chesterton (caller confirmed unique).
4. **Lakehouse adoption (cross-repo, separate lakehouse PR — the structural win):** the lakehouse
   `idsse.py` imports the port and **deletes its private parser**. One DFL parser for dev *and* prod.
   Adoption checklist (S3) carried in the handoff: (a) lakehouse `pyproject.toml` requires
   `silly-kicks[parse-dfl]` (it is now a **hard** runtime dep, not optional, for the lakehouse);
   (b) the terraform serverless env blocks get the `==` pin mirroring `uv.lock` (lakehouse ADR-046 /
   `test_terraform_env_dep_parity.py`); (c) the PEP-723 trainer footgun (a `[parse-dfl]`-less
   resolution silently downgrades); (d) confirm void-strip / GS-dedup stay consumer-side.

This **dissolves the old "events-shim infeasible" risk**: `_parse_events_xml` is a working,
battle-tested DFL-event→native-shape parser — the upstream *is* the events answer.

**Ball-at-wrong-end** (separately tracked) is re-checked on the native path (Gate T3b); if it persists
natively it is a distinct ball-data bug — isolated and fixed or documented-with-evidence.

### 4.5 Smoothing + velocity — one shared canonical callable, gated (PR-S95)

Data-quality (Savitzky–Golay smoothing, **velocity derivation** (B3), void-strip, dedup) stays
**consumer-side** and **out of the port** (the port is bytes→typed-rows). Consumers compose it
explicitly (§4.4 S1). The two numeric derivations (smooth + velocity) are the residual drift after the
parser is single-sourced, and they are **y-independent** — the y-identity golden cannot see them — so
they get a **blocking gate, not a footnote (B4)**:

- **Acceptance criterion (PR-S95):** the dev harness and production compose the **same** smoothing +
  velocity callable, asserted by an **end-to-end** parity test (C1) —
  `convert(velocity(smooth(parse(slice)))) == production_frames` over a committed DFL slice — **not**
  per-layer, so a composition-order swap (velocity-before-smooth) is caught. **OR**, if a residual is
  accepted, it must (C3): (a) be **bounded below a stated threshold tied to a feature-impact argument**
  (e.g. < X m → < Y change in `nearest_defender_distance` / `pressure_on_actor`), AND (b) carry a
  **tracked TODO follow-up to converge** the smoother/velocity. An "accepted residual + dated decision"
  with no bound and no follow-up is permanent drift with a paper trail — not allowed.
- Cleanest target: silly-kicks exposes its canonical `PreprocessConfig`-driven smoother + velocity as
  ready-to-compose callables; production and the dev harness compose those. **Decision (lakehouse's):**
  which is the shared canonical callable (silly-kicks' or theirs ported in) + the SG window/poly to
  reconcile (the lakehouse `_derive_velocities_savgol` is parity-by-comment with `_velocity.py` today).

---

## 5. Validation gates (all on the DGX — canonical compute)

Gates 0/A/B → **PR-S94**; Gate C + T3a/T3b → **PR-S95**; **Gate D = DONE (clean, §4.3)**. Gate E
(cross-provider golden) spans PR-S94 (SC+Metrica) and PR-S95 (native-IDSSE).

- **Gate 0 — DONE (PR-S94 linchpin):** SkillCorner ~220k player-frames — candidate **B (CS-only)** gives
  `max|x−cur.x|=0` and `max|y−(68−cur.y)|=0`; candidate A (keep `to_pitch_dimensions`) leaves y inverted.
  Signature pinned to CS-only (§4.2). Repro `scripts/_tf48_cspin_equiv.py`.
- **Gate A — DONE (PR-S94, was a HARD pre-merge gate):** Metrica (kloppy open-data, ~145k frames,
  tracking A/B) — `max|new.y−(68−cur.y)|=0`, `max|new.y−cur.y|=74.8` → **Metrica was y-INVERTED (like
  SkillCorner); the CS-pin flips it to canonical.** N4 resolved: **values moved → the Metrica (calibration)
  retrain trigger FIRES** (not a no-op). Neither real kloppy provider is canonical, so the no-op guard
  uses canonicalized real data (Task 3.2). Repro `scripts/_tf48_gate_a_metrica.py`.
- **Gate B — DONE (PR-S94, SkillCorner re-verify):** post-fix full-match SC action↔shooter
  (n=497 off-centre HOME): `d_identity=0.16 m`, `d_yflip=41.1 m`, `dx=0.15 m` → **IDENTITY (fix correct
  on the full match).** SC events converter needs no flip (confirmed). Repro `scripts/_tf48_gate_ab.py`.
- **Gate C — RESOLVED (rev-4, against the lakehouse repo):** per the lakehouse AC dispatch
  (`src/analytics/action_context/pipeline.py:229 _convert_tracking_batch`): **IDSSE** → native
  `tracking.sportec.convert_to_frames` (Gate-D clean ✓); **GradientSports** → native
  `tracking.gradientsports` (out of scope ✓); **Metrica** → lakehouse-OWN `_bronze_metrica_to_frames`
  (`convert.py:279`); **SkillCorner** → lakehouse-OWN `_bronze_skillcorner_to_frames` (`convert.py:390`).
  **The silly-kicks `tracking/kloppy.py` gateway is used by NONE of the lakehouse prod paths** — only by
  the pining/calibration harness + external gateway consumers. So PR-S94 fixes calibration, not lakehouse
  prod. **RED FLAG (lakehouse-side, separate investigation):** the two local builders handle vertical
  *inconsistently* — Metrica flips (`y=(1−y01)*68`, `convert.py:335`), SkillCorner does not
  (`y=y_center+34`, `convert.py:427`) — and neither is y-guarded (the lakehouse orientation golden is
  x-based, structurally blind to a y-mirror; the `correct_frames_to_home_ltr` 180° net can't rescue a
  single-axis flip). **Gate-C handoff (to the lakehouse):** run the off-centre action↔shooter y-identity
  diagnostic on `_bronze_metrica_to_frames` + `_bronze_skillcorner_to_frames` output and add a lakehouse
  y-identity golden distinct from the orientation golden — the lakehouse's half of Gate E (currently a
  coverage hole). NOT asserted broken without that run (evidence discipline); but the asymmetry is the
  same smell the silly-kicks bug had.
- **Gate D — DONE (guard rides in PR-S95):** native sportec handedness measured **clean per period**
  (P1+P2, exact, max=0.00; §4.3). **ET (P3/P4) unverified** (no ET in the IDSSE test set) — flagged as a
  known gap; if an ET-bearing DFL match becomes available, re-run before relying on native ET frames.
  No fix, no retrain.
- **Gate T3a (PR-S95 — parse-port parity + numeric single-source):** (1) port `parse_dfl_*` output ==
  the lakehouse parser output by **semantic equality** — documented float tolerance + canonical
  column/row ordering, byte-exact only for genuinely exact fields (ids, `ball_state` enums) (S2);
  (2) the smoothing+velocity single-source assertion is **end-to-end** (C1):
  `convert(velocity(smooth(parse(slice)))) == production_frames`, not per-layer, so a composition-order
  swap is caught (§4.5 / B4).
- **Gate T3b (PR-S95 — re-route correctness + ball):** post-re-route, IDSSE actions↔frames pass the
  action↔shooter identity check on the native path (expected green — Gate D); re-check the
  ball-at-wrong-end natively (gone, or root-caused as a separate bug).
- **Gate E (y-identity golden — TWO repos, distinct paths):**
  - **silly-kicks half (this spec):** exercises `tracking.kloppy.convert_to_frames` (SkillCorner +
    Metrica) — committed **RED first** (reproduces the gateway inversion), green after the CS-pin; plus
    the **native-IDSSE** slice in PR-S95, which is a **green-from-start regression guard** (Gate D = clean,
    nothing to drive out — NOT mislabeled RED-first).
  - **lakehouse half (Gate-C handoff):** a *distinct* y-identity golden on `_bronze_metrica_to_frames` /
    `_bronze_skillcorner_to_frames` output, in the lakehouse repo. The phrase "production convert path"
    means the silly-kicks gateway *inside silly-kicks* and the lakehouse builders *inside the lakehouse* —
    they are different code; each repo guards its own.

DGX state: `ssh karsten@192.168.68.73`, venv `~/sk-s93-venv` (kloppy 3.19), `source ~/.pining_env`,
`export PINING_CACHE_DIR=~/Development/silly-kicks/xt_bandwidth_run/artifact_cache` (all 81 pining
matches cached; the lakehouse repo is at `~/Development/luxury-lakehouse-fourier-promote`). The DGX
silly-kicks tree had stale uncommitted prior-session state; the native sportec converter is `c2e9d65`
(4.20.1) and **unchanged through merged main**, so Gate D at the current tree tests the byte-identical
production converter. Repro: `scripts/_tf48_*.py` (incl. `_tf48_gate_d.py`).

---

## 6. Blast radius (for ADR-031)

SkillCorner / Metrica (kloppy gateway) + IDSSE *dev harness* (kloppy path), action-anchor × frame-y:

- **CORRUPTED (fixed):** `add_action_context` (`nearest_defender_distance`,
  `defenders_in_triangle_to_goal`, `receiver_zone_density`); `add_pressure_on_actor` all flavors incl.
  vy; `add_pre_shot_gk_*` distances/angles; `add_shot_goalmouth`.
- **ISOMETRY-IMMUNE / already correct:** x-only + frame-integrated aggregates — `team_shape`
  spread/area/compactness/inter-line-gaps/height; `add_defensive_line` height + spread.
- **MIRRORED-position (harmless until y consumed):** absolute `team_shape_centroid_y`,
  `defensive_line_y`, `pre_shot_gk_y`.
- **VERIFY-PER-FEATURE** (sampled at an action coord → reads the mirror cell; softer because PC/xT
  surfaces are smooth + y-symmetric): `obso`, `pitch_control`, `space_creation`, `pausa`, `das`,
  `get_xc`, `gk_influence`, `player_influence`, `xt_gk`. The §7.5 A/B per-column delta is a
  **committed e2e test** (`tests/tracking/test_y_blast_radius_ab.py`, offline geometric subset).
- **Measured A/B (committed e2e, 2026-06-16) — refines this taxonomy:** under a frames-only y-flip,
  `nearest_defender_distance`, `pre_shot_gk_distance_to_shot`, and `pre_shot_gk_y` **change**;
  `actor_speed` (magnitude), `pre_shot_gk_x` (x-only), and **`pre_shot_gk_distance_to_goal`** do **not**
  — the last because the goal sits at **centre y=34**, so the distance is y-symmetric (§6 had over-listed
  it among the corrupted distances; corrected). `receiver_zone_density` + `defenders_in_triangle_to_goal`
  are structurally 0 on the test slice (no players in the synthesized actions' receiver-zones/triangles),
  so the committed fixture cannot exercise their y-sensitivity — they remain corrupted *in principle*;
  documented, not silently capped.
- **NOT affected:** native sportec/IDSSE (Gate D clean); Gradient Sports native; event-only providers.

---

## 7. Testing strategy (TDD / hexagonal / committed)

1. **Cross-provider y-identity golden (keystone, Gate E).** One committed real slice **per provider**
   (SkillCorner + Metrica in PR-S94; native-IDSSE in PR-S95) through the production convert path,
   asserting acting-player frame-y ≈ action `start_y` off-centre (`|start_y−34|>8`) — the
   `test_frame_orientation_golden.py` shape (one slice per provider, geometric invariant).
   **Cross-provider is mandatory** (the bug hid because coverage was per-provider-incomplete).
   Synthetic fixtures structurally cannot catch a self-consistent y-mirror. **Commit RED first** on
   the providers that reproduce the bug (proves sensitivity); green after. Fixtures stay in-repo
   (minimal off-centre slices; three-tier per `feedback_lakehouse_derived_ci_fixtures`).
2. **Parity contract tests (drift-impossible, not just shared).**
   - **PR-S94:** event-gateway canonical-y == tracking-gateway canonical-y on one fixture (precedent:
     lakehouse `test_convert_drift.py`). Commit RED first.
   - **PR-S95:** parse-port output == lakehouse parser output by **semantic equality** (float
     tolerance + canonical ordering; byte-exact only ids/enums — S2) on a committed DFL slice; plus
     the smoothing+velocity parity assertion (§4.5).
3. **CS-pin unit test (PR-S94).** On a kloppy fixture / stub `TrackingDataset`, assert the gateway pins
   the CS / produces canonical handedness; red before, green after.
4. **Event-path byte-equivalence (PR-S94).** Assert the `_kloppy_coordinates` extraction leaves the
   `spadl/kloppy.py` event conversion byte-identical via the **WC2018 golden** (N3: drop the weak
   import-identity assertion — it proves re-export, not unchanged output).
5. **Blast-radius A/B e2e (committed).** The §6 VERIFY-PER-FEATURE per-column delta (y vs 68−y on a few
   SC matches) is a committed e2e test asserting "these N columns change, those M don't" — keeps §6
   honest after future refactors, not a one-off plan run. **Extend it (rev-4 E2E):** assert the
   native-IDSSE path's corrupted-feature columns (the §6 CORRUPTED list) match a known-good reference
   end-to-end — making "production IDSSE is correct" a durable *feature-level* claim, not only the
   coordinate-level identity Gate D proved.
6. **Diagnosis discipline** (`feedback_coordinate_mismatch_diagnosis_rigor`): every localization uses
   the **library** transform (`tracking._action_orientation.*`), filters degenerate near-axis cases,
   and compares action↔**shooter** (same player/instant), never action↔ball. No hand-rolled 4-way menu.
7. Per PR: `ruff format --check`, `ruff check`, `pyright silly_kicks/`, `pytest tests/ -m "not e2e"`.

---

## 8. Retrain trigger + ADR-031

- **ADR-031 (silly-kicks)** records: the root cause (event/tracking gateway asymmetry), the CS-pin fix
  (and why NOT a blanket flip), the §6 blast-radius taxonomy + the A/B verdict, the Gate-C
  cross-repo finding, the **Gate-D verdict (native clean)**, the four-layer drift + parse-port decision
  (§4.4), and the smoothing+velocity single-source choice (§4.5). References (N2 — repo-qualified)
  silly-kicks ADR-004 / ADR-006 / ADR-019 / ADR-028 / ADR-029; the lakehouse's own ADR-029 (ET guard)
  and ADR-046 (terraform dep parity) are **distinct** documents in the lakehouse repo. Written in
  PR-S94, amended by PR-S95. Also records the **Gate-C resolution** (PR-S94 fixes calibration, not
  lakehouse prod; the lakehouse owns its SkillCorner/Metrica builder y-correctness) and **C4 — the
  parse-port release-coupling trade-off:** once the lakehouse deletes its private parser and depends on
  `silly-kicks[parse-dfl]`, every future DFL-parser change routes through a silly-kicks release
  (PyPI → wheel → terraform `==` pin → ADR-046 parity) instead of a one-repo lakehouse patch — the right
  hexagonal direction (drift-elimination) but raises lakehouse change latency + couples cadence; a
  **chosen** cost, recorded so it isn't a surprise.
- **Retrain trigger, scoped per PR:**
  - **PR-S94:** **calibration recommendations only** (the pining harness now sees correct gateway
    frames) + any external kloppy-gateway consumer. **NOT lakehouse VAEP/tracking** for SkillCorner/
    Metrica — the lakehouse builds those via its own `convert.py`, untouched by this PR (Gate C). A
    lakehouse SC/Metrica retrain, if its own builder check finds a y-bug, is a separate lakehouse PR.
  - **PR-S95:** **IDSSE calibration consumers** (harness now feeds native-parsed frames); lakehouse
    re-materializes IDSSE tracking features after adopting the port + shared data-quality callable.
  - **T2 / native sportec: NO retrain** (Gate D confirmed clean).
  - Hyrum: any lakehouse table from the affected providers' tracking features re-materializes.
    CHANGELOG + TODO note each trigger against its PR.

---

## 9. Risks / open questions

- **R1 — `transform()` signature (Gate 0).** CS alongside `to_pitch_dimensions`, or dims dropped?
  Decided empirically against the proven `(x, 68−y)` output.
- **R2 — RESOLVED (Gate D clean).** The "native sportec mis-handed" hypothesis is refuted (§4.3); T2
  is a documented no-op. No `tracking/sportec.py` y-map change.
- **R3 — Cross-repo parser adoption (PR-S95).** silly-kicks ships the port; the lakehouse adopts it +
  deletes its copy in a separate PR, touching the S3 checklist (pyproject hard-dep, terraform `==`
  pins / ADR-046, PEP-723 footgun, void-strip/dedup stay consumer-side). The parity test (§7.2) guards
  the upstream until adoption. (The "events-shim infeasibility" risk is **dissolved**.)
- **R4 — game_id-None workaround retires (PR-S95).** Native sets `game_id` from `match_id`, so the
  `_loader_pining.py:313-314` stamp + `test_calibrate_cli.py:36` expectation update. **grep ALL
  consumers of the None behavior — incl. any lakehouse loader reading silly-kicks output — before
  flipping.**
- **R5 — Fixture size.** Cross-provider golden + parity fixtures stay in-repo (not >100 MB blobs);
  extract minimal off-centre slices per provider.
- **R6 — Numeric single-source is a GATE, not a footnote (B4).** Smoothing + velocity drift is
  y-independent (golden-invisible); PR-S95 either asserts dev+prod compose the same callable (parity
  test) or records an explicitly-accepted measured residual in ADR-031.
- **R7 — New namespace + extra + shape-coupling (PR-S95).** `silly_kicks/providers/sportec/` + the
  `[parse-dfl]` extra; the port emits the converter's input shape (S4), so reuse for a future provider
  is parse-only. Confirm `providers/` is the right generalizable home.
- **R8 — Lakehouse builder y-asymmetry (Gate C, lakehouse-side).** `_bronze_metrica_to_frames` flips y;
  `_bronze_skillcorner_to_frames` does not — one convention may be wrong vs SPADL event y, and no
  existing test catches a y-mirror (orientation golden is x-based). NOT a silly-kicks fix (it's the
  lakehouse's code) — carried in the Gate-C handoff for the lakehouse to run the action↔shooter
  diagnostic + add its own y-identity golden. Surfaced here so PR-S94's scope isn't mistaken for closing
  it.
- **(Resolved) old R6 "single commit, broad change"** — eliminated by the PR split.

---

## 10. Anchors (for cross-session verification)

- `silly_kicks/spadl/kloppy.py:196-202` (event transform, CS pinned), `:293-326`
  (`_SoccerActionCoordinateSystem`), `:9-43` (kloppy-domain imports).
- `silly_kicks/tracking/kloppy.py:104-113` (tracking transform, CS NOT pinned — the bug).
- `silly_kicks/tracking/sportec.py:131-146` (native y map + per-period flip — Gate-D CLEAN), `:37-54`
  (`EXPECTED_INPUT_COLUMNS` = the native-input contract the port must emit).
- `silly_kicks/spadl/sportec.py:1012-1015` (events `start_y` from `rows["y"]`).
- `scripts/_loader_pining.py:253-257` (`_preprocess` = `smooth_frames` + `derive_velocities`),
  `:300-315` (IDSSE branch), `:318-359` (`_kloppy_tracking_to_frames`, retires).
- `scripts/_loader_databricks.py:157-192` (IDSSE bronze → native converters; raises SC/Metrica;
  `home_start_left=True` placeholder `:174`).
- `scripts/_tf48_gate_d.py` (Gate-D measurement, DGX).
- **Lakehouse repo `~/Development/luxury-lakehouse-fourier-promote` (verified on the DGX; line numbers
  differ from the rev-2/rev-3 reviewer's newer snapshot — re-locate on adoption):**
  `src/ingestion/idsse.py` (`_parse_teams`, `_parse_positions_xml` — raw DFL X/Y, no flip, confirmed;
  `_smooth_tracking`/smoothing); `tracking_context.py` (`_bronze_idsse_to_sportec_input`,
  `_derive_velocities_savgol`); `spadl_adapter.py` (`derive_idsse_home_team_start_left`); precedents
  `src/tests/action_context/test_convert_drift.py`, `test_frame_orientation_golden.py`.
- Report: `docs/research/bug_kloppy_tracking_y_inverted.md`. TODO: "### Confirmed bugs".
