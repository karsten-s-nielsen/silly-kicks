# Spec — DFL parse+shape port (`silly-kicks[parse-dfl]`): single-source the IDSSE/Sportec parser

**Status:** REVISED ×2 — lakehouse review (A–F) + re-review (N1 bronze-ownership/naming, N2 committed-golden parity, N3 B-ii sync, N4 events-bronze, N5 test_convert_drift, N6 correctness framing) all applied. Lakehouse verdict: "plan-ready once N1/N2 are nailed" — done.
**Date:** 2026-06-16
**Author:** main session (Karsten / Claude)
**PR:** PR-S95 / **T3** of ADR-031 (follows PR-S94 / T1, shipped 4.29.0)
**Base:** `main` @ v4.29.0
**Decision record:** ADR-031 (amend) — the parse-port decision is §4.4 there; this spec is the
standalone, source-verified design.
**Target version:** 4.30.0 · **Retrain (CORRECTNESS remediation, N6):** the dev-harness IDSSE path
(`_kloppy_tracking_to_frames`) was NOT touched by PR-S94 and has been emitting **y-inverted** frames
all along; the re-route to the native (Gate-D-correct) path **corrects** them, so the existing IDSSE
**calibration artifacts** (xT bandwidth, carrier defaults, …) were fit on inverted-y frames and must be
re-run. Not a cosmetic "switch to native-parsed frames."

---

## 1. Problem (the T3 four-layer drift)

For IDSSE/Sportec, the dev/calibration harness and production use **different code at every layer**
(established in ADR-031 §4.4, rev-2 review against the lakehouse repo):

| Layer | Dev harness (`scripts/_loader_pining.py`) | Production (lakehouse `src/ingestion/idsse.py`) |
|-------|-------------------------------------------|--------------------------------------------------|
| Parse | kloppy `sportec.load_event`/`load_tracking` | hand-rolled stdlib `xml.etree`, kloppy-free |
| Smooth | silly-kicks `smooth_frames` | lakehouse `analytics.smoothing.smooth_positions` |
| Velocity | silly-kicks `derive_velocities` | lakehouse `_derive_velocities_savgol` (parity-by-comment) |
| Convert | kloppy `spadl.kloppy` + `_kloppy_tracking_to_frames` | native `spadl.sportec` / `tracking.sportec` |

PR-S94 (T1) fixed the kloppy *gateway* y-inversion, but the dev harness still parses IDSSE with kloppy
while production parses with the hand-rolled `xml.etree` parser — a train/serve skew at the **parser**
layer. T3 single-sources the parser by **upstreaming the lakehouse's pure DFL parse functions into
silly-kicks as a reusable parse+shape port**, consumed by both.

**Why a port, not a kloppy shim:** the kloppy-shim leaves parse/smooth/velocity divergent. The port
makes silly-kicks own the DFL→native parse it already converts from (hexagonal; the lakehouse is a
consumer), and is the generalizable `parse_<provider>_*` shape for any future provider.

**Source confirmed (cloned lakehouse `main` @ `0efac60`, 2026-06-16):** the current parser is complete
(captures ball `Z/S/BallStatus` — unlike the stale DGX `fourier-promote` snapshot) and the parse
functions are pure (ET + stdlib + pandas; no lakehouse-internal calls inside them). The lakehouse pin
is `silly-kicks>=4.27.0,<5` — it adopts the port *downstream*, so silly-kicks ships PR-S95 first.

**Scope + residual (review C — explicit, not buried):** PR-S95 single-sources the **parse** layer
only. The **smooth + velocity** layers stay consumer-side (§2.4); their dev-vs-prod convergence is the
lakehouse's downstream decision (R3). **Success criterion: "single-source the parser" ✓; NOT "eliminate
all train/serve drift"** — two numeric layers remain (tracked follow-ups). The §1 table is the *map*,
not PR-S95's full conquest. **But note (N6): the re-route is ALSO a correctness fix** — the harness
IDSSE path (`_kloppy_tracking_to_frames`, retiring here) was y-inverted and untouched by PR-S94, so the
re-route un-inverts the harness IDSSE frames; the IDSSE-calibration re-run is **correctness
remediation** (artifacts were fit on inverted-y frames), not a refresh.

---

## 2. The parse+shape port

### 2.1 Module + extra

New package `silly_kicks/providers/sportec/` (a new top-level `providers/` layer — the generalizable
per-provider parse-port home; R7), with `parse.py`. Behind a new optional extra
**`silly-kicks[parse-dfl]`** (the port is stdlib-`xml.etree` + pandas; the extra keeps it opt-in and
gives the lakehouse a hard-dep handle on adoption). No new heavy core deps.

### 2.2 Public surface — SPLIT at the bronze seam (review A), typed returns

The lakehouse production pipeline **persists bronze** between parse and shape
(`bronze.idsse_tracking` → many consumers: the AC pipeline reads bronze, not raw XML). A port that
emitted the converter's post-shaper input would skip bronze and be un-adoptable by production. So the
port is **two composables split at the bronze seam**:

```python
# PARSE: bytes -> sportec-canonical BRONZE rows (the faithful parse port; the PRIMARY return)
def parse_dfl_match_info(info_xml: str | Path) -> MatchInfo
def parse_dfl_events(events_xml: str | Path, *, player_team_map, ...) -> list[SportecEventBronze]
def parse_dfl_tracking(positions_xml: str | Path, *, player_team_map, ...) -> SportecTrackingBronze
# SHAPE: BRONZE -> native converter input (converter-coupled glue; a SECOND composable)
def shape_tracking_to_native(bronze: SportecTrackingBronze) -> NativeTrackingInput  # EXPECTED_INPUT_COLUMNS
def shape_events_to_native(bronze: list[SportecEventBronze]) -> NativeEventsInput     # spadl.sportec input
```

- **Production:** `parse → bronze → [persist bronze] → shape → convert`. **Harness:** `parse → shape →
  convert` (no persist). **Both run the identical `parse` AND identical `shape`** → parser drift truly
  eliminated, the bronze medallion layer preserved.
- `MatchInfo` (dataclass): home/away team ids, `player_team_map`, GK ids, `home_team_start_left` (incl.
  the `extraTimeFirstHalf` ET variant) — seeds `_parse_teams` (idsse.py:437), `_parse_match_metadata`
  (:517), `derive_idsse_home_team_start_left` (spadl_adapter.py:438).
- `SportecTrackingBronze` — **silly-kicks' own domain name** (N1), field-identical today to the
  lakehouse `_IDSSE_TRACKING_BRONZE_COLS` (idsse.py:846; `bronze.idsse_tracking`: `match_id, period,
  frame, timestamp, x, y, s, ball_x/ball_y/ball_z, ball_status, …`) — seeds `_parse_positions_xml`
  (:620). `SportecEventBronze` — field-identical to `_IDSSE_EVENTS_BRONZE_COLS` (idsse.py:354;
  `bronze.idsse_events`) — seeds `_parse_events_xml` (:1356) / `_build_event_row` (:1147).
- `NativeTrackingInput` = `tracking.sportec.EXPECTED_INPUT_COLUMNS`; `NativeEventsInput` =
  `spadl.sportec.convert_to_actions` input — the shapers seed `_bronze_idsse_to_sportec_input`
  (`action_context/convert.py` / `tracking_context.py`) + the events adapter (`spadl_adapter.py`).

**N1 — bronze-schema ownership (a cross-repo contract under BOTH models).** Emitting the bronze shape
is what makes drop-in adoption possible, but it inverts ownership: silly-kicks' bronze types and the
lakehouse bronze DDL (`bronze.idsse_tracking`/`idsse_events`) become a **versioned cross-repo
contract** — a bronze-column rename/add is now a coordinated silly-kicks-port change. This coupling
holds under B-ii (parity) too, not just B-i. We mitigate by naming the types in silly-kicks' OWN domain
(`SportecTrackingBronze`, NOT "the lakehouse's `_IDSSE_TRACKING_BRONZE_COLS`"), field-identical today.
**Cleaner-decoupling option (the lakehouse's call):** the port emits a sportec-canonical bronze and the
lakehouse keeps a thin, trivial, in-repo rename adapter to its DDL columns — freeing the lakehouse to
evolve its own bronze table names independently. Recorded; the lakehouse picks drop-in vs thin-adapter.

**Hexagonal boundary (sharpened, S4):** the **parse port is `bytes → bronze rows`** (faithful, the
provider-domain shape the lakehouse persists); the **shaper is `bronze → converter input`**
(converter-coupled glue). Typed returns (dataclass/TypedDict) pin both shapes as APIs; field renames
are breaking.

### 2.3 What gets lifted (verified pure on `0efac60`)

Copied (NOT imported — `idsse.py` imports `ingestion.guards`/`utils`/`workflows` at module top), at a
**pinned lakehouse commit** (`0efac60`; the source is live — review F): `_parse_teams`,
`_parse_match_metadata`, `_parse_positions_xml`, `_parse_events_xml`/`_build_event_row`,
`derive_idsse_home_team_start_left`, the bronze→native shaper(s), + the small helpers
`_parse_float_or_none` / `_parse_bool_or_none` / `_SECTION_TO_PERIOD`.

**Function-level, not file-level (review minor):** the lakehouse shaper lives in
`action_context/convert.py` / `tracking_context.py` **shared** with the Metrica/SkillCorner/GS builders
+ the velocity helper (and the two copies are drift-locked by the lakehouse's own `test_convert_drift`).
The lift copies the **IDSSE parse + IDSSE-shape functions only**; nothing deletes a shared file or
another provider's builder.

Provenance: a module docstring + code comment noting the functions were upstreamed from
`luxury-lakehouse src/ingestion/idsse.py @ 0efac60` (both repos owner-owned; silly-kicks is MIT). Not a
published methodology → no academic NOTICE entry; a provenance note suffices.

### 2.4 Data-quality stays consumer-side (NOT in the port)

The port is a faithful bytes→typed-rows boundary. Smoothing (`analytics.smoothing.smooth_positions`),
velocity (`_derive_velocities_savgol`), void-strip, dedup are **modeling** decisions and do **not**
move into the port (ADR-031 §4.5). Consumers compose explicitly: `velocity(smooth(parse(...)))`
(smooth-then-velocity — the canonical order). For PR-S95 the dev harness composes silly-kicks'
existing `smooth_frames`+`derive_velocities` (its current `_preprocess`); the *shared canonical
callable* convergence is the lakehouse's downstream decision and does NOT block the port.

---

## 3. Re-route the dev harness + retire the kloppy IDSSE path

- **Re-route** `_loader_pining.py` IDSSE branch (`:300-315`) → `parse_dfl_*` → native
  `spadl.sportec.convert_to_actions` + `tracking.sportec.convert_to_frames`. **No kloppy parser in the
  loop.** `home_team_start_left` from `parse_dfl_match_info` (NOT the `_loader_databricks.py:174`
  `True` placeholder).
- **Retire** `_kloppy_tracking_to_frames` (`:318-359`, single caller `:306`) + the kloppy IDSSE events
  path (`:304`) — National Park / Chesterton (caller confirmed unique in PR-S94).
- **Retire the `game_id`-None workaround** (`:313-314`) — the native converter sets `game_id` from
  `match_id`; update the `test_calibrate_cli.py:36` expectation (it encodes "spadl_kloppy leaves
  game_id None"). grep ALL consumers of the None behaviour first.

---

## 4. Testing

1. **Parse(+shape)-port parity (keystone) — RED-first (E), against a COMMITTED GOLDEN (N2), SHA-gated
   (F).** silly-kicks CI is standalone — it cannot clone/import the lakehouse — so the parity reference
   is **golden parquets captured once from lakehouse@`0efac60` and committed** into silly-kicks
   alongside the DFL fixture: `idsse_parse_bronze_golden.parquet` + `idsse_shape_native_golden.parquet`
   (+ the events equivalents). The test asserts the port reproduces them by **semantic equality**
   (documented float tolerance + canonical column/row ordering; byte-exact only for ids / `ball_state`
   enums — S2), comparing **both** the bronze-seam output AND the shaped native output (so an A-split
   regression in either half is caught). **Proven RED on an empty/stub port first** (it has teeth),
   then GREEN. The golden IS the frozen `0efac60` snapshot → **re-pinning = regenerating the golden in
   one reviewed commit** (makes F operationally real); when the lakehouse adopts + deletes its copy,
   the golden becomes the port's own.
2. **Re-route correctness.** Post-re-route, native-IDSSE action↔shooter identity (off-centre) — green
   per Gate D (the native path is already y-correct); the IDSSE ball-at-wrong-end re-checked natively.
3. **Cross-provider y-identity golden.** Add the **native-IDSSE** slice to
   `tests/tracking/test_kloppy_y_identity_golden.py` (the PR-S94 SC+Metrica golden) — green-from-start
   (Gate D), via the production native convert path.
4. **Calibration-invariance e2e (review minor + N6 tolerance).** A committed harness e2e asserting
   harness IDSSE output is **invariant pre/post re-route** on everything except the **y-fix delta** —
   which is **large** (a full un-inversion `|68−2y|` on y-anchored features, NOT a rounding wobble),
   while non-y features (kloppy-parse vs native-parse) differ by **~0**. So the test asserts: y-anchored
   columns move (the correctness fix), x-only / y-symmetric / frame-integrated columns are ~unchanged
   (tolerance sized for parse-engine numeric noise, not for a flip). Stronger than the calibration
   re-run alone.
5. **Fixtures — a real plan task, not a truncation (D).** DFL position XML is nested `FrameSet`/`Frame`;
   a naive byte-cut yields invalid XML the parser rejects. The capture must emit **schema-valid reduced
   DFL XML** (drop whole `FrameSet`s / trim `Frame` ranges while preserving structure) for positions +
   events + metadata, and the capture script must **assert the lifted parser accepts it**. Sized to
   stay in-repo (mirrors the PR-S94 `yident` capture, but XML-structure-aware).
6. Run `ruff format --check`, `ruff check`, `pyright silly_kicks/`, **full `tests/`** (`-m "not e2e"` —
   the whole dir, not a subset; the PR-S94 CI miss).

---

## 5. Lakehouse adoption — the LAKEHOUSE's downstream decision (review B)

**PR-S95 (silly-kicks side) is invariant to this choice** — it ships the bronze-split port + the
parity test, and the dev harness re-routes onto it, regardless. How the *lakehouse* adopts is a
separate lakehouse PR + decision, between two models:

- **(B-i) Delete-and-depend:** the lakehouse imports `silly_kicks.providers.sportec` (the IDSSE
  parse+shape functions), deletes its own, pins `silly-kicks[parse-dfl] >= 4.30.0`. Zero
  dual-maintenance; pays **C4 release-coupling** — every future DFL-parser change routes lakehouse →
  PyPI release → wheel → terraform pin.
- **(B-ii) Keep-both-and-parity:** the lakehouse keeps its parser and parity-tests it against the
  silly-kicks port (the pattern it already uses internally via `test_convert_drift`). No
  release-coupling for routine lakehouse parser iteration, but **NOT zero ongoing cost (N3)**: the
  lakehouse-side parity test is locked to a silly-kicks port version, so a silly-kicks release that
  changes the port forces a **bump-time sync** (re-pin + reconcile). So B-ii = "dual-maintenance +
  bump-time sync" — genuinely lower coupling than B-i, but eyes-open.

The **parity test (§4.1) is the actual drift-eliminator and works under either model.** Given the
lakehouse DFL parser is **actively iterating** (Z/S/BallStatus just added; more changes expected),
**sequence the lakehouse adoption AFTER the parser churn settles** — lifting/locking a snapshot while
the source moves risks a born-stale port (the SHA pin in §4.1/§2.3 makes "what was lifted" explicit).
silly-kicks ships the port + the SHA-pinned parity test now; the lakehouse picks B-i vs B-ii later.

**Adoption checklist (S3, carried in the handoff, applies to B-i):** (a) lakehouse `pyproject.toml`
requires `[parse-dfl]` (a HARD runtime dep); (b) terraform serverless env `==` pin mirroring `uv.lock`
(lakehouse ADR-046 / `test_terraform_env_dep_parity.py`); (c) the PEP-723 trainer downgrade footgun;
(d) void-strip / dedup stay consumer-side; (e) **rework `test_convert_drift` (N5)** — once the lakehouse
imports `shape_tracking_to_native` and deletes both `_bronze_idsse_to_sportec_input` copies,
`test_idsse_converter_no_drift` (`src/tests/action_context/test_convert_drift.py:30`, AST-compares the
two copies) is moot → drop the IDSSE-shaper drift assertion (other-provider assertions stay).

---

## 6. Risks / open questions

- **R1 — DFL fixture is schema-valid-reduction, not truncation (review D).** Nested `FrameSet`/`Frame`
  XML; the capture must emit structurally-valid reduced DFL + assert the parser accepts it (§4.5). A
  real plan task, not a one-liner.
- **R2 — Events-side shaper liftability.** The bronze-events→`spadl.sportec.convert_to_actions`-input
  mapping (in `spadl_adapter.py`) must lift as cleanly as the tracking shaper — adapter layer confirmed
  at the spec level; per-function lifting is a plan task.
- **R3 — RESIDUAL: smooth + velocity drift remain (review C).** Stated explicitly in §1 — PR-S95 closes
  parse only; the numeric-layer convergence (one shared smoother+velocity callable) is the lakehouse's
  downstream decision (ADR-031 §4.5). Tracked follow-up, NOT a PR-S95 blocker.
- **R4 — `providers/` namespace.** New top-level package; confirm it's the right generalizable home
  (vs folding under `spadl`/`tracking`).
- **R5 — Moving target / lift fidelity (review F).** The lakehouse parser is **live** (Z/S/BallStatus
  just added; more expected). The lift is pinned to `0efac60` + the parity test gates on that SHA
  (§2.3/§4.1). Sequence the lakehouse adoption AFTER the churn settles (§5) to avoid a born-stale port.
  (The in-flight lakehouse change set — 4.29.0 adoption + a Metrica builder y-fix — touches the Metrica
  builder + velocity helper, NOT the IDSSE parse/shaper functions being lifted, so no collision.)
- **R6 — Adoption model is the lakehouse's call (review B).** Delete-and-depend (B-i, release-coupling)
  vs keep-both-and-parity (B-ii, dual-maintenance) — §5. PR-S95 is invariant to it.

---

## 7. Anchors (for cross-session verification)

- **Lakehouse, PINNED @ `0efac60` (cloned on the DGX `~/Development/luxury-lakehouse`; the parity test
  gates on this SHA — F):** `src/ingestion/idsse.py` — `_parse_teams:437`, `_parse_match_metadata:517`,
  `_parse_positions_xml:620` (two-pass; PASS-1 ball `X/Y/Z/S/A/D/M/T/BallPossession/BallStatus`),
  `_build_event_row:1147`, `_parse_events_xml:1356`, `_IDSSE_TRACKING_BRONZE_COLS:846`
  (`bronze.idsse_tracking` shape = the tracking parse port's primary return),
  `_IDSSE_EVENTS_BRONZE_COLS:354` (`bronze.idsse_events` shape = the events parse port's primary
  return — N4), helpers `_parse_float_or_none`/`_parse_bool_or_none`/`_SECTION_TO_PERIOD`;
  `src/ingestion/spadl_adapter.py:438` `derive_idsse_home_team_start_left` (+ the bronze-events→native
  adapter); `src/analytics/action_context/convert.py` + `tracking_context.py`
  (`_bronze_idsse_to_sportec_input` / `x_centered`; shared with other-provider builders + drift-locked
  by `src/tests/action_context/test_convert_drift.py:30 test_idsse_converter_no_drift` — lift the IDSSE
  functions only; B-i moots that assertion, N5).
- **silly-kicks:** `silly_kicks/tracking/sportec.py:37-54` (`EXPECTED_INPUT_COLUMNS`) + `:60` signature;
  `silly_kicks/spadl/sportec.py` `convert_to_actions` input contract; `scripts/_loader_pining.py:300-359`
  (IDSSE branch + `_kloppy_tracking_to_frames` to retire); `tests/calibration/test_calibrate_cli.py:36`
  (game_id-None expectation). ADR-031 §4.4/§4.5; PR-S94 = the shipped T1.
