# Spec: TF-23 — SkillCorner + Metrica bronze→frame builders (single-source the coordinate/orientation/clock truth)

**Date:** 2026-06-18
**Status:** Proposed — **Revision 2** (post luxury-lakehouse review); for re-review before any build
**Target:** next minor (4.33.0 tentative); additive to silly-kicks, no breaking silly-kicks API change
**Decision record:** ADR-034 (to author) — **supersedes the "decided against native converter" clause of ADR-029**, **harmonizes with lakehouse ADR-053** (promotes its geometric frame-LTR net into the library), extends ADR-031 (cross-repo bronze contract) to SkillCorner/Metrica
**Origin:** TF-23 re-scope 2026-06-18. Sportec-XML half shipped as the DFL parse-port (4.30.0, PR-S95, ADR-031 T3). Grounded in a live audit of the lakehouse SC/Metrica builders + the live Databricks bronze schemas (2026-06-18).

### Revision 2 — what changed after the lakehouse review (2026-06-18)

The review verified the thesis and the testing strategy but surfaced three load-bearing facts I had missed; all three are confirmed against live lakehouse source and are now integrated:

1. **There is a *second* lakehouse builder copy** — `src/analytics/action_context/convert.py::_bronze_{metrica,skillcorner}_to_frames` is the builder for the **live `fct_action_context` mart** (the published dataset + all tracking features). It is a near-verbatim duplicate of the `tracking_context.py` copy the v1 spec audited. So there are **three** coordinate truths, not two. The migration must retire **both** lakehouse copies, with `convert.py` named **primary** (`tracking_context.py`/`fct_tracking_context` is a legacy path the lakehouse is retiring). — *§2, §7*
2. **Orientation is already solved better in the lakehouse** — `analytics.action_context.pipeline.correct_frames_to_home_ltr` (ADR-053, shipped 2026-06-14) is a flag-free, per-period **GK-geometry** orienter, idempotent, that also fixes the GS extra-time per-feed flip. It supersedes the v1 spec's §4.3 bootstrap. **Resolution: promote it into the library** (per the "best practice for all consumers" mandate; ADR-053 itself relayed it upstream). — *§4.3, §11*
3. **`ball_z` recovery is a capability unlock, not a side effect** — `bronze.skillcorner_tracking.ball_z` is 100% populated (9,606,256 rows, −0.66→14.7 m, 99.5% non-zero). Both builders discard it (`z=NaN`), so SkillCorner `shot_on_target_derived` / `shot_crossing_z` / `shot_z_profile` are **silently null in production**. Recovering it unlocks on-target derivation + PSxG ball-height for an entire provider, and is independently shippable. — *§1, §2.2*

Also adopted from the review: §5.3 is reframed as a diff-explainer (not a correctness gate); a **kloppy-independent, event-anchored** correctness gate is added as the primary Gate-C closer (§5.1); an **extra-time** fixture is added (§5.2).

### Revision 3 — second lakehouse review (2026-06-18)

Near ship-ready; four nail-downs integrated, two resolved by verifying source rather than assuming:

1. **§5.1(A) is self-contained in silly-kicks (Gate C closes by construction).** Verified: the event-anchored gate uses silly-kicks' own `tracking.utils.link_actions_to_frames` (utils.py:297 — "nearest tracking frame in time within tolerance," per-period), **not** the lakehouse AC linker. No cross-repo dependency; the keystone gate runs on every CI leg. — *§5.1(A)*
2. **The z-in-parity "contradiction" is not real — kloppy carries SkillCorner ball-z.** Verified against the installed kloppy SkillCorner deserializer (`skillcorner.py:127, 249-251`): the ball is `Point3D(x, y, z=float(z))`; players are 2D `Point`. z is invariant under the x/y CS-pin + orientation flip, so oracle-ball-z == builder-ball-z == raw bronze z. **z stays in the oracle-parity set** (it is the *expected diff* in the §5.3 old-vs-new check — the old builder drops it); plus an **independent z validation** (range + airborne spot-check) is added as defense-in-depth. — *§5.1(B)*
3. **The orienter promotion is a schema-adapted port, not a verbatim lift.** `correct_frames_to_home_ltr` is a transform over an already-schema'd frame; the lift adapts column conventions to `KLOPPY_TRACKING_FRAMES_COLUMNS`. Acceptance oracle = ADR-053's existing tests (`test_frame_ltr_correction.py` + `test_frame_orientation_golden.py`), mirrored into silly-kicks as the cross-repo equivalence contract. — *§4.3*
4. **End-state = retire the lakehouse orientation net entirely, gated on TF-23b.** The interim provider-conditional net (kept for idsse/GS, removed for SC/Metrica) is a temporary, idempotent GS-ET-only backstop; TF-23b (geometric net on the native adapters) is **the gate that lets the lakehouse delete its net entirely** — elevated from "fast-follow." — *§7, §8, O6*

Plus: metrica-y is named as a non-skippable gate case (§5.1A); a GK-derivation risk line added (§10); long-term flag-based-orienter deprecation tracked (§9).

### Execution findings (2026-06-18, on real Databricks/kloppy data)

The builders, orienter, and gates were implemented and run against real data. Two findings refine §5.1 and surface a relay item:

1. **Metrica builder is canonical-correct AND has an input contract.** On contract-honoring bronze, `tracking.metrica.convert_to_frames` matches the kloppy gateway **byte-for-byte (dx=dy=0)** on open-data game 1 (incl. LTR orientation). The contract: **bronze `y` must be SPADL bottom-to-top** — kloppy's metrica NATIVE CS is top-to-bottom, so a consumer landing bronze straight from a kloppy `TrackingDataset` MUST flip `y` (`1−y`); the lakehouse bronze already does (its "metrica y is already SPADL bottom-to-top"). Documented on the builder + a network-gated open-data parity test (`test_metrica_builder_matches_kloppy_oracle_open_data`).
2. **Event-anchored gate is SkillCorner (real Databricks slice), metrica via kloppy parity.** §5.1(A) was implemented as a **ball-position** (identity-free) probe — better than the acting-player probe (no player-id bridge, and the ball is the on-target Gate-C probe). The SkillCorner gate **passes** on a real `skillcorner_tracking ⋈ skillcorner_matches` + `spadl_actions` slice (both teams, both periods). The metrica event-anchored path was retired because the lakehouse metrica `spadl_actions` **event-y did not co-locate with the canonical tracking ball** in the pilot — metrica Gate-C is the kloppy-oracle parity instead (the proper canonical check).

**RELAY to luxury-lakehouse:** the metrica `bronze.spadl_actions` event-y appears inconsistent with the canonical metrica tracking-ball y (pilot: a Home P1 action at start_y=16.8 m vs canonical ball ≈33 m at the linked frame). Worth a lakehouse look at the metrica events→SPADL y handling — separate from TF-23 (the TF-23 builder is validated canonical-correct).

---

## 1. Executive summary

silly-kicks and the luxury-lakehouse maintain **three** copies of the SkillCorner/Metrica
bronze→frame coordinate transform: silly-kicks' kloppy gateway
(`tracking.kloppy.convert_to_frames`, the oracle), and **two** lakehouse copies —
`analytics/action_context/convert.py` (the live `fct_action_context` mart path) and
the legacy `ingestion/tracking_context.py`. The lakehouse copies exist because the
lakehouse holds **bronze DataFrames**, not kloppy `TrackingDataset` objects, and
cannot feed the gateway. That triplication is the structural source of the
y-inversion / direction-double-flip / period-clock-mismatch parade of the last weeks
(ADR-019/028/029/031/053). silly-kicks' own 4.29.0 kloppy y-fix (ADR-031) **does not
reach lakehouse prod** for these providers, and ADR-031 **Gate C** is still open.

This spec adds two **pure, bronze-consuming** frame builders —
`tracking.skillcorner.convert_to_frames(bronze, …)` /
`tracking.metrica.convert_to_frames(bronze, …)` — parallel to `tracking.sportec` /
`tracking.gradientsports`, owning the full bronze→canonical-oriented-frame transform
(rescale + period-clock re-base + id-namespacing + **`ball_z` recovery** + GK
derivation + speed + LTR orientation). They emit the kloppy-variant schema and are
validated by **two correctness gates**: a kloppy-independent **event-anchored
action↔frame y-identity** check (the actual Gate-C invariant), and **parity to the
kloppy oracle**. The lakehouse then deletes **both** its builders + the duplicated
clock constant **and the downstream orientation net**, depending on these (ADR-031
delete-and-depend). This **structurally closes Gate C** for the path that ships.

Two consolidations ride along, both worth their own line:

- **Orientation is unified.** The lakehouse's flag-free geometric orienter
  (`correct_frames_to_home_ltr`, ADR-053) — strictly more robust than silly-kicks'
  flag-based `orient_frames_to_ltr`, and the only mechanism that fixes the GS-ET
  per-feed flip — is **promoted into the library** as the canonical orienter the
  builders use, so every consumer (not just the lakehouse) inherits it.
- **`ball_z` is recovered.** SkillCorner shot-height features that are silently null
  in production today (`shot_on_target_derived`, `shot_crossing_z`, `shot_z_profile`,
  PSxG) light up for the whole provider.

This is **not** a reversal of ADR-029: that ruled against a *raw-file* converter;
these consume *bronze*. `orient_frames_to_ltr` is retained for authoritative-flag
callers; only the duplicated coordinate/clock/orientation truths are consolidated.

---

## 2. Background — what exists today (verified 2026-06-18)

| Path | Frame builder | Orientation | Status |
|---|---|---|---|
| silly-kicks gateway | `tracking.kloppy.convert_to_frames` (needs `TrackingDataset`) | labeled LTR via `play_left_to_right` | **oracle** (CS-pinned, ADR-031) |
| **lakehouse AC mart** | `analytics/action_context/convert.py::_bronze_{m,sc}_to_frames` | absolute → **`correct_frames_to_home_ltr`** (ADR-053 net, `pipeline.py`) | **live `fct_action_context`** — primary target |
| lakehouse legacy | `ingestion/tracking_context.py::_bronze_{m,sc}_to_frames` | absolute, **net NOT applied** (ADR-053 §Negative) | legacy `fct_tracking_context`, being retired — still buggy |
| sportec / idsse | `tracking.sportec.convert_to_frames(ltr)` | labeled LTR | lakehouse already delegates (4.30.0) |
| gradientsports | `tracking.gradientsports.convert_to_frames(ltr)` | labeled LTR (+ net corrects ET) | native |

The **kloppy gateway is the oracle** (`silly_kicks/tracking/kloppy.py`): pins the
canonical SPADL coordinate system (`socceraction_coordinate_system`, ADR-031 — the
y-fix), labels `team_attacking_direction`, derives GK + speed, applies
`play_left_to_right`. But it requires a kloppy `TrackingDataset` (raw files + parse),
which the lakehouse does not have at bronze.

### 2.1 The triplication, concretely

The two lakehouse builders (`convert.py` + `tracking_context.py`) are **near-verbatim
duplicates of each other** — identical docstrings, identical
`Metrica y*68 no-flip "verified live vs event y"` (convert.py:149 / tc.py:993),
identical SkillCorner `z=NaN` (convert.py / tc.py:1153). Each hand-maintains four
truths independent of silly-kicks:

1. **Coordinate rescale** — Metrica `*105/*68`; SkillCorner `+52.5/+34.0`.
2. **y-flip decision** — hand-verified "NO flip; verified live vs event y."
3. **Period-clock offsets** — `_SKILLCORNER_PERIOD_START_SECONDS` = a verbatim copy of
   silly-kicks' `skillcorner.py::_PERIOD_START_SECONDS`. Silent desync on retune.
4. **Orientation** — builders emit `team_attacking_direction=None`; orientation is a
   *separate downstream step* (`correct_frames_to_home_ltr`) wired on the AC path
   only — the legacy path has no orientation at all.

silly-kicks' 4.29.0 kloppy y-fix **does not reach** any of these. The three truths
are kept aligned only by independent manual re-verification; Gate C is the open proof.

### 2.2 Live bronze schemas + the `ball_z` capability unlock

```
metrica_tracking:    period, frame, timestamp, ball_x, ball_y,
                     home_players(JSON), away_players(JSON), match_id,
                     frame_rate, gk_jersey_numbers(JSON),
                     pitch_length_m, pitch_width_m, is_anonymized   — NO ball z (z=NaN correct)
skillcorner_tracking: match_id, period, frame, timestamp, player_id(LONG),
                     x, y, is_visible, ball_x, ball_y, ball_z,
                     ball_is_detected, frame_rate                   — ball_z PRESENT
skillcorner_matches: ..., team_id, position_acronym, home_team_id, away_team_id,
                     pitch_length, pitch_width, period_boundaries(JSON), ...
```

- **`ball_z` recovery — capability unlock (not a side effect).** Lakehouse-verified:
  `skillcorner_tracking.ball_z` is **100% populated, 9.6M rows, −0.66→14.7 m, 99.5%
  non-zero** — real ball height. Both lakehouse builders discard it (`z=NaN`), so the
  SkillCorner post-shot height features — `shot_on_target_derived`, `shot_crossing_z`,
  `shot_z_profile` (TF-48), and any PSxG ball-height consumer — are **silently null in
  production for the entire provider**. The consolidated builder maps `ball_z → z`
  (and `is_visible → visibility`), lighting them up. This is independently valuable
  and ships even under the O1 "keep ADR-029 line" fallback. Same class as the GS `z=0`
  hardcode TF-48 fixed.
- **No provider bronze carries `home_team_start_left`.** The kloppy gateway infers
  orientation from kloppy's transform; the bronze builders must derive it from
  geometry (§4.3). **O3 RESOLVED (verified 2026-06-18):** the nominal `_PERIOD_START_SECONDS`
  is EXACT, not an approximation — every SkillCorner match's `timestamp` is nominal-aligned
  per period (P2 starts at exactly 2700.0 in all 10 pining matches, regardless of P1
  stoppage), so `timestamp − nominal_start` is exactly period-relative-from-0 and matches
  the events. `skillcorner_matches.period_boundaries` is **frame-index structure**
  (`start_frame`/`end_frame`/`duration_minutes`), NOT a seconds-clock — switching to it would
  *diverge* from the events. Keep nominal; O3 is closed (no change).
- **Metrica has no ball z** — `z=NaN` is correct there (asymmetric handling is right).

---

## 3. The central decision: where is the bounded-context boundary?

This is the question the review must ratify, because it moves work **out of** the
lakehouse ingestion context. ADR-029 drew the line at "orientation only":
> *"That mapping (bronze schema, coord scaling) is correctly lakehouse-owned … the
> orientation is the only part that belongs upstream — hence option (b), not a native
> converter (option a)."*

**This spec argues that line is wrong**, and the triplication is the proof: the parts
ADR-029 left lakehouse-owned (rescale, clock, ball-z mapping, the orientation
*bootstrap*) are exactly where the three truths diverge and re-break. Rescale +
clock-rebase + pitch geometry are **domain logic** (the physics/geometry of the
pitch), not ingestion plumbing. The cleaner boundary:

- **Lakehouse (ingestion) — unchanged:** raw files → bronze Delta; Spark read; the
  `skillcorner_tracking ⋈ skillcorner_matches` team/GK/meta join; Spark→pandas.
- **silly-kicks (frame construction) — NEW:** bronze pandas DataFrame → canonical
  oriented frames (rescale + clock + id-namespace + `ball_z` + GK + speed + orient).

ADR-029's `orient_frames_to_ltr` and ADR-053's `correct_frames_to_home_ltr` were the
lakehouse's two successive attempts to fix *just orientation* from the consumer side;
this spec completes the consolidation so all three duplicated truths live once,
upstream. **Review note (lakehouse endorsed O1, strongly, conditioned on retiring
both copies + the net — §7.)**

---

## 4. Design

### 4.1 Public surface (mirrors `tracking.sportec`)

Two new pure modules, exported as submodules from `tracking/__init__.py`:

```python
# silly_kicks/tracking/skillcorner.py
def convert_to_frames(
    bronze: pd.DataFrame,            # post-join narrow bronze (4.2)
    *,
    home_team_id: Any,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,        # None ⇒ geometric orient (4.3)
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]: ...

# silly_kicks/tracking/metrica.py
def convert_to_frames(
    bronze: pd.DataFrame,            # frame-level bronze (JSON player cols)
    *,
    home_team_id: Any = "Home",
    jersey_to_player_id: dict[str, dict[str, str]] | None = None,  # team→jersey→pid
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]: ...
```

- **Pure** (pandas in → pandas out; no Spark, no I/O) — hexagonal, matches every
  converter. Returns `(frames, TrackingConversionReport)`.
- `output_convention` mirrors the gateway (ADR-006). Default `"ltr"` orients via the
  promoted geometric net (§4.3). `"absolute_frame"` emits labeled absolute frames
  (no flip) for callers that orient later — and is the clean lever for the
  double-orientation decision (§7 / O-new).
- `preprocess` thread-through, **off by default** (cosmetic data-quality stays
  consumer-side; DFL-port precedent).

### 4.2 Bronze input contract (`EXPECTED_INPUT_COLUMNS`)

Each module pins an explicit input-column contract (raise loud on missing columns).
The team/GK/meta **join is the consumer's** (ingestion); the builder consumes the
post-join narrow frame:

- **SkillCorner** (post `⋈ skillcorner_matches`): `match_id`, `period`, `frame`,
  `timestamp`, `player_id`, `team_id`, `is_goalkeeper`, `x`, `y`, `ball_x`, `ball_y`,
  **`ball_z`**, **`is_visible`**, `frame_rate`. (`ball_z`/`is_visible` newly consumed.)
- **Metrica**: `period`, `frame`, `timestamp`, `ball_x`, `ball_y`, `home_players`,
  `away_players`, `gk_jersey_numbers`, `frame_rate` (+ optional `pitch_length_m/width_m`,
  informational — 0–1 → 105×68 standardization; confirm in test).

### 4.3 Orientation — promote the geometric net (drop the bootstrap)

The v1 spec proposed a new `infer_home_team_start_left` (period-1 GK side + alternation).
**The lakehouse already shipped a strictly better mechanism** —
`correct_frames_to_home_ltr` (ADR-053, 2026-06-14) — and the review correctly flagged
re-implementing it as a third orienter. Per the "best practice for all consumers"
mandate, the decision is to **promote it into silly-kicks** as the canonical orienter:

- New public `tracking.orient_frames_to_ltr_by_geometry(frames, *, home_team_id, …)`
  (name TBD), a **schema-adapted port** of `correct_frames_to_home_ltr` — NOT a
  byte-verbatim lift (it is a transform over an already-schema'd frame, so the port
  adapts column conventions to `KLOPPY_TRACKING_FRAMES_COLUMNS`: `x/y/team_id/
  period_id/is_ball/is_goalkeeper/team_attacking_direction` align; `vx/vy` are flipped
  *when present* — silly-kicks frames carry them only post-`derive_velocities`, which
  the source already handles conditionally; pitch constants source from `spadlconfig`,
  not lakehouse literals; `ids_match` is already silly-kicks' own). Per period, the
  home GK median x is the directional anchor; any period with home-GK median x on the
  attacking half (`>52.5`) is point-reflected (`x→105−x`, `y→68−y`, `vx→−vx`, `vy→−vy`;
  speed unchanged); labels populated when null; zero-home-match guard; every flip logged.
  **Acceptance oracle = ADR-053's existing tests** (`test_frame_ltr_correction.py` unit
  + `test_frame_orientation_golden.py` cross-provider real-slice golden), mirrored into
  silly-kicks to prove the port equivalent to the lakehouse original. After promotion,
  the lakehouse retires its copy and depends on the library function, so that golden
  becomes a **cross-repo equivalence contract** — the lakehouse behavior is the source
  of truth the port must reproduce.
- Why it's better than the dropped bootstrap **and** than flag-based
  `orient_frames_to_ltr`: it reads orientation from **data**, so it is robust to the
  absent/defaulted `home_team_start_left` (no bronze field) **and** fixes the GS-ET
  per-feed flip that *no single flag can* (SPADL actions + frames share one ET flag,
  but GS ships ET tracking end-flipped vs events for some matches — ADR-053 §Context).
- **Idempotent** — re-running on correctly-oriented frames is a no-op (home GK already
  low-x). This neutralizes the *severity* of the double-orientation hazard (§7) but
  does not excuse leaving the duplicate wired.
- The SC/Metrica builders call it internally when `output_convention="ltr"`. An
  explicit `home_team_start_left=` override remains for callers with authoritative
  metadata (routes through flag-based `orient_frames_to_ltr` instead — kept, not removed).

**Three orientation mechanisms, reconciled:** flag-based `orient_frames_to_ltr`
(ADR-029) stays for authoritative-flag callers; the v1 bootstrap is **dropped**; the
geometric net (ADR-053) is **promoted to the library** as the default for bronze
builders and a reusable correctness net for any consumer. The lakehouse retires its
copy (§7).

**Follow-on (out of scope, flagged):** the promoted geometric net is the tool that
would fix the **GS native adapter's** ET flip in the library too (it currently trusts
the unreliable `home_team_start_left_extratime` for ET tracking). Applying it as a net
to the native adapters is a GS-adapter change + GS retrain trigger — a strongly
recommended fast-follow, **not** bundled here. Tracked as TF-23b.

### 4.4 Transform pipeline (both builders)

1. Shape bronze → long-form rows (Metrica: explode player JSON; SkillCorner: rename +
   dedup ball rows).
2. Rescale to 105×68 (Metrica `*105/*68`; SkillCorner `+52.5/+34.0`).
3. Map `ball_z → z`, `is_visible → visibility` (SkillCorner); Metrica `z=NaN`.
4. Period clock → period-relative, **importing `_PERIOD_START_SECONDS` from
   `skillcorner.py`** (kills duplicated-truth #3 — do not copy).
5. Id-namespacing → object strings (match the gateway's `str(...)` ids).
6. GK derivation via `_gk_identification.derive_goalkeepers`.
7. Speed via `_derive_speed` (or `preprocess`).
8. Orient via the promoted geometric net (§4.3) when `output_convention="ltr"`.
9. Emit `SKILLCORNER_TRACKING_FRAMES_COLUMNS` / `METRICA_TRACKING_FRAMES_COLUMNS`
   (= `KLOPPY_TRACKING_FRAMES_COLUMNS`; named aliases in `schema.py`).

---

## 5. Testing strategy (TDD; red-first)

Two correctness gates, because **kloppy self-consistency alone cannot prove the fix**
(the gateway itself was y-wrong pre-4.29.0). The event-anchored gate is the real
Gate-C invariant; the kloppy-parity gate is the structural agreement-with-the-blessed-path.

### 5.1 Correctness gates

**(A) Event-anchored action↔frame y-identity — the primary Gate-C closer
(kloppy-independent, self-contained in silly-kicks).** Link the match's SPADL actions
to the builder's frames using **silly-kicks' own `tracking.utils.link_actions_to_frames`**
(nearest frame by `(period_id, time_seconds)` within tolerance — utils.py:297; NOT the
lakehouse AC linker, so the gate is fully in-library and Gate C closes *by
construction*). Assert the **acting player's frame position at the action instant ≈ the
action's start coordinate** (after ADR-028 reprojection to the action-LTR frame),
within tolerance, **for both teams and at off-centre y** (the error `|68−2y|` vanishes
at y=34 — centre cases must be excluded, per `feedback_coordinate_mismatch_diagnosis_rigor`).
Plus field invariants: both teams' shots cluster high-x post-LTR; each defending GK
sits at its attacked goal. This is independent of kloppy and catches
builder-y-vs-event-y disagreement — the exact bug-class.
- **Metrica-y is a NAMED, non-skippable case** (not folded into a generic loop):
  Metrica was the historical y-bug and the highest-risk axis (the kloppy oracle was
  itself y-wrong pre-4.29.0). The committed Metrica fixture must exercise **both teams,
  off-centre y, and an ET period** — a centre-y Metrica case proves nothing.

**(B) Parity to the kloppy oracle — structural agreement.** Owner-gated e2e: oracle =
`tracking.kloppy.convert_to_frames(ds, output_convention="ltr")`; builder =
`tracking.{skillcorner,metrica}.convert_to_frames(bronze, …)` on the same match.
Assert frame-equal on the coordinate truth (`x, y, z, time_seconds, period_id,
is_goalkeeper, team_attacking_direction`, per-player position keyed on a normalized
identity) within a tight float tolerance. **Ids compared up to provider convention**:
kloppy's anonymized Metrica teams are `"home"/"away"` vs the builder's labels, so
Metrica asserts identity up to home/away+jersey; SkillCorner asserts raw id equality
(both use the SkillCorner numeric team id stringified). Bronze fixtures extracted once
from Databricks (documented `SOURCE_SHA`, DFL-port precedent); a committed reduced
slice + committed golden run on every CI leg (pandas2/3 string-dtype handled per
`feedback_pandas2_vs_pandas3_string_dtype_parity`).

**On `z` in the parity set (resolved):** verified that the kloppy SkillCorner
deserializer *does* carry ball z (`skillcorner.py:127, 249-251` — ball is `Point3D`,
players are 2D `Point`), and the gateway reads it (`kloppy.py:158-160`). z is invariant
under the x/y CS-pin and the orientation flip, so oracle-ball-z == builder-ball-z ==
raw bronze z — **keep z in parity** (it is the *expected diff* vs. the old lakehouse
builder in §5.3, which drops it). Independently, validate z semantics (not just
agreement): ball z ∈ [0, 10] m, and a spot-check that a known airborne event
(header/lob/save) shows `z > 0` at the right frame — the physical check the
agreement-to-oracle gate cannot make. (Player z is NaN on both sides — bronze carries
no player z; consistent.)

### 5.2 Unit / invariant (committed)

- **Orientation involution** (per `symmetry_test_insufficient_pin_ground_truth`):
  asymmetric + extreme ground-truth fixture (home AND away; GK at one goal → asserts
  post-orient GK lands at the attacked goal — pins the involution direction), plus the
  home==away mirror test; plus an **idempotence** test (orienting twice == once).
- **Extra-time fixture** (NEW per review): real-or-synthetic periods 3/4 exercising
  `require_et_direction` + the 5400/6300 clock offsets + the GS-ET-class per-period
  geometric flip. ET is historically where these bugs hide.
- **Data recovery**: `ball_z` carried through (not NaN); `is_visible → visibility`.
- **Clock single-source**: guard asserting the builder imports `_PERIOD_START_SECONDS`
  from `skillcorner.py` (no local copy) — regression test for duplicated-truth #3.
- **Schema/constraints**: output passes `TRACKING_CONSTRAINTS`; dtypes match
  `KLOPPY_TRACKING_FRAMES_COLUMNS`.
- **Loud guards**: missing input column → raise; ET without flag → raise;
  zero-home-team match (ADR-019) → raise; geometric-orient zero-GK-anchor → documented
  fallback/warn (ADR-053 semantics preserved).
- **Id-dtype seams**: numeric vs string id inputs (ADR-019 fixtures).

### 5.3 Cross-repo migration gate — a diff-explainer, NOT a correctness gate

Old-builder-vs-new-builder on the same bronze fixture is a **change-surface check
only**: the old builder is the *suspect*, so it cannot be the oracle. The correctness
gates are §5.1(A)+(B). The diff is expected and explained: `z` now populated;
`team_attacking_direction` now labeled + geometrically oriented; everything else
byte-equal. It is the lakehouse's delete-and-depend safety net before deletion.

---

## 6. CDF-readiness (forward-looking; not a dependency)

The `convert_to_frames(bronze, …)` boundary is the seam for a future
`tracking.cdf.convert_to_frames`. **Do not build for CDF now** — it is v0.2.2 and the
schema does not yet specify origin/units/orientation (verified 2026-06-18). The only
requirement here: keep each builder's transform a thin, well-named `bronze→frames`
step so a CDF reader slots in beside `skillcorner`/`metrica`. See TF-38.

---

## 7. Rollout / migration (delete-and-depend)

1. Ship the two builders + the **promoted geometric orienter** + both correctness
   gates in silly-kicks — additive, no silly-kicks retrain (new modules + new public
   orienter; existing converters/gateway untouched; in no default xfn list).
2. Lakehouse bumps the pin and migrates **both** copies:
   - **`analytics/action_context/convert.py` (primary, live `fct_action_context`):**
     swap `_bronze_{m,sc}_to_frames` → `tracking.{skillcorner,metrica}.convert_to_frames`,
     **and remove the `correct_frames_to_home_ltr` call for SC/Metrica** (the builder
     now orients). Decision to ratify (O-new): either the builder emits `ltr` and the
     net is retired for these providers, OR the builder emits `absolute_frame` and the
     net keeps orienting — **pick one owner**. Recommended: builder emits `ltr`, net
     retired for SC/Metrica.
   - **End-state: the lakehouse orientation net is retired ENTIRELY (not left
     provider-conditional).** A provider-conditional net (kept for idsse/GS, removed for
     SC/Metrica) is a footgun (add a provider, forget the branch). Once all four
     builders emit oriented LTR upstream — idsse/GS already do via
     `convert_to_frames(ltr)`, SC/Metrica via these new builders — the net's *only*
     remaining job is the GS-ET per-feed flip. **TF-23b** (the geometric net on the GS
     native adapter, §8) eliminates that, after which `correct_frames_to_home_ltr` is
     fully redundant and deleted. Until TF-23b lands, the net stays as a **temporary,
     idempotent, GS-ET-only backstop** — explicitly tracked against TF-23b so the
     interim conditional does not ossify into permanent.
   - **`ingestion/tracking_context.py` (legacy `fct_tracking_context`):** same swap;
     delete the builder + `_SKILLCORNER_PERIOD_START_SECONDS`. (ADR-053 §Negative
     records this path was never covered by the net — consolidation fixes it.)
   - Run the §5.3 migration gate on both; re-materialize SC/Metrica marts. **This is
     the lakehouse's retrain trigger, not silly-kicks'** (`z` populated, orientation
     correct where absolute) — re-materialize + downstream VAEP/AC consumers.
3. ADR-031 **Gate C is closed by construction** — the event-anchored gate replaces
   manual y-verification, on the path that ships.

**Precondition (owner/lakehouse call):** the payoff exists only if the lakehouse
delete-and-depends **both** copies. The spec ships nothing that *forces* adoption.

---

## 8. Non-goals

- **No raw-file loader** (ADR-029 option a). Builders consume *bronze*; raw→bronze
  stays lakehouse ingestion.
- **No change to the kloppy gateway / sportec / gradientsports adapters** here
  (Chesterton). **Applying the promoted geometric net to the native adapters to fix
  GS-ET in the library is TF-23b** — not bundled (it is a GS retrain trigger), but
  elevated from "fast-follow" to **the gate that lets the lakehouse delete its
  orientation net entirely** (§7.2). It must be scheduled, not deprioritized, or the
  interim provider-conditional net becomes permanent.
- **No removal of `orient_frames_to_ltr`** — retained for authoritative-flag callers.
- **No GS bronze-builder work** (GS converters already native).
- **No CDF reader** (§6).
- **No switch to `period_boundaries` clock** — O3 RESOLVED (§2.2): nominal is exact for
  SkillCorner; `period_boundaries` is frame-index structure, not a seconds-clock.
- **No new action-coupled aggregator** — the surface is two bronze→frame converters + a
  geometric orienter (C4-free; the C4 aggregator count stays **28**).

## 9. Open decisions for the lakehouse re-review

- **O1 — bounded-context boundary (§3).** Move the transform to silly-kicks.
  *Review: endorsed, strongly, conditioned on retiring both copies + the net.*
- **O2 — orientation (§4.3).** **Promote `correct_frames_to_home_ltr` into the
  library** (revised from the v1 bootstrap). *Review: required — do not build a third
  orienter.* Decision: lift it (golden-gated), lakehouse retires its copy.
- **O-new — double-orientation owner (§7.2).** Builder emits `ltr` + net retired for
  SC/Metrica (recommended), vs. builder emits `absolute_frame` + net keeps orienting.
  Pick one owner explicitly.
- **O3 — clock source. RESOLVED + CLOSED (verified 2026-06-18).** Nominal
  `_PERIOD_START_SECONDS` is EXACT for SkillCorner (every match's `timestamp` is
  nominal-aligned per period — P2 = exactly 2700.0 across all 10 pining matches despite
  P1 stoppage), so `period_boundaries` (frame indices, not seconds) is the wrong source
  and switching would diverge from the events. No follow-up.
- **O4 — Metrica roster mapping.** Pass `team→jersey→player_id` in. *Review: agree
  (lakehouse owns identity).*
- **O5 — post-join SkillCorner bronze.** Keep the `⋈ skillcorner_matches` join
  consumer-side. *Review: agree (ingestion-context).*
- **O6 — TF-23b (GS-ET net upstream).** Schedule the native-adapter geometric net —
  **the gate for full deletion of the lakehouse net** (§7.2/§8). *Review: yes, separate
  PR (GS retrain); elevate, do not deprioritize.*
- **O7 — long-term: two orienters. NOT resolvable now — BLOCKED on TF-23b.** The
  flag-based `orient_frames_to_ltr` (and the underlying `compute_attacking_direction` +
  `play_left_to_right`) has LIVE callers today: the native sportec/GS adapters supply an
  authoritative `home_team_start_left` from DFL/GS metadata and orient via the flag path.
  So it cannot be deprecated until TF-23b migrates those adapters onto the geometric net.
  And even then it may resolve as **keep both** (flag-based is the *authoritative* path
  when a reliable flag exists; the geometric net is the robust no-flag/backstop path) — so
  O7 is "revisit after TF-23b", not a committed deprecation. Track; do not act now.

## 10. Risks

- **Parity tolerance / float + pandas 2↔3 drift** — committed-golden multi-hash set +
  `check_dtype=False` string normalization.
- **Geometric orient on a pathological match** (no GK anchor in a period) — ADR-053's
  documented fallback (away-GK anchor, else warn-leave-as-is) is preserved + the
  event-anchored gate (§5.1A) catches a wrong result rather than shipping it.
- **Orientation correctness depends on GK-derivation correctness.** The geometric net
  anchors on the home-GK median x; a wrong `derive_goalkeepers` pick (highest risk:
  anonymized Metrica from `gk_jersey_numbers`) flips orientation the wrong way. Pipeline
  order is deliberate (GK derive @ step 6 *before* orient @ step 8); a wrong GK pick is
  the **one input that silently defeats the net** — caught only by the event-anchored
  gate (§5.1A), which is why that gate, not oracle-parity, is the primary Gate-C closer.
- **kloppy oracle is only as correct as ADR-031's CS-pin** — mitigated by making
  §5.1(A) the *primary* gate (event-anchored, kloppy-independent) and freezing the
  agreed truth in the committed golden.
- **Adoption gap** — if the lakehouse does not retire both copies + the net, the
  triplication persists; §7 precondition.

## 11. ADR note (ADR-034, to author with the feature commit)

- **Supersedes** ADR-029's "decided against a native `metrica`/`skillcorner.convert_to_frames`"
  clause — the *bronze-consuming* builder is the correct consolidation;
  `orient_frames_to_ltr` retained.
- **Harmonizes with ADR-053** — promotes its flag-free geometric frame-LTR net into the
  library (ADR-053 itself "relayed upstream" and listed "orientation stays in
  silly-kicks" as the appeal of its rejected flag-based option; this realizes that with
  ADR-053's robustness). The lakehouse net is retired in favor of the library function.
- **Extends** ADR-031's cross-repo bronze contract + delete-and-depend from Sportec/DFL
  to SkillCorner/Metrica; closes Gate C on the shipping path.
- **Records** the bounded-context boundary (§3), the promote-not-reinvent orientation
  decision (§4.3), and the `ball_z` capability recovery (§2.2).
