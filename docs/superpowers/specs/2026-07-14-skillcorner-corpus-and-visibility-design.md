# SkillCorner corpus expansion, visibility surfacing, and expanded-corpus retrain re-registration

**Status:** rev 4 — APPROVED subject to corrections, all applied; ready for the plan
**Date:** 2026-07-14
**Cycle:** TF-19 GKDV, between PR-1 (4.47.0, shipped) and PR-2 (weights + chirality enforcement)
**Supersedes nothing. Blocks:** PR-2 (weights are not frozen until the registered runs below complete).

---

## Executive summary (read this first)

The pining API's SkillCorner listing grew from 10 to 108 matches. Investigation of the 98 new
ones produced four findings, in descending order of consequence:

1. **They are owner-tier, not public.** `visibility: "private"`, licence *"Restricted;
   redistribution not permitted"*. The public arm stays at 17 matches (10 SkillCorner + 7 IDSSE),
   so **the paired tests already run are not invalidated**. What the 98 can expand is the
   *owner/full* arm: 81 → 179 matches.

2. **A licensing landmine sits directly in that path.** Both trainers classify public-vs-owner by
   *provider name* (`_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}`). The 98 carry provider
   `skillcorner`. Wire them in naively and they are absorbed into the **public** arm — and the
   model we ship *because* it is reproducible from redistributable data would be trained on
   restricted data. This must be made impossible before the data is reachable.

3. **`is_detected` has been in the SkillCorner feed all along, and we throw it away.** The
   *native* bronze builder (`tracking/skillcorner.py:149`) already maps it to `visibility`; the
   *kloppy gateway* (`tracking/kloppy.py:154,180`) hard-codes `None`. So the research/pining path
   is degraded relative to the lakehouse path. Measured on real data: **goalkeepers are detected
   in only 19.6% of frames** (outfield 66.6%) — i.e. ~80% of SkillCorner keeper positions are
   inferred, not observed. This is quantitative vindication of two rules we adopted on judgement:
   the registered "GKDV measurement runs on Gradient Sports frames only", and ADR-024/PR-S104's
   distrust of SkillCorner keeper origins.

4. **Neither tracking path agrees with our own events converter on a non-105×68 pitch.** The native
   builder applies a fixed `+52.5 / +34.0` offset with no pitch-dimension input, so on the
   **4 of 10 public matches** that are 104/106 m — and the 7 of 98 down at 101×67 — it mis-places the
   goal line by up to **2.0 m** (the lakehouse consumes this today). kloppy scales, but *not* on the
   events convention: measured mirror-invariantly on 956,076 rows it assumes a pitch **103.48 m**
   long where the metadata declares **104.00**, diverging **0.263 m** at the goal line (width and
   centre spot are correct). The right transform is neither kloppy's nor a new one — it is the
   **affine part** of `spadl/skillcorner.py::_transform_coords`, which SkillCorner **events** already
   use. §3.4 single-sources that affine part — and, critically, **not** the clamp that follows it,
   which is safe for events and destructive for tracking (§3.4).

The work below (a) makes the 98 reachable and safely classified, (b) surfaces `visibility` by routing
SkillCorner through the native builder, (c) fixes the pitch-dimension defect that this routing would
otherwise inherit — **without** inheriting the events converter's coordinate clamp, which is safe for
events and destroys tracking (§3.4) — and (d) **registers, before any fit sees data**, the rules by
which the expanded corpus is admitted:

- a three-arm paired test in a **fixed sequence**, with **tuning nested inside the outer CV** so that
  no arm's hyperparameters ever see the fold they are scored on (§4.1);
- a ghost-GK admission test on **detected-keeper targets only**, over a **common keeper domain**, by
  paired per-fold sign-consistency — with an automatic refusal if the expanded model improves on
  all frames while degrading on the frames where the keeper was actually *seen* (§4.3);
- a **staged retrain** so the pipeline change cannot be confounded with the corpus change (§4.2);
- a permanent exclusion of the 98 from GKDV measurement (§4.5).

Registration-before-fitting is the point of this document; the loader mechanics are the easy part.

---

## 1. Findings (evidence)

All figures below were measured, not assumed. Probes and raw outputs are in the session
scratchpad (`sc_new_schema/`), to be copied into `docs/research/skillcorner_corpus/` with the
implementing PR.

### 1.1 What the 98 matches are

| | Public (canonical) | Private (new) |
|---|---|---|
| n | 10 | 98 |
| Source | SkillCorner Open Data (MIT) | SkillCorner, *"Restricted; redistribution not permitted"* |
| `visibility` | `public` | `private` |
| S3 prefix | `skillcorner/…` | `skillcorner/_private/…` |
| Competition | A-League 2024/25 | LaLiga 23/24 (37), LaLiga 24/25 (34), UCL 24/25 (14), UCL 23/24 (13) |
| Teams | mixed | **Real Madrid in all 98** (49 home, 49 away) |
| Size | — | ~10.8 MB/match, ~1.05 GB total |

Manifests are uniform across all 98 (identical artifact maps; 20/20 serially sampled matches have
all five roles, non-degenerate).

### 1.2 Same product, different containers

The new artifacts are **the same SkillCorner V3 product**, not a different tier:

- Tracking frame keys identical (`ball_data, frame, image_corners_projection, period,
  player_data, possession, timestamp`); player-row keys identical (`{x, y, player_id,
  is_detected}`); **10.000 fps** exactly in both; ball `z` present in both.
- Events: `events.parquet` vs `dynamic_events.csv` — **294 columns each, zero difference**, same
  `event_type` taxonomy. The SkillCorner SPADL converter runs unchanged (1415 actions,
  `result_source=native` on 1383).
- Geometry is clean: same-player event-vs-tracking identity reconciles at **0.0000 m RMSE** under
  identity or a 180° point reflection, per (team, period), for 100% of events. A y-only flip is
  never the best fit → **no y-inversion**; ADR-031's CS pin holds and ADR-028 handles rotation.
- Linkage is the best in the corpus: `link_actions_to_frames` gives **link_rate 1.000, median
  offset 0.000 s** — the events carry a native `frame_start` that *is* a tracking frame id.

Only the containers and manifest key names changed (CSV→Parquet, JSONL→gzipped JSON array,
`{id}_tracking_extrapolated` → `tracking`).

### 1.3 Why the loader fails (three points, all reproduced live)

| Site | Failure |
|---|---|
| `_loader_pining.py:125-130` `_artifact_key()` | resolves artifacts by filename **suffix**; raises `KeyError: no artifact ending with '_dynamic_events.csv' in ['events','freeze_frames','metadata','physical','tracking']` |
| `_loader_pining.py:98` `_download_to_temp()` | names the temp file `f"{provider}_{match_id}_{artifact_key}"` — **dropping the extension**. kloppy's `identify_data_version` sniffs the first byte, sees the gzip magic `0x1f`, and raises `DeserializationError`. Verified: with a `.json.gz` name, kloppy returns `V3` and parses. |
| `_loader_pining.py:291` `_build_skillcorner()` | uses `pd.read_csv`; needs a `.parquet` dispatch (`pyarrow>=14` is already a declared dep). |

### 1.4 The keeper confound (decisive for how the 98 may be used)

Real Madrid supplies **99 of 198 GK team-sheet slots (50%)** across the 98. Three keepers —
**Courtois (45), Lunin (35), Kepa (18)** — account for **~49.5%** of all keeper-slots, across a
cohort of 56 distinct keepers.

### 1.5 Detection rates (the `is_detected` finding)

Measured on the SkillCorner feed (present in **both** schemas):

| | detected | extrapolated |
|---|---|---|
| **Goalkeepers** | **19.6%** | **80.4%** |
| Outfield | 66.6% | 33.4% |

`silly_kicks/tracking/_gk_geometry.py:189` already gates detected-keeper resolution on
`_truthy_bool(frames["visibility"])`. On the kloppy path that column is `None` everywhere, so the
gate matches **zero** keepers and SkillCorner goal-kick origins always fall back to the rule-point
prior. The information to do better has been in the feed the whole time.

### 1.6 Pitch dimensions (a live defect in the native builder)

`tracking/skillcorner.py` has **no** `pitch_length` / `pitch_width` input (`EXPECTED_INPUT_COLUMNS`
omits them) and transforms with a fixed `x + 52.5`, `y + 34.0`. Measured across all 108 matches:

| pitch | n | goal-line error under the fixed offset |
|---|---|---|
| 105 × 68 | 97 | 0.0 m |
| 104 × 68 | 2 | 0.5 m |
| 106 × 68 | 2 | −0.5 m |
| 103 × 67 / 103 × 68 | 4 | 1.0 m |
| 101 × 67 / 101 × 68 | 3 | **2.0 m** |

**Four of the ten public matches** (1886347, 1899585 at 104; 2013725, 2015213 at 106) are affected.
The lakehouse builds SkillCorner frames with this builder, so its SkillCorner geometry carries this
error today.

#### 1.6.1 What the correct transform is — settled by measurement (rev 2)

Rev 1 asserted "the kloppy path scales from metadata and is correct". **That assertion was wrong,
and is retracted.** The standard is not kloppy — it is **this library's own SkillCorner events
converter**, because action↔frame co-location (ADR-028) is the contract that matters.

`silly_kicks/spadl/skillcorner.py::_transform_coords` computes (its affine part)

```
x_out = (x / (L/2)) * 52.5 + 52.5        # ≡ (x + L/2) * (105 / L)
y_out = (y / (W/2)) * 34.0 + 34.0        # ≡ (y + W/2) * (68  / W)
```

whose **affine part** is algebraically identical to §3.4's tracking transform (verified to
floating-point). kloppy does *not* compute this.

**Measured (rev 3, mirror-invariant, match 1886347 @ 104 × 68, 1:1 join on `(period, frame,
player)`, n = 956,076).** Distance from the centre spot is invariant under the 180° flip the kloppy
gateway applies, so regressing `|x_kloppy − 52.5|` on `|x_events − 52.5|` recovers the ratio of the
two effective scales without any orientation contamination:

| quantity | measured |
|---|---|
| x-slope (kloppy ÷ events) | **1.00502** |
| kloppy's effective **pitch length** | **103.48 m** (the metadata declares **104.00**) |
| kloppy's effective width | 68.00 m — **correct** |
| centre spot | kloppy **52.492** vs events 52.500 — **kloppy preserves it** |
| divergence at the goal line | **0.263 m** |

So kloppy silently assumes a pitch ~0.5 m shorter than the one SkillCorner declares, in **x only**.
(The reviewer's independent measurement agrees on structure and on the centre spot, and puts the
effective length at 103.71 / the divergence at ~0.15 m. The residual disagreement between our two
figures is unexplained and is *not* decision-relevant — see §1.6.2 — but it is recorded rather than
reconciled away.)

**Consequences:**

- The **lakehouse** error is the larger one (up to 2.0 m, §1.6 table) and is native-vs-truth.
- The **research path is also not clean**: kloppy-built tracking disagrees with our own SkillCorner
  *events* on the four non-105 public matches — a small, pre-existing action↔frame inconsistency that
  §3.3's re-route fixes. **Magnitude: see the caveat below — rev 2's "0.29 m" is withdrawn.**

#### 1.6.2 What single-sourcing does and does not prove (rev 3, reviewer's fallacy catch)

Rev 2 argued that single-sourcing removes the need for a landmark check, "because tracking and events
are computed by the same function, so they cannot disagree". **That is a fallacy and it is
withdrawn.** Single-sourcing guarantees *consistency*, not *correctness*: it makes tracking equal to
events, and if events are wrong then both are wrong together — unfalsifiably, since every §6 gate
compares them to each other.

The two candidate transforms have the same form and differ in exactly one number — the assumed pitch
length: **104 m** (the value the match metadata declares, and the value our events converter already
trusts) versus kloppy's smaller effective figure. What the choice rests on:

- **Provenance, not agreement.** 104 m is what SkillCorner *states* in `pitch_length`. Our SPADL
  events have been built on that number for every SkillCorner match ever converted. Choosing it for
  tracking preserves the events contract; choosing kloppy's number would mean **our events have been
  wrong all along** — a separate, pre-existing defect requiring its own evidence and its own fix.
- **The residual uncertainty is registered, not hidden.** If SkillCorner's declared `pitch_length` is
  not the length its coordinates are actually normalised against, then events *and* tracking are both
  off by the same small factor, and this spec does not fix that. The cheap empirical routes are
  closed (the feed's `image_corners_projection` is all-null; the events taxonomy has no set-piece
  type to use as a fixed-geometry landmark). **Action item, carried into the plan: ask SkillCorner.**
  Until answered, 104 is the registered choice on provenance grounds, and the choice is *recorded as
  a choice*.

**Provenance of the magnitude figures (rev 3).** Rev 2 published a divergence table (centre spot at
52.647; up to 0.29 m) computed from a scale factor inherited from a review pass since retracted as
fan-out-contaminated. My first re-measurement was contaminated too — by *orientation* (the kloppy
gateway emits LTR-oriented frames; comparing them against an unoriented affine map produces ±105 m
artefacts). **Both are withdrawn.** The table in §1.6.1 is the completed mirror-invariant
re-measurement (n = 956,076) and is the number the rest of this document cites.

**Why our figure and the reviewer's differ (103.48 vs 103.71).** Recorded, not reconciled: kloppy's
map is **not affine** — a clean affine fit leaves a ~0.14 m residual — so "effective pitch length" is
a *fit artefact*, and different estimators on different samples land in different places. This is
itself informative: **nobody has actually characterised what kloppy's SkillCorner transform does**,
which is a further reason to prefer the provenance-backed 104 m over reverse-engineering it. It goes
into the SkillCorner email as a question, not into this spec as a decision.

None of this changes any decision here: the structural facts — the native builder's fixed offset is
wrong, tracking must not inherit the events clamp, and 104 m is the provenance-backed length —
stand independently of the exact magnitude.

### 1.7 The other two artifact roles

- **`freeze_frames.parquet`** — per-event snapshots (5,556 frames × 23 rows). 100% of its
  `(period, frame)` keys already exist in the full tracking → **redundant**. Its only novel field
  is a per-player `visible_area` camera polygon.
- **`physical.parquet`** — 29 rows × 36 cols of match-level running aggregates (distance, HSR,
  sprints, accelerations, psv99). No per-frame content, no consumer in silly-kicks → **not used**.

---

## 2. Scope

**In scope**

- Loader support for the role-keyed artifact schema (both container variants).
- A licensing-safe corpus taxonomy keyed on `visibility`, fail-closed.
- Routing the SkillCorner pining path through the native bronze builder (surfacing `visibility`
  and `ball_z`; retiring a duplicated-truth fork).
- Fixing the native builder's pitch-dimension defect.
- **Registering** the expanded-corpus retrain protocol (§4). The runs themselves are owner-run and
  follow this document; they do not start before it is approved.

**Out of scope (explicit)**

- **GKDV measurement on the 98.** Never (see §4.5). Measurement stays Gradient-Sports-only.
- New SkillCorner-trained `GkCompletionModel` / `GkRetentionModel` variants on the expanded cohort
  (a genuine follow-up, not this cycle).
- The `freeze_frames` `visible_area` polygon (no consumer; revisit if a camera-coverage feature is
  ever specified).
- Any change to Gradient Sports or IDSSE handling.
- Upstreaming `is_detected` into kloppy (documented as a gateway limitation instead; the native
  builder is the supported path).

---

## 3. Design

### 3.1 Loader: role-aware resolution (`scripts/_loader_pining.py`)

Three surgical changes, matching the three failure points in §1.3:

1. `_artifact_key(artifacts, *, suffix, role)` — try the filename suffix first (canonical schema),
   then fall back to the **role key** (`events`, `metadata`, `tracking`). `_download_artifacts`
   already carries an `else role` fallback that is currently unreachable because `_artifact_key`
   raises first.
2. `_download_to_temp` — derive the destination name from the manifest's *filename value*, so the
   extension survives (`tracking.json.gz`, `events.parquet`). Safe for IDSSE/GS, which sniff magic
   bytes. **Consequence:** cache keys change → a one-time re-download of the SkillCorner cache.
3. `_build_skillcorner` — dispatch on suffix: `.parquet` → `pd.read_parquet`, `.csv` → `pd.read_csv`.

### 3.2 Corpus taxonomy: keyed on `visibility`, fail-closed

The manifest already carries `visibility: "public" | "private"` per match (§1.1). Therefore:

- `load_matches` surfaces it. To avoid breaking the five-tuple yield that every caller unpacks, the
  loader gains a **separate, cheap accessor** — `match_visibility(providers) -> dict[(provider,
  match_id) -> str]` — reading the same manifest it already fetches. No signature change.
- Both trainers already build a **per-row provider array** during extraction
  (`parts_p.append(np.array([prov] * len(X)))`, `train_xshot_occurrence.py:65`) but discard
  `match_id`. They gain a parallel **per-row visibility array** built in the same loop from the
  accessor, defaulting to `"private"` when the match is absent from the map:

  ```python
  # was: is_public = np.isin(providers, list(_PUBLIC_PROVIDERS))
  parts_v.append(np.array([vis.get((prov, mid), "private")] * len(X)))   # extraction
  is_public = visibilities == "public"                                    # arm split
  ```

  **Fail-closed by construction**: an unknown or missing visibility is treated as *restricted*. A
  new match can never silently enter the public arm. `_PUBLIC_PROVIDERS` is deleted, not merely
  bypassed, so the name-keyed rule cannot be reintroduced by accident.
- `_extract`'s return grows from a 4-tuple to a 5-tuple, and the feature cache gains
  `visibility.npy`.

**Rev 2 — closing the two holes the reviewer found (B3).** Rev 1's control was incomplete in two
ways, and both are load-bearing:

1. **The cache-hit predicate defeats it.** Both trainers gate on
   `(cache / "features.parquet").exists()`. Rev 1 said a stale `_feature_cache/` "must be treated as
   a miss" and gave **no mechanism** — and the 2026-07-13/14 owner runs have already populated that
   cache on the DGX. Registered fix: the cache directory carries a `cache_meta.json`
   (`schema_version` + a corpus fingerprint = sorted `(provider, match_id, visibility)` triples);
   **any mismatch, or an absent `cache_meta.json`, is a miss**. The plan additionally deletes the
   existing DGX caches explicitly rather than trusting the predicate.
2. **`_PUBLIC_PROVIDERS` has six sites, not two — and one of them sets the shipped label.** Rev 1
   patched only `is_public`. But `provset <= _PUBLIC_PROVIDERS` at
   `train_xshot_occurrence.py:313` / `train_xcross_attempt.py:398` decides the artifact's *label*.
   Trace an `sc_extended`-shaped run: providers are `{skillcorner, idsse}`, no Gradient Sports, so
   `two_candidate` is False, the `else` branch runs, `provset ⊆ _PUBLIC_PROVIDERS` is True — and a
   model trained on **98 restricted matches ships labelled `"public"`**. That is the licensing
   landmine alive in the one place the control did not look. **Verified in code.**

Registered fix: **`_PUBLIC_PROVIDERS` is deleted outright** (all six sites), and the artifact label
is derived from the **visibility composition of the shipped training mask**, not from provider
names:

| label | condition on the ship mask |
|---|---|
| `public` | every row has `visibility == "public"` |
| `sc_extended` | contains restricted SkillCorner rows, no Gradient Sports |
| `full` | contains Gradient Sports rows |

**CI guard (registered):** a behavioural test asserts that a corpus containing any restricted match
can never produce `shipped == "public"` — driven red-first against today's code, which fails it.
- A committed **`PUBLIC_CORPUS` assertion**: the public arm must resolve to exactly the known 17
  (10 SkillCorner ids + 7 IDSSE ids). Any drift fails the run loudly rather than shipping a model
  trained on data we may not redistribute.

This is the one change in this document that is a *compliance* control, not an engineering
preference. It lands whether or not the 98 are ever admitted.

### 3.3 SkillCorner pining path → the native bronze builder

Today the pining path builds SkillCorner frames through the **kloppy gateway**; the lakehouse
builds them through the **native builder** (`tracking/skillcorner.py`, TF-23/ADR-034). The two
paths disagree, and the research path is the poorer one:

| | kloppy gateway (pining today) | native builder (lakehouse) |
|---|---|---|
| `visibility` | hard-coded `None` | **`is_visible` → `visibility`** |
| `ball_z` | discarded | **recovered** |
| clock | kloppy | single-sourced from the events converter's `_PERIOD_START_SECONDS` |
| GK identity | derived | native roster (ADR-034 trust rule) |
| pitch dims | scales, but **not on the events convention** — assumes 103.48 m where the metadata says 104.00, diverging 0.263 m at the goal line (§1.6.1) | **fixed +52.5/+34 — wrong by up to 2.0 m (§1.6)** |

**Decision: route the pining SkillCorner path onto the native builder**, after fixing the pitch
defect (§3.4). The loader shapes raw SkillCorner V3 (either container) into the builder's
`EXPECTED_INPUT_COLUMNS` bronze — a mechanical mapping:

| bronze column | source |
|---|---|
| `match_id`, `frame_rate` | metadata |
| `period`, `frame`, `timestamp` | frame record |
| `player_id`, `x`, `y` | `player_data[]` |
| `is_visible` | `player_data[].is_detected` |
| `team_id`, `is_goalkeeper` | metadata roster |
| `ball_x`, `ball_y`, `ball_z` | `ball_data` |

Gains: `visibility` and `ball_z` on the research path, one SkillCorner truth instead of two, and
no kloppy dependency for SkillCorner. The kloppy gateway remains for Metrica and for external
users, with its `visibility` limitation documented.

**This changes the SkillCorner frames the models train on** — which is exactly why the retrain is
staged (§4.2).

### 3.4 Pitch-dimension normalization: single-source the *affine* transform (rev 3)

**Rev 2 was wrong and would have shipped a serious regression.** It proposed single-sourcing
`spadl/skillcorner.py::_transform_coords` — which (a) is not the name rev 2 used three times
(`_rescale_coordinates` does not exist), and (b) **ends with a clamp**:

```python
x_out = x_out.clip(lower=0.0, upper=105.0)
y_out = y_out.clip(lower=0.0, upper=68.0)
```

For **events** that is harmless: an action's location is on the pitch by construction. For
**tracking** it is destructive, because tracking is *full of legitimately off-pitch positions*.
Measured on match 1886347 (956,076 player rows, 43,458 ball rows):

| rows | clamped | max displacement |
|---|---|---|
| players | 0.71% | 6.33 m |
| **ball** | **11.31%** | **9.00 m** |

And the decisive case: **1,391 ball rows (3.2%) lie beyond the goal line** — true scaled x range
`[−5.76, 114.00]`, clamped to `[0, 105]`. **A ball nine metres behind the goal becomes a ball on the
goal line** — the difference between a goal and a save, erased, with `z` untouched so the ball also
acquires an impossible height-on-the-line. The clamp is *unconditional*, so it would fire on all 108
matches, including the 97 already at 105×68 that this change was supposed to leave untouched.

**Registered fix (rev 3):** split the pure affine map from the clamp.

- `_scale_to_spadl(x, y, L, W)` — the affine part, **no clamp**. This is the single-sourced seam.
- `_transform_coords` = `_scale_to_spadl` **+ clamp**, unchanged, and remains what the events
  converter calls.
- `tracking/skillcorner.py` imports and calls **`_scale_to_spadl` only**.

Tracking therefore inherits the events' *geometry* without inheriting a domain assumption that is
false for tracking. `EXPECTED_INPUT_COLUMNS` gains `pitch_length`, `pitch_width`.

**Consequence for §4.4's within-pitch invariant:** rev 2's clamp would have made that guard
*vacuously true* — a brand-new gate that could never fail, the exact defect class the previous round
removed in B4. With the clamp gone, the invariant is live again and must be able to fire.

**Missing dimensions fail closed (rev 2, reviewer m1).** A bronze lacking `pitch_length` /
`pitch_width` **raises**. It does *not* silently default to 105×68 — that default would
reproduce the exact defect being fixed, and a warning is invisible in a DGX batch log. A caller
that genuinely knows its pitch is standard passes `assume_standard_pitch=True` explicitly.

**Why scale rather than preserve true metres** (unchanged, and the reviewer independently agrees):
SPADL is a normalized 105×68 space, the events side is *already* normalized into it, and every
geometric constant in the library (goal at x=105, penalty-area bounds, the ghost-GK training box
x∈[0,30]) assumes it. Preserving true metres in tracking would put the goal line and the shot
coordinates in different places by up to 2 m. The cost — a ~4% distance distortion on a 101 m pitch
— is the lesser evil, and it is the distortion the events already carry.

### 3.5 What is *not* changed

`silly_kicks/tracking/kloppy.py` keeps its `visibility: None` for providers that genuinely have no
detection flag (Metrica is anonymized; Sportec/DFL and Gradient Sports set `None` natively). The
gateway is not the place to fix SkillCorner — the native builder is.

---

## 4. Registered protocol (locked before any fit sees data)

This section is the reason the document exists. Everything below is fixed **now**; results are
reported against it whatever they say.

### 4.1 Paired-test candidates — three arms, tested in a fixed sequence

The **decision rule** is unchanged from 4.9.0/4.18.0: for each public outer fold *k*, score the
candidate and `public` on the same public held-out fold, take Δ_k = PR-AUC(candidate) −
PR-AUC(public), and **ship the candidate iff Δ_k > 0 in ≥ K−1 of K folds AND mean Δ > 0.** Folds:
`StratifiedGroupKFold` over **public** game_ids, `n_splits = max(2, min(5, #public games))`,
`shuffle=True`, `random_state=42` — unchanged.

What *does* change is **how each candidate is fitted** — see the hyperparameter subsection
immediately below, which is a rev-2 correction and the single most consequential change in this
revision.

#### The hyperparameter asymmetry, and what now decides (rev 2, reviewer M4)

The historical protocol fits **every candidate at the public-optimal hyperparameters** — tuned by
HPO on 17 matches, then applied to a 179-match fit. Larger corpora want more capacity and less
regularisation, so the expansion arms are evaluated **under-tuned**: a systematic handicap pointing
in exactly one direction, *"more data looks worse"*. That protocol is 0-for-2, and rev 1 leaned on
those two losses to justify both the ordering and the accepted cost — pricing a decision off a prior
that the protocol itself may have manufactured.

This is a real defect and it is not one we can discover after the fact, so it is fixed in the
registration:

- **PRIMARY (decides what ships): best-vs-best with NESTED tuning (rev 3).**

  Rev 2 said "each candidate at its own HPO parameters" and stopped there — which, as the reviewer
  showed, **introduces a fresh bias pointing the same way as the one it removed**. `_hpo_once` runs
  *once, outside the outer CV, on the candidate's full data*. So the `public` arm would tune its
  hyperparameters on exactly the 17 matches that constitute the entire evaluation universe — maximal
  selection leakage, optimistic score — while `full` dilutes those 17 across 179 and leaks far less.
  The old shared-params protocol at least made that leakage **common-mode**; rev 2 would have
  converted it into a **differential bias favouring `public`** — and then let it decide what ships.

  Registered fix: **tuning is nested inside the outer CV.** For each outer fold *k*, every candidate
  is tuned on **its own training data with fold *k*'s public games excluded**, then fitted at those
  parameters and scored on fold *k*. No candidate's hyperparameters ever see the fold they are
  scored on. Budget: `n_trials=50`, `seed=42`, identical for every arm and every fold.

  **Cost, stated honestly:** nesting multiplies HPO by K. Estimated ~35–45 DGX-hours for xS + xCross
  (ghost-GK has no HPO and is unaffected), against ~10 h for the un-nested version. The house rule is
  that expense is not a reason to measure the wrong thing; if the owner caps the budget, the cap is
  recorded in the model card as a known bias, not absorbed silently.
- **SECONDARY (reported, never decides): the shared-params data-effect contrast.** The historical
  fixed-hyperparameter comparison is still computed and published for all three arms, because it is
  the scientifically clean *data* contrast and it preserves comparability with the 4.9.0 / 4.18.0
  records.

Both are pre-registered here, together with which one decides. Reporting both is what makes a loss
interpretable: a candidate that loses best-vs-best has lost on merit; a candidate that wins
best-vs-best but loses shared-params was capacity-starved, not data-starved.

**Registered power limitation.** The public arm is 17 matches; with K=5 that is ~3.4 held-out
matches per fold. Every verdict from this protocol — old and new — rests on that. The expansion does
not fix it (the new data is not public), and no amount of owner-tier data can. This is stated in the
model cards and is the honest ceiling on what any of these paired tests can establish.

What changes is the **candidate set**:

| candidate | training data |
|---|---|
| `public` | SkillCorner-public (10) + IDSSE (7) |
| `sc_extended` | public + SkillCorner-private (98) |
| `full` | public + SkillCorner-private (98) + Gradient Sports (64) |

**Rationale for the third arm:** the full candidate has lost twice (4.9.0, 4.18.0), and both times
the added data was a *different product* (Gradient Sports). `sc_extended` isolates "does
same-product owner data help?" from "does Gradient Sports hurt?". Without it, a `full` win or loss
is uninterpretable.

#### The selection procedure: fixed-sequence testing (registered)

Adding a second *shipping* candidate would inflate the chance of shipping a model that only looks
better. The ship rule is a sign-consistency criterion, not a p-value: under a symmetric null
(fold-level deltas as coin flips) a single candidate clears `Δ > 0 in ≥ 4 of 5 folds` with
probability **6/32 ≈ 0.19**. That is the error rate the two-arm protocol has always carried.
Testing two shipping candidates independently would take it to roughly **1 − (1 − 0.19)² ≈ 0.34**,
and no alpha correction applies to a rule of this shape.

The registered procedure is therefore a **fixed sequence** (a standard family-wise-error control:
because the order is fixed in advance, no correction is required and the effective error rate stays
at the single-test level):

1. Test **`sc_extended`** against `public` by the unchanged rule. If it clears → it is the
   **provisional ship**, and step 2 runs.
2. Test **`full`** against `public` by the unchanged rule.
   - **If step 1 failed, stop.** Ship `public`. `full` is still fitted and reported, but it cannot
     ship on this registration.
   - If step 1 cleared, `full` displaces `sc_extended` **only** if it clears against `public` **and**
     the **per-fold `full`-vs-`sc_extended` contrast** clears the same sign-consistency rule
     (`Δ_k > 0` in ≥ K−1 of K folds AND mean Δ > 0). Otherwise `sc_extended` ships. **Ties go to less
     data, not to noise.**

> **Rev 3 note:** rev 2 left a superseded rule inside this numbered procedure — *"ship `full` iff its
> mean Δ exceeds `sc_extended`'s"* — a bare point estimate, contradicting the sign-consistency
> tie-break registered ten lines below it. In a pre-registration, an ambiguity about **which rule
> decides the ship** is exactly the failure the registration exists to prevent. The procedure above
> is now the single statement of the rule; the paragraph below explains it and adds nothing.

**Order justification (a priori, not post hoc):** `sc_extended` is tested first because the
same-product hypothesis is the stronger one — identical SkillCorner V3 schema, identical 10 fps,
clean geometry, no domain shift (§1.2, all measured *before* any fit). That argument stands on its
own evidence and does not depend on the historical record.

**Tie-break, registered (rev 2, reviewer M3.2).** Rev 1 broke a tie on a bare mean-Δ comparison — a
single point estimate with no sign-consistency and no error control, deciding what ships. It is
replaced by the same rule used everywhere else in this document: `full` displaces `sc_extended` only
if the **per-fold `full`-vs-`sc_extended` contrast** clears `Δ_k > 0 in ≥ K−1 of K folds AND mean
Δ > 0`. Otherwise the simpler arm ships. Ties go to less data, not to noise.

**Accepted cost, honestly priced (rev 2, reviewer M3.1).** Rev 1 priced this cost using Gradient
Sports' two prior losses — but the `full` that lost twice was **public + GS (81 matches)**, and the
`full` here is **public + 98 + GS (179)**. The question has changed, so the prior does not transfer;
worse, those losses came from the shared-params protocol that M4 shows is biased against added-data
arms. The cost is therefore **larger than rev 1 claimed**, and is registered as follows:

> If `sc_extended` fails, we ship `public` and `full` cannot ship **on this registration** — even if
> its reported deltas would have cleared. Should that happen (the 98 and GS each helping
> sub-threshold, but jointly clearing), it is recorded as a **finding that triggers a new
> registration**, not a silent loss and not an after-the-fact ship. The fixed sequence buys error
> control at the price of one deferred cycle; that price is now stated rather than hand-waved.

**Correlation note (rev 2, reviewer M3.3).** The ~0.34 two-candidate inflation figure assumes
independence. `full ⊃ sc_extended`: they share the entire public arm and most training data, so
their fold deltas are strongly positively correlated and the true inflation is **below** 0.34. The
fixed sequence is kept anyway — it costs one deferred cycle in a branch we consider unlikely, and it
removes the question entirely — but the problem it solves is smaller than rev 1 implied.

All three arms' per-fold deltas are reported under **both** scoring protocols regardless of which
ships — the `sc_extended`-vs-`full` contrast is the interpretable quantity, not the mere fact that
something won.

### 4.2 Staged retrain (the confound control)

§3.3 changes the SkillCorner frames themselves (visibility, ball_z, pitch scaling). Expanding the
corpus *and* changing the pipeline in one fit would make any paired-test movement unattributable.
Therefore the retrain runs in **two stages**:

- **Stage A — new baseline.** Re-fit **xS, xCross _and ghost-GK_** on the **same 81-match corpus**
  with the new SkillCorner pipeline. This is the apples-to-apples control against the 2026-07-13 run
  (xS paired mean −0.0094; xCross −0.0217) and becomes the reference the expansion is judged against.
- **Stage B — expansion.** Fit the three candidates of §4.1 on the 179-match corpus (and the
  ghost-GK expanded run of §4.3).

**Ghost-GK is included in Stage A by explicit registration (rev 2, reviewer M2).** Rev 1 re-baselined
xS and xCross only — yet ghost-GK trains on SkillCorner frames, and §3.3 changes them. Had §4.3's
baseline been the archived 81-match number while its expanded run used the new pipeline, the
ghost-GK gate would have confounded pipeline with corpus: precisely the error Stage A exists to
prevent, for the one model whose *target* the pipeline change touches most. **Both ghost-GK runs are
fresh, under the new pipeline, with the §4.3 detected-keeper rule applied to both.**

**Registered leakage check on `ball_z` (rev 2, reviewer M2).** After §3.3, SkillCorner gains a real
`ball_z` where it previously had none. Since `z` is an xS/xCross feature, `z`-missingness could
correlate with provider and act as a leakage channel the public-folds evaluation cannot see. The
check is registered *before* the run: report `z`-missingness by provider pre- and post-change, and
each model's split frequency on `z`. Our expectation is that the change **reduces** the channel
(SkillCorner was the only provider missing `z`), but expectation is not evidence and this is
measured either way.

Both stages are reported. If Stage A alone moves the paired verdict, that is a finding about the
pipeline, not the corpus, and it is recorded as such.

### 4.3 Ghost-GK: honest targets, keeper-grouped CV, paired admission (rev 2)

Rev 1's gate was **circular**, and the reviewer is right that this was the most serious defect in
the document. Ghost-GK's *target is the keeper's position*. On SkillCorner **80.4% of keeper
positions are interpolator output** (§1.5). The expansion takes SkillCorner from 10 of 81 matches to
108 of 179 — so the share of ghost-GK's training target that is *inferred rather than observed*
would go from roughly 10% to roughly half. Rev 1 then measured admission on those same extrapolated
targets: **a model that got better at predicting SkillCorner's interpolator, and worse at predicting
real keepers, would have passed.** §4.5 bans the 98 from GKDV measurement using exactly this
argument and then permitted them in ghost-GK training, "controlled by §4.3" — a control that could
not see the failure it existed to prevent.

The fix is already in this PR: §3.3 surfaces `visibility`.

**Registered — honest targets (rev 3: fail-CLOSED on unknown detection).** Ghost-GK trains **only on
frames where the keeper was actually detected**, wherever detection is knowable.

Rev 2 expressed this as "`visibility` truthy, and null means keep" — which is **fail-open on exactly
the field this spec exists to fix**. Gradient Sports and IDSSE legitimately carry no detection flag,
so a null there means *"this provider observes everyone"*. But the kloppy gateway **also** emits null
— and there it means *"we threw the information away"*. Rev 2's rule would have read the second null
as the first: the same failure shape as the licensing landmine, one section later. Registered fix:

```python
_DETECTION_AWARE_PROVIDERS = frozenset({"skillcorner"})   # feed carries is_detected
_FULLY_OBSERVED_PROVIDERS  = frozenset({"gradientsports", "sportec", "idsse"})
```

- provider in `_DETECTION_AWARE_PROVIDERS` and `visibility` is null → **raise** (the pipeline dropped
  the flag; do not silently train on interpolator output).
- provider in `_FULLY_OBSERVED_PROVIDERS` → keep all frames (no flag exists, none is needed).
- **any other provider → raise.** Unknown providers are not assumed observed.

This bites SkillCorner only in practice. The rule applies **identically to the baseline and the
expanded run**, so it is not a confound — and it also cleans the *incumbent* corpus, which already
carries ~10% interpolated keeper targets.

**Registered — CV protocol.** `GroupKFold(5)` grouped by **keeper `player_id`** (not by match): the
model's target is keeper positioning, and half the new cohort's keeper-slots are three Real Madrid
keepers (§1.4), so match-grouped folds would let Courtois appear in both train and test.

**Registered — paired admission rule (replaces the 0.05 m band).** Rev 1's band was not gamed; it
was **never costed**. The same acceptance block tolerates `cross_fold_std < 0.5 m`, so the standard
error on a single run's mean MAE is ~0.22 m and the SE of a difference of two independent runs is
~0.3 m — a 0.05 m threshold on that is a coin flip in both directions. The reviewer's remedy is the
one already used in §4.1, and it is adopted:

**The common evaluation domain (rev 3).** Rev 2 said "the same keeper groups, the same fold ids" —
which **cannot be done as written**, because the two corpora have different keeper populations, so
there is no shared domain to hold fixed. §4.1 gets this right ("the *common* public held-out fold");
§4.3 needs the same construction spelled out:

- The evaluation domain is the **keeper universe of the 81-match baseline**, restricted to keepers
  who do **not** appear in the 98. (They overlap: Courtois is in the WC2022 Gradient Sports corpus
  *and* in 45 of the 98 — so this exclusion is load-bearing, not hypothetical.)
- `GroupKFold(5)` folds are constructed **once** over that domain, with fixed ids and seed, and both
  models are scored on the identical held-out keepers.
- **Asserted, not assumed:** no test-fold keeper appears anywhere in the 98. A violation is a hard
  failure, not a warning — it would mean the expanded model saw its own test keeper in training.
- **Reported before Stage B runs (rev 3):** the **surviving keeper-domain size** — the count of
  keepers left after excluding everyone who appears in the 98, and the per-fold count. This is a
  registered pre-condition, not a post-hoc observation: if the surviving domain is too small to
  support `GroupKFold(5)` with a usable number of keepers per fold, the ghost-GK admission test is
  **underpowered and is reported as such**, not run and interpreted. (Rough expectation ~50–60
  keepers, 10–12 per fold — but an expectation is not a measurement, which is exactly why it is
  registered to be counted first.)

For each fold *k*, Δ_k = MAE_expanded(k) − MAE_baseline(k), computed **on detected-keeper held-out
frames only**.

> **Admit the 98 to ghost-GK training iff Δ_k < 0 in ≥ K−1 of K folds AND mean Δ < 0** — i.e. a
> demonstrated improvement under sign-consistency, not a "no material harm" wash. A wash leaves the
> status quo (81-match corpus) in place.

Plus the unchanged structural gates: per-provider MAE < 3.0 m, cross-fold std < 0.5 m, and the
bundled `default` artifact under 15 MB (measured on the **shipped file set** — see §6/m3).

**The interpolator-tell refusal is RETIRED (rev 5).** Rev 2 registered it, rev 3 kept it, and the
implementation plan proved it is **dead by construction**:

```
improves_detected = clears_rule([-d for d in detected_only_deltas])
if improves_all and not improves_detected:   # only reachable when improves_detected is False
    return False                             # ...which the fall-through returns anyway
return improves_detected
```

The refusal can never change a verdict, because admission **already requires** detected-only
improvement. The test written to cover it passed with the entire block deleted.

**The root cause is mine, not the code's.** The tell was designed in rev 2, when ghost-GK trained on
*all* frames — there, a model could genuinely learn SkillCorner's interpolator from ~50% interpolated
targets, and the tell would catch it. **Rev 3 then changed the training rule to detected-keeper
targets only**, which removes the *mechanism*: the model never sees an interpolated target. I changed
the regime and left its guard standing. A guard for a regime that no longer exists is not a
conservative extra — it is a green light that proves nothing.

Nor is a "divergence" variant (refuse when the model improves *more* on interpolated frames than on
detected ones) the right replacement: interpolated positions are smoother and therefore *easier* to
predict, so almost any model will improve more there. Such a rule would refuse models that are
genuinely better on real keepers. It is retired, not re-specified.

**What replaces it — a reported diagnostic, honestly labelled:** the expanded-vs-baseline deltas on
**all frames** and on **interpolated-only frames** are computed and published alongside the
detected-only deltas that decide. They are informative — they show what the 98 taught the model — and
they **decide nothing**. Saying so is the point.

**And the limitation the detected-only rule actually introduces — REGISTERED (rev 5).** SkillCorner
detects the keeper when the broadcast camera sees him, which correlates with the ball being near him.
So detected-keeper frames are a **selection-biased sample**: the keeper is disproportionately
*engaged* rather than deep, off-ball, or holding a sweeper line — which is precisely the regime GKDV
cares about. Training and evaluating on that sample is internally consistent but externally biased,
and no gate in this document detects it.

Registered: before Stage B is interpreted, **characterise the bias** — report the distributions of
(ball-to-keeper distance, keeper depth) for detected versus undetected keeper frames. This is a
measurement and a stated limitation in the model card, not a gate. It is the honest residual risk of
choosing observed-but-biased targets over unbiased-but-fabricated ones; that trade is still the right
one, and it is now recorded as a trade rather than presented as a solution.

**Reporting.** The shipped artifact's headline metrics stay match-grouped (`StratifiedGroupKFold` by
`game_id`, stratified by provider) so they remain comparable with the 4.14.0 and 2026-07-13 records.
The keeper-grouped, detected-only numbers are published **alongside** them as the admission evidence.
Two scoring schemes, one artifact, both published — never a silent swap of the metric that decides.

### 4.4 Extra time and pitch anomalies

- **Extra time.** 13 of the 98 are UCL knockout ties with `potential_overtime = True`. The native
  builder handles periods 3–5 via the single-sourced `_PERIOD_START_SECONDS`. **Registered check:**
  for each match with period ∈ {3,4} frames, assert the post-rebase clock is monotone within the
  period and bounded by [0, 900] s. A match failing the check contributes **regulation periods
  only**; its extra-time frames are dropped and counted. No guessing.
- **Geometry admission — rebuilt (rev 3, reviewer R1).** Rev 2 registered an exclusion **nothing
  could perform**. The builder's within-pitch invariant *warns and counts*; it never raises
  (`skillcorner.py:200`: "warn + count; NEVER clamp/crash"), its systematic backstop is a
  **deferred** CI rate-gate (`:67`), and its ball tolerance `_TOL_BALL = 30.0 m` is flagged in-code
  as *"provisional — re-calibrate from the measured bronze on the pining corpus"* against a real
  ball excursion of ~9 m. So a broken match produced a warning in a DGX batch log — the very thing
  §3.4 argues is invisible. This was a fourth consecutive can't-fail guard, standing between the
  corpus and the newest, never-validated data.

  **Calibrated on the known-good public 10** (10.0 M rows; calibrating on the 98 would be circular),
  under the correct §3.4 transform:

  | statistic | public-10 worst | catastrophic break (measured) |
  |---|---|---|
  | player rows > 3 m off-pitch | 0.00086 | **0.34139** |
  | ball rows > 10 m off-pitch | 0.00000 | — |
  | player max excursion | 11.01 m | 63.34 m |
  | ball max excursion | **9.00 m** | — |

  **Registered:**
  1. `_TOL_BALL`: **30.0 → 15.0 m** (public-10 max is 9.00 m; 67% headroom; zero public rows exceed
     it). At 30 m the ball tolerance could not fire on any real break.
  2. **Implement the deferred rate-gate**, per match, on the correct transform:
     `player_frac(>3 m) > 0.005` **or** `ball_frac(>10 m) > 0.0005` → **the match is EXCLUDED**, not
     warned. (Margins: 5.8× over the worst clean public match; the catastrophic break exceeds it by
     68×.)
  3. **Mutation-tested three ways:** a catastrophic-break fixture must be excluded; a clean match
     must not be; and the *known limitation* below must be pinned by a test, so no future reader
     assumes coverage the gate does not have.

  **What this gate does NOT do (measured, registered).** It **cannot detect a pitch-dimension
  error.** Transforming a real 105 m match as though it were 101 m — a 2 m goal-line error, the
  worst case in this corpus — moves `player_frac(>3 m)` from 0.00047 to 0.00095, *inside* the clean
  public range. Nor can action↔frame co-location see it: events and tracking both read the same
  metadata, so a wrong `pitch_length` moves them **together**. The only instruments are provenance
  (§1.6.2) and the SkillCorner question. Registering a gate that appears to cover this, and does
  not, would be worse than registering none.

### 4.5 The 98 are never used for GKDV measurement (registered exclusion)

Probes, causal arms, and gates continue to run on **Gradient Sports frames only**. The 98 are
excluded from measurement permanently, for two independent reasons:

1. **Keeper confound:** ~50% of keeper-slots are three keepers, against a clean between-keeper ICC
   of ~0.015–0.026 (the 4.46.0 anchor). A corpus that concentrated is unfit for measuring
   between-keeper variance.
2. **Extrapolated keepers:** 80.4% of SkillCorner keeper positions are inferred (§1.5). A
   GK-substitution statistic computed there would substantially measure the interpolator.

Training use is unaffected by either objection — keeper identity is not the target for xS/xCross,
and ghost-GK's exposure is controlled by §4.3.

---

## 5. Impact and Hyrum surface

| Change | Who is affected | Trigger |
|---|---|---|
| Pitch-dimension scaling (§3.4) | **Lakehouse** — SkillCorner frames on non-105×68 pitches move by up to **2.0 m** | **Re-materialize.** It is a correctness fix; the previous geometry was wrong. |
| SkillCorner → native builder (§3.3) | Research/pining path — geometry moves by **≈ 0.26 m at the goal line** on the four non-105 public matches (kloppy's 103.48 m → the events convention's 104.00 m, §1.6.1), plus `visibility` and `ball_z` appear | New training inputs → Stage A re-baseline (§4.2). Rev 1 wrongly implied the research path was already clean; it is not, and this fixes it. |
| `visibility` surfaced | Pining path: `xt_gk` SkillCorner keeper-origin resolution activates for the ~19.6% of frames where the keeper is actually detected | No lakehouse impact (it already had `visibility`). |
| Corpus taxonomy (§3.2) | Trainers only | None externally; closes a compliance hole. |
| Ghost-GK size gate fix (§6) | Trainer only | None. |

**No change to the shipped library API.** Every item above is either a bronze/loader concern or a
correctness fix behind an existing surface.

---

## 6. Testing

Every gate below must be able to **fail**. Rev 1 shipped two that could not (reviewer B4); they are
replaced, and each replacement names the mutation that kills it.

- **Loader:** both container variants resolve (role-key and suffix); the extension survives into
  the temp name; parquet and csv both parse. A real two-match smoke (one public, one private).
- **Cross-provider regression for the `_download_to_temp` rename (rev 2, reviewer m2):** the rename
  touches **every** provider's cache keys and temp filenames. Rev 1 asserted "safe for IDSSE/GS,
  which sniff magic bytes" without testing it. Registered: an IDSSE and a Gradient Sports match must
  load end-to-end after the rename, and a pre-rename cache entry must be treated as a miss rather
  than silently mis-resolved.
- **Coordinate transform — structural:** `tracking.skillcorner` and `spadl.skillcorner` must produce
  **bit-identical** output *for on-pitch coordinates*, because after §3.4 both call the same
  `_scale_to_spadl`. **Mutation:** re-introduce the fixed `+52.5` offset → fails on a 104 m match.
  (Rev 1's "native and kloppy differ on a 104 m match" is deleted: it passed with *or without* the
  fix — 0.50 m before, 0.26 m after.)
- **Off-pitch positions survive (rev 3 — the clamp regression):** tracking coordinates outside the
  pitch rectangle must pass through **unmodified**. Specifically: a ball beyond the goal line keeps
  `x > 105`; a keeper behind his line keeps `x < 0`; a ball out of play keeps `y > 68`.
  **Mutation:** route tracking through the clamping `_transform_coords` (i.e. rev 2's design) → the
  test fails, because 11.3% of real ball rows and 0.7% of player rows get snapped, with
  displacements up to 9.0 m. This gate exists because rev 2 would have shipped exactly that.
- **The within-pitch invariant must still be able to fire (rev 3):** `tracking/skillcorner.py`'s
  existing gross-off-pitch warning counts real off-pitch rows. Assert it is **non-zero** on a real
  match — under rev 2's clamp it would have been identically zero, i.e. a guard that could never
  fail.
- **Action↔frame co-location (the behavioural gate that actually matters):** on a non-105 m match, a
  same-player event and its linked tracking frame must agree within tolerance. **This fails today**
  (~0.19 m via kloppy, ~0.50 m via the native builder) and passes only after §3.3 + §3.4 — the gate
  and the fix are therefore inseparable.
- **Missing pitch dimensions raise** (§3.4, fail-closed); `assume_standard_pitch=True` is the only
  way through. **Mutation:** restore the silent 105/68 default → the test fails.
- **`visibility` round-trip:** `is_detected` → `frames["visibility"]`; `_tracking_gk_xy_detected`
  selects only detected keepers; a frame whose keeper is extrapolated is **not** used as a keeper
  origin, and is **not** used as a ghost-GK training target (§4.3).
- **Corpus taxonomy (compliance gate, red-first):** a corpus containing any restricted match can
  never yield `shipped == "public"` — asserted against the *label* path, not just the arm split, and
  driven red-first against today's code (which fails it, §3.2). Missing `visibility` ⇒ restricted.
  The `PUBLIC_CORPUS` assertion catches drift in the known 17.
- **Cache invalidation:** a `_feature_cache/` written under the old schema (no `cache_meta.json`, no
  `visibility.npy`) must be treated as a **miss**. **Mutation:** restore the bare
  `features.parquet` existence predicate → the test fails.
- **Registered-rule unit tests:** the fixed sequence, the ship rule, the `full`-vs-`sc_extended`
  tie-break, and the ghost-GK paired admission rule are pure functions with table tests, including
  the "both qualify", "none qualify", and "expanded wins all-frames but loses detected-only"
  (interpolator-tell refusal) cells.
- **Ghost-GK keeper grouping:** a fixture where one keeper appears in several matches must never
  land in both train and test folds.
- **Size gate measures the shipped set (rev 2, reviewer m3):** the fixed `artifact_bytes` must sum
  exactly the files that are copied into `_ghost_gk_weights/default/` — not a directory walk that
  includes `_feature_cache/` (today's bug), and not one that silently omits a file that ships. With
  ~2.4% headroom against the 15 MB cap this is not academic.

---

## 7. Risks and open questions

1. **The pitch fix is a lakehouse Hyrum event.** It is a correctness fix, but it moves production
   geometry. It could be split into its own release to decouple it from the corpus work. **My
   recommendation: keep it here** — it is a prerequisite for §3.3, and shipping the corpus work on
   top of known-wrong geometry would be worse. Flagged for review.
2. **Three arms could have raised the noise-win risk** — handled by the fixed-sequence procedure
   (§4.1), which holds the effective error rate at the historical single-test level (~0.19 under a
   symmetric null) instead of the ~0.34 that two independent shipping candidates would carry. The
   accepted cost is registered: a `full` that would have cleared cannot ship if `sc_extended` fails
   first.
3. **The public arm remains tiny (17 matches).** Every paired verdict — old and new — is decided on
   ~3.4 held-out matches per fold. The expansion does not fix this, because the new data is not
   public. This is an honest, structural limit of the paired test and is stated in the model cards.
4. **Stage A could move the baseline.** If the new SkillCorner pipeline alone changes the paired
   verdict, the 2026-07-13 numbers are superseded before the expansion is even tested. That is a
   finding, not a failure — but it means PR-2's weights depend on Stage A completing.
5. **Cost.** ~1.05 GB of new downloads; Stage A + Stage B + the two ghost-GK protocol runs are
   roughly **45–60 DGX-hours** in total: ~35–45 h for the nested-tuning paired tests on xS + xCross
   (§4.1 — this figure supersedes rev 1's 30–40 h, which predated the nesting decision), plus Stage A
   (~10 h) and the two ghost-GK protocol runs. The owner approves the budget off **this** line.

---

## 8. Sequencing

1. **PR-A (this spec):** loader + taxonomy + native SkillCorner route + pitch fix + visibility +
   the ghost-GK size-gate fix + the `--match-ids-json` corpus pin (already written during the
   TF-19 owner runs, currently uncommitted). Code and tests only — **no weights**.
2. **Owner runs (registered, §4):** Stage A → Stage B → the ghost-GK keeper-CV pair.
3. **PR-B (the former PR-2):** bundle the final weights, turn on fail-closed chirality enforcement,
   record every verdict from the TF-19 re-gate and from §4 here. **One weights release, one
   lakehouse re-materialization.**
4. **PR-3 (`gkdv/`)** proceeds in parallel with (2) — it is a code build with no weight dependency.

---

## 9. Revision log

**Rev 5 (2026-07-14)** — after the **plan** review. Writing the implementation exposed a defect in
the registration itself.

| Item | Disposition |
|---|---|
| **The interpolator-tell refusal is dead code** | **Accepted; the rule is RETIRED, not repaired.** It can never change a verdict — admission already requires detected-only improvement, so the refusal branch is only reachable when the fall-through returns `False` anyway. **Root cause (mine):** it was designed in rev 2, when ghost-GK trained on *all* frames and a model could genuinely learn the interpolator; **rev 3 changed the training rule to detected-keeper targets only**, removing the mechanism, and I left the guard standing. A "divergence" replacement is also rejected — interpolated positions are *smoother*, so almost any model improves more on them, and such a rule would refuse models that are genuinely better on real keepers. Replaced by an explicitly non-deciding **diagnostic** (all-frames and interpolated-only deltas, published beside the deltas that decide). |
| **What the detected-only rule actually risks** | **Newly registered.** SkillCorner detects the keeper when the camera sees him — which correlates with the ball being near him. Detected frames are therefore a **selection-biased** sample: the keeper is disproportionately *engaged* rather than deep or off-ball, which is the regime GKDV cares about. Registered as a **measurement + stated limitation** (characterise ball-to-keeper distance and keeper depth, detected vs undetected, before Stage B is interpreted), not as a gate. Choosing observed-but-biased targets over unbiased-but-fabricated ones remains right; it is now recorded as a trade rather than presented as a solution. |

**Rev 4 (2026-07-14)** — after the third review round (approval subject to four corrections).

| Item | Disposition |
|---|---|
| **R1 · §4.4 registers an exclusion that nothing can perform** | **Accepted — the one piece of real work, and the reviewer is right that it was a fourth consecutive can't-fail guard, protecting the newest data.** Verified in code: the invariant warns and counts but never raises (`:200`), its systematic backstop is *deferred* (`:67`), and `_TOL_BALL = 30.0 m` — flagged in-code as provisional pending exactly this calibration — sits 3× above the largest real ball excursion. **Calibrated on the known-good public 10** (10.0 M rows; calibrating on the 98 would be circular): ball max 9.00 m, player max 11.01 m, worst clean `player_frac(>3 m)` = 0.00086. Registered: `_TOL_BALL` → **15.0 m**; the deferred **rate-gate is implemented** (`player_frac(>3 m) > 0.005` or `ball_frac(>10 m) > 0.0005` → **exclude the match**); mutation-tested three ways. **And its power was measured, not assumed:** a catastrophic break trips it at 0.34 (400× the clean baseline), but a **pitch-dimension error is invisible to it** (0.00095 vs a clean 0.00086 — inside the public range), and action↔frame co-location cannot see one either, since events and tracking read the same metadata and move together. That limitation is now stated and **pinned by a test**, rather than papered over by a gate that appears to cover it. |
| **§4.1's numbered procedure still carried the rev-1 tie-break** | **Accepted.** A superseded bare-mean-Δ rule was sitting in the procedure an implementer follows, contradicting the sign-consistency tie-break ten lines below. In a pre-registration, ambiguity about *which rule decides the ship* is precisely the failure the registration exists to prevent. The procedure is now the single statement of the rule. |
| **§1.6.1's caveat contradicted the table above it** | **Accepted.** The mirror-invariant measurement is complete; the paragraph no longer says it "is being re-measured". |
| **§7.5's budget predates the nested-tuning decision** | **Accepted.** 30–40 h → **45–60 h**, itemised. The owner approves off that line. |
| *(non-blocking)* the 103.48-vs-103.71 gap is explainable — kloppy's map is **not affine** (~0.14 m non-affine residual), so "effective pitch length" is a fit artefact | **Adopted into §1.6.2.** It strengthens the case for provenance over reverse-engineering: nobody has characterised what kloppy's SkillCorner transform actually does. It goes into the SkillCorner email as a question. |
| *(non-blocking)* report the surviving ghost-GK keeper-domain size before Stage B | **Registered as a pre-condition** in §4.3 — counted and reported *before* the run, with an explicit underpowered verdict if the domain is too small. An expectation (~50–60 keepers) is not a measurement. |

**Rev 3 (2026-07-14)** — after the second review round
(`…-REVIEW-2.md`). Three of rev 2's fixes were themselves wrong; the worst was **created by** a
rev-2 fix. All accepted.

| Item | Disposition |
|---|---|
| **§3.4 single-sources a function that CLAMPS** — and tracking is not events | **Accepted; this was the most serious defect in the document, and rev 2 introduced it.** Independently reproduced on match 1886347: the clamp would snap **11.31% of ball rows** and 0.71% of player rows, displacing by up to **9.00 m** — and **1,391 ball rows (3.2%) lie beyond the goal line**, so *a ball nine metres behind the goal becomes a ball on the goal line*: goal and save, made indistinguishable. It is unconditional, so it fires on all 108 matches, including the 97 this change was meant to leave untouched. Fixed by splitting the pure affine map (`_scale_to_spadl`, single-sourced into tracking) from the clamp (events only). Two new gates in §6, each naming the mutation that kills it — including one asserting the within-pitch invariant is **non-vacuous**, which rev 2's clamp would have made vacuously true. |
| **§4.3's detected-only rule is fail-OPEN on null `visibility`** | **Accepted.** The null that means "this provider observes everyone" (GS/IDSSE) is indistinguishable from the null that means "the kloppy gateway threw the flag away" — the same failure shape as the licensing landmine, one section later. Replaced with an explicit provider allowlist that **raises** on a detection-aware provider with null visibility, and on any unknown provider. |
| **Best-vs-best carries a fresh bias pointing the same way as the one it fixed** | **Accepted.** `_hpo_once` tunes outside the outer CV on the candidate's full data, so `public` would tune on exactly the 17 matches that *are* the evaluation universe — differential leakage favouring `public`, deciding what ships. Fixed by **nesting tuning inside the outer CV** (each candidate tuned per fold with that fold's public games excluded). Cost stated: ~35–45 DGX-h, up from ~10. |
| **§1.6.1 deletes the landmark check on a fallacy** | **Accepted.** "They call the same function so they cannot disagree" proves *consistency*, not *correctness*. §1.6.2 now states the real basis for choosing 104 m — **provenance** (SkillCorner's declared `pitch_length`, which our events have always used) — records that kloppy's unexplained shorter figure would imply *our events have been wrong all along*, and carries the action item: **ask SkillCorner**. |
| **The rev-2 kloppy divergence table does not reproduce** | **Accepted; both prior figures withdrawn.** Rev 2's numbers came from a scale factor inherited from a retracted review pass; my own re-measurement was contaminated by *orientation* (±105 m artefacts). Re-measured mirror-invariantly (n=956,076, 1:1 join): kloppy assumes **103.48 m** vs the declared 104.00, diverging **0.263 m** at the goal line, **centre spot preserved** (52.492). The reviewer's independent figure (103.71 / ~0.15 m) agrees on structure and on the centre spot; the residual gap between our two numbers is unexplained, recorded, and decision-irrelevant. |
| **§4.3's "same keeper groups, same fold ids" cannot be done as written** | **Accepted.** The two corpora have different keeper populations. §4.3 now specifies the common domain explicitly (the baseline's keeper universe minus any keeper appearing in the 98 — Courtois is in *both*, so the exclusion is load-bearing) plus a hard assertion that no test-fold keeper appears in the 98. |
| **Dueling text: §3.3's table still said kloppy "scaled correctly"** | **Fixed.** Swept, along with three references to a function name that does not exist (`_rescale_coordinates` → `_transform_coords`). |

**Rev 2 (2026-07-14)** — after the first review round

**Rev 2 (2026-07-14)** — after external review
(`2026-07-14-skillcorner-corpus-and-visibility-REVIEW.md`). Every blocker and major was accepted;
two spec claims were **retracted**, and one reviewer premise was corrected by measurement.

| Item | Disposition |
|---|---|
| **B1** ghost-GK gate is circular (measures MAE on interpolated targets) | **Accepted, and fixed deeper than proposed.** §4.3 now restricts ghost-GK *training targets* to detected keepers — not merely the admission metric — plus a registered interpolator-tell refusal. |
| **B2** the 0.05 m band was never costed | **Accepted.** Replaced by a paired per-fold ΔMAE under the same sign-consistency rule used in §4.1 (same folds, groups, seed). |
| **B3** fail-closed defeated by the cache predicate; `_PUBLIC_PROVIDERS` has six sites | **Accepted; verified in code.** `provset <= _PUBLIC_PROVIDERS` at `:313`/`:398` labels an `sc_extended`-shaped run `"public"`. `_PUBLIC_PROVIDERS` is deleted; the label derives from the ship mask's visibility composition; the cache gains a schema+fingerprint predicate; a red-first CI guard is registered. |
| **B4** the parity test cannot fail | **Accepted.** Deleted and replaced by a structural single-source identity plus a behavioural action↔frame co-location gate that fails today. |
| **M1** "one of them is wrong, and the spec asserts kloppy is correct without testing it" | **Accepted — and settled the other way.** §3.4's formula is *algebraically identical* to this library's own SkillCorner **events** converter; kloppy is the outlier (≤0.29 m divergence, centre spot at 52.65). Rev 1's "kloppy is correct" is retracted (§1.6.1). §3.4 now **single-sources the events transform** rather than re-deriving it, which makes the invariant structural. The reviewer's landmark check is therefore unnecessary — a stronger invariant replaces it. |
| **M2** ghost-GK has no pipeline-matched baseline; `ball_z` leakage | **Accepted.** Ghost-GK joins Stage A; the `ball_z` provider-correlation check is registered before the run. |
| **M3** cost priced off a candidate that no longer exists; tie-break is the weakest rule; 0.34 assumes independence | **All accepted.** Cost re-priced and the deferred-cycle consequence stated; tie-break moved to sign-consistency; the correlation caveat recorded. |
| **M4** the protocol is structurally biased against the arms it tests | **Accepted, and it changes what decides.** Best-vs-best (each candidate at its own HPO parameters) becomes the **primary** ship rule; the shared-params contrast is retained as a reported scientific quantity. The power ceiling (~3.4 held-out matches per fold) is registered. |
| **m1** §3.4's default was fail-open | **Accepted.** Missing pitch dimensions now **raise**; `assume_standard_pitch=True` is the explicit opt-in. |
| **m2** no regression test for IDSSE/GS after the temp-name rename | **Accepted.** Registered in §6. |
| **m3** the size gate excludes a file that ships | **Accepted.** The gate must sum the shipped file set exactly. |
| **m4** `PUBLIC_CORPUS` couples us to the pining mirror | **Acknowledged; correct behaviour.** It fires if pining's public listing changes — which is the point. |

**Rev 1 (2026-07-14)** — initial draft.

---

## References

- TF-19 cycle: `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`, ADR-037.
- Native SkillCorner builder: TF-23, ADR-034.
- Coordinate-system pin on the kloppy tracking gateway: ADR-031.
- Trained-model lifecycle (variant selection, paired test): ADR-011.
- SkillCorner keeper-origin distrust: ADR-024 / PR-S104.
