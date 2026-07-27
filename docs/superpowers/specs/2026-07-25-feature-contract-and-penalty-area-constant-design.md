# Trained-model feature contract + canonical penalty-area constant — design

**Date:** 2026-07-25
**Target:** silly-kicks (next-free at commit-prep; provisional `4.64.0 / PR-S135 / ADR-050`) — do not pre-claim
**Status:** design **rev 3 — SPEC REVIEW CLOSED, ready to plan.** Round 1 (3 blockers / 6 major / 5 minor)
and round 2 (0 blockers / 3 major / 4 minor) both applied in full. Every anchor in both rounds was
re-verified against source before adoption; both round-1 blockers were reproduced independently. Round 2's
three majors were each direct consequences of round-1's own fixes — the agreed stopping signal — so per the
reviewer's disposition the remaining review budget goes to the **plan**, not another spec pass.
**Source:** TF-51 v2 (ADR-049) re-flagged the penalty-area drift; ADR-011 (trained-model lifecycle); ADR-040
(`_chirality` fail-closed precedent); ADR-038 (cache fingerprint deferral)
**Scope class:** library integrity machinery — no new aggregator, no `*_xfns`, no VAEP consumer →
**C4-free (count stays 32), no retrain trigger**

---

## 1. Problem

Three modules hard-code the penalty-area half-width at **two different values**, and the divergent one
feeds a trained model.

| Site | Value | Verified |
|---|---|---|
| `tracking/defensive_credit/_params.py:20` | `_BOX_HALF_WIDTH_M = 20.16` | yes |
| `tracking/_xcross_attempt.py:70` | `_BOX_HALF_WIDTH_M = 20.16` (used `:208`) | yes |
| `tracking/_ghost_gk.py:235-236` | bounds from `40.3` ⇒ half-width **20.15** | yes |

**20.16 is the Law value** (FIFA penalty area 40.32 m wide); `40.3` is a rounding. `_params.py:14-18`
already records the discrepancy in prose and flags a canonical constant as a follow-up.

**The divergent value is a trained-model input.** `_ghost_gk.py:608` uses those bounds to compute
`attackers_in_box`, and `attackers_in_box` is one of the **26 names in `GHOST_GK_FEATURE_NAMES`**
(verified by import). The bundled ghost weights were fit at `40.3` (`corpus_provenance`: providers
`gradientsports`/`skillcorner`/`sportec`, 179 games). So unifying the constant naively changes a served
feature against weights trained under the old one — **train/serve skew inside our own artifact**, the class
ADR-040's chirality work exists to prevent. Downstream re-materialization does not fix this; it propagates it.

**Measured, not assumed** (real GS match 10502, via the pining loader). A row can only move the feature if it
is **both** in the 1 cm `y`-band (`[13.84, 13.85)` or `(54.15, 54.16]`) **and** within the 16.5 m box depth of
the goal it attacks:

| quantity | value |
|---|---|
| outfield player rows (finite coords) | 3,519,597 |
| rows in the 1 cm y-band | 844 |
| rows **also** within box depth — the set that can actually flip | **70** |
| distinct frames that can flip `attackers_in_box` | **70 of 175,969 = 0.0398 %** |

An earlier draft quoted only the y-band figure (0.0218 % of rows / 0.47 % of frames), which is an upper bound
on *rows*, not the feature delta (review R1 m3). **0.0398 % of frames is the number that sizes D3** and the one
ADR-050 should carry. Small, but non-zero and real — and the point of D3 is that "small" is not "zero".

### 1.1 Why the existing guards do not cover it

- `GEOMETRY_VERSION = "goal-relative-1"` (`tracking/_geometry.py:23`) is documented as *"Bump when the
  goal-relative **transform's** NUMERIC output changes"*. A feature-extractor constant is not the transform,
  so bumping it for a box change would assert something false.
- xS (`_xshot_occurrence.py:461`, `:515-518`) and xCross (`_xcross_attempt.py:498`, `:545-548`) record and
  check `geometry_version`, but at identical pitch dims they **warn**, not raise.
- **Ghost-GK records no `geometry_version` at all** (verified on the bundled `default/metadata.json` and on
  the HF artifact). The one model consuming the divergent constant has no geometry contract whatsoever.
- The chirality fingerprint probes y-asymmetry; a 1 cm box change does not move it.

⇒ Changing only the number fixes one centimetre and leaves intact the mechanism that let it skew silently.
This spec closes the class.

---

## 2. Resolved decisions (owner, 2026-07-25)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Behavioural fingerprint gates; declared constants explain** | The repo's consistent lesson (ADR-043's deleted lint, auto-enumerating gates, chirality) is that behavioural gates survive where declaration-by-discipline does not. A declared-only contract would have missed this bug had nobody registered the constant. Constants ride along so a failure says *why*, not just *that*. |
| D2 | **Missing contract → WARN; mismatch → RAISE.** All three trained models | Additive: nothing that works today breaks. Chirality raised on *missing* because every pre-PR-2 artifact was known-bad; these artifacts are merely undeclared, so refusing them would be self-inflicted breakage — including HF `from_hub` consumers. Full teeth apply to every artifact saved from here on. |
| D3 | **`_ghost_gk`'s constant flips in the RE-FIT commit, not this cycle** | We are shipping a mechanism whose purpose is "never serve features the weights weren't fit under". Shipping a known skew in the same release would be that mechanism's first exception. Zero skew window instead of a small one. NOT the rejected "permanent documented divergence" — same destination, sequenced. |

**Explicitly rejected:** permanently pinning `_ghost_gk` at 20.15 with a documented divergence. It
institutionalises a wrong physical constant, leaves the library with two penalty areas, and relies on a prose
contract — which this repo has watched fail over six releases before.

---

## 3. The feature contract

New **private** `silly_kicks/tracking/_feature_contract.py`, a deliberate sibling of `_chirality.py` (same
shape, same idiom, same failure taxonomy).

```
contract_probe_frame() -> pd.DataFrame
feature_contract(extract_fn, *, constants: dict) -> dict
    # {"version": ..., "probe_sha256": ..., "fingerprint": [...], "constants": {...}}
verify_feature_contract(recomputed, stored, *, legacy_override, model_name, error_cls=None) -> None
```

**The `verify_*` signature is byte-for-byte the sibling's** (`_chirality.py:134-141`:
`verify_chirality(recomputed, stored, *, legacy_override, model_name, error_cls=None)`) — **current
first, stored second**. An earlier draft of this spec inverted them and dropped `model_name`; since both
arguments are `dict`, the swap is not a type error — it makes the `is None` branch test the wrong side, so a
*missing* contract would stop warning and start raising, silently inverting D2. "Same shape, same idiom" is
therefore literal, not approximate. `model_name` is carried because three models share this module and every
chirality error message names the failing one. (Review R1 M2.)

- **`feature_contract`** runs the extractor on the probe and returns the feature vector plus the declared
  constants. Called in each model's `save()`; stored under a `feature_contract` key in `metadata.json`.
  The three extractors take **different signatures** (`extract_xshot_features(grp, *, gk_team_id, goal_x)`
  is not shaped like `extract_ghost_gk_features`), so `extract_fn` is a **zero-argument closure** each model
  supplies, binding its own extractor to the probe — the contract module stays extractor-agnostic and does
  not need to know any of the three call forms.
- **`verify_feature_contract`** re-runs the extractor at `load()` and compares. **Missing → `warnings.warn`
  once, then proceed** (D2). **Mismatch → raise**, using the caller-supplied `error_cls` so each model keeps
  its own `IntegrityError` taxonomy — the same parameterisation `verify_chirality` uses, so a consumer
  catching `_ghost_gk.IntegrityError` catches this too.
- **`legacy_override=True`** escapes with a warning, mirroring the chirality escape hatch.

**Tolerance is chosen here, NOT inherited (review R1 B2i).** An earlier draft said "reuses the chirality
constants". That is wrong: `rtol=1e-2` was sized for a gross sign flip on a *probability* output across an
aarch64-save / x86-load (`_chirality.py:127-129`). A feature vector spans metres, counts and radians —
measured `max |feature| = 32.5`, and `nearest_attacker_to_goal_x ≈ 17` gives a tolerance of
`1e-3 + 1e-2 × 17 ≈ 0.17 m`. **The change this entire spec exists for is 0.01 m.** Had the box constant moved
a *distance* feature rather than an integer *count*, the inherited tolerance would not have seen it. So:
**`rtol=0`, `atol` set from a MEASURED DGX-vs-x86 feature-vector delta**, with `atol=1e-6` as the defensible
starting value pending that measurement (features are deterministic arithmetic; the Qhull-based
`defending_team_compactness` is the only plausibly wobbly one). Inheriting an unmeasured tolerance is the same
shape as the "≤1 ulp, unmeasured" claim a prior review round caught.

**That measurement gets an owner and a trigger, or it repeats the trap one level down (review R2 N3).**
A "pending" obligation with no trigger and no failing test is exactly the shape this paragraph exists to
avoid. Two commitments make it concrete:

1. **The §4 ghost re-save runs on x86** — the same platform the lakehouse loads on. That makes `atol=1e-6`
   trivially safe for the *only* artifact that carries a fingerprint this cycle, because no cross-platform
   comparison occurs. (Had it been produced on the DGX, every x86 drain worker would be comparing
   cross-platform against an unvalidated tolerance from day one.)
2. **The measurement is a prerequisite of the first DGX-produced fingerprint** — whichever comes first, the
   xS/xCross next train or the ghost re-fit — carried as a TODO row. Same move §4 already makes for the
   constant flip: a named trigger rather than a floating intention.

**NaN policy is stated, NOT inherited (review R1 B1) — this is a ship-blocker if skipped.** Chirality
fingerprints *outputs* and its own docstring records why NaNs are harmless there: *"Sparse-frame NaN features
are fine --- the booster treats NaN as missing and the served (x, y) is deterministic."* A **feature** contract
destroys that property. Measured on the current probe: 26 features, **3 structurally NaN**
(`defending_team_compactness` needs ≥3 defenders for a ConvexHull; `defensive_line_speed` and
`defending_centroid_vx` need the `prev_*` kwargs, which are arguments, not frame rows), and
`np.allclose(v, v, atol=1e-3, rtol=1e-2)` returns **False for a vector against itself** — i.e. `save()` →
`load()` would raise on a byte-identical artifact. Therefore:

1. compare with **`equal_nan=True`** (this still catches NaN→finite drift, which is the direction that matters);
2. the probe closure **passes the `prev_*` kwargs and seats ≥3 defenders**, so any surviving NaN is deliberate;
3. §6 asserts the recorded fingerprint contains **zero** NaNs — a NaN feature is a feature the contract cannot
   gate at all.

Applies to all three trained models — `extract_ghost_gk_features`, `extract_xshot_features`,
`extract_xcross_features`. xCross carries its own `_BOX_HALF_WIDTH_M`, so it is equally exposed to the next
box change; covering only ghost-GK would leave the class half-closed.

### 3.1 The probe must have teeth — the load-bearing requirement

The existing `canonical_probe_frame` (`_chirality.py:21`) is **6 rows / 5 players / teams A,B** (verified).
It is not reusable: if no attacker sits inside the penalty box near its edge, `attackers_in_box` is `0` under
both 20.15 and 20.16, the fingerprint is byte-identical either way, and **the guard is blind to the exact bug
it exists for**.

`contract_probe_frame` is therefore purpose-built to a stated rule:

> **Every constant the contract declares must be load-bearing on at least one feature of the probe.**

Concretely it seats an attacker **in the 1 cm y-band AND inside the 16.5 m box depth** — both conditions, not
just the band. §1's measurement is exactly why: 844 rows sat in the band but only **70** were also within
depth, so a probe player placed in the band at the wrong `x` would move nothing and the meta-test would report
the constant as uncovered. It also seats **≥3 defenders** (so the ConvexHull-based
`defending_team_compactness` is finite) and a keeper, and the closure passes the `prev_*` kwargs, so the three
extractors produce non-degenerate, NaN-free values per the §3 NaN policy.

This is enforced, not asserted: a **meta-test mutates each declared constant in turn and requires
`verify_feature_contract` to RAISE**. Note the wording — an earlier draft said "requires the fingerprint to
move", which a 1e-9 shift satisfies while remaining invisible to the gate (review R1 B2ii). The assertion must
be that the *gate fires*, not that a number changed. A constant the probe cannot see is not covered, and the
meta-test says so. (This repo has shipped vacuous fixtures before — the y-axis-vacuous team-shape test and the
`ball_speed=0` xCross directional fixture — so "the probe has teeth" is a gate, not a comment.)

**Completeness is bounded by the PROBE, not by the declared dict — and that bound is closed by enumeration
(review R1 M1).** The review argued the fingerprint's completeness equals the hand-maintained `constants`
dict. That specific mechanism is not right: the fingerprint is taken over the **feature vector**, so an
*undeclared* constant that moves any probe-exercised feature still fires the gate — the dict is not in the
**fingerprint** gate's path (it *is* separately compared, and can itself raise — see §3.2; review R2 N5).
But the conclusion holds by a different route: because §3.1 builds the probe to be sensitive *to the declared
constants*, an undeclared one may simply never be exercised. The real bound is probe sensitivity.

So the durable fix is the repo's own idiom, and it is the one ADR-043 actually set: that decision did not
replace declaration with behaviour, it replaced an **incomplete-by-heuristic lint** with a registry **complete
by ENUMERATION**. Accordingly this cycle adds an **auto-enumeration gate**: an AST pass over `_ghost_gk.py`,
`_xshot_occurrence.py`, `_xcross_attempt.py` and `defensive_credit/_params.py` asserting that every
module-level geometry constant is either declared in some model's `constants` dict or carries an explicit
exemption with a reason. That converts §3.1's rule from self-referential ("the probe sees what you already
told it about") into complete, and it is the durable output of this cycle — more than the constant fix itself.

### 3.2 Probe identity, and why this probe's lifecycle differs from chirality's (review R1 M3)

The contract dict carries **`version` + `probe_sha256`**, mirroring chirality (`_CHIRALITY_VERSION` at
`_chirality.py:18`, `frame_sha256` at `:122`). Without them, a probe change reports as a constant/extractor
skew and sends the reader hunting the wrong thing.

But the two probes have **opposite lifecycles**, and that forces one deliberate asymmetry. Chirality's probe is
frozen. This one is *specified to grow*: §3.1's rule ("every declared constant must be load-bearing on the
probe") guarantees the probe is extended every time a constant is added. Under chirality's semantics — probe
mismatch raises (`:174-176`) — every future constant addition would brick all three models' artifacts until
each is re-saved.

So on a **probe mismatch this contract WARNS AND SKIPS THE FINGERPRINT COMPARISON ONLY** — with a distinct
message ("cannot verify the fingerprint — probe changed; re-save to regain teeth") — rather than raising. That
is consistent with D2's additive principle, and the asymmetry with chirality is justified by the differing
probe lifecycle rather than by convenience. The rejected alternative — raise, for strict parity — costs a
forced re-save of all three artifacts on every constant addition.

**Declared constants are also compared, as a cheap second net, and the comparison SURVIVES a probe change
(review R2 N1).** The fingerprint alone misses a sub-probe-resolution change (20.16 → 20.161 moves no
feature). Comparing the **intersection** of declared keys closes it at no cost: new keys ignored (additive),
changed values raise, removed keys warn.

Critically, **constants are probe-independent**, so a probe change is no reason to stop comparing them. An
earlier draft's "warn and skip" swallowed the constants net along with the fingerprint, which opened a hole
the two halves otherwise close for each other: change the probe *and* a constant in one commit and nothing
would raise. Under chirality parity that combination raises, and it must here too. This is also the strongest
argument for keeping the constants net at all — it is the only part of the contract that survives probe
evolution.

### 3.3 The missing-contract warning is a named category (review R1 M4)

D2's warn path has a known weakness recorded in this repo already — `_xshot_occurrence.py:504-506`, verbatim:
*"a `warnings.warn` is invisible in a swallowed-stdout Spark/batch serve, so we never rely on it for the case
that actually skews."* Since §4 establishes that **no shipped artifact carries a fingerprint yet**, 100 % of
loads take that path, so a bare warning would make the mechanism a silent no-op in the lakehouse drain.

Therefore the warning is a **named `MissingFeatureContractWarning(UserWarning)` in
`tracking/_warnings.py`**, re-exported via `tracking.__all__` per ADR-041's convention (*"one module, every
category re-exported ... so a consumer's `filterwarnings` line has a single stable import path"*). Chirality
predates ADR-041; a new module should not.

**Register it in every place the existing three categories appear (review R2 N6)** — `_warnings.__all__`,
`tracking/__init__.py`'s `__all__`, and the public-surface test/doc that covers them. This is the classic
add-to-one-list-forget-the-others failure, and the whole value of the named category is that a consumer's
`filterwarnings` import path is stable. This also *fixes* the invisibility rather than merely noting it: a
consumer can set `filterwarnings("error", category=MissingFeatureContractWarning)` and get D2's additive
semantics for external/HF consumers **and** fail-closed semantics in their own serve.

**Load cost:** the contract re-runs feature extraction on top of chirality's extract-and-predict, on a larger
probe. Negligible per load, but it is paid per worker in a fan-out drain; recorded so it is a known quantity
rather than a surprise.

### 3.4 Relationship to `geometry_version`

Untouched. The fingerprint supersedes it behaviourally, but churning a working guard is not this cycle's job.
Retiring or folding it in is a later question, recorded here so the next reader does not assume an oversight.

**Ghost additionally gains the fail-closed pitch-dimension guard it lacks (review R1 m1).** xS
(`_xshot_occurrence.py:507-513`) and xCross raise on a `pitch_length`/`pitch_width` mismatch; ghost records
neither field, so it has no fail-closed geometry guard at all. That is ~6 lines in ghost's `save()`/`load()`,
is orthogonal to the contract and to D3, and is stronger than anything §3 provides for the dimension case.

---

## 4. The canonical constant — partial by design (D3)

Add to `spadl/config.py`:

```
penalty_area_half_width = 20.16   # FIFA Laws: penalty area 40.32 m wide
penalty_area_depth      = 16.5
```

Migrate the two consumers that **already hold 20.16** — `_xcross_attempt` and `defensive_credit/_params` —
onto them. Byte-identical; guarded by an explicit value-equality test.

**Unify the PREDICATE, not just the scalar (review R1 M6).** Two hand-rolled membership tests exist and they
already disagree on the boundary:

- `_xcross_attempt.py:209` — `(gr_x <= _BOX_DEPTH_M) & (abs(y - GOAL_Y) <= _BOX_HALF_WIDTH_M)`
- `_ghost_gk.py:608` — `(atk_xs < _PENALTY_AREA_X) & (atk_ys >= _MIN) & (atk_ys <= _MAX)`

Non-strict vs **strict** on `x`: a player exactly on the 16.5 m line is in-box for xCross and out-of-box for
ghost. Numerically negligible, contractually not — after this section lands the docs assert a *canonical*
penalty area while two membership predicates remain. By §1's own logic ("changing only the number fixes one
centimetre and leaves the mechanism intact"), unifying only the scalars would repeat the error one level up.

**But the helper MUST NOT take a single `goal_x` (review R2 N2) — the three sites differ on FRAME, not just
on strictness:**

| Site | Frame | Reference goal | x test |
|---|---|---|---|
| `defensive_credit/_params.py:78-81` | **absolute** (action-LTR) | **attacked**, x=105 | `x >= 105 − 16.5` |
| `_xcross_attempt.py:209` | **goal-relative** | **attacked**, gr_x=0 | `gr_x <= 16.5` |
| `_ghost_gk.py:234/608` | **goal-relative** | **defended**, gr_x=0 | `atk_xs < 16.5` |

And `goal_x` already has a fixed meaning in this repo — `_geometry._flip(goal_x) → goal_x > 50.0`, i.e. the
*absolute* x of a reference goal — so a `goal_x` parameter would mean one thing to `_geometry` and another at
the xCross call site. Same name, three readings, and the failure mode is not an epsilon: it is a box at the
**wrong end of the pitch**, an 88.5 m error.

So ship **two named entry points**, with no default and no overloading:

```
in_penalty_area_absolute(x, y, *, attacked_goal_x)   # action-LTR / absolute frame
in_penalty_area_goal_relative(gr_x, y)               # already goal-relative; caller owns WHICH goal
```

The goal-relative form deliberately takes **no goal argument at all** — the caller has already resolved
attacked-vs-defended by producing `gr_x`, so the ambiguity cannot re-enter the helper. Boundary convention is
**decided and tested**: non-strict `<=` on both axes, matching the Law's "the area includes its lines".

Migrate the two 20.16 sites now (byte-identical: both are already non-strict on x and use the same
`abs(y − 34) <= half_width` form); `_ghost_gk` migrates in the re-fit commit alongside the constant, per D3,
which is also where its strict `<` becomes non-strict.

**Tested at the edges, not mid-box:** `x` exactly 16.5 and exactly 88.5, `y` exactly 13.84 and 54.16, plus one
mirrored input per frame. A mid-box fixture passes under every wrong convention and would prove nothing.

`_ghost_gk` **keeps `40.3` this cycle**, and gains an **anti-premature-unification guard**: a test pinning
the current value whose docstring states that unifying it before the ghost re-fit silently skews the bundled
weights, plus a TODO row carrying the obligation. Without that guard the natural next contributor "finishes
the job" and reintroduces precisely the skew D3 sequences around.

**The bundled ghost artifact IS re-saved this cycle — D3's guard must not be prose (review R1 M5).** §2
rejected the permanent-divergence option precisely because it "relies on a prose contract — which this repo has
watched fail over six releases before". D3's anti-premature-unification guard, as first drafted, was that same
instrument: a docstring plus a pinning test a future contributor can delete when it goes red. So this cycle
**re-saves (does NOT re-fit) the bundled ghost artifact** to stamp its fingerprint **at the current 40.3**.
Flipping the constant without re-fitting then makes `load()` **raise** — the guard becomes the mechanism this
spec is shipping rather than a comment. This is a supported path, not a novelty: `_ghost_gk.py:1859-1860`
("Preserve the recorded training-time version across a **load->save migration**; only stamp the runtime version
for a genuinely fresh fit"), and the bundled `corpus_provenance` records it being used at the 4.54.0
parameters-only migration ("This artifact was NOT retrained"). Byte delta is metadata-only; weights untouched.

The re-save **runs on x86**, per §3's N3 commitment — the same platform the lakehouse loads on, so the one
fingerprinted artifact shipping this cycle is never compared cross-platform against an unvalidated tolerance.

> **HF is explicitly OUT — standing owner hold.** The review's M5 recommended "one HF re-upload". That is
> **blocked**: there is a standing instruction not to touch the ghost-GK HF repo, token or wheels, from the
> raw-keeper-position disclosure remediation. The reviewing session has no visibility into it. So the re-save
> covers the **in-wheel bundled artifact only**.
>
> **The hold costs less than it looks (review R2 N7, verified).** `_resolve_model` (`_ghost_gk.py:175+`)
> cascades *caller → `SILLY_KICKS_GHOST_GK_PATH` env → variant*, and `"default"` is **bundled in the wheel**;
> only `variant="full"` reaches `from_hub`. The lakehouse AC-1 path resolves the default, so **the bundled
> re-save covers the production consumer** and the hold does not weaken M5's guard.
>
> **Residual, dated and triggered:** once the re-fit flips the constant, the bundled artifact raises on a
> stale load (intended) while the HF `full` variant — 40.3-fit weights, 20.16 features, no fingerprint —
> would *warn and load*, i.e. become actively skewed and unguarded at a known future date. Trigger, recorded
> in §7: **when the hold lifts, `full` is re-uploaded or deprecated; until then it must not be selected.**

**ALL THREE bundled artifacts are stamped — amended after plan review round 2.** This paragraph previously
recorded a "known and accepted gap": that xS and xCross would not be re-saved this cycle, so their shipped
artifacts would carry no fingerprint and take the D2 warn path until their next training run. Review S1 then
proposed *counting* that gap by escalating the warning in CI behind ~14 module-level test opt-outs.

Both rested on an assumption that turned out to be false. **Verified:** all three bundled artifacts have the
identical structure — `metadata.json` + `SHA256SUMS` + a weights file — and the xS/xCross `metadata.json`
already carries `chirality`, `geometry_version`, `pitch_length` and `pitch_width`, exactly the neighbours a
`feature_contract` key sits beside. The metadata-only migration this section already specifies for ghost
therefore applies **verbatim** to xS and xCross. So all three are stamped, and the CI escalation of
`MissingFeatureContractWarning` runs with an **empty** opt-out list: fail-closed from day one, no ledger to
maintain, nothing to retire. It is also cheaper than counting the gap — one migration script covering three
directories, versus opt-outs in ~14 test files.

**What the stamp attests, stated precisely so it is not over-read.** It records what the *current* library's
extractor produces on the probe. The guarantee is **forward-looking**: from this point, any change to an
extractor or a declared constant makes `load()` raise. It does **not** retroactively prove these are the
features each model was trained on. That limit is not introduced here — it is exactly what this section already
accepted for ghost, on the load→save migration path `_ghost_gk.py:1859-1860` documents and the 4.54.0
parameters-only migration used. The supporting evidence is identical for all three: fail-closed chirality
verification passes at load (ADR-040), and chirality is model output on a fixed frame, which flows through the
extractor — evidence of stability, though not proof for a feature carrying near-zero weight.

**The residual gap is real and irreducible this cycle:** the HF-hosted `full` (ghost) and `sc_extended`
(xS/xCross) variants cannot be re-uploaded under the standing owner hold, so they carry no contract and take
the D2 warn path. That is correct behaviour rather than an oversight, and §3.3's named warning category is what
keeps it from being invisible in a batch serve. The trigger recorded above — when the hold lifts, re-upload or
deprecate — now covers all three repos, not ghost alone.

---

## 5. The two bundled items

**5.1 Cache corpus-drift fingerprint (ADR-038 follow-up).** The machinery already exists — `corpus_fingerprint(rows)`
(`scripts/_cache.py:19`), `write_cache_meta` (`:30`), `cache_is_valid` (`:38-49`) — and `cache_is_valid`
already compares a recorded fingerprint. What is missing is the trainers calling it with real data: its own
docstring (`:22`) records *"the trainers currently gate on the constant `CACHE_SCHEMA_VERSION` token"*. Wire
**`train_xcross_attempt.py` and `train_xshot_occurrence.py`** — named, because they are the *only* two of the
five trainers that use `scripts/_cache.py` (each currently gating on `_CACHE_FINGERPRINT = "schema-v2"`,
`:38` and `:37`) — to pass sorted `(provider, match_id, visibility)` triples; delete the deferral note.

**Operator cost, stated here rather than discovered on the DGX (review R1 m2):** swapping a constant token for
a live corpus hash **invalidates every existing feature cache** for those two trainers — a one-time full
re-extraction on the next run.

**5.1a `train_ghost_gk.py` must be brought onto the same gate THIS cycle — this is the blocker (review R1 B3).**
Ghost does **not** use `scripts/_cache.py`. It keeps its own cache (`args.output_dir/"ghost_gk_v1"/"_feature_cache"`,
`:238-243`) whose predicate is **bare file existence** (`:256-262`), and it caches **extracted features**
(`features.parquet`, written `:383`, read back `:265`) — i.e. `attackers_in_box` computed at **40.3**. Its own
comment concedes the gap: *"a schema-version bump lands later"* (`:254-255`).

Consequence if left: the D3 re-fit, reusing an existing `--output-dir`, silently fits weights on 40.3 features
**while stamping a fingerprint computed at 20.16** — precisely the train/serve skew this spec exists to
prevent, arriving through the one door §5.1 would otherwise leave open, on the very next operation the spec
queues. Deferring it to the re-fit commit is not viable: *the re-fit commit is the one that gets it wrong.*

Fix: bring `train_ghost_gk.py` onto `scripts/_cache.py`. If that surface proves too large during planning, the
**minimum acceptable** substitute is a version token in its predicate — but that token must be **derived from
the geometry constants themselves** (e.g. include `_PENALTY_AREA_Y_MIN` in the token string), **not** a
hand-bumped literal (review R2 N4).

A hand-bumped literal does not survive the cycle it exists to protect: it forces one re-extraction, then goes
constant again, so *within* the re-fit cycle — extract, inspect, flip the constant, re-run — the second run
happily reuses the first run's 40.3 features. That is B3 all over again, one iteration later. Deriving the
token from the constants makes it **auto-invalidate on the flip with zero discipline required** — behavioural,
not declarative, which is this spec's own stated principle. Either way it lands here, not later.

**5.2 Ghost-GK trainer startup fail-fast.** `_ghost_gk.py:257-262` already fail-closes on an unclassified
provider, but *inside* `keeper_detection_mask` — i.e. mid-run, after fitting may have begun. Add a startup
check validating every corpus provider against `_DETECTION_AWARE_PROVIDERS ∪ _FULLY_OBSERVED_PROVIDERS`
(`:245`, `:250`) before any fitting. Behaviour-preserving; it only moves the failure earlier.

**Extract the membership logic rather than copying it (review R1 m4):** a second inline
`provider ∈ _DETECTION_AWARE ∪ _FULLY_OBSERVED` would drift the next time a provider is added. Ship one
`validate_provider(provider)` and have both the startup check and `keeper_detection_mask` call it.

---

## 6. Testing

Both sides throughout — a one-sided assertion passes identically when the machinery silently does nothing.

- **Round-trip on an UNMODIFIED artifact passes.** This is the B1 regression: with the inherited
  `equal_nan=False` it fails on a byte-identical artifact (measured). Assert it explicitly, or the NaN policy
  can silently regress.
- **Mismatch RAISES:** mutate a declared constant, assert the model's own `IntegrityError`. Red-first.
- **Missing WARNS and does NOT raise** (D2) — an artifact without the key still loads — and the warning is
  the named `MissingFeatureContractWarning`, asserted **by category**, not by message text.
- **`legacy_override=True`** escapes with a warning.
- **Probe fingerprint contains ZERO NaNs.** Replaces the weaker "not all-NaN / all-zero" — the measured
  3-NaN vector passes *that* and is still ungateable (review R1 B2ii).
- **Probe identity:** a changed probe warns-and-skips the *fingerprint* with the distinct §3.2 message; it
  does **not** raise and does **not** report as a constant skew.
- **Probe changed AND a constant changed → still RAISES** (review R2 N1). The constants intersection survives
  a probe change; without this bullet the two nets cancel and the combination passes silently.
- **`in_penalty_area_*` edge cases (R2 N2):** `x` exactly 16.5 and exactly 88.5, `y` exactly 13.84 and 54.16,
  plus one mirrored input per frame. A mid-box fixture passes under every wrong convention.
- **Per-constant teeth (§3.1):** mutating each declared constant makes `verify_feature_contract` **RAISE** —
  not merely "moves the fingerprint", which a 1e-9 shift satisfies invisibly.
- **Auto-enumeration (M1):** every module-level geometry constant in the four modules is declared or
  explicitly exempted; an undeclared new constant fails the gate.
- **Declared-constant comparison:** a sub-probe-resolution change (20.16 → 20.161) that moves no feature is
  still caught by the constants intersection.
- **Constant migration:** `_xcross_attempt` and `defensive_credit/_params` values are unchanged (20.16), by
  explicit value-equality assertion, not by inspection.
- **Anti-premature-unification:** `_ghost_gk`'s box value is pinned at `40.3` with the re-fit rationale.
- **Cache:** identical corpus → HIT; a changed `(provider, match_id, visibility)` set → MISS.
- **Trainer fail-fast:** an unclassified provider raises **before** any fit call — spy the fit and assert it
  is never reached (otherwise the test passes on the pre-existing mid-run raise and proves nothing).

---

## 7. Out of scope

- **The ghost-GK re-fit itself** — a separate owner-run DGX cycle. It flips `_ghost_gk` to the canonical
  constant, migrates it onto `in_penalty_area`, and produces ghost's first *fitted* fingerprint (the §4
  re-save supplies an interim one at 40.3).

  **It must be SCHEDULED, not merely deferred — cross-repo coupling (review R1 X1).** The lakehouse consumes
  ghost-GK in its AC-1 path, so re-fitting ghost changes served ghost `(x, y)` ⇒ AC mart columns change ⇒ a
  full drain recompute (~5.5 h) **plus regeneration of both mini-golden fixtures**. A lakehouse AC recompute
  is *already queued* against the 4.52.0 xT-EPV / TF-35 work. **Sequence the ghost re-fit into that same
  window or the drain is paid twice.** Recorded here so it is a scheduling input rather than a surprise.
- **Retiring `geometry_version`** (§3.2).
- **Back-filling fingerprints into the xS and xCross artifacts by re-saving them.** Rejected for those two:
  D2 already makes them load cleanly, so a re-save buys no behavioural gain. **Ghost is the exception and is
  re-saved (§4)** — there the gain is real, because ghost is the model D3 defers and the fingerprint is what
  converts D3's guard from prose into a raise.
- **Re-uploading anything to the ghost-GK HF repo** — standing owner hold (§4); the HF `full` variant stays
  on the D2 warn path.

  **Dated accepted risk with a trigger (review R2 N7):** after the re-fit flips the constant, `full` holds
  40.3-fit weights that would be served 20.16 features, with no fingerprint to catch it — actively skewed and
  unguarded from that date. **Trigger: when the hold lifts, `full` is re-uploaded or deprecated; until then it
  must not be selected.** The bundled `"default"` — which is what the lakehouse resolves — is unaffected.

- **The DGX-vs-x86 tolerance measurement** — deliberately deferred with a named trigger (§3, R2 N3): it is a
  prerequisite of the **first DGX-produced fingerprint**, not of this cycle, because the only artifact
  fingerprinted here is re-saved on x86 and never compared cross-platform.

**Cross-repo dependency (review R1 X2):** the lakehouse silly-kicks version bump is currently held, so
everything in this spec is inert in lakehouse production until it ships. When it does, the bump is lockstep:
`pyproject` → `uv lock` → their TF-env pin sync script — never hand-edited.

**Verified already-closed, recorded so review does not re-raise them:** the OAuth loader follow-up is DONE
(`scripts/_loader_databricks._connect:36-58` is OAuth-native — PAT wins, else `databricks.sdk.core.Config`;
shipped 4.55.1), and the ghost-GK HF artifact is FIXED (`metadata.json` carries `chirality` and
`stores_training_data: false`; `GhostGkModel.from_hub()` verified loading 2026-07-25).

---

## 7a. Review round-1 close-out

Recorded so round 2 is a confirmations pass, not a re-read. **Adopted in full:** B1 (NaN policy stated, §3),
B2 (tolerance chosen not inherited + gate-fires assertion, §3/§3.1/§6), B3 (ghost trainer cache, §5.1a),
M2 (sibling signature, §3), M3 (probe identity + warn-and-skip + constants comparison, §3.2), M4 (named
warning category, §3.3), M5 (bundled ghost re-save, §4), M6 (shared `in_penalty_area` predicate, §4),
m1 (ghost pitch-dims guard, §3.4), m2 (trainers named + cache-invalidation cost, §5.1), m3 (the real
0.0398 % figure, §1), m4 (shared `validate_provider`, §5.2), m5 (load cost, §3.3), X1 (re-fit scheduling,
§7), X2 (lockstep bump, §7).

**Adopted with the reasoning corrected — M1.** The review held that the fingerprint's completeness equals the
hand-maintained `constants` dict. It does not: the fingerprint is over the feature vector, so an undeclared
constant that moves a probe-exercised feature still fires the gate — the dict is not in the gating path. The
conclusion survives by a different mechanism (probe *sensitivity* is designed around the declared set), and
the recommended auto-enumeration gate is adopted precisely because it forces probe coverage per constant.
Recorded because the distinction matters for anyone extending this later. See §3.1.

**Adopted in modified form — M5's HF leg.** The review recommended re-saving *and* re-uploading to HF. The
HF half is blocked by a standing owner hold the reviewing session could not see; the bundled artifact is
re-saved, HF is untouched. See §4.

**One defect found in this spec by its own author while applying the review:** §3.1 originally specified the
probe player as sitting "inside the 1 cm boundary band", which m3's measurement then proved insufficient —
844 band rows collapsed to 70 once box *depth* was also required. A probe built to the original wording would
have moved nothing and reported the box constant as uncovered. Fixed in §3.1.

## 7b. Review round-2 close-out — spec review CLOSED

Round 2 returned **0 blockers / 3 major / 4 minor**, with all three majors direct consequences of round-1's
own fixes. That is the pre-agreed signal that the round is manufacturing as much as it finds, so the spec
review stops here and the next lens is the plan.

**All seven folded into rev 3.** N1 — probe mismatch skips only the *fingerprint*; the constants intersection
still runs and can still raise, closing the "change the probe and a constant together" hole (§3.2 + a §6
bullet). N2 — the single `in_penalty_area(x, y, *, goal_x)` is **replaced by two frame-explicit entry points**;
verified that the three sites differ on *frame* and *reference goal*, not just strictness, and that `goal_x`
already means an absolute reference-goal x in `_geometry`, so overloading it risked an 88.5 m error rather
than an epsilon (§4). N3 — the tolerance measurement gains an owner and a trigger, and the §4 re-save is
pinned to **x86** so `atol=1e-6` is safe for the only artifact fingerprinted this cycle (§3, §4, §7). N4 —
the §5.1a fallback token is **derived from the geometry constants** so it auto-invalidates on the flip,
instead of a hand-bumped literal that would go stale inside the very re-fit cycle it protects (§5.1a). N5 —
§3.1's "not in the gating path" qualified to "not in the *fingerprint* gate's path", cross-referencing §3.2.
N6 — the new warning registers in all three surfaces, not one. N7 — **verified** that the default variant is
bundled in-wheel and only `"full"` reaches the Hub, so the bundled re-save covers the lakehouse production
path; the residual `full` skew is recorded in §7 as a dated risk with a trigger.

**Reviewer concessions recorded, since rejections should be as visible as adoptions:** round 2 conceded that
round 1's M1 mechanism was wrong (the fingerprint is over the feature vector, so the dict is not the
completeness bound) and that the §3.1 probe-construction defect was theirs to miss. Round 2 also
independently cross-checked the m3 arithmetic (3,871,535 − 2 × 175,969 = 3,519,597 exactly), which is
reasonable evidence the harness run was real rather than asserted.

---

## 8. Attribution / C4 / retrain

No new aggregator, backend or trained model → **C4-free (count stays 32)**; confirm by running `/c4` at
commit-prep rather than asserting it. In no default xfn list, no VAEP consumer → **no retrain trigger**. The
penalty-area figure is the FIFA Laws of the Game landmark (40.32 m); no new NOTICE entry is required beyond
citing it inline at the constant.
