# PR 5 — the chiral goal-relative transform + xS/xCross retrain

**Status:** design, approved 2026-08-01; revised 2026-08-02 after cross-session review
**Cycle:** ADR-051 / ADR-028 orientation defect class, PR 5 of 5 — the last one
**Source of truth for the defect:** §8b of `2026-07-29-adr028-orientation-defect-class-design.md`
**New ADR:** none. §8b is already recorded in ADR-051, exactly as PRs 2–4 shipped. The
`sc_extended` decision in §5.3 lands as an ADR-051 amendment note.
**Version:** not assigned. Numbers are taken at commit-prep, after merging `origin/main` — five
register collisions in this cycle say the window can only be closed, never bounded.

---

## 0. What the cross-session review changed

The part-deux review verified every load-bearing claim against source and re-derived the headline
counts independently. Three findings changed the work; the rest were accuracy. Recorded here so the
reasoning is not re-derived:

| # | Finding | Effect |
|---|---|---|
| F1 | "One seam" is **false for xShot** — y is read at four sites | §3 now pre-transforms the frame |
| F2 | §7 gate 4 was **green by construction** | §4 rationale replaced, gate 4 replaced |
| F3 | One commit is **incompatible** with the provenance rule | §9 is two commits (owner ruling) |
| F4 | The rejected-alternative cited a call site that does not exist | §3 rationale replaced |
| F5 | `to_goal_relative_vx` has **zero** production callers | §3 says so explicitly |
| F6 | "~1800 cells" is stale; it is **805** | §4 fixes the docstring in the same edit |
| F7 | A derived anchor beats a magic `1.0` + a res-specific comment | §4 adopts the formula |
| F9 | The backwards ADR-038 question is already answered | §6.1 closes it by citation |
| F10 | The registry's own prose denies the markers exist | §7 deletes it with them |
| F11 | The `atol` policy is already recorded in code | §10 risk 1 shrinks to one step |
| F8 | `sc_extended` re-upload never stated visibility or labels | §5.3 is now a four-row label table |
| F12 | "Physically mirrored" can invert the gate | §7.2 says *rotation* |

Review #2 (2026-08-02) **executed** the claims rather than re-reading them, and closed §10.6:

| # | Finding | Effect |
|---|---|---|
| — | F2 **executed and confirmed** on both models | §10.6 closed; §4 keeps its replacement rationale |
| N1 | §6.1 closed F9 with the wrong warrant (right answer) | §6.1 cites the artifact's own `metrics.json` |
| N2 | The **default** GS variant will be labelled `"full"` once ADR-038 is wired | §6.1 — **owner decision**, surfaced not taken |
| N3 | Gate 4's "landed red" is true of the **y** assertion only | §7 gate 4 says so |
| N4 | Gate 2 is partly vacuous on `canonical_scene()` — measured | §7 gate 2 records it and enriches the scene |
| N5 | Not committing the spec strands the design record | §9 — owner has ruled; consequence recorded |
| N6 | F8 missing from this ledger | fixed above |

Two claims the review did **not** cover, verified here: pre-transforming the xShot frame is safe
(every x already routes through `gx()` — `bx`, both `.map(gx)` calls, `gkx`, so no site wants
absolute coordinates), and `openGoal` stays invariant because the goal-mouth clamps
(`_xshot_occurrence.py:101-102`) are symmetric about y=34.

## 1. Why this PR exists

The ADR-051 cycle's remaining debt is a ledger of 10 strict xfails:

```
Gate A   add_xcross_attempt, add_xshot_occurrence                                    2  <- this PR
Gate B   add_cover_shadows, add_defensive_line, add_gk_influence, add_line_break,
         add_off_ball_context, add_packing, add_player_influence, add_structural_pass 8  <- PR 6
```

This PR closes the 2 Gate A markers. PR 6 closes the 8 Gate B markers (D3 re-key), PR 7 completes
ADR-050's penalty-area unification, and TF-24's refresh follows PR 7. All in one session; each
separately revertable, which is why they are not one change.

## 2. The defect

`silly_kicks/tracking/_geometry.py` has `to_goal_relative_x` and `to_goal_relative_vx` and **no
`to_goal_relative_y`**. With `_flip(goal_x) = goal_x > 50.0`:

- `goal_x=105` maps `(x, y) -> (105-x, y)` — an x-only **mirror**, determinant **-1**
- `goal_x=0` is the **identity**, determinant **+1**

The two goal ends therefore use frames of **opposite handedness**.

**The mechanism.** Both extractors build bearings as `atan2(y - GOAL_Y, gr_x)` where `gr_x` is
flipped at the high-x end and `y` never is. Radials are `hypot(gr_x, y - GOAL_Y)`, and since
`(y-34)^2 == (34-y)^2` they are **byte-identical**; every bearing **negates**. That is why it
survived: distances and radii all agree.

**Measured** on `canonical_scene()`: xS **12 of 27** features flip sign, output 0.01113 -> 0.01293
(**+16.2%**); xCross **3 of 16**, 0.00168 -> 0.00113 (**-33.0%**) — exactly ADR-037's
"sign-inconsistent" counts, reached independently from the ADR-028 side. PR-S118 retrained the
**symptom** and left the transform chiral.

Review re-derived the counts by inspection: xS flips `theta`, `GK_theta`, `DefAngle_0..4`,
`OffAngle_0..4`; xCross flips `ball_theta`, `gk_theta`, `gk_lateral_offset`. Two near-misses are
**not** flips — `gk_carrier_side` (`_xcross_attempt.py:210`) is a product of two negating factors,
and `openGoal` is invariant under symmetric clamps.

**Production consequence.** One physical scene scores differently depending which END the acting
team attacks: a systematic home-vs-away split *inside a single match*.

## 3. The fix

Add to `_geometry.py`:

```python
def to_goal_relative_y(y: float, *, goal_x: float) -> float:   # (PITCH_WIDTH - y) if _flip else y
def to_goal_relative_vy(vy: float, *, goal_x: float) -> float: # -vy if _flip else vy
```

so `(to_goal_relative_x, to_goal_relative_y)` is the 180-degree point reflection
`(x, y) -> (105-x, 68-y)` and the two ends differ by a **rotation**.

The module defines `FIELD_LENGTH`, `GOAL_Y`, `PITCH_LENGTH` and `PITCH_WIDTH`. There is **no**
`FIELD_WIDTH` — the y counterpart is `PITCH_WIDTH`.

**`to_goal_relative_vy` will be dead code, and so is its sibling.** `to_goal_relative_vx` has
**zero** production callers — the only references are its own doctests (`_geometry.py:52-55`) and
`tests/tracking/test_xshot_occurrence.py:17-28`. Neither extractor uses a directional velocity;
xS's `bvx`/`bvy` (`:192-193`) and xCross's (`:167-169`) feed only `hypot`. `_vy` is added for
symmetry and **cannot be exercised by §7 gate 2**, because no feature depends on it. Note also that
CI does not execute `_geometry.py` doctests — the public-surface doctest step ignores
single-underscore private modules, so those examples are documentation, not a gate.

### 3.1 Where the reflection is applied — different answers per extractor

**xCross genuinely is one site.** `_xcross_attempt.py:158-159` builds `gr_x` and `y` together;
convert `y -> gr_y` there and everything downstream follows.

**xShot is not.** `_xshot_occurrence.py:180-181` defines `def gx(x)` — an **x-only** helper. There
is no site at which to convert y, and y is read at **four independent** places:

| site | expression |
|---|---|
| `:191` | `by = float(ball["y"].iloc[0])` |
| `:216` | `defending["y"].to_numpy(dtype=float)` |
| `:226` | `attacking["y"].to_numpy(dtype=float)` |
| `:236` | `gky = float(gk_rows["y"].iloc[0])` |

So the earlier claim — "one seam, no call site can be missed" — was true for the model with 3 sign
flips and **false for the model with 12**. A missed site reproduces the original defect's silent
shape exactly: distances stay right, only some bearings move.

**Resolution: transform the frame once at the top of `extract_xshot_features`** and delete `gx`.

```python
fd = frame_data.assign(x=..., y=...)   # goal-relative once, for every consumer
```

This is numerically identical to adding a `gy()` sibling and enumerating the four sites, but it
makes "no call site can be missed" true **by construction** rather than by assertion — which is the
whole point, given the defect being fixed is a missed-site defect.

**Verified safe:** every x in the extractor already routes through `gx()` (`bx = gx(bx_raw)`,
`defending["x"].map(gx)`, `attacking["x"].map(gx)`, `gkx = gx(...)`), so no site wants absolute
coordinates and pre-transforming cannot double-apply. The implementation must still confirm no
downstream reader is added that expects absolute coords, and rename the now-misleading `bx_raw`.

**Rejected alternative:** an offset helper `to_goal_relative_dy(dy, *, goal_x) -> -dy if flip`.
The originally-recorded reason was wrong — it cited `in_penalty_area_goal_relative`, which neither
extractor calls (its only caller is `_geometry.py:99`, inside `in_penalty_area_absolute`), and
whose `abs(y - 34) <= half_width` is y-symmetric anyway, so it could never have discriminated. The
two reasons that **are** true: a dy helper cannot express `gk_lateral_offset`
(`_xcross_attempt.py:203`) or `side` (`:205`), which are absolute-y reads rather than offsets; and
it cannot reach the `space_controlled` grid at all.

**Docstring.** `_geometry.py:3-6` claims LTR and RTL frames "map to identical feature values". That
is falsified today and becomes true under this fix. Restate it as the invariant plus the gate that
enforces it (§7) — not softened, not deleted.

## 4. Rides with it — the dominant-region grid

`_xcross_attempt._dominant_region_area` (`:115-116`):

```python
xs = np.arange(res / 2, _geo.PITCH_LENGTH, res)   # 1.5 .. 103.5, 35 centres
ys = np.arange(res / 2, _geo.PITCH_WIDTH,  res)   # 1.5 ..  67.5, 23 centres, centred on 34.5
```

**The original rationale was wrong and is replaced.** §8b reasoned that "under `y -> 68-y` a centre
at 1.5 maps to 66.5, which is not a grid point (`space_controlled` 328.17 -> 310.43)". That is a
**pre-fix** measurement and it evaporates once y is transformed: at `:191` the inputs are
`all_xy = [(gr_x[i], y[i]) ...]`, so after §3 both legs feed **bit-identical** goal-relative arrays
and every function of them agrees — grid included.

**The residual defect is real but different: a left-right handedness bias, measured at 5.4%.**
Centres 1.5…67.5 cover `[0, 69]`, so 1 m of the sampled band sits beyond the far touchline and 0 m
beyond the near one. Stated that way it sounds like a sampling nicety; the measured consequence is
sharper. A scene and its **left-right mirror at the same goal end** differ by **17.74 m^2 — 5.4% —
in `space_controlled`, xCross model feature #3**:

```
left-right mirror (y -> 68-y) at a FIXED goal end:
  space_controlled  310.43478 vs 328.17391   delta = 17.73913 (5.4%)
  grid inputs identical between legs? False
```

In goal-relative space that axis is **left wing versus right wing** — for a *cross* model, the axis
that matters most. So this is a quantified handedness bias in a live feature, not an accuracy
rounding issue. It is still **not** an orientation defect (it is identical at both goal ends), which
is why §7 gate 4 had to change rather than the conclusion.

**Fix: a derived anchor, not a magic number.** For `n = round(L / res)` cells the symmetric anchor
is

```python
a = L / 2 - (n - 1) * res / 2
```

which yields **1.5 for (105, 3)** — byte-identical to the shipped x grid — and **1.0 for (68, 3)**,
and stays symmetric for *any* `res`. Use it for both axes.

Rejected: hard-coding `1.0` with an inline comment that "105 is divisible by 3, 68 is not". That is
true only at `res=3.0` and becomes actively misleading the moment `res` is coarsened — which
`:110-111` explicitly anticipates. The failure direction even inverts: at `res=2.0` it is the **x**
grid that is asymmetric (centres 1, 3, …, 103; `105 - 1 = 104` is not a centre). F6 is proof `res`
has already moved once. The formula also introduces no new module-level geometry constant, so
`tests/tracking/test_geometry_constant_enumeration.py` is unaffected.

**Fix the stale docstring in the same edit.** `:109` says "~1800 cells x ~22 players at res=3.0".
The real count is `35 x 23 = 805`; ~1800 matches `res=2.0` (52 x 34 = 1768) and went stale when
`res` moved. The PR edits the two lines directly beneath it.

**Why the same PR:** `space_controlled` is xCross model feature #3, so it shares the retrain
regardless. Only the reasoning changed, not the conclusion.

## 5. Retrain and artifact lifecycle

### 5.1 Atomic by construction

Both bundled artifacts carry `chirality` (ADR-040) **and** `feature_contract` (ADR-050) stamps,
both fail-closed:

- changed transform -> changed feature VALUES -> ADR-050 fingerprint mismatch -> `load()` **raises**
- changed features -> changed model OUTPUTS -> ADR-040 fingerprint mismatch -> `load()` **raises**

Review confirmed both fingerprints actually move: each probe frame is y-asymmetric by construction
(`_chirality.py:22` "all rows deliberately OFF the y=34 mirror axis"; `contract_probe_frame` places
A1 at y=13.845). So code fix, retrain and re-stamp are **one indivisible change** — a code-only
wheel's own weights refuse to load.

### 5.2 Blast radius (verified narrow)

`to_goal_relative_x` has exactly two callers: `_xshot_occurrence.py:181` and
`_xcross_attempt.py:158`. **`_ghost_gk` does not consume it** — it has its own `_defending_goal`,
and `_ghost_gk.py:1928-1929` writes pitch dims with **no** `geometry_version`, independently
corroborating the boundary.

silly-kicks retrains `_xshot_weights/` and `_xcross_weights/`. The lakehouse re-materializes, and
retrains only if it fit models on those columns — `xshot_occurrence_xfns`/`xcross_attempt_xfns` are
wired into `pre_shot_gk_full_default_xfns` ONLY. The rho retention model uses marts features and is
unaffected.

**This is a house-convention retrain trigger** and must be declared explicitly in **CHANGELOG** and
**CLAUDE.md** — the spec names the consequence, those two artifacts record it.

**Declare the SHAPE of the change, not just its existence** — "half the rows, plus a two-sided
feature-#3 shift" is far more useful to a lakehouse consumer than "feature values change". Measured:

```
same scene, pre-fix vs post-fix
  xShot   goal_x=  0 : max delta 0.000e+00      goal_x=105 : max delta 6.124e+00
  xCross  space_controlled (shipped -> derived anchor)
          goal_x=105 : 310.43478 -> 310.43478  (+0.00%)
          goal_x=  0 : 328.17391 -> 310.43478  (-5.41%)
```

`to_goal_relative_y(y, goal_x=0)` is the identity, so **the transform fix moves only rows attacking
the high-x goal** — roughly half the corpus, and precisely the "home-vs-away split inside a single
match" of §2. The **grid** change is two-sided: it can move any row, and on this fixture landed at
0% at one end and −5.4% at the other. That split is scene-dependent, not a structural rule.

### 5.3 Arms, visibility, and expected labels

Four fits: xS and xCross x {`public`, `sc_extended`}.

`from_variant("sc_extended")` routes to `from_hub` for both models, and those Hub artifacts carry
chirality stamps computed on the **old** transform, so after this PR they mismatch and `load()`
raises. They are retrained and re-uploaded here (owner decision, 2026-08-01).

**Expected outcome, stated rather than assumed** (the machinery is already wired at
`train_xshot_occurrence.py:442-452, 517`):

| fit | corpus | expected `artifact_label` | destination |
|---|---|---|---|
| xS `public` | 17 (10 SkillCorner + 7 IDSSE) | public | bundled in the wheel |
| xCross `public` | 17 | public | bundled in the wheel |
| xS `sc_extended` | public + 98 owner-tier | **restricted** | HF `silly-kicks/xshot-occurrence-v1` |
| xCross `sc_extended` | public + 98 owner-tier | **restricted** | HF `silly-kicks/xcross-attempt-v1` |

Two of the four are restricted by construction, and ADR-038's `is_public_row` / `artifact_label` /
`assert_public_corpus` must produce that verdict rather than the fits being labelled by hand. The
exact `sc_extended` match count is an inference; `scripts/_paired.py` defines the three-candidate
`public`/`sc_extended`/`full` sequence and is the authority — confirm at build time.

### 5.4 Stamping platform — confirm one step, not a policy

`_feature_contract.py:37-40` already records the policy:

> The atol is pending a measured DGX-vs-x86 delta; until then every fingerprinted artifact is
> produced on x86 so no cross-platform comparison happens against an unvalidated tolerance.

So there is nothing to decide, only one step to confirm: that the runbook copies weights to x86
before `stamp_feature_contracts.py`. This PR still **measures** the DGX-vs-x86 probe delta while
both platforms are first in play (one probe-extractor call per platform) and records it for PR 7.

Per ADR-050 §1 the answer may not be "widen the number": if a covering tolerance would also swallow
a real 1 cm geometry change, the honest conclusion is that fingerprints are **platform-scoped** —
verify on the stamping platform, and have the other skip the fingerprint prong only, since the
constants prong is platform-independent by construction.

**One line on the declared-constants prong:** both probes run at `goal_x=105.0`, so after the fix
`PITCH_WIDTH` becomes load-bearing on both feature vectors. The separate fail-closed pitch-dims
prong (`_xshot_occurrence.py:535`, `_xcross_attempt.py:585`) is expected to cover it; confirm
rather than assume.

### 5.5 `GEOMETRY_VERSION` must be bumped

`_geometry.py:25` defines `GEOMETRY_VERSION = "goal-relative-1"` with the instruction:

> Bump when the goal-relative transform's NUMERIC output changes (NOT for a pure origin
> translation like TF-38, which is invariant). Consumed by trained-model metadata as the
> coordinate-change fail-closed guard.

This PR changes exactly that, so the bump to `"goal-relative-2"` is **mandatory by the constant's
own contract**. It is written at save by `_xshot_occurrence.py:486` / `_xcross_attempt.py:543` and
recorded in both artifacts' `metadata.json`.

**The load policy is weaker than the name suggests.** At `_xshot_occurrence.py:529` and
`_xcross_attempt.py:582` the fail-closed prong is on pitch **dims**; a bare `geometry_version`
mismatch only **warns**, on the recorded rationale that such a change "at identical dims is
translation-invariant". **This change is not translation-invariant.** Net protection stays
fail-closed via chirality and feature_contract — but that is where it comes from, and a future
change relaxing those two must know `geometry_version` alone will not hold the line.

**Concrete test breakage:** `tests/tracking/test_xshot_occurrence.py:381`
(`test_load_warns_on_geometry_version_only`) monkeypatches the library constant to
`"goal-relative-2"` to force a mismatch. Once the library **is** `"goal-relative-2"` it compares a
value to itself and silently stops testing. Its sentinel must move.

## 6. Folded-in scope

Approved 2026-08-01 after reviewing all 19 On-Deck items.

### 6.1 Trainer plumbing (TODO L26, closes the row)

`_loader_pining.load_matches` takes `cache_dir` (`:224`) which persists downloaded artifacts per
match and reuses them; without it the loader falls to `tempfile.TemporaryDirectory()` and
**re-downloads the whole corpus every run**. Verified occurrences per trainer:

```
11  train_ghost_gk.py       (its FEATURE cache, not the download cache)
 0  train_gk_completion.py
 0  train_gk_retention.py
 0  train_xcross_attempt.py    <- this PR runs it
 0  train_xshot_occurrence.py  <- this PR runs it
```

**This is a consequence of an existing decision, not a new observation.** Both trainers already
adopt `scripts/_driver.py`'s `for_each`, and ADR-052's recorded rule is that `for_each` "resumes
WORK, never the PRODUCTION of its items" — which is precisely why sharding does not save the
download and why `cache_dir` is the right lever.

At PR-S141's measured rates (~24 s/match solo, ~90 s/match under contention) this is on the order
of **hours per pass**; the precise figure moves with the §5.3 match count, the decision does not.
**Land it before the first DGX run.**

Also wire ADR-038 taxonomy into `train_gk_completion.py`, which calls `require_clean_tree`
(`:647-649`) but imports none of `is_public_row` / `artifact_label` / `assert_public_corpus`.

**The backwards question is answered — by the artifact's own record, not by CLAUDE.md.** That
model's SkillCorner variant was retrained and bundled in PR-S141 / 4.73.0 with no taxonomy guard,
so: was the shipped artifact's corpus public? An earlier draft answered by citing CLAUDE.md's
"10-match PUBLIC arm" — but that describes what the arm **is**, and F9's whole premise is that the
trainer enforces no correspondence between the arm and the run. A general fact cannot settle a
particular run. The run's own record can, and does:

```
silly_kicks/tracking/_gk_completion_weights/skillcorner/metrics.json
  n_matches = 10   bundled = True
  run_commit = 4b153655f2388c4c1f4009d5abb0955b114222f1   run_tree_dirty = False
```

Ten matches, clean tree, resolvable commit. The shipped artifact is fine; the guard is still
missing, which is what this closes.

**N2 — wiring ADR-038 has a visible consequence on the OTHER bundled variant, and it is an owner
decision.** `_gk_completion_weights/default/metrics.json` records `providers = ['gradientsports']`,
and `scripts/_corpus.py`'s `artifact_label` returns **`"full"`** — the most restricted tier — for
any non-all-public run containing `gradientsports`. Today that variant's `metrics.json` carries no
label field at all. Once ADR-038 is wired in, its next run is labelled `"full"` rather than
unlabelled.

This will **not** break the build: `assert_public_corpus` only subset-gates the `public` claim, and
its docstring names a GS-only run as legitimate. But it puts a new, publicly visible restricted-tier
label on an artifact that **ships in the wheel**, which is a licensing statement rather than a
mechanical change.

**Owner decision, 2026-08-02: wire it in and ship the `"full"` label.** The reasoning recorded so it
is not re-opened: the guard creates no new exposure — those GS-derived coefficients already ship, so
the label documents an existing situation rather than changing what is distributed. Withholding the
label would also make this artifact the only one of the three whose tier is undeclared, which a
future reader would reasonably "fix".

Keep the two risks distinct: what TODO L26 actually names is the **SkillCorner** gap (a defaulted
`--max-per-provider 64` run pulling 54 restricted matches into a distributable artifact with nothing
refusing it or labelling the result). That is what the guard closes. The **Gradient Sports** `"full"`
label is a side effect of the same wiring.

**Caveat to state rather than discover:** a download cache keyed on `{provider}/{match_id}` serves
stale bytes if an upstream artifact is ever revised. These are immutable historical matches, so the
risk is low — but it is a risk, not an absence of one.

### 6.2 Driver encoding (TODO L35)

16 drivers crash on `--help` under cp1252 — hit twice while preparing this spec, once while
enumerating the project's own TODO (`UnicodeEncodeError: 'charmap' codec can't encode '→'`).

**Target printed strings specifically.** 21 files under `scripts/` contain non-ASCII bytes, but only
*printed* text can crash — argparse `help`/`description`/`epilog` and `print()`. Comments and
docstrings cannot, so a blanket ASCII sweep would be both larger and beside the point. The gate is
`--help` exits 0 on a cp1252 console.

### 6.3 TF-19 probe re-run

This PR supersedes xS/xCross verdicts cited in six directories:

```
docs/research/  tf19_entanglement  tf19_pr2  tf19_pr3b
                tf19_pr3b_xs_v2    tf19_signoff_power   xcross_causal
```

They cite numbers measured on the **chiral** weights, including the two verdicts the cycle leans on
(the xS-v2 probe `pass`; xCross `tf19_ready=false` / `gated_clean_fail`). This is the failure mode
the repo already has a rule for: an artifact whose inputs came from another run needs provenance on
**both**, or a clean SHA launders a stale input.

Re-run the two **registered** probes with constants unchanged — `scripts/validate_xs_probe.py` and
the xCross substitution probe in `_xcross_eval.py` — and update the affected directories.

**A verdict may flip. That is a result, not a failure of the PR** — and it is why the re-run is in
scope rather than deferred. Not in scope: the TF-19 composition + audit harness (TODO L43).

## 7. Acceptance gates

1. **The 2 strict Gate A xfails must be deleted.** `strict=True` means XPASS fails the build, so
   the fix cannot land without removing its own markers. Already written; this is the primary gate.
2. **Feature-vector identity under a 180-degree POINT REFLECTION.** On `canonical_scene()`, a scene
   reflected in **both** axes (`x -> 105-x` and `y -> 68-y`, as `test_mirror_registry.py:159` does)
   must produce identical feature vectors at `goal_x=0` and `goal_x=105`: the 12/27 and 3/16 sign
   flips go to **0**. The word *rotation* is load-bearing — an **x-only** mirror yields identical
   features under the current chiral transform and different ones under the fixed one, so a reader
   implementing "physically mirrored" literally builds a gate that passes today and fails after the
   fix.
   **Gate 2 is partly vacuous on `canonical_scene()` as it stands — measured, not suspected.** Four
   features sit at degenerate values in **both** legs, so the identity assertion proves nothing
   about them: xS `openGoal` = 1.000000 (saturated — no defender's shadow survives the
   `[30.34, 37.66]` goal-mouth clamp, so `_open_goal_fraction` returns its empty-interval default),
   and xCross `box_off_def_ratio` = 0.0, `ten_minute_warning` = 0, `score_differential` = 0. So §0's
   claim that `openGoal` is invariant under symmetric clamps is true **by argument and untested by
   the gate meant to demonstrate it**, and `box_off_def_ratio` — the only consumer of the `in_box`
   y-predicate — is likewise unexercised. **Enrich the scene** (one defender inside the ball->goal
   cone, one attacker inside the box) so the gate's discriminating power does not rest on the
   bearing features alone. If any feature remains degenerate afterwards, record it the way the
   registry already does elsewhere — "EMPTY BY MEASUREMENT, not by omission".
3. **Non-vacuity partner for (2).** Shown **failing against the pre-fix transform** — proven by
   planting, not by passing. Already demonstrated by review #2: the pre-fix leg reproduces the 12
   (xS) and 3 (xCross) sign flips and a max delta of 6.124, so the harness provably sees the defect.
4. **Grid symmetry, asserted on the property the anchor actually controls.** The previous gate
   ("dominant-region value equal at both ends") was **green by construction**: after §3 both legs
   feed bit-identical goal-relative arrays, so it passes with the grid untouched and cannot fail for
   the reason it was registered. Replace with: the centre SET is invariant
   (`set(ys) == set(PITCH_WIDTH - ys)`, and the same for x), plus a left-right mirror
   (`y -> 68-y` at a **fixed** goal end) invariance test on `_dominant_region_area` — measured to
   move by 17.74 m^2 (5.4%), so the replacement gate provably *can* fail where the one it replaces
   could not.

   **"Landed red" is true of the y assertion ONLY.** Measured: the x centre set is **already**
   invariant today (`set(xs) == set(105 - xs)` -> True). If gate 4 is written as one parameterized
   test over both axes, the x case is green-by-construction. That is fine and wanted as a regression
   guard — but ADR-051's detection-first rule is about *observing* a gate fail, and half of this one
   cannot. State it, or the next reader reads "landed red" as covering both axes.
5. **`GEOMETRY_VERSION` bumped to `"goal-relative-2"`, and its sentinel moved** (§5.5). Needs its
   own gate precisely because a forgotten bump **fails nothing** — the mismatch path only warns.
   Assert the library constant, both re-stamped `metadata.json` values, and that
   `test_load_warns_on_geometry_version_only` still produces a genuine mismatch.
6. **Stale registry prose deleted with the markers.**
   `tests/tracking/_mirror_entries/trained_and_das.py:11-14` says the two entries are "Registered at
   the exact tolerance **with no defect marker**" and `:16` says "NOTE (finding, deliberately NOT
   xfail-ed)" — but `:161` and `:218` both pass `defect=`, which becomes
   `pytest.mark.xfail(strict=True)`. The prose already denies a gate that exists; delete it with the
   markers, or the next reader inherits a doc that contradicts the code. Its module note describing
   the x-only transform also goes stale on merge.
7. **Standing gates:** `python -m ruff check` + `ruff format --check`, `python -m pyright` **bare
   over the whole repo including `tests/`** (a scoped run is green while CI is red), full non-e2e
   suite, C4. `/final-review` runs **before** each commit and its findings are fixed in the working
   tree.
8. **Retrain acceptance:** each of the four fits clears its recorded gates, and every artifact
   `load()`s under fail-closed chirality **and** feature-contract enforcement — the strongest
   end-to-end check that code, weights and stamps agree.

## 8. Out of scope

- **The 8 D3 Gate B xfails** — PR 6, a different defect family.
- **Ghost-GK box constant** — PR 7. Ghost does not consume the chiral transform.
- **TF-24 calibration refresh** — after PR 7, same session.
- **TF-30 (b)** — PR 6 unblocks it, but its design questions are unanswered (spec §10.1: the RQ1
  lane target is circular). Own spec.
- **TF-51 Item 4 / Track B, Metrica GK identification, SkillCorner keeper-origin, TF-50/52/53/54/55,
  the QA bundle** — unrelated to orientation or this retrain.
- **ADR-code reconciliation sweep** (TODO L36) — PR 7, once at cycle end.

## 9. Sequencing — TWO commits, and why it cannot be one

`scripts/train_xshot_occurrence.py:392`, `scripts/train_xcross_attempt.py:438` and
`scripts/validate_xs_probe.py:438` all call
`require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` from `main()`, on the recorded
rationale that "an artifact whose provenance is unknown is one nobody can reproduce or audit later".

The fits must run against **final** geometry. With nothing committed the tree is dirty at every fit,
so each run either refuses or is forced through `--allow-dirty` and records `dirty: true` — on four
artifacts that ship in the wheel and six research directories. CLAUDE.md is explicit that the hatch
"must never launder the fact". Amending afterwards is worse: the artifacts would cite a rewritten
SHA, which is exactly why 4.72.0 was merged rather than squashed.

**Owner ruling (2026-08-02, revised after review #3): THREE commits on one feature branch.**

Two commits are not enough, and the reason is the same rule one layer further on. Everything in
Phase B re-dirties the tree: `--output-dir` inside the repo makes fit 2 refuse on `?? runs/`, and
`validate_xs_probe.py:438` calls `require_clean_tree` while needing to run **with** weights that
would not be committed until after it. That is circular. Three commits break it.

```
commit 1  code + THIS SPEC + the plan: §6.1 cache_dir + taxonomy, §6.2 cp1252, §3 transform,
          §4 grid, §5.5 GEOMETRY_VERSION bump, §7 gates (incl. the plant and the red grid gate)
          -> a CLEAN SHA the fits can cite; legitimately broken on its own, since §5.1 means a
             code-only tree's bundled weights refuse to load BY DESIGN

   then   4 fits on DGX (--output-dir OUTSIDE the repo) + §5.4 atol measurement -> x86 re-stamp

commit 2  weights + re-stamped metadata + SHA256SUMS + HF/model-card updates

   then   §6.3 probe re-run, from a clean tree containing the weights under test

commit 3  the six research directories + CHANGELOG + CLAUDE.md retrain-trigger declaration

   then   commit-prep: merge origin/main, THEN number
```

Commit 2 before the probes is **stronger provenance**, not a concession: each research directory
then cites a SHA that actually contains the weights that produced its numbers.

**This is three substantial commits on ONE feature branch, not a WIP sequence and not pushes to
`main`.** None is microscopic, none is squashed away.

**The spec and the plan are committed WITH commit 1** — never as a commit of their own. That is the
long-standing pattern (109 specs and 109 plans are tracked), and it is operationally required:
`_provenance.py:73` counts untracked files as dirty on purpose, so leaving them in the working tree
would make **every** `require_clean_tree` refuse, starting with fit 1. It also resolves review #2's
N5 — the design record lands in git.

**All three commits live on the SAME feature branch, and the PR is merged with `--merge`, NEVER
squashed.** This is not a preference: commit 1's SHA is stamped into four wheel artifacts and
commit 2's into six research directories, and a squash rewrites both, leaving every one of those
citations unresolvable.
It is the same reason 4.73.0 was merged rather than squashed so `metrics.json`'s `run_commit` stayed
resolvable — and the reason `4b15365` and `94e05d1` both survive on `main` today.

Nothing else is committed or staged; `/final-review` runs before each commit.

**Superseded paragraph removed (review #4, Q3).** An earlier revision of this section stated the
opposite — that the spec is never committed — on a misreading of the 2026-08-02 instruction. The
owner clarified it the same day: *"we never commit specs and plans by themselves, to go in with the
relevant code changes. nothing new in that pattern either, it is what we have done for months."* The
rule bans a **standalone** spec/plan commit, not inclusion in the feature commit — consistent with
the 109 specs and 109 plans already tracked, and with §8b's spec being cited here by path. See the
paragraph above for what actually happens.

## 10. Risks and open questions

1. **`sc_extended` match count (§5.3)** is inferred. It drives the compute estimate, not the
   decision.
2. **A TF-19 verdict may flip (§6.3).** Planned for.
3. **Confirm the runbook copies weights to x86 before stamping (§5.4).** Reduced from "the one
   unverified assumption with the power to change scope" — the policy is already recorded in
   `_feature_contract.py:37-40`, so only the step needs confirming.
4. **PR 5 is large** — §8b plus four folded workstreams plus four fits. The honest risk of the
   "close it all out" directive, recorded so the tradeoff stays visible.
5. **Prevalence figures** in §2 are measured on `canonical_scene()` and one match per provider. The
   away-share component is structural; magnitude is scene-dependent.
6. **F2 — CLOSED 2026-08-02, executed on both models.** Built from the shipped source body with the
   §3/§3.1 edit applied by source rewrite (not hand-transcribed, and deliberately not by
   pre-reflecting the input, which would have made the comparison tautological):

   ```
   xCross  PRE-FIX  : 3 sign flips; space_controlled 328.17391 vs 310.43478; grid inputs differ
           POST-FIX : max abs delta over 16 features = 0.000e+00
                      _dominant_region_area inputs BIT-identical, output identical -- GRID UNTOUCHED
   xShot   PRE-FIX  : 12 sign flips (theta, GK_theta, DefAngle_0..4, OffAngle_0..4), max delta 6.124
           POST-FIX : max abs delta over 27 features = 0.000e+00, 0 sign flips
   ```

   The pre-fix leg reproduces §2/§4's cited 328.17 / 310.43 to five decimals, so the harness
   demonstrably sees the defect; with the grid untouched the §3 transform alone drives every
   feature to exactly zero delta. **The old gate 4 was green by construction and §4's replacement
   rationale is correct.** §3.1's pre-transform is also confirmed behaviourally — exactly the 12
   named features, driven to zero, no double-application and no missed site.

7. **Nothing in §3, §3.1 or §4 remains unexecuted.** The only unrun items are inherently build-time:
   the four fits and the §5.4 DGX-vs-x86 probe delta.
