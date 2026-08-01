# ADR-028 orientation defect class — design

**Date:** 2026-07-29
**Status:** design approved (owner), plans not yet written
**Supersedes:** ADR-045 D6 (partially — see §4.3)
**Amends:** ADR-028, ADR-041
**Cycle shape:** 4 PRs. Detection first, then one correction per root cause.

---

## 1. Origin and scope

This cycle began as TF-30 (b) — a cover-shadow RQ1 validation harness. Specing it surfaced a live
ADR-028 defect in `add_cover_shadows`, and verifying *that* surfaced three more root causes plus two
classes of test-fixture defect. **TF-30 (b) itself is not in this cycle** and is deferred until the
seam it would measure is sound; its own open design questions are recorded in §10.

Scope is the orientation defect class: places where a value in one coordinate convention is combined
with a value in another. Two conventions are in play (ADR-028):

- **action-LTR** — SPADL actions; the ACTING team attacks `x=105`.
- **frame-LTR** — `convert_to_frames` output; the HOME team attacks `x=105` in every period.

For an away-team action these are a 180° point reflection apart: `x -> 105-x` **and** `y -> 68-y`.

---

## 2. Evidence

All figures are measured on real matches at HEAD `894458a` (4.68.0). The merge that produced that
HEAD touched none of the files in this spec, so the measurements describe current code.

### 2.1 Instrument

**Physical mirror.** Reflect the frames (`x -> 105-x`, `y -> 68-y`, swap every row's
`team_attacking_direction`), leave the ACTIONS unchanged (they are action-LTR and therefore invariant
under a physical mirror), and pass `home_team_id` = the other team. A correct action-LTR aggregator
emits identical values in both runs.

This was chosen over patch-and-diff deliberately: it requires no "corrected" implementation, so a
mistake in the fix cannot be mistaken for a defect magnitude. For an away row the base run is the
shipped value and the mirrored run is the correct one.

**KNOWN BLIND SPOT — the mirror instrument cannot detect identity-keyed direction inference.**
Swapping `home_team_id` restores the very invariant identity-keying assumes ("the team called home
attacks +x"). Trace an away action: base `same_id(AWAY, HOME)` = False and away attacks low x —
correct; mirrored with `home_team_id=AWAY`, `same_id(AWAY, AWAY)` = True and away attacks high x —
also correct. **An identity-keyed aggregator is invariant under this instrument by construction.**

So every "invariant" reading in §2.2 and §3.1 is evidence about CONVENTION MIXING only, and is silent
about the D1 defect class. Detecting identity-keying needs a second, behavioural gate — §6's D1
invariance test. This limitation is recorded here rather than in a footnote because it is the same
shape as the §2.5 authoring error: an instrument that could not have seen the claim it was cited for.

### 2.2 Result — two valid providers

**These are ONE-MATCH point estimates per provider, not corpus rates.** The three-significant-figure
precision below describes the sample, not the population; quote them as "GS match 10502" / "IDSSE
DFL-MAT-J03WMX", never as a rate. The away-share component is structural and should generalise;
magnitude is scene-dependent (see §10.2).

| Defect | Gradient Sports 10502 | IDSSE DFL-MAT-J03WMX |
|---|---|---|
| `cover_shadows` `max_single_defender_blocking_score` | 90.7% | **100%** |
| `cover_shadows` `n_blocked_receivers` | 77.8% (median 1.0) | **85.0% (median 2.0)** |
| `space_creation` (both columns) | 47.4%, max 0.140 m² | 60.0%, max **0.880 m²** |
| `xt_gk` composite | 19.0% | **0%** |
| `gk_completion` | 17.4% | **0%** |
| `restart` `enriched_start_x` | 2.55%, max **101.24 m** | 1.11%, max **99.58 m** |

Percentages are the share of AWAY rows in each aggregator's own domain whose value changes. Not-home
rows are 801/1414 (GS) and 718/1363 (IDSSE). On GS, `acting_team_attacks_rtl` resolves True on
**791** of those 801 — the other 10 are ADR-027 NaN-team GS null-actor duel/foul rows, which are
not-home by the `ids_match` test but have no away team to flip. On IDSSE flip is True on exactly
718/718.

### 2.3 Controls

**Valid control (IDSSE):** `add_pitch_control` moved `0%`, max `2.66e-15`, on the same rows where
`n_blocked_receivers` moved on 85%. IDSSE has dense 25 Hz tracking, so pitch control's degenerate
fallback never fires.

**In-aggregator control (both providers):** inside the same `add_cover_shadows` call, on the same
rows, `blocking_score` moved `4.4e-16` and `blocked_threat_fraction` `5.6e-17`. Both are
passer-INDEPENDENT (`compute_blocking_score` takes no passer); the two columns that moved are
passer-dependent. Same call, same frames — the instrument validates itself from the inside.

**`add_pitch_control` is NOT a valid control on GS or SkillCorner.** ~half of real query points return
exactly `0.5` — Spearman's documented degenerate/no-information fallback — and that fallback is not
mirror-symmetric. Measured: GS 55/79 rows exactly 0.5, control max 0.5. This is already recorded in
the repo (`test_action_ltr_mirror_invariance.py`'s `_OBSO_MIRROR_TOL` comment: pitch control returns
exactly 0.5 at one point and 1.0 at its mirror). A single-point sample is unsuitable; the
surface-integral quantities above are not.

### 2.4 The SkillCorner leg is VOID — do not reuse its numbers

Measured: SkillCorner frames from `scripts/_loader_pining` carry `team_attacking_direction` NULL on
**999,534 / 999,534 rows (100%)**, and `acting_team_attacks_rtl` returns **True on 0/1197 actions**.
No reprojection ever runs, so defective and fixed code are byte-identical and a mirror measurement
there measures nothing. Compare GS (flip True **791**/1414) and IDSSE (**718**/1363), both valid.

Root cause: `scripts/_loader_pining.py:477-479` requests `output_convention="absolute_frame"` and never
orients; `silly_kicks/tracking/skillcorner.py:282` sets `df["team_attacking_direction"] = None`
unconditionally and labels only inside the `output_convention == "ltr"` branch. Gradient Sports
escapes because `silly_kicks/tracking/gradientsports.py:129` runs `direction.finalize_orientation`
BEFORE its convention branch, so even its `absolute_frame` output is labelled.

This void is itself root cause 4 (§3.4).

### 2.5 Two authoring errors, recorded

1. **Predicted the wrong sign.** The convention-broken fixture was predicted to produce absurd values;
   it produces *more plausible* ones. Metrica action 1 emits `nearest_defender_distance` **1.029355 m**
   where the true value is **18.328738 m**. 1.03 m reads like tight marking. That is why it survived.
2. **Reported results from a leg whose control had already failed.** The SkillCorner control changed on
   94% of rows and was explained away as the degenerate-0.5 fallback — correct for GS, wrong for
   SkillCorner. An adversarial confound pass caught it. The void was caused by precisely the
   silent-no-flip class identified two steps earlier as the highest-value fix, applied to everything
   except the instrument itself.

Both are recorded because the second one is the argument for D2 (§4.2).

---

## 3. Root causes

### 3.1 RC1 — `add_cover_shadows` uses a raw action-LTR passer

`silly_kicks/tracking/features.py:3698` and `:3861-3864` both build
`passer_xy = (float(row["start_x"]), float(row["start_y"]))` and pass it against frame-coordinate
player positions. Neither site calls `acting_team_attacks_rtl`.

`silly_kicks/tracking/_cover_shadows.py:1164-1168` is the defect in one screenshot: it reprojects the
RECEIVER to action-LTR for the xT lookup (`q_x, q_y = ... 105.0 - recv_x ...`), then two lines later
builds `receiver = np.array([recv_x, recv_y])` in raw FRAME coords and differences it against the
action-LTR `passer`. The module knows the distinction and mixes conventions anyway.

**Scope, measured per column:** `n_blocked_receivers` and (cheap-path only)
`max_single_defender_blocking_score` are affected; `blocking_score`, `blocked_threat_fraction` and
`n_potential_receivers` are not (passer-independent). Under `detailed=True` the max-single column is
computed by the pitch-control counterfactual and is also unaffected.

The atomic mirror at `silly_kicks/atomic/tracking/features.py:1169` is a thin rename-and-delegate
adapter and inherits the fix; it is not a third site.

**Also in RC1:** `_cover_shadows.py:1030` keys `attacking_toward_high_x` on
`same_id(attacking_team_id, home_team_id)`. On canonical converter frames identity and direction agree
**by construction** (`finalize_orientation` sets the label FROM the home flag), so this is fragility
rather than a live defect on that input, and it is re-keyed under D3.

**That verdict is REASONING, not measurement.** A 5.68e-14 mirror reading was originally cited here as
evidence and has been WITHDRAWN: the mirror instrument is structurally blind to identity-keying
(§2.1), so 5.68e-14 is the expected reading whether the keying is safe or not. The claim is unproven
by measurement and stands only as an argument about canonical frames — which matters, because RC4
proves non-canonical frames ship from the library's own loader, and that is exactly where
identity-keying goes live. §6's D1 invariance gate is what will actually test it.

### 3.2 RC2 — `_gk_geometry` writes frame coordinates into action-LTR quantities

`silly_kicks/tracking/_gk_geometry.py:149-151` (`_tracking_gk_xy`) and `:261` (`_tracking_ball_xy`)
read a frame `x`/`y` and write it straight into an action-LTR quantity, with no reprojection.

The pairing is the tell: the sibling `_tracking_gk_xy_detected` at `:167-172` carries a docstring
explicitly stating it **re-projects to action-LTR (ADR-028)** because "without this, away-team origins
land at the wrong end of the pitch", and does so at `:220-221`. One sibling is correct and documented;
the other is not.

**Consumers:** `add_xt_gk`, `add_gk_completion`, `add_restart_coordinates`.

**Why the magnitude is bimodal.** The clamp at `:150` (`if gx <= _GOAL_AREA_DEPTH`) means a
mis-oriented away keeper lands at high `x`, FAILS the clamp, and falls through to the rule point
`(5.5, 34)` — which is orientation-independent. So the defect usually converts into a systematic
*loss of the tracking tier* for away teams rather than a wild coordinate. `_tracking_ball_xy` has no
such clamp, which is why restart origins move by up to a full pitch length.

**Why the rate differs by provider.** It is governed entirely by ADR-024 native-origin trust, not by
geometry: IDSSE/sportec natives are trusted and present, so the tracking tier never runs (0%); GS has
~67% NaN native goal-kick origins, so it runs often (19.0%, which is exactly 4/21 of its scored away
rows — the tracking-tier count).

**Weights consequence.** `prepare_gk_completion_training_data` shares this seam, so the bundled
`GkCompletionModel` was fit with away-team origins systematically routed to the rule-point fallback.
That is a train/serve distribution difference aligned with team, not added noise. **Retrain required.**

### 3.3 RC3 — `_space_creation` applies an unrotated EPV multiplier

`silly_kicks/tracking/_space_creation.py:216` builds
`obso_multiplier = effective_transition * epv_interp` and applies this attack-LTR multiplier to
frame-LTR pitch control. `_compute_space_creation_for_action` (`features.py:5557`) accepts
`home_team_id` and never calls `acting_team_attacks_rtl`; the parameter is dead (verified by
output-identity across `home_team_id` ∈ {HOME, AWAY, 999}).

Structurally the two emitted columns are exchanged for away actions. On real data they are
numerically close (GS p90 3.1e-5 m²), so the practical impact is small even though the labelling is
wrong — but IDSSE reaches 0.880 m², 6× GS, so "immaterial" is not a safe generalisation.

**CLAUDE.md is factually wrong, and about TWO of the six aggregators it names.** Its ADR-028 bullet
states that obso, `space_creation`, pausa, `player_influence`, `cover_shadows` and `gk_influence` are
"now reprojected at their own query seam". Two corrections are needed:

- `space_creation` — **false outright**. ADR-041 gave it only an unconditional `axis=(0,1)` opponent
  mirror; it never calls `acting_team_attacks_rtl` (RC3).
- `cover_shadows` — **half true, which is worse than false** because it reads as settled. The
  RECEIVER is reprojected at its query seam (`_cover_shadows.py:1164`); the PASSER never is (RC1).

**The correction ships in PR 1, not with the code fixes.** PR 1 lands `xfail` markers documenting
these defects; shipping it while the documentation still asserts they are fixed reproduces the
stale-claim failure mode this repo hit as recently as 4.68.0, where four registration sites went
stale simultaneously (code docstring, CLAUDE.md, TODO and the ADR).

### 3.4 RC4 — the pining loader ships SkillCorner frames unoriented

See §2.4 for the measurement and the two source anchors. This is a corpus-integrity defect, not a
value-correctness one: the library's own research/calibration path silently disables ADR-028 for an
entire provider.

**Verified consumer list** (by reading each trainer's data path):

| Consumer | Path | Affected |
|---|---|---|
| `train_xshot_occurrence` | `load_matches` -> `prepare_xshot_training_data` -> `_gk_resolve.defended_goal_x` | **No** |
| `train_xcross_attempt` | `load_matches` -> `prepare_xcross_training_data` -> `_xcross_attempt._build_goal_map` | **No** |
| `train_ghost_gk` | `--data-dir` parquets -> `_ghost_gk._defending_goal` (GK mean-x) | **No** |
| `train_gk_retention` | marts-based, no frames | **No** |
| `train_gk_completion` | `load_matches` -> `prepare_gk_completion_training_data` -> reads `flip` | **Yes** |
| `calibrate_tracking_defaults` (TF-24) | `load_matches` -> action-coupled features | **Yes** |

The three large trained models are insulated because each resolves the defended goal **geometrically
from GK mean-x** — via THREE separate implementations (`_gk_resolve.defended_goal_x:323-353`,
`_xcross_attempt._build_goal_map:257`, `_ghost_gk._defending_goal:814-818`), none of which reads an
orientation label. That geometry works in all three is the existence proof behind D1/D3.

Two precision notes, because the single-resolver story is tempting and wrong:

- `_xcross_attempt.py` mentions `_defended_goal_x` only in a **docstring** (`:259`); it calls
  `_build_goal_map` (`:306`, `:697`).
- `_ghost_gk.py:843` carries an identity **fallback** on its `.get()` default:
  `goal_x = _defending_goal.get((gid, pid, gk_team), 0.0 if same_id(gk_team, home_team_id) else _FIELD_LENGTH)`.
  It is unreachable while the `(game, period, team)` group has a GK row (the loop's GK mask at `:830`
  is the same mask that built the dict at `:814`), which is why the **No** verdict holds — but it IS a
  D3 re-key target. `_gk_resolve.py:334-337` already warns in-code that "a second implementation is a
  fork that can disagree with the first"; there are three.

---

## 4. Architecture decisions

### 4.1 D1 — orientation precedence: label -> geometry -> fail loud. Never identity.

Three sources exist and they are not equally good:

| Source | Mechanism | Fails when |
|---|---|---|
| Identity | `same_id(team, home_team_id)` | frames are not home-attacks-right |
| Label | `acting_team_attacks_rtl` | label absent -> silent all-False (RC4) |
| Geometry | `defended_goal_x` | needs neither identity nor label |

**Rule: attacking direction is read from the frames, never inferred from team identity.**
`home_team_id` may answer "which team is home"; it must never answer "which way do they attack".

### 4.2 D2 — the orientation seam fails loud

`silly_kicks/tracking/_action_orientation.py` returns an all-False flip silently. It has **four**
`return flip` sites, and **three of them are silent failures**:

| Line | Condition | Status |
|---|---|---|
| `:145` | `len(actions) == 0 or len(frames) == 0` | **split it**: silent iff `len(actions) == 0`; empty FRAMES with non-empty actions is a caller error and warns |
| `:147` | `"team_attacking_direction" not in frames.columns` | **silent failure** — absent column |
| `:155` | `"team_id" not in keys or "period_id" not in keys` | **silent failure** — join keys don't align |
| `:160` | `players.empty` after the `notna()` filter | **silent failure** — column present, all-null |

Its docstring justifies the no-flip default with "such actions produce NaN geometry anyway" — true for
a per-action unresolvable direction, false for all three of these, where actions link fine and simply
come out unflipped.

Measured on one canonical y-asymmetric away action, labelled vs the same frames with the column
dropped: `nearest_defender_distance` **7.6158 -> 19.6977**, `receiver_zone_density` **1 -> 0**,
`defenders_in_triangle_to_goal` **1 -> 0**.

The measurement exercised `:147`; **RC4 exits through `:160`** (SkillCorner sets the column to `None`,
so it exists and is entirely NA). All three return the identical all-False Series, so the measurement
transfers to each.

**Specify the fix by OUTCOME, not by enumerated condition:** the seam warns whenever it returns an
all-False flip for any reason other than there being no actions to flip. The single carve-out is
`len(actions) == 0` — deliberately narrower than `:145`'s current disjunction, because when a rule is
specified by outcome its one carve-out is where the rule leaks, and "no frames but plenty of actions"
is a caller error rather than a no-op. An
enumerated-condition fix implemented literally would leave `:155` silent — and `:155` is reachable by
passing frames whose join keys do not align with the actions, producing exactly the failure D2 exists
to catch. Do NOT assert `:155` is unreachable without evidence: this repo has two recent
"unreachable" claims that needed qualifying (ADR-043's deliberately-ungated site, and
`_ghost_gk.py:843` inside this very spec).

**Decision: warn, do not raise**, with a dedicated public warning category and CI escalation — the
ADR-045 `on_unknown="warn"` precedent. A raise has no reachable remedy for a consumer legitimately
holding absolute frames (ADR-029 documents that they exist), and RC4 proves the library itself ships
them. Fail-closed lives in the CI gate, not the runtime call.

This is the highest-value single change in the cycle: it is what would have caught RC4 and the void
measurement in §2.4.

### 4.3 D3 — `home_team_id` retires by disuse; it is not removed

Supersedes ADR-045 D6's "other action-coupled aggregators still take `home_team_id` by design", on
measured evidence (the absolute-frames divergence in §2.4).

`home_team_id` stays in signatures — no breaking change mid-cleanup — but stops being READ for
direction. The re-key targets are the **7 AGGREGATORS** the sweep classified as fixture artifacts and
which need a code change (of 8 such verdicts; `add_pausa` needs none — §10.4). They are byte-identical
on converter output (verified, worst 8.53e-14), so this costs no re-materialize. Once nothing reads
it, removal is a mechanical no-risk change in a later major, and the §6 gate is what proves nothing
reads it.

**Expect the enumeration to return MORE than 7, and do not treat that as a gate bug.** Those 7 are
aggregators; the gate enumerates *code sites*, and at least one lies outside that set —
`_ghost_gk.py:843`'s identity fallback (§3.4), in a module the sweep verdicts scored as unaffected
because the fallback is unreachable. An enumerating gate returning more members than the hand-built
list is the ADR-043 idiom working as intended.

**Honest limit:** geometry resolves the absolute-frames case. It does NOT rescue
`snapshot_to_tracking_frames` output — each frame there is oriented for its own action, so a
per-`(game, period, team)` aggregation mixes orientations. That case is structurally ambiguous under
every key and D2 is its only correct answer. Re-keying was measured NOT to fix it: on
`add_defensive_line` with both teams labelled `"ltr"`, a direction key leaves away rows wrong and
makes the home row worse (62.0 vs a correct 80.0).

### 4.4 D4 — fixtures: correctness in the shared helper, coverage as a parameter

`tests/tracking/_provider_inputs.py::synthesize_actions` stamps `start_x`/`start_y` from the frame
row's raw `x`/`y` (`:169-178`), so with `play_left_to_right`-normalised frames its away actions are in
FRAME convention, not action-LTR. Measured: 9/10 actions equal the actor's raw frame position exactly,
**0/10** equal the point reflection. The team mix is 9:1 in every fixture, but WHICH team dominates is
provider-dependent — sportec, skillcorner and gradientsports are 9 home / 1 away, while **metrica is
1 home / 9 away** (which is exactly why metrica is the fixture whose baseline moves).

Consequences: an ADR-028 passer defect is UNEXPRESSIBLE on this fixture (raw `start_x` is accidentally
correct), and a CORRECT implementation is wrong on it for away rows.

- **Convention — fixed in place, unconditionally.** Measured blast radius: metrica moves 9 rows;
  sportec, skillcorner and gradientsports move **zero** (their single away action is the NaN
  keeper_save). One non-empty baseline diff.
- **Team balance — an opt-in parameter** (`balance_teams=False` default). Changing sampling rewrites
  every action in every fixture and destroys all four baselines' regression history for no correctness
  gain. The mirror gate opts in; nothing else changes.

Rationale: a correctness defect in a shared helper belongs in the shared helper; a sampling policy
belongs at the call site. The 9:1 skew is an artifact of `drop_duplicates(...).head(n)` picking an
arbitrary first-listed player per frame — today's low exposure is sort order, not design.

### 4.5 D5 — GS fixture direction labels are derived, not hardcoded

`_provider_inputs.py:71` sets `"team_attacking_direction": "ltr"` as a scalar across every row, so
both teams claim to attack the same way and no GS fixture can exercise any orientation path.

Derived from geometry: GK median `x` is team 100 -> 20.52, team 200 -> 60.52 in both periods (outfield
medians 32.5 / 72.5), giving `{100: "ltr", 200: "rtl"}`. Ends do not swap between periods — it is a
synthetic fixture. Caveat: team 200's keeper sits ~8 m past halfway rather than in a goalmouth, so the
"keeper marks the defended end" inference is weaker here than on real data; what is unambiguous is
that 200 is the high-x team and 100 the low-x team, so they cannot both attack the same way.

**Measured: this fix is LATENT on its own** — exactly one action flips (action 9, `type_id=14`
keeper_save), its `add_action_context` output is already all-NaN, and every column diffs on 0 rows.
its baseline does not move. Under D4's convention fix it stops being latent: reflecting away
actions requires knowing which team attacks rtl, so a wrong label produces wrong coordinates directly
(GS action 9's `start_x` becomes 44.31 instead of 60.69). One assertion — synthesized actions are in
action-LTR — guards D4 and D5 together.

---

## 5. PR decomposition

### PR 1 — Detection (test-only; value-neutral for library consumers)

1. D4 + D5 fixture repair; regenerate `metrica_expected.parquet` (9 rows); confirm the other three
   baselines are byte-unchanged.
2. D2 fail-loud seam + public warning category + CI escalation, covering all three silent branches
   (§4.2).
3. The 33-aggregator registry (§6) — **BOTH gates**: Gate A (ADR-028 mirror) and Gate B (D1
   `home_team_id` invariance), with the four root causes registered `xfail(strict=True)` and the
   per-entry `home_team_id_role` declaration. Gate B is not optional polish: without it D3 is enforced
   only by static enumeration, which proves nobody calls `home_team_id` and not that nobody depends
   on it.
4. **The CLAUDE.md ADR-028 correction (§3.3)** — lands HERE, with the `xfail` markers it describes,
   not with the code fixes.
5. **Own `_defensive_line.py:73`.** It is a public export with three consumers
   (`_packing.py:166`, `_gk_influence.py:398`, plus `_defensive_line` itself) and is the named
   incomplete-repair risk in §10.5. PR 1 does not re-key it, but the §6 registry MUST enumerate it and
   its three consumers as one D3 unit, so that a later partial re-key fails the gate rather than
   shipping. A named risk with no owner is how partial repairs ship — which is that risk's own point.

**Why first:** a gate that lands after the fixes arrives green and is never observed failing — the
green-by-construction trap TF-30 (a) existed to clean up. Landing it red-via-xfail means each
correction PR deletes exactly one marker, and `strict=True` makes that mandatory.

### PR 2 — RC1 cover shadows

Reproject the passer into FRAME coords at `features.py:3698` and `:3861-3864`; the xfns cache key
must be built from the REPROJECTED passer. Direction of reprojection: passer -> frame coords, because
everything downstream of `passer_xy` is frame-convention and the one place that steps out to
action-LTR (`_cover_shadows.py:1164`) already reprojects itself.

**Re-materialize trigger:** `n_blocked_receivers` and `max_single_defender_blocking_score` change on
~40-45% of all actions. `cover_shadow_xfns` is a factory in no default xfn list, so **no forced VAEP
retrain**.

### PR 3 — RC2 + RC3 (value corrections)

Two independent CODE fixes, kept as separate commits so either can be reverted alone. **The
`GkCompletionModel` retrain is NOT here — it moved to PR 4** (weights have their own lifecycle per
ADR-011; bundling them means a failed ECE/slope gate holds two unrelated value corrections hostage).

- **RC2:** reproject in `_tracking_gk_xy` and `_tracking_ball_xy`.
- **RC3:** point-reflect the EPV/threat grid (`[::-1, ::-1]` — BOTH axes) keyed on
  `acting_team_attacks_rtl`. An x-only mirror is exact only for a y-symmetric grid, which the
  synthetic ramp IS — this is the incomplete repair ADR-041 already shipped once, so the gate fixture
  must be deliberately y-asymmetric.

> **RELEASE CONSTRAINT — PR 3 MUST NOT SHIP IN A RELEASE WITHOUT PR 4.**
> Today `GkCompletionModel` is train/serve *consistent*: fit on fabricated away origins, served on
> fabricated away origins. PR 3 corrects the serving geometry while the bundled weights are still the
> old ones, which **introduces** a train/serve skew that does not exist today (GS: 17.4% of away
> `gk_completion` rows change, max 0.063). The window is acceptable in repo history and NOT acceptable
> in a published artifact. Tag only after PR 4.
> The separate xT-GK v2 hazard also lives here: its coherence check raises on a half-refreshed mart,
> so the mart refresh and the weights must land together.

**Atomic mirrors — verdicts, not silence.** All three corrections propagate to atomic without separate
edits, verified by reading each surface: `atomic.tracking.features.add_cover_shadows:1169` and
`add_xt_gk:448-465` are thin adapters that synthesize endpoints, delegate to the standard aggregator
and drop the synthesized columns; there is **no** atomic `add_space_creation` at all, and atomic
re-exports the *same* `space_creation_xfns` function object (`:55`, `:131`), so it inherits RC3 by
identity. No atomic-specific work in any PR.

### PR 4 — RC4 loader / corpus integrity + all weights work

Orient SkillCorner frames in `scripts/_loader_pining`, or request `output_convention="ltr"`.

Then, in one place, everything weights-and-corpus — which is why the retrain belongs here rather than
in PR 3: **both** items retrain or re-assess the *same* model, and both depend on a corrected corpus.

1. **Retrain `GkCompletionModel` `default`** on the RC2-corrected seam (PR 3 is its prerequisite).
   Must clear its recorded ECE / reliability-slope gate; if it does not, PR 3 is held unreleased
   (§5 PR 3 release constraint) rather than shipped with skew.
2. **Re-assess the `skillcorner` variant**, which is compromised by BOTH root causes at once — RC2's
   seam and RC4's unoriented frames. Confirm its recorded corpus provenance before acting (§10.3).
3. **Re-assess TF-24 SkillCorner calibration corpora.**

xS/xCross/ghost are verified unaffected (§3.4) — do not retrain them on this basis.

---

## 6. The mirror registry (PR 1)

Registry-driven parameterized gate in the `PURITY_ENTRIES` / id-scalar-registry idiom. One
`MIRROR_ENTRIES` table over all **33** registered `tracking` `add_*`, each declaring input construction,
a per-column mirror class, a tolerance with a recorded reason, and a non-vacuity anchor. Two
meta-assertions pin the registry to `tracking.__all__` in both directions.

**TWO gates, because one instrument cannot see both defect classes.**

**Gate A — the ADR-028 mirror** (§2.1). Detects convention mixing. `home_team_id` is swapped for
EVERY entry that takes it: after a physical mirror, the team attacking +x really is the other one, so
swapping is the semantically correct re-expression of a frame-orientation input. **20 of the 33
`add_*` take the parameter** (measured via `inspect.signature` over `tracking.__all__`). Where an
entry uses it for ATTRIBUTION rather than direction (team scoping, column labelling), the affected
columns are declared `exempt` with a reason — the existing per-column vocabulary already handles this,
so no separate mechanism is needed.

**Gate B — the D1 invariance test.** Detects identity-keyed direction inference, which Gate A is
structurally blind to (§2.1). On FIXED canonical frames, vary `home_team_id` over
`{home, away, a nonsense id}`; **every column declared `invariant` must not move.** A mirror-invariant
column is action-LTR geometry by definition, so it cannot legitimately depend on which team is home —
if a counterexample appears it declares `exempt` with a reason.

Gate B is deliberately simpler than the mirror: it needs no transformed frames, so it never runs an
aggregator outside the `convert_to_frames` home-attacks-right contract. It is also strictly more
discriminating than a no-swap mirror variant, because the nonsense id catches
`same_id(x, home) else ...` branches that a two-team swap can leave looking correct.

**Gate B is what makes D3 real.** §4.3's "no geometry module reads `home_team_id` for direction" is
specified as a static enumeration, and enumeration only proves nobody *calls* it — Gate B proves
nobody *depends* on it. The precedent is exact: the sweep established
`_compute_space_creation_for_action`'s parameter was dead by output-identity across
`home_team_id ∈ {HOME, AWAY, 999}`, which is Gate B applied by hand to one aggregator.

Each entry still declares `home_team_id_role ∈ {direction_only, attribution, unused}` — it documents
intent and drives the per-column exemptions in both gates.

**Per-column mirror classes** (closed vocabulary):

- `invariant` — action-LTR geometry; base and mirror identical. The default.
- `mirrored_pitch_absolute` — deliberately pitch-absolute; must equal its own reflection. At least one
  real case: the shape-graph lateral label, settled pitch-absolute per ADR-045 D5.
- `exempt` — non-deterministic or undefined under mirror, WITH a written reason.

**Per-entry tolerance, never global.** `_OBSO_MIRROR_TOL = 0.02` exists because pitch control is
genuinely not mirror-symmetric (measured 1.3e-2); the ghost entry's `_GHOST_Y_TOL` is 3.0 m against a
measured 1.26 m of real model asymmetry. Both are tolerances sized ABOVE their measurement, with
headroom — record each entry's tolerance and its measured basis separately, so a future re-fit that
moves the measurement does not silently consume the headroom. One global tolerance either catches
nothing or flags correct code.

**Three vacuity guards, each addressing a way these gates die green:**

1. *Computed nothing* — invariant columns must be non-null on the away rows specifically.
2. *No away rows* — fixture-level assertion on away-action count (this is why D4's balance parameter
   is load-bearing, not tidiness).
3. *y-symmetric scene* — the fixture is y-asymmetric by construction, asserted. An x-only repair is
   exact on a y-symmetric scene; ADR-041 shipped exactly that incomplete repair and only a
   y-asymmetric oracle caught it.

**Base and mirror legs must NEVER share a `PitchControlCache`.** It keys on frame IDENTITY
(`game_id, period_id, frame_id, team, method, params, ball_position, decompose`) and excludes player
positions, so a mirrored frame carrying its twin's identity would be served the base leg's surface and
every pitch-control-family test would pass at exactly zero difference. This is the ADR-043 ghost-frame
failure mode reappearing inside a test harness.

**CI placement:** not `@pytest.mark.slow`. ADR-023 scopes `slow` to expensive AND platform-invariant
work and explicitly keeps behavioural-contract guards on every leg. Budget basis: the existing
5-aggregator gate is **7 tests in 1.63s** including a trained-model load; the plan measures the real
number rather than extrapolating, since the richer multi-frame fixture the pre-window and goalmouth
aggregators need will cost more.

**Also gated (D3):** no geometry module reads `home_team_id` for direction — an enumerable property in
the ADR-043 registry idiom.

---

## 7. Hyrum / re-materialize / retrain summary

| PR | Value change | Re-materialize | Retrain |
|---|---|---|---|
| 1 | none (test-only) | no | no |
| 2 | `n_blocked_receivers`, `max_single_defender_blocking_score` on ~40-45% of rows | **yes** | no |
| 3 | `xt_gk*`, `gk_completion`, `enriched_*` restart coords, `space_created_m2` / `space_denied_m2_opponent` | **yes** | no (moved to PR 4) |
| 4 | none in the library; research corpora change | n/a | **`GkCompletionModel` `default`** + re-assess SC variant + TF-24 |

**PR 3 and PR 4 are release-coupled** (§5): PR 3 alone would introduce a `GkCompletionModel`
train/serve skew that does not exist today. Tag only once both have landed.

No VAEP retrain is forced by any PR: `cover_shadow_xfns` and `space_creation_xfns` are factories in no
default xfn list, and `xt_gk` is opt-in.

---

## 8. ADR requirements

- **New ADR** for this class: records D1-D5, supersedes ADR-045 D6, amends ADR-028's repair table
  (which wrongly lists `space_creation` and `cover_shadows` as repaired) and ADR-041.
- CLAUDE.md ADR-028 bullet corrected (§3.3).

---

## 8b. PR 5 — the chiral goal-relative transform (FOUND BY THE GATE, deferred to its own cycle)

Gate A surfaced a root cause the audit missed, in a DIFFERENT defect family, and it does not fit
any of PRs 1-4.

**What it is.** `silly_kicks/tracking/_geometry.py` has `to_goal_relative_x` and
`to_goal_relative_vx` and **no `to_goal_relative_y`**. So `goal_x=105` maps `(x,y) -> (105-x, y)` --
an x-only MIRROR, determinant -1 -- while `goal_x=0` is the identity, determinant +1. The two ends
therefore use frames of OPPOSITE HANDEDNESS. Composed with ADR-028's point reflection this leaves
every RADIAL feature byte-identical and NEGATES every BEARING.

This is the ADR-041 incomplete-repair pattern ("an x-only mirror where a point reflection is
required") one layer down: in the goal-relative transform rather than the action/frame transform.
The fix is to make it the 180-degree point reflection `(x,y) -> (105-x, 68-y)`, so both ends differ
by a ROTATION.

**Measured** on `canonical_scene()`: xS **12 of 27** features flip sign, output 0.01113 -> 0.01293
(+16.2%); xCross **3 of 16**, 0.00168 -> 0.00113 (-33.0%). Those are exactly the counts ADR-037
records as "sign-inconsistent", reached independently from the ADR-028 side -- i.e. PR-S118's
retrain addressed the SYMPTOM while the transform stayed chiral. Consequence in production: one
physical scene scores differently depending on which END the acting team attacks -- a systematic
home-vs-away split INSIDE a single match. `_geometry.py`'s docstring claim that "LTR and RTL frames
map to identical feature values" is FALSIFIED by measurement.

**Rides with it:** `_xcross_attempt._dominant_region_area`'s grid is
`np.arange(1.5, 68.0, 3.0)`. 105/3 = 35 exactly so x tiles the pitch and centres on 52.5; 68/3 =
22.67 so y runs 1.5..67.5 and centres on **34.5**, and under `y -> 68-y` a centre at 1.5 maps to
66.5, which is not a grid point (`space_controlled` 328.17 -> 310.43, 5.4%). It ships in the SAME PR
because `space_controlled` is xCross model feature #3 -- splitting them means retraining the same
model twice.

**Why it cannot be folded into PR 4.** Both bundled artifacts carry `chirality` AND
`feature_contract` stamps (verified in their `metadata.json`). Changing the transform changes
feature VALUES -> the ADR-050 contract fingerprint mismatches -> `load()` RAISES; changed features
change model OUTPUTS -> the ADR-040 chirality fingerprint mismatches -> `load()` RAISES. Both are
fail-closed by design, so **the code fix, the retrain and the re-stamp are ATOMIC** -- a code-only
PR would ship a wheel whose own bundled weights refuse to load. That is stricter than PR 3/PR 4's
release coupling, which is a silent skew rather than a hard error.

**Blast radius (verified, and NARROWER than first reported).** Only two consumers import the
helpers: `_xcross_attempt.py:158` and `_xshot_occurrence.py:181`. **`_ghost_gk` does NOT** -- it has
its own `_defending_goal`; an early report speculated otherwise and would have widened the retrain by
a model. So: **silly-kicks retrains** (`_xshot_weights/`, `_xcross_weights/` ship in the wheel);
**the lakehouse re-materializes**, and retrains only if it fit models on those columns --
`xshot_occurrence_xfns`/`xcross_attempt_xfns` are wired into `pre_shot_gk_full_default_xfns` ONLY.
The rho retention model uses marts features, not xS/xCross, and is unaffected.

**Sequencing.** DGX work. A natural SESSION companion to the already-queued ghost-GK re-fit onto the
canonical box constant (also DGX, also a re-stamp) -- but NOT the same PR, since ghost does not
consume the chiral transform.

## 9. Out of scope

- **TF-30 (b)** RQ1 validation — deferred until PR 2 lands.
- Removing `home_team_id` from signatures (D3 leaves it in place deliberately).
- The `snapshot_to_tracking_frames` orientation ambiguity — D2 is its answer; a structural fix is a
  separate question.
- The 8 AGGREGATORS the sweep verified as fixture artifacts — byte-identical on converter output
  (worst 8.53e-14). 7 are re-keyed under D3; `add_pausa` needs no code change (§10.4). None is
  corrected as a defect.

---

## 10. Open questions and risks

1. **TF-30 (b)'s own design questions survive** and are unanswered: the RQ1 lane target is CIRCULAR
   for the failure class (`spadl/base.py:59-64` overwrites a pass's `end_x`/`end_y` with the NEXT
   action's start, so for a failed pass `end_xy` IS the interceptor's position; and
   `resolve_next_touch_receiver` returns `<NA>` on opponent-next, deleting the failure class
   entirely). The receiver definition must be outcome-blind. Also open: decision rule, corpus window,
   and that precision is a lower bound by construction.
2. **Prevalence is measured on one match per provider.** The away-share component is structural and
   should generalise; magnitude is scene-dependent.
3. **The `skillcorner` `GkCompletionModel` variant's corpus provenance** should be confirmed against
   its recorded metadata before PR 4 acts on it.
4. **`add_pausa` needs no code change** — its sweep finding was a false docstring plus an undocumented
   `epv_grid=` (attack-LTR) vs `transition_grid=` (frame-absolute) API split, unreachable today.
5. **`_defensive_line.py:73` is a public export with three consumers** — a partial D3 re-key there is
   the incomplete-repair pattern this repo has shipped before. **OWNED: PR 1 item 5** — the registry
   enumerates it and its consumers as a single D3 unit, so a partial re-key fails the gate. This is no
   longer an unowned risk.

---

## 11. AMENDMENT (2026-08-01) — PR 4 re-planned after ADR-052 / 4.72.0

PR 4 was implemented, reviewed, **parked**, and is now **re-planned from scratch on merged main**.
Two things happened between the original decomposition and today: an adversarial `/final-review`
falsified several of PR 4's own claims, and a parallel session shipped ADR-052 (4.72.0), which
independently built four of PR 4's five work items — three of them better.

This section records what changed. §§1–10 are left as written: they are what was believed at spec
time, and §2.5 already establishes that convention.

### 11.1 A FABRICATED measurement, and the retraction it caused

PR 4's implementation extended RC4 from SkillCorner to **IDSSE**, reporting
*"IDSSE 1.0000 → 0.0000 unlabelled, flip fraction 0.3155"* **as measured. It was not.**

`sportec.py:137` calls `finalize_orientation` **unconditionally, BEFORE** the `output_convention`
branch at `:194`. IDSSE frames were therefore always labelled; `absolute_frame` and `ltr` are
byte-identical there, at 0.0% unlabelled. Independently reproduced on the committed IDSSE slice.

The mechanism of the error: the smoke test ran only **after** the loader was changed, so it measured
the post-fix side and the pre-fix side was **assumed** from SkillCorner's, because both call sites
carried the same override keyword. **§3.4 of this spec was correct all along** — its title and its
consumer table name SkillCorner alone. The extension was invented at implementation time.

Worse, §2.2's own IDSSE datum — *"On IDSSE flip is True on exactly 718/718"* — is impossible if those
frames were unlabelled, and was never checked. On the fabricated premise PR 4 **retracted the §2.2
IDSSE column as "provisional"**, propagating that retraction into CHANGELOG, CLAUDE.md, TODO.md,
ADR-051, a research findings doc, a loader comment, a test docstring and a commit message.

**§2.2's IDSSE column stands. The retraction is withdrawn.** Rule carried forward: an `X → Y` claim
requires both sides measured; a sibling call site is not evidence; and retracting someone else's
measurement demands a stronger instrument than asserting a new one.

**CLOSED by measurement (2026-08-01).** §2.2's "718/718" is no longer merely un-refuted — it has been
independently reproduced. A full-frame run of both sides
(`docs/research/adr028_rc4_orientation/`, `tracking_limit=None`) returns IDSSE **`n_flip_true: 718`
of 1,363 actions = `0.5267791636096845`, byte-identical before and after RC4**. Two instruments, one
number.

**And it explains where `0.3155` came from, which is worth more than the vindication.** That figure
was never invented — it is `430/1363`, the *real* IDSSE flip **under a `tracking_limit=3000` cap**
that the measurement did not record. The orientation lookup groups non-ball frame rows by
`(game_id, period_id, team_id)`, and an action whose key is missing **defaults to no-flip silently**
(`OrientationUnresolvedWarning` fires only when *nothing* resolves), so a truncated frame set
*depresses* the flip fraction. The fabrication of §11.1 was therefore a **real number attached to an
invented transition** — the harder variant to catch, because the digits survive any check that only
asks "does this number exist somewhere?".

The same cap silently understated the RC4 headline: SkillCorner's post-fix flip is **0.4728**, not
the 0.2398 first published. The corrected value also stops being anomalous — 0.4728 sits with IDSSE
0.5268 and GS 0.5665 as an ordinary away share, where 0.2398 was half of anything else measured and
should itself have prompted the question.

### 11.2 Four of five work items are superseded by ADR-052

| PR 4 item | Disposition |
|---|---|
| `--mode {rebundle,retrain}` on `train_gk_completion.py` | **Superseded** — main's is stronger |
| Provenance wiring + `ARTIFACT_DRIVERS` registration | **Superseded** — main enrolled all five trainers |
| Startup provenance capture (commit `1c81099`) | **Superseded** — main already does it, plus `run_tree_state` |
| Task 17b (`measure_cover_shadow_argmax_agreement` passer) | **Superseded** — on main line-for-line (`2424608`) |
| **RC4 SkillCorner half** | **SURVIVES** — `_loader_pining.py` untouched by 4.72.0 |

The `--mode` collision is the instructive one, and main's basis is **better**: it asserts the
**served probabilities** moved (`predictions_moved`, reconstructing the sigmoid over two probe design
matrices), where PR 4 asserted a **parameter delta**. PR 4's own reviewers had already found that
defective — `max()` over `coef/intercept/mean/std` lets a *translation* move `mean` by metres and
pass with a coefficient delta of ~3.5e-17. Standardisation absorbs a translation exactly, so served
output is unchanged. A pure coordinate correction is precisely that case, i.e. **exactly the class
this cycle produces.** Main's docstring argues a parameter rule "is wrong in BOTH directions" and
tests both sides.

Four small items of PR 4's remain genuinely additive and may be carried as a follow-up:
`metrics.json` recording the **served** coefficients rather than the probe's (main still writes
`model._coef` while `served` may be `committed`); an actionable `SystemExit` on rebundle drift
instead of a bare `assert_allclose`; `weight_deltas`/`coefficients_changed` in metrics; and test
coverage of the real `save`/`load` branches (main's trainer tests are entirely duck-typed and import
no model).

### 11.3 The retrain must now declare `--feature-space moved`

Main gates `--mode retrain` behind `--feature-space {unchanged,moved}`, and `moved` requires
`--probe-old`: the design matrix the **committed** model was fit on. Its help defines `moved` as
*"a geometry/coordinate correction changed the raw features"* — i.e. feature **values**, not columns.

**This cycle's retrain is unambiguously `moved`.** Two consequences:

1. **The parked weights are not reproducible under main's trainer** and must be regenerated. They
   were produced under a CLI that no longer exists.
2. `--probe-old` cannot be reconstructed from the artifact — `to_dict()` persists only
   `coef/intercept/mean/std/base_rates/gate`, no design matrix. It must be re-extracted from a
   pre-change commit.

**The probe's VINTAGE is the hard part, and "pre-RC2" is the wrong answer** (part-deux review 1).
`--probe-old` must be the design matrix the **committed weights** were fit under. Those coefficients
were fit at **`e3d5e92`** — 2026-06-09, **4.21.0**, the original bundling — roughly fifty releases
before RC2, with several geometry changes in between (ADR-024 amendments, PR-S104 SkillCorner
keeper-origin). A pre-RC2 checkout yields "just before the most recent change", which is not the same
thing. Get it wrong and `predictions_moved` compares the committed model against coordinates **it
never saw either** — meaningless, and nothing reports it.

**A CORRECTION worth reading, because it is this cycle's own failure shape** (part-deux review 2).
Revision 1 of this section named `7e875c8` (4.21.4) as where the weights were "last written",
inferred from `git log -1 -- model.json`. That finds the last **touch**, not the last **fit**:
`7e875c8`'s diff on `model.json` is **purely additive** — `version` 1.0.0→1.1.0 plus
`type_serve_mode`/`type_gate_metrics` — with **no `coef` line changed**. Only two commits have ever
touched that file, and the same holds for `skillcorner/model.json`, so the trap applies to both
variants. A proxy was reported as the thing, exactly as in §11.1.

**`7e875c8` remains admissible — by VALIDATION, not by recency**, and the distinction is the lesson.
CLAUDE.md's PR-S91 bullet records that 4.21.4 attached the gate onto the *loaded committed* model with
coefficients **byte-unchanged**, the fresh fit serving only as a corpus-identity probe. So the
validation this section proposes was already executed there, and already passed.

**How to validate a candidate vintage.** At the candidate commit, **run the trainer** and let its
corpus-identity assertion decide. Do **not** write `--mode rebundle`: that flag is 4.72.0's and does
not exist at 4.21.x (measured at `7e875c8`: 0 occurrences of `--mode`, 0 of `rebundle`). The mechanism
is there regardless — `_CORPUS_IDENTITY_ATOL` appears 9 times and the identity assertion runs
**unconditionally**, because at 4.21.4 rebundle is the only behaviour. Applying HEAD's vocabulary to
an old checkout without executing it there is the same class of error as §11.1.

**And that settles WHICH commit, more sharply than "admissible by validation" did — execution found
it, reading did not.** The validation was attempted at `e3d5e92` first, since that is where the
coefficients were fit and therefore the most faithful probe. It cannot run there:
`_CORPUS_IDENTITY_ATOL` occurs **0 times** at `e3d5e92` and **9** at `7e875c8`, because
**`7e875c8` is the commit that INTRODUCED the assertion** (`git log -S`, `scripts/train_gk_completion.py`
— only two commits ever touch it: `7e875c8`, then ADR-052's driver migration). At 4.21.0 the trainer
simply fits and saves; there is no instrument to disagree with, so a run there produces neither a
verdict nor — since the script does not dump the design matrix — a probe.

So `7e875c8` is not merely *an* admissible vintage. It is the **earliest commit at which the check
exists at all**, it is where that check was originally run against these very coefficients, and it
passed. Reading `git log` could establish that `7e875c8` was additive; only running it established
that `e3d5e92` is **unvalidatable**. The paragraph above still holds — do not apply HEAD's vocabulary
to an old checkout — with the corollary that an old checkout may not carry the instrument the
technique depends on. **Check that the instrument is present before trusting a run that uses it**, or
a vacuous pass is indistinguishable from a real one.

#### 11.3.1 THE VINTAGE TARGET ABOVE IS WRONG — `probe_old` must be ROW-ALIGNED with `probe_new`

**Everything from "The probe's VINTAGE is the hard part" down to here identifies the wrong artifact,
and the correction comes from running `predictions_moved` rather than reading it.** The function ends
in

```python
np.allclose(_serve(old, probe_old), _serve(new, probe_new), atol=atol, rtol=0.0)
```

which is **element-wise**. Handing it a 1666-row historical matrix against the current corpus's
~3491 rows does not produce a wrong answer — it raises
`ValueError: operands could not be broadcast together with shapes (1666,) (3491,)`. The retrain would
crash at the guard, after paying for the whole corpus pass.

So `probe_old` is **not** "the matrix the committed weights were fit on" in the archaeological sense
this section spent three revisions pursuing. Two independent things in the codebase say so:

1. The docstring's own next clause — *"they are the SAME array whenever the feature space did not
   move, which is the ordinary case."* A historical training matrix can never be the same array as
   the current `X_all`; only a same-corpus extraction can.
2. The registered test `test_a_rebundle_across_a_MOVED_feature_space_must_still_abort` constructs
   the moved case as `X_new = X_old + 5.0` — **same rows, shifted geometry**, never a different
   corpus.

**The correct probe is: the SAME corpus as the fresh fit, extracted under the PRE-CHANGE geometry.**
The guard asks a *serving* question — "does what production emits change?" — so the two legs are
what production emitted yesterday and what it will emit tomorrow, over one population of actions.
Fitting provenance is a different question that this guard does not ask.

Which means **"pre-RC2", the answer §11.3 opens by calling wrong, is right** — though not for the
reason it was originally offered, and the original objection was sound on its own terms. The
objection was that the committed weights would be compared "against coordinates it never saw". True,
and unavoidable: those weights have been served against every geometry change since 4.21.0, so
"coordinates it never saw" is simply the production reality this PR is correcting. The probe vintage
is therefore **`641dadf` (4.70.0)** — the commit immediately before RC2/RC3/RC5 landed in `89dd9af`
(4.71.0) — extracted on the same Gradient Sports corpus the retrain uses. RC1 (4.70.0) is safely
inside that baseline: it reprojects the cover-shadow passer, and no `GK_COMPLETION_FEATURE_NAMES`
entry reads cover shadows.

**Two alignment properties must be verified, not assumed, and only one of them is loud.** A row-COUNT
mismatch raises. A row-ORDER mismatch does not: it silently compares action *i*'s old prediction
against action *j*'s new one, which will almost certainly differ and so reports `moved=True` — the
guard passes, for a reason that has nothing to do with the model. ADR-052 changed extraction to
sharded `for_each`, so ordering across the two vintages is exactly the kind of thing that could have
shifted underneath.

**VERIFIED, by reading the two orderings rather than paying for a second corpus pass:**

1. HEAD's `_extract` combines `[... for k in res.keys]` — **this pass's own keys, deliberately not
   `_driver.reconcile`** (whose `sorted(glob("*.parquet"))` *would* have re-ordered by filename, and
   GS ids mix widths — `"10502" < "3857"` lexicographically but not numerically). `for_each`
   accumulates `own_keys.append(k)` inside the iteration loop, and its docstring states `keys` is
   "every joined shard key THIS pass covered, **in order**".
2. `641dadf`'s pre-ADR-052 `_extract` appends inside the same `load_matches(...)` loop.
3. The **entire** `641dadf..4b15365` diff to `scripts/_loader_pining.py` is the RC4 SkillCorner
   change — 18 insertions, all in `build_skillcorner_frames`. The Gradient Sports path is untouched,
   so `load_matches(providers=["gradientsports"], ...)` yields the same matches in the same order at
   both commits.

Within a match, row order follows the selection in action order, which the corrections do not
reorder. So ORDER agrees by construction; the row COUNT does **not**, and an earlier draft of this
paragraph claimed it did.

**Membership is NOT invariant by construction — that claim was wrong.** The trainer's filter is not
`_gk_distribution_mask` alone. `prepare_gk_completion_training_data` computes
`keep = geom_ok & id_ok`, where
`geom_ok = np.isfinite(X["length"]) & np.isfinite(X["dest_x"])` — and `length`/`dest_x` derive from
the **resolved** geometry that RC2 rewrites. A coordinate that was NaN and resolves finite (or the
reverse, e.g. a goal-kick falling from the tracking-GK tier to the rule-point prior) adds or removes
a row. So the count is an **empirical** question and the reasoning must not pretend otherwise.

Which is fine, because the check is loud and free: compare `probe_old`'s `n_rows` against the `N=`
the trainer prints. **Do not rely on the broadcast error to catch a mismatch** — that is true only
of shapes that fail to broadcast (1666 vs 3491 does; a **1-row probe does NOT**, it broadcasts and
answers silently, measured). `_assert_retrain_moved_predictions` compares row counts explicitly,
which matters because ADR-052's own follow-up proposes persisting a fixed probe *sample*. The
probe dump keeps `_group` anyway (`--probe-old` selects `[feats]`, so extra columns ride along
harmlessly) so the claim stays falsifiable later.

**Confirmed on SkillCorner — but be precise about WHICH comparison shows what.** Two distinct
identities landed, and only the second bears on the corrections:

1. `641dadf` probe at `--max-per-provider 10` → **`n_rows: 542`, `n_groups: 10`**, matching the
   committed variant's **`n_rows: 542`, `n_matches: 10`**. **Both legs are PRE-correction** (the
   committed weights were fit at 4.21.4 geometry, the probe ran at 4.70.0), so this says the cap
   selects the corpus the committed weights were fit on — and *nothing whatever* about RC2/RC4. An
   earlier draft cited it as evidence that "the domain is unchanged by the corrections", which it
   structurally cannot be.
2. `641dadf` probe (542) vs the retrain's own `N=542` at `4b15365`. **This one straddles the
   corrections**, and it is the row-count identity the alignment argument actually needs.

It is also a **count** identity, not a corpus identity — 542 rows over 10 groups could in principle
be a different 542. The `_group` order digest is dumped for exactly that reason, and the same
skepticism this section applies to Gradient Sports applies here.

Separately, it shows the corpus-drift problem is **GS-specific**: SkillCorner's committed corpus and
its current public arm are the same ten matches, while GS grew 30 → 64.

The extraction emitted the RC4 defect live while running — `OrientationUnresolvedWarning:
acting_team_attacks_rtl: returning an all-False flip (team_attacking_direction is present but
entirely null)` — which is exactly the state `probe_old` is supposed to capture, and independent
evidence that the PYTHONPATH guard did its job rather than silently resolving HEAD's corrected code.

**Cost of the error, recorded because it is the reusable part:** ~30 minutes of DGX compute — one
cap-64 vintage run to completion (~26 min) plus a cap-30 run killed early once the premise collapsed — chasing a corpus-size discrepancy that was real but irrelevant. The
discriminator table added to the plan caught the *symptom* (N=3491 vs a committed 1666) and correctly
refused to read the failure as a vintage verdict; it could not catch the *premise*, because it was
built on the same wrong target. The check that would have caught it in seconds was calling
`predictions_moved` with two mismatched shapes — one line, no corpus, no DGX.

**The check is ONE-SIDED, and both directions must be stated** (part-deux review 2, their own
correction to phrasing this section had adopted verbatim). A **pass** is strong: code and corpus must
*both* reproduce the committed weights. A **fail is ambiguous** — wrong vintage *or* corpus drift. The
corpus is not frozen (4.49.0 and 4.50.0 each shipped GS-only retrain triggers: dribble end-derivation
and `ballCarryOutcome` results), and `_CORPUS_IDENTITY_ATOL` is `0.05`, loose enough that a pass means
something but a fail does not localise. Do not let a first failure send someone hunting a vintage
problem that may not exist.

**Long-term fix, and the right moment to take it:** ADR-052 records *"persist a fixed probe sample
beside the weights"* as an ADR-011 follow-up. Every future geometry correction otherwise repeats this
archaeology. Since this PR regenerates the artifact anyway, persisting the probe belongs here.

**But NOT inside the artifact — ADR-044 forbids it** (part-deux review 2; the ADR-052 follow-up was
written without checking it). `ADR-044:25`: *"Distributed model artifacts carry **learned parameters
only, not per-sample training data.**"* A persisted design matrix is per-sample training data. ADR-044
is titled ghost-scoped, so this is a tension to reconcile rather than a rule already broken — but
silence would be the wrong resolution.

**Decision: persist OUT-OF-BAND.** The probe lives under `docs/research/`, referenced from
`metrics.json` by path **and SHA256**. That keeps the wheel parameters-only (ADR-044 intact), keeps
the citation resolvable, and keeps the artifact's integrity envelope honest. The alternatives were
in-artifact with an explicit ADR-044 amendment — defensible, since a fixed ~100-row probe is not a
corpus, but it needs the ADR edit rather than silence — or an explicit defer. Recorded here so the
choice is visible rather than dropped under "optional".

Note either way: adding a file to the weights directory changes `SHA256SUMS`, so a checksum-pinning
consumer sees a diff with no value change — the ADR-050 precedent, where three `metadata.json` gained
`feature_contract` and the sums moved.

### 11.4 Sequencing constraint: COMMIT RC4 (on the branch) before any trainer run

ADR-052 shards `_extract` per match, keyed on
`token_inputs={"extractor": <literal name>, "tracking_limit": ...}`. **That token captures neither
geometry nor library version** — `_driver.generation_dir`'s own docstring says the token cannot be
checked for completeness.

The handoff framed this as a pre-RC2 hazard. That precondition is **unconstructible**: shards did not
exist at RC2 (the mechanism arrived *in* 4.72.0), and none exist on disk. **The live hazard is
pre-RC4** — run the trainer on rebased main *before* applying RC4 and that run mints a generation;
apply RC4 afterwards and the SkillCorner frames change while the token does not, so the retrain reads
stale features silently. The only signal is a per-item `skip (shard exists)`, and `metrics.json`
records `run_commit` but no generation digest, so a clean SHA would describe the fit and not the
extraction.

**Decision: commit RC4 first — on the BRANCH, not merged to main.** That makes the failure
*unreachable* rather than guarded, and costs
nothing because no shards exist yet. A fresh `--shard-dir` per run is the hand-maintained discipline
`train_ghost_gk`'s own docstring records going stale inside the cycle it protects. Adding a derived
geometry entry to `token_inputs` is worth doing as well, but as defence in depth, not as the primary
control — note `cache_token()` reads penalty-area constants and would have missed **both** RC2 and
RC4.

Also: **do not pass `--cache-features` on an RC4 run.** It is a bare `Path(cache).exists()` check
that bypasses the generation directory entirely, so no token fix can protect it.

### 11.5 Superseded measurement: the cover-shadow argmax agreement

4.72.0 fixed the RC1 defect in `measure_cover_shadow_argmax_agreement.py` — a site that imports
`_compute_cover_shadow_dict` directly and so was never a registered RC1 site — and re-measured from a
clean tree:

| | pre-RC1-fix | post |
|---|---|---|
| agreement | 0.1567 | **0.0443** |
| vs ~0.10 chance | 1.6× | **0.44× — worse than random** |

**The defect had been inflating the number.** The 4.67.0 verdict (gate
`max_single_defender_player_id` to `detailed=True`) is unchanged and better supported: the upper CI
bound is 0.059 against a 0.90 floor. Any prose citing **0.157 / 0.1992 / 0.723** is stale in **digit
and direction**.

Main was itself stale in two places — its shipped 4.72.0 CHANGELOG and its CLAUDE.md `PR-S136`
bullet. **The part-deux session has taken that correction as PR #190** (docs-only, CHANGELOG +
CLAUDE.md), on the principle that a release's own error is that release's to fix, which also
removes a conflict from this diff. Their handling of 4.67.0 is the right call and worth copying:
the shipped entry is left **unedited** and flagged with a superseded-by note, because rewriting a
shipped release note is worse than leaving it, while leaving it unmarked invites a forward
citation.

**Still ours:** `0.723` survives on main at
`docs/superpowers/plans/2026-07-29-adr028-orientation-defect-class.md` (search `0.723`; the line number moves as the plan grows) — Task 17b's
worst-case bound, written in PR 3 and merged. The reviewer searched CHANGELOG and CLAUDE.md and
did not find it; it is in the plan. It rests on the pre-fix 0.157 and is stale in the same
direction.

### 11.6 Register and tag state

4.72.0 and PR-S140 are **taken**. This PR is **4.73.0 / PR-S141**. No new ADR — ADR-051 already
scopes "PR 4 of 5".

**Both 4.71.0 and 4.72.0 are unreleased** (latest tag `v4.70.0`). The window did not close; it
widened. The first tag pushed from here ships RC2+RC3 *and* ADR-052 without the paired
`GkCompletionModel` retrain — so this PR is still what makes a tag safe. ADR-051's status header and
CHANGELOG's 4.71.0 note both still say "ships within 4.72.0 alongside PR 4", which is now false and
must be corrected here.

### 11.7 TF-24 is RC4's second consumer — a STATED no-op, not silence

§3.4's verified table names **two** affected consumers: `train_gk_completion` **and
`calibrate_tracking_defaults` (TF-24)**. The first draft of this amendment mentioned TF-24 zero times
(part-deux review, finding 1). Since §3.4's own analysis put it in the blast radius, silence is the
one disposition not available.

**Disposition: RC4 changes NO shipped TF-24-calibrated constant.** Evidence, per stage:

- **Stage 1's params ARE shipped, and PARTLY TF-24-calibrated** — `infer_ball_carrier`'s
  `tolerance_m=3.0, beta=0.0, gamma=0.25`. Precisely: the docstring
  (`_ball_carrier.py:352-353`, *"The `beta` and `gamma` defaults are Optuna-calibrated (TF-24) at the
  held `tolerance_m=3.0`"*) records only **two** of the three as calibrated; `tolerance_m` was HELD,
  making it an engineering default. An earlier draft of this bullet called all three calibrated.
  They were calibrated against a
  3-provider fold that **included the unoriented SkillCorner frames**. They are nonetheless
  unaffected, because carrier inference is **orientation-invariant**: measured, 40/40 identical
  carrier assignments under an exact point reflection, carrier distance unchanged to <1e-9, and
  (the figure once quoted here as `1.01e-14` does not reproduce -- the registered test's own
  fixture gives ~1e-14 at a scale that depends on the fixture, so the assertion is stated as a
  bound rather than a digit), and
  `_ball_carrier.py` contains no orientation reads at all. The unoriented frames gave the same answer,
  so the calibration stands.
- **Stage 2's params are NOT TF-24-set.** `k3` (`pressure.py:61`) and `min_displacement_m`
  (`_off_ball_runs.py:100`, `_run_values.py:121`) ship as **engineering** defaults — `pressure.py:50`
  says so in as many words — consistent with the standing rule that TF-24 recommends and never changes
  library constants.

So RC4 invalidates a Stage-2 *recommendation that was never run on corrected frames and never
applied*, which is already tracked as the deferred TF-24 item. **No re-sweep trigger; no shipped value
moves.** Recorded here so the **No** verdict is auditable rather than absent.

### 11.8 Merge strategy: a MERGE COMMIT, not a squash

`metrics.json` records `run_commit`. **A squash merge breaks that citation** — the SHA never exists on
`main`. The part-deux session hit this on #189 and landed it as a merge commit for exactly this
reason.

**The repo now allows merge commits** — verified `allow_merge_commit: true`. It was squash-only until
2026-08-01, so any memory, habit or doc saying "squash-only" is stale. This PR must merge with
`--merge` explicitly rather than take a default, and verify afterwards:

    git merge-base --is-ancestor <the SHA metrics.json records> origin/main

**ONE merge, not several.** The multi-commit structure (RC4, then weights, then docs) exists so the
trainer executes against RC4-corrected code and so ``run_commit`` names a real commit — both are
satisfied on the BRANCH. A merge commit preserves those commits, which is the whole reason this PR
does not squash.

Merging RC4 to main on its own would be actively wrong: it would put corrected serving geometry on
main **without** the paired retrained weights — a smaller instance of exactly the train/serve skew
that left 4.71.0 untagged. It would widen the open window rather than close it.

### 11.9 The RC4 guard needs strengthening, not just porting

PR 4's AST guard on `_loader_pining.py` had two holes its own review found:

- **A third call site is invisible to it.** `_build_gradientsports` omits `output_convention`
  *entirely*, so there is no keyword for a keyword-matcher to see. (Behaviourally benign — GS resolves
  via the same unconditional `finalize_orientation` — but the guard cannot say so.)
- **Its non-vacuity partner does not exercise the guard.** It re-implements a weaker matcher inline,
  dropping the `name != "convert_to_frames"` filter, so it passes even if the guard is dead.

The re-planned guard must assert on the **resolved convention per builder**, not on the presence of a
literal keyword, and its non-vacuity test must call the guard's own body.
