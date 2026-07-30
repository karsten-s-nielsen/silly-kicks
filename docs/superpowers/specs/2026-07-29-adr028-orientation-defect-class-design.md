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
