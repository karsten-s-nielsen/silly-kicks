# ADR-028 RC4 — the pining loader shipped UNORIENTED SkillCorner frames

Both sides of the RC4 claim, measured on real matches at **full frame depth**. ADR-051; spec
`docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md` §3.4 / §11.

## The defect

`scripts/_loader_pining.py::build_skillcorner_frames` forced
`output_convention="absolute_frame"`, which leaves `team_attacking_direction` **NULL on every row**.
`acting_team_attacks_rtl` then resolves nothing, returns an all-False flip, and the **entire ADR-028
per-action re-projection layer silently no-ops** — so every away-team action in the research corpus
carried mixed-convention geometry while looking perfectly healthy.

## Results

| metric | SkillCorner pre-fix | SkillCorner post-fix | IDSSE pre-fix | IDSSE post-fix |
|---|---|---|---|---|
| `unlabelled_fraction` | **1.0000** | **0.0000** | 0.0000 | 0.0000 |
| `distinct_labels` | `[]` | `['ltr','rtl']` | `['ltr','rtl']` | `['ltr','rtl']` |
| `n_flip_true` | **0** / 1197 | **566** / 1197 | 718 / 1363 | 718 / 1363 |
| `flip_true_fraction` | **0.0000** | **0.4728487886382623** | 0.5267791636096845 | 0.5267791636096845 |
| `orientation_warnings` | 1 | 0 | 0 | 0 |

SkillCorner match `1886347` — 43,458 frames, 956,076 player rows, 1,197 actions.
IDSSE `DFL-MAT-J03WMX` — 145,967 frames, 3,211,274 player rows, 1,363 actions.
Both sides `tracking_limit=None`, `max_per_provider=1`; pre-fix at `5a67212`, post-fix at `4b15365`.

## IDSSE is the CONTROL, and it is deliberately NOT changed

`sportec.py` calls `finalize_orientation` **unconditionally**, before its own convention branch, so
IDSSE frames are already labelled regardless of what the loader requests. Its numbers are
**byte-identical across the fix** — `718` flipped actions on both sides, `0.5267791636096845` to all
17 significant digits.

That invariance is the point. An earlier pass in this cycle reported IDSSE as transitioning
`1.0000 → 0.0000` unlabelled — a figure **assumed** from SkillCorner's behaviour rather than measured,
which then triggered a wrongful retraction of a correct finding (spec §11.1). A control that does not
move is what distinguishes "the fix worked" from "the fix touched everything it happened to run past",
and it is retained for exactly that reason.

**Honest about what this control can and cannot fail.** The RC4 diff is confined to a single
function, `build_skillcorner_frames`, so the two runs execute byte-identical IDSSE code and the
invariance is guaranteed by the diff's shape — it could not have come out otherwise. Its value is
therefore not that it *might* have moved, but that it makes the *scope* claim checkable: a later
change that reached beyond SkillCorner would break it, and the previous cycle's error was precisely a
scope claim ("RC4 also affects IDSSE") asserted without one. Read it as a scope guard, not as
independent evidence that the SkillCorner fix works.

**It also settles an open conflict.** Spec §2.2 independently reports *"on IDSSE flip is True on
exactly 718/718"* of the not-home actions. This measurement returns `n_flip_true: 718` of 1,363 —
the same 718. §2.2 is confirmed by a second instrument.

## The first version of this measurement was CAPPED, and said so nowhere

The original run passed **`tracking_limit=3000`** and the artifact did not record it. That is this
cycle's named failure mode — a cap presented as a corpus — committed in the document that argues
against doing it. It was caught by arithmetic during review: 66,000 IDSSE player rows is exactly
3,000 frames × 22 players, and the published IDSSE flip `0.31548055759354365` is exactly `430/1363`
where §2.2 independently said `718/1363`.

**The mechanism, which fixes the direction of the error.** The orientation lookup is built by grouping
non-ball frame rows on `(game_id, period_id, team_id)`; an action whose key is absent from that lookup
**defaults to no-flip**, silently — `OrientationUnresolvedWarning` fires only when *nothing* resolves,
never on a partial miss. A truncated frame set therefore **depresses** `flip_true_fraction`. Both
published flip numbers were lower bounds:

| | capped (3,000 frames) | uncapped | understated by |
|---|---|---|---|
| SkillCorner post-fix flip | 0.2398 | **0.4728** | ~2× |
| IDSSE flip (both sides) | 0.3155 | **0.5268** | ~1.7× |

**What the cap did NOT affect: `unlabelled_fraction`.** A cap cannot make labels appear, so the
`1.0000 → 0.0000` finding — the defect itself — stood on its own throughout, and the RC4 fix was never
in doubt. What was wrong was the *magnitude*, understated in the direction that made the fix look
smaller than it is. The corrected SkillCorner figure also stops being an outlier: 0.4728 now sits
alongside IDSSE 0.5268 and Gradient Sports 0.5665 (§2.2) as an ordinary away-action share, whereas
0.2398 was half of anything else measured and should have prompted a question.

Both JSON artifacts now carry a `_provenance` block recording `tracking_limit`, `max_per_provider`
and the resolved commit, so the corpus can be checked rather than inferred from divisibility.

## Provenance and self-authentication

| side | commit | loader read |
|---|---|---|
| `prefix_measurement.json` | `5a67212864660fe0c6571778f6b897dbec5b3606` | `output_convention="absolute_frame"` |
| `postfix_measurement.json` | `4b153655f2388c4c1f4009d5abb0955b114222f1` | `output_convention="ltr"` |

These were written by an ad-hoc measurement pass, so the `scripts/_provenance.py` machinery did not
apply — the `_provenance` block above is hand-rolled and records the commit it resolved at runtime,
under the key `commit` rather than ADR-052's `run_commit`.

**A committed producer now exists: `scripts/measure_rc4_orientation.py`** (4.73.0, registered in
`ARTIFACT_DRIVERS`). It emits the ADR-052 vocabulary — `run_commit` / `run_tree_dirty` /
`run_tree_state` — plus `tracking_limit`, and refuses a dirty tree unless `--allow-dirty` is passed.
Re-running it regenerates these two files in that vocabulary — but the two sides are **not
symmetric**, and an earlier version of this section gave a recipe that cannot work.

The **postfix** side is a plain clean-tree run at a commit containing the fix:

```
python scripts/measure_rc4_orientation.py --label postfix
```

The **prefix** side cannot be a bare `git checkout <pre-fix commit>`: this script does not exist at
any commit before 4.73.0, so the checkout deletes the very thing you are running. Copy it into a
worktree at the older commit:

```
git worktree add --detach /tmp/rc4-pre <pre-fix commit>
cp scripts/measure_rc4_orientation.py /tmp/rc4-pre/scripts/
cd /tmp/rc4-pre && python scripts/measure_rc4_orientation.py --label prefix --allow-dirty
```

`--allow-dirty` is REQUIRED there, and the resulting artifact will record `run_tree_dirty: true` —
which is correct rather than a wart: the code that ran is that commit **plus an imported file**, so
it is not that commit. The driver's own module docstring carries the same procedure.

The files here were NOT regenerated for 4.73.0, and the reason is worth stating rather than hiding: a
clean-tree run is only possible once this release is committed, and re-running under `--allow-dirty`
purely to change key names would stamp `run_tree_dirty: true` on numbers that were in fact measured
from clean checkouts (`5a67212` and `4b15365`, both recorded above). The values are what matter and
they are unchanged; the key naming converges on the next re-run.

**An earlier version of this section claimed the prefix artifact was SELF-AUTHENTICATING. It is
not, and the correction is worth keeping.** The argument was that `unlabelled_fraction == 1.0` with
`distinct_labels == []` is unreachable under `"ltr"`, since that branch's body is the orientation
call. That reasoned from the call site and never read the callee.
`orient_frames_to_ltr_by_geometry` (`tracking/direction.py:175`) has an early return: when **no home
players match** — the ADR-019 id-dtype-mismatch case, a documented real defect class — it warns
*"orientation left as-is"* (`:262`) and returns, **bypassing the tail at `:303`** that would otherwise
populate the labels. (An earlier draft cited `:81-89` and `:129`, which land inside
`require_et_direction`, ~170 lines away: those numbers were read from a `sed` extraction of the
function body and are RELATIVE offsets, not absolute lines.) So an all-null match under `"ltr"` is reachable, and the artifact cannot date itself from its
own contents.

What the pre-fix side *does* rest on is ordinary evidence: the `_provenance` block records the
resolved commit `5a67212…`, and the run printed that commit's `output_convention="absolute_frame"`
before measuring. Weaker than a self-proving artifact, and stated as such. The asymmetry still holds
in one direction only — the post-fix `0.4728` flip **requires** labels to exist, so it cannot have
come from the pre-fix loader.

The durable guard is `tests/scripts/test_loader_orientation.py`, which parses each builder's
`convert_to_frames` call and pins the whole mapping, so a silent flip on **any** builder fails CI. Its
reach is precise rather than total: it resolves the **argument at the call site** — a string literal,
or its absence — not the convention that ultimately takes effect. For `_build_gradientsports` it
records `None`, meaning only *the kwarg is absent*.

**And that absence does NOT mean `"ltr"` — an earlier draft of this paragraph said it did, citing
`tracking/skillcorner.py:156`, which is the wrong converter entirely.** In this document, of all
places: a sibling call site is not evidence. `_build_gradientsports` calls the *gradientsports*
converter, which routes `None` through `sportec._resolve_output_convention` and resolves it to
**`absolute_frame`** with a `DeprecationWarning` (ADR-006). Only SkillCorner defaults to `"ltr"`.

It is benign anyway, and for the **same reason IDSSE is**, which is the reason worth carrying:
`gradientsports.py:129` calls `finalize_orientation` **unconditionally**, before the
`output_convention == "ltr"` branch at `:187`, so its frames carry `team_attacking_direction` under
either convention. RC4 is a SkillCorner defect precisely because `tracking/skillcorner.py` has no
such unconditional labelling step.
