# Plan review — `2026-07-14-skillcorner-corpus-and-visibility.md`

**Reviewer:** external session · **Date:** 2026-07-14 · **Verdict: not ready to execute.**

The plan's *architecture* is right: task order respects the dependency graph, the TDD shape is real,
the blast radius is stated rather than hidden, and your four self-flagged gaps are all genuine —
particularly #2 (`prepare_ghost_gk_training_data` exposes neither `player_id` nor `visibility`, so
**both** registered ghost-GK rules were impossible without a library change). Surfacing a missing
prerequisite the spec didn't know it had is the plan doing its job.

But it cannot be executed as written, and — the serious part — **three of the four registered safety
rules it exists to implement are inert.** Every finding below is verified against the source, not
inferred from the plan's prose.

---

## BLOCKERS — registered rules that cannot fire

### P1 · The interpolator-tell refusal is dead code — and so is the spec rule behind it

Task 10, `scripts/_paired.py`:

```python
improves_detected = clears_rule([-d for d in detected_only_deltas])
if all_frames_deltas is not None:
    improves_all = clears_rule([-d for d in all_frames_deltas])
    if improves_all and not improves_detected:
        return False            # <-- reachable ONLY when improves_detected is False
return improves_detected        # <-- which returns False anyway
```

The refusal branch fires **iff `improves_detected` is False** — in which case the fall-through
returns `False` identically. So `ghost_admission(a, b) ≡ clears_rule([-d for d in a])` **for every
input**. The `all_frames_deltas` argument is inert.

The test written to cover it:

```python
assert ghost_admission(detected_only_deltas=[0.1] * 5, all_frames_deltas=[-0.2] * 5) is False
```

`[0.1]*5` negated never clears, so this passes **with the entire refusal block deleted**.

**This is a spec defect, not a coding slip.** §4.3's rule — *"if all-frames MAE improves while
detected-only MAE degrades, refuse"* — is **logically subsumed by the admission rule**, because
admission already *requires* detected-only improvement. It can never change a verdict. I called this
"the strongest single addition in rev 2". I was wrong, and the plan is the first thing to make that
visible.

Two ways out, and they are different rules:

- **Demote it to a diagnostic.** Return a refusal *reason*, so "interpolator tell" is distinguishable
  from "no improvement" in the record. Honest, cheap, and preserves the reporting value — but admit
  it changes no decision.
- **Give it independent content.** The dangerous case it was *meant* to catch is not "detected
  degrades" — it is **"the model improves on both, but far more on the interpolated frames."** That
  case currently **admits**. A divergence rule — refuse when the all-frames improvement materially
  exceeds the detected-only improvement — actually bites. If you want a real guard, this is it.

### P2 · Task 4 defines the rate-gate and never calls it

The Files header promises *"a new pure gate function **and its call in `convert_to_frames`**"*. Step 3
adds `geometry_rate_gate(...)`, `GeometryGateReport`, and the thresholds — and **no step wires the
call**. No step says what `excluded=True` *does*: raise? drop the match? a report field? All four
tests call the pure function directly.

Delete the (never-written) call site and the suite stays green. **§4.4's exclusion mechanism — the
R1 blocker from the last review, the entire reason Task 4 exists — still does not exist.** This is
the fourth consecutive round of the same defect class, on the item raised specifically to end it.

*(Credit where due: Task 4's calibration comment is the most honest paragraph in the plan — it states
outright that a 4 m pitch-dimension error measures 0.00095 and **does not trip**, that this gate
cannot see one, and that action↔frame co-location cannot either "because events and tracking read the
same metadata and move together". That is exactly the right acknowledgement of the
consistency-≠-correctness limit, volunteered rather than hidden.)*

### P3 · The compliance control is tested at the wrong layer

Spec §6 registered this gate as: *"asserted against the **label path**, not just the arm split, and
driven **red-first against today's code (which fails it)**."*

Task 9 writes `tests/scripts/test_corpus_taxonomy.py`, which imports `PUBLIC_CORPUS`,
`artifact_label`, `is_public_row` from a **brand-new** `scripts/_corpus.py`. Step 2's expected failure
is:

> `ModuleNotFoundError: No module named '_corpus'`

That is not red-first against today's code. That is a new file not existing yet — true of every new
file ever written. **Nothing in the suite pins the trainer's `shipped` label.** The actual shipped bug
— `provset <= _PUBLIC_PROVIDERS` at `train_xshot_occurrence.py:313` / `train_xcross_attempt.py:398`
labelling an `sc_extended`-shaped run `"public"` — is verified only by a `grep`. Rewire the label
branch wrongly and every test still passes.

This is the wrong-layer pattern exactly: the guard is validated at the pure-function layer, not at the
layer where the defect lives. The test you need asserts on the **trainer's** `shipped` value for a
corpus containing a restricted match — and it must fail on today's code for the *right* reason.

### P4 · Task 12 changes what ghost-GK learns and writes no tests — and its one assertion is a tautology

Task 12 implements **all four** registered §4.3 rules — detected-only targets, keeper-grouped CV, the
common-domain exclusion, two-scheme reporting — across six steps, **none of which writes a test.** It
runs pre-existing suites only. The plan's own rule is that every gate names the mutation that kills
it.

And the one assertion it does contain cannot fire:

```python
domain = np.array([k not in expansion_keepers for k in keepers], dtype=bool)   # <- defines domain
...
assert not (set(keepers[domain].tolist()) & expansion_keepers), "test-fold keeper leaked into the 98"
```

`keepers[domain]` is **by construction** the set of keepers not in `expansion_keepers`. The
intersection is empty by definition, for any input. This is the §4.3 leakage check the spec insisted
be *"asserted, not assumed"* — and it is an assumption wearing an assert's clothes. The real check
must run **after the split**, against each fold's *test keepers*, and against the actual keeper sets
of the two corpora.

*(Good, and worth keeping: Step 2 does print `n_domain_keepers` and warns when it drops below
`cv_folds * 2` — that answers the "quantify the domain" note from the last review.)*

---

## MECHANICAL — the plan will not run

| # | Task | Defect |
|---|---|---|
| M1 | 3 (×5 tests), 7 (production) | `convert_to_frames(bronze, *, home_team_id, …)` — `home_team_id` is **required keyword-only** (`skillcorner.py:91-99`; every existing test passes it). All the plan's calls omit it → `TypeError`. Worse: Task 3's `pytest.raises(ValueError, match="pitch_length")` sees a `TypeError`, so "watch it fail" reports the wrong failure and the test can never go green. |
| M2 | 3, 4 | `tests/tracking/test_skillcorner.py` **does not exist** (the real files are `test_skillcorner_builder.py`, `_gk_roster_trust.py`, `_within_pitch_invariant.py`). Both Step 4s pytest it → file-not-found. These are precisely the fixtures Known-Risk-1 is about, and none is named. |
| M3 | 8 | `_token()` does not exist — it is `_resolve_token` (`_loader_pining.py:66`) → `NameError`. |
| M4 | 11 | `_fit_score`: `_pinned_params` is a **function-local** import in the trainers (`:121`, `:179`), not module-level → `NameError`. It also **already returns** `tree_method`/`n_jobs`/`random_state`/`eval_metric`, which `_fit_score` passes again → `TypeError: got multiple values`. And `X[te_idx]` on a DataFrame with an integer array is **column** selection → `KeyError` (the replaced code used `.iloc`). It silently drops `base_score = ytr.mean()` and the `negative_subsample` train-fold thinning — so "the same fit/score the old `_paired_data_effect` performed inline" is not true. |
| M5 | 11 | **The NaN-fold semantics you flagged in your own self-review are prose, not code.** The body appends `s_nested - base_nested` unconditionally, so a single-class fold yields `NaN`; `clears_rule` then silently returns `False` (`NaN > 0` is False, mean is NaN). The old code dropped NaN folds explicitly (`:205`). One degenerate fold now **flips the verdict to "don't ship"** instead of dropping out — and `clears_rule`'s table test has no NaN row. |
| M6 | 12, 6 | Step 1 uses `prov` **before assignment** (`train_ghost_gk.py:262` assigns it *after* the append) and misnames the accumulators (`all_feats`/`gid` vs the real `all_features`/`game_id`). Task 6 Step 5 patches only the *tail* return of `prepare_ghost_gk_training_data`; the early return at `_ghost_gk.py:889-893` still yields a 2-tuple → `ValueError: not enough values to unpack` on any game with no GK frames. |
| M7 | 12 | `np.load(args.expansion_keepers)` on the object-dtype array Step 1 saves → `ValueError: Object arrays cannot be loaded when allow_pickle=False`. Every other object-array load in that file passes `allow_pickle=True` (`:214-215`). |

---

## Minor

- **`_TOL_BALL` 30 → 15 is the right call, but its test is a tautology.** The *implementation* comment
  is well-reasoned (67% headroom over the worst observed 9.00 m; zero public rows exceed it). The
  *test* is `assert _TOL_BALL == 15.0` with the message *"30.0 m sat above every real excursion — the
  gate could not trip"* — which is incoherent, because **15.0 also sits above every real excursion**,
  and must, or it would fire on good data. A constant-equality check proves nothing behavioural.
  Replace with: a broken transform trips it; real data does not.
- `assert_public_corpus` is wired into both trainers but never tested (replace its body with `pass`
  and the suite is green) — and as written it demands the manifest's public set equal **all 17**, so a
  `--providers gradientsports` run yields the empty set and dies on `SystemExit`. Pin the intended
  behaviour before an owner run discovers it.
- `cache_is_valid` checks only `cache_meta.json` and **replaces** the `features.parquet` existence
  predicate — so a directory with a valid meta and no features file is a **hit** →
  `FileNotFoundError`. The plan's test (`write_cache_meta` only, then assert hit) enshrines it.
- Lint gate (Task 15) will fail: `E402` (Task 7 appends `import pytest` after a def), several `F401`
  unused imports in the new test files, `F841` (`is_gs`, Task 11).

---

## What to do

The mechanical breaks (M1–M7) are an afternoon — they are real but shallow, and the two riskiest of
them (`home_team_id`, the NaN folds) would have surfaced on first run.

**P1–P4 are the ones that matter, because each one produces a green suite that proves nothing.** Three
of the four registered safety rules in §4 — the interpolator tell, the geometry exclusion, and the
keeper-leakage assertion — currently cannot fail, and the fourth (the compliance label) is asserted
one layer away from the defect it guards. That is the same pattern this spec has now shed four times;
it keeps reappearing because each round fixes the *instance* and not the *habit*.

The habit is cheap to install, and it is already the plan's own stated rule: **for every guard, name
the single line of production code whose deletion makes the test fail — then delete it and watch.**
P1, P2 and P4 all die instantly under that check. Run it across all 16 tasks before executing any of
them.

**P1 additionally needs an owner decision, because it is a spec change**: is the interpolator tell a
reported diagnostic, or is it the divergence rule (refuse when all-frames improvement materially
exceeds detected-only improvement)? As written it is neither — it is a no-op.
