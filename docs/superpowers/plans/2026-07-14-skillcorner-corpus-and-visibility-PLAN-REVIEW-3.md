# Plan review 3 — `2026-07-14-skillcorner-corpus-and-visibility.md` (rev 3)

**Reviewer:** external session · **Date:** 2026-07-14 · **Verdict: one blocker left, in Task 11. Everything else is ready.**

## The five blockers from last round are fixed — verified at the code, not the prose

| | fix | verified against source |
|---|---|---|
| **B1** `match_id` from a path split | `_build_skillcorner` now threads `match_id=str(match_id)` into `build_skillcorner_frames`; a new `test_game_id_is_the_REAL_match_id_not_a_path_fragment` names the old `split("_")[-2]` as its kill-line | ✅ Correct — and the regression test is exactly the collapse-to-one-group bug |
| **B2** `assert_public_corpus` too strict | now a **subset** assertion by default (licensing), with `expect_full_public_arm=True` opt-in for the equality/drift check; the two failure modes are named | ✅ The flagship licensing test can now reach its assertion |
| **B3** `home_team_id` omitted in production | `build_skillcorner_frames` derives `home_team_id = str(meta["home_team"]["id"])` and passes it; the route call is now correct | ✅ |
| **B4** `parts_m` never initialised | `parts_x, parts_y, parts_g, parts_p, parts_m = [], [], [], [], []` | ✅ |
| **B5** `all_feats` | corrected to `all_features` **and quoted from source** (`:224`, `:259`, "NOT all_feats") | ✅ |

**The symbol-provenance table (lines 60-66) is the right second habit.** It does for symbol names what
the kill-line table did for guards — every accumulator, every helper, cited to its definition line.
This is the discipline I asked for last round, generalised into a checklist artifact. Where the plan
applies it, it is now correct throughout.

Which is why the one place it *didn't* reach is the one thing left.

---

## BLOCKER

### C1 · Task 11 Step 3 — the nested-HPO rewrite calls a `_fit_score` that its own note forbids writing, and that cannot exist as described

The rewrite **replaces** `_paired_data_effect` wholesale. The new body calls, three times:

```python
base_nested = _fit_score(X, y, candidates["public"] & trainable, fold_params["public"], te_idx, seed)
s_nested    = _fit_score(X, y, m, fold_params[name],   te_idx, seed)
s_shared    = _fit_score(X, y, m, fold_params["public"], te_idx, seed)
```

That is a **6-argument** `_fit_score(X, y, mask, params, te_idx, seed)`: full arrays, a row mask, and
— critically — a **`params` argument**, because the whole point of the M4 fix is to score the same
candidate at two different hyperparameter sets (`fold_params[name]` for nested, `fold_params["public"]`
for shared).

Immediately below, the plan says:

> **"Do NOT write a new `_fit_score`.** One already exists as a closure inside `_paired_data_effect`
> (`train_xshot_occurrence.py:185`), and it is correct in three ways a rewrite would break."

I opened `:185`. The existing closure is:

```python
def _fit_score(Xtr, ytr, te_idx):              # THREE args — pre-masked arrays
    ...
    p_ = dict(_pinned_params(shared_params))    # hardcodes shared_params — no params argument
```

Three contradictions, any one of which stops the task:

1. **It is deleted by the very rewrite that calls it.** The closure lives *inside* the old
   `_paired_data_effect`. The new body replaces that function and does not define `_fit_score`. After
   the edit, the three calls hit an undefined name → `NameError`. The note points at a closure that no
   longer exists.
2. **Its signature is wrong** — 3 args (`Xtr, ytr, te_idx`) versus the 6 the new code passes, and a
   different calling convention (pre-masked arrays vs `X, y, mask`).
3. **It cannot do the job even if kept.** It hardcodes `_pinned_params(shared_params)`. The nested
   protocol *requires* per-candidate params (`fold_params[name]`). A `_fit_score` with no `params`
   argument cannot produce `s_nested` at all — the primary quantity that decides the ship.

There is also a dropped behaviour: the 6-arg calls pass no `negative_subsample`, yet the note itself
lists train-fold subsampling as one of the "three ways a rewrite would break". A correct new
`_fit_score` must be a nested closure that captures `negative_subsample` from the enclosing scope — i.e.
exactly the "new `_fit_score`" the note forbids.

**And the same block has a second break:** the in-fold tuning calls

```python
fold_params[name] = _hpo_once(X[m], y[m], groups[m], out_dir=None, tag=..., ...)
```

`_hpo_once` (`:75`) does `store=StoreConfig(kind="sqlite", path=str(out_dir / f"study_{tag}.db"))`.
`None / f"..."` → `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'`. Nested tuning
runs K×3 = 15 studies per model; either give each a real path (a per-fold temp dir) or add an
in-memory store branch to `_hpo_once`. As written, the first fold crashes.

**The fix is small but it must be written down**, because Task 11 is the M4 correction — the reason the
budget is 45–60 DGX-hours, and the rule that decides what ships. Replace the "do not write one" note
with the actual definition:

```python
    def _fit_score(X, y, mask, params, te_idx, seed):
        Xtr, ytr = X[mask], y[mask]
        if len(np.unique(ytr)) < 2 or len(np.unique(y[te_idx])) < 2:
            return float("nan")
        if negative_subsample:                      # captured from the enclosing scope
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed)
            if len(np.unique(ytr)) < 2:
                return float("nan")
        p_ = dict(_pinned_params(params))           # params is now an ARGUMENT, not shared_params
        p_["base_score"] = float(ytr.mean())
        c = xgb.XGBClassifier(**p_)
        c.fit(Xtr.to_numpy(float), ytr)
        return average_precision_score(y[te_idx], c.predict_proba(X.iloc[te_idx].to_numpy(float))[:, 1])
```

Keep the NaN-drop at the call site (`if not np.isnan(...)`), which the plan already preserves.

**Note the pattern.** This is the same defect class as the last two rounds — the plan asserting a fact
about existing code (*"`_fit_score` already exists and is correct"*) that does not hold once you open
the file. The symbol-provenance table was built to catch exactly this, and it lists the *accumulators*
and *helpers by name* — but it never checked `_fit_score`'s **arity and parameters** against the new
call sites. The table needs one more column: not just "does this symbol exist at line N", but "does the
signature at line N match how the new code calls it".

---

## Minor (none blocking; fix in passing)

- **`_TOL_BALL` test is still a constant-equality tautology.** `assert _TOL_BALL == 15.0` proves
  nothing behavioural, and its message is self-contradictory (15 m also sits above the 9 m maximum, by
  design). It is now *harmless* — the gate's firing is genuinely covered by `test_a_wild_ball_is_excluded`
  and `test_catastrophic_break_is_excluded` — so this is a mislabeled redundant check, not a vacuous
  guard. Delete it or rename it to what it is: a calibration pin.
- **`cache_is_valid` drops the `features.parquet` existence check.** In the happy path this is safe
  *if* `write_cache_meta` is always called after the arrays are saved (meta present ⟹ features
  present). It is not robust to a half-written cache (meta present, parquet absent → `FileNotFoundError`
  at read). One `and (cache_dir / "features.parquet").exists()` closes it cheaply; low severity because
  the write order makes it rare.
- **Task 12 still has duplicate step numbers** (1, 2, 3, 4, **3**, **4**, 5, 6). A tick-off hazard in a
  checklist an implementer works down. Renumber to 1–8.

---

## Bottom line

Four rounds ago the defect class was *guards that cannot fail*; the kill-line table closed it. Two
rounds ago it was *symbols asserted from memory*; the provenance table closed most of it. **C1 is the
last instance of that same class** — a helper asserted to "already exist and be correct" that, once you
open `:185`, has the wrong arity, hardcodes the parameter the new design must vary, and is deleted by
the rewrite that calls it.

Fix C1 (define the params-aware `_fit_score`; give `_hpo_once` a real store path), clear the three
minors, and the plan is executable. Everything else — all five prior blockers, both discipline tables,
the whole of Tasks 1–10 and 12–16 — I could not fault on this pass.
