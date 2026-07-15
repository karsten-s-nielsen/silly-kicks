# Plan review 2 — `2026-07-14-skillcorner-corpus-and-visibility.md` (rev 2)

**Reviewer:** external session · **Date:** 2026-07-14 · **Verdict: five mechanical breaks — two silent and severe. Not ready, but the hard part is done.**

## The four blockers are genuinely fixed, and the habit is installed

All verified against the source, not the prose:

| | fix | verdict |
|---|---|---|
| **P1** interpolator tell | **Retired** (spec → rev 5) and replaced by `ghost_admission_report() -> (verdict, reason)`. The test asserts out loud that it decides nothing: `assert ghost_admission([0.1]*5) is False`. | ✅ And the plan found a *better* argument than mine: under rev 3's detected-only **training** rule the model never sees an interpolated target, so the mechanism the tell guarded no longer exists. Retiring it beats my divergence-rule suggestion. |
| **P2** rate-gate never called | Gate now runs at the end of `convert_to_frames`, surfaces on `TrackingConversionReport`, and the loader **`continue`s** on `geometry_excluded` — with that `continue` named as the kill-line. | ✅ §4.4's exclusion mechanism finally exists |
| **P3** compliance tested at the wrong layer | New red-first `test_a_restricted_corpus_NEVER_ships_a_public_label` drives `tr.main()` and asserts on `metrics["shipped_variant"]` (key verified at `train_xshot_occurrence.py:371`), with the kill-line named and "red for the wrong reason proves nothing" stated. | ✅ (but see **B2** — it cannot currently run) |
| **P4** Task 12 untested + tautological assert | Tests written **first**, kill-lines named; the tautology is replaced by a real non-vacuity check: `assert report.n_excluded_keepers > 0, "the domain exclusion removed nobody -- it is inert"`. `_ghost_domain.py` extracted so it is testable without a fit. | ✅ |

M2 (`test_skillcorner.py`), M3 (`_resolve_token`), M4 (`_pinned_params` / `base_score` /
`negative_subsample` / `.iloc`) and M5 (NaN folds now dropped in `ghost_admission`) are all fixed.

**The "Guard discipline" section with its kill-line table is the right artifact** — it converts a
recurring defect into a required step. That is the thing that stops the fifth round.

**And you found something I missed.** Task 12 Step 4's **detection selection bias** — detected frames
over-represent the *engaged* keeper because the camera follows the ball, so training detected-only
under-samples exactly the deep sweeper regime GKDV cares about — is a real cost of the B1 fix that
neither the spec nor I had surfaced. Registering it as a measured, stated limitation rather than a
gate is the right disposition.

---

## BLOCKERS

### B1 · `match_id` is parsed out of a temp filename — and it is wrong for **all ten public matches**

Task 7, Step 3:

```python
bronze = _skillcorner_bronze(raw, meta, match_id=str(paths["tracking"]).split("_")[-2])
```

I executed the plan's own `_dest_name` (`f"{provider}_{match_id}_{artifact_key}{ext}"`) against both
schemas:

| corpus | temp filename | `split("_")[-2]` |
|---|---|---|
| **canonical 10** (suffix-resolved key = the file stem) | `skillcorner_1886347_1886347_tracking_extrapolated.jsonl` | **`"tracking"`** ❌ |
| private 98 (role key) | `skillcorner_1021404_tracking.json.gz` | `"1021404"` ✅ |

`match_id` becomes `game_id` for every frame (`skillcorner.py:138`), and **`game_id` is the grouping
key for `StratifiedGroupKFold`**. So all ten public SkillCorner matches collapse into a **single CV
group** called `"tracking"`, and the public arm — the arm that decides what ships — silently drops
from 17 groups to 8. Every paired verdict in §4.1 is computed on that fold structure.

Nothing catches it: no test covers `build_skillcorner_frames` (the unit test calls
`_skillcorner_bronze` directly with `match_id="m1"`), and `PUBLIC_CORPUS` checks the *manifest*, not
`game_id`.

**Fix:** never derive an identity from a temp path. `_build_skillcorner(paths, match_id,
tracking_limit)` **already receives `match_id`** — thread it into `build_skillcorner_frames`.

### B2 · Task 9's flagship test cannot pass — `assert_public_corpus` kills it first

`assert_public_corpus` is called **unconditionally** in the arm-split path and demands the manifest's
public set equal **all 17**:

```python
    vis = match_visibility(sorted(set(providers.tolist())))
    assert_public_corpus(vis)               # raises SystemExit unless seen == the full 17
```

But Task 9 Step 2's own red-first test runs:

```python
    tr.main(["--providers", "skillcorner", "--output-dir", str(out), "--n-trials", "1"])
```

with `match_visibility` monkeypatched to two entries. So `seen = {("skillcorner","1886347")}` ≠ the
17 → **`SystemExit`**, before the `shipped_variant` assertion is ever reached. The test that fixes
the licensing landmine cannot run.

The same guard also kills any legitimate partial run: `--providers gradientsports` yields
`seen = ∅` ≠ 17 → `SystemExit`.

The *intent* (drift detection) is right; the *scope* is wrong. Assert **`seen ⊆ PUBLIC_CORPUS`** — no
match outside the registry may ever be classified public — and demand equality only when the run
actually requests the full public arm.

### B3 · The `home_team_id` break survives in production (tests were fixed; the route was not)

Task 7, Step 3:

```python
frames, _report = tracking_sk.convert_to_frames(bronze, output_convention="absolute_frame")
```

`convert_to_frames(bronze, *, home_team_id, …)` — **required keyword-only** (`skillcorner.py:91-99`).
→ `TypeError`, on the new SkillCorner route itself.

The note beneath it is a misread of the code:

> "*`_build_skillcorner` already reads `meta` for `home_team_id` and passes it on — leave that alone.*"

It does not. `_build_skillcorner` computes `home_team_id` **after** calling `build_skillcorner_frames`,
for its own return tuple. The function being rewritten never has it. `meta` is right there in scope —
pass `str(meta["home_team"]["id"])`.

### B4 · `parts_m` is appended to but never initialised

Task 9 Step 4 adds `parts_m.append(np.array([str(mid)] * len(X)))`, but `_extract` initialises only
four accumulators (`train_xshot_occurrence.py:52`: `parts_x, parts_y, parts_g, parts_p = [], [], [], []`).
The plan says "return the 5-tuple" and never adds the fifth list. → `NameError`.

### B5 · Task 12 still uses `all_feats` — and *restates* the wrong fact rather than checking it

> "*The extraction loop currently accumulates `all_feats`, `all_labels`, `all_game_ids`,
> `all_providers`.*"

It does not. The real name is **`all_features`** (`train_ghost_gk.py:224`, `:259`). The plan then
writes `all_feats.append(feats)` → `NameError`. This was in the previous review and was not fixed;
the revision asserted the claim more confidently instead of opening the file.

---

## Minor

- **Task 12 has duplicate step numbers:** 1, 2, 3, 4, **3**, **4**, 5, 6. In a checklist an
  implementer ticks off, that is a skip hazard — "Step 3" and "Step 4" each appear twice with
  different content.
- **`cache_is_valid` drops the existence check.** It tests only `cache_meta.json` +
  `schema_version` + fingerprint, and it *replaces* the `features.parquet` existence predicate. A
  directory with a valid meta and no features file is now a **hit** → `FileNotFoundError`. Add
  `and (cache_dir / "features.parquet").exists()`.
- Task 9's licensing test calls `json.loads` without importing `json`; `subprocess`, `numpy`, `pandas`
  are imported unused. Task 15 gates on `ruff`, so this fails the gate (F401).
- **The `_TOL_BALL` test is still a tautology.** `assert _TOL_BALL == 15.0` is a constant-equality
  check, and its message — *"30.0 m sat above every real excursion — the gate could not trip"* — is
  incoherent: **15.0 also sits above every real excursion** (9.00 m), and must, or it would fire on
  correct data. The *value* is well-reasoned in the implementation comment (67% headroom); the *test*
  proves nothing behavioural. Replace with: a broken transform trips it, real data does not.

---

## The pattern has shifted — and so should the check

Last round the defect class was **guards that cannot fail**. The kill-line table fixed it, and fixed
it properly: I could not find a new vacuous guard anywhere in rev 2.

This round the defect class is different: **the plan asserts facts about code it did not open.**
`all_feats` (it is `all_features`), the `home_team_id` note (it does the opposite of what the note
says), `parts_m` (never initialised), `match_id` from a path split (wrong for 10 of 108 matches). Same
root cause as before — writing from memory rather than from the file — but it slips past a kill-line
check, because a guard can be perfectly designed and still name a symbol that does not exist.

The companion habit is as cheap as the first: **for every symbol the plan names, quote its definition
line from the source next to it** — as Task 8 already does (`_resolve_token` … "*real helper names,
verified: `_loader_pining.py:66`*"). Where the plan does that, it is correct. Where it does not, it is
wrong. Do that pass across all 16 tasks.

**B1 is the one to fear.** The other four break loudly on the first run. B1 runs green, trains a model,
produces a paired verdict, and ships it — on a public arm whose ten SkillCorner matches have quietly
become one.
