# PR-S81 — Ghost-GK 4.7.0 re-fit + serve-carrier consistency fix + R3

**Status:** approved 2026-06-02
**Target version:** next minor off 4.9.0 (provisional — whoever merges re-bumps per the version-bump hard gate)
**Branch:** `pr-s81-ghost-gk-refit`

## 1. Problem

Two coupled defects on the Ghost-GK `team_in_possession` feature (`_ghost_gk.py:541`),
both found during the TF-16 weights cycle (PR-S80, 2026-06-02), plus a stale-corpus
re-fit obligation created when the 4.7.0 carrier defaults changed.

1. **Serve-carrier train/serve skew.** `compute_ghost_gk` (serve path, `_ghost_gk.py:1732`)
   calls `_extract_all_ghost_gk_features` **without** `carrier=`. With `carrier=None`,
   `_extract_one_frame` (`:541`/`:544`) sets `team_in_poss = 0.0` unconditionally. Meanwhile
   `prepare_ghost_gk_training_data` (train, `:860`) computes the real carrier via
   `infer_ball_carrier(frames)`. So the feature is correct at train and **always 0.0 at
   serve** — a pre-existing mismatch that contradicts the TF-18 spec §5 (which required serve
   to always compute the carrier). The divergence likely entered with the PR-S66
   frame-restriction or a part-deux fft refactor.

2. **No R3 carrier-param provenance.** Even once serve computes the carrier, Ghost-GK calls
   `infer_ball_carrier` with bare library defaults at both ends and never records/consumes the
   carrier params in `metadata.json` the way xS does ("R3"). A future `infer_ball_carrier`
   default change would silently re-skew the bundled model's serve output.

3. **Stale corpus.** The bundled weights (v1.0.0) were trained before the 4.7.0 carrier
   defaults (`beta` 0.5→0.0, `gamma` 1.0→0.25, `tolerance_m` held 3.0; PR-S79). The
   `team_in_possession` feature was therefore generated under the old carrier scoring.

`team_in_possession` is a long-tail feature (≪ `defensive_line_x` ≈ 15.2 in importance), so the
serve-output change is small — but it is a Hyrum-observable change for consumers incl. lakehouse.

## 2. Scope & sequencing

Per owner decision: **code-fix-first, re-fit last**, shipped as one PR-S81.

- **Part 1 (code):** serve-carrier fix + R3 record/consume, landed and tested against the
  *incumbent* weights first.
- **Part 2 (compute):** the 4.7.0 re-fit, driven by the maintainer over SSH on DGX Spark, run
  *after* Part 1 so the new weights bake in the corrected serve path + recorded carrier params.
- **Part 3 (validate + bundle + release):** per-variant non-regression gate, weight bundling,
  CHANGELOG/NOTICE, lakehouse heads-up.

Rationale (precise — see P5): the bundled **weights** are `f(prepare features, labels,
carrier_params)` and cannot encode a *serve* path; the serve fix (§3.1) touches only
`compute_ghost_gk`. So code-fix-first is **not** a weight-correctness requirement for the serve
path. It *is* load-bearing for two train-side reasons: (a) the R3 threading of §3.2 changes
`prepare_ghost_gk_training_data` (the train path), so the re-fit must run against that committed
code; and (b) it lets us validate `serve == train` feature parity on the freshly-trained weights.
Honors *Verify prereqs before long jobs*.

## 3. Design — Part 1 (code)

### 3.1 Serve-carrier consistency (`compute_ghost_gk`)

Compute the carrier on **full** frames and thread it into extraction:

```python
resolved = _resolve_model(model)
carrier_raw = infer_ball_carrier(frames, **resolved.carrier_params)
carrier_cols = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]
batch_features, meta = _extract_all_ghost_gk_features(
    frames,
    home_team_id=home_team_id,
    carrier=carrier_cols,
    score_at_time=score_fn,
    phase_at_time=phase_fn,
    link_frame_ids=link_frame_ids,
)
```

**Key correctness invariant:** the carrier is computed on the **full** `frames`, never the
`link_frame_ids` subset. `_extract_all_ghost_gk_features` already walks all frames (for the
per-period defending-goal mean-x and the cross-period velocity state) and the carrier lookup in
`_extract_one_frame` is keyed per `(game_id, period_id, frame_id)` — per-frame independent. So
the PR-S66 frame-restriction stays **byte-identical** for kept frames. This mirrors xS
(`_xshot_occurrence.py:776-781`), which explicitly computes the carrier on full frames even when
restricting the expensive per-frame work.

`add_ghost_gk` and `ghost_gk_xfns` both funnel through `compute_ghost_gk`; the atomic mirror
re-exports it. So the single edit fixes every serve surface.

**Optional carrier passthrough (N5 — cache convention).** The fix adds an `infer_ball_carrier`
call to **every** `compute_ghost_gk` invocation (serve previously skipped it — that was the bug).
`infer_ball_carrier` is non-trivial and its dominant cost is the param-invariant `pre` index,
which is exactly why the codebase has the precompute-cache convention (`links`,
`pitch_control_cache`, and `ball_carrier_at_action`'s `pre`/`links`; ADR-008 / TF-24
invariant-prepare). A pipeline calling ghost-GK repeatedly per match — or across families that
*also* compute the carrier (xS, possession) — would recompute it each time. So, mirroring `links`:

- `compute_ghost_gk(..., carrier: pd.DataFrame | None = None)` — when supplied (the
  `["game_id","period_id","frame_id","ball_carrier_team_id"]` projection of an
  `infer_ball_carrier` result), skip the internal call and use it directly; when `None`, compute
  it internally (default, self-contained).
- Thread the same optional `carrier` through `add_ghost_gk` and `ghost_gk_xfns` so a pipeline
  caller computes `infer_ball_carrier(frames, **model.carrier_params)` **once** and passes it to
  all ghost-GK surfaces (like pre-linking + `pitch_control_cache`). `ghost_gk_xfns` already calls
  `compute_ghost_gk` once for the gamestate union, so it passes its single carrier straight through.

This keeps ghost-GK consistent with every other tracking aggregator. Validation note: a
caller-supplied `carrier` must be computed on the **full** frames with the model's
`carrier_params` for the §3.1 byte-identical invariant to hold (documented on the kwarg).

### 3.2 R3 record/consume (`GhostGkModel`) — xS pattern, single-source

- `__init__`: add `self.carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)`
  (import the shared `DEFAULT_CARRIER_PARAMS` from `silly_kicks.tracking._ball_carrier` —
  the same single-source-of-truth constant xS consumes; no reflection).
- `fit(features, labels, *, carrier_params: dict | None = None)`:
  `self.carrier_params = dict(carrier_params) if carrier_params else dict(DEFAULT_CARRIER_PARAMS)`.
- `save`: add `"carrier_params": self.carrier_params` to `metadata`; bump
  `"version": "1.0.0" → "1.1.0"`. The npz/SHA256SUMS machinery is unchanged.
- `load`: `model.carrier_params = metadata.get("carrier_params", dict(DEFAULT_CARRIER_PARAMS))`.
  Back-compat: a v1.0.0 artifact lacking the field loads with the library default.

`compute_ghost_gk` consumes `resolved.carrier_params` (§3.1), so serve resolves possession
identically to how the loaded model was trained.

**Single-source train-side wiring (P1 — prevents R3 recording a lie; N1 — non-breaking).** Today
`prepare_ghost_gk_training_data` (`:810`) takes no `carrier_params` and computes the training
carrier with `infer_ball_carrier(frames)` **bare** (`:860`) — the library default. If `fit`
records a *separately-supplied* `carrier_params`, the two are independent sources that agree only
by coincidence (today's bare default *equals* 4.7.0). The instant `infer_ball_carrier`'s default
changes again — the exact scenario R3 exists to survive — a re-fit passing an explicit
`carrier_params` would record params that were **not** the ones used to compute the training
carrier.

`prepare_ghost_gk_training_data` returns a **documented public 2-tuple** `(features, labels)`
(its docstring example is `features, labels = prepare_ghost_gk_training_data(...)`). So it must
**stay a 2-tuple** — returning the params (a 3-tuple / result object) would break every existing
caller (Hyrum). The single source therefore lives in the **trainer**, not in a return value:

- `prepare_ghost_gk_training_data(frames, *, carrier_params: dict | None = None, ...)` —
  **additive kwarg**, uses `infer_ball_carrier(frames, **(carrier_params or DEFAULT_CARRIER_PARAMS))`.
  Default `None` is byte-identical to today's bare call because `DEFAULT_CARRIER_PARAMS` *equals*
  `infer_ball_carrier`'s signature defaults (no behavior change for existing callers). Return type
  unchanged.
- `train_ghost_gk.py` resolves **one** local `cp = carrier_params or DEFAULT_CARRIER_PARAMS` and
  passes the *same* `cp` to both `prepare(carrier_params=cp)` and `model.fit(carrier_params=cp)`.

Single source = the trainer's `cp`; recorded == used by construction; cannot desync; no public
break. Test #3 (§5.2) guards the wiring.

### 3.3 Out of scope (rejected alternative)

Dropping `team_in_possession` (26→25 features) to remove the carrier dependency outright.
Rejected: bigger break, loses signal, and R3 is the principled fix per the TODO.

## 4. Design — Part 2 (re-fit)

**Why re-fit at all (precise — P5):** the re-fit is justified by two *train-side* facts, not the
serve fix: (a) the **4.7.0 carrier defaults** (`beta` 0.5→0.0, `gamma` 1.0→0.25) change the
training `team_in_possession`, and (b) **R3 recording** needs the params baked into metadata.
Weights cannot encode a serve path.

- **Corpus:** re-pull ~81 matches via the pining owner token (`~/.pining_env`); skillcorner +
  idsse + gradientsports. The xS run cached only the xS feature matrix (not raw frames), and xS
  features are not reusable for Ghost-GK, so PR-S81 re-pulls. See the PR-S80 DGX-Spark recipe.
- **Train against a committed SHA (P6).** Do **not** train against a `git apply`'d working-tree
  patch — if the patch is edited before the single commit, the shipped weights won't match the
  committed train/extraction code. Instead: land Part 1 as a commit on the branch, push it (or a
  scratch ref) to the box, train against that checked-out SHA, and record `training_commit` in
  metadata (§7). The artifact is then traceable to its source. (This is compatible with the
  one-commit policy: the final squash collapses to one commit; the box just needs a reachable SHA
  of the Part-1 code, amended/rebased into the final commit before merge.)
- **Carrier params at train (single source, §3.2/P1/N1):** `train_ghost_gk.py` resolves one local
  `cp={beta:0.0, gamma:0.25, tolerance_m:3.0}` (4.7.0 defaults) and passes the **same `cp`** to both
  `prepare_ghost_gk_training_data(carrier_params=cp)` and `model.fit(carrier_params=cp)`, so
  metadata records exactly the params used to compute the training carrier. `prepare` stays a
  2-tuple (no public break).
- **Variants (axis = sample count → wheel size, NOT provenance):**
  - `full` — all in-domain samples, hosted on HuggingFace Hub (`silly-kicks/ghost-gk-v1`), ≤ ~91 MB.
  - `default` — subsampled slice bundled in the wheel, ≤ ~9 MB (PyPI 100 MB limit; hatch exclude
    pattern already excludes `_ghost_gk_weights/full/`).
  - Both record `carrier_params={beta:0.0, gamma:0.25, tolerance_m:3.0}`, version `1.1.0`.
- **GS inclusion:** GS is included for maximum KDE data. Ghost-GK is a data-hungry density
  *regression*; the xS "GS degraded public generalization" *classification* lesson does not
  transfer, and the bundled weights are not a public-reproducibility artifact.

## 5. Design — Part 3 (validation, gate, release)

### 5.1 Per-variant non-regression gate (apples-to-apples — P2)

Ship a re-fit variant only if it does not regress against the incumbent **on a common held-out
set**. The frozen incumbent MAEs (`default` 1.135 m, `full` 1.045 m, from their own old
`metrics.json`) were measured on *different* held-out sets, so comparing them to the re-fit's
held-out MAE conflates "different data" with "different model" — the exact H3 flaw the TF-16 spec
corrected for xS. Instead:

- Evaluate the **incumbent model on the re-fit's held-out folds** (cheap: the incumbent loads +
  predicts via numpy tree traversal, no retrain) and compare both MAEs on the **identical** set.
- Gate: re-fit ships if `MAE_refit ≤ MAE_incumbent_on_same_holdout + ε` (ε small, ~0.02 m or
  expressed as a small % of incumbent MAE — final value owner-confirmed at execution).
- **L3 overlap caveat (carried from PR-S80):** if the re-fit's held-out frames overlap the
  incumbent's *training* corpus, the incumbent looks unfairly strong (it has seen them), biasing
  the gate toward keeping the incumbent. Use a demonstrably disjoint held-out, or state the
  comparison is conservative-toward-incumbent and accept that bias.

**Keeping the incumbent is availability-safe, not correctness-safe (N4).** The §5.1 fallback keeps
incumbent weights (trained under the **old** `beta=0.5/gamma=1.0` carrier) and backfills their
metadata with those old params — R3-self-consistent, but the bundled model then serves a carrier
regime that **diverges from the library default** (`beta=0.0/gamma=0.25`, used by every direct
`infer_ball_carrier` call and the rest of the library). Combined with the L3 bias, a "conservative"
gate can **reject a perfectly good re-fit and preserve exactly the stale-corpus problem (§1 #3)
this PR set out to fix.** So: keeping the incumbent is the *availability*-safe direction, not the
*correctness*-safe one. If the gate rejects the re-fit, **record why** and treat a genuine refresh
(bronze-scale corpus) as a **tracked follow-up** (TODO row) — do **not** declare the staleness
resolved.

**Fallback ("if pining too thin → keep incumbent + record"):** if a variant regresses
(81 matches may be thinner than the incumbent `full`'s corpus), keep that variant's incumbent
weights but **backfill its `metadata.json` with `carrier_params={beta:0.5, gamma:1.0,
tolerance_m:3.0}`** (its actual training-time defaults) and bump version to `1.1.0`, so R3 keeps
its serve path self-consistent even without a re-fit.

### 5.2 Tests (TDD — RED first)

1. **Serve produces real `team_in_poss` (feature-matrix assertion — P7).** Assert on the internal
   feature matrix `compute_ghost_gk` builds, not just the final `ghost_gk_x/y`: a synthetic
   frames fixture where the GK's team is in possession at some frame yields `team_in_possession==0`
   today (RED) and the real carrier-derived value post-fix.
2. **Train==serve feature parity (the invariant the bug violated — P7).** Extract
   `team_in_possession` via `prepare_ghost_gk_training_data` vs via `compute_ghost_gk`'s internal
   extraction on the *same* frames + same `carrier_params` → identical. This is strictly stronger
   than "serve != 0".
3. **R3 recorded == used (P1; fixture must bite — N2).** Train with a **non-default**
   `carrier_params` (e.g. `beta=0.9`); assert (a) `metadata.json` records exactly those, **and**
   (b) the resulting `team_in_possession` reflects *that* carrier. `beta` (velocity-toward-ball)
   only changes the carrier on a **near-tie** frame, so the fixture must include a frame where two
   players are ~equidistant from the ball and one is moving toward it — making `beta=0.9` vs
   default **demonstrably flip** the carrier on the asserted frame. Otherwise (b) passes vacuously
   and only the metadata half bites. (Same "make the test bite" discipline as the 4.2.0 DAS onside
   fixture invoked in §5.2a.)
4. **Frame-restriction stays byte-identical** — extend `TestExtractionRestriction`: full vs
   `link_frame_ids`-restricted `compute_ghost_gk` produce identical predictions for kept frames,
   now with the carrier active (carrier computed on full frames).
5. **R3 round-trips** — `fit(carrier_params=...)` → `save` → `load` preserves `carrier_params`;
   metadata JSON contains the field; version is `1.1.0`.
6. **Back-compat load** — a v1.0.0 metadata (no `carrier_params`) loads with
   `DEFAULT_CARRIER_PARAMS`.
7. **Atomic mirror** — atomic re-export inherits the serve fix (carrier active at serve).

### 5.2a Measured serve-output impact (P3 — not asserted from importance)

The serve fix flips `team_in_possession` from **always-0** to its **real** value on possession
frames — a *maximal* change for that feature, not a perturbation. "Low importance ⇒ small output
shift" is the exact inference that produced the false 4.2.0 DAS "value-neutral" claim (a lakehouse
golden later caught it; corrected in 4.4.1). So **measure it**: on a real match (pining), compute
`ghost_gk_x/y` with the buggy serve vs the fixed serve and report the actual **max + median
delta** (metres). That measured number — not the word "small" — is what the §5.3 CHANGELOG note
and the lakehouse heads-up carry.

### 5.3 Release & Hyrum

- Minor bump off 4.9.0 (provisional; re-bump at merge per the version-bump hard gate:
  `pyproject.toml` + `__init__.py` + `TODO.md` + `CHANGELOG.md` must all match).
- CHANGELOG `### Changed`: bundled Ghost-GK **serve output changes**, carrying the **measured**
  max/median `ghost_gk_x/y` delta from §5.2a (not the word "small") — Hyrum-flagged for consumers
  incl. lakehouse. NOTICE unchanged (no new methodology; R3 is provenance plumbing).
- **Driven by the bug fix, not (only) the re-fit (P4).** The serve-carrier fix (Part 1) shifts
  served `ghost_gk_x/y` on possession frames for **every** variant — including the §5.1 fallback
  where incumbent weights are kept (serve now computes the real carrier with the backfilled
  params). So the Hyrum/lakehouse change ships even if the re-fit is skipped entirely; state this
  explicitly in CHANGELOG.
- **Lakehouse heads-up** rides along, carrying the measured delta.
- TODO.md: delete the PR-S81 row; the structural R3 work is now done (no longer "a further
  separate PR").

## 6. Files touched (Part 1)

- `silly_kicks/tracking/_ghost_gk.py` — `compute_ghost_gk` serve fix + optional `carrier=None`
  passthrough (N5; lazy function-scope `from ._ball_carrier import infer_ball_carrier` matching
  prepare's `:853` import — P9); `GhostGkModel` `__init__`/`fit`/`save`/`load` R3;
  `prepare_ghost_gk_training_data` gains additive `carrier_params` kwarg, **return type unchanged**
  (stays `(features, labels)` 2-tuple — N1).
- `silly_kicks/tracking/features.py` — thread optional `carrier=None` through `add_ghost_gk` and
  `ghost_gk_xfns` (N5).
- `scripts/train_ghost_gk.py` — resolve one `cp` and pass the same `cp` to both
  `prepare(carrier_params=cp)` and `model.fit(carrier_params=cp)` (N1); record `training_commit`/
  `sklearn_version`/`training_platform` (§7).
- `tests/tracking/test_ghost_gk*.py` — the §5.2 tests (+ atomic mirror test).
- Atomic surface inherits via re-export (verify, no new logic).

## 7. Metadata provenance (P6/P8 — mandatory, not optional)

Record in `metadata.json` at re-fit (mirrors xS L2 — the artifact is ARM-trained on Spark but
served on x86/CI):

- `carrier_params` (R3, §3.2).
- `training_commit` — the committed branch SHA the weights were trained against (P6); makes the
  bundled artifact traceable to its source code. **N3 caveat:** under the one-commit policy the
  Part-1 commit is squashed at merge, so this SHA is the **pre-squash PR-S81 branch SHA**
  (preserved on the PR / reflog, **not** an ancestor of main) — `git show <training_commit>` works
  on the PR refs, not on main. Documented as such so nobody expects main to contain it.
- `sklearn_version` + `training_platform` — `HistGradientBoostingRegressor` leaf assignment must
  be reproducible across the ARM-train / x86-serve boundary. Pin `random_state` (already in
  trainer) and document the determinism contract (single-thread if needed).

## 8. Open risks

- **Corpus thinness** — 81 matches may yield a worse `full` than the incumbent. Mitigated by the
  §5.1 apples-to-apples gate + backfill fallback.
- **Cross-platform determinism** — see §7; if HGBR leaf assignment proves non-reproducible
  ARM↔x86, the byte-identical-load contract is via the stored node arrays (load uses numpy
  traversal, not sklearn), so *inference* is platform-stable; only *training* reproducibility is
  at issue, and that only affects exact weight regeneration, not served output.
