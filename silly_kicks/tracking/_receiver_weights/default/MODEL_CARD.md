# Expected-receiver model — `default` variant (StatsBomb 360, positions-only)

**What it is.** The intended-receiver probability `P(this candidate is the pass's intended target |
pre-pass state)`, one row per candidate teammate. Logistic regression; sklearn at fit, pure-numpy
`sigmoid(Xβ)` at serve (no runtime sklearn). Loaded via `ReceiverModel.from_variant("default")` and
auto-selected for every provider: `variant_key_for_provider` maps `statsbomb → default`, and the
provider-preference keys `gs_owner`/`skillcorner` **degrade to this `default`** because no owner variant
earned separate bundling (see "Provider strategy").

**Feature set — POSITIONS ONLY, leakage-free (`ball_dist`, `lane_pressure`, `space`).** The model reads
only pre-pass state; it NEVER reads the pass end/loss location (the pass-event angle is origin→end and is
banned). The velocity-derived `release_dir_align` + `closing_speed` of the `owner` feature set are
absent here — velocity-less SB360 freeze frames cannot carry a leakage-free release direction, so the
bundled default is the positions-only construct that is valid on any provider. The ban is pinned by an
output-invariance guard: perturbing `end_x`/`end_y` leaves the extracted features byte-identical
(`tests/tracking/test_receiver_leakage_guard.py`).

**Label construct — TRAJECTORY (SB360 carries no player identity).** SB360 freeze-frame rows are
numbered, not identified (ADR-062), so an id-based receiver label yields zero positives. The intended
receiver is instead the teammate nearest the release→next-touch-START ray within a tight lane; an
ambiguous pass is DROPPED (never all-zero, never nearest-guess). Strategy is fixed per frame-set by
`labeling_strategy_for_provider` (`statsbomb → trajectory`).

**Training corpus + gate.** 30 WC2022 StatsBomb 360 open-data matches (the public open-data corpus,
`corpus_visibility: public`). Match-stratified 5-fold `GroupKFold`: **top-1 CV 0.510**
(folds 0.523 / 0.510 / 0.493 / 0.521 / 0.501), n_passes 7229 KEPT of 16635 completed
(43.5% label coverage — the drop-on-ambiguous thinning is deliberate and visible); candidate-set size
mean 6.4 (p10 4 / p50 6 / p90 9). See `metrics.json`.

**Provider strategy — SB360-primary; GS earned neither pooling nor a separate bundle.** A held-out
`pooling_gate` tested pooling Gradient Sports into this model: **REJECTED** (pooled top-1 0.488 <
primary 0.510, margin −0.022), so the default stays SB360-only. The `owner` (GS, +velocity) variant was
trained on 64 GS matches (top-1 CV 0.403) and put through the M-A(ii) deployment gate on GS-failed
passes: **non-decisive** (owner 0.299 vs public 0.266, margin +0.033 < the 0.05 `MIN_RECEIVER_MARGIN`
floor; velocity added only +0.34 pp on completed passes). Per the pre-registered gate the GS variant is
NOT bundled — the honest-null the design anticipated. A future cycle that earns `gs_owner` simply ships
that dir and `from_variant` uses it automatically.

**Missing-value policy.** Features are standardized (stored `mean`/`std`); a NaN feature on a linked
frame mean-imputes to the neutral post-standardization value. A pass with no usable frame link is not
scored (the caller keeps its 4.87.0 leaked-`end_xy` behaviour when no model is supplied).

**Integrity (ADR-011).** Parameters-only, pickle-free: `model.json` + `SHA256SUMS`. `load()` verifies
the SHA (CRLF-normalized, so git line-ending translation is safe), a behavioural **chirality**
fingerprint (a y-mirror-mis-served artifact raises), and the **feature contract** (the feature vector +
geometry constants the extractor consumes). Serve is pure-numpy — sklearn is never imported at inference.

**Provenance + reproduction.** `metrics.json` records `run_commit`
`08347cd6d6f0451707f29209dc1e4fa73a0f4b90` and `run_tree_dirty: false` (produced from a clean tree).
Reproduce with:

```
python -m scripts.train_receiver_model --feature-set public --provider statsbomb \
  --out <dir> --shard-root <shards> --cache-dir <cache>
```

**Attribution.** Power, Hobbs, Ruiz, Wei & Lucey, "Not All Passes Are Created Equal" (KDD 2017). See
`NOTICE` for full bibliographic citations.
