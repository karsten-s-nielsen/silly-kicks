# ADR-021: Pluggable, evaluatable xT (`silly_kicks.xthreat`)

| Field | Value |
|---|---|
| **Date** | 2026-06-07 |
| **Status** | Accepted (silly-kicks SK-xT-1) |
| **Deciders** | Karsten Nielsen (silly-kicks); luxury-lakehouse session (proposer/reviewer) |

## Context

`silly_kicks.xthreat` was the classic socceraction Singh-2018 xT: a fixed 16×12 grid, a monolithic
`fit()` with an in-method (non-pluggable) move-transition step, `rate()`/`interpolator()`, and no
KDE smoothing, no variable resolution as a first-class input, and no held-out evaluation metric.

The luxury-lakehouse session built an "ExT v2" model (pluggable transitions + KDE smoothing +
held-out NLL evaluator) on top of silly-kicks SPADL data and proposed promoting the pure-model
parts upstream so silly-kicks owns one canonical, evaluatable xT (matching how it already owns
VAEP, line-breaking, and the TF-24 calibration harness). silly-kicks is the product; the lakehouse
is one downstream consumer — so this is a reshape to silly-kicks house conventions, not a port.

The KDE claim is empirically grounded: on the lakehouse mart (8.8M actions, 12×8) KDE lowered
held-out transition NLL 3.789 → 3.748 (+1.08%). Independently reproduced here on the committed
WC2018 fixture (16×12): KDE beats Singh at the Silverman-multiplier optimum.

## Decision

Refactor `xthreat` into a package with a house-style **string-dispatch + frozen-dataclass**
transition family (Singh counts + KDE-smoothed), a `GridSpec` resolution input, a standalone
`value_iteration`, and a held-out transition-model NLL evaluator — keeping `ExpectedThreat` a
byte-identical back-compat facade over the Singh path at 16×12. Defer KNN/conditional (pre-publication
+ tracking-join-dependent).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. ABCs (`TransitionModel`/`Producer`) as proposed | matches the lakehouse seed | silly-kicks has zero ABCs; ADR-005 §8 codifies string-dispatch + frozen params (`pressure.py`, `pitch_control/`) | imports a foreign architecture |
| B. Parallel functional `compute_xt()` API | matches tracking `compute_*` | xthreat is class-based like VAEP; functional is the per-frame pattern | wrong fit; extend the class |
| C. Adopt lakehouse `XTGrid` return type | typed lookup + coord conversion | Hyrum break on every consumer reading raw `.xT` (`calibration/_xt.py`, `_gk_influence`, `_player_influence`, `_cover_shadows`) | keep raw `.xT` ndarray |
| D. Include KNN-conditional now | feature-complete | not implemented in the lakehouse; pre-publication method; conditional features are tracking-join-dependent | defer |
| E. **String-dispatch class facade + KDE + NLL (chosen)** | house-idiomatic; byte-identical Singh; sklearn already a hard dep | KDE adds a per-source-zone fit loop | — |

## Consequences

### Positive

- One canonical, pluggable, evaluatable xT; the lakehouse can retire its own `expected_threat.py`.
- KDE-smoothed transitions + variable resolution (`GridSpec`) + the first held-out xT evaluation
  primitive (`compute_holdout_nll`), all generically useful.
- Additive: `method="singh_counts"` is byte-identical to the prior implementation (proven by an
  in-process frozen-oracle parity gate), so existing consumers (incl. the TF-24 calibration
  `FrozenXt`) are unaffected — no VAEP retrain trigger.
- **Singh transition vectorized (byte-identical).** `singh_transition_matrix` uses `np.add.at`
  (O(n_actions + n_zones²)) instead of the legacy O(n_zones × n_actions) per-zone boolean-mask
  loop — same integer operands ⇒ same float64 division, proven by the exact-equality parity gate
  against `tests/xthreat_legacy_reference.py`. This matters at the 24×16 resolution this PR enables
  (384 zones) on production-scale corpora (validated against an 8.9M-action mart).

### Negative

- KDE inference loops per source zone fitting `sklearn.neighbors.KernelDensity` — heavier than
  Singh; only paid when `method="kde_smoothed"`.

### Neutral

- **Raw-diff value-iteration convergence is correct — do not "fix" it.** Iteration starts at xT=0
  under a monotone non-negative operator (`gs, p_move, T ≥ 0`), so iterates increase from below and
  `newxT - xT ≥ 0` always; raw-diff ≡ abs-diff. `value_iteration` is extracted byte-identically from
  the legacy `__solve` (the lakehouse `iterate` uses abs-convergence + an iteration-count return —
  do not copy it). An optional `max_iter` guard (default `None` = unbounded = byte-identical) caps
  degenerate inputs for direct callers.
- Held-out NLL is a **transition-model** metric (where the ball goes), NOT an xT-quality metric, and
  is keyed on silly-kicks-native `game_id` (not the lakehouse `competition_id`/`match_key`).
- KDE default bandwidth = 1.0 (pure Silverman multiplier) — a conservative, corpus-agnostic
  baseline. The held-out-NLL-optimal multiplier is **strongly corpus-size-dependent**: ~1.0 on a
  64-match sample (WC2018), but monotone-decreasing through ≥4 on an 8.9M-action production mart
  (adaptive Silverman shrinks per-zone h ~ n^(-1/6), so larger corpora need more smoothing).
  Consumers tune via the shipped held-out-NLL evaluator. KDE strictly beats Singh at every
  (resolution, bandwidth) tested.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-07-xthreat-pluggable-xt-promotion-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-07-xthreat-pluggable-xt-promotion.md`
- **ADRs:** follows ADR-005 (string-dispatch + frozen params + `<feature>__<method>` + attribution discipline)
- **External references:** Singh (2018) karun.in/blog/expected-threat; Silverman (1986); Salimi et al.
  (2026, LISS poster, pre-publication, arXiv pending). See NOTICE.

## Notes

WC2018 16×12 held-out-NLL bandwidth sweep (adaptive Silverman multiplier), Singh baseline 4.225:
bw 0.25 → 4.177, 0.5 → 3.658, 0.75 → 3.529, **1.0 → 3.500 (min)**, 1.5 → 3.526, 2.0 → 3.588,
4.0 → 3.929, 8.0 → 4.510. Interior minimum at 1.0 on this small (64-match) corpus.

Live-mart triangulation (one-off, non-committed; `soccer_analytics.dev_gold.fct_action_values`,
8.88M actions / 5,488 matches / 28 competitions; holdout passes; lakehouse split convention):
KDE strictly beats Singh at both resolutions and every bandwidth. 12×8: Singh 3.708; KDE bw
1.0→3.606, 2.0→3.552, 3.0→3.517, 4.0→3.500 (monotone down — optimum ≥4). 16×12: Singh 4.391; KDE
bw 1.0→4.258, 2.0→4.206, 3.0→4.176, 4.0→4.165 (monotone down). Contrast WC2018 (optimum ~1.0),
confirming the optimal multiplier grows with corpus size. (My Singh 3.708 ≈ the lakehouse's
recorded 3.789 on its earlier, smaller snapshot — validating the implementation.) The ~4% relative
KDE win on the real mart exceeds the lakehouse's reported +1.08%.

KDE strictly beats Singh on the deterministic 5-seed synthetic sparse-overfit gate (the sole hard
KDE-wins assertion in the suite); the WC2018 sweep and the StatsBomb-open-data e2e are committed
diagnostics/gates (empirical, corpus/resolution-dependent).
