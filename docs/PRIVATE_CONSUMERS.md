# Private-module consumers

Downstream code that knowingly imports silly-kicks **private** (underscore-prefixed) modules or
pins their paths. Underscore modules carry **no stability promise** — this file exists so a
refactor can see its blast radius, not to turn these into supported API.

**Contributor rule:** before renaming, splitting or relocating any `silly_kicks/**/_*.py` listed
here, treat it as cross-repo coordination. A path pin in particular fails **silently** — no
`ImportError`, just a degraded consumer.

Verified 2026-07-18 with the luxury-lakehouse session. Line numbers are theirs and will drift;
the module/consumer pairing is the durable part.

| silly-kicks private | What is used | Consumer (luxury-lakehouse) | Why | Exit condition |
|---|---|---|---|---|
| `tracking/_xt_gk.py` | `XtGkParams`, **private fn** `_resolve_completion_for_frames` | `src/analytics/action_context/enrich.py:490` | No public seam exposes per-frame completion resolution | **Lakehouse migrates to xT-GK v2** (`xtgk.compute_xt_gk_v2`). v1 is already frozen and is removed ≥1 release after that migration. |
| `tracking/_xt_gk.py` | `XtGkReport` | `src/analytics/action_context/pipeline.py:98` | Aggregate QA type is not re-exported publicly | Same v2 migration; or promote the report type if v2 keeps an equivalent |
| `tracking/_ghost_gk.py`, `_xt_gk.py`, `_gk_completion.py`, `_gk_geometry.py` | **module PATHS as hardcoded strings** | `src/ingestion/exec_visibility.py:467-472` (their ADR-044 executor-env drift guard) | Needs stable module identities to detect executor-env drift | A public introspection surface for shipped-module identity, **or** an accepted standing pin coordinated on rename. **Highest-risk entry: degrades silently.** |

**Public output value changes** — not a private-module import, but the same blast-radius discipline:
a public column whose VALUES moved (not its schema) can silently break a downstream consumer that
built assumptions on the old distribution.

| silly-kicks column | What changed | Consumer (luxury-lakehouse) | Why | Exit condition |
|---|---|---|---|---|
| `spadl.statsbomb` output `cross_blocked` | Flips from all-`pd.NA` to a real `True`/`False` mask (ADR-046 amendment, `4.86.0`) — an open-play `cross` whose `related_events` links to an OPPOSING-team `Block` (not `block.offensive`) | Any consumer schema/pipeline that added `cross_blocked` on StatsBomb assuming a stable all-`pd.NA` column (e.g. a lakehouse mart column typed/partitioned around "always null") | The ADR-046 deferral (n=1-verified, "fragile join") is discharged — a pre-registered probe over ~510 open-data matches passed all three ship rules (`docs/research/sb360_cross_blocked/`) | Re-check any downstream null-handling / dtype assumption on StatsBomb `cross_blocked`; the column now carries real signal on open-play crosses (still `pd.NA` on set-piece crosses and non-cross rows) |

**Note:** the public TF-51 `compute_bravery` (`tracking/defensive_credit/_bravery.py`) reads
`cross_blocked` directly (no `*_xfns`, so still no retrain); `add_press_commitment` does NOT.
A downstream consumer computing TF-51 bravery via silly-kicks on StatsBomb sees the
`bravery_open_play_crosses` leg move from NA-unknown to cross-inclusive. The luxury-lakehouse
materializes `press_commitment` (which does not read `cross_blocked`) and DEFERS `compute_bravery`,
so it is UNAFFECTED by this change.

**In-repo (first-party) consumers** — not the lakehouse, but recorded under the same discipline
(a permanent CI test coupling to a private module is worth a rename's blast-radius, even in-repo):

| silly-kicks private | What is used | Consumer (in-repo) | Why | Exit condition |
|---|---|---|---|---|
| `tracking/_model_eval.py` | probe symbols `xs_substitution_probe`, `xs_substitution_probe_v2`, `evaluate_xs_probe`, `substitution_deltas`, `regate_verdict`, `PROBE_WRAPPERS`, `_validate_targets` (+ gkdv-declared `_TARGET_COLUMNS`) | `tests/gkdv/test_xs_probe_wiring.py` + `tests/scripts/test_validate_xs_probe.py` (permanent CI) and `scripts/validate_xs_probe.py` (TF-19 PR-3b driver) | The registered TF-19 probe is a first-party **research instrument** (in no xfn list); ADR-037 kept `_model_eval` private on purpose (out of production coupling). These are real `import`s, so a rename fails **loudly** at collection, not silently. | Promote to `silly_kicks.tracking.__all__` **only if a cross-package consumer appears** — the lakehouse wanting the verdict, or the Part B §6.4 harness importing the probe. Until then, recording here is sufficient and no promotion is warranted (YAGNI + ADR-037). |
| `tracking/_cover_shadows.py`, `tracking/_geometry.py` | `lane_control`, `LaneControlResult`, `CoverShadowParams`; `GEOMETRY_VERSION` | `scripts/build_rq_pass_scores.py` + `scripts/_rq_corpus.py` (the cover-shadow RQ1 + pass-risk validation cycle, 2026-08-19) | The per-`(passer, receiver)` lane primitive + its frozen params have no public re-export; the driver validates the SHIPPED cover-shadow model against real GS pass outcomes, so it needs the low-level lane call (not the aggregate `add_cover_shadows`). Real `import`s -> a rename fails **loudly** at collection. | Promote `lane_control` to `tracking.__all__` only if a cross-package consumer appears; until then recording here suffices (YAGNI). |

**Retired entries** (kept so the question is not re-asked):

- `tracking/_id_compat.py` → `ids_match`, consumed by the luxury-lakehouse
  `src/tests/action_context/test_frame_orientation_golden.py:49` (test only). Its exit condition was
  *"promote the `_id_compat` helpers to a public surface"*. **That condition is met in 4.53.0
  (PR-S120):** the module is now the **public** `silly_kicks/id_compat.py`. ADR-019 requires every
  consumer in the repo to route id comparisons through it, and a mandatory seam is public API by
  definition — the underscore was a false signal, and it was blocking `gkdv` from truthfully
  claiming it depends on public tracking seams only.

  **Deliberate clean break — no compatibility shim at the old private path.** The consumer pin is an
  `import`, so it fails **loudly** (`ImportError` at collection) rather than degrading silently; the
  silent-degradation risk this file exists to catch belongs to the *path-string* pin below, not to
  this one. A shim at `tracking/_id_compat.py` would also have left the promotion cosmetic: nothing
  would ever migrate, and the private path would stay the one everyone cites.

  **Lakehouse migration — one line, mechanical:**

  ```diff
  - from silly_kicks.tracking._id_compat import ids_match
  + from silly_kicks.id_compat import ids_match
  ```

  Behaviour is unchanged for `ids_match`. Two other changes ride along in the same release and are
  worth knowing about: `canonical_id_series` no longer raises on infinities or on floats outside
  int64 range (it now matches the scalar `canonical_id`, which always handled them), and the
  action↔frame masks in `_resolve_action_frame_context` now content-probe object id columns, so a
  **boxed-numeric** object id column (e.g. one carrying `2.0` rather than `"2"`) resolves instead of
  silently matching nothing.

- `tracking/_das.py` → `get_das`, recorded in the lakehouse
  `docs/superpowers/specs/2026-05-14-tracking-context-oom-bekkers-fix-design.md:225`. The consumer
  reached for the private function because it needed `chunk_size`, which `add_das` did not expose
  at the time, and stated its own exit condition: *switch back once `add_das` exposes `chunk_size`*.
  **That condition is met.** Verified 2026-07-18 on `origin/main` (`ec543cc`, 4.51.0):
  `silly_kicks/tracking/features.py:2478` defines
  `add_das(actions, frames, *, links=None, chunk_size=None, attacking_direction_col=None)`.
  The public seam is sufficient, so the coupling is recorded as retired rather than live. If the
  lakehouse has not yet switched over, that is a consumer-side migration, not a silly-kicks pin.

**Not consumers** (checked, recorded so the question is not re-asked): `_calibration_metrics.py`
and `_group_metrics.py` have **no** downstream consumer — the lakehouse computes its statistical
gates lakehouse-side (`src/analytics/xg_calibration.py`) and consumes model-validation results as
verdicts, not as computations.
