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
