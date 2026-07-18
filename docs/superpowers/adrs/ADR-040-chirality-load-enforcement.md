# ADR-040: chirality `load()` enforcement + xgboost `base_score` compatibility guard

| Field | Value |
|---|---|
| **Date** | 2026-07-17 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; TF-19 PR-2 (silly-kicks session) |
| **Supersedes / amends** | Completes ADR-037 §9 (chirality fingerprint: PR-1 emission, this ADR's PR-2 enforcement); new decision (not previously recorded anywhere): the `base_score` cross-version compatibility guard |
| **Source plan** | `docs/superpowers/plans/2026-07-17-tf19-pr2-weights-bundle.md` |

## Context

ADR-037 §9 designed a behavioral chirality fingerprint (`silly_kicks/tracking/_chirality.py`:
`chirality_fingerprint` + `canonical_probe_frame`) to catch the class of bug that motivated
the whole TF-19 re-gate cycle: xS, xCross, and ghost-GK were trained on y-mirrored frames
(pre-ADR-031) and served on y-correct frames from 4.29.0/4.30.0 onward, silently negating
3/16 xCross features and 12/27 xS features for every y-correct provider. ADR-037 §9 shipped
in two parts by design: **PR-1** (4.47.0) emits the fingerprint into every `save()` path's
`metadata.json`; **PR-2** (this ADR) hardens `load()` to actually check it. Until this PR,
every bundled artifact — including the corrected DGX Stage A/B retrains — could be loaded
without any verification that the served outputs matched what was fingerprinted at training
time. A fingerprint nobody checks is documentation, not a guard.

A second, unplanned defect surfaced while validating this enforcement. The DGX training
environment runs xgboost 3.2.0; the library's runtime dependency floor is `xgboost>=2.0`,
and the developer/serve environment used xgboost 2.1.4. xgboost 3.x serializes
`learner.learner_model_param.base_score` in `model.json` as a bracketed **string** (e.g.
`"[2.19E-1]"`); xgboost 2.x's `Booster.load_model` does not parse that form and silently
falls back to the `0.5` default — a mis-served intercept, not a load failure, so nothing
raises. On the DGX-trained xS artifact this shifted a probe prediction from the true
**0.0325** to **0.107** on the chirality probe frame — over 3x off, entirely from a
base_score serialization mismatch with zero relation to the mirror bug the fingerprint was
built to catch. The chirality enforcement is what surfaced it: Task 9's golden test
(`test_weights_bundle_golden.py`, `from_variant("default")` running the strict, non-override
`load()` path) failed with an output mismatch on the DGX-retrained default weights before
any fix was applied — the fingerprint recorded at training time (correct base_score) did
not reproduce at load time (base_score silently reset to 0.5), and the guard refused to load
the artifact rather than silently serving wrong numbers.

## Decision

`load()` on all three trained-model classes (`XShotOccurrenceModel`, `XCrossAttemptModel`,
`GhostGkModel`) now fails closed on chirality: it re-runs the model's own
`_chirality_block` on the canonical y-asymmetric probe frame and compares against the
`chirality` block stored in `metadata.json`, raising `IntegrityError` on a **missing**
fingerprint (every pre-PR-2 artifact) or a **mismatch** beyond a cross-platform float
tolerance, with an explicit `legacy_override: bool = False` escape hatch that warns instead
of raising. Separately, `load_xgb_booster_base_score_safe` normalizes a bracketed
`base_score` string at load time so the library keeps working across the xgboost 2.x/3.x
serialization boundary without narrowing its dependency floor.

## Part 1 — fail-closed chirality enforcement

### Shared helper

`silly_kicks/tracking/_chirality.py` gains `verify_chirality(recomputed, stored, *,
legacy_override, model_name, error_cls=None)`:

The `error_cls` parameter is the exception each caller's `load()` raises for artifact-integrity
failures, so the chirality error shares that `load()`'s taxonomy — a consumer catching the model's
own `IntegrityError` catches the chirality failure too. It defaults to
`_xshot_occurrence.IntegrityError` (the type xS *and* xCross use throughout, since xCross imports
it); `_ghost_gk` passes its own module-local `IntegrityError` so that ghost's `load()` raises a
single integrity type for SHA-256, pitch-dimension, *and* chirality failures alike.

- `stored is None` (no `chirality` block in `metadata.json`) -> raises `IntegrityError`
  unless `legacy_override=True`, in which case it `warnings.warn`s
  (`stacklevel=2`) and returns. The message is explicit that every pre-TF-19-PR-2 artifact
  is exactly the mis-served class of bug, so overriding is only safe for an artifact the
  caller has independently verified.
- `recomputed["frame_sha256"] != stored["frame_sha256"]` -> raises. The probe frame itself
  is versioned; a frame change means the two fingerprints are not comparable, which is a
  version-skew condition distinct from an actual chirality mismatch.
- Output arrays compared with `np.allclose(atol=1e-3, rtol=1e-2)` -> raises
  `IntegrityError` on a mismatch beyond tolerance.

`chirality_fingerprint` also gained a finiteness guard (`np.isfinite`) — a NaN output would
otherwise serialize as non-standard JSON `NaN` and the `==`-adjacent comparison logic could
behave unpredictably; this fails at fingerprint-computation time instead, before it ever
reaches `metadata.json`.

### Tolerance rationale

`atol=1e-3`, `rtol=1e-2`. The fingerprint is computed once at training time on the DGX
(aarch64) and re-verified at every `load()` on the serving platform (x86-64 in CI and in
production). xgboost/numpy floating-point evaluation is not bit-identical across that
platform boundary, but the noise floor is small — empirically around 1e-6 on the probe
frame's outputs. A genuine y-mirror on the deliberately y-asymmetric probe frame
(`canonical_probe_frame`, every row off the `y=34` axis, goal at `x=105`) moves outputs by
O(0.01-1) — two to three orders of magnitude above the cross-platform noise floor. The
chosen tolerance sits comfortably above the noise floor and comfortably below the mirror
signature, so it discriminates the two without needing per-platform calibration.

### `legacy_override`

Every model class's `load()` gained `legacy_override: bool = False`. This is not a
convenience flag — no code path in this PR sets it `True` by default, and `from_variant` /
`from_hub` call `load()` at the default (strict). It exists so a caller holding a
verified-safe pre-PR-2 artifact (for example, one independently confirmed never to have been
served on mirrored frames) is not permanently locked out, without weakening the default
posture for the artifacts the fingerprint exists to catch.

### Validation: the cross-platform golden test

`tests/tracking/test_weights_bundle_golden.py` parametrizes over all three model classes and
calls `from_variant("default")` — the strict, non-override path — asserting it loads
without raising. This is the single test that validates the entire cross-platform tolerance
design in one shot: a pass means the DGX-computed fingerprint reproduces on the CI/serving
platform within `atol=1e-3`/`rtol=1e-2`. Its docstring is explicit that a future failure here
must not be resolved by loosening the tolerance — it means a genuine train/serve
inconsistency has been (re-)introduced, and the correct response is to find and fix it, the
same way the base_score defect below was found and fixed rather than tolerated.

`tests/tracking/test_chirality_enforcement.py` covers the unit-level contract for all three
model classes: raises on a tampered/gross output mismatch, raises on a missing fingerprint,
`legacy_override=True` loads a fingerprint-less artifact with a warning, and a probe-frame
SHA change raises. It also covers the finiteness guard and the `sc_extended` ->
`from_hub` routing added in the same PR.

## Part 2 — `base_score` cross-version compatibility guard

`load_xgb_booster_base_score_safe(model_json_path)` (`silly_kicks/tracking/_xshot_occurrence.py`,
imported by `_xcross_attempt.py`) reads `model.json` as UTF-8 JSON (xgboost writes UTF-8; the
platform default would be cp1252 on Windows and raise `UnicodeDecodeError` on a non-ASCII feature
name) before handing it to `xgboost.Booster.load_model`. If
`learner.learner_model_param.base_score` is a string of the form `"[...]"`, it strips the
brackets and loads the corrected in-memory JSON; any other form (a bare scalar, 2.x-native
or already-fixed) passes through unchanged with no behavioral difference. This is
defensive, not merely a fix for the currently-bundled artifacts: the library's dependency
floor is `xgboost>=2.0`, spanning both the 2.x and 3.x serialization conventions, so
`load()` must tolerate whichever convention produced the artifact on disk.

Both the bundled default weights and the staged Hub-bound `sc_extended` weights were also
**re-saved to clean 2.x scalar `base_score` format** as part of this PR, so the guard is a
belt-and-suspenders defense on top of a corrected artifact, not the only line of defense.

**Recommended operational discipline** (recorded, not code-enforced): a DGX or other
training environment that produces artifacts for this library should pin `xgboost<3.0` to
match the serving environment's major version, eliminating the serialization mismatch at
the source. Where that is not practical — for example, a training environment that must
track a newer xgboost for other reasons — `load_xgb_booster_base_score_safe` is the
fallback that keeps `load()` correct regardless. Both are legitimate; the guard exists
specifically so the second case does not silently corrupt predictions the way it did before
this PR.

### Discovery mechanism — the design validating itself

This defect was not found by manual inspection or a targeted xgboost-version test. It was
found because Part 1's enforcement existed: the DGX training run stamped a chirality
fingerprint using the correctly-computed `base_score`; loading that same artifact in the
serving environment recomputed the fingerprint with `base_score` silently defaulted to
`0.5`, and the two disagreed by far more than the cross-platform tolerance. The golden test
failed loud, with the actual predicted values in the assertion message, rather than shipping
a wheel with a silently 3x-off xS model. This is the intended failure mode of a
behavioral, not self-declared, guard (ADR-037 §9): it does not need to know in advance what
kind of train/serve divergence to look for.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Warn instead of raise on a missing/mismatched fingerprint | Never breaks an existing caller's `load()` call | Defeats the purpose — the whole cycle exists because a silent mis-serve went undetected for multiple releases; a warning is exactly what would have been ignored | Fail-closed is the point; ADR-037 §9 already specified "fail-closed on a mismatched fingerprint AND on a missing one" |
| B. Pin `xgboost<3.0` in `pyproject.toml` instead of adding a load-time guard | Simpler; no new code path | Breaks any consumer who has a legitimate reason to run xgboost 3.x elsewhere in their stack; does not protect an artifact trained by a future/rogue environment that still uses 3.x | The guard is strictly more defensive and does not narrow the existing `>=2.0` floor; pinning is recommended as *training-environment* discipline, not enforced as a library constraint |
| C. Loosen `_CHIRALITY_ATOL`/`_CHIRALITY_RTOL` when the golden test first failed | Fastest path to green | Would have papered over the base_score defect instead of finding it — exactly the anti-pattern the plan's "KEY RISK" note warns against | The failure was investigated to its root cause instead; the tolerance was validated, not adjusted |
| D. Fail-closed chirality + defensive base_score guard, tolerance derived from measured cross-platform noise (chosen) | Catches the mirror class of bug by construction; caught an unrelated real defect on first real-data run; supports the existing `xgboost>=2.0` floor | Two new failure modes callers must understand (`IntegrityError` on missing/mismatched fingerprint); `legacy_override` is an escape hatch that must be used carefully | — |

## Consequences

### Positive

- A mis-served (mirrored, or otherwise silently wrong) trained artifact can no longer be
  loaded without an explicit, logged decision (`legacy_override=True` + a warning) — closing
  the exact gap that let the original mis-serving bug ship undetected across multiple
  releases.
- The base_score guard keeps the library correct across the xgboost 2.x/3.x serialization
  boundary without narrowing the `xgboost>=2.0` dependency floor.
- Demonstrated real value on first use: the enforcement caught a genuine, unrelated
  serialization defect before it shipped, on the very first cross-platform golden-test run
  against the corrected retrains.

### Negative

- Every pre-PR-2 weights artifact anywhere (a user's own fine-tuned model, an old cached
  Hub download, a `save()`d artifact from before this release) now fails to load without
  `legacy_override=True`. This is `!`-worthy (breaking) for any caller holding such an
  artifact; the CHANGELOG entry for 4.51.0 must and does flag it.
- `legacy_override` is an escape hatch that, if used carelessly (e.g. blanket-set `True` in
  a wrapper to silence the warning), reintroduces exactly the risk this ADR closes. It is
  intentionally not a config default anywhere in the library.
- Two additional failure modes now exist at `load()` time (`IntegrityError` from a missing
  fingerprint, `IntegrityError` from a mismatched one) that callers integrating this library
  need to be aware can be raised, in addition to the pre-existing SHA256SUMS integrity check.

### Neutral

- No new C4 element, aggregator, or backend — this is a hardening of an existing `load()`
  contract on three existing model classes, not a new capability surface.
- The `base_score` guard is transparent to a caller: it changes nothing about the public
  API, only what `load()` does internally before handing bytes to xgboost.

## CI gates

- `tests/tracking/test_chirality_enforcement.py` — unit-level: raises on mismatch, raises on
  missing, `legacy_override` loads with a warning, finiteness guard, probe-frame SHA change,
  `sc_extended` -> Hub routing, and the base_score bracket-normalization guard
  (`test_load_xgb_booster_base_score_safe_normalizes_bracketed`).
- `tests/tracking/test_weights_bundle_golden.py` — the cross-platform validation gate: the
  bundled `default` weights for all three model classes load and re-verify chirality on
  whatever platform CI runs on. This is the test that would have failed loud on the
  base_score defect had it shipped unfixed, and is the ongoing regression guard against any
  future train/serve divergence of this shape.

## Related

- **ADRs:** completes ADR-037 §9 (chirality fingerprint design; PR-1 emission / PR-2
  enforcement split); references ADR-031 (the kloppy tracking y-inversion fix that made the
  original mis-serve possible), ADR-011 (trained-model lifecycle).
- **Plans:** `docs/superpowers/plans/2026-07-17-tf19-pr2-weights-bundle.md`.
- **Research:** `docs/research/tf19_pr2/decision_table.md` (the re-gate verdict measured on
  these corrected, chirality-verified weights), `docs/research/tf19_pr2/hf_upload_instructions.md`
  (the HF-only `sc_extended`/`full` variants this PR's weights include).
- **External references:** xgboost `model.json` serialization format changes between the
  2.x and 3.x major series (`learner.learner_model_param.base_score`).
