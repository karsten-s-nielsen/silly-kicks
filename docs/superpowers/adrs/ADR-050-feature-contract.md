# ADR-050: Trained-model feature contract + the canonical penalty-area constant

| Field | Value |
|---|---|
| **Date** | 2026-07-27 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |
| **Supersedes / amends** | ADR-021 (the tracked "canonical `spadlconfig` penalty-area constant" follow-up); ADR-038 (the deferred live corpus fingerprint); extends ADR-040's fail-closed `load()` guards |

## Context

Two penalty-area half-widths exist in this repo and always have: `_xcross_attempt.py` and
`defensive_credit/_params.py` use **20.16** (FIFA's 40.32 m), `_ghost_gk.py` uses **20.15** (40.3 m).
Neither cites the other. ADR-047 flagged the discrepancy in a comment and recorded "a canonical
`spadlconfig` penalty-area constant" as a follow-up.

The obvious fix — unify on the Law value — is **not behaviour-free**, and that is the finding this
cycle turns on. `_ghost_gk.py:608` uses the constant to compute `attackers_in_box`, which is one of the
26 `GHOST_GK_FEATURE_NAMES`: a real input to a bundled trained model. Flipping the constant changes a
feature the shipped weights were fit on, and nothing in the codebase would have noticed.

Measured on a real WC2022 match: **70 of 175,969 frames (0.0398%)** can flip `attackers_in_box`. Small,
and entirely beside the point — the number is not the risk. The risk is the *mechanism*: a geometry
constant edited far from any model silently re-defines that model's inputs, and the existing guards
cannot see it. `chirality` (ADR-040) fingerprints model OUTPUT on a fixed frame, which a feature
carrying little weight can shift without moving; `geometry_version` covers the coordinate transform,
not the constants; `pitch_length`/`pitch_width` cover pitch scale, and ghost recorded neither.

So the choice was never "which number". It was whether to change one centimetre and leave the
mechanism intact, or build the guard that makes the class of change safe — for this constant and for
the next one.

## Decision

Ship the guard first, then the constant, sequenced so the flip is a checkable event.

### 1. The feature contract (`silly_kicks/tracking/_feature_contract.py`)

A sibling of `_chirality.py` — probe, fingerprint, verify-at-load — recording **two** things per
trained-model artifact: the feature VECTOR its extractor produces on a fixed synthetic probe frame,
and the geometry CONSTANTS that extractor consumes.

Three policy differences from chirality, each deliberate:

- **A missing contract WARNS, does not raise.** A pre-contract artifact is undeclared, not known-bad.
- **A probe change WARNS and skips the fingerprint comparison only.** Declaring a new constant
  *requires* extending the probe (see §2), which changes the probe hash for every previously-saved
  artifact; those must keep loading.
- **A fingerprint or declared-constant mismatch RAISES**, with the model's own `IntegrityError`.

Constants are compared **first and always** — including when the probe changed. The two nets must not
cancel: a probe change is no reason to stop comparing 20.16 against 20.15. A sub-probe-resolution
change (20.16 → 20.161) moves no feature at all, so the declared constant is the *only* thing that can
catch it.

`equal_nan=True` on the fingerprint comparison is not cosmetic: `np.allclose(v, v)` is **False** for any
vector containing a NaN, so without it a round trip on an unmodified artifact fails. The builder also
refuses a non-finite vector at save time — a NaN feature is one the contract could never gate, and
allowing it would ship a fingerprint with silent holes.

**Tolerance is chosen, not inherited.** Chirality's `rtol=1e-2` was sized for a gross sign flip on a
probability; a feature vector spans metres, counts and radians, where `rtol=1e-2` on a ~17 m feature is
a 0.17 m blind spot — 17× the change this exists to catch. Ours is `atol=1e-6`, `rtol=0`, with every
fingerprinted artifact produced on x86 so no cross-platform comparison happens against a
not-yet-measured tolerance.

### 2. The probe must make every declared constant load-bearing

A declared constant the probe cannot move is a guard that fires when nothing changed — which is how
`legacy_override` becomes reflex. `contract_probe_frame()` is built so that `attackers_in_box` is
**0 at half-width 20.15 and 1 at 20.16** (verified through the real extractor, not a re-implementation
of its predicate).

Every element of the probe is load-bearing and measured: five attackers and five defenders (with four
and three, xS returned **7 NaN features**), a ball row carrying `z` (without it xS's `z` is NaN), ≥3
non-collinear defenders (ghost's ConvexHull compactness), and one attacker at (90.0, 13.845) — inside
the 16.5 m depth *and* inside `[13.84, …]` but outside `[13.85, …]`. Being in the y-band alone is not
enough: of 844 band rows on a real match, only 70 were also within depth.

### 3. Completeness by ENUMERATION, not by remembering

`tests/tracking/test_geometry_constant_enumeration.py` walks the four extractor modules with AST,
finds every module-level geometry-named constant (**14**), and requires each to be either declared in
`DECLARED_CONSTANT_SOURCES` or listed in `_EXEMPT` **with a reason**. This is the ADR-043 idiom that
replaced the id-compat lint: complete by enumeration where a heuristic was complete by hope.

The gate reads the **built** contracts, not just the registry — it saves all three models and compares
both directions (every registry key is stamped by some model; no model stamps an unknown key). That
check is what caught the defect below.

The derived-constant rule is written down, because the two cases look identical: derived from a
declared constant → map it to that constant's key (`GOAL_Y_MIN/MAX` move iff `goal_width` moves);
derived from pitch dimensions → exempt (`GOAL_Y_CENTRE` is just `PITCH_WIDTH/2`, already covered by the
pitch-dims guard).

**Per-model declarations, pinned in a test:**

| model | declares | why |
|---|---|---|
| xS | `goal_width` | its ONLY geometry constant; drives `openGoal` |
| xCross | `penalty_area_half_width`, `penalty_area_depth`, `goal_width` | box ratio + post distances |
| ghost | `penalty_area_half_width`, `penalty_area_depth` | `attackers_in_box`; values are the frozen 40.3 pair |

**A defect this found.** An earlier draft had xS declaring `penalty_area_half_width`. Verified:
`_xshot_occurrence.py` contains no penalty-area constant or predicate at all. That declaration would
have made the canonical flip raise on every xS load with xS's features provably unchanged. The
enumeration gate surfaced it by forcing the question "what does each model actually consume?".

### 4. Two warning categories, not one

`MissingFeatureContractWarning` (no contract at all — the escalatable case) and
`UnverifiableFeatureContractWarning` (probe changed, constant no longer declared, or a mismatch waved
through by `legacy_override`).

Separate **because** escalating the first must not escalate the second. Extending the probe is
mandatory whenever a constant is declared; if one umbrella category covered both, a consumer escalating
the missing-contract case would silently turn every probe extension into a hard failure across every
artifact not yet re-saved — the exact outcome §1's warn-and-skip was designed to avoid. Neither may
subclass the other, and a test enforces it, because subclassing is precisely how someone would "tidy"
these later. This follows ADR-041's reasoning for keeping warning categories separate.

### 5. All three bundled artifacts are stamped — not just ghost

The design originally stamped only ghost (it has the constant divergence to pin) and accepted xS/xCross
shipping without a contract as a known gap, to be *counted* via CI opt-outs.

Both positions rested on an assumption that proved false. All three bundled artifacts have identical
structure — `metadata.json` + `SHA256SUMS` + weights — and xS/xCross metadata already carries
`chirality`, `geometry_version`, `pitch_length`, `pitch_width`, exactly where `feature_contract`
belongs. The metadata-only migration works verbatim for all three, via a committed, re-runnable
`scripts/stamp_feature_contracts.py`.

Closing the gap costs **less** than counting it (one script over three directories, versus opt-outs in
~14 test files), leaves no ledger to maintain, and lets the CI escalation ship with an **empty**
opt-out list — fail-closed from day one.

**What the stamp attests, stated precisely.** It records what the *current* library's extractor
produces; the guarantee is **forward-looking**. It does NOT retroactively prove these are the features
each model was trained on. That limit is not new — it is what the load→save migration path
(`_ghost_gk.py:1859-1860`, used at the 4.54.0 parameters-only migration) already implied. Supporting
evidence is the same for all three: fail-closed chirality verification passes at load, and chirality
flows through the extractor. Evidence of stability, not proof for a near-zero-weight feature.

**The migration must not call `save()`.** Ghost's `save()` unconditionally rewrites the npz via
`np.savez_compressed`, whose ZIP members carry mtimes — so the bytes differ even when every array is
bit-identical; xS/xCross re-serialize `model.json` through xgboost. Verified after the run: **654 ghost
arrays bit-identical, both boosters identical, all three metadata deltas additive-only** (no key
removed, none changed).

**Residual, irreducible this cycle:** the HF-hosted `full` (ghost) and `sc_extended` (xS/xCross)
variants cannot be re-uploaded under the standing owner hold, so they still take the warn path. That is
correct — they genuinely carry no contract — and the existing trigger covers them.

### 6. The canonical constant, and what did NOT move

`spadlconfig.penalty_area_half_width = 20.16` / `penalty_area_depth = 16.5`, plus two **frame-explicit**
predicates in `_geometry.py`:

```
in_penalty_area_absolute(x, y, *, attacked_goal_x)   # action-LTR / absolute frame
in_penalty_area_goal_relative(gr_x, y)               # already goal-relative; caller owns WHICH goal
```

Two entry points, not one `goal_x`-taking helper: the three call sites differ on **frame**, not just on
strictness, and `goal_x` already means "the *defended* goal" elsewhere in `_geometry`. Same name, two
readings, and the failure mode is a box at the wrong end of the pitch — an 88.5 m error, not an epsilon.
The goal-relative form takes no goal argument at all, so the ambiguity cannot re-enter.

Migrated: `defensive_credit/_params.py` (scalar → calls the helper; its two box constants **deleted**,
not aliased, since nothing else read them and a dead alias would satisfy the enumeration gate
vacuously) and `_xcross_attempt.py` (its predicate is **vectorized** over numpy arrays, so it keeps its
expression and rebinds the constants as module aliases — the single source here is the CONSTANT, not
the predicate). The rule in both cases: *a module-level constant exists iff something in that module
reads it.* Byte-identity proven by a grid sweep over both sites; **159 tests** across the xcross and
defensive-credit suites pass unchanged.

**`_ghost_gk` keeps 40.3.** Its weights were fit on it. The contract now records that, so flipping it
without a re-fit makes `load()` **raise** — the "do not unify before the re-fit" instruction became a
mechanism instead of a comment a future contributor can delete. Note also that ghost's depth test is
**strict** `<` where the canonical helper is `<=`; it could not have been migrated even without the
weights issue.

`in_penalty_area_absolute` has an upper bound the old `x >= 105 - 16.5` form lacks (they disagree for
x > 121.5). Documented and pinned, **not** claimed unreachable: the nearest cap is `_SPADL_X_MAX = 120.0`
inside `derive_goalkeepers`, which validates *tracking* coords, while this helper's only caller works on
*action* coords — a different path.

### 7. Trainer-side guards the same failure mode implies

- **Ghost's feature cache is keyed on the geometry constants** (`cache_token()`), not a hand-bumped
  literal. A literal goes stale inside the very re-fit cycle it protects: extract, flip the constant,
  re-run — and the second run reuses the first run's 40.3 features while stamping a 20.16 contract.
- **The xS/xCross cache fingerprint is now LIVE per-corpus** (closing ADR-038's deferral). The constant
  `"schema-v2"` token could invalidate a pre-schema cache but was blind to corpus drift, which is why
  "use a fresh `--output-dir` per corpus" had to be a discipline. It is keyed on the **requested**
  corpus (not the extracted one — `load_matches` may drop a match at runtime, which would otherwise
  cause a permanent miss) via a `select_match_ids` / `_wanted_for_provider` helper **shared** with
  `load_matches`, so the fingerprint cannot describe a corpus the extraction never loaded.
- **Provider classification fails at startup**, via a shared `_ghost_gk.validate_provider` that both the
  trainer's pre-flight and `keeper_detection_mask` call. Same rule, same source; only the moment it
  fires changes, from after the full extraction to immediately.

### 8. CI escalates the missing-contract category

`error::silly_kicks.tracking.MissingFeatureContractWarning` in `pyproject.toml`, adopting the mechanism
ADR-041 established for `SyntheticEPVWarning` — with **no** opt-outs, because §5 leaves nothing to opt
out of. An opt-out appearing there later means an artifact shipped un-stamped, which is the thing to
fix rather than annotate.

## Consequences

- **No retrain, no value change.** No weights change, no model output changes, no feature value changes
  (the constant migration is byte-identical; ghost keeps its constant). Three `metadata.json` files gain
  a `feature_contract` key and their `SHA256SUMS` change — a consumer pinning artifact checksums sees a
  diff. Not a Hyrum break in behaviour, but worth the changelog line.
- **New public surface:** `spadlconfig.penalty_area_half_width` / `penalty_area_depth`;
  `tracking.MissingFeatureContractWarning` / `UnverifiableFeatureContractWarning`.
- **The next constant change is a checkable event**, for any constant any model declares — which is the
  whole point, and the reason this ADR is about a mechanism rather than about 1 cm.
- **C4-free** (no new action-coupled aggregator, model or backend; count stays 32).
- **One follow-on, and it should be NEXT, not someday.** The ghost re-fit — flip the constant, migrate
  `_ghost_gk` onto `in_penalty_area_goal_relative` (including its strict-`<` boundary), re-fit, re-stamp
  — is what actually completes the unification. It is a genuine retrain (new weights, moved outputs), so
  it is a separate PR from this one, and it is blocked **only on compute**: the 179-match owner corpus
  needs the DGX and the pining token.

  It should be sequenced **ahead of any downstream recompute**, so the new weights ride that pass instead
  of forcing a second one. An earlier draft of this ADR had the reasoning exactly inverted — it argued for
  *waiting for* the recompute window, which is what would cause the double drain. Recorded because the
  inverted version is the intuitive one and will be re-derived by the next reader otherwise.

  The **DGX-vs-x86 `atol` measurement is a gate inside that re-fit**, not a parallel item: it costs one
  probe-extractor call per platform, it needs exactly the DGX session the re-fit needs, and it must precede
  the re-stamp so a too-tight tolerance is not baked into a shipped artifact. Its answer may not be "widen
  the number" — if the cross-platform delta is large enough that a covering tolerance would also swallow a
  real 1 cm geometry change, the honest conclusion is that fingerprints are **platform-scoped**: verify on
  the stamping platform and skip only the fingerprint prong elsewhere, since the constants prong is
  platform-independent by construction.

## Alternatives considered

- **Just change the number.** Rejected: it silently re-defines a trained model's input, which is the
  actual defect. Fixes one centimetre, leaves the mechanism intact.
- **Permanent divergence, documented.** Rejected: relies on a prose contract, which this repo has
  watched fail before. The re-save turns the pin into a raise.
- **Count the xS/xCross gap with CI opt-outs** (the review's proposal). Rejected in favour of closing
  it: cheaper, no ledger, fail-closed immediately.
- **One warning category.** Rejected: escalating it would brick every artifact on any probe extension.
- **Unify the predicate as well as the constant at every site.** Not possible at `_xcross_attempt.py:209`
  (vectorized) without a per-element loop, and not safe at `_ghost_gk.py:608` (strict `<`, trained
  weights). Recorded rather than half-done.
