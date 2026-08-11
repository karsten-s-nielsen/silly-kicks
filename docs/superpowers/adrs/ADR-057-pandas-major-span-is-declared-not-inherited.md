# ADR-057: The pandas-major span is DECLARED, not inherited

| Field | Value |
|---|---|
| **Date** | 2026-08-09 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

> **Numbering note — this ADR was RENUMBERED on landing.** It was drafted as ADR-056 while a
> parallel session ran, on the recorded basis that `origin/cycleb-artifact-contracts` added no
> ADR. That basis expired: the branch merged as **4.78.0 carrying ADR-056**. The rule the draft
> stated — *whoever lands second renumbers* — is what was applied, and this cycle landed second.
> Recorded rather than quietly rewritten, because a provisional number that turned out to be
> taken is exactly the case the rule exists for.

## Context

`pyproject.toml` pins `pandas>=2.1.1,!=3.0.4` — deliberately permitting pandas 3, with the exclusion
carrying its own comment about a bisected 3.0.4 SIGSEGV. There is **no upper bound**, so pip resolves
the newest compatible pandas per interpreter, and pandas 3 requires Python >= 3.11 (verified on the
index: 3.0.5 declares `requires_python >=3.11`, 2.3.3 declares `>=3.9`).

The CI matrix is OS x Python with no pandas axis. Measured on run `31316804815`:

| CI leg | pandas |
|---|---|
| `ubuntu-latest, 3.10` | **2.3.3** |
| `ubuntu-latest, 3.11` | 3.0.5 |
| `ubuntu-latest, 3.12` (primary) | 3.0.5 |
| `windows-latest, 3.12` | 3.0.5 |

**Three of four legs already run pandas 3, and nothing anywhere said so.** `TODO.md` asserted the
opposite — that a pandas-3 environment "CI does not have" was the blocker for an open question —
and that false premise had stood for two cycles, deferring a measurement that was already free.

The forcing function is that this coverage is **real but accidental**. It is a side effect of the
Python matrix crossed with pandas' own `requires_python`, so it can vanish with no diff and no
signal. This repo has one measured instance of the class it protects against: DAS went silently
all-NaN on pandas 3.

## Decision

**The pandas-major span is a declared property of CI, asserted from two sides:** a structural guard
over the resolved leg set (`tests/test_ci_pandas_span_wired.py`) and a `pandas-span` aggregation job
that unions what each leg actually installed. Neither is sufficient alone, and the span is never
inferred from `pyproject.toml`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Add an explicit `pandas` axis to the matrix | Every combination chosen, not inferred | Multiplies legs and wall-clock; the repo already gates slow tests to one leg to bound runtime (ADR-023) | Cost is real and recurring; the span is obtainable without it |
| B. Structural guard only (read `ci.yml`) | Cheap, fast, local | **Cannot see a span collapse caused by a dependency constraint.** Adding `pandas<3` to `pyproject.toml` leaves `ci.yml` untouched and puts every leg on pandas 2 | Misses a hazard this repo has already demonstrated the practice for (`!=3.0.4`) |
| C. Aggregation job only (observe installs) | Observes ground truth | Only fails AFTER a full matrix run; says nothing at edit time, and a matrix edit is the likelier hazard | Slow feedback for the common case |
| D. **Both (chosen)** | Edit-time signal for matrix changes; run-time truth for dependency changes | 28 lines of `ci.yml` and one extra job (~20s) | — |

Option B was the original design and was rejected only after the `pyproject` hazard was identified;
that hazard is why the pair exists rather than the single guard.

## Consequences

### Positive

- The differential coverage is **declared**, so losing it is a failing build rather than a silent
  change in what CI verifies.
- The structural half fails at edit time with a message naming the property and its assumption.
- The aggregation half catches the dependency-constraint case no `ci.yml` reader can see, and
  refuses to pass on zero artifacts — so a broken recording step cannot masquerade as success.
- It made an existing false claim visible: the `TODO.md` row asserting CI had no pandas-3
  environment was wrong, and the question it was blocking (`snapshot_to_tracking_frames` id dtypes)
  was answerable immediately. Measured: nullable `Int64` is preserved on pandas 2 and promoted to
  `Float64` on pandas 3, while the ADR-019 `id_compat` property holds on both. Answering it then
  exposed the schema defect *underneath* it — `TRACKING_FRAMES_COLUMNS` declaring a non-nullable
  `int64` for columns that are NA on the ball row by construction — which is **ADR-058**. Two
  cycles of "unimplementable" traced back to a false premise recorded here.

### Negative

- 28 lines added to a 100-line `ci.yml`, plus one job that can fail for its own reasons.
- The structural half rests on an ASSUMPTION — that pandas 3 requires Python >= 3.11 — encoded as
  `_PANDAS3_MIN_PY`. If pandas changes its floor, that constant must move. The guard's failure
  message says so explicitly, because the tempting wrong fix is to redefine the boundary to match a
  matrix that lost its leg.
- Artifact upload/download is a new failure surface in a workflow that previously had none.

### Neutral

- No library code changes; no runtime dependency changes; no retrain trigger.
- The span asserted is "at least two majors", not specific versions. A routine pandas bump does not
  touch it, which is the point — a guard that fails on every dependency update trains its reader to
  edit the expectation without thinking.

## Verification

- Structural half observed RED against the HAZARD, not the implementation: the pandas-2 leg pruned
  via `exclude` while `"3.10"` remains in the `python-version` axis — the case an axis-based
  assertion passes. The axis-deletion mutation was also run, to show the two differ.
- Aggregation script executed against fabricated artifact trees, the body EXTRACTED from `ci.yml`
  rather than retyped: `{2,3}` passes; `{3}` fails; `{2}` fails; zero artifacts fails.
- Both wiring assertions observed RED (`needs: test` removed; artifact name made non-per-leg, which
  would collide every leg onto one artifact and read as a lost span).
- `ci.yml` restored byte-identical after every mutation run.
