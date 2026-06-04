# Tracking time-base contract + loud low-coverage guard for `link_actions_to_frames`

| Field | Value |
|---|---|
| **Date** | 2026-06-04 |
| **Status** | Reviewed — lakehouse comments incorporated 2026-06-04; ready for implementation plan |
| **Authors** | Karsten S. Nielsen, Claude Opus 4.8 (1M) |
| **Reviewers** | luxury-lakehouse AC-1 session (design Q1–Q3 answered; spec reviewed — 2 HIGH + 3 MEDIUM incorporated, see §8) |
| **ADR** | ADR-017 (to be authored, bundled into the feature commit) |
| **Origin** | luxury-lakehouse change request, 2026-06-04 (GradientSports period-2 action↔frame time-base mismatch → silent ~81% data loss) |

## 1. Problem

AC-1 enrichment silently dropped ~81% of **GradientSports period-2** actions in lakehouse production. Coverage (output rows / source SPADL actions) per (provider, match, period):

| provider | match | period | source actions | enriched | coverage |
|---|---|---|---|---|---|
| gradientsports | 10502 | 2 | 740 | 98 | 13% |
| gradientsports | 10503 | 2 | 762 | 145 | 19% |
| gradientsports | * | 1 | — | — | 100% |
| idsse | J03WMX | 1 & 2 | — | — | 100% |

**Root cause (confirmed):** two different time bases were compared. GS *frames* are timestamped **period-relative** (`period_elapsed_time`, resets to 0 each period); GS *actions* carry **absolute match-clock** time (`period_game_clock_time`, continuous across periods). The action↔frame match only succeeded where the two ranges accidentally overlapped — GS p2 output range `[2700, 3142]` is exactly `intersection(actions [2700, 5835], frames-elapsed [0, 3142.8])`. No error was raised; a manual source-vs-output audit weeks later found it.

The proximate column-choice fix is lakehouse-side. This spec is the **library-side, long-term** half: make this class of bug impossible to ship silently.

## 2. The decisive finding: silly-kicks' `time_seconds` convention is **period-relative**, not absolute

The change request proposed standardizing on *"absolute seconds from kickoff, continuous across periods."* Investigation shows that is **the opposite** of what silly-kicks actually produces, and adopting it would conflict with every existing events converter:

- **Opta** (`spadl/opta.py:166-173`) explicitly subtracts cumulative period durations to force period-relative time:
  ```python
  actions["time_seconds"] = (60 * events.minute + events.second
      - ((events.period_id > 1) * 45 * 60)    # −45 min in P2
      - ((events.period_id > 2) * 45 * 60)
      - ((events.period_id > 3) * 15 * 60)
      - ((events.period_id > 4) * 15 * 60))
  ```
- **StatsBomb** (`spadl/statsbomb.py:237`) — `pd.to_timedelta(events.timestamp)`; StatsBomb `timestamp` is elapsed-within-period, resets each half.
- **GS *frames*** already emit `period_elapsed_time` (period-relative) — **conforming.**
- **GS *actions*** as fed by the lakehouse use `period_game_clock_time` (absolute) — the **lone non-conformer**, injected upstream. The GS events converter (`spadl/gradientsports.py:416`) passes `time_seconds` straight through, so the library does not mandate absolute for GS — the absolute value comes from the lakehouse's bronze→input mapping.

**The linker is already per-period scoped.** `link_actions_to_frames` (`tracking/utils.py:238-239`) groups actions by `period_id` and `merge_asof`-matches only against same-period frames. Cross-period continuity is therefore irrelevant to silly-kicks. The only invariant that matters is:

> **Within each period, frames and actions must share the same time origin.**

GS p2 breaks it because actions are absolute (`[2700, 5835]`) while frames are relative (`[0, 3142]`). Either internally-consistent choice (both absolute, or both period-relative) fixes the per-period linker, but **period-relative** is the choice that matches silly-kicks' canonical convention and every other provider, leaving no latent cross-provider footgun. We document period-relative as canonical.

## 3. Goals / non-goals

**Goals**
- Document the canonical `time_seconds` convention (period-relative) on the converter + linker + schema surfaces so a consumer cannot silently hand-pick a mismatched column.
- Make a low-coverage link outcome **loud** (warn by default; opt-in raise), evaluated **per-period**, with a diagnostic message that names a suspected time-base mismatch.
- Provide a thin public `validate_time_base` affordance for cheap pre-link / CI assertions.

**Non-goals**
- Owning GS bronze-schema time normalization in the library (change request §3c). The GS `convert_to_frames` receives `time_seconds` already-extracted in `EXPECTED_INPUT_COLUMNS`; it never sees the raw bronze columns. Owning normalization would require ingesting multi-column bronze, crossing the I/O boundary the hexagonal architecture deliberately keeps caller-side. **Rejected.** The documented contract (§4.1) + linker guard (§4.3) give the same safety without dragging bronze-schema knowledge into the library.
- Changing the per-period matching algorithm. The linker stays per-period; we only add diagnostics + docs.
- Forcing the lakehouse's fix direction. The lakehouse will make GS actions+frames agree per period (their proximate fix); this spec is independent of which side they normalize.

## 4. Design

### 4.1 Documentation — the canonical convention (the real fix for *this* bug)

State explicitly, on the surfaces below:

> `time_seconds` is **seconds since the start of its period, resetting to 0 each period** — monotonic within a period, **NOT** absolute match-clock / continuous across periods. Frames and actions linked together must share this per-period origin.

Surfaces:
- Tracking `convert_to_frames` docstrings + `EXPECTED_INPUT_COLUMNS` (all native adapters: `gradientsports`, `sportec`; note kloppy gateway inherits kloppy's own period-relative timestamps).
- `link_actions_to_frames` docstring (Parameters + a "Time-base contract" note).
- `slice_around_event` docstring (same per-period assumption).
- Schema modules: `tracking/schema.py` (`TRACKING_FRAMES_COLUMNS`) and `spadl/schema.py` (`SPADL_COLUMNS`) — a comment on the `time_seconds` entry.

This is the single change that would have prevented *this* bug (the lakehouse pre-filter dropped the actions before they reached the linker, so the runtime guard in §4.3 would **not** have fired on this specific incident — documentation is the higher-leverage fix; the guard is the general-class backstop).

### 4.2 Pure detector core (one implementation, two surfaces — DRY + SoC)

`_diagnose_time_base(actions, frames) -> TimeBaseDiagnosis` — internal, in `tracking/utils.py`. Pure: no warn, no raise, no I/O. NaN-tolerant (drops NaN `time_seconds` before computing ranges). **Vectorized** — per-period ranges come from a single `groupby("period_id")["time_seconds"].agg(["min", "max"])` on each side, **not** the `iterrows` pattern in the neighboring `_count_candidates_within_tolerance` (which we deliberately do not copy).

`TimeBaseDiagnosis` (frozen dataclass, in `tracking/schema.py` next to `LinkReport`):

| field | type | meaning |
|---|---|---|
| `per_period_action_range` | `dict[int, tuple[float, float]]` | `(min, max)` action `time_seconds` per period |
| `per_period_frame_range` | `dict[int, tuple[float, float]]` | `(min, max)` frame `time_seconds` per period |
| `per_period_overlap_fraction` | `dict[int, float]` | fraction of the action span covered by the frame span, per period |
| `suspected_mismatch_periods` | `tuple[int, ...]` | periods where `overlap_fraction < MISMATCH_OVERLAP_FLOOR`, **ordered worst-first** (lowest overlap first) |
| `message` | `str` | human-readable summary **enumerating all suspected periods worst-first**, e.g. `"period 2: actions [2700, 5835] vs frames [0, 3142] — near-disjoint (overlap 0.14); period 1: ... ; suspected period-relative/absolute time-base mismatch"` |

Per-period overlap fraction:
```
overlap = max(0, min(a_max, f_max) - max(a_min, f_min))
overlap_fraction = overlap / (a_max - a_min)   # 1.0 = frames fully span the actions; 0.0 = disjoint
# degenerate a_max == a_min (single action): overlap_fraction = 1.0 if the point lies in [f_min, f_max] else 0.0
```
`MISMATCH_OVERLAP_FLOOR` (module constant) = **`0.2`**, **decoupled from `min_link_rate`** (lakehouse MEDIUM). The two thresholds answer different questions: `min_link_rate` (0.5) governs the *symptom* (is coverage low enough to warn?); `MISMATCH_OVERLAP_FLOOR` (0.2) governs the *cause hypothesis* (is the low coverage specifically a near-disjoint time-base mismatch, vs. ordinary sparsity?). 0.2 keeps the hypothesis **specific**: the GS case is overlap ≈ 0.14, so 0.2 catches it, while a period that is merely sparse (overlap high, link-rate moderate) is *not* mislabeled a time-base mismatch. A period can trip `min_link_rate` (warns) without tripping `MISMATCH_OVERLAP_FLOOR` (no mismatch hint in the message) — the message keeps symptom and cause distinct.

Periods present in actions but with zero frames → `frame_range` absent, `overlap_fraction = 0.0`, period flagged. Empty actions → empty diagnosis (all dicts empty, no suspected periods).

### 4.3 Surface A — linker guard (primary; automatic, lazy, per-period)

`link_actions_to_frames` signature gains two keyword-only parameters:

```python
def link_actions_to_frames(
    actions, frames,
    tolerance_seconds: float = 0.2,
    *,
    min_link_rate: float = 0.5,
    on_low_coverage: Literal["warn", "raise", "ignore"] = "warn",
) -> tuple[pd.DataFrame, LinkReport]:
```

- **Per-period evaluation (load-bearing).** The guard tests `min(report.per_period_link_rate.values()) < min_link_rate`, **never** the match aggregate. A match-aggregate test would launder GS p2 behind p1: GS 10503 whole-match = `(803 + 145) / (803 + 762) = 60.6%`, sailing past any 0.5 floor; per-period reads p2 = 19%, cratering decisively. `LinkReport` gains `per_period_link_rate: dict[int, float]` alongside the existing `per_provider_link_rate`.
- **`per_period_link_rate` is computed from `merged_all`, not from the returned `pointers` (lakehouse MEDIUM).** The returned `pointers` DataFrame projects to `action_id / frame_id / time_offset_seconds / n_candidate_frames / link_quality_score` (`utils.py:272-280`) — it has **dropped `period_id`**. Computing the per-period rate off `pointers` would have no period column to group on (or force a re-join that re-launders). The intermediate `merged_all` (`utils.py:259`) still carries `period_id` (each `a_group` was grouped on it), so per-period rate = `merged_all.groupby("period_id")["frame_id"].apply(lambda s: s.notna().mean())`. This is the one impl detail a naive implementation gets wrong.
- **Lazy diagnosis.** Only when the guard trips do we call `_diagnose_time_base` (range math is paid on degraded matches only, never on healthy ones).
- **Diagnostic-rich message.** The warning/error text carries `link_rate` (worst period), `n_actions_unlinked`, and — when `_diagnose_time_base` flags the worst period — the time-base hint string. Example:
  > `link_actions_to_frames: period 2 link_rate 0.19 (762 actions, 617 unlinked) below min_link_rate 0.5. period 2: actions [2700, 5835] vs frames [0, 3142] — near-disjoint (overlap 0.14); suspected period-relative/absolute time-base mismatch. See the time-base contract in the docstring.`
- **Policy dispatch.** `"warn"` → `warnings.warn(..., UserWarning, stacklevel=...)`; `"raise"` → `ValueError`; `"ignore"` → silent (report still populated). One `UserWarning` per offending period (dedup), `stacklevel` set so the warning points at the consumer's call site, not into the linker internals.
- **Rationale (documented in docstring + ADR):** warn-by-default because low coverage is a quality *continuum*, not a structurally-impossible input (unlike `require_et_direction`, which raises) — a legitimately-partial match (camera dropout, missing period) can hit it, so raise-by-default would break honest callers. Warn is non-fatal, visible (surfaces on stderr, capturable by pytest, escalates under `-W error`), and escalatable. `min_link_rate=0.5` placed where defects (which crater coverage to 13–19%) fire with 30+ points of headroom while staying quiet on legitimate sparsity (0.7–0.95 band: low-frame-rate broadcast tracking, substitution gaps, ball-out stretches, whistle-edge frames). Matches the existing GS-roster `≥50% unmatched` precedent (`tracking/gradientsports.py:373`) — same line for the same "structurally wrong" judgment (Chesterton).

### 4.4 Surface B — `validate_time_base` affordance (not the safety net)

```python
def validate_time_base(
    actions, frames, *,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> TimeBaseDiagnosis:
```

Public, in `tracking/utils.py`, re-exported from `silly_kicks.tracking`. A thin wrapper over the same pure `_diagnose_time_base`. Returns the diagnosis; on a suspected mismatch, raises `ValueError` (`on_mismatch="raise"`, the default for an explicitly-invoked pre-flight assertion) or warns / stays silent.

**Surface B is the *primary* guard for any consumer that pre-filters actions by time before linking — not a bonus (lakehouse HIGH).** The linker guard (§4.3) only sees the actions that reach it. A consumer whose pipeline pre-filters actions to the frame time-window before calling `link_actions_to_frames` (e.g. the lakehouse `enrich_batch` batching actions by `floor(frame_num/250)` and pre-windowing to `[frames.t.min()-buf, frames.t.max()+buf]`) drops the out-of-range p2 actions **upstream** — the linker then sees ~100%-linkable survivors and Surface A stays silent. **This is exactly how the original bug stayed invisible.** For such consumers, `validate_time_base` called on the **unfiltered** actions + frames at work-unit entry is the real safety net. Documentation must therefore frame Surface B as:

> If you filter, window, or batch actions by time before linking, call `validate_time_base(actions, frames)` on the **unfiltered** inputs first — the linker's `on_low_coverage` guard cannot see actions your pre-filter has already dropped.

The CI-contract use (synthetic period-relative-vs-absolute inputs in the lakehouse `test_silly_kicks_boundary.py`, proving the detector fires without a full link or tracking fixture) remains, but is secondary to the pre-filter-consumer guard role. (Adoption note for the lakehouse: wire `validate_time_base` at work-unit entry, not only `on_low_coverage="raise"` on the linker — the pre-filter means the linker alone would not have caught this.)

### 4.5 Consumer posture (informative, not part of the library)

The lakehouse will opt **up** to `on_low_coverage="raise"` with `min_link_rate≈0.9` for AC-1 (hard-fail-first UDF semantics; a silently half-empty period is worse than a work-unit that fails loud and quarantines). The library default protects the naive caller; strict consumers tighten by policy. No conflict with lakehouse ADR-002's ban on telemetry-swallowing `logger.warning` — Python `warnings.warn(UserWarning)` is a different, genuinely-visible mechanism.

## 5. Testing

Non-e2e, fixture-light (synthetic frames/actions). The tests that **must bite**:

- **Convention-pinning tests — make §4.1 executable (lakehouse HIGH).** Prose docs drift; the period-relative invariant is now load-bearing (the linker, `validate_time_base`, and every consumer depend on it), yet nothing currently asserts the real converters honor it — a future converter refactor could silently emit absolute time and every other test would still pass (the exact "undocumented convention" failure that caused the original bug). Add one convention-pinning test **per events converter that the library controls the time arithmetic for**: a multi-period synthetic fixture asserting `time_seconds` **resets toward 0 at the start of each period** (concretely: `min(time_seconds | period==2)` is small / near period start, and is **not** ≥ `max(time_seconds | period==1)` as an absolute clock would be). Cover the converters whose `time_seconds` the library computes — **Opta** (`opta.py:166` arithmetic), **StatsBomb** (`statsbomb.py:237` period-elapsed `timestamp`). For pass-through converters (**Sportec** `sportec.py:949`, **GS events** `gradientsports.py:416`, **kloppy** gateway) the library does not own the arithmetic — assert instead that the converter **passes the period-relative input through unchanged** (a period-relative synthetic in stays period-relative out), documenting that the convention is the caller's responsibility there. These tests are what convert §4.1 from prose into an enforced contract.
- **Laundering guard (the keystone).** A 2-period synthetic shaped like GS 10503 — aggregate `> 0.5` but p2 `< 0.5` (e.g. p1 100/100 linked, p2 19/100 linked → aggregate 59.5%). Assert the `"warn"` path **fires** and the message names period 2. This test fails if anyone "simplifies" the guard to match-aggregate.
- **Detector purity / correctness.** Disjoint per-period ranges → `suspected_mismatch_periods == (2,)`, `overlap_fraction[2] ≈ 0.14` (`< MISMATCH_OVERLAP_FLOOR` 0.2); fully-overlapping ranges → empty `suspected_mismatch_periods`. Multiple suspected periods → ordered **worst-first** (lowest overlap first). Single-action period (degenerate span) → in-range point gives fraction 1.0, out-of-range gives 0.0. NaN `time_seconds` rows dropped, not crashing. Vectorized (no `iterrows`).
- **Policy paths.** `on_low_coverage="warn"` emits exactly one `UserWarning` per offending period (dedup); `"raise"` raises `ValueError` with the diagnostic text; `"ignore"` emits nothing yet still populates `per_period_link_rate`. `stacklevel` points at the caller (assert via `warnings.catch_warnings(record=True)` frame check or pytest `filterwarnings`).
- **Sparsity ≠ mismatch (decoupled thresholds).** A period at link_rate 0.6 with high overlap (uniform sparsity, overlap ≥ 0.2) → if `min_link_rate=0.5`, no warning; if tightened to `min_link_rate=0.8`, the warning fires (symptom) but the message does **not** assert a time-base mismatch because `overlap ≥ MISMATCH_OVERLAP_FLOOR` (cause hypothesis not met). Proves `min_link_rate` and `MISMATCH_OVERLAP_FLOOR` are independent.
- **`validate_time_base`.** Synthetic period-relative-vs-absolute inputs → `on_mismatch="raise"` raises; `"warn"` warns + returns; `"ignore"` returns the diagnosis silently.
- **Backcompat.** Existing callers (`link_actions_to_frames(actions, frames)` and the positional `LinkReport(...)` construction at `utils.py:222`) still work; `LinkReport` gains `per_period_link_rate` **as the last field with `field(default_factory=dict)`** so positional construction is unbroken. Default behavior on a healthy match emits no warning.
- **Healthy-match silence.** A 100%-linked 2-period match → zero warnings, `per_period_link_rate == {1: 1.0, 2: 1.0}`.

## 6. Edge cases

- Empty actions → existing early return; extend the empty `LinkReport` with `per_period_link_rate={}`; guard no-ops (empty min → skip).
- Period with actions but zero frames → `per_period_link_rate[p] = 0.0`, flagged; message names it (overlap 0.0). This is desired loudness (e.g. tracking missing for a period); `"ignore"` is the escape hatch for known-legitimate cases.
- All-unlinked match → worst period rate 0.0 → fires.
- Float fragility: overlap fractions compared against `OVERLAP_FLOOR` with a small tolerance; message ranges rounded for readability but comparisons use raw floats. Tests use `pytest.approx`.

## 7. Housekeeping

- **ADR-017** (time-base contract + loud low-coverage guard) — authored and bundled into the single feature commit (per the project's no-standalone-doc-commits rule). Mirrors ADR-010's fail-loud structure; documents warn-vs-raise rationale + the per-period caveat + the §3c rejection.
- **CLAUDE.md** — one architecture line under the tracking namespace section noting the time-base contract + `link_actions_to_frames` guard params + `validate_time_base`; amend the "Linkage primitive" sentence.
- **C4** — no container-structure change (no new model / KDE backend / aggregator). The `tracking` container description does not enumerate the linker's guard params; confirm no token/count drift during `/final-review`. If the container description's linkage sentence needs the contract noted, edit the one string + regenerate.
- **NOTICE** — no entry (no published methodology implemented).
- **Version** — next minor. The parallel session has 4.12.0 + ADR-016 claimed (in-flight ghost-gk serve-mean). Reconcile the exact number against `origin/main` + the parallel session's status at release time per the version-bump checklist (5 sites + `uv lock`); if 4.12.0 lands first, this ships 4.13.0.
- **Lint trio** before push: `ruff check` + `ruff format --check` + `pyright silly_kicks/` (whole package), on the pinned tool versions.

## 8. Resolved (lakehouse review, 2026-06-04)

1. **Mismatch floor — decoupled, lowered to 0.2.** `MISMATCH_OVERLAP_FLOOR = 0.2` is a separate module constant from `min_link_rate` (0.5). Symptom (`min_link_rate`) and cause-hypothesis (`MISMATCH_OVERLAP_FLOOR`) are independent; 0.2 keeps the mismatch hypothesis specific (catches GS's 0.14, ignores moderate sparsity). See §4.2.
2. **`validate_time_base` default `on_mismatch="raise"` — confirmed correct.** The asymmetry with the linker's `on_low_coverage="warn"` default is intended, not inconsistent: an explicitly-invoked pre-flight assertion *should* fail loud (you call it because you want the contract enforced), whereas the linker runs on every link and must not break legitimately-partial matches. See §4.4.
3. **Message enumerates all suspected periods, worst-first.** Not worst-only. `suspected_mismatch_periods` is an ordered `tuple` (lowest overlap first) and the message text lists each. See §4.2.

### Incorporated from review
- **HIGH** — convention-pinning tests per events converter (§5) make §4.1 an enforced contract, not drift-prone prose.
- **HIGH** — Surface B (§4.4) reframed as the *primary* guard for pre-filtering consumers (the linker cannot see actions a pre-filter dropped); lakehouse adoption wires `validate_time_base` at work-unit entry.
- **MEDIUM** — `per_period_link_rate` computed from `merged_all` (retains `period_id`), not the returned `pointers` (drops it), avoiding a re-laundering impl (§4.3).
- **MEDIUM** — `MISMATCH_OVERLAP_FLOOR` decoupled + set to 0.2 (§4.2).
- **MEDIUM** — detector vectorized via `groupby().agg`, not the neighboring `iterrows` pattern (§4.2).
