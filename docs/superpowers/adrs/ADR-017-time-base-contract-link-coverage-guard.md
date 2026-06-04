# ADR-017: Period-relative `time_seconds` contract + loud per-period link-coverage guard

| Field | Value |
|---|---|
| **Date** | 2026-06-04 |
| **Status** | Accepted — pending implementation (silly-kicks 4.12.0) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.8 (1M); luxury-lakehouse AC-1 session (3-round spec + plan review) |

## Context

luxury-lakehouse AC-1 enrichment silently dropped ~81% of **GradientSports period-2** actions in production (GS 10502 p2: 13%, GS 10503 p2: 19%; period 1 and all other providers 100%). The root cause: GS *frames* are timestamped **period-relative** (`period_elapsed_time`, resets to 0 each period) while GS *actions* carry **absolute** match-clock time (`period_game_clock_time`). The action↔frame match only succeeded where the two ranges accidentally overlapped — GS p2 output `[2700, 3142]` is exactly `intersection(actions [2700, 5835], frames-elapsed [0, 3142.8])`. No error was raised; a manual source-vs-output coverage audit weeks later found it.

A source audit established the decisive fact: **silly_kicks' canonical `time_seconds` convention is period-relative, not absolute.** Opta (`spadl/opta.py:166`) explicitly subtracts cumulative period durations; StatsBomb (`spadl/statsbomb.py:237`) uses the period-elapsed `timestamp`; GS *frames* already conform. The GS *actions* being absolute is the lone non-conformer, injected upstream by the lakehouse bronze→input mapping (the GS events converter passes `time_seconds` through verbatim at `spadl/gradientsports.py:416`). This convention was undocumented — which is exactly how a consumer hand-picked a mismatched column with no signal.

The linker `link_actions_to_frames` is already **per-period scoped** (groups by `period_id`, `merge_asof` within period), so cross-period continuity is irrelevant; the only load-bearing invariant is per-period origin agreement between actions and frames. `LinkReport` already carried `link_rate` / `n_actions_unlinked` but the linker never warned or raised on low coverage.

## Decision

1. **Document period-relative `time_seconds` as the canonical convention** on the tracking + events converter docstrings, `link_actions_to_frames` / `slice_around_event`, and the SPADL + tracking schemas, and **pin it with convention lock tests** (Opta + StatsBomb).
2. **`link_actions_to_frames` gains a per-period low-coverage guard:** `min_link_rate: float = 0.5` + `on_low_coverage: Literal["warn","raise","ignore"] = "warn"`. Evaluated **per period** (worst period), never the match aggregate. `LinkReport` gains `per_period_link_rate`, computed from the internal per-period merge (not the returned pointers, which drop `period_id`).
3. **A decoupled time-base-mismatch diagnostic:** `MISMATCH_OVERLAP_FLOOR = 0.2` governs the *cause hypothesis* (near-disjoint per-period ranges), distinct from `min_link_rate` (the *symptom*). A pure `_diagnose_time_base` feeds both the linker message and the public affordance.
4. **Public `validate_time_base(actions, frames, *, on_mismatch="raise")`** — the primary guard for consumers that pre-filter / window / batch actions by time before linking (the linker guard cannot see actions a pre-filter already dropped).
5. **Reject library-owned GS bronze time-normalization** — the converter receives `time_seconds` already-extracted; owning normalization would require ingesting multi-column bronze, crossing the hexagonal I/O boundary.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Absolute, continuous-across-periods convention | matches the lakehouse's first instinct | conflicts with every events converter (Opta/StatsBomb period-relative); larger change | rejected — inverts the de-facto convention |
| B. Raise-by-default on low coverage | loudest | low coverage is a continuum (camera dropout, missing periods are legitimately partial) → breaks honest callers | rejected — warn is the right default loudness for a continuum |
| C. Match-aggregate threshold | simpler | launders GS 10503's 60.6% whole-match over its 19% p2 — misses the motivating bug | rejected — must be per-period |
| D. Silent opt-in guard | zero behavior change | protects exactly the discard-the-report population that needs it least | rejected — diagnostics nobody reads aren't diagnostics |
| E. (chosen) period-relative docs + per-period warn-default guard + decoupled mismatch diagnostic + `validate_time_base` | enforced contract; loud per-period; pre-filter consumers covered | Hyrum: callers under `-W error` now fail on degraded matches | — |

## Consequences

### Positive
- A converter/linker pipeline can no longer silently drop the majority of a period's actions; the contract is documented and enforced by lock tests.
- Pre-filtering consumers have a real guard (`validate_time_base` on unfiltered inputs at work-unit entry).

### Negative
- **Hyrum's Law:** warn-by-default changes observable behavior — consumers running `-W error` / `filterwarnings=error` will start failing on genuinely-degraded matches. This is the intended shift-left, not a regression to apologize for; a `UserWarning` is strictly gentler than the loudness bar `require_et_direction` (ADR-010) already set.
- Internal callers of `link_actions_to_frames` (the `add_*` aggregators, `_resolve_action_frame_context`) may emit the new warning on low-coverage fixtures; non-fatal (no global `-W error`).

### Neutral
- The lakehouse opts **up** to `on_low_coverage="raise"` with `min_link_rate≈0.9` for AC-1 (hard-fail-first UDF semantics) and wires `validate_time_base` at work-unit entry. The library default protects naive callers; strict consumers tighten by policy.

### Scope note (coverage not overstated)
The convention-pinning lock tests enforce the contract only for converters whose `time_seconds` arithmetic the library **owns** (**Opta, StatsBomb**). **GradientSports `time_seconds` is a verbatim pass-through (`spadl/gradientsports.py:416`) originating upstream in the lakehouse — it is guarded lakehouse-side (`validate_time_base` at work-unit entry + the lakehouse boundary test), NOT by these library tests.** Sportec/kloppy are also pass-through (lower drift risk; not covered here).

## Related

- **Specs:** `docs/superpowers/specs/2026-06-04-tracking-time-base-contract-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-04-tracking-time-base-contract.md`
- **ADRs:** ADR-010 (symmetric fail-loud `require_et_direction` — the precedent this mirrors, one notch softer for a continuous signal)
- **Origin:** luxury-lakehouse AC-1 change request, 2026-06-04
