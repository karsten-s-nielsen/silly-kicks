# ADR-010: Symmetric fail-loud extra-time direction handling + public `require_et_direction` guard

| Field | Value |
|---|---|
| **Date** | 2026-05-30 |
| **Status** | Accepted — implemented in silly-kicks **4.0.0** |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.8 (1M); luxury-lakehouse maintainer (2-round spec second-opinion review) |

## Context

Per-period-absolute providers (Sportec/IDSSE, Metrica, Gradient Sports) flip tracking/event coordinates **per period** by the home team's start direction (ADR-006). Extra time (periods 3/4) requires a separate `home_team_start_left_extratime` flag because ET ends are not a simple mirror of regular time.

The native converters handled a **missing** ET flag **inconsistently**:

- **Gradient Sports** (`tracking/gradientsports.py:114`, `spadl/gradientsports.py:326`) — **raises `ValueError`**.
- **Sportec events** (`spadl/sportec.py`) and **Metrica events** (`spadl/metrica.py`) — **raise `ValueError`** (added in PR-S23 / silly-kicks 3.0.1; "Sportec + Metrica per-period direction-of-play correctness"), but with **per-converter ad-hoc message wording** rather than a shared guard.
- **Sportec tracking** (`tracking/sportec.py`) — **silently defaults** ET orientation via `direction.home_attacks_right_per_period` (p3→`False`, p4→`True`), producing **geometrically wrong ET coordinates with no signal**. This was the one remaining silent sibling.

Surfaced by the TF-24 calibration dry-run: a GS ET match crashed (GS raises); a source audit confirmed the events converters already raise (since 3.0.1) but inconsistently worded, while **Sportec tracking** still silently mis-orients. Silent wrong-geometry corrupts every downstream geometric feature for ET periods (DAS, pitch control, distances, embeddings, model training data) and violates the project's fail-loud / no-silent-degradation rule. This change closes the one silent gap **and** unifies all five converters behind a single shared guard with one message.

## Decision

1. **Symmetric fail-loud.** All per-period-absolute converters (Sportec + Metrica + GS, tracking **and** events) **raise** on ET-without-flag, via a single shared guard. **Sportec tracking gains a new raise** (it silently defaulted); GS tracking + GS events + Sportec events + Metrica events **refactor their existing inline raises** to the shared guard (standardizing the message — the events converters have raised since 3.0.1). Identical exception type + message across providers (parity-tested).

2. **Promote the guard to public API:** `silly_kicks.tracking.require_et_direction(period_ids, home_team_start_left_extratime, *, source)` (re-exported from `silly_kicks.spadl` for events). Rationale: downstream consumers (the lakehouse) **pre-flight-validate** a batch before converting, and the cross-repo **sentinel test** (see Consequences) calls it to detect a pin/metadata mismatch in CI. An internal-only guard could not serve either.

3. **Full module rename** `tracking/_direction.py → tracking/direction.py` (single public home; no private/public mirror to keep in sync). All importers updated.

4. **Public, calibration-labelled filter helper** `silly_kicks.tracking.utils.filter_extratime_frames` — drops ET periods for **sampling/calibration only**, with a docstring stating production must source the real flag, not drop ET. DRYs the TF-24/TF-25 loaders without inviting the filter into production paths.

5. **SemVer = 4.0.0.** A raise where there was none is a breaking behavioural change in an active path. A major bump is the honest signal to **all** downstream consumers (the lakehouse is one of potentially many). The lakehouse's mechanical bump cost (`bump_wheel.py` + `_REQUIRED_SK_MIN`) is scripted and not a deciding factor.

## Consequences

### Positive
- No converter can silently ship wrong ET geometry again; behaviour is consistent across providers and across the tracking/events boundary.
- The public guard lets consumers fail fast (pre-flight) and lets CI catch a pin/metadata mismatch via the sentinel.

### Negative / Breaking
- ET matches processed **without** the flag now **raise** (were silently mis-oriented). Consumers must source `home_team_start_left_extratime` and pass it.
- 4.0.0 cascades a wider version bump in consumers than a minor would (scripted; acceptable).

### Cross-repo coordination (sentinel pattern)
Phased, CI-enforced — **not** "lockstep":
- **Phase A (consumer, silly-kicks pin unchanged):** add an ET start-direction field to the consumer's match metadata (extending bronze ingestion first if the field is absent there — a Phase A.0 prerequisite); pass it to both `convert_to_frames` and `convert_to_actions`; test ET-present + ET-absent paths.
- **Phase B (silly-kicks 4.0.0 ships, then consumer pin bump):** because Phase A already passes the flag, the bump cannot break prod.
- **Sentinel test (consumer):** a per-batch pre-flight `require_et_direction` call so that a pin/metadata mismatch (guard present + ET field missing + ET in scope) fails loudly in CI rather than at runtime.
- **Ship gate:** silly-kicks 4.0.0 does **not** ship until the consumer's historical-data audit (count of already-processed ET matches + silent-mis-orientation rate) is reported back, so the remediation blast radius is known.

## Alternatives considered
- **Warn + default** — rejected: silent-wrong persists behind an unseen warning.
- **One-release deprecation cycle (3.31 warns → 4.0 raises)** — rejected: ships fail-soft for one release window, violating the no-silent-degradation rule; its only benefit (decoupled adoption) is already provided by the Phase-A-first sequence + sentinel.
- **Leave the asymmetry / call-site-only patches** — rejected: latent correctness bug + guaranteed recurrence; loader `_apply_et_direction` is the interim calibration unblock only.

## References
- ADR-006 (direction-of-play handling) + 3.0.1 erratum.
- Spec: `docs/superpowers/specs/2026-05-30-et-direction-converter-consistency-design.md`.
- Paired luxury-lakehouse ADR (MatchMeta ET field + pre-flight sentinel + version-coupled coordination) — to be authored lakehouse-side; cross-references this ADR.
