# ADR-069: Detection-aware provider visibility guardrails

| Field | Value |
|---|---|
| **Date** | 2026-08-24 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The ghost-GK model trains keeper *detection*, so its labels are valid only where the keeper was
actually observed. SkillCorner is the one bundled provider carrying a per-player detection flag
(`_DETECTION_AWARE_PROVIDERS = {"skillcorner"}`); every other bundled provider is fully observed by
construction (`_FULLY_OBSERVED_PROVIDERS = {"gradientsports", "sportec", "idsse", "metrica"}`). The
**native** `tracking.skillcorner` builder preserves the flag as `visibility`; the **kloppy** gateway
hard-codes `visibility=None`, which the contract treats as "detection discarded" — reading it as
"observed" would train ghost-GK on interpolator output (~80% of SkillCorner keeper positions are
extrapolated). `keeper_detection_mask` already encodes the rule (a detection-aware provider whose
`visibility` is entirely null RAISES; spec 4.3) and it is correct — it is what *caught* the bad data.

The rule fired **per-frame, inside the corpus pass**. During the position-only cycle a `clean_cache`
held 10 SkillCorner games, all kloppy-built (all-null `visibility`), grouped at corpus positions
72–81; the ghost run reached them last, `keeper_detection_mask` raised on each, and `for_each`'s
3-consecutive-failure backstop aborted at absolute game 74/81 — an hour in, with an ordering-dependent
message. The pre-flight `validate_corpus_providers` validated provider *classification* but not that a
detection-aware provider's frames actually *carry* visibility, so a stale/kloppy corpus passed the
pre-flight and failed deep in extraction.

## Decision

A **visibility-usability contract for detection-aware providers**, enforced at two layers around one
shared rule, failing **loud** with the native-rebuild remedy — never silently excluding. The taxonomy
+ rule move to a neutral private module `tracking/_provider_visibility.py`; `keeper_detection_mask`,
the tc3 materializer (build-time, Layer 1), and the ghost trainer (consume-time pre-flight, Layer 2)
all route through it.

- **Shared rule** `assert_detection_aware_visibility(visibility, *, provider)` — raises iff
  `provider in _DETECTION_AWARE_PROVIDERS and visibility.isna().all()` (empty raises too, matching the
  original `keeper_detection_mask` semantics — no `len()` guard). `keeper_detection_mask` delegates;
  the raise trigger and returned mask are preserved, only the message is unified (provider-generic,
  since it is now surfaced from three sites where a `keeper_detection_mask:` prefix would mislead).
- **Layer 1 (build-time):** `materialize_tc3_frames._guard_provider_frames(frames, provider)`, called
  by `_work` before it returns frames, so a detection-discarding shard is never written. Requires the
  `visibility` column present (a dropped column is a discarding regression too) and non-all-null.
- **Layer 2 (consume-time):** `train_ghost_gk.validate_corpus_visibility(provider_by_path)`, called in
  the discovery pre-flight after `validate_corpus_providers`, reads parquet `null_count` **metadata**
  (zero data pages) for **every** detection-aware shard — so a MIXED corpus (one tail-kloppy shard)
  is caught, not just a systematic one. Provider is resolved from `source_provider` (the flat
  `for_each` generation has no `{provider}/` path segment).

**Module placement (the neutral home).** "Which providers carry a detection flag" is a property of
provider DATA, not of the ghost model — Layer 1 is the tell, since a general tc3 builder is not
ghost-specific. This repo already ruled on this shape: 4.53.0 promoted `_id_compat` out of `tracking/`
because "a mandatory seam is public API by definition — the underscore was a false signal"
(`docs/PRIVATE_CONSUMERS.md`). The move is a **true clean break via a module alias**: `_ghost_gk` does
`from . import _provider_visibility as _pv` and references `_pv.*`, so no moved name enters
`_ghost_gk`'s namespace and `_ghost_gk.validate_provider` genuinely fails (a bare
`from ._provider_visibility import validate_provider, _FULLY_OBSERVED_PROVIDERS` would transitively
re-export those two — the shim the move claims to avoid — while `_DETECTION_AWARE_PROVIDERS` broke: an
asymmetric half-clean break). A negative test pins that `_ghost_gk` exposes none of the four moved
names. Private, not public/top-level: every consumer is tracking-adjacent (YAGNI; promotion follows
the id_compat path only if an external consumer appears).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep the taxonomy in `_ghost_gk.py`, import it into the materializer | Smaller change | A general tc3 builder imports from a ghost-private module — the exact layering smell 4.53.0 wrote an ADR to kill; and `keeper_detection_mask` consuming two moved names makes the break asymmetric | Re-introduces the shape id_compat removed |
| B. Sample one detection-aware shard per provider | Cheap | Misses a MIXED corpus (a tail-kloppy rebuild slips past) | Superseded by reading every shard's `null_count` — same cost class, complete |
| C. Pre-flight only (no build-time Layer 1) | One place | Misses a future materializer regression that writes poisoned shards | The build layer is one shared call and stops the poison at the source |
| D. Exclude-and-count the bad shards | No crash | Trains on a quietly-truncated corpus while reporting success (the ADR-043 "all-NaN is indistinguishable" trap) | Silent truncation is the failure this exists to prevent |
| E. **Two layers + one shared rule in a neutral module (chosen)** | Fails loud at build AND consume; catches mixed; single source; correct ownership | A new private module to maintain; a metadata read per detection-aware shard at pre-flight | — |

## Consequences

### Positive

- A kloppy/stale detection-aware corpus now fails **immediately** — at materialize (Layer 1, the
  poison never becomes a shard) and at the trainer pre-flight (Layer 2, before extraction) — instead
  of an hour into a run via the `for_each` abort.
- Layer 2's `null_count`-metadata scan of every shard catches a MIXED corpus, closing the tail-kloppy
  gap a systematic-only check would miss.
- Single source for the provider visibility taxonomy + rule, with correct ownership (provider-data,
  not ghost-model) and a genuine clean break (no transitive re-export).

### Negative

- A new private module (`tracking/_provider_visibility.py`) to maintain, and 3 in-repo import sites
  migrated (clean break, no shim).
- Layer 2 opens each detection-aware shard's metadata at pre-flight (metadata-only; a read fallback
  only if statistics are absent).

### Neutral

- The `keeper_detection_mask` all-null message loses its `keeper_detection_mask:` prefix (unified,
  provider-generic). No existing test pinned the prefix (`match` strings were `discarded|null`,
  `skillcorner`, `unknown` — all preserved).
- **No model change, no retrain trigger.** The native path already carries a real `visibility`, so
  both layers are no-ops on every currently-bundled corpus; they fire only on a detection-discarding
  builder (a future revert).

## Related

- **Specs:** `docs/superpowers/specs/2026-08-24-detection-aware-visibility-guardrails-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-24-detection-aware-visibility-guardrails.md`
- **ADRs:** ADR-038 (SkillCorner visibility gating / the kloppy `visibility=None`); ADR-043 (an
  all-NaN column is indistinguishable downstream — fail loud, don't degrade); the 4.53.0 `_id_compat`
  promotion precedent (`docs/PRIVATE_CONSUMERS.md`).
- **In-repo consumers:** `docs/PRIVATE_CONSUMERS.md` (new `tracking/_provider_visibility.py` entry).

## Notes

The parquet `null_count` path was verified on a real native SkillCorner shard: `visibility` is arrow
`bool` / pandas `object` and its nulls are real parquet nulls (metadata `null_count` 46962 of
1,080,126 = the observed 4.3%), not float-NaN — so the metadata read is reliable. Should a future
producer store `visibility` as float, the read fallback still decides correctly; re-check the metadata
path then.

## Amendment 2026-10-26 -- Gradient Sports is detection-aware (TF-46 Part 2 spec section 27.15, finding G)

Source: the DL design G1-G8 (2026-10-26), saved with the owner decisions at
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_v3_g_slow_error_2026-10-26.md`; owner decision 2026-10-26: in
the C2 amend, in full. The premise "every other bundled provider is fully observed by construction" was wrong for
Gradient Sports.

- **Fact.** The raw GS records are broadcast-derived tracking: every player and ball object carries `visibility`
  (`VISIBLE` / `ESTIMATED`) and `confidence` (`LOW` / `MEDIUM` / `HIGH`). In match 10502 only 0.31 of the player
  positions and 0.63 of the ball records are `VISIBLE`; 26 % of the records have no ball.
- **Decision.** `_DETECTION_AWARE_PROVIDERS = {"skillcorner", "gradientsports"}`; `_FULLY_OBSERVED_PROVIDERS =
  {"sportec", "idsse", "metrica"}`; `_DEAD_BALL_OBSERVED_PROVIDERS = {"sportec", "idsse"}` (the GS loader writes a
  constant `"alive"`; the schema has no null value; a dead-ball state derived from `game_event` has no validation
  truth on GS and is not built). The GS adapter accepts an optional input column `visibility_native` (closed
  vocabulary: `VISIBLE` -> True, `ESTIMATED` -> False, any other value raises) and writes the boolean `visibility`;
  without the column `visibility` is None and every detection consumer fails loud (as the SkillCorner adapter). The
  loader reads the raw arrays with their own flags, never the `*Smoothed` copies; `confidence` is not carried.
- **Detection-aware does NOT imply a camera footprint.** GS has a detection flag and no footprint. A footprint
  consumer tests for the footprint itself (`detect_ball_contacts` reads `footprint_active` from the data profile's
  gate entry; the `_camera_footprint` functions take the footprint or `in_camera_view` as input); it never infers a
  footprint from provider membership. No library consumer infers one (sweep 2026-10-26).
- **Consequences (Hyrum's law).** Every detection consumer now gates GS: the keeper detection gate
  (`resolve_detected_keeper_xy`, `keeper_detected`, the action-context defending keeper, GK influence, GKDV, rest
  defence, GK decision), `detected_mask` / `keeper_detection_mask` (ghost-GK, coordination), ball-contact attribution,
  the kick resolver's launch rule; an ESTIMATED GS keeper is honest-NaN, as on SkillCorner. GS frames built before
  this amendment (no flag) fail loud at those consumers until re-staged. The models and research artifacts that read
  a GS detection mask, GS ball z / `is_aerial` or the GS dead-ball state are classified (G7) and re-run or retrained
  before the joint release; each model publish needs the owner's approval (ADR-088).
