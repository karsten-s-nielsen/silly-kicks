# ADR-058: Tracking-frame identifier dtypes are NULLABLE, because the ball row has no identity

| Field | Value |
|---|---|
| **Date** | 2026-08-09 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

> **Numbering note — this ADR was RENUMBERED on landing.** It was drafted as ADR-057 while a
> parallel session ran, on the recorded basis that `origin/cycleb-artifact-contracts` added no
> ADR. That basis expired: the branch merged as **4.78.0 carrying ADR-056**. The rule the draft
> stated — *whoever lands second renumbers* — is what was applied, and this cycle landed second.
> Recorded rather than quietly rewritten, because a provisional number that turned out to be
> taken is exactly the case the rule exists for.

## Context

`silly_kicks/tracking/schema.py` declared:

```python
TRACKING_FRAMES_COLUMNS = {
    ...
    "player_id": "int64",
    "team_id": "int64",
}
```

Every tracking frame set carries a **ball row**. The ball belongs to no team and holds no player, so
`player_id` and `team_id` are NA on it **by construction** — this is not a data-quality accident, it
is what the row means. numpy `int64` cannot represent NA, so casting a real frame set to the
declared schema raised `IntCastingNaNError`. Always. On every producer.

The declaration was therefore satisfied by **nothing**:

| variant | `player_id` / `team_id` | satisfied by |
|---|---|---|
| `TRACKING_FRAMES_COLUMNS` (base) | `int64` | *nothing* |
| `KLOPPY` / `SPORTEC` / `SKILLCORNER` / `METRICA` | `object` | those four adapters |
| `GRADIENTSPORTS` | `Int64` | the GS adapter |

All five provider variants overrode it. The Gradient Sports variant's own docstring said the quiet
part out loud — *"nullable Int64 identifiers … allows NaN on ball rows"* — i.e. one variant had
already worked out the correct answer and applied it locally instead of to the base.

### Why this survived so long

4.77.0/ADR-055 planned a dtype pin for `snapshot_to_tracking_frames`, hit `IntCastingNaNError`, and
recorded the pin as **unimplementable**. That conclusion attached the failure to the *pin* rather
than to the *declaration the pin was pinning to*. The follow-on `restore_id_dtype`-based attempt
then changed nothing measurable (0 of 2 tests written for it went red), which read as further
evidence that the question was inert — when in fact both attempts were bouncing off the same wrong
constant.

The only place the base was ever cast to was `_snapshot._empty_frames()`, which builds an EMPTY
frame — where there is no NA to fail on. So the broken declaration was reachable only in the one
case that could not expose it, while the populated path (`_snapshot.py:172`) selects the 20 columns
without applying dtypes at all. Empty and populated snapshots therefore disagreed about
`player_id`'s dtype, and nothing said so.

## Decision

**1. The base declares `Int64` (nullable).**

It is the dtype that is true for every producer, including the one that cannot express its ids any
other way.

**2. Not `object` for the base.**

`object` would also hold NA, and four variants use it legitimately. But `id_compat`'s both-object
fast path is **CONTENT-probed**, not free (~15% of the comparison per side, ~30% for the guard),
because an object column of boxed floats raw-compares False against the same id held as a string.
Defaulting every producer onto that path taxes the comparison seam ADR-019 makes mandatory. The four
`object` variants are KEPT — kloppy domain types and SkillCorner's stringified ids are genuinely
strings, not workarounds — and this ADR does not touch them.

**3. `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` becomes an ALIAS, not a deletion.**

With the base at `Int64` the GS override is redundant. It is aliased (`= TRACKING_FRAMES_COLUMNS`)
rather than deleted, for two reasons: it is exported in `silly_kicks.tracking.__all__`, so deleting
it is a breaking change to public API for no functional gain; and aliasing is already this file's
idiom for *"this provider's schema equals another's"* — `SPORTEC`, `SKILLCORNER` and `METRICA` are
each already `= KLOPPY_TRACKING_FRAMES_COLUMNS`. Restating the two columns as a literal would leave
a duplicate that a future edit could silently diverge from.

End state: **two honest dtypes plus four aliases**, replacing six declarations of which five
contradicted the sixth.

## Consequences

**Behaviour.** The only cast site touching the base is `_snapshot._empty_frames()`; its empty
`player_id`/`team_id` move `int64` → `Int64`. Provider adapters cast to their own variants and are
unchanged. No retrain, no re-materialization, no golden movement.

**The populated snapshot path still does not cast.** That is now *implementable* for the first time,
and deliberately NOT done here: it is a Hyrum-visible output change (a consumer reading today's
`float64` would see `Int64`), and the property downstream code actually depends on — that
`id_compat` comparisons keep working — is already pinned behaviourally on both pandas majors by
`tests/tracking/test_snapshot_id_dtype_across_pandas.py`. Recorded in `TODO.md` with that reasoning.

**Gates.** Two tests in `tests/test_tracking_schema.py`, both landed RED and observed failing on the
`int64` declaration:

- one pins the base directly, and asserts the cast preserves NA rather than producing a sentinel —
  ADR-027 records a non-NA sentinel as a crash source in downstream opponent guards, which would be
  worse than the raise it replaced;
- one is **complete by ENUMERATION** over every `*_TRACKING_FRAMES_COLUMNS` in the module, so a
  future provider variant added with a non-nullable id dtype fails CI. This follows the ADR-043
  idiom: where a rule has a knowable finite surface, enumerate it rather than trusting a reviewer to
  remember.

## The durable lesson

**When a cast to a schema constant fails, the constant is a suspect.** Two consecutive attempts read
`IntCastingNaNError` as a property of the code doing the casting. The tell that it was the
declaration instead was available the whole time and is now the rule: **a constant that every one of
its own variants overrides is a DEFAULT masquerading as a CONTRACT.** It describes one producer's
happy path, and the overrides are the evidence.

The corollary for review: when a schema constant grows a per-provider variant, ask whether the
variant is expressing a genuine provider difference (kloppy's string domain types — yes) or
repairing the base (Gradient Sports' `Int64` — no). The second kind should be fixed at the base the
day it is written.
