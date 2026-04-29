# ADR-001: Converter identifier conventions — caller's team_id / player_id are sacred

| Field | Value |
|---|---|
| **Date** | 2026-04-29 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks's SPADL DataFrame converters accept a `team` column on input
events (with `home_team_id` as a string label compared against it) and
emit `team_id` on the output actions. The implicit contract — natural
given the API surface — is that `team_id` mirrors `team`. Pre-2.0.0,
`silly_kicks/spadl/sportec.py:559-565` violated this contract by
overriding both `team_id` and `player_id` for tackle rows from the raw
DFL `tackle_winner` / `tackle_winner_team` qualifier columns. The override
emitted DFL identifiers (`DFL-CLU-...` / `DFL-OBJ-...`) regardless of the
caller's `team` convention, silently corrupting downstream output.

luxury-lakehouse PR-LL2 close-out (2026-04-29) measured the empirical
impact on `bronze.spadl_actions` for IDSSE (2522 rows): 1412 rows (56%)
had their caller-supplied `home`/`away` team labels silently rewritten
to raw DFL CLU ids on the tackle subset. The lakehouse-side workaround
is a 3-line `_team_label_to_dfl_id` shim that maps DFL- prefixed strings
back to the caller's convention. This shim only solves the lakehouse
case; future consumers normalizing to other conventions (integer team
ids, abbreviated codes, team names) hit the same hazard.

A cross-converter audit (this session) confirmed sportec.tackle is the
unique violator of the implicit contract. The other 5 SPADL DataFrame
converters (metrica, wyscout, statsbomb, opta, kloppy gateway) honor
the caller's `team_id` / `player_id` columns verbatim. The principled
fix is to lock the contract in across the converter surface and document
it as a first-class ADR, so future converter additions don't reintroduce
the override pattern.

## Decision

silly-kicks SPADL DataFrame converters never override the caller's
`team_id` / `player_id` from provider-specific qualifiers. Qualifier-
derived facts surface as dedicated output columns with explicit naming
(e.g. `tackle_winner_team_id`). Caller-supplied `team` / `player_id`
values mirror verbatim into the canonical output fields.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Documentation only + downstream shim | minimal silly-kicks change | pushes a footgun onto every future consumer; lakehouse-specific shim doesn't help unknown future consumers | violates "best practice, long term" |
| B. Deprecation warning in 1.11.0 + flip default in 2.0.0 | softer migration | emits wrong-format values until 2.0.0 anyway; adds log noise without fixing anything | clean break is cleaner |
| C (chosen). Single coherent break in 2.0.0 with `use_tackle_winner_as_actor` migration helper | cleanest contract; one CHANGELOG entry; locks the principle for future converters | breaking change for sportec consumers relying on the override (mitigation: helper) | — |

## Consequences

### Positive

- Caller's identifier conventions preserved across all converters.
- Provider-specific qualifier values surfaced with structure not silent-mutation.
- Future converters with similar quirks follow the same pattern.
- ADR pattern adopted in silly-kicks (lakehouse template vendored verbatim).
- The cross-provider parity regression gate (added in this cycle) locks the contract per-provider going forward.

### Negative

- silly-kicks 2.0.0 is a breaking change for sportec consumers that:
  (a) relied on tackle "actor = winner" SPADL semantic, AND
  (b) had upstream-supplied `team` / `player_id` columns in the same
      format as DFL's `tackle_winner_team` / `tackle_winner` qualifiers
      (raw DFL ids).
  Mitigation: `silly_kicks.spadl.use_tackle_winner_as_actor` helper
  restores pre-2.0.0 behavior in one line.
- Sportec output schema gains 4 columns (sportec consumers must update
  schema-equality assertions to use `SPORTEC_SPADL_COLUMNS`).

### Neutral

- `KLOPPY_SPADL_COLUMNS` / `SPADL_COLUMNS` unchanged; non-sportec
  converters unaffected.
- Adds the per-provider `<PROVIDER>_SPADL_COLUMNS` extension precedent
  for future converters that need to surface provider-specific qualifier
  columns.

## CLAUDE.md Amendment

Adds one rule to the project's `CLAUDE.md` "Key conventions" section:

> Converter identifier conventions are sacred. SPADL DataFrame converters
> never override the caller's `team_id` / `player_id` columns from
> provider-specific qualifiers. Qualifier-derived facts surface as
> dedicated output columns (see `tackle_winner_*` / `tackle_loser_*` on
> sportec). Decision: ADR-001.

Scope: every SPADL DataFrame converter and any future provider added to
the silly-kicks 5-converter surface.

## Related

- **Spec:** `docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md`
- **Issues / PRs:** luxury-lakehouse PR-LL2 close-out report (2026-04-29); silly-kicks #16 (this PR)
- **External references:** kloppy upstream — no equivalent contract
  documented; silly-kicks's pre-2.0.0 override pattern was inherited
  from socceraction v1.5.3 era and never re-examined.

## Notes

### Cross-converter audit (2026-04-29)

| Converter | Override `player_id` / `team_id` from qualifiers? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | YES (tackle_winner / tackle_winner_team) | Removed in 2.0.0; replaced with 4 dedicated columns. |
| `silly_kicks.spadl.metrica` | NO | The 1.10.0 `goalkeeper_ids` routing only changes `type_id` / `bodypart_id`. ✓ |
| `silly_kicks.spadl.wyscout` | NO | The 1.0.0 `goalkeeper_ids` aerial-duel reclassification only changes `type_id` / `subtype_id` on the input events. ✓ |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides. ✓ |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. ✓ |
| `silly_kicks.spadl.kloppy` | NO | Gateway path; relies on the dedicated converters above. ✓ |

### Empirical impact (luxury-lakehouse PR-LL2 close-out)

`bronze.spadl_actions` for IDSSE (2522 rows total):

| Bucket | Rows | Cause |
|---|---|---|
| Non-NULL team_id_native | 164 | TacklingGame rows where `tackle_winner` was empty/NaN — override didn't fire. |
| **NULL — override fired (THE BUG)** | **1412** | TacklingGame rows where `tackle_winner_team` was a raw DFL CLU id; rewrote caller's `home`/`away` label. |
| NULL — bronze team='unknown' | 759 | Foul/Caution/FairPlay/etc. without team attribution — legitimate. |
| NULL — synthetic dribble rows | 187 | Inserted by `_add_dribbles`, propagated the prior overridden `team_id`. |

1412 / 2522 = 56% silent corruption rate on tackle rows.
