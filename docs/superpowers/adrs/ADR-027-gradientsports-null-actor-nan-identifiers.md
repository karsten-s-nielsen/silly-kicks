# ADR-027: Gradient Sports null-actor duel/foul events carry NaN identifiers (nullable Int64), never a sentinel

| Field | Value |
|---|---|
| **Date** | 2026-06-11 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (luxury-lakehouse production report + same-session withdrawal of its original prescription) |

## Context

The Gradient Sports SPADL converter (`spadl/gradientsports.py`) emitted the integer
sentinel `0` as `team_id`/`player_id` on null-actor events. It did so because
`SPADL_COLUMNS` types both ids as non-nullable `int64`, which cannot hold NaN, so the
converter did `events["team_id"].astype("Int64").fillna(0).astype("int64")`. Gradient
Sports is the only int-id provider; the kloppy-family providers (sportec, skillcorner)
carry object-string ids where an absent actor is naturally `None` (`pd.isna`-routable),
which is why no other provider ever produced the sentinel.

Because `0` is non-NaN, it masqueraded as a real team id: it bypassed every downstream
`pd.isna` NaN-route and reached the strict two-team guard in
`tracking._space_creation._resolve_opponent_team_id`, which raised
`ValueError: attacking_team_id '0' does not uniquely match the frame team ids [...]`.
This took down every Gradient Sports unit in the luxury-lakehouse action-context pipeline
(2026-06-11). Under ≤4.22.1 the same rows produced silent NaN space values; 4.23.0's loud
two-team guard turned the latent corruption into a hard failure. Good guard, bad input.

The lakehouse's report prescribed resolving the real team from "the acting player's
roster." Ground truth from the canonical PFF WC2022 feed (64 matches) contradicts that
premise: the null-team events are **594 `OTB`+`CH` challenges + 28 `FOUL`+`FO` fouls**,
and on **100%** of them `gameEvents.playerId` is ALSO null. A challenge is an inherently
two-sided 50/50 duel (it carries `homeDuelPlayerId` AND `awayDuelPlayerId`) — there is no
single owning team, which is precisely why PFF leaves `teamId` null. The only
team-bearing ids that exist (`challengerPlayerId`, `challengeWinnerPlayerId`,
`onFieldCulpritPlayerId`) are possession-event **qualifiers**. Synthesizing `team_id`
from them is exactly the ADR-001 violation that silly-kicks 2.0.0 removed (the sportec
tackle-winner override the lakehouse itself reported in PR-LL2); ADR-001 even classifies
team-less fouls as *legitimate NULL*. On surfacing this, the lakehouse withdrew its
prescription and confirmed the NaN approach.

## Decision

1. **`GRADIENTSPORTS_SPADL_COLUMNS` types `team_id`/`player_id` as nullable `Int64`**
   (overriding the `SPADL_COLUMNS` `int64`). They mirror the canonical `gameEvents`
   actor verbatim and carry **NaN where the actor is absent — never a sentinel `0`**.
   NaN is `pd.isna`-routable, so the existing downstream NaN-row defaults handle these
   rows; the strict opponent guard stays strict (it must still raise on a genuinely
   unmatched non-NaN id).

2. **ADR-001-legal canonical self-heal** (`_resolve_team_ids`): where a row has a real
   `player_id` but a null `team_id`, the team is derived from that player's other
   same-match rows (a player belongs to exactly one team per match). This keys ONLY on
   the canonical `player_id` column, NEVER on a duel/foul qualifier; an ambiguous mapping
   (a player attributed to >1 team) raises rather than guesses. On the canonical feed it
   resolves nothing (player_id is null wherever team_id is) → all null-actor rows are NaN;
   it self-heals only genuine player-present/team-absent rows, and the ADR-001 boundary
   stays inside silly-kicks (it does not depend on any lakehouse-side `player_id`
   enrichment).

3. **No converter path may synthesize `team_id`/`player_id` from a qualifier** (ADR-001
   reaffirmed). Duel participants remain available in the dedicated `tackle_winner_*` /
   `tackle_loser_*` columns for consumer-side attribution.

4. **Downstream NaN-safety is verified, not assumed.** A NaN-team action on a healthy
   two-team frame must NaN-route through every frame-aware consumer. Auditing this
   surfaced — and this ADR's change fixes — a masked second crash and a silent miscompute
   in `tracking._line_breaking` (`add_line_break(method="ward")`): a NaN-team action hit a
   raw `t != action_team` list-comp (`TypeError: boolean value of NA is ambiguous`), and
   that same raw `!=` compared an Int64 action team against object-string frame teams
   (always True), silently keeping the actor's own team as the "opponent" for every GS
   Ward action. Both are fixed via a `pd.isna` NaN-route plus the ADR-019 `same_id`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep `0`, document it | no dtype change | a sentinel only some code paths understand is a trap; this incident is the proof; non-NaN ⇒ bypasses every `pd.isna` route | the trap is the bug |
| B. Resolve `team_id` from duel/foul qualifiers (lakehouse's original ask) | the 622 rows get teams + enrichment | the acting player does not exist on the feed; a duel has no single team; re-commits the exact ADR-001 violation from PR-LL2; encodes a guess as ground truth (Hyrum) | withdrawn by the lakehouse once the false premise was shown |
| C. Relax the downstream guard to silently NaN-route an unmatched id | unblocks fast | re-opens the silent-corruption class the guard exists to catch (it caught THIS within one production run) | the guard stays strict; the input gets fixed |
| **D (chosen).** Nullable `Int64` + NaN-not-sentinel + canonical-only self-heal | ADR-001/003/019-clean; fixes the crash via the existing NaN routing; keeps every door open (consumer-side attribution from the dedicated columns) | GS id dtype changes (Hyrum); the 622 rows carry NaN (no enrichment) | — chosen |

## Consequences

### Positive

- The SPADL corpus stops carrying a sentinel; every silly-kicks consumer of GS
  conversions is fixed at once, not just the lakehouse.
- The crash is fixed through the existing `pd.isna` routing — the strict guard is
  untouched and keeps catching genuine corruption.
- A masked second crash + a silent Ward miscompute (latent for all GS Ward actions) are
  fixed as a direct result of verifying — not assuming — downstream NaN-safety.

### Negative

- GS `team_id`/`player_id` dtype changes `int64` → `Int64`; consumers asserting `int64`
  must update. The ~622 null-actor rows per WC2022 corpus flip `0` → NaN and carry NO
  action-context enrichment (honest for a contested duel / stoppage). **Hyrum / retrain
  trigger:** VAEP/tracking consumers re-materialize GS.

### Neutral

- Atomic-SPADL conversion now preserves the source id dtype instead of force-casting to
  the atomic schema's `int64` (a latent crash for GS NaN AND for kloppy-family
  object-string ids).
- The ADR-019 AST lint (`tests/tracking/test_id_compat_lint.py`) is a name-heuristic: it
  missed the Ward `!=` because the operands are named `t`/`action_team`, not `*_id`. The
  behavioral NaN-safety harness, not the lint, is what caught it.

## CLAUDE.md Amendment

Extends the `spadl/` converters note and the ADR-019 conventions note (see CLAUDE.md diff
in this change): GS `team_id`/`player_id` are nullable `Int64` and carry NaN (never a
sentinel `0`) on the null-actor duel/foul events; the ADR-019 lint is a name-heuristic
backstopped by the behavioral NaN-safety gate.

## Related

- **ADRs:** ADR-001 (no qualifier→identifier override — reaffirmed), ADR-003 (NaN-safe
  enrichment), ADR-019 (tracking id-dtype contract / `_id_compat`).
- **Report:** luxury-lakehouse `tmp/silly_kicks_gs_sentinel_team_bug_20260611.md` (and the
  same-session withdrawal of its qualifier-resolution prescription).
- **Tests:** `tests/spadl/test_gradientsports.py::TestGradientsportsTeamAttributionNoSentinel`,
  `tests/tracking/test_line_breaking.py` (NaN-route + cross-dtype opponent),
  `tests/tracking/test_space_creation.py::...::test_nan_team_action_on_healthy_frame_returns_nan_row`.
- **Ships in:** silly-kicks 4.25.0.
