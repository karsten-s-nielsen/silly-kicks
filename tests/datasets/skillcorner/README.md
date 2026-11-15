# SkillCorner converter fixtures

`basic_possessions.csv` and `derived_actions.csv` are hand-written synthetic dynamic events (no generator). They cover the converter's dispatch: one row per end type and set-piece kind, and the derived interception, recovery, tackle and keeper-save rows.

`time_end` (ADR-133, spec 2026-11-13-cross-definition r8 6.6 A) was added on 2026-11-14 with one rule: every `player_possession` row has `time_end = time_start`, a zero-length possession (24 % of real possessions are zero-length). The times of the existing rows therefore do not change; the native rows now start at `x_end`/`y_end`, the on-ball action at the possession end. `on_ball_engagement` rows leave `time_end` empty, because the converter reads it only on possession rows.

`match_metadata.json` gives a 105 x 68 pitch. `raw/1886347_events_slim.csv` is a five-column slice of a real match for the invariant tests.
