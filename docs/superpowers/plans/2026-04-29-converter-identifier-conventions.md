# Converter Identifier Conventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock in the silly-kicks SPADL converter contract — the caller's `team_id` / `player_id` are sacred, never overridden from provider-specific qualifiers — by removing the sportec tackle override, surfacing DFL qualifier values via 4 new dedicated columns, adopting an ADR pattern (vendored from luxury-lakehouse) to record the decision, and shipping all of it as silly-kicks 2.0.0 (first major-version break).

**Architecture:** Hexagonal — pure-function converters, zero I/O, zero global state mutation. The new `SPORTEC_SPADL_COLUMNS` schema constant in `silly_kicks/spadl/schema.py` extends `KLOPPY_SPADL_COLUMNS` with 4 sportec-specific columns. The new `use_tackle_winner_as_actor` migration helper lives in `silly_kicks/spadl/sportec.py` and re-exports from `silly_kicks/spadl/__init__.py`. The contract is captured in `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` (with `ADR-TEMPLATE.md` vendored from luxury-lakehouse), reinforced by `CLAUDE.md`'s "Key conventions" section, and enforced by a new cross-provider parity regression gate.

**Tech Stack:** Python 3.10+, pandas 2.x (with pandas-stubs 2.3.3.260113 pinned), numpy 2.x, pytest, ruff 0.15.7, pyright 1.1.395. No new runtime deps.

**Convention deviation from writing-plans defaults:** silly-kicks ships ONE commit per branch (per `feedback_commit_policy` memory). Tasks below stage changes (`git add`) but DO NOT commit. The single commit happens in Task 11 after `/final-review` and explicit user approval (sentinel-gated by hook).

**Spec:** `docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md`

---

## Branch + Worktree

**Branch:** `feat/converter-identifier-conventions`
**Worktree:** None — single-commit ritual operates directly on the main folder. (No concurrent silly-kicks work expected; matches PR-S10 1.10.0 cycle.)

## File Structure

| Path | Action | Δ Lines | Purpose |
|---|---|---|---|
| `docs/superpowers/adrs/ADR-TEMPLATE.md` | NEW | ~65 lines | Vendored verbatim from `D:\Development\karstenskyt__luxury-lakehouse\docs\superpowers\adrs\ADR-TEMPLATE.md`. Establishes silly-kicks ADR pattern. |
| `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` | NEW | ~90 lines | First silly-kicks ADR. Captures the contract + audit + alternatives + migration. |
| `CLAUDE.md` | MOD | +1 / 0 | "Key conventions" section gains one rule line citing ADR-001. |
| `silly_kicks/spadl/schema.py` | MOD | +9 / 0 | Add `SPORTEC_SPADL_COLUMNS` constant after `KLOPPY_SPADL_COLUMNS`. |
| `silly_kicks/spadl/sportec.py` | MOD | +90 / -10 | Remove tackle override; populate 4 new tackle_* columns; add `use_tackle_winner_as_actor` helper; update module docstring with the contract. |
| `silly_kicks/spadl/__init__.py` | MOD | +3 / 0 | Re-export `SPORTEC_SPADL_COLUMNS` + `use_tackle_winner_as_actor`. |
| `tests/spadl/test_sportec.py` | MOD | +200 / -10 | TestSportecTackleNoOverride + TestSportecTackleWinnerColumns + TestSportecTackleLoserColumns + TestSportecOutputSchema. Delete pre-2.0.0 `test_tackle_uses_winner_as_actor`. |
| `tests/spadl/test_use_tackle_winner_as_actor.py` | NEW | ~140 lines | TestUseTackleWinnerAsActorContract / Correctness / Degenerate (~15 tests). |
| `tests/spadl/test_cross_provider_parity.py` | MOD | +80 / -2 | `test_team_id_mirrors_input_team` parametrized over all 5 converters; sportec branch updated to `SPORTEC_SPADL_COLUMNS`; new `TestSportecAdrContractOnProductionFixture` for the IDSSE e2e. |
| `pyproject.toml` | MOD | +1 / -1 | Version `1.10.0` → `2.0.0`. |
| `CHANGELOG.md` | MOD | +95 lines | `## [2.0.0]` entry with `### ⚠️ Breaking` and `### Migration` sections. |
| `TODO.md` | MOD | ~+3 / -1 | PR-S11 → shipped (referenced from this plan); PR-S12 (was PR-S11 add_possessions improvement) bumped in queue. |
| `docs/c4/architecture.dsl` | MOD | +1 / -1 | Spadl container description mentions `SPORTEC_SPADL_COLUMNS` + `use_tackle_winner_as_actor`. |
| `docs/c4/architecture.html` | REGEN | (script) | Regenerated from .dsl by `/final-review`. |
| `docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md` | TRACK | (spec) | Already exists; bundled into the single commit. |
| `docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md` | THIS FILE | (plan) | Bundled into the single commit. |

---

## Task 0: Pre-flight verification + branch creation

**Files:** none (shell-only)

- [ ] **Step 0.1: Verify clean working tree on main at v1.10.0**

Run:
```bash
git status
git log --oneline -1
grep "^version" pyproject.toml
```

Expected: `On branch main`, `Your branch is up to date with 'origin/main'`, only `README.md.backup` + `uv.lock` + the spec/plan files as untracked; HEAD is `d27dc48 ... silly-kicks 1.10.0 (#15)`; pyproject.toml shows `version = "1.10.0"`.

If anything else is dirty, STOP and ask the user before proceeding.

- [ ] **Step 0.2: Create + checkout the feature branch**

Run:
```bash
git checkout -b feat/converter-identifier-conventions
```

Expected: `Switched to a new branch 'feat/converter-identifier-conventions'`.

- [ ] **Step 0.3: Verify CI pin install**

Run (with explicit timeout, may take ~5-30s):
```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: install succeeds with no errors. Cached from prior cycles, should be fast.

- [ ] **Step 0.4: Baseline test pass + lint clean**

Run (in parallel where independent):
```bash
uv run ruff check silly_kicks/ tests/ scripts/ 2>&1 | tail -3
uv run ruff format --check silly_kicks/ tests/ scripts/ 2>&1 | tail -3
uv run pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -5
uv run pyright silly_kicks/ 2>&1 | tail -3
```

Expected: all green; pytest ~627 passed (post-1.10.0 baseline); ruff/format/pyright zero errors. This is the baseline against which we'll measure later.

If anything fails, STOP and investigate before adding new code on top.

---

## Task 1: ADR pattern adoption — ADR-TEMPLATE.md + ADR-001 + CLAUDE.md amendment

**Files:**
- Create: `docs/superpowers/adrs/ADR-TEMPLATE.md` (verbatim from luxury-lakehouse)
- Create: `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`
- Modify: `CLAUDE.md` (Key conventions section)

This task establishes the ADR pattern in silly-kicks before any code changes. ADR-001 captures the contract that the subsequent tasks implement.

- [ ] **Step 1.1: Create the `docs/superpowers/adrs/` directory + vendor the template**

Run:
```bash
mkdir -p docs/superpowers/adrs
cp "D:/Development/karstenskyt__luxury-lakehouse/docs/superpowers/adrs/ADR-TEMPLATE.md" docs/superpowers/adrs/ADR-TEMPLATE.md
```

Expected: `docs/superpowers/adrs/ADR-TEMPLATE.md` is a verbatim copy of the lakehouse template.

If `cp` is unavailable on the local shell, manually open the source file and write to the destination via the Write tool.

- [ ] **Step 1.2: Write `ADR-001-converter-identifier-conventions.md`**

Create `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`:

```markdown
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
```

- [ ] **Step 1.3: Amend CLAUDE.md "Key conventions" section**

Read `CLAUDE.md` to find the "## Key conventions" section. Insert a new bullet at the end of that bulleted list:

```markdown
- **Converter identifier conventions are sacred.** SPADL DataFrame converters never override the caller's `team_id` / `player_id` columns from provider-specific qualifiers. Qualifier-derived facts surface as dedicated output columns (see `tackle_winner_*` / `tackle_loser_*` on sportec). Decision: ADR-001.
```

If the existing list ends with the line:

```markdown
- ML naming conventions (uppercase `X`, `Y`, `Pscores`) are allowed in `vaep/` and `xthreat.py` per ruff per-file-ignores.
```

then insert the new bullet immediately after that line.

- [ ] **Step 1.4: Stage Task 1 changes (DO NOT commit)**

Run:
```bash
git add docs/superpowers/adrs/ADR-TEMPLATE.md docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md CLAUDE.md
git status
```

Expected: 3 new/modified files staged.

---

## Task 2: `SPORTEC_SPADL_COLUMNS` schema constant + re-export

**Files:**
- Modify: `silly_kicks/spadl/schema.py:41` (insert after `KLOPPY_SPADL_COLUMNS` definition)
- Modify: `silly_kicks/spadl/__init__.py` (re-export)

This task introduces the new schema without changing any converter behavior. Sportec converter still emits 14-column output until Task 3.

- [ ] **Step 2.1: Add `SPORTEC_SPADL_COLUMNS` to `schema.py`**

Edit `silly_kicks/spadl/schema.py`. Find the existing `KLOPPY_SPADL_COLUMNS` definition:

```python
KLOPPY_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "game_id": "object",
    "team_id": "object",
    "player_id": "object",
}
```

Insert immediately after, before the `@dataclasses.dataclass(frozen=True)` line for `ConversionReport`:

```python
SPORTEC_SPADL_COLUMNS: dict[str, str] = {
    **KLOPPY_SPADL_COLUMNS,
    "tackle_winner_player_id": "object",
    "tackle_winner_team_id": "object",
    "tackle_loser_player_id": "object",
    "tackle_loser_team_id": "object",
}
"""Sportec SPADL output schema: KLOPPY_SPADL_COLUMNS + 4 sportec-specific
qualifier passthrough columns surfacing DFL ``tackle_winner`` /
``tackle_winner_team`` / ``tackle_loser`` / ``tackle_loser_team`` qualifier
values verbatim. NaN on rows where the qualifier is absent in the source;
always NaN on non-tackle rows. See ADR-001 for the contract rationale."""
```

- [ ] **Step 2.2: Re-export from `silly_kicks/spadl/__init__.py`**

Edit `silly_kicks/spadl/__init__.py`. Find the existing `__all__` list. Insert `"SPORTEC_SPADL_COLUMNS"` alphabetically:

```python
__all__ = [
    "SPADL_COLUMNS",
    "SPORTEC_SPADL_COLUMNS",
    "BoundaryMetrics",
    "ConversionReport",
    "CoverageMetrics",
    "actiontypes_df",
    # ... rest unchanged ...
]
```

Then update the `from .schema import` block:

```python
from .schema import SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
```

(Note: `use_tackle_winner_as_actor` re-export is added in Task 4. Don't add it here yet.)

- [ ] **Step 2.3: Smoke-test the re-export**

Run:
```bash
uv run python -c "from silly_kicks.spadl import SPORTEC_SPADL_COLUMNS; assert len(SPORTEC_SPADL_COLUMNS) == 18, f'expected 18 columns, got {len(SPORTEC_SPADL_COLUMNS)}'; assert 'tackle_winner_player_id' in SPORTEC_SPADL_COLUMNS; print('SPORTEC_SPADL_COLUMNS OK')"
```

Expected: `SPORTEC_SPADL_COLUMNS OK`.

- [ ] **Step 2.4: Run pytest to confirm no regression**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=line -q 2>&1 | tail -3
```

Expected: 627 passed (no test should break — schema is additive and unused by the converter yet).

- [ ] **Step 2.5: Stage Task 2 changes**

```bash
git add silly_kicks/spadl/schema.py silly_kicks/spadl/__init__.py
```

---

## Task 3: sportec converter — remove tackle override + add 4 new columns (TDD)

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (~lines 559-565: remove override; lines 600-630: add 4 new columns to actions DataFrame; convert_to_actions: switch schema to `SPORTEC_SPADL_COLUMNS`)
- Modify: `tests/spadl/test_sportec.py` (add 4 new test classes; update / delete pre-2.0.0 override-asserting tests)

This is the largest task — the core ADR-001 implementation. TDD: write failing tests first, see them fail, then implement, see them pass.

- [ ] **Step 3.1: Add new test classes to `tests/spadl/test_sportec.py`**

Append after the existing test classes (after `TestSportecGoalkeeperIdsSupplementary` class which lives at the bottom from PR-S10):

```python
# ---------------------------------------------------------------------------
# ADR-001: tackle override removal — caller's team_id / player_id are sacred
# (2.0.0; supersedes pre-2.0.0 sportec.py:559-565 override that silently
# rewrote 56% of tackle rows on consumers using a normalized team convention)
# ---------------------------------------------------------------------------


def _df_tackle_with_qualifiers() -> pd.DataFrame:
    """One TacklingGame event with all 4 DFL qualifier columns populated."""
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["TacklingGame"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P-LOSER"],  # event-level recorded player (DFL records from loser perspective)
            "team": ["T-AWAY"],  # caller's team label, NOT raw DFL CLU id
            "x": [50.0],
            "y": [34.0],
            "tackle_winner": ["P-WINNER"],
            "tackle_winner_team": ["DFL-CLU-WINNER"],  # raw DFL CLU id (different convention from caller's team)
            "tackle_loser": ["P-LOSER"],
            "tackle_loser_team": ["DFL-CLU-LOSER"],
        }
    )


def _df_tackle_no_qualifiers() -> pd.DataFrame:
    """One TacklingGame event with no winner/loser qualifiers (DFL records 50/50 duels this way)."""
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["TacklingGame"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [34.0],
        }
    )


class TestSportecTackleNoOverride:
    """ADR-001: caller's team_id / player_id mirror input verbatim — never
    overridden from tackle_winner / tackle_winner_team qualifiers.
    """

    def test_tackle_team_id_mirrors_input_team(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["team_id"].iloc[0] == "T-AWAY", (
            f"team_id should mirror input 'T-AWAY', got {actions['team_id'].iloc[0]!r}"
        )

    def test_tackle_player_id_mirrors_input_player_id(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["player_id"].iloc[0] == "P-LOSER"

    def test_legacy_home_label_survives(self):
        df = _df_tackle_with_qualifiers()
        df["team"] = ["home"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="home")
        assert actions["team_id"].iloc[0] == "home"

    def test_legacy_away_label_survives(self):
        df = _df_tackle_with_qualifiers()
        df["team"] = ["away"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="home")
        assert actions["team_id"].iloc[0] == "away"

    def test_mixed_tackle_rows_preserve_per_row_labels(self):
        df = pd.DataFrame(
            {
                "match_id": ["M1"] * 2,
                "event_id": ["e1", "e2"],
                "event_type": ["TacklingGame"] * 2,
                "period": [1, 1],
                "timestamp_seconds": [10.0, 20.0],
                "player_id": ["P1", "P2"],
                "team": ["home", "away"],
                "x": [50.0, 60.0],
                "y": [34.0, 34.0],
                "tackle_winner": ["P-W1", "P-W2"],
                "tackle_winner_team": ["DFL-CLU-A", "DFL-CLU-B"],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="home")
        # Take only the tackle rows (no synthetic dribbles).
        tackle_rows = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]].reset_index(drop=True)
        assert tackle_rows["team_id"].tolist() == ["home", "away"]

    def test_tackle_with_no_qualifier_unchanged(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        assert actions["team_id"].iloc[0] == "T-HOME"
        assert actions["player_id"].iloc[0] == "P1"


class TestSportecTackleWinnerColumns:
    """The 4 new DFL qualifier passthrough columns: populated verbatim
    when present in input; NaN otherwise; always NaN on non-tackle rows."""

    def test_tackle_winner_player_id_populated_from_qualifier(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["tackle_winner_player_id"].iloc[0] == "P-WINNER"

    def test_tackle_winner_team_id_populated_from_qualifier(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["tackle_winner_team_id"].iloc[0] == "DFL-CLU-WINNER"

    def test_winner_columns_nan_when_qualifier_absent(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        assert pd.isna(actions["tackle_winner_player_id"].iloc[0])
        assert pd.isna(actions["tackle_winner_team_id"].iloc[0])

    def test_winner_columns_nan_on_non_tackle_rows(self):
        # Mix tackle + pass-class (Play). Assert pass row has NaN winner cols.
        df = pd.DataFrame(
            {
                "match_id": ["M1"] * 2,
                "event_id": ["e1", "e2"],
                "event_type": ["TacklingGame", "Play"],
                "period": [1, 1],
                "timestamp_seconds": [10.0, 20.0],
                "player_id": ["P1", "P2"],
                "team": ["T-HOME", "T-HOME"],
                "x": [50.0, 60.0],
                "y": [34.0, 34.0],
                "tackle_winner": ["P-WIN", None],
                "tackle_winner_team": ["DFL-CLU-X", None],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        # Find the pass row by type_id.
        pass_row = actions[actions["type_id"] == spadlconfig.actiontype_id["pass"]].iloc[0]
        assert pd.isna(pass_row["tackle_winner_player_id"])
        assert pd.isna(pass_row["tackle_winner_team_id"])

    def test_winner_columns_unaffected_by_extraneous_input_columns(self):
        # tackle_winner column on a non-TacklingGame row must NOT leak.
        df = pd.DataFrame(
            {
                "match_id": ["M1"],
                "event_id": ["e1"],
                "event_type": ["Play"],  # NOT TacklingGame
                "period": [1],
                "timestamp_seconds": [10.0],
                "player_id": ["P1"],
                "team": ["T-HOME"],
                "x": [50.0],
                "y": [34.0],
                "tackle_winner": ["P-WIN"],  # Extraneous: caller has this column on a non-tackle row
                "tackle_winner_team": ["DFL-CLU-X"],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        # Non-tackle row must have NaN winner cols regardless of input.
        assert pd.isna(actions["tackle_winner_player_id"].iloc[0])
        assert pd.isna(actions["tackle_winner_team_id"].iloc[0])

    def test_winner_columns_present_in_output_schema(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        for col in ("tackle_winner_player_id", "tackle_winner_team_id"):
            assert col in actions.columns


class TestSportecTackleLoserColumns:
    """Symmetric to winner: tackle_loser_* qualifiers also surface verbatim."""

    def test_tackle_loser_player_id_populated_from_qualifier(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["tackle_loser_player_id"].iloc[0] == "P-LOSER"

    def test_tackle_loser_team_id_populated_from_qualifier(self):
        df = _df_tackle_with_qualifiers()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["tackle_loser_team_id"].iloc[0] == "DFL-CLU-LOSER"

    def test_loser_columns_nan_when_qualifier_absent(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        assert pd.isna(actions["tackle_loser_player_id"].iloc[0])
        assert pd.isna(actions["tackle_loser_team_id"].iloc[0])

    def test_loser_columns_nan_on_non_tackle_rows(self):
        df = _df_pass_default()  # pass-class Play event (no qualifier columns at all)
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert pd.isna(actions["tackle_loser_player_id"].iloc[0])
        assert pd.isna(actions["tackle_loser_team_id"].iloc[0])


class TestSportecOutputSchema:
    """Sportec output now uses SPORTEC_SPADL_COLUMNS (= KLOPPY_SPADL_COLUMNS + 4)."""

    def test_output_columns_match_sportec_spadl_columns_keys(self):
        from silly_kicks.spadl.schema import SPORTEC_SPADL_COLUMNS

        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        assert list(actions.columns) == list(SPORTEC_SPADL_COLUMNS.keys())

    def test_output_dtypes_match_sportec_spadl_columns(self):
        from silly_kicks.spadl.schema import SPORTEC_SPADL_COLUMNS

        actions, _ = sportec_mod.convert_to_actions(_df_tackle_no_qualifiers(), home_team_id="T-HOME")
        for col, expected in SPORTEC_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected, f"{col}: got {actions[col].dtype}, expected {expected}"

    def test_synthetic_dribble_rows_have_nan_in_tackle_columns(self):
        # Two same-team passes with positional gap → _add_dribbles inserts a
        # synthetic dribble. The dribble must have NaN in tackle_* columns.
        events = pd.DataFrame(
            {
                "match_id": ["M1"] * 2,
                "event_id": ["e1", "e2"],
                "event_type": ["Play", "Play"],
                "period": [1, 1],
                "timestamp_seconds": [10.0, 12.0],
                "player_id": ["P1", "P2"],
                "team": ["T-HOME"] * 2,
                "x": [30.0, 60.0],
                "y": [34.0, 34.0],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        dribble_rows = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
        assert len(dribble_rows) >= 1
        for col in ("tackle_winner_player_id", "tackle_winner_team_id", "tackle_loser_player_id", "tackle_loser_team_id"):
            assert dribble_rows[col].isna().all(), f"{col} should be NaN on synthetic dribble rows"
```

- [ ] **Step 3.2: Run new tests, verify they fail correctly**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecTackleNoOverride tests/spadl/test_sportec.py::TestSportecTackleWinnerColumns tests/spadl/test_sportec.py::TestSportecTackleLoserColumns tests/spadl/test_sportec.py::TestSportecOutputSchema -v --tb=line 2>&1 | tail -25
```

Expected: most tests fail. Specifically:
- `TestSportecTackleNoOverride::test_tackle_team_id_mirrors_input_team` → FAIL (override still rewrites to DFL-CLU-WINNER)
- `TestSportecTackleNoOverride::test_legacy_home_label_survives` → FAIL (override rewrites to DFL-CLU-A)
- `TestSportecTackleWinnerColumns::test_winner_columns_present_in_output_schema` → FAIL (column doesn't exist yet)
- `TestSportecTackleLoserColumns::test_tackle_loser_player_id_populated_from_qualifier` → FAIL (column doesn't exist yet)
- `TestSportecOutputSchema::test_output_columns_match_sportec_spadl_columns_keys` → FAIL (output still has 14 columns, not 18)

A few tests may pass coincidentally (e.g., the no-qualifier case where override doesn't fire) — that's fine.

- [ ] **Step 3.3: Implement the sportec.py changes**

Edit `silly_kicks/spadl/sportec.py`. Three coordinated changes.

**Change 1 — Remove the tackle override block** at the existing tackle dispatch site (search for `is_tackle = et == "TacklingGame"`). Replace this block:

```python
    # --- TacklingGame: actor = tackle_winner if present, else generic player_id/team ---
    is_tackle = et == "TacklingGame"
    type_ids[is_tackle] = spadlconfig.actiontype_id["tackle"]
    result_ids[is_tackle] = spadlconfig.result_id["success"]
    if is_tackle.any():
        winner_p = _opt("tackle_winner", None)
        winner_t = _opt("tackle_winner_team", None)
        override_mask = is_tackle & winner_p.notna().to_numpy()
        if override_mask.any():
            rows.loc[override_mask, "player_id"] = winner_p[override_mask].values
            rows.loc[override_mask, "team"] = winner_t[override_mask].values
```

With:

```python
    # --- TacklingGame ---
    # Per ADR-001: caller's team / player_id mirror verbatim; the converter
    # never overrides them from DFL qualifier columns. Winner / loser ids
    # surface as dedicated tackle_*_player_id / tackle_*_team_id output
    # columns below. Callers wanting the SPADL-canonical "actor = winner"
    # semantic apply silly_kicks.spadl.use_tackle_winner_as_actor() post-conversion.
    is_tackle = et == "TacklingGame"
    type_ids[is_tackle] = spadlconfig.actiontype_id["tackle"]
    result_ids[is_tackle] = spadlconfig.result_id["success"]
```

**Change 2 — Compute 4 new tackle qualifier passthrough arrays** before the actions DataFrame assembly. After the line `is_play_unrecognized_gk = is_play & (play_gk != "") & ~is_play_known_qualifier` and before `# Assemble main actions DataFrame (1:1 with rows).`:

```python
    # --- ADR-001 qualifier passthrough columns for TacklingGame rows ---
    # Surface DFL tackle_winner / tackle_winner_team / tackle_loser /
    # tackle_loser_team verbatim, NaN on non-tackle rows. The np.where
    # masks out qualifier values from non-tackle rows defensively (in
    # case caller leaves these columns populated on unrelated events).
    tackle_winner_player_arr = np.where(
        is_tackle, _opt("tackle_winner", np.nan).to_numpy(dtype=object), np.nan,
    ).astype(object)
    tackle_winner_team_arr = np.where(
        is_tackle, _opt("tackle_winner_team", np.nan).to_numpy(dtype=object), np.nan,
    ).astype(object)
    tackle_loser_player_arr = np.where(
        is_tackle, _opt("tackle_loser", np.nan).to_numpy(dtype=object), np.nan,
    ).astype(object)
    tackle_loser_team_arr = np.where(
        is_tackle, _opt("tackle_loser_team", np.nan).to_numpy(dtype=object), np.nan,
    ).astype(object)
```

**Change 3 — Add the 4 new columns to the actions DataFrame assembly**. Find the existing `actions = pd.DataFrame({...})` block in `_build_raw_actions`. Add the 4 new keys at the end of the dict:

```python
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["timestamp_seconds"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player_id"].astype("object"),
            "start_x": rows["x"].astype(np.float64),
            "start_y": rows["y"].astype(np.float64),
            "end_x": rows["x"].astype(np.float64),
            "end_y": rows["y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
            "tackle_winner_player_id": tackle_winner_player_arr,
            "tackle_winner_team_id": tackle_winner_team_arr,
            "tackle_loser_player_id": tackle_loser_player_arr,
            "tackle_loser_team_id": tackle_loser_team_arr,
        }
    )
```

**Change 4 — Update `_finalize_output` schema arg in `convert_to_actions`**. Find the line:

```python
    actions = _finalize_output(actions, schema=KLOPPY_SPADL_COLUMNS, extra_columns=extras)
```

Replace with:

```python
    actions = _finalize_output(actions, schema=SPORTEC_SPADL_COLUMNS, extra_columns=extras)
```

**Change 5 — Update the import + the `_empty_raw_actions` helper.** At the top of the file, find:

```python
from .schema import KLOPPY_SPADL_COLUMNS, ConversionReport
```

Replace with:

```python
from .schema import KLOPPY_SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
```

Then update `_empty_raw_actions` to use the new schema:

```python
def _empty_raw_actions(preserve_native: list[str] | None, events: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical SPADL schema plus extras."""
    cols = {col: pd.Series(dtype=dtype) for col, dtype in SPORTEC_SPADL_COLUMNS.items()}
    if preserve_native:
        for col in preserve_native:
            cols[col] = pd.Series(dtype=events[col].dtype if col in events.columns else "object")
    return pd.DataFrame(cols)
```

(Note: the function signature is unchanged; only the schema reference changed inside.)

- [ ] **Step 3.4: Run new tests, verify they pass**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecTackleNoOverride tests/spadl/test_sportec.py::TestSportecTackleWinnerColumns tests/spadl/test_sportec.py::TestSportecTackleLoserColumns tests/spadl/test_sportec.py::TestSportecOutputSchema -v --tb=line 2>&1 | tail -20
```

Expected: all ~17 new tests pass.

- [ ] **Step 3.5: Run all sportec tests, identify breakage in pre-2.0.0 tests**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py -v --tb=line 2>&1 | tail -25
```

Expected: 1 failure — `TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor` at line ~372 (it asserts the now-removed override behavior). Other pre-existing tests should pass — `_df_tackle_winner` fixture data still works (the existing test uses caller-friendly identifiers).

- [ ] **Step 3.6: Delete the pre-2.0.0 `test_tackle_uses_winner_as_actor` test**

Edit `tests/spadl/test_sportec.py`. Find and delete the entire test method (~6 lines):

```python
    def test_tackle_uses_winner_as_actor(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_winner(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["tackle"]
        assert actions["player_id"].iloc[0] == "P-WINNER"
        assert actions["team_id"].iloc[0] == "T-HOME"
```

The test was asserting the override behavior. ADR-001 inverts this: tackle's `player_id` and `team_id` mirror input, NOT winner. The new contract is covered by `TestSportecTackleNoOverride` and `TestSportecTackleWinnerColumns`.

The test's helper `_df_tackle_winner()` (~line 283) is still used by other tests — leave it in place. Audit: it's referenced by the deleted test only. If unused after deletion, also delete the helper:

```bash
uv run grep -n "_df_tackle_winner" tests/spadl/test_sportec.py
```

Expected (after the deletion above): only the helper definition remains, no callers. If so, delete the helper too:

```python
def _df_tackle_winner() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["TacklingGame"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P-LOSER"],
            "team": ["T-AWAY"],
            "x": [50.0],
            "y": [34.0],
            "tackle_winner": ["P-WINNER"],
            "tackle_winner_team": ["T-HOME"],
            "tackle_loser": ["P-LOSER"],
            "tackle_loser_team": ["T-AWAY"],
        }
    )
```

- [ ] **Step 3.7: Re-run all sportec tests, confirm all pass**

```bash
uv run pytest tests/spadl/test_sportec.py -v --tb=line -q 2>&1 | tail -5
```

Expected: all sportec tests pass (~67 baseline + 17 new = ~84 tests).

- [ ] **Step 3.8: Stage Task 3 changes**

```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec.py
```

---

## Task 4: `use_tackle_winner_as_actor` migration helper (TDD)

**Files:**
- Create: `tests/spadl/test_use_tackle_winner_as_actor.py`
- Modify: `silly_kicks/spadl/sportec.py` (add the helper near the bottom of the file)
- Modify: `silly_kicks/spadl/__init__.py` (re-export)

The helper restores pre-2.0.0 SPADL "actor = winner" semantic for sportec consumers whose upstream `team` / `player_id` columns happen to be in the same identifier convention as DFL's `tackle_winner_team` / `tackle_winner` qualifiers. Pure post-conversion enrichment, mirrors the `add_*` helper family pattern.

- [ ] **Step 4.1: Create the failing test file**

Create `tests/spadl/test_use_tackle_winner_as_actor.py`:

```python
"""Tests for ``silly_kicks.spadl.use_tackle_winner_as_actor`` (added in 2.0.0).

Mirrors the PR-S8 ``boundary_metrics`` test discipline. The helper is a
post-conversion enrichment that restores pre-2.0.0 sportec SPADL
"actor = winner" semantic for consumers whose upstream identifier
conventions match DFL's tackle_winner_* qualifier format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import use_tackle_winner_as_actor

_TACKLE_ID = spadlconfig.actiontype_id["tackle"]


def _actions_one_tackle(
    *,
    team_id: object = "home",
    player_id: object = "P-LOSER",
    winner_player_id: object = "P-WINNER",
    winner_team_id: object = "DFL-CLU-X",
    type_id: int = _TACKLE_ID,
) -> pd.DataFrame:
    """Build a one-row sportec-shape SPADL action DataFrame."""
    return pd.DataFrame(
        {
            "team_id": [team_id],
            "player_id": [player_id],
            "tackle_winner_player_id": [winner_player_id],
            "tackle_winner_team_id": [winner_team_id],
            "tackle_loser_player_id": ["P-LOSER"],
            "tackle_loser_team_id": ["DFL-CLU-Y"],
            "type_id": [type_id],
            "extra_col": ["preserved"],
        }
    )


class TestUseTackleWinnerAsActorContract:
    def test_returns_dataframe(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self):
        actions = _actions_one_tackle()
        original_team = actions["team_id"].iloc[0]
        use_tackle_winner_as_actor(actions)
        assert actions["team_id"].iloc[0] == original_team

    def test_missing_team_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["team_id"])
        with pytest.raises(ValueError, match="team_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_player_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["player_id"])
        with pytest.raises(ValueError, match="player_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_tackle_winner_player_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["tackle_winner_player_id"])
        with pytest.raises(ValueError, match="tackle_winner_player_id"):
            use_tackle_winner_as_actor(actions)

    def test_missing_tackle_winner_team_id_column_raises(self):
        actions = _actions_one_tackle().drop(columns=["tackle_winner_team_id"])
        with pytest.raises(ValueError, match="tackle_winner_team_id"):
            use_tackle_winner_as_actor(actions)


class TestUseTackleWinnerAsActorCorrectness:
    def test_overwrites_team_id_from_winner_team_when_non_null(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert result["team_id"].iloc[0] == "DFL-CLU-X"

    def test_overwrites_player_id_from_winner_player_when_non_null(self):
        result = use_tackle_winner_as_actor(_actions_one_tackle())
        assert result["player_id"].iloc[0] == "P-WINNER"

    def test_atomic_overwrite_both_or_neither(self):
        # Two rows: one with full winner cols, one with NaN.
        actions = pd.DataFrame(
            {
                "team_id": ["home", "home"],
                "player_id": ["P-LOSER1", "P-LOSER2"],
                "tackle_winner_player_id": ["P-WIN1", np.nan],
                "tackle_winner_team_id": ["DFL-CLU-A", np.nan],
                "tackle_loser_player_id": ["P-LOSER1", "P-LOSER2"],
                "tackle_loser_team_id": ["DFL-CLU-B", "DFL-CLU-B"],
                "type_id": [_TACKLE_ID, _TACKLE_ID],
            }
        )
        result = use_tackle_winner_as_actor(actions)
        # Row 0: both team + player overwritten.
        assert result["team_id"].iloc[0] == "DFL-CLU-A"
        assert result["player_id"].iloc[0] == "P-WIN1"
        # Row 1: neither overwritten.
        assert result["team_id"].iloc[1] == "home"
        assert result["player_id"].iloc[1] == "P-LOSER2"

    def test_rows_with_nan_winner_left_unchanged(self):
        actions = _actions_one_tackle(winner_player_id=np.nan, winner_team_id=np.nan)
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].iloc[0] == "home"
        assert result["player_id"].iloc[0] == "P-LOSER"

    def test_preserves_all_other_columns(self):
        actions = _actions_one_tackle()
        result = use_tackle_winner_as_actor(actions)
        assert result["extra_col"].iloc[0] == "preserved"
        assert result["type_id"].iloc[0] == _TACKLE_ID

    def test_preserves_tackle_winner_columns_themselves(self):
        # The helper writes from winner cols TO team_id/player_id, but
        # leaves the winner cols themselves intact for downstream inspection.
        actions = _actions_one_tackle()
        result = use_tackle_winner_as_actor(actions)
        assert result["tackle_winner_player_id"].iloc[0] == "P-WINNER"
        assert result["tackle_winner_team_id"].iloc[0] == "DFL-CLU-X"

    def test_non_tackle_rows_unchanged(self):
        # Even if a non-tackle row had non-null winner cols (which shouldn't
        # happen in practice), the helper acts only when winner cols are
        # non-null — so it would write. But the converter guarantees winner
        # cols are NaN on non-tackle rows. This test asserts the helper's
        # NaN-aware behavior independent of type_id.
        actions = _actions_one_tackle(type_id=spadlconfig.actiontype_id["pass"])
        result = use_tackle_winner_as_actor(actions)
        # Helper acts purely on tackle_winner_*_id non-null status, not
        # type_id. With non-null winner cols, the swap fires regardless.
        assert result["team_id"].iloc[0] == "DFL-CLU-X"


class TestUseTackleWinnerAsActorDegenerate:
    def test_empty_dataframe_returns_empty(self):
        actions = pd.DataFrame(
            {
                "team_id": pd.Series([], dtype="object"),
                "player_id": pd.Series([], dtype="object"),
                "tackle_winner_player_id": pd.Series([], dtype="object"),
                "tackle_winner_team_id": pd.Series([], dtype="object"),
            }
        )
        result = use_tackle_winner_as_actor(actions)
        assert len(result) == 0
        assert list(result.columns) == ["team_id", "player_id", "tackle_winner_player_id", "tackle_winner_team_id"]

    def test_all_nan_winner_returns_identity(self):
        actions = pd.DataFrame(
            {
                "team_id": ["home", "away"],
                "player_id": ["P1", "P2"],
                "tackle_winner_player_id": [np.nan, np.nan],
                "tackle_winner_team_id": [np.nan, np.nan],
            }
        )
        result = use_tackle_winner_as_actor(actions)
        assert result["team_id"].tolist() == ["home", "away"]
        assert result["player_id"].tolist() == ["P1", "P2"]

    def test_preserves_preserve_native_columns(self):
        actions = _actions_one_tackle()
        actions["my_preserved"] = ["xyz"]
        result = use_tackle_winner_as_actor(actions)
        assert "my_preserved" in result.columns
        assert result["my_preserved"].iloc[0] == "xyz"
```

- [ ] **Step 4.2: Run tests, verify they fail with ImportError**

Run:
```bash
uv run pytest tests/spadl/test_use_tackle_winner_as_actor.py -v --tb=short 2>&1 | tail -10
```

Expected: collection error — `ImportError: cannot import name 'use_tackle_winner_as_actor' from 'silly_kicks.spadl'`.

- [ ] **Step 4.3: Implement the helper in `silly_kicks/spadl/sportec.py`**

Append to the bottom of `silly_kicks/spadl/sportec.py` (after `_synthesize_gk_distribution_actions`):

```python
_USE_TACKLE_WINNER_AS_ACTOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "team_id",
    "player_id",
    "tackle_winner_player_id",
    "tackle_winner_team_id",
)


def use_tackle_winner_as_actor(actions: pd.DataFrame) -> pd.DataFrame:
    """Re-attribute SPADL tackle rows to the winning duelist (pre-2.0.0 semantic).

    Sportec converter output emits ``team_id`` / ``player_id`` mirroring the
    caller's input verbatim per ADR-001. For tackle rows where DFL recorded
    a winner via ``tackle_winner`` / ``tackle_winner_team`` qualifiers, the
    winner ids are surfaced as ``tackle_winner_player_id`` /
    ``tackle_winner_team_id`` columns alongside.

    This helper applies the SPADL-canonical "tackle.actor = winner" semantic
    by overwriting ``player_id`` / ``team_id`` from the winner columns where
    those columns are non-null. Pre-2.0.0 sportec consumers can call it
    post-conversion to restore the prior behavior — but ONLY if their
    upstream-supplied ``team`` column is already in the same identifier
    convention as DFL's ``tackle_winner_team`` qualifier (raw ``DFL-CLU-...``).
    Mismatched conventions are the bug ADR-001 fixes; this helper does NOT
    resolve identifier formats.

    Parameters
    ----------
    actions : pd.DataFrame
        Sportec converter output. Must contain ``team_id``, ``player_id``,
        ``tackle_winner_player_id``, ``tackle_winner_team_id``.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with ``team_id`` and ``player_id`` overwritten
        from the winner columns on rows where those columns are non-null.
        All other columns unchanged.

    Raises
    ------
    ValueError
        If a required column is missing.

    Examples
    --------
    Restore SPADL "actor = winner" semantic on sportec output::

        from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
        actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
        actions = use_tackle_winner_as_actor(actions)
    """
    missing = [c for c in _USE_TACKLE_WINNER_AS_ACTOR_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"use_tackle_winner_as_actor: actions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actions.columns)}"
        )

    result = actions.copy()
    if len(result) == 0:
        return result

    winner_player = result["tackle_winner_player_id"]
    winner_team = result["tackle_winner_team_id"]
    # Atomic per-row: only overwrite when BOTH winner columns are non-null
    # (matches DFL's qualifier pairing — both populated together or both
    # absent). Rows with mismatched NaN status (a possibility only with
    # pathological hand-crafted fixtures) are conservatively left unchanged.
    overwrite_mask = winner_player.notna() & winner_team.notna()
    if overwrite_mask.any():
        result.loc[overwrite_mask, "player_id"] = winner_player[overwrite_mask].to_numpy()
        result.loc[overwrite_mask, "team_id"] = winner_team[overwrite_mask].to_numpy()

    return result
```

- [ ] **Step 4.4: Re-export from `silly_kicks/spadl/__init__.py`**

Edit `silly_kicks/spadl/__init__.py`. Update `__all__` to add `"use_tackle_winner_as_actor"` alphabetically:

```python
__all__ = [
    # ... existing entries ...
    "use_tackle_winner_as_actor",
    # ... rest ...
]
```

Update the import block at the bottom — add to the existing `from .sportec import` block (or create one if not present):

```python
from .sportec import use_tackle_winner_as_actor
```

(Note: the `from . import config, opta, statsbomb, wyscout` line imports modules; that's separate. The new import is for a specific function from sportec.)

- [ ] **Step 4.5: Run tests, verify all PASS**

Run:
```bash
uv run pytest tests/spadl/test_use_tackle_winner_as_actor.py -v --tb=short 2>&1 | tail -20
```

Expected: all 15 tests pass.

- [ ] **Step 4.6: Stage Task 4 changes**

```bash
git add silly_kicks/spadl/sportec.py silly_kicks/spadl/__init__.py tests/spadl/test_use_tackle_winner_as_actor.py
```

---

## Task 5: Cross-provider parity regression gate

**Files:**
- Modify: `tests/spadl/test_cross_provider_parity.py` (new parametrized test class)

This task adds the parametrized regression gate that locks ADR-001 across all 5 DataFrame converters. Would have caught the 1.7.0 sportec bug if it had existed.

- [ ] **Step 5.1: Add `test_team_id_mirrors_input_team` to `tests/spadl/test_cross_provider_parity.py`**

Append to `tests/spadl/test_cross_provider_parity.py` (after the existing `test_converter_returns_canonical_spadl_columns` parametrized test):

```python
# ---------------------------------------------------------------------------
# ADR-001: caller's team_id mirrors input — never overridden from qualifiers
# (2.0.0; locks the converter contract per-provider as a regression gate)
# ---------------------------------------------------------------------------


def _input_team_values_for(provider: str) -> set[str]:
    """Capture the unique team values from each provider's input fixture.

    Returns the set of team identifiers the caller passed in. ADR-001 says
    output team_id values must be a subset of this set.
    """
    if provider == "sportec":
        parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)
        return set(events["team"].dropna().astype(str).tolist())
    if provider == "metrica":
        parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)
        return set(events["team"].dropna().astype(str).tolist())
    if provider == "statsbomb":
        fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / "7298.json"
        with open(fixture_path, encoding="utf-8") as f:
            events_raw = json.load(f)
        return {str((e.get("team") or {}).get("id")) for e in events_raw if (e.get("team") or {}).get("id") is not None}
    if provider == "opta":
        # Synthetic fixture (see _load_opta_fixture); team_ids are 100, 200.
        return {"100", "200"}
    if provider == "wyscout":
        # Synthetic fixture (see _load_wyscout_fixture); team_id is 100.
        return {"100"}
    raise ValueError(f"unknown provider {provider!r}")


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_team_id_mirrors_input_team(provider: str):
    """ADR-001 contract: every converter's output team_id values must be a
    subset of the input team values. Locks the no-override contract
    per-provider as a regression gate going forward.

    Pre-2.0.0 sportec failed this gate: tackle rows had raw DFL CLU ids
    (e.g., 'DFL-CLU-000005') in team_id when the caller passed 'home' /
    'away' in the input team column.
    """
    actions = _PROVIDER_LOADERS[provider]()
    expected = _input_team_values_for(provider)

    # Cast output team_id to str for comparison (some providers use int,
    # but membership comparison via str is robust across dtypes).
    actual = set(actions["team_id"].dropna().astype(str).tolist())

    leaked = actual - expected
    assert not leaked, (
        f"Provider {provider!r} emitted team_id values not present in input: {sorted(leaked)}. "
        f"Expected subset of input team values: {sorted(expected)[:10]}{'...' if len(expected) > 10 else ''}. "
        f"This is the ADR-001 violation pattern (cf. silly-kicks pre-2.0.0 sportec tackle override)."
    )
```

- [ ] **Step 5.2: Update `test_converter_returns_canonical_spadl_columns` for sportec's new schema**

Find the existing `test_converter_returns_canonical_spadl_columns` test in `tests/spadl/test_cross_provider_parity.py`. The pre-2.0.0 version asserts only `"type_id" in actions.columns`. This task makes the test schema-aware for sportec specifically.

Replace:

```python
@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_returns_canonical_spadl_columns(provider: str):
    """Sanity check: every converter returns a DataFrame with type_id present."""
    actions = _PROVIDER_LOADERS[provider]()
    assert "type_id" in actions.columns
    assert len(actions) > 0
```

With:

```python
@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_returns_canonical_spadl_columns(provider: str):
    """Sanity check: every converter returns a DataFrame with type_id present.
    Sportec output uses SPORTEC_SPADL_COLUMNS (ADR-001); other providers use
    KLOPPY_SPADL_COLUMNS or SPADL_COLUMNS.
    """
    actions = _PROVIDER_LOADERS[provider]()
    assert "type_id" in actions.columns
    assert len(actions) > 0

    if provider == "sportec":
        # Post-2.0.0: sportec output includes the 4 tackle_*_*_id columns.
        for col in ("tackle_winner_player_id", "tackle_winner_team_id", "tackle_loser_player_id", "tackle_loser_team_id"):
            assert col in actions.columns, f"sportec output missing ADR-001 column {col!r}"
```

- [ ] **Step 5.3: Run cross-provider parity tests**

Run:
```bash
uv run pytest tests/spadl/test_cross_provider_parity.py -v --tb=short 2>&1 | tail -20
```

Expected: all 15+ tests pass (10 existing + 5 new `test_team_id_mirrors_input_team`).

- [ ] **Step 5.4: Stage Task 5 changes**

```bash
git add tests/spadl/test_cross_provider_parity.py
```

---

## Task 6: e2e on the IDSSE production fixture

**Files:**
- Modify: `tests/spadl/test_cross_provider_parity.py` (new test class `TestSportecAdrContractOnProductionFixture`)

This task runs ADR-001 verification against the vendored IDSSE production fixture (`tests/datasets/idsse/sample_match.parquet`, 308 rows, vendored 1.10.0). Confirms the fix works on production-shape data, not just synthetic fixtures.

- [ ] **Step 6.1: Add the e2e test class**

Append to `tests/spadl/test_cross_provider_parity.py`:

```python
# ---------------------------------------------------------------------------
# ADR-001 e2e on the IDSSE production fixture
# (Verifies the contract works on production-shape data — 308 rows from
# soccer_analytics.bronze.idsse_events match J03WMX, vendored 1.10.0)
# ---------------------------------------------------------------------------


class TestSportecAdrContractOnProductionFixture:
    """ADR-001 verification on production-shape IDSSE data."""

    @staticmethod
    def _load_actions():
        from silly_kicks.spadl import sportec

        parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)

        # Pretend the caller normalized team to home/away labels (the
        # luxury-lakehouse adapter pattern) — overwrite events["team"]
        # using the first non-null team value as "home" and others as "away".
        first_team = events["team"].dropna().iloc[0]
        events["team"] = events["team"].apply(
            lambda t: "home" if t == first_team else ("away" if pd.notna(t) else t)
        )

        actions, _ = sportec.convert_to_actions(events, home_team_id="home")
        return actions, events

    def test_no_dfl_clu_strings_leak_into_team_id(self):
        """The PR-LL2 bug: caller passed 'home'/'away' but team_id rows had
        raw 'DFL-CLU-...' strings. ADR-001 makes this impossible."""
        actions, _ = self._load_actions()
        leaked = actions["team_id"].dropna().astype(str)
        dfl_leakage = leaked[leaked.str.startswith("DFL-CLU-")]
        assert len(dfl_leakage) == 0, (
            f"DFL-CLU-... strings leaked into team_id ({len(dfl_leakage)} rows). "
            f"This is the ADR-001 violation pattern. Sample leaks: {dfl_leakage.head(3).tolist()}"
        )

    def test_team_id_only_contains_home_or_away(self):
        actions, _ = self._load_actions()
        unique_team_ids = set(actions["team_id"].dropna().astype(str).tolist())
        # The caller normalized to {home, away}. Synthetic dribble rows
        # inherit prior action's team_id (which is also home/away).
        assert unique_team_ids <= {"home", "away"}, (
            f"team_id contains values outside {{home, away}}: {sorted(unique_team_ids)}"
        )

    def test_tackle_winner_team_id_populated_for_qualifier_rows(self):
        """Some tackle rows in the fixture have tackle_winner_team qualifier
        populated (DFL-CLU-... values). The new column surfaces those verbatim."""
        actions, events = self._load_actions()

        # Count input rows where the qualifier was set.
        if "tackle_winner_team" not in events.columns:
            pytest.skip("IDSSE fixture lacks tackle_winner_team column entirely")
        input_winner_rows = events[events["tackle_winner_team"].notna()]
        if len(input_winner_rows) == 0:
            pytest.skip("IDSSE fixture has no tackle rows with tackle_winner_team qualifier")

        # Output: tackle_winner_team_id should be populated on (at least
        # some of) those rows.
        output_winner_rows = actions[actions["tackle_winner_team_id"].notna()]
        assert len(output_winner_rows) > 0, (
            "Sportec output has zero rows with tackle_winner_team_id populated, "
            "but input has rows with the tackle_winner_team qualifier."
        )

    def test_use_tackle_winner_as_actor_round_trips(self):
        """The migration helper restores pre-2.0.0 'actor = winner' semantic."""
        from silly_kicks.spadl import use_tackle_winner_as_actor

        actions, _ = self._load_actions()

        n_winner_rows = int(actions["tackle_winner_team_id"].notna().sum())
        if n_winner_rows == 0:
            pytest.skip("IDSSE fixture has no rows with tackle_winner_team_id; helper has nothing to swap")

        rotated = use_tackle_winner_as_actor(actions)

        # On rows with non-null winner cols, team_id is now the winner team
        # id (not 'home'/'away').
        rotated_winner_rows = rotated[actions["tackle_winner_team_id"].notna()]
        assert (rotated_winner_rows["team_id"] == rotated_winner_rows["tackle_winner_team_id"]).all(), (
            "use_tackle_winner_as_actor failed to overwrite team_id on rows with non-null winner cols"
        )

    def test_keeper_coverage_preserved_from_1_10_0(self):
        """ADR-001 changes don't regress the 1.10.0 keeper coverage fix."""
        from silly_kicks.spadl import coverage_metrics

        actions, _ = self._load_actions()
        m = coverage_metrics(actions=actions, expected_action_types={
            "keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up",
        })
        keeper_total = sum(m["counts"].get(t, 0) for t in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"))
        # IDSSE fixture has 7 throwOut + 1 punt qualifier rows from PR-S10;
        # each produces a keeper_pick_up. Plus possibly save/claim/punch rows.
        # Assert at least 1 (the 1.10.0 floor); typical is ~8.
        assert keeper_total > 0, (
            f"1.10.0 keeper coverage regressed under ADR-001 changes. "
            f"counts={m['counts']}"
        )
```

- [ ] **Step 6.2: Run the e2e tests**

Run:
```bash
uv run pytest tests/spadl/test_cross_provider_parity.py::TestSportecAdrContractOnProductionFixture -v --tb=short 2>&1 | tail -15
```

Expected: all 5 e2e tests pass.

- [ ] **Step 6.3: Stage Task 6 changes**

```bash
git add tests/spadl/test_cross_provider_parity.py
```

(Already staged from Task 5; this confirms the file is up to date.)

---

## Task 7: sportec module docstring update

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (module docstring)

The 1.10.0 sportec module docstring documents the DFL event_type vocabulary + GK qualifier vocabulary + bug history. ADR-001 adds a new section about the identifier-conventions contract.

- [ ] **Step 7.1: Append the ADR-001 contract section to the sportec module docstring**

Edit the top of `silly_kicks/spadl/sportec.py`. Find the closing `"""` of the existing module docstring (after the "Bug history" section, before `import warnings`). Insert before the closing triple-quotes:

```rst

ADR-001: identifier conventions are sacred
-------------------------------------------

The converter never overrides ``team_id`` / ``player_id`` from DFL
qualifier columns. Caller-supplied ``team`` / ``player_id`` values
mirror verbatim into the output. DFL ``tackle_winner`` /
``tackle_winner_team`` / ``tackle_loser`` / ``tackle_loser_team``
qualifier values surface via dedicated output columns:

================== ============================ ============================
Output column      DFL qualifier source         NaN when
================== ============================ ============================
``tackle_winner_player_id`` ``tackle_winner``    qualifier absent OR non-tackle row
``tackle_winner_team_id``   ``tackle_winner_team``  qualifier absent OR non-tackle row
``tackle_loser_player_id``  ``tackle_loser``     qualifier absent OR non-tackle row
``tackle_loser_team_id``    ``tackle_loser_team``  qualifier absent OR non-tackle row
================== ============================ ============================

The output schema is :data:`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`
(extends :data:`silly_kicks.spadl.KLOPPY_SPADL_COLUMNS` with the 4 tackle
columns).

Pre-2.0.0 callers relying on the SPADL "tackle.actor = winner" semantic —
specifically those whose upstream-supplied ``team`` / ``player_id``
columns happened to be in the same identifier convention as DFL's
``tackle_winner_team`` / ``tackle_winner`` qualifiers (raw
``DFL-CLU-...`` / ``DFL-OBJ-...``) — restore the prior behavior with the
:func:`silly_kicks.spadl.use_tackle_winner_as_actor` helper:

.. code-block:: python

    from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
    actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
    actions = use_tackle_winner_as_actor(actions)

See ``docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md``
for the contract rationale and audit findings.
```

- [ ] **Step 7.2: Stage Task 7 changes**

```bash
git add silly_kicks/spadl/sportec.py
```

---

## Task 8: Release artifacts — version, CHANGELOG, TODO, C4

**Files:**
- Modify: `pyproject.toml` (version `1.10.0` → `2.0.0`)
- Modify: `CHANGELOG.md` (add `## [2.0.0]` entry)
- Modify: `TODO.md` (close PR-S11, bump PR-S12)
- Modify: `docs/c4/architecture.dsl` (mention `SPORTEC_SPADL_COLUMNS` + helper)

- [ ] **Step 8.1: Bump version in pyproject.toml**

Edit `pyproject.toml`:

```diff
-version = "1.10.0"
+version = "2.0.0"
```

- [ ] **Step 8.2: Add CHANGELOG entry**

Insert at the top of `CHANGELOG.md` (after the header lines and before the existing `## [1.10.0]` entry):

```markdown
## [2.0.0] — 2026-04-29

### ⚠️ Breaking

- **`silly_kicks.spadl.sportec.convert_to_actions` no longer overrides
  `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team`
  qualifiers.** Per ADR-001
  (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`),
  the SPADL converter contract is "caller's identifier conventions are
  sacred — never overridden from qualifiers." Caller-supplied `team` /
  `player_id` values mirror verbatim into the output. Pre-2.0.0 behavior
  silently rewrote ~56% of tackle rows on consumers using a
  caller-normalized `team` convention (see luxury-lakehouse PR-LL2
  close-out report).
- **Sportec output schema changes from `KLOPPY_SPADL_COLUMNS` to
  `SPORTEC_SPADL_COLUMNS`** — 14 + 4 = 18 columns. The 4 new columns
  surface DFL qualifier values: `tackle_winner_player_id`,
  `tackle_winner_team_id`, `tackle_loser_player_id`,
  `tackle_loser_team_id`. NaN on non-tackle rows; NaN when the qualifier
  is absent. Sportec consumers asserting against `KLOPPY_SPADL_COLUMNS`
  must switch to `SPORTEC_SPADL_COLUMNS`.

### Migration

If your pre-2.0.0 sportec consumer relied on the tackle-winner override
AND your upstream `team` / `player_id` columns are in the same
identifier convention as DFL's `tackle_winner_team` / `tackle_winner`
qualifiers (raw `DFL-CLU-...` / `DFL-OBJ-...`), call the new helper
post-conversion:

```python
from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
actions = use_tackle_winner_as_actor(actions)
```

If your `team` / `player_id` columns use any other convention, the
post-1.10.0 behavior already preserved your conventions correctly — no
migration needed; the bug fix is automatic on upgrade.

### Added

- **First silly-kicks ADR.** `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`
  + `docs/superpowers/adrs/ADR-TEMPLATE.md` (vendored verbatim from
  luxury-lakehouse) establish the silly-kicks ADR pattern. Future
  decisions that add an exception to project-wide conventions, change
  schema ownership, or hardcode a workaround for a platform constraint
  get an ADR.
- **`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`** schema constant (18-key
  dict) — extends `KLOPPY_SPADL_COLUMNS` with the 4 tackle qualifier
  passthrough columns. Re-exported from `silly_kicks.spadl`.
- **`silly_kicks.spadl.use_tackle_winner_as_actor(actions) -> pd.DataFrame`**
  — pure post-conversion enrichment that restores pre-2.0.0 sportec
  SPADL "actor = winner" semantic for callers whose upstream identifier
  convention matches DFL's qualifier format. Raises `ValueError` early
  on missing required columns. Mirrors the `add_*` helper family pattern.
- **Cross-provider parity regression gate**
  (`tests/spadl/test_cross_provider_parity.py::test_team_id_mirrors_input_team`).
  Parametrized over all 5 DataFrame converters; asserts each output's
  `team_id` values are a subset of the input `team` values. Locks the
  ADR-001 contract per-provider going forward; would have caught the
  1.7.0 sportec bug.
- **e2e on the IDSSE production fixture**
  (`TestSportecAdrContractOnProductionFixture`, 5 tests). Verifies the
  contract works on production-shape data: caller's labels survive
  through the converter; the 4 new columns are populated for qualifier
  rows; the migration helper round-trips correctly; 1.10.0 keeper
  coverage is preserved.

### Changed

- **CLAUDE.md "Key conventions" section** gains one rule citing ADR-001:
  "Converter identifier conventions are sacred. SPADL DataFrame
  converters never override the caller's `team_id` / `player_id`
  columns from provider-specific qualifiers..."
- **Sportec module docstring** documents the 4 tackle qualifier
  passthrough columns + the `SPORTEC_SPADL_COLUMNS` schema + the
  migration helper. References ADR-001.

### Removed

- **`silly_kicks.spadl.sportec` tackle override block** at the previous
  `sportec.py:559-565`. The 6-line override that silently rewrote
  `team_id` / `player_id` from raw DFL qualifier values is gone.
- **`tests/spadl/test_sportec.py::TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor`**
  — was asserting the now-removed override. Covered by the new
  `TestSportecTackleNoOverride` + `TestSportecTackleWinnerColumns`
  classes.

### Audit findings

Manual cross-converter review (this cycle) confirmed sportec.tackle
was the unique violator of the ADR-001 contract:

| Converter | Override `player_id` / `team_id`? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | YES (removed) | The bug. |
| `silly_kicks.spadl.metrica` | NO | 1.10.0 GK routing only changes `type_id` / `bodypart_id`. |
| `silly_kicks.spadl.wyscout` | NO | 1.0.0 aerial-duel reclassification only changes `type_id` / `subtype_id`. |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.kloppy` | NO | Gateway path. |

The 2.0.0 change is surgical (one converter), but the parity gate locks
the contract for all future converter additions.

### Notes

- silly-kicks 2.0.0 is the project's first semver-major release. The
  library is ~3 weeks old (0.1.0 shipped 2026-04-06); major versions
  aren't precious — bumping locks the contract before more downstream
  consumers pin against pre-2.0.0 behavior.
- luxury-lakehouse can bump `silly-kicks>=2.0.0,<3.0` and (optionally)
  drop their `_team_label_to_dfl_id` shim from PR-LL2 close-out, OR
  keep it as a documented winner-attribution post-conversion pattern.

```

- [ ] **Step 8.3: Update TODO.md**

Edit `TODO.md`. Find the `## Open PRs` table. Replace the existing PR-S11 row:

```markdown
| PR-S11 | Medium-Large | `add_possessions` algorithmic precision improvement | ... |
```

With:

```markdown
| PR-S12 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. The 64-match WorldCup HDF5 from PR-S9 is available for parameter sweeping. Re-numbered from PR-S11 (which became the converter identifier conventions / sportec tackle override removal work, shipped in silly-kicks 2.0.0). |
```

- [ ] **Step 8.4: Update C4 architecture.dsl**

Edit `docs/c4/architecture.dsl`. Find the spadl container description (in the `model { sillyKicks = softwareSystem ... { spadl = container ... } }` block):

```diff
-            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 6 dedicated DataFrame converters with preserve_native + goalkeeper_ids passthrough (StatsBomb, Opta, Wyscout, Sportec, Metrica) plus a kloppy gateway covering the same providers, output coords clamped to [0, 105] x [0, 68] AND unified to canonical 'all-actions-LTR' SPADL orientation across all paths, ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context); boundary_metrics + coverage_metrics utilities for validating add_possessions output and per-action-type coverage" "Python" "Library"
+            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 6 dedicated DataFrame converters with preserve_native + goalkeeper_ids passthrough (StatsBomb, Opta, Wyscout, Sportec, Metrica) plus a kloppy gateway covering the same providers, output coords clamped to [0, 105] x [0, 68] AND unified to canonical 'all-actions-LTR' SPADL orientation across all paths, ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context, use_tackle_winner_as_actor); boundary_metrics + coverage_metrics utilities for validating add_possessions output and per-action-type coverage; ADR-001 caller-conventions contract — Sportec output uses SPORTEC_SPADL_COLUMNS (KLOPPY_SPADL_COLUMNS + 4 tackle qualifier passthrough columns)" "Python" "Library"
```

(Note: the `architecture.html` regeneration happens in Task 11 via `/final-review`.)

- [ ] **Step 8.5: Stage Task 8 changes**

```bash
git add pyproject.toml CHANGELOG.md TODO.md docs/c4/architecture.dsl
git status
```

---

## Task 9: Verification gates

**Files:** none (run-only)

Run the same gates that CI will run, against the staged + working-tree state, before the single commit.

- [ ] **Step 9.1: Re-pin CI tools (defensive)**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: succeeds (already installed in Task 0; this is a defensive re-check).

- [ ] **Step 9.2: Lint clean**

```bash
uv run ruff check silly_kicks/ tests/ scripts/
```

Expected: zero errors.

If any errors surface (likely candidates: N802 false positive on test names with uppercase letters; S105 false positive on hard-coded strings comparing to pass_type-named variables), add per-file-ignores to `pyproject.toml` rather than disabling rules globally.

- [ ] **Step 9.3: Format check**

```bash
uv run ruff format --check silly_kicks/ tests/ scripts/
```

If anything fails, run `uv run ruff format silly_kicks/ tests/ scripts/` to auto-fix, then re-stage the affected files.

- [ ] **Step 9.4: Pyright type check**

```bash
uv run pyright silly_kicks/ 2>&1 | tail -5
```

Expected: zero errors. If any surface (likely candidates: numpy `np.where` narrowing, `pd.Series` dtype object narrowing on the new tackle columns), fix inline. Use the existing 1.10.0 patterns:
- `pd.Series(suffix, index=src.index, dtype="object")` for explicitly-typed object Series
- Narrow `.to_numpy(dtype=object)` for arrays that need object dtype

- [ ] **Step 9.5: Full test suite**

```bash
uv run pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -10
```

Expected: ~660 passed (627 baseline from 1.10.0 + ~32 new), 4 skipped (pre-existing). Zero failures.

If any tests fail at this stage, STOP and investigate.

- [ ] **Step 9.6: Optional — run e2e tests too**

```bash
uv run pytest tests/ -v --tb=short -q 2>&1 | tail -10
```

Expected: same as Step 9.5 plus the WorldCup-2018 prediction tests (which run in the regular suite since 1.9.0). If anything regresses, STOP.

- [ ] **Step 9.7: Stage any auto-fixed format changes**

```bash
git status
```

If `ruff format` modified any files in Step 9.3, re-stage them:

```bash
git add silly_kicks/ tests/ scripts/
```

---

## Task 10: /final-review + 5 user-gated steps (commit, push, PR, merge, tag)

**Files:** the spec + plan + ADR-001 + ADR-TEMPLATE + CLAUDE.md amendment + all preceding changes — bundled into ONE commit.

Per `feedback_commit_policy` memory and the narrowed hook from PR-S8 (only `git commit`, `git push --force`, `git reset --hard`, `git rebase` are sentinel-gated; push/PR/merge/tag are chat-only). User must explicitly approve at each gate.

- [ ] **Step 10.1: Stage the spec + this plan into the same commit**

```bash
git add docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md
git status
```

Expected: ~14 files staged (per the file structure at the top of this plan). Verify the list looks complete:

- `docs/superpowers/adrs/ADR-TEMPLATE.md` (new)
- `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` (new)
- `CLAUDE.md`
- `silly_kicks/spadl/schema.py`
- `silly_kicks/spadl/sportec.py`
- `silly_kicks/spadl/__init__.py`
- `tests/spadl/test_sportec.py`
- `tests/spadl/test_use_tackle_winner_as_actor.py` (new)
- `tests/spadl/test_cross_provider_parity.py`
- `pyproject.toml`
- `CHANGELOG.md`
- `TODO.md`
- `docs/c4/architecture.dsl`
- `docs/c4/architecture.html` (regenerated by /final-review)
- `docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md` (existing untracked from spec phase)
- `docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md` (this file)

- [ ] **Step 10.2: Run `/final-review` skill (regenerates C4 HTML, runs all gates)**

Invoke the `mad-scientist-skills:final-review` skill via the Skill tool. The skill regenerates the C4 architecture HTML from the .dsl, runs ruff / pyright / pytest, and reviews documentation consistency. It will surface any remaining gaps.

If `/final-review` modifies any files (likely just `docs/c4/architecture.html`), re-stage them:

```bash
git add docs/c4/architecture.html
```

Address any other findings from `/final-review` inline.

- [ ] **Step 10.3: Present summary to user; await explicit "approved" + sentinel touch**

Present to the user (in chat):

```
✅ All 11 task gates green:
- ADR-TEMPLATE.md + ADR-001 + CLAUDE.md amendment
- SPORTEC_SPADL_COLUMNS schema constant
- sportec tackle override removed (the ADR-001 fix)
- 4 new tackle_*_*_id columns populated from DFL qualifiers
- use_tackle_winner_as_actor migration helper + 15 tests
- Cross-provider parity regression gate (5 providers)
- e2e on IDSSE production fixture (5 tests)
- Sportec module docstring updated
- pyproject 1.10.0 → 2.0.0
- CHANGELOG with ⚠️ Breaking + Migration sections
- TODO PR-S12 bump
- /final-review pass: ruff + format + pyright + pytest all green
- C4 architecture diagram regenerated

Ready for the single commit. Per silly-kicks policy:
1. Run `!touch ~/.claude-git-approval` to release the sentinel
2. Reply "approved" to authorize the commit
```

WAIT for explicit user approval + sentinel touch.

- [ ] **Step 10.4: Single commit on user approval (sentinel-gated)**

After user approval + sentinel:

```bash
git commit -m "$(cat <<'EOF'
feat(spadl)!: ADR-001 -- caller's team_id / player_id are sacred -- silly-kicks 2.0.0

BREAKING CHANGE: silly_kicks.spadl.sportec.convert_to_actions no longer
overrides team_id / player_id from DFL tackle_winner / tackle_winner_team
qualifiers. Per ADR-001, caller-supplied team / player_id values mirror
verbatim into the output. DFL qualifier values surface via 4 new dedicated
columns (tackle_winner_player_id, tackle_winner_team_id, tackle_loser_player_id,
tackle_loser_team_id). Sportec output schema changes from KLOPPY_SPADL_COLUMNS
to SPORTEC_SPADL_COLUMNS (14 -> 18 columns).

Triggered by luxury-lakehouse PR-LL2 close-out report (2026-04-29):
1412 / 2522 = 56% of IDSSE tackle rows had their caller-supplied
home/away team labels silently rewritten to raw DFL CLU ids. The
override emitted DFL identifiers regardless of the caller's team
convention.

Migration: pre-2.0.0 sportec consumers relying on the tackle-winner
override AND whose upstream team / player_id columns happened to be in
the same identifier convention as DFL's tackle_winner_team / tackle_winner
qualifiers (raw DFL-CLU- / DFL-OBJ-) call:

    from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
    actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
    actions = use_tackle_winner_as_actor(actions)

Other callers' bug-fix is automatic on upgrade.

ADR pattern adopted (vendored ADR-TEMPLATE from luxury-lakehouse).
Cross-converter audit confirms sportec.tackle is the unique violator;
metrica / wyscout / statsbomb / opta / kloppy already honor the contract.
Cross-provider parity regression gate added — would have caught the
1.7.0 bug.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run `git status` after to verify the commit succeeded.

- [ ] **Step 10.5: User approval → push -u**

WAIT for chat approval. Then:

```bash
git push -u origin feat/converter-identifier-conventions
```

- [ ] **Step 10.6: User approval → gh pr create**

WAIT for chat approval. Then:

```bash
gh pr create --title "feat(spadl)!: ADR-001 -- caller's team_id / player_id are sacred -- silly-kicks 2.0.0" --body "$(cat <<'EOF'
## Summary

silly-kicks 2.0.0: first semver-major release. Locks in the SPADL converter contract per ADR-001 (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`): "caller's `team_id` / `player_id` are sacred — never overridden from provider-specific qualifiers."

**Triggered by:** luxury-lakehouse PR-LL2 close-out report (2026-04-29). The pre-2.0.0 sportec converter overrode `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team` qualifiers, silently rewriting 1412 / 2522 = 56% of IDSSE tackle rows.

**Changes:**
- ⚠️ Breaking: sportec converter no longer overrides team_id / player_id; output schema changes from KLOPPY_SPADL_COLUMNS to SPORTEC_SPADL_COLUMNS.
- 4 new sportec output columns: tackle_winner_player_id, tackle_winner_team_id, tackle_loser_player_id, tackle_loser_team_id (verbatim from DFL qualifiers; NaN otherwise).
- New `silly_kicks.spadl.use_tackle_winner_as_actor(actions)` migration helper for callers wanting the pre-2.0.0 SPADL "actor = winner" semantic.
- New ADR pattern (ADR-001 + ADR-TEMPLATE vendored from luxury-lakehouse).
- Cross-provider parity regression gate (would have caught the 1.7.0 bug).
- e2e on the IDSSE production fixture (vendored 1.10.0).

ADR-001 audit confirms sportec.tackle was the unique violator; metrica / wyscout / statsbomb / opta / kloppy already honor the contract.

Spec: `docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md`
Plan: `docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md`

## Test plan

- [x] ADR-001 + ADR-TEMPLATE + CLAUDE.md amendment landed
- [x] SPORTEC_SPADL_COLUMNS schema constant + re-export
- [x] Sportec tackle override removed; 4 new columns populated from DFL qualifiers
- [x] use_tackle_winner_as_actor helper + 15 unit tests
- [x] Cross-provider parity regression gate (5 providers)
- [x] e2e on IDSSE production fixture (5 tests)
- [x] All pre-existing tests pass (660 total, 0 failures)
- [ ] CI matrix (ubuntu 3.10/3.11/3.12 + windows 3.12) all green before merge
- [ ] luxury-lakehouse PR (separate session): bump silly-kicks>=2.0.0,<3.0; optionally drop the PR-LL2 _team_label_to_dfl_id shim

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Capture the PR URL and present it to the user.

- [ ] **Step 10.7: Wait for CI green → user approval → merge**

Watch CI status:

```bash
gh pr checks --watch
```

When green, WAIT for user approval. Then:

```bash
gh pr merge --admin --squash --delete-branch
```

- [ ] **Step 10.8: User approval → tag v2.0.0 → PyPI auto-publish**

WAIT for user approval. Then on `main`:

```bash
git checkout main
git pull origin main
git tag v2.0.0
git push origin v2.0.0
```

The PyPI auto-publish workflow fires on tag push.

- [ ] **Step 10.9: Verify PyPI publish + final state**

After ~2-5 min:

```bash
gh run list --workflow=publish.yml --limit 3
```

Or directly:

```bash
curl -s "https://pypi.org/pypi/silly-kicks/2.0.0/json" | python -c "import sys, json; d = json.load(sys.stdin); print('version:', d.get('info', {}).get('version'), 'released:', d.get('urls', [{}])[0].get('upload_time'))"
```

Expected: `version: 2.0.0 released: 2026-04-29...`.

Update memories:
- `project_release_state.md` → 2.0.0 current; ADR-001 entry; first major-version note
- `project_followup_prs.md` → PR-S11 SHIPPED in 2.0.0 (with implementation notes); PR-S12 (was-PR-S11 add_possessions improvement) is now next-in-queue

Done.

---

## Self-review checklist (run before presenting plan to user)

1. **Spec coverage:** every section of the spec maps to at least one task above. ✓
   - § 2 Goal 1 (Sportec contract fix) → Task 3
   - § 2 Goal 2 (4 new tackle_* columns) → Task 3
   - § 2 Goal 3 (SPORTEC_SPADL_COLUMNS) → Task 2
   - § 2 Goal 4 (use_tackle_winner_as_actor helper) → Task 4
   - § 2 Goal 5 (ADR-001 + ADR-TEMPLATE) → Task 1
   - § 2 Goal 6 (CLAUDE.md amendment) → Task 1
   - § 2 Goal 7 (Cross-provider parity gate) → Task 5
   - § 2 Goal 8 (e2e on IDSSE fixture) → Task 6
   - § 5 Test plan (32 new tests, distributed across Tasks 3, 4, 5, 6) ✓
   - § 6 ADR-001 outline → Task 1.2 (full content)
   - § 7 Verification gates → Task 9
   - § 8 Commit cycle → Task 10
   - § 9 CHANGELOG migration block → Task 8

2. **Type / signature consistency:**
   - `use_tackle_winner_as_actor(actions: pd.DataFrame) -> pd.DataFrame` — signature consistent across spec § 3, § 4.4, plan Task 4. ✓
   - 4 column names (`tackle_winner_player_id`, `tackle_winner_team_id`, `tackle_loser_player_id`, `tackle_loser_team_id`) — consistent across spec § 4.2, § 4.3, plan Task 2, Task 3. ✓
   - `SPORTEC_SPADL_COLUMNS` schema constant — consistent across spec § 4.2, plan Task 2. ✓
   - The 18-column count claim — consistent across spec § 9 ("14 + 4 = 18"), plan Task 2.3 smoke test. ✓

3. **No placeholders:** every step contains actual code, exact paths, exact commands. ✓

4. **Single-commit ritual:** every task ends with `git add` (stage), no intermediate `git commit`. The single commit happens in Task 10. ✓

5. **Hook scope honoured:** sentinel-gated only on `git commit` (Task 10.4). Push / PR / merge / tag (Tasks 10.5-10.8) are chat-only approval. ✓

