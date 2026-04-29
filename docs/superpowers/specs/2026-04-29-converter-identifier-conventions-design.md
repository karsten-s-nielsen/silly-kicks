# SPADL converter identifier conventions — caller's `team_id` / `player_id` are sacred

**Status:** Approved (design)
**Target release:** silly-kicks 2.0.0 (first semver-major)
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-29
**Predecessor:** 1.10.0 (`docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md`)
**Triggered by:** luxury-lakehouse PR-LL2 close-out report (2026-04-29) — `bronze.spadl_actions` for IDSSE has 1412/2522 (56%) tackle rows with silently-rewritten team_ids because `silly_kicks.spadl.sportec.convert_to_actions` overrides the caller's `team` column from raw DFL `tackle_winner_team` qualifier values

---

## 1. Problem

`silly_kicks/spadl/sportec.py:559-565` (post-1.10.0 main, the same code lived since 1.7.0) overrides both `player_id` and `team` on TacklingGame rows when DFL's `tackle_winner` / `tackle_winner_team` qualifiers are populated:

```python
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

Then assembled at `:605`:

```python
"team_id": rows["team"].astype("object"),
```

### 1.1 Empirical impact (luxury-lakehouse PR-LL2 close-out, 2026-04-29)

`bronze.spadl_actions` for IDSSE (2522 rows total):

| Bucket | Rows | Cause |
|---|---|---|
| Non-NULL team_id_native | 164 | TacklingGame rows where `tackle_winner` was empty/NaN — override didn't fire. |
| **NULL — override fired (THE BUG)** | **1412** | TacklingGame rows where `tackle_winner_team` was a raw DFL CLU id; rewrote caller's `home`/`away` label. |
| NULL — bronze team='unknown' | 759 | DFL events without team attribution per the DFL spec (Foul/Caution/FairPlay/etc.) — legitimate. |
| NULL — synthetic dribble rows | 187 | Inserted by `_add_dribbles`, propagated the prior overridden `team_id`. |

**1412 / 2522 = 56% silent corruption of caller-supplied team labels** on tackle rows.

### 1.2 Why this is an API problem (general — not lakehouse-specific)

`convert_to_actions(events, home_team_id="home")` accepts a string for `home_team_id` and uses it as a label compared against the input `team` column. The implicit contract: `team_id` in the output mirrors `team` in the input. The TacklingGame override breaks this contract on a subset of rows of a single event type, with no warning and no opt-out.

Any consumer who normalizes the input `team` column to a convention OTHER than DFL's raw `tackle_winner_team` format (`DFL-CLU-...`) gets silently corrupted output for tackle rows. The luxury-lakehouse adapter normalizes to `home`/`away`/`unknown` labels — the most natural choice given the API surface — and gets bit hardest. Future consumers normalizing to other formats (integer team_ids, team names, abbreviated codes) face the same hazard.

### 1.3 General-cause analysis (the contract that's missing)

The override exists because DFL's TacklingGame XML records the event from the duel-recorder's perspective (typically the loser side per DFL conventions), with the actual SPADL tackler's identity in the `tackle_winner` qualifier. SPADL semantically wants `tackle.player_id = winner` — the converter is doing the right thing semantically but pulls override values from raw qualifier columns without normalization.

**Generalizing:** any time a converter would override `player_id` / `team_id` from a source-specific qualifier, that's a code smell. The converter has no way to translate the qualifier's identifier convention into the caller's, and silly-kicks deliberately doesn't accept resolution metadata (no `team_id_resolver` callable, no `dim_players` join). The principled answer: never override.

### 1.4 Cross-converter audit (this session, 2026-04-29)

Manual review of all 5 SPADL DataFrame converters confirmed the override is unique to sportec.tackle:

| Converter | Override `player_id` / `team_id`? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | **YES (tackle_winner / tackle_winner_team)** | The bug. Removed in 2.0.0. |
| `silly_kicks.spadl.metrica` | NO | The 1.10.0 `goalkeeper_ids` routing only changes `type_id` / `bodypart_id`. |
| `silly_kicks.spadl.wyscout` | NO | The 1.0.0 `goalkeeper_ids` aerial-duel reclassification only changes `type_id` / `subtype_id` on the input events (not the output actions). |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides; `player_id` / `team_id` derive from `events.player_id` / `events.team_id` directly. |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.kloppy` | NO | Gateway path; relies on the dedicated converters above. |

The 2.0.0 change is surgical (one converter), but locks the contract across the whole 5-converter surface for future converter additions.

## 2. Goals

1. **Sportec converter contract fix** — remove the tackle override; `team_id` / `player_id` mirror input `team` / `player_id` verbatim per the universal contract.
2. **Surface qualifier values via dedicated columns** — add 4 new output columns (`tackle_winner_player_id`, `tackle_winner_team_id`, `tackle_loser_player_id`, `tackle_loser_team_id`) to the sportec output. Verbatim from DFL qualifiers; NaN on rows where the qualifier is absent; NaN on non-tackle rows always.
3. **`SPORTEC_SPADL_COLUMNS` schema constant** — extends `KLOPPY_SPADL_COLUMNS` with the 4 new columns. Sportec's `convert_to_actions` returns DataFrame with this schema; other converters unchanged.
4. **`silly_kicks.spadl.use_tackle_winner_as_actor(actions)` migration helper** — pure post-conversion enrichment that overwrites `team_id` / `player_id` from the new winner columns where non-null. Pre-2.0.0 sportec consumers wanting the old SPADL "actor = winner" semantic call this helper post-conversion.
5. **ADR-001** at `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` — captures the contract and the audit. Plus `docs/superpowers/adrs/ADR-TEMPLATE.md` vendored verbatim from luxury-lakehouse to establish the silly-kicks ADR pattern going forward.
6. **`CLAUDE.md` "Key conventions" amendment** — one rule line ("Converter identifier conventions are sacred...") so future-Claude sessions don't reintroduce the override pattern.
7. **Cross-provider parity regression gate** — extend `tests/spadl/test_cross_provider_parity.py` with `TestNoTeamOverride[provider]` parametrized over all 5 converters: `set(actions["team_id"].unique()) ⊆ set(input.team.unique())`. Locks the contract per-provider going forward; would have caught the 1.7.0 bug.
8. **e2e on the vendored IDSSE production fixture** (`tests/datasets/idsse/sample_match.parquet` from 1.10.0) — verify caller's labels preserved, winner columns populated, helper round-trips, no regression in existing GK coverage from 1.10.0.

## 3. Non-goals

1. **No `team_id_resolver` / `player_id_resolver` callable** — silly-kicks deliberately doesn't accept identifier-translation metadata. Callers normalize upstream OR resolve downstream via their own joins. Adding resolver callables would invite a category of bug: callers passing wrong resolvers and getting silently mismatched outputs.
2. **No symmetric `use_tackle_loser_as_actor` helper** — SPADL never has "tackle.actor = loser" semantics; the helper would invite confusion. Loser data lives in `tackle_loser_*` columns for downstream defensive-action analytics; consumers can build their own swaps if needed.
3. **No deprecation-warning cycle** — emitting wrong-format values until 2.0.0 anyway adds log noise without fixing anything. Single coherent break is cleaner.
4. **No changes to other converters' output schemas** — `KLOPPY_SPADL_COLUMNS` and `SPADL_COLUMNS` unchanged; metrica/wyscout/kloppy/opta/statsbomb outputs continue to use the existing canonical 14-column shape.
5. **No bundled cleanup** — 2.0.0 is laser-focused on the override removal + the principle. Other major-version cleanups (deprecated APIs, anything else worth a breaking change) come in their own cycle.
6. **No public exposure of `_RECOGNIZED_QUALIFIER_COLUMNS` / `_CONSULTED_QUALIFIER_COLUMNS`** — internal implementation details; do not become part of the 2.0.0 contract.
7. **No lakehouse-side changes** — luxury-lakehouse's `_team_label_to_dfl_id` shim from PR-LL2 close-out becomes the documented "winner-attribution post-conversion" pattern. Lakehouse can drop the shim or keep it independently. Pin shift from `>=1.7.0` to `>=2.0.0,<3.0` is lakehouse's territory in their own PR.
8. **No new ADRs beyond ADR-001** — the audit findings live in ADR-001's "Notes" section. Future converter additions get reviewed against ADR-001; new ADRs only when a new principle emerges.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/schema.py                                   MOD  +8 / 0      Add SPORTEC_SPADL_COLUMNS = {**KLOPPY_SPADL_COLUMNS, ...4 new}
silly_kicks/spadl/sportec.py                                  MOD  +60 / -10   Remove tackle override; populate 4 new columns; add use_tackle_winner_as_actor helper; update module docstring with the contract
silly_kicks/spadl/__init__.py                                 MOD  +3 / 0      Re-export SPORTEC_SPADL_COLUMNS + use_tackle_winner_as_actor
tests/spadl/test_sportec.py                                   MOD  +200 / -10  TestSportecTackleNoOverride + TestSportecTackleWinnerColumns + TestSportecTackleLoserColumns + TestSportecOutputSchema; remove pre-2.0.0 override-asserting tests; update existing tackle test to mirror new contract
tests/spadl/test_use_tackle_winner_as_actor.py                NEW  ~140 lines  TestUseTackleWinnerAsActorContract / Correctness / Degenerate (mirrors PR-S8 boundary_metrics test discipline)
tests/spadl/test_cross_provider_parity.py                     MOD  +50 / 0     TestNoTeamOverride parametrized regression gate; TestSportecAdrContractOnProductionFixture
docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md  NEW  ~80 lines  First silly-kicks ADR; lakehouse template structure
docs/superpowers/adrs/ADR-TEMPLATE.md                         NEW  ~65 lines  Vendored verbatim from luxury-lakehouse
docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md  THIS FILE  Bundled into single commit
docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md  NEW   Implementation plan, bundled into single commit
CLAUDE.md                                                     MOD  +1 / 0      "Key conventions" gains one rule
pyproject.toml                                                MOD  +1 / -1     Version 1.10.0 → 2.0.0
CHANGELOG.md                                                  MOD  +90 lines   ## [2.0.0] entry with ⚠️ Breaking section + migration block
TODO.md                                                       MOD  ~+3 / -0    Track follow-up: PR-S12 (was-PR-S11 add_possessions improvement) bumps in queue
docs/c4/architecture.dsl + .html                              MOD  +1 / -1     Add SPORTEC_SPADL_COLUMNS to spadl container public-helper enumeration; mention use_tackle_winner_as_actor
```

### 4.2 `SPORTEC_SPADL_COLUMNS` schema

In `silly_kicks/spadl/schema.py`, after the existing `KLOPPY_SPADL_COLUMNS` definition:

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

Re-exported from `silly_kicks/spadl/__init__.py`'s `__all__`.

### 4.3 Sportec converter behavior change

Replace the override block with a no-op + per-row qualifier extraction:

```python
# OLD (1.10.0):
if is_tackle.any():
    winner_p = _opt("tackle_winner", None)
    winner_t = _opt("tackle_winner_team", None)
    override_mask = is_tackle & winner_p.notna().to_numpy()
    if override_mask.any():
        rows.loc[override_mask, "player_id"] = winner_p[override_mask].values
        rows.loc[override_mask, "team"] = winner_t[override_mask].values

# NEW (2.0.0):
# No override per ADR-001: team_id / player_id mirror input verbatim.
# Winner / loser qualifier values surface as dedicated columns below.
```

Then in the actions DataFrame assembly, the 4 new columns are populated from the source qualifier columns (NaN on non-tackle rows by virtue of `is_tackle` masking):

```python
# Build 4 qualifier passthrough columns at the rows level (same length).
tackle_winner_player_id_arr = np.where(
    is_tackle, _opt("tackle_winner", np.nan).to_numpy(), np.nan
)
tackle_winner_team_id_arr = np.where(
    is_tackle, _opt("tackle_winner_team", np.nan).to_numpy(), np.nan
)
tackle_loser_player_id_arr = np.where(
    is_tackle, _opt("tackle_loser", np.nan).to_numpy(), np.nan
)
tackle_loser_team_id_arr = np.where(
    is_tackle, _opt("tackle_loser_team", np.nan).to_numpy(), np.nan
)

# In the main actions DataFrame construction:
actions = pd.DataFrame({
    "game_id": ...,
    # ... existing 13 canonical columns ...
    "tackle_winner_player_id": tackle_winner_player_id_arr,
    "tackle_winner_team_id": tackle_winner_team_id_arr,
    "tackle_loser_player_id": tackle_loser_player_id_arr,
    "tackle_loser_team_id": tackle_loser_team_id_arr,
})
```

`_finalize_output(actions, schema=SPORTEC_SPADL_COLUMNS, extra_columns=preserve_native)` — the existing schema-aware finalize accepts the new schema constant unchanged.

### 4.4 `use_tackle_winner_as_actor` migration helper

Defined in `silly_kicks/spadl/sportec.py` (provenance with the producer); re-exported from `silly_kicks/spadl/__init__.py`:

```python
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
```

**Behavior contract:**
- Pure function (no input mutation).
- Raises `ValueError` early on any missing required column (matches the `add_*` enrichment family pattern).
- No-op on rows where `tackle_winner_player_id` is NaN (`team_id` / `player_id` left at original values).
- Both `player_id` AND `team_id` overwritten atomically per row — never one without the other (matches DFL's qualifier pairing semantic; tests assert this).
- Does NOT touch the four `tackle_*` columns — they remain on the output for downstream inspection.

### 4.5 ADR-001 contents

Mirrors lakehouse template structure exactly. See Section 6 below for the field-by-field outline that goes into the ADR file.

### 4.6 `CLAUDE.md` amendment

Inserted as a single bullet in the project's `CLAUDE.md` "Key conventions" section:

```markdown
- **Converter identifier conventions are sacred.** SPADL DataFrame converters never override the caller's `team_id` / `player_id` columns from provider-specific qualifiers. Qualifier-derived facts surface as dedicated output columns (see `tackle_winner_*` / `tackle_loser_*` on sportec). Decision: ADR-001.
```

## 5. Test plan

TDD discipline: all new tests written and verified failing BEFORE the code change.

### 5.1 Unit tests — sportec converter

`tests/spadl/test_sportec.py` — new test classes (replacing the pre-2.0.0 `TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor` test which asserted the now-removed override).

```
TestSportecTackleNoOverride
├── test_tackle_team_id_mirrors_input_team
├── test_tackle_player_id_mirrors_input_player_id
├── test_legacy_home_label_survives
├── test_legacy_away_label_survives
├── test_mixed_tackle_rows_preserve_per_row_labels
└── test_tackle_with_no_qualifier_unchanged

TestSportecTackleWinnerColumns
├── test_tackle_winner_player_id_populated_from_qualifier
├── test_tackle_winner_team_id_populated_from_qualifier
├── test_winner_columns_nan_when_qualifier_absent
├── test_winner_columns_nan_on_non_tackle_rows
├── test_winner_columns_unaffected_by_extraneous_input_columns
└── test_winner_columns_present_in_output_schema

TestSportecTackleLoserColumns
├── test_tackle_loser_player_id_populated_from_qualifier
├── test_tackle_loser_team_id_populated_from_qualifier
├── test_loser_columns_nan_when_qualifier_absent
└── test_loser_columns_nan_on_non_tackle_rows

TestSportecOutputSchema
├── test_output_columns_match_sportec_spadl_columns_keys
├── test_output_dtypes_match_sportec_spadl_columns
└── test_synthetic_dribble_rows_have_nan_in_tackle_columns
```

### 5.2 Unit tests — migration helper

New file `tests/spadl/test_use_tackle_winner_as_actor.py`. Mirrors PR-S8 `boundary_metrics` test discipline.

```
TestUseTackleWinnerAsActorContract
├── test_returns_dataframe
├── test_does_not_mutate_input
├── test_missing_team_id_column_raises
├── test_missing_player_id_column_raises
├── test_missing_tackle_winner_player_id_column_raises
└── test_missing_tackle_winner_team_id_column_raises

TestUseTackleWinnerAsActorCorrectness
├── test_overwrites_team_id_from_winner_team_when_non_null
├── test_overwrites_player_id_from_winner_player_when_non_null
├── test_atomic_overwrite_both_or_neither
├── test_rows_with_nan_winner_left_unchanged
├── test_preserves_all_other_columns
└── test_preserves_tackle_winner_columns_themselves

TestUseTackleWinnerAsActorDegenerate
├── test_empty_dataframe_returns_empty
├── test_all_nan_winner_returns_identity
└── test_preserves_preserve_native_columns
```

### 5.3 Cross-provider parity regression gate

`tests/spadl/test_cross_provider_parity.py` — new parametrized class:

```python
@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_team_id_mirrors_input_team(provider: str):
    """ADR-001 contract: every converter's output team_id must be a subset
    of the input team values. Locks the no-override contract per-provider
    as a regression gate going forward."""
    # Per-provider: load fixture, capture input team values, run converter,
    # assert set(actions["team_id"].unique()) ⊆ set(input_team_values).
```

Plus the pre-existing 1.10.0 cross-provider tests (`test_converter_emits_at_least_one_keeper_action`, `test_converter_returns_canonical_spadl_columns`) are kept; the latter is updated to use `SPORTEC_SPADL_COLUMNS` for the sportec branch.

### 5.4 e2e on the IDSSE production fixture

New class in `tests/spadl/test_cross_provider_parity.py` (or as a separate e2e-marker-free module — per the 1.9.0 pattern, fixtures committed mean tests run in the regular suite):

```
TestSportecAdrContractOnProductionFixture
├── test_no_dfl_clu_strings_leak_into_team_id
├── test_tackle_winner_team_id_populated_for_qualified_rows
├── test_use_tackle_winner_as_actor_round_trips_to_pre_2_0_0_behavior
└── test_keeper_coverage_preserved_from_1_10_0
```

### 5.5 Existing test updates

Pre-2.0.0 tests asserting the override behavior must be removed / updated:

| Existing test | Action |
|---|---|
| `tests/spadl/test_sportec.py::TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor` | DELETE — was asserting the buggy override. Replaced by `TestSportecTackleNoOverride`. |
| Any other test asserting `actions["team_id"].iloc[i] == "T-HOME"` after a tackle override fired | Update to assert the new contract (caller's team label preserved). |

## 6. ADR-001 outline

Field-by-field structure, mirroring `D:\Development\karstenskyt__luxury-lakehouse\docs\superpowers\adrs\ADR-TEMPLATE.md`:

```
# ADR-001: Converter identifier conventions — caller's team_id / player_id are sacred

| Field | Value |
|---|---|
| Date | 2026-04-29 |
| Status | Accepted |
| Deciders | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context
[3 paragraphs: the bug, the empirical impact (1412/2522 IDSSE rows = 56%),
the cross-converter audit confirming sportec is the unique offender]

## Decision
silly-kicks SPADL converters never override the caller's team_id /
player_id from provider-specific qualifiers. Qualifier-derived facts
surface as dedicated output columns with explicit naming (e.g.
tackle_winner_team_id). Caller-supplied team / player_id values mirror
verbatim into the canonical output fields.

## Alternatives considered
| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Documentation only + downstream shim | minimal silly-kicks change | pushes a footgun onto every future consumer; lakehouse-specific shim doesn't help unknown future consumers | violates "best practice, long term" |
| B. Deprecation warning in 1.11.0 + flip default in 2.0.0 | softer migration | emits wrong-format values until 2.0.0 anyway; adds log noise without fixing anything | clean break is cleaner |
| C (chosen). Single coherent break in 2.0.0 with use_tackle_winner_as_actor migration helper | cleanest contract; one CHANGELOG entry; locks the principle for future converters | breaking change for sportec consumers relying on the override (mitigation: helper) | — |

## Consequences

### Positive
- Caller's identifier conventions preserved across all converters.
- Provider-specific qualifier values surfaced with structure not silent-mutation.
- Future converters with similar quirks follow the same pattern.
- ADR pattern adopted in silly-kicks (lakehouse template vendored verbatim).

### Negative
- silly-kicks 2.0.0 is a breaking change for sportec consumers that:
  - relied on tackle "actor = winner" SPADL semantic, AND
  - had upstream-supplied `team` / `player_id` columns in the same format
    as DFL's `tackle_winner_team` / `tackle_winner` qualifiers (raw DFL ids).
  - Mitigation: use_tackle_winner_as_actor helper restores pre-2.0.0
    behavior for these consumers in one line.
- Sportec output schema gains 4 columns (sportec consumers must update
  schema-equality assertions).

### Neutral
- KLOPPY_SPADL_COLUMNS / SPADL_COLUMNS unchanged; non-sportec converters
  unaffected.
- Adds the per-provider <PROVIDER>_SPADL_COLUMNS extension precedent.

## CLAUDE.md Amendment
> Adds to the project's CLAUDE.md "Key conventions" section:
> "Converter identifier conventions are sacred. SPADL DataFrame converters
> never override the caller's team_id / player_id columns from
> provider-specific qualifiers. Qualifier-derived facts surface as
> dedicated output columns (see tackle_winner_* / tackle_loser_* on
> sportec). Decision: ADR-001."
> Scope: every SPADL DataFrame converter and any future provider.

## Related
- Spec: docs/superpowers/specs/2026-04-29-converter-identifier-conventions-design.md
- Plan: docs/superpowers/plans/2026-04-29-converter-identifier-conventions.md
- Issues / PRs: luxury-lakehouse PR-LL2 close-out report (2026-04-29);
  silly-kicks #16 (this PR)
- External references: kloppy upstream (no equivalent contract documented;
  silly-kicks's override pattern was inherited from socceraction v1.5.3
  era and never re-examined)

## Notes
[Verbatim audit table from spec § 1.4]
[Empirical impact table from spec § 1.1]
```

## 7. Verification gates

```bash
# CI pin (per feedback_ci_cross_version memory)
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/ scripts/
uv run ruff format --check silly_kicks/ tests/ scripts/
uv run pyright silly_kicks/
uv run pytest tests/ -m "not e2e" -v --tb=short
```

**Expected post-2.0.0 baseline:** ~660 passed (627 from 1.10.0 + ~32 new), 4 skipped (pre-existing). Pyright + ruff zero errors.

## 8. Commit cycle

Per `feedback_commit_policy` memory + narrowed hook (only `git commit` + `git push --force` + `git reset --hard` + `git rebase` sentinel-gated):

```
1. Branch: feat/converter-identifier-conventions from main (currently d27dc48 = 1.10.0 merge).
2. /final-review pass + all gates green
3. User approves → sentinel touch → git commit (ONE commit)
4. User approves → git push -u origin feat/...
5. User approves → gh pr create
6. CI green → user approves → gh pr merge --admin --squash --delete-branch
7. User approves → git tag v2.0.0 + git push origin v2.0.0  # PyPI auto-publish
```

Single-commit ritual unchanged from 1.10.0.

## 9. CHANGELOG migration block

Top of `## [2.0.0] — 2026-04-29` entry:

```markdown
### ⚠️ Breaking

- **`silly_kicks.spadl.sportec.convert_to_actions` no longer overrides
  `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team`
  qualifiers.** Per ADR-001 (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`),
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

    from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
    actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
    actions = use_tackle_winner_as_actor(actions)

If your `team` / `player_id` columns use any other convention, the
post-1.10.0 behavior already preserved your conventions correctly — no
migration needed; the bugfix is automatic on upgrade.
```

## 10. Risks + mitigations

| Risk | Mitigation |
|---|---|
| **Sportec consumers relying on the override + matching qualifier format get bit on upgrade** | `use_tackle_winner_as_actor` helper restores pre-2.0.0 behavior in one line; explicit migration block in CHANGELOG. |
| **Cross-provider parity test flakiness on the new `TestNoTeamOverride` gate** if any provider's converter has a non-trivial transformation of `team_id` we missed | The audit (spec § 1.4) explicitly walks every converter; the gate would catch this BEFORE the 2.0.0 commit lands. Per-row `set(actions["team_id"].unique()) ⊆ set(input.team.unique())` is a strict-but-correct contract. |
| **Pyright complains about the new `tackle_*` columns** if pandas-stubs narrows the np.where output too tightly | Use explicit `dtype="object"` on the np.where outputs; if pyright still complains, fall back to `pd.Series(..., dtype="object")` per the 1.10.0 sportec synthesis pattern. |
| **CI cross-platform encoding** of new tackle_winner / tackle_loser values (object dtype, mixed string + NaN) | Tests assert dtype = "object" explicitly; 1.10.0 cycle confirmed this dtype round-trips cleanly through parquet. |
| **The audit misses a corner case** in some less-traveled provider path | The cross-provider parity gate is the safety net: any future audit miss surfaces as a test failure on the next cycle. |
| **Contract creep** — future converters tempted to "just override one field" for the same SPADL-semantic reasons | ADR-001 + CLAUDE.md amendment make the contract explicit; ADR forces a decision-trail for any future override (must supersede ADR-001). |
| **2.0.0 is a major-version bump** for a young library | silly-kicks is ~3 weeks old (0.1.0 shipped 2026-04-06); major versions aren't precious. Bumping locks the contract before more consumers downstream-pin. |

## 11. Out of scope (queued follow-ups)

### PR-S12 — `add_possessions` algorithmic precision improvement (was PR-S11)

Re-numbered for the third time as silly-kicks production-bug PRs took the slots. Original design (brief-opposing-action merge rule + max_gap_seconds parameter sweep) unchanged. See `project_followup_prs.md`. The 64-match WorldCup HDF5 from PR-S9 + the 3-fixture StatsBomb open-data set from PR-S8 + the IDSSE / Metrica fixtures from 1.10.0 give plenty of empirical surface for parameter tuning.

### Atomic-SPADL `coverage_metrics` parity (TODO.md tech debt C-1)

Unchanged from 1.10.0 disposition. Defer until concrete consumer ask.

## 12. Acceptance criteria

1. `silly_kicks.spadl.schema.SPORTEC_SPADL_COLUMNS` exists; equals `KLOPPY_SPADL_COLUMNS` plus 4 named columns; re-exported from `silly_kicks.spadl`.
2. `silly_kicks.spadl.sportec.convert_to_actions` returns DataFrame matching `SPORTEC_SPADL_COLUMNS` shape (column names + dtypes).
3. The 4 new `tackle_*` columns are populated verbatim from DFL qualifier columns when present; NaN when absent; NaN on non-tackle rows always.
4. Tackle rows' `team_id` / `player_id` mirror the input `team` / `player_id` columns verbatim. `set(actions["team_id"].unique()) ⊆ set(input.team.unique())` for every tackle subset.
5. `silly_kicks.spadl.use_tackle_winner_as_actor(actions) -> pd.DataFrame` is public, importable from `silly_kicks.spadl`, returns a DataFrame, raises `ValueError` on missing required columns, doesn't mutate input.
6. `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` exists with the structure outlined in § 6.
7. `docs/superpowers/adrs/ADR-TEMPLATE.md` exists, vendored verbatim from luxury-lakehouse.
8. `CLAUDE.md` "Key conventions" section gains the one-bullet rule from § 4.6.
9. `tests/spadl/test_cross_provider_parity.py::test_team_id_mirrors_input_team` passes for all 5 DataFrame converters (sportec, metrica, statsbomb, opta, wyscout).
10. `tests/spadl/test_use_tackle_winner_as_actor.py` exists with the 15 tests outlined in § 5.2; all pass.
11. e2e on the IDSSE production fixture (vendored 1.10.0): `team_id` for tackle rows preserves caller's home_team_id convention (no `DFL-CLU-...` leakage); `tackle_winner_team_id` populated for qualifier-positive rows; helper round-trips correctly.
12. `pyproject.toml` version is `2.0.0`. Tag `v2.0.0` is pushed → PyPI auto-publish workflow succeeds.
13. `CHANGELOG.md` has a `## [2.0.0] — 2026-04-29` entry with the explicit `### ⚠️ Breaking` and `### Migration` sections from § 9.
14. Verification gates (ruff/format/pyright/pytest) all pass before commit. /final-review passes. C4 architecture diagram regenerated to mention the new helper + schema.
15. CI matrix (lint + ubuntu 3.10/3.11/3.12 + windows 3.12) all green before merge.
16. luxury-lakehouse can bump `silly-kicks>=2.0.0,<3.0` and (optionally) drop their `_team_label_to_dfl_id` shim from PR-LL2 close-out, OR keep it as a documented winner-attribution post-conversion pattern. *(Acceptance is on lakehouse side; silly-kicks 2.0.0 ships the contract; lakehouse decides their own pattern.)*
