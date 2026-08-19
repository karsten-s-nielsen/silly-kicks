# SB360 completion — `cross_blocked` enablement + licensed coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un-defer StatsBomb `cross_blocked` (conditional on a pre-registered corpus probe) and render the committed licensed SB360 coverage parquet as a human-readable `coverage.md`.

**Architecture:** Part 1 — an ad-hoc, print-only probe measures the `related_events`→`Block` join over StatsBomb open data against a pre-registered R1–R3 rule; if it passes, the converter gains one masked column mirroring the existing `shot_blocked` site, verified offline on committed fixtures. Part 2 — a deterministic render script (the `render_sb360_matrix.py` class: reads a committed artifact, writes a doc, unguarded, enrolled as a non-driver) emits `coverage.md`. The two halves are independent.

**Tech Stack:** Python, pandas (nullable `"boolean"` dtype), numpy, pytest. `silly_kicks.spadl.statsbomb`, `silly_kicks.spadl.utils._blocked_flag`, `silly_kicks.id_compat.same_id`. Optional `statsbombpy` (probe fetch only).

## Global Constraints

- **No version number is claimed.** It is resolved from `main` only at completion (Task 8). If Part 1 keep-defers, there is **no release** — a plain `main` commit, no bump, no tag, no PyPI publish (spec §7.2).
- **One commit for the whole cycle.** No task writes a commit step. No provenance-guarded artifact driver is added: the probe prints to stdout (no `write_text`); the render reads a committed parquet and is unguarded.
- **`cross_blocked` scope:** open-play `cross` type ONLY (`applicable`); set-piece crosses (`corner_crossed`/`freekick_crossed`) stay `pd.NA`. `blocked` = an open-play cross whose `related_events` links to a `Block` by the **opposing** team.
- **Dtype safety (ADR-019):** the opposing-team compare uses `silly_kicks.id_compat.same_id` (scalar↔scalar, NA-safe). The uuid lookup keys on the string `event_id` — **never** `.astype(str)` on any numeric id.
- **Additive / no silly-kicks retrain:** `cross_blocked` is consumed by no `vaep/`/`atomic/` feature (verified); ADR-045 declares block-detection columns reflection-`"invariant"`.
- **Committed fixtures:** `tests/datasets/statsbomb/raw/events/{7298,7584,3754058}.json`. The one genuine blocked cross is in `7584`: `original_event_id == "e8edd276-8490-456c-b221-240d128f61f1"` (min 89). `7298`/`3754058` have **zero** blocked crosses.
- **Licensed artifact (Part 2 input, committed):** `docs/research/sb360_licensed_coverage/coverage.parquet` (7,740 rows; cols `match_id, kind, subject, metric, value, denominator, detail`) + `manifest_all.json` (`generation="100c80a1b37b40a6"`, `n_attempted=30`, `run_commit="cf2f155…"`). Ground facts to render: **31 of 230** battery columns fully-NaN; roster resolution **1.0**; pitch coverage mean **0.273**.
- **Render determinism:** fixed decimals + sorted groupings; produce/verify on `.venv312`. Not golden-gated; a cheap staleness guard checks the stamped generation hash only.
- **Test command:** `python -m pytest tests/ -m "not e2e" -v --tb=short`. Lint at CI scope: `python -m ruff check silly_kicks/ tests/ scripts/` + `ruff format --check …`. `pyright` bare.

---

## Task 1: The probe script (print-only, ad-hoc)

**Files:**
- Create: `scripts/probe_sb_cross_blocked.py`

**Interfaces:**
- Produces: a CLI that prints a per-match + aggregate table of the join measurements (base rate, ambiguity, team-side, set-piece leakage, symmetry). No importable API relied on by later tasks. **No `write_text` / file output** (keeps it off the artifact-driver population; spec §3.1).

- [ ] **Step 1: Write the probe.** It measures the offline core (the three committed fixtures) unconditionally and an optional wider open-data slice via `statsbombpy` (guarded import). It reuses the converter's open-play-cross rule inline (same predicate as `statsbomb.py:466`) so it measures the same mask the converter would emit.

```python
"""Probe the StatsBomb `cross → related_events → Block` join for `cross_blocked` (BD-2).

Ad-hoc, print-only investigation (spec §3.1): prints the R1-R3 evidence to stdout and writes
NOTHING. The offline core is the three committed fixtures; `--open "comp/season"` additionally
fetches a wider StatsBomb OPEN-data slice via statsbombpy (guarded import) for corpus-scale
base-rate + edge-case evidence. Open data is a proxy for the whole `statsbomb` provider:
`related_events` is a standard field and the same converter path serves licensed SB360, so the
measurement generalises by construction, not by measuring the un-probeable licensed rows.

Decision rule (pre-registered, spec §3.2): SHIP iff R1 (< 5% of open-play crosses have absent
`related_events`) AND R2 (same-team Block links absent, or < 1% of linked crosses) AND R3 (the
">=1 opposing Block" rule is well-defined on 100% of linked cases -- it is, by construction).

Team comparison here uses raw `!=` (measurement-only, on raw-int open-data ids); the CONVERTER
(Task 4) uses `id_compat.same_id` for the NA-safe production path -- they agree except on an
NA-team edge case, which the probe reports separately. The R2 `linked` denominator
(`blocked + same_team_block`) slightly double-counts a cross carrying BOTH an opposing and a
same-team Block -- a rough bound for the < 1% check, acknowledged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "tests" / "datasets" / "statsbomb" / "raw" / "events"
_SET_PIECE = {"Corner", "Free Kick", "Goal Kick", "Throw-in"}


def _is_open_play_cross(e: dict) -> bool:
    if (e.get("type") or {}).get("name") != "Pass":
        return False
    p = e.get("pass") or {}
    if not p.get("cross"):
        return False
    return (p.get("type") or {}).get("name") not in _SET_PIECE


def measure(events: list[dict]) -> dict:
    """Per-match join measurements over one events list."""
    by_id = {e.get("id"): e for e in events}
    n_cross = n_absent_rel = n_blocked = n_multi_block = n_same_team = n_asym = 0
    for e in events:
        if not _is_open_play_cross(e):
            continue
        n_cross += 1
        rel = e.get("related_events") or []
        if not rel:
            n_absent_rel += 1
            continue
        blocks = [by_id[r] for r in rel if r in by_id and (by_id[r].get("type") or {}).get("name") == "Block"]
        if not blocks:
            continue
        if len(blocks) > 1:
            n_multi_block += 1
        my_team = (e.get("team") or {}).get("id")
        opposing = [b for b in blocks if (b.get("team") or {}).get("id") != my_team]
        same = [b for b in blocks if (b.get("team") or {}).get("id") == my_team]
        if same:
            n_same_team += 1
        if opposing:
            n_blocked += 1
            # symmetry: does at least one opposing Block list this cross back?
            if not any(e.get("id") in (b.get("related_events") or []) for b in opposing):
                n_asym += 1
    return {
        "open_play_crosses": n_cross,
        "blocked": n_blocked,
        "absent_related_events": n_absent_rel,
        "multi_block": n_multi_block,
        "same_team_block": n_same_team,
        "asymmetric": n_asym,
    }


def _fixture_events() -> dict[str, list[dict]]:
    out = {}
    for mid in ("7298", "7584", "3754058"):
        out[mid] = json.loads((_FIXTURES / f"{mid}.json").read_text(encoding="utf-8"))
    return out


def _open_events(comp_season: str, limit: int) -> dict[str, list[dict]]:
    try:
        from statsbombpy import sb  # guarded: scripts-only optional dep
    except ImportError:
        print("statsbombpy not installed; skipping --open fetch", file=sys.stderr)
        return {}
    comp, season = (int(x) for x in comp_season.split("/"))
    matches = sb.matches(competition_id=comp, season_id=season)
    out = {}
    for mid in list(matches["match_id"])[:limit]:
        out[str(mid)] = list(sb.events(match_id=int(mid), fmt="dict").values())
    return out


def _print_report(title: str, per_match: dict[str, dict]) -> None:
    agg = {k: sum(m[k] for m in per_match.values()) for k in next(iter(per_match.values()))}
    print(f"\n=== {title} ({len(per_match)} matches) ===")
    for mid, m in sorted(per_match.items()):
        print(f"  {mid}: {m}")
    print(f"  AGG: {agg}")
    c = agg["open_play_crosses"] or 1
    linked = agg["blocked"] + agg["same_team_block"] or 1
    print(f"  base rate blocked/open-play-cross = {agg['blocked'] / c:.4f}")
    print(f"  R1 absent-related_events rate = {agg['absent_related_events'] / c:.4f}  (ship iff < 0.05)")
    print(f"  R2 same-team-link rate (of linked) = {agg['same_team_block'] / linked:.4f}  (ship iff < 0.01)")
    print(f"  R3 multi-block crosses = {agg['multi_block']} (rule is monotone: >=1 opposing Block)")
    print(f"  symmetry: asymmetric links = {agg['asymmetric']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", default=None, help='StatsBomb open comp/season, e.g. "43/106"')
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    _print_report("offline committed fixtures", {mid: measure(ev) for mid, ev in _fixture_events().items()})
    if args.open:
        wider = _open_events(args.open, args.limit)
        if wider:
            _print_report(f"open data {args.open}", {mid: measure(ev) for mid, ev in wider.items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the offline core and confirm the known numbers.**

Run: `python scripts/probe_sb_cross_blocked.py`
Expected: `7584` reports `blocked: 1`, `same_team_block: 0`, `asymmetric: 0`; `7298`/`3754058` report `blocked: 0`. AGG `open_play_crosses: 81`, `blocked: 1`, base rate `0.0123`, R1 rate `0.0000`, R2 rate `0.0000`.

- [ ] **Step 3: Confirm it is not flagged by any population gate.**

Run: `python -m pytest tests/scripts/test_provenance_wiring.py tests/scripts/test_corpus_driver_resilience.py tests/scripts/test_input_contracts.py -q`
Expected: PASS. The probe has no `write_text`/`to_parquet` (not an artifact driver) and calls no `_CORPUS_CALLS` and carries no `--match-ids-json`-style corpus args (not a corpus driver), so it enrolls in nothing.

---

## Task 2: Run the wider probe, record the measurement, decide, amend ADR-046

This task PRODUCES the ship/keep-defer decision that gates Tasks 3–5. Part 2 (Tasks 6–7) is independent and runs regardless.

**Files:**
- Create: `docs/research/sb360_cross_blocked/README.md`
- Modify: `docs/superpowers/adrs/ADR-046-block-detection-converter-columns.md`

- [ ] **Step 1: Run the wider probe if network is available.**

Run: `python scripts/probe_sb_cross_blocked.py --open 43/106 --limit 40` (FIFA World Cup 2022 open slice; any open comp/season works). If `statsbombpy` is absent or offline, the offline core (Task 1 Step 2) stands as the decisive mechanism evidence and R1–R3 are evaluated on it.

- [ ] **Step 2: Evaluate R1–R3 and record the numbers.** Write `docs/research/sb360_cross_blocked/README.md` — a hand-written note (numbers pasted from stdout; no machine artifact, no provenance guard). It must state: the probe corpus (offline fixtures + any open slice), the base rate, the R1/R2/R3 actuals, the symmetry finding, and the DECISION (ship / keep-defer) with its reason. Template:

```markdown
# SB360 `cross_blocked` — probe measurement and decision

**Question:** is the StatsBomb `cross → related_events → Block` join reliable enough to un-defer
`cross_blocked` (ADR-046 BD-2, deferred at n=1)?

## Corpus
- Offline: committed fixtures `7298`, `7584`, `3754058` (81 open-play crosses).
- Open (if run): `<comp/season>`, `<N>` matches (`<M>` open-play crosses).

## Measurements (from `scripts/probe_sb_cross_blocked.py`)
- Base rate blocked / open-play cross: `<x>`.
- R1 absent-`related_events` rate: `<x>`  (ship iff < 0.05).
- R2 same-team-link rate (of linked): `<x>`  (ship iff < 0.01).
- R3 multi-`Block` crosses: `<n>` — the ">= 1 opposing Block" rule is monotone, so a second
  Block cannot change the boolean; well-defined on 100% of linked cases by construction.
- Symmetry: `<n>` asymmetric links.

## Decision
**<SHIP | KEEP DEFERRED>.** <one paragraph tying the numbers to R1-R3.>
```

- [ ] **Step 3: Amend ADR-046.** Append an "Amendment (2026-08-19): StatsBomb `cross_blocked` un-deferral" section stating the probe outcome and, on the ship path, the mechanism (`applicable`=open-play cross, `blocked`=opposing-team `Block` via `related_events`) and the Hyrum note (downstream consumers see StatsBomb `cross_blocked` flip from all-`pd.NA` to real values; silly-kicks-side additive/no-retrain). On the keep-defer path, record the measured reason (a strict upgrade over the original `n=1` sentence).

- [ ] **Step 4: Gate.** If the decision is **keep-defer**, STOP here for Part 1 — skip Tasks 3–5, make no converter change, and proceed to Task 6. If **ship**, continue to Task 3.

---

## Task 3 (ship-path): Factor the open-play-cross predicate into one shared spelling

A pure, behavior-invariant refactor so the type dispatch and the `cross_blocked` mask cannot drift (spec §3.3.2, Risk row).

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py`
- Test: `tests/spadl/test_statsbomb.py`

**Interfaces:**
- Produces: `_open_play_cross_mask(events: pd.DataFrame) -> np.ndarray` (module-level helper in `statsbomb.py`), consumed by `_vectorized_type_id` and by Task 4's mask.

- [ ] **Step 1: Write the failing test** — a NON-tautological invariance guard. It must NOT compare the helper against `_vectorized_type_id` (which will CALL the helper after Step 3 → passes trivially). Instead it asserts the helper equals an INDEPENDENT inline spelling of the rule on real flattened events, plus a non-vacuity assertion that the True branch is exercised.

```python
def test_open_play_cross_mask_equals_inline_predicate():
    """The extracted helper must equal an independent inline spelling of the open-play-cross rule
    on real events. Deliberately NOT compared against `_vectorized_type_id` (post-refactor it calls
    the helper, so that comparison is tautological); the inline copy here is the invariance anchor.
    """
    import json
    import pathlib

    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl.statsbomb import _flatten_extra, _open_play_cross_mask

    raw = json.loads(
        pathlib.Path("tests/datasets/statsbomb/raw/events/7584.json").read_text(encoding="utf-8")
    )
    events = flatten_events(raw, 7584)
    events["extra"] = events["extra"].fillna({})  # mirror convert_to_actions
    events = _flatten_extra(events)

    inline = (
        (events["type_name"] == "Pass")
        & (events["_pass_cross"] == True)  # noqa: E712
        & ~events["_pass_type"].isin(["Free Kick", "Corner", "Goal Kick", "Throw-in"])
    ).to_numpy()
    mask = _open_play_cross_mask(events)
    assert mask.tolist() == inline.tolist()
    assert mask.sum() > 0  # non-vacuous: the fixture actually exercises the True branch
```

- [ ] **Step 2: Run it to verify it fails** (import error: `_open_play_cross_mask` undefined).

Run: `pytest tests/spadl/test_statsbomb.py::test_open_play_cross_mask_equals_inline_predicate -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add the helper and route the type dispatch through it.** In `statsbomb.py`, add near the other module helpers:

```python
def _open_play_cross_mask(events: pd.DataFrame) -> np.ndarray:
    """Open-play cross: a Pass with ``pass.cross`` True whose ``pass.type`` is not a set piece.

    ONE spelling, shared by ``_vectorized_type_id`` (the SPADL ``cross`` type) and the
    ``cross_blocked`` mask, so the two definitions cannot drift. Requires the ``_flatten_extra``
    columns (``_pass_cross``, ``_pass_type``) to be present.
    """
    is_pass = events["type_name"] == "Pass"
    return (
        is_pass
        & (events["_pass_cross"] == True)  # noqa: E712
        & ~events["_pass_type"].isin(["Free Kick", "Corner", "Goal Kick", "Throw-in"])
    ).to_numpy()
```

Then in `_vectorized_type_id`, replace the inline cross condition (currently `statsbomb.py:466`) with the shared mask:

```python
        # Cross (not a set piece)
        _open_play_cross_mask(events),
```

- [ ] **Step 4: Run the full statsbomb + block-detection suites to confirm no dispatch change.**

Run: `pytest tests/spadl/test_statsbomb.py tests/spadl/test_block_detection_contract.py tests/spadl/test_output_contract.py -v`
Expected: PASS (the refactor is behavior-invariant; only `test_statsbomb.py:413` will change later, in Task 4).

---

## Task 4 (ship-path): Lift `_related_events`, build the `cross_blocked` mask, flip the deferred assertions

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py` (`_flatten_extra`; a new `_cross_blocked_flag`; the `cross_blocked` site at `:282`)
- Test: `tests/spadl/test_statsbomb.py`

**Interfaces:**
- Consumes: `_open_play_cross_mask` (Task 3), `_blocked_flag` (`spadl/utils.py:1559`), `silly_kicks.id_compat.same_id`.
- Produces: `actions["cross_blocked"]` as a real `"boolean"` mask for StatsBomb.

- [ ] **Step 1: Write the failing positive + negative tests.**

```python
def test_cross_blocked_true_on_real_blocked_cross():
    from silly_kicks.spadl.utils import add_names
    from tests.invariants._loaders import load_statsbomb
    actions, _ = load_statsbomb(7584)
    actions = add_names(actions)
    crosses = actions[actions["type_name"] == "cross"]
    # exactly one genuine blocked cross in 7584 (min 89)
    assert (crosses["cross_blocked"] == True).sum() == 1  # noqa: E712
    blocked = crosses[crosses["cross_blocked"] == True]  # noqa: E712
    assert blocked["original_event_id"].iloc[0] == "e8edd276-8490-456c-b221-240d128f61f1"
    # every open-play cross carries a real True/False, never NA
    assert crosses["cross_blocked"].notna().all()
    # non-open-play rows (regular passes, set-piece crosses, non-passes) stay NA
    assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()


def test_cross_blocked_false_when_no_block_related():
    from silly_kicks.spadl.utils import add_names
    from tests.invariants._loaders import load_statsbomb
    for mid in (7298, 3754058):  # zero blocked crosses in these matches
        actions, _ = load_statsbomb(mid)
        actions = add_names(actions)
        crosses = actions[actions["type_name"] == "cross"]
        assert crosses["cross_blocked"].notna().all()
        assert (crosses["cross_blocked"] == False).all()  # noqa: E712
        assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()
```

- [ ] **Step 2: Run to verify they fail** (StatsBomb `cross_blocked` is still all-`pd.NA`).

Run: `pytest tests/spadl/test_statsbomb.py::test_cross_blocked_true_on_real_blocked_cross tests/spadl/test_statsbomb.py::test_cross_blocked_false_when_no_block_related -v`
Expected: FAIL (assertion: NA where True/False expected).

- [ ] **Step 3: Lift `_related_events`.** In `_flatten_extra`, after the `_carry_end_location` line:

```python
    events["_related_events"] = extra.str.get("related_events")
```

- [ ] **Step 4: Add the mask builder and wire it in.** Add the helper (near `_open_play_cross_mask`):

```python
def _cross_blocked_flag(events: pd.DataFrame) -> "pd.arrays.BooleanArray":
    """StatsBomb ``cross_blocked``: an open-play cross whose ``related_events`` links to a
    ``Block`` by the OPPOSING team. Built on the PRE-FILTER events frame so every uuid resolves;
    aligned to ``actions`` exactly as ``shot_blocked`` is (same call site, same length).
    """
    from silly_kicks.id_compat import same_id  # local import mirrors the module's id_compat usage

    applicable = _open_play_cross_mask(events)
    # uuid -> (type_name, team_id) over the full events. event_id is a genuine string uuid;
    # team_id is numeric -> compared via same_id, never stringified (ADR-019 dict-key trap).
    lookup = dict(
        zip(events["event_id"], zip(events["type_name"], events["team_id"], strict=True), strict=True)
    )
    blocked = np.zeros(len(events), dtype=bool)
    for i, (is_x, rel_ids, my_team) in enumerate(
        zip(applicable, events["_related_events"], events["team_id"], strict=True)
    ):
        if not is_x or not isinstance(rel_ids, list) or pd.isna(my_team):
            continue
        for r in rel_ids:
            info = lookup.get(r)
            if info is None:
                continue
            rtype, rteam = info
            if rtype == "Block" and pd.notna(rteam) and not same_id(my_team, rteam):
                blocked[i] = True
                break
    return _blocked_flag(len(events), applicable=applicable, blocked=blocked)
```

Then replace the deferred site (`statsbomb.py:282`). The `assert` makes the events↔actions
alignment the mask depends on fail LOUDLY if a future edit drops a row before this pre-filter,
pre-sort site (rather than silently misaligning one mask):

```python
    assert len(actions) == len(events)  # cross_blocked mask is events-aligned at this site
    actions["cross_blocked"] = _cross_blocked_flag(events)
```

- [ ] **Step 5: Flip the stale deferred assertion.** In `test_shot_blocked_true_on_real_blocked_shot` (`test_statsbomb.py:413`), replace the single line `assert actions["cross_blocked"].isna().all()` with the shipped contract for `7298` (the function already ran `actions = add_names(actions)` at `:407`, so `type_name` is populated — no new import):

```python
    _crosses = actions[actions["type_name"] == "cross"]
    assert _crosses["cross_blocked"].notna().all()  # open-play crosses carry True/False
    assert (_crosses["cross_blocked"] == False).all()  # 7298 has no blocked crosses  # noqa: E712
    assert actions[actions["type_name"] != "cross"]["cross_blocked"].isna().all()
```

- [ ] **Step 6: Run the statsbomb + block-detection suites.**

Run: `pytest tests/spadl/test_statsbomb.py tests/spadl/test_block_detection_contract.py -v`
Expected: PASS. `test_cross_blocked_is_subset_of_open_play_cross_type` now also gains StatsBomb non-NA teeth (7298 crosses are `False`, correctly typed `cross`) and still holds.

- [ ] **Step 7: Lint + type-check the converter change.**

Run: `python -m ruff check silly_kicks/spadl/statsbomb.py tests/spadl/test_statsbomb.py && python -m pyright silly_kicks/spadl/statsbomb.py`
Expected: clean.

---

## Task 5 (ship-path): Confirm no unintended cross-provider / reflection breakage

**Files:**
- Test only (no source change): `tests/spadl/`, `tests/invariants/`

- [ ] **Step 1: Run the reflection + cross-provider parity suites.**

Run: `pytest tests/spadl/test_block_detection_contract.py tests/spadl/test_cross_provider_parity.py tests/test_reflection.py -v`
Expected: PASS — `cross_blocked` is reflection-`"invariant"` (boolean), and only StatsBomb's value path changed (GS already real; opta/skillcorner/sportec/metrica/kloppy unchanged).

- [ ] **Step 2: Full offline suite green for the converter half.**

Run: `python -m pytest tests/spadl/ tests/invariants/ -m "not e2e" -q`
Expected: PASS.

---

## Task 6: Render script + enrollment + tests (Part 2 — runs regardless of Part 1)

**Files:**
- Create: `scripts/render_sb360_licensed_coverage.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (enroll in `_NOT_A_DRIVER`)
- Create: `tests/scripts/test_render_sb360_licensed_coverage.py`

**Interfaces:**
- Produces: `render(df: pd.DataFrame, meta: dict) -> str` (pure) and `main()` (reads committed dir, writes `coverage.md`). The pure `render` is what the smoke test calls.

- [ ] **Step 1: Write the render script.** Mirrors `render_sb360_matrix.py` (unguarded, docstring states why; `render()` pure, `main()` writes). Deterministic: round to 3 decimals, sort groupings.

```python
"""Render the licensed SB360 coverage parquet as `coverage.md`.

No provenance guard, deliberately -- same class as `render_sb360_matrix.py`: reads a COMMITTED
artifact (`coverage.parquet` + `manifest_all.json`) and writes a document. It does no corpus work
and consumes no external data, so a guard would add nothing and would make the report unrenderable
during the session that produces it. Provenance travels BY REFERENCE to the manifest it stamps.

Usage::

    python scripts/render_sb360_licensed_coverage.py
    python scripts/render_sb360_licensed_coverage.py --dir docs/research/sb360_licensed_coverage
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_DEFAULT_DIR = "docs/research/sb360_licensed_coverage"
_CAVEAT = (
    "_Battery numbers are STRUCTURAL coverage (did the aggregator run + fraction populated on real "
    "freeze-frames), NOT tactical values -- they are synthetic-input hybrids; a coverage fraction is "
    "a denominator, never a signal (ADR-042)._"
)


def _frame_coverage(df: pd.DataFrame) -> list[str]:
    fc = df[df["kind"] == "frame_coverage"]
    rows = []
    for subj, g in fc.groupby("subject"):
        actions = float(g["denominator"].sum())
        rate = (g["value"] * g["denominator"]).sum() / actions if actions else float("nan")
        rows.append((subj, int(g["match_id"].nunique()), int(actions), rate))
    rows.sort(key=lambda r: (r[0] != "all", -r[3]))
    out = ["## Frame-existence coverage (per GK-domain type)", "",
           "| Type | matches | actions | frame-existence |", "|---|---|---|---|"]
    for subj, m, a, rate in rows:
        out.append(f"| `{subj}` | {m} | {a} | {rate:.3f} |")
    return out + [""]


def _battery(df: pd.DataFrame) -> list[str]:
    bc = df[df["kind"] == "battery_column"]
    means = bc.groupby("subject")["value"].mean()
    full = sorted(means[means == 0.0].index)
    out = ["## Battery aggregator coverage", "", _CAVEAT, "",
           f"**{len(full)} of {means.shape[0]}** battery columns are fully-NaN across the corpus "
           "(mean populated fraction 0) -- the velocity-derived, ADR-063 Tier-2-suppressed, and "
           "constitutively-tracking columns. `add_visible_area_coverage`-style coverage fractions "
           "are denominators, not signals.", "",
           "<details><summary>The fully-NaN columns</summary>", ""]
    out += [f"- `{c}`" for c in full]
    out += ["", "</details>", ""]
    return out


def _companion(df: pd.DataFrame) -> list[str]:
    cs = df[df["kind"] == "companion_source"]
    cf = df[df["kind"] == "companion_fraction"]  # mean_observed_fraction per feature
    out = ["## ADR-062 visibility companions", "",
           "Per count feature: the source-token breakdown (row counts) and the mean observed "
           "fraction. _An observed fraction is a coverage denominator, not a signal (ADR-042)._ "
           "Fractions are UNWEIGHTED per-match means; the frame-existence table above is "
           "denominator-weighted -- do not cross-compare them as the same statistic.", "",
           "| Feature | source | total rows |", "|---|---|---|"]
    for subj, v in cs.groupby("subject")["value"].sum().sort_index().items():
        feature, _, token = subj.partition(".")
        out.append(f"| `{feature}` | `{token}` | {int(v)} |")
    out += ["", "| Feature | mean observed fraction |", "|---|---|"]
    for subj, v in cf.groupby("subject")["value"].mean().sort_index().items():
        out.append(f"| `{subj}` | {v:.3f} |")
    return out + [""]


def _pitch_roster_raises(df: pd.DataFrame) -> list[str]:
    pc = df[df["kind"] == "pitch_coverage"]["value"]
    pcs = df[df["kind"] == "pitch_coverage_source"]  # source-token counts (observed/no_polygon/…)
    ros = df[df["kind"] == "roster"]["value"]
    raises = df[df["kind"] == "battery_raises"]
    out = ["## Pitch coverage, roster, raises", "",
           f"- **Observed pitch fraction** (real `visible_area`): mean {pc.mean():.3f}, "
           f"min {pc.min():.3f}, max {pc.max():.3f} over {pc.shape[0]} matches. "
           "_A coverage denominator, not a signal._"]
    if not pcs.empty:
        toks = ", ".join(
            f"`{t}` {int(v)}" for t, v in pcs.groupby("subject")["value"].sum().sort_index().items()
        )
        out.append(f"- **Pitch-coverage source tokens** (summed rows): {toks}.")
    out.append(f"- **Roster keeper-resolution rate**: mean {ros.mean():.3f} over {ros.shape[0]} matches.")
    if not raises.empty:
        by = raises.groupby("subject")["match_id"].nunique().sort_index()
        out.append("- **Aggregators that raised** (an honest refusal, not a defect):")
        for subj, n in by.items():
            out.append(f"  - `{subj}`: {int(n)} matches "
                       "(freeze-frame carried only one team's players near the action).")
    return out + [""]


def render(df: pd.DataFrame, meta: dict) -> str:
    lines = [
        "# SB360 licensed-corpus coverage",
        "",
        "What the library produces on the **licensed** StatsBomb 360 corpus (30 matches). "
        "The companion to the open-data `../sb360_coverage/coverage.md`.",
        "",
        "## Provenance",
        "",
        "| | |", "|---|---|",
        f"| Driver | `scripts/validate_sb360_licensed_corpus.py` |",
        f"| Generation | `{meta['generation']}` |",
        f"| Matches | {meta['n_attempted']} attempted, {meta['n_failed']} failed |",
        f"| Commit | `{meta['run_commit']}` |",
        f"| Tree | {'dirty' if meta.get('run_tree_dirty') else 'clean'} |",
        "",
        "Rendered from the committed `coverage.parquet`; licensed data is never committed.",
        "",
    ]
    lines += _frame_coverage(df)
    lines += _battery(df)
    lines += _companion(df)
    lines += _pitch_roster_raises(df)
    lines += [
        "## The 40 -> 31 fully-NaN lift",
        "",
        "The 4.85.0 velocity-less lift (ADR-063) moved the fully-NaN battery count from **40** "
        "(prior state) to the **31** this parquet records: velocity-requiring pitch-control "
        "aggregators now serve the zero-velocity positional model on declared freeze-frames.",
        "",
        "## Reading limits / reproducing",
        "",
        "- The battery per-column numbers are structural coverage, not tactics (see the caveat above).",
        "- Licensed data is never committed. Refresh the parquet with "
        "`python scripts/validate_sb360_licensed_corpus.py` (owner token required), then re-render with "
        "`python scripts/render_sb360_licensed_coverage.py`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path(_DEFAULT_DIR))
    args = ap.parse_args()
    df = pd.read_parquet(args.dir / "coverage.parquet")
    meta = json.loads((args.dir / "manifest_all.json").read_text(encoding="utf-8"))
    out = args.dir / "coverage.md"
    out.write_text(render(df, meta), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Enroll the renderer as a non-driver.** In `tests/scripts/test_provenance_wiring.py`, add to `_NOT_A_DRIVER`:

```python
    "render_sb360_licensed_coverage": (
        "reads the COMMITTED licensed coverage parquet + manifest and writes coverage.md. It does "
        "no corpus work and consumes no external data, so a provenance guard would add nothing and "
        "would make the report unrenderable during the session that produces it. Same class as "
        "`render_sb360_matrix`; provenance travels by reference to the manifest it stamps."
    ),
```

- [ ] **Step 3: Write the smoke + staleness tests.**

```python
import json
import pathlib

import scripts.render_sb360_licensed_coverage as r

_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "research" / "sb360_licensed_coverage"


def test_render_emits_expected_sections():
    import pandas as pd
    df = pd.read_parquet(_DIR / "coverage.parquet")
    meta = json.loads((_DIR / "manifest_all.json").read_text(encoding="utf-8"))
    text = r.render(df, meta)
    for header in ("## Provenance", "## Frame-existence coverage", "## Battery aggregator coverage",
                   "## ADR-062 visibility companions", "## Pitch coverage, roster, raises",
                   "| Feature | mean observed fraction |"):  # pins the F2 companion_fraction sub-table
        assert header in text, f"missing section/table: {header}"


def test_coverage_md_stamps_manifest_generation():
    """Staleness guard (spec §4.1): a parquet refreshed out-of-band without a re-render is caught,
    without pinning any value."""
    assert (_DIR / "coverage.md").exists(), "coverage.md not rendered yet -- run the render script"
    gen = json.loads((_DIR / "manifest_all.json").read_text(encoding="utf-8"))["generation"]
    md = (_DIR / "coverage.md").read_text(encoding="utf-8")
    assert gen in md, f"coverage.md does not stamp manifest generation {gen} -- re-render"
```

- [ ] **Step 4: Run to verify the smoke test passes and the staleness test FAILS (coverage.md not rendered yet).**

Run: `pytest tests/scripts/test_render_sb360_licensed_coverage.py -v`
Expected: `test_render_emits_expected_sections` PASS; `test_coverage_md_stamps_manifest_generation` FAIL (assertion: "coverage.md not rendered yet", a clean AssertionError rather than FileNotFoundError).

---

## Task 7: Render `coverage.md` and confirm the guards

**Files:**
- Create: `docs/research/sb360_licensed_coverage/coverage.md`

- [ ] **Step 1: Render the doc on `.venv312`.**

Run: `python scripts/render_sb360_licensed_coverage.py`
Expected: writes `docs/research/sb360_licensed_coverage/coverage.md`. Eyeball: frame-existence table (`all`, `shot`, `cross`, `goalkick`, `keeper_save`, …), 31 fully-NaN listed, roster 1.000, pitch mean 0.273.

- [ ] **Step 2: Confirm both render tests and all three population gates pass.**

Run: `pytest tests/scripts/test_render_sb360_licensed_coverage.py tests/scripts/test_provenance_wiring.py tests/scripts/test_corpus_driver_resilience.py tests/scripts/test_input_contracts.py -v`
Expected: PASS (staleness guard now green; the renderer is enrolled in `_NOT_A_DRIVER`; the population-EXACT gate is satisfied).

- [ ] **Step 3: Lint the new script + test.**

Run: `python -m ruff check scripts/render_sb360_licensed_coverage.py tests/scripts/test_render_sb360_licensed_coverage.py && python -m ruff format --check scripts/render_sb360_licensed_coverage.py`
Expected: clean.

---

## Task 8: Docs, CHANGELOG, and version resolution (at completion)

**Files:**
- Modify: `CLAUDE.md` (ADR-046 block-detection bullet), `TODO.md`, `CHANGELOG.md`
- Modify (ship-path only, downstream pin): `docs/PRIVATE_CONSUMERS.md`
- Modify (at completion, only if Part 1 ships): the five version sites incl. `uv.lock`

- [ ] **Step 1: Update the CLAUDE.md ADR-046 bullet** — remove "StatsBomb `cross_blocked` deferred (n=1-verified, fragile `related_events` join)" and state, on the ship path, that StatsBomb now emits a real open-play-cross mask (opposing-team `Block` via `related_events`); on the keep-defer path, that it remains `pd.NA` with the measured reason recorded in `docs/research/sb360_cross_blocked/`.

- [ ] **Step 2: Add the TODO.md entry** for this cycle (mark the SB360 cross_blocked + licensed-coverage item done/updated).

- [ ] **Step 3: Add the CHANGELOG entry.** On the ship path, include the **Hyrum note**: StatsBomb `cross_blocked` flips from all-`pd.NA` to real values (additive; no silly-kicks retrain; a live-surface value change for downstream consumers). Plus the coverage-render + ADR amendment. On the keep-defer path: coverage render + ADR amendment only (no library change).

- [ ] **Step 4 (ship-path only): record the downstream pin.** Add a `docs/PRIVATE_CONSUMERS.md` entry if the lakehouse depends on the deferred all-`pd.NA` StatsBomb `cross_blocked` state (spec §7.3).

- [ ] **Step 5 (at completion): resolve the version.** ONLY if Part 1 ships a library change: take the next version from `main` (do not pre-write a number), and update all five sites incl. `uv.lock`; verify each site fresh. If keep-defer: no bump, no tag, no publish (spec §7.2).

- [ ] **Step 6: Full suite + lint + pyright green.**

Run: `python -m pytest tests/ -m "not e2e" -q && python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/`
Expected: PASS / clean. Then STOP and hand back for review before any commit.

---

## Self-review notes (spec coverage)

- Spec §3.1 probe → Task 1; §3.2 decision rule + §3.5 record/ADR → Task 2; §3.3 impl → Tasks 3–4; §3.4 verification → Tasks 4–5. §4 render → Tasks 6–7; §4.1 staleness guard → Task 6 Step 3 / Task 7 Step 2; §4.2.3 inline caveat → `render` `_CAVEAT` + the per-value labels. §7.2 versioning + §7.3 coordination → Task 8. §6 testing folded per-task.
- Keep-defer branch is explicit (Task 2 Step 4 gate) and leaves the library untouched; Part 2 is independent and unconditional.
- Type consistency: `_open_play_cross_mask` (Task 3) is the only cross predicate, consumed by `_cross_blocked_flag` (Task 4); `render(df, meta)` (Task 6) is called identically by the smoke test and `main()`.
