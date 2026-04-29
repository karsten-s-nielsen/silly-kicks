# WorldCup HDF5 fixture + e2e prediction tests in CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship silly-kicks 1.9.0 — restore actual CI coverage on the 5 `test_predict*` cases (currently silently skipping everywhere because `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` doesn't exist) by vendoring the HDF5 + committing a regenerator script (`scripts/build_worldcup_fixture.py`); also bundle a small DEFERRED.md → TODO.md tech-debt-section migration (National Park Principle) since we're already touching TODO.md.

**Architecture:** New build script under `scripts/` downloads StatsBomb open-data WorldCup-2018 raw events, runs them through `silly_kicks.spadl.statsbomb.convert_to_actions`, writes a multi-key HDFStore matching what `tests/conftest.py::sb_worldcup_data` expects (`games` table + per-game `actions/game_<id>` keys). Hexagonal — pure stdlib + pandas; no new dependencies. Raw events cached at `tests/datasets/statsbomb/raw/.cache/` (gitignored). HDF5 vendored.

**Tech Stack:** Python 3.10+ stdlib (`urllib.request`, `argparse`, `json`, `pathlib`), pandas, pytables (already in `[test]` extras). No new dependencies.

**Important — silly-kicks commit discipline (per `feedback_commit_policy` memory):**
- **Literally ONE commit per branch.** No per-task commits. All changes accumulate locally; the final task commits everything as a single squash-ready commit.
- The `git commit` step is at the end of the plan (Task 11), not per-task.
- Branch name: `feat/worldcup-hdf5-e2e-prediction-tests`.
- Hook scope (narrowed during PR-S8): only `git commit` and destructive ops are sentinel-gated. Routine push, PR ops, tag pushes proceed on chat approval alone.
- User approval gates apply at: (a) the single commit (sentinel), (b) push, (c) PR open, (d) PR merge, (e) tag push. Stop and ask before each.

**Cross-version pin (per `feedback_ci_cross_version` memory):** before running any verification gate, install the EXACT CI pin so local pyright/ruff matches CI bit-for-bit:
```
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `scripts/build_worldcup_fixture.py` | Create (~180 LOC) | Downloads matches manifest + 64 raw event JSONs (cached), converts each via `statsbomb.convert_to_actions`, builds `games` metadata DataFrame, writes HDFStore with `games` + per-game `actions/game_<id>` keys. CLI: `--output`, `--cache-dir`, `--no-cache`, `--verbose`, `--quiet`. |
| `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` | Create (~10-30 MB) | Vendored fixture. Multi-key HDFStore: `games` + `actions/game_<game_id>` per match. |
| `tests/conftest.py` | Modify (~+5 / -3 lines) | `pytest.skip(...)` → `pytest.fail(...)` with build-script reference (matches PR-S8 `pytest.fail` pattern for committed fixtures). |
| `tests/vaep/test_vaep.py` | Modify (-2 lines) | Drop `@pytest.mark.e2e` on `test_predict` (line 71) and `test_predict_with_missing_features` (line 81). |
| `tests/test_xthreat.py` | Modify (-2 lines) | Drop `@pytest.mark.e2e` on `test_predict` (line 218) and `test_predict_with_interpolation` (line 228). |
| `tests/atomic/test_atomic_vaep.py` | Modify (-1 line) | Drop `@pytest.mark.e2e` on `test_predict` (line 23). |
| `.gitignore` | Modify (+3 lines) | Add `tests/datasets/statsbomb/raw/.cache/` exclusion + comment. |
| `.github/workflows/ci.yml` | Modify (+0 / -0 net, edit 2 lines) | Add `scripts/` to ruff lint paths. |
| `pyproject.toml` | Modify (+1 / -1) | Version `1.8.0` → `1.9.0`. |
| `CHANGELOG.md` | Modify (+50 lines) | New `## [1.9.0] — 2026-04-29` entry. |
| `TODO.md` | Modify (~+15 / -7) | Remove PR-S9 line from `## Open PRs`; add `## Tech Debt` section with 4 migrated items (A19, D-9, O-M1, O-M6). |
| `CLAUDE.md` | Modify (+0 / -1) | Remove `Audit history in [docs/DEFERRED.md]...` reference. |
| `docs/DEFERRED.md` | Delete | History preserved in `git log -- docs/DEFERRED.md`. |
| `docs/superpowers/specs/2026-04-29-worldcup-hdf5-e2e-prediction-tests-design.md` | Already exists | Bundled into the single commit. |
| `docs/superpowers/plans/2026-04-29-worldcup-hdf5-e2e-prediction-tests.md` | This file | Bundled into the single commit. |

---

## Task 0: Pre-flight — verify clean baseline

**Files:** None (verification only).

- [ ] **Step 1: Verify clean working tree on branch `main`**

```bash
git status
git log --oneline -1
grep "^version" pyproject.toml
```

Expected output:
- `git status`: clean working tree (allowed untracked: `README.md.backup`, `uv.lock`).
- `git log`: `96ea786 feat(spadl): public boundary_metrics utility + recall-based add_possessions CI gate — silly-kicks 1.8.0 (#13)`.
- `pyproject.toml`: `version = "1.8.0"`.

If any of these don't match, STOP and surface to the user.

- [ ] **Step 2: Create feature branch**

```bash
git checkout -b feat/worldcup-hdf5-e2e-prediction-tests
```

- [ ] **Step 3: Install exact CI pin**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: clean install, no version conflicts.

- [ ] **Step 4: Run baseline tests**

```bash
uv run pytest tests/ --tb=short -q 2>&1 | tail -10
```

Expected: 555 passed, 9 skipped (post-1.8.0 baseline). The 5 prediction tests skip (PR-S9 territory). No failures.

- [ ] **Step 5: Run baseline lint + pyright**

```bash
uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/
```

Expected: zero errors from each. If any fail, STOP — fix the baseline first.

---

## Task 1: Write `scripts/build_worldcup_fixture.py`

**Files:**
- Create: `scripts/build_worldcup_fixture.py`

- [ ] **Step 1: Verify scripts/ directory does not yet exist**

```bash
ls scripts/ 2>/dev/null || echo "(does not exist — will be created)"
```

Expected: `(does not exist — will be created)`. If the directory already exists, surface to the user before continuing.

- [ ] **Step 2: Create `scripts/build_worldcup_fixture.py`**

Write the full script content below to `scripts/build_worldcup_fixture.py`:

```python
"""Build the WorldCup-2018 SPADL fixture for silly-kicks e2e prediction tests.

Downloads StatsBomb open-data raw events for the 64 FIFA World Cup 2018 matches,
converts each via :func:`silly_kicks.spadl.statsbomb.convert_to_actions`, and
writes a multi-key HDFStore at ``tests/datasets/statsbomb/spadl-WorldCup-2018.h5``
with the structure expected by ``tests/conftest.py::sb_worldcup_data``:

- ``games`` table — per-match metadata (game_id, home_team_id, away_team_id, scores,
  competition_id, season_id, kick_off, match_date)
- ``actions/game_<game_id>`` keys — per-match SPADL action DataFrames

Raw event JSONs are cached at ``tests/datasets/statsbomb/raw/.cache/`` (gitignored)
so re-runs are fast. The cache is keyed by match_id only — if StatsBomb updates
a match upstream, the cached file stays stale until ``--no-cache`` or manual
deletion. Acceptable for a vendored fixture.

Usage:

    # Default: write to tests/datasets/statsbomb/spadl-WorldCup-2018.h5,
    # cache raw events at tests/datasets/statsbomb/raw/.cache/
    python scripts/build_worldcup_fixture.py --verbose

    # Custom output path
    python scripts/build_worldcup_fixture.py --output /tmp/wc.h5

    # Force re-download (ignore cache)
    python scripts/build_worldcup_fixture.py --no-cache

License: StatsBomb open-data is non-commercial. The vendored HDF5 is a derivative
work covered by the same license. Attribution lives in
``tests/datasets/statsbomb/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

# Late import to avoid silly-kicks import cost when running --help.
# Imported lazily inside main().

# StatsBomb open-data WorldCup-2018: competition_id=43, season_id=3.
_COMPETITION_ID = 43
_SEASON_ID = 3
_OPEN_DATA_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_MANIFEST_URL = f"{_OPEN_DATA_BASE}/matches/{_COMPETITION_ID}/{_SEASON_ID}.json"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "spadl-WorldCup-2018.h5"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / ".cache"

_MANIFEST_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_SECONDS = (1, 2, 4)
_HDF_COMPLIB = "zlib"
_HDF_COMPLEVEL = 9
_HDF_SIZE_WARN_BYTES = 50 * 1024 * 1024  # 50 MB
_MIN_ACTIONS_PER_MATCH = 100  # sanity floor

_TOP_LEVEL_KEYS = frozenset({"id", "period", "timestamp", "team", "player", "type", "location"})


def _log(message: str, *, level: str = "info", verbose: bool = False, quiet: bool = False) -> None:
    """Logger. ``verbose`` adds info/debug messages; ``quiet`` suppresses info."""
    if level == "error":
        print(f"ERROR: {message}", file=sys.stderr)
    elif level == "warn":
        print(f"WARN: {message}", file=sys.stderr)
    elif level == "info" and not quiet:
        print(message)
    elif level == "debug" and verbose:
        print(f"  {message}")


def _fetch_url(url: str, *, retries: int = _DOWNLOAD_RETRIES) -> bytes:
    """Fetch a URL with retry-with-backoff on transient failures.

    Raises urllib.error.URLError after ``retries`` attempts.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(_DOWNLOAD_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last_error}")


def _fetch_manifest(cache_dir: Path, *, no_cache: bool, verbose: bool, quiet: bool) -> list[dict[str, Any]]:
    """Fetch the WorldCup-2018 matches manifest. Cached for 7 days unless --no-cache."""
    cache_path = cache_dir / "manifest-43-3.json"
    if not no_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < _MANIFEST_CACHE_MAX_AGE_SECONDS:
            _log(f"manifest: cache hit ({cache_path.relative_to(_REPO_ROOT)}, age={age / 3600:.1f}h)",
                 level="debug", verbose=verbose, quiet=quiet)
            return json.loads(cache_path.read_text(encoding="utf-8"))

    _log(f"manifest: downloading from {_MANIFEST_URL}", level="info", verbose=verbose, quiet=quiet)
    raw = _fetch_url(_MANIFEST_URL)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def _fetch_raw_events(match_id: int, cache_dir: Path, *, no_cache: bool, verbose: bool, quiet: bool) -> list[dict[str, Any]]:
    """Fetch raw events JSON for a match. Cached indefinitely unless --no-cache."""
    events_dir = cache_dir / "events"
    cache_path = events_dir / f"{match_id}.json"
    if not no_cache and cache_path.exists():
        _log(f"  match {match_id}: cache hit", level="debug", verbose=verbose, quiet=quiet)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = f"{_OPEN_DATA_BASE}/events/{match_id}.json"
    raw = _fetch_url(url)
    events_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    size_kb = len(raw) / 1024
    _log(f"  match {match_id}: downloaded {size_kb:.0f} KB", level="debug", verbose=verbose, quiet=quiet)
    return json.loads(raw.decode("utf-8"))


def _adapt_events_to_silly_kicks_input(events: list[dict[str, Any]], match_id: int) -> pd.DataFrame:
    """Adapt StatsBomb open-data event JSON to silly-kicks's expected input shape.

    Same adapter pattern used in tests/spadl/test_add_possessions.py.
    """
    return pd.DataFrame([
        {
            "game_id": match_id,
            "event_id": e.get("id"),
            "period_id": e.get("period"),
            "timestamp": e.get("timestamp"),
            "team_id": (e.get("team") or {}).get("id"),
            "player_id": (e.get("player") or {}).get("id"),
            "type_name": (e.get("type") or {}).get("name"),
            "location": e.get("location"),
            "extra": {k: v for k, v in e.items() if k not in _TOP_LEVEL_KEYS},
        }
        for e in events
    ])


def _build_games_metadata_df(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the ``games`` metadata DataFrame from the matches manifest."""
    rows = []
    for m in matches:
        rows.append({
            "game_id": int(m["match_id"]),
            "home_team_id": int(m["home_team"]["home_team_id"]),
            "away_team_id": int(m["away_team"]["away_team_id"]),
            "home_score": int(m["home_score"]) if m.get("home_score") is not None else 0,
            "away_score": int(m["away_score"]) if m.get("away_score") is not None else 0,
            "kick_off": m.get("kick_off", ""),
            "match_date": m.get("match_date", ""),
            "competition_id": int(m["competition"]["competition_id"]),
            "season_id": int(m["season"]["season_id"]),
            "competition_name": m["competition"].get("competition_name", ""),
        })
    df = pd.DataFrame(rows).sort_values("game_id", kind="mergesort").reset_index(drop=True)
    return df


def _convert_match(
    match: dict[str, Any],
    cache_dir: Path,
    *,
    no_cache: bool,
    verbose: bool,
    quiet: bool,
) -> tuple[int, pd.DataFrame]:
    """Download + convert a single match. Returns (match_id, spadl_actions_df)."""
    from silly_kicks.spadl import statsbomb

    match_id = int(match["match_id"])
    home_team_id = int(match["home_team"]["home_team_id"])

    events = _fetch_raw_events(match_id, cache_dir, no_cache=no_cache, verbose=verbose, quiet=quiet)
    adapted = _adapt_events_to_silly_kicks_input(events, match_id)
    actions, _report = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id, preserve_native=None)
    return match_id, actions


def _validate_output(games: pd.DataFrame, actions_by_id: dict[int, pd.DataFrame], output_path: Path,
                     *, verbose: bool, quiet: bool) -> None:
    """Validate the produced HDF5: structure + sanity floors."""
    games_ids = set(games["game_id"].to_list())
    actions_ids = set(actions_by_id.keys())
    missing_in_actions = games_ids - actions_ids
    missing_in_games = actions_ids - games_ids
    if missing_in_actions:
        raise RuntimeError(
            f"validation: {len(missing_in_actions)} games in metadata table have no actions: "
            f"{sorted(missing_in_actions)[:10]}{'...' if len(missing_in_actions) > 10 else ''}"
        )
    if missing_in_games:
        raise RuntimeError(
            f"validation: {len(missing_in_games)} action keys have no metadata row: "
            f"{sorted(missing_in_games)[:10]}{'...' if len(missing_in_games) > 10 else ''}"
        )

    short_matches = [(mid, len(df)) for mid, df in actions_by_id.items() if len(df) < _MIN_ACTIONS_PER_MATCH]
    if short_matches:
        raise RuntimeError(
            f"validation: {len(short_matches)} matches have fewer than {_MIN_ACTIONS_PER_MATCH} actions "
            f"(suspicious): {short_matches[:5]}{'...' if len(short_matches) > 5 else ''}"
        )

    file_size_bytes = output_path.stat().st_size
    if file_size_bytes > _HDF_SIZE_WARN_BYTES:
        _log(
            f"output file size {file_size_bytes / 1024 / 1024:.1f} MB exceeds {_HDF_SIZE_WARN_BYTES / 1024 / 1024:.0f} MB warn threshold",
            level="warn", verbose=verbose, quiet=quiet,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"HDF5 output path (default: {_DEFAULT_OUTPUT.relative_to(_REPO_ROOT)})")
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR,
                        help=f"Raw event cache dir (default: {_DEFAULT_CACHE_DIR.relative_to(_REPO_ROOT)})")
    parser.add_argument("--no-cache", action="store_true", help="Re-download all files even if cached")
    parser.add_argument("--verbose", action="store_true", help="Per-match progress logging")
    parser.add_argument("--quiet", action="store_true", help="Errors only")
    args = parser.parse_args()

    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")

    output_path: Path = args.output.resolve()
    cache_dir: Path = args.cache_dir.resolve()
    verbose: bool = args.verbose
    quiet: bool = args.quiet

    _log(f"output:    {output_path}", level="info", verbose=verbose, quiet=quiet)
    _log(f"cache:     {cache_dir}", level="info", verbose=verbose, quiet=quiet)
    _log(f"no_cache:  {args.no_cache}", level="info", verbose=verbose, quiet=quiet)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()

    matches = _fetch_manifest(cache_dir, no_cache=args.no_cache, verbose=verbose, quiet=quiet)
    if len(matches) != 64:
        _log(f"manifest contains {len(matches)} matches (expected 64) — proceeding anyway",
             level="warn", verbose=verbose, quiet=quiet)

    matches.sort(key=lambda m: int(m["match_id"]))

    actions_by_id: dict[int, pd.DataFrame] = {}
    skipped: list[tuple[int, str]] = []

    for i, match in enumerate(matches, 1):
        match_id = int(match["match_id"])
        _log(f"[{i}/{len(matches)}] match {match_id}", level="info", verbose=verbose, quiet=quiet)
        try:
            mid, actions = _convert_match(match, cache_dir, no_cache=args.no_cache, verbose=verbose, quiet=quiet)
            actions_by_id[mid] = actions
        except Exception as e:
            _log(f"match {match_id}: conversion failed: {e}", level="error", verbose=verbose, quiet=quiet)
            skipped.append((match_id, str(e)))

    if skipped:
        _log(f"{len(skipped)} matches were skipped due to errors", level="error", verbose=verbose, quiet=quiet)
        for mid, err in skipped:
            _log(f"  match {mid}: {err}", level="error", verbose=verbose, quiet=quiet)
        return 1

    games_df = _build_games_metadata_df(matches)
    games_df = games_df[games_df["game_id"].isin(actions_by_id.keys())].reset_index(drop=True)

    _log(f"writing HDFStore: {len(actions_by_id)} matches, {sum(len(df) for df in actions_by_id.values())} total actions",
         level="info", verbose=verbose, quiet=quiet)

    with pd.HDFStore(str(output_path), mode="w", complib=_HDF_COMPLIB, complevel=_HDF_COMPLEVEL) as store:
        store.put("games", games_df, format="table")
        for game_id, actions_df in actions_by_id.items():
            store.put(f"actions/game_{game_id}", actions_df, format="table")

    _validate_output(games_df, actions_by_id, output_path, verbose=verbose, quiet=quiet)

    elapsed = time.time() - start_ts
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    _log(
        f"done: {len(actions_by_id)} matches, {file_size_mb:.1f} MB HDF5, {elapsed:.0f}s",
        level="info", verbose=verbose, quiet=quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the script's `--help` works**

```bash
uv run python scripts/build_worldcup_fixture.py --help 2>&1 | head -15
```

Expected: argparse help output. If syntax errors, surface and fix.

---

## Task 2: Run the build script — generate the WorldCup HDF5

**Files:**
- Generated: `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`
- Generated: `tests/datasets/statsbomb/raw/.cache/manifest-43-3.json`
- Generated: `tests/datasets/statsbomb/raw/.cache/events/<match_id>.json` × 64

- [ ] **Step 1: Run the build script in verbose mode**

```bash
uv run python scripts/build_worldcup_fixture.py --verbose
```

Use `run_in_background=true` because cold-cache run takes 3-8 minutes. Poll the output periodically.

Expected progress shape:
```
output:    .../tests/datasets/statsbomb/spadl-WorldCup-2018.h5
cache:     .../tests/datasets/statsbomb/raw/.cache
no_cache:  False
manifest: downloading from https://raw.githubusercontent.com/.../matches/43/3.json
[1/64] match <id1>
  match <id1>: downloaded ~XX KB
[2/64] match <id2>
  match <id2>: downloaded ~XX KB
... (62 more matches)
writing HDFStore: 64 matches, ~95000 total actions
done: 64 matches, ~XX.X MB HDF5, ~NNNs
```

If any individual match fails (network or conversion), the script exits 1 and lists skipped matches. Surface to user — investigate before retrying.

- [ ] **Step 2: Verify the HDF5 file**

```bash
ls -lh tests/datasets/statsbomb/spadl-WorldCup-2018.h5
uv run python -c "
import pandas as pd
with pd.HDFStore('tests/datasets/statsbomb/spadl-WorldCup-2018.h5', mode='r') as store:
    games = store['games']
    print(f'games table: {len(games)} rows, columns: {list(games.columns)}')
    keys = [k for k in store.keys() if k.startswith('/actions/game_')]
    print(f'actions keys: {len(keys)}')
    sample_actions = store[keys[0]]
    print(f'sample actions ({keys[0]}): {len(sample_actions)} rows, columns: {list(sample_actions.columns)}')
"
```

Expected:
- File size 10-30 MB.
- `games table: 64 rows, columns: ['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'kick_off', 'match_date', 'competition_id', 'season_id', 'competition_name']`
- `actions keys: 64`
- Sample actions: ~1500-2500 rows with the standard SPADL columns (`game_id`, `period_id`, `time_seconds`, `team_id`, `player_id`, `start_x`, ..., `bodypart_id`).

If file size > 50 MB, see risks section in spec — switch to blosc compression or report.

- [ ] **Step 3: Verify the cache directory layout**

```bash
ls tests/datasets/statsbomb/raw/.cache/ && ls tests/datasets/statsbomb/raw/.cache/events/ | head -5
```

Expected: `manifest-43-3.json` + `events/` subdir containing 64 JSON files. These are NOT committed (gitignored in Task 6).

---

## Task 3: Update conftest — `pytest.skip` → `pytest.fail`

**Files:**
- Modify: `tests/conftest.py:17-26`

- [ ] **Step 1: Replace the fixture body**

In `tests/conftest.py`, find:

```python
@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5")
    if not os.path.exists(hdf_file):
        pytest.skip(
            f"e2e dataset {hdf_file!r} not present — download required to run e2e-marked tests "
            f"(StatsBomb WorldCup-2018 SPADL fixture)."
        )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()
```

Replace with:

```python
@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5")
    if not os.path.exists(hdf_file):
        pytest.fail(
            f"WorldCup-2018 SPADL fixture not found at {hdf_file!r}. "
            f"This fixture is committed to the repo as of silly-kicks 1.9.0. "
            f"If absent, regenerate via: python scripts/build_worldcup_fixture.py "
            f"(or check for accidental .gitignore exclusion)."
        )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()
```

---

## Task 4: Drop `@pytest.mark.e2e` markers on the 5 prediction tests

**Files:**
- Modify: `tests/vaep/test_vaep.py:71` and `:81`
- Modify: `tests/test_xthreat.py:218` and `:228`
- Modify: `tests/atomic/test_atomic_vaep.py:23`

- [ ] **Step 1: Remove markers in `tests/vaep/test_vaep.py`**

Find:

```python
@pytest.mark.e2e
def test_predict(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
```

Replace with:

```python
def test_predict(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
```

Then find:

```python
@pytest.mark.e2e
def test_predict_with_missing_features(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
```

Replace with:

```python
def test_predict_with_missing_features(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
```

- [ ] **Step 2: Remove markers in `tests/test_xthreat.py`**

Find:

```python
@pytest.mark.e2e
def test_predict(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
```

Replace with:

```python
def test_predict(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
```

Then find:

```python
@pytest.mark.e2e
def test_predict_with_interpolation(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
```

Replace with:

```python
def test_predict_with_interpolation(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
```

- [ ] **Step 3: Remove marker in `tests/atomic/test_atomic_vaep.py`**

Find:

```python
@pytest.mark.e2e
def test_predict(sb_worldcup_data: pd.HDFStore) -> None:
```

Replace with:

```python
def test_predict(sb_worldcup_data: pd.HDFStore) -> None:
```

---

## Task 5: Run the prediction tests — confirm they pass with the new fixture

**Files:** None (verification only).

- [ ] **Step 1: Run only the 5 newly-unmarked prediction tests**

```bash
uv run pytest tests/vaep/test_vaep.py::test_predict tests/vaep/test_vaep.py::test_predict_with_missing_features tests/test_xthreat.py::test_predict tests/test_xthreat.py::test_predict_with_interpolation tests/atomic/test_atomic_vaep.py::test_predict -v --tb=short
```

Use `run_in_background=true` because the session-scoped `vaep_model` and `xt_model` fixtures train models (xgboost VAEP fit takes 20-60 sec; xT fit a few sec). Total ~1-2 min.

Expected: all 5 tests pass.

If any test fails:
- This is a real previously-latent bug (these tests have not actually run for ~9 release cycles per the spec).
- STOP. Surface the failure to the user with full traceback and the test source.
- Decide inline: (a) small fix (1-2 hours) → bundle into PR-S9; (b) larger fix → defer to a focused follow-up PR, restore the e2e markers in this PR, document the deferred fix in CHANGELOG.

- [ ] **Step 2: Run the full pytest suite**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -10
```

Expected: ~560 passed, ~4 skipped. Compared to baseline (555/9): +5 (the 5 prediction tests now pass), -5 from skipped (those same 5 no longer skip).

If the math doesn't match, surface and investigate.

---

## Task 6: Update `.gitignore` — add cache exclusion

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add cache directory exclusion**

In `.gitignore`, find:

```
# Downloaded test datasets (not committed)
tests/datasets/wyscout_public/
```

Replace with:

```
# Downloaded test datasets (not committed)
tests/datasets/wyscout_public/

# StatsBomb open-data raw events cache for scripts/build_worldcup_fixture.py
tests/datasets/statsbomb/raw/.cache/
```

- [ ] **Step 2: Verify cache files are excluded from git**

```bash
git status --short tests/datasets/statsbomb/raw/.cache/ 2>&1 | head -3
```

Expected: empty output (gitignore working — directory not tracked).

```bash
git status --short tests/datasets/statsbomb/spadl-WorldCup-2018.h5
```

Expected: `?? tests/datasets/statsbomb/spadl-WorldCup-2018.h5` (new file ready to be added).

---

## Task 7: Update CI workflow — add `scripts/` to lint paths

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Update both ruff invocations**

In `.github/workflows/ci.yml`, find:

```yaml
      - run: ruff check silly_kicks/ tests/
      - run: ruff format --check silly_kicks/ tests/
```

Replace with:

```yaml
      - run: ruff check silly_kicks/ tests/ scripts/
      - run: ruff format --check silly_kicks/ tests/ scripts/
```

- [ ] **Step 2: Verify locally with the same invocation**

```bash
uv run ruff check silly_kicks/ tests/ scripts/
uv run ruff format --check silly_kicks/ tests/ scripts/
```

Expected: zero errors, "65+ files already formatted" (slightly higher count than 1.8.0's 65 because we added a script file).

---

## Task 8: Version bump + CHANGELOG entry

**Files:**
- Modify: `pyproject.toml:7`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, find:

```toml
version = "1.8.0"
```

Replace with:

```toml
version = "1.9.0"
```

- [ ] **Step 2: Add CHANGELOG entry**

In `CHANGELOG.md`, find:

```markdown
## [1.8.0] — 2026-04-29
```

Insert this new entry directly above it:

```markdown
## [1.9.0] — 2026-04-29

### Added
- **Vendored `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** — committed
  HDF5 fixture for the FIFA World Cup 2018 (64 matches, all 5 prediction
  pipeline tests in `tests/vaep/`, `tests/test_xthreat.py`, and
  `tests/atomic/` now run on every PR + push). Pre-1.9.0 these tests
  silently skipped in CI and locally because the fixture was never
  committed. Net: ~9 release cycles of zero coverage on the prediction
  pipeline (VAEP fit/rate, xT fit/rate, atomic VAEP fit/rate) is now
  closed.
- **`scripts/build_worldcup_fixture.py`** — reproducible HDF5 generator.
  Downloads StatsBomb open-data WorldCup-2018 raw events (cached at
  `tests/datasets/statsbomb/raw/.cache/`, gitignored), converts each via
  `silly_kicks.spadl.statsbomb.convert_to_actions`, writes the multi-key
  HDFStore. CLI: `--output`, `--cache-dir`, `--no-cache`, `--verbose`,
  `--quiet`. Cold-cache run: 3-8 min. Warm-cache re-run: ~30 sec. No new
  dependencies (stdlib + pandas + already-present pytables).
- **`scripts/` is now linted in CI** — `.github/workflows/ci.yml` runs
  `ruff check` and `ruff format --check` on `silly_kicks/`, `tests/`,
  AND `scripts/`. Pyright include stays `silly_kicks/` only — build
  scripts aren't worth full type-checking.

### Changed
- **`tests/conftest.py::sb_worldcup_data` calls `pytest.fail` instead of
  `pytest.skip` when the HDF5 is absent.** Matches the PR-S8 pattern for
  committed fixtures: once a fixture is committed, "missing" is a
  packaging error worth surfacing prominently — not a silent skip that
  lets CI quietly regress. Failure message points at the build script
  for regeneration.
- The 5 `test_predict*` cases (`tests/vaep/test_vaep.py::test_predict`,
  `tests/vaep/test_vaep.py::test_predict_with_missing_features`,
  `tests/test_xthreat.py::test_predict`,
  `tests/test_xthreat.py::test_predict_with_interpolation`,
  `tests/atomic/test_atomic_vaep.py::test_predict`) no longer carry the
  `@pytest.mark.e2e` marker. They run in the regular suite on every CI
  matrix slot (4 slots, ~5-15 sec overhead per slot — negligible).

### Documentation
- **`docs/DEFERRED.md` deleted; live items migrated to a new `## Tech
  Debt` section in `TODO.md`.** Per the National Park Principle —
  bundle the cleanup of the rotting parallel doc into this cycle since
  we're already touching `TODO.md` anyway. Audit history preserved in
  `git log -- docs/DEFERRED.md`. Migrated items: A19 (default
  hyperparameters scattered), D-9 (5 xthreat module-level functions
  naming), O-M1 (StatsBomb `events.copy()`), O-M6 (StatsBomb fidelity
  version check temporary DataFrame). Items judged "by design / accept"
  and not migrated: A15 (kloppy LSP differs by design), A16 (no plugin
  registry — YAGNI for 4 converters), A17 (`_fit_*` coupling — partial
  refactor done, diminishing returns), S5 (optional ML deps no upper
  bounds — librarian convention).
- `CLAUDE.md` no longer references `docs/DEFERRED.md` (file removed).

### Notes
- WorldCup HDF5 file size: ~XX MB (measured during build — record
  observed value in commit message). Well under GitHub's 50 MB soft /
  100 MB hard thresholds. No Git LFS needed.
- The `tests/datasets/statsbomb/raw/.cache/` directory is gitignored —
  raw event JSONs (~192 MB total) are downloaded on demand by the build
  script and never committed.

```

(The `~XX MB` placeholder gets filled in during Task 11 with the actual
measured size from the build run.)

---

## Task 9: DEFERRED.md migration → TODO.md tech debt + CLAUDE.md edit

**Files:**
- Modify: `TODO.md` (remove PR-S9 line; add `## Tech Debt` section)
- Modify: `CLAUDE.md` (remove DEFERRED.md reference)
- Delete: `docs/DEFERRED.md`

- [ ] **Step 1: Update `TODO.md` — remove PR-S9 line, add Tech Debt section**

In `TODO.md`, find the `## Open PRs` section (added in PR-S8):

```markdown
## Open PRs

| # | Size | Item | Context |
|---|------|------|---------|
| PR-S9 | Medium | e2e prediction tests in CI via WorldCup HDF5 generation | Generate `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` from open-data raw events (~64 matches × ~3 MB). Output structure: `games` table + `actions/game_<id>` per match (see `tests/vaep/test_vaep.py:48` for shape contract). Drop `@pytest.mark.e2e` on the 5 `test_predict*` cases (`tests/vaep/test_vaep.py:72,82`, `tests/test_xthreat.py:219,229`, `tests/atomic/test_atomic_vaep.py:24`). Conversion script committed at `scripts/build_worldcup_fixture.py`. Estimated 1-2 hours. See `docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md` § 10 for design notes. |
| PR-S10 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the new `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. Best done AFTER PR-S9 — 64-match WorldCup fixture is more reliable than PR-S8's 3-match set for parameter-tuning. |
```

Replace with (PR-S9 line removed, plus a new `## Tech Debt` section appended):

```markdown
## Open PRs

| # | Size | Item | Context |
|---|------|------|---------|
| PR-S10 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the new `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. The 64-match WorldCup HDF5 from PR-S9 is now available for parameter sweeping (vs PR-S8's 3-match set). |

## Tech Debt

| # | Sev | Item | Context |
|---|-----|------|---------|
| A19 | Low | Default hyperparameters scattered across 3 learner functions | Extracted to named constants in `learners.py`; could centralize further but low impact. Audit-source: DEFERRED.md (Phase 2 architecture audit, migrated 1.9.0). |
| D-9 | Low | 5 xthreat module-level functions (`scoring_prob`, `get_move_actions`, etc.) not underscore-prefixed but not re-exported | Implementation helpers technically public API. Audit-source: DEFERRED.md (migrated 1.9.0). |
| O-M1 | Low | Full `events.copy()` at top of StatsBomb `convert_to_actions` (`spadl/statsbomb.py:78`) | Defensive copy — could shrink on demand. Audit-source: DEFERRED.md (migrated 1.9.0). |
| O-M6 | Low | Temporary n×3 DataFrame for StatsBomb fidelity version check (`spadl/statsbomb.py:171`) | Audit-source: DEFERRED.md (migrated 1.9.0). |
```

- [ ] **Step 2: Update `CLAUDE.md` — remove DEFERRED.md reference**

In `CLAUDE.md`, find:

```markdown
See [TODO.md](TODO.md) for tracked work. Audit history in [docs/DEFERRED.md](docs/DEFERRED.md).
```

Replace with:

```markdown
See [TODO.md](TODO.md) for tracked work.
```

- [ ] **Step 3: Delete `docs/DEFERRED.md`**

```bash
git rm docs/DEFERRED.md
```

Expected: `rm 'docs/DEFERRED.md'`.

- [ ] **Step 4: Verify the file references are consistent**

```bash
grep -rn "DEFERRED" --include="*.md" --include="*.py" --include="*.toml" --include="*.yml" 2>&1 | head -10
```

Expected: zero matches in tracked files. If any, surface and fix.

---

## Task 10: Verification gates — full local CI parity

**Files:** None (verification only).

- [ ] **Step 1: Re-confirm exact CI pin (in case shells were restarted)**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

- [ ] **Step 2: Ruff lint (now includes scripts/)**

```bash
uv run ruff check silly_kicks/ tests/ scripts/
```

Expected: zero errors. If any, fix and re-run.

- [ ] **Step 3: Ruff format check (now includes scripts/)**

```bash
uv run ruff format --check silly_kicks/ tests/ scripts/
```

Expected: zero formatting issues. If any, run `uv run ruff format silly_kicks/ tests/ scripts/` and re-check.

- [ ] **Step 4: Pyright (still silly_kicks/ only)**

```bash
uv run pyright silly_kicks/
```

Expected: zero errors.

- [ ] **Step 5: Full pytest suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -50
```

Expected: ~560 passed, ~4 skipped. Specifically the 5 `test_predict*` tests pass (no longer skip).

If pass/skip count is off, surface and investigate before commit.

- [ ] **Step 6: Update CHANGELOG with measured HDF5 size**

In `CHANGELOG.md` Notes section, replace the `~XX MB` placeholder with the actual measured size from Task 2's verification (e.g., `~12 MB`). Get the exact value via:

```bash
ls -lh tests/datasets/statsbomb/spadl-WorldCup-2018.h5 | awk '{print $5}'
```

---

## Task 11: `/final-review` skill + single commit + 5 user-gated steps

**Files:** All accumulated changes commit together.

- [ ] **Step 1: Run /final-review skill**

Invoke `mad-scientist-skills:final-review` (per `feedback_commit_policy` memory: mandatory before commit).

Expected: passes with no critical issues. Address any flagged issues inline. The C4 architecture diagram regen is part of /final-review — although this PR doesn't change architecture (the build script is a one-off, not a deployable component), confirm whether the spadl container description needs a minor refresh.

- [ ] **Step 2: Stage all changes — explicit user approval gate (sentinel-required)**

STOP HERE. Surface to the user the full diff/file list and the proposed commit message:
> "All implementation tasks complete + verification gates green + /final-review passed. Ready to stage + commit. Approve?"

Wait for user approval AND sentinel touch (`!touch ~/.claude-git-approval`). The narrowed hook (per PR-S8) gates `git commit` only; subsequent gates are chat-only.

- [ ] **Step 3: Stage + commit (single commit, on user approval)**

```bash
git add scripts/build_worldcup_fixture.py tests/datasets/statsbomb/spadl-WorldCup-2018.h5 tests/conftest.py tests/vaep/test_vaep.py tests/test_xthreat.py tests/atomic/test_atomic_vaep.py .gitignore .github/workflows/ci.yml pyproject.toml CHANGELOG.md TODO.md CLAUDE.md docs/superpowers/specs/2026-04-29-worldcup-hdf5-e2e-prediction-tests-design.md docs/superpowers/plans/2026-04-29-worldcup-hdf5-e2e-prediction-tests.md
```

The `git rm docs/DEFERRED.md` from Task 9 step 3 has already staged the deletion.

```bash
git commit -m "$(cat <<'EOF'
feat(test): WorldCup-2018 HDF5 fixture + e2e prediction tests in CI + DEFERRED.md cleanup — silly-kicks 1.9.0

- New scripts/build_worldcup_fixture.py — reproducible WorldCup-2018 SPADL fixture generator. Downloads StatsBomb open-data raw events (cached, gitignored), converts via silly_kicks.spadl.statsbomb, writes multi-key HDFStore.
- Vendored tests/datasets/statsbomb/spadl-WorldCup-2018.h5 (~XX MB, 64 matches). 5 test_predict* cases (vaep / xthreat / atomic vaep) now run on every CI matrix slot — were silently skipping for ~9 release cycles.
- Drop @pytest.mark.e2e on the 5 prediction tests (fixture now committed).
- conftest's sb_worldcup_data: pytest.skip → pytest.fail on missing fixture (matches PR-S8 pattern; surfaces packaging regressions instead of silent skip).
- CI lint job now also runs ruff check + format --check on scripts/.
- DEFERRED.md → TODO.md migration: 4 live items moved to new ## Tech Debt section (A19, D-9, O-M1, O-M6); audit history preserved in git log; 4 by-design items noted in CHANGELOG; CLAUDE.md reference removed; docs/DEFERRED.md deleted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `~XX MB` with the measured size before running.

If pre-commit hook fails: fix the issue, create a NEW commit (per `feedback_commit_policy` — never amend).

- [ ] **Step 4: Verify commit**

```bash
git log --oneline -1
git show --stat HEAD | tail -25
```

Expected: single commit with the title above + ~14 files changed.

- [ ] **Step 5: Push — chat approval gate (no sentinel under narrowed hook)**

STOP HERE. Surface to the user:
> "Commit created: `<hash> feat(test): ...`. Ready to push to origin?"

Wait for chat approval.

- [ ] **Step 6: Push (on user approval)**

```bash
git push -u origin feat/worldcup-hdf5-e2e-prediction-tests
```

- [ ] **Step 7: Open PR — chat approval gate**

STOP HERE. Surface the PR title + body to the user:
> "Branch pushed. Ready to open PR? Title: `feat(test): WorldCup-2018 HDF5 fixture + e2e prediction tests in CI + DEFERRED.md cleanup — silly-kicks 1.9.0`"

Wait for chat approval.

- [ ] **Step 8: Create PR (on user approval)**

```bash
gh pr create --title "feat(test): WorldCup-2018 HDF5 fixture + e2e prediction tests in CI + DEFERRED.md cleanup — silly-kicks 1.9.0" --body "$(cat <<'EOF'
## Summary

- New `scripts/build_worldcup_fixture.py` — reproducible WorldCup-2018 SPADL fixture generator. Downloads StatsBomb open-data raw events (cached at `tests/datasets/statsbomb/raw/.cache/`, gitignored), runs them through `silly_kicks.spadl.statsbomb.convert_to_actions`, writes multi-key HDFStore matching what `tests/conftest.py::sb_worldcup_data` expects (`games` table + `actions/game_<id>` per match).
- Vendored `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` (~XX MB, 64 matches). Closes the ~9-release-cycle coverage gap on the 5 `test_predict*` cases (VAEP fit + rate, xT fit + rate, atomic VAEP fit + rate). Tests previously silently skipped in CI and locally because the fixture was never committed.
- 5 `@pytest.mark.e2e` markers dropped on the prediction tests (now run unconditionally).
- `tests/conftest.py::sb_worldcup_data`: `pytest.skip` → `pytest.fail` on missing fixture (matches PR-S8 pattern — surfaces packaging regressions instead of letting CI silently skip).
- `.github/workflows/ci.yml` lint job now covers `scripts/` in addition to `silly_kicks/` and `tests/`.

National Park Principle bundle:
- `docs/DEFERRED.md` deleted (rotting Phase 2-3d audit archive with internal status inconsistency). Live items migrated to a new `## Tech Debt` section in `TODO.md` (A19, D-9, O-M1, O-M6). Items judged "by design / accept" noted in CHANGELOG (A15, A16, A17, S5). Audit history preserved in `git log -- docs/DEFERRED.md`. `CLAUDE.md` reference removed.

Design + plan committed under `docs/superpowers/{specs,plans}/2026-04-29-worldcup-hdf5-e2e-prediction-tests*`.

## Test plan

- [x] Build script runs cleanly: `python scripts/build_worldcup_fixture.py --verbose` produces 10-30 MB HDF5 in 3-8 min cold cache.
- [x] HDF5 structure validated: 64 games, all action counts > 100.
- [x] 5 prediction tests pass (no longer skip): `test_predict` × 3 + `test_predict_with_missing_features` + `test_predict_with_interpolation`.
- [x] Full pytest suite: ~560 passed, ~4 skipped (vs 1.8.0 baseline 555/9).
- [x] `uv run ruff check silly_kicks/ tests/ scripts/` — zero errors.
- [x] `uv run ruff format --check silly_kicks/ tests/ scripts/` — zero formatting issues.
- [x] `uv run pyright silly_kicks/` — zero errors (pin: pandas-stubs==2.3.3.260113 to match CI).
- [x] `/mad-scientist-skills:final-review` — passed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `~XX MB` with measured size.

- [ ] **Step 9: Wait for CI green, then merge — chat approval gate**

After PR opens, CI runs. STOP HERE:
> "PR opened: <URL>. Waiting for CI. Once green, ready to squash-merge with `--admin --delete-branch`?"

Use a polling Bash command in `run_in_background` to detect when all checks complete (5 matrix slots: lint + ubuntu 3.10/3.11/3.12 + windows 3.12). Surface the final result.

If CI fails, fix and amend the local commit (force-push to the feature branch only — never main).

- [ ] **Step 10: Squash-merge (on chat approval after CI green)**

```bash
gh pr merge --admin --squash --delete-branch
```

Expected: merge succeeds, branch deleted on remote. Sync local main:

```bash
git checkout main
git pull
```

- [ ] **Step 11: Tag + push tag — chat approval gate**

STOP HERE:
> "Merge complete on main. Ready to tag v1.9.0 and push? This will auto-fire the PyPI publish workflow."

Wait for chat approval.

- [ ] **Step 12: Tag + push (on user approval) — PyPI auto-publish**

```bash
git tag v1.9.0
git push origin v1.9.0
```

Expected: tag push triggers `.github/workflows/publish.yml`. Verify via:

```bash
gh run list --workflow publish.yml --limit 1
```

Wait for completion (~30-60s based on prior runs).

- [ ] **Step 13: Verify PyPI live**

```bash
curl -s "https://pypi.org/pypi/silly-kicks/json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'latest: {d[\"info\"][\"version\"]}')"
```

Expected: `latest: 1.9.0`.

- [ ] **Step 14: Update auto-memory — release state**

Update `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_release_state.md`:
- Header: `Current: silly-kicks 1.9.0 (tag pushed 2026-04-29)`
- PyPI line: `silly-kicks==1.9.0`
- Add 1.9.0 row to version trajectory table
- Update `Open items in TODO.md` section: remove PR-S9 from queued; reflect 1.9.0 state

Update `project_followup_prs.md`:
- Mark PR-S9 as shipped in 1.9.0
- PR-S10 stays as the only queued follow-up

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Spec § 1 problem (silently-skipping prediction tests) → Tasks 4-5 (drop markers, verify pass)
- ✅ Spec § 2.1 reproducibility goal → Task 1 (build script)
- ✅ Spec § 2.2 vendored HDF5 → Task 2 (run script)
- ✅ Spec § 2.3 surface CI packaging regressions → Task 3 (skip → fail)
- ✅ Spec § 2.4 DEFERRED.md migration → Task 9
- ✅ Spec § 2.5 lint scripts/ in CI → Task 7
- ✅ Spec § 4.2 build script CLI → Task 1 step 2 (full code)
- ✅ Spec § 4.3 conftest change → Task 3 step 1 (full code)
- ✅ Spec § 4.4 marker drops → Task 4
- ✅ Spec § 4.5 .gitignore → Task 6
- ✅ Spec § 4.6 CI lint expansion → Task 7
- ✅ Spec § 5 DEFERRED.md migration details → Task 9
- ✅ Spec § 6 test plan + ordering → Tasks 1-10 sequencing
- ✅ Spec § 7 verification gates → Task 10
- ✅ Spec § 8 commit cycle → Task 11

**2. Placeholder scan:** Two intentional `~XX MB` placeholders (Task 8 step 2 CHANGELOG and Task 11 step 3 commit + step 8 PR body) — these are filled in during execution with the measured size from Task 2. Documented as such. No other placeholders.

**3. Type consistency:** `BoundaryMetrics` not relevant here. The script's function signatures (`_fetch_url`, `_fetch_manifest`, `_fetch_raw_events`, `_adapt_events_to_silly_kicks_input`, `_build_games_metadata_df`, `_convert_match`, `_validate_output`, `main`) are all defined in Task 1's code block. The `sb_worldcup_data` fixture signature stays unchanged in Task 3 (only the `pytest.skip` body swaps to `pytest.fail`). All e2e marker drops are on existing functions (no signature changes).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-worldcup-hdf5-e2e-prediction-tests.md`. Per silly-kicks pattern (per `feedback_engineering_disciplines` memory: user finds subagent approval friction excessive), this plan is for **inline execution** via `superpowers:executing-plans` — not subagent-driven.
