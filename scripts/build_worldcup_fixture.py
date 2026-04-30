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

    Raises RuntimeError after ``retries`` attempts.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
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
            _log(
                f"manifest: cache hit ({cache_path.relative_to(_REPO_ROOT)}, age={age / 3600:.1f}h)",
                level="debug",
                verbose=verbose,
                quiet=quiet,
            )
            return json.loads(cache_path.read_text(encoding="utf-8"))

    _log(f"manifest: downloading from {_MANIFEST_URL}", level="info", verbose=verbose, quiet=quiet)
    raw = _fetch_url(_MANIFEST_URL)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def _fetch_raw_events(
    match_id: int,
    cache_dir: Path,
    *,
    no_cache: bool,
    verbose: bool,
    quiet: bool,
) -> list[dict[str, Any]]:
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
    return pd.DataFrame(
        [
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
                # PR-S12 (silly-kicks 2.1.0): preserve native possession sequence
                # for downstream regression validation against add_possessions.
                "possession": e.get("possession"),
            }
            for e in events
        ]
    )


def _build_games_metadata_df(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the ``games`` metadata DataFrame from the matches manifest."""
    rows = []
    for m in matches:
        rows.append(
            {
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
            }
        )
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
    actions, _report = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id, preserve_native=["possession"])
    return match_id, actions


def _validate_output(
    games: pd.DataFrame,
    actions_by_id: dict[int, pd.DataFrame],
    output_path: Path,
    *,
    verbose: bool,
    quiet: bool,
) -> None:
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
            f"output file size {file_size_bytes / 1024 / 1024:.1f} MB exceeds "
            f"{_HDF_SIZE_WARN_BYTES / 1024 / 1024:.0f} MB warn threshold",
            level="warn",
            verbose=verbose,
            quiet=quiet,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"HDF5 output path (default: {_DEFAULT_OUTPUT.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help=f"Raw event cache dir (default: {_DEFAULT_CACHE_DIR.relative_to(_REPO_ROOT)})",
    )
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
        _log(
            f"manifest contains {len(matches)} matches (expected 64) — proceeding anyway",
            level="warn",
            verbose=verbose,
            quiet=quiet,
        )

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

    _log(
        f"writing HDFStore: {len(actions_by_id)} matches, "
        f"{sum(len(df) for df in actions_by_id.values())} total actions",
        level="info",
        verbose=verbose,
        quiet=quiet,
    )

    with pd.HDFStore(str(output_path), mode="w", complib=_HDF_COMPLIB, complevel=_HDF_COMPLEVEL) as store:
        store.put("games", games_df, format="table")
        for game_id, actions_df in actions_by_id.items():
            store.put(f"actions/game_{game_id}", actions_df, format="table")

    _validate_output(games_df, actions_by_id, output_path, verbose=verbose, quiet=quiet)

    elapsed = time.time() - start_ts
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    _log(
        f"done: {len(actions_by_id)} matches, {file_size_mb:.1f} MB HDF5, {elapsed:.0f}s",
        level="info",
        verbose=verbose,
        quiet=quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
