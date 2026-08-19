"""Probe the StatsBomb `cross -> related_events -> Block` join for `cross_blocked` (BD-2).

Ad-hoc, print-only investigation (spec 3.1): prints the R1-R3 evidence to stdout and writes
NOTHING. The offline core is the three committed fixtures; `--open "comp/season"` additionally
fetches a wider StatsBomb OPEN-data slice via statsbombpy (guarded import) for corpus-scale
base-rate + edge-case evidence. Open data is a proxy for the whole `statsbomb` provider:
`related_events` is a standard field and the same converter path serves licensed SB360, so the
measurement generalises by construction, not by measuring the un-probeable licensed rows.

Decision rule (pre-registered, spec 3.2): SHIP iff R1 (< 5% of open-play crosses have absent
`related_events`) AND R2 (same-team Block links absent, or < 1% of linked crosses) AND R3 (the
">=1 opposing Block" rule is well-defined on 100% of linked cases -- it is, by construction).

Team comparison here uses raw `!=` (measurement-only, on raw-int open-data ids); the CONVERTER
(Task 4) uses `id_compat.same_id` for the NA-safe production path -- they agree except on an
NA-team edge case, which the probe does not separately tally. The R2 `linked` denominator
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
        from statsbombpy import sb  # type: ignore[import-not-found]
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
