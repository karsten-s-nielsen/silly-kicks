"""Measure real StatsBomb 360 freeze-frame coverage across the three-cell design.

Layer B of the SB360 coverage audit. Layer A establishes what the CODE does on freeze-frames;
this establishes what real freeze-frames CONTAIN -- the keeper-visibility rate that decides
whether the GK surface is usable at all.

Corpus pass, so it adopts the ADR-052 seam (``for_each``, per-match shards, resumable) and the
ADR-037 provenance rule (``require_clean_tree`` in ``main()``, before any corpus work).

``statsbombpy`` is imported LAZILY inside the work functions so ``--help`` and the unit tests
never need the optional dependency.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._driver import for_each
from scripts._provenance import git_provenance, require_clean_tree
from silly_kicks.providers.statsbomb import (
    acting_side_gk_visible,
    defending_gk_visible,
    visible_fraction,
)
from silly_kicks.spadl import _sb_coordinates as _sb_coords

#: (competition_id, season_id) -> expected name. Asserted at run time: prose verification does
#: not survive an upstream renumber, and a silently-wrong sample is worse than a crash.
EXPECTED_NAMES = {
    (72, 107): "Women's World Cup",
    (43, 106): "FIFA World Cup",
    (44, 107): "Major League Soccer",
}

#: SPADL type names where the DEFENDING keeper is the metric's subject.
#:
#: These are SPADL names and are matched against REAL SPADL actions produced by the converter,
#: never against StatsBomb's own taxonomy -- which contains none of them. StatsBomb has no
#: `Cross` type (it is `pass_cross == True`), no `Goal Kick` type (it is
#: `pass_type == "Goal Kick"`), and its keeper type is `"Goal Keeper"`. Matching this constant
#: against StatsBomb names would reduce `is_gk_domain` to "pass or shot" -- most of the match --
#: while goal kicks, the spec's named GK-domain event, folded into `Pass` and were never
#: flagged.
GK_DOMAIN_TYPES = ("shot", "shot_penalty", "shot_freekick", "cross", "goalkick", "keeper_save")

#: Default shard root. TOP-LEVEL and ending in ``_shards`` so the anchored ``/*_shards/`` glob
#: at .gitignore:90 covers it. A nested path such as ``docs/research/.../_shards`` is NOT
#: covered -- the anchor is deliberate, so an unanchored glob cannot silence tracked paths at
#: depth -- and would dirty the tree on every run.
DEFAULT_SHARD_ROOT = "sb360_coverage_shards"

#: StatsBomb's pitch. `visible_area` is delivered in THIS frame, not SPADL's 105x68; dividing
#: by 105*68 yields ~1.34 for a fully-visible frame, i.e. a "fraction" above 1.
# Re-exported from the port so the script and the library cannot disagree about the SB grid.
SB_PITCH_LENGTH = _sb_coords.SB_FIELD_LENGTH
SB_PITCH_WIDTH = _sb_coords.SB_FIELD_WIDTH

#: Keys the converter reads from the top level; everything else rides in `extra`.
#: Mirrors tests/test_xthreat_statsbomb_e2e.py::_adapt.
_TOP_LEVEL_KEYS = frozenset({"id", "period", "timestamp", "team", "player", "type", "location"})


class CompetitionMismatchError(ValueError):
    """A competition id resolved to the wrong name, or to nothing.

    A raise rather than an ``assert``: this validates EXTERNAL data, and an assert disappears
    under ``python -O`` -- taking the guard with it precisely when a run is being optimised.
    """


@functools.cache
def _type_id_to_name() -> dict[int, str]:
    """SPADL ``type_id`` -> ``type_name``, read from the library's own config table."""
    import silly_kicks.spadl as spadl

    df = spadl.actiontypes_df()
    return {int(i): str(n) for i, n in zip(df["type_id"], df["type_name"], strict=True)}


def _retry(fn, attempts: int = 4, base_sleep: float = 5.0):
    """Ride out transient network blips (DNS, resets) on the SB open-data calls.

    Mirrors ``scripts/validate_shot_goalmouth_sb.py::_retry``, which found this necessary at
    ~17 calls. This driver makes roughly three per match, so a full three-cell run at eight
    matches per cell is ~72 -- and ``for_each`` records a failure rather than losing the pass,
    but a match lost to one failed DNS lookup is still a match re-fetched later for no reason.
    """
    import time

    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(base_sleep * (2**k))
    raise AssertionError("unreachable: the final attempt either returned or raised")


def _values(payload):
    """statsbombpy returns dict-keyed-by-id or a list depending on call and version."""
    return list(payload.values()) if isinstance(payload, dict) else list(payload)


def resolve_competition(comp_id: int, season_id: int, *, catalogue, expect_name: str) -> dict:
    """Resolve one competition/season and assert its NAME, not just its id."""
    for row in catalogue:
        if row["competition_id"] == comp_id and row["season_id"] == season_id:
            actual = row["competition_name"]
            if actual != expect_name:
                raise CompetitionMismatchError(
                    f"competition {comp_id}/{season_id} resolved to {actual!r}, expected "
                    f"{expect_name!r} -- upstream ids have drifted and sampling would be silent"
                )
            return row
    raise CompetitionMismatchError(f"competition {comp_id}/{season_id} not found in catalogue")


def _ids_for_cell(override, comp_id: int, season_id: int) -> tuple[bool, list | None]:
    """Return ``(drop_this_cell, ids)``; ``ids is None`` means "take the default".

    A partition naming NO ids for a cell must DROP it. An empty list and an absent key are BOTH
    falsy, and conflating them with "unpartitioned" makes every worker load the entire unsliced
    manifest -- ADR-052's measured defect. Returned as an explicit pair rather than a sentinel
    so the three states are distinguishable to a reader and to a type checker.
    """
    if override is None:
        return (False, None)
    ids = override.get(f"{comp_id}:{season_id}")
    if not ids:
        return (True, None)
    return (False, list(ids))


def _load_catalogue() -> list[dict]:
    from statsbombpy import sb  # type: ignore[import-not-found]

    return _values(_retry(lambda: sb.competitions(fmt="dict")))


def _adapt_events(events: list[dict], match_id: int):
    """Raw StatsBomb event dicts -> the silly-kicks converter's input contract."""
    import pandas as pd

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
            }
            for e in events
        ]
    )


def measure_match(match):
    """One match -> tidy per-(SPADL action_type) coverage rows. Rates carry denominators."""
    import pandas as pd
    from statsbombpy import sb  # type: ignore[import-not-found]

    from silly_kicks.spadl.statsbomb import convert_to_actions

    comp_id, season_id, match_id, home_team_id = match

    # Frame records carry event_uuid, visible_area and freeze_frame -- NOT the event type.
    # The join runs through the REAL converter for two reasons: the spec asks for coverage
    # "per SPADL action type" and StatsBomb's taxonomy cannot express those types; and it
    # exercises the converter -- the path NWSL data will take -- for the same reason Leg A of
    # the synthetic fixture is built by the real producer.
    events = _values(_retry(lambda: sb.events(match_id=match_id, fmt="dict")))
    actions, _report = convert_to_actions(_adapt_events(events, match_id), home_team_id)
    # SPADL emits `type_id`, NOT `type_name` -- the name is a config-table lookup, and
    # `SPADL_COLUMNS` has no `type_name` at all. The synthetic Layer A fixture carries
    # `type_name` as a CONVENIENCE column alongside the schema, and writing this against that
    # shape raised `KeyError: 'type_name'` on the first real match. A fixture's convenience is
    # not a contract.
    frames_raw = _values(_retry(lambda: sb.frames(match_id=match_id, fmt="dict")))
    id_to_name = _type_id_to_name()
    # statsbomb.py:235 sets original_event_id = events.event_id.astype(str).
    type_by_uuid = {
        str(uuid): id_to_name.get(int(tid), "unknown_type")
        for uuid, tid in zip(actions["original_event_id"].astype(str), actions["type_id"], strict=True)
    }

    # FRAME-EXISTENCE, counted from the ACTION side. The per-frame metrics below can only
    # describe frames that exist; they are structurally blind to an action that got no frame at
    # all. Measured across 9 matches: only 30/129 goal kicks (23.3%) carry a 360 frame, with a
    # per-match range of 0-50% -- and goal kicks are xT-GK's core distribution domain. Reporting
    # only within-frame keeper visibility would have described a quarter of the domain as if it
    # were the whole of it.
    frame_uuids = {str(f.get("event_uuid")) for f in frames_raw}
    actions_per_type: dict[str, dict[str, int]] = {}
    for uuid, tid in zip(actions["original_event_id"].astype(str), actions["type_id"], strict=True):
        name = id_to_name.get(int(tid), "unknown_type")
        b = actions_per_type.setdefault(name, {"n_actions": 0, "n_with_frame": 0})
        b["n_actions"] += 1
        b["n_with_frame"] += int(str(uuid) in frame_uuids)

    # JOIN INTEGRITY. Measured on the first real pass: 3 of 22 open matches (all MLS 2023) ship
    # a 360 file whose `event_uuid`s have ZERO overlap with their own events file, while
    # correctly claiming the same `match_id`. That is an upstream inconsistency, not a converter
    # or join defect -- verified against the RAW events, not just the SPADL actions.
    #
    # It must be COUNTED, not silently averaged over. Such a match previously produced a
    # one-row shard whose single bucket was `unmapped`, indistinguishable at a glance from a
    # quiet match, and it would have diluted every aggregate it entered.
    mapped = sum(1 for f in frames_raw if str(f.get("event_uuid")) in type_by_uuid)
    join_rate = mapped / len(frames_raw) if frames_raw else float("nan")
    if frames_raw and mapped == 0:
        warnings.warn(
            f"match {match_id}: {len(frames_raw)} freeze-frames and NONE join to an event "
            f"(`event_uuid` has zero overlap with the events file). Upstream data "
            f"inconsistency -- rows are emitted with match_join_rate=0.0 so a consumer can "
            f"exclude and COUNT them.",
            stacklevel=2,
        )

    per_type: dict[str, dict[str, float]] = {}
    for ff in frames_raw:
        players = ff.get("freeze_frame") or []
        # "unmapped" rather than "unknown": a frame whose event the converter dropped as a
        # non_action is a REAL category, and conflating it with a failed lookup would hide a
        # broken join inside a legitimate bucket.
        type_name = type_by_uuid.get(str(ff.get("event_uuid")), "unmapped")
        bucket = per_type.setdefault(
            type_name,
            {
                "n_events": 0,
                "n_defending_gk_visible": 0,
                "n_acting_side_gk_visible": 0,
                "sum_visible": 0.0,
                "sum_area": 0.0,
            },
        )
        bucket["n_events"] += 1
        bucket["n_defending_gk_visible"] += int(defending_gk_visible(players))
        bucket["n_acting_side_gk_visible"] += int(acting_side_gk_visible(players))
        bucket["sum_visible"] += len(players)
        bucket["sum_area"] += visible_fraction(ff.get("visible_area") or [])

    rows = []
    for type_name, b in per_type.items():
        n = b["n_events"]
        rows.append(
            {
                "competition_id": comp_id,
                "season_id": season_id,
                "match_id": match_id,
                "action_type": type_name,
                # Denominators travel WITH every rate: a rate alone invites a reader to treat
                # an 8-event cell as an 800-event one.
                "n_events": n,
                "n_defending_gk_visible": b["n_defending_gk_visible"],
                "defending_gk_visible_rate": b["n_defending_gk_visible"] / n if n else float("nan"),
                # The acting side's keeper -- the relevant one for GK distribution and
                # saves, where the defending rate is 0 by construction.
                "n_acting_side_gk_visible": b["n_acting_side_gk_visible"],
                "acting_side_gk_visible_rate": (b["n_acting_side_gk_visible"] / n if n else float("nan")),
                # Roster completeness -- the honest quantity where a feature's support is
                # defined by the visible players themselves, since a coverage fraction there is
                # circular (the hull over visible players is 100% observed by construction).
                "mean_players_visible": b["sum_visible"] / n if n else float("nan"),
                "mean_visible_pitch_fraction": b["sum_area"] / n if n else float("nan"),
                # How many SPADL actions of this type EXIST, and how many got a frame.
                # NaN rather than 0 where the type produced no actions -- an "unmapped" bucket
                # has frames but no actions by definition, and 0/0 is not a rate of zero.
                "n_actions": actions_per_type.get(type_name, {}).get("n_actions", 0),
                "n_actions_with_frame": actions_per_type.get(type_name, {}).get("n_with_frame", 0),
                "frame_existence_rate": (
                    actions_per_type[type_name]["n_with_frame"] / actions_per_type[type_name]["n_actions"]
                    if actions_per_type.get(type_name, {}).get("n_actions")
                    else float("nan")
                ),
                "is_gk_domain": type_name.lower() in GK_DOMAIN_TYPES,
                # Fraction of this match's frames that resolved to an action at all.
                # 0.0 marks an upstream uuid mismatch: exclude, and count the exclusion.
                "match_join_rate": join_rate,
            }
        )
    return pd.DataFrame(rows)


def _iter_matches(selected, args):
    """STREAM, never list() -- the ADR-052 rule; a match pull is expensive per item.

    Yields ``(competition_id, season_id, match_id, home_team_id)``. The home team id rides along
    because ``convert_to_actions`` requires it and re-fetching inside ``work`` would repeat a
    network call per item.
    """
    from statsbombpy import sb  # type: ignore[import-not-found]

    override = json.loads(args.match_ids_json.read_text()) if args.match_ids_json else None
    for row in selected:
        c, s = row["competition_id"], row["season_id"]
        drop, ids = _ids_for_cell(override, c, s)
        if drop:
            continue
        # Loop variables bound as defaults: `_retry` happens to invoke immediately, so a
        # late-binding capture would work by luck rather than by construction (ruff B023).
        raw = _retry(lambda c=c, s=s: sb.matches(competition_id=c, season_id=s, fmt="dict"))
        matches = {m["match_id"]: m for m in _values(raw)}
        if ids is None:
            ids = sorted(matches)[: args.matches_per_cell]
        for mid in ids:
            home = matches[mid].get("home_team") or {}
            home_id = home.get("home_team_id", home.get("id")) if isinstance(home, dict) else home
            yield (c, s, mid, home_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Top-level `*_shards/`, matching the seven sibling drivers and the ANCHORED `/*_shards/`
    # glob at .gitignore:90. Not a tidiness choice: an un-ignored shard root makes the NEXT
    # artifact-writing run refuse on its predecessor's own scratch, and the likely operator
    # response is `--allow-dirty`, which stamps `run_tree_dirty: true` onto a run whose CODE was
    # clean -- laundering the exact fact ADR-052 wired the gate to preserve. Verified with
    # `git check-ignore`: `sb360_coverage_shards/` is covered, `docs/research/.../_shards/` is
    # NOT, because the glob is anchored to the repo root on purpose.
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(DEFAULT_SHARD_ROOT))
    ap.add_argument("--competitions", nargs="+", default=["72:107", "43:106", "44:107"])
    ap.add_argument("--matches-per-cell", type=int, default=8)
    ap.add_argument("--match-ids-json", type=pathlib.Path, default=None)
    ap.add_argument("--list-matches", action="store_true")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    # FIRST, before any corpus work: `git rev-parse HEAD` returns the same SHA whether or not
    # the tree is modified, so a driver stamping the bare SHA records a commit that does not
    # describe the code that ran (ADR-037).
    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    cells = [tuple(int(p) for p in c.split(":")) for c in args.competitions]
    catalogue = _load_catalogue()
    selected = [resolve_competition(c, s, catalogue=catalogue, expect_name=EXPECTED_NAMES[(c, s)]) for c, s in cells]

    if args.list_matches:
        print(json.dumps([m[2] for m in _iter_matches(selected, args)]))
        return

    res = for_each(
        _iter_matches(selected, args),
        key=lambda m: f"{m[0]}_{m[1]}_{m[2]}",
        work=measure_match,
        shard_root=args.out,
        token_inputs={
            "competitions": sorted(f"{c}:{s}" for c, s in cells),
            "matches_per_cell": args.matches_per_cell,
            "schema": "sb360-coverage-2",
        },
        tag=args.tag,
        label="match",
    )
    (args.out / f"manifest_{args.tag}.json").write_text(
        json.dumps({**res.manifest(), **prov}, indent=2), encoding="utf-8"
    )
    # CorpusPassResult (scripts/_driver.py:511-518) carries shard_dir, attempted, skipped,
    # failed, failures, counters, keys, counters_unrecorded. There is NO `processed`.
    print(
        f"attempted={res.attempted} processed={res.attempted - res.skipped - res.failed} "
        f"skipped={res.skipped} failed={res.failed}"
    )


if __name__ == "__main__":
    main()
