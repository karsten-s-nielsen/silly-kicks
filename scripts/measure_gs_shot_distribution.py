"""Measure the real per-(match, team, period) shot distribution on owner-tier Gradient Sports.

Item 23 step 2a (ADR-054). The GS input-convention guard cannot see the case it governs: the
committed fixture `tests/datasets/gradientsports/synthetic_match.json` DEFERS, so
`detect_input_convention` never classifies and the declared-vs-detected disagreement never
surfaces in CI.

The binding constraint is NOT per-group shot count, which an earlier reading assumed. The fixture
has 10 shots in (team 100, period 1) -- AT `min_shots_per_group_high` -- and only ONE team has
shots at all, so it defers on the *fewer-than-two-reliable-groups* clause
(`silly_kicks/spadl/orientation.py`). Reshaping to raise per-group counts would not have made CI
see the case; giving a SECOND team or period a reliable group is what does.

This driver measures what real GS actually looks like, so the reshape targets a recorded
distribution instead of a guess. A fixture shaped to an unrecorded number rebuilds the exact
failure this cycle removes.

WHAT TRAVELS, AND WHAT DOES NOT
-------------------------------
Gradient Sports data is owner-tier. This driver emits COUNTS ONLY -- shots per (team, period),
how many groups clear each reliability threshold, and the detector's own verdict. No coordinates,
no player or team names, no event ids, nothing from which a position could be reconstructed. The
`team_id` values are replaced by a per-match dense rank, so even the identifier does not travel.

Usage (on the box, scripts/ on sys.path, pining token in env):

    python scripts/measure_gs_shot_distribution.py --out docs/research/gs_input_convention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._driver import for_each, reconcile
from scripts._input_contract import declare_inputs
from scripts._provenance import git_provenance, require_clean_tree

#: The thresholds the detector itself uses. Read from the library rather than restated, so a change
#: to either default moves this measurement instead of silently invalidating it.
from silly_kicks.spadl.orientation import detect_input_convention

_DEFAULT_HIGH = 10
_DEFAULT_MEDIUM = 5


def input_contract() -> dict:
    """Declare WHICH SYMBOLS this measurement depends on (ADR-054)."""
    import inspect

    sig = inspect.signature(detect_input_convention)
    return declare_inputs(
        driver="measure_gs_shot_distribution",
        detector={
            "min_shots_per_group_high": sig.parameters["min_shots_per_group_high"].default,
            "min_shots_per_group_medium": sig.parameters["min_shots_per_group_medium"].default,
        },
        extractors=("silly_kicks.spadl.orientation",),
    )


def measure_match(item) -> pd.DataFrame:
    """One match -> one tidy row per (team, period) group, counts only.

    An empty result still writes a shard: absent means "not yet run", present-and-empty means
    "ran, produced nothing", and conflating them recomputes every barren match forever (ADR-052).
    """
    provider, match_id, actions, _frames, _home = item
    import silly_kicks.spadl.config as spadlcfg

    shot_ids = {spadlcfg.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick")}
    shots = actions[actions["type_id"].isin(shot_ids)]
    if shots.empty:
        return pd.DataFrame(
            columns=["provider", "match_id", "team_rank", "period_id", "n_shots", "reliable_high", "reliable_medium"]
        )

    # Dense-rank the team ids so no real identifier travels.
    ranks = {t: i for i, t in enumerate(sorted(shots["team_id"].dropna().unique()))}
    grouped = shots.groupby(["team_id", "period_id"]).size().reset_index(name="n_shots")
    grouped["provider"] = provider
    grouped["match_id"] = str(match_id)
    grouped["team_rank"] = grouped["team_id"].map(ranks)
    grouped["reliable_high"] = grouped["n_shots"] >= _DEFAULT_HIGH
    grouped["reliable_medium"] = grouped["n_shots"] >= _DEFAULT_MEDIUM
    return grouped.drop(columns=["team_id"])[
        ["provider", "match_id", "team_rank", "period_id", "n_shots", "reliable_high", "reliable_medium"]
    ]


def summarise(table: pd.DataFrame) -> dict:
    """Corpus-level summary: the numbers the fixture reshape is shaped to."""
    if table.empty:
        return {"n_matches": 0, "note": "no shots in corpus"}
    per_match = table.groupby(["provider", "match_id"]).agg(
        n_groups=("n_shots", "size"),
        n_reliable_high=("reliable_high", "sum"),
        n_reliable_medium=("reliable_medium", "sum"),
        max_shots_in_a_group=("n_shots", "max"),
    )
    return {
        "n_matches": len(per_match),
        "n_groups_total": len(table),
        "shots_per_group": {
            "min": int(table["n_shots"].min()),
            "median": float(table["n_shots"].median()),
            "max": int(table["n_shots"].max()),
        },
        "reliable_groups_per_match_high": {
            "min": int(per_match["n_reliable_high"].min()),
            "median": float(per_match["n_reliable_high"].median()),
            "max": int(per_match["n_reliable_high"].max()),
        },
        "reliable_groups_per_match_medium": {
            "min": int(per_match["n_reliable_medium"].min()),
            "median": float(per_match["n_reliable_medium"].median()),
            "max": int(per_match["n_reliable_medium"].max()),
        },
        # THE number the fixture defect turns on: how often does a real match clear the
        # two-reliable-groups bar the committed fixture fails?
        "matches_with_ge2_reliable_high": int((per_match["n_reliable_high"] >= 2).sum()),
        "matches_with_ge2_reliable_medium": int((per_match["n_reliable_medium"] >= 2).sum()),
        "thresholds": {"high": _DEFAULT_HIGH, "medium": _DEFAULT_MEDIUM},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output DIRECTORY: shards, combined table, metrics.json.")
    ap.add_argument("--providers", nargs="+", default=["gradientsports"])
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=1, help="Frames per match; this pass needs EVENTS only.")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dev run from a modified tree; the artifact still records dirty: true",
    )
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    from _loader_pining import load_matches

    out = Path(args.out)
    res = for_each(
        load_matches(
            providers=list(args.providers),
            tracking_limit=args.tracking_limit,
            max_per_provider=args.max_per_provider,
            cache_dir=args.cache_dir,
        ),
        key=lambda m: f"{m[0]}_{m[1]}",
        work=measure_match,
        shard_root=out,
        token_inputs={
            "providers": sorted(args.providers),
            "thresholds": [_DEFAULT_HIGH, _DEFAULT_MEDIUM],
            "schema": "gs-shot-distribution-1",
        },
        label="match",
    )
    table = reconcile(res.shard_dir, out / "shot_distribution.parquet", tag="all")
    metrics = {
        **summarise(table),
        # Record the SCOPE, not just the result. `measure_rc4_orientation`'s predecessor shipped a
        # `tracking_limit=3000` cap recorded NOWHERE and halved a published headline; the artifact
        # was cited, uncheckable and wrong. Verified for this driver: `tracking_limit` slices
        # `frames_json` only (`_loader_pining._build_gradientsports`), so it caps FRAMES and leaves
        # the action set -- which is why a shot-count pass can set it to 1. Recorded anyway, because
        # "it does not affect this measurement" is exactly the claim a reader must be able to check.
        "scope": {
            "providers": sorted(args.providers),
            "max_per_provider": args.max_per_provider,
            "tracking_limit": args.tracking_limit,
            "tracking_limit_caps": "frames per match, NOT actions -- shot counts come from events",
        },
        "input_contract": input_contract(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        **res.manifest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "input_contract"}, indent=2))


if __name__ == "__main__":
    main()
