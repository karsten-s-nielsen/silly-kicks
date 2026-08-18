"""Validate the licensed StatsBomb 360 corpus through the library (SB360 enablement cycle).

For each SB360 match this measures THREE segregated things (round-4 review), kept apart on purpose:

1. **Battery population -- VERDICTS/coverage, never synthetic values.** ``run_add_star_battery``
   runs every registered ``add_*`` with the shared exercise adapters (synthetic xt/xg, and a fixed
   half-pitch for ``add_visible_area_coverage``), so its per-column NUMBERS are synthetic-input
   hybrids. Only the structural fact is recorded: did the aggregator RUN, and what fraction of each
   emitted column is populated on real freeze-frames. A ``~0.5`` half-pitch fraction reported as a
   corpus number would be the ADR-042 coverage-denominator-as-signal trap.
2. **Count-feature companions -- REAL visible_area.** ``add_action_context(..., visible_area=<real>)``
   gives the three count features' ``observed_source``/``observed_fraction`` distributions.
3. **Pitch coverage -- REAL visible_area, explicit and segregated.**
   ``add_visible_area_coverage(..., visible_area=<real>)`` gives the real observed-pitch-fraction
   distribution -- the honest counterpart to (1), which neither (1) nor (2) measures.

Plus frame-existence coverage per GK-domain action type and the roster keeper-resolution rate.

Licensed data is NEVER committed (ADR-009:11; ADR-038 fail-closed default): per-match shards go to a
GITIGNORED ``--shard-root``; only the reconciled aggregate lands under ``--out``. Adopts the ADR-052
seam (``for_each``, per-match shards, resumable) and the ADR-037 provenance rule
(``require_clean_tree`` in ``main()``, before any corpus work; ``run_commit``/``run_tree_dirty``
stamped into the artifact).
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
from typing import TYPE_CHECKING

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._driver import for_each, reconcile
from scripts._provenance import git_provenance, require_clean_tree
from scripts._sb_battery import run_add_star_battery

if TYPE_CHECKING:
    import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: TOP-LEVEL and ending in ``_shards`` so the anchored ``/*_shards/`` .gitignore glob covers it -- a
#: nested ``docs/research/.../_shards`` is NOT covered, and a licensed shard leaking there would be
#: committed. The reconciled aggregate is the ONLY thing that lands under ``--out``.
DEFAULT_SHARD_ROOT = "sb360_licensed_shards"

#: SPADL type names where the DEFENDING keeper is the metric's subject (matches build_sb360_coverage).
_GK_DOMAIN_TYPES = ("shot", "shot_penalty", "shot_freekick", "cross", "goalkick", "keeper_save")

_COMPANION_FEATURES = ("nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal")

#: The tidy shard schema and its generation token, pinned TOGETHER (4.77.1): the ``for_each``
#: fingerprint digests ``token_inputs`` only, so changing the emitted columns while leaving the token
#: resolves to the SAME generation directory and silently combines the old schema. A run-time check
#: in ``measure_match`` fails at the first shard if the two drift.
_SHARD_SCHEMA_VERSION = "sb360-licensed-1"
_EMITTED_SHARD_COLUMNS = ("match_id", "kind", "subject", "metric", "value", "denominator", "detail")


@functools.cache
def _type_id_to_name() -> dict[int, str]:
    import silly_kicks.spadl as spadl

    df = spadl.actiontypes_df()
    return {int(i): str(n) for i, n in zip(df["type_id"], df["type_name"], strict=True)}


def _row(match_id, kind, subject, metric, value, denominator=float("nan"), detail=None) -> dict:
    return {
        "match_id": str(match_id),
        "kind": kind,
        "subject": subject,
        "metric": metric,
        "value": None if value is None else float(value),
        "denominator": float(denominator) if denominator is not None else float("nan"),
        "detail": detail,
    }


def measure_match(item) -> pd.DataFrame:
    """One SB360 match -> a tidy long-form coverage/verdict table (one metric per row)."""
    import pandas as pd

    from silly_kicks.id_compat import canonical_id
    from silly_kicks.tracking import (
        REGION_OBSERVATION_SOURCE_VALUES,
        VISIBLE_AREA_SOURCE_VALUES,
        add_visible_area_coverage,
        link_actions_to_frames,
    )
    from silly_kicks.tracking.features import add_action_context

    match_id, actions, frames, visible_area, home = item
    if not isinstance(visible_area, pd.DataFrame):  # narrows for the type checker; always true for statsbomb
        raise TypeError("a statsbomb match must carry a visible_area DataFrame")
    n = len(actions)
    rows: list[dict] = []

    # Pre-link ONCE and thread `links` to every consumer (the CLAUDE.md pre-linking pattern). Without
    # it the battery runs with links=None, so a links-dependent aggregator (add_sync_score) can't run
    # and is recorded as `raises` -- a HARNESS artifact, not a library refusal (the ADR-053 mis-call
    # class). `on_low_coverage="ignore"`: freeze-frame link rate is legitimately partial, not an error.
    links = link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]

    # --- 1. Frame-existence coverage per GK-domain action type -------------------------------
    has_frame = (
        {canonical_id(a) for a in visible_area["action_id"]}
        if visible_area is not None and len(visible_area)
        else set()
    )
    type_name = actions["type_id"].map(_type_id_to_name())
    action_has_frame = actions["action_id"].map(lambda a: canonical_id(a) in has_frame)
    rows.append(_row(match_id, "frame_coverage", "all", "frame_existence_rate", action_has_frame.mean(), n))
    for atype in _GK_DOMAIN_TYPES:
        mask = type_name == atype
        k = int(mask.sum())
        if k:
            rows.append(
                _row(match_id, "frame_coverage", atype, "frame_existence_rate", action_has_frame[mask].mean(), k)
            )

    # --- 2. Battery population: VERDICTS/coverage, never synthetic values ---------------------
    battery = run_add_star_battery(actions, frames, links=links, home_team_id=home)
    for name, result in battery.items():
        if isinstance(result, str):  # a real-data refusal is a recorded result, not a crash
            rows.append(_row(match_id, "battery_raises", name, "raised", 1.0, detail=result))
            continue
        for col in result.columns:
            non_nan = float(result[col].notna().mean()) if len(result) else float("nan")
            rows.append(_row(match_id, "battery_column", f"{name}.{col}", "non_nan_fraction", non_nan, len(result)))

    # --- 3. Count-feature companions under the REAL visible_area ------------------------------
    ctx = add_action_context(actions, frames, links=links, visible_area=visible_area)
    for feat in _COMPANION_FEATURES:
        src_col, frac_col = f"{feat}_observed_source", f"{feat}_observed_fraction"
        counts = ctx[src_col].value_counts()
        for token in (*REGION_OBSERVATION_SOURCE_VALUES, "unlinked"):
            rows.append(_row(match_id, "companion_source", f"{feat}.{token}", "count", int(counts.get(token, 0)), n))
        observed = ctx.loc[ctx[src_col] == "observed", frac_col]
        rows.append(
            _row(
                match_id,
                "companion_fraction",
                feat,
                "mean_observed_fraction",
                float(observed.mean()) if len(observed) else float("nan"),
                len(observed),
            )
        )

    # --- 4. Pitch-level coverage under the REAL visible_area (explicit, segregated) -----------
    cov = add_visible_area_coverage(actions, visible_area=visible_area, links=links)
    obs = cov.loc[cov["visible_area_source"] == "observed", "visible_area_fraction"]
    rows.append(
        _row(
            match_id,
            "pitch_coverage",
            "observed_pitch_fraction",
            "mean",
            float(obs.mean()) if len(obs) else float("nan"),
            len(obs),
        )
    )
    src_counts = cov["visible_area_source"].value_counts()
    for token in VISIBLE_AREA_SOURCE_VALUES:
        rows.append(_row(match_id, "pitch_coverage_source", token, "count", int(src_counts.get(token, 0)), n))

    # --- 5. Roster keeper-resolution rate ----------------------------------------------------
    resolved = float(actions["player_name"].notna().mean()) if "player_name" in actions.columns else 0.0
    rows.append(_row(match_id, "roster", "player_identity", "resolution_rate", resolved, n))

    frame = pd.DataFrame(rows)
    # 4.77.1: check the keys the rows ACTUALLY carry against the declaration -- never build with
    # `columns=_EMITTED_SHARD_COLUMNS`, which SELECTS to it (a dropped key vanishes, a missing one
    # arrives as NaN, and the guard certifies both). A raise, not an assert: this fires at the first
    # shard and must survive `python -O`, which strips asserts (the ADR-037 CompetitionMismatchError rule).
    if set(frame.columns) != set(_EMITTED_SHARD_COLUMNS):
        raise RuntimeError(
            f"shard schema drift: {set(frame.columns) ^ set(_EMITTED_SHARD_COLUMNS)}; bump _SHARD_SCHEMA_VERSION"
        )
    return frame[list(_EMITTED_SHARD_COLUMNS)]


def _fixture_items():
    """The committed open-360 golden slice (WWC2023, 6 freeze-frames) -- CI-reachable, no network."""
    from scripts._loader_pining import build_statsbomb_match

    slice_dir = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "three-sixty"
    paths = {"events": slice_dir / "events.json", "freeze_frames": slice_dir / "frames.json"}
    actions, frames, home, visible_area, _report = build_statsbomb_match(paths, 3893795)
    yield (3893795, actions, frames, visible_area, home)


def _pining_items(match_ids, token):
    from scripts._loader_pining import load_statsbomb_matches

    for _prov, mid, actions, frames, home, visible_area in load_statsbomb_matches(match_ids=match_ids, token=token):
        yield (mid, actions, frames, visible_area, home)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-root", type=pathlib.Path, default=pathlib.Path(DEFAULT_SHARD_ROOT))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/research/sb360_licensed_coverage"))
    ap.add_argument("--match-ids-json", type=pathlib.Path, default=None, help="JSON list of match ids to slice.")
    ap.add_argument("--fixture-only", action="store_true", help="Run the committed open-360 slice; no network.")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    # FIRST, before any corpus work (ADR-037): the bare HEAD SHA is identical clean-or-dirty.
    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    if args.fixture_only:
        items = _fixture_items()
    else:
        match_ids = json.loads(args.match_ids_json.read_text()) if args.match_ids_json else None
        items = _pining_items(match_ids, token=None)

    res = for_each(
        items,
        key=lambda it: str(it[0]),
        work=measure_match,
        shard_root=args.shard_root,
        token_inputs={"schema": _SHARD_SCHEMA_VERSION},
        tag=args.tag,
        label="match",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    reconcile(res.shard_dir, args.out / "coverage.parquet", tag=args.tag)
    # Stamp the CANONICAL research-artifact provenance (ADR-037/ADR-056): `run_commit` +
    # `run_tree_dirty` (+ `run_tree_state`), NOT git_provenance()'s raw `commit`/`dirty` keys -- the
    # `test_artifact_provenance_output` gate reads `run_commit`/`run_tree_dirty` by name. Matches the
    # convention in build_gkdv_arm_values.py.
    (args.out / f"manifest_{args.tag}.json").write_text(
        json.dumps(
            {
                **res.manifest(),
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "run_tree_state": prov["tree_state"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"attempted={res.attempted} processed={res.attempted - res.skipped - res.failed} "
        f"skipped={res.skipped} failed={res.failed}"
    )


if __name__ == "__main__":
    main()
