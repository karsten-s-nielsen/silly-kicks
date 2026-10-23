"""ADR-129: the reusable `extract` and `pool` stages of `scripts.train_receiver_model`; their closure must
not contain the trainer (`test_stage_modules`). The fit-side stages (fit/deploy/assemble) stay in the
trainer and import the shared helpers (`_feature_names`, `labeling_strategy_for_provider`,
`_namespace_game_ids`, `_staged_refs`, `_load_staged`, `_concat_non_empty`) from here."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts._stages import ItemStage
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.spadl.config import actiontypes, results
from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.tracking._receiver import _OWNER_EXTRA_COLS, _PUBLIC_COLS, receiver_candidate_features

_T = {n: i for i, n in enumerate(actiontypes)}
_R = {n: i for i, n in enumerate(results)}
_PASS_TYPES = {_T["pass"], _T["cross"]}
_SUCCESS = _R["success"]
_SHARD_SCHEMA_VERSION = "receiver-rows-2"  # v2: per-provider labeling (id / trajectory), drop-on-ambiguous
#: Q2 label-admission lane width -- DELIBERATELY tighter than trajectory_weak_labels' 5.0 m TRAVEL gate
#: (which it overloads as its perpendicular bound). A candidate farther than this from the
#: release->reception ray is not "clearly the target": the pass is DROPPED (counted), never guessed onto
#: the nearest visible teammate -- which, on visibility-truncated SB360, would be a confident mislabel.
_LABEL_LANE_WIDTH_M = 2.0


def _feature_names(feature_set: str) -> list[str]:
    return list(_PUBLIC_COLS + (_OWNER_EXTRA_COLS if feature_set == "owner" else []))


def _emitted_columns(feature_set: str) -> list[str]:
    """The shard schema (4.77.1). Feature-set-dependent, so it is threaded into the generation token so
    an in-set column addition to _PUBLIC_COLS/_OWNER_EXTRA_COLS moves the digest (no stale-shard reuse)."""
    return [*_feature_names(feature_set), "candidate_id", "label", "game_id", "action_id", "n_candidates"]


def _assert_emitted_schema(df: pd.DataFrame, declared: list[str]) -> None:
    """4.77.1: compare the keys the rows ACTUALLY carry to the declaration -- NEVER
    ``pd.DataFrame(rows, columns=declared)`` as the CHECK, which SELECTS to the declaration (a dropped key
    vanishes, a missing one arrives as NaN). Fails at the FIRST shard on any drift."""
    got, want = set(df.columns), set(declared)
    if got != want:
        raise AssertionError(f"shard schema drift: missing {sorted(want - got)}, extra {sorted(got - want)}")


def labeling_strategy_for_provider(provider: str) -> str:
    """Which receiver-labeling regime a provider's frames admit (Q4) -- decided at FRAME-SET granularity,
    NEVER per-pass. A provider whose freeze/tracking frames carry real player identity uses clean
    ``"id"`` labels; an identity-less freeze-frame provider (SB360, whose rows are numbered -- ADR-062)
    uses ``"trajectory"`` labels.

    An identity provider's pass whose receiver has no in-frame id match is DROPPED (counted), NEVER
    trajectory-labeled: a per-pass "id-match else trajectory" fallback would silently trajectory-label a
    GS pass whose receiver ran off-frame, among the VISIBLE candidates -- a confident mislabel that erodes
    the clean-label guarantee making the identity leg the trusted one in the pool.
    """
    return "trajectory" if provider == "statsbomb" else "id"


def _trajectory_winner(teammates: pd.DataFrame, release: np.ndarray, reception: np.ndarray, lane_width: float):
    """``candidate_id`` of the teammate nearest the release->reception ray (forward of the passer), within
    ``lane_width`` (Q1 reception anchor + Q2 tight bound). ``None`` when none is clearly on the ray -- the
    pass is then dropped, never guessed onto the nearest visible teammate."""
    ray = reception - release
    length = float(np.linalg.norm(ray))
    if length < 1e-9:
        return None
    u = ray / length
    best_perp, best_id = np.inf, None
    for _, tm in teammates.iterrows():
        rel = np.array([float(tm["x"]), float(tm["y"])], dtype=np.float64) - release
        proj = float(rel @ u)
        if proj <= 0.0:  # forward of the passer only
            continue
        perp = float(np.linalg.norm(rel - proj * u))
        if perp < best_perp:
            best_perp, best_id = perp, canonical_id(tm["player_id"])
    return best_id if best_perp <= lane_width else None


def extract_candidate_rows(
    actions: pd.DataFrame, frames: pd.DataFrame, *, feature_set: str = "public", labeling_strategy: str = "id"
) -> pd.DataFrame:
    """Per completed pass, one row per teammate candidate: features + a single-receiver label.

    ``labeling_strategy`` (Q4, from :func:`labeling_strategy_for_provider`): ``"id"`` labels the candidate
    whose real id is the observed next-touch receiver (identity providers; a no-match pass is DROPPED);
    ``"trajectory"`` labels the candidate nearest the release->reception ray within ``_LABEL_LANE_WIDTH_M``
    (identity-less SB360; an ambiguous pass is DROPPED). Both anchor the reception on the SAME next-touch
    action the label identity comes from (Q1). A ball-less owner frame is skipped (Q5), never a crash.
    """
    from silly_kicks.spadl.utils import _resolve_next_touch_positions, resolve_next_touch_receiver
    from silly_kicks.tracking._receiver import (
        NoReleaseDirectionError,
        _acting_attacks_rtl,
        _passer_xy,
        _split_players,
    )

    links, _ = link_actions_to_frames(actions, frames)
    a = actions.reset_index(drop=True)  # positional alignment for _resolve_next_touch_positions/receiver
    positions = _resolve_next_touch_positions(a)  # Int64 positional index of the next same-team touch
    receiver = resolve_next_touch_receiver(a, positions=positions)  # id, from that SAME action (Q1)
    frame_of = dict(zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=True))
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}
    rows = []
    for p, (_idx, act) in enumerate(a.iterrows()):  # p is the positional index (a is reset_index)
        if act["type_id"] not in _PASS_TYPES or int(act["result_id"]) != _SUCCESS:
            continue  # train on COMPLETED passes only
        fr = by_frame.get(canonical_id(frame_of.get(act["action_id"])))
        if fr is None:
            continue
        try:
            feats = receiver_candidate_features(act, fr, feature_set=feature_set)
        except NoReleaseDirectionError:
            continue  # Q5: ball-less owner frame -> skip this pass (never crash the match)
        if feats.empty:
            continue
        if labeling_strategy == "id":
            obs = receiver.iloc[p]
            if obs is None or (isinstance(obs, float) and np.isnan(obs)) or obs is pd.NA:
                continue  # no resolvable receiver -> drop
            matches = [cid for cid in feats["candidate_id"] if same_id(cid, obs)]
            if len(matches) != 1:  # Q4: identity provider, receiver off-frame / not unique -> DROP
                continue
            winner = matches[0]
        else:  # trajectory
            npos = positions.iloc[p]
            if pd.isna(npos):
                continue  # no next touch -> no reception anchor -> drop
            nxt = a.iloc[int(npos)]
            attacks_rtl = _acting_attacks_rtl(fr, act["team_id"])  # reproject action coords -> frame coords

            def _to_frame(x, y, _rtl=attacks_rtl):
                return (105.0 - float(x), 68.0 - float(y)) if _rtl else (float(x), float(y))

            release = np.array(_to_frame(act["start_x"], act["start_y"]), dtype=np.float64)
            reception = np.array(_to_frame(nxt["start_x"], nxt["start_y"]), dtype=np.float64)
            # same passer-exclusion as the feature candidates, so the winner is always among them (Q4/A-F1)
            teammates, _opp = _split_players(fr, act["team_id"], act["player_id"], passer_xy=_passer_xy(act, fr))
            winner = _trajectory_winner(teammates, release, reception, _LABEL_LANE_WIDTH_M)
            if winner is None:  # Q2: ambiguous -> DROP (never all-zero, never label-nearest-regardless)
                continue
        for _, cf in feats.iterrows():
            rows.append(
                {
                    **{c: cf[c] for c in _feature_names(feature_set)},
                    "candidate_id": cf["candidate_id"],
                    "label": 1 if same_id(cf["candidate_id"], winner) else 0,
                    "game_id": act["game_id"],
                    "action_id": act["action_id"],
                    "n_candidates": len(feats),
                }
            )
    cols = _emitted_columns(feature_set)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    _assert_emitted_schema(df, cols)  # keys rows ACTUALLY carry vs the declaration (4.77.1)
    return df[cols]  # deterministic column order, AFTER the schema assertion


def _coverage_counters(item, frame) -> dict:
    """for_each counters closure: per-match label coverage. Survives even a fully-dropped match (empty
    frame) via its own sidecar, so drop-on-ambiguous thinning is visible in the manifest, not silent."""
    completed = int((item[1]["type_id"].isin(list(_PASS_TYPES)) & (item[1]["result_id"] == _SUCCESS)).sum())
    kept = int(frame.groupby(["game_id", "action_id"]).ngroups) if len(frame) else 0
    return {"n_completed_passes": completed, "n_kept_passes": kept}


def _namespace_game_ids(rows, provider):
    """F3: provider-prefix ``game_id`` so a pooled corpus can never collide ids across providers (which
    would under-count ``n_matches`` and make the post-pool CV provider-blind). The prefix is bijective, so
    grouping/CV is unchanged for a single provider. Single-sourced so the primary and the pool stage
    produce the SAME in-memory rows."""
    if not len(rows):
        return rows
    return rows.assign(game_id=f"{provider}:" + rows["game_id"].astype(str))


def _staged_refs(ctx, param: str):
    """The staged units of the dataset named by ``param`` (none when it is unset)."""
    from scripts._driver_sources import staged_flat_refs

    dataset = ctx.params[param]
    if not dataset:
        return []
    refs = staged_flat_refs(ctx.data_root, dataset)
    if param == "dataset" and ctx.params["match_ids_json"]:
        keep = {str(m) for m in json.loads(ctx.params["match_ids_json"])}
        refs = [r for r in refs if r.stem in keep]
    return refs


def _load_staged(ref, ctx, param: str):
    """``(match_id, actions, frames)`` of one staged unit -- the legacy ``_data_dir_source`` shape."""
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    root = dataset_dir(ctx.data_root, ctx.params[param])
    frames = read_dataset_parquet(root / ref.files[0])
    apath = root / ref.files[1]
    actions = read_dataset_parquet(apath) if apath.exists() else pd.DataFrame()
    return (ref.stem, actions, frames)


def _rows_declare(provider_param: str):
    def declare(p):
        provider = p[provider_param]
        return {
            "schema": _SHARD_SCHEMA_VERSION,
            "feature_set": p["feature_set"],
            "provider": provider,
            "labeling_strategy": labeling_strategy_for_provider(provider) if provider else None,
            "columns": _emitted_columns(p["feature_set"]),
        }

    return declare


def _rows_work(provider_param: str):
    def work(item, ctx):
        _mid, actions, frames = item
        strategy = labeling_strategy_for_provider(ctx.params[provider_param])
        return extract_candidate_rows(
            actions, frames, feature_set=ctx.params["feature_set"], labeling_strategy=strategy
        )

    return work


def _concat_non_empty(results, ctx):
    """The legacy ``reconcile``: non-empty shards in sorted key order; a column-less frame when none."""
    from scripts._stages import combine_frames

    return combine_frames([f for f in results.values() if len(f)], ids=("game_id", "player_id", "team_id"))


#: The reusable `extract` and `pool` stages (ADR-129); the lambdas are declared here (``test_stage_modules``).
_EXTRACT = ItemStage(
    name="extract",
    refs=lambda ctx: _staged_refs(ctx, "dataset"),
    key=lambda ref: ref.stem,  # pre-migration key was the shard stem (match_id); _KEY_EXCEPTIONS
    load=lambda ref, ctx: _load_staged(ref, ctx, "dataset"),
    work=_rows_work("provider"),
    counters=_coverage_counters,
    reduce=_concat_non_empty,
    params=("feature_set", "provider", "match_ids_json"),
    declare=_rows_declare("provider"),
    uses_dataset=True,
    max_failures=None,  # faithful: the legacy for_each recorded a failed match and fitted the rest
)
_POOL = ItemStage(
    name="pool",
    refs=lambda ctx: _staged_refs(ctx, "pool_dataset"),
    key=lambda ref: ref.stem,
    load=lambda ref, ctx: _load_staged(ref, ctx, "pool_dataset"),
    work=_rows_work("pool_provider"),
    counters=_coverage_counters,
    reduce=_concat_non_empty,
    params=("feature_set", "pool_dataset", "pool_provider"),
    declare=_rows_declare("pool_provider"),
    # It reads `--pool-dataset`: that dataset's identity keys the token (it was keyed by the name
    # only), and a download job fetches its units.
    uses_dataset=True,
    dataset_param="pool_dataset",
    max_failures=None,
)
STAGES = {"extract": _EXTRACT, "pool": _POOL}
