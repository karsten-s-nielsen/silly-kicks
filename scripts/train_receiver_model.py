"""Task 6: train the PUBLIC receiver model on SB360 open-data (D1).

SB360 freeze frames carry the pre-pass positions with real team ids (ADR-062) but NO velocity, so the
public model is positions-only (leakage-free everywhere). Trained on COMPLETED passes, where
``resolve_next_touch_receiver`` gives the observed receiver as ground truth. ADR-052 shards candidate
rows per match; ADR-037 stamps provenance; the bundle is ADR-011 (SHA + chirality + feature contract).

M2 -- the SB360 visible area truncates the NEGATIVE candidate set (the model ranks among visible
teammates in training but ALL teammates at GS serve), so the per-frame candidate-count distribution is
recorded for the train-vs-serve shift check.

ADR-128 stage contract (decisions P5, P17): ``extract[match]`` over the staged ``--dataset`` ->
``pool[match]`` (optional ``--pool-dataset``) -> ``fit`` -> ``deploy[match]`` (owner variant with
``--public-bundle``, over ``--deployment-dataset``) -> ``assemble`` (the bundle). Every corpus is a STAGED
neutral dataset; the corpus visibility and the requested corpus size arrive as parameters
(``--corpus-visibility``, ``--corpus-n-matches``, from ``datasets.toml`` + the staging lock), so the module
never reaches the pining client (the I1 gate). `--feature-set owner --provider gradientsports` trains the
GS velocity variant (Task 6b); the default trains the public SB360 model. The DGX runs the same CLI on
locally materialised staged data (ruling 11).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from scripts._cover_shadow_thresholds import MIN_RECEIVER_MARGIN
from scripts._stages import ItemStage, Param, Pipeline, Stage, main
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.spadl.config import actiontypes, results
from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.tracking._receiver import _OWNER_EXTRA_COLS, _PUBLIC_COLS, ReceiverModel, receiver_candidate_features

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


def _top1_accuracy(model, rows: pd.DataFrame, feature_set: str) -> float:
    """Per-pass top-1: the argmax-scored candidate is the labeled receiver. Shared by cv_top1 + the
    pooling gate so 'train here, evaluate there' uses ONE scoring definition."""
    names = _feature_names(feature_set)
    test = rows.reset_index(drop=True).copy()  # unique index -> grp.loc[idxmax] is a scalar (L5)
    test["_p"] = model.predict_candidates(test[names])
    hits = n = 0
    for _, grp in test.groupby(["game_id", "action_id"]):
        n += 1
        hits += int(grp.loc[grp["_p"].idxmax(), "label"] == 1)
    return hits / n if n else float("nan")


def cv_top1(rows: pd.DataFrame, feature_set: str, n_splits: int = 5) -> tuple[float, list[float]]:
    """Held-out top-1 accuracy (per pass: the argmax candidate is the observed receiver), GroupKFold on game."""
    from sklearn.model_selection import GroupKFold

    names = _feature_names(feature_set)
    n_groups = len(pd.unique(rows["game_id"].to_numpy()))
    k = min(n_splits, n_groups)
    if k < 2:
        return float("nan"), []
    fold_top1 = []
    for tr, te in GroupKFold(n_splits=k).split(rows, groups=rows["game_id"].to_numpy()):
        m = ReceiverModel(feature_set).fit(rows.iloc[tr][names], rows.iloc[tr]["label"])
        fold_top1.append(_top1_accuracy(m, rows.iloc[te], feature_set))
    return float(np.nanmean(fold_top1)), fold_top1


def pooling_gate(primary: pd.DataFrame, pool: pd.DataFrame, feature_set: str, *, n_splits: int = 5) -> dict:
    """Q3: does adding ``pool`` (clean-id GS) regress top-1 on the PRIMARY (SB360) held-out? GroupKFold on
    the PRIMARY games only; the pool is added to TRAIN, NEVER to TEST. Keep the pool iff pooled >=
    primary-only (no regression) -- SB360 is the serve distribution and the leakage-free-everywhere target,
    so the noisier GS leg must EARN inclusion rather than be pooled in naively."""
    from sklearn.model_selection import GroupKFold

    names = _feature_names(feature_set)
    games = primary["game_id"].to_numpy()
    k = min(n_splits, len(pd.unique(games)))
    if k < 2:
        return {
            "keep_pool": False,
            "reason": "too few primary games to gate",
            "primary_only_top1": float("nan"),
            "pooled_top1": float("nan"),
            "margin": float("nan"),
        }
    prim_only, pooled = [], []
    for tr, te in GroupKFold(n_splits=k).split(primary, groups=games):
        train_prim, test_prim = primary.iloc[tr], primary.iloc[te]
        m0 = ReceiverModel(feature_set).fit(train_prim[names], train_prim["label"])
        prim_only.append(_top1_accuracy(m0, test_prim, feature_set))
        train_pooled = pd.concat([train_prim, pool], ignore_index=True)
        m1 = ReceiverModel(feature_set).fit(train_pooled[names], train_pooled["label"])
        pooled.append(_top1_accuracy(m1, test_prim, feature_set))  # evaluated on held-out PRIMARY only
    p0, p1 = float(np.nanmean(prim_only)), float(np.nanmean(pooled))
    return {"primary_only_top1": p0, "pooled_top1": p1, "margin": p1 - p0, "keep_pool": bool(p1 >= p0)}


def _coverage_counters(item, frame) -> dict:
    """for_each counters closure: per-match label coverage. Survives even a fully-dropped match (empty
    frame) via its own sidecar, so drop-on-ambiguous thinning is visible in the manifest, not silent."""
    completed = int((item[1]["type_id"].isin(list(_PASS_TYPES)) & (item[1]["result_id"] == _SUCCESS)).sum())
    kept = int(frame.groupby(["game_id", "action_id"]).ngroups) if len(frame) else 0
    return {"n_completed_passes": completed, "n_kept_passes": kept}


def velocity_ablation_completed(rows: pd.DataFrame) -> dict:
    """Task 6b M-A(i): velocity's contribution on held-out COMPLETED passes (full ground truth).

    ``rows`` = GS candidate rows extracted with ``feature_set="owner"`` (both subsets present), so the
    ablation is same-corpus, off the R1 failed-easy-tail where positions already resolve the receiver.
    """
    pos_top1, _ = cv_top1(rows, "public")  # positions only
    posvel_top1, _ = cv_top1(rows, "owner")  # positions + velocity
    return {
        "top1_positions": pos_top1,
        "top1_positions_velocity": posvel_top1,
        "velocity_delta": posvel_top1 - pos_top1,
        "caveat": "measured on COMPLETED passes as the best proxy for velocity's failed-pass value "
        "(the completed->failed transfer, H1, is the model's caveat, not velocity's)",
    }


def deployment_decision(pub: dict, own: dict, *, min_margin=MIN_RECEIVER_MARGIN) -> dict:
    """Pure M-A(ii) decision from two ``receiver_failed_pass_accuracy`` dicts (single-match OR pooled).

    Kept separate from the model-application so the corpus driver can POOL per-match counts and call this
    on the pooled ``top1``/``n_scored`` -- a single arithmetic, one place, single- or multi-match.

    R1: a NON-decisive result reads *the GS variant's advantage is unmeasurable on the easy tail*, NOT
    "velocity/GS-native adds nothing" -- ``velocity_ablation_completed`` is the un-easy-tail-limited read.
    """
    both_scored = bool(own["n_scored"] and pub["n_scored"])
    margin = (own["top1"] - pub["top1"]) if both_scored else float("nan")
    return {
        "public_top1": pub["top1"],
        "owner_top1": own["top1"],
        "margin": margin,
        "decisive": bool(both_scored and margin >= min_margin),
        "coverage": own.get("coverage"),
        "n_scored": int(own["n_scored"]),
        "r1_caveat": "non-decisive = unmeasurable on the easy tail, NOT 'adds nothing' "
        "(see velocity_ablation_completed for the un-easy-tail-limited read)",
    }


def deployment_gate(
    public_model, gs_owner_model, actions, frames, *, min_margin=MIN_RECEIVER_MARGIN, links=None
) -> dict:
    """Task 6b M-A(ii): which variant to SERVE on GS-failed (the bundling decision), on the validated
    subset -- single-match convenience over :func:`deployment_decision`."""
    from scripts._receiver_validation import receiver_failed_pass_accuracy

    pub = receiver_failed_pass_accuracy(public_model, actions, frames, links=links)
    own = receiver_failed_pass_accuracy(gs_owner_model, actions, frames, links=links)
    return deployment_decision(pub, own, min_margin=min_margin)


def _deployment_counts_for_match(public_model, gs_owner_model, actions, frames, *, match_id) -> pd.DataFrame:
    """One row of RAW pooled-able counts for a match: hits + n for each model, so the corpus driver can
    sum them and compute one pooled :func:`deployment_decision` (a per-match margin would be noise)."""
    from scripts._receiver_validation import receiver_failed_pass_accuracy

    links, _ = link_actions_to_frames(actions, frames)
    pub = receiver_failed_pass_accuracy(public_model, actions, frames, links=links)
    own = receiver_failed_pass_accuracy(gs_owner_model, actions, frames, links=links)

    def _hits(acc: dict) -> int:
        return round(acc["top1"] * acc["n_scored"]) if acc["n_scored"] else 0

    return pd.DataFrame(
        [
            {
                "match_id": str(match_id),
                "pub_n_scored": int(pub["n_scored"]),
                "pub_hits": _hits(pub),
                "own_n_scored": int(own["n_scored"]),
                "own_hits": _hits(own),
                "n_covered": int(own["n_covered"]),
                "n_intercepted": int(own["n_intercepted"]),
            }
        ]
    )


_DEPLOY_SHARD_COLUMNS = [
    "match_id",
    "pub_n_scored",
    "pub_hits",
    "own_n_scored",
    "own_hits",
    "n_covered",
    "n_intercepted",
]


def _pooled_deployment(counts: pd.DataFrame) -> dict:
    """Pool per-match deployment counts into ONE :func:`deployment_decision` on the corpus-wide top-1."""

    def _acc(n_col: str, hit_col: str) -> dict:
        n = int(counts[n_col].sum())
        return {"n_scored": n, "top1": (float(counts[hit_col].sum()) / n) if n else float("nan")}

    decision = deployment_decision(_acc("pub_n_scored", "pub_hits"), _acc("own_n_scored", "own_hits"))
    n_covered, n_intercepted = int(counts["n_covered"].sum()), int(counts["n_intercepted"].sum())
    decision["coverage"] = (n_covered / n_intercepted) if n_intercepted else float("nan")
    decision["n_covered"] = n_covered
    decision["n_intercepted"] = n_intercepted
    return decision


def _candidate_count_distribution(rows: pd.DataFrame) -> dict:
    per_pass = rows.groupby(["game_id", "action_id"])["n_candidates"].first()
    if per_pass.empty:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(per_pass.mean()),
        "p10": float(per_pass.quantile(0.10)),
        "p50": float(per_pass.quantile(0.50)),
        "p90": float(per_pass.quantile(0.90)),
    }


def _namespace_game_ids(rows, provider):
    """F3: provider-prefix ``game_id`` so a pooled corpus can never collide ids across providers (which
    would under-count ``n_matches`` and make the post-pool CV provider-blind). The prefix is bijective, so
    grouping/CV is unchanged for a single provider. Single-sourced so the primary and the pool stage
    produce the SAME in-memory rows."""
    if not len(rows):
        return rows
    return rows.assign(game_id=f"{provider}:" + rows["game_id"].astype(str))


# ---- ADR-128 stage contract -------------------------------------------------------------------------
#
# extract[match] (primary dataset) -> pool[match] (optional pool dataset) -> fit (one unit) ->
# deploy[match] (owner variant with a public bundle only) -> assemble (one unit, the bundle).
#
# Decision P5 / P17: every input is a STAGED dataset parameter; corpus visibility and the requested
# corpus size are explicit parameters from `datasets.toml` + `datasets.lock.json` (the campaign
# controller passes them), so nothing in this module's import closure reads the pining client (I1).

_PARAMS = (
    Param("dataset", str, None, required=True, help="the PRIMARY staged dataset (the serve target)"),
    Param("feature_set", str, "public", choices=("public", "owner")),
    Param("provider", str, "statsbomb", help="the primary provider: selects the labeling strategy"),
    Param("pool_dataset", str, None, help="a SECOND staged dataset to POOL in if it earns inclusion (Q3)"),
    Param("pool_provider", str, None, help="the pool dataset's provider (its labeling strategy)"),
    Param("deployment_dataset", str, None, help="owner variant: the dataset the M-A(ii) gate scores"),
    Param("public_bundle", str, None, help="owner variant: the public bundle dir for the M-A(ii) gate"),
    Param("match_ids_json", str, None, help="JSON list of PRIMARY unit stems to keep (default: all)"),
    Param("min_rows", int, 1),
    # A corpus-volume floor so a partial download cannot silently bundle a model. Injectable; RAISE it to
    # the corpus's expected pass volume for a production bundling run (mirrors build_rq_pass_scores).
    Param("min_passes", int, 200),
    Param("corpus_visibility", str, None, required=True, choices=("public", "restricted")),
    Param("corpus_n_matches", int, None, required=True, help="the requested corpus size (the staging lock)"),
)


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


def _fit_work(key, ctx):
    import tempfile

    from scripts._corpus_identity import reproducibility
    from scripts._stages import FileSet, item_counters

    p = ctx.params
    fs, provider = p["feature_set"], p["provider"]
    rows = _namespace_game_ids(ctx.upstream("extract"), provider)
    coverage, _missing = item_counters(ctx, "extract")
    failed = ctx.summary("extract")["n_failed"]
    if failed:  # a thinned corpus is VISIBLE in metrics.json, never silent (the legacy fit the rest)
        coverage = {**coverage, "n_failed_matches": failed}

    # `reconcile` returns a COLUMN-LESS frame when no shard is non-empty, so guard `both_classes` on
    # `len(rows)` (F2) -- else `rows["label"]` raises KeyError before the vacuity message can fire.
    n_passes = int(rows.groupby(["game_id", "action_id"]).ngroups) if len(rows) else 0
    both_classes = len(rows) > 0 and bool((rows["label"] == 1).any()) and bool((rows["label"] == 0).any())
    if len(rows) < p["min_rows"] or n_passes < p["min_passes"] or not both_classes:
        raise RuntimeError(
            f"vacuous training set: {len(rows)} rows / {n_passes} passes "
            f"(need >= {p['min_rows']} rows, >= {p['min_passes']} passes, BOTH classes present)"
        )

    # Q3: a POOL provider (GS) earns inclusion only if it does NOT regress the PRIMARY held-out top-1.
    providers, pool_gate = {provider}, None
    if p["pool_dataset"]:
        pool_rows = _namespace_game_ids(ctx.upstream("pool"), p["pool_provider"])
        pool_coverage, _ = item_counters(ctx, "pool")
        pool_gate = pooling_gate(rows, pool_rows, fs)
        # F1: an EMPTY pool ties the gate (margin 0 -> keep_pool True) but contributes nothing; gating on
        # len(pool_rows) stops a zero-contribution pool from falsely stamping providers_trained.
        if pool_gate["keep_pool"] and len(pool_rows):
            rows = pd.concat([rows, pool_rows], ignore_index=True)
            providers.add(p["pool_provider"])
        pool_gate = {**pool_gate, "pool_provider": p["pool_provider"], "pool_coverage": pool_coverage}

    names = _feature_names(fs)
    model = ReceiverModel(fs).fit(rows[names], rows["label"])
    model.shipped_variant = fs
    top1, fold_top1 = cv_top1(rows, fs)
    corpus_label = p["corpus_visibility"]  # decision P5: a parameter, never read in-job
    prov = ctx.prov
    manifest = {
        "schema": _SHARD_SCHEMA_VERSION,
        "feature_set": fs,
        "provider": provider,
        "labeling_strategy": labeling_strategy_for_provider(provider),
        "providers_trained": sorted(providers),  # the actual pool the shipped model saw
        "n_matches": int(rows["game_id"].nunique()),
        "n_passes": n_passes,
        "n_candidate_rows": len(rows),
        "top1_cv": top1,
        "top1_by_fold": fold_top1,
        "candidate_count_distribution": _candidate_count_distribution(rows),  # M2
        "label_coverage": coverage,  # kept vs completed passes -> drop-on-ambiguous thinning is visible
        "visible_area_caveat": "SB360 truncates the negative candidate set; see the train-vs-serve shift (M2).",
        "corpus_visibility": corpus_label,
        "corpus_n_matches": int(p["corpus_n_matches"]),
        **reproducibility(
            "public" if corpus_label == "public" else "restricted", sorted(providers), training_commit=prov["commit"]
        ),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    if pool_gate is not None:  # Q3 gate outcome + the pool's own coverage (load-bearing: did GS earn inclusion?)
        manifest["pooling_gate"] = pool_gate
    if fs == "owner":  # M-A(i): velocity's contribution on COMPLETED passes
        manifest["velocity_ablation"] = velocity_ablation_completed(rows)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(pathlib.Path(tmp) / "model")
        files = {
            f"model/{q.name}": q.read_bytes() for q in sorted((pathlib.Path(tmp) / "model").iterdir()) if q.is_file()
        }
    files["manifest.json"] = json.dumps(manifest).encode("utf-8")
    return FileSet(files)


def _deploy_refs(ctx):
    if ctx.params["feature_set"] != "owner" or not ctx.params["public_bundle"]:
        return []
    return _staged_refs(ctx, "deployment_dataset")


_DEPLOY_MODELS: dict = {}


def _deploy_models(ctx):
    """The public bundle and this run's owner model, loaded ONCE per worker (keyed by the fit token)."""
    import tempfile

    from scripts._stages import materialize_fileset

    tok = ctx.tokens["fit"]
    if tok not in _DEPLOY_MODELS:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="sk-recv-"))
        materialize_fileset(ctx, "fit", "_output", tmp)
        _DEPLOY_MODELS.clear()
        _DEPLOY_MODELS[tok] = (
            ReceiverModel.load(pathlib.Path(ctx.params["public_bundle"])),
            ReceiverModel.load(tmp / "model"),
        )
    return _DEPLOY_MODELS[tok]


def _deploy_work(item, ctx):
    mid, actions, frames = item
    public_model, owner_model = _deploy_models(ctx)
    return _deployment_counts_for_match(public_model, owner_model, actions, frames, match_id=mid)


def _deploy_reduce(results, ctx):
    if not results:
        return {}
    return _pooled_deployment(_concat_non_empty(results, ctx))


def _assemble_work(key, ctx):
    from scripts._stages import FileSet

    out = ctx.upstream("fit")
    manifest = json.loads(out.files["manifest.json"])
    # M-A(ii): the deployment gate on GS-FAILED, stamped into the same provenanced manifest (L4a).
    if ctx.params["feature_set"] == "owner" and ctx.params["public_bundle"]:
        manifest["deployment"] = ctx.upstream("deploy")
    files = {name.removeprefix("model/"): data for name, data in out.files.items() if name.startswith("model/")}
    files["metrics.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    print(json.dumps(manifest, indent=2))
    return FileSet(files)


PIPELINE = Pipeline(
    name="receiver_model",
    module="scripts.train_receiver_model",
    params=_PARAMS,
    stages=(
        ItemStage(
            name="extract",
            refs=lambda ctx: _staged_refs(ctx, "dataset"),
            key=lambda ref: ref.stem,  # pre-migration key was the shard stem (match_id); _KEY_EXCEPTIONS
            load=lambda ref, ctx: _load_staged(ref, ctx, "dataset"),
            work=_rows_work("provider"),
            counters=_coverage_counters,
            reduce=_concat_non_empty,
            params=("feature_set", "provider", "match_ids_json"),
            declare=_rows_declare("provider"),
            cross_commit_version="receiver-rows-2",
            uses_dataset=True,
            max_failures=None,  # faithful: the legacy for_each recorded a failed match and fitted the rest
        ),
        ItemStage(
            name="pool",
            refs=lambda ctx: _staged_refs(ctx, "pool_dataset"),
            key=lambda ref: ref.stem,
            load=lambda ref, ctx: _load_staged(ref, ctx, "pool_dataset"),
            work=_rows_work("pool_provider"),
            counters=_coverage_counters,
            reduce=_concat_non_empty,
            params=("feature_set", "pool_dataset", "pool_provider"),
            declare=_rows_declare("pool_provider"),
            cross_commit_version="receiver-rows-2",
            # It reads `--pool-dataset`: that dataset's identity keys the token (it was keyed by the name
            # only), and a download job fetches its units.
            uses_dataset=True,
            dataset_param="pool_dataset",
            max_failures=None,
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("extract", "pool"),
            reduce=lambda r, ctx: r["only"],
            params=(
                "feature_set",
                "provider",
                "pool_dataset",
                "pool_provider",
                "min_rows",
                "min_passes",
                "corpus_visibility",
                "corpus_n_matches",
            ),
        ),
        ItemStage(
            name="deploy",
            refs=_deploy_refs,
            key=lambda ref: ref.stem,  # pre-migration key was the match_id; _KEY_EXCEPTIONS
            load=lambda ref, ctx: _load_staged(ref, ctx, "deployment_dataset"),
            work=_deploy_work,
            needs=("fit",),
            reduce=_deploy_reduce,
            params=("feature_set", "public_bundle", "deployment_dataset"),
            declare=lambda p: {
                "schema": "receiver-deploy-1",
                "provider": p["provider"],
                "columns": _DEPLOY_SHARD_COLUMNS,
            },
            max_failures=None,
        ),
        Stage(
            name="assemble",
            units=lambda ctx: ["only"],
            work=_assemble_work,
            needs=("fit", "deploy"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
