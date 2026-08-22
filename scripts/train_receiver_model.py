"""Task 6: train the PUBLIC receiver model on SB360 open-data (D1).

SB360 freeze frames carry the pre-pass positions with real team ids (ADR-062) but NO velocity, so the
public model is positions-only (leakage-free everywhere). Trained on COMPLETED passes, where
``resolve_next_touch_receiver`` gives the observed receiver as ground truth. ADR-052 shards candidate
rows per match; ADR-037 stamps provenance; the bundle is ADR-011 (SHA + chirality + feature contract).

M2 -- the SB360 visible area truncates the NEGATIVE candidate set (the model ranks among visible
teammates in training but ALL teammates at GS serve), so the per-frame candidate-count distribution is
recorded for the train-vs-serve shift check.

Owner-run, local pining (NOT DGX). `--feature-set owner --provider gradientsports` trains the GS
velocity variant (Task 6b); the default trains the public SB360 model.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._corpus import artifact_label
from scripts._cover_shadow_thresholds import MIN_RECEIVER_MARGIN
from scripts._driver import for_each, reconcile
from scripts._provenance import git_provenance, require_clean_tree
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


def _load_corpus(provider: str, cache_dir):
    """Stream ``(match_id, actions, frames)`` for the trained provider (ADR-052: a generator, never
    list()). ``statsbomb`` -> SB360 freeze frames (positions-only, public variant); any tracking provider
    (e.g. ``gradientsports``) -> real tracking frames WITH velocity, which the owner variant requires."""
    if provider == "statsbomb":
        from scripts._loader_pining import load_statsbomb_matches

        return ((m[1], m[2], m[3]) for m in load_statsbomb_matches(cache_dir=cache_dir))  # F5: honor --cache-dir
    from scripts._loader_pining import load_matches

    return ((m[1], m[2], m[3]) for m in load_matches(providers=[provider], cache_dir=cache_dir))


def _resolve_deployment(public_bundle, owner_model, provider, cache_dir, shard_root, out_dir, prov_commit) -> dict:
    """M-A(ii): pool per-match public-vs-owner failed-pass accuracy over the trained provider's corpus and
    reduce to ONE :func:`deployment_decision`. A SECOND sharded pass (its own generation), resumable."""
    public_model = ReceiverModel.load(public_bundle)
    res = for_each(
        _load_corpus(provider, cache_dir),
        key=lambda t: str(t[0]),
        work=lambda t: _deployment_counts_for_match(public_model, owner_model, t[1], t[2], match_id=t[0]),
        shard_root=shard_root,
        token_inputs={
            "schema": "receiver-deploy-1",
            "provider": provider,  # F4: two owner runs with different --provider must not share a generation
            "columns": _DEPLOY_SHARD_COLUMNS,
            "commit": prov_commit,
        },
        label="deploy",
    )
    pooled = reconcile(res.shard_dir, out_dir / "deployment_counts.parquet", tag="deploy")
    return _pooled_deployment(pooled)


def _extract_provider_rows(provider, feature_set, shard_root, cache_dir, out_path, tag):
    """Shard + reconcile one provider's candidate rows with ITS per-provider labeling strategy (Q4), and
    return ``(rows, coverage_counters)``. Primary and pool providers get DISTINCT generations (the token
    keys on ``provider``), so they never collide under one ``shard_root``."""
    strategy = labeling_strategy_for_provider(provider)
    res = for_each(
        _load_corpus(provider, cache_dir),
        key=lambda t: str(t[0]),
        work=lambda t: extract_candidate_rows(t[1], t[2], feature_set=feature_set, labeling_strategy=strategy),
        shard_root=shard_root,
        token_inputs={
            "schema": _SHARD_SCHEMA_VERSION,
            "feature_set": feature_set,
            "provider": provider,
            "labeling_strategy": strategy,
            "columns": _emitted_columns(feature_set),
        },
        counters=_coverage_counters,
        label=tag,
    )
    rows = reconcile(res.shard_dir, out_path, tag=tag)
    # F3: namespace game_id by provider so a pooled corpus can never collide game_ids across providers
    # (which would under-count n_matches and make the post-pool CV provider-blind). Per-provider parquet
    # keeps raw ids; the returned in-memory rows -- what the model fits + the manifest counts -- are unique.
    if len(rows):
        rows = rows.assign(game_id=f"{provider}:" + rows["game_id"].astype(str))
    return rows, dict(res.counters)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the public (SB360) / owner (GS) receiver model.")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--shard-root", type=pathlib.Path, required=True)
    ap.add_argument("--feature-set", choices=["public", "owner"], default="public")
    ap.add_argument("--provider", default="statsbomb")
    ap.add_argument(
        "--pool-provider",
        default=None,
        help="a SECOND provider to POOL into the public model IF it earns inclusion on the primary held-out "
        "(Q3) -- e.g. --provider statsbomb --pool-provider gradientsports. The primary is the serve target.",
    )
    ap.add_argument("--cache-dir", default=None, help="pining cache dir for a tracking provider (owner variant)")
    ap.add_argument(
        "--public-bundle",
        type=pathlib.Path,
        default=None,
        help="public bundle dir -> run the M-A(ii) deployment gate (public vs this owner variant) on GS-failed",
    )
    ap.add_argument(
        "--owner-rows",
        type=pathlib.Path,
        default=None,
        help="reuse a pre-extracted candidate_rows.parquet and SKIP the training corpus re-parse (the "
        "deployment gate still parses the provider once). Lean path for completing the owner M-A resolution "
        "-- a GS match is ~4M frames / ~74s to parse, so re-parsing it just to skip an existing shard is pure "
        "waste (ADR-052: for_each resumes work, not item production).",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--min-rows", type=int, default=1)
    # A corpus-volume floor so a partial download cannot silently bundle a model. Injectable; RAISE it to
    # the corpus's expected pass volume for a production bundling run (mirrors build_rq_pass_scores).
    ap.add_argument("--min-passes", type=int, default=200)
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.owner_rows is not None:
        # Lean path: reuse the already-extracted rows; only the deployment gate re-parses the provider.
        rows, coverage = pd.read_parquet(args.owner_rows), {"note": "rows from --owner-rows; coverage not recomputed"}
    else:
        # PRIMARY corpus (the serve target; SB360 -> trajectory labels, a tracking provider -> id labels).
        rows, coverage = _extract_provider_rows(
            args.provider, args.feature_set, args.shard_root, args.cache_dir, args.out / "candidate_rows.parquet", "all"
        )

    # `reconcile` returns a COLUMN-LESS frame when no shard is non-empty, so guard `both_classes` on
    # `len(rows)` (F2) -- else `rows["label"]` raises KeyError before the vacuity message can fire.
    n_passes = int(rows.groupby(["game_id", "action_id"]).ngroups) if len(rows) else 0
    both_classes = len(rows) > 0 and bool((rows["label"] == 1).any()) and bool((rows["label"] == 0).any())
    if len(rows) < args.min_rows or n_passes < args.min_passes or not both_classes:
        raise SystemExit(
            f"vacuous training set: {len(rows)} rows / {n_passes} passes "
            f"(need >= {args.min_rows} rows, >= {args.min_passes} passes, BOTH classes present)"
        )

    # Q3: a POOL provider (GS) earns inclusion only if it does NOT regress the PRIMARY held-out top-1.
    providers, pool_gate, pool_coverage = {args.provider}, None, None
    if args.pool_provider:
        pool_rows, pool_coverage = _extract_provider_rows(
            args.pool_provider,
            args.feature_set,
            args.shard_root,
            args.cache_dir,
            args.out / "pool_rows.parquet",
            "pool",
        )
        pool_gate = pooling_gate(rows, pool_rows, args.feature_set)
        # F1: an EMPTY pool ties the gate (margin 0 -> keep_pool True) but contributes nothing; gating on
        # len(pool_rows) stops a zero-contribution pool from falsely stamping providers_trained/visibility.
        if pool_gate["keep_pool"] and len(pool_rows):
            rows = pd.concat([rows, pool_rows], ignore_index=True)
            providers.add(args.pool_provider)

    names = _feature_names(args.feature_set)
    model = ReceiverModel(args.feature_set).fit(rows[names], rows["label"])
    model.shipped_variant = args.feature_set
    top1, fold_top1 = cv_top1(rows, args.feature_set)
    model.save(args.out / "model")

    corpus_label = artifact_label(providers=providers, all_public=providers.issubset({"statsbomb"}))
    manifest = {
        "schema": _SHARD_SCHEMA_VERSION,
        "feature_set": args.feature_set,
        "provider": args.provider,
        "labeling_strategy": labeling_strategy_for_provider(args.provider),
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
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    if pool_gate is not None:  # Q3 gate outcome + the pool's own coverage (load-bearing: did GS earn inclusion?)
        manifest["pooling_gate"] = {**pool_gate, "pool_provider": args.pool_provider, "pool_coverage": pool_coverage}
    # M-A resolution (owner variant): velocity's contribution on COMPLETED passes (i), and -- when a public
    # bundle is supplied -- the deployment gate on GS-FAILED (ii). Both stamped into the same provenanced
    # manifest, so the bundling decision is an artifact, not an unrecorded interactive call (L4a).
    if args.feature_set == "owner":
        manifest["velocity_ablation"] = velocity_ablation_completed(rows)  # M-A(i)
        if args.public_bundle is not None:
            manifest["deployment"] = _resolve_deployment(  # M-A(ii)
                args.public_bundle, model, args.provider, args.cache_dir, args.shard_root, args.out, prov["commit"]
            )
    (args.out / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
