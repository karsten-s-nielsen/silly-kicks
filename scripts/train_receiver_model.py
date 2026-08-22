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
_SHARD_SCHEMA_VERSION = "receiver-rows-1"


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


def extract_candidate_rows(actions: pd.DataFrame, frames: pd.DataFrame, *, feature_set: str = "public") -> pd.DataFrame:
    """Per completed pass, one row per teammate candidate: features + label (1 = observed receiver)."""
    from silly_kicks.spadl.utils import resolve_next_touch_receiver

    links, _ = link_actions_to_frames(actions, frames)
    # resolve_next_touch_receiver is POSITIONALLY aligned with actions -> key by action_id explicitly
    receiver = dict(zip(actions["action_id"].to_numpy(), resolve_next_touch_receiver(actions).to_numpy(), strict=True))
    frame_of = dict(zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=True))
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}
    rows = []
    for _, act in actions.iterrows():
        if act["type_id"] not in _PASS_TYPES or int(act["result_id"]) != _SUCCESS:
            continue  # train on COMPLETED passes only (observed receiver = ground truth)
        obs = receiver.get(act["action_id"])
        fr = by_frame.get(canonical_id(frame_of.get(act["action_id"])))
        if obs is None or (isinstance(obs, float) and np.isnan(obs)) or obs is pd.NA or fr is None:
            continue
        feats = receiver_candidate_features(act, fr, feature_set=feature_set)
        if feats.empty:
            continue
        for _, cf in feats.iterrows():
            rows.append(
                {
                    **{c: cf[c] for c in _feature_names(feature_set)},
                    "candidate_id": cf["candidate_id"],
                    "label": 1 if same_id(cf["candidate_id"], obs) else 0,
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


def cv_top1(rows: pd.DataFrame, feature_set: str, n_splits: int = 5) -> tuple[float, list[float]]:
    """Held-out top-1 accuracy (per pass: the argmax candidate is the observed receiver), GroupKFold on game."""
    from sklearn.model_selection import GroupKFold

    names = _feature_names(feature_set)
    groups = rows["game_id"].to_numpy()
    n_groups = len(pd.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return float("nan"), []
    fold_top1 = []
    for tr, te in GroupKFold(n_splits=k).split(rows, groups=groups):
        m = ReceiverModel(feature_set).fit(rows.iloc[tr][names], rows.iloc[tr]["label"])
        test = rows.iloc[te].copy().reset_index(drop=True)  # unique index -> grp.loc[idxmax] is a scalar (L5)
        test["_p"] = m.predict_candidates(test[names])
        hits = n = 0
        for _, grp in test.groupby(["game_id", "action_id"]):
            n += 1
            hits += int(grp.loc[grp["_p"].idxmax(), "label"] == 1)
        fold_top1.append(hits / n if n else float("nan"))
    return float(np.nanmean(fold_top1)), fold_top1


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

        return ((m[1], m[2], m[3]) for m in load_statsbomb_matches())
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
        token_inputs={"schema": "receiver-deploy-1", "columns": _DEPLOY_SHARD_COLUMNS, "commit": prov_commit},
        label="deploy",
    )
    pooled = reconcile(res.shard_dir, out_dir / "deployment_counts.parquet", tag="deploy")
    return _pooled_deployment(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the public (SB360) / owner (GS) receiver model.")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--shard-root", type=pathlib.Path, required=True)
    ap.add_argument("--feature-set", choices=["public", "owner"], default="public")
    ap.add_argument("--provider", default="statsbomb")
    ap.add_argument("--cache-dir", default=None, help="pining cache dir for a tracking provider (owner variant)")
    ap.add_argument(
        "--public-bundle",
        type=pathlib.Path,
        default=None,
        help="public bundle dir -> run the M-A(ii) deployment gate (public vs this owner variant) on GS-failed",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--min-rows", type=int, default=1)
    # A corpus-volume floor so a partial download cannot silently bundle a model. Injectable; RAISE it to
    # the corpus's expected pass volume for a production bundling run (mirrors build_rq_pass_scores).
    ap.add_argument("--min-passes", type=int, default=200)
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    res = for_each(
        _load_corpus(args.provider, args.cache_dir),  # (match_id, actions, frames)
        key=lambda t: str(t[0]),
        work=lambda t: extract_candidate_rows(t[1], t[2], feature_set=args.feature_set),
        shard_root=args.shard_root,
        # emitted columns IN the generation token (4.77.1): an in-set column addition moves the digest.
        token_inputs={
            "schema": _SHARD_SCHEMA_VERSION,
            "feature_set": args.feature_set,
            "provider": args.provider,
            "columns": _emitted_columns(args.feature_set),
        },
        label="match",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = reconcile(res.shard_dir, args.out / "candidate_rows.parquet", tag="all")

    n_passes = int(rows.groupby(["game_id", "action_id"]).ngroups) if len(rows) else 0
    both_classes = bool((rows["label"] == 1).any()) and bool((rows["label"] == 0).any())
    if len(rows) < args.min_rows or n_passes < args.min_passes or not both_classes:
        raise SystemExit(
            f"vacuous training set: {len(rows)} rows / {n_passes} passes "
            f"(need >= {args.min_rows} rows, >= {args.min_passes} passes, BOTH classes present)"
        )

    names = _feature_names(args.feature_set)
    model = ReceiverModel(args.feature_set).fit(rows[names], rows["label"])
    model.shipped_variant = args.feature_set
    top1, fold_top1 = cv_top1(rows, args.feature_set)
    model.save(args.out / "model")

    corpus_label = artifact_label(providers={args.provider}, all_public=(args.feature_set == "public"))
    manifest = {
        "schema": _SHARD_SCHEMA_VERSION,
        "feature_set": args.feature_set,
        "provider": args.provider,
        "n_matches": int(rows["game_id"].nunique()),
        "n_passes": n_passes,
        "n_candidate_rows": len(rows),
        "top1_cv": top1,
        "top1_by_fold": fold_top1,
        "candidate_count_distribution": _candidate_count_distribution(rows),  # M2
        "visible_area_caveat": "SB360 truncates the negative candidate set; see the train-vs-serve shift (M2).",
        "corpus_visibility": corpus_label,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
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
