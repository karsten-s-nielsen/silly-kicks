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
from scripts._receiver_stages import (
    _LABEL_LANE_WIDTH_M,  # noqa: F401 -- re-export: tests
    _R,  # noqa: F401 -- re-export: tests
    _SHARD_SCHEMA_VERSION,
    _T,  # noqa: F401 -- re-export: tests
    _concat_non_empty,
    _feature_names,
    _load_staged,
    _namespace_game_ids,
    _staged_refs,
    _trajectory_winner,  # noqa: F401 -- re-export: tests
    extract_candidate_rows,  # noqa: F401 -- re-export: tests
    labeling_strategy_for_provider,
)
from scripts._receiver_stages import STAGES as _REUSABLE  # ADR-129: the reusable extract/pool stages
from scripts._stages import ItemStage, Param, Pipeline, Stage, main
from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.tracking._receiver import ReceiverModel


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
        # Commit-free (S5.10): the trainer emits the reproducibility LABEL; adopt stamps the commit + note.
        **reproducibility("public" if corpus_label == "public" else "restricted", sorted(providers)),
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
    from scripts.hf_jobs.identity import commit_free

    return FileSet(commit_free(files))  # S5.10: commit-free stage output


PIPELINE = Pipeline(
    name="receiver_model",
    module="scripts.train_receiver_model",
    params=_PARAMS,
    stages=(
        _REUSABLE["extract"],  # ADR-129: declared in scripts._receiver_stages
        _REUSABLE["pool"],
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
