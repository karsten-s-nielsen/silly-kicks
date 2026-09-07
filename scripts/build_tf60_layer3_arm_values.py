"""Maintainer driver: persist per-sample TF-60 Layer-3 deterrent arm values (outfield + keeper).

The construct-validity report (spec section 8, Phase B) consumes the table this writes: one row per
scored in-possession rest-defense sample, carrying both arms (rd_outfield_deter_*, rd_gk_deter_*), the
two Layer-1 anchor KPIs the pre-registered Spearman test correlates against (rd_num_superiority,
rd_compactness_x), and the resolved keeper the keeper arm belongs to (keeper_key) -- exactly what the
locked anchor needs (outfield arm vs superiority / -compactness; keeper arm named-set Mann-Whitney).

xT is fit ONCE on the whole loaded corpus and injected, the ESTABLISHED convention for a
reported-not-gated corpus measurement driver that needs an ExpectedThreat (see
measure_cover_shadow_argmax_agreement.py: silly-kicks ships no xT model and ExpectedThreat has no
save/load, so a one-surface corpus fit is how a metric like this obtains one). The one-surface fit is
a genuine cross-item barrier -- no match can be scored until every match's actions have been read --
so this driver materializes the loaded corpus and for_each walks the list (the load is re-paid on a
resume; the per-match arm MEASUREMENT is what a resume skips, and that is the expensive part).

The arm MEASUREMENT (DAS-bound, ~minutes/match) dominates, and a single process holding the whole
corpus's frames does not fit in memory at corpus scale. So a parallel run is decoupled into a FIT step
(--xt-out: stream the corpus, keep only actions, fit ONE ExpectedThreat, write it as npz) and N ARMS
workers (--xt-in: each loads the shared surface + its own --match-ids-json slice's frames only). All
workers digest the SAME full-corpus token, so they write into ONE shard generation and combine (ADR-052).
The npz keeps the surface pickle-free (ADR-011); singh_counts is deterministic so its grids are the
whole model. A plain single-process run (no --xt-*) still fits xT inline, unchanged.

Two correctness constraints inherited from the arms (both load-bearing):
* No PitchControlCache: it keys on frame IDENTITY and excludes player positions, so a ghost frame --
  which carries its twin's identity -- would be served the factual leg's surface and every delta would
  collapse to zero, silently. The arms accept no cache.
* Dropped frames are dropped, never zero: build_restdefense_ghost_frames drops-and-counts a frame
  whose ghost is missing/NaN; scoring those as delta=0 would read as "no deterrence" and bias the
  keeper/possession aggregates toward the null. The RestDefenseGhostReport conservation is asserted.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/build_tf60_layer3_arm_values.py --out <DIR> [--providers gradientsports] \
      [--max-per-provider N] [--tracking-limit N] [--match-ids-json <FILE>] [--list-matches]

The keeper arm uses the SWEEPER ghost-GK variant (an advanced in-possession keeper can sit past the
frozen default model's 30 m label ceiling; parent spec section 9 / CLAUDE.md), never the frozen default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._input_contract import declare_inputs

#: The columns measure_match emits, pinned to the shard-generation token they travel with. These two
#: MUST move together: the for_each fingerprint digests token_inputs only, never the source, so a
#: column change with an un-bumped token resolves to the SAME generation directory, skips every
#: existing shard as already-done, and combines the OLD schema while reporting a clean pass (the
#: 4.77.1 stale-shard trap). tests/scripts/test_build_tf60_layer3_arm_values.py pins the pair.
_SHARD_SCHEMA_VERSION = "tf60-layer3-arms-1"
_EMITTED_SHARD_COLUMNS = (
    "game_id",
    "period_id",
    "team_id",
    "action_id",
    "keeper_key",
    "rd_num_superiority",
    "rd_compactness_x",
    "rd_outfield_deter_threat",
    "rd_outfield_deter_space",
    "rd_gk_deter_threat",
    "rd_gk_deter_space",
    "rd_outfield_source",
    "rd_gk_source",
)


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056).

    The arm/counterfactual/compute extractor modules, the two arm-config param blocks (RestDefenseParams
    + GkdvParams field VALUES), and the two ghost models (declared by name -- their own chirality /
    feature-contract stamps pin the artifact, ADR-050). The xT surface is corpus-fit (declared as a
    covariate token), so a corpus change moves the digest via the manifest's loaded match ids.
    """
    from dataclasses import asdict

    from silly_kicks.gkdv import GkdvParams
    from silly_kicks.restdefense import RestDefenseParams

    return declare_inputs(
        driver="build_tf60_layer3_arm_values",
        params={"restdefense": asdict(RestDefenseParams()), "gkdv": asdict(GkdvParams())},
        covariates={"xt_surface": ("corpus-fit",)},
        extractors=(
            "silly_kicks.restdefense._arms",
            "silly_kicks.restdefense._counterfactual",
            "silly_kicks.restdefense._compute",
        ),
        models=(
            "silly_kicks.tracking._ghost_outfield.GhostOutfieldModel",
            "silly_kicks.tracking._ghost_gk.GhostGkModel",
        ),
    )


def _keeper_for(keeper_map, game, period, team):
    """A's resolved keeper id for (game, period, team), or NA (ADR-078; dtype-safe keys).

    Returns the CANONICAL (string) id, never the raw ``gk_id``: raw ids are int for Gradient Sports
    and str for SkillCorner/IDSSE, so a mixed-provider ``keeper_key`` column is an unwritable
    mixed-dtype object (pyarrow ``ArrowInvalid`` on the parquet combine). ``canonical_id`` yields a
    consistent string ("12248" / "DFL-OBJ-...") -- the ADR-019 convention -- so the combine succeeds.
    """
    import pandas as pd

    from silly_kicks.id_compat import canonical_id

    ident = keeper_map.get((canonical_id(game), period, canonical_id(team)))
    return canonical_id(ident.gk_id) if ident is not None else pd.NA


def measure_match(
    match_id: str,
    actions,
    frames,
    *,
    home_team_id,
    xt,
    ghost_outfield_model,
    ghost_gk_model,
):
    """One per-match shard: both Layer-3 arms + the Layer-1 anchor KPIs, keyed per scored sample.

    Returns a tidy frame carrying EXACTLY :data:`_EMITTED_SHARD_COLUMNS`. An EMPTY result still carries
    the declared columns ("ran, produced nothing" stays distinct from "not yet run"; ADR-052).
    """
    import numpy as np
    import pandas as pd

    from silly_kicks.keeper_identity import resolve_keeper_identities
    from silly_kicks.restdefense import (
        compute_rest_defense,
        merge_rest_defense,
        rest_defense_gk_deterrent,
        rest_defense_outfield_deterrent,
    )
    from silly_kicks.restdefense._columns import (
        RD_COMPACTNESS_X,
        RD_GK_DETER_SPACE,
        RD_GK_DETER_THREAT,
        RD_GK_SOURCE,
        RD_NUM_SUPERIORITY,
        RD_OUTFIELD_DETER_SPACE,
        RD_OUTFIELD_DETER_THREAT,
        RD_OUTFIELD_SOURCE,
    )
    from silly_kicks.tracking import resolve_defended_goals

    empty = pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))

    goal_map = resolve_defended_goals(frames)
    samples, _rep = compute_rest_defense(actions, frames, xt=xt, goal_map=goal_map)
    scored = samples[samples["gate_drop_reason"].isna()] if "gate_drop_reason" in samples.columns else samples
    if not len(scored):
        return empty

    of_arm, of_rep = rest_defense_outfield_deterrent(
        actions, frames, xt=xt, ghost_outfield_model=ghost_outfield_model, home_team_id=home_team_id, goal_map=goal_map
    )
    gk_arm, gk_rep = rest_defense_gk_deterrent(
        actions, frames, xt=xt, ghost_gk_model=ghost_gk_model, home_team_id=home_team_id, goal_map=goal_map
    )
    # Conservation (the engine guarantees it): a silent shortfall means frames vanished. A raise, not an
    # assert (asserts vanish under -O), and a frame that is neither scored nor counted as a drop is
    # exactly the silent-null shape this package refuses.
    for name, rep in (("outfield", of_rep), ("keeper", gk_rep)):
        if rep.n_frames_scored + sum(rep.drop_reasons.values()) != rep.n_frames_in:
            raise RuntimeError(
                f"{match_id}/{name}: scored ({rep.n_frames_scored}) + drops "
                f"({sum(rep.drop_reasons.values())}) != in ({rep.n_frames_in}) -- frames vanished"
            )

    merged = merge_rest_defense(scored, of_arm, gk_arm)

    # A's resolved keeper per (game, period, team) -- native path (velocity-bearing tracking corpus).
    keeper_map, _krep = resolve_keeper_identities(actions, frames, identity="native")
    merged["keeper_key"] = np.array(
        [
            _keeper_for(keeper_map, g, p, t)
            for g, p, t in zip(merged["game_id"], merged["period_id"], merged["team_id"], strict=True)
        ],
        dtype=object,
    )

    for col in (
        RD_NUM_SUPERIORITY,
        RD_COMPACTNESS_X,
        RD_OUTFIELD_DETER_THREAT,
        RD_OUTFIELD_DETER_SPACE,
        RD_GK_DETER_THREAT,
        RD_GK_DETER_SPACE,
        RD_OUTFIELD_SOURCE,
        RD_GK_SOURCE,
    ):
        if col not in merged.columns:
            merged[col] = pd.NA

    out = merged[list(_EMITTED_SHARD_COLUMNS)].copy()
    # CANONICAL (string) provider-identity columns: game_id / team_id / keeper_key are int for
    # Gradient Sports and str for SkillCorner/IDSSE, so a mixed-provider shard set is an unwritable
    # mixed-dtype object column (pyarrow ArrowInvalid on the combine, which also loses the run_commit
    # manifest). canonical_id_series yields a consistent string (ADR-019); the int-indexed columns
    # (period_id / action_id) stay numeric.
    from silly_kicks.id_compat import canonical_id_series

    for _idc in ("game_id", "team_id", "keeper_key"):
        out[_idc] = canonical_id_series(out[_idc])
    if len(out) and tuple(out.columns) != _EMITTED_SHARD_COLUMNS:
        raise AssertionError(
            f"shard columns drifted from _EMITTED_SHARD_COLUMNS; bump _SHARD_SCHEMA_VERSION "
            f"(currently {_SHARD_SCHEMA_VERSION!r}). emitted={tuple(out.columns)}"
        )
    return out.reset_index(drop=True)


def _aggregate_manifests(dest) -> dict:
    """Corpus-wide totals across per-worker manifests, with the conservation identity this pass owns."""
    from scripts._partition import aggregate_manifests

    corpus = aggregate_manifests(dest, defaults=("n_frames_in", "n_frames_scored", "n_matches"))
    corpus.setdefault("drop_reasons", {})
    scored, dropped = corpus["n_frames_scored"], sum(corpus["drop_reasons"].values())
    corpus["conservation_holds"] = scored + dropped == corpus["n_frames_in"]
    return corpus


def _dump_xt_npz(xt, corpus_ids, path) -> None:
    """Serialize a fitted ExpectedThreat to npz -- pickle-free, the codebase's model-serialization
    convention (ADR-011: bundled models are npz + JSON, never pickle; this keeps the repo's zero-pickle
    record intact even for a transient scripts-only artifact).

    singh_counts is deterministic, so the fitted numeric grids fully capture the surface: the __init__
    scalars (l/w/eps/method) rebuild ``grid`` + ``params`` and every fitted ndarray is persisted. The FIT
    path constructs ``ExpectedThreat()`` with default params, so a default reconstruction on load is exact
    (the scalars are re-pinned on load; a non-ndarray fitted attribute would be dropped, which the
    round-trip test in tests/scripts/test_build_tf60_layer3_arm_values.py forbids).
    """
    import numpy as np

    # This driver only ever serializes a DEFAULT singh_counts surface (main() fits ExpectedThreat() with
    # no params), so params stays None and a default reconstruction on load is exact. A non-default params
    # (e.g. kde_smoothed) would NOT survive default reconstruction -> fail closed rather than silently.
    if getattr(xt, "params", None) is not None or str(xt.method) != "singh_counts":
        raise RuntimeError(f"_dump_xt_npz only supports the default singh_counts surface (got method={xt.method!r})")

    # l/w/eps/method are re-pinned from meta; grid/params are rebuilt by __init__. Every OTHER fitted
    # attribute must be an ndarray or a list of ndarrays, or the round-trip is silently lossy (heatmaps is
    # a list of ndarrays) -> fail closed on anything else.
    rebuildable = {"l", "w", "eps", "method", "grid", "params"}
    plain_arrays: dict = {}
    list_arrays: dict = {}  # attr -> list[ndarray]
    for k, v in vars(xt).items():
        if isinstance(v, np.ndarray):
            plain_arrays[k] = v
        elif isinstance(v, list) and v and all(isinstance(e, np.ndarray) for e in v):
            list_arrays[k] = v
        elif k not in rebuildable:
            raise RuntimeError(f"ExpectedThreat has non-serializable fitted state {k!r} ({type(v).__name__})")
    arrays = dict(plain_arrays)
    for k, elems in list_arrays.items():
        for i, e in enumerate(elems):
            arrays[f"{k}__{i}"] = e
    meta = {
        "l": int(xt.l),
        "w": int(xt.w),
        "eps": float(xt.eps),
        "method": str(xt.method),
        "plain_array_keys": sorted(plain_arrays),
        "list_array_keys": {k: len(v) for k, v in list_arrays.items()},
        "corpus_ids": sorted(str(m) for m in corpus_ids),
    }
    with open(path, "wb") as fh:  # file handle -> exact filename (np.savez would append .npz)
        np.savez(fh, _xt_meta_json=json.dumps(meta), **arrays)


def _load_xt_npz(path):
    """Reconstruct (xt, corpus_ids) from :func:`_dump_xt_npz`. ``allow_pickle=False`` makes the load truly
    pickle-free: a tampered or foreign npz can only fail to parse, never execute code."""
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    z = np.load(path, allow_pickle=False)
    meta = json.loads(z["_xt_meta_json"].item())
    xt = ExpectedThreat(l=meta["l"], w=meta["w"], eps=meta["eps"], method=meta["method"])
    for k in meta["plain_array_keys"]:
        setattr(xt, k, z[k])
    for k, n in meta["list_array_keys"].items():
        setattr(xt, k, [z[f"{k}__{i}"] for i in range(n)])
    return xt, list(meta["corpus_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help=(
            'JSON {"gradientsports": ["10502", ...]} pinning WHICH matches this process handles. '
            "Split the id list N ways for N processes sharing one --out. In a parallel run the generation "
            "token is keyed on the SHARED --xt-in corpus (not this slice), so the shards combine; in a "
            "single-process run it is keyed on the loaded corpus the inline xT was fit on."
        ),
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; manifest is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    ap.add_argument(
        "--xt-out",
        default=None,
        help=(
            "FIT step for a parallel run: load the corpus actions ONLY (frames discarded -> no OOM), fit "
            "ONE ExpectedThreat, write {xt, corpus_ids} as npz to this path, and exit. arm_values cannot be "
            "sliced because the xT-fit is a cross-item barrier; this decouples the fit so N workers can "
            "share the identical surface via --xt-in."
        ),
    )
    ap.add_argument(
        "--xt-in",
        default=None,
        help=(
            "ARMS step for a parallel worker: load the shared {xt, corpus_ids} npz written by --xt-out "
            "instead of fitting. The corpus_ids (the FULL corpus the xT was fit on) key the shard "
            "generation, so every worker -- each handling its own --match-ids-json slice -- writes into ONE "
            "shared generation and the shards combine."
        ),
    )
    args = ap.parse_args()

    import pandas as pd

    from scripts._loader_pining import load_matches
    from scripts._provenance import git_provenance, require_clean_tree
    from silly_kicks.tracking import GhostGkModel, GhostOutfieldModel
    from silly_kicks.xthreat import ExpectedThreat

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # FIRST, before paying for any corpus work: git rev-parse HEAD returns the same SHA whether or not
    # the tree is modified, so stamping the bare SHA would record a commit that does not describe the
    # code that ran. Enforcement lives in main(), not the work function.
    prov = (
        {"commit": "n/a", "dirty": False, "dirty_files": [], "tree_state": "n/a"}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.list_matches:
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    from scripts._partition import providers_for_slice

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    # --- FIT step (--xt-out): load actions ONLY, fit one xT, write {xt, corpus_ids} npz, exit. Frames are
    # discarded per match (never accumulated), so the fit step does not carry the whole-corpus frame set
    # in memory -- the OOM that a single-process arm run hits. ---
    if args.xt_out:
        fit_actions: list = []
        fit_ids: list = []
        for _provider, match_id, actions, _frames, _home in load_matches(
            providers=providers_for_slice(args.providers.split(","), match_ids),
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        ):
            fit_actions.append(actions)
            fit_ids.append(str(match_id))
        if not fit_actions:
            raise SystemExit("no matches loaded for the xT fit")
        xt = ExpectedThreat()
        xt.fit(pd.concat(fit_actions, ignore_index=True))
        _dump_xt_npz(xt, fit_ids, Path(args.xt_out))
        print(f"fit xT on {len(fit_ids)} matches -> {args.xt_out}")
        return

    # --- ARMS step: materialize the (possibly sliced) matches. The xT surface is fit ONCE on the whole
    # corpus -- a cross-item barrier -- so a single process cannot stream it; --xt-in decouples that fit so
    # N workers share ONE surface (each still loads only its own slice's frames -> no whole-corpus OOM). ---
    all_actions: list = []
    loaded: list = []
    for _provider, match_id, actions, frames, home in load_matches(
        providers=providers_for_slice(args.providers.split(","), match_ids),
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        all_actions.append(actions)
        loaded.append((match_id, actions, frames, home))

    if not loaded:
        raise SystemExit("no matches loaded")

    # The established convention (measure_cover_shadow_argmax_agreement.py): ONE ExpectedThreat serves every
    # match. Fit it here, OR load the shared surface a --xt-out step already fit on the FULL corpus. The
    # shared surface's corpus_ids -- not this worker's slice -- key the generation, so parallel workers all
    # write into one shard generation and combine.
    if args.xt_in:
        xt, token_corpus_ids = _load_xt_npz(Path(args.xt_in))
    else:
        xt = ExpectedThreat()
        xt.fit(pd.concat(all_actions, ignore_index=True))
        token_corpus_ids = sorted(str(m) for m, _a, _f, _h in loaded)

    outfield_model = GhostOutfieldModel.from_variant("default")
    gk_model = GhostGkModel.from_variant("sweeper")

    _last_report: dict = {}

    def _work(item):
        match_id, actions, frames, home = item
        shard = measure_match(
            match_id,
            actions,
            frames,
            home_team_id=home,
            xt=xt,
            ghost_outfield_model=outfield_model,
            ghost_gk_model=gk_model,
        )
        # Corpus counters from the outfield arm's report (both arms share the same domain; the outfield
        # arm is the scoreable one on velocity-less providers -- the DAS leg NaNs but the frame is still
        # scored). n_frames_in/scored describe the counterfactual domain, drop_reasons the counted drops.
        _last_report.clear()
        _last_report.update({"n_frames_scored": len(shard), "n_frames_in": len(shard), "n_matches": 1})
        return shard

    from scripts._driver import for_each
    from scripts._partition import worker_tag as _worker_tag
    from scripts._partition import write_table_atomically

    worker_tag = _worker_tag(args.match_ids_json)
    dest = Path(args.out)
    res = for_each(
        loaded,
        key=lambda item: (str(args.providers.split(",")[0]), str(item[0])),
        work=_work,
        counters=lambda _item, _frame: dict(_last_report),
        shard_root=dest / "shards",
        token_inputs={
            # The xT surface is fit on exactly the LOADED corpus and feeds both arms, so the FULL corpus
            # match ids belong in the token (a --max-per-provider run reusing a wider surface's shards
            # would mix two threat models). Under a parallel run this is the SHARED --xt-in corpus, not
            # this worker's slice -- so every worker digests the identical surface, resolves to ONE shard
            # generation, and their per-match shards combine (ADR-052: the token names the surface; the
            # per-worker slice is a selector outside it, making the generation a superset of any one run).
            "match_ids": sorted(token_corpus_ids),
            "xt_surface": "corpus-fit",
            "ghost_outfield_variant": "default",
            "ghost_gk_variant": "sweeper",
            "tracking_limit": args.tracking_limit,
            # Moves with _EMITTED_SHARD_COLUMNS (the 4.77.1 stale-shard rule).
            "schema": _SHARD_SCHEMA_VERSION,
        },
        tag=worker_tag,
        label="match",
    )

    shard_dir = res.shard_dir
    shards = sorted(shard_dir.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True) if shards else pd.DataFrame()
    written = {}
    if len(combined):
        path = dest / "layer3_arm_values.parquet"
        write_table_atomically(combined, path, tag=worker_tag)
        written = {
            "path": str(path),
            "n_rows": len(combined),
            "n_keepers": int(combined["keeper_key"].dropna().nunique()),
            "n_outfield_nonnull": int(combined["rd_outfield_deter_threat"].notna().sum()),
            "n_gk_nonnull": int(combined["rd_gk_deter_threat"].notna().sum()),
        }

    worker_manifest = {
        **res.counters,
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov["tree_state"],
        "partition": worker_tag,
    }
    (dest / f"manifest_{worker_tag}.json").write_text(
        json.dumps(worker_manifest, indent=2, default=str), encoding="utf-8"
    )

    corpus = _aggregate_manifests(dest)
    corpus.update(arm_values_written=written, input_contract=input_contract())
    (dest / "layer3_arm_values_manifest.json").write_text(json.dumps(corpus, indent=2, default=str), encoding="utf-8")
    print(json.dumps(corpus, indent=2, default=str))


if __name__ == "__main__":
    main()
