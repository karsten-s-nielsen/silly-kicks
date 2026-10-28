"""Owner-run pining cross-check of the pass-completion trainer (TF-54b): ``--source pining``.

DGX only: it needs the pining token in the environment. The weights it writes are NOT guaranteed public or
redistributable, so they MUST NOT be bundled or published (IMPL-05).

``scripts/train_pass_completion.py`` runs on HF (a stage-contract pipeline, ADR-128), so it never imports the
pining client: the I1 gate walks every HF job's static import closure and allows no exemption (ADR-123
Amendment B). This module owns the pining source AND the cross-check's run flow: the legacy ``for_each`` flow
(the clean-tree guard, the shards, the token, the fit, the metrics and the card), which moved here when the HF
trainer became a pipeline (HFU R3-10). The fit, the out-of-fold metrics and the card are the trainer's own pure
functions.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/_pass_completion_pining.py --out <DIR> --source pining [--providers statsbomb] \
      [--max-per-provider N] [--tracking-limit N] [--match-ids-json FILE]
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any


def pining_events_source(
    args: argparse.Namespace, match_ids: dict | None
) -> tuple[list[Any], Callable[[Any], Any], str, dict]:
    """The pining corpus as ``(refs, load, corpus_label, source_token)``, EVENTS ONLY.

    PassCompletionModel is event-only, so this parses no tracking. The events-only actions are byte-identical
    to a full load's, so the shards and the token are unchanged. A SkillCorner ref goes through the admission
    edge (fail-closed); GS needs no verdict.
    """
    from scripts._events_admission import events_only_loader
    from scripts._loader_pining import list_match_refs
    from scripts._partition import providers_for_slice

    # IMPL-05 footgun guard: pining weights are NOT guaranteed public. The bundled default must be
    # public-corpus-trained (the redistributability contract), so name the trap loudly here.
    warnings.warn(
        "Training from --source pining: these weights are NOT guaranteed public/redistributable and "
        "MUST NOT be bundled or published. The pining 'statsbomb' provider is a PRIVATE women's corpus, "
        "not the public men's WC2022 the bundled default uses -- pass --source open-data instead.",
        stacklevel=2,
    )
    refs = list_match_refs(
        providers=providers_for_slice(args.providers.split(","), match_ids),
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
    )
    load, _admission = events_only_loader(refs, cache_dir=args.cache_dir)
    token = {"source": "pining", "providers": args.providers, "tracking_limit": args.tracking_limit}
    return refs, load, f"pining:{args.providers}", token


def main(argv: list[str] | None = None) -> None:
    """The owner-tier pining cross-check CLI (the legacy trainer's ``--source pining`` path, unchanged)."""
    import json

    import pandas as pd

    from scripts._driver import for_each
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts.train_pass_completion import (
        _SHARD_COLUMNS,
        cross_val_metrics,
        pass_rows_for_match,
        render_model_card,
    )
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.expected_passing._features import FEATURE_NAMES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + weights/ + metrics.json)")
    ap.add_argument("--source", choices=("pining",), default="pining", help="the pining providers (owner tier)")
    ap.add_argument("--providers", default="statsbomb", help="comma-separated pining providers")
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap the number of matches")
    ap.add_argument("--tracking-limit", type=int, default=None, help="cap frames parsed per match")
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches this process handles.',
    )
    ap.add_argument("--cache-dir", default=None, help="raw-artifact cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact marked dirty)")
    args = ap.parse_args(argv)

    # Clean-tree guard FIRST, before any corpus work (ADR-052): an artifact whose provenance is unknown is one
    # nobody can reproduce or audit later.
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)
    refs, load, corpus_label, source_token = pining_events_source(args, match_ids)

    def _work(item):
        _provider, match_id, actions, _frames, _home, *_ = item
        return pass_rows_for_match(actions, match_id)

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        shard_root=dest / "shards",
        token_inputs={"model": "PassCompletionModel", "feature_names": list(FEATURE_NAMES), **source_token},
        label="match",
    )
    frames = [pd.read_parquet(s) for s in sorted(res.shard_dir.glob("*.parquet"))]
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)
    if not len(pooled):
        raise SystemExit("no pass rows collected from the corpus; nothing to fit")

    metrics = cross_val_metrics(pooled)
    model = PassCompletionModel().fit(pooled)
    wdir = dest / "weights"
    model.save(wdir)
    out = {
        **metrics,
        "n_pass_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": corpus_label,
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **res.manifest(),
    }
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (wdir / "MODEL_CARD.md").write_text(render_model_card(out), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"bundled weights + MODEL_CARD.md -> {wdir}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
