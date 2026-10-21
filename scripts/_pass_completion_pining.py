"""Owner-run pining cross-check of the pass-completion trainer (TF-54b): ``--source pining``.

DGX only: it needs the pining token in the environment. The weights it writes are NOT guaranteed public or
redistributable, so they MUST NOT be bundled or published (IMPL-05).

``scripts/train_pass_completion.py`` runs on HF (the one-unit adapter stage, ADR-128), so it never imports the
pining client: the I1 gate walks every HF job's static import closure and allows no exemption (ADR-123
Amendment B). This module owns the pining source and injects it into that trainer's ``main``. The CLI, the
clean-tree guard, the shards, the token, the fit and the card stay the trainer's own.

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
    from scripts.train_pass_completion import main as train_main

    train_main(argv, pining_source=pining_events_source)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
