"""Shared consume-time corpus-visibility pre-flight for the ghost-GK trainer.

Single source for the "a detection-aware shard must not have discarded its ``visibility`` flag" rule
at consume time (``docs/PRIVATE_CONSUMERS.md``). The build-time guard
(``materialize_tc3_frames._guard_provider_frames``) is the primary defense, but a corpus can predate it
or be hand-assembled, so the trainer re-checks here before the expensive extraction. Kept out of
``train_ghost_gk.py`` so the check is importable and unit-testable without loading the trainer's argparse
surface; ``train_ghost_gk`` re-imports it under the same name (the re-import is load-bearing, spec 7.1).
"""

from __future__ import annotations

from pathlib import Path


def validate_corpus_visibility(provider_by_path: dict[Path, str]) -> None:
    """Fail BEFORE extraction on a detection-aware shard whose ``visibility`` was discarded.

    The build-time guard (``materialize_tc3_frames._guard_provider_frames``) is the primary defense,
    but a corpus can predate it or be hand-assembled, so the trainer re-checks at consume time -- the
    same class of pre-flight as :func:`validate_corpus_providers`, one stage earlier than the per-frame
    :func:`keeper_detection_mask` (which fires only after the expensive extraction).

    Reads parquet ``null_count`` METADATA (zero data pages) for EVERY detection-aware shard, so a
    MIXED corpus (one tail-kloppy shard among good ones) is caught, not just a systematic one. Raises
    the shared remedy message (rebuild via ``tracking.skillcorner``); also raises if a detection-aware
    shard dropped the ``visibility`` column entirely (M2, consume side). A shard whose statistics are
    unavailable falls back to reading the one column.
    """
    import pyarrow.parquet as pq

    from silly_kicks.tracking._provider_visibility import (
        _DETECTION_AWARE_PROVIDERS,
        _detection_discarded_message,
    )

    for path, provider in provider_by_path.items():
        if provider not in _DETECTION_AWARE_PROVIDERS:
            continue
        pf = pq.ParquetFile(path)
        if "visibility" not in pf.schema_arrow.names:
            raise ValueError(
                f"{path.name}: provider {provider!r} carries a detection flag, but the shard has NO "
                "`visibility` column -- the pipeline dropped it. Build these frames with "
                "tracking.skillcorner instead (spec 4.3)."
            )
        meta = pf.metadata
        num_rows = meta.num_rows
        if num_rows == 0:
            continue  # an empty shard is not a discarded-flag signal
        # Flat tc3 schema: arrow field order == parquet leaf-column order.
        col_idx = pf.schema_arrow.names.index("visibility")
        null_count = 0
        stats_ok = True
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(col_idx).statistics
            nc = getattr(stats, "null_count", None) if stats is not None else None
            if nc is None:
                stats_ok = False
                break
            null_count += nc
        if not stats_ok:
            # No usable metadata -> read the one column and decide honestly (rare path).
            vis = pf.read(columns=["visibility"]).column("visibility").to_pandas()
            if vis.isna().all():
                raise ValueError(f"{path.name}: {_detection_discarded_message(provider)}")
            continue
        if null_count == num_rows:
            raise ValueError(f"{path.name}: {_detection_discarded_message(provider)}")
