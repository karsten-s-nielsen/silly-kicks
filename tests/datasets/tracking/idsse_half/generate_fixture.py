"""Reproduce the committed ``idsse_half`` TF-58 fixture from the Sportec Open DFL Dataset (R2).

A real, public, long-enough tracking fixture so §9.5 liveness can exercise ``coord_median_freq_cpm`` and the
four ``coord_coh_*`` columns on real data (the committed synthetic/provider fixtures are all < 260 s -- too
short for the spectral and coherence minima). Source match: **DFL-MAT-J03WMX** (the same match and licence as
``tests/datasets/elastic_sync/j03wmx_slice``), loaded through the repo's own ``pining`` loader.

Reduction: period 1, ``time_seconds < 1200``; keep every other frame (25 Hz -> 12.5 Hz); the
``TRACKING_FRAMES_COLUMNS`` schema; frames sorted by ``(frame_id, player_id)``. Parquet is written with pinned
settings (pyarrow, zstd, ``index=False``) so the ADR-056 e2e reproduction test is byte-for-byte -- under the writer
the README states (a parquet footer records the pandas and pyarrow versions; regenerating under another writer means
restating it there).

Run (needs ``PINING_FOR_THE_DATA_TOKEN`` in the environment)::

    python -m tests.datasets.tracking.idsse_half.generate_fixture
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from silly_kicks.tracking import TRACKING_FRAMES_COLUMNS

HERE = Path(__file__).resolve().parent
MATCH_ID = "DFL-MAT-J03WMX"
PROVIDER = "idsse"
PERIOD = 1
CUTOFF_S = 1200.0
DECIMATE = 2
SIZE_LIMIT_MB = 6.0


def _load_raw(token: str | None = None, cache_dir: str | None = None):
    """Load the full match through the repo's pining loader (events + tracking)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from _loader_pining import list_match_refs, load_match

    refs = {r.match_id: r for r in list_match_refs(providers=[PROVIDER], token=token)}
    if MATCH_ID not in refs:
        raise RuntimeError(f"{MATCH_ID} not offered by provider {PROVIDER!r}; got {sorted(refs)}")
    loaded = load_match(refs[MATCH_ID], events_only=False, cache_dir=cache_dir, token=token)
    if loaded.frames is None or loaded.frames.empty:
        raise RuntimeError("loader returned no tracking frames")
    return loaded.actions, loaded.frames


def build_fixture(token: str | None = None, cache_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the reduced ``(frames, actions)`` deterministically (no RNG, stable sort)."""
    actions, frames = _load_raw(token=token, cache_dir=cache_dir)

    p1 = frames[(frames["period_id"] == PERIOD) & (frames["time_seconds"] < CUTOFF_S)].copy()
    keep = pd.unique(p1["frame_id"].sort_values().to_numpy())[::DECIMATE]
    native_hz = float(p1["frame_rate"].iloc[0])
    p1 = p1[p1["frame_id"].isin(keep)].copy()
    p1["frame_rate"] = native_hz / DECIMATE
    frames_out = (
        p1[list(TRACKING_FRAMES_COLUMNS)]
        .sort_values(["frame_id", "player_id"], kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )

    lo, hi = float(frames_out["time_seconds"].min()), float(frames_out["time_seconds"].max())
    a1 = actions[(actions["period_id"] == PERIOD) & actions["time_seconds"].between(lo, hi)].copy()
    actions_out = a1.sort_values("action_id", kind="mergesort").reset_index(drop=True)
    return frames_out, actions_out


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def main() -> None:
    frames, actions = build_fixture()
    fpath, apath = HERE / "frames.parquet", HERE / "actions.parquet"
    _write(frames, fpath)
    _write(actions, apath)
    fmb = fpath.stat().st_size / 1e6
    amb = apath.stat().st_size / 1e6
    print(f"frames.parquet: {len(frames):,} rows, {fmb:.3f} MB")
    print(f"actions.parquet: {len(actions):,} rows, {amb:.3f} MB")
    import pyarrow

    print(f"written with pandas {pd.__version__} and pyarrow {pyarrow.__version__} (state it in README.md)")
    print(
        f"duration: {frames['time_seconds'].max() - frames['time_seconds'].min():.1f} s at "
        f"{frames['frame_rate'].iloc[0]:.1f} Hz"
    )
    if fmb > SIZE_LIMIT_MB:
        print(f"STOP: frames.parquet {fmb:.3f} MB exceeds the {SIZE_LIMIT_MB} MB target -- report to the owner.")


if __name__ == "__main__":
    main()
