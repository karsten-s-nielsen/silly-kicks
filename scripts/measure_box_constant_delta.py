"""Measure what the ghost box-constant unification actually changes, before re-fitting.

Spec D2: the re-fit happens either way (the contract raises on an unaccompanied flip), but the SHIP
CLAIM depends on this count. Zero-versus-nonzero is the whole question, so no threshold is needed --
but the count must be ATTRIBUTABLE to the band vs the depth boundary, or it is a number nobody can
reason about next cycle.

Also counts the behind-the-line population (`gr_x < 0`) while the rows are in hand: ADR-050 parks
"should a behind-the-line point count as in-box", and answering it later would otherwise cost a
second corpus pass over data we are already holding.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts._provenance import git_provenance, require_clean_tree

_LEGACY_BOX_WIDTH = 40.3
_LEGACY_DEPTH = 16.5


def frame_parquets(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every TRACKING-FRAME parquet under `data_dir`, and nothing else.

    Accepts BOTH layouts. `materialize_tc3_frames` adopts ADR-052's `for_each`, which writes
    `shard_root/<token>/<key>.parquet` -- NOT `{provider}/{id}/frames.parquet`. A
    `**/frames.parquet` glob silently finds nothing there and this driver exits having measured no
    rows, which on a remote box reads as "the corpus is empty" rather than "you pointed at the wrong
    directory".

    **Sidecar directories are EXCLUDED, and that is a regression fix, not tidiness.** The bare
    `**/*.parquet` fallback was safe only while frames were the sole parquet under `--out`. Once
    `materialize_tc3_frames` began emitting `_actions/<key>.parquet` for the trainer, the fallback
    swept those up too and this driver died mid-corpus on
    `ArrowInvalid: No match for FieldRef.Name(x)` -- SPADL actions carry `start_x`/`end_x`, never
    `x`. Measured on the real 179-match pass.

    The rule is the leading underscore: `_actions/`, `_home/` and any future sidecar are siblings of
    the shard generation under one `--out`, and none of them holds frames. Excluding by NAME rather
    than by probing each file's schema keeps this cheap and, more importantly, keeps a genuinely
    malformed FRAME shard loud instead of silently skipped.
    """
    named = sorted(data_dir.glob("**/frames.parquet"))
    if named:
        return named
    return sorted(
        p
        for p in data_dir.glob("**/*.parquet")
        if not any(part.startswith("_") for part in p.relative_to(data_dir).parts[:-1])
    )


_GOAL_Y = 34.0


def _legacy_y_in_band(y: np.ndarray) -> np.ndarray:
    """The MIN/MAX BAND form ghost actually shipped -- NOT the abs form.

    Do not "simplify" this to `np.abs(y - 34.0) <= 40.3/2`. Spec 1.1 item 3 proves the two forms
    equivalent at the CANONICAL constant and explicitly records that they DISAGREE at the LEGACY
    one, at exactly `y = 13.85`: the double sits fractionally below `(68-40.3)/2`, so the band says
    outside while the abs form says inside. Modelling legacy with the abs form makes that row a
    no-flip when it is a real flip -- an undercount at precisely the boundary this driver measures.
    """
    return (y >= (68.0 - _LEGACY_BOX_WIDTH) / 2.0) & (y <= (68.0 + _LEGACY_BOX_WIDTH) / 2.0)


def classify_flips(gr_x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Split the legacy-vs-canonical disagreement by CAUSE, three ways.

    A three-way split states a fact; a two-way split forces a convention. A row where both changes
    are individually NECESSARY (y in the 1 cm strip AND x exactly on the depth line) belongs to
    neither pure bucket, and folding it into one makes the other a systematic undercount.
    """
    legacy_y = _legacy_y_in_band(y)
    canon_y = np.abs(y - _GOAL_Y) <= spadlconfig.penalty_area_half_width
    legacy_x = gr_x < _LEGACY_DEPTH
    canon_x = gr_x <= spadlconfig.penalty_area_depth

    flipped = (legacy_x & legacy_y) != (canon_x & canon_y)
    y_agrees = legacy_y == canon_y
    x_agrees = legacy_x == canon_x
    return {
        "n_flipped": int(flipped.sum()),
        "n_flipped_band_only": int((flipped & ~y_agrees & x_agrees).sum()),
        "n_flipped_boundary_only": int((flipped & y_agrees & ~x_agrees).sum()),
        "n_flipped_both": int((flipped & ~y_agrees & ~x_agrees).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the ghost box-constant feature delta.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    paths = frame_parquets(args.data_dir)
    xs_all, ys_all = [], []
    for pq in paths:
        df = pd.read_parquet(pq, columns=["x", "y"])
        xs_all.append(df["x"].to_numpy(dtype=float))
        ys_all.append(df["y"].to_numpy(dtype=float))
    if not xs_all:
        raise SystemExit(
            f"no *.parquet under {args.data_dir}. Run materialize_tc3_frames first, and point "
            f"--data-dir at its --out (the shard generation directory beneath it is searched too)."
        )

    gr_x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)

    out = classify_flips(gr_x, y)
    out["n_rows"] = int(gr_x.size)
    out["n_behind_line"] = int((gr_x < 0).sum())
    # `git_provenance()` returns a DICT, not an object -- `prov.commit` raises AttributeError, and
    # it would do so AFTER the whole corpus pass, in the cheap write step. pyright caught this;
    # ruff and the unit tests could not, because nothing exercises `main()`.
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = prov["dirty"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
