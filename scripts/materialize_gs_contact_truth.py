"""Materialize the ``gs_contact_truth`` staged dataset (scorer spec section 4; DGX, where the pining token lives).

One pergame unit per Gradient Sports WC2022 match, ``gradientsports-<game_id>/`` (``pergame_name``):

* ``frames.parquet`` -- the tracking frames exactly as the pining loader returns them at this commit (the
  period-relative clock, ``z_source`` and every other column), plus ``source_provider`` (read by ``datasets
  record``); the id columns are stored as plain values, because a categorical id cannot be classified by
  ``datasets id-kinds``;
* ``actions.parquet`` -- the GS SPADL actions the converter returns (the standard pergame pair);
* ``sequences.parquet`` -- the PFF on-ball sequences (``scripts/_gs_truth.otb_sequences``), the contact truth.

The frames are not taken from ``tc3``: it was staged before the period-relative GS clock and ``z_source`` existed.
``_materialize.json`` stamps the run commit (ADR-037). Resume is ``for_each`` over cheap refs (ADR-052); the
item shards live in ``--work-dir``, outside the dataset, so ``stage`` uploads only the units. Then, each with the
owner's go: ``python -m scripts.hf_jobs stage``, ``datasets record gs_contact_truth``, ``datasets id-kinds
gs_contact_truth``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from scripts._gs_truth import otb_sequences
from scripts.hf_jobs.datasets import pergame_name

PROVIDER = "gradientsports"


def _plain_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Store id columns as their values: a categorical id with numeric categories becomes float (NaN where missing;
    the same id under ``id_compat``), any other categorical becomes object."""
    out = df.copy()
    for col in ("game_id", "player_id", "team_id"):
        if col in out and isinstance(out[col].dtype, pd.CategoricalDtype):
            numeric = pd.api.types.is_numeric_dtype(out[col].cat.categories)
            out[col] = out[col].astype(float) if numeric else out[col].astype(object)
    return out


def write_unit(out: Path, game: str, frames: pd.DataFrame, actions: pd.DataFrame | None, events_json) -> Path:
    """Write one unit atomically (a temporary directory renamed into place)."""
    final = out / pergame_name(PROVIDER, str(game))
    tmp = out / f".{final.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _plain_ids(frames.assign(source_provider=PROVIDER)).to_parquet(tmp / "frames.parquet", index=False)
    acts = actions if actions is not None else pd.DataFrame({"game_id": [], "player_id": [], "team_id": []})
    _plain_ids(acts).to_parquet(tmp / "actions.parquet", index=False)
    otb_sequences(events_json).to_parquet(tmp / "sequences.parquet", index=False)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def run(out, *, work_dir, provenance=None, cache_dir=None, source=None) -> dict:
    """Materialize every GS match the pining loader lists into ``out``. ``source=(refs, load)`` injects a loader
    for tests, where ``load(ref)`` returns ``(loaded_match, events_json)``."""
    from scripts._driver import for_each
    from scripts._provenance import git_provenance

    out, work_dir = Path(out), Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance or git_provenance()
    if source is None:
        from _loader_pining import gradientsports_events_json, pining_source, resolve_cache_dir

        cd = resolve_cache_dir(cache_dir)
        refs, load_full = pining_source([PROVIDER], token=None, cache_dir=cd)

        def load(ref):
            return load_full(ref), gradientsports_events_json(ref, cache_dir=cd)
    else:
        refs, load = source

    def _work(item):
        lm, events_json = item
        unit = write_unit(out, str(lm[1]), lm[3], lm[2], events_json)
        return pd.DataFrame({"unit": [unit.name], "n_frame_rows": [len(lm[3])]})

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        counters=lambda _item, _frame: {"n_units": 1},
        shard_root=work_dir / "shards",
        token_inputs={"mode": "gs_contact_truth", "schema": 1},
        tag="gs_contact_truth",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"gs_contact_truth failures: {res.failures}")
    units = sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith("."))
    stamp = {"run_commit": prov.get("commit"), "run_tree_dirty": prov.get("dirty"), "n_units": len(units)}
    (out / "_materialize.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the gs_contact_truth staged dataset (DGX).")
    parser.add_argument("--out", type=Path, required=True, help="the dataset directory (staged as is)")
    parser.add_argument("--work-dir", type=Path, required=True, help="resume shards, outside the dataset")
    parser.add_argument("--cache-dir", type=Path, default=None, help="the pining cache")
    parser.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (provenance stamps it)")
    args = parser.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    stamp = run(args.out, work_dir=args.work_dir, provenance=prov, cache_dir=args.cache_dir)
    print(f"gs_contact_truth: {stamp['n_units']} units at commit {stamp['run_commit']}")


if __name__ == "__main__":
    main()
