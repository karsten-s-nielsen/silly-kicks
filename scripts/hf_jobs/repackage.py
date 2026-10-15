"""Repackage the staged flat TC3 corpus into the per-game layout the xshot/xcross ``--data-dir`` path
needs, IN-JOB (owner_pergame source mode).

The flat corpus is ``shards/<gen>/<provider>__<game>.parquet`` (frames) + ``_actions/<provider>__<game>.parquet``
(SPADL actions). The trainers' ``_iter_matches_from_dir`` expects, per match, a subdirectory holding
``frames.parquet`` + ``shots.parquet`` (shots = SPADL shot actions, from which the label horizon is built).

Verified on the DGX (2026-10-13): pointing the trainers at the flat layout yields "No usable training data"
(``iterdir()`` finds no game subdirs); the per-game layout extracts rows and runs the HPO. The per-game dir
name must NOT contain ``__`` (the ``scripts/_driver`` key separator rejects it), so it is ``{provider}-{game}``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

#: SPADL shot action type names whose rows become ``shots.parquet`` (prepare_xshot/xcross's default filter).
_SHOT_TYPE_NAMES = ("shot", "shot_penalty", "shot_freekick")


def _shot_type_ids() -> set[int]:
    from silly_kicks.spadl import config as spadlconfig

    at = spadlconfig.actiontypes_df()
    return set(at.loc[at["type_name"].isin(_SHOT_TYPE_NAMES), "type_id"])


def _id_to_name() -> dict:
    from silly_kicks.spadl import config as spadlconfig

    at = spadlconfig.actiontypes_df()
    return dict(zip(at["type_id"], at["type_name"], strict=True))


def _flat_shards(flat_dir: Path) -> list[Path]:
    gen = sorted(flat_dir.glob("shards/*"))
    shards = [p for g in gen for p in sorted(g.glob("*.parquet"))] if gen else []
    if not shards:  # flat *.parquet directly under the corpus root
        shards = sorted(flat_dir.glob("*.parquet"))
    return shards


def _shots_frame(actions: pd.DataFrame, shot_ids: set[int], id_to_name: dict) -> pd.DataFrame:
    if "type_name" in actions.columns:
        shots = actions[actions["type_name"].isin(_SHOT_TYPE_NAMES)].copy()
    else:
        shots = actions[actions["type_id"].isin(shot_ids)].copy()
        shots["type_name"] = shots["type_id"].map(id_to_name)
    return shots


def flat_to_pergame(flat_dir: str | Path, out_dir: str | Path) -> int:
    """Materialise ``{out_dir}/{provider}-{game}/{frames.parquet,shots.parquet}`` from the flat corpus.

    ``frames.parquet`` is symlinked (no copy -- the shards are large); ``shots.parquet`` is the filtered
    SPADL shot actions. Returns the number of per-game dirs written.
    """
    flat_dir, out_dir = Path(flat_dir), Path(out_dir)
    actions_dir = flat_dir / "_actions"
    shot_ids, id_to_name = _shot_type_ids(), _id_to_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard in _flat_shards(flat_dir):
        stem = shard.stem  # "<provider>__<game>"
        provider, _, game = stem.partition("__")
        name = f"{provider}-{game}" if game else provider  # bare dir name: no "__" (driver key separator)
        gdir = out_dir / name
        gdir.mkdir(parents=True, exist_ok=True)
        link = gdir / "frames.parquet"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.symlink(shard.resolve(), link)  # cheap: the shards are large, avoid a copy (Linux HF job)
        except (OSError, NotImplementedError):  # unprivileged symlink (e.g. Windows CI) -> copy
            shutil.copy(shard, link)
        apath = actions_dir / shard.name
        actions = pd.read_parquet(apath) if apath.exists() else pd.DataFrame(columns=["type_id"])
        _shots_frame(actions, shot_ids, id_to_name).to_parquet(gdir / "shots.parquet")
        # xcross's _iter_matches_from_dir reads {game}/actions.parquet (the FULL SPADL actions, to extract
        # crosses); emit it (symlink like frames.parquet -- the _actions shards are large) when present.
        if apath.exists():
            alink = gdir / "actions.parquet"
            if alink.exists() or alink.is_symlink():
                alink.unlink()
            try:
                os.symlink(apath.resolve(), alink)
            except (OSError, NotImplementedError):  # unprivileged symlink (e.g. Windows CI) -> copy
                shutil.copy(apath, alink)
        written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: python -m scripts.hf_jobs.repackage <flat_corpus_dir> <per_game_out_dir>")
    n = flat_to_pergame(args[0], args[1])
    print(f"HF_JOBS_REPACKAGE_OK {n} per-game dirs -> {args[1]}")


if __name__ == "__main__":
    main()
