"""Build the held-out Gradient Sports ``xcross_probe`` corpus (TF-46 Task 5).

A pure staging WRAPPER: it drives :mod:`scripts.materialize_tc3_frames` (which owns the ``for_each``, the
stoppage staging flag and the provenance) to a flat cache, then :func:`scripts.hf_jobs.repackage.flat_to_pergame`
to the per-game layout the xcross trainer's ``--probe-dataset`` reads, and FAILS LOUD (O6) on a probe match whose
SPADL actions carry no restart or goal -- ``events_v3`` would then derive no dead-ball span. The wrapper writes
no committed ARTIFACT of its own (materialize + repackage do); its only write is a TEMP roster JSON for the
materialize subprocess. It is OUTSIDE the compute-contract population (``tests/scripts/test_compute_contract.py``;
no ``for_each``) and is enrolled in ``_NOT_A_DRIVER`` in ``tests/scripts/test_provenance_wiring.py`` -- the temp
roster is not a cited artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


class ProbeBuildError(RuntimeError):
    """O6: a probe match whose events carry no restart/goal, so ``events_v3`` derives no dead-ball span."""


def _match_has_stoppage_events(actions: pd.DataFrame) -> bool:
    """True when the match's SPADL actions carry at least one restart (each closes the span the previous
    action opened) or goal (each opens a span). The two together are the ``events_v3`` dead-ball spans."""
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking._stoppages import RESTART_TYPES, SHOT_TYPES

    if not len(actions):
        return False
    if "type_name" in actions.columns:
        names = actions["type_name"].astype("string")
    elif "type_id" in actions.columns:
        at = spadlconfig.actiontypes_df()
        id_to_name = dict(zip(at["type_id"], at["type_name"], strict=True))
        names = actions["type_id"].map(id_to_name).astype("string")
    else:
        return False
    if names.isin(RESTART_TYPES).any():
        return True
    is_shot = names.isin(SHOT_TYPES)
    if "result_name" in actions.columns:
        goals = is_shot & (actions["result_name"].astype("string") == "success")
        if bool(goals.any()):
            return True
    if "owngoal" in actions.columns and bool(actions["owngoal"].fillna(False).astype(bool).any()):
        return True
    return False


def assert_probe_units_have_spans(pergame_dir: str | Path) -> None:
    """O6, in the BUILD step (not a library change to ``silly_kicks._stoppages``): every per-game unit's SPADL
    actions carry a restart or goal, else ``events_v3`` derives no dead-ball span and the probe match cannot
    anchor a stoppage. Raise :class:`ProbeBuildError` naming the unit."""
    root = Path(pergame_dir)
    for unit in sorted(d for d in root.iterdir() if d.is_dir()):
        apath = unit / "actions.parquet"
        actions = pd.read_parquet(apath) if apath.exists() else pd.DataFrame()
        if not _match_has_stoppage_events(actions):
            raise ProbeBuildError(
                f"xcross_probe unit {unit.name!r} has no restart/goal action: events_v3 derives no dead-ball span "
                "(the build fails loud rather than ship a probe match with a degenerate stoppage; O6)"
            )


def build_probe_pergame(flat_dir: str | Path, out_dir: str | Path) -> int:
    """Repackage a staged FLAT probe corpus to the per-game layout, then run the O6 guard. Returns the unit count."""
    from scripts.hf_jobs.repackage import flat_to_pergame

    n = flat_to_pergame(flat_dir, out_dir)
    assert_probe_units_have_spans(out_dir)
    return n


def probe_roster(dataset: str = "xcross_probe") -> dict[str, list[str]]:
    """The held-out probe roster ``{provider: [native match_id]}`` from the dataset lock (Task 5 / r4 section4.7).

    The lock's ``units`` carry ``source_provider`` + a pergame ``game`` (``{provider}-{id}``); the native
    match id the loader allowlist keys on is that ``game`` with the provider prefix stripped. Deriving the
    roster here means the build ALWAYS stages the lock's M6 held-out pair, never an arbitrary first-N of the
    Gradient Sports catalog (which could overlap the training matches, destroying the held-out property)."""
    from scripts.hf_jobs.datasets import locked

    roster: dict[str, list[str]] = {}
    for u in locked(dataset)["units"]:
        provider = str(u["source_provider"])
        game = str(u["game"])
        native = game[len(provider) + 1 :] if game.startswith(f"{provider}-") else game
        roster.setdefault(provider, []).append(native)
    return roster


def _materialize(
    cache_dir: Path,
    flat_dir: Path,
    providers: list[str],
    max_per_provider: int | None,
    match_ids_json: Path | None = None,
) -> None:
    """Drive materialize_tc3_frames (stoppage staging flag ON by default there) to a flat cache (its own
    for_each + provenance). A subprocess, so this wrapper calls no for_each / write of its own.

    ``match_ids_json`` constrains the corpus to the held-out probe roster (Task 5); without it the
    materializer would stage the whole catalog (or an arbitrary ``--max-per-provider`` first-N)."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.materialize_tc3_frames",
        "--cache-dir",
        str(cache_dir),
        "--out",
        str(flat_dir),
        "--providers",
        *providers,
    ]
    if match_ids_json is not None:
        cmd += ["--match-ids-json", str(match_ids_json)]
    if max_per_provider is not None:
        cmd += ["--max-per-provider", str(max_per_provider)]
    subprocess.run(cmd, check=True)  # noqa: S603 -- fixed argv; materialize owns the stoppage flag + provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the held-out Gradient Sports xcross_probe corpus (TF-46).")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="the per-game probe corpus output dir")
    ap.add_argument("--providers", nargs="+", default=["gradientsports"])
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        type=Path,
        default=None,
        help="JSON {provider: [match_id, ...]} roster. Omit = the held-out probe roster from the dataset "
        "lock (Task 5); pass it to override. Without a roster the probe would NOT be held out.",
    )
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp) / "flat"
        # Default to the lock's held-out M6 roster so O-1 cannot accidentally stage the whole GS catalog
        # (r4 section4.7 / Task 5); an explicit --match-ids-json overrides it.
        roster_path = args.match_ids_json
        if roster_path is None:
            roster_path = Path(tmp) / "probe_roster.json"
            roster_path.write_text(json.dumps(probe_roster()), encoding="utf-8")
        _materialize(args.cache_dir, flat, args.providers, args.max_per_provider, roster_path)
        n = build_probe_pergame(flat, args.out)
    print(f"BUILD_XCROSS_PROBE_OK {n} per-game units -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
