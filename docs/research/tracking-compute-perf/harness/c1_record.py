"""C1 record (spec 2026-11-08 tracking-batch, section 7, owner ruling C1): the lakehouse's own calls, before and after.

One input, two trees, one machine (the DGX), one venv. Run the same file in each tree with ``PYTHONPATH`` set to that
tree, then compare the two output folders:

    python c1_record.py build-input --provider gradientsports --match 10517 --cache-dir <pining cache> --out <in>
    PYTHONPATH=<main tree>   python c1_record.py run  --input <in> --out <out>/main   --label main
    PYTHONPATH=<frozen tree> python c1_record.py run  --input <in> --out <out>/frozen --label frozen
    PYTHONPATH=<frozen tree> python c1_record.py peak --input <in> --out <out>/frozen --backend numba
    PYTHONPATH=<frozen tree> python c1_record.py peak --input <in> --out <out>/frozen --backend numpy
    python c1_record.py compare --main <out>/main --frozen <out>/frozen --out <out>/compare.json

The calls are the lakehouse's (``luxury-lakehouse`` ``v2/src/lakehouse/compute/pitch_control.py`` and
``shape_graphs.py``): ``smooth_frames`` + ``derive_velocities`` with ``PreprocessConfig.default()``; per
(period, frame) ``compute_pitch_control(grp, attacking_team_id=home)``; per (period, frame, team) on the outfield rows
``compute_shape_graph(positions)`` + ``infer_positions(graph, positions, 1.0)``. The batch calls are
``compute_pitch_control_batch`` with one request per frame and ``compute_shape_graphs``.

Bits: each pitch-control surface is reduced to its SHA-256 (in frame order); each shape graph to the SHA-256 of its
edges, stabilities and sorted faces; labels are kept as strings. Nothing is rounded. Timing is wall clock
(``time.perf_counter``) after one warm-up call, so a numba compile is not counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

_KEYS = ["game_id", "period_id", "frame_id"]
_MIRROR = {"L": "R", "LC": "RC", "C": "C", "RC": "LC", "R": "L"}


# --------------------------------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------------------------------


def _read(path: pathlib.Path) -> pd.DataFrame:
    # use_threads=False: a threaded Arrow read can abort the process at exit on the DGX (Part Deux, 2026-11-23)
    return pd.read_parquet(path, use_threads=False)


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _env(label: str) -> dict[str, Any]:
    import numba
    import scipy

    import silly_kicks

    return {
        "label": label,
        "silly_kicks_file": silly_kicks.__file__,
        "silly_kicks_version": getattr(silly_kicks, "__version__", None),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "machine": platform.machine(),
        "node": platform.node(),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_CACHE_DIR", "PYTHONPATH")},
    }


# --------------------------------------------------------------------------------------------------------------------
# build-input: one match, written once, read by both trees
# --------------------------------------------------------------------------------------------------------------------


def build_input(args: argparse.Namespace) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        from tests._prod_fixtures import make_production_frames, make_single_game_inputs

        frames = make_production_frames(n_frames=args.synthetic, n_periods=2)
        actions, _ = make_single_game_inputs()
        home = int(pd.unique(frames["team_id"].dropna())[0])
    else:
        from scripts import _loader_pining as lp

        refs = lp.list_match_refs(providers=[args.provider], match_ids={args.provider: [args.match]})
        if len(refs) != 1:
            raise SystemExit(f"c1_record: {args.provider} match {args.match} not found exactly once in pining")
        loaded = lp.load_match(refs[0], events_only=False, cache_dir=args.cache_dir, with_velocity=False)
        frames, actions, home = loaded.frames, loaded.actions, int(loaded.home_team_id)
    frames.to_parquet(out / "frames.parquet", index=False)
    actions.to_parquet(out / "actions.parquet", index=False)
    teams = sorted({int(t) for t in frames.loc[~frames["is_ball"].astype(bool), "team_id"].dropna().unique()})
    away = next((t for t in teams if t != home), home)
    _write_json(
        out / "meta.json",
        {
            "provider": args.provider,
            "match": args.match,
            "synthetic": args.synthetic,
            "home_team_id": home,
            "away_team_id": away,
            "n_rows": len(frames),
            "n_frames": int(frames[_KEYS].drop_duplicates().shape[0]),
            "n_actions": len(actions),
            "frames_sha256": _sha256_file(out / "frames.parquet"),
            "actions_sha256": _sha256_file(out / "actions.parquet"),
            "built_by": _env("build-input"),
        },
    )


# --------------------------------------------------------------------------------------------------------------------
# run: every lakehouse call in one tree
# --------------------------------------------------------------------------------------------------------------------


def _digest(*arrays: np.ndarray) -> bytes:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.digest()


def _graph_digest(graph: Any) -> bytes:
    faces = sorted(tuple(sorted(int(i) for i in f)) for f in graph.faces)  # merged faces differ in size
    flat = np.asarray([v for face in faces for v in (len(face), *face)], dtype=np.int64)  # length, then vertices
    return _digest(np.asarray(graph.edges), np.asarray(graph.stabilities), flat)


def _preprocess(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames

    cfg = PreprocessConfig.default()
    t0 = time.perf_counter()
    smoothed = smooth_frames(frames, config=cfg)
    t1 = time.perf_counter()
    velocity = derive_velocities(smoothed, config=cfg)
    t2 = time.perf_counter()
    numeric = [c for c in velocity.columns if velocity[c].dtype.kind == "f"]
    digests = {c: _digest(velocity[c].to_numpy()).hex() for c in sorted(numeric)}
    return velocity, {"smooth_frames_s": t1 - t0, "derive_velocities_s": t2 - t1, "column_sha256": digests}


def _pc_loop(velocity: pd.DataFrame, home: int) -> tuple[list[bytes], np.ndarray, float]:
    """The lakehouse's ``pitch_control_values`` loop, verbatim in its calls (no bronze table)."""
    from silly_kicks.tracking.pitch_control import compute_pitch_control

    groups = list(velocity.groupby(["period_id", "frame_id"], sort=True))
    compute_pitch_control(groups[0][1], attacking_team_id=home)  # warm-up (numba compile)
    digests: list[bytes] = []
    fractions = np.empty(len(groups))
    t0 = time.perf_counter()
    for i, (_key, grp) in enumerate(groups):
        surface = compute_pitch_control(grp, attacking_team_id=home)
        fractions[i] = float(np.mean(surface.surface))
        digests.append(_digest(surface.surface))
    return digests, fractions, time.perf_counter() - t0


def _pc_requests(velocity: pd.DataFrame, home: int) -> list:
    keys = velocity[_KEYS].drop_duplicates().sort_values(["period_id", "frame_id"], kind="stable")
    return [(tuple(k), home, False) for k in keys.itertuples(index=False, name=None)]


def _pc_batch(velocity: pd.DataFrame, home: int) -> tuple[list[bytes], np.ndarray, float]:
    from silly_kicks.tracking.pitch_control import compute_pitch_control_batch

    requests = _pc_requests(velocity, home)
    compute_pitch_control_batch(velocity, requests[:8], method="spearman")  # warm-up (numba compile)
    t0 = time.perf_counter()
    surfaces = compute_pitch_control_batch(velocity, requests, method="spearman")
    elapsed = time.perf_counter() - t0
    digests = [_digest(s.surface) for s in surfaces]
    fractions = np.array([float(np.mean(s.surface)) for s in surfaces])
    return digests, fractions, elapsed


def _sg_loop(frames: pd.DataFrame, direction_of=None) -> tuple[pd.DataFrame, list[bytes], float, int]:
    """The lakehouse's ``shape_graph_values`` loop, verbatim in its calls. ``direction_of`` (a mapping from
    (period, frame, team) to +1.0 / -1.0) replaces the lakehouse's constant 1.0; a team-frame missing from it is
    skipped."""
    from silly_kicks.tracking._shape_graph import compute_shape_graph, infer_positions

    outfield = frames[
        (~frames["is_ball"].astype("boolean").fillna(False))
        & (~frames["is_goalkeeper"].astype("boolean").fillna(False))
        & frames["x"].notna()
        & frames["y"].notna()
    ]
    grouped = list(outfield.groupby(["period_id", "frame_id", "team_id"], sort=True, observed=True))
    warm = next(g for _k, g in grouped if len(g) >= 3)
    compute_shape_graph(warm[["x", "y"]].to_numpy(dtype="float64"))  # warm-up
    rows: list[tuple] = []
    graphs: list[bytes] = []
    n_team_frames = 0
    t0 = time.perf_counter()
    for (period_id, frame_id, team), grp in grouped:
        if len(grp) < 3:
            continue
        direction = 1.0
        if direction_of is not None:
            direction = direction_of.get((int(period_id), int(frame_id), int(team)))
            if direction is None:
                continue
        positions = grp[["x", "y"]].to_numpy(dtype="float64")
        player_ids = [int(p) for p in grp["player_id"].tolist()]
        graph = compute_shape_graph(positions)
        inferred = list(infer_positions(graph, positions, direction))
        n_team_frames += 1
        graphs.append(_graph_digest(graph))
        if len(inferred) != len(player_ids):
            continue
        for player_id, label in zip(player_ids, inferred, strict=True):
            rows.append((int(period_id), int(frame_id), int(team), player_id, str(label.label)))
    elapsed = time.perf_counter() - t0
    table = pd.DataFrame(rows, columns=["period_id", "frame_id", "team_id", "player_id", "label"])
    return table, graphs, elapsed, n_team_frames


def _frame_directions(frames: pd.DataFrame) -> dict[tuple, float] | None:
    if "team_attacking_direction" not in frames.columns:
        return None
    sel = frames.loc[~frames["is_ball"].astype(bool), ["period_id", "frame_id", "team_id", "team_attacking_direction"]]
    sel = sel.dropna().drop_duplicates()
    value = {"ltr": 1.0, "rtl": -1.0}
    return {
        (int(p), int(f), int(t)): value[d]
        for p, f, t, d in sel.itertuples(index=False, name=None)
        if d in value
    }


def run(args: argparse.Namespace) -> None:
    inp, out = pathlib.Path(args.input), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "meta.json").read_text(encoding="utf-8"))
    frames, actions = _read(inp / "frames.parquet"), _read(inp / "actions.parquet")
    home = int(meta["home_team_id"])
    record: dict[str, Any] = {"env": _env(args.label), "input": meta}

    velocity, record["preprocess"] = _preprocess(frames)

    digests, fractions, elapsed = _pc_loop(velocity, home)
    np.save(out / "pc_loop_digests.npy", np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(-1, 32))
    np.save(out / "pc_loop_fractions.npy", fractions)
    record["pc_loop"] = {"n_frames": len(digests), "seconds": elapsed, "ms_per_frame": 1e3 * elapsed / len(digests)}

    digests_b, fractions_b, elapsed_b = _pc_batch(velocity, home)
    np.save(out / "pc_batch_digests.npy", np.frombuffer(b"".join(digests_b), dtype=np.uint8).reshape(-1, 32))
    np.save(out / "pc_batch_fractions.npy", fractions_b)
    record["pc_batch"] = {
        "n_frames": len(digests_b),
        "seconds": elapsed_b,
        "ms_per_frame": 1e3 * elapsed_b / len(digests_b),
        "equal_to_loop_in_this_tree": int(sum(a == b for a, b in zip(digests, digests_b, strict=True))),
    }
    del velocity

    labels, graphs, elapsed_sg, n_tf = _sg_loop(frames)
    labels.to_parquet(out / "sg_loop_labels.parquet", index=False)
    np.save(out / "sg_loop_graph_digests.npy", np.frombuffer(b"".join(graphs), dtype=np.uint8).reshape(-1, 32))
    n_frames = int(frames[_KEYS].drop_duplicates().shape[0])
    record["sg_loop"] = {
        "n_team_frames": n_tf,
        "seconds": elapsed_sg,
        "ms_per_frame": 1e3 * elapsed_sg / n_frames,
        "ms_per_team_frame": 1e3 * elapsed_sg / n_tf,
    }

    try:
        from silly_kicks.tracking import compute_shape_graphs
    except ImportError:
        compute_shape_graphs = None
    if compute_shape_graphs is not None:
        directions = _frame_directions(frames)
        if directions is not None:
            labels_d, _g, elapsed_d, n_tf_d = _sg_loop(frames, direction_of=directions)
            labels_d.to_parquet(out / "sg_loop_framedir_labels.parquet", index=False)
            record["sg_loop_framedir"] = {"n_team_frames": n_tf_d, "seconds": elapsed_d}
        compute_shape_graphs(frames.head(200))  # warm-up
        t0 = time.perf_counter()
        batch = compute_shape_graphs(frames)
        elapsed_batch = time.perf_counter() - t0
        players = batch.players
        ok = players[players["shape_graph_source"] == "ok"]
        ok = ok.assign(label=ok["position_label"].astype(str))[["period_id", "frame_id", "team_id", "player_id", "label"]]
        ok.astype({"period_id": "int64", "frame_id": "int64", "team_id": "int64", "player_id": "int64"}).to_parquet(
            out / "sg_batch_labels.parquet", index=False
        )
        record["sg_batch"] = {"seconds": elapsed_batch, "ms_per_frame": 1e3 * elapsed_batch / n_frames}

    record["fingerprint"] = _fingerprint(actions, frames)
    _write_json(out / "record.json", record)


def _fingerprint(actions: pd.DataFrame, frames: pd.DataFrame) -> dict[str, Any]:
    from silly_kicks.tracking.utils import link_actions_to_frames

    link_actions_to_frames(actions.head(5), frames)  # warm-up
    t0 = time.perf_counter()
    link_actions_to_frames(actions, frames)
    link_s = time.perf_counter() - t0
    try:
        from silly_kicks.tracking._link_identity import action_fingerprints
    except ImportError:
        return {"link_actions_to_frames_s": link_s, "fingerprint_s": None}
    action_fingerprints(actions.head(5))
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        action_fingerprints(actions)
    return {"link_actions_to_frames_s": link_s, "fingerprint_s": (time.perf_counter() - t0) / reps, "n_actions": len(actions)}


# --------------------------------------------------------------------------------------------------------------------
# peak: tracemalloc peak of the batch call on the lakehouse workload (one request per frame)
# --------------------------------------------------------------------------------------------------------------------


def peak(args: argparse.Namespace) -> None:
    if args.backend == "numpy":
        os.environ["SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY"] = "1"
    inp, out = pathlib.Path(args.input), pathlib.Path(args.out)
    meta = json.loads((inp / "meta.json").read_text(encoding="utf-8"))
    frames = _read(inp / "frames.parquet")
    velocity, _ = _preprocess(frames)
    del frames
    home = int(meta["home_team_id"])
    from silly_kicks.backends import active_backend
    from silly_kicks.tracking.pitch_control import compute_pitch_control_batch

    requests = _pc_requests(velocity, home)
    compute_pitch_control_batch(velocity, requests[:8], method="spearman")  # warm-up outside the trace
    tracemalloc.start()
    t0 = time.perf_counter()
    surfaces = compute_pitch_control_batch(velocity, requests, method="spearman")
    elapsed = time.perf_counter() - t0
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _write_json(
        out / f"peak_{args.backend}.json",
        {
            "env": _env(f"peak-{args.backend}"),
            "backend_requested": args.backend,
            "active_backend": active_backend("PITCH_CONTROL"),
            "n_requests": len(requests),
            "n_surfaces": len(surfaces),
            "tracemalloc_peak_bytes": peak_bytes,
            "seconds_under_trace": elapsed,
        },
    )


# --------------------------------------------------------------------------------------------------------------------
# compare: main against frozen
# --------------------------------------------------------------------------------------------------------------------


def _label_parts(label: str) -> tuple[str, str]:
    for h in ("LC", "RC", "L", "C", "R"):
        if label.startswith(h):
            return h, label[len(h) :]
    raise ValueError(label)


def _labels_join(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["period_id", "frame_id", "team_id", "player_id"]
    return a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)


def compare(args: argparse.Namespace) -> None:
    main, frozen = pathlib.Path(args.main), pathlib.Path(args.frozen)
    rm = json.loads((main / "record.json").read_text(encoding="utf-8"))
    rf = json.loads((frozen / "record.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {"main": rm["env"], "frozen": rf["env"]}
    if rm["input"]["frames_sha256"] != rf["input"]["frames_sha256"]:
        raise SystemExit("c1_record compare: the two runs did not read the same input")

    cm, cf = rm["preprocess"]["column_sha256"], rf["preprocess"]["column_sha256"]
    result["preprocess_columns_equal"] = sorted(c for c in cm if cf.get(c) == cm[c])
    result["preprocess_columns_differ"] = sorted(c for c in set(cm) | set(cf) if cm.get(c) != cf.get(c))

    def _eq(x: str, y: str, dx: pathlib.Path, dy: pathlib.Path) -> dict[str, int]:
        a, b = np.load(dx / x), np.load(dy / y)
        same = a.shape == b.shape
        n_equal = int((a == b).all(axis=1).sum()) if same else -1
        return {"n": int(a.shape[0]), "n_equal": n_equal, "shape_match": int(same)}

    result["pc_loop_main_vs_frozen"] = _eq("pc_loop_digests.npy", "pc_loop_digests.npy", main, frozen)
    result["pc_batch_vs_loop_frozen"] = _eq("pc_batch_digests.npy", "pc_loop_digests.npy", frozen, frozen)
    result["pc_batch_vs_loop_main"] = _eq("pc_batch_digests.npy", "pc_loop_digests.npy", main, main)
    result["pc_batch_frozen_vs_loop_main"] = _eq("pc_batch_digests.npy", "pc_loop_digests.npy", frozen, main)
    result["sg_graphs_main_vs_frozen"] = _eq("sg_loop_graph_digests.npy", "sg_loop_graph_digests.npy", main, frozen)

    joined = _labels_join(_read(main / "sg_loop_labels.parquet"), _read(frozen / "sg_loop_labels.parquet"))
    both = joined[joined["_merge"] == "both"]
    changed = both[both["label_a"] != both["label_b"]]
    mirror_only = 0
    for la, lb in zip(changed["label_a"], changed["label_b"], strict=True):
        (ha, va), (hb, vb) = _label_parts(la), _label_parts(lb)
        mirror_only += int(va == vb and _MIRROR[ha] == hb)
    result["sg_labels_constant_direction_main_vs_frozen"] = {
        "n_rows_main": int((joined["_merge"] != "right_only").sum()),
        "n_rows_frozen": int((joined["_merge"] != "left_only").sum()),
        "n_both": len(both),
        "n_equal": int(len(both) - len(changed)),
        "n_changed": len(changed),
        "n_changed_lateral_mirror_only": mirror_only,
        "changed_by_team": {str(k): int(v) for k, v in changed.groupby("team_id").size().items()},
    }

    if (frozen / "sg_batch_labels.parquet").exists() and (frozen / "sg_loop_framedir_labels.parquet").exists():
        jb = _labels_join(_read(frozen / "sg_loop_framedir_labels.parquet"), _read(frozen / "sg_batch_labels.parquet"))
        bb = jb[jb["_merge"] == "both"]
        result["sg_batch_vs_framedir_loop_frozen"] = {
            "n_loop": int((jb["_merge"] != "right_only").sum()),
            "n_batch": int((jb["_merge"] != "left_only").sum()),
            "n_both": len(bb),
            "n_equal": int((bb["label_a"] == bb["label_b"]).sum()),
        }

    timing: dict[str, Any] = {}
    for step in ("pc_loop", "pc_batch", "sg_loop", "sg_batch"):
        timing[step] = {"main": rm.get(step, {}).get("ms_per_frame"), "frozen": rf.get(step, {}).get("ms_per_frame")}
    for step in ("smooth_frames_s", "derive_velocities_s"):
        timing[step] = {"main": rm["preprocess"][step], "frozen": rf["preprocess"][step]}
    result["timing"] = timing

    fp = rf.get("fingerprint", {})
    if fp.get("fingerprint_s") is not None:
        steps = {
            "smooth_frames": rf["preprocess"]["smooth_frames_s"],
            "derive_velocities": rf["preprocess"]["derive_velocities_s"],
            "pc_loop": rf["pc_loop"]["seconds"],
            "pc_batch": rf["pc_batch"]["seconds"],
            "sg_loop": rf["sg_loop"]["seconds"],
            "link_actions_to_frames": fp["link_actions_to_frames_s"],
        }
        if "sg_batch" in rf:
            steps["sg_batch"] = rf["sg_batch"]["seconds"]
        ratios = {k: fp["fingerprint_s"] / v for k, v in steps.items() if v > 0}
        result["fingerprint"] = {
            "fingerprint_s": fp["fingerprint_s"],
            "n_actions": fp["n_actions"],
            "ratio_to_step": ratios,
            "max_ratio": max(ratios.values()),
            "within_2_percent_of_every_step": bool(max(ratios.values()) <= 0.02),
        }
    _write_json(pathlib.Path(args.out), result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-input")
    b.add_argument("--provider", default="gradientsports")
    b.add_argument("--match", default="10517")
    b.add_argument("--cache-dir", default=None)
    b.add_argument("--synthetic", type=int, default=0, help="n frames of the test fixture instead of pining (smoke)")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", required=True)
    p = sub.add_parser("peak")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=("numba", "numpy"), required=True)
    c = sub.add_parser("compare")
    c.add_argument("--main", required=True)
    c.add_argument("--frozen", required=True)
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"build-input": build_input, "run": run, "peak": peak, "compare": compare}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
