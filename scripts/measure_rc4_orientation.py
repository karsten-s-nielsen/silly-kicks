"""Measure BOTH sides of the ADR-051 RC4 claim: does the pining loader ship ORIENTED frames?

Produces `docs/research/adr028_rc4_orientation/{prefix,postfix}_measurement.json`. Committed because
an artifact nobody can re-run is an artifact nobody can check -- the first version of this
measurement was an ad-hoc pass, and it shipped a `tracking_limit=3000` cap that it recorded nowhere,
halving a published headline (spec 11.1 / the README beside the artifacts).

WHAT IT MEASURES. For one match per provider: what fraction of player rows carry
`team_attacking_direction`, and what fraction of actions `acting_team_attacks_rtl` flips. RC4 is the
defect where `build_skillcorner_frames` forced `output_convention="absolute_frame"`, leaving the
label NULL on every row so the ADR-028 re-projection silently no-opped.

IDSSE IS THE CONTROL and must come out byte-identical on both sides: `sportec.py` calls
`finalize_orientation` unconditionally, before its own convention branch, so its frames are labelled
regardless. A previous cycle "extended" RC4 to IDSSE on an assumed -- never measured -- premise; this
control is what refutes that, and it is why the script measures a provider it expects not to move.

RUNNING THE TWO SIDES, and the awkwardness is real rather than a wording problem. The **postfix**
side is a plain clean-tree run at a commit that contains the fix:

    python scripts/measure_rc4_orientation.py --label postfix

The **prefix** side cannot be a bare `git checkout <pre-fix commit>` -- this script does not exist at
any commit before 4.73.0, so the checkout removes the very thing you are trying to run. Copy it into
a worktree at the older commit instead:

    git worktree add --detach /tmp/rc4-pre <pre-fix commit>
    cp scripts/measure_rc4_orientation.py /tmp/rc4-pre/scripts/
    cd /tmp/rc4-pre && python scripts/measure_rc4_orientation.py --label prefix --allow-dirty

`--allow-dirty` is REQUIRED there and the artifact will record `run_tree_dirty: true` -- which is
correct, not a wart: the code that ran is that commit PLUS an imported file, so it is not that commit.
An artifact claiming otherwise would be the false-provenance bug this driver exists to avoid. The two
committed artifacts predate this script and were produced from genuinely clean checkouts (`5a67212`
and `4b15365`); see the README beside them.

Requires pining access (owner token). `--tracking-limit` defaults to None -- FULL frames -- and is
recorded in the artifact either way. Do not pass a cap and then cite the numbers: a truncated frame
set leaves `(game_id, period_id, team_id)` keys out of the orientation lookup, and those actions
default to no-flip SILENTLY, so every flip fraction it produces is a lower bound.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "research" / "adr028_rc4_orientation"
_PROVIDERS = ("skillcorner", "idsse")


def measure(provider: str, *, tracking_limit: int | None, cache_dir: str | None) -> dict:
    """One match's orientation state. A load failure PROPAGATES -- it is NOT a result.

    This used to catch `Exception` and return ``{"error": ...}``, which made the driver structurally
    unable to fail, and MEASURED: a tokenless run returned normally, `_work` wrapped the error dict in
    an ordinary one-row frame, `for_each` wrote it as a healthy shard, `res.failures` stayed empty so
    `run()`'s guard never fired, and `main()` wrote the artifact into
    ``docs/research/adr028_rc4_orientation`` -- the DEFAULT ``--out-dir``, home of the two committed,
    cited artifacts -- and exited 0. A non-measurement published as the measurement, inside the driver
    written to prevent exactly that.

    It was worse than a one-off: because the error row was written as a shard, `already_done()` is
    true forever, so every later run reported ``skip (shard exists)`` and re-published the memoized
    error. Recovery would have needed an operator to delete a 16-hex generation directory by hand.
    Raising instead is what lets `for_each` record a FAILURE and write NO shard, so a resume redoes
    the item -- the property this driver adopted the ADR-052 seam to buy.
    """
    from _loader_pining import load_matches

    from silly_kicks.tracking import OrientationUnresolvedWarning
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

    loaded = next(
        iter(
            load_matches(
                providers=[provider],
                max_per_provider=1,
                tracking_limit=tracking_limit,
                cache_dir=cache_dir,
            )
        ),
        None,
    )
    if loaded is None:
        raise RuntimeError(
            f"{provider}: load_matches yielded no match, so there is nothing to measure. "
            "Recording an absence as a result would publish it as one."
        )

    _provider, match_id, actions, frames, _home = loaded
    players = frames[~frames["is_ball"].astype(bool)]
    label = players["team_attacking_direction"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flip = acting_team_attacks_rtl(actions, frames)
    n_warn = sum(1 for w in caught if issubclass(w.category, OrientationUnresolvedWarning))

    return {
        "match_id": str(match_id),
        "n_frames": int(frames["frame_id"].nunique()),
        "player_rows": len(players),
        "unlabelled_fraction": float(label.isna().mean()),
        "distinct_labels": sorted(str(x) for x in label.dropna().unique()),
        "n_actions": len(actions),
        # The COUNT rides beside the fraction on purpose: it is what lets a later reader divide by 22
        # and notice a frame cap, which is how the first version's undisclosed cap was caught.
        "n_flip_true": int(flip.sum()),
        "flip_true_fraction": float(flip.mean()),
        "orientation_warnings": n_warn,
    }


def run(*, label: str, tracking_limit: int | None, cache_dir: str | None, shard_dir: str, run_prov: dict) -> dict:
    """Walk the two providers through the ADR-052 shared seam.

    Two items is a small corpus, but adoption is universal by decision, not by size: "expensive
    enough to need resume" is a judgement that rots, and the seam also buys the conservation check
    and the declared-input generation token for free. Each provider's result is a ONE-ROW frame,
    which is what the `work -> tidy frame` contract wants (ADR-052 D7 -- no per-item metadata rides
    outside the frame).
    """
    import pandas as pd
    from _driver import for_each, shard_path

    def _work(provider: str):
        print(f"\n===== {provider} ({label}) =====", flush=True)
        result = measure(provider, tracking_limit=tracking_limit, cache_dir=cache_dir)
        for k, v in result.items():
            print(f"  {k:22s} {v}", flush=True)
        # distinct_labels is a list; json-encode so the row stays scalar-valued and round-trips.
        row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in result.items()}
        row["provider"] = provider
        return pd.DataFrame([row])

    res = for_each(
        list(_PROVIDERS),
        key=lambda provider: (provider,),
        work=_work,
        shard_root=Path(shard_dir) / label,
        # What determines a shard's CONTENT -- and `run_commit` is LOAD-BEARING here, not decoration.
        # This driver's whole subject is that the CODE differs between the two sides, and a token of
        # {measurement, tracking_limit, label} does not capture code at all: two runs at different
        # commits under the same label would share a generation, so the second would silently REUSE
        # the first's shards while stamping its own `run_commit` into the artifact. That is the
        # false-provenance failure this driver exists to prevent, reintroduced one level down. With
        # the commit in the token, a code change starts a new generation instead.
        token_inputs={
            "measurement": "rc4_orientation",
            "tracking_limit": tracking_limit,
            "label": label,
            "run_commit": run_prov["commit"],
            "run_tree_dirty": run_prov["dirty"],
        },
        tag="rc4_orientation",
        label="provider",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} provider(s) failed: {res.failures}")

    record = {
        "_provenance": {
            "label": label,
            "run_commit": run_prov["commit"],
            "run_tree_dirty": run_prov["dirty"],
            "run_tree_state": run_prov["tree_state"],
            "tracking_limit": tracking_limit,
            "max_per_provider": 1,
        }
    }
    for k in res.keys:
        frame = pd.read_parquet(shard_path(res.shard_dir, k))
        if not len(frame):
            raise RuntimeError(
                f"shard {k!r} is empty: that provider produced no measurement row. Skipping it would "
                "drop a provider from an artifact that still reads as complete."
            )
        # `.to_dict()` is typed dict[Hashable, Any]; the keys are our own column names, so narrow
        # to str explicitly rather than assigning a Hashable-keyed dict into a str-keyed record.
        row = {str(k): v for k, v in frame.iloc[0].to_dict().items()}
        provider = str(row.pop("provider"))
        record[provider] = {
            kk: (json.loads(vv) if kk == "distinct_labels" else vv) for kk, vv in row.items() if vv is not None
        }
    # Defence in depth, and CURRENTLY UNREACHABLE -- stated rather than dressed up as a live guard.
    # Every path that could shorten `record` is already closed upstream: `for_each` raises on a
    # non-injective key (ADR-052 `_require_injective`), `res.failures` raises above, and an empty
    # shard raises just above. So this cannot fire today, and there is deliberately NO test for it: a
    # contrived trigger would be a guard that only looks like it guards, which is the failure mode
    # this cycle keeps finding. It stays because it is one line and it catches a FUTURE refactor that
    # reintroduces skipping -- an artifact citing fewer providers than it set out to measure still
    # READS as complete, and IDSSE is the CONTROL, so an artifact without it proves nothing about
    # SkillCorner.
    missing = sorted(set(_PROVIDERS) - set(record))
    if missing:
        raise RuntimeError(f"artifact is missing {missing}; without them it would read as complete")
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=["prefix", "postfix"], help="which side of the fix this run is")
    ap.add_argument(
        "--tracking-limit",
        type=int,
        default=None,
        help="frames per match; default None = FULL frames. A cap DEPRESSES flip_true_fraction and is "
        "recorded in the artifact -- do not cap and then cite the numbers.",
    )
    ap.add_argument("--cache-dir", default=None, help="persist downloaded artifacts here to speed a re-run")
    ap.add_argument(
        "--out-dir",
        default=str(_OUT_DIR),
        help="defaults to the COMMITTED artifact directory, so a successful run regenerates the "
        "cited files in place -- which is the intended way to refresh them.",
    )
    ap.add_argument(
        "--allow-capped-overwrite",
        action="store_true",
        help="permit a CAPPED run to overwrite the committed artifacts. Refused by default: an "
        "undisclosed cap is the exact defect this driver was written after.",
    )
    ap.add_argument(
        "--shard-dir",
        default="rc4_orientation_shards",
        help="per-provider resumable shards (ADR-052). Root-relative by default, which .gitignore covers.",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Measure from a modified working tree. The artifact still records run_tree_dirty=true -- "
        "the hatch permits a dev run, it never launders the fact.",
    )
    args = ap.parse_args()

    # FIRST, before any corpus work: this writes a cited artifact, and a measurement whose code state
    # is unknown cannot be checked later. Enforcement lives here rather than in `run()` so `run()`
    # stays testable without mocking git.
    from _provenance import git_provenance, require_clean_tree

    run_prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    # SAME REASON as the provenance check above, and it belongs in the same place: BEFORE any corpus
    # work. This sat after `run()` first, which meant an operator paid for the whole pass and was
    # then refused -- a refusal that arrives after the cost is a worse version of no refusal.
    out_dir = Path(args.out_dir)
    if args.tracking_limit is not None and out_dir.resolve() == _OUT_DIR.resolve() and not args.allow_capped_overwrite:
        raise SystemExit(
            f"refusing to overwrite the committed artifacts in {out_dir} with a CAPPED measurement "
            f"(--tracking-limit {args.tracking_limit}). A cap DEPRESSES flip_true_fraction, so these "
            "numbers would be lower bounds replacing full-frame ones. Write elsewhere with "
            "--out-dir, or pass --allow-capped-overwrite if that is genuinely what you want."
        )

    record = run(
        label=args.label,
        tracking_limit=args.tracking_limit,
        cache_dir=args.cache_dir,
        shard_dir=args.shard_dir,
        run_prov=run_prov,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}_measurement.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
