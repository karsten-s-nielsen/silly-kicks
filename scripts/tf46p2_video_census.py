"""TF-46 Arm 1 Part 2 pre-flight run 15: the SkillCorner video-overlap census (spec section 16.2, C-05).

Video availability is not derivable from the data, so the DL / owner compiles a reviewed ``match_id ->
source`` list OUTSIDE the repo (passed by ``--source-list``), applying three criteria: the full match,
publicly accessible, the same fixture. This driver validates the list schema and the criteria and writes
COUNTS ONLY per data profile (ADR-116: no match id or source enters the artifact). Video stays never-a-gate.
Refuses a dirty tree (ADR-037); writes under ``docs/research/tf46_part2/``.

Usage:

    python scripts/tf46p2_video_census.py --out docs/research/tf46_part2 --source-list reviewed_video.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._provenance import git_provenance, require_clean_tree
from scripts._tf46p2_common import RESEARCH_ROOT, DataProfile, video_census, write_run_aggregate

_RUN = "video_census"


def _profiles(spec_path: Path) -> dict[str, DataProfile]:
    """Load the ``match_id -> DataProfile`` mapping (a JSON of ``{match_id: [provider, tracking, gi, cs]}``)."""
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    return {mid: DataProfile(p[0], p[1], (p[2] or None), p[3]) for mid, p in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=RESEARCH_ROOT, help="Output DIRECTORY for the census aggregate.")
    ap.add_argument("--source-list", required=True, help="Reviewed match_id -> source JSON (outside the repo).")
    ap.add_argument("--profiles", required=True, help="match_id -> DataProfile JSON mapping.")
    ap.add_argument("--allow-dirty", action="store_true", help="dev run from a modified tree; recorded dirty: true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = video_census(Path(args.source_list), _profiles(Path(args.profiles)))
    write_run_aggregate(table, out, _RUN, prov)
    metrics = {"run": _RUN, "n_rows": len(table), "run_commit": prov["commit"], "run_tree_dirty": prov["dirty"]}
    (out / f"{_RUN}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
