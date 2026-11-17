"""Build ``tests/datasets/idsse/per_period_match.parquet`` from the raw public DFL files (R64, bar A26).

The fixture is the bronze of match ``DFL-MAT-J03WMX`` as THIS library's parser gives it:
``parse_dfl_match_info`` + ``parse_dfl_events`` on the raw DFL match-information and event XML files. It reads no
Databricks and no lakehouse table (the former source of this file was the luxury-lakehouse production bronze through
``scripts/extract_provider_fixtures.py``; that path is retired for this file).

Source of the raw files: the DFL Bundesliga Match Tracking Data -- Open Match Data 2022/23 (Bassek et al. 2025,
"An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data,
doi:10.1038/s41597-025-04505-y; CC BY 4.0), match ``DFL-MAT-J03WMX``. See ``NOTICE`` and
``tests/datasets/idsse/README.md``, which record the sha256 of the raw files and of the built file.

The file is the bronze exactly as the parser gives it: no column is added or changed after the parse, so its
``match_id`` is the parser's form, the bare DFL id ``J03WMX``.

Usage::

    python scripts/build_idsse_per_period_fixture.py \\
        --match-info <path>/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml \\
        --events <path>/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "tests" / "datasets" / "idsse" / "per_period_match.parquet"
_DFL_MATCH_ID = "J03WMX"  # the bare DFL MatchId; the parser keeps it as the match_id (idsse_native_match_id)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(match_info: Path, events: Path, out: Path) -> dict[str, object]:
    """Parse the two raw files and write the bronze fixture; return the provenance (paths, sha256, shape)."""
    from silly_kicks.providers.sportec import parse_dfl_events, parse_dfl_match_info

    info = parse_dfl_match_info(str(match_info))
    bronze = parse_dfl_events(str(events), match_info=info, match_id=_DFL_MATCH_ID)
    out.parent.mkdir(parents=True, exist_ok=True)
    bronze.to_parquet(out, index=False)
    return {
        "match_info": str(match_info),
        "match_info_sha256": _sha256(match_info),
        "events": str(events),
        "events_sha256": _sha256(events),
        "out": str(out),
        "out_sha256": _sha256(out),
        "rows": len(bronze),
        "columns": len(bronze.columns),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--match-info", type=Path, required=True, help="raw DFL match-information XML of J03WMX")
    parser.add_argument("--events", type=Path, required=True, help="raw DFL event XML of J03WMX")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help=f"output parquet (default: {_DEFAULT_OUT})")
    args = parser.parse_args(argv)
    for key, value in build(args.match_info, args.events, args.out).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
