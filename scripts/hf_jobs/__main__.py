"""`python -m scripts.hf_jobs {campaign,datasets,scaffold} ...` (ADR-128)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("campaign", "datasets", "scaffold"):
        print("usage: python -m scripts.hf_jobs {campaign,datasets,scaffold} ...", file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "campaign":
        from scripts.hf_jobs.campaign import main as run
    elif cmd == "datasets":
        from scripts.hf_jobs.datasets import main as run
    else:
        from scripts.hf_jobs.scaffold import main as run
    return int(run(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
