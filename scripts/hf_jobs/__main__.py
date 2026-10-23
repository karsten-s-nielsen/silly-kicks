"""`python -m scripts.hf_jobs {campaign,datasets,scaffold,closures} ...` (ADR-128/ADR-129)."""

from __future__ import annotations

import sys

_COMMANDS = ("campaign", "datasets", "scaffold", "closures", "image")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in _COMMANDS:
        print(f"usage: python -m scripts.hf_jobs {{{','.join(_COMMANDS)}}} ...", file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "campaign":
        from scripts.hf_jobs.campaign import main as run
    elif cmd == "datasets":
        from scripts.hf_jobs.datasets import main as run
    elif cmd == "closures":
        from scripts.hf_jobs.closures import main as run
    elif cmd == "image":
        from scripts.hf_jobs.image import _cli

        return int(_cli(rest) or 0)
    else:
        from scripts.hf_jobs.scaffold import main as run
    return int(run(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
