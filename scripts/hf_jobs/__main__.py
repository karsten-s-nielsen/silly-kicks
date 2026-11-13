"""`python -m scripts.hf_jobs {campaign,datasets,scaffold} ...` (ADR-128)."""

from __future__ import annotations

import sys


def _record(rest: list[str]) -> int:
    """`datasets record <name>`: measure a staged dataset into the lock, with the launcher's corpus policy."""
    import argparse
    import json

    from scripts.hf_jobs import arm_policy, datasets

    ap = argparse.ArgumentParser(prog="python -m scripts.hf_jobs datasets record")
    ap.add_argument("name")
    a = ap.parse_args(rest[1:])
    entry = datasets.record(a.name, policy=arm_policy)
    print(json.dumps({k: v for k, v in entry.items() if k != "units"}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("campaign", "datasets", "scaffold"):
        print("usage: python -m scripts.hf_jobs {campaign,datasets,scaffold} ...", file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    from scripts.hf_jobs.hub_client import install_timeouts

    install_timeouts()  # every Hub call of the controller is bounded (HFU spec 13.4)
    if cmd == "campaign":
        from scripts.hf_jobs.driver import main as run  # the minimal driver is the controller (ADR-128)
    elif cmd == "datasets" and rest[:1] == ["record"]:
        run = _record  # ADR-132: the launcher injects the corpus policy; datasets.py stays taxonomy-blind
    elif cmd == "datasets":
        from scripts.hf_jobs.datasets import main as run
    else:
        from scripts.hf_jobs.scaffold import main as run
    return int(run(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
