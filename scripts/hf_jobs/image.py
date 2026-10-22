"""The prebuilt HF job image (ADR-129 Cycle 1b, S5.12): a digest-pinned environment the job CHECKS, never
installs.

ADR-128 Cycle 1 built the environment in every job (`pip install` + `uv sync`), paying the install on every
unit job. Cycle 1b builds the locked third-party environment ONCE into a GHCR image and the job only verifies
it (`uv sync --check`). The image is derived purely from `uv.lock`, the Dockerfile, the base digest, the extras
and the `uv` pin, so its tag is reproducible and a tag that does not match its inputs is refused in CI.

This module is STDLIB-ONLY and imports nothing from the repo: `python3 -m scripts.hf_jobs image tag` runs under
the system interpreter in CI (it only hashes files), and `_stages`/`launch` import it without dragging a
visibility concept into an HF job's import closure (the I1 gate, ADR-123 Amendment B). ``EXTRAS`` is a literal
whose coverage of the registry's extras is asserted by `tests/scripts/hf_jobs/test_image_lock.py`.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
DOCKERFILE = HERE / "image" / "Dockerfile"
LOCK = HERE / "image.lock.json"

#: The Dockerfile's ``FROM`` (kept equal to ``launch.IMAGE`` / ``launch.BASE_IMAGE`` by a test).
BASE_IMAGE = "python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0"
#: The ``uv`` pin installed into the image (kept equal to ``launch.UV_REQUIREMENT`` by a test).
UV_REQUIREMENT = "uv==0.9.28 --hash=sha256:7b8460a2b624d8ab27cb293a2c9f2393f9efc4e36e0fb886a6c2360e23fb48be"
#: The pyproject OPTIONAL-DEPENDENCY extras the image passes as `--extra` beyond `train`. Empty today: the one
#: registry "extra", ``statsbombpy``, is a dependency-GROUP (`--group hfjobs`), not a pyproject extra -- `uv`
#: rejects `--extra statsbombpy` ("not defined in optional-dependencies"). Group-provided extras are listed in
#: ``GROUP_PROVIDED`` so `test_image_lock.py` can assert the image holds every registry extra one way or the other.
EXTRAS: tuple[str, ...] = ()
#: Registry "extras" the image holds via the `hfjobs` dependency group rather than a `--extra` flag.
GROUP_PROVIDED: tuple[str, ...] = ("statsbombpy",)
#: The GHCR repository the image publishes to (owner lower-cased, as `job-image.yml` derives it).
REGISTRY_REF = "ghcr.io/karsten-s-nielsen/silly-kicks-job"
#: Where the in-image environment lives (the Dockerfile's `VIRTUAL_ENV`).
VENV = "/env/.venv"


def _base_digest() -> str:
    return BASE_IMAGE.partition("@")[2]


def _extra_flags() -> str:
    return " ".join(f"--extra {e}" for e in EXTRAS)


def _lf(p: pathlib.Path) -> bytes:
    """File bytes with CRLF normalized to LF, so the tag is the SAME on a CRLF Windows checkout and the LF
    tree CI/GHCR build from (CCTP-IMPL-33). ``.gitattributes`` also pins these files to LF."""
    return p.read_bytes().replace(b"\r\n", b"\n")


def inputs_sha256(root: pathlib.Path | None = None) -> str:
    """The SHA256 of every input the image is derived from: ``uv.lock`` + the Dockerfile + the base digest +
    the extras + the ``uv`` pin. Line endings are normalized, so the tag is checkout-independent. The tag is
    this digest's 12-char prefix."""
    root = root or REPO_ROOT
    h = hashlib.sha256()
    h.update(_lf(root / "uv.lock"))
    h.update(_lf(DOCKERFILE))
    h.update(_base_digest().encode("utf-8"))
    h.update(",".join(EXTRAS).encode("utf-8"))
    h.update(UV_REQUIREMENT.encode("utf-8"))
    return h.hexdigest()


def derived_tag(root: pathlib.Path | None = None) -> str:
    """The image tag derived from its inputs (`inputs_sha256`); `job-image.yml` refuses a push whose tag does
    not equal this."""
    return "env-" + inputs_sha256(root)[:12]


def load_lock() -> dict:
    """The committed image lock (`image.lock.json`): ``tag``, ``inputs_sha256``, ``registry_ref``, ``digest``,
    ``built_at``, ``startup_s``, ``startup_samples``, ``startup_jobs``. ``digest``/``startup_s`` are filled by
    Task 22's build + startup probe."""
    return json.loads(LOCK.read_text(encoding="utf-8"))


def image_ref() -> str:
    """``<registry_ref>@<digest>`` from the lock (what a job pulls). Before the image is built (Task 22 fills
    ``digest``) it falls back to ``BASE_IMAGE``, so the launcher imports resolve; no real job launches until
    the digest is recorded, and `preflight.check_image` refuses a lock without one."""
    lock = load_lock()
    if not lock.get("digest"):
        return BASE_IMAGE
    return f"{lock['registry_ref']}@{lock['digest']}"


#: The job's environment check (S5.12): verify the pinned environment matches, install nothing. Run from
#: ``/work`` against the fetched ``uv.lock`` with ``UV_PROJECT_ENVIRONMENT=/env/.venv``.
CHECK_COMMAND = " ".join(
    part
    for part in (
        f"UV_PROJECT_ENVIRONMENT={VENV}",
        "uv sync --frozen --check --no-install-project",
        f"--python {VENV}/bin/python",
        "--extra train",
        _extra_flags(),
        "--group hfjobs",
    )
    if part
)


def build_args() -> str:
    """The Docker build arguments for `job-image.yml` (one ``KEY=VALUE`` per line): the ``uv`` pin and the
    ``--extra`` flags that select every held extra."""
    return f"UV_REQ={UV_REQUIREMENT}\nEXTRAS={_extra_flags()}"


def _cli(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m scripts.hf_jobs image {tag,build-args,probe-registry,probe-startup}", file=sys.stderr)
        return 2
    cmd = argv[0]
    if cmd == "tag":
        print(derived_tag())
        return 0
    if cmd == "build-args":
        print(build_args())
        return 0
    if cmd in ("probe-registry", "probe-startup"):
        from scripts.hf_jobs import image_probe  # third-party (HfApi); imported only for the probes

        return image_probe.run(cmd, argv[1:])
    print(f"unknown image command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
