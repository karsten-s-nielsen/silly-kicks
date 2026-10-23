"""S5.12: the image lock matches its inputs; the job command checks the environment, never installs; the
build workflow is pinned and owner-gated."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.hf_jobs import image, launch

WORKFLOW = Path(".github/workflows/job-image.yml")


def _spec_cmd(spec=None):
    from tests.scripts.hf_jobs.test_launch_typed import _spec

    return launch.job_command(_spec() if spec is None else spec)[2]


def test_the_derived_tag_equals_the_locked_tag():
    assert image.derived_tag() == image.load_lock()["tag"]


def test_the_base_image_matches_the_launcher():
    assert image.BASE_IMAGE == launch.BASE_IMAGE == launch.IMAGE


def test_the_job_command_checks_the_environment_once_and_installs_nothing():
    cmd = _spec_cmd()
    assert cmd.count(image.CHECK_COMMAND) == 1
    assert "pip install" not in cmd
    assert all("--check" in line for line in cmd.splitlines() if "uv sync" in line)


def test_the_job_command_marks_the_environment_check():
    assert "phases mark after_env_check" in _spec_cmd()


def test_one_image_serves_every_spec():
    # ONE image holds every extra via the hfjobs group, so the environment check is a FIXED command that does
    # not vary per job (`JobSpec` carries no extras after the ADR-129 launcher rewrite).
    from tests.scripts.hf_jobs.test_launch_typed import _spec

    cmd = launch.job_command(_spec())[2]
    assert cmd.count(image.CHECK_COMMAND) == 1
    assert "extras" not in launch.JobSpec.__dataclass_fields__  # the image holds every extra via the group


def test_the_image_holds_every_registry_extra():
    from scripts.hf_jobs.registry import REGISTRY

    # Every registry "extra" is held either as a pyproject `--extra` (image.EXTRAS) or via the hfjobs
    # dependency group (image.GROUP_PROVIDED): statsbombpy is group-provided, not a pyproject extra.
    held = set(image.EXTRAS) | set(image.GROUP_PROVIDED)
    assert {e for spec in REGISTRY.values() for e in spec.extras} <= held


def test_the_check_command_passes_no_undefined_extra():
    # `uv` rejects `--extra statsbombpy` (it is a dependency group, not a pyproject extra). The image holds
    # statsbombpy via `--group hfjobs`; the check must never pass it as `--extra`.
    assert "--extra statsbombpy" not in image.CHECK_COMMAND
    assert "--group hfjobs" in image.CHECK_COMMAND


def test_the_derived_tag_is_line_ending_invariant(tmp_path):
    # B#33 regression: the tag must be the same on a CRLF Windows checkout and the LF CI/GHCR build.
    lf = tmp_path / "lf.lock"
    crlf = tmp_path / "crlf.lock"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert image._lf(lf) == image._lf(crlf)


def test_group_provided_extras_live_in_the_hfjobs_group():
    # image.GROUP_PROVIDED claims the hfjobs group supplies these (so no `--extra` is needed). Verify against
    # pyproject so the literal cannot drift from the real group (B#42).
    pp = Path("pyproject.toml").read_text(encoding="utf-8")
    block = pp.split("hfjobs = [", 1)[1].split("]", 1)[0]
    for e in image.GROUP_PROVIDED:
        assert e in block, f"{e!r} not found in the pyproject hfjobs dependency group"


def test_a_lock_without_startup_s_refuses_preflight(monkeypatch):
    from scripts.hf_jobs import preflight

    base = image.load_lock()
    monkeypatch.setattr(image, "load_lock", lambda: {**base, "startup_s": None})
    assert not preflight.check_image(resolve=lambda ref: True).ok


def test_check_image_fails_when_provenance_does_not_verify(monkeypatch):
    from scripts.hf_jobs import preflight

    lock = {**image.load_lock(), "startup_s": 42.0, "digest": "sha256:" + "a" * 64}
    monkeypatch.setattr(image, "load_lock", lambda: lock)
    out = preflight.check_image(resolve=lambda ref: True, attested=lambda ref: False)
    assert not out.ok and "provenance" in out.detail


def test_build_args_lists_the_uv_pin_and_the_extras():
    lines = dict(line.split("=", 1) for line in image.build_args().splitlines())
    assert lines["UV_REQ"] == image.UV_REQUIREMENT
    assert "statsbombpy" not in lines["EXTRAS"]  # group-provided, never a `--extra`
    assert lines["EXTRAS"] == image._extra_flags()


# ---- the build workflow (E-1): owner-gated, SHA-pinned ----------------------------------------------


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_the_workflow_triggers_only_on_the_image_tag():
    txt = _workflow_text()
    assert 'tags: ["job-image/env-*"]' in txt
    assert "environment: job-image" in txt


def test_every_action_is_pinned_by_a_full_commit_sha():
    uses = re.findall(r"uses:\s*(\S+)", _workflow_text())
    assert uses, "no `uses:` found"
    for u in uses:
        ref = u.partition("@")[2]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"action {u!r} is not pinned by a 40-hex SHA"


def test_the_login_precedes_the_manifest_inspect_and_the_attestation_follows_the_push():
    txt = _workflow_text()
    assert txt.index("docker/login-action") < txt.index("docker manifest inspect")
    assert txt.index("docker/build-push-action") < txt.index("attest-build-provenance")


def test_the_tag_comes_from_the_derived_output_not_the_ref_name():
    txt = _workflow_text()
    assert "steps.meta.outputs.tag" in txt
    assert "tags: ${{ github.ref_name }}" not in txt


def test_the_registry_probe_names_every_job_within_the_hf_label_limit(monkeypatch):
    """Without an explicit `name`, hf_hub derives the `name` label from the image; the digest-pinned GHCR ref
    (113 chars) broke HF's 100-character label limit (HTTP 400, 2026-10-23). Each probe job is named."""
    import types

    from scripts.hf_jobs import image_probe

    calls: list[dict] = []

    class _Api:
        def run_job(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(id=f"j{len(calls)}")

    long_ref = "ghcr.io/karsten-s-nielsen/silly-kicks-job@sha256:" + "a" * 64
    monkeypatch.setattr(image_probe, "_candidates", lambda: (long_ref, image.BASE_IMAGE))
    monkeypatch.setattr(image_probe, "_wait", lambda api, job_id, timeout_s=900: "COMPLETED")
    out = image_probe.probe_registry(api=_Api())
    assert len(long_ref) > launch.LABEL_MAX  # the case that failed
    assert [c["image"] for c in calls] == [long_ref, image.BASE_IMAGE]
    assert all(c.get("name") and len(c["name"]) <= launch.LABEL_MAX for c in calls)
    assert out["pulled"] == [long_ref, image.BASE_IMAGE]
