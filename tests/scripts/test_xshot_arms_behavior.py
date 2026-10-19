import subprocess
import sys

import pytest

#: Each visibility-gated trainer's stage-CLI argv WITHOUT the launcher's arm policy (ADR-128).
_NO_ARMS = {
    "scripts.train_xshot_occurrence": ["--dataset", "x"],
    "scripts.train_xcross_attempt": ["--dataset", "x", "--probe-dataset", "p"],
}


@pytest.mark.parametrize("module", sorted(_NO_ARMS))
def test_a_run_without_arms_json_fails_loud(module, tmp_path):
    """I4 (ADR-123 Amendment B) on the ADR-128 stage CLI: `--arms-json` is REQUIRED, so a run without the
    launcher's policy fails loud at parse time -- never an implicit all-private (which silently shipped the
    wrong candidate). `--plan` needs no clean tree, so this holds regardless of the working-tree state."""
    r = subprocess.run(  # noqa: S603 -- fixed test argv (sys.executable + a trainer module), never user input
        [sys.executable, "-m", module, *_NO_ARMS[module], "--store", str(tmp_path / "s"), "--plan"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "--arms-json" in (r.stdout + r.stderr)


def test_an_arms_payload_with_no_member_fails_loud():
    from scripts.hf_jobs.arms import ArmsUnavailableError, parse_arms

    with pytest.raises(ArmsUnavailableError, match="no non-empty arm"):
        parse_arms('{"arms": {"public": []}}')
    assert parse_arms('{"arms": {"public": ["a__b"]}}').ship_order == ("public",)
