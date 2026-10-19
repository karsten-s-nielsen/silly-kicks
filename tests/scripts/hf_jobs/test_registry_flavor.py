from scripts.hf_jobs import registry as reg


def test_every_trainer_runs_on_cpu_upgrade():
    """Owner directive 2026-10-14: cpu-upgrade only (INV-4). ADR-128: the flavor is ONE constant, the
    launcher's `FLAVOR`; no registry entry can name another (the per-entry field is gone)."""
    from scripts.hf_jobs import launch

    assert launch.FLAVOR == "cpu-upgrade"
    for name, spec in {**reg.REGISTRY, **reg.BLOCKED}.items():
        assert not hasattr(spec, "flavor"), f"{name} carries its own flavor"


def test_blocked_trainer_is_untouched():
    # gk_retention stays BLOCKED (Databricks-marts); not re-flavored into REGISTRY.
    assert "gk_retention" in reg.BLOCKED
    assert "gk_retention" not in reg.REGISTRY
