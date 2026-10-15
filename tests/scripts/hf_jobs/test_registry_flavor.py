from scripts.hf_jobs import registry as reg


def test_every_trainer_runs_on_cpu_upgrade():
    # Owner directive 2026-10-14: cpu-upgrade only until further notice (spec Global Constraints).
    for name, spec in reg.REGISTRY.items():
        assert spec.flavor == "cpu-upgrade", f"{name} is {spec.flavor}"


def test_blocked_trainer_is_untouched():
    # gk_retention stays BLOCKED (Databricks-marts); not re-flavored into REGISTRY.
    assert "gk_retention" in reg.BLOCKED
    assert "gk_retention" not in reg.REGISTRY
