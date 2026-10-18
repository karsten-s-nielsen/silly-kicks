import pathlib

from scripts.hf_jobs import registry as reg

REPO = pathlib.Path(__file__).resolve().parents[3]


def _population() -> set[str]:
    # ADR-056: derive the population, assert it exactly. A trainer = scripts/train_<name>.py.
    return {p.stem[len("train_") :] for p in (REPO / "scripts").glob("train_*.py")}


def test_registry_covers_population_exactly():
    covered = set(reg.REGISTRY) | set(reg.BLOCKED)
    assert covered == _population(), covered.symmetric_difference(_population())


def test_blocked_is_exactly_gk_retention_and_nonempty():
    assert set(reg.BLOCKED) == {"gk_retention"}
    assert reg.BLOCKED["gk_retention"].databricks_blocked is True


def test_every_weights_path_has_a_loadable_committed_model():
    # Head-to-head LOADS the committed model from these dirs (not a metrics.json -- 3 trainers ship none).
    for spec in reg.REGISTRY.values():
        for wp in spec.weights_paths:
            d = REPO / "silly_kicks" / wp
            assert d.is_dir(), f"{spec.name}: missing {d}"
            assert any((d / f).exists() for f in ("model.json", "rfcde_weights.npz", "metadata.json")), (
                f"{spec.name}: no loadable committed model artifact in {wp}"
            )


def test_floor_config_is_sane_and_has_a_scorer():
    from scripts.hf_jobs.headtohead import SCORERS  # one registered head-to-head scorer per trainer

    for spec in reg.REGISTRY.values():
        assert spec.metric_direction in {"lower_better", "higher_better"}
        assert spec.delta > 0
        assert spec.name in SCORERS, f"{spec.name}: no head-to-head scorer registered"
