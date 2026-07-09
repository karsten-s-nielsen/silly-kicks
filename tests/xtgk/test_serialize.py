import numpy as np
import pytest

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from tests.xtgk.conftest import three_band_cohort


def test_save_load_roundtrip_is_exact(tmp_path):
    a = three_band_cohort()
    m = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")
    out = tmp_path / "surface"
    m.save(out)
    m2 = MarkovPossessionValue.load(out)
    for p in (1, 2, 3):
        assert np.array_equal(m.surface(p), m2.surface(p))
        assert np.array_equal(m.support(p), m2.support(p))
    assert m2.provenance["xg_column"] == "xg"
    assert m.pressure_levels is not None and m2.pressure_levels is not None
    assert m2.pressure_levels.cutpoints == m.pressure_levels.cutpoints
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) == m2.value(z, 1)


def test_load_detects_tampering(tmp_path):
    a = three_band_cohort()
    m = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")
    out = tmp_path / "surface"
    m.save(out)
    (out / "SHA256SUMS").write_text("deadbeef  surfaces.npz\n")
    with pytest.raises(ValueError, match="checksum"):
        MarkovPossessionValue.load(out)
