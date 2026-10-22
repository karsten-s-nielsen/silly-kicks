from __future__ import annotations

from scripts._toml import load_toml, loads_toml


def test_load_toml_reads_tables_and_arrays(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('[campaign]\nname = "cycle1"\n[[jobs]]\njob = "ghost_gk"\nvariants = ["default"]\n', encoding="utf-8")
    d = load_toml(p)
    assert d["campaign"]["name"] == "cycle1"
    assert d["jobs"][0]["variants"] == ["default"]


def test_loads_toml_round_trips_text():
    assert loads_toml("a = 1\n") == {"a": 1}
