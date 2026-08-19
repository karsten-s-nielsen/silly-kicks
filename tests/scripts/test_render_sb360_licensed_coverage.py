import json
import pathlib

import pandas as pd

import scripts.render_sb360_licensed_coverage as r

_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "research" / "sb360_licensed_coverage"


def test_render_emits_expected_sections():
    df = pd.read_parquet(_DIR / "coverage.parquet")
    meta = json.loads((_DIR / "manifest_all.json").read_text(encoding="utf-8"))
    text = r.render(df, meta)
    for header in (
        "## Provenance",
        "## Frame-existence coverage",
        "## Battery aggregator coverage",
        "## ADR-062 visibility companions",
        "## Pitch coverage, roster, raises",
        "| Feature | mean observed fraction |",
    ):  # pins the F2 companion_fraction sub-table
        assert header in text, f"missing section/table: {header}"


def test_coverage_md_stamps_manifest_generation():
    """Staleness guard (spec §4.1): a parquet refreshed out-of-band without a re-render is caught,
    without pinning any value."""
    assert (_DIR / "coverage.md").exists(), "coverage.md not rendered yet -- run the render script"
    gen = json.loads((_DIR / "manifest_all.json").read_text(encoding="utf-8"))["generation"]
    md = (_DIR / "coverage.md").read_text(encoding="utf-8")
    assert gen in md, f"coverage.md does not stamp manifest generation {gen} -- re-render"
