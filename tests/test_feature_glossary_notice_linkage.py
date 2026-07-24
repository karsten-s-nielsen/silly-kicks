"""Gate: every non-None ``attribution`` token in FEATURE_GLOSSARY appears verbatim in NOTICE (Task 12).

Empty registry => trivially green; becomes load-bearing as attributed entries are authored (ADR-005
attribution discipline -- do not blank an attribution to pass, add the citation to NOTICE).
"""

from pathlib import Path

from silly_kicks.feature_glossary import FEATURE_GLOSSARY


def test_every_attribution_token_is_in_notice():
    notice = Path("NOTICE").read_text(encoding="utf-8")
    missing = sorted(
        {
            fc.attribution
            for fc in FEATURE_GLOSSARY.values()
            if fc.attribution is not None and fc.attribution not in notice
        }
    )
    assert not missing, f"attribution tokens absent from NOTICE (add the citation): {missing}"
