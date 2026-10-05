"""No committed research artifact carries a non-public corpus id (combined-cycle spec 11.8;
widened for B-C2-01 + owner ruling 2026-10-05, to ALL of docs/research/).

Owner tier rule (2026-10-05):
  - **Gradient Sports** match ids: PUBLIC at source -> readable everywhere.
  - **idsse** (``DFL-MAT-...``) + **DFL-OBJ** object ids: public DFL open dataset -> readable.
  - **player_id / team_id**: pseudonymous aggregate grain, non-reversible -> readable.
  - **SkillCorner**: ONLY the 20 public A-League Open-Data matches are readable; every other SkillCorner
    match id is owner-tier and must be pseudonymized (``scp_<12hex of sha256(id)>``) or absent.

The scan is recursive over JSON keys+values, parquet columns, and md/csv prose. A non-public SkillCorner
id is detected as a ``skillcorner__<id>`` shard token, a ``skillcorner/<id>`` or ``skillcorner_<id>``
prose token, or a 7-digit (``1``/``2``-leading) SkillCorner-shaped id inside a match-id list. Gradient
Sports ids are <=5 digits, so they never match the shape rule.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

_RESEARCH = Path("docs/research")
_DIRS = sorted(d for d in _RESEARCH.glob("*") if d.is_dir())

# the 20 public A-League SkillCorner Open-Data matches (MIT) -- the only readable SkillCorner ids
_PUB20 = frozenset(
    {
        "1874553",
        "1886347",
        "1899585",
        "1925299",
        "1927964",
        "1953632",
        "1959846",
        "1986691",
        "1996435",
        "1996436",
        "2006229",
        "2006363",
        "2007448",
        "2007721",
        "2010085",
        "2011166",
        "2013725",
        "2015213",
        "2016236",
        "2017461",
    }
)

_SC_SHARD = re.compile(r"skillcorner__(\d{6,})")  # a "<provider>__<id>" shard token
_SC_PROSE = re.compile(r"skillcorner[/_](\d{6,})")  # a provider-joined id in prose / csv
_SC_SHAPE = re.compile(r"^[12]\d{6}$")  # a 7-digit SkillCorner-shaped match id (GS ids are <=5 digits)
# a JSON key (or its parent) whose scalar list enumerates match identities
_LIST_KEY = re.compile(r"^(match_key|match_ids?|game_ids?|corpus_match_ids|requested_match_ids|skillcorner)$", re.I)
# NOTE: a parquet match_id/game_id column of PUBLIC ids (Gradient Sports, StatsBomb open data,
# the 20 A-League) is legitimate; only non-public SkillCorner VALUES are forbidden. StatsBomb-LICENSED
# ids are a separate tier not yet ruled (surfaced 2026-10-05) -- see docs/research/sb360_licensed_coverage.


def _bad_sc(tok: str) -> bool:
    return tok not in _PUB20


def _scan_json(node, parent_key, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, list) and v and all(isinstance(x, (str, int)) for x in v):
                if _LIST_KEY.match(str(k)) or _LIST_KEY.match(str(parent_key or "")):
                    for x in v:
                        if _SC_SHAPE.match(str(x)) and _bad_sc(str(x)):
                            out.append(f"non-public SkillCorner id {x!r} in list {k!r}")
            _scan_json(v, k, out)
    elif isinstance(node, list):
        for x in node:
            _scan_json(x, parent_key, out)
    elif isinstance(node, str):
        for mid in _SC_SHARD.findall(node):
            if _bad_sc(mid):
                out.append(f"non-public SkillCorner shard id {mid!r}")


_JSON = sorted(p for d in _DIRS for p in d.glob("**/*.json"))
_PARQUET = sorted(p for d in _DIRS for p in d.glob("**/*.parquet"))
_TEXT = sorted(p for d in _DIRS for p in [*d.glob("**/*.md"), *d.glob("**/*.csv")])


@pytest.mark.parametrize("path", _JSON, ids=str)
def test_no_nonpublic_ids_in_json(path):
    out: list[str] = []
    _scan_json(json.loads(path.read_text(encoding="utf-8")), None, out)
    assert not out, f"{path}: {out[:5]}"


@pytest.mark.parametrize("path", _PARQUET, ids=str)
def test_no_nonpublic_skillcorner_in_tables(path):
    df = pd.read_parquet(path)
    bad = {}
    for c in df.columns:
        vals = {str(v) for v in df[c].dropna().unique()}
        hit = sorted(v for v in vals if _SC_SHAPE.match(v) and _bad_sc(v))
        if hit:
            bad[c] = hit[:5]
    assert not bad, f"{path}: non-public SkillCorner values {bad}"


@pytest.mark.parametrize("path", _TEXT, ids=str)
def test_no_nonpublic_ids_in_text(path):
    text = path.read_text(encoding="utf-8")
    hits = [m for m in _SC_SHARD.findall(text) + _SC_PROSE.findall(text) if _bad_sc(m)]
    assert not hits, f"{path}: non-public SkillCorner ids {hits[:5]}"


@pytest.mark.parametrize("d", _DIRS, ids=str)
def test_the_scan_sees_every_dir(d):
    """Every research dir is covered -- an empty glob here would be a silent coverage hole."""
    assert any(d.glob("**/*")), d


def test_the_scans_are_non_vacuous():
    assert _JSON and _PARQUET and _TEXT, "a committed research dir should carry json + parquet + text"


def test_the_detector_actually_fires():
    """A planted non-public SkillCorner id must be caught in each surface (anti-vacuity)."""
    j: list[str] = []
    _scan_json({"match_ids": {"skillcorner": ["1999999"]}}, None, j)
    assert j, "json detector missed a planted non-public SkillCorner list id"
    assert [m for m in _SC_SHARD.findall("skillcorner__1999999") if _bad_sc(m)], "shard detector missed"
    assert "1886347" in _PUB20 and _bad_sc("1999999"), "the public-20 allowance is wired"
    # a Gradient Sports id (<=5 digits) and the 20 A-League ids are NOT flagged
    assert not _SC_SHAPE.match("10502") and not _bad_sc("1886347")
