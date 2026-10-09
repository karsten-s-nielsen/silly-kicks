"""No committed research artifact carries UNPUBLISHABLE content.

Publish policy (owner, standing): the only bar is REVERSIBILITY -- derived aggregates are shareable,
raw/reversible provider data is not. Corpus ids (``match_id`` / ``game_id`` / ``player_id`` / ``team_id``,
every provider) are non-reversible REFERENCES, not data: they let anyone WITH data access (the owner, a
licensee) VERIFY a result against the source, while anyone WITHOUT access cannot reconstruct the data from
an id. So ids are readable everywhere. An earlier SkillCorner id-pseudonymization tier ("only the 20 public
A-League ids readable") was a mis-recording of this policy; ids were always kept. Corrected by ADR-116; this
gate no longer inspects ids (the filename is legacy -- kept to avoid churn in the artifacts that reference it).

Research artifacts under ``docs/research/`` are curated aggregates (reliability, params, occlusion error,
hypothesis results, numerics), non-reversible by construction. Non-reversibility itself is a review
property, not a regex. The one publishable-leak a scan CAN catch is a LOCAL FILESYSTEM PATH -- an absolute
user/host/temp path that leaks a run environment and never belongs in a committed artifact (the standing
no-local-paths-in-committed-docs rule). That is this gate's mechanical teeth.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

_RESEARCH = Path("docs/research")
_DIRS = sorted(d for d in _RESEARCH.glob("*") if d.is_dir())

# A LOCAL FILESYSTEM PATH that must never enter a committed artifact: an absolute POSIX user/host/temp path, a
# home-relative ``~/`` path (the project's runbook style), or a Windows drive path (backslash OR forward slash).
# Relative repo paths (``docs/research/...``, ``scripts/...``) are fine and common, so the patterns anchor on an
# environment-revealing root.
_LOCAL_PATH = re.compile(
    r"(?:/home/[^\s\"']+"  # /home/<user>/...
    r"|/Users/[^\s\"']+"  # macOS /Users/<user>/...
    r"|/root/[^\s\"']+"  # /root/...
    r"|/tmp/[^\s\"']+"  # temp dir
    r"|/mnt/[^\s\"']+"  # /mnt/...
    r"|~/[^\s\"']+"  # home-relative (~/Development/..., ~/.venv/...): the runbook style
    # Windows drive path, backslash (C:\) OR forward slash (D:/); the (?<![A-Za-z]) lookbehind keeps a single
    # drive letter from matching a URL scheme's trailing letter (``https://`` -> ``s:/`` must NOT match).
    r"|(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"']+)"
)


def _scan_strings(node, out):
    """Collect every string scalar reachable in a JSON node (keys are config, not data -- values carry it)."""
    if isinstance(node, dict):
        for v in node.values():
            _scan_strings(v, out)
    elif isinstance(node, list):
        for x in node:
            _scan_strings(x, out)
    elif isinstance(node, str):
        for hit in _LOCAL_PATH.findall(node):
            out.append(hit)


_JSON = sorted(p for d in _DIRS for p in d.glob("**/*.json"))
_PARQUET = sorted(p for d in _DIRS for p in d.glob("**/*.parquet"))
_TEXT = sorted(p for d in _DIRS for p in [*d.glob("**/*.md"), *d.glob("**/*.csv")])


@pytest.mark.parametrize("path", _JSON, ids=str)
def test_no_local_paths_in_json(path):
    out: list[str] = []
    _scan_strings(json.loads(path.read_text(encoding="utf-8")), out)
    assert not out, f"{path}: local filesystem path(s) {out[:5]}"


@pytest.mark.parametrize("path", _PARQUET, ids=str)
def test_no_local_paths_in_tables(path):
    df = pd.read_parquet(path)
    bad = {}
    for c in df.columns:
        vals = {str(v) for v in df[c].dropna().unique()}
        hit = sorted(v for v in vals if _LOCAL_PATH.search(v))
        if hit:
            bad[c] = hit[:5]
    assert not bad, f"{path}: local filesystem path(s) {bad}"


@pytest.mark.parametrize("path", _TEXT, ids=str)
def test_no_local_paths_in_text(path):
    hits = _LOCAL_PATH.findall(path.read_text(encoding="utf-8"))
    assert not hits, f"{path}: local filesystem path(s) {hits[:5]}"


@pytest.mark.parametrize("d", _DIRS, ids=str)
def test_the_scan_sees_every_dir(d):
    """Every research dir is covered -- an empty glob here would be a silent coverage hole."""
    assert any(d.glob("**/*")), d


def test_the_scans_are_non_vacuous():
    assert _JSON and _PARQUET and _TEXT, "a committed research dir should carry json + parquet + text"


def test_the_detector_actually_fires():
    """A planted local path must be caught in each surface (anti-vacuity); corpus ids must NOT fire."""
    j: list[str] = []
    _scan_strings({"run": {"out": "/home/karsten/tf58/out/d1"}}, j)
    assert j, "json detector missed a planted /home path"
    assert _LOCAL_PATH.search(r"C:\Users\Karsten\scratch"), "windows backslash-path detector missed"
    assert _LOCAL_PATH.search("D:/Development/silly-kicks/x"), "windows forward-slash-path detector missed"
    assert _LOCAL_PATH.search("~/Development/silly-kicks/.venv/bin/python"), "home-relative (~/) detector missed"
    assert _LOCAL_PATH.search("/tmp/declared_excluded.json"), "temp-dir detector missed"  # noqa: S108
    # a corpus id and a RELATIVE repo path are NOT flagged (ids are non-reversible references; ADR-116)
    assert not _LOCAL_PATH.search("skillcorner__1912996"), "a corpus id must not be flagged as a path"
    assert not _LOCAL_PATH.search("docs/research/tf58_team_coordination/metrics.json"), "relative path flagged"
