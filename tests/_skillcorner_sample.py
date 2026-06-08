"""Shared SkillCorner pining-sample test support (TF-27 + the SPADL e2e).

Pure, network-free helpers for the GK-roster verification harness live here so the
e2e gate and the CI synthetic guard call the SAME comparator (no drifting second
path). Filesystem/sample constants are shared with tests/spadl/test_skillcorner_e2e.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- sample-dir convention (mirrors the maintainer's pre-download layout) ---

SAMPLE_DIR = Path(os.environ.get("SKILLCORNER_SAMPLE_DIR", r"C:\Users\Karsten\AppData\Local\Temp\skillcorner_sample"))

MATCH_IDS = [
    "1886347",
    "1899585",
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
]


def find_artifact(match_dir: Path, suffix: str) -> Path | None:
    """Find an artifact by filename suffix (bare or {id}-prefixed)."""
    candidates = list(match_dir.glob(f"*{suffix}"))
    return candidates[0] if candidates else None


def available_matches(required_suffixes: tuple[str, ...]) -> list[str]:
    """Match ids whose dir contains every required artifact suffix."""
    out = []
    for mid in MATCH_IDS:
        d = SAMPLE_DIR / mid
        if all(find_artifact(d, s) is not None for s in required_suffixes):
            out.append(mid)
    return out


# --- ground-truth extraction (pure) ---


def build_skillcorner_gk_truth(meta: dict) -> dict[str, list[str]]:
    """Map {str(team_id): [str(gk_player_id), ...]} from a SkillCorner match.json.

    Ground truth is players whose ``player_role.acronym == "GK"`` (the starter; subs
    carry "SUB"). Teams with zero GK-acronym players are omitted (no anchor). Never
    raises on cardinality.
    """
    truth: dict[str, list[str]] = {}
    for p in meta.get("players", []):
        role = (p.get("player_role") or {}).get("acronym")
        if role == "GK":
            truth.setdefault(str(p["team_id"]), []).append(str(p["id"]))
    return truth


# --- comparison (pure; the single CI-tested gate) ---


@dataclass(frozen=True)
class Mismatch:
    match_id: str
    team_id: str
    expected: tuple[str, ...]
    got: tuple[str, ...]
    rule: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgreementResult:
    matched: tuple[tuple[str, str], ...] = ()
    mismatched: tuple[Mismatch, ...] = ()
    no_roster_gk: tuple[tuple[str, str], ...] = ()

    @classmethod
    def empty(cls) -> AgreementResult:
        return cls()

    @property
    def is_perfect(self) -> bool:
        return len(self.mismatched) == 0

    def __add__(self, other: AgreementResult) -> AgreementResult:
        return AgreementResult(
            self.matched + other.matched,
            self.mismatched + other.mismatched,
            self.no_roster_gk + other.no_roster_gk,
        )

    def summary(self) -> str:
        lines = [f"matched={len(self.matched)} mismatched={len(self.mismatched)} no_roster_gk={len(self.no_roster_gk)}"]
        for m in self.mismatched:
            names = f" ({', '.join(m.names)})" if m.names else ""
            lines.append(
                f"  MISMATCH match={m.match_id} team={m.team_id} "
                f"expected={list(m.expected)} got={list(m.got)} rule={m.rule}{names}"
            )
        return "\n".join(lines)


def compare_gk_picks(
    truth: dict[str, list[str]],
    derived_picks: dict[tuple, list[str]],
    *,
    match_id: str | int,
    subset_allowlist: frozenset[tuple[str, str]] = frozenset(),
    name_map: dict[str, str] | None = None,
) -> AgreementResult:
    """Compare one match's derived GK picks against its roster truth.

    Default rule: exact set equality per team (catches over-identification). For a
    team in ``subset_allowlist`` (set of ``(str(match_id), str(team_id))``) the rule
    relaxes to ``truth[team] <= picks[team]``. Teams with no roster GK -> no_roster_gk
    (not a failure). String-casts the team key on both sides (derived keys carry int
    team_id). ``name_map`` (id -> short_name) is used only for diagnostics.
    """
    mid = str(match_id)
    names = name_map or {}
    derived: dict[str, list[str]] = {str(tid): [str(p) for p in pids] for (_g, tid), pids in derived_picks.items()}
    matched: list[tuple[str, str]] = []
    mismatched: list[Mismatch] = []
    no_roster: list[tuple[str, str]] = []
    for team in sorted(set(truth) | set(derived)):
        if team not in truth:
            no_roster.append((mid, team))
            continue
        expected = set(truth[team])
        got = set(derived.get(team, []))
        allow = (mid, team) in subset_allowlist
        ok = expected <= got if allow else expected == got
        if ok:
            matched.append((mid, team))
        else:
            ids = sorted(expected | got)
            mismatched.append(
                Mismatch(
                    mid,
                    team,
                    tuple(sorted(expected)),
                    tuple(sorted(got)),
                    "subset" if allow else "exact",
                    tuple(names.get(i, i) for i in ids),
                )
            )
    return AgreementResult(tuple(matched), tuple(mismatched), tuple(no_roster))
