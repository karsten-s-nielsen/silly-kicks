"""Fakes for tests of drivers that list refs and load one match per item (ADR-052 D14).

Consumed by every migrated driver's tests (Tasks 9-15). `install_fake_corpus` patches a driver
MODULE's own ``list_match_refs`` / ``load_match`` names, and replaces any stream wrapper it still
exposes (``load_matches`` / ``load_statsbomb_matches``) with one that FAILS loudly -- so a test aimed
at a driver that has not migrated yet names the cause instead of silently reaching the real network.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import _loader_pining as lp
import pandas as pd


def make_ref(provider: str, match_id: str, artifacts: Mapping[str, str] | None = None) -> lp.MatchRef:
    return lp.MatchRef(provider, str(match_id), dict(artifacts or {}))


def make_loaded(
    provider: str,
    match_id: str,
    *,
    actions: pd.DataFrame | None = None,
    frames: pd.DataFrame | None = None,
    home_team_id: object = "H",
    visible_area: pd.DataFrame | None = None,
    report: object | None = None,
) -> lp.LoadedMatch:
    return lp.LoadedMatch(
        provider,
        str(match_id),
        pd.DataFrame({"game_id": [str(match_id)]}) if actions is None else actions,
        frames,
        home_team_id,
        visible_area,
        report,
    )


class SpyLoader:
    """A ``load_match`` stand-in: records ``(ref.key, kwargs)`` per call; serves a `LoadedMatch` per key.

    ``fail`` keys raise ``RuntimeError`` (a recorded failure in `for_each`); ``exclude`` keys raise
    `MatchExcluded` with the given reason -- the same class the real loader raises.
    """

    def __init__(
        self,
        matches: Mapping[tuple[str, str], lp.LoadedMatch],
        *,
        fail: Collection[tuple[str, str]] = (),
        exclude: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self.matches = dict(matches)
        self.fail = set(fail)
        self.exclude = dict(exclude or {})
        self.calls: list[tuple[tuple[str, str], dict]] = []

    def __call__(self, ref: lp.MatchRef, **kwargs) -> lp.LoadedMatch:
        self.calls.append((ref.key, dict(kwargs)))
        if ref.key in self.fail:
            raise RuntimeError(f"fake load failure for {ref.key}")
        if ref.key in self.exclude:
            raise lp.MatchExcluded(
                self.exclude[ref.key], details={"player_off_pitch_rate": 0.0, "ball_off_pitch_rate": 0.001}
            )
        return self.matches[ref.key]


def _stream_refused(name: str):
    def _refuse(*_args, **_kwargs):
        raise AssertionError(f"{name} was called: this driver still streams its corpus (ADR-052 D14)")

    return _refuse


def install_fake_corpus(monkeypatch, module, *, refs: list[lp.MatchRef], loader: SpyLoader) -> None:
    """Fake the corpus seam on ``module``: ``refs`` and the spy loader for whichever of the load-seam
    names it exposes. Handles both shapes -- a driver that calls the primitives directly
    (``list_match_refs`` + ``load_match``) and one that goes through the source factory
    (``pining_source``, owner-ratified reuse) -- by patching each only when present, so one helper
    serves every migrated driver regardless of which seam it imported. A stream wrapper the module
    still exposes is replaced by a function that FAILS, so a test aimed at a not-yet-migrated driver
    fails fast and names the cause instead of reaching the network.
    """
    if hasattr(module, "pining_source"):
        monkeypatch.setattr(module, "pining_source", lambda *_a, **_kw: (list(refs), loader))
    if hasattr(module, "list_match_refs"):
        monkeypatch.setattr(module, "list_match_refs", lambda **_kw: list(refs))
    if hasattr(module, "load_match"):
        monkeypatch.setattr(module, "load_match", loader)
    for stream in ("load_matches", "load_statsbomb_matches"):
        if hasattr(module, stream):
            monkeypatch.setattr(module, stream, _stream_refused(stream))
