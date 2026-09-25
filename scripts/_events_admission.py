"""Events-only SkillCorner admission -- an edge POLICY layer, not part of the loader (spec §4.1, §8).

The SkillCorner S1 geometry gate needs built tracking, so ``load_match(events_only=True)`` cannot run
it and stays a PURE loader (it consults no artifact). An events-only consumer admits an S1-excluded
match ONLY if its EVENTS pass the Task-0 event-side check, recorded in a committed, provenance-stamped
verdict artifact. "Policy lives at the edge, never in the shared engine" (AGENTS.md): this module is
that edge, and Rule D makes it the one sanctioned events-only entry for consumers.

Imports are QUALIFIED (``from scripts._xxx import ...``), not bare: this module is imported as
``scripts._events_admission`` by consumers OUTSIDE tests/scripts/ (the migrated calibrate_* drivers
and their tests/calibration/ suite), where ``tests/scripts/conftest.py`` -- the only thing that puts
scripts/ on sys.path -- has not run. A bare ``from _loader_pining import ...`` dies there with
ModuleNotFoundError; the qualified form resolves wherever ``scripts`` is importable as a package.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from scripts._loader_pining import LoadedMatch, MatchExcluded, MatchRef, load_match

#: The Task-0 verdict artifact (committed at commit 2). Required only when a SkillCorner ref is loaded.
ADMISSION_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "docs" / "research" / "skillcorner_s1_event_validity" / "verdicts.json"
)

#: The per-match tracking-side status the Task-0 reduce assigns every listed match (spec §8).
STATUSES = ("s1_passed", "s1_excluded", "tracking_unloadable", "events_unloadable")

_PROVIDER = "skillcorner"


class AdmissionRefusedError(RuntimeError):
    """The events-only pass refuses BEFORE loading: a missing/unprovenanced/dirty/tampered artifact, or
    an unmeasured requested match without ``--allow-unmeasured`` (spec §4.1). Refusal, not silent
    exclusion, so the corpus never shrinks unnoticed."""


def listing_digest(matches: Mapping[str, Mapping[str, object]]) -> str:
    """A stable digest of the verdict listing -- single-sourced by the Task-0 producer and this reader.

    Over the sorted ``(key, status, event_verdict)`` triples, so a regenerated artifact whose verdicts
    changed gets a new digest (which joins the pass's ``token_inputs``), and a tampered listing fails
    the load-time check.
    """
    h = hashlib.sha256()
    for key in sorted(matches):
        rec = matches[key]
        h.update(f"{key}\x1f{rec.get('status')}\x1f{rec.get('event_verdict')}\x1e".encode())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class AdmissionRecord:
    """What the preflight decided, threaded into ``XtFitProvenance`` / ``token_inputs`` (spec §4.4)."""

    digest: str | None  # None when no SkillCorner ref was requested (no artifact consulted)
    unmeasured_admitted: tuple[str, ...]  # join-key strings admitted under allow_unmeasured, else ()


@dataclasses.dataclass(frozen=True)
class EventsOnlyAdmission:
    """The loaded, fail-closed verdict artifact + the per-ref admission decision."""

    digest: str
    _matches: Mapping[str, Mapping[str, object]]

    @classmethod
    def load(cls, path: Path = ADMISSION_ARTIFACT) -> EventsOnlyAdmission:
        """Read + validate the verdict artifact, FAIL-CLOSED (spec §4.1).

        Refuses a missing file, one lacking provenance (``run_commit``), one built from a dirty tree
        (``run_tree_dirty``), or one whose stored ``digest`` does not match its listing (tamper). All
        four are `AdmissionRefusedError`, in seconds, before any match loads.
        """
        if not Path(path).is_file():
            raise AdmissionRefusedError(
                f"events-only SkillCorner admission needs the verdict artifact {path}, which is absent. "
                f"Run scripts/build_skillcorner_s1_event_validity.py (resumable) or request no SkillCorner ref."
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not payload.get("run_commit"):
            raise AdmissionRefusedError(f"{path} has no run_commit: an unprovenanced verdict artifact is refused.")
        if payload.get("run_tree_dirty"):
            raise AdmissionRefusedError(f"{path} was built from a dirty tree (run_tree_dirty=true); refused.")
        matches = payload.get("matches") or {}
        if payload.get("digest") != listing_digest(matches):
            raise AdmissionRefusedError(f"{path} digest does not match its listing (tampered or truncated); refused.")
        return cls(digest=str(payload["digest"]), _matches=matches)

    def preflight(self, refs: Iterable[MatchRef], *, allow_unmeasured: bool) -> AdmissionRecord:
        """Refuse the pass if any requested SkillCorner ref is UNMEASURED, unless ``allow_unmeasured``.

        Excluding unmeasured matches by default would silently shrink the corpus, so the default is to
        refuse and name them; ``allow_unmeasured`` admits and RECORDS their keys (the ``--allow-*`` idiom).
        """
        unmeasured = sorted(_join(r.key) for r in refs if r.provider == _PROVIDER and _join(r.key) not in self._matches)
        if unmeasured and not allow_unmeasured:
            raise AdmissionRefusedError(
                f"{len(unmeasured)} requested SkillCorner match(es) are unmeasured by the verdict artifact: "
                f"{unmeasured}. Re-run scripts/build_skillcorner_s1_event_validity.py (resumable, cheap on a "
                f"warm cache), or pass --allow-unmeasured to admit and record them."
            )
        return AdmissionRecord(digest=self.digest, unmeasured_admitted=tuple(unmeasured) if allow_unmeasured else ())

    def check(self, ref: MatchRef) -> None:
        """Raise `MatchExcluded` if this SkillCorner ref is not admitted; a no-op otherwise.

        ``s1_passed`` -> admitted. ``s1_excluded`` / ``tracking_unloadable`` -> admitted iff the event
        verdict is ``sound``. ``events_unloadable`` -> never (its events cannot load). Unmeasured (only
        reachable under ``--allow-unmeasured``, since preflight would otherwise have refused) ->
        admitted. Non-SkillCorner refs are never gated.
        """
        if ref.provider != _PROVIDER:
            return
        rec = self._matches.get(_join(ref.key))
        if rec is None:
            return  # unmeasured but admitted under allow_unmeasured (preflight recorded it)
        status, verdict = rec.get("status"), rec.get("event_verdict")
        if status == "s1_passed":
            return
        if status in ("s1_excluded", "tracking_unloadable") and verdict == "sound":
            return
        raise MatchExcluded(
            f"events-only SkillCorner {ref.match_id}: status={status}, event_verdict={verdict} "
            f"({rec.get('reason', '')})",
            details={"status": status, "event_verdict": verdict, "reason": rec.get("reason", "")},
        )


def _join(key: tuple[str, str]) -> str:
    return f"{key[0]}__{key[1]}"


def _no_admission_check(ref: MatchRef) -> None:
    """The events-only check for a pass with no SkillCorner ref -- there is no artifact to consult."""
    return None


def events_only_loader(
    refs: Iterable[MatchRef],
    *,
    allow_unmeasured: bool = False,
    cache_dir=None,
    artifact_path: Path = ADMISSION_ARTIFACT,
) -> tuple[Callable[[MatchRef], LoadedMatch], AdmissionRecord]:
    """The ONE sanctioned events-only entry for consumers (Rule D): ``(load, record)``.

    Runs the preflight once (loading the verdict artifact only when a SkillCorner ref is requested),
    then returns a per-item ``load(ref)`` that CHECKS admission and then calls
    ``load_match(ref, events_only=True, ...)``. The returned callable is a closure -- Rule D keys on
    this enclosing function, so the ``load_match(events_only=True)`` inside it is the allowed call.
    """
    refs = list(refs)
    if any(r.provider == _PROVIDER for r in refs):
        admission = EventsOnlyAdmission.load(artifact_path)
        record = admission.preflight(refs, allow_unmeasured=allow_unmeasured)
        check = admission.check
    else:
        record = AdmissionRecord(digest=None, unmeasured_admitted=())
        check = _no_admission_check

    def load(ref: MatchRef) -> LoadedMatch:
        check(ref)
        return load_match(ref, events_only=True, cache_dir=cache_dir)

    return load, record
