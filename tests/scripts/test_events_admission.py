"""`scripts/_events_admission.py` -- events-only SkillCorner admission at the edge (spec §4.1, §8).

`load_match(events_only=True)` stays a PURE loader; admission (against the Task-0 verdict artifact)
is policy at the edge. tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import json

import pytest
from _fake_corpus import make_loaded, make_ref

# QUALIFIED, not bare: `scripts._events_admission` imports `MatchExcluded` from
# `scripts._loader_pining` (the seam's qualified-import convention). A bare `import _loader_pining`
# here would be a DIFFERENT module object, so `pytest.raises(lp.MatchExcluded)` would not match the
# class `EventsOnlyAdmission.check` actually raises. Importing the same qualified module aligns them.
import scripts._events_admission as ea
import scripts._loader_pining as lp


def _write_artifact(path, matches, *, run_commit="abc123", dirty=False, digest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "digest": digest if digest is not None else ea.listing_digest(matches),
        "run_commit": run_commit,
        "run_tree_dirty": dirty,
        "matches": matches,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_MATCHES = {
    "skillcorner__pass": {"status": "s1_passed", "event_verdict": "sound", "reason": ""},
    "skillcorner__excl_sound": {"status": "s1_excluded", "event_verdict": "sound", "reason": "S1 ball off-pitch"},
    "skillcorner__excl_bad": {"status": "s1_excluded", "event_verdict": "reversed", "reason": "S1 + reversed events"},
    "skillcorner__track_unl": {"status": "tracking_unloadable", "event_verdict": "sound", "reason": "tracking DNF"},
    "skillcorner__ev_unl": {"status": "events_unloadable", "event_verdict": None, "reason": "events DNF"},
}


@pytest.fixture
def artifact(tmp_path):
    return _write_artifact(tmp_path / "verdicts.json", _MATCHES)


# ---- Step 1: per-status admission ----------------------------------------------------------------


def test_admission_per_status(artifact):
    adm = ea.EventsOnlyAdmission.load(artifact)
    adm.check(make_ref("skillcorner", "pass"))  # s1_passed -> admitted (no raise)
    adm.check(make_ref("skillcorner", "excl_sound"))  # s1_excluded + sound -> admitted
    adm.check(make_ref("skillcorner", "track_unl"))  # tracking_unloadable + sound -> admitted
    for mid in ("excl_bad", "ev_unl"):
        with pytest.raises(lp.MatchExcluded):
            adm.check(make_ref("skillcorner", mid))


def test_non_skillcorner_ref_is_never_checked(artifact):
    ea.EventsOnlyAdmission.load(artifact).check(make_ref("idsse", "anything"))  # passes through, no raise


def test_matchexcluded_carries_status_and_verdict(artifact):
    adm = ea.EventsOnlyAdmission.load(artifact)
    with pytest.raises(lp.MatchExcluded) as exc:
        adm.check(make_ref("skillcorner", "excl_bad"))
    assert exc.value.details.get("status") == "s1_excluded"
    assert exc.value.details.get("event_verdict") == "reversed"


# ---- Step 2: preflight refuses unmeasured --------------------------------------------------------


def test_preflight_refuses_unmeasured_by_name(artifact):
    adm = ea.EventsOnlyAdmission.load(artifact)
    refs = [make_ref("skillcorner", "pass"), make_ref("skillcorner", "NEW_UNLISTED")]
    with pytest.raises(ea.AdmissionRefusedError, match="unmeasured"):
        adm.preflight(refs, allow_unmeasured=False)
    rec = adm.preflight(refs, allow_unmeasured=True)
    assert "skillcorner__NEW_UNLISTED" in rec.unmeasured_admitted
    assert rec.digest == adm.digest


def test_a_pass_with_no_skillcorner_ref_needs_no_artifact(tmp_path):
    # events_only_loader over idsse-only refs: no artifact loaded, digest is None.
    load, rec = ea.events_only_loader([make_ref("idsse", "m1")], artifact_path=tmp_path / "does_not_exist.json")
    assert rec.digest is None and rec.unmeasured_admitted == ()
    assert callable(load)


# ---- Step 3: fail-closed load --------------------------------------------------------------------


def test_missing_artifact_refuses_when_skillcorner_requested(tmp_path):
    with pytest.raises(ea.AdmissionRefusedError):
        ea.events_only_loader([make_ref("skillcorner", "pass")], artifact_path=tmp_path / "nope.json")


def test_unprovenanced_or_dirty_artifact_refuses(tmp_path):
    no_commit = tmp_path / "a.json"
    no_commit.write_text(json.dumps({"matches": _MATCHES, "digest": "x"}), encoding="utf-8")
    with pytest.raises(ea.AdmissionRefusedError, match="unprovenanced"):
        ea.EventsOnlyAdmission.load(no_commit)
    dirty = _write_artifact(tmp_path / "b.json", _MATCHES, dirty=True)
    with pytest.raises(ea.AdmissionRefusedError, match="dirty"):
        ea.EventsOnlyAdmission.load(dirty)


def test_tampered_digest_refuses(tmp_path):
    bad = _write_artifact(tmp_path / "c.json", _MATCHES, digest="not-the-real-digest")
    with pytest.raises(ea.AdmissionRefusedError, match="digest"):
        ea.EventsOnlyAdmission.load(bad)


# ---- Step 4: events_only_loader wiring -----------------------------------------------------------


def test_events_only_loader_checks_before_loading(artifact, monkeypatch):
    order: list = []

    def _fake_load_match(ref, *, events_only, cache_dir=None, **kw):
        order.append(("load", ref.key, events_only))
        return make_loaded(ref.provider, ref.match_id)

    monkeypatch.setattr(ea, "load_match", _fake_load_match)
    load, rec = ea.events_only_loader([make_ref("skillcorner", "pass")], artifact_path=artifact)
    assert rec.digest is not None
    m = load(make_ref("skillcorner", "pass"))
    assert order == [("load", ("skillcorner", "pass"), True)] and m.match_id == "pass"

    # a non-admitted match: check raises BEFORE load_match runs (order unchanged)
    order.clear()
    with pytest.raises(lp.MatchExcluded):
        load(make_ref("skillcorner", "excl_bad"))
    assert order == [], "load_match ran despite a failed admission check"
