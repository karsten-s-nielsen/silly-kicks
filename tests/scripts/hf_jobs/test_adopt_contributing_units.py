"""TF-46 Task 6: adopt's fail-closed per-SHIPPED-ARM contributing-unit floor (spec SPEC-19, A PLAN-03).

The INTENDED membership comes from the arm policy (``--arms-json``); the ACTUAL from the refit's POST-DROP
``metrics.json``. The two come from DIFFERENT sources, so a dropped/emptied match (ACTUAL < INTENDED) is
caught -- a same-source compare would always pass. A missing count fails closed (no ``{}`` fallback)."""

from __future__ import annotations

import json

import pytest

from scripts.hf_jobs.adopt import (
    AdoptionRefusedError,
    _arms_json_from_args,
    _intended_units,
    check_contributing_units,
)


def _key(provider: str, game: str) -> str:
    """The in-job arm key the off-HF policy emits for a provider whose converter name equals its identity:
    ``{provider}__{provider}-{game}`` (the DFL converter != identity case is in ``test_arm_policy``)."""
    return f"{provider}__{provider}-{game}"


def _arms(**arms: list[str]) -> str:
    ship_order = [a for a in ("public", "sc_extended", "full") if a in arms]
    return json.dumps({"arms": arms, "ship_order": ship_order})


def test_arms_json_from_args_and_intended_units():
    # the wiring reads --arms-json off the instance args; intended is per-provider, keyed on the provider prefix
    args = ("--dataset", "d", "--arms-json", _arms(public=[_key("gradientsports", "1"), _key("gradientsports", "2")]))
    arms_json = _arms_json_from_args(args)
    assert arms_json is not None
    assert _intended_units(arms_json, "public") == {"gradientsports": 2}
    assert _arms_json_from_args(("--dataset", "d")) is None  # no arms-json -> nothing to enforce


def test_adopt_refuses_missing_metrics_json():
    # SPEC-19: no per-provider contributing_units key -> refuse (no {} fallback)
    arms = _arms(public=[_key("gradientsports", "1")])
    with pytest.raises(AdoptionRefusedError, match="contributing_units"):
        check_contributing_units(arms, "public", {"shipped_variant": "public"})


@pytest.mark.parametrize(
    ("shipped", "contributing", "match"),
    [
        # INTENDED public = 2 gs matches (the arm policy); ACTUAL = 1 (one dropped post-stoppage) -> refuse. A
        # same-source compare (ACTUAL from the same post-drop masks) would wrongly pass (A PLAN-03).
        ("public", 1, "below the INTENDED"),
        # SPEC-19 fail closed: a shipped_variant that names NO arm would give an empty INTENDED and pass open.
        ("sc_extended", 2, "names no arm"),
    ],
    ids=["actual_below_intended", "unknown_shipped_arm"],
)
def test_adopt_refuses(shipped, contributing, match):
    arms = _arms(public=[_key("gradientsports", "1"), _key("gradientsports", "2")])
    metrics = {"shipped_variant": shipped, "contributing_units": {"gradientsports": contributing}}
    with pytest.raises(AdoptionRefusedError, match=match):
        check_contributing_units(arms, shipped, metrics)


def test_adopt_accepts_public_shipped_xshot115():
    # B-02: an sc_extended-tier campaign whose fixed-sequence ship stopped at the PUBLIC arm is accepted against
    # the PUBLIC intended membership (not sc_extended's), because shipped_variant == "public".
    arms = _arms(
        public=[_key("gradientsports", "1"), _key("gradientsports", "2")],
        sc_extended=[_key("gradientsports", "1"), _key("gradientsports", "2"), _key("skillcorner", "9")],
    )
    metrics = {"shipped_variant": "public", "contributing_units": {"gradientsports": 2}}
    check_contributing_units(arms, "public", metrics)  # no raise: matches the public membership


def test_adopt_accepts_full_corpus():
    arms = _arms(
        public=[_key("gradientsports", "1")],
        full=[_key("gradientsports", "1"), _key("skillcorner", "9"), _key("sportec", "7")],
    )
    metrics = {
        "shipped_variant": "full",
        "contributing_units": {"gradientsports": 1, "skillcorner": 1, "sportec": 1},
    }
    check_contributing_units(arms, "full", metrics)  # no raise: ACTUAL == INTENDED for every provider


def test_verify_enforces_the_floor_wiring(monkeypatch, tmp_path):
    """Drive ``adopt.verify()`` with an arms-carrying instance whose metrics undershoot the INTENDED
    membership: verify() must refuse THROUGH the wiring (``adopt.py`` ``_arms_json_from_args`` +
    ``check_contributing_units``). Deleting those lines makes verify() skip the check and complete, so this
    test stops raising -- it guards the WIRING, not just the helper (B05). The heavy collaborators
    (instances/spec_for/artifacts_for/load_check/_certify) are stubbed so only the floor wiring runs."""
    import types

    from scripts.hf_jobs import adopt as ad

    inst = types.SimpleNamespace(
        id="inst1",
        job="ghost_gk",  # a LOADERS job NOT in INTEGRITY_FAMILIES (so _integrity is skipped)
        tier="public",
        variant="public",
        args=("--arms-json", _arms(public=[_key("gradientsports", "1"), _key("gradientsports", "2")])),
    )
    (tmp_path / "inst1").mkdir()
    (tmp_path / "inst1.record.json").write_text(json.dumps({"instance": "inst1"}), encoding="utf-8")
    (tmp_path / "inst1" / "metrics.json").write_text(
        json.dumps({"shipped_variant": "public", "contributing_units": {"gradientsports": 1}}), encoding="utf-8"
    )  # ACTUAL 1 < INTENDED 2
    fake_artifact = types.SimpleNamespace(kind="wheel", id="a1", job="ghost_gk", variant="public", tier="public")
    monkeypatch.setattr(ad, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr("scripts.hf_jobs.registry.spec_for", lambda job: types.SimpleNamespace())
    monkeypatch.setattr(ad, "artifacts_for", lambda inst, man: [fake_artifact])
    monkeypatch.setattr(ad, "load_check", lambda *a, **k: None)
    monkeypatch.setattr(ad, "_certify", lambda *a, **k: {"ok": True})
    c = types.SimpleNamespace(name="test-campaign", bucket="bucket", mode="rematerialize")
    with pytest.raises(AdoptionRefusedError, match="below the INTENDED"):
        ad.verify(c, "deadbeef", staging=tmp_path)  # pyright: ignore[reportArgumentType]  (c is a minimal fake)
