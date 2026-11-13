"""Corpus widening spec 6.1 / 6.2 (ADR-132): every dataset declares `raw` + `provenance`; `datasets record` checks
the provenance against the registries and records `public_ids` for the public-arm units; every repo this repo
STAGES is a private Hub repo (record and the campaign mount preflight refuse otherwise, fail-closed)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from scripts.hf_jobs import arm_policy
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs import preflight as P

_HEAD = """[datasets.{name}]
repo = "silly-kicks/{repo}"
kind = "flat_tc3"
access = "mount"
read_concurrency = 4
"""


def _toml(tmp_path, body: str) -> pathlib.Path:
    p = tmp_path / "datasets.toml"
    p.write_text(body, encoding="utf-8")
    return p


def _fixture(tmp_path, monkeypatch, entries: dict[str, tuple[str, str, list[str]]]):
    """Fixture datasets ``name -> (raw, provenance, stems)``; staged flat shards; providers from the stem."""
    body = ""
    stems_of = {}
    for name, (raw, prov, stems) in entries.items():
        body += _HEAD.format(name=name, repo=f"fx-{name}") + f'raw = "{raw}"\nprovenance = "{prov}"\n\n'
        stems_of[f"silly-kicks/fx-{name}"] = stems
    monkeypatch.setattr(D, "DECLARED", _toml(tmp_path, body))
    conv = {"idsse": "sportec"}
    monkeypatch.setattr(
        D,
        "_source_provider",
        lambda fs, path: conv.get(path.split("/")[-1].split("__")[0], path.split("/")[-1].split("__")[0]),
    )
    monkeypatch.setattr(D, "measure_id_kinds", lambda *a, **k: {})
    return stems_of


class _Api:
    def __init__(self, stems_of, private: Any = True, fail=False):
        self.stems_of, self.private, self.fail = stems_of, private, fail

    def dataset_info(self, repo, revision=None, **kw):
        if self.fail:
            raise ConnectionError("network down")
        private = self.private(repo) if callable(self.private) else self.private
        return type("I", (), {"sha": "f" * 40, "private": private})()

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return ["home_teams.json"] + [f"shards/g/{s}.parquet" for s in self.stems_of.get(repo, [])]


_OBTAINABLE = ["skillcorner__1874553", "idsse__DFL-MAT-J03WMX", "gradientsports__3812"]
_PROBE = ["gradientsports__10502"]
_LICENSED = ["skillcorner__1021404"]


def test_fields_required(tmp_path, monkeypatch):
    base = _HEAD.format(name="a", repo="fx-a")
    for body in (base + 'raw = "restricted"\n', base + 'provenance = "public"\n'):
        monkeypatch.setattr(D, "DECLARED", _toml(tmp_path, body))
        with pytest.raises(ValueError, match=r"raw.*provenance|provenance.*raw"):
            D.declared()
    monkeypatch.setattr(
        D, "DECLARED", _toml(tmp_path, base + 'raw = "restricted"\nprovenance = "public"\nvisibility = "public"\n')
    )
    with pytest.raises(ValueError, match="visibility"):
        D.declared()
    monkeypatch.setattr(D, "DECLARED", _toml(tmp_path, base + 'raw = "open"\nprovenance = "public"\n'))
    with pytest.raises(ValueError, match="raw"):
        D.declared()


def test_committed_declarations_carry_the_spec_values():
    want = {
        "tc3": ("restricted", "extended"),
        "xshot17": ("redistributable", "public"),
        "xshot115": ("restricted", "extended"),
        "xshot187": ("restricted", "extended"),
        "xcross_probe": ("restricted", "public"),
        "sb360": ("restricted", "extended"),
        "sb_open": ("redistributable", "public"),
    }
    got = {n: (d.raw, d.provenance) for n, d in D.declared().items()}
    assert got == want


def test_receiver_mapping():
    import scripts.hf_jobs.campaign as C

    insts = C.instances(C.load_campaign(pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")), "0" * 40)
    recv = next(i for i in insts if i.job == "receiver_model")
    args = list(recv.args)
    assert args[args.index("--corpus-visibility") + 1] == "restricted"  # sb360
    assert C.corpus_visibility_arg(D.declared()["sb360"]) == "restricted"
    assert C.corpus_visibility_arg(D.declared()["sb_open"]) == "public"


def test_provenance_checked_at_record(tmp_path, monkeypatch):
    stems_of = _fixture(
        tmp_path,
        monkeypatch,
        {
            "ok": ("restricted", "public", _OBTAINABLE + _PROBE),
            "bad": ("restricted", "public", _OBTAINABLE + _LICENSED),
            "ext": ("restricted", "extended", _OBTAINABLE + _LICENSED),
            "under": ("restricted", "extended", _OBTAINABLE),
        },
    )
    api = _Api(stems_of)
    lock = tmp_path / "lock.json"
    D.record("ok", policy=arm_policy, api=api, fs=object(), lock_path=lock)
    D.record("ext", policy=arm_policy, api=api, fs=object(), lock_path=lock)
    with pytest.raises(ValueError, match="provenance"):
        D.record("bad", policy=arm_policy, api=api, fs=object(), lock_path=lock)
    with pytest.raises(ValueError, match="provenance"):
        D.record("under", policy=arm_policy, api=api, fs=object(), lock_path=lock)
    assert set(json.loads(lock.read_text(encoding="utf-8"))) == {"ok", "ext"}


def test_public_ids_recorded_for_public_arm_units(tmp_path, monkeypatch):
    stems_of = _fixture(tmp_path, monkeypatch, {"ext": ("restricted", "extended", _OBTAINABLE + _PROBE + _LICENSED)})
    committed = D.LOCK.read_bytes()
    entry = D.record("ext", policy=arm_policy, api=_Api(stems_of), fs=object(), lock_path=tmp_path / "lock.json")
    assert entry["public_ids"] == sorted(
        [["gradientsports", "3812"], ["idsse", "DFL-MAT-J03WMX"], ["skillcorner", "1874553"]]
    )
    assert D.LOCK.read_bytes() == committed  # the committed lock is not re-recorded


def test_public_ids_absent_without_public_arm_units(tmp_path, monkeypatch):
    stems_of = _fixture(tmp_path, monkeypatch, {"probe": ("restricted", "public", _PROBE)})
    entry = D.record("probe", policy=arm_policy, api=_Api(stems_of), fs=object(), lock_path=tmp_path / "lock.json")
    assert entry["public_ids"] is None


def test_private_repo_required(tmp_path, monkeypatch):
    stems_of = _fixture(tmp_path, monkeypatch, {"ok": ("restricted", "public", _OBTAINABLE)})
    with pytest.raises(ValueError, match="not a private"):
        D.record(
            "ok", policy=arm_policy, api=_Api(stems_of, private=False), fs=object(), lock_path=tmp_path / "lock.json"
        )
    with pytest.raises(ValueError, match="not a private"):
        D.require_private("silly-kicks/x", api=_Api({}, private=None))  # unknown visibility is not private
    D.require_private("silly-kicks/x", api=_Api({}, private=True))


def test_staged_repos_exempt_only_the_upstream_open_source():
    sb_open = D.declared()["sb_open"]
    assert D.staged_repos(sb_open) == ["silly-kicks/hf-refit-sb-open"]  # the upstream statsbomb/open-data is exempt
    assert D.staged_repos(D.declared()["tc3"]) == ["silly-kicks/hf-refit-tc3"]


def test_private_check_fails_closed_on_api_error():
    with pytest.raises(ValueError, match="network down"):
        D.require_private("silly-kicks/x", api=_Api({}, fail=True))


def test_campaign_preflight_refuses_public_staged_repo():
    class _PubApi(_Api):
        def list_repo_files(self, repo, repo_type=None, revision=None):
            raise AssertionError("the private check runs first")

    c = P.check_datasets(
        ["xshot17", "sb_open"], api=_PubApi({}, private=lambda repo: repo != "silly-kicks/hf-refit-xshot17")
    )
    assert not c.ok and "xshot17" in c.detail and "not a private" in c.detail
    report = P.assemble_report("fx", "0" * 40, "h", [c])
    assert not report["ok"]


def test_statsbomb_datasets_keep_their_declared_provenance():
    # sb360 is kind flat_tc3 but outside the tracking registries: no provenance check, no public_ids (spec 6.2).
    # The unit shape is the locked one (source_provider "snapshot": the SB360 freeze-frame shards).
    units = [{"stem": "statsbomb__3986784", "game": "3986784", "source_provider": "snapshot"}]
    arm_policy.check_provenance("sb360", D.declared()["sb360"], units)
    assert arm_policy.public_ids(units) is None
    with pytest.raises(ValueError, match="mixes"):
        arm_policy.public_ids(
            [*units, {"stem": "skillcorner__1874553", "game": "1874553", "source_provider": "skillcorner"}]
        )


def test_an_unregistered_provider_refuses_the_provenance_check():
    # Review B CW-IB-03 (spec 3, fail-closed): a provider that is neither in the tracking registries nor a known
    # non-tracking source cannot be classified, so `datasets record` refuses it instead of keeping its declared value.
    units = [{"stem": "metrica__1", "game": "1", "source_provider": "metrica"}]
    with pytest.raises(ValueError, match="cannot classify"):
        arm_policy.check_provenance("mx", D.declared()["sb360"], units)
    with pytest.raises(ValueError, match="cannot classify"):
        arm_policy.public_ids(units)


def test_every_locked_dataset_still_classifies():
    # Anti-rot for the refusal above: the real staged units of every locked dataset classify without error.
    lock = json.loads(D.LOCK.read_text(encoding="utf-8"))
    staged = {name: e["units"] for name, e in lock.items() if e.get("units")}
    assert {"sb360", "tc3", "xcross_probe", "xshot115", "xshot17"} <= set(staged)
    for name, units in staged.items():
        arm_policy.check_provenance(name, D.declared()[name], units)
        arm_policy.public_ids(units)


def test_the_record_cli_injects_the_launcher_policy(monkeypatch, capsys):
    # datasets.py is in the in-job import closure and names no taxonomy symbol; `datasets record` routes through
    # the hf_jobs entry point, which injects arm_policy (ADR-123 Amendment B / ADR-132).
    from scripts.hf_jobs import __main__ as entry

    seen = {}

    def fake_record(name, *, policy, **kw):
        seen.update(name=name, policy=policy)
        return {"repo": "r", "units": []}

    monkeypatch.setattr(D, "record", fake_record)
    monkeypatch.setattr("scripts.hf_jobs.hub_client.install_timeouts", lambda: None)
    assert entry.main(["datasets", "record", "xshot17"]) == 0
    assert seen == {"name": "xshot17", "policy": arm_policy}
    assert '"repo": "r"' in capsys.readouterr().out
    with pytest.raises(TypeError):
        D.record("xshot17")  # pyright: ignore[reportCallIssue]  -- no default policy: a caller must inject it
