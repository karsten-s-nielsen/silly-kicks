"""Datasets: declared once, measured at staging into the lock, identity computed like the in-job trainer."""

from __future__ import annotations

import json

import pytest

from scripts import _corpus, _corpus_identity
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs.repackage import flat_to_pergame


def test_declared_kinds_access_and_visibility_are_valid():
    for name, d in D.declared().items():
        assert d.kind in D.KINDS and d.access in D.ACCESS and d.visibility in ("public", "restricted"), name
        assert (d.revision is not None) == (d.kind == "statsbomb_open"), name


def test_every_mount_declares_its_read_concurrency_and_no_other_access_does():
    """The dataset read gate (`scripts._dataset_reads`): a mounted dataset's job-wide whole-file read cap
    is declared beside its access; tc3 measured clean at 4 and failed at 8 (2026-10-20 probes)."""
    for name, d in D.declared().items():
        if d.access == "mount":
            assert isinstance(d.read_concurrency, int) and d.read_concurrency >= 1, name
        else:
            assert d.read_concurrency is None, name
    assert D.declared()["tc3"].access == "mount" and D.declared()["tc3"].read_concurrency == 4


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"access": "mount"}, "read_concurrency"),
        ({"access": "mount", "read_concurrency": 0}, "read_concurrency"),
        ({"access": "mount", "read_concurrency": "4"}, "read_concurrency"),
        ({"access": "download", "read_concurrency": 4}, "read_concurrency"),
    ],
)
def test_a_bad_read_concurrency_declaration_refuses(entry, match):
    raw = {"repo": "r/x", "kind": "flat_tc3", "visibility": "restricted", **entry}
    with pytest.raises(ValueError, match=match):
        D._declared_entry("x", raw)


def test_pergame_name_matches_the_repackage_tool(tmp_path):
    from scripts._toy_corpora import build_flat_corpus

    root = build_flat_corpus(tmp_path / "flat", n_games=2, providers=("skillcorner",))
    flat_to_pergame(root, tmp_path / "pg")
    stems = sorted(p.stem for p in (root / "shards").glob("*/*.parquet"))
    want = sorted(D.pergame_name(*s.split("__", 1)) for s in stems)
    assert sorted(p.name for p in (tmp_path / "pg").iterdir()) == want


def test_every_campaign_dataset_is_locked():
    lock = json.loads(D.LOCK.read_text(encoding="utf-8"))
    assert set(lock) == set(D.declared())
    for name, rec in lock.items():
        assert rec["revision"] and rec["kind"] == D.declared()[name].kind


def test_public_dataset_ids_agree_with_the_in_repo_registry():
    """Two independent sources (INV-9): the staging measurement and the PUBLIC_CORPUS registry."""
    assert D.locked("xshot17")["public_ids"] == _corpus.bundled_public_arm_pairs()


def test_expected_identity_uses_the_trainer_function():
    rec = D.locked("xshot115")
    units = rec["units"]
    providers = [u["source_provider"] for u in units]
    names = [D.pergame_name(u["stem"].split("__", 1)[0], u["game"]) for u in units]
    want = _corpus_identity.corpus_identity(providers, names, all_public=False)
    assert D.expected_identity("xshot115", "pergame") == want
    assert want["corpus_n_matches"] == 115


def test_record_refuses_a_kind_mismatch(tmp_path):
    class _Api:
        def dataset_info(self, repo, revision=None):
            return type("I", (), {"sha": "f" * 40})()

        def list_repo_files(self, repo, repo_type=None, revision=None):
            return ["foo/frames.parquet"]  # pergame layout, declared flat_tc3

    with pytest.raises(ValueError, match="declared kind"):
        D.record("tc3", api=_Api(), fs=None, lock_path=tmp_path / "lock.json")


# ---- the fixture-fidelity guard (ADR-128): toy ids must mirror the staged data's ids --------------------


def test_every_staged_dataset_records_its_providers_id_kinds():
    """Without the record the guard below could pass vacuously."""
    for name, d in D.declared().items():
        if d.kind == "statsbomb_open":
            continue
        kinds = D.locked(name).get("provider_id_kinds")
        assert kinds, f"{name}: run `python -m scripts.hf_jobs datasets id-kinds {name}`"


def test_the_toy_id_profile_matches_every_staged_provider():
    """Cycle 1's mixed-id defects reached real HF because every toy carried ONE id type. The toy profile
    (`_toy_corpora.PROVIDER_ID_KIND`) must equal what `datasets` measured in the staged data, provider by
    provider; a new provider or a changed dtype fails here until the toys follow."""
    from scripts._toy_corpora import PROVIDER_ID_KIND

    seen = {}
    for name, d in D.declared().items():
        if d.kind == "statsbomb_open":
            continue
        for provider, kind in D.locked(name)["provider_id_kinds"].items():
            assert provider in PROVIDER_ID_KIND, f"{name}: staged provider {provider!r} has no toy id profile"
            assert PROVIDER_ID_KIND[provider] == kind, (name, provider, kind, PROVIDER_ID_KIND[provider])
            seen[provider] = kind
    assert {"int", "str"} <= set(seen.values())  # non-vacuity: the staged data really mixes both


@pytest.mark.parametrize(
    ("arrow", "kind"),
    [
        ("int64", "int"),
        ("int32", "int"),
        ("double", "int"),
        ("string", "str"),
        ("large_string", "str"),
        ("dictionary<values=string, indices=int8, ordered=0>", "str"),
        ("null", None),
    ],
)
def test_an_arrow_id_type_maps_to_its_kind(arrow, kind):
    assert D._id_kind(arrow) == kind


def test_measure_id_kinds_reads_one_frames_and_one_actions_file_per_provider(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = {}

    def put(rel, table):
        p = tmp_path / rel.replace("/", "_")
        pq.write_table(table, p)
        files[rel] = p

    ints = pa.table({"game_id": pa.array([1]), "player_id": pa.array([2]), "team_id": pa.array([3.0])})
    strs = pa.table(
        {"game_id": pa.array(["a"]), "player_id": pa.array(["b"]), "team_id": pa.array(["c"]).dictionary_encode()}
    )
    for stem, t in (("gradientsports__1", ints), ("skillcorner__2", strs)):
        put(f"shards/g/{stem}.parquet", t)
        put(f"_actions/{stem}.parquet", t)

    class _Fs:
        def open(self, path, mode="rb"):
            return open(files[path.split("@rev/", 1)[1]], mode)

    rec = {"kind": "flat_tc3", "generation": "g", "units": [{"stem": "gradientsports__1"}, {"stem": "skillcorner__2"}]}
    assert D.measure_id_kinds("x", rec, repo="r/x", revision="rev", fs=_Fs()) == {
        "gradientsports": "int",
        "skillcorner": "str",
    }
