"""The pinned open-data snapshot (HFU spec 14.7, Change 7f; launch-3 findings F14, F15).

A job reads StatsBomb open data from a staged, read-only snapshot of the pinned GitHub commit: the consumer
cache, filled by the same fetch functions. A cache hit returns what a fetch returns; an in-job miss raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import _sb_open_data as O
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs import launch as L

REV = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SNAP = "f" * 40


@pytest.fixture
def toy_server(tmp_path, monkeypatch):
    """The toy open-data tree over local http (``SILLY_KICKS_OPEN_DATA_BASE``), plus one 360 file."""
    pytest.importorskip("statsbombpy")
    from statsbombpy import config  # type: ignore[import-not-found]

    from scripts._toy_corpora import build_open_data, serve_open_data

    root = build_open_data(tmp_path / "od")
    (root / "data" / "three-sixty").mkdir()
    frames = [{"event_uuid": "e1", "visible_area": [0, 0, 1, 1], "freeze_frame": [{"teammate": True}]}]
    (root / "data" / "three-sixty" / "7298.json").write_text(json.dumps(frames), encoding="utf-8")
    manifest = root / "data" / "matches" / "43" / "106.json"
    matches = json.loads(manifest.read_text(encoding="utf-8"))
    for m in matches:  # the manifest field that says a match has 360 frames
        m["match_status_360"] = "available" if m["match_id"] == 7298 else "unscheduled"
    manifest.write_text(json.dumps(matches), encoding="utf-8")
    saved = dict(config.OPEN_DATA_PATHS)
    with serve_open_data(root) as url:
        monkeypatch.setenv(O.OPEN_DATA_BASE_ENV, url)
        O.pin_open_data_revision(REV)
        yield root
    config.OPEN_DATA_PATHS.clear()
    config.OPEN_DATA_PATHS.update(saved)


def _kinds():
    from statsbombpy import sb  # type: ignore[import-not-found]

    return {
        "competitions": (lambda r: O._open_competitions(r), lambda: O._values(sb.competitions(fmt="dict"))),
        "matches": (
            lambda r: O._open_matches(43, 106, r),
            lambda: sb.matches(competition_id=43, season_id=106, fmt="dict"),
        ),
        "lineups": (lambda r: O.open_lineups("7298", r), lambda: sb.lineups(match_id=7298, fmt="dict")),
        "events": (lambda r: O._open_events("7298", r), lambda: O._values(sb.events(match_id=7298, fmt="dict"))),
        "frames": (lambda r: O._open_frames("7298", r), lambda: O._values(sb.frames(match_id=7298, fmt="dict"))),
    }


@pytest.mark.parametrize("kind", ["competitions", "matches", "lineups", "events", "frames"])
def test_open_data_cache_hit_equals_fetch(kind, toy_server, tmp_path, monkeypatch):
    cached, fetch = _kinds()[kind]
    want = fetch()
    cache = tmp_path / "cache"
    assert cached(cache) == want  # a miss fetches and writes the cache
    monkeypatch.setenv(O.OPEN_DATA_BASE_ENV, "http://127.0.0.1:9")  # nothing listens: a hit must not fetch
    O.pin_open_data_revision(REV)
    monkeypatch.setenv("SILLY_KICKS_IN_HF_JOB", "1")  # and the in-job rule does not fire on a hit
    assert cached(cache) == want


def test_snapshot_miss_raises_only_in_job(tmp_path, monkeypatch):
    calls: list[int] = []

    def fetch():
        calls.append(1)
        return [{"a": 1}]

    monkeypatch.setenv("SILLY_KICKS_IN_HF_JOB", "1")
    with pytest.raises(O.OpenDataSnapshotMissError, match=r"99\.json"):
        O._cached_json(tmp_path, "statsbomb_open/99.json", fetch)
    assert calls == []  # never falls back to GitHub
    monkeypatch.delenv("SILLY_KICKS_IN_HF_JOB")
    assert O._cached_json(tmp_path, "statsbomb_open/99.json", fetch) == [{"a": 1}] and calls == [1]
    assert O._cached_json(None, "statsbomb_open/99.json", fetch) == [{"a": 1}] and calls == [1, 1]


def test_verify_cache_fills_from_the_snapshot(tmp_path, monkeypatch):
    """Off-HF with the snapshot named (verify's public builders): a miss downloads that file from the
    snapshot at its revision, never from GitHub."""
    import huggingface_hub

    asked: list[tuple] = []

    def fake_download(repo_id, filename, *, repo_type, revision, local_dir):
        asked.append((repo_id, filename, repo_type, revision))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([{"b": 2}]), encoding="utf-8")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv(O.OPEN_DATA_SNAPSHOT_ENV, f"silly-kicks/hf-refit-sb-open@{SNAP}")
    got = O._cached_json(tmp_path, "statsbomb_open/7.json", lambda: pytest.fail("fetched from GitHub"))
    assert got == [{"b": 2}]
    assert asked == [("silly-kicks/hf-refit-sb-open", "statsbomb_open/7.json", "dataset", SNAP)]


def _sb_open_mount(**kw) -> L.DatasetMount:
    base = dict(name="sb_open", repo="statsbomb/open-data", revision=REV, access="open")
    return L.DatasetMount(**{**base, **kw})


def _spec(*datasets: L.DatasetMount) -> L.JobSpec:
    return L.JobSpec(
        campaign="c", run_key="c-abcdef1-xsuccess-public", job="xsuccess", variant="default", stages=("run",),
        shard=(0, 1), module="m", args=(), commit="a" * 40, bucket="b/b", workers=1, inner_workers=1,
        cap_bytes=None, peak_rss_bytes=None, timeout_s=60, attempts=1, datasets=datasets, extras=(),
    )  # fmt: skip


def test_job_env_mounts_snapshot():
    from scripts._dataset_reads import READ_CAP_ENV

    d = _sb_open_mount(snapshot_repo="silly-kicks/hf-refit-sb-open", snapshot_revision=SNAP, read_concurrency=4)
    env = L.job_env(_spec(d))
    assert env[O.OPEN_DATA_REVISION_ENV] == REV
    assert env["SILLY_KICKS_CORPUS_CACHE_DIR"] == d.path and env[READ_CAP_ENV] == "4"
    vols = [v for v in L.volumes_for(_spec(d)) if v.type == "dataset"]
    assert [(v.source, v.mount_path, v.revision, v.read_only) for v in vols] == [
        ("silly-kicks/hf-refit-sb-open", d.path, SNAP, True)
    ]
    bare = L.job_env(_spec(_sb_open_mount()))
    assert "SILLY_KICKS_CORPUS_CACHE_DIR" not in bare and READ_CAP_ENV not in bare


def test_a_snapshot_mount_needs_its_read_cap():
    with pytest.raises(ValueError, match="read_concurrency"):
        _sb_open_mount(snapshot_repo="silly-kicks/hf-refit-sb-open", snapshot_revision=SNAP)


def test_sb_open_read_concurrency_allowed_with_snapshot():
    base = {"repo": "statsbomb/open-data", "kind": "statsbomb_open", "visibility": "public", "access": "open"}
    d = D._declared_entry("sb_open", {**base, "snapshot_repo": "silly-kicks/hf-refit-sb-open", "read_concurrency": 4})
    assert d.snapshot_repo == "silly-kicks/hf-refit-sb-open" and d.read_concurrency == 4
    with pytest.raises(ValueError, match="read_concurrency"):
        D._declared_entry("sb_open", {**base, "read_concurrency": 4})
    with pytest.raises(ValueError, match="read_concurrency"):
        D._declared_entry("sb_open", {**base, "snapshot_repo": "silly-kicks/hf-refit-sb-open"})


class _Api:
    def __init__(self, sha):
        self.sha = sha

    def dataset_info(self, repo, revision=None):
        import types

        return types.SimpleNamespace(sha=self.sha)


def _record_sb_open(tmp_path, monkeypatch, source_revision: str):
    import huggingface_hub

    monkeypatch.setattr(O, "pin_open_data_revision", lambda rev=None: rev)
    monkeypatch.setattr(O, "all_open_competitions", lambda: [(43, 106)])
    monkeypatch.setattr(O, "list_open_data_refs", lambda comps: [O.OpenDataRef(43, 106, "1", 10)])

    def fake_download(repo_id, filename, *, repo_type, revision):
        p = tmp_path / "snap" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"source_revision": source_revision, "n_files": 5, "bytes": 123}), encoding="utf-8")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    lock = tmp_path / "lock.json"
    return D.record("sb_open", api=_Api(SNAP), now="2026-11-07T00:00:00+00:00", lock_path=lock)


def test_record_writes_the_snapshot(tmp_path, monkeypatch):
    rec = _record_sb_open(tmp_path, monkeypatch, REV)
    assert rec["snapshot"] == {"revision": SNAP, "source_revision": REV, "n_files": 5, "bytes": 123, "unreadable": []}


def test_record_refuses_foreign_snapshot(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="source_revision"):
        _record_sb_open(tmp_path, monkeypatch, "0" * 40)


def test_verify_fails_wrong_snapshot_revision(tmp_path, monkeypatch):
    from scripts.hf_jobs import adopt as A
    from scripts.hf_jobs import manifest as M

    real, orig = D.locked("sb_open"), D.locked
    monkeypatch.setattr(
        D,
        "locked",
        lambda name, **kw: {**real, "snapshot": {"revision": SNAP}} if name == "sb_open" else orig(name, **kw),
    )
    fresh = tmp_path / "f"
    fresh.mkdir()
    (fresh / "metadata.json").write_text(json.dumps({"training_commit": "c" * 40}), encoding="utf-8")
    (fresh / "metrics.json").write_text(json.dumps(real["identity"]["open"]), encoding="utf-8")

    def record(snapshot):
        return {"token_inputs": {"run": {"dataset": {"name": "sb_open", "revision": REV, "snapshot": snapshot}}}}

    a = M.load().artifacts["xsuccess"]
    assert A.expectation_violations(a, fresh, record(SNAP), "c" * 40) == []
    bad = A.expectation_violations(a, fresh, record("e" * 40), "c" * 40)
    assert any("snapshot" in v for v in bad), bad


def test_stage_open_data_layout(toy_server, tmp_path):
    from scripts.hf_jobs.stage import stage_open_data

    out = tmp_path / "snap"
    got = stage_open_data(REV, out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    listed = json.loads((toy_server / "data" / "matches" / "43" / "106.json").read_text(encoding="utf-8"))
    ids = [str(m["match_id"]) for m in listed]
    assert len(ids) >= 3  # non-vacuous: several matches, one of them with 360 frames
    want = ["_snapshot.json", "statsbomb_open/competitions.json", "statsbomb_open/matches/43_106.json"]
    want += [f"statsbomb_open/{i}.json" for i in ids] + [f"statsbomb_open/lineups/{i}.json" for i in ids]
    want += ["statsbomb_open/7298.360.json"]  # only the match the manifest marks 360-available
    assert files == sorted(want)
    meta = json.loads((out / "_snapshot.json").read_text(encoding="utf-8"))
    assert got == meta and meta["source_revision"] == REV and meta["n_files"] == len(files) - 1
    assert meta["bytes"] == sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and p.name != "_snapshot.json")


def test_public_builders_read_the_snapshot(tmp_path, monkeypatch):
    """Verify's public head-to-head reads the evaluation matches from the snapshot at the locked revision,
    into the staging folder, with the GitHub pin set (HFU spec 14.7)."""
    from scripts.hf_jobs import adopt as A
    from scripts.hf_jobs import campaign as Camp
    from scripts.hf_jobs import manifest as M
    from scripts.hf_jobs.registry import get
    from tests.scripts.hf_jobs.test_adopt import _Run

    real, orig = D.locked("sb_open"), D.locked
    monkeypatch.setattr(
        D,
        "locked",
        lambda name, **kw: {**real, "snapshot": {"revision": SNAP}} if name == "sb_open" else orig(name, **kw),
    )
    c = Camp.load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    inst = next(i for i in Camp.instances(c, "c" * 40) if i.job == "win_probability")
    staging = tmp_path / "s"
    fresh = staging / inst.id
    fresh.mkdir(parents=True)
    ref = M.load().artifacts["win_probability"]
    run = _Run({"committed_ece": 0.5, "fresh_ece": 0.5})
    A._certify(get("win_probability"), inst, ref, fresh, A._committed_metrics(ref, {}), {}, run=run)
    _argv, env = run.calls[0]
    assert env[O.OPEN_DATA_REVISION_ENV] == real["revision"]
    assert env[O.OPEN_DATA_SNAPSHOT_ENV] == f"silly-kicks/hf-refit-sb-open@{SNAP}"
    assert env["SILLY_KICKS_CORPUS_CACHE_DIR"] == str(staging / "_open_data" / inst.id)  # one per instance (HFU 17.5)


def _fake_open_data(monkeypatch, fail_events: int):
    """Stage over a 2-match fake corpus whose events fetch fails ``fail_events`` times first (a truncated read).
    Staging runs off-HF in the hfjobs environment, where statsbombpy brings ``requests``."""
    requests = pytest.importorskip("requests")

    calls = {"events": 0}

    def events(match_id, root):
        calls["events"] += 1
        if calls["events"] <= fail_events:
            raise requests.exceptions.JSONDecodeError("Expecting ',' delimiter", "{", 1)
        (root / "statsbomb_open").mkdir(parents=True, exist_ok=True)
        (root / "statsbomb_open" / f"{match_id}.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(O, "pin_open_data_revision", lambda rev=None: rev)
    monkeypatch.setattr(O, "assert_statsbomb_open_data_mode", lambda: None)
    monkeypatch.setattr(O, "_open_competitions", lambda root: [{"competition_id": 1, "season_id": 2}])
    monkeypatch.setattr(O, "_open_matches", lambda c, s, root: {1: {}, 2: {}})
    monkeypatch.setattr(O, "_open_events", events)
    monkeypatch.setattr(O, "open_lineups", lambda match_id, root: None)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


def test_stage_open_data_retries_a_transient_fetch(tmp_path, monkeypatch):
    from scripts.hf_jobs.stage import stage_open_data

    calls = _fake_open_data(monkeypatch, fail_events=2)
    meta = stage_open_data(REV, tmp_path / "snap", workers=1)
    assert calls["events"] == 4 and meta["n_files"] == 2  # match 1 failed twice, then both matches staged


def test_stage_open_data_records_an_upstream_corrupt_file(tmp_path, monkeypatch):
    """An upstream file that never parses (StatsBomb's three-sixty/3845506.json at the pin holds NUL bytes) is
    recorded ``unreadable`` and staging goes on: a direct fetch of it fails too, so a job miss is faithful."""
    from scripts.hf_jobs.stage import stage_open_data

    _fake_open_data(monkeypatch, fail_events=3)  # match 1's events fail all 3 attempts; match 2 stages
    meta = stage_open_data(REV, tmp_path / "snap", workers=1)
    assert list(meta["unreadable"]) == ["statsbomb_open/1.json"] and meta["n_files"] == 1
    assert "Expecting ',' delimiter" in meta["unreadable"]["statsbomb_open/1.json"]


def test_stage_open_data_stops_on_a_network_failure(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")

    from scripts.hf_jobs.stage import stage_open_data

    _fake_open_data(monkeypatch, fail_events=0)
    monkeypatch.setattr(
        O, "open_lineups", lambda match_id, root: (_ for _ in ()).throw(requests.exceptions.ConnectionError("down"))
    )
    with pytest.raises(requests.exceptions.ConnectionError):
        stage_open_data(REV, tmp_path / "snap", workers=1)
