"""Adopt's public-tier head-to-head must hand the eval its corpus (A-O r6, out-of-delta finding).

xshot/xcross certify their PUBLIC tier with the head-to-head CLI, but their eval reads the staged corpus, never
public open data. Without ``--corpus-dir`` the eval raises ``EvalDataUnavailableError`` and certification can
never pass, so the refit could not be adopted. The corpus is the instance's own locked ``--dataset``."""

from __future__ import annotations

import types

import pytest

from scripts.hf_jobs import adopt as ad
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs import manifest as mf
from scripts.hf_jobs.campaign import Instance

_REV = "c" * 40


def _inst(job: str, args: tuple[str, ...]) -> Instance:
    return Instance(job=job, tier="public", variant="default", module=f"m.{job}", args=args, datasets=(), run_key="rk")


def test_fetch_locked_downloads_the_pinned_revision_once(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "locked", lambda name: {"repo": "silly-kicks/x", "revision": _REV})
    calls = []

    def download(repo, *, repo_type, revision, local_dir):
        calls.append((repo, repo_type, revision, local_dir))

    dest = tmp_path / "_eval" / "xshot17"
    dest.mkdir(parents=True)
    assert D.fetch_locked("xshot17", dest=dest, download=download) == dest
    assert calls == [("silly-kicks/x", "dataset", _REV, str(dest))]  # the PINNED revision, never the head
    D.fetch_locked("xshot17", dest=dest, download=download)
    assert len(calls) == 1  # the same revision is already there: no second download


@pytest.mark.parametrize("job", ["xshot_occurrence", "xcross_attempt"])
def test_eval_corpus_args_fetch_the_instance_dataset(monkeypatch, tmp_path, job):
    fetched = {}

    def fake_fetch(name, *, dest, download=None):
        fetched[name] = dest
        return dest

    monkeypatch.setattr(D, "fetch_locked", fake_fetch)
    args = ad._eval_corpus_args(_inst(job, ("--dataset", "xshot17", "--n-trials", "2")), tmp_path)
    assert args == ("--corpus-dir", str(tmp_path / "_eval" / "xshot17"))
    assert fetched == {"xshot17": tmp_path / "_eval" / "xshot17"}


def test_eval_corpus_args_are_empty_for_a_public_open_data_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "fetch_locked", lambda *a, **k: pytest.fail("a public open-data eval fetches no corpus"))
    assert ad._eval_corpus_args(_inst("win_probability", ("--all-competitions",)), tmp_path) == ()


def test_eval_corpus_args_refuse_an_instance_without_a_dataset(tmp_path):
    with pytest.raises(ad.AdoptionRefusedError, match="no --dataset"):
        ad._eval_corpus_args(_inst("xshot_occurrence", ("--n-trials", "2")), tmp_path)


def test_certify_passes_the_corpus_dir_to_the_headtohead(monkeypatch, tmp_path):
    # The wiring: _certify's head-to-head argv carries --corpus-dir for a public-tier xshot instance. Dropping it
    # from the argv (the pre-fix state) makes this go red.
    from scripts.hf_jobs.registry import get

    monkeypatch.setattr(D, "fetch_locked", lambda name, *, dest, download=None: dest)
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=1, stderr="stop here")

    fresh = tmp_path / "rk" / "default"
    fresh.mkdir(parents=True)
    ref = mf.Artifact(
        id="a1",
        destination="wheel:silly_kicks/tracking/_xshot_weights/default",
        job="xshot_occurrence",
        variant="default",
        tier="public",
        certification="public",
    )
    inst = _inst("xshot_occurrence", ("--dataset", "xshot17"))
    out = ad._certify(get("xshot_occurrence"), inst, ref, fresh, {}, {}, run=run)
    assert out["ok"] is False  # the stubbed CLI failed; only the argv is under test
    argv = seen["argv"]
    assert argv[argv.index("--corpus-dir") + 1] == str(tmp_path / "rk" / "_eval" / "xshot17")
