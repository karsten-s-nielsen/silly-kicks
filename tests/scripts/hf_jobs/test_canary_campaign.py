"""HFU rung 5 (spec `docs/superpowers/specs/2026-10-31-hfu-rung5-canary-design.md`): the canary campaign, its
forced-halt arm on the toy pipeline (canary-only), the `--reports` record of a watch, the scripted `canary-check`,
and the owner ruling of 2026-10-31 that the ERROR-in-a-row stop is an instance halt that cancels the instance's
live jobs, like the failed-units halt."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import types

import pytest

from scripts.hf_jobs import campaign as C

SHA = "abcdef1" + "0" * 33
CANARY_TOML = pathlib.Path("scripts/hf_jobs/campaigns/canary.toml")
CYCLE1_TOML = pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")
REPO = pathlib.Path(__file__).resolve().parents[3]
HALT_RK = "canary-abcdef1-canary_halt-none"


# ---- C3: the halt arm's registry entry and the canary-only guards --------------------------------------------------


def test_the_halt_arm_is_a_canary_spec_on_the_toy_pipeline():
    from scripts.hf_jobs.registry import CanarySpec, canary_spec

    spec = canary_spec("canary_halt")
    assert isinstance(spec, CanarySpec)
    assert spec.module == "tests.scripts._toy_pipeline"
    assert spec.base_args == ("--n-items", "16", "--fail-keys", "0000", "--hold-s", "3600")
    assert canary_spec("ghost_gk") is None  # a real job is never a canary spec


def _toml(tmp_path: pathlib.Path, name: str, jobs: list[tuple[str, str, str]]) -> pathlib.Path:
    body = (
        f'[campaign]\nname = "{name}"\nbucket = "b/x"\ntarget_minutes = 30\ndispatch_cap = 25\nshard_cap = 20\n'
        "attempts = 3\n"
    )
    for job, tier, dataset in jobs:
        body += f'\n[[jobs]]\njob = "{job}"\ntier = "{tier}"\ndataset = "{dataset}"\nvariants = ["default"]\n'
    path = tmp_path / f"{name}.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_canary_job_is_refused_outside_a_canary_campaign(tmp_path):
    with pytest.raises(ValueError, match="canary"):
        C.load_campaign(_toml(tmp_path, "cycle9", [("canary_halt", "canary", "none")]))
    assert C.load_campaign(_toml(tmp_path, "canary-x", [("canary_halt", "canary", "none")])).name == "canary-x"


def test_a_canary_instance_runs_the_toy_with_no_dataset(tmp_path):
    c = C.load_campaign(_toml(tmp_path, "canary", [("canary_halt", "canary", "none")]))
    (inst,) = C.instances(c, SHA)
    assert inst.module == "tests.scripts._toy_pipeline"
    assert inst.args == ("--n-items", "16", "--fail-keys", "0000", "--hold-s", "3600")
    assert inst.datasets == () and inst.run_key == HALT_RK


@pytest.mark.parametrize("command", ["retrieve", "verify", "adopt", "publish"])
def test_a_canary_campaign_never_retrieves_verifies_adopts_or_publishes(command, monkeypatch):
    monkeypatch.setattr(C, "_head", lambda: SHA)
    monkeypatch.setattr(C, "_tree_identity", lambda: "clean")
    with pytest.raises(SystemExit, match="canary"):
        C.main([str(CANARY_TOML), command, "--staging", str(REPO.parent / "_hf_staging" / "canary")])


def test_the_toy_runs_as_a_module_from_the_repository_root(tmp_path):
    """Review R5-SPEC-03: the halt arm's module must import as the HF job runs it (`python -m` from the checkout
    root), or the arm would halt on an import error instead of on `strict`."""
    argv = [sys.executable, "-m", "tests.scripts._toy_pipeline", "--n-items", "16", "--fail-keys", "0000"]
    argv += ["--hold-s", "3600", "--plan", "--store", str(tmp_path / "store")]
    proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, check=False)  # noqa: S603 -- fixed argv
    assert proc.returncode == 0, proc.stderr[-2000:]
    stages = {s["name"] for s in json.loads(proc.stdout)["stages"]}
    assert {"strict", "hold"} <= stages


# ---- C1, C2: the committed campaign file (owner ruling 2026-10-31: 13 instances plus the halt arm) -----------------


def test_the_committed_canary_is_the_owner_ruled_cycle1_subset_plus_the_halt_arm():
    canary, cycle1 = C.load_campaign(CANARY_TOML), C.load_campaign(CYCLE1_TOML)
    assert canary.name == "canary" and canary.bucket == cycle1.bucket  # C1: the MEM history is per bucket
    for field in ("target_minutes", "dispatch_cap", "shard_cap", "attempts"):
        assert getattr(canary, field) == getattr(cycle1, field), field
    by_entry = {(j.job, j.tier, j.dataset, j.probe_dataset): set(j.variants) for j in cycle1.jobs}
    real = [j for j in canary.jobs if j.job != "canary_halt"]
    for j in real:  # C2: each entry is a cycle-1 computation (same job, tier, dataset, probe); its variants a subset
        assert set(j.variants) <= by_entry[(j.job, j.tier, j.dataset, j.probe_dataset)], j
    runs = {(j.job, j.tier, v) for j in real for v in j.variants}
    assert len(runs) == 13
    assert {("ghost_gk", "owner", "default"), ("ghost_gk", "owner", "position_only")} <= runs  # the variant pair
    for job in ("xshot_occurrence", "xcross_attempt"):  # two tiers of one module on one shared run key
        assert {(job, "public", "default"), (job, "sc_extended", "default")} <= runs
    assert {j.job for j in real} == {j.job for j in cycle1.jobs}  # every module
    (halt,) = [j for j in canary.jobs if j.job == "canary_halt"]
    assert halt.variants == ("default",) and halt.dataset == "none"


# ---- C4: the watch record ----------------------------------------------------------------------------------------


def test_watch_reports_record_every_step_and_the_settled_exit_code(tmp_path, monkeypatch):
    import huggingface_hub

    reps = iter(
        [
            {"i": {"action": "run", "stages": ["gen"], "jobs": ["j1"], "failed": None}},
            {"i": {"action": "halt", "stages": ["strict"], "jobs": [], "failed": "strict: halted"}},
        ]
    )
    monkeypatch.setattr(C, "_head", lambda: SHA)
    monkeypatch.setattr(C, "step", lambda *a, **k: next(reps))
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    out = tmp_path / "reports.jsonl"
    rc = C.main([str(CANARY_TOML), "resume", "--watch", "--reports", str(out)])
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert rc == 1
    assert [ln["report"]["i"]["action"] for ln in lines[:2]] == ["run", "halt"]
    assert lines[-1] == {"exit_code": 1}


def test_the_reports_path_is_refused_inside_the_repository(monkeypatch):
    monkeypatch.setattr(C, "_head", lambda: SHA)
    with pytest.raises(SystemExit, match="outside the repository"):
        C.main([str(CANARY_TOML), "resume", "--watch", "--reports", str(REPO / "canary-reports.jsonl")])


# ---- C3 + owner ruling: both HF behaviours halt the arm by name, cancel `hold`, and a settled watch exits 1 -------


def _stage(name: str, *, n: int, done: int, failed: dict | None = None, pending: list | None = None) -> dict:
    units = [f"{i:04d}" for i in range(n)]
    return {
        "name": name,
        "n_units": n,
        "done": done,
        "pending": len(pending) if pending is not None else n - done,
        "failed": len(failed or {}),
        "excluded": 0,
        "ready": True,
        "reduced": False,
        "inner_parallel": False,
        "inner_width": 1,
        "history_key": name,
        "units": units,
        "pending_units": pending if pending is not None else units[done:],
        "failed_units": dict(failed or {}),
        "max_failures": 0,
        "token": f"t_{name}",
        "needs": [],
    }


def _toy_job(stage: str, status: str, *, jid: str, created_at: int, message: str = "exit 3") -> types.SimpleNamespace:
    labels = {"run_key": HALT_RK, "tier": "canary", "variant": "default", "stage": stage, "gens": f"t_{stage}"[:8]}
    return types.SimpleNamespace(
        id=jid, labels=labels, status=types.SimpleNamespace(stage=status, message=message), created_at=created_at
    )


@pytest.mark.parametrize(
    ("hf_behaviour", "strict_failed", "strict_pending", "strict_jobs", "reason_fragment", "strict_message"),
    [
        # HF retries a failed job inside the job id (measured 2026-10-31): 0000 failed in all 3 in-job attempts
        ("in-job retries", {"0000": 3}, [], 1, "unit(s) failed 2 time(s), over the stage's max_failures=0", "exit 3"),
        # no in-job retries: one failure per job id; `next_actions` halts on the spent unit before any resume check
        (
            "no in-job retries",
            {"0000": 2},
            [],
            2,
            "unit(s) failed 2 time(s), over the stage's max_failures=0",
            "exit 3",
        ),
        # the REAL run-3 shape: a single crash before any unit writes a record (the strict profile job exits 3).
        # The non-resumable-ERROR instance halt stops it on the FIRST ERROR, named by stage (HFU rung 5).
        ("single crash, no record", {}, ["0000"], 1, "ended in ERROR, nothing to resume", "exit 3"),
        # two no-record errors in a row reach the ERROR-in-a-row bound. A crash PAIR cannot occur now (the first
        # crash halts), so this models repeated started-then-timed-out jobs: resumable, relaunched, errored twice.
        ("repeated timeout", {}, ["0000"], 2, "2 job(s) in a row ended in ERROR", "started, then Job timeout"),
    ],
)
def test_the_halt_arm_halts_named_cancels_hold_and_a_settled_watch_exits_1(
    monkeypatch, hf_behaviour, strict_failed, strict_pending, strict_jobs, reason_fragment, strict_message
):
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import preflight

    inst = C.Instance("canary_halt", "canary", "default", "tests.scripts._toy_pipeline", (), (), HALT_RK)
    c = C.Campaign(
        name="canary",
        bucket="b/x",
        target_minutes=30,
        dispatch_cap=25,
        shard_cap=20,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    plan = {
        "pipeline": "toy",
        "stages": [
            _stage("strict", n=6, done=5, failed=strict_failed, pending=strict_pending),
            _stage("hold", n=1, done=0),
        ],
    }
    jobs = [
        _toy_job("strict", "ERROR", jid=f"s{i}", created_at=i, message=strict_message)
        for i in range(1, strict_jobs + 1)
    ]
    jobs.append(_toy_job("hold", "RUNNING", jid="hold-job", created_at=0))
    cancelled: list[str] = []

    class _Api:
        def list_bucket_tree(self, bucket, prefix=None, *, recursive=None):
            return []

        def list_jobs(self, *, labels=None):
            return [j for j in jobs if all(j.labels.get(k) == v for k, v in (labels or {}).items() if k in j.labels)]

        def cancel_job(self, *, job_id):
            cancelled.append(job_id)

    monkeypatch.setattr(preflight, "require_preflight", lambda *a, **k: {})
    monkeypatch.setattr(C, "instances", lambda c, commit, replica=None: [inst])
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: plan)
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "live_history", lambda plan, store, cache: {})
    monkeypatch.setattr(C, "_tree_identity", lambda: "t")
    monkeypatch.setattr(C, "_job_specs", lambda c, commit, inst, plan, a, *, store, unit_peaks=None: ["spec"])
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: pytest.fail(f"{hf_behaviour}: nothing may launch"))
    monkeypatch.setattr(L, "jobs_for", lambda api, **k: jobs)
    rep = C.step(c, SHA, api=_Api(), token="hf_x")
    entry = rep[inst.id]
    assert entry["action"] == "halt", (hf_behaviour, entry)
    assert entry["failed"].startswith("strict: ") and reason_fragment in entry["failed"], entry["failed"]
    assert cancelled == ["hold-job"], hf_behaviour  # the sibling `hold` job is cancelled on BOTH paths
    assert C.canary_halt_reason(entry["failed"]), entry["failed"]  # the check's structural match accepts it
    settled = {inst.id: {"action": "halt", "stages": entry["stages"], "jobs": [], "failed": entry["failed"]}}
    assert C.all_settled(settled) and C.watch_exit_code(settled) == 1


def test_the_halt_reason_match_is_structural_and_names_strict():
    assert C.canary_halt_reason("strict: 1 unit(s) failed 2 time(s), over the stage's max_failures=0: {'0000': 3}")
    assert C.canary_halt_reason("strict: 2 job(s) in a row ended in ERROR")
    assert C.canary_halt_reason("strict: latest job s1 ended in ERROR, nothing to resume")  # non-resumable crash
    assert not C.canary_halt_reason("hold: 2 job(s) in a row ended in ERROR")  # another stage
    assert not C.canary_halt_reason("hold: latest job s1 ended in ERROR, nothing to resume")  # another stage
    assert not C.canary_halt_reason(
        "strict: zero progress -- its 2 most recent settled jobs (s1, s2) committed nothing (progress mark 5)"
    )  # a zero-progress halt
    assert not C.canary_halt_reason(None)


# ---- C4: the scripted check ----------------------------------------------------------------------------------------


def _passing_inputs() -> dict:
    halt_id = f"{HALT_RK}/canary/default"
    reports = [
        {"report": {"a/owner/default": {"action": "run", "jobs": ["j1"], "failed": None}}},
        {
            "report": {
                "a/owner/default": {"action": "done", "jobs": [], "failed": None},
                "canary-abcdef1-win_probability-sb_open/public/default": {"action": "done", "jobs": [], "failed": None},
                halt_id: {
                    "action": "halt",
                    "stages": ["strict"],
                    "jobs": [],
                    "failed": "strict: 1 unit(s) failed 2 time(s), over the stage's max_failures=0: {'0000': 3}",
                    "cancelled": ["hold-job"],
                },
            }
        },
        {"exit_code": 1},
    ]
    summary = {
        "stages": [
            {"instance": "a/owner/default", "stage": "extract", "audit": {"ok": True}, "n_units": 10},
            {"instance": "a/owner/default", "stage": "select", "audit": None, "n_units": 10},  # shared: audited once
            {
                "instance": "canary-abcdef1-win_probability-sb_open/public/default",
                "stage": "extract",
                "audit": {"ok": True},
                "n_units": 3961,
            },
        ]
    }
    return {
        "reports": reports,
        "summary": summary,
        "listing": ["canary-abcdef1-a-tc3/ghost_gk/extract/t/u.parquet"],
        "jobs": [{"id": "hold-job", "labels": {"run_key": HALT_RK, "stage": "hold"}, "status": "CANCELED"}],
        "halt_instance": halt_id,
        "wp_reduce": {
            "canary-abcdef1-win_probability-sb_open/public/default": {
                "n_units": 3961,
                "n_done": 3958,
                "n_excluded": 3,
                "n_failed": 0,
            }
        },
    }


def test_canary_check_passes_on_a_clean_run():
    out = C.canary_check(**_passing_inputs())
    assert out["ok"], out
    assert out["halt_reason_kind"] == "failed-units"


@pytest.mark.parametrize(
    ("mutate", "failing"),
    [
        (lambda d: d["reports"][1]["report"]["a/owner/default"].update(action="waiting"), "instances_done"),
        (lambda d: d["summary"]["stages"][0]["audit"].update(ok=False), "r1_audit"),
        (lambda d: d["reports"][0]["report"]["a/owner/default"].update(lost="extract: 3 lost"), "no_lost_output"),
        (lambda d: d["listing"].append("canary-abcdef1-a-tc3/ghost_gk/extract/t/.tmp-u.parquet"), "no_orphan_temp"),
        (
            lambda d: d["reports"][1]["report"]["a/owner/default"].update(
                action="halt", failed="extract: zero progress -- its 2 most recent settled jobs (j1, j2) committed"
            ),
            "no_zero_progress_halt",
        ),
        (lambda d: d["summary"]["stages"][2].update(n_units=3960), "win_probability_extract_3961"),
        (lambda d: d["jobs"][0].update(status="COMPLETED"), "halt_arm"),
        (lambda d: d["reports"].__setitem__(-1, {"exit_code": 0}), "halt_arm"),
    ],
)
def test_canary_check_fails_each_criterion(mutate, failing):
    inputs = _passing_inputs()
    mutate(inputs)
    out = C.canary_check(**inputs)
    assert not out["ok"] and not out["checks"][failing], out


# ---- implementation review round 1 (B): preflight, the C4.5 reduced count, the check_plans key, a fresh record ----


def test_preflight_of_the_committed_canary_skips_the_halt_arms_dataset(monkeypatch):
    """Review R5B-09: `launch` needs the preflight report, and the halt arm declares no dataset (`none`); preflight
    checked every job's dataset and raised KeyError 'none' before any API call."""
    from scripts.hf_jobs import preflight
    from scripts.hf_jobs.datasets import declared

    seen: list[str] = []

    def check_datasets(names, *, api):
        for name in names:
            declared()[name]  # the real lookup: an undeclared name raises, as on HF
            seen.append(name)
        return preflight.Check("datasets", True, "ok")

    ok = preflight.Check("x", True, "ok")
    monkeypatch.setattr(preflight, "check_ref", lambda commit: ok)
    monkeypatch.setattr(preflight, "check_lock", lambda: ok)
    monkeypatch.setattr(preflight, "check_api", lambda api: ok)
    monkeypatch.setattr(preflight, "check_datasets", check_datasets)
    monkeypatch.setattr(preflight, "cached_check_plans", lambda insts, *, store, campaign_sha256: ok)
    monkeypatch.setattr(C, "committed_history", dict)
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: (_ for _ in ()).throw(RuntimeError("no plan")))

    class _Api:
        def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
            return None

    report = C.cmd_preflight(C.load_campaign(CANARY_TOML), SHA, api=_Api())
    assert "none" not in seen and {"tc3", "xshot115", "xcross_probe", "sb360", "sb_open"} == set(seen)
    assert report["campaign"] == "canary"


def test_the_check_plans_key_covers_the_halt_arms_module(tmp_path, monkeypatch):
    """Review R5B-10 (b): `check_plans` runs the halt arm's `--plan`, so its module is a cache-key input."""
    from scripts.hf_jobs import preflight

    toy = tmp_path / "tests" / "scripts" / "_toy_pipeline.py"
    toy.parent.mkdir(parents=True)
    toy.write_text("A = 1\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "_ROOT", tmp_path)
    before = preflight.check_plans_key("h", image="img")
    toy.write_text("A = 2\n", encoding="utf-8")
    assert preflight.check_plans_key("h", image="img") != before


def test_launch_refuses_a_reports_record_that_already_exists(tmp_path, monkeypatch):
    """Review R5B-10 (c): `--reports` appends, so a `launch` onto an earlier record would mix two runs; `resume`
    continues the same run and appends."""
    monkeypatch.setattr(C, "_head", lambda: SHA)
    old = tmp_path / "reports.jsonl"
    old.write_text('{"exit_code": 0}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        C.main([str(CANARY_TOML), "launch", "--watch", "--reports", str(old)])


def test_canary_check_needs_the_win_probability_instance_done():
    """Review R5B-10 (a): 3,961 is the planned unit count; the check also needs the instance reduced (done)."""
    inputs = _passing_inputs()
    inputs["reports"][1]["report"]["canary-abcdef1-win_probability-sb_open/public/default"]["action"] = "waiting"
    out = C.canary_check(**inputs)
    assert not out["ok"] and not out["checks"]["win_probability_extract_3961"]


# ---- implementation review round 2 (B): C4.5 reads the reduced count ----------------------------------------------

_WP_ID = "canary-abcdef1-win_probability-sb_open/public/default"


@pytest.mark.parametrize(
    "mutate",
    [
        # the extract tolerates failures (`max_failures=None`), so it reduces with spent units and still plans 3,961
        lambda r: r[_WP_ID].update(n_done=3956, n_failed=2),
        lambda r: r.__setitem__(_WP_ID, None),  # no reduce wrote a `_summary.json`
        lambda r: r.clear(),  # no record read for the instance
        lambda r: r[_WP_ID].update(n_done=3957),  # done + excluded short of the planned units
    ],
)
def test_canary_check_needs_every_win_probability_unit_reduced(mutate):
    """Review R5B-11: the instance's `done` and the planned 3,961 do not prove the reduce took every unit; the
    reduce's own `_summary.json` counts do (no failed unit, done + excluded = the 3,961 planned)."""
    inputs = _passing_inputs()
    mutate(inputs["wp_reduce"])
    out = C.canary_check(**inputs)
    assert not out["ok"] and not out["checks"]["win_probability_extract_3961"], out
    assert out["wp_reduce"] == inputs["wp_reduce"]  # the counts are in the result, for the report


def test_the_win_probability_reduce_record_is_read_from_the_extract_generation():
    """Review R5B-11: `canary-check` reads the `_summary.json` the extract's reduce wrote, at the plan's generation."""
    insts = list(C.instances(C.load_campaign(CANARY_TOML), SHA))
    wp = [i for i in insts if i.job == "win_probability"]
    assert len(wp) == 1
    plans: dict[str, object] = {
        i.id: {"stages": [{"name": "extract", "generation": f"{i.job}/extract/tok"}, {"name": "cv", "generation": "x"}]}
        for i in insts
    }
    raw = {"win_probability/extract/tok/_summary.json": b'{"n_done": 3961, "n_excluded": 0, "n_failed": 0}'}

    class _Snap:
        def read_bytes(self, rel):
            return raw.get(rel)

    snaps = {i.run_key: _Snap() for i in insts}
    got = C.wp_extract_reduces(insts, plans, snaps)
    assert got == {wp[0].id: {"n_done": 3961, "n_excluded": 0, "n_failed": 0}}
    raw.clear()
    assert C.wp_extract_reduces(insts, plans, snaps) == {wp[0].id: None}
    plans[wp[0].id] = RuntimeError("no plan")
    with pytest.raises(RuntimeError, match="no plan"):
        C.wp_extract_reduces(insts, plans, snaps)
