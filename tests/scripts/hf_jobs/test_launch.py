from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import (
    BUCKET_MOUNT,
    build_argv,
    build_inner_command,
    launch_all_variants,
    launch_variant_job,
    poll_until_terminal,
)

BUCKET = "silly-kicks/hf-refit-scratch"


def _default(name):
    spec = reg.REGISTRY[name]
    return spec, spec.builds[0]


def _variant(name, variant):
    spec = reg.REGISTRY[name]
    return spec, next(b for b in spec.builds if b.variant == variant)


def test_install_includes_huggingface_hub_for_in_job_hf_download():
    # Owner jobs run `hf download` to fetch the staged corpus; `.[train]` does not pull huggingface_hub
    # and the python:3.12 base lacks it (mini-wave ModuleNotFoundError). Install it explicitly.
    for name, spec in reg.REGISTRY.items():
        inner = build_inner_command(
            spec, spec.builds[0], tag="r", run_key="r1", corpus_repo="ns/tc3", sb360_repo="ns/sb360"
        )
        assert "huggingface_hub" in inner, name


def test_outputs_go_to_the_mounted_bucket_scratch_not_tmp():
    # Gold standard: output + shards land in the read-write bucket mount, so a crash/timeout keeps them
    # and --attempts resumes (corpus-driver for_each skips done shards). Nothing training-side on /tmp.
    spec, b = _default("win_probability")
    inner = build_inner_command(spec, b, tag="r", run_key="r1")
    assert "--out /scratch/r1/win_probability/default" in inner
    assert "--fresh-dir /scratch/r1/win_probability/default/weights" in inner
    assert "--metrics-json /scratch/r1/win_probability/default/metrics.json" in inner
    assert "/tmp/out" not in inner  # noqa: S108 -- asserting the old ephemeral output path is GONE


def test_no_in_job_upload_step_outputs_live_in_the_bucket():
    # Outputs persist in the bucket; retrieval is local (hf cp from the bucket). No in-job HfApi upload.
    spec, b = _default("win_probability")
    inner = build_inner_command(spec, b, tag="r", run_key="r1")
    assert "upload_folder" not in inner and "HfApi" not in inner


def test_one_job_runs_exactly_one_variant():
    spec, sweeper = _variant("ghost_gk", "sweeper")
    inner = build_inner_command(spec, sweeper, tag="r", run_key="r1", corpus_repo="ns/tc3")
    assert inner.count("python -m scripts.train_ghost_gk") == 1
    assert "--grid-x-max 52.5 --variant sweeper" in inner
    assert "sweeper_position_only" not in inner  # sibling variant must not leak in


def test_headtohead_gates_the_given_variant():
    spec, pos = _variant("ghost_gk", "position_only")
    inner = build_inner_command(spec, pos, tag="r", run_key="r1", corpus_repo="ns/tc3")
    assert "--committed-dir silly_kicks/tracking/_ghost_gk_weights/position_only" in inner
    assert "--fresh-dir /scratch/r1/ghost_gk/position_only/ghost_gk_v1" in inner
    assert "--output-dir /scratch/r1/ghost_gk/position_only" in inner  # ghost writes via --output-dir


def test_match_outcome_fresh_dir_is_the_out_dir_itself():
    spec, b = _default("match_outcome_dependence")
    inner = build_inner_command(spec, b, tag="r", run_key="r1")
    assert "--fresh-dir /scratch/r1/match_outcome_dependence/default --metrics-json" in inner


def test_public_single_variant_payload():
    spec, b = _default("win_probability")
    inner = build_inner_command(spec, b, tag="relref", run_key="r1")
    assert "git clone --depth 1 --branch relref" in inner
    assert "uv pip install --system -q -e '.[train]' statsbombpy huggingface_hub" in inner
    assert (
        "python -m scripts.train_win_probability --all-competitions --out /scratch/r1/win_probability/default" in inner
    )
    assert "python -m scripts.hf_jobs.headtohead win_probability" in inner
    assert "--committed-dir silly_kicks/win_probability/weights" in inner
    assert "hf download" not in inner  # public trainer needs no staged corpus


def test_trainers_run_as_modules_never_as_file_paths():
    for name, spec in reg.REGISTRY.items():
        inner = build_inner_command(
            spec, spec.builds[0], tag="relref", run_key="r1", corpus_repo="ns/tc3", sb360_repo="ns/sb360"
        )
        assert f"python -m {spec.module}" in inner, name
        assert "python scripts/train_" not in inner, name


def test_owner_flat_downloads_corpus_and_resolves_gen_dir():
    spec, b = _default("ghost_gk")
    inner = build_inner_command(spec, b, tag="relref", run_key="r1", corpus_repo="ns/tc3")
    assert "hf download ns/tc3 --repo-type dataset --local-dir /tmp/corpus" in inner
    assert '--data-dir "$(ls -d /tmp/corpus/shards/*/ | head -1)"' in inner
    assert "--actions-dir /tmp/corpus/_actions --home-teams /tmp/corpus/home_teams.json" in inner
    assert "--corpus-dir /tmp/corpus" in inner  # owner head-to-head eval reads the flat TC3 corpus


def test_owner_pergame_repackages_flat_to_per_game_in_job():
    spec, b = _default("xshot_occurrence")
    inner = build_inner_command(spec, b, tag="relref", run_key="r1", corpus_repo="ns/xshot")
    assert "python -m scripts.hf_jobs.repackage /tmp/corpus /tmp/pergame" in inner
    assert "--data-dir /tmp/pergame --output-dir /scratch/r1/xshot_occurrence/default" in inner
    assert "--fresh-dir /scratch/r1/xshot_occurrence/default/xshot_occurrence_v1" in inner


def test_owner_sb360_is_offline_no_token_no_network():
    spec, b = _default("receiver_model")
    inner = build_inner_command(spec, b, tag="relref", run_key="r1", corpus_repo="ns/tc3", sb360_repo="ns/sb360")
    assert "hf download ns/sb360 --repo-type dataset --local-dir /tmp/sb360" in inner
    assert "export SILLY_KICKS_PINING_MANIFEST_DIR=/tmp/sb360/manifest" in inner
    assert "--cache-dir /tmp/sb360/cache" in inner and "--match-ids-json /tmp/sb360/match_ids.json" in inner
    assert "--shard-root /scratch/r1/receiver_model/default/scratch" in inner  # extraction scratch in the bucket
    assert "--provider statsbomb --feature-set public" in inner
    assert "--fresh-dir /scratch/r1/receiver_model/default/model" in inner
    assert (
        "hf download ns/tc3 --repo-type dataset --local-dir /tmp/corpus" in inner
        and "--corpus-dir /tmp/corpus" in inner
    )


def test_owner_trainer_without_corpus_repo_is_a_loud_error():
    import pytest

    spec, b = _default("ghost_gk")
    with pytest.raises(ValueError, match="needs corpus_repo"):
        build_inner_command(spec, b, tag="t", run_key="r")
    rspec, rb = _default("receiver_model")
    with pytest.raises(ValueError, match="needs sb360_repo"):
        build_inner_command(rspec, rb, tag="t", run_key="r", corpus_repo="ns/tc3")


def test_argv_mounts_bucket_sets_attempts_and_keeps_the_dash_dash_marker():
    spec, b = _default("win_probability")
    inner = build_inner_command(spec, b, tag="relref", run_key="r1")
    argv = build_argv(spec, inner, bucket=BUCKET, attempts=3)
    assert argv[:3] == ["hf", "jobs", "run"]
    assert "--name" not in argv
    assert argv[argv.index("--flavor") + 1] == spec.flavor
    # token rides the env (`--secrets HF_TOKEN`), never the cmdline/process listing
    assert argv[argv.index("--secrets") + 1] == "HF_TOKEN"
    assert not any("hf_FAKE" in a for a in argv)
    assert argv[argv.index("--attempts") + 1] == "3"  # native auto-retry (resume via the persistent bucket)
    assert argv[argv.index("-v") + 1] == f"hf://buckets/{BUCKET}:{BUCKET_MOUNT}"  # read-write scratch mount
    # the end-of-options marker still sits immediately before `bash` (without it v2.2.0 eats `-lc`)
    assert argv[-4:] == ["--", "bash", "-lc", inner]
    assert argv[argv.index("--") - 1] == "python:3.12"


def test_launch_variant_job_passes_built_argv_and_returns_id():
    spec, b = _default("win_probability")
    calls = {}

    def fake_runner(argv):
        calls["argv"] = argv
        # v2.2.0 `-d` prints `id=<hex> name=... url=.../jobs/<ns>/<hex>` (NOT v1.7.1 "Job started with ID:").
        return "id=6abc123def456 name=python-3-12-x url=https://huggingface.co/jobs/karstenskyt/6abc123def456\n", 0

    jid = launch_variant_job(
        spec, b, tag="relref", run_key="r1", token="hf_FAKE", bucket=BUCKET, attempts=3, runner=fake_runner
    )
    assert jid == "karstenskyt/6abc123def456"  # the namespaced id (inspect/cancel/logs take it)
    assert "--attempts" in calls["argv"] and "-v" in calls["argv"]


def test_launch_all_variants_launches_one_job_per_build():
    spec = reg.REGISTRY["ghost_gk"]  # 4 variants
    seen = []

    def fake_runner(argv):
        seen.append(argv)
        h = f"6abcde{len(seen)}"  # >=6 hex for the id regex
        return f"id={h} name=x url=https://huggingface.co/jobs/karstenskyt/{h}\n", 0

    ids = launch_all_variants(
        spec,
        tag="relref",
        run_key="r1",
        corpus_repo="ns/tc3",
        token="hf_FAKE",
        bucket=BUCKET,
        attempts=3,
        runner=fake_runner,
    )
    assert set(ids) == {"default", "position_only", "sweeper", "sweeper_position_only"}
    assert len(seen) == 4


def test_poll_returns_terminal_stage_from_json_format():
    import json

    seq = iter(
        [
            json.dumps([{"id": "6abc123", "status": {"stage": "RUNNING"}}]),
            json.dumps([{"id": "6abc123", "status": {"stage": "COMPLETED"}}]),
        ]
    )

    def fake_runner(argv):
        assert argv[:3] == ["hf", "jobs", "inspect"]
        assert "--format" in argv and argv[argv.index("--format") + 1] == "json"
        return next(seq), 0

    assert poll_until_terminal("6abc123", runner=fake_runner, sleep=lambda _: None) == "COMPLETED"


def test_poll_does_not_false_positive_on_a_message_substring():
    import json

    out = json.dumps([{"id": "j", "status": {"stage": "RUNNING", "message": "retrying after an Error"}}])
    from scripts.hf_jobs.launch import _terminal_stage

    assert _terminal_stage(out) is None


def test_inner_command_is_syntactically_valid_bash():
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - CI (ubuntu) always has bash
        import pytest

        pytest.skip("bash not available")
    for name in ("win_probability", "ghost_gk", "xshot_occurrence", "receiver_model"):
        spec = reg.REGISTRY[name]
        inner = build_inner_command(
            spec, spec.builds[0], tag="relref", run_key="r1", corpus_repo="ns/tc3", sb360_repo="ns/sb360"
        )
        r = subprocess.run([bash, "-n", "-c", inner], capture_output=True, text=True)  # noqa: S603
        assert r.returncode == 0, f"{name}: invalid bash: {r.stderr}"
