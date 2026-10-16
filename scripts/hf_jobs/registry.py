"""Single source of truth for how each trainer runs as an HF Job + its quality-floor config.

Every fact here is **execution-verified on the DGX** against the real trainers (2026-10-13): the
module path (trainers run as ``python -m scripts.train_<name>`` -- the file form
``python scripts/train_<name>.py`` raises ``ModuleNotFoundError: No module named 'scripts'`` because
there is no ``scripts/__init__.py`` and ``sys.path[0]`` becomes ``scripts/``), the output flag
(``--out`` vs ``--output-dir``, which differ per trainer), the per-variant ``run_args`` (pinned from
each committed variant's ``metadata.json``), and where the fresh weights land under the output dir
(``fresh_subdir``). ``launch.py`` consumes this; nothing about the invocation is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Corpus source modes (how the HF job feeds the trainer -- see launch.build_inner_command):
#   "public"        -- no staged corpus; the trainer pulls StatsBomb open-data in-job.
#   "owner_flat"    -- the staged flat TC3 corpus (shards/<gen>/*.parquet + _actions/ + home_teams.json),
#                      consumed directly via --data-dir/--actions-dir/--home-teams.
#   "owner_pergame" -- the same flat corpus, repackaged in-job to the per-game {game}/frames.parquet +
#                      shots.parquet layout the xshot/xcross --data-dir path requires.
#   "owner_sb360"   -- the staged SB360 cache + offline manifest (zero pining token/network in-job).
SOURCES = ("public", "owner_flat", "owner_pergame", "owner_sb360")


@dataclass(frozen=True)
class VariantBuild:
    """One committed model variant a trainer run produces, and how to reproduce + score it.

    ``run_args`` are the variant-only trainer args (pinned from the committed variant's metadata);
    ``committed_path`` is the repo dir (under ``silly_kicks/``) holding the committed variant that the
    head-to-head loads as the baseline; ``fresh_subdir`` is where THIS run writes the fresh weights
    under its own output dir (so the head-to-head's ``--fresh-dir`` is ``{out}/{variant}/{fresh_subdir}``).
    """

    variant: str
    run_args: tuple[str, ...]
    committed_path: str
    fresh_subdir: str
    #: The feature vintage this variant was fit at. The head-to-head eval extracts features at THIS
    #: set so a velocity-free (position_only) booster is scored on a matching frame -- xgboost
    #: predict_proba is strict on feature_names (the floor-crash fix, ADR-123 / spec §5). Default
    #: "faithful" for the velocity-bearing variants and the public trainers.
    feature_set: str = "faithful"


@dataclass(frozen=True)
class TrainerSpec:
    name: str  # "win_probability"
    module: str  # "scripts.train_win_probability" (run via python -m; NEVER the file path)
    source: str  # one of SOURCES
    flavor: str  # HF Jobs flavor
    extras: tuple[str, ...]  # extra pip args beyond ".[train]" (e.g. ("statsbombpy",))
    timeout: str  # HF --timeout, e.g. "45m" / "6h"
    output_flag: str  # "--out" | "--output-dir" (differs per trainer -- verified)
    base_args: tuple[str, ...]  # trainer args shared across every variant
    builds: tuple[VariantBuild, ...]  # one per committed variant (>=1)
    primary_metric: str  # the metric the head-to-head scorer EMITS for both models
    metric_direction: str  # "lower_better" | "higher_better"
    delta: float  # pinned per-trainer regression tolerance
    databricks_blocked: bool = False
    #: This trainer reproduces a held-out TF-19 probe from a STAGED per-game probe dataset (ADR-123
    #: dataset-agnostic: --probe-data-dir, never in-job pining). launch downloads probe_repo -> /tmp/probe
    #: and threads --probe-data-dir. Only xcross (the committed GS 10502/10503 held-out probe).
    probe: bool = False
    #: This trainer ships a VISIBILITY-GATED candidate (public / sc_extended / full). The public/owner
    #: policy is computed OFF-HF by the launcher (``arm_policy.compute_arms`` over ``PUBLIC_CORPUS``) and
    #: passed as the neutral ``--arms`` job parameter; the in-job trainer has NO public/private concept
    #: (ADR-123 Amendment B). Only xshot_occurrence + xcross_attempt.
    visibility_gated: bool = False

    @property
    def weights_paths(self) -> tuple[str, ...]:
        """Every committed variant dir -- the head-to-head baselines (kept for the registry gates)."""
        return tuple(b.committed_path for b in self.builds)


# ``primary_metric`` is the metric name the head-to-head scorer (scripts/hf_jobs/headtohead.py) computes
# IDENTICALLY for both the committed and the fresh model on the same held-out split -- NOT a key read from
# any committed metrics.json (3 public trainers ship none). The floor gates EVERY variant (no silent
# regression on a non-default variant). ``delta`` is the pinned per-trainer tolerance; OWNER-RATIFIED,
# FAIL-CLOSED (missing/NaN blocks). A noisier trainer needs a LARGER delta; tightened by the owner after
# the first head-to-head deltas, never loosened silently.
REGISTRY: dict[str, TrainerSpec] = {
    "win_probability": TrainerSpec(
        name="win_probability",
        module="scripts.train_win_probability",
        source="public",
        flavor="cpu-upgrade",
        extras=("statsbombpy",),
        timeout="2h",  # 3961 matches (--all-competitions) + per-shard bucket-FUSE writes (~31m on local /tmp)
        output_flag="--out",
        base_args=(),
        builds=(VariantBuild("default", ("--all-competitions",), "win_probability/weights", "weights"),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "pass_completion": TrainerSpec(
        name="pass_completion",
        module="scripts.train_pass_completion",
        source="public",
        flavor="cpu-upgrade",
        extras=("statsbombpy",),
        timeout="45m",
        output_flag="--out",
        base_args=("--source", "open-data"),
        builds=(VariantBuild("default", (), "expected_passing/weights", "weights"),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "match_outcome_dependence": TrainerSpec(
        name="match_outcome_dependence",
        module="scripts.train_match_outcome_dependence",
        source="public",
        flavor="cpu-upgrade",
        extras=("statsbombpy",),
        timeout="45m",
        output_flag="--out",
        base_args=(),
        # match_outcome writes the artifact DIRECTLY to --out (no weights/ subdir) -- verified.
        builds=(VariantBuild("default", (), "match_outcome/weights", ""),),
        primary_metric="log_loss",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "xsuccess": TrainerSpec(
        name="xsuccess",
        module="scripts.train_xsuccess",
        source="public",
        flavor="cpu-upgrade",
        extras=("statsbombpy",),
        timeout="45m",
        output_flag="--out",
        base_args=(),
        builds=(VariantBuild("default", (), "xsuccess/weights", "weights"),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "ghost_gk": TrainerSpec(
        name="ghost_gk",
        module="scripts.train_ghost_gk",
        source="owner_flat",
        flavor="cpu-upgrade",  # owner directive 2026-10-14: cpu-upgrade only
        extras=(),
        timeout="3h",  # assemble-phase ceiling; extraction + CV folds fan out (shorter) per-phase in launch
        output_flag="--output-dir",
        base_args=(),
        # 4 variants, each writing {output-dir}/ghost_gk_v1, so each needs its OWN output dir (launch
        # gives every build {OUT}/{variant}). run_args pinned from the committed variants' metadata
        # (feature_set + grid_spec): default/position_only use the 30 m DEFAULT_GHOST_GRID; the sweeper
        # pair lift the grid ceiling to 52.5 m.
        builds=(
            VariantBuild("default", ("--feature-set", "faithful"), "tracking/_ghost_gk_weights/default", "ghost_gk_v1"),
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only"),
                "tracking/_ghost_gk_weights/position_only",
                "ghost_gk_v1",
                feature_set="position_only",
            ),
            VariantBuild(
                "sweeper",
                ("--feature-set", "faithful", "--grid-x-max", "52.5", "--variant", "sweeper"),
                "tracking/_ghost_gk_weights/sweeper",
                "ghost_gk_v1",
            ),
            VariantBuild(
                "sweeper_position_only",
                ("--feature-set", "position_only", "--grid-x-max", "52.5", "--variant", "sweeper_position_only"),
                "tracking/_ghost_gk_weights/sweeper_position_only",
                "ghost_gk_v1",
                feature_set="position_only",
            ),
        ),
        primary_metric="cv_mae_euclidean",
        metric_direction="lower_better",
        delta=0.05,
    ),  # metres; committed default ~1.13 m
    "ghost_outfield": TrainerSpec(
        name="ghost_outfield",
        module="scripts.train_ghost_outfield",
        source="owner_flat",
        flavor="cpu-upgrade",  # owner directive 2026-10-14: cpu-upgrade only
        extras=(),
        timeout="3h",  # assemble-phase ceiling; extraction + CV folds fan out per-phase in launch
        output_flag="--output-dir",
        base_args=(),
        # 2 variants; the trainer writes {output-dir}/{variant}, so the variant is in the path -- a shared
        # output dir would NOT collide, but launch keeps the uniform {OUT}/{variant} rule anyway.
        builds=(
            VariantBuild("default", ("--variant", "default"), "tracking/_ghost_outfield_weights/default", "default"),
            VariantBuild(
                "position_only",
                ("--variant", "position_only"),
                "tracking/_ghost_outfield_weights/position_only",
                "position_only",
                feature_set="position_only",
            ),
        ),
        primary_metric="cv_mae",
        metric_direction="lower_better",
        delta=0.10,
    ),  # metres; committed default ~6.0 m
    "xshot_occurrence": TrainerSpec(
        name="xshot_occurrence",
        module="scripts.train_xshot_occurrence",
        source="owner_pergame",
        flavor="cpu-upgrade",  # owner directive 2026-10-14: cpu-upgrade only
        extras=(),
        timeout="2h",  # assemble-phase ceiling; extraction + HPO studies fan out per-phase in launch
        output_flag="--output-dir",
        base_args=("--n-trials", "50"),
        visibility_gated=True,  # ships public/sc_extended/full; --arms policy computed off-HF (ADR-123 Amd B)
        builds=(
            VariantBuild(
                "default", ("--feature-set", "faithful"), "tracking/_xshot_weights/default", "xshot_occurrence_v1"
            ),
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only"),
                "tracking/_xshot_weights/position_only",
                "xshot_occurrence_v1",
                feature_set="position_only",
            ),
        ),
        primary_metric="pr_auc",
        metric_direction="higher_better",
        delta=0.01,
    ),
    "xcross_attempt": TrainerSpec(
        name="xcross_attempt",
        module="scripts.train_xcross_attempt",
        source="owner_pergame",
        flavor="cpu-upgrade",  # owner directive 2026-10-14: cpu-upgrade only
        extras=(),
        timeout="2h",  # assemble-phase ceiling; extraction + HPO studies fan out per-phase in launch
        output_flag="--output-dir",
        base_args=("--n-trials", "50"),
        probe=True,  # held-out TF-19 probe from a staged per-game probe dataset (ADR-123: --probe-data-dir, no pining)
        visibility_gated=True,  # ships public/sc_extended/full; --arms policy computed off-HF (ADR-123 Amd B)
        builds=(
            VariantBuild(
                "default", ("--feature-set", "faithful"), "tracking/_xcross_weights/default", "xcross_attempt_v1"
            ),
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only"),
                "tracking/_xcross_weights/position_only",
                "xcross_attempt_v1",
                feature_set="position_only",
            ),
        ),
        primary_metric="pr_auc",
        metric_direction="higher_better",
        delta=0.01,
    ),
    "receiver_model": TrainerSpec(
        name="receiver_model",
        module="scripts.train_receiver_model",
        source="owner_sb360",
        flavor="cpu-upgrade",  # owner directive 2026-10-14: cpu-upgrade only
        extras=(),
        timeout="2h",  # assemble-phase ceiling; extraction fans out per-phase in launch
        output_flag="--out",
        # ADR-123 dataset-AGNOSTIC: launch injects --data-dir (the STAGED flat corpus) + --shard-root
        # (extraction scratch in the bucket). NO pining manifest / --cache-dir / --match-ids-json in-job.
        # --provider still selects the labeling strategy; --feature-set public is the shipped variant.
        base_args=("--provider", "statsbomb", "--feature-set", "public"),
        builds=(VariantBuild("default", (), "tracking/_receiver_weights/default", "model"),),
        primary_metric="top1",
        metric_direction="higher_better",
        delta=0.01,
    ),
}

BLOCKED: dict[str, TrainerSpec] = {
    "gk_retention": TrainerSpec(
        name="gk_retention",
        module="scripts.train_gk_retention",
        source="owner_sb360",  # nominal; never launched -- Databricks-marts corpus has no pining/staging path
        flavor="cpu-xl",
        extras=(),
        timeout="4h",
        output_flag="--out",
        base_args=(),
        builds=(VariantBuild("default", (), "xtgk/_retention_weights/default", "model"),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
        databricks_blocked=True,
    ),
}


def trainable_names() -> list[str]:
    return sorted(REGISTRY)


def blocked_names() -> list[str]:
    return sorted(BLOCKED)


def get(name: str) -> TrainerSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    return BLOCKED[name]
