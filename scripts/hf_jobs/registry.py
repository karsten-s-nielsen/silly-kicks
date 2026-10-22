"""Single source of truth for how each trainer runs as a stage-contract pipeline and how it is certified.

ADR-128: a migrated trainer runs as ``python -m <pipeline>`` (the uniform stage CLI, one ``--out``); an
unmigrated public trainer runs its own CLI through the one-unit adapter (``adapter``). Every module runs as
``python -m`` -- the file form raises ``ModuleNotFoundError: No module named 'scripts'`` (there is no
``scripts/__init__.py``). The per-variant ``run_args`` are pinned from each committed variant's
``metadata.json``; ``fresh_subdir`` is where the artifact lands inside ``--out`` (``""`` for a pipeline).
The flavor is ONE constant (``launch.FLAVOR``, INV-4); job size and timeout are the planner's (spec 5.4);
the corpus shape is the dataset's (``datasets.toml``). The campaign controller consumes this.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    extras: tuple[str, ...]  # extra pip args beyond ".[train]" (e.g. ("statsbombpy",))
    base_args: tuple[str, ...]  # trainer args shared across every variant
    builds: tuple[VariantBuild, ...]  # one per committed variant (>=1)
    primary_metric: str  # the metric the head-to-head scorer EMITS for both models
    metric_direction: str  # "lower_better" | "higher_better"
    delta: float  # pinned per-trainer regression tolerance
    databricks_blocked: bool = False
    #: This trainer reproduces a held-out TF-19 probe from a STAGED per-game probe dataset (ADR-123
    #: dataset-agnostic, never in-job pining). ADR-128: the campaign names it (`probe_dataset`) and the job
    #: mounts it; the trainer's `probe` stage reads it via --probe-dataset. Only xcross.
    probe: bool = False
    #: This trainer ships a VISIBILITY-GATED candidate (public / sc_extended / full). The public/owner
    #: policy is computed OFF-HF by the controller (``arm_policy.compute_arms`` over ``PUBLIC_CORPUS``) and
    #: passed as the neutral ``--arms-json`` job parameter; the in-job trainer has NO public/private concept
    #: (ADR-123 Amendment B). Only xshot_occurrence + xcross_attempt.
    visibility_gated: bool = False
    #: ADR-128: how an artifact of this trainer is certified at adoption (spec 5.5, single-sourced):
    #: ``public`` = head-to-head floor + ADR-126 integrity gate; ``owner`` = CV-parity against the
    #: committed ``metrics.json`` + integrity gate (head-to-head diagnostic only); ``rebaseline`` = the
    #: criterion-3 identity comparison. ``tier_certification`` overrides it per campaign tier.
    certification: str = "owner"
    tier_certification: dict[str, str] | None = None
    #: ADR-128: the dotted module path whose ``PIPELINE`` the campaign launches; ``""`` until the module is
    #: migrated, and for an ``adapter`` trainer (its unmigrated CLI runs as ONE unit, spec 5.6).
    pipeline: str = ""
    adapter: bool = False
    #: ADR-128 owner certification (CV parity): the dotted path of the CV metric in BOTH the fresh and the
    #: committed ``metrics.json`` (``{shipped_variant}`` is filled from that file); ``""`` = not CV-certified.
    cv_metric: str = ""

    @property
    def weights_paths(self) -> tuple[str, ...]:
        """Every committed WHEEL variant dir -- the head-to-head baselines (kept for the registry gates).
        A Hub-only build (empty ``committed_path``, e.g. ghost ``full``) has no wheel dir."""
        return tuple(b.committed_path for b in self.builds if b.committed_path)

    def certification_for(self, tier: str) -> str:
        """The certification policy for a campaign ``tier`` (spec 5.5)."""
        if self.tier_certification is not None:
            return self.tier_certification[tier]
        return self.certification


@dataclass(frozen=True)
class DriverSpec:
    """A corpus driver the campaign launches (ADR-128 spec 5.8): its pipeline module, how its output is
    certified, and where its committed output lives (``output_rel``, repo-relative)."""

    name: str
    module: str
    certification: str
    output_rel: str


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
        extras=("statsbombpy",),
        base_args=(),
        # ADR-128: `--run-all --out` writes the bundle files directly (no weights/ subdir).
        builds=(VariantBuild("default", ("--all-competitions",), "win_probability/weights", ""),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
        certification="public",
        pipeline="scripts.train_win_probability",
    ),
    "pass_completion": TrainerSpec(
        name="pass_completion",
        module="scripts.train_pass_completion",
        extras=("statsbombpy",),
        base_args=("--source", "open-data"),
        builds=(VariantBuild("default", (), "expected_passing/weights", "weights"),),
        certification="public",
        adapter=True,  # ADR-128 spec 5.6: runs via the one-unit adapter stage until Cycle 2
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "match_outcome_dependence": TrainerSpec(
        name="match_outcome_dependence",
        module="scripts.train_match_outcome_dependence",
        extras=("statsbombpy",),
        base_args=(),
        # match_outcome writes the artifact DIRECTLY to --out (no weights/ subdir) -- verified.
        builds=(VariantBuild("default", (), "match_outcome/weights", ""),),
        certification="public",
        adapter=True,  # ADR-128 spec 5.6: runs via the one-unit adapter stage until Cycle 2
        primary_metric="log_loss",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "xsuccess": TrainerSpec(
        name="xsuccess",
        module="scripts.train_xsuccess",
        extras=("statsbombpy",),
        base_args=(),
        builds=(VariantBuild("default", (), "xsuccess/weights", "weights"),),
        certification="public",
        adapter=True,  # ADR-128 spec 5.6: runs via the one-unit adapter stage until Cycle 2
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
    ),
    "ghost_gk": TrainerSpec(
        name="ghost_gk",
        module="scripts.train_ghost_gk",
        extras=(),
        base_args=(),
        # 5 builds (4 wheel variants + the Hub-only `full`); ADR-128: `--run-all --out` writes the bundle
        # files directly (no ghost_gk_v1/ subdir). run_args pinned from the committed variants' metadata
        # (feature_set + grid_spec): default/position_only use the 30 m DEFAULT_GHOST_GRID; the sweeper
        # pair lift the grid ceiling to 52.5 m. `default` and `full` share extract/cache/cv/fit (E11).
        builds=(
            # Decision P10 (ADR-128): the committed default bundle records "variant": "default", so the build
            # passes it explicitly (the trainer's own default is "full").
            VariantBuild(
                "default",
                ("--feature-set", "faithful", "--variant", "default"),
                "tracking/_ghost_gk_weights/default",
                "",
            ),
            # P10 applies here too: the committed position_only bundle ALSO records "variant": "default".
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only", "--variant", "default"),
                "tracking/_ghost_gk_weights/position_only",
                "",
                feature_set="position_only",
            ),
            VariantBuild(
                "sweeper",
                ("--feature-set", "faithful", "--grid-x-max", "52.5", "--variant", "sweeper"),
                "tracking/_ghost_gk_weights/sweeper",
                "",
            ),
            VariantBuild(
                "sweeper_position_only",
                ("--feature-set", "position_only", "--grid-x-max", "52.5", "--variant", "sweeper_position_only"),
                "tracking/_ghost_gk_weights/sweeper_position_only",
                "",
                feature_set="position_only",
            ),
            # Hub-only (silly-kicks/ghost-gk-v1): no wheel dir; the artifact manifest names its destination.
            VariantBuild("full", ("--feature-set", "faithful", "--variant", "full"), "", ""),
        ),
        pipeline="scripts.train_ghost_gk",
        cv_metric="cv_mae_euclidean_mean",
        primary_metric="cv_mae_euclidean",
        metric_direction="lower_better",
        delta=0.05,
    ),  # metres; committed default ~1.13 m
    "ghost_outfield": TrainerSpec(
        name="ghost_outfield",
        module="scripts.train_ghost_outfield",
        extras=(),
        base_args=(),
        builds=(
            # ADR-128: `--run-all --out` writes the bundle files directly (no <variant>/ subdir).
            VariantBuild("default", ("--variant", "default"), "tracking/_ghost_outfield_weights/default", ""),
            VariantBuild(
                "position_only",
                ("--variant", "position_only"),
                "tracking/_ghost_outfield_weights/position_only",
                "",
                feature_set="position_only",
            ),
        ),
        pipeline="scripts.train_ghost_outfield",
        cv_metric="cv_mae",
        primary_metric="cv_mae",
        metric_direction="lower_better",
        delta=0.10,
    ),  # metres; committed default ~6.0 m
    "xshot_occurrence": TrainerSpec(
        name="xshot_occurrence",
        module="scripts.train_xshot_occurrence",
        extras=(),
        base_args=("--n-trials", "50"),
        visibility_gated=True,  # ships public/sc_extended/full; --arms policy computed off-HF (ADR-123 Amd B)
        # ADR-128: `--run-all --out` writes the bundle files directly (no xshot_occurrence_v1/ subdir); the
        # controller adds `--dataset` and `--arms-json` (the off-HF arm policy).
        builds=(
            VariantBuild("default", ("--feature-set", "faithful"), "tracking/_xshot_weights/default", ""),
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only"),
                "tracking/_xshot_weights/position_only",
                "",
                feature_set="position_only",
            ),
        ),
        pipeline="scripts.train_xshot_occurrence",
        cv_metric="candidates.{shipped_variant}.metrics.pr_auc",
        primary_metric="pr_auc",
        metric_direction="higher_better",
        delta=0.01,
        tier_certification={"public": "public", "sc_extended": "owner"},
    ),
    "xcross_attempt": TrainerSpec(
        name="xcross_attempt",
        module="scripts.train_xcross_attempt",
        extras=(),
        base_args=("--n-trials", "50"),
        probe=True,  # held-out TF-19 probe from a staged per-game probe dataset (ADR-123: --probe-data-dir, no pining)
        visibility_gated=True,  # ships public/sc_extended/full; --arms policy computed off-HF (ADR-123 Amd B)
        # ADR-128: `--run-all --out` writes the bundle files directly (no xcross_attempt_v1/ subdir); the
        # controller adds `--dataset`, `--probe-dataset` and `--arms-json` (the off-HF arm policy).
        builds=(
            VariantBuild("default", ("--feature-set", "faithful"), "tracking/_xcross_weights/default", ""),
            VariantBuild(
                "position_only",
                ("--feature-set", "position_only"),
                "tracking/_xcross_weights/position_only",
                "",
                feature_set="position_only",
            ),
        ),
        pipeline="scripts.train_xcross_attempt",
        cv_metric="candidates.{shipped_variant}.metrics.pr_auc",
        primary_metric="pr_auc",
        metric_direction="higher_better",
        delta=0.01,
        tier_certification={"public": "public", "sc_extended": "owner"},
    ),
    "receiver_model": TrainerSpec(
        name="receiver_model",
        module="scripts.train_receiver_model",
        extras=(),
        # ADR-123 dataset-AGNOSTIC: the job reads the STAGED dataset (--dataset); the controller passes the
        # visibility as parameters (--corpus-visibility / --corpus-n-matches, decision P5). NO pining in-job.
        # --provider still selects the labeling strategy; --feature-set public is the shipped variant.
        base_args=("--provider", "statsbomb", "--feature-set", "public"),
        # ADR-128: `--run-all --out` writes the bundle files directly (no model/ subdir).
        builds=(VariantBuild("default", (), "tracking/_receiver_weights/default", ""),),
        pipeline="scripts.train_receiver_model",
        cv_metric="top1_cv",
        primary_metric="top1",
        metric_direction="higher_better",
        delta=0.01,
    ),
    # --- scaffold: trainers above ---
}

BLOCKED: dict[str, TrainerSpec] = {
    "gk_retention": TrainerSpec(
        name="gk_retention",
        module="scripts.train_gk_retention",
        extras=(),
        base_args=(),
        builds=(VariantBuild("default", (), "xtgk/_retention_weights/default", "model"),),
        primary_metric="ece",
        metric_direction="lower_better",
        delta=0.01,
        databricks_blocked=True,
    ),
}


#: ADR-128 spec 5.8: the corpus drivers a campaign launches (the census in Cycle 1).
DRIVERS: dict[str, DriverSpec] = {
    "build_territory_ranking_census": DriverSpec(
        "build_territory_ranking_census",
        "scripts.build_territory_ranking_census",
        "rebaseline",
        "docs/research/territory_ranking_census",
    ),
    # --- scaffold: drivers above ---
}


def spec_for(name: str) -> TrainerSpec | DriverSpec:
    """The trainer or driver spec for a campaign job name (REGISTRY, then DRIVERS, then BLOCKED)."""
    for table in (REGISTRY, DRIVERS, BLOCKED):
        if name in table:
            return table[name]
    raise KeyError(f"no trainer or driver named {name!r} in the job registry")


def get(name: str) -> TrainerSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    return BLOCKED[name]
