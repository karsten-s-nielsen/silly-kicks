"""Crosser-anchored opportunity-row builder for the xCross causal harness (ADR-015).

A per-(game,period) spell state-machine: one row per continuous wide-area possession-spell, anchored
at entry (the paper's sender-level unit). The spell end serves as the dedup boundary AND the ceiling
on the treatment window (R3-M1):
  Z = 1 iff a possessing-team cross occurs in (entry, min(entry + EXPOSURE_WINDOW_SECONDS, spell_end)];
      the fixed T cap keeps Z-exposure bounded (no spell-length confounding -- Y's window is already
      fixed, so clamping to spell_end adds no duration->Y path), and the spell_end cap prevents
      misattributing a cross from a LATER re-possession phase to this opportunity.
  Y = a possessing-team shot in (anchor, anchor + OUTCOME_WINDOW_SECONDS], anchor = t_cross for
      treated (strictly post-cross -> no reverse-direction leakage, R2-M1) else entry for controls;
      Y is NOT possession-clamped (documented modeling choice -- treated/control windows time-shifted).
X = the 7 paper confounders (imported from _xcross_attempt._CONFOUNDERS -- single source, R2-M2) + 6
GK columns; ball-geometry features are excluded (surface-model inputs, not paper confounders). Pure;
no I/O. Reuses the shipped xCross domain/carrier/feature helpers so the matched corpus is the model's
training domain by construction. Dedup R2-M1: a new spell starts only on a possession break or a
wide-area domain exit; a mid-spell carrier hand-off stays one row.

TF-19/ADR-037: every knob above is the ``xcross_config`` DEFAULT of the parameterized
``OpportunityConfig`` surface (``build_opportunities(config=...)``; ``config=None`` is byte-identical
to the legacy xCross path); the §3.3 shot arm (``shot_arm_config``) re-targets treatment/outcome/
domain/extractor purely as arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match, same_id
from silly_kicks.spadl import config as _spc
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
from silly_kicks.tracking._xcross_attempt import (
    _ADVANCE_M,
    _CONFOUNDERS,
    XCROSS_FEATURE_NAMES_FAITHFUL,
    _build_goal_map,
    _has_results,
    _in_wide_area,
)

# X split (M3 + R2-M2): the 7 paper confounders are the SINGLE-SOURCE _CONFOUNDERS (not re-literal'd);
# ball_r/theta/speed are surface-model inputs, NOT paper confounders, and are excluded from the causal X.
PAPER_CONFOUNDERS = list(_CONFOUNDERS)
GK_BLOCK = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL if c.startswith("gk_")]

# Pre-registered windows (named + asserted). The treatment/outcome windows are bounded by a fixed cap
# (R2-H3), so they are NOT a function of spell length; MAX_SPELL_SECONDS only bounds the dedup machine.
MAX_SPELL_SECONDS = 30.0  # dedup cap: split a pathological never-closing in-domain run
EXPOSURE_WINDOW_SECONDS = 8.0  # T: Z = cross in (entry, min(entry+T, spell_end)]
OUTCOME_WINDOW_SECONDS = 6.0  # W: Y = shot in (anchor, anchor+W]

_PROV_COLS = [
    "game_id",
    "period_id",
    "entry_frame_id",
    "entry_time",
    "end_time",
    "spell_duration_seconds",
    "possessing_team",
    "carrier_resolved",
]


@dataclass(frozen=True)
class OpportunityConfig:
    """Full builder surface (ADR-037/M8): everything a consumer arm needs, as arguments."""

    treatment_type_names: tuple[str, ...]
    outcome_type_names: tuple[str, ...]
    outcome_result_ids: tuple[int, ...] | None = None  # None = type-only (legacy xCross)
    outcome_window_seconds: float = OUTCOME_WINDOW_SECONDS  # ALWAYS a window (R8: the
    # own-result 'None' form was structurally degenerate for controls and is banned)
    outcome_window_anchor_inclusive: bool = False  # False = legacy strict-post (xCross)
    exposure_window_seconds: float = EXPOSURE_WINDOW_SECONDS
    max_spell_seconds: float = MAX_SPELL_SECONDS  # THREADED into the spell loop
    confounders: tuple[str, ...] = tuple(PAPER_CONFOUNDERS)
    gk_block: tuple[str, ...] = tuple(GK_BLOCK)
    domain: str = "wide_area"  # "wide_area" | "attacking_third"
    extractor: str = "xcross"  # "xcross" | "xs" -- threaded via _extract_row adapters
    # --- TF-19 sign-off package (D3/D5). All defaulted: every shipped config is unchanged. ---
    outcome_max_distance_m: float | None = None  # D3: None = no spatial filter (legacy)
    emit_outcome_partition: bool = False  # Layer 2: Y_attempt / Y_close_attempt / Y_far_attempt
    treatment_covariate: str | None = None  # D5: None = action-occurrence treatment (legacy)
    treatment_threshold_m: float | None = None


def xcross_config(model_metadata: dict) -> OpportunityConfig:
    """The legacy xCross harness configuration (the ``config=None`` default path).

    Examples
    --------
    >>> cfg = xcross_config({"cross_types": ["cross"]})
    >>> cfg.treatment_type_names
    ('cross',)
    >>> cfg.extractor, cfg.domain, cfg.outcome_window_anchor_inclusive
    ('xcross', 'wide_area', False)
    """
    return OpportunityConfig(
        treatment_type_names=tuple(model_metadata.get("cross_types", ("cross",))),
        outcome_type_names=("shot", "shot_freekick", "shot_penalty"),
    )


#: The §3.3 xS-side confounder list -- a FRESH registered decision (xS has no _CONFOUNDERS
#: constant to reuse): the ball-geometry trio is the xS surface's core positional state.
SHOT_ARM_CONFOUNDERS = ("r", "theta", "speed", "openGoal", "DefDist_0", "DefDist_1")


def shot_arm_config(model_metadata: dict) -> OpportunityConfig:
    """The TF-19 §3.3 shot-arm configuration (ADR-037), expressed purely as builder arguments.

    Examples
    --------
    >>> cfg = shot_arm_config({})
    >>> cfg.treatment_type_names
    ('shot', 'shot_freekick', 'shot_penalty')
    >>> cfg.extractor, cfg.domain, cfg.outcome_window_anchor_inclusive
    ('xs', 'attacking_third', True)
    """
    # Outcome (P1 re-registration): ANCHOR-INCLUSIVE success window -- ts >= anchor,
    # result_id == success, 6 s. Y = the anchor shot's own goal OR a rebound goal for
    # treated spells, and a within-window goal for CONTROLS (anchor = entry). The
    # earlier own-result-only registration made control Y ≡ 0 by construction (controls
    # have no anchor action), which made the ATT confounder-INVARIANT and the
    # entanglement gate structurally dead. Anchor-inclusion also moots the
    # np.isclose time-scan concern (the window catches the anchor action).
    return OpportunityConfig(
        treatment_type_names=("shot", "shot_freekick", "shot_penalty"),
        outcome_type_names=("shot", "shot_freekick", "shot_penalty"),
        outcome_result_ids=(_spc.result_id["success"],),
        outcome_window_seconds=OUTCOME_WINDOW_SECONDS,
        outcome_window_anchor_inclusive=True,  # NEW config field: (ts >= anchor) vs legacy (ts > anchor)
        domain="attacking_third",
        extractor="xs",  # P2: the extractor AXIS (see _extract_row)
        confounders=SHOT_ARM_CONFOUNDERS,
        gk_block=("GK_r", "GK_theta"),  # P2: xS GK names -- xcross gk_* don't exist in xS features
    )


#: BUILD-TIME confounders for Layer 2: EXTRACTOR-PRODUCED COLUMNS ONLY.
#:
#: ``_row`` reads every ``cfg.confounders`` name straight out of the xS feature dict with a HARD key
#: lookup, so a name the extractor does not emit raises ``KeyError`` at BUILD time -- not NaN.
#: VERIFIED against ``XSHOT_FEATURE_NAMES_FAITHFUL``: ``r``/``theta``/``DefDist_0``/``DefDist_1`` are
#: present; ``defensive_line_height``, ``defensive_line_compactness``,
#: ``pressure_on_actor__bekkers_pi``, ``score_differential`` and ``time_remaining_s`` are NOT.
LAYER2_BUILD_CONFOUNDERS: tuple[str, ...] = ("r", "theta", "DefDist_0", "DefDist_1")

#: ANALYSIS-TIME design matrix: the build-time set PLUS what ``causal/_confounders.py`` joins on
#: afterwards. This is what the propensity model's ``X`` is assembled from -- never
#: ``cfg.confounders``. Split deliberately: the tracking confounders are per-spell joins, not
#: extractor features, and forcing them through the extractor contract would mean either a silent
#: NaN-filling ``_row`` (hiding genuine join failures) or teaching the xS extractor about defensive
#: lines it has no business knowing. ``score_differential`` is emitted by ``_row`` itself.
LAYER2_CONFOUNDERS: tuple[str, ...] = (
    *LAYER2_BUILD_CONFOUNDERS,
    "defensive_line_height",
    "defensive_line_compactness",
    "pressure_on_actor__bekkers_pi",
    "score_differential",
    "time_remaining_s",
)


def layer2_config(model_metadata: dict) -> OpportunityConfig:
    """TF-19 §6.4 Layer 2: the H1-vs-H2 decider's DESIGN (sign-off package D5).

    Treatment is keeper DEPTH at spell entry binarised at the penalty-area line -- Law-defined and
    data-independent, so the entire decider is untuned. The outcome is an ATTEMPT; contrast
    :func:`shot_arm_config`, whose outcome is a GOAL and whose treatment is roughly this outcome.
    That distinction is why a power curve for Layer 2 cannot be borrowed from the shot arm.

    Building this config does NOT run Layer 2 -- see the FIREWALL in :mod:`silly_kicks.causal.power`.

    Examples
    --------
    >>> cfg = layer2_config({})
    >>> cfg.treatment_covariate, cfg.treatment_threshold_m
    ('gk_depth_x', 16.5)
    >>> cfg.outcome_result_ids is None  # an ATTEMPT, not a goal
    True
    """
    return OpportunityConfig(
        treatment_type_names=(),  # unused: the covariate axis supersedes it
        treatment_covariate="gk_depth_x",
        treatment_threshold_m=16.5,
        outcome_type_names=("shot", "shot_freekick", "shot_penalty"),
        outcome_result_ids=None,
        outcome_window_anchor_inclusive=True,
        outcome_max_distance_m=16.5,
        emit_outcome_partition=True,
        domain="attacking_third",
        extractor="xs",
        confounders=LAYER2_BUILD_CONFOUNDERS,  # NOT LAYER2_CONFOUNDERS -- see the constant's note
        gk_block=("GK_r", "GK_theta"),
    )


def _extract_row(cfg, grp, *, gk_team_id, goal_x, carrier_pid, sd):
    """Per-extractor ADAPTER closure (P2): the two real signatures differ -- xcross takes
    ``carrier_player_id`` + ``score_differential``, xS does not."""
    if cfg.extractor == "xs":
        from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

        return extract_xshot_features(grp, gk_team_id=gk_team_id, goal_x=goal_x).iloc[0]
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features

    return extract_xcross_features(
        grp, gk_team_id=gk_team_id, goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
    ).iloc[0]


def build_opportunities(
    frames,
    actions,
    *,
    home_team_id,
    model_metadata,
    advance_m=_ADVANCE_M,
    config: OpportunityConfig | None = None,
) -> pd.DataFrame:
    """Return one row per continuous in-domain possession-spell: the configured confounders + GK
    block, treatment ``Z``, outcome ``Y``, and provenance. Pure; no I/O.

    ``config=None`` constructs ``xcross_config(model_metadata)`` -- the legacy xCross path,
    byte-identical to the pre-config builder. Pass ``shot_arm_config(...)`` (or any
    ``OpportunityConfig``) to re-target treatment/outcome/domain/extractor.

    Examples
    --------
    >>> import pandas as pd
    >>> from tests.causal._fixtures import META, WIDE, actions, frames  # doctest: +SKIP
    >>> f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})  # doctest: +SKIP
    >>> build_opportunities(f, actions([]), home_team_id=5, model_metadata=META).shape[0]  # doctest: +SKIP
    1
    >>> shot_arm = build_opportunities(  # doctest: +SKIP
    ...     f, actions([]), home_team_id=5, model_metadata=META, config=shot_arm_config(META)
    ... )
    """
    cfg = xcross_config(model_metadata) if config is None else config
    if cfg.domain not in ("wide_area", "attacking_third"):
        raise ValueError(f"OpportunityConfig.domain must be 'wide_area' or 'attacking_third', got {cfg.domain!r}")
    if cfg.extractor not in ("xcross", "xs"):
        raise ValueError(f"OpportunityConfig.extractor must be 'xcross' or 'xs', got {cfg.extractor!r}")
    carrier_params = dict(model_metadata.get("carrier_params", {}))
    carrier = infer_ball_carrier(frames, **carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)
    score_fn = None
    if _has_results(actions) and home_team_id is not None:
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        score_fn = _build_score_lookup(actions, home_team_id)

    spells: list[dict] = []
    for (gid, per), g in poss.groupby(["game_id", "period_id"], sort=False):
        g = g.sort_values(["time_seconds", "frame_id"])
        frame_keys = list(dict.fromkeys(zip(g["frame_id"].tolist(), g["time_seconds"].tolist(), strict=True)))
        spell: dict | None = None
        for fid, t in frame_keys:
            grp = g[g["frame_id"] == fid]
            team, goal_x, in_dom = _frame_domain_state(grp, goal_map, gid, per, advance_m, cfg)
            if (  # spell continues iff same team, still in domain, under the dedup cap (R2-L2: same_id)
                spell is not None
                and in_dom
                and same_id(team, spell["team"])
                and (t - spell["entry_time"]) <= cfg.max_spell_seconds
            ):
                spell["end_time"], spell["end_frame_id"] = float(t), fid
                continue
            if spell is not None:
                spells.append(spell)
                spell = None
            if in_dom:
                spell = dict(
                    gid=gid,
                    per=per,
                    team=team,
                    goal_x=goal_x,
                    grp=grp,
                    entry_frame_id=fid,
                    entry_time=float(t),
                    end_time=float(t),
                    end_frame_id=fid,
                )
        if spell is not None:
            spells.append(spell)

    rows = [_row(sp, actions, cfg, score_fn, home_team_id) for sp in spells]
    cols = list(cfg.confounders) + list(cfg.gk_block) + _PROV_COLS + ["Z", "Y"]
    # The column list is EXPLICIT (it fixes ordering and drops anything unregistered), so a config
    # that emits extra columns must extend it -- otherwise `_row` builds them and `pd.DataFrame`
    # silently discards them. Caught by the end-to-end build test, never by the unit tests on
    # `_partition_from_distances`, which is exactly why that test exists.
    if cfg.emit_outcome_partition:
        cols += ["Y_attempt", "Y_close_attempt", "Y_far_attempt", "score_differential"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _frame_domain_state(grp, goal_map, gid, per, advance_m, cfg):
    in_poss = grp["team_in_possession"].dropna()
    if in_poss.empty:
        return None, None, False
    poss_team = in_poss.iloc[0]
    ball = grp[grp["is_ball"]]
    if "ball_state" in grp.columns and len(ball) and str(ball["ball_state"].iloc[0]) == "dead":
        return poss_team, None, False
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, poss_team)]
    if not defending:
        return poss_team, None, False
    goal_x = goal_map.get((gid, per, defending[0]))
    if goal_x is None:
        return poss_team, None, False
    bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
    by = float(ball["y"].iloc[0]) if len(ball) else np.nan
    if cfg.domain == "attacking_third":  # advanced only; no wide-corridor requirement
        return poss_team, goal_x, bool(abs(bx - goal_x) <= advance_m)
    return poss_team, goal_x, _in_wide_area(bx, by, goal_x, advance_m)


def _row(sp, actions, cfg, score_fn, home_team_id) -> dict:
    grp, gid, per, team, goal_x = sp["grp"], sp["gid"], sp["per"], sp["team"], sp["goal_x"]
    carrier_s = grp["ball_carrier_player_id"].dropna()
    carrier_pid = carrier_s.iloc[0] if not carrier_s.empty else None
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, team)]
    sd = np.nan
    if score_fn is not None:
        # R2-L1: _build_score_lookup returns a _zero callback when no goals -> raw is never None/NaN.
        raw = score_fn(gid, sp["entry_time"])  # home - away
        sd = float(raw) if same_id(team, home_team_id) else -float(raw)
    feats = _extract_row(cfg, grp, gk_team_id=defending[0], goal_x=goal_x, carrier_pid=carrier_pid, sd=sd)
    row = {c: float(feats[c]) for c in list(cfg.confounders) + list(cfg.gk_block)}
    entry = sp["entry_time"]
    if cfg.treatment_covariate is not None:
        # D5: covariate-threshold treatment. No treatment ACTION exists, so there is no anchor to
        # inherit -- `_resolve_anchor` puts BOTH arms on the spell entry.
        z, t_anchor = _label_treatment_covariate(feats, cfg.treatment_covariate, cfg.treatment_threshold_m), None
    else:
        z, t_anchor = _label_treatment(actions, gid, per, team, cfg, entry, sp["end_time"])
    anchor = _resolve_anchor(z=z, t_anchor=t_anchor, entry=entry)
    row.update(
        game_id=gid,
        period_id=per,
        entry_frame_id=sp["entry_frame_id"],
        entry_time=entry,
        end_time=sp["end_time"],
        spell_duration_seconds=sp["end_time"] - entry,
        possessing_team=team,
        carrier_resolved=carrier_pid is not None,
        Z=z,
        Y=_label_outcome(actions, gid, per, team, anchor, cfg),
    )
    if cfg.emit_outcome_partition:
        d = _outcome_distances(actions, gid, per, team, anchor, cfg)
        y_att, y_close, y_far = _partition_from_distances(d, float(cfg.outcome_max_distance_m or 16.5))
        row.update(Y_attempt=y_att, Y_close_attempt=y_close, Y_far_attempt=y_far)
        # `sd` is ALREADY computed above and available for every config -- `score_fn` is built
        # whenever the actions carry results, regardless of extractor. It is simply never EMITTED,
        # because the xS extractor adapter ignores its `sd` argument (xcross takes
        # score_differential, xS does not). Emitting it here POPULATES Layer 2's confounder rather
        # than leaving it all-NaN, which would reach `fit_propensity` and die as
        # "Input X contains NaN" (measured) during the run.
        row.update(score_differential=sd)
    return row


def _team_period_action_times(actions, gid, per, team, type_names) -> np.ndarray:
    type_ids = {_spc.actiontype_id[n] for n in type_names}
    sel = (  # ids_match: dtype-safe action<->frame team/game id seam (ADR-019)
        ids_match(actions["game_id"], gid)
        & (actions["period_id"] == per)
        & ids_match(actions["team_id"], team)
        & actions["type_id"].isin(type_ids)
    )
    return np.sort(actions.loc[sel, "time_seconds"].to_numpy(dtype=float))


def _covariate_depth(feats) -> float:
    """Goal-relative DEPTH (x) of the keeper, from the shipped POLAR GK block.

    TF-19 spec §6.4 registers the binarisation at goal-relative *x* = 16.5 m (the penalty-area line),
    but the shipped block is polar (``gk_block=("GK_r","GK_theta")``). These agree ONLY on the goal's
    centre line and diverge off-centre, so thresholding ``GK_r`` directly would silently mis-assign
    treatment for wide spells while looking entirely reasonable. ``_xshot_occurrence`` builds
    ``gk_r = hypot(gkx, gky - GOAL_Y)`` and ``gk_theta = atan2(gky - GOAL_Y, gkx)``, so
    ``gkx == GK_r * cos(GK_theta)`` identically.
    """
    return float(feats["GK_r"]) * float(np.cos(float(feats["GK_theta"])))


_COVARIATES = {"gk_depth_x": _covariate_depth}


def _label_treatment_covariate(feats, covariate: str, threshold: float) -> int:
    """``Z = 1`` when the covariate is AT OR BEYOND the threshold.

    For ``gk_depth_x`` at 16.5 m that means **treated == the keeper is ADVANCED beyond the
    penalty-area line** (further from his own goal). Stated this way deliberately: "deep" in football
    means close to one's OWN goal -- the CONTROL arm here -- and the treated arm's identity flows into
    the sign of every ATT this design produces.
    """
    try:
        fn = _COVARIATES[covariate]
    except KeyError:
        raise ValueError(f"unknown treatment_covariate {covariate!r}") from None
    return int(fn(feats) >= float(threshold))


def _resolve_anchor(*, z: int, t_anchor: float | None, entry: float) -> float:
    """Entry anchors BOTH arms when there is no treatment action (D5).

    The action path takes its anchor from the treatment action; a covariate treatment has none, so
    without this a treated row would take ``anchor=None`` and the outcome window would explode. It
    also removes the treated-vs-control time shift this module's docstring flags for the action arms.
    """
    return float(t_anchor) if (z and t_anchor is not None) else float(entry)


def _label_treatment(actions, gid, per, team, cfg, entry, end_time) -> tuple[int, float | None]:
    hi = min(entry + cfg.exposure_window_seconds, end_time)  # R3-M1: clamp the Z-window to possession continuity
    ts = _team_period_action_times(actions, gid, per, team, cfg.treatment_type_names)
    win = ts[(ts > entry) & (ts <= hi)]
    return (1, float(win[0])) if len(win) else (0, None)


_GOAL_XY = (105.0, 34.0)  # SPADL action-LTR: the attacked goal centre, for BOTH teams


def _outcome_distance_m(start_x: float, start_y: float) -> float:
    """Distance from an outcome action's SPADL origin to the attacked goal centre (action-LTR)."""
    return float(np.hypot(_GOAL_XY[0] - float(start_x), _GOAL_XY[1] - float(start_y)))


def _partition_from_distances(distances: np.ndarray, d_max: float) -> tuple[int, int, int]:
    """``(Y_attempt, Y_close_attempt, Y_far_attempt)`` from ONE set of in-window outcome distances.

    ``Y_far := Y_attempt AND NOT Y_close`` is the registered PARTITION (TF-19 spec §6.4 N4), NOT "an
    attempt beyond D": under the looser reading a spell holding BOTH a close and a far attempt would
    count in both indicators, so ``ATT(close) + ATT(far) != ATT(attempt)`` and the coherence check
    §6.4 relies on would be unlicensed arithmetic. Computing all three from a single pass is also
    what guarantees identical row masks across the three outcomes.
    """
    if distances.size == 0:
        return (0, 0, 0)
    close = int(bool((distances <= d_max).any()))
    return (1, close, int(not close))


def _outcome_distances(actions, gid, per, team, anchor, cfg) -> np.ndarray:
    """In-window outcome actions' distances to the attacked goal centre (empty if none)."""
    type_ids = {_spc.actiontype_id[n] for n in cfg.outcome_type_names}
    sel = (
        ids_match(actions["game_id"], gid)
        & (actions["period_id"] == per)
        & ids_match(actions["team_id"], team)
        & actions["type_id"].isin(type_ids)
    )
    if cfg.outcome_result_ids is not None:
        sel &= actions["result_id"].isin(cfg.outcome_result_ids)
    sub = actions.loc[sel]
    ts = sub["time_seconds"].to_numpy(dtype=float)
    # Anchor-inclusive (shot arm) vs legacy strictly-post (xCross) -- P1: an own-result
    # 'None' window is banned (control Y would be structurally 0).
    in_window = (ts >= anchor) if cfg.outcome_window_anchor_inclusive else (ts > anchor)
    keep = in_window & (ts <= anchor + cfg.outcome_window_seconds)
    if not keep.any():
        return np.empty(0, dtype=float)
    xs = sub.loc[keep, "start_x"].to_numpy(dtype=float)
    ys = sub.loc[keep, "start_y"].to_numpy(dtype=float)
    return np.hypot(_GOAL_XY[0] - xs, _GOAL_XY[1] - ys)


def _label_outcome(actions, gid, per, team, anchor, cfg) -> int:
    d = _outcome_distances(actions, gid, per, team, anchor, cfg)
    if cfg.outcome_max_distance_m is None:
        return int(d.size > 0)  # legacy: presence only, byte-identical to the pre-D3 form
    return int(bool((d <= cfg.outcome_max_distance_m).any()))
