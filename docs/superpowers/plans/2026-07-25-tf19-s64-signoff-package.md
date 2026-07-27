# TF-19 §6.4 sign-off package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make TF-19 spec §6.4 signable — build the plasmode power simulator (ICC + ATT modes), land Layer 2's
design in the causal builder, derive the two placeholder constants, split verdict from routing, and run the
shipped-but-unrun §3.3 shot-arm harness.

**Architecture:** Three homes by import direction — domain-free ICC power in `silly_kicks/_group_metrics.py`,
generic ATT power in a NEW PUBLIC `silly_kicks/causal/power.py`, GKDV-specific constants and injection in
`silly_kicks/gkdv/_validate.py`. The causal builder gains a covariate-threshold treatment axis alongside its
existing action-occurrence one, default-off and byte-identical for both shipped configs.

**Tech Stack:** Python 3.10+, numpy, pandas; pytest; ruff + pyright (CI `lint`).

**Design source:** `docs/superpowers/specs/2026-07-25-tf19-s64-signoff-package-design.md` (F1–F7, D1–D5, two
cross-session review rounds applied). Section refs below (`spec §N`) point there for rationale — this plan is
the *how*. **Read spec §5.1's FIREWALL and §9 before starting**: this cycle builds the machinery that could
answer TF-19's open question ahead of the sign-off meant to authorise it, and the guard against that is a test.

**Branch:** `feat/tf19-signoff-package` (already created, based on `4ab63fe`/4.61.0).

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `silly_kicks/_group_metrics.py` | + ICC power curve (domain-free: values/groups/blocks). Docstring fence corrected. | Modify |
| `silly_kicks/causal/power.py` | Generic plasmode ATT power loop + `InjectionSpec` recipe + firewall guard (accepts no outcome vector). | **Create (public)** |
| `silly_kicks/causal/opportunities.py` | + covariate-threshold treatment, entry-anchor rule, `outcome_max_distance_m`, outcome partition, `layer2_config()`. | Modify |
| `silly_kicks/causal/_confounders.py` | Layer 2's tracking-confounder join (TF-14 line + `bekkers_pi`), computed FRESH from frames. | **Create (private)** |
| `silly_kicks/causal/__init__.py` | Export `power` surface + `layer2_config`. | Modify |
| `silly_kicks/gkdv/_validate.py` | + `ATT_RELATIVE_ANCHORS`, `LAYER3_HEADROOM_RANGE_FRACTION`, `N_MIN_MATCHED`, GKDV injection. | Modify |
| `silly_kicks/tracking/_model_eval.py` | + `regate_routing` + `REGATE_ROUTING_VALUES`. `regate_verdict` UNCHANGED. | Modify |
| `scripts/derive_opengoal_range.py` | Layer 3 derivation driver (marginal `openGoal` distribution). | **Create** |
| `scripts/run_signoff_power.py` | DGX driver: ICC + ATT power curves → artifacts. | **Create** |
| `tests/causal/test_power.py`, `test_covariate_treatment.py`, `test_outcome_partition.py`, `test_confounders.py` | New gates. | Create |
| `tests/tracking/test_regate_routing.py`, `tests/test_group_metrics_power.py` | New gates. | Create |

**Ordering rationale:** Tasks 1–2 are independent and land first (fast green). Tasks 3–6 build the Layer 2
design bottom-up. Task 7 is the firewall — **it must land before any ATT power code can run on real spells**.
Tasks 8–11 are derivations/runs. Task 12–13 are record + release.

---

## Task 0: Verify the starting state

- [ ] **Step 1: Confirm branch and clean tree**

```bash
git -C "D:/Development/karstenskyt__silly-kicks_part-deux" branch --show-current
git -C "D:/Development/karstenskyt__silly-kicks_part-deux" status --short
```
Expected: `feat/tf19-signoff-package`, and only the untracked spec + this plan.

- [ ] **Step 2: Baseline the suite**

```bash
.venv312/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q --benchmark-skip 2>&1 | tail -5
```
Expected: all pass. Record the count — every later "still green" claim compares against it.

---

## Task 1: `regate_routing` — split routing from verdict (spec §7, F4)

**Files:** `silly_kicks/tracking/_model_eval.py`; `tests/tracking/test_regate_routing.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_regate_routing.py
import itertools
import pytest
from silly_kicks.tracking._model_eval import (
    REGATE_ROUTING_VALUES, _ENTANGLEMENT, _PROBE_VERDICTS, regate_routing, regate_verdict,
)


def test_gated_clean_fail_no_longer_routes_to_gk_feature_engineering():
    """The pre-registered disclosure: `gated_clean_fail` must stop hard-coding H1."""
    assert regate_routing("gated_clean_fail") == "pending_layer2"


@pytest.mark.parametrize("verdict", sorted({
    regate_verdict(arm=a, probe_verdict=p, entanglement=e)
    for a, p, e in itertools.product(("shot", "cross"), _PROBE_VERDICTS, _ENTANGLEMENT)
}))
def test_every_reachable_verdict_has_a_routing_in_the_closed_vocabulary(verdict):
    assert regate_routing(verdict) in REGATE_ROUTING_VALUES


def test_unknown_verdict_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown verdict"):
        regate_routing("not_a_verdict")


def test_regate_verdict_is_byte_identical_over_every_input_combination():
    """Golden pin: no recorded verdict may move. 4.60.0's `joins_with_caveat` and 4.51.0's
    `gated_clean_fail` are published in metrics.json artifacts."""
    got = {
        (a, p, e): regate_verdict(arm=a, probe_verdict=p, entanglement=e)
        for a, p, e in itertools.product(("shot", "cross"), _PROBE_VERDICTS, _ENTANGLEMENT)
    }
    assert got[("shot", "pass", "inside_band")] == "joins_with_caveat"
    assert got[("shot", "pass", "clears")] == "joins"
    assert got[("cross", "fail", "inside_band")] == "gated_clean_fail"
    assert got[("shot", "no_valid_placebo", "clears")] == "unmeasurable_at_dose"
    assert got[("shot", "instrument_invalid", "clears")] == "verdict_void"
    assert got[("shot", "band_pass_flat_dose_response", "clears")] == "gated_flat_dose_response"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/tracking/test_regate_routing.py -q`
Expected: `ImportError: cannot import name 'REGATE_ROUTING_VALUES'`

- [ ] **Step 3: Implement**

Append to `silly_kicks/tracking/_model_eval.py`, immediately after `regate_verdict`:

```python
#: Closed routing vocabulary (the DAS_SOURCE_VALUES pattern): a consumer CASE/enum pins to this set.
REGATE_ROUTING_VALUES: tuple[str, ...] = (
    "pending_layer2",
    "gk_feature_engineering",
    "fix_the_instrument",
    "corpus_or_sampling",
    "joins_the_metric",
)

_ROUTING: dict[str, str] = {
    # ADR-037's routing rule AMENDED here, against its own pre-registered disclosure (spec §6.4
    # Registration disclosures: "`regate_verdict`'s routing needs amending ... since that hard-codes
    # H1"). `gated_clean_fail` previously routed UNCONDITIONALLY to GK feature engineering, which
    # made H2 unreachable by construction. H2 stays reachable ONLY through row 7 of
    # `gkdv_discrimination_verdict` (PR-3b); this opens the channel without pre-empting the decider.
    "gated_clean_fail": "pending_layer2",
    "gated_flat_dose_response": "pending_layer2",
    "unmeasurable_at_dose": "corpus_or_sampling",
    "verdict_void": "fix_the_instrument",
    "joins": "joins_the_metric",
    "joins_with_caveat": "joins_the_metric",
}


def regate_routing(verdict: str) -> str:
    """What to DO about a `regate_verdict` result -- deliberately a separate function.

    The verdict answers "what did the probe say"; the routing answers "what should we do about it",
    and only the second may depend on Layer 2. Conflating them is what hard-coded H1.

    Examples
    --------
    >>> regate_routing("gated_clean_fail")
    'pending_layer2'
    """
    try:
        return _ROUTING[verdict]
    except KeyError:
        raise ValueError(f"regate_routing: unknown verdict {verdict!r}") from None
```

- [ ] **Step 4: Verify pass**

Run: `.venv312/Scripts/python.exe -m pytest tests/tracking/test_regate_routing.py -q`
Expected: PASS (10 tests).

---

## Task 2: ICC power curve + the `_group_metrics` fence correction (spec §5.3, F2)

**Files:** `silly_kicks/_group_metrics.py`; `tests/test_group_metrics_power.py` (create)

- [ ] **Step 1: Write the failing tests — BOTH sides, plus the clustering claim**

```python
# tests/test_group_metrics_power.py
import numpy as np
import pytest
from silly_kicks._group_metrics import icc_power_curve


def _corpus(n_groups=30, per_group=40, seed=0):
    """Keepers must SPAN matches, and matches must hold more than one keeper.

    MEASURED, do not "simplify" this: with `blocks` 1:1 with `groups` (one keeper per match) the
    block-permutation null is a pure RELABELLING of an identical partition, and `icc_one_way` is
    label-invariant -- observed ICC 0.213228 and all five nulls 0.213228, so nothing is ever
    detectable. Spec §6.1's own floor language ("for a single-match keeper, keeper == match")
    excludes exactly this shape.

    A 2-keepers-per-block fixture does NOT fix it either: the null stops equalling the observed
    value but stops VARYING (constant 0.162025), so the test goes green on a fixed comparison that
    proves nothing -- a false pass. Keepers spanning matches is what makes the null vary.
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"k{i}" for i in range(n_groups)], per_group)
    # each keeper's observations are chopped across many matches; each match holds 2 keepers
    blocks = np.array([f"m{(i // 10) % (n_groups * per_group // 20)}" for i in range(n_groups * per_group)])
    values = rng.normal(0.0, 1.0, size=n_groups * per_group)
    return values, groups, blocks


def test_power_is_high_at_a_large_injected_effect():
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.30,), n_replicates=40, rng_seed=1)
    assert out["power"][0.30] >= 0.80


def test_power_collapses_to_alpha_at_zero_injected_effect():
    """The other side. A one-sided 'power is high' assertion passes identically when the
    simulator silently produces nothing."""
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.0,), n_replicates=200, rng_seed=1)
    assert out["power"][0.0] <= 0.15  # alpha=0.05 + Monte-Carlo slack at 200 replicates


def test_injection_measurably_moves_the_data_non_vacuity():
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.30,), n_replicates=10, rng_seed=1)
    assert out["mean_observed_icc"][0.30] > out["mean_observed_icc_at_zero"] + 0.05


def test_block_structure_inflates_the_NULL_which_is_what_plasmode_means():
    """'Plasmode, not i.i.d.' has teeth only if block structure changes the null. Assert on the null
    MEAN -- the only form of this claim that is stable.

    MEASURED over 20 seeds x 40 permutations, do not "improve" this into another statistic:
        null MEAN clustered > iid : 20/20 at anchor 0.02, 20/20 at anchor 0.30
        null P95  clustered > iid : 13/20 at anchor 0.02, 20/20 at anchor 0.30
        POWER     iid >= clustered: FAILS (0.000 vs 0.013 -- both on the noise floor)
    Block permutation reassigns whole CHUNKS, so the permuted grouping retains real clustering and
    the null sits higher; i.i.d. permutation fully randomises and collapses it toward zero. That is
    a structural property of the null, which is why it survives at both anchors, while power and
    p95 comparisons are noise-dominated at the small ones.
    """
    values, groups, real_blocks = _corpus()
    iid_blocks = np.array([f"b{i}" for i in range(len(values))])  # every obs its own block
    clustered = icc_power_curve(values, groups, real_blocks, anchors=(0.02,), n_replicates=40, rng_seed=3)
    iid = icc_power_curve(values, groups, iid_blocks, anchors=(0.02,), n_replicates=40, rng_seed=3)
    assert clustered["mean_null_icc"][0.02] > iid["mean_null_icc"][0.02]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_group_metrics_power.py -q`
Expected: `ImportError: cannot import name 'icc_power_curve'`

- [ ] **Step 3: Correct the fence docstring FIRST (spec §4)**

In `silly_kicks/_group_metrics.py`, replace lines 30–32:

```python
SCOPE: this module holds domain-free grouped statistics -- ICC, spread, and the ICC POWER
SIMULATOR. The power sim was originally deferred to PR-3b; that deferral was OVERRIDDEN because
spec §6.1 registers the power curve as a PRECONDITION on the ICC gate ("the gate is registered only
if detection at the anchor ... is >= 0.8"), so `ICC_ANCHORS` shipped promising a curve nothing could
produce. The permutation BAND remains PR-3b and is deliberately absent.
```

- [ ] **Step 4: Implement the power curve**

Append to `silly_kicks/_group_metrics.py`:

```python
def _inject_group_effect(values, groups, target_icc, rng):
    """Group-centre the values (removing any existing group structure), then add a group-level
    effect sized so the between-group variance share equals `target_icc`. Returns new values."""
    vals = np.asarray(values, dtype=float).copy()
    keys, inv = np.unique(np.asarray(groups), return_inverse=True)
    means = np.zeros(len(keys))
    for i in range(len(keys)):
        means[i] = vals[inv == i].mean()
    centred = vals - means[inv]
    within_var = float(np.var(centred, ddof=0))
    if target_icc <= 0.0 or within_var <= 0.0:
        return centred
    between_var = target_icc / (1.0 - target_icc) * within_var
    effects = rng.normal(0.0, np.sqrt(between_var), size=len(keys))
    return centred + effects[inv]


def _permute_groups_by_block(groups, blocks, rng):
    """Match-block label permutation: shuffle which GROUP label attaches to each BLOCK, so
    within-block clustering survives. An i.i.d. shuffle of observations would not."""
    groups = np.asarray(groups)
    blocks = np.asarray(blocks)
    bkeys = np.unique(blocks)
    # one representative group label per block, then permute the block -> label map
    reps = np.array([groups[blocks == b][0] for b in bkeys])
    permuted = rng.permutation(reps)
    mapping = dict(zip(bkeys.tolist(), permuted.tolist()))
    return np.array([mapping[b] for b in blocks.tolist()])


def icc_power_curve(values, groups, blocks, *, anchors, n_replicates, alpha=0.05, rng_seed=0):
    """Plasmode power to DETECT a keeper-level ICC at each anchor (spec §5.3; TF-19 §6.1).

    Real values, real clustering, injected known effects -- never i.i.d. simulation, which would
    inherit none of the clustering and could pass while the real instrument is simultaneously
    underpowered and anti-conservative.

    Returns a dict with ``power`` (anchor -> detected fraction), ``mean_observed_icc``,
    ``mean_observed_icc_at_zero`` (the non-vacuity reference), ``n_replicates`` and ``alpha``.
    """
    rng = np.random.default_rng(rng_seed)
    power, mean_icc, mean_null = {}, {}, {}
    zero_iccs = []
    for anchor in anchors:
        detected, obs, nulls = 0, [], []
        for _ in range(int(n_replicates)):
            injected = _inject_group_effect(values, groups, float(anchor), rng)
            observed = icc_one_way(injected, np.asarray(groups))
            null = np.array([
                icc_one_way(injected, _permute_groups_by_block(groups, blocks, rng))
                for _ in range(30)
            ])
            crit = float(np.quantile(null, 1.0 - alpha))
            detected += int(observed > crit)
            obs.append(observed)
            nulls.append(float(np.mean(null)))
        power[anchor] = detected / float(n_replicates)
        mean_icc[anchor] = float(np.mean(obs))
        # Reported so the block-structure claim is assertable: an i.i.d. blocking collapses this
        # toward zero while a real one holds it up (20/20 seeds at both anchors).
        mean_null[anchor] = float(np.mean(nulls))
        if float(anchor) == 0.0:
            zero_iccs = obs
    if not zero_iccs:
        zero_iccs = [
            icc_one_way(_inject_group_effect(values, groups, 0.0, rng), np.asarray(groups))
            for _ in range(10)
        ]
    return {
        "power": power,
        "mean_observed_icc": mean_icc,
        "mean_null_icc": mean_null,
        "mean_observed_icc_at_zero": float(np.mean(zero_iccs)),
        "n_replicates": int(n_replicates),
        "alpha": float(alpha),
    }
```

- [ ] **Step 5: Verify pass**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_group_metrics_power.py -q`
Expected: PASS (4 tests).

**Why the smoke uses anchor 0.30 and not the registered 0.015–0.026.** On a 1200-observation
fixture the observed ICC at 0.02 is inside the noise (**measured −0.0112**, i.e. negative), so a CI
test at the registered anchors would assert on noise. The smoke therefore proves the *machinery*
at a large anchor; the registered anchors are only meaningful on the real corpus. Corollary worth
holding before Task 10: **real-corpus power at 0.015–0.026 may genuinely come back below 0.8**, and
§6.1 says that is a result to report, not a number to tune toward.

**Troubleshooting, pointed at the right test.** If `test_power_is_high_at_a_large_injected_effect`
returns ~0, the fixture's block structure has collapsed — print the observed ICC alongside the
permutation nulls and check whether they are IDENTICAL. If they are, the permutation is relabelling a
fixed partition, and the fix is the FIXTURE (keepers must span matches), not the null. Do **not**
lower the 0.80 bound. If `test_power_collapses_to_alpha_at_zero_injected_effect` fails high, that
one is a genuinely mis-specified null.

---

## Task 3: `outcome_max_distance_m` + the outcome partition (D3, spec §5.1)

**Files:** `silly_kicks/causal/opportunities.py`; `tests/causal/test_outcome_partition.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_outcome_partition.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.causal.opportunities import OpportunityConfig, shot_arm_config, xcross_config


def test_legacy_configs_are_byte_identical_with_the_new_field_defaulted():
    """D3 is additive: a default alone is not evidence -- pin both shipped configs."""
    for cfg in (xcross_config({}), shot_arm_config({})):
        assert cfg.outcome_max_distance_m is None
        assert cfg.emit_outcome_partition is False


def test_close_outcome_uses_action_ltr_distance_to_the_attacked_goal():
    from silly_kicks.causal.opportunities import _outcome_distance_m
    # SPADL action-LTR: attacked goal centre is (105, 34)
    assert _outcome_distance_m(105.0, 34.0) == pytest.approx(0.0)
    assert _outcome_distance_m(88.5, 34.0) == pytest.approx(16.5)
    assert _outcome_distance_m(105.0, 17.5) == pytest.approx(16.5)


def test_partition_is_exact_by_construction_not_by_two_passes():
    """Y_far := Y_attempt AND NOT Y_close, computed from ONE labelling pass so the three
    indicators share identical row masks (review R2 #4)."""
    from silly_kicks.causal.opportunities import _partition_from_distances
    # A spell containing BOTH a close and a far attempt is classified CLOSE, not both. Y_far is
    # `Y_attempt AND NOT Y_close` (parent spec N4) -- under the looser "an attempt beyond D" reading
    # this row would score (1, 1, 1), the indicators would overlap, and
    # ATT(close) + ATT(far) == ATT(attempt) would fail. If this assertion ever goes red, fix the
    # CALLER, never this literal.
    assert _partition_from_distances(np.array([10.0, 30.0]), 16.5) == (1, 1, 0)
    assert _partition_from_distances(np.array([30.0]), 16.5) == (1, 0, 1)
    assert _partition_from_distances(np.array([10.0]), 16.5) == (1, 1, 0)
    assert _partition_from_distances(np.array([]), 16.5) == (0, 0, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/causal/test_outcome_partition.py -q`
Expected: `AttributeError: 'OpportunityConfig' object has no attribute 'outcome_max_distance_m'`

- [ ] **Step 3: Implement**

In `silly_kicks/causal/opportunities.py`, add two fields to `OpportunityConfig` (after `extractor`,
line 82) — both defaulted so every shipped config is unchanged:

```python
    outcome_max_distance_m: float | None = None  # D3: None = no spatial filter (legacy)
    emit_outcome_partition: bool = False  # Layer 2: emit Y_attempt / Y_close_attempt / Y_far_attempt
```

Add module-level helpers near `_label_outcome`:

```python
_GOAL_XY = (105.0, 34.0)  # SPADL action-LTR: the attacked goal centre, both teams


def _outcome_distance_m(start_x: float, start_y: float) -> float:
    """Distance from an outcome action's SPADL origin to the attacked goal centre (action-LTR)."""
    return float(np.hypot(_GOAL_XY[0] - float(start_x), _GOAL_XY[1] - float(start_y)))


def _partition_from_distances(distances: np.ndarray, d_max: float) -> tuple[int, int, int]:
    """(Y_attempt, Y_close_attempt, Y_far_attempt) from ONE set of in-window outcome distances.

    `Y_far := Y_attempt AND NOT Y_close` is the registered PARTITION (spec §6.4 N4), not "an attempt
    beyond D" -- under the looser reading a spell with both a close AND a far attempt would count in
    both indicators and ATT(close) + ATT(far) != ATT(attempt).
    """
    if distances.size == 0:
        return (0, 0, 0)
    close = int(bool((distances <= d_max).any()))
    return (1, close, int(not close))
```

Rewrite `_label_outcome` to return the in-window distances, and add a thin back-compatible wrapper:

```python
def _outcome_distances(actions, gid, per, team, anchor, cfg) -> np.ndarray:
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
        return int(d.size > 0)
    return int(bool((d <= cfg.outcome_max_distance_m).any()))
```

In `_row` (line ~281), emit the partition when configured:

```python
        Y=_label_outcome(actions, gid, per, team, anchor, cfg),
    )
    if cfg.emit_outcome_partition:
        d = _outcome_distances(actions, gid, per, team, anchor, cfg)
        y_att, y_close, y_far = _partition_from_distances(d, float(cfg.outcome_max_distance_m or 16.5))
        row.update(Y_attempt=y_att, Y_close_attempt=y_close, Y_far_attempt=y_far)
        # `sd` is ALREADY computed above (`_row` lines 261-265) and is available for every config --
        # `score_fn` is built whenever the actions carry results, regardless of extractor. It is
        # simply never emitted, because the xS extractor adapter ignores its `sd` argument (the
        # adapter docstring at :140 says xcross takes score_differential and xS does not). Emitting
        # it here POPULATES Layer 2's confounder rather than leaving it all-NaN -- which would reach
        # `fit_propensity` and die as "Input X contains NaN" (MEASURED) during the DGX run.
        row.update(score_differential=sd)
    return row
```

- [ ] **Step 4: Verify pass + legacy identity**

```bash
.venv312/Scripts/python.exe -m pytest tests/causal/ -q
```
Expected: all pass, including the pre-existing `test_builder_surface.py` legacy-identity checks.

---

## Task 4: Covariate-threshold treatment axis + entry anchor (D5, F7, spec §5.1)

**Files:** `silly_kicks/causal/opportunities.py`; `tests/causal/test_covariate_treatment.py` (create)

- [ ] **Step 1: Write the failing tests — including the vacuity guard**

```python
# tests/causal/test_covariate_treatment.py
import numpy as np
import pytest
from silly_kicks.causal.opportunities import (
    OpportunityConfig, _covariate_depth, _label_treatment_covariate, shot_arm_config, xcross_config,
)


def test_action_path_is_untouched_when_the_covariate_axis_is_unset():
    for cfg in (xcross_config({}), shot_arm_config({})):
        assert cfg.treatment_covariate is None
        assert cfg.treatment_threshold_m is None


@pytest.mark.parametrize("gk_r,gk_theta,expected_z", [(20.0, 0.0, 1), (10.0, 0.0, 0)])
def test_treatment_binarises_at_the_penalty_area_line(gk_r, gk_theta, expected_z):
    feats = {"GK_r": gk_r, "GK_theta": gk_theta}
    assert _label_treatment_covariate(feats, "gk_depth_x", 16.5) == expected_z


def test_depth_is_the_x_component_not_the_radius():
    """Vacuity guard: a test using only on-axis keepers passes identically if GK_r is thresholded
    directly. This wide case is the one that discriminates -- r=20 but x=14.1, so a keeper 20 m
    from goal on a diagonal is INSIDE the 16.5 m depth line."""
    feats = {"GK_r": 20.0, "GK_theta": np.pi / 4}
    assert _covariate_depth(feats) == pytest.approx(20.0 * np.cos(np.pi / 4))
    assert _covariate_depth(feats) < 16.5
    assert _label_treatment_covariate(feats, "gk_depth_x", 16.5) == 0


def test_covariate_treated_rows_anchor_at_ENTRY_not_at_none():
    """`_row` computes `anchor = t_anchor if z else entry`. A covariate treatment has NO anchor
    action, so a treated row would take anchor=None and the outcome window would explode."""
    from silly_kicks.causal.opportunities import _resolve_anchor
    assert _resolve_anchor(z=1, t_anchor=None, entry=12.5) == 12.5
    assert _resolve_anchor(z=1, t_anchor=30.0, entry=12.5) == 30.0
    assert _resolve_anchor(z=0, t_anchor=None, entry=12.5) == 12.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/causal/test_covariate_treatment.py -q`
Expected: `ImportError: cannot import name '_covariate_depth'`

- [ ] **Step 3: Implement**

Add two more `OpportunityConfig` fields (defaulted — the action path stays the default):

```python
    treatment_covariate: str | None = None  # D5: None = action-occurrence treatment (legacy)
    treatment_threshold_m: float | None = None
```

Add near `_label_treatment`:

```python
def _covariate_depth(feats) -> float:
    """Goal-relative DEPTH (x) of the keeper, from the shipped polar GK block.

    Spec §6.4 registers the binarisation at goal-relative *x* = 16.5 m (the penalty-area line), but
    the shipped block is polar (`gk_block=("GK_r","GK_theta")`). These agree only on the goal's
    centre line. `_xshot_occurrence` builds gk_r = hypot(gkx, gky-GOAL_Y) and
    gk_theta = atan2(gky-GOAL_Y, gkx), so gkx == GK_r * cos(GK_theta) identically.
    """
    return float(feats["GK_r"]) * float(np.cos(float(feats["GK_theta"])))


_COVARIATES = {"gk_depth_x": _covariate_depth}


def _label_treatment_covariate(feats, covariate: str, threshold: float) -> int:
    """Z = 1 when the covariate is AT OR BEYOND the threshold.

    For `gk_depth_x` at 16.5 m that means **treated == the keeper is ADVANCED beyond the
    penalty-area line** (further from his own goal). Stated this way deliberately: "deep" in football
    means close to one's OWN goal, i.e. the CONTROL arm here, and the treated arm's identity flows
    into the sign of every ATT this design produces.
    """
    try:
        fn = _COVARIATES[covariate]
    except KeyError:
        raise ValueError(f"unknown treatment_covariate {covariate!r}") from None
    return int(fn(feats) >= float(threshold))


def _resolve_anchor(*, z: int, t_anchor: float | None, entry: float) -> float:
    """Entry anchors BOTH arms when there is no treatment action (D5). This also removes the
    treated-vs-control time shift the module docstring flags at :11-12 for the action arms."""
    return float(t_anchor) if (z and t_anchor is not None) else float(entry)
```

In `_row`, replace lines 269–270:

```python
    if cfg.treatment_covariate is not None:
        z, t_anchor = _label_treatment_covariate(feats, cfg.treatment_covariate, cfg.treatment_threshold_m), None
    else:
        z, t_anchor = _label_treatment(actions, gid, per, team, cfg, entry, sp["end_time"])
    anchor = _resolve_anchor(z=z, t_anchor=t_anchor, entry=entry)
```

- [ ] **Step 4: Verify pass, then prove legacy identity behaviourally**

```bash
.venv312/Scripts/python.exe -m pytest tests/causal/ -q
```
Expected: all pass. `test_builder_surface.py`'s `config=None` regression check is the byte-identity
gate for the xCross path; the parametrized config test above covers the shot arm.

---

## Task 5: `layer2_config()` (D5, spec §5.1)

**Files:** `silly_kicks/causal/opportunities.py`, `silly_kicks/causal/__init__.py`; `tests/causal/test_covariate_treatment.py`

- [ ] **Step 1: Write the failing test**

```python
def test_layer2_config_registers_the_landmark_design():
    from silly_kicks.causal import layer2_config
    cfg = layer2_config({})
    assert cfg.treatment_covariate == "gk_depth_x"
    assert cfg.treatment_threshold_m == 16.5          # Law-defined, data-independent
    assert cfg.outcome_type_names == ("shot", "shot_freekick", "shot_penalty")
    assert cfg.outcome_result_ids is None             # an ATTEMPT, not a goal
    assert cfg.outcome_max_distance_m == 16.5
    assert cfg.emit_outcome_partition is True
    assert cfg.domain == "attacking_third"
    assert cfg.extractor == "xs"


def test_every_build_confounder_is_actually_emitted_by_the_xs_extractor():
    """`_row` reads each confounder out of the feature dict with a HARD key lookup, so an
    unproduced name is a build-time KeyError, not a NaN."""
    from silly_kicks.causal import layer2_config
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL
    cfg = layer2_config({})
    for name in tuple(cfg.confounders) + tuple(cfg.gk_block):
        assert name in XSHOT_FEATURE_NAMES_FAITHFUL, f"{name} would KeyError in _row"


def test_layer2_config_actually_BUILDS_opportunities(layer2_fixture):
    """The field-only test above passes even when the config cannot be built. This one is the gate:
    it calls the builder, which is where a bad confounder name surfaces."""
    from silly_kicks.causal import build_opportunities, layer2_config
    out = build_opportunities(
        layer2_fixture["frames"], layer2_fixture["actions"],
        home_team_id=layer2_fixture["home_team_id"], model_metadata={},
        config=layer2_config({}),
    )
    assert len(out) > 0, "fixture produced no spells -- the test would be vacuous"
    for col in ("Z", "Y_attempt", "Y_close_attempt", "Y_far_attempt"):
        assert col in out.columns
    assert set(out["Z"].unique()) <= {0, 1}
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: cannot import name 'layer2_config'`

- [ ] **Step 3: Implement**

```python
#: BUILD-TIME confounders: EXTRACTOR-PRODUCED COLUMNS ONLY.
#:
#: `_row` (opportunities.py:267) reads every `cfg.confounders` name straight out of the xS feature
#: dict with a hard key lookup -- `{c: float(feats[c]) for c in ...}` -- so a name the extractor does
#: not emit raises KeyError at BUILD time, not NaN. VERIFIED against
#: `XSHOT_FEATURE_NAMES_FAITHFUL`: `r`, `theta`, `DefDist_0`, `DefDist_1` are PRESENT;
#: `defensive_line_height`, `defensive_line_compactness`, `pressure_on_actor__bekkers_pi`,
#: `score_differential` and `time_remaining_s` are ABSENT (the extractor adapter docstring at
#: opportunities.py:140 states xS takes no `score_differential` at all).
LAYER2_BUILD_CONFOUNDERS = ("r", "theta", "DefDist_0", "DefDist_1")

#: ANALYSIS-TIME design matrix: the build-time set PLUS the columns `causal/_confounders.py` joins on
#: afterwards. This is what `att_power_curve`'s `X` is assembled from -- never `cfg.confounders`.
#: Split deliberately (review R2/HIGH-3 resolution): the tracking confounders are per-spell joins,
#: not extractor features, and forcing them through the extractor contract would mean either a
#: silent NaN-filling `_row` (hiding genuine join failures) or teaching the xS extractor about
#: defensive lines it has no business knowing.
LAYER2_CONFOUNDERS = LAYER2_BUILD_CONFOUNDERS + (
    "defensive_line_height", "defensive_line_compactness",
    "pressure_on_actor__bekkers_pi", "score_differential", "time_remaining_s",
)


def layer2_config(model_metadata: dict) -> OpportunityConfig:
    """TF-19 §6.4 Layer 2: the H1-vs-H2 decider's DESIGN (spec D5).

    Treatment is keeper DEPTH at spell entry binarised at the penalty-area line -- Law-defined and
    data-independent, so the entire decider is untuned. The outcome is an ATTEMPT (contrast
    `shot_arm_config`, whose outcome is a GOAL and whose treatment is roughly this outcome).

    Building this config does NOT run Layer 2: see spec §5.1's FIREWALL.
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
```

Export `layer2_config` + `LAYER2_CONFOUNDERS` from `silly_kicks/causal/__init__.py`'s `__all__`.

- [ ] **Step 4: Verify pass** — `.venv312/Scripts/python.exe -m pytest tests/causal/ -q`

---

## Task 6: The confounder join — **its own gated task** (review R2 #5)

**Files:** `silly_kicks/causal/_confounders.py` (create); `tests/causal/test_confounders.py` (create)

> **Gate:** this is the piece most likely to blow the estimate. If it overruns, the recorded fallback
> (spec header scope note) is to register the anchor rule now and derive `N_min` in PR-3b. Raise it
> with the owner rather than absorbing the overrun silently.

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_confounders.py
import pandas as pd
import pytest
from silly_kicks.causal._confounders import CONFOUNDER_SOURCE, join_layer2_confounders


def test_provenance_is_declared_as_frames_computed():
    """Spec §5.1: a mart-sourced join would hand Layer 2 pre-ADR-045 away-team pressure."""
    assert CONFOUNDER_SOURCE == "frames_computed"


def test_mart_sourced_frames_are_refused():
    spells = pd.DataFrame({"game_id": [1], "period_id": [1], "entry_frame_id": [10], "possessing_team": [5]})
    with pytest.raises(ValueError, match="fct_action_context"):
        join_layer2_confounders(
            spells, frames=None, actions=None, home_team_id=5, source="fct_action_context"
        )


def test_every_layer2_confounder_column_is_present_after_the_join(layer2_fixture):
    from silly_kicks.causal.opportunities import LAYER2_CONFOUNDERS
    out = join_layer2_confounders(**layer2_fixture)
    for col in LAYER2_CONFOUNDERS:
        assert col in out.columns, f"missing confounder {col}"


def test_the_join_does_not_mutate_the_input_spells(layer2_fixture):
    before = layer2_fixture["spells"].copy(deep=True)
    join_layer2_confounders(**layer2_fixture)
    pd.testing.assert_frame_equal(layer2_fixture["spells"], before)
```

Build `layer2_fixture` in `tests/causal/_fixtures.py` by extending the existing `frames`/`spell`
builders with two keepers at differing depths and a defensive line of four defenders.

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: silly_kicks.causal._confounders`

- [ ] **Step 3: Implement**

```python
# silly_kicks/causal/_confounders.py
"""Layer 2's tracking-confounder join (spec §5.1, D5).

PROVENANCE IS REGISTERED: every tracking confounder is computed FRESH from frames. It must NOT be
sourced from `fct_action_context` -- ADR-045/4.55.0 fixed `pressure_on_actor__bekkers_pi` (away-team
velocity re-projection; away values changed, home byte-identical) and the lakehouse
re-materialization of that column is still an open owner action. A mart-sourced join would silently
hand Layer 2's design pre-fix away-team pressure, in a confounder chosen because it is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONFOUNDER_SOURCE = "frames_computed"


def join_layer2_confounders(spells, *, frames, actions, home_team_id, source=CONFOUNDER_SOURCE):
    """Return a NEW spells frame with Layer 2's tracking confounders attached at spell entry."""
    if source != CONFOUNDER_SOURCE:
        raise ValueError(
            f"Layer 2 confounders must be {CONFOUNDER_SOURCE!r}, got {source!r}. "
            "fct_action_context.pressure_on_actor__bekkers_pi is pre-ADR-045 for away teams "
            "until the lakehouse re-materializes; see spec §5.1."
        )
    from silly_kicks.tracking import add_pressure_on_actor, compute_defensive_line

    out = spells.copy()
    # SIGNATURES VERIFIED BY EXECUTION (do not "simplify" these two calls):
    #   compute_defensive_line(frames, *, home_team_id, n=4, adaptive_max_n=5)
    #   add_pressure_on_actor(actions, frames, *, links=None, methods=(...), ...)   <- methods is
    #   PLURAL and a TUPLE; a singular `method="bekkers_pi"` raises TypeError.
    line = compute_defensive_line(frames, home_team_id=home_team_id)
    # COLUMNS VERIFIED (docstring): game_id, period_id, frame_id, team_id, defensive_line_x,
    # back_line_high_x, compactness_x, lateral_width, max_lateral_gap. There is NO
    # `defensive_line_spread` -- compactness is `compactness_x`.
    #
    # The key has FOUR levels and the function "Computes for BOTH teams", so the DEFENDING team must
    # be selected: Layer 2's confounder is the line the ATTACKING spell faces, i.e. the team that is
    # NOT `possessing_team`. Joining on three levels would silently pick an arbitrary team's row.
    keyed = line.set_index(["game_id", "period_id", "frame_id", "team_id"])
    defending = _defending_team_id(out, frames)
    idx = pd.MultiIndex.from_arrays(
        [out["game_id"], out["period_id"], out["entry_frame_id"], defending]
    )
    out["defensive_line_height"] = keyed["defensive_line_x"].reindex(idx).to_numpy()
    out["defensive_line_compactness"] = keyed["compactness_x"].reindex(idx).to_numpy()

    pressed = add_pressure_on_actor(actions, frames, methods=("bekkers_pi",))
    pcol = pressed.set_index("action_id")["pressure_on_actor__bekkers_pi"]
    out["pressure_on_actor__bekkers_pi"] = pcol.reindex(out.get("entry_action_id", pd.Series(dtype=float))).to_numpy()

    out["time_remaining_s"] = _time_remaining(out)
    # score_differential is emitted by `_row` (Task 3) when `emit_outcome_partition` is set -- it is
    # NOT filled here. A NaN-fill would look tolerant and then raise inside `fit_propensity`.
    if "score_differential" not in out.columns:
        raise ValueError(
            "score_differential absent: build the spells with a config that emits it "
            "(layer2_config), or the design matrix will fail in fit_propensity."
        )
    return out


def _defending_team_id(spells, frames) -> pd.Series:
    """The team NOT in possession, per spell -- ADR-019 dtype-safe (`ids_differ`, never raw `!=`)."""
    from silly_kicks.id_compat import ids_differ

    teams = frames.loc[~frames["is_ball"].astype(bool), ["game_id", "team_id"]].dropna()
    out = []
    for gid, poss in zip(spells["game_id"], spells["possessing_team"]):
        cand = teams.loc[teams["game_id"] == gid, "team_id"].unique()
        other = [t for t in cand if ids_differ(pd.Series([t]), pd.Series([poss])).iloc[0]]
        out.append(other[0] if other else np.nan)
    return pd.Series(out, index=spells.index)


def _time_remaining(spells) -> np.ndarray:
    """Seconds left in the period, from the period's own observed maximum end_time."""
    end = spells.groupby(["game_id", "period_id"])["end_time"].transform("max")
    return (end - spells["entry_time"]).to_numpy(dtype=float)
```

> **Implementer note:** verify `compute_defensive_line`'s actual returned column names before
> wiring (`.venv312/Scripts/python.exe -c "from silly_kicks.tracking import compute_defensive_line; help(compute_defensive_line)"`)
> and adapt the two `keyed[...]` reads. Do not guess — the spec's own history has two phantom-citation
> incidents from exactly this shortcut.

- [ ] **Step 4: Verify pass** — `.venv312/Scripts/python.exe -m pytest tests/causal/test_confounders.py -q`

---

## Task 7: `causal/power.py` — ATT power + **the FIREWALL** (spec §5.1, §5.4)

**Files:** `silly_kicks/causal/power.py` (create), `silly_kicks/causal/__init__.py`; `tests/causal/test_power.py` (create)

- [ ] **Step 1: Write the failing tests — the firewall FIRST, with its RED demonstration**

```python
# tests/causal/test_power.py
import numpy as np
import pytest
from silly_kicks.causal.power import InjectionSpec, att_power_curve


def _spells(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "Z": rng.integers(0, 2, size=n),
        "X": rng.normal(size=(n, 3)),
        "clusters": np.repeat(np.arange(n // 20), 20),
    }


def test_firewall_refuses_a_bare_outcome_array():
    """THE gate this cycle rests on. A call-count spy on `estimate_att` is VACUOUS -- the harness
    always calls it -- so the guard is provenance, and this is its RED side."""
    s = _spells()
    with pytest.raises(ValueError, match="not an InjectionSpec"):
        att_power_curve(Z=s["Z"], injection=np.zeros(len(s["Z"])), X=s["X"],
                        clusters=s["clusters"], sizes=(200,), n_replicates=2, rng_seed=0)


def test_firewall_refuses_a_stamp_lookalike():
    s = _spells()
    class _Fake:
        base_rate, relative_effect, stamp = 0.15, 0.0, "not-the-stamp"
        def draw(self, Z, rng):
            return np.zeros(len(Z))
    with pytest.raises(ValueError, match="not an InjectionSpec"):
        att_power_curve(Z=s["Z"], injection=_Fake(), X=s["X"], clusters=s["clusters"],
                        sizes=(200,), n_replicates=2, rng_seed=0)


def test_a_fresh_outcome_is_drawn_PER_REPLICATE_not_once():
    """Spec §5.4: 'Per replicate: ... inject a treatment effect.' Freezing one realisation and
    reusing it makes every replicate the same dataset reordered, which understates variance and
    turns the power estimate into a single-draw accident."""
    s = _spells()
    spec = InjectionSpec(base_rate=0.15, relative_effect=0.20)
    out = att_power_curve(Z=s["Z"], injection=spec, X=s["X"], clusters=s["clusters"],
                          sizes=(400,), n_replicates=30, rng_seed=0)
    assert out["n_distinct_outcome_draws"][400] == 30


def test_power_rises_with_the_injected_effect_both_sides():
    s = _spells(n=800)
    lo = att_power_curve(Z=s["Z"], injection=InjectionSpec(base_rate=0.15, relative_effect=0.0),
                         X=s["X"], clusters=s["clusters"], sizes=(800,), n_replicates=60, rng_seed=0)
    hi = att_power_curve(Z=s["Z"], injection=InjectionSpec(base_rate=0.15, relative_effect=0.60),
                         X=s["X"], clusters=s["clusters"], sizes=(800,), n_replicates=60, rng_seed=0)
    assert lo["power_by_size"][800] <= 0.20
    assert hi["power_by_size"][800] >= 0.60


def test_matched_n_is_the_MATCHED_count_not_the_subsample_size():
    """`N_MIN_MATCHED` is spec-defined as 'the smallest matched-n bin at which power >= 0.80'.
    Recording `idx.size` would record the SUBSAMPLE SIZE -- identically the input, since the
    resampler truncates to exactly `size`. ATT's focal set is the TREATED units only, so a correct
    matched-n is strictly smaller than the subsample."""
    s = _spells()
    out = att_power_curve(Z=s["Z"], injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
                          X=s["X"], clusters=s["clusters"], sizes=(200, 400),
                          n_replicates=10, rng_seed=0)
    assert out["matched_n_by_size"][400] < 400, "matched_n is echoing the subsample size"
    assert out["matched_n_by_size"][200] < out["matched_n_by_size"][400]
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: silly_kicks.causal.power`

- [ ] **Step 3: Implement**

```python
# silly_kicks/causal/power.py
"""Plasmode ATT power (spec §5.4). Generic: the GKDV-specific injection lives in `gkdv/_validate.py`.

FIREWALL (spec §5.1). Once Layer 2's design is expressible in code, this module could also RUN it --
producing the H1-vs-H2 answer before the sign-off meant to authorise it. :func:`att_power_curve`
therefore takes **no outcome vector at all**: it accepts an :class:`InjectionSpec` recipe and draws
the outcome itself, so an observed outcome is not merely refused but unrepresentable. A call-count
spy on ``estimate_att`` would NOT catch a breach, because the harness always calls it -- the guard
has to be provenance, and its RED side is demonstrated in ``tests/causal/test_power.py``.

Examples
--------
Power to detect a 20 % relative lift on a 15 % base rate::

    import numpy as np
    from silly_kicks.causal.power import InjectionSpec, att_power_curve

    rng = np.random.default_rng(0)
    Z = rng.integers(0, 2, size=400)
    out = att_power_curve(Z=Z, injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
                          X=rng.normal(size=(400, 3)),
                          clusters=np.repeat(np.arange(20), 20),
                          sizes=(400,), n_replicates=20, rng_seed=0)
    out["power_by_size"][400]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from silly_kicks.causal.matching import estimate_att, fit_propensity

_STAMP = "silly_kicks.causal.power:injected:v1"


@dataclass(frozen=True)
class InjectionSpec:
    """A RECIPE for a known effect, not a frozen realisation.

    The outcome vector is drawn inside the replicate loop (spec §5.4: "Per replicate: ... inject a
    treatment effect"), so replicates differ in outcome noise rather than being one dataset
    reordered. Passing a recipe rather than a vector also strengthens the FIREWALL: an observed
    outcome is not merely refused, it is UNREPRESENTABLE -- `att_power_curve` accepts no `Y` at all.
    """

    base_rate: float
    relative_effect: float
    stamp: str = field(default=_STAMP)

    @property
    def true_effect(self) -> float:
        return float(self.base_rate) * float(self.relative_effect)

    def draw(self, Z, rng) -> np.ndarray:
        """Bernoulli outcomes at `base_rate`, lifted by `relative_effect` x base_rate when treated."""
        Z = np.asarray(Z)
        p = np.full(Z.shape, float(self.base_rate), dtype=float)
        p[Z == 1] = float(self.base_rate) * (1.0 + float(self.relative_effect))
        return (rng.random(Z.shape) < np.clip(p, 0.0, 1.0)).astype(float)


def _require_spec(injection) -> InjectionSpec:
    if not isinstance(injection, InjectionSpec) or getattr(injection, "stamp", None) != _STAMP:
        raise ValueError(
            "att_power_curve was given something that is not an InjectionSpec (spec §5.1 FIREWALL): "
            "computing power on the OBSERVED outcome would answer Layer 2 before sign-off."
        )
    return injection


def att_power_curve(*, Z, injection, X, clusters, sizes, n_replicates, alpha_z=2.0, rng_seed=0) -> dict:
    """Cluster-resampled plasmode power.

    `matched_n` is an OUTPUT, never a dial: matching CONSUMES spells and YIELDS a focal count, so the
    loop records `(matched_n, detected)` per replicate and bins power by the requested subsample size.
    """
    spec = _require_spec(injection)
    rng = np.random.default_rng(rng_seed)
    Z, X, clusters = np.asarray(Z), np.asarray(X, dtype=float), np.asarray(clusters)
    ukeys = np.unique(clusters)
    power, matched, draws = {}, {}, {}
    for size in sizes:
        detected, m_ns, seen = 0, [], set()
        for _ in range(int(n_replicates)):
            idx = _resample_clusters(clusters, ukeys, int(size), rng)
            y = spec.draw(Z[idx], rng)  # FRESH per replicate (spec §5.4)
            seen.add(hash(y.tobytes()))
            ps, _ = fit_propensity(X[idx], Z[idx], seed=int(rng.integers(0, 2**31 - 1)))
            est = estimate_att(y, Z[idx], ps, X[idx])
            # est.n_focal is the MATCHED FOCAL (treated) count -- NOT idx.size, which is the
            # subsample size the resampler was asked for and would echo the input identically.
            m_ns.append(int(est.n_focal))
            if est.se and np.isfinite(est.se) and est.se > 0:
                detected += int(abs(est.estimate) / est.se >= alpha_z)
        power[size] = detected / float(n_replicates)
        matched[size] = int(np.mean(m_ns))
        draws[size] = len(seen)
    return {
        "power_by_size": power,
        "matched_n_by_size": matched,
        "n_distinct_outcome_draws": draws,
        "true_effect": spec.true_effect,
        "base_rate": float(spec.base_rate),
        "n_replicates": int(n_replicates),
    }


def _resample_clusters(clusters, ukeys, target_size: int, rng) -> np.ndarray:
    """Resample WHOLE clusters until the target row count is reached -- never individual rows."""
    picked, total = [], 0
    order = rng.permutation(ukeys)
    for k in order:
        members = np.flatnonzero(clusters == k)
        picked.append(members)
        total += members.size
        if total >= target_size:
            break
    return np.concatenate(picked)[:target_size]
```

Export `power` from `silly_kicks/causal/__init__.py` and add `silly_kicks/causal/power.py` to
`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`.

- [ ] **Step 4: Verify pass + public-module gates**

```bash
.venv312/Scripts/python.exe -m pytest tests/causal/test_power.py tests/test_public_api_examples.py -q
.venv312/Scripts/python.exe -m pytest --doctest-modules silly_kicks/causal/power.py -q
```

- [ ] **Step 5: Demonstrate the firewall RED (spec §10 — a demonstrated property, not a claim)**

Temporarily change `_require_spec` to `return injection` (dropping both the isinstance and the stamp
check), run `.venv312/Scripts/python.exe -m pytest tests/causal/test_power.py -k firewall -q`, and
confirm **both firewall tests FAIL**. Revert. Paste the failure output into the PR body — the guard's
non-vacuity is evidence, not assertion.

---

## Task 8: Register the GKDV constants (spec §5.4, §6)

**Files:** `silly_kicks/gkdv/_validate.py`; `tests/gkdv/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_signoff_constants_are_registered_blind():
    from silly_kicks.gkdv._validate import ATT_RELATIVE_ANCHORS, LAYER3_HEADROOM_RANGE_FRACTION
    assert ATT_RELATIVE_ANCHORS == (0.10, 0.15, 0.20)
    assert LAYER3_HEADROOM_RANGE_FRACTION == 0.02
```

- [ ] **Step 2: Run to verify it fails** — `ImportError`

- [ ] **Step 3: Implement** — append to `gkdv/_validate.py`:

```python
#: Row 5's ATT effect-size anchors: a RANGE, mirroring ICC_ANCHORS, expressed as a RELATIVE change
#: in the outcome base rate (spec D2; §1.3: "scale-free relative criteria ... are the honest idiom
#: for small-probability quantities"). N_MIN_MATCHED is registered at the 0.15 anchor.
ATT_RELATIVE_ANCHORS: tuple[float, float, float] = (0.10, 0.15, 0.20)

#: Layer 3's headroom threshold as a fraction of `openGoal`'s OBSERVED range. COMMITTED BEFORE the
#: measurement (spec §6): measuring first and choosing the fraction after would make the threshold
#: tunable to any desired Layer 3 outcome.
LAYER3_HEADROOM_RANGE_FRACTION: float = 0.02

#: Derived by `scripts/run_signoff_power.py` on the locked corpus; filled in Task 11.
N_MIN_MATCHED: int | None = None
```

- [ ] **Step 4: Verify pass** — `.venv312/Scripts/python.exe -m pytest tests/gkdv/ -q`

---

## Task 9: `openGoal` range derivation driver (spec §6)

**Files:** `scripts/derive_opengoal_range.py` (create)

- [ ] **Step 1: Write the driver**

A `--out DIR` script that loads the corpus via the existing `scripts/_loader_pining.py` pattern,
calls `extract_xshot_features` per domain frame, and writes `opengoal_distribution.json` with
`{"n": N, "min": .., "p01": .., "median": .., "p99": .., "max": .., "observed_range": max-min,
"layer3_threshold": observed_range * LAYER3_HEADROOM_RANGE_FRACTION}`.

- [ ] **Step 2: Assert the boundary in the script itself**

```python
# The derivation measures the MARGINAL distribution of the shipped feature. It must NOT run the
# ghost substitution or compute any delta -- that is Layer 3's probe, and it is PR-3b (spec §6).
assert "gkdv" not in sys.modules, "derivation must not import the ghost-substitution engine"
```

- [ ] **Step 3: Smoke it locally on one match**

```bash
.venv312/Scripts/python.exe scripts/derive_opengoal_range.py --out /tmp/og --max-per-provider 1
```
Expected: a JSON with `0 <= min <= max <= 1` (openGoal is a fraction by construction).

---

## Task 9b: `run_signoff_power.py` — the power driver (review R3 MEDIUM 2)

**Files:** `scripts/run_signoff_power.py` (create); `tests/scripts/test_run_signoff_power.py` (create)

> This script was referenced by Task 10, by the File Structure table and by `N_MIN_MATCHED`'s
> comment, but **no task authored it** — Task 9 writes the openGoal driver only. It is also where the
> analysis design matrix is assembled, which is where MEDIUM 3 lands.

- [ ] **Step 1: Write the failing test for the design-matrix assembly**

```python
# tests/scripts/test_run_signoff_power.py
import numpy as np
import pandas as pd
import pytest

import scripts.run_signoff_power as mod  # bare import: tests/scripts/ has NO __init__.py


def test_all_nan_confounder_raises_naming_the_column():
    """An all-NaN column reaches `fit_propensity` and dies inside sklearn with a message that does
    not name the culprit: `ValueError: Input X contains NaN. LogisticRegression does not accept
    missing value` (MEASURED). Fail earlier and say which column."""
    spells = pd.DataFrame({"r": [1.0, 2.0], "theta": [0.1, 0.2], "score_differential": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="score_differential"):
        mod.build_design_matrix(spells, ("r", "theta", "score_differential"))


def test_design_matrix_returns_columns_in_the_registered_order():
    spells = pd.DataFrame({"r": [1.0, 2.0], "theta": [0.1, 0.2]})
    X = mod.build_design_matrix(spells, ("r", "theta"))
    assert X.shape == (2, 2)
    assert X[0, 0] == 1.0 and X[0, 1] == 0.1
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: scripts.run_signoff_power`

- [ ] **Step 3: Implement the driver**

`--out DIR`, `--providers`, `--seed`. Loads the corpus with the same pattern
`scripts/validate_xs_probe.py` uses, then:

1. builds Layer 2 spells via `build_opportunities(..., config=layer2_config({}))`;
2. joins the tracking confounders via `join_layer2_confounders(..., home_team_id=...)`;
3. assembles `X` with `build_design_matrix(spells, LAYER2_CONFOUNDERS)`;
4. runs `att_power_curve` once per `ATT_RELATIVE_ANCHORS` entry **for each of `Y_attempt` and
   `Y_close_attempt`**, taking `N_MIN_MATCHED` as the **maximum** of the two (spec §5.4);
5. runs `icc_power_curve` on the arm values at all three `ICC_ANCHORS`;
6. writes `metrics.json` with `lock_commit`/`run_commit` and the measured base rates.

```python
def build_design_matrix(spells, confounders) -> np.ndarray:
    """Assemble X, failing LOUD on an unusable column rather than inside sklearn.

    A dropped-silently confounder would weaken the registered design without any record; sklearn's
    own message names no column. Both are worse than stopping here.
    """
    missing = [c for c in confounders if c not in spells.columns]
    if missing:
        raise ValueError(f"design matrix: confounders absent from spells: {missing}")
    dead = [c for c in confounders if not np.isfinite(spells[c].to_numpy(dtype=float)).any()]
    if dead:
        raise ValueError(
            f"design matrix: confounders are entirely non-finite: {dead}. "
            "fit_propensity would raise 'Input X contains NaN' without naming them."
        )
    return spells.loc[:, list(confounders)].to_numpy(dtype=float)
```

- [ ] **Step 4: Verify pass** — `.venv312/Scripts/python.exe -m pytest tests/scripts/test_run_signoff_power.py -q`

---

## Task 10: DGX run — power curves + openGoal (owner)

- [ ] **Step 1: Record the lock commit**

```bash
git rev-parse --short HEAD
```
The artifacts must cite `lock_commit == run_commit` (the xS-v2 blindness idiom).

- [ ] **Step 2: Run on DGX** (`ssh karsten@192.168.68.73`, repo `~/Development/silly-kicks`)

```bash
python scripts/run_signoff_power.py --out docs/research/tf19_s64_signoff --providers gradientsports
python scripts/derive_opengoal_range.py --out docs/research/tf19_s64_signoff
```

- [ ] **Step 3: Read the ICC result honestly**

If §6.1's ICC power is **< 0.8** at the anchor: per §6.1 the gate is NOT registered and floors/sampling
are adjusted first. Report it as prominently as a pass — do not proceed to register the ICC gate.

---

## Task 11: DGX run — §3.3 shot arm (D4, F6)

- [ ] **Step 1: Run the shipped driver, unmodified**

```bash
python scripts/validate_xshot_causal.py --out docs/research/tf19_causal/xshot --providers gradientsports
```

- [ ] **Step 2: Record the outcome against the three registered possibilities (spec §5.5)**

`clears` → the xS re-gate becomes `joins` (measured); `inside_band` → 4.60.0's `joins_with_caveat`
is confirmed and becomes measured; `degenerate` → the driver refuses by design and the caveat is
re-described as honestly unmeasured. **All three are reportable results.**

- [ ] **Step 3: Fill in `N_MIN_MATCHED`** in `gkdv/_validate.py` from Task 10's ATT curve at the
0.15 anchor, and add the artifact-derived value to the test in Task 8.

---

## Task 12: Record corrections + amendments (spec §8)

**Files:** `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md` (§6.4),
`docs/superpowers/adrs/ADR-037-*.md`, `TODO.md`, `CLAUDE.md`, `scripts/validate_xs_probe.py`

- [ ] **Step 1: Amend §6.4** — the arm divergence; the registered SYMMETRY (a live probe is Layer-1
responsiveness, NOT H1 support — both hypotheses remain Layer-2-only); row 5 re-specified with its
own relative anchor and the `"at ICC 0.015–0.026"` clause struck; the derived constants; the routing
amendment.

- [ ] **Step 2: Fix the F6 record everywhere it appears** — TODO.md's TF-19 row and research-program
note, CLAUDE.md's TF-19 bullet, ADR-037. Replace "the banked SHOT causal arm" with what it was: a
registered default that became decision-relevant only when the probe passed, now superseded by
Task 11's measurement.

- [ ] **Step 3: Fix the CLI help string** (`scripts/validate_xs_probe.py:290`) — it calls the value
the *"banked shot-arm causal result"* for a run that had never happened, and names
`docs/research/tf19_causal/xshot/`, the path Task 11 now writes.

---

## Task 13: Release

- [ ] **Step 1: Full suite on the CI-repro venv**

```bash
.venv312/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip 2>&1 | tail -5
```

- [ ] **Step 2: Lint + types**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m pyright
```

- [ ] **Step 3: 5-site version bump** — `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`,
`TODO.md` header, then `uv lock`. Confirm the next-free version against `origin/main` at commit-prep;
**never reserve a number early**.

- [ ] **Step 4: `/mad-scientist-skills:final-review`, including `/c4`** — expected C4-free (count stays
32: a power simulator and a builder axis are not action-coupled aggregators). Verify by running it.

- [ ] **Step 5: Single commit, PR, watch CI, admin merge, tag** — one commit for the whole branch per
house convention. Commit only with explicit approval.

---

## Self-review

**Spec coverage:** F1→Tasks 2/7; F2→Task 2; F3→Tasks 5/8/12; F4→Task 1; F5→Task 9; F6→Tasks 11/12;
F7→Task 4. D1→Tasks 2/7; D2→Task 8; D3→Task 3; D4→Task 11; D5→Tasks 4/5/6. Review R2: #1→Task 6,
#2→Task 7 Steps 1/5, #3→Tasks 7/11 scoping, #4→Task 3, #5→Task 6's gate.

**Known gaps carried deliberately:** Task 6's `compute_defensive_line` column names are flagged
verify-before-wiring rather than guessed; `score_differential` needs the `_build_score_lookup` path
already used by `_row` and is left NaN-tolerant. `run_signoff_power.py`'s loader wiring follows
`scripts/validate_xs_probe.py`'s existing corpus pattern.
