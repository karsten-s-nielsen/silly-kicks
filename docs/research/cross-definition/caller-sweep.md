# Cross definition (ADR-133): caller sweep

Spec §9 lists four seams that change. This memo classifies every reader in `silly_kicks/` and `scripts/` with its evidence. The sweep is the full `git grep` at the cycle's base `0cc07e6c` plus this branch. The spec's floor lists are a subset of the hits below.

Classes:
- **intended shift**: the value moves because the type now means the geometric delivery (ADR-133). The release notice lists it.
- **unaffected (union)**: the reader takes `pass` ∪ `cross` (or every set-piece sibling), so a row that moves between the types does not change it.
- **changed in this cycle**: the reader moved to the new meaning, with tests (Tasks 4, 6, 7).
- **retrain owed**: a trained artifact whose training rows or features read the type. The retrain is handed to Part Deux (spec decision 6).

## Seam 1: the open-play `cross` type

| Reader | Reads | Class | Evidence |
|---|---|---|---|
| `spadl/base.py:18` `_DERIVE_END_TYPE_IDS` | pass, cross, set pieces (end derivation) | unaffected (union) | the set holds both types |
| `spadl/config.py:53` | the action-type list | unaffected | the id list does not change |
| `atomic/spadl/base.py:99` | atomic type mapping | unaffected (union) | pass and cross map to their atomic twins one to one |
| `atomic/tracking/features.py:266,305,632,675` | defensive-credit domain, pass-like mask, packing | unaffected (union) | each lists `("pass", "cross")` |
| `atomic/vaep/features.py:404` | move names | unaffected (union) | `("pass", "dribble", "cross")` |
| `vaep/features/specialty.py:47,130,140` | `cross_zone` / cross-type features | intended shift | `is_cross` reads the type; a VAEP model fitted before ADR-133 sees a new cross population |
| `xthreat/_grid.py:132`, `xthreat/_model.py:97` | xT move counts | unaffected (union) | pass, cross and dribble are all moves |
| `xtgk/_moves.py:29,31` | xT-GK moves | unaffected (union) | pass and cross both listed |
| `calibration/_carrier_objective.py:33` | carrier action types | unaffected (union) | `{"pass", "cross", "shot", "dribble"}` |
| `causal/opportunities.py:96-103` | treatment rows | changed in this cycle (Task 6) | label v2 selects by `cross_attempt` (`xcross_label_version`) |
| `providers/sportec/parse.py:243` | bronze column prefix | unaffected | `"Cross"` -> `cross_*` columns; not a type decision |
| `tracking/_xcross_attempt.py` | xCross label | changed in this cycle (Task 6) | label v2 reads `cross_attempt` |
| `tracking/_run_values.py:419` | pass ids | unaffected (union) | `{pass, cross}` |
| `tracking/_packing.py:33-38` | packing move types | unaffected (union) | pass, cross and every set-piece sibling |
| `tracking/_elastic_sync.py:380-390` | ELASTIC event kinds | unaffected (union) | pass, cross and every set-piece sibling |
| `tracking/_line_breaking.py:37`, `tracking/features.py:5789` | pass type ids | unaffected (union) | `("pass", "cross")` |
| `tracking/_model_eval.py:728` | arm name `"cross"` | unaffected | a string label for the xCross arm |
| `tracking/defensive_credit/_rules.py:52,437,456,478`, `_params.py:54` (`ANCHOR_CROSS`) | the cross-defence credit rule | intended shift | the rule fires on a successful geometric `cross` |
| `tracking/defensive_credit/_bravery.py:17-21` | crosses faced | changed in this cycle (Task 4) | reads `cross_native` |
| `scripts/_rq_corpus.py:15,16` | pass/cross; crosses excluded from the pass-only headline | intended shift | the excluded rows are the geometric crosses |
| `scripts/build_sb360_coverage.py:56`, `validate_sb360_licensed_corpus.py:52` | GK domain types | intended shift | the GK domain counts geometric crosses |
| `scripts/_toy_corpora.py:608` | toy cross rows | changed in this cycle (Task 6) | the toy stages `cross_attempt` |
| `scripts/train_receiver_model.py:39`, `scripts/_receiver_validation.py:31` | completed passes | unaffected (union) | `_PASS_TYPES = {pass, cross}` |

### Readers of `pass` alone (rows move both ways between `pass` and `cross`)

| Reader | Reads | Class | Evidence |
|---|---|---|---|
| `expected_passing/_model.py:26,133`, `scripts/train_pass_completion.py:47` | PassCompletionModel training rows | **retrain owed** | trains on `type_id == pass`; bundled `expected_passing/weights` |
| `team_metrics/_pressing.py:22`, `_buildup.py:32`, `_progression.py:24` | pass counts (PPDA, build-up, progression) | intended shift | geometric crosses leave and SkillCorner zone "crosses" join the pass count |
| `territory/_compute.py:62` | territory passes | intended shift | pass-only |
| `tracking/defensive_credit/_line_break_signal.py:20`, `_rules.py:53` | pass-only defensive rules | intended shift | pass-only |
| `tracking/_gk_resolve.py:35,89` | GK distribution `is_open` | unaffected in practice | a geometric cross starts at or beyond `CROSS_START_X` in the attacking half, so a keeper distribution stays a `pass` |
| `causal/_confounders.py:93` | synthesized spell-entry rows | unaffected | the rows are built as `pass`, not read |
| `xsuccess/_features.py:19,48` | the action-type one-hot | **retrain owed** | the feature set is every action type; bundled `xsuccess/weights` |
| `xsuccess/_features.py:71`, `_model.py:49`, `scripts/publish_xsuccess.py:41` | docstring / probe rows | unaffected | literal example rows |
| `tracking/_stoppages.py:123` | docstring example | unaffected | literal example |
| `scripts/hf_jobs/headtohead.py:304,336` | synthetic test rows | unaffected | built, not read |
| `scripts/hf_jobs/headtohead.py:503`, `scripts/validate_territory_counterfactual.py:413,587`, `scripts/measure_f1b_feature_delta.py:240` | pass-only research measures | intended shift | pass-only |

## Seam 2: the set-piece types

| Reader | Reads | Class | Evidence |
|---|---|---|---|
| `team_metrics/_possession.py:29-32`, `coordination/_windows.py:28-31`, `tracking/_stoppages.py:43-46`, `tracking/_elastic_sync.py:387-398`, `tracking/_packing.py:35-38`, `tracking/_ghost_gk.py:645`, `tracking/_gk_geometry.py:30,40` | restarts / set pieces | unaffected (union) | each lists both siblings of a kind |
| `atomic/spadl/base.py:101-104,180-181,281-284`, `atomic/spadl/utils.py`, `atomic/tracking/features.py:211-233` | atomic `corner` / `freekick` collapse | unaffected (union) | both siblings collapse to one atomic type |
| `vaep/formula.py:76` | corner rows | unaffected (union) | `["corner_crossed", "corner_short"]` |
| `vaep/features/specialty.py:48,49,130-137` | crossed set pieces in `cross_zone` / `cross_types` | intended shift | reads the crossed siblings |
| `corners/_identify.py:24,25,95` | `corner_kind` | changed in this cycle (Task 7) | `cross_attempt` decides when present (`corner_kind_source`) |
| `tracking/_xcross_attempt.py:362,363` | set-piece label kinds | changed in this cycle (Task 6) | `_CROSS_TYPE_KINDS` |
| `tracking/defensive_credit/_bravery.py:21` | set-piece crosses faced | changed in this cycle (Task 4) | reads `cross_native` |
| `scripts/_receiver_validation.py:32`, `scripts/build_skillcorner_s1_event_validity.py:54,66` | restart / anchored types | unaffected (union) | both siblings listed |

## Seam 3: the new column `cross_native`

`spadl/schema.py` (all schema dicts), `_finalize_output`, `convert_to_atomic(preserve_native=…)`, `reflection.py:204`, `atomic/tracking/features.py:330,462-491` (atomic bravery) and `tracking/defensive_credit/_bravery.py`. All changed in this cycle (Tasks 2 and 4). Every converter emits the column (Task 3).

## Seam 4: `cross_blocked` applicability

`tracking/defensive_credit/_bravery.py:42,90,93` and `atomic/tracking/features.py:323-357,433` read `cross_blocked` on rows with `cross_native == True` (Task 4). `scripts/probe_sb_cross_blocked.py` is a one-off probe of the StatsBomb native block join, unaffected (it reads raw events).

## Trained artifacts

| Artifact | Reads SPADL type? | Retrain |
|---|---|---|
| `tracking/_xcross_weights` (xCross) | the label | **owed** (label v2) |
| `expected_passing/weights` (PassCompletionModel) | trains on `pass` rows | **owed** |
| `xsuccess/weights` (XSuccessModel) | action-type one-hot feature | **owed** |
| `corners/_routine_weights`, `corners/_run_zone_weights` (TF-46) | trained on `corner_kind == "direct"` corners | **owed** (tracking decides `corner_kind`) |
| `tracking/_receiver_weights` | completed `pass` ∪ `cross` | not owed (union) |
| `tracking/_xshot_weights` | shot types only (`_xshot_occurrence.py:338`) | not owed |
| `match_outcome/weights`, `win_probability/weights` | shot types only (`_compute.py:23`, `_state.py:18`) | not owed |
| `tracking/_ghost_gk_weights`, `_ghost_outfield_weights` | shots and the set-piece union | not owed |
| `xtgk/_retention_weights` | the move union (`_moves.py`) | not owed |
| VAEP | no bundled weights | none (users refit) |

## Raw hit lists

The raw `git grep` output for the four seams (JSON weight files excluded). Every line is classified above.

```text
=== SEAM1 cross
scripts/_rq_corpus.py:15:_PASS_TYPES = {spc.actiontype_id[t] for t in ("pass", "cross")}  # spec 4: pass/cross ONLY
scripts/_rq_corpus.py:16:_CROSS = spc.actiontype_id["cross"]  # crosses are aerial -> Driver A headline is pass-only
scripts/_toy_corpora.py:608:            "type_id": [spadlconfig.actiontype_id["cross"]] * len(cross_times),
scripts/build_sb360_coverage.py:56:GK_DOMAIN_TYPES = ("shot", "shot_penalty", "shot_freekick", "cross", "goalkick", "keeper_save")
scripts/validate_sb360_licensed_corpus.py:52:_GK_DOMAIN_TYPES = ("shot", "shot_penalty", "shot_freekick", "cross", "goalkick", "keeper_save")
silly_kicks/atomic/spadl/base.py:99:        "cross",
silly_kicks/atomic/tracking/features.py:266:_DC_DOMAIN_TYPES = ("pass", "cross", "shot", "shot_penalty", "take_on", "bad_touch")
silly_kicks/atomic/tracking/features.py:305:    is_pass_like = np.isin(type_id, [atomicconfig.actiontype_id["pass"], atomicconfig.actiontype_id["cross"]])
silly_kicks/atomic/tracking/features.py:632:    adapted = _packing_atomic_adapter(actions, PackingParams(action_types=("pass", "cross")))
silly_kicks/atomic/tracking/features.py:675:    adapter_params = PackingParams(action_types=("pass", "cross"))
silly_kicks/atomic/vaep/features.py:404:    move_names = ("pass", "dribble", "cross")
silly_kicks/calibration/_carrier_objective.py:33:_CARRIER_ACTION_TYPES = {"pass", "cross", "shot", "dribble"}
silly_kicks/causal/opportunities.py:101:    ('cross',)
silly_kicks/providers/sportec/parse.py:243:    "Cross": "cross",
silly_kicks/tracking/_elastic_sync.py:380:        "cross",
silly_kicks/tracking/_line_breaking.py:37:    spadlconfig.actiontype_id[n] for n in ("pass", "cross") if n in spadlconfig.actiontype_id
silly_kicks/tracking/_model_eval.py:728:    if arm not in ("shot", "cross") or probe_verdict not in _PROBE_VERDICTS or entanglement not in _ENTANGLEMENT:
silly_kicks/tracking/_packing.py:33:    "cross",
silly_kicks/tracking/_run_values.py:419:    pass_ids = {spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]}
silly_kicks/tracking/defensive_credit/_bravery.py:19:_OPEN_PLAY = frozenset({spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]})
silly_kicks/tracking/defensive_credit/_rules.py:52:_CROSS = spadlconfig.actiontype_id["cross"]
silly_kicks/tracking/features.py:5789:_PASS_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("pass", "cross") if n in spadlconfig.actiontype_id)
silly_kicks/vaep/features/specialty.py:47:            spadlcfg.actiontype_id["cross"],
silly_kicks/xtgk/_moves.py:31:    spadlconfig.actiontype_id["cross"],
silly_kicks/xthreat/_grid.py:132:        | (actions.type_id == spadlconfig.actiontype_id["cross"])
silly_kicks/xthreat/_model.py:97:    MOVE_TYPE_NAMES: tuple[str, str, str] = ("pass", "dribble", "cross")
=== SEAM1b pass-alone
scripts/hf_jobs/headtohead.py:304:        typ = np.full(n, spadlconfig.actiontype_id["pass"])
scripts/hf_jobs/headtohead.py:336:                    "type_id": [spadlconfig.actiontype_id["pass"]] * k,
scripts/hf_jobs/headtohead.py:503:    passes = acts[acts["type_id"] == spadlconfig.actiontype_id["pass"]]
scripts/measure_f1b_feature_delta.py:240:    passes = actions[actions.get("type_name", pd.Series(index=actions.index, dtype=object)) == "pass"]
scripts/publish_xsuccess.py:41:            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
scripts/train_pass_completion.py:47:    pass_id = spadlconfig.actiontype_id["pass"]
scripts/validate_territory_counterfactual.py:413:    pass_id = spadlconfig.actiontype_id["pass"]
scripts/validate_territory_counterfactual.py:587:    pass_id = spadlconfig.actiontype_id["pass"]
silly_kicks/atomic/tracking/features.py:305:    is_pass_like = np.isin(type_id, [atomicconfig.actiontype_id["pass"], atomicconfig.actiontype_id["cross"]])
silly_kicks/causal/_confounders.py:93:            "type_id": _cfg.actiontype_id["pass"],
silly_kicks/expected_passing/_model.py:26:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/team_metrics/_buildup.py:32:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/team_metrics/_pressing.py:22:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/team_metrics/_progression.py:24:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/territory/_compute.py:62:_PASS_TYPE_ID = spadlconfig.actiontype_id["pass"]  # SPADL uses type_id; type_name is add_names-only
silly_kicks/tracking/_gk_resolve.py:35:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/tracking/_run_values.py:419:    pass_ids = {spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]}
silly_kicks/tracking/_stoppages.py:123:    ...     "type_id": [c.actiontype_id["pass"], c.actiontype_id["throw_in"], c.actiontype_id["pass"]],
silly_kicks/tracking/defensive_credit/_bravery.py:19:_OPEN_PLAY = frozenset({spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]})
silly_kicks/tracking/defensive_credit/_line_break_signal.py:20:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/tracking/defensive_credit/_rules.py:53:_PASS = spadlconfig.actiontype_id["pass"]
silly_kicks/xsuccess/_features.py:71:            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
silly_kicks/xsuccess/_model.py:49:            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
silly_kicks/xtgk/_moves.py:29:    spadlconfig.actiontype_id["pass"],
silly_kicks/xthreat/_grid.py:130:        (actions.type_id == spadlconfig.actiontype_id["pass"])
=== SEAM2 setpiece
scripts/_receiver_validation.py:32:_RESTART_TYPES = {_T["throw_in"], _T["goalkick"], _T["corner_crossed"], _T["corner_short"]}
scripts/build_skillcorner_s1_event_validity.py:54:ATTACK_HALF_ANCHORED = ("shot", "corner_short", "corner_crossed")
scripts/build_skillcorner_s1_event_validity.py:66:    {"goalkick", "corner_crossed", "corner_short", "throw_in", "freekick_crossed", "freekick_short", "shot_freekick"}
silly_kicks/atomic/spadl/base.py:101:        "freekick_short",
silly_kicks/atomic/spadl/base.py:102:        "freekick_crossed",
silly_kicks/atomic/spadl/base.py:103:        "corner_crossed",
silly_kicks/atomic/spadl/base.py:104:        "corner_short",
silly_kicks/atomic/spadl/base.py:180:            _atomicspadl.actiontype_id["corner_crossed"],
silly_kicks/atomic/spadl/base.py:181:            _atomicspadl.actiontype_id["corner_short"],
silly_kicks/atomic/spadl/base.py:281:    cornerlike = ["corner_crossed", "corner_short"]
silly_kicks/atomic/spadl/base.py:284:    freekicklike = ["freekick_crossed", "freekick_short", "shot_freekick"]
silly_kicks/atomic/spadl/utils.py:34:collapsed names: standard SPADL ``corner_short`` / ``corner_crossed`` →
silly_kicks/atomic/spadl/utils.py:35:``corner`` and ``freekick_short`` / ``freekick_crossed`` / ``shot_freekick`` →
silly_kicks/atomic/spadl/utils.py:265:       (NOT ``corner_short`` / ``corner_crossed``) and ``freekick`` (NOT
silly_kicks/atomic/spadl/utils.py:266:       ``freekick_short`` / ``freekick_crossed`` / ``shot_freekick``).
silly_kicks/atomic/spadl/utils.py:560:``long_threshold``. Standard SPADL's ``freekick_short`` / ``freekick_crossed``
silly_kicks/atomic/spadl/utils.py:1171:        standard on collapsed names — pass ``"corner"`` (not ``"corner_short"``)
silly_kicks/atomic/spadl/utils.py:1172:        and ``"freekick"`` (not ``"freekick_short"``).
silly_kicks/atomic/tracking/features.py:211:    ``_simplify`` re-types corner_crossed/corner_short -> atomic ``corner`` and
silly_kicks/atomic/tracking/features.py:212:    freekick_crossed/freekick_short/shot_freekick -> atomic ``freekick``, so the
silly_kicks/atomic/tracking/features.py:232:        ("corner", ("corner_crossed", "corner_short")),
silly_kicks/atomic/tracking/features.py:233:        ("freekick", ("freekick_crossed", "freekick_short")),
silly_kicks/coordination/_windows.py:28:    "freekick_crossed",
silly_kicks/coordination/_windows.py:29:    "freekick_short",
silly_kicks/coordination/_windows.py:30:    "corner_crossed",
silly_kicks/coordination/_windows.py:31:    "corner_short",
silly_kicks/corners/_identify.py:3:Anchors every corner on its SPADL ``corner_crossed`` / ``corner_short`` action (present for every
silly_kicks/corners/_identify.py:4:provider -- the converters map a ``corner_for`` restart to one of these). ``corner_short`` -> a short
silly_kicks/corners/_identify.py:5:corner; ``corner_crossed`` -> a direct corner. For SkillCorner, an optional ``gi`` frame carrying the
silly_kicks/corners/_identify.py:24:_CORNER_CROSSED_ID = spadlconfig.actiontype_id["corner_crossed"]
silly_kicks/corners/_identify.py:25:_CORNER_SHORT_ID = spadlconfig.actiontype_id["corner_short"]
silly_kicks/team_metrics/_possession.py:29:        "freekick_crossed",
silly_kicks/team_metrics/_possession.py:30:        "freekick_short",
silly_kicks/team_metrics/_possession.py:31:        "corner_crossed",
silly_kicks/team_metrics/_possession.py:32:        "corner_short",
silly_kicks/tracking/_elastic_sync.py:387:        "freekick_crossed",
silly_kicks/tracking/_elastic_sync.py:388:        "freekick_short",
silly_kicks/tracking/_elastic_sync.py:389:        "corner_crossed",
silly_kicks/tracking/_elastic_sync.py:390:        "corner_short",
silly_kicks/tracking/_elastic_sync.py:398:_OUT_TRIGGER_TYPES = frozenset({"throw_in", "goalkick", "corner_crossed", "corner_short"})
silly_kicks/tracking/_ghost_gk.py:645:    _SP_TYPES = {"freekick_crossed", "freekick_short", "corner_crossed", "corner_short"}
silly_kicks/tracking/_gk_geometry.py:30:_CORNER_TYPES = (spadlconfig.actiontype_id["corner_crossed"], spadlconfig.actiontype_id["corner_short"])
silly_kicks/tracking/_gk_geometry.py:464:            continue  # open-play / freekick_short -> no rule-point
silly_kicks/tracking/_packing.py:35:    "freekick_crossed",
silly_kicks/tracking/_packing.py:36:    "freekick_short",
silly_kicks/tracking/_packing.py:37:    "corner_crossed",
silly_kicks/tracking/_packing.py:38:    "corner_short",
silly_kicks/tracking/_stoppages.py:43:    "freekick_crossed",
silly_kicks/tracking/_stoppages.py:44:    "freekick_short",
silly_kicks/tracking/_stoppages.py:45:    "corner_crossed",
silly_kicks/tracking/_stoppages.py:46:    "corner_short",
silly_kicks/tracking/_xcross_attempt.py:362:    "corner_crossed": ("corner_crossed", "corner_short"),
silly_kicks/tracking/_xcross_attempt.py:363:    "freekick_crossed": ("freekick_crossed", "freekick_short"),
silly_kicks/tracking/_xcross_attempt.py:401:    ``"cross"`` selects open-play ``pass``/``cross`` rows, and ``"corner_crossed"`` / ``"freekick_crossed"``
silly_kicks/tracking/defensive_credit/_bravery.py:21:    spadlconfig.actiontype_id[t] for t in ("corner_crossed", "corner_short", "freekick_crossed", "freekick_short")
silly_kicks/vaep/features/specialty.py:48:            spadlcfg.actiontype_id["corner_crossed"],
silly_kicks/vaep/features/specialty.py:49:            spadlcfg.actiontype_id["freekick_crossed"],
silly_kicks/vaep/features/specialty.py:130:    cross_types = {spadlcfg.actiontype_id[t] for t in ["cross", "corner_crossed", "freekick_crossed"]}
silly_kicks/vaep/features/specialty.py:134:            "corner_crossed",
silly_kicks/vaep/features/specialty.py:135:            "corner_short",
silly_kicks/vaep/features/specialty.py:136:            "freekick_crossed",
silly_kicks/vaep/features/specialty.py:137:            "freekick_short",
silly_kicks/vaep/formula.py:76:    corner_idx = actions.type_name.isin(["corner_crossed", "corner_short"])
=== SEAM3/4
scripts/_toy_corpora.py:280:                "cross_blocked": pd.NA,
scripts/_toy_corpora.py:281:                "cross_native": pd.NA,  # ADR-133 SPADL column
scripts/probe_sb_cross_blocked.py:1:"""Probe the StatsBomb `cross -> related_events -> Block` join for `cross_blocked` (BD-2).
silly_kicks/atomic/tracking/features.py:323:    ``cross_blocked`` reach the mirror only when the caller threaded them through ``preserve_native``.
silly_kicks/atomic/tracking/features.py:330:            f"convert_to_atomic(std_actions, preserve_native=['xg', 'shot_blocked', 'cross_blocked', 'cross_native'])."
silly_kicks/atomic/tracking/features.py:357:        atomic = convert_to_atomic(std_actions, preserve_native=["xg", "shot_blocked", "cross_blocked"])
silly_kicks/atomic/tracking/features.py:433:        atomic = convert_to_atomic(std_actions, preserve_native=["xg", "shot_blocked", "cross_blocked"])
silly_kicks/atomic/tracking/features.py:462:    cross_blocked_column="cross_blocked",
silly_kicks/atomic/tracking/features.py:463:    cross_native_column="cross_native",
silly_kicks/atomic/tracking/features.py:468:    provider's own cross (``cross_native``, ADR-133), so open-play bravery is exact. Atomic ``_simplify``
silly_kicks/atomic/tracking/features.py:478:        atomic = convert_to_atomic(std_actions, preserve_native=["shot_blocked", "cross_blocked", "cross_native"])
silly_kicks/atomic/tracking/features.py:485:        actions, [shot_blocked_column, cross_blocked_column, cross_native_column], fn="compute_bravery"
silly_kicks/atomic/tracking/features.py:490:        cross_blocked_column=cross_blocked_column,
silly_kicks/atomic/tracking/features.py:491:        cross_native_column=cross_native_column,
silly_kicks/reflection.py:202:    "cross_blocked": "invariant",
silly_kicks/reflection.py:204:    "cross_native": "invariant",
silly_kicks/tracking/defensive_credit/_bravery.py:17:# ADR-133: a cross faced is the PROVIDER's own cross (cross_native True), not the geometric SPADL type, so the
silly_kicks/tracking/defensive_credit/_bravery.py:42:    cross_blocked_column: str = "cross_blocked",
silly_kicks/tracking/defensive_credit/_bravery.py:43:    cross_native_column: str = "cross_native",
silly_kicks/tracking/defensive_credit/_bravery.py:47:    A cross faced is the provider's own cross (``cross_native`` True, ADR-133): an open-play pass-class row
silly_kicks/tracking/defensive_credit/_bravery.py:53:        If ``cross_native_column`` is absent (SPADL produced before ADR-133: re-convert first).
silly_kicks/tracking/defensive_credit/_bravery.py:63:    if cross_native_column not in a.columns:
silly_kicks/tracking/defensive_credit/_bravery.py:65:            f"compute_bravery: column {cross_native_column!r} is absent. Bravery counts the provider's own "
silly_kicks/tracking/defensive_credit/_bravery.py:68:    native = a[cross_native_column].fillna(False).astype(bool)
silly_kicks/tracking/defensive_credit/_bravery.py:90:        b_open = _rate(g, "open", cross_blocked_column)
silly_kicks/tracking/defensive_credit/_bravery.py:93:        known_blocked, known_faced = _known_domain(g, shot_blocked_column, cross_blocked_column)
```
