# Form-B search output (at `2901fa5c`, tree `527bf49c`): 336 hits

A `.to_numpy(dtype=float|"float64"|np.float64)` read with no `na_value`, then the classification by `scripts/classify_b.py` against the columns that a library schema declares nullable.

```text
$ rg -nP '\.to_numpy\((?![^)]*na_value)[^)]*dtype\s*=\s*(float|"float64"|np\.float64)\b' silly_kicks scripts   # also the 'float64' spelling
silly_kicks/gk_decision/_reconstruct.py:216:            d = np.hypot(cand["x"].to_numpy(dtype=float) - pass_end[0], cand["y"].to_numpy(dtype=float) - pass_end[1])
silly_kicks/gk_decision/_reconstruct.py:223:        cand_x = cand["x"].to_numpy(dtype=float)
silly_kicks/gk_decision/_reconstruct.py:224:        cand_y = cand["y"].to_numpy(dtype=float)
silly_kicks/corners/_zones.py:342:    return sub[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/positioning/_probe.py:203:        out[float(h)] = scored.to_numpy(dtype=float)
silly_kicks/restdefense/_structure.py:163:    x = a["x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:111:        out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/gkdv/_probe.py:112:        out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/gkdv/_probe.py:153:    defended = targets["defended_goal_x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:154:    actual_x = targets["actual_x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:155:    actual_y = targets["actual_y"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:171:        imp_x = targets["ghost_x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:172:        imp_y = targets["ghost_y"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:173:        keep = targets["displacement_m"].to_numpy(dtype=float) >= REALISTIC_MIN_DISP_M
silly_kicks/gkdv/_probe.py:344:    actual_x = targets["actual_x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:345:    actual_y = targets["actual_y"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:346:    imp_x = targets["imp_x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:347:    imp_y = targets["imp_y"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:359:        dist = np.hypot(sub["x"].to_numpy(dtype=float) - ax, sub["y"].to_numpy(dtype=float) - ay)
silly_kicks/gkdv/_probe.py:376:            cf.loc[labels, "x"] = (cf.loc[labels, "x"].to_numpy(dtype=float) + dxs).astype(str(cf["x"].dtype))
silly_kicks/gkdv/_probe.py:377:            cf.loc[labels, "y"] = (cf.loc[labels, "y"].to_numpy(dtype=float) + dys).astype(str(cf["y"].dtype))
silly_kicks/vaep/labels.py:313:        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).to_numpy(dtype=float)
silly_kicks/corners/_encode.py:57:    out["y"] = _FIELD_WIDTH - out["y"].to_numpy(dtype=np.float64)
silly_kicks/corners/_encode.py:78:    init_xy = merged[["x_i", "y_i"]].to_numpy(dtype=np.float64)
silly_kicks/corners/_encode.py:79:    target_xy = merged[["x_t", "y_t"]].to_numpy(dtype=np.float64)
silly_kicks/xthreat/_transitions.py:138:    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
silly_kicks/restdefense/_probe.py:171:    own_goal = targets["own_goal_x"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:172:    actual_x = targets["actual_x"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:173:    actual_y = targets["actual_y"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:188:        imp_x = targets["ghost_x"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:189:        imp_y = targets["ghost_y"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:193:        disp = targets["displacement_m"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:222:    dx = targets["imp_x"].to_numpy(dtype=float) - targets["actual_x"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:223:    dy = targets["imp_y"].to_numpy(dtype=float) - targets["actual_y"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:229:    out["frame_vec_x"] = means["_dx"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:230:    out["frame_vec_y"] = means["_dy"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:252:        out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/restdefense/_probe.py:253:        out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/restdefense/_probe.py:316:    cxs = frame_ref["cx"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:317:    cys = frame_ref["cy"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:318:    vxs = frame_ref["vx"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:319:    vys = frame_ref["vy"].to_numpy(dtype=float)
silly_kicks/restdefense/_probe.py:327:        dist = np.hypot(sub["x"].to_numpy(dtype=float) - cx, sub["y"].to_numpy(dtype=float) - cy)
silly_kicks/restdefense/_probe.py:342:            cf.loc[labels, "x"] = (cf.loc[labels, "x"].to_numpy(dtype=float) + dxs).astype(str(cf["x"].dtype))
silly_kicks/restdefense/_probe.py:343:            cf.loc[labels, "y"] = (cf.loc[labels, "y"].to_numpy(dtype=float) + dys).astype(str(cf["y"].dtype))
silly_kicks/gkdv/_engine.py:340:        work["ball_x"].to_numpy(dtype=float) - work["attacked_goal_x"].to_numpy(dtype=float),
silly_kicks/gkdv/_engine.py:341:        work["ball_y"].to_numpy(dtype=float) - _GOAL_Y,
silly_kicks/gkdv/_engine.py:461:            scored["ghost_x"].to_numpy(dtype=float) - scored["actual_x"].to_numpy(dtype=float),
silly_kicks/gkdv/_engine.py:462:            scored["ghost_y"].to_numpy(dtype=float) - scored["actual_y"].to_numpy(dtype=float),
silly_kicks/gkdv/_engine.py:528:        out.loc[hits, "x"] = joined.loc[hits, "ghost_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/gkdv/_engine.py:529:        out.loc[hits, "y"] = joined.loc[hits, "ghost_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/gkdv/_engine.py:721:    if len(scored_rows) and not np.isfinite(scored_rows[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
silly_kicks/restdefense/_counterfactual.py:276:    if len(computed) and not np.isfinite(computed[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
silly_kicks/restdefense/_counterfactual.py:309:            scored["ghost_x"].to_numpy(dtype=float) - scored["actual_x"].to_numpy(dtype=float),
silly_kicks/restdefense/_counterfactual.py:310:            scored["ghost_y"].to_numpy(dtype=float) - scored["actual_y"].to_numpy(dtype=float),
silly_kicks/restdefense/_counterfactual.py:364:        out.loc[idx, "x"] = joined.loc[idx, "ghost_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/restdefense/_counterfactual.py:365:        out.loc[idx, "y"] = joined.loc[idx, "ghost_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/atomic/vaep/features.py:439:            "start_x": sub["x"].to_numpy(dtype=float),
silly_kicks/atomic/vaep/features.py:440:            "start_y": sub["y"].to_numpy(dtype=float),
silly_kicks/atomic/vaep/features.py:441:            "end_x": (sub["x"] + sub["dx"]).to_numpy(dtype=float),
silly_kicks/atomic/vaep/features.py:442:            "end_y": (sub["y"] + sub["dy"]).to_numpy(dtype=float),
silly_kicks/restdefense/_counting.py:54:    xs = own["x"].to_numpy(dtype=float)
silly_kicks/corners/_effectiveness.py:74:            sum_xg = float(np.nansum(xg[shot_win].to_numpy(dtype=float))) if shot_win.any() else 0.0
silly_kicks/corners/_effectiveness.py:110:        w = joined[fc].to_numpy(dtype=float)
silly_kicks/corners/_routines.py:249:    coeffs = features[feat_cols].to_numpy(dtype=np.float64)
silly_kicks/corners/_routines.py:280:    x = encoding[run_cols].to_numpy(dtype=np.float64)
scripts/run_signoff_power.py:57:        c: int((~np.isfinite(spells[c].to_numpy(dtype=float))).sum())
scripts/run_signoff_power.py:59:        if not np.isfinite(spells[c].to_numpy(dtype=float)).all()
scripts/run_signoff_power.py:69:    return spells.loc[:, list(confounders)].to_numpy(dtype=float)
scripts/run_signoff_power.py:293:            av["arm_value"].to_numpy(dtype=float),
scripts/calibrate_coordination.py:230:        vals = g[col].to_numpy(dtype=np.float64)
silly_kicks/xtgk/_xg_reward.py:25:    np.add.at(out, flat, w.to_numpy(dtype=np.float64))
scripts/build_cross_definition.py:105:            "start_x": a["start_x"].to_numpy(dtype=float)[keep],
silly_kicks/coordination/_windows.py:185:    vals = uniq["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_windows.py:251:    times = ctx["time_seconds"].to_numpy(dtype=np.float64).copy()
silly_kicks/coordination/_windows.py:347:        t = grp["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_windows.py:460:        t = grp["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_windows.py:496:        t = grp["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/spadl/base.py:195:    sx = actions["start_x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/spadl/base.py:196:    sy = actions["start_y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/spadl/base.py:247:    sx = out["start_x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/spadl/base.py:248:    sy = out["start_y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/spadl/utils.py:212:    start_x = sorted_actions["start_x"].to_numpy(dtype=np.float64)
silly_kicks/spadl/utils.py:445:    start_x = out["start_x"].to_numpy(dtype=np.float64)
silly_kicks/spadl/utils.py:446:    start_y = out["start_y"].to_numpy(dtype=np.float64)
silly_kicks/spadl/utils.py:447:    end_x = out["end_x"].to_numpy(dtype=np.float64)
silly_kicks/spadl/utils.py:448:    end_y = out["end_y"].to_numpy(dtype=np.float64)
silly_kicks/spadl/utils.py:665:    time_seconds_arr = sorted_actions["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:176:    x = sorted_actions["x"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:708:    x = out["x"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:709:    y = out["y"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:710:    dx = out["dx"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:711:    dy = out["dy"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:934:    time_seconds_arr = sorted_actions["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/xtgk/_turnover.py:232:            np.add.at(num, flat, sub["_opp_shot_xg"].to_numpy(dtype=float))
silly_kicks/xtgk/_turnover.py:264:        xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
silly_kicks/xtgk/_turnover.py:265:        t = a["time_seconds"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:177:    t = actions["time_seconds"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:256:    gap = df["positioning_gap"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:257:    c5 = df["conceded_threat_5s"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:258:    c10 = df["conceded_threat_10s"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:259:    feas = df["n_feasible_proposals"].to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:271:        "unit_gaps": unit.to_numpy(dtype=float),
scripts/build_tf56_positioning_validity.py:272:        "unit_gap_std": float(np.nanstd(unit.to_numpy(dtype=float))) if unit.shape[0] else float("nan"),
scripts/hf_jobs/headtohead.py:506:        "ox": passes["start_x"].to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:507:        "oy": passes["start_y"].to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:508:        "tx": passes["end_x"].to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:509:        "ty": passes["end_y"].to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:510:        "y": (passes["result_id"] == succ).to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:518:    return {"actions": acts, "y": (acts["result_id"] == spadlconfig.result_id["success"]).to_numpy(dtype=float)}
scripts/hf_jobs/headtohead.py:613:        true_list.append(labs[["gk_x", "gk_y"]].to_numpy(dtype=float))
scripts/hf_jobs/headtohead.py:693:        true_list.append(data[["target_x", "target_y"]].to_numpy(dtype=float))
silly_kicks/tracking/_cover_shadows.py:606:        lane_blockers[["x", "y"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:607:        lane_blockers[["vx", "vy"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:609:        attackers_all[["x", "y"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:610:        attackers_all[["vx", "vy"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:809:    all_att_pos = attackers_outfield[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1321:        lb_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1322:        lb_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1323:        att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1324:        att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)
scripts/build_corner_routines.py:92:    init_xy = snapshots[["init_x", "init_y"]].to_numpy(dtype=float)
scripts/build_corner_routines.py:93:    target_xy = snapshots[["target_x", "target_y"]].to_numpy(dtype=float)
scripts/build_corner_routines.py:105:            grp[["init_x", "init_y"]].to_numpy(dtype=float),
scripts/build_corner_routines.py:106:            grp[["target_x", "target_y"]].to_numpy(dtype=float),
silly_kicks/coordination/_signals.py:492:    present_t = pg["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_signals.py:495:    tp = src["time_seconds"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_signals.py:497:    xr = src["x"].to_numpy(dtype=np.float64)[live]
silly_kicks/coordination/_signals.py:498:    yr = src["y"].to_numpy(dtype=np.float64)[live]
silly_kicks/coordination/_signals.py:515:    observed = _detected_grid(det["time_seconds"].to_numpy(dtype=np.float64), grid_n, fs) & on_pitch
silly_kicks/tracking/_pausa.py:105:    actual = result["obso_actual"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_pausa.py:106:    peak = result["obso_peak"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_pausa.py:107:    optimal = result["obso_optimal"].to_numpy(dtype=np.float64)
scripts/measure_cover_shadow_argmax_agreement.py:237:    all_max_def = df["max_def_exact"].to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence_objective.py:97:    clf.fit(Xtr.to_numpy(dtype=float), ytr)
silly_kicks/tracking/_xshot_occurrence_objective.py:98:    p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
silly_kicks/tracking/preprocess/_velocity.py:163:    x_sm = sorted_frames["x_smoothed"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_velocity.py:164:    y_sm = sorted_frames["y_smoothed"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_velocity.py:165:    raw_x = sorted_frames["x"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_velocity.py:166:    raw_y = sorted_frames["y"].to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:56:    bx = ball["x"].reindex(idx).to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:57:    by = ball["y"].reindex(idx).to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:82:        d2 = (grp["x"].to_numpy(dtype=float) - cx) ** 2 + (grp["y"].to_numpy(dtype=float) - cy) ** 2
silly_kicks/causal/_confounders.py:92:            "time_seconds": spells["entry_time"].to_numpy(dtype=float),
silly_kicks/causal/_confounders.py:102:    return pressed["pressure_on_actor__bekkers_pi"].to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:108:    return (end - spells["entry_time"]).to_numpy(dtype=float)
scripts/_coordination_occlusion.py:24:    x = frames["x"].to_numpy(dtype=np.float64)
scripts/_coordination_occlusion.py:39:    return ball_x.reindex(idx).to_numpy(dtype=np.float64)
scripts/_coordination_occlusion.py:56:            vals = out[axis].to_numpy(dtype=np.float64)[sub]
scripts/build_tf19_instrument_responsiveness.py:176:        real = g["realistic_abs"].to_numpy(dtype=float)
scripts/build_tf19_instrument_responsiveness.py:177:        sat = g["saturating_abs"].to_numpy(dtype=float)
scripts/build_tf19_instrument_responsiveness.py:178:        gk = g["gk_abs"].to_numpy(dtype=float)
scripts/build_tf19_instrument_responsiveness.py:179:        nd = g["nd_abs"].to_numpy(dtype=float)
scripts/build_tf19_instrument_responsiveness.py:181:        placebo = g[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
scripts/build_tf19_instrument_responsiveness.py:442:    defended = base["defended_goal_x"].to_numpy(dtype=float)
scripts/build_tf19_instrument_responsiveness.py:443:    actual_x = base["actual_x"].to_numpy(dtype=float)
silly_kicks/tracking/_ball_contact.py:191:    px = players["x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:192:    py = players["y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:391:        tt = seg_df["time_seconds"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:392:        bx = seg_df["x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:393:        by = seg_df["y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:394:        bz = seg_df["z"].to_numpy(dtype=np.float64, copy=True) if "z" in seg_df else np.full(len(seg_df), np.nan)
silly_kicks/tracking/_ball_contact.py:403:        fr = seg_df["frame_rate"].to_numpy(dtype=np.float64, copy=True) if "frame_rate" in seg_df else np.array([])
silly_kicks/tracking/_ball_contact.py:577:        tvals = out[tcol].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_contact.py:589:            c_t = cper["time_seconds"].to_numpy(dtype=np.float64)
scripts/build_territory_ranking_census.py:192:    y = pd.to_numeric(df[_METRIC_COL], errors="coerce").to_numpy(dtype=float)
silly_kicks/tracking/_occurrence_labels.py:31:        ev_groups[key] = np.sort(grp["time_seconds"].to_numpy(dtype=float))
silly_kicks/tracking/_occurrence_labels.py:35:    ts = frames_index["time_seconds"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:152:        xs = atk["x"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:153:        ys = atk["y"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:236:    base = feats["box_off_def_ratio"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:237:    clamp = meta["box_off_def_ratio_clamped"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:334:    gx = frames["x"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:335:    gy = frames["y"].to_numpy(dtype=float)
silly_kicks/xtgk/_resolved_geometry.py:116:        return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence.py:240:                defending["x"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:241:                defending["y"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:250:                attacking["x"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:251:                attacking["y"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:417:            .to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence.py:509:        clf.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
silly_kicks/tracking/_xshot_occurrence.py:526:        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
scripts/hf_jobs/adopt.py:628:                x, y = a[col].to_numpy(dtype=np.float64), b[col].to_numpy(dtype=np.float64)
silly_kicks/tracking/preprocess/_smoothing.py:222:        x_vals = sorted_frames.loc[idx_arr, "x"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_smoothing.py:223:        y_vals = sorted_frames.loc[idx_arr, "y"].to_numpy(dtype=float)
silly_kicks/causal/opportunities.py:386:    return np.sort(actions.loc[sel, "time_seconds"].to_numpy(dtype=float))
silly_kicks/causal/opportunities.py:473:    ts = sub["time_seconds"].to_numpy(dtype=float)
silly_kicks/causal/opportunities.py:480:    xs = sub.loc[keep, "start_x"].to_numpy(dtype=float)
silly_kicks/causal/opportunities.py:481:    ys = sub.loc[keep, "start_y"].to_numpy(dtype=float)
scripts/_coordination_hypotheses.py:51:    t = uniq["time_seconds"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:178:    long_deg = tt.loc[tt["signal_a"] == "centroid_x", "coord_rp_mean_deg"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:197:    r = tt["coord_xc_r_at_max"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:198:    lag = tt["coord_xc_lag_s"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:233:        shot = early.loc[early["terminal_action"] == "shot", col].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:234:        tackle = early.loc[early["terminal_action"] == "tackle", col].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:259:        freqs = s["coord_median_freq_cpm"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:313:    diffs = (in_mean - out_mean).dropna().to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:333:        v = dyad.loc[dyad["axis"] == axis, "coord_rp_pct_near_in_phase"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:355:    bc = r["coord_rsi_bimodality_coefficient"].to_numpy(dtype=np.float64) if len(r) else np.array([])
scripts/_coordination_hypotheses.py:408:            change_by_half[key if isinstance(key, tuple) else (key,)] = g["time"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:412:        sw = g["time"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_carrier.py:97:    bx_arr = frame_ball["bx"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_carrier.py:98:    by_arr = frame_ball["by"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_carrier.py:114:        .to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_carrier.py:134:    px[f_idx, slot] = placed["x"].to_numpy(dtype=np.float64)[order]
silly_kicks/tracking/_ball_carrier.py:135:    py[f_idx, slot] = placed["y"].to_numpy(dtype=np.float64)[order]
silly_kicks/tracking/_ball_carrier.py:137:        pvx[f_idx, slot] = placed["vx"].to_numpy(dtype=np.float64)[order]
silly_kicks/tracking/_ball_carrier.py:138:        pvy[f_idx, slot] = placed["vy"].to_numpy(dtype=np.float64)[order]
scripts/derive_coordination_params.py:150:    u_m = grp["m"].transform("size").to_numpy(dtype=np.float64)  # units in each (provider, match)
scripts/derive_coordination_params.py:154:        m_p = df["p"].map(matches_per_p).to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:552:        tp = pg["time_seconds"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:554:            vals = pg[axis].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:611:        out[str(provider)] = provider_cutoff(g["cutoff_hz"].to_numpy(dtype=np.float64), w)
scripts/derive_coordination_params.py:808:    w = agg["width_m"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:809:    rate = (agg["det"] / agg["tot"].where(agg["tot"] > 0, 1)).to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:826:    t = frames["time_seconds"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:848:    x = frames["x"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:849:    y = frames["y"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:888:            v = pg[axis].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:942:                gfull = grp["_full"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1113:        vals = g["value"].to_numpy(dtype=np.float64) if g is not None else np.empty(0)
scripts/derive_coordination_params.py:1130:    vals = med_df["value"].to_numpy(dtype=np.float64) if len(med_df) else np.empty(0)
scripts/derive_coordination_params.py:1157:        vals = grp["value"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1249:        full_vals = gf["value"].to_numpy(dtype=np.float64) if len(gf) else np.empty(0)
scripts/derive_coordination_params.py:1270:                vals = g["value"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1376:        nv = noise["value"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1387:            vals = grp["value"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1434:            a_df["cutoff_hz"].to_numpy(dtype=np.float64),
scripts/derive_coordination_params.py:1514:        return _summary(df[col].to_numpy(dtype=np.float64), df["provider"].to_numpy(), df["match_id"].to_numpy())
scripts/derive_coordination_params.py:1546:        vals = u["value"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:1686:    gk = occ_df[occ_df["quantity"] == "gk_rate"]["value"].to_numpy(dtype=np.float64) if len(occ_df) else np.array([])
scripts/check_stage1_argmax.py:138:    empty = [c for c in _VELOCITY_COLUMNS if not np.isfinite(frames[c].to_numpy(dtype=float)).any()]
scripts/measure_ball_contact_validation.py:45:        vals = ts[period == p].to_numpy(dtype=float)
scripts/measure_ball_contact_validation.py:56:        out[(per, pid)] = np.sort(g["time_seconds"].to_numpy(dtype=np.float64))
scripts/measure_ball_contact_validation.py:126:    offsets = ev["value"].to_numpy(dtype=float)
scripts/measure_ball_contact_validation.py:131:    conf = ct["value"].to_numpy(dtype=float)
scripts/measure_ball_contact_validation.py:132:    dv = ct["dv"].to_numpy(dtype=float)
silly_kicks/xtgk/_pressure_levels.py:60:        p_all = pressure.to_numpy(dtype=float)
silly_kicks/xtgk/_pressure_levels.py:102:        p = pressure.to_numpy(dtype=float)
silly_kicks/tracking/features.py:487:        return actions[col].to_numpy(dtype=float) if arr is None else arr
silly_kicks/tracking/features.py:492:        ex=actions["end_x"].to_numpy(dtype=float),
silly_kicks/tracking/features.py:493:        ey=actions["end_y"].to_numpy(dtype=float),
silly_kicks/tracking/features.py:6602:        _frac = _cov["visible_area_fraction"].to_numpy(dtype=float)
scripts/gen_space_creation_mirror_golden.py:58:    created = out["space_created_m2"].to_numpy(dtype=float)
scripts/gen_space_creation_mirror_golden.py:59:    denied = out["space_denied_m2_opponent"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_interpolation.py:93:    x = sorted_frames["x"].to_numpy(dtype=float, copy=True)
silly_kicks/tracking/preprocess/_interpolation.py:94:    y = sorted_frames["y"].to_numpy(dtype=float, copy=True)
silly_kicks/tracking/preprocess/_interpolation.py:95:    t = sorted_frames["time_seconds"].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:564:    vec = feats.sort_values("slot_index")[names].iloc[0].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:663:        X = data[self._feature_names()].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_outfield.py:664:        y_x = data["target_x"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_outfield.py:665:        y_y = data["target_y"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_outfield.py:713:        X = features[self._feature_names()].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_outfield.py:1152:        _grx = out["ghost_gr_x"].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:1153:        _gry = out["ghost_gr_y"].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:1189:        ys = gg["ghost_gr_y"].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:1190:        xs = gg["ghost_gr_x"].to_numpy(dtype=float)
scripts/_coordination_corpus.py:717:        times = g["start_time_s"].to_numpy(dtype=float)
silly_kicks/territory/_counterfactual.py:164:            end_x = opp["end_x"].to_numpy(dtype=float)
silly_kicks/territory/_counterfactual.py:165:            end_y = opp["end_y"].to_numpy(dtype=float)
silly_kicks/territory/_counterfactual.py:166:            start_x = opp["start_x"].to_numpy(dtype=float)
silly_kicks/territory/_counterfactual.py:167:            start_y = opp["start_y"].to_numpy(dtype=float)
silly_kicks/territory/_counterfactual.py:168:            xt_end = opp["_xt_end"].to_numpy(dtype=float)
silly_kicks/tracking/utils.py:588:    a_times = actions["time_seconds"].to_numpy(dtype=float)
silly_kicks/tracking/utils.py:1018:    z = ball["z"].to_numpy(dtype=float) if "z" in ball.columns else np.array([], dtype=float)
scripts/make_xcross_directional_fixture.py:238:    bx = frozen["ball_x"].to_numpy(dtype=float)
scripts/make_xcross_directional_fixture.py:239:    by = frozen["ball_y"].to_numpy(dtype=float)
scripts/make_xcross_directional_fixture.py:240:    gx = frozen["goal_x"].to_numpy(dtype=float)
scripts/make_ghost_gk_golden.py:55:        ghost_gk_x=gk["ghost_gk_x"].to_numpy(dtype=float),
scripts/make_ghost_gk_golden.py:56:        ghost_gk_y=gk["ghost_gk_y"].to_numpy(dtype=float),
scripts/make_ghost_gk_golden.py:57:        ghost_gk_density_spread=gk["ghost_gk_density_spread"].to_numpy(dtype=float),
silly_kicks/xtgk/_possession_value.py:84:        mask &= np.isfinite(pd.to_numeric(actions[col], errors="coerce").to_numpy(dtype=float))
silly_kicks/tracking/_xcross_attempt_objective.py:90:    clf.fit(Xtr.to_numpy(dtype=float), ytr)
silly_kicks/tracking/_xcross_attempt_objective.py:91:    p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
silly_kicks/tracking/preprocess/_guard.py:36:        speed = out["speed"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_guard.py:46:        accel = out["accel"].to_numpy(dtype=float)
scripts/validate_das_native_parity.py:328:            "team_as": team["as"].to_numpy(dtype=float),
scripts/validate_das_native_parity.py:329:            "team_das": team["das"].to_numpy(dtype=float),
scripts/validate_das_native_parity.py:331:            "player_as": player["as"].to_numpy(dtype=float),
scripts/validate_das_native_parity.py:332:            "player_das": player["das"].to_numpy(dtype=float),
scripts/train_gk_retention.py:40:    y_full = retains(actions).to_numpy(dtype=float)  # 1.0 / 0.0 / NaN (truncated windows)
silly_kicks/territory/_compute.py:180:        def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)
silly_kicks/territory/_compute.py:268:        def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)
scripts/train_ghost_gk.py:463:                _depth = meta["gk_x_gr"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_densify.py:31:    x = gi["x"].reindex(full).to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_densify.py:32:    y = gi["y"].reindex(full).to_numpy(dtype=float)
silly_kicks/tracking/_ghost_gk.py:552:    ts = pd.to_numeric(nat["team_score"], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_gk.py:553:    os_ = pd.to_numeric(nat["opponent_team_score"], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_gk.py:1859:            .to_numpy(dtype=float)
silly_kicks/xtgk/_moves.py:93:    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_xcross_attempt.py:237:    y = np.array([_geo.to_goal_relative_y(v, goal_x=goal_x) for v in f["y"].to_numpy(dtype=float)])
silly_kicks/tracking/_xcross_attempt.py:618:            .to_numpy(dtype=float)
silly_kicks/tracking/_xcross_attempt.py:698:        clf.fit(features.to_numpy(dtype=float), np.asarray(labels, dtype=int))  # M1: numpy
silly_kicks/tracking/_xcross_attempt.py:710:        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
scripts/validate_territory_counterfactual.py:271:    sx = completed_passes["start_x"].to_numpy(dtype=float)
scripts/validate_territory_counterfactual.py:272:    sy = completed_passes["start_y"].to_numpy(dtype=float)
scripts/validate_territory_counterfactual.py:273:    ex = completed_passes["end_x"].to_numpy(dtype=float)
scripts/validate_territory_counterfactual.py:274:    ey = completed_passes["end_y"].to_numpy(dtype=float)
scripts/train_xshot_occurrence.py:298:    Mirrors the shipped fit path (``.to_numpy(dtype=float)`` then ``booster.feature_names =
scripts/train_xshot_occurrence.py:325:    clf.fit(X.to_numpy(dtype=float), y)
scripts/validate_coordination_numerics.py:141:                x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
silly_kicks/tracking/_elastic_sync.py:234:        merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:235:        merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:238:    zc = merged["ball_z"].to_numpy(dtype=float)
silly_kicks/tracking/_elastic_sync.py:252:        accel = bgrp["ball_accel"].to_numpy(dtype=float)
silly_kicks/tracking/_elastic_sync.py:253:        boundary = bgrp["boundary_dist"].to_numpy(dtype=float)
silly_kicks/tracking/_elastic_sync.py:613:            merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:614:            merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
scripts/validate_corners.py:147:    off = (s[possession_col] - s[contact_col]).to_numpy(dtype=float)
scripts/validate_corners.py:417:        feats["dom_routine"] = dominant_routine(feats[feat_cols].to_numpy(dtype=float))
scripts/validate_corners.py:512:                div[["init_x", "init_y"]].to_numpy(dtype=float),
scripts/validate_corners.py:513:                div[["kick_x", "kick_y"]].to_numpy(dtype=float),
scripts/validate_team_coordination.py:182:            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
scripts/validate_team_coordination.py:238:            v = gg["value"].to_numpy(dtype=np.float64)
scripts/validate_team_coordination.py:252:    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/xtgk/_metric.py:46:        got = pd.to_numeric(retention_features[col], errors="coerce").to_numpy(dtype=float)
silly_kicks/xtgk/_metric.py:47:        exp = expected[col].to_numpy(dtype=float)
silly_kicks/territorial_defense/_probe.py:81:    x = out["x"].to_numpy(dtype=float, copy=True)
silly_kicks/territorial_defense/_probe.py:82:    y = out["y"].to_numpy(dtype=float, copy=True)
silly_kicks/territorial_defense/_probe.py:112:    xs = frame["x"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_probe.py:113:    ys = frame["y"].to_numpy(dtype=float)
silly_kicks/xtgk/_empirical.py:42:        xg = grp[xg_column].fillna(0.0).to_numpy(dtype=float)
silly_kicks/xtgk/_empirical.py:91:            np.add.at(num, flat, sub["_outcome"].to_numpy(dtype=float))
scripts/validate_territorial_defense.py:379:    real = df["realistic_abs"].to_numpy(dtype=float)
scripts/validate_territorial_defense.py:380:    sat = df["saturating_abs"].to_numpy(dtype=float)
scripts/validate_territorial_defense.py:381:    nd = df["nd_abs"].to_numpy(dtype=float)
scripts/validate_territorial_defense.py:383:    placebo = df[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
silly_kicks/tracking/_receiver.py:133:        d = np.linalg.norm(teammates[["x", "y"]].to_numpy(dtype=np.float64) - pxy, axis=1)
silly_kicks/tracking/_receiver.py:205:    def_pos = opp_outfield[["x", "y"]].to_numpy(dtype=np.float64)
scripts/train_xcross_attempt.py:448:    clf.fit(X.to_numpy(dtype=float), y)
scripts/train_xcross_attempt.py:775:    sd = X["score_differential"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_compute.py:211:        own_half_xy = grp[grp["start_x"] < params.own_half_max_x][["start_x", "start_y"]].to_numpy(dtype=float)
silly_kicks/territorial_defense/_compute.py:227:        end_x = opp["end_x"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_compute.py:228:        end_y = opp["end_y"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_arms.py:132:    xs = frame["x"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_arms.py:133:    ys = frame["y"].to_numpy(dtype=float)
```

## Classification

```text
$ python docs/research/action-time-repair/pandas-floor/scripts/classify_b.py . <the hits above>
declared masked columns: 74
AT_RISK: 54
SAFE: 160
UNKNOWN: 122
--- AT_RISK
silly_kicks/gk_decision/_reconstruct.py:216 ['x', 'y'] | d = np.hypot(cand["x"].to_numpy(dtype=float) - pass_end[0], cand["y"].to_numpy(dtype=float) - pass_end[1])
silly_kicks/gk_decision/_reconstruct.py:223 ['x'] | cand_x = cand["x"].to_numpy(dtype=float)
silly_kicks/gk_decision/_reconstruct.py:224 ['y'] | cand_y = cand["y"].to_numpy(dtype=float)
silly_kicks/restdefense/_structure.py:163 ['x'] | x = a["x"].to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:359 ['x', 'y'] | dist = np.hypot(sub["x"].to_numpy(dtype=float) - ax, sub["y"].to_numpy(dtype=float) - ay)
silly_kicks/corners/_encode.py:57 ['y'] | out["y"] = _FIELD_WIDTH - out["y"].to_numpy(dtype=np.float64)
silly_kicks/restdefense/_probe.py:327 ['x', 'y'] | dist = np.hypot(sub["x"].to_numpy(dtype=float) - cx, sub["y"].to_numpy(dtype=float) - cy)
silly_kicks/gkdv/_engine.py:340 ['ball_x'] | work["ball_x"].to_numpy(dtype=float) - work["attacked_goal_x"].to_numpy(dtype=float),
silly_kicks/gkdv/_engine.py:341 ['ball_y'] | work["ball_y"].to_numpy(dtype=float) - _GOAL_Y,
silly_kicks/atomic/vaep/features.py:439 ['x'] | "start_x": sub["x"].to_numpy(dtype=float),
silly_kicks/atomic/vaep/features.py:440 ['y'] | "start_y": sub["y"].to_numpy(dtype=float),
silly_kicks/restdefense/_counting.py:54 ['x'] | xs = own["x"].to_numpy(dtype=float)
silly_kicks/atomic/spadl/utils.py:176 ['x'] | x = sorted_actions["x"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:708 ['x'] | x = out["x"].to_numpy(dtype=np.float64)
silly_kicks/atomic/spadl/utils.py:709 ['y'] | y = out["y"].to_numpy(dtype=np.float64)
silly_kicks/coordination/_signals.py:497 ['x'] | xr = src["x"].to_numpy(dtype=np.float64)[live]
silly_kicks/coordination/_signals.py:498 ['y'] | yr = src["y"].to_numpy(dtype=np.float64)[live]
silly_kicks/tracking/preprocess/_velocity.py:165 ['x'] | raw_x = sorted_frames["x"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_velocity.py:166 ['y'] | raw_y = sorted_frames["y"].to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:82 ['x', 'y'] | d2 = (grp["x"].to_numpy(dtype=float) - cx) ** 2 + (grp["y"].to_numpy(dtype=float) - cy) ** 2
scripts/_coordination_occlusion.py:24 ['x'] | x = frames["x"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_contact.py:191 ['x'] | px = players["x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:192 ['y'] | py = players["y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:392 ['x'] | bx = seg_df["x"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:393 ['y'] | by = seg_df["y"].to_numpy(dtype=np.float64, copy=True)
silly_kicks/tracking/_ball_contact.py:403 ['frame_rate'] | fr = seg_df["frame_rate"].to_numpy(dtype=np.float64, copy=True) if "frame_rate" in seg_df else np.array([])
scripts/measure_box_constant_delta.py:152 ['x'] | xs = atk["x"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:153 ['y'] | ys = atk["y"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:334 ['x'] | gx = frames["x"].to_numpy(dtype=float)
scripts/measure_box_constant_delta.py:335 ['y'] | gy = frames["y"].to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence.py:240 ['x'] | defending["x"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:241 ['y'] | defending["y"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:250 ['x'] | attacking["x"].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence.py:251 ['y'] | attacking["y"].to_numpy(dtype=float),
silly_kicks/tracking/_ball_carrier.py:134 ['x'] | px[f_idx, slot] = placed["x"].to_numpy(dtype=np.float64)[order]
silly_kicks/tracking/_ball_carrier.py:135 ['y'] | py[f_idx, slot] = placed["y"].to_numpy(dtype=np.float64)[order]
scripts/derive_coordination_params.py:848 ['x'] | x = frames["x"].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:849 ['y'] | y = frames["y"].to_numpy(dtype=np.float64)
silly_kicks/tracking/preprocess/_interpolation.py:93 ['x'] | x = sorted_frames["x"].to_numpy(dtype=float, copy=True)
silly_kicks/tracking/preprocess/_interpolation.py:94 ['y'] | y = sorted_frames["y"].to_numpy(dtype=float, copy=True)
scripts/make_xcross_directional_fixture.py:238 ['ball_x'] | bx = frozen["ball_x"].to_numpy(dtype=float)
scripts/make_xcross_directional_fixture.py:239 ['ball_y'] | by = frozen["ball_y"].to_numpy(dtype=float)
silly_kicks/tracking/_xcross_attempt.py:237 ['y'] | y = np.array([_geo.to_goal_relative_y(v, goal_x=goal_x) for v in f["y"].to_numpy(dtype=float)])
silly_kicks/tracking/_elastic_sync.py:234 ['x', 'ball_x'] | merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:235 ['y', 'ball_y'] | merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:238 ['ball_z'] | zc = merged["ball_z"].to_numpy(dtype=float)
silly_kicks/tracking/_elastic_sync.py:613 ['x', 'ball_x'] | merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
silly_kicks/tracking/_elastic_sync.py:614 ['y', 'ball_y'] | merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
silly_kicks/territorial_defense/_probe.py:81 ['x'] | x = out["x"].to_numpy(dtype=float, copy=True)
silly_kicks/territorial_defense/_probe.py:82 ['y'] | y = out["y"].to_numpy(dtype=float, copy=True)
silly_kicks/territorial_defense/_probe.py:112 ['x'] | xs = frame["x"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_probe.py:113 ['y'] | ys = frame["y"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_arms.py:132 ['x'] | xs = frame["x"].to_numpy(dtype=float)
silly_kicks/territorial_defense/_arms.py:133 ['y'] | ys = frame["y"].to_numpy(dtype=float)
--- SAFE column names (count)
[('time_seconds', 23), ('value', 12), ('start_x', 8), ('actual_x', 7), ('actual_y', 6), ('start_y', 6), ('end_x', 6), ('end_y', 6), ('ghost_x', 4), ('ghost_y', 4), ('defended_goal_x', 2), ('displacement_m', 2), ('imp_x', 2), ('imp_y', 2), ('vx', 2), ('vy', 2), ('realistic_abs', 2), ('saturating_abs', 2), ('nd_abs', 2), ('z', 2), ('time', 2), ('cutoff_hz', 2), ('ghost_gr_x', 2), ('ghost_gr_y', 2), ('as', 2), ('das', 2), ('own_goal_x', 1), ('_dx', 1), ('_dy', 1), ('cx', 1), ('cy', 1), ('arm_value', 1), ('dx', 1), ('dy', 1), ('_opp_shot_xg', 1), ('positioning_gap', 1), ('conceded_threat_5s', 1), ('conceded_threat_10s', 1), ('n_feasible_proposals', 1), ('obso_actual', 1)]
--- UNKNOWN (a variable subject: searched, not read; R80)
silly_kicks/corners/_zones.py:342 | return sub[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/positioning/_probe.py:203 | out[float(h)] = scored.to_numpy(dtype=float)
silly_kicks/gkdv/_probe.py:111 | out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/gkdv/_probe.py:112 | out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/gkdv/_probe.py:376 | cf.loc[labels, "x"] = (cf.loc[labels, "x"].to_numpy(dtype=float) + dxs).astype(str(cf["x"].dtype))
silly_kicks/gkdv/_probe.py:377 | cf.loc[labels, "y"] = (cf.loc[labels, "y"].to_numpy(dtype=float) + dys).astype(str(cf["y"].dtype))
silly_kicks/vaep/labels.py:313 | xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0).to_numpy(dtype=float)
silly_kicks/corners/_encode.py:78 | init_xy = merged[["x_i", "y_i"]].to_numpy(dtype=np.float64)
silly_kicks/corners/_encode.py:79 | target_xy = merged[["x_t", "y_t"]].to_numpy(dtype=np.float64)
silly_kicks/xthreat/_transitions.py:138 | end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
silly_kicks/restdefense/_probe.py:252 | out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/restdefense/_probe.py:253 | out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/restdefense/_probe.py:342 | cf.loc[labels, "x"] = (cf.loc[labels, "x"].to_numpy(dtype=float) + dxs).astype(str(cf["x"].dtype))
silly_kicks/restdefense/_probe.py:343 | cf.loc[labels, "y"] = (cf.loc[labels, "y"].to_numpy(dtype=float) + dys).astype(str(cf["y"].dtype))
silly_kicks/gkdv/_engine.py:528 | out.loc[hits, "x"] = joined.loc[hits, "ghost_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/gkdv/_engine.py:529 | out.loc[hits, "y"] = joined.loc[hits, "ghost_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/gkdv/_engine.py:721 | if len(scored_rows) and not np.isfinite(scored_rows[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
silly_kicks/restdefense/_counterfactual.py:276 | if len(computed) and not np.isfinite(computed[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
silly_kicks/restdefense/_counterfactual.py:364 | out.loc[idx, "x"] = joined.loc[idx, "ghost_x"].to_numpy(dtype=float).astype(str(out["x"].dtype))
silly_kicks/restdefense/_counterfactual.py:365 | out.loc[idx, "y"] = joined.loc[idx, "ghost_y"].to_numpy(dtype=float).astype(str(out["y"].dtype))
silly_kicks/atomic/vaep/features.py:441 | "end_x": (sub["x"] + sub["dx"]).to_numpy(dtype=float),
silly_kicks/atomic/vaep/features.py:442 | "end_y": (sub["y"] + sub["dy"]).to_numpy(dtype=float),
silly_kicks/corners/_effectiveness.py:74 | sum_xg = float(np.nansum(xg[shot_win].to_numpy(dtype=float))) if shot_win.any() else 0.0
silly_kicks/corners/_effectiveness.py:110 | w = joined[fc].to_numpy(dtype=float)
silly_kicks/corners/_routines.py:249 | coeffs = features[feat_cols].to_numpy(dtype=np.float64)
silly_kicks/corners/_routines.py:280 | x = encoding[run_cols].to_numpy(dtype=np.float64)
scripts/run_signoff_power.py:57 | c: int((~np.isfinite(spells[c].to_numpy(dtype=float))).sum())
scripts/run_signoff_power.py:59 | if not np.isfinite(spells[c].to_numpy(dtype=float)).all()
scripts/run_signoff_power.py:69 | return spells.loc[:, list(confounders)].to_numpy(dtype=float)
scripts/calibrate_coordination.py:230 | vals = g[col].to_numpy(dtype=np.float64)
silly_kicks/xtgk/_xg_reward.py:25 | np.add.at(out, flat, w.to_numpy(dtype=np.float64))
silly_kicks/xtgk/_turnover.py:264 | xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
scripts/build_tf56_positioning_validity.py:271 | "unit_gaps": unit.to_numpy(dtype=float),
scripts/build_tf56_positioning_validity.py:272 | "unit_gap_std": float(np.nanstd(unit.to_numpy(dtype=float))) if unit.shape[0] else float("nan"),
scripts/hf_jobs/headtohead.py:510 | "y": (passes["result_id"] == succ).to_numpy(dtype=float),
scripts/hf_jobs/headtohead.py:518 | return {"actions": acts, "y": (acts["result_id"] == spadlconfig.result_id["success"]).to_numpy(dtype=float)}
scripts/hf_jobs/headtohead.py:613 | true_list.append(labs[["gk_x", "gk_y"]].to_numpy(dtype=float))
scripts/hf_jobs/headtohead.py:693 | true_list.append(data[["target_x", "target_y"]].to_numpy(dtype=float))
silly_kicks/tracking/_cover_shadows.py:606 | lane_blockers[["x", "y"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:607 | lane_blockers[["vx", "vy"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:609 | attackers_all[["x", "y"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:610 | attackers_all[["vx", "vy"]].to_numpy(dtype=np.float64),
silly_kicks/tracking/_cover_shadows.py:809 | all_att_pos = attackers_outfield[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1321 | lb_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1322 | lb_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1323 | att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_cover_shadows.py:1324 | att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)
scripts/build_corner_routines.py:92 | init_xy = snapshots[["init_x", "init_y"]].to_numpy(dtype=float)
scripts/build_corner_routines.py:93 | target_xy = snapshots[["target_x", "target_y"]].to_numpy(dtype=float)
scripts/build_corner_routines.py:105 | grp[["init_x", "init_y"]].to_numpy(dtype=float),
scripts/build_corner_routines.py:106 | grp[["target_x", "target_y"]].to_numpy(dtype=float),
silly_kicks/tracking/_xshot_occurrence_objective.py:97 | clf.fit(Xtr.to_numpy(dtype=float), ytr)
silly_kicks/tracking/_xshot_occurrence_objective.py:98 | p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
silly_kicks/causal/_confounders.py:56 | bx = ball["x"].reindex(idx).to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:57 | by = ball["y"].reindex(idx).to_numpy(dtype=float)
silly_kicks/causal/_confounders.py:108 | return (end - spells["entry_time"]).to_numpy(dtype=float)
scripts/_coordination_occlusion.py:39 | return ball_x.reindex(idx).to_numpy(dtype=np.float64)
scripts/_coordination_occlusion.py:56 | vals = out[axis].to_numpy(dtype=np.float64)[sub]
scripts/build_tf19_instrument_responsiveness.py:181 | placebo = g[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
silly_kicks/tracking/_ball_contact.py:577 | tvals = out[tcol].to_numpy(dtype=np.float64)
scripts/build_territory_ranking_census.py:192 | y = pd.to_numeric(df[_METRIC_COL], errors="coerce").to_numpy(dtype=float)
silly_kicks/xtgk/_resolved_geometry.py:116 | return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence.py:417 | .to_numpy(dtype=float)
silly_kicks/tracking/_xshot_occurrence.py:509 | clf.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
silly_kicks/tracking/_xshot_occurrence.py:526 | dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
scripts/hf_jobs/adopt.py:628 | x, y = a[col].to_numpy(dtype=np.float64), b[col].to_numpy(dtype=np.float64)
silly_kicks/tracking/preprocess/_smoothing.py:222 | x_vals = sorted_frames.loc[idx_arr, "x"].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_smoothing.py:223 | y_vals = sorted_frames.loc[idx_arr, "y"].to_numpy(dtype=float)
silly_kicks/causal/opportunities.py:386 | return np.sort(actions.loc[sel, "time_seconds"].to_numpy(dtype=float))
silly_kicks/causal/opportunities.py:480 | xs = sub.loc[keep, "start_x"].to_numpy(dtype=float)
silly_kicks/causal/opportunities.py:481 | ys = sub.loc[keep, "start_y"].to_numpy(dtype=float)
scripts/_coordination_hypotheses.py:178 | long_deg = tt.loc[tt["signal_a"] == "centroid_x", "coord_rp_mean_deg"].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:233 | shot = early.loc[early["terminal_action"] == "shot", col].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:234 | tackle = early.loc[early["terminal_action"] == "tackle", col].to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:313 | diffs = (in_mean - out_mean).dropna().to_numpy(dtype=np.float64)
scripts/_coordination_hypotheses.py:333 | v = dyad.loc[dyad["axis"] == axis, "coord_rp_pct_near_in_phase"].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ball_carrier.py:114 | .to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:150 | u_m = grp["m"].transform("size").to_numpy(dtype=np.float64)  # units in each (provider, match)
scripts/derive_coordination_params.py:154 | m_p = df["p"].map(matches_per_p).to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:554 | vals = pg[axis].to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:809 | rate = (agg["det"] / agg["tot"].where(agg["tot"] > 0, 1)).to_numpy(dtype=np.float64)
scripts/derive_coordination_params.py:888 | v = pg[axis].to_numpy(dtype=np.float64)
scripts/check_stage1_argmax.py:138 | empty = [c for c in _VELOCITY_COLUMNS if not np.isfinite(frames[c].to_numpy(dtype=float)).any()]
scripts/measure_ball_contact_validation.py:45 | vals = ts[period == p].to_numpy(dtype=float)
silly_kicks/xtgk/_pressure_levels.py:60 | p_all = pressure.to_numpy(dtype=float)
silly_kicks/xtgk/_pressure_levels.py:102 | p = pressure.to_numpy(dtype=float)
silly_kicks/tracking/features.py:487 | return actions[col].to_numpy(dtype=float) if arr is None else arr
silly_kicks/tracking/_ghost_outfield.py:564 | vec = feats.sort_values("slot_index")[names].iloc[0].to_numpy(dtype=float)
silly_kicks/tracking/_ghost_outfield.py:663 | X = data[self._feature_names()].to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_outfield.py:713 | X = features[self._feature_names()].to_numpy(dtype=np.float64)
silly_kicks/xtgk/_possession_value.py:84 | mask &= np.isfinite(pd.to_numeric(actions[col], errors="coerce").to_numpy(dtype=float))
silly_kicks/tracking/_xcross_attempt_objective.py:90 | clf.fit(Xtr.to_numpy(dtype=float), ytr)
silly_kicks/tracking/_xcross_attempt_objective.py:91 | p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
scripts/train_gk_retention.py:40 | y_full = retains(actions).to_numpy(dtype=float)  # 1.0 / 0.0 / NaN (truncated windows)
silly_kicks/territory/_compute.py:180 | def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)
silly_kicks/territory/_compute.py:268 | def_xy = drows[["start_x", "start_y"]].to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_densify.py:31 | x = gi["x"].reindex(full).to_numpy(dtype=float)
silly_kicks/tracking/preprocess/_densify.py:32 | y = gi["y"].reindex(full).to_numpy(dtype=float)
silly_kicks/tracking/_ghost_gk.py:552 | ts = pd.to_numeric(nat["team_score"], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_gk.py:553 | os_ = pd.to_numeric(nat["opponent_team_score"], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/tracking/_ghost_gk.py:1859 | .to_numpy(dtype=float)
silly_kicks/xtgk/_moves.py:93 | end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
silly_kicks/tracking/_xcross_attempt.py:618 | .to_numpy(dtype=float)
silly_kicks/tracking/_xcross_attempt.py:698 | clf.fit(features.to_numpy(dtype=float), np.asarray(labels, dtype=int))  # M1: numpy
silly_kicks/tracking/_xcross_attempt.py:710 | dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
scripts/train_xshot_occurrence.py:298 | Mirrors the shipped fit path (``.to_numpy(dtype=float)`` then ``booster.feature_names =
scripts/train_xshot_occurrence.py:325 | clf.fit(X.to_numpy(dtype=float), y)
scripts/validate_coordination_numerics.py:141 | x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
scripts/validate_corners.py:147 | off = (s[possession_col] - s[contact_col]).to_numpy(dtype=float)
scripts/validate_corners.py:417 | feats["dom_routine"] = dominant_routine(feats[feat_cols].to_numpy(dtype=float))
scripts/validate_corners.py:512 | div[["init_x", "init_y"]].to_numpy(dtype=float),
scripts/validate_corners.py:513 | div[["kick_x", "kick_y"]].to_numpy(dtype=float),
scripts/validate_team_coordination.py:182 | v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
scripts/validate_team_coordination.py:252 | v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
silly_kicks/xtgk/_metric.py:46 | got = pd.to_numeric(retention_features[col], errors="coerce").to_numpy(dtype=float)
silly_kicks/xtgk/_metric.py:47 | exp = expected[col].to_numpy(dtype=float)
silly_kicks/xtgk/_empirical.py:42 | xg = grp[xg_column].fillna(0.0).to_numpy(dtype=float)
scripts/validate_territorial_defense.py:383 | placebo = df[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
silly_kicks/tracking/_receiver.py:133 | d = np.linalg.norm(teammates[["x", "y"]].to_numpy(dtype=np.float64) - pxy, axis=1)
silly_kicks/tracking/_receiver.py:205 | def_pos = opp_outfield[["x", "y"]].to_numpy(dtype=np.float64)
scripts/train_xcross_attempt.py:448 | clf.fit(X.to_numpy(dtype=float), y)
silly_kicks/territorial_defense/_compute.py:211 | own_half_xy = grp[grp["start_x"] < params.own_half_max_x][["start_x", "start_y"]].to_numpy(dtype=float)
```
