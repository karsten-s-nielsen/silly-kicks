workspace "silly-kicks" "Football action classification (SPADL) and valuation (VAEP) library" {

    model {
        // --- Actors ---
        analyst = person "Soccer Analytics Practitioner" "Data scientist or analyst who classifies and values football actions"
        pipeline = person "Downstream Pipeline" "Production data pipeline that calls silly-kicks inside Spark UDFs"

        // --- External Systems ---
        kloppy = softwareSystem "kloppy" "PySport event data normalization library" "External"
        mlLibs = softwareSystem "ML Libraries" "XGBoost, CatBoost, LightGBM gradient boosting frameworks" "External"

        // --- The System ---
        sillyKicks = softwareSystem "silly-kicks" "Classifies football actions into SPADL representation and values them via VAEP" {

            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 6 dedicated DataFrame converters with preserve_native + goalkeeper_ids passthrough (StatsBomb, Opta, Wyscout, Sportec, Metrica, PFF FC) plus a kloppy gateway covering StatsBomb / Sportec / Metrica, output coords clamped to [0, 105] x [0, 68] AND unified to canonical 'all-actions-LTR' SPADL orientation across all paths via single-call to_spadl_ltr() dispatcher (ADR-006, silly-kicks 3.0.0; 3.0.1 erratum: native sportec + metrica converters declare PER_PERIOD_ABSOLUTE not ABSOLUTE_FRAME_HOME_RIGHT, expose home_team_start_left + home_attacks_right_per_period kwargs); orientation.py defines InputConvention enum (POSSESSION_PERSPECTIVE / ABSOLUTE_FRAME_HOME_RIGHT / PER_PERIOD_ABSOLUTE) + detect_input_convention (TF-22-hardened against sparse per-team-period-asymmetric data) + validate_input_convention validator wired into every converter to surface upstream loader regressions; ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context, use_tackle_winner_as_actor); boundary_metrics + coverage_metrics utilities for validating add_possessions output and per-action-type coverage; ADR-001 caller-conventions contract — Sportec output uses SPORTEC_SPADL_COLUMNS (KLOPPY_SPADL_COLUMNS + 4 object tackle qualifier passthrough columns); PFF output uses PFF_SPADL_COLUMNS (SPADL_COLUMNS + 4 nullable Int64 tackle passthrough columns)" "Python" "Library"
            vaep = container "silly_kicks.vaep" "VAEP framework: feature extraction, label generation (binary + xG), model training, action valuation. Includes HybridVAEP (result-leakage-free). compute_features / rate accept optional frames= kwarg dispatching frame-aware xfns (ADR-005); compute_features also accepts frames_convention='absolute_frame'|'ltr' kwarg to control internal tracking-frame normalisation (ADR-006). Internal play_left_to_right call removed in 3.0.0 -- converter output is canonical SPADL LTR (ADR-006)" "Python" "Library"
            tracking = container "silly_kicks.tracking" "Tracking namespace (ADR-004): 19-column long-form per-frame schema, native Sportec + PFF adapters, kloppy gateway for Metrica + SkillCorner, link_actions_to_frames + slice_around_event linkage primitives. PR-S20 (ADR-005) shipped the first tracking-aware features: nearest_defender_distance, actor_speed, receiver_zone_density, defenders_in_triangle_to_goal + add_action_context aggregator + tracking_default_xfns. PR-S21 added pre-shot GK position: pre_shot_gk_x / _y / _distance_to_goal / _distance_to_shot + add_pre_shot_gk_position aggregator + pre_shot_gk_default_xfns. PR-S22 (ADR-006, 3.0.0) added output_convention='absolute_frame'|'ltr' kwarg to all 3 adapters; default behaviour preserved (DeprecationWarning recommends explicit). PR-S24 (3.1.0) ships TF-6 sync_score (per-action tracking↔events sync-quality, 3 aggregations) + TF-12 pre_shot_gk_angle_to_shot_trajectory / _off_goal_line (signed angles) + add_pre_shot_gk_angle aggregator + pre_shot_gk_angle_default_xfns / pre_shot_gk_full_default_xfns. NEW silly_kicks.tracking.preprocess submodule ships TF-8 smooth_frames (Savitzky-Golay / EMA -- additive x_smoothed/y_smoothed/_preprocessed_with cols) + derive_velocities (vx/vy/speed via SG-derivative) + TF-9 interpolate_frames (linear NaN gap-fill up to max_gap_seconds; cubic deferred to TF-9-cubic) + shared PreprocessConfig dataclass with flag-based is_default() + per-provider defaults from a codegen-from-JSON pipeline (probe_preprocess_baseline.py -> preprocess_baseline.json -> regenerate_provider_defaults.py -> _provider_defaults_generated.py). Tracking converters take optional preprocess: PreprocessConfig | None = None kwarg with auto-promotion via _resolve.resolve_preprocess (UserWarning + force_universal fallback for unsupported providers). Schema-agnostic kernels in _kernels shared with atomic SPADL surface" "Python" "Library"
            atomic = container "silly_kicks.atomic" "Atomic SPADL/VAEP: continuous action representation with 33 extended action types, deferred single-sort conversion, full parity for the post-conversion enrichment helper family (preserve_native, add_possessions, GK analytics suite, validate_atomic_spadl). atomic.tracking.features mirrors tracking.features for atomic-shaped column reads (x, y, dx, dy)" "Python" "Library"
            xthreat = container "silly_kicks.xthreat" "Expected Threat model: pitch grid value surface via dynamic programming" "Python" "Library"
        }

        // --- Relationships: Context level ---
        analyst -> sillyKicks "Converts event data and values actions using" "Python API"
        pipeline -> sillyKicks "Calls inside Spark applyInPandas UDFs via" "Python import"
        sillyKicks -> kloppy "Accepts EventDataset from" "kloppy bridge"
        sillyKicks -> mlLibs "Trains and predicts with" "Python API"

        // --- Relationships: Container level ---
        analyst -> spadl "Converts raw events to SPADL actions and enriches via" "convert_to_actions() + add_*() helper family"
        analyst -> tracking "Converts raw tracking data to long-form frames + enriches via" "convert_to_frames() + add_action_context()"
        analyst -> vaep "Values actions via" "VAEP.fit() / VAEP.rate() / HybridVAEP (with optional frames=)"
        analyst -> xthreat "Computes pitch value surface via" "ExpectedThreat.fit()"

        pipeline -> spadl "Passes per-game DataFrames to" "lazy import inside UDF"
        pipeline -> tracking "Passes per-match tracking frames to" "lazy import inside UDF"
        pipeline -> vaep "Scores actions with pre-trained models via" "VAEP.rate()"

        spadl -> kloppy "Accepts kloppy EventDataset in kloppy converter" "kloppy bridge"
        tracking -> kloppy "Accepts kloppy TrackingDataset in kloppy gateway (Metrica, SkillCorner)" "kloppy bridge"

        vaep -> spadl "Reads SPADL config, schema constants, and action names from" "Python import"
        vaep -> mlLibs "Delegates model training to" "fit() dispatch via _LEARNER_REGISTRY"
        tracking -> vaep "Imports frame_aware decorator + Frames type alias from" "vaep.feature_framework"
        vaep -> tracking "Lazy-imports tracking.utils.play_left_to_right when frames= is supplied (no module-import-time cycle, ADR-005)" "lazy import"
        spadl -> tracking "Lazy-imports tracking.features.add_pre_shot_gk_position + add_pre_shot_gk_angle when frames= is supplied to add_pre_shot_gk_context — emits 6 cols when frames=..., 4 cols when frames=None (no module-import-time cycle, ADR-005, PR-S21 + PR-S24)" "lazy import"
        atomic -> spadl "Extends SPADL with atomic action types via" "Python import"
        atomic -> vaep "Inherits VAEP pipeline via AtomicVAEP subclass; auto-inherits frames= extension (ADR-005)" "Python import"
        atomic -> tracking "Reuses _kernels + lift_to_states from tracking namespace" "Python import"
        xthreat -> spadl "Reads SPADL config and schema from" "Python import"
    }

    views {
        systemContext sillyKicks "SystemContext" {
            include *
            autoLayout
        }

        container sillyKicks "Containers" {
            include *
            autoLayout
        }

        styles {
            element "Person" {
                shape Person
                background #08427B
                color #ffffff
            }
            element "Software System" {
                background #1168BD
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438DD5
                color #ffffff
            }
            element "Library" {
                shape RoundedBox
            }
            element "Database" {
                shape Cylinder
            }
            element "Component" {
                background #85BBF0
                color #000000
            }
            relationship "Relationship" {
                color #707070
            }
        }
    }

}
