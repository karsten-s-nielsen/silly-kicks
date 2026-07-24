workspace "silly-kicks" "Football action classification (SPADL) and valuation (VAEP) library" {

    model {
        // --- Actors ---
        analyst = person "Soccer Analytics Practitioner" "Data scientist or analyst who classifies and values football actions"
        pipeline = person "Downstream Pipeline" "Production data pipeline that calls silly-kicks inside Spark UDFs"
        maintainer = person "Library Maintainer" "Runs the TF-24 calibration sweep to recommend tuned tracking defaults"

        // --- External Systems ---
        kloppy = softwareSystem "kloppy" "PySport event/tracking data normalization library" "External"
        mlLibs = softwareSystem "ML Libraries" "XGBoost, CatBoost, LightGBM gradient boosting frameworks" "External"
        hfHub = softwareSystem "HuggingFace Hub" "Model artifact hosting for pre-trained xS / xCross / Ghost-GK weights (Hub-only variants)" "External"
        accessibleSpace = softwareSystem "accessible-space" "DAS (Dangerous Accessible Space) surface computation" "External"
        ruthless = softwareSystem "ruthless-efficiency" "Optuna/evolutionary optimization substrate (OptunaStrategy + CachedObjective)" "External"
        pining = softwareSystem "pining-for-the-data" "Gated mock provider REST API (SkillCorner/IDSSE public, Gradient Sports owner-tier) over S3" "External"
        databricks = softwareSystem "Databricks Lakehouse" "bronze.* SPADL/tracking + spadl_actions xT corpus + dev_gold action/shot marts (fct_action_values/context, fct_shot_xg, dim_matches) for the xT-GK v2 gate + rho retention" "External"

        // --- The System ---
        sillyKicks = softwareSystem "silly-kicks" "Classifies football actions into SPADL representation and values them via VAEP" {

            spadl = container "silly_kicks.spadl" "SPADL event conversion (23 action types) from 7 providers + a kloppy gateway; enrichments (possessions, game state, GK analytics, naming) + block-detection columns (ADR-046); canonical LTR coords." "Python" "Library"
            vaep = container "silly_kicks.vaep" "VAEP action valuation: features, action/possession/time-windowed labels, and gradient-boosted models. HybridVAEP removes result leakage; optional Expected-Threat feature factory." "Python" "Library"
            tracking = container "silly_kicks.tracking" "Per-frame tracking: schema, adapters, linkage, spatial/GKDV models (pitch control, OBSO, DAS, ghost-GK, xS/xCross/xT-GK, run valuation, defensive credit), 31 action-coupled aggregators. ADR-042." "Python" "Library"
            atomic = container "silly_kicks.atomic" "Atomic SPADL/VAEP: continuous 33-type action representation with full enrichment parity. Mirrors tracking.features for atomic-shaped columns." "Python" "Library"
            xthreat = container "silly_kicks.xthreat" "Expected Threat (xT): pluggable transition family (Singh counts / KDE-smoothed) + value iteration on a variable-resolution grid; held-out transition-NLL evaluator; physical_grid resampling. ADR-041." "Python" "Library"
            xtgk = container "silly_kicks.xtgk" "xT-GK v2: possession value V(z,p) (Markov surface + deep-zone gate), metric compute_xt_gk_v2 over 3 injected ports, resolved-GK-geometry edge (apply_resolved_gk_geometry), bundled rho weights." "Python" "Library"
            gkdv = container "silly_kicks.gkdv" "GKDV v1 (TF-19): ghost-substitution engine (build_ghost_frames) + two gate-independent physics arms (delta-DAS, delta-threat-suppression) in attacker-value units (negative = deterrent). ADR-043." "Python" "Library"
            calibration = container "silly_kicks.calibration + scripts/" "Optuna calibration harness (pure objectives/CV/gates + frozen exogenous xT artifact) + scripts/ CLI + pining/Databricks loaders. Recommends tuned tracking/xT defaults; never changes library constants." "Python (optional [calibration] extra)" "Library"
            providers = container "silly_kicks.providers" "Per-provider raw-data parse ports (bytes -> provider bronze -> converter input). The Sportec/DFL parse+shape port single-sources the lakehouse DFL parser (golden-pinned). Behind the [parse-dfl] extra." "Python (optional [parse-dfl] extra)" "Library"
        }

        // --- Relationships: Context level ---
        analyst -> sillyKicks "Converts event data and values actions using" "Python API"
        pipeline -> sillyKicks "Calls inside Spark applyInPandas UDFs via" "Python import"
        maintainer -> sillyKicks "Calibrates tracking defaults via the calibration CLI" "scripts/calibrate_tracking_defaults.py"
        sillyKicks -> kloppy "Accepts EventDataset / TrackingDataset from" "kloppy bridge"
        sillyKicks -> mlLibs "Trains and predicts with" "Python API"
        sillyKicks -> hfHub "Downloads pre-trained xS / xCross / Ghost-GK models from" "huggingface_hub"
        sillyKicks -> accessibleSpace "Computes DAS surfaces via" "accessible-space API"
        sillyKicks -> ruthless "Runs Optuna calibration studies via" "OptunaStrategy"
        sillyKicks -> pining "Loads calibration match data from" "Bearer -> presigned S3"
        sillyKicks -> databricks "Loads bronze tables + xT corpus + dev_gold action/shot marts (xT-GK v2 gate, rho retention, resolved GK origins xt_gk_origin_x/y) from" "databricks-sql-connector"

        // --- Relationships: Container level ---
        analyst -> spadl "Converts raw events to SPADL actions and enriches via" "convert_to_actions() + add_*() helper family"
        analyst -> tracking "Converts raw tracking data to long-form frames + enriches via" "convert_to_frames() + add_action_context()"
        analyst -> vaep "Values actions via" "VAEP.fit() / VAEP.rate() / HybridVAEP (with optional frames=)"
        analyst -> xthreat "Computes pitch value surface via" "ExpectedThreat.fit()"
        maintainer -> calibration "Runs the two-stage Optuna sweep (carrier accuracy, then held-out Brier) via" "calibrate_tracking_defaults.py"

        pipeline -> spadl "Passes per-game DataFrames to" "lazy import inside UDF"
        pipeline -> tracking "Passes per-match tracking frames to" "lazy import inside UDF"
        pipeline -> vaep "Scores actions with pre-trained models via" "VAEP.rate()"

        spadl -> kloppy "Accepts kloppy EventDataset (derives game_id from dataset metadata) in kloppy converter" "kloppy bridge"
        tracking -> kloppy "Accepts kloppy TrackingDataset in kloppy gateway" "kloppy bridge"
        tracking -> hfHub "Lazy-downloads xS / xCross / Ghost-GK model weights via" "huggingface_hub"
        tracking -> accessibleSpace "Computes DAS via" "get_individual_das()"

        vaep -> spadl "Reads SPADL config, schema constants, and action names from" "Python import"
        vaep -> mlLibs "Delegates model training to" "fit() dispatch"
        tracking -> vaep "Imports frame_aware decorator + Frames type alias from" "vaep.feature_framework"
        vaep -> tracking "Lazy-imports play_left_to_right when frames= is supplied" "lazy import"
        spadl -> tracking "Lazy-imports tracking GK features when frames= is supplied to add_pre_shot_gk_context" "lazy import"
        atomic -> spadl "Extends SPADL with atomic action types via" "Python import"
        atomic -> vaep "Inherits VAEP pipeline via AtomicVAEP subclass" "Python import"
        atomic -> tracking "Reuses _kernels + lift_to_states from tracking namespace" "Python import"
        xthreat -> spadl "Reads SPADL config and schema from" "Python import"
        tracking -> xthreat "Weights pitch control by real threat (OBSO EPV, GK + player influence, cover shadows, run valuation) via" "physical_grid / values_at_points"

        // --- Relationships: xT-GK v2 possession value (ADR-036) ---
        analyst -> xtgk "Fits the possession-value surface V(z,p) with an injected per-shot xg_column via" "MarkovPossessionValue.fit()"
        xtgk -> xthreat "Reuses value_iteration + low-level transition/grid seams (no xthreat edits) via" "Python import"
        xtgk -> spadl "Reads SPADL config + action-type ids from" "Python import"

        // --- Relationships: GKDV v1 (TF-19 PR-3) ---
        analyst -> gkdv "Values keeper positioning against a league-average ghost via" "build_ghost_frames + delta_das / delta_threat_suppression"
        gkdv -> tracking "Consumes PUBLIC tracking seams + ONE confined private DAS port (_das_port.py); never the reverse (allowlist-gated)" "Python import"
        gkdv -> xthreat "Weights the pitch-control field by per-cell threat with an injected fitted model" "ExpectedThreat"
        gkdv -> accessibleSpace "Sums per-player DAS under ONE direction pinned on the factual frames via" "_das_port / get_individual_das()"

        // --- Relationships: Calibration harness (TF-24) ---
        calibration -> ruthless "Drives Optuna TPE studies (CachedObjective fast path) via" "OptunaStrategy"
        calibration -> tracking "Enriches frames + infers ball carrier via" "add_* aggregators + infer_ball_carrier"
        calibration -> vaep "Computes held-out scores/concedes labels via" "vaep.labels"
        calibration -> xthreat "Fits the frozen exogenous xT grid via" "ExpectedThreat.fit() on a disjoint corpus"
        calibration -> spadl "Converts provider events to SPADL via" "convert_to_actions()"
        calibration -> mlLibs "Trains disposable XGBoost classifiers (deterministic) via" "XGBoost"
        calibration -> kloppy "Parses SkillCorner/Sportec provider data via" "kloppy.skillcorner / kloppy.sportec"
        calibration -> pining "Loads SkillCorner/IDSSE/Gradient-Sports matches from" "Bearer -> 302 -> presigned S3"
        calibration -> databricks "Loads bronze tables + the spadl_actions xT corpus from" "databricks-sql-connector"

        // --- Relationships: DFL parse port (ADR-031 T3) ---
        calibration -> providers "Parses IDSSE/Sportec DFL XML + shapes to native converter input via" "parse_dfl_* / shape_*_to_native"
        providers -> spadl "Emits silly_kicks.spadl.sportec convert_to_actions input via" "shape_events_to_native (DataFrame contract)"
        providers -> tracking "Emits silly_kicks.tracking.sportec convert_to_frames input via" "shape_tracking_to_native (DataFrame contract)"
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
