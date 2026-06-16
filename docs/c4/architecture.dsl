workspace "silly-kicks" "Football action classification (SPADL) and valuation (VAEP) library" {

    model {
        // --- Actors ---
        analyst = person "Soccer Analytics Practitioner" "Data scientist or analyst who classifies and values football actions"
        pipeline = person "Downstream Pipeline" "Production data pipeline that calls silly-kicks inside Spark UDFs"
        maintainer = person "Library Maintainer" "Runs the TF-24 calibration sweep to recommend tuned tracking defaults"

        // --- External Systems ---
        kloppy = softwareSystem "kloppy" "PySport event/tracking data normalization library" "External"
        mlLibs = softwareSystem "ML Libraries" "XGBoost, CatBoost, LightGBM gradient boosting frameworks" "External"
        hfHub = softwareSystem "HuggingFace Hub" "Model artifact hosting for pre-trained Ghost-GK weights" "External"
        accessibleSpace = softwareSystem "accessible-space" "DAS (Dangerous Accessible Space) surface computation" "External"
        ruthless = softwareSystem "ruthless-efficiency" "Optuna/evolutionary optimization substrate (OptunaStrategy + CachedObjective)" "External"
        pining = softwareSystem "pining-for-the-data" "Gated mock provider REST API (SkillCorner/IDSSE public, Gradient Sports owner-tier) over S3" "External"
        databricks = softwareSystem "Databricks Lakehouse" "bronze.* SPADL/tracking tables + spadl_actions xT corpus" "External"

        // --- The System ---
        sillyKicks = softwareSystem "silly-kicks" "Classifies football actions into SPADL representation and values them via VAEP" {

            spadl = container "silly_kicks.spadl" "SPADL event conversion (23 action types) from 7 providers + kloppy gateway. Post-conversion enrichments: possessions, game state, GK analytics, naming. Canonical LTR orientation with auto-detected input conventions. Owns the shared canonical kloppy coordinate system (_kloppy_coordinates), pinned by BOTH the event and tracking kloppy gateways so their vertical orientation cannot drift (ADR-031)." "Python" "Library"
            vaep = container "silly_kicks.vaep" "VAEP action valuation: features, labels (action/possession/time windowing), model training. HybridVAEP removes result leakage. Goalscore-free xfn variants + opt-in xt__<method> Expected-Threat feature factory (wraps a caller-supplied fitted ExpectedThreat; atomic mirror)." "Python" "Library"
            tracking = container "silly_kicks.tracking" "Per-frame tracking data: schema, provider adapters, event-frame linkage, preprocessing, pitch control, OBSO, DAS, space creation, structural-pass primitives (LBS/SGM/SDI), ghost-GK positioning (selectable scipy/vectorized/cpu-numba/fft/fft-cic KDE backends; serves the exact pickle-free boosted-mean estimator), shot-occurrence (xS), cross-attempt (xCross), Eyestone xT-GK (GK-distribution value; RAV completion via a fitted GkCompletionModel with a per-type base-rate serve gate + scoped goal-kick coordinate derivation), post-shot goalmouth crossing geometry (TF-48: pure trajectory-fit engine with sample-and-hold collapse, contact anchoring/existence, z-aware flight and an extrapolation-leverage cap; lakehouse-PSxG feed, holdout-validated), and 28 action-coupled aggregators. Enforced consumer contracts: period-relative time base (validate_time_base, ADR-017), dtype-safe id comparison/merge at the feature seams (_id_compat + validate_id_dtypes, ADR-019), position-based frame_id resolution in frame-aware xfns (resolve_frame_ids_by_position, ADR-020), per-action SPADL-LTR re-projection of emitted geometry at the feature seams (_action_orientation, ADR-028), single-sourced frame-LTR orientation for consumer-built frames (orient_frames_to_ltr, ADR-029), and the kloppy tracking gateway pinning the canonical SPADL coordinate system shared with the event gateway (was y-inverted; ADR-031)." "Python" "Library"
            atomic = container "silly_kicks.atomic" "Atomic SPADL/VAEP: continuous 33-type action representation with full enrichment parity. Mirrors tracking.features for atomic-shaped columns." "Python" "Library"
            xthreat = container "silly_kicks.xthreat" "Expected Threat (xT): pluggable transition family (Singh counts / KDE-smoothed) + value iteration on a variable-resolution grid; held-out transition-NLL evaluator. ExpectedThreat facade (byte-identical Singh default)." "Python" "Library"
            calibration = container "silly_kicks.calibration + scripts/" "Optuna calibration harness: pure objectives/CV/gates + frozen exogenous xT artifact. scripts/ CLI + pining + Databricks loaders. Recommends infer_ball_carrier / LinkParams.k3 / off-ball-run defaults (carrier accuracy + held-out VAEP Brier) and per-corpus xT KDE bandwidth/resolution (held-out transition-NLL). Does not change library constants." "Python (optional [calibration] extra)" "Library"
            providers = container "silly_kicks.providers" "Per-provider raw-data parse ports (bytes -> provider-canonical bronze -> converter input). The Sportec/DFL parse+shape port (parse_dfl_* / shape_*_to_native) single-sources the IDSSE/Sportec parser as a verbatim lift of the lakehouse DFL parser, pinned by a golden parity test; emits RAW bronze (data-quality stays consumer-side). Behind the [parse-dfl] extra (ADR-031 T3)." "Python (optional [parse-dfl] extra)" "Library"
        }

        // --- Relationships: Context level ---
        analyst -> sillyKicks "Converts event data and values actions using" "Python API"
        pipeline -> sillyKicks "Calls inside Spark applyInPandas UDFs via" "Python import"
        maintainer -> sillyKicks "Calibrates tracking defaults via the calibration CLI" "scripts/calibrate_tracking_defaults.py"
        sillyKicks -> kloppy "Accepts EventDataset / TrackingDataset from" "kloppy bridge"
        sillyKicks -> mlLibs "Trains and predicts with" "Python API"
        sillyKicks -> hfHub "Downloads pre-trained Ghost-GK model from" "huggingface_hub"
        sillyKicks -> accessibleSpace "Computes DAS surfaces via" "accessible-space API"
        sillyKicks -> ruthless "Runs Optuna calibration studies via" "OptunaStrategy"
        sillyKicks -> pining "Loads calibration match data from" "Bearer -> presigned S3"
        sillyKicks -> databricks "Loads bronze tables + xT corpus from" "databricks-sql-connector"

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
        tracking -> hfHub "Lazy-downloads Ghost-GK model weights via" "huggingface_hub"
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
