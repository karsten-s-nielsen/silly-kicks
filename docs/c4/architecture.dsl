workspace "silly-kicks" "Football action classification (SPADL) and valuation (VAEP) library" {

    model {
        // --- Actors ---
        analyst = person "Soccer Analytics Practitioner" "Data scientist or analyst who classifies and values football actions"
        pipeline = person "Downstream Pipeline" "Production data pipeline that calls silly-kicks inside Spark UDFs"

        // --- External Systems ---
        kloppy = softwareSystem "kloppy" "PySport event data normalization library" "External"
        mlLibs = softwareSystem "ML Libraries" "XGBoost, CatBoost, LightGBM gradient boosting frameworks" "External"
        hfHub = softwareSystem "HuggingFace Hub" "Model artifact hosting for pre-trained Ghost-GK weights" "External"

        // --- The System ---
        sillyKicks = softwareSystem "silly-kicks" "Classifies football actions into SPADL representation and values them via VAEP" {

            spadl = container "silly_kicks.spadl" "SPADL event conversion (23 action types) from 7 providers + kloppy gateway. Post-conversion enrichments: possessions, game state, GK analytics, naming. Canonical LTR orientation with auto-detected input conventions." "Python" "Library"
            vaep = container "silly_kicks.vaep" "VAEP action valuation: features, labels (action/possession/time windowing), model training. HybridVAEP removes result leakage. Goalscore-free xfn variants." "Python" "Library"
            tracking = container "silly_kicks.tracking" "Per-frame tracking data: schema, provider adapters, event-frame linkage, preprocessing, pitch control, OBSO, DAS, space creation, and 21 action-coupled aggregators." "Python" "Library"
            atomic = container "silly_kicks.atomic" "Atomic SPADL/VAEP: continuous 33-type action representation with full enrichment parity. Mirrors tracking.features for atomic-shaped columns." "Python" "Library"
            xthreat = container "silly_kicks.xthreat" "Expected Threat model: pitch grid value surface via dynamic programming" "Python" "Library"
        }

        // --- Relationships: Context level ---
        analyst -> sillyKicks "Converts event data and values actions using" "Python API"
        pipeline -> sillyKicks "Calls inside Spark applyInPandas UDFs via" "Python import"
        sillyKicks -> kloppy "Accepts EventDataset from" "kloppy bridge"
        sillyKicks -> mlLibs "Trains and predicts with" "Python API"
        sillyKicks -> hfHub "Downloads pre-trained Ghost-GK model from" "huggingface_hub"

        // --- Relationships: Container level ---
        analyst -> spadl "Converts raw events to SPADL actions and enriches via" "convert_to_actions() + add_*() helper family"
        analyst -> tracking "Converts raw tracking data to long-form frames + enriches via" "convert_to_frames() + add_action_context()"
        analyst -> vaep "Values actions via" "VAEP.fit() / VAEP.rate() / HybridVAEP (with optional frames=)"
        analyst -> xthreat "Computes pitch value surface via" "ExpectedThreat.fit()"

        pipeline -> spadl "Passes per-game DataFrames to" "lazy import inside UDF"
        pipeline -> tracking "Passes per-match tracking frames to" "lazy import inside UDF"
        pipeline -> vaep "Scores actions with pre-trained models via" "VAEP.rate()"

        spadl -> kloppy "Accepts kloppy EventDataset in kloppy converter" "kloppy bridge"
        tracking -> kloppy "Accepts kloppy TrackingDataset in kloppy gateway" "kloppy bridge"
        tracking -> hfHub "Lazy-downloads Ghost-GK model weights via" "huggingface_hub"

        vaep -> spadl "Reads SPADL config, schema constants, and action names from" "Python import"
        vaep -> mlLibs "Delegates model training to" "fit() dispatch"
        tracking -> vaep "Imports frame_aware decorator + Frames type alias from" "vaep.feature_framework"
        vaep -> tracking "Lazy-imports play_left_to_right when frames= is supplied" "lazy import"
        spadl -> tracking "Lazy-imports tracking GK features when frames= is supplied to add_pre_shot_gk_context" "lazy import"
        atomic -> spadl "Extends SPADL with atomic action types via" "Python import"
        atomic -> vaep "Inherits VAEP pipeline via AtomicVAEP subclass" "Python import"
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
