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

            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 4 provider converters with preserve_native passthrough, ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context)" "Python" "Library"
            vaep = container "silly_kicks.vaep" "VAEP framework: feature extraction, label generation (binary + xG), model training, action valuation. Includes HybridVAEP (result-leakage-free)" "Python" "Library"
            atomic = container "silly_kicks.atomic" "Atomic SPADL/VAEP: continuous action representation with 33 extended action types, deferred single-sort conversion, full parity for the post-conversion enrichment helper family (preserve_native, add_possessions, GK analytics suite, validate_atomic_spadl)" "Python" "Library"
            xthreat = container "silly_kicks.xthreat" "Expected Threat model: pitch grid value surface via dynamic programming" "Python" "Library"
        }

        // --- Relationships: Context level ---
        analyst -> sillyKicks "Converts event data and values actions using" "Python API"
        pipeline -> sillyKicks "Calls inside Spark applyInPandas UDFs via" "Python import"
        sillyKicks -> kloppy "Accepts EventDataset from" "kloppy bridge"
        sillyKicks -> mlLibs "Trains and predicts with" "Python API"

        // --- Relationships: Container level ---
        analyst -> spadl "Converts raw events to SPADL actions and enriches via" "convert_to_actions() + add_*() helper family"
        analyst -> vaep "Values actions via" "VAEP.fit() / VAEP.rate() / HybridVAEP"
        analyst -> xthreat "Computes pitch value surface via" "ExpectedThreat.fit()"

        pipeline -> spadl "Passes per-game DataFrames to" "lazy import inside UDF"
        pipeline -> vaep "Scores actions with pre-trained models via" "VAEP.rate()"

        spadl -> kloppy "Accepts kloppy EventDataset in kloppy converter" "kloppy bridge"

        vaep -> spadl "Reads SPADL config, schema constants, and action names from" "Python import"
        vaep -> mlLibs "Delegates model training to" "fit() dispatch via _LEARNER_REGISTRY"
        atomic -> spadl "Extends SPADL with atomic action types via" "Python import"
        atomic -> vaep "Inherits VAEP pipeline via AtomicVAEP subclass" "Python import"
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
