"""Private causal-validation port (ADR-015). Pure numpy/sklearn matching estimators +
opportunity-row builder for the TF-17 xCross causal harness. NOT imported by
silly_kicks/__init__; promote to a public silly_kicks/causal/ only when a 2nd consumer
(TF-19) lands. No public API is exported here by design."""
