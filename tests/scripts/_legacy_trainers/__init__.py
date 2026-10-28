"""Frozen copies of the three adapter trainers' legacy CLIs, kept VERBATIM as test oracles (HFU R3-10 / D12).

Each module is byte-for-byte the ``scripts/train_*.py`` file at ``2474b73`` (git blob ids
``77329b0`` match_outcome_dependence, ``0cc3642`` pass_completion, ``ad232d2`` xsuccess), before the trainer
became a stage-contract pipeline. ``tests/scripts/test_adapter_migration_identity.py`` runs an oracle and its
pipeline in the SAME environment and compares the artifacts byte for byte.

Why an oracle and not a committed digest: the legacy CLIs give different numbers on different platforms and
library versions (x86_64 with the ``uv.lock`` versions against aarch64 with newer ones: the pass-completion
coefficients and the xsuccess booster differ), so only a same-environment comparison proves the migration.
Never edit these files; they are the reference the pipelines are held to.
"""
