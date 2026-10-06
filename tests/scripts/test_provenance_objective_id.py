"""TF-58 Task 21 (D21): the ``objective_id`` provenance helper.

``objective_id(objective, inputs, *, prov=None)`` returns ``"<module>.<qualname>@<commit>:<digest>"``,
where the digest MUST come from :func:`scripts._input_contract.declare_inputs` (``ValueError`` otherwise),
and a ``tree_state`` other than ``"clean"`` appends ``"+dirty-<uuid4 hex>"`` so a dirty or unknown tree
never resumes an Optuna store across runs (C4, the fail-closed reading of §12).
"""

from __future__ import annotations

import pytest

from scripts._input_contract import declare_inputs
from scripts._provenance import objective_id, store_path_for


class _DummyObjective:
    """A stand-in objective; only its ``__module__``/``__qualname__`` matter to the id."""


_CLEAN = {"commit": "abc123def456", "tree_state": "clean"}
_DIRTY = {"commit": "abc123def456", "tree_state": "dirty"}
_UNKNOWN = {"commit": "unknown", "tree_state": "unknown"}
_PREFIX = f"{_DummyObjective.__module__}.{_DummyObjective.__qualname__}"


def test_format_on_clean_tree():
    inputs = declare_inputs(driver="dummy", match_ids=["m1", "m2"])
    oid = objective_id(_DummyObjective, inputs, prov=_CLEAN)
    assert oid == f"{_PREFIX}@{_CLEAN['commit']}:{inputs['digest']}"
    assert "+dirty-" not in oid


def test_dirty_and_unknown_trees_get_a_unique_nonce():
    inputs = declare_inputs(driver="dummy", match_ids=["m1"])
    a = objective_id(_DummyObjective, inputs, prov=_DIRTY)
    b = objective_id(_DummyObjective, inputs, prov=_DIRTY)
    assert a != b, "a dirty tree must never produce a resumable (stable) id"
    assert "+dirty-" in a and "+dirty-" in b
    # unknown provenance is untrustworthy exactly like dirty (git_provenance sets tree_state='unknown')
    assert "+dirty-" in objective_id(_DummyObjective, inputs, prov=_UNKNOWN)


def test_requires_declared_digest():
    with pytest.raises(ValueError, match="digest"):
        objective_id(_DummyObjective, {"driver": "dummy"}, prov=_CLEAN)  # not from declare_inputs


def test_class_and_instance_give_same_prefix():
    inputs = declare_inputs(driver="dummy")
    from_class = objective_id(_DummyObjective, inputs, prov=_CLEAN)
    from_instance = objective_id(_DummyObjective(), inputs, prov=_CLEAN)
    assert from_class == from_instance
    assert from_class.startswith(f"{_PREFIX}@")


# --------------------------------------------------------------------------- store_path_for (C4 made true)
def test_clean_id_keeps_the_store_path_so_a_rerun_resumes(tmp_path):
    inputs = declare_inputs(driver="dummy")
    oid = objective_id(_DummyObjective, inputs, prov=_CLEAN)
    assert store_path_for(tmp_path / "study.db", oid) == str(tmp_path / "study.db")


def test_dirty_id_gets_a_fresh_store_per_call(tmp_path):
    # Each dirty call carries its own nonce, so each opens its OWN sibling store: it never resumes, and never reopens a
    # store an earlier call wrote (which ruthless refuses: "written for a different objective").
    inputs = declare_inputs(driver="dummy")
    a = store_path_for(tmp_path / "study.db", objective_id(_DummyObjective, inputs, prov=_DIRTY))
    b = store_path_for(tmp_path / "study.db", objective_id(_DummyObjective, inputs, prov=_UNKNOWN))
    assert len({a, b, str(tmp_path / "study.db")}) == 3
    assert a.startswith(str(tmp_path / "study.dirty-")) and a.endswith(".db")


def test_in_memory_store_is_left_alone():
    oid = objective_id(_DummyObjective, declare_inputs(driver="dummy"), prov=_DIRTY)
    assert store_path_for(":memory:", oid) == ":memory:"
