"""Neutral candidate-arm membership, parsed from the --arms job parameter (ADR-123 Amendment B).

The in-job trainers build their candidate row masks from THIS alone -- a mapping of dataset match-keys to
arm memberships, supplied by the launcher. There is NO public/private rule here and NO provider-name
logic: an arm is just a set of keys. Policy (which keys are ``public``) lives in the launcher, off-HF.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ArmsUnavailableError(RuntimeError):
    """The --arms parameter is absent or defines no non-empty arm (I4: fail-loud, never all-private)."""


class ArmKeyMismatchError(ValueError):
    """An arm key names no unit of the dataset the job reads: the launcher and the job disagree on the match key
    (HFU R4-10, 2026-10-30). Raised before any row is cut, so no tier loses a match silently."""


@dataclass(frozen=True)
class ArmSpec:
    arms: dict[str, frozenset[str]]
    ship_order: tuple[str, ...]


def match_key(provider: str, match_id: str) -> str:
    """The dataset's own match key: ``<source_provider>__<match_id>`` (shard-filename / for_each key)."""
    return f"{provider}__{match_id}"


def parse_arms(text: str, *, source: str = "--arms-json") -> ArmSpec:
    """The arm spec from the payload's JSON TEXT (ADR-128: the campaign passes it as the ``--arms-json``
    job parameter). Raises `ArmsUnavailableError` when no arm is non-empty (I4: never all-private)."""
    raw = json.loads(text)
    arms = {name: frozenset(map(str, members)) for name, members in raw.get("arms", {}).items()}
    if not any(arms.values()):
        raise ArmsUnavailableError(f"--arms defines no non-empty arm: {source}")
    ship_order = tuple(raw.get("ship_order") or sorted(arms))
    return ArmSpec(arms=arms, ship_order=ship_order)


def load_arms(path: str | Path) -> ArmSpec:
    p = Path(path)
    if not p.exists():
        raise ArmsUnavailableError(f"--arms file not found: {p}")
    return parse_arms(p.read_text(), source=str(p))


def arm_masks(match_keys: Sequence[str], spec: ArmSpec) -> dict[str, np.ndarray]:
    keys = np.asarray([str(k) for k in match_keys])
    return {name: np.isin(keys, sorted(members)) for name, members in spec.arms.items()}


def arm_keys(spec: ArmSpec) -> frozenset[str]:
    """Every match key of every arm: the rows a tier may read (spec r4 D18 (c))."""
    return frozenset().union(*spec.arms.values())


def require_arm_keys_in(spec: ArmSpec, unit_keys: Iterable[str]) -> None:
    """Fail closed (HFU R4-10): every arm key must be the in-job key of a unit of the dataset the job reads. A unit
    outside every arm is fine; an arm key that names no unit is a key mismatch, and the row filter
    (`restrict_to_arms`) or the masks (`arm_masks`) would drop that match silently."""
    stray = sorted(arm_keys(spec) - frozenset(unit_keys))
    if stray:
        raise ArmKeyMismatchError(
            f"{len(stray)} arm key(s) name no unit of this dataset, e.g. {stray[:3]}: the launcher must key a unit "
            "by the in-job match key (its source_provider and its unit directory name)"
        )


def restrict_to_arms(frame, spec: ArmSpec, *, provider_col: str = "_provider", match_col: str = "_match_id"):
    """``frame`` cut to the rows of the union of ``spec``'s arms, in order, index reset (spec r4 D18 (c)). Tiers that
    read one dataset share its extraction; each reader keeps only its own tier's matches first, so it sees exactly
    what a dataset of those matches alone gives it. A no-op (same rows) when the arms cover the dataset."""
    keys = [match_key(str(p), str(m)) for p, m in zip(frame[provider_col], frame[match_col], strict=True)]
    keep = np.isin(np.asarray(keys, dtype=object), sorted(arm_keys(spec)))
    return frame[keep].reset_index(drop=True)
