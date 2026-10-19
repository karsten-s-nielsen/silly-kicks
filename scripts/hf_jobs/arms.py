"""Neutral candidate-arm membership, parsed from the --arms job parameter (ADR-123 Amendment B).

The in-job trainers build their candidate row masks from THIS alone -- a mapping of dataset match-keys to
arm memberships, supplied by the launcher. There is NO public/private rule here and NO provider-name
logic: an arm is just a set of keys. Policy (which keys are ``public``) lives in the launcher, off-HF.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ArmsUnavailableError(RuntimeError):
    """The --arms parameter is absent or defines no non-empty arm (I4: fail-loud, never all-private)."""


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
