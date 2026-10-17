"""ADR-123: xshot/xcross gain a --shard-index extraction fan-out flag (the study seam already exists).
Offline checks -- the flag parses, and the xcross held-out-probe coupling fails fast. The real byte-
identical fanned==serial extraction runs on CI / the DGX e2e (extraction needs pining/statsbombpy data).
The disjoint-cover partition itself is tested in tests/scripts/test_driver_shard_select.py.
"""

from __future__ import annotations

import pytest

import scripts.train_xshot_occurrence as xshot  # noqa: F401 -- import-smoke the edited module
from scripts.train_xcross_attempt import main as xcross_main


def test_xcross_shard_index_requires_probe_data_dir(capsys):
    # Fan-out of the training extraction loses the in-line probe cohort on a resumed prep (ADR-119), so the
    # held-out TF-19 probe MUST be the separate staged pass. The trainer refuses --shard-index without
    # --probe-data-dir (ADR-123 Amendment B renamed the pining --probe-match-ids-json to the agnostic
    # --probe-data-dir), BEFORE any corpus work (so this runs offline). --allow-dirty so the dirty-tree
    # refusal doesn't pre-empt the guard under test.
    with pytest.raises(SystemExit):
        xcross_main(["--output-dir", "x", "--data-dir", "d", "--shard-index", "0/2", "--allow-dirty"])
    assert "--shard-index requires --probe-data-dir" in capsys.readouterr().err
