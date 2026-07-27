"""Stamp feature contracts into the three bundled artifacts (ADR-050) -- METADATA ONLY.

Deliberately does NOT call any model's ``save()``. Ghost's ``save()`` unconditionally rewrites
``rfcde_weights.npz`` via ``np.savez_compressed``, which writes a fresh ZIP whose members carry
mtimes -- so the npz bytes differ even when every array is bit-identical. The xS/xCross ``save()``
paths likewise re-serialize ``model.json`` through xgboost. Either would break the "byte delta is
metadata-only" property this migration promises, and would trip its own verification step on a
correct run.

Re-runnable: it rebuilds the contract from the current library each time. Run it again after any
change to a declared constant or to a feature extractor -- that is exactly when the recorded
fingerprint needs to be re-derived rather than hand-edited.

    python scripts/stamp_feature_contracts.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import silly_kicks.tracking._ghost_gk as gg
import silly_kicks.tracking._xcross_attempt as xc
import silly_kicks.tracking._xshot_occurrence as xs
from silly_kicks.tracking import _geometry as _geo

ROOT = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking"

#: (directory, SHA256SUMS file list, contract builder, extra metadata to inject)
#:
#: The file lists are READ FROM each model's own save(), not inferred. They are two entries each
#: and deliberately EXCLUDE metrics.json, which sits in the xS/xCross directories but is not
#: hashed. An inferred list that disagrees with what load() verifies produces an IntegrityError on
#: the very next load.
TARGETS = [
    (
        ROOT / "_ghost_gk_weights" / "default",
        ["rfcde_weights.npz", "metadata.json"],
        gg._feature_contract_block,
        # ghost alone lacked pitch dims; xS and xCross have recorded them since their first release
        {"pitch_length": _geo.PITCH_LENGTH, "pitch_width": _geo.PITCH_WIDTH},
    ),
    (
        ROOT / "_xshot_weights" / "default",
        ["model.json", "metadata.json"],
        xs._feature_contract_block,
        {},
    ),
    (
        ROOT / "_xcross_weights" / "default",
        ["model.json", "metadata.json"],
        xc._feature_contract_block,
        {},
    ),
]


def main() -> None:
    for path, sum_files, build, extra in TARGETS:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        meta.update(extra)
        meta["feature_contract"] = build()

        # Byte-for-byte the writer save() uses: LF endings, indent=2.
        with open(path / "metadata.json", "w", newline="\n", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Recompute sums exactly as save() does, including the CRLF->LF normalisation that keeps
        # the hash platform-independent.
        with open(path / "SHA256SUMS", "w", newline="\n", encoding="utf-8") as f:
            for fname in sum_files:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

        print(f"stamped {path.name}: {sorted(meta['feature_contract']['constants'])}")


if __name__ == "__main__":
    main()
