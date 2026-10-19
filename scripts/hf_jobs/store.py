"""Read-only `Store` over the HF bucket API (ADR-128): the controller's view of a campaign store.

Jobs write through the bucket MOUNT (a `LocalStore` at /scratch/<run_key>); the stateless controller only
lists and reads small files (plan state, summaries, ledger records) through the API, so it can stop and
restart at any time."""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from typing import Any


@dataclasses.dataclass
class BucketStore:
    bucket: str  # "<namespace>/<bucket>"
    prefix: str  # the run key
    api: Any = None

    @classmethod
    def from_uri(cls, uri: str) -> BucketStore:
        rest = uri.removeprefix("hf://buckets/")
        ns, bucket, *prefix = rest.split("/")
        return cls(f"{ns}/{bucket}", "/".join(prefix))

    def _api(self):
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi()
        return self.api

    def _full(self, rel: str) -> str:
        return "/".join(p for p in (self.prefix, rel.strip("/")) if p)

    def names(self, rel_dir: str) -> set[str]:
        base = self._full(rel_dir) + "/"
        out = set()
        for entry in self._api().list_bucket_tree(self.bucket, prefix=base, recursive=False):
            if getattr(entry, "type", None) == "file":
                tail = entry.path[len(base) :]
                if "/" not in tail:
                    out.add(tail)
        return out

    def read_bytes(self, rel: str) -> bytes | None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "f"
            self._api().download_bucket_files(self.bucket, [(self._full(rel), str(dest))], raise_on_missing_files=False)
            return dest.read_bytes() if dest.is_file() else None

    def write_bytes(self, rel: str, data: bytes) -> None:
        raise PermissionError("BucketStore is read-only: jobs write through the bucket mount")

    def upload_bytes(self, rel: str, data: bytes) -> None:
        """The controller's ONLY write: its own records under `_campaign/` and `_partitions/` (preflight
        report, planned partitions). Never a unit result -- those are written in-job through the mount."""
        if not (rel.startswith("_campaign/") or rel.startswith("_partitions/")):
            raise PermissionError(f"controller uploads are limited to _campaign/ and _partitions/, got {rel!r}")
        self._api().batch_bucket_files(self.bucket, add=[(data, self._full(rel))])

    def delete(self, rel: str) -> None:
        raise PermissionError("BucketStore is read-only: jobs write through the bucket mount")

    def local_path(self, rel: str) -> pathlib.Path | None:
        return None
