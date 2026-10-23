"""Read-only `Store` over the HF bucket API (ADR-128): the controller's view of a campaign store.

Jobs write through the bucket MOUNT (a `LocalStore` at /scratch/<run_key>); the stateless controller only
lists and reads small files (plan state, summaries, ledger records) through the API, so it can stop and
restart at any time."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import time
from typing import Any

#: The controller's write prefixes (S5.13): preflight report + predictions (`_campaign/`), planned partitions
#: (`_partitions/`), action ownership records (`_actions/`), operator halt-clear records (`_operator/`) and
#: attestations (`_attest/`). A unit result is NEVER one of these; `_quarantine/` moves go only through
#: `quarantine_generation`.
CONTROLLER_PREFIXES = ("_campaign/", "_partitions/", "_actions/", "_operator/", "_attest/")


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

    _UPLOAD_PREFIXES = CONTROLLER_PREFIXES

    def upload_bytes(self, rel: str, data: bytes) -> None:
        """The controller's ONLY write, limited to its own record prefixes (S5.13). Never a unit result."""
        if not rel.startswith(self._UPLOAD_PREFIXES):
            raise PermissionError(f"controller uploads are limited to {self._UPLOAD_PREFIXES}, got {rel!r}")
        self._api().batch_bucket_files(self.bucket, add=[(data, self._full(rel))])

    def delete(self, rel: str) -> None:
        raise PermissionError("BucketStore is read-only: jobs write through the bucket mount")

    def _rel_of(self, full: str) -> str:
        """The path of a listed file RELATIVE to the run key (store prefix)."""
        return full[len(self.prefix) + 1 :] if self.prefix else full

    def _list_files(self, rel_prefix: str) -> list[tuple[str, str | None]]:
        """Every FILE under ``rel_prefix`` (recursive) as ``(path-relative-to-run-key, xet_hash)``. The xet hash
        drives the server-side copy (``batch_bucket_files(copy=)`` copies BY XET HASH, not by path)."""
        base = self._full(rel_prefix)
        out = []
        for entry in self._api().list_bucket_tree(self.bucket, prefix=base, recursive=True):
            if getattr(entry, "type", "file") == "file":
                out.append((self._rel_of(entry.path), getattr(entry, "xet_hash", None)))
        return out

    def quarantine_generation(self, pipeline: str, stage: str, token: str, record: Any) -> list[str]:
        """Move a generation and its attestations under ``_quarantine/{stage}/{token}/<utc>/`` (ADR-129 canary
        mismatch, S5.10). Copies every file of ``{pipeline}/{stage}/{token}/`` and ``_attest/{stage}/{token}/``
        server-side BY XET HASH (``copy=[(repo_type, repo_id, xet_hash, dest)]``, huggingface_hub 2.2), adds a
        record, THEN deletes the originals -- the ONLY method that deletes. Returns the moved paths (full,
        store-relative)."""
        items = sorted(self._list_files(f"{pipeline}/{stage}/{token}/") + self._list_files(f"_attest/{stage}/{token}/"))
        moved = [rel for rel, _ in items]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest_root = f"_quarantine/{stage}/{token}/{stamp}"
        copies = [("bucket", self.bucket, xh, self._full(f"{dest_root}/{rel}")) for rel, xh in items]
        rec = (json.dumps({**dict(record), "moved": moved, "utc": stamp}, sort_keys=True) + "\n").encode()
        self._api().batch_bucket_files(self.bucket, copy=copies, add=[(rec, self._full(f"{dest_root}/record.json"))])
        self._api().batch_bucket_files(self.bucket, delete=[self._full(rel) for rel in moved])
        return [self._full(rel) for rel in moved]

    def list_tree(self, rel_dir: str) -> set[str]:
        """Every FILE under ``rel_dir`` (recursive) as a path RELATIVE to ``rel_dir`` (ADR-129 generation
        compare). One recursive listing; the attestation diff reads each file by name afterward."""
        base = self._full(rel_dir).rstrip("/") + "/"
        out: set[str] = set()
        for entry in self._api().list_bucket_tree(self.bucket, prefix=base, recursive=True):
            if getattr(entry, "type", "file") == "file" and entry.path.startswith(base):
                out.add(entry.path[len(base) :])
        return out

    def size_bytes(self) -> int:
        """The sum of every file size under the run key (one recursive listing; `campaign status` reports it)."""
        total = 0
        for entry in self._api().list_bucket_tree(self.bucket, prefix=self.prefix, recursive=True):
            if getattr(entry, "type", "file") == "file":
                total += int(getattr(entry, "size", 0) or 0)
        return total

    def local_path(self, rel: str) -> pathlib.Path | None:
        return None


class CanaryReadThroughStore:
    """A canary instance's store (ADR-129 S5.10): it READS a path from the ``main`` store when that path lies
    inside an ALLOWED generation (an upstream generation reused from main; the allow set is serializable so a
    worker rebuilds the same store) and the local store lacks it, and WRITES only to ``local``. So the canary
    re-runs one unit over the reused upstream and its own output is compared to main byte for byte, never mixing
    into the reused stores.

    ``allow`` holds the full reused GENERATION strings (e.g. ``census/fit_prepass/<token>``); a path is served
    from main when it equals, or is nested under, one of them -- NOT by its first path segment, which is only
    the pipeline name (CCTP-IMPL-29). It lives here (the visibility-blind store module, not the controller) so an
    HF job's import closure never reaches `campaign` -> `registry` (the I1 gate, ADR-123 Amendment B)."""

    def __init__(self, local: Any, main: Any, *, allow: set[str]) -> None:
        self.local = local
        self.main = main
        self.allow = set(allow)

    def _allowed(self, rel: str) -> bool:
        return any(rel == g or rel.startswith(g.rstrip("/") + "/") for g in self.allow)

    def read_bytes(self, rel: str) -> bytes | None:
        data = self.local.read_bytes(rel)
        if data is not None:
            return data
        return self.main.read_bytes(rel) if self._allowed(rel) else None

    def names(self, rel_dir: str) -> set[str]:
        local = self.local.names(rel_dir)
        return local | self.main.names(rel_dir) if self._allowed(rel_dir) else local

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.local.write_bytes(rel, data)  # every write goes to the canary store, never main

    def delete(self, rel: str) -> None:
        self.local.delete(rel)

    def local_path(self, rel: str):
        p = getattr(self.local, "local_path", lambda _r: None)(rel)
        if p is not None:
            return p
        main_lp = getattr(self.main, "local_path", lambda _r: None)
        return main_lp(rel) if self._allowed(rel) else None
