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

    def write_if_absent(self, rel: str, data: bytes) -> bool:
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


def _dir_prefix(rel_dir: str) -> str:
    rel = rel_dir.strip("/")
    return f"{rel}/" if rel else ""


def _is_record(name: str) -> bool:
    """A small controller record read in bulk (a unit ledger or a stage profile), never a large unit result
    such as a ``.parquet``. These are the files `ledger.unit_records` and `live_history` read for every done
    stage of every instance -- the 804 s step's per-file downloads (TF-85)."""
    return name.endswith(".ledger.json") or name in ("_profile.json", "_reduce.json")


class StoreSnapshot:
    """One controller step's read view of a `BucketStore` (TF-85). The store is listed ONCE, recursively, on
    first use; `names`, `list_tree` and a missing file are answered from that listing with no further call. The
    first read of a record file (a ``.ledger.json`` or ``_profile.json``) downloads every uncached record
    sibling in its directory in ONE batch, so a done stage's unit ledgers come back in a single call instead of
    one per unit; a large result file beside them is read alone. ``content`` caches bytes by xet hash across
    steps, so a later ``--watch`` step downloads only the files that changed. Uploads go to the wrapped store.
    Reads may run on several threads; an upload must never run beside a read (the step uploads only in its
    serial grant loop)."""

    def __init__(self, store: BucketStore, content: dict[str, bytes] | None = None) -> None:
        self._store = store
        self._content = {} if content is None else content
        self._files: dict[str, str | None] | None = None  # path relative to the run key -> xet hash
        self._local: dict[str, bytes] = {}  # this step's bytes of a file listed without a xet hash
        self._digest: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "_store":  # not set yet (a copy under construction): never recurse
            raise AttributeError(name)
        return getattr(self._store, name)

    def _listing(self) -> dict[str, str | None]:
        if self._files is None:
            base = self._store._full("") + "/" if self._store.prefix else ""  # "rk/": never a sibling "rk2/"
            self._files = {
                entry.path[len(base) :]: getattr(entry, "xet_hash", None)
                for entry in self._store._api().list_bucket_tree(self._store.bucket, prefix=base, recursive=True)
                if getattr(entry, "type", "file") == "file" and entry.path.startswith(base)
            }
            self._files.update(dict.fromkeys(self._local))  # an upload made before the first listing
        return self._files

    def digest(self) -> str:
        """A content hash of the whole listing: equal digests mean an unchanged store (the plan-cache key)."""
        if self._digest is None:
            import hashlib

            h = hashlib.sha256()
            for rel, xh in sorted(self._listing().items()):
                h.update(f"{rel}\0{xh}\n".encode())
            self._digest = h.hexdigest()
        return self._digest

    def xet_hashes(self) -> set[str]:
        """Every xet hash in the listing: the content a later step may still read."""
        return {xh for xh in self._listing().values() if xh is not None}

    def names(self, rel_dir: str) -> set[str]:
        base = _dir_prefix(rel_dir)
        return {p[len(base) :] for p in tuple(self._listing()) if p.startswith(base) and "/" not in p[len(base) :]}

    def list_tree(self, rel_dir: str) -> set[str]:
        base = _dir_prefix(rel_dir)
        return {p[len(base) :] for p in tuple(self._listing()) if p.startswith(base)}

    def upload_bytes(self, rel: str, data: bytes) -> None:
        """Upload through the store, then serve the new bytes to this step's later reads (read-after-write)."""
        self._store.upload_bytes(rel, data)
        rel = rel.strip("/")
        self._local[rel] = data
        self._digest = None
        if self._files is not None:
            self._files[rel] = None

    def _cached(self, rel: str) -> bytes | None:
        xh = self._listing().get(rel)
        return self._content.get(xh) if xh is not None else self._local.get(rel)

    def read_bytes(self, rel: str) -> bytes | None:
        rel = rel.strip("/")
        if rel not in self._listing():
            return None
        hit = self._cached(rel)
        if hit is not None:
            return hit
        parent, _, name = rel.rpartition("/")
        if _is_record(name):  # fetch every uncached record sibling (unit ledgers + profile) in one call
            base = _dir_prefix(parent)
            batch = sorted(
                f"{base}{n}" for n in self.names(parent) if _is_record(n) and self._cached(f"{base}{n}") is None
            )
        else:
            batch = [rel]
        self._download(batch)
        return self._cached(rel)

    def _download(self, rels: list[str]) -> None:
        files = self._listing()
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [(self._store._full(r), str(pathlib.Path(tmp) / str(i))) for i, r in enumerate(rels)]
            self._store._api().download_bucket_files(self._store.bucket, pairs, raise_on_missing_files=False)
            for r, (_, dest) in zip(rels, pairs, strict=True):
                p = pathlib.Path(dest)
                if not p.is_file():
                    continue
                xh = files.get(r)
                if xh is not None:
                    self._content[xh] = p.read_bytes()
                else:
                    self._local[r] = p.read_bytes()
