"""Classify the form-B search hits: which `.to_numpy(dtype=float)` reads (no `na_value`) can see a nullable column.

Usage, from the repository root (the B hits are the `rg` output lines, `path:line:text`):

    python docs/research/action-time-repair/pandas-floor/scripts/classify_b.py . <file with the B hits>

Masked (nullable) column names = every column a library schema dict declares as Int*/UInt*/Float*/boolean (the
capitalised pandas extension dtypes). A hit whose subject literal names such a column is AT_RISK at pandas 2.1;
a hit on a literal column that no schema declares masked is SAFE (numpy float/int storage); a hit on a variable
(no literal) is UNKNOWN and needs a manual read. AT_RISK is a name match only: the README records which of those
names are masked only in a provider bronze map, not in the canonical schema.
"""

import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
hits_file = pathlib.Path(sys.argv[2])
decl = re.compile(
    r"""["']([A-Za-z_]\w*)["']\s*:\s*["'](Int8|Int16|Int32|Int64|UInt8|UInt16|UInt32|UInt64|Float32|Float64|boolean)["']"""
)
masked: dict[str, set[str]] = collections.defaultdict(set)
for p in (root / "silly_kicks").rglob("*.py"):
    for m in decl.finditer(p.read_text(encoding="utf-8")):
        masked[m.group(1)].add(m.group(2))

hit = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<text>.*)$")
col = re.compile(r"""\[\s*["']([A-Za-z_]\w*)["']\s*\]\s*\.to_numpy\(|\.(?P<attr>[A-Za-z_]\w*)\.to_numpy\(""")
buckets: dict[str, list[tuple[str, list[str], str]]] = {"AT_RISK": [], "SAFE": [], "UNKNOWN": []}
for raw in hits_file.read_text(encoding="utf-8").splitlines():
    m = hit.match(raw.replace("\\", "/"))
    if not m:
        continue
    text = m.group("text")
    names = [n for c in col.finditer(text) if (n := c.group(1) or c.group("attr"))]
    where = f"{m.group('file')}:{m.group('line')}"
    if any(n in masked for n in names):
        buckets["AT_RISK"].append((where, [n for n in names if n in masked], text.strip()[:110]))
    elif names:
        buckets["SAFE"].append((where, names, ""))
    else:
        buckets["UNKNOWN"].append((where, [], text.strip()[:110]))
print("declared masked columns:", len(masked))
for k, v in buckets.items():
    print(f"{k}: {len(v)}")
print("--- AT_RISK")
for w, n, t in buckets["AT_RISK"]:
    print(w, n, "|", t)
print("--- SAFE column names (count)")
print(collections.Counter(n for _, ns, _ in buckets["SAFE"] for n in ns).most_common(40))
print("--- UNKNOWN (a variable subject: searched, not read; R80)")
for w, _, t in buckets["UNKNOWN"]:
    print(w, "|", t)
