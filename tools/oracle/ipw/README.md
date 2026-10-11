# TF-71 `ipw.R` oracle generator (dev-only)

Generates `tests/causal/fixtures/ipw_oracle.json` — the **independent R reference** that
`tests/causal/test_sensitivity.py` gate 2 compares the Python ZSB reimplementation against.

**Why Docker.** R is intentionally absent on dev hosts; a pinned `rocker/r-ver` image makes the
oracle regeneration **repeatable and version-pinned** (like the lakehouse's Spark-local Docker and the
`das-reference` frozen-oracle discipline). The Python reimplementation never produces these numbers — that
would make the faithfulness gate circular.

**License (important).** `bootsens` is **GPL-3**. It is **installed into the ephemeral dev image at
build time** (pinned by commit SHA) and is **never vendored into or distributed by** the (MIT) silly-kicks
repo. Only the generated numeric fixture (facts, not copyrightable) is committed. Do not copy `bootsens`
source into this repo.

**Independence HARD RULE.** The `<expected>` targets in `ipw_oracle.json` come ONLY from this R run.
Never hand-edit them and never fill them from the Python port. If the container has not produced the
fixture, Task 1 — and the commit — is **blocked**.

**Regenerate:**
```bash
docker build -t sk-ipw-oracle:4.4.1 tools/oracle/ipw
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)":/work sk-ipw-oracle:4.4.1
```
(On non-Windows: `-v "$(pwd)":/work`.) The container runs `gen_ipw_oracle.R`, which `library(bootsens)`s
the installed package, computes `get.extrema` / `extrema.os(att)` (incl. the `reg.adjust=TRUE` AIPW path)
on the pinned fixture, and writes the JSON with a provenance block (R version, pinned `bootsens` commit +
version). A reviewer can re-run this and diff the JSON.

**Files:** `Dockerfile` (pinned `rocker/r-ver` + `remotes` + `jsonlite`; installs `bootsens` GPL-3 pinned
to commit `6008a8d8`), `gen_ipw_oracle.R` (fixture inputs + computation + JSON). No `bootsens` source is
committed here.
