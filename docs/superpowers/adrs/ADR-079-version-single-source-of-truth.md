# ADR-079: Single source of truth for the package version

| Field | Value |
|---|---|
| **Date** | 2026-08-29 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

The package version was hand-maintained in **two** places with nothing enforcing agreement:
`pyproject.toml` `[project] version = "4.100.0"` (the packaging/wheel-metadata version) and
`silly_kicks/__init__.py` `__version__ = "4.100.0"` (the runtime attribute). They are two copies of
one number, and every release had to keep them in sync by hand — the drift that keeps being caught
manually at release time. A stale copy is silent: a wheel could publish one version while
`silly_kicks.__version__` reported another, and nothing in the build or the test suite would object.

`ruthless-efficiency` already solved this (its ADR-002, decisions 5 & 6): a dedicated
`ruthless/_version.py` holds the one literal, `pyproject` declares `dynamic = ["version"]` with
`[tool.hatch.version] path`, and `__init__` re-exports — *"Deleting the duplication beats testing for
it. Bump `ruthless/_version.py`; everything else follows."* silly-kicks' build backend is already
`hatchling` (`requires = ["hatchling>=1.27,<2"]`), so the same pattern works out of the box.

One difference from ruthless is worth recording so the rationale is not miscopied: ruthless *needed*
`_version.py` to be a separate module because `ruthless/__init__.py` imports a strategy, and any core
module reading the version from the package root would transitively import `ruthless.strategies` and
break its `core-isolation` import-linter contract. **silly-kicks has no such import contract.** It
adopts the dedicated module purely for the SSOT benefit and because an import-free `_version.py` lets
tooling read the version without importing the (heavy) `silly_kicks` package.

## Decision

Single-source the version in a new **`silly_kicks/_version.py`** (one import-free `__version__`
literal). `pyproject.toml` `[project]` declares **`dynamic = ["version"]`** and a
**`[tool.hatch.version] path = "silly_kicks/_version.py"`** table makes hatchling derive the
wheel/sdist metadata version from that module. `silly_kicks/__init__.py` **re-exports** it
(`from silly_kicks._version import __version__`, keeping `__all__ = ["__version__"]`). Bumping
`_version.py` is the whole source change for a release; `uv.lock` follows from `uv lock`, and the
wheel/sdist metadata follow from hatchling.

Deleting the duplication is preferred over a *consistency* test (which can only fail after the
inconsistency has already shipped to a reader — ADR-002's reasoning). One **structural guard** test is
added, which is a different thing: it fails the moment someone re-adds a static `version` literal to
`[project]` (or lets `__init__` hard-code a literal instead of re-exporting) — the *reintroduction*
that deletion alone cannot prevent, caught in the PR's CI rather than after a wrong version ships. It
reads repo files only (no install-time metadata), so it is not editable-install-fragile.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep both literals + add a consistency test asserting they are equal | No new module | A consistency test can only fail *after* the two disagree — i.e. after drift ships | Rejected — ADR-002's own reasoning; treats the symptom, not the cause |
| B. Point `[tool.hatch.version] path` at `silly_kicks/__init__.py` (no new file) | Single-sources with zero new files | Version literal stays buried in a file that also carries a docstring + imports; tools must read/execute more to get it; diverges from the ruthless template the spec names | Rejected — a dedicated import-free `_version.py` is the smallest bump target and readable without importing the package |
| C. Dedicated `silly_kicks/_version.py` + `dynamic` pyproject + re-export (**chosen**) | One editable site; wheel + runtime derive from one source; import-free; mirrors ruthless ADR-002 | One extra tiny module; contributors must know to bump `_version.py`, not pyproject | — |

## Consequences

### Positive

- **One editable version site.** A release bumps `silly_kicks/_version.py` and nothing else by hand.
- **Drift is impossible by construction.** Wheel/sdist metadata and the runtime `silly_kicks.__version__` both derive from `_version.py`; they cannot disagree.
- **A broken wiring fails loud, early.** CI runs `pip install -e ".[…]"` in three jobs, so a typo in `[tool.hatch.version] path` fails at install time — it cannot lie dormant until a release.
- **The publish guard keeps working unchanged.** `publish.yml` compares the git tag to the **built wheel** version (not pyproject), which hatchling now derives from `_version.py`.

### Negative

- **One more tiny module** to be aware of.
- **A contributor must know to bump `_version.py`, not `pyproject`.** Mitigated by the ADR and a one-line CLAUDE.md convention.

### Neutral

- `uv.lock`'s `silly-kicks` version line follows from `uv lock` (never hand-edited).
- `CHANGELOG.md` / `TODO.md` remain release notes, not a version mirror.
- `silly_kicks.__version__` stays the editable-install-safe way to read the version (the calibration scripts deliberately use it over `importlib.metadata.version(...)`); the re-export preserves that.

## CLAUDE.md Amendment

Versioning is a repo-wide policy, so (if approved) add one durable line under **Key conventions**:

> The package version is single-sourced in `silly_kicks/_version.py` (ADR-079) — `pyproject` is
> `dynamic = ["version"]`, and `silly_kicks/__init__.py` re-exports it. Bump that **one** file for a
> release; `uv.lock` follows from `uv lock` (never hand-edit it), and the wheel/sdist metadata follow
> from hatchling.

The line is version-free, so it never drifts. This is an additive convention, not an edit to any
frozen historical version number.

## Related

- **Spec:** `HANDOFF-version-ssot.md` (repo root)
- **Plan:** `docs/superpowers/plans/2026-08-29-version-ssot.md`
- **Template:** `ruthless-efficiency` ADR-002 (`docs/adr/ADR-002-cache-identity-and-code-provenance.md`, decisions 5 & 6) + `ruthless/_version.py` + its `[tool.hatch.version]`
- **Guard test:** `tests/test_version_single_source.py`

## Notes

The five files carrying the literal `4.100.0` before this change: `pyproject.toml` (→ dynamic),
`silly_kicks/__init__.py` (→ re-export), `uv.lock` (→ `uv lock`), `CHANGELOG.md` (history — frozen),
`TODO.md` "Current" summary (release note). After the change the number is editable in exactly one
place, `silly_kicks/_version.py`; the rest are derived or are release notes.
