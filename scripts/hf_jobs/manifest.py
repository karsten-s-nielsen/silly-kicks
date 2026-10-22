"""The artifact manifest (ADR-128 spec 5.7): every bundled and Hub artifact, with its declared EXPECTATIONS.

``docs/artifacts/manifest.toml`` is human-written, with ONE exception (spec INV-9, 5.7 step 6): `adopt` sets
an adopted artifact's ``training_commit`` to the verified campaign commit. It replaces the scattered test
constants (commits, events, classification, card sources, Hub roles) with ONE declared source, read by the
artifact tests, the publisher, the Hub validator and `adopt`. Observed fields (sha256, metrics summary, env,
campaign, adopted_at) live in ``docs/artifacts/observed.json``, written only by `adopt` -- never compared
against themselves (INV-9).

Generated card regions: a card wraps each generated value in inline markers
``<!-- generated:<name> -->...<!-- /generated:<name> -->``; `render_card` re-renders them from the manifest
and the artifact files, `stale_regions` names the ones that differ, and `strip_markers` removes the markers
before a card reaches the Hub (decision P7).
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: The repository root: manifest paths (wheel dirs, cards) are repo-relative, whatever the working dir.
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "artifacts" / "manifest.toml"
OBSERVED = REPO_ROOT / "docs" / "artifacts" / "observed.json"

#: One generated region: name in group 1, content in group 2.
REGION = re.compile(r"<!-- generated:([a-z0-9_:-]+) -->(.*?)<!-- /generated:\1 -->", re.S)
_MARKER = re.compile(r"<!-- /?generated:[a-z0-9_:-]+ -->")


@dataclasses.dataclass(frozen=True)
class RefitEvent:
    id: str
    version: str
    adr: str
    title: str
    basis: str
    cite: str


@dataclasses.dataclass(frozen=True)
class Artifact:
    id: str
    destination: str
    job: str
    variant: str
    tier: str
    certification: str
    dataset: str | None = None
    naming: str | None = None
    corpus: str | None = None
    training_commit: str | None = None
    commit_field: str | None = None
    refit_event: str | None = None
    card: str | None = None
    wheel_sibling: str | None = None
    mirror_of: str | None = None
    hub_class: str | None = None
    hub_role: str | None = None
    #: ADR-129 adoption provenance (written to ``observed.json`` by `adopt`, never to the human manifest; the
    #: fields exist on the dataclass so a future manifest may carry them). ``adopted_at_commit`` is the clean
    #: HEAD the reuse landed at; ``attested_by`` is the oldest ancestor commit that attested the generation.
    adopted_at_commit: str | None = None
    attested_by: str | None = None
    expect: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.destination.partition(":")[0]

    @property
    def path(self) -> str:
        if self.kind != "wheel":
            raise ValueError(f"{self.id} is not a wheel artifact")
        return self.destination.partition(":")[2]

    @property
    def repo_id(self) -> str:
        if self.kind != "hub":
            raise ValueError(f"{self.id} is not a Hub artifact")
        return self.destination.partition(":")[2]


@dataclasses.dataclass(frozen=True)
class Manifest:
    events: Mapping[str, RefitEvent]
    artifacts: Mapping[str, Artifact]
    roots: Mapping[str, Mapping[str, str]]
    #: The repo root the artifact paths resolve against (the manifest's ``../..``).
    base: Path = REPO_ROOT


_FIELDS = {f.name for f in dataclasses.fields(Artifact)}


def load(path: str | Path = MANIFEST) -> Manifest:
    from scripts._toml import load_toml

    raw = load_toml(Path(path))
    events = {k: RefitEvent(id=k, **v) for k, v in raw.get("events", {}).items()}
    artifacts: dict[str, Artifact] = {}
    for a in raw.get("artifacts", []):
        unknown = set(a) - _FIELDS
        if unknown:
            raise ValueError(f"manifest artifact {a.get('id')!r}: unknown field(s) {sorted(unknown)}")
        if a["id"] in artifacts:
            raise ValueError(f"manifest artifact {a['id']!r} is declared twice")
        if a["destination"].partition(":")[0] not in ("wheel", "hub"):
            raise ValueError(f"manifest artifact {a['id']!r}: destination must be wheel:<dir> or hub:<repo>")
        artifacts[a["id"]] = Artifact(**a)
    base = Path(path).resolve().parents[2]
    return Manifest(events=events, artifacts=artifacts, roots=dict(raw.get("roots", {})), base=base)


def weights_classification(man: Manifest | None = None) -> dict[str, tuple[str, str]]:
    """``{weights root: (classification, reason)}`` (the F1b classification gate's registry)."""
    man = man or load()
    return {root: (r["classification"], r["reason"]) for root, r in man.roots.items()}


def card_sources(man: Manifest | None = None) -> dict[str, str]:
    """``{hub repo: in-repo card}`` (the publisher's CARD_SOURCE)."""
    man = man or load()
    return {a.repo_id: a.card for a in man.artifacts.values() if a.kind == "hub" and a.card}


def hub_registry(man: Manifest | None = None) -> dict[str, tuple[str, str]]:
    """``{hub repo: (silly_kicks.tracking class, role)}`` (the Hub validator's HUB_REGISTRY)."""
    man = man or load()
    return {a.repo_id: (a.hub_class or "", a.hub_role or "") for a in man.artifacts.values() if a.kind == "hub"}


def _get(doc: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        doc = doc[part]
    return doc


def _wheel_reader(a: Artifact) -> Callable[[str], dict]:
    return lambda fname: json.loads((REPO_ROOT / a.path / fname).read_text(encoding="utf-8"))


def violations(a: Artifact, read: Callable[[str], dict] | None = None) -> list[str]:
    """The expectations ``a``'s files do not satisfy (empty == compliant): every ``expect`` entry, the
    ``commit_field`` == ``training_commit``, and the corpus rule. Corpus messages contain ``corpus``."""
    from scripts.hf_jobs.datasets import expected_identity, locked

    read = read or _wheel_reader(a)
    bad: list[str] = []

    def value(spec: str) -> Any:
        fname, _, key = spec.partition(":")
        return _get(read(fname), key)

    for spec, want in a.expect.items():
        try:
            got = value(spec)
        except (FileNotFoundError, KeyError) as exc:
            bad.append(f"{a.id}: {spec} missing ({exc!r})")
            continue
        if got != want:
            bad.append(f"{a.id}: {spec} = {got!r}, expected {want!r}")
    if a.commit_field:
        try:
            got = value(a.commit_field)
        except (FileNotFoundError, KeyError) as exc:
            bad.append(f"{a.id}: {a.commit_field} missing ({exc!r})")
        else:
            if got != a.training_commit:
                bad.append(f"{a.id}: {a.commit_field} = {got!r}, expected {a.training_commit!r}")
    if a.corpus == "public_ids":
        want = locked(a.dataset or "")["public_ids"]
        try:
            got = read("metrics.json")["corpus_match_ids"]
        except (FileNotFoundError, KeyError) as exc:
            bad.append(f"{a.id}: corpus_match_ids missing ({exc!r})")
        else:
            if got != want:
                bad.append(f"{a.id}: corpus ids {got!r} differ from the staging record {want!r}")
    elif a.corpus == "digest":
        want = expected_identity(a.dataset or "", a.naming or "flat")
        m = read("metrics.json")
        got = {k: m.get(k) for k in want}
        if got != want:
            bad.append(f"{a.id}: corpus identity {got!r} differs from the staging record {want!r}")
    return bad


# ---- generated card regions ---------------------------------------------------------------------------


def _metrics(a: Artifact, base: Path = REPO_ROOT) -> dict:
    return json.loads((base / a.path / "metrics.json").read_text(encoding="utf-8"))


def _short(a: Artifact) -> str:
    return (a.training_commit or "")[:7]


def _event(man: Manifest, a: Artifact) -> RefitEvent:
    if not a.refit_event:
        raise ValueError(f"{a.id} declares no refit_event")
    return man.events[a.refit_event]


#: Mirror-card table rows: label -> [(metrics key, format)] (formats copied from the card test).
_GK_ROWS: dict[str, list[tuple[Any, str]]] = {
    "Held-out CV euclidean MAE": [
        ("cv_mae_euclidean_mean", ".3f"),
        *[(("per_provider_mae_euclidean", p), ".3f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "> 30 m high-sweeper stratum MAE": [("high_sweeper_stratum_mae_mean", ".2f")],
    "Training corpus": [("n_games", "d"), ("n_samples", "d")],
}
_OUTFIELD_ROWS: dict[str, list[tuple[Any, str]]] = {
    "Held-out CV euclidean MAE": [
        ("cv_mae", ".2f"),
        *[(("cv_mae_by_provider", p), ".2f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "Per-possession CV MAE": [(("cv_mae_by_possession", p), ".2f") for p in ("in_possession", "out_of_possession")],
    "Per-slot CV MAE (slots 1&ndash;4)": [(("cv_mae_by_slot", s), ".2f") for s in ("1", "2", "3", "4")],
    "Training corpus": [("n_games", "d"), ("n_rows", "d")],
}


#: Table label -> region slug (``row:<slug>``). A slug carries no digit, so a marker never adds a number
#: to the value cell it wraps.
_ROW_SLUG: dict[str, str] = {
    "Held-out CV euclidean MAE": "cv-mae",
    "> 30 m high-sweeper stratum MAE": "high-sweeper-mae",
    "Training corpus": "corpus",
    "Per-possession CV MAE": "possession-mae",
    "Per-slot CV MAE (slots 1&ndash;4)": "slot-mae",
}


def table_rows(a: Artifact) -> dict[str, list[tuple[Any, str]]]:
    """The metrics-table rows a mirror card of wheel artifact ``a`` carries: label -> [(metrics key, format)]."""
    return _OUTFIELD_ROWS if a.job == "ghost_outfield" else _GK_ROWS


def row_region(label: str) -> str:
    """The region name of the table row labelled ``label``."""
    return f"row:{_ROW_SLUG[label]}"


def row_values(w: Artifact, label: str, base: Path = REPO_ROOT) -> list[str]:
    """The formatted numbers the ``label`` row of a mirror card of wheel artifact ``w`` must show."""
    m = _metrics(w, base)
    return [format(_key(m, key), fmt) for key, fmt in table_rows(w)[label]]


def _key(doc: dict, key: Any) -> Any:
    return doc[key] if isinstance(key, str) else doc[key[0]][key[1]]


def _card_artifact(card: str, man: Manifest) -> Artifact:
    hits = [a for a in man.artifacts.values() if a.card == card]
    if len(hits) != 1:
        raise ValueError(f"card {card!r} belongs to {len(hits)} manifest artifacts, expected exactly one")
    return hits[0]


def required_regions(card: str, man: Manifest) -> set[str]:
    """The regions a card must carry: a mirror card's refit head + one row per table label; a Hub-only
    card's wheel-sibling note + its own re-fit sentence; a wheel card's provenance line."""
    a = _card_artifact(card, man)
    if a.kind == "hub" and a.mirror_of:
        return {"refit_head"} | {row_region(label) for label in table_rows(man.artifacts[a.mirror_of])}
    if a.kind == "hub" and a.wheel_sibling:
        return {"hub_note", "hub_refit"}
    if a.kind == "wheel":
        return {"provenance"}
    return set()


def render_region(name: str, card: str, man: Manifest) -> str:
    """The text of generated region ``name`` of ``card``, from the manifest and the artifact files."""
    a = _card_artifact(card, man)
    if name == "refit_head":
        w = man.artifacts[a.mirror_of or ""]
        e = _event(man, w)
        return f"**{e.title} (silly-kicks {e.version} / {e.adr}; `training_commit={_short(w)}`).**"
    if name.startswith("row:"):
        raise ValueError(f"{name!r} is a table value cell: render it with render_row")
    if name == "hub_note":
        w = man.artifacts[a.wheel_sibling or ""]
        e = _event(man, w)
        leaf = Path(w.path).name
        return (
            f"In silly-kicks {e.version} the wheel's bundled `{leaf}` was re-fit {e.basis} "
            f"({e.cite}`training_commit={_short(w)}`)."
        )
    if name == "hub_refit":
        e = _event(man, a)
        return f"This Hub artifact was last re-fit in silly-kicks {e.version} ({e.adr}, `training_commit={_short(a)}`)."
    if name == "provenance":
        m = _metrics(a, man.base)
        e = _event(man, a)
        fields = [f"`run_commit={m['run_commit'][:7]}`"]
        fields += [f"`{k}: {m[k]}`" for k in ("corpus_visibility", "artifact_label") if k in m]
        fields += [f"{m[k]} {unit}" for k, unit in (("n_matches", "matches"), ("n_rows", "rows")) if k in m]
        return f"**Provenance (silly-kicks {e.version}).** " + " · ".join(fields)
    raise ValueError(f"unknown generated region {name!r}")


_NUM = re.compile(r"\d+(?:\.\d+)?")


def render_row(content: str, card: str, name: str, man: Manifest) -> str:
    """A mirror card's table VALUE cell (region ``name``) with its numbers replaced, in order, by the
    mirrored bundle's metrics (the prose around them is the card's own). Refuses an unknown row and a cell
    whose number count differs from the bundle's."""
    w = man.artifacts[_card_artifact(card, man).mirror_of or ""]
    labels = [label for label in table_rows(w) if row_region(label) == name]
    if len(labels) != 1:
        raise ValueError(f"{card}: {name!r} names no metrics-table row of {w.id}")
    want = row_values(w, labels[0], man.base)
    found = _NUM.findall(content)
    if len(found) != len(want):
        raise ValueError(f"{card} {name}: {len(found)} numbers in the cell, the bundle has {len(want)}")
    values = iter(want)
    return _NUM.sub(lambda _m: next(values), content)


def _fresh(name: str, content: str, card: str, man: Manifest) -> str:
    if name.startswith("row:"):
        return render_row(content, card, name, man)
    return render_region(name, card, man)


def _card_id(path: Path, base: Path) -> str:
    """The manifest ``card`` value of ``path``: repo-relative posix (an absolute path inside the repo is
    made relative; any other path is taken as given)."""
    resolved = path.resolve()
    if resolved.is_relative_to(base):
        return resolved.relative_to(base).as_posix()
    return path.as_posix()


def render_card(path: str | Path, man: Manifest | None = None, *, card_id: str | None = None) -> str:
    """The card text with every generated region re-rendered (markers kept)."""
    man = man or load()
    path = Path(path)
    card = card_id or _card_id(path, man.base)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    def sub(m: re.Match) -> str:
        name = m.group(1)
        return f"<!-- generated:{name} -->{_fresh(name, m.group(2), card, man)}<!-- /generated:{name} -->"

    return REGION.sub(sub, text)


def stale_regions(path: str | Path, man: Manifest | None = None, *, card_id: str | None = None) -> list[str]:
    """The generated regions whose content differs from their re-render."""
    man = man or load()
    path = Path(path)
    card = card_id or _card_id(path, man.base)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    stale = []
    for m in REGION.finditer(text):
        try:
            fresh = _fresh(m.group(1), m.group(2), card, man)
        except ValueError:
            stale.append(m.group(1))
            continue
        if fresh != m.group(2):
            stale.append(m.group(1))
    return stale


def strip_markers(text: str) -> str:
    """The card without its inline region markers (the Hub README, decision P7)."""
    return _MARKER.sub("", text)
