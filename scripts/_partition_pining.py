"""Pining-backed match selection, severed from the generic ADR-052 partition seam (ADR-123 Amendment B).

``list_match_ids`` is the one ``_partition`` helper that reached the pining loader
(``_loader_pining``, which carries the banned visibility symbols). It lives HERE,
not in ``_partition``, so the in-job trainer's static import closure
(``train_xshot_occurrence`` / ``train_xcross_attempt`` -> ``_driver`` -> ``_partition``) never reaches
``_loader_pining``. That makes the I1 visibility-blind gate a pure STATIC-closure check needing no
``_INFRA_LEAVES`` exemption (the gold-standard severance that replaced it). Maintainer-only: the
``--providers`` corpus-selection path imports it; an HF in-job trainer never does.
"""

from __future__ import annotations


def list_match_ids(providers: list[str]) -> dict[str, list[str]]:
    """Every available match id per provider, as the JSON a ``--match-ids-json`` split is built from.

    Consumes the loader's own ``_list_matches`` -- the exact call ``load_matches`` makes internally,
    so the id set cannot drift from what a run would actually fetch.
    """
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token

    tok, base = _resolve_token(None), _base_url()
    return {p: [str(m["id"]) for m in _list_matches(p, tok, base)] for p in providers}
