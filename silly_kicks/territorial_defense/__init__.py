"""silly_kicks.territorial_defense -- TF-54b SB360 territorial-defense counterfactual (DEMOTED, EXPERIMENTAL).

How much a defender's POSITIONING suppresses the attacking team's threat, via a model-free removal
(marginal-contribution) counterfactual on SB360 freeze-frames. Two arms: Arm A (action-anchored,
identity-exact) and Arm B (hull-based, attribution-approximate). Tracking-consuming sibling of
``gkdv`` / ``restdefense`` -- imports ``silly_kicks.tracking`` / ``keeper_identity`` / ``territory``
PUBLIC seams only; NOTHING imports it.

DEMOTED TO EXPERIMENTAL (ADR-090). The owner-run construct-validity battery (321 SB360 matches;
``docs/research/territorial_defense_construct_validity/``) found the removal arm ``instrument_void`` /
``not_responsive``: the dose or removal of a SINGLE defender barely moves the pitch-control threat
(the remaining players re-cover the vacated space), so the per-defender number is NOT a usable
defender valuation. Per the standing rule -- a metric that fails its validation must not ship AS that
metric -- the public metric surface (``compute_territorial_defense`` + its types + the three
feature-glossary columns + the SB360 boundary-audit verdict) is REMOVED. The code is RETAINED in the
private ``._compute`` / ``._arms`` / ``._engine`` / ``._config`` / ``._report`` / ``._columns`` modules
for the replacement-ghost redesign (the removal mechanism's deferred sibling); a ``scripts``/tests
consumer imports it from the private path. Mirrors the TF-60 Layer-3 demotion (ADR-089).

See NOTICE for full bibliographic citations.
"""

# DEMOTED (ADR-090): no public metric surface. Everything is retained as PRIVATE modules for the
# redesign; there is intentionally nothing to re-export here.
__all__: list[str] = []
