# ADR-116: reversibility is the only publish bar — corpus ids are readable

| Field | Value |
|---|---|
| **Date** | 2026-10-08 |
| **Status** | Accepted |
| **Deciders** | owner (Karsten) |

## Context

The owner's standing data-redistribution policy is: **derived work is shareable; the underlying provider data is not. The only test is REVERSIBILITY.** A committed research artifact may publish anything a reader without data access cannot use to reconstruct the provider's raw tracking/events.

Corpus ids — `match_id` / `game_id` / `player_id` / `team_id`, every provider — are non-reversible REFERENCES, not data. They let anyone WITH data access (the owner, a licensee) verify a result against the source; anyone WITHOUT access cannot reconstruct anything from an id. They are therefore readable everywhere, and always have been.

During the combined-provenance and TF-58 cycles, a stricter SkillCorner id tier was introduced and encoded in a test (`tests/test_research_artifacts_carry_no_ids.py`) and a session memory: "only the 20 public A-League SkillCorner match ids are readable; every other SkillCorner id must be pseudonymized (`scp_<12hex>`) or absent", attributed to an "owner ruling 2026-10-05". That tier was a mis-recording of the reversibility policy, not an owner decision — the owner's position (ids stay) was consistent throughout. The gate would have forced pseudonymizing ~900 SkillCorner match ids out of the TF-58 corpus artifacts, which both contradicts the policy and destroys the verification affordance the ids exist to provide.

## Decision

The publish bar is reversibility alone. Corpus ids are readable in committed artifacts, every provider, no pseudonymization.

`tests/test_research_artifacts_carry_no_ids.py` is repurposed: it no longer inspects ids. It scans `docs/research/` (JSON values, parquet columns, md/csv prose) for the one publishable-leak a scan can mechanically catch — a LOCAL FILESYSTEM PATH (an absolute user/host/temp path, or a Windows drive path) that would leak a run environment and never belongs in a committed artifact (the standing no-local-paths-in-committed-docs rule). Non-reversibility of the aggregates themselves is a review property, not a regex. The filename is kept (legacy) to avoid churn in the artifacts and plan docs that reference it; its docstring records the correction.

## Consequences

- TF-58's corpus artifacts (`docs/research/tf58_team_coordination/`) and all future research artifacts carry their real corpus ids. A reader with data access can verify any row against the source.
- The `scp_<12hex>` pseudonymization mechanism and the public-20 allowlist are removed from the gate; no artifact needs an id scrub.
- The gate still fails loud on a leaked local path, and its coverage + anti-vacuity scaffolding is retained (a planted path is caught; corpus ids and relative repo paths are not).
- Supersedes the mis-recorded 2026-10-05 SkillCorner id tier. The reversibility policy itself is unchanged — this ADR records it correctly.
