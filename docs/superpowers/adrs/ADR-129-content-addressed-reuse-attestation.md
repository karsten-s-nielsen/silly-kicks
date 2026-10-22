# ADR-129: Campaign throughput, linear reduce and content-addressed reuse with attestation

| Field | Value |
|---|---|
| **Date** | 2026-10-21 |
| **Status** | Proposed (number claimed 2026-10-21; the full decision record lands with this cycle's code commit) |
| **Deciders** | Karsten Nielsen (owner rulings 1–13, 2026-10-21); reviewers A and B (spec r1–r9, plan r1–r4) |

## Context

The ADR-128 Cycle 1 campaign ran on real HF (21 instances, 145 `cpu-upgrade` jobs, 38.1 job-hours). It measured
three classes of waste: a quadratic stage reduce (one directory listing per unit over the bucket mount), a serial
critical path (one stage at a time per instance, every job filled to 60 minutes, fixed instance order), and no
result reuse across commits (the run key contains the commit). The controller's cost-history read took 1978 s for
the 3963 units of win_probability. ADR-128 "Cycle 1 outcome" records the evidence; the owner cut Cycle 1's
artifact work so that the re-fit runs on fixed code.

## Decision

The campaign-throughput cycle makes the reduce linear and verified-before-write, schedules DAG-parallel stages
critical-path first under a makespan planner (the job wall is the LPT makespan of its units on its W workers),
and makes stores content-addressed per `{job}-{tier}`. A stage's token digests its locked import closure, its
declared data files and Hub files, `uv.lock`, the image digest and the output-affecting environment, so a
result is reused only when HEAD would produce the same bytes (INV-10). Adoption is by token equality at a clean
HEAD, backed by an attestation (a fresh replica proved the token byte-identical on HEAD or an ancestor of HEAD) and
a per-campaign canary unit. Jobs run a prebuilt, digest-pinned, provenance-attested environment image.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Runtime dependency manifest per result (ccache / Bazel style) | precise | more store I/O; complex | owner ruling 3 |
| B. Keep `cross_commit_version` and add a CI gate | small change | a human still bumps a string | owner ruling 3 |
| C. A locked static closure + fail-closed runtime checks | automatic, gated, sound | coarse while `silly_kicks/__init__.py` imports most of the library | — |

## Consequences

To be completed with the code commit: the spec and the plan (approved 2026-10-21) are under review outside the
repo and land with this ADR's code commit, with ADR-128 P9 and P18 and ADR-037 amended by reference.

## Related

- **ADRs:** amends ADR-128 (P9 code identity, P18 same-commit replica, the controller and planner) and ADR-037 (the
  adoption rule); builds on ADR-052, ADR-068, ADR-073, ADR-088.
