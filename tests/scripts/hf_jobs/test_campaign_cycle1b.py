"""The committed Cycle 1b campaign file: the Cycle 1 population at concurrency_cap = 20 (S5.5)."""

from __future__ import annotations


def test_cycle1b_is_the_cycle1_population_at_cap_20():
    from scripts.hf_jobs import campaign as C

    one = C.load_campaign("scripts/hf_jobs/campaigns/cycle1.toml")
    two = C.load_campaign("scripts/hf_jobs/campaigns/cycle1b.toml")
    assert two.name == "cycle1b" and two.concurrency_cap == 20
    assert [(j.job, j.tier, j.dataset, j.variants) for j in two.jobs] == [
        (j.job, j.tier, j.dataset, j.variants) for j in one.jobs
    ]
