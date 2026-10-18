import pytest

from scripts._driver import shard_select


def test_disjoint_cover_over_two_or_more_shards():
    seq = [f"m{i}" for i in range(7)]
    parts = [shard_select(seq, i, 3) for i in range(3)]
    # disjoint
    seen = [x for p in parts for x in p]
    assert len(seen) == len(set(seen))
    # complete cover, order-preserving union
    assert sorted(seen) == sorted(seq)
    # every shard non-trivial for N <= len (the >=2-shards determinism precondition)
    assert all(len(p) >= 1 for p in parts)


def test_n_one_is_identity():
    seq = [1, 2, 3]
    assert shard_select(seq, 0, 1) == [1, 2, 3]


@pytest.mark.parametrize("index,n", [(-1, 3), (3, 3), (0, 0)])
def test_rejects_bad_bounds(index, n):
    with pytest.raises(ValueError):
        shard_select([1, 2, 3], index, n)
