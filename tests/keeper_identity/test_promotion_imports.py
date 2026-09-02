"""Promotion guard: the keeper-identity resolver's public home is ``silly_kicks.keeper_identity``.

The module was moved out of ``silly_kicks/tracking/_keeper_identity.py`` (a breaking import move,
no shim, fail-loud). These three checks pin the new home, the clean break from ``tracking``, and the
tracking-free import (importing the resolver must not drag in ``silly_kicks.tracking`` -- numba plus
~30 submodules).
"""

from __future__ import annotations


def test_public_home_exports_the_resolver():
    from silly_kicks.keeper_identity import (
        KEEPER_ID_SOURCE_VALUES,
        KeeperIdentity,
        add_defending_gk_player_id,
        apply_keeper_identities_to_frames,
        resolve_keeper_identities,
    )

    assert callable(resolve_keeper_identities)
    assert callable(add_defending_gk_player_id)
    assert callable(apply_keeper_identities_to_frames)
    assert KeeperIdentity._fields == ("gk_id", "source", "conflict")
    assert set(KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}


#: The full keeper surface that moved to ``silly_kicks.keeper_identity`` (breaking move, no shim).
_MOVED_SYMBOLS = (
    "resolve_keeper_identities",
    "add_defending_gk_player_id",
    "apply_keeper_identities_to_frames",
    "KeeperIdentity",
    "KeeperIdentityMap",
    "KeeperIdentityReport",
    "KEEPER_ID_SOURCE_EVENT",
    "KEEPER_ID_SOURCE_ROSTER",
    "KEEPER_ID_SOURCE_NATIVE",
    "KEEPER_ID_SOURCE_DERIVED",
    "KEEPER_ID_SOURCE_UNRESOLVED",
    "KEEPER_ID_SOURCE_VALUES",
)


def test_old_tracking_path_is_a_clean_break():
    import silly_kicks.tracking as T

    for name in _MOVED_SYMBOLS:
        assert not hasattr(T, name), f"{name} must no longer be re-exported from tracking (breaking, no shim)"


def test_tracking_features_submodule_is_also_a_clean_break():
    """No residual soft shim: the resolver must not be reachable via ``tracking.features`` either.

    ``tracking.features`` used to re-export the whole keeper surface for the package ``__init__`` to
    pull; the promotion removed both the import and the ``__all__`` entries, so this path must be gone
    too (controller ruling: full relocation, not a re-export).
    """
    import silly_kicks.tracking.features as TF

    leaked = [name for name in _MOVED_SYMBOLS if hasattr(TF, name)]
    assert not leaked, f"tracking.features still exposes moved keeper symbols (residual shim): {leaked}"


def test_importing_keeper_identity_does_not_import_tracking():
    """Importing the resolver must not drag in ``silly_kicks.tracking`` (numba + ~30 submodules).

    Run in a SUBPROCESS with a fresh interpreter. The in-process form (``del sys.modules[...tracking...]``
    then re-import) would POLLUTE the shared ``sys.modules`` for the rest of the pytest session: a
    later re-import of ``silly_kicks.tracking`` creates DUPLICATE module objects that desync the
    numba-compiled kernels / ``isinstance`` identity other tests rely on (measured: it silently failed
    ~29 downstream velocity-aggregator tests). A subprocess needs no ``sys.modules`` surgery and cannot
    leak state back into the parent.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import silly_kicks.keeper_identity\n"
        "assert 'silly_kicks.tracking' not in sys.modules, "
        "'keeper_identity must stay tracking-free at import'\n"
    )
    # controlled: sys.executable + a fixed in-repo code string, no untrusted input.
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, f"tracking-free import check failed:\n{result.stderr}"
