"""The top-level interface: core types and subpackages, reachable from `import isinglib`."""

from __future__ import annotations

import isinglib


def test_subpackages_are_reachable_without_importing_them() -> None:
    for name in ("generators", "io", "mappings", "solvers", "states", "topology"):
        assert hasattr(isinglib, name)


def test_every_exported_name_exists() -> None:
    for name in isinglib.__all__:
        assert hasattr(isinglib, name)


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(isinglib.__version__, str) and isinglib.__version__
