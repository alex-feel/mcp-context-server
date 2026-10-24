"""Tests that the app.migrations package describes and exports every migration module.

The package docstring lists each public migration module, the package re-exports every
``apply_*`` entry point those modules define, and the database preparation phase takes
each migration from the package rather than from the defining module.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import app.migrations as migrations
import app.startup.database_setup as database_setup


def _public_module_names() -> list[str]:
    return sorted(info.name for info in pkgutil.iter_modules(migrations.__path__) if not info.name.startswith('_'))


def test_every_public_migration_module_is_described_in_the_package_docstring() -> None:
    docstring = migrations.__doc__ or ''

    undescribed = [name for name in _public_module_names() if f'- {name}.py:' not in docstring]

    assert undescribed == []


def test_every_migration_entry_point_is_exported_by_the_package() -> None:
    unexported: list[str] = []
    for name in _public_module_names():
        module = importlib.import_module(f'app.migrations.{name}')
        for attribute, value in vars(module).items():
            defined_here = inspect.iscoroutinefunction(value) and value.__module__ == module.__name__
            if attribute.startswith('apply_') and defined_here and (
                attribute not in migrations.__all__ or getattr(migrations, attribute, None) is not value
            ):
                unexported.append(f'{name}.{attribute}')

    assert unexported == []


def test_the_database_preparation_phase_imports_migrations_from_the_package() -> None:
    tree = ast.parse(Path(inspect.getfile(database_setup)).read_text(encoding='utf-8'))

    submodule_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith('app.migrations.')
    ]

    assert submodule_imports == []
