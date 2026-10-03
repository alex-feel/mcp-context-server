"""The two-principal case registry: the cases of every seam group in one tuple.

Each seam group keeps its cases in its own ``_access_scope_cases_<group>.py`` module, built
on the shared scaffold of :mod:`tests.repositories._access_scope_cases`. This module only
collects them, so the scaffold never imports a group and no import cycle forms. Both entry
points parametrize over :data:`CASES`.
"""

from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases_dedup import DEDUP_CASES
from tests.repositories._access_scope_cases_reads import READ_CASES

# The registered cases, each proving one seam for every scope it names.
CASES: tuple[AccessCase, ...] = (*DEDUP_CASES, *READ_CASES)
