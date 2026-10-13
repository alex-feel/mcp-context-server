"""The two-principal case registry: the cases of every seam group in one tuple.

Each seam group keeps its cases in its own ``_access_scope_cases_<group>.py`` module, built
on the shared scaffold of :mod:`tests.repositories._access_scope_cases`. This module only
collects them, so the scaffold never imports a group and no import cycle forms. Both entry
points parametrize over :data:`CASES`; the PostgreSQL entry point also runs
:data:`RLS_CASES` under row-level security and :data:`READ_DECIDED_CASES` with the database
as the only filter.
"""

from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases_aggregates import AGGREGATE_CASES
from tests.repositories._access_scope_cases_dedup import DEDUP_CASES
from tests.repositories._access_scope_cases_deletes import DELETE_CASES
from tests.repositories._access_scope_cases_ranked import RANKED_CASES
from tests.repositories._access_scope_cases_reads import READ_CASES
from tests.repositories._access_scope_cases_search import SEARCH_CASES
from tests.repositories._access_scope_cases_updates import RLS_UPDATE_CASES
from tests.repositories._access_scope_cases_updates import UPDATE_CASES

# The registered cases, each proving one seam for every scope it names.
CASES: tuple[AccessCase, ...] = (
    *DEDUP_CASES, *READ_CASES, *SEARCH_CASES, *UPDATE_CASES, *DELETE_CASES, *RANKED_CASES, *AGGREGATE_CASES,
)

# The cases that only a database enforcing row-level security runs, beside every case of CASES.
RLS_CASES: tuple[AccessCase, ...] = RLS_UPDATE_CASES

# The cases of the read, search, ranked and aggregate groups whose observable the rows each
# scope may read decide alone, so the database's answer for the scope's identity must
# reproduce it without the application predicate.
READ_DECIDED_CASES: tuple[AccessCase, ...] = tuple(
    case for case in (*READ_CASES, *SEARCH_CASES, *RANKED_CASES, *AGGREGATE_CASES) if not case.reports_write_access
)
