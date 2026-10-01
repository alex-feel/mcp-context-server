"""IN / NOT_IN membership predicates for metadata filters.

``MembershipConditionsMixin`` builds the type-aware list-membership predicates
that ``MetadataQueryBuilder`` inherits; the builder supplies the backend type,
the placeholder offset, and the accumulating conditions and bound parameters.
"""

from typing import Any

from app.metadata_sql import in_list_text
from app.metadata_sql import pg_ci
from app.metadata_sql import pg_int_origin_probe
from app.metadata_sql import pg_json_accessor
from app.metadata_sql import pg_number_guard
from app.metadata_sql import pg_safe_float8
from app.metadata_sql import pg_text_accessor
from app.metadata_sql import pg_text_guard
from app.metadata_sql import sqlite_bool_guard
from app.metadata_sql import sqlite_number_guard
from app.metadata_sql import sqlite_text_guard


class MembershipConditionsMixin:
    """IN / NOT_IN list-membership conditions for the metadata query builder.

    Builds one type-aware predicate per IN or NOT_IN filter, matching string,
    numeric, and boolean members only against stored values of the same JSON
    type so SQLite and PostgreSQL return identical rows. The predicate numbers
    its own PostgreSQL placeholders from ``param_offset`` and the parameters
    bound so far, and appends to ``conditions`` and ``parameters``, which the
    inheriting builder owns and initializes per instance.
    """

    backend_type: str
    param_offset: int
    parameters: list[Any]
    conditions: list[str]

    def _membership_match_sql(
        self,
        json_path: str,
        key_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> str:
        """Build a SQL predicate true when the stored value matches any list member.

        Members are matched TYPE-AWARE so the SQLite and PostgreSQL result sets are
        identical and no member is compared across JSON types:

        - STRING members match a JSON-STRING-typed stored value only (text guard), with
          optional ASCII case-fold. A stored JSON NUMBER is NOT compared as text -- its
          text form diverges across backends for out-of-int64 / high-precision values
          (SQLite double-rendered vs PostgreSQL exact ``->>``), mirroring the string-only
          EQ/NE/comparison contract.
        - NUMERIC members match a JSON-NUMBER-typed stored value NUMERICALLY (number
          guard / numeric accessor), so ``in [1, 2, 3]`` keeps matching stored numbers
          identically on both backends without any text comparison.
        - BOOLEAN members match a JSON-BOOLEAN-typed stored value only (bool guard), so a
          boolean member never collides with a same-text integer 1/0 or string
          'true'/'false'.

        Bound params are appended in placeholder order; used by IN (the predicate) and
        NOT_IN (its negation under an IS NOT NULL presence guard).

        Args:
            json_path: The validated ``$.``-prefixed metadata path (SQLite).
            key_path: The same path without the ``$.`` prefix (PostgreSQL accessor).
            values: The IN / NOT_IN member list.
            case_sensitive: When False, string members fold case (numbers/booleans never do).

        Returns:
            A parenthesized SQL predicate; appends its bound params to ``self.parameters``.
        """
        # bool is a subclass of int, so partition booleans out FIRST.
        bool_members = [v for v in values if isinstance(v, bool)]
        numeric_members = [v for v in values if not isinstance(v, bool) and isinstance(v, (int, float))]
        string_members = [v for v in values if isinstance(v, str)]
        fold = not case_sensitive and bool(string_members)
        parts: list[str] = []
        if self.backend_type == 'sqlite':
            if string_members:
                text_acc = f"json_extract(metadata, '{json_path}')"
                lhs = f'LOWER({text_acc})' if fold else text_acc
                ph = ', '.join('?' for _ in string_members)
                parts.append(f'({sqlite_text_guard(json_path)} AND {lhs} IN ({ph}))')
                self.parameters.extend(in_list_text(v, backend_type=self.backend_type, lower=fold) for v in string_members)
            if numeric_members:
                ph = ', '.join('?' for _ in numeric_members)
                parts.append(
                    f"({sqlite_number_guard(json_path)} AND "
                    f"json_extract(metadata, '{json_path}') IN ({ph}))",
                )
                self.parameters.extend(numeric_members)
            if bool_members:
                acc = f"CAST(json_extract(metadata, '{json_path}') AS TEXT)"
                bph = ', '.join('?' for _ in bool_members)
                parts.append(f'({sqlite_bool_guard(json_path)} AND {acc} IN ({bph}))')
                self.parameters.extend(in_list_text(v, backend_type=self.backend_type, lower=False) for v in bool_members)
        else:  # postgresql
            acc = pg_text_accessor(key_path)
            json_acc = pg_json_accessor(key_path)
            if string_members:
                start = self.param_offset + len(self.parameters) + 1
                ph = ', '.join(f'${start + i}::TEXT' for i in range(len(string_members)))
                lhs = pg_ci(acc) if fold else acc  # ASCII-only fold (parity with SQLite LOWER)
                parts.append(f'({pg_text_guard(key_path)} AND {lhs} IN ({ph}))')
                self.parameters.extend(in_list_text(v, backend_type=self.backend_type, lower=fold) for v in string_members)
            if numeric_members:
                # Numeric equality with SQLite's exact integer/double semantics, emitted ONCE
                # PER FILTER rather than once per member. The exact/double discriminator
                # (pg_int_origin_probe) and the double fallback (pg_safe_float8, which
                # inlines two long decimal literals) depend only on the STORED value, not on
                # the member, so the whole member list rides ONE discriminator with an IN list
                # on each arm. Rebuilding the discriminator per member made the generated
                # statement text grow by kilobytes per FLOAT member while the bind count grew
                # by one, so a fully cap-legal request (100 filters x 100 float members) built
                # tens of megabytes of SQL on PostgreSQL and kilobytes on SQLite.
                # INT and FLOAT members stay separated because their semantics differ: an int
                # param compares exact NUMERIC for EVERY stored value, while a float param
                # takes the discriminator (see pg_numeric_compare).
                # Wrap the whole OR-group in an explicit jsonb_typeof = 'number' boolean guard
                # (pg_number_guard), mirroring SQLite's sqlite_number_guard AND-form: an
                # explicit boolean guard (rather than a value-or-NULL accessor) is required for
                # NOT_IN, which is `present AND NOT match` -- for a present NON-number stored
                # value a NULL match would make NOT NULL stay NULL and drop the row on
                # PostgreSQL, while SQLite's explicit FALSE guard yields NOT FALSE -> TRUE and
                # KEEPS it. The guard forces a deterministic FALSE for a non-number on BOTH
                # backends, so IN and NOT_IN agree.
                num_guard = pg_number_guard(key_path)
                num = f'({acc})::NUMERIC'
                int_placeholders: list[str] = []
                float_placeholders: list[str] = []
                for m in numeric_members:
                    pos = self.param_offset + len(self.parameters) + 1
                    # bool is already partitioned out, so isinstance(m, int) is an exact
                    # int-vs-float discrimination here. Members keep their original binding
                    # order; only their SQL grouping differs.
                    target = int_placeholders if isinstance(m, int) else float_placeholders
                    target.append(f'${pos}')
                    self.parameters.append(m)
                num_parts: list[str] = []
                if int_placeholders:
                    num_parts.append(f'({num} IN (' + ', '.join(int_placeholders) + '))')
                if float_placeholders:
                    exact_list = ', '.join(float_placeholders)
                    double_list = ', '.join(f'({ph})::float8' for ph in float_placeholders)
                    num_parts.append(
                        f'(CASE WHEN {pg_int_origin_probe(num)} '
                        f'THEN {num} IN ({exact_list}) '
                        f'ELSE {pg_safe_float8(num)} IN ({double_list}) END)',
                    )
                parts.append(f'({num_guard} AND (' + ' OR '.join(num_parts) + '))')
            if bool_members:
                bguard = f"jsonb_typeof({json_acc}) = 'boolean'"
                start = self.param_offset + len(self.parameters) + 1
                bph = ', '.join(f'${start + i}::TEXT' for i in range(len(bool_members)))
                parts.append(f'({bguard} AND {acc} IN ({bph}))')
                self.parameters.extend(in_list_text(v, backend_type=self.backend_type, lower=False) for v in bool_members)
        return '(' + ' OR '.join(parts) + ')'

    def _add_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add an IN condition for list membership.

        Delegates to :meth:`_membership_match_sql`, which compares members as TEXT but
        gates boolean members on the JSON type so a boolean member matches only a JSON
        boolean (never a same-text integer 0/1 on SQLite or string 'true'/'false' on
        PostgreSQL), keeping the backends identical.
        """
        if not values:
            self.conditions.append('0 = 1')
            return
        self.conditions.append(self._membership_match_sql(json_path, json_path[2:], values, case_sensitive))

    def _add_not_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add a NOT IN condition.

        The match predicate is :meth:`_membership_match_sql` (type-aware for booleans);
        NOT_IN is its negation under an explicit presence guard so a missing key (or JSON
        null) is excluded -- the same three-valued-logic outcome the prior bare
        ``... NOT IN (...)`` produced via NULL propagation, now with boolean members
        matched JSON-boolean-only on both backends.
        """
        if not values:
            self.conditions.append('1 = 1')
            return

        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            present = f"json_extract(metadata, '{json_path}') IS NOT NULL"
        else:  # postgresql
            present = f'{pg_text_accessor(key_path)} IS NOT NULL'
        match = self._membership_match_sql(json_path, key_path, values, case_sensitive)
        self.conditions.append(f'({present} AND NOT {match})')
