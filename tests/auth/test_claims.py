"""Tests for IdP-agnostic claim extraction helpers.

Covers the claim key resolution order (flat key, full-URL key, dotted path),
string-or-list coercion, and the fail-closed Entra ID group-overage rule in
app.auth.claims.
"""

import logging

import pytest

from app.auth.claims import coerce_string_list
from app.auth.claims import extract_claim
from app.auth.claims import extract_groups
from app.auth.claims import extract_roles


class TestExtractClaim:
    """Tests for extract_claim key resolution."""

    def test_flat_key_lookup(self) -> None:
        """A plain top-level key resolves directly."""
        assert extract_claim({'groups': ['a']}, 'groups') == ['a']

    def test_missing_key_returns_none(self) -> None:
        """An absent key resolves to None."""
        assert extract_claim({'other': 1}, 'groups') is None

    def test_dotted_path_traversal(self) -> None:
        """A dotted key traverses nested claim objects (Keycloak realm_access.roles)."""
        claims = {'realm_access': {'roles': ['admin', 'user']}}
        assert extract_claim(claims, 'realm_access.roles') == ['admin', 'user']

    def test_exact_flat_key_wins_over_dotted_traversal(self) -> None:
        """A literal flat key containing dots takes precedence over traversal."""
        claims = {
            'realm_access.roles': ['flat'],
            'realm_access': {'roles': ['nested']},
        }
        assert extract_claim(claims, 'realm_access.roles') == ['flat']

    def test_full_url_key_is_looked_up_verbatim(self) -> None:
        """A full-URL claim key (Auth0 namespaced) is never split on dots."""
        claims = {'https://example.com/groups': ['kb-readers']}
        assert extract_claim(claims, 'https://example.com/groups') == ['kb-readers']

    def test_full_url_key_missing_returns_none_without_traversal(self) -> None:
        """A missing full-URL key returns None even when URL segments exist as nesting."""
        claims = {'https://example': {'com/groups': ['x']}}
        assert extract_claim(claims, 'https://example.com/groups') is None

    def test_dotted_path_through_non_mapping_returns_none(self) -> None:
        """Traversal through a non-object value resolves to None."""
        claims = {'realm_access': 'not-an-object'}
        assert extract_claim(claims, 'realm_access.roles') is None

    def test_dotted_path_missing_leaf_returns_none(self) -> None:
        """Traversal to an absent leaf resolves to None."""
        claims: dict[str, object] = {'realm_access': {'other': []}}
        assert extract_claim(claims, 'realm_access.roles') is None

    def test_deep_dotted_path(self) -> None:
        """Traversal handles more than two segments."""
        claims = {'a': {'b': {'c': 'leaf'}}}
        assert extract_claim(claims, 'a.b.c') == 'leaf'


class TestCoerceStringList:
    """Tests for coerce_string_list value coercion."""

    def test_none_is_empty(self) -> None:
        """None coerces to an empty list."""
        assert coerce_string_list(None) == []

    def test_string_becomes_single_element(self) -> None:
        """A plain string coerces to a one-element list."""
        assert coerce_string_list('editors') == ['editors']

    def test_empty_string_is_empty(self) -> None:
        """An empty string coerces to an empty list."""
        assert coerce_string_list('') == []

    def test_string_list_passes_through(self) -> None:
        """A list of strings passes through unchanged."""
        assert coerce_string_list(['a', 'b']) == ['a', 'b']

    def test_integer_elements_are_stringified(self) -> None:
        """Integer list elements are coerced to strings."""
        assert coerce_string_list(['a', 7]) == ['a', '7']

    def test_boolean_elements_are_skipped(self) -> None:
        """Boolean list elements are skipped, not stringified."""
        assert coerce_string_list([True, 'a', False]) == ['a']

    def test_empty_string_elements_are_skipped(self) -> None:
        """Empty string list elements are dropped."""
        assert coerce_string_list(['', 'a']) == ['a']

    def test_nested_structures_are_skipped(self) -> None:
        """Dict and list elements inside the list are skipped."""
        assert coerce_string_list([{'x': 1}, ['y'], 'z']) == ['z']

    def test_non_string_non_list_value_is_empty(self) -> None:
        """A dict or numeric claim value coerces to an empty list."""
        assert coerce_string_list({'groups': ['a']}) == []
        assert coerce_string_list(42) == []


class TestExtractGroups:
    """Tests for extract_groups including the Entra overage rule."""

    def test_groups_present(self) -> None:
        """A present groups claim resolves normally."""
        assert extract_groups({'groups': ['g1', 'g2']}, 'groups') == ['g1', 'g2']

    def test_groups_absent_without_markers(self) -> None:
        """An absent groups claim without overage markers is simply empty."""
        assert extract_groups({'sub': 'alice'}, 'groups') == []

    def test_overage_markers_fail_closed_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Absent groups claim plus Entra overage markers yields empty groups and a warning."""
        claims = {
            'sub': 'alice',
            '_claim_names': {'groups': 'src1'},
            '_claim_sources': {'src1': {'endpoint': 'https://graph.microsoft.com/v1.0/users/x/getMemberObjects'}},
        }
        with caplog.at_level(logging.WARNING, logger='app.auth.claims'):
            assert extract_groups(claims, 'groups') == []
        assert any('overage' in record.message for record in caplog.records)

    def test_single_overage_marker_also_fails_closed(self) -> None:
        """Either overage marker alone triggers the fail-closed empty list."""
        assert extract_groups({'_claim_names': {'groups': 'src1'}}, 'groups') == []

    def test_groups_present_with_markers_resolves_normally(self, caplog: pytest.LogCaptureFixture) -> None:
        """A present groups claim wins even when marker-shaped keys exist."""
        claims = {'groups': ['g1'], '_claim_names': {'other': 'src1'}}
        with caplog.at_level(logging.WARNING, logger='app.auth.claims'):
            assert extract_groups(claims, 'groups') == ['g1']
        assert not any('overage' in record.message for record in caplog.records)

    def test_custom_url_groups_claim(self) -> None:
        """A configured full-URL groups claim resolves verbatim."""
        claims = {'https://example.com/groups': ['kb']}
        assert extract_groups(claims, 'https://example.com/groups') == ['kb']


class TestExtractRoles:
    """Tests for extract_roles."""

    def test_roles_present(self) -> None:
        """A present roles claim resolves normally."""
        assert extract_roles({'roles': ['publisher']}, 'roles') == ['publisher']

    def test_roles_absent(self) -> None:
        """An absent roles claim is empty."""
        assert extract_roles({}, 'roles') == []

    def test_dotted_roles_claim(self) -> None:
        """A dotted roles claim traverses nested objects (Keycloak)."""
        claims = {'realm_access': {'roles': ['admin']}}
        assert extract_roles(claims, 'realm_access.roles') == ['admin']

    def test_single_string_role(self) -> None:
        """A scalar string role coerces to a one-element list."""
        assert extract_roles({'roles': 'publisher'}, 'roles') == ['publisher']
