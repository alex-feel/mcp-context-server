"""Claim extraction helpers for IdP-agnostic JWT claims mapping.

Verified JWT claims differ in shape across identity providers: Keycloak nests
realm roles under ``realm_access.roles``, Auth0 requires namespaced full-URL
claim keys such as ``https://example.com/groups``, and Microsoft Entra ID
replaces the ``groups`` claim with overage markers when a user belongs to too
many groups. This module normalizes those shapes into plain string lists so
the rest of the application never inspects raw claim dictionaries.

Claim key resolution order:
1. A key containing ``://`` is a full-URL claim key (Auth0 namespaced custom
   claims); it is looked up verbatim only -- the dots inside the URL are part
   of the key, never a traversal path.
2. Any other key is first looked up verbatim at the top level; an exact flat
   key always wins.
3. When the flat lookup misses and the key contains dots, it is traversed as
   a dotted path through nested objects (Keycloak ``realm_access.roles``).

Entra ID group overage is handled fail-closed: when the configured groups
claim is absent and the token carries the ``_claim_names``/``_claim_sources``
overage markers, the group list is treated as empty and a warning is logged.
No Microsoft Graph call is made -- resolving overage requires an outbound
authenticated Graph request, which a token verifier must never perform.
"""

import logging
from collections.abc import Mapping
from collections.abc import Sequence
from typing import cast

logger = logging.getLogger(__name__)

# Entra ID emits these top-level markers instead of the groups claim when the
# user's group count exceeds the token size limit (the "overage" case).
_ENTRA_OVERAGE_MARKERS = ('_claim_names', '_claim_sources')


def extract_claim(claims: Mapping[str, object], key: str) -> object:
    """Resolve a claim value by flat key, full-URL key, or dotted path.

    Args:
        claims: Verified JWT claims.
        key: Claim key -- a plain name, a full-URL key (contains ``://``),
            or a dotted path into nested claim objects.

    Returns:
        The resolved claim value, or None when the key does not resolve.
    """
    if key in claims:
        return claims[key]

    # Full-URL claim keys are opaque: their dots belong to the URL.
    if '://' in key or '.' not in key:
        return None

    current: object = claims
    for segment in key.split('.'):
        if not isinstance(current, Mapping):
            return None
        mapping: Mapping[str, object] = current
        if segment not in mapping:
            return None
        current = mapping[segment]
    return current


def coerce_string_list(value: object) -> list[str]:
    """Coerce a claim value into a list of strings.

    Args:
        value: A claim value -- a string, a list, or anything else.

    Returns:
        A single-element list for a non-empty string value, string elements
        (with integer scalars stringified) for a list value, and an empty
        list for None or any other shape.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        # cast: isinstance narrows only to list[Unknown]; the element type is
        # opaque JSON from the token, so object is the honest element type.
        items = cast('Sequence[object]', value)
        result: list[str] = []
        for item in items:
            if isinstance(item, str):
                if item:
                    result.append(item)
            elif isinstance(item, bool):
                logger.debug('Skipping boolean claim list element')
            elif isinstance(item, int):
                result.append(str(item))
            else:
                logger.debug('Skipping non-string claim list element of type %s', type(item).__name__)
        return result
    logger.debug('Claim value of type %s is not a string or list; treating as empty', type(value).__name__)
    return []


def extract_groups(claims: Mapping[str, object], groups_claim: str) -> list[str]:
    """Extract group memberships from verified claims, fail-closed on Entra overage.

    Args:
        claims: Verified JWT claims.
        groups_claim: Configured claim key carrying group memberships.

    Returns:
        The caller's group list; empty when the claim is absent, including
        the Entra ID overage case (markers present, groups omitted), which
        logs a warning and never triggers a Microsoft Graph call.
    """
    value = extract_claim(claims, groups_claim)
    if value is None and any(marker in claims for marker in _ENTRA_OVERAGE_MARKERS):
        logger.warning(
            'Groups claim %r is absent but the token carries Entra ID group-overage markers '
            '(_claim_names/_claim_sources). Treating the group list as EMPTY (fail-closed); '
            'no Graph call is made. Reduce group count, filter groups in the app registration, '
            'or emit group claims via optional claims configuration to avoid overage.',
            groups_claim,
        )
        return []
    return coerce_string_list(value)


def extract_roles(claims: Mapping[str, object], roles_claim: str) -> list[str]:
    """Extract roles from verified claims.

    Args:
        claims: Verified JWT claims.
        roles_claim: Configured claim key carrying roles.

    Returns:
        The caller's role list; empty when the claim is absent.
    """
    return coerce_string_list(extract_claim(claims, roles_claim))
