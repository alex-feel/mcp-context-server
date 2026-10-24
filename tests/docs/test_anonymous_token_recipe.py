"""The anonymous-token recipe of docs/authentication.md mints a token the server accepts.

The "Anonymous Read Access" section is what an operator follows to mint the shared read-only
credential, so the recipe is run exactly as written, with only its algorithm and its issuer
and audience placeholders filled in, for one recommended and one non-recommended algorithm
of every key family the server accepts. The printed token must pass the server's own JWT
verifier for the anonymous subject, carry an expiry, and carry no group or role claim; any
warning the recipe raises fails the run, so a deprecated algorithm name cannot creep in.
"""

import contextlib
import io
import re
import secrets
import time
import warnings
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.asymmetric import rsa

from app.auth import _create_jwt_verifier
from app.settings.auth import AuthSettings

_DOC = Path(__file__).resolve().parents[2] / 'docs' / 'authentication.md'
_SECTION = 'Anonymous Read Access'
_HEADING = re.compile(r'^(?P<marks>#{1,6})[ \t]+(?P<text>.+?)[ \t]*$')
_FENCE = re.compile(r'^[ \t]*```(?P<info>\S*)')
_ALGORITHM_LINE = re.compile(r"^ALGORITHM = '[^']*'", re.MULTILINE)
_ISSUER_PLACEHOLDER = '<MCP_AUTH_JWT_ISSUER>'
_AUDIENCE_PLACEHOLDER = '<MCP_AUTH_JWT_AUDIENCE>'
_ISSUER = 'https://idp.example.com/realms/main'
_AUDIENCE = 'mcp-context-server'
_SUBJECT = 'anonymous'
_KEY_FILE = 'private.pem'

# One recommended and one non-recommended algorithm per key family: joserfc refuses the
# non-recommended ones unless the call names the algorithm, which the recipe must do.
_ALGORITHMS = ('RS256', 'RS512', 'PS256', 'ES256', 'ES384', 'Ed25519', 'HS512')


def _section_lines() -> list[str]:
    """Return the lines of the "Anonymous Read Access" section, its subsections included."""
    lines = _DOC.read_text(encoding='utf-8').splitlines()
    level: int | None = None
    section: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading is not None:
            depth = len(heading['marks'])
            if level is not None and depth <= level:
                break
            if level is None and heading['text'] == _SECTION:
                level = depth
                continue
        if level is not None:
            section.append(line)
    assert level is not None, f'{_DOC.name} has no "{_SECTION}" heading'
    return section


def _python_blocks() -> list[str]:
    """Return the source of every fenced Python block of the section, in document order."""
    blocks: list[str] = []
    current: list[str] | None = None
    for line in _section_lines():
        fence = _FENCE.match(line)
        if current is None:
            if fence is not None and fence['info'] == 'python':
                current = []
        elif fence is not None:
            blocks.append('\n'.join(current) + '\n')
            current = None
        else:
            current.append(line)
    return blocks


def _recipe() -> str:
    """Return the one Python block of the section that sets the signing algorithm."""
    recipes = [block for block in _python_blocks() if _ALGORITHM_LINE.search(block)]
    assert len(recipes) == 1, f'expected one Python block setting ALGORITHM under "{_SECTION}", found {len(recipes)}'
    return recipes[0]


def _key_material(algorithm: str) -> tuple[str, str]:
    """Generate the signing key of ``algorithm``'s family.

    Asymmetric families write a PKCS#8 PEM private key and verify with its PEM public key;
    HS* writes the shared secret itself, with the trailing newline a text editor leaves,
    and the server verifies with the bare secret.

    Returns:
        The key file text the recipe signs with, and the key the server verifies with.
    """
    family = algorithm[:2]
    if family == 'HS':
        secret = secrets.token_urlsafe(64)
        return f'{secret}\n', secret
    if family in {'RS', 'PS'}:
        private_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey | ed25519.Ed25519PrivateKey = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
        )
    elif algorithm == 'ES256':
        private_key = ec.generate_private_key(ec.SECP256R1())
    elif algorithm == 'ES384':
        private_key = ec.generate_private_key(ec.SECP384R1())
    else:
        private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _mint(recipe: str, algorithm: str) -> str:
    """Run the recipe in the current directory for ``algorithm`` and return the token it prints."""
    assert _ISSUER_PLACEHOLDER in recipe
    assert _AUDIENCE_PLACEHOLDER in recipe
    source, replaced = _ALGORITHM_LINE.subn(f"ALGORITHM = '{algorithm}'", recipe)
    assert replaced == 1
    source = source.replace(_ISSUER_PLACEHOLDER, _ISSUER).replace(_AUDIENCE_PLACEHOLDER, _AUDIENCE)
    printed = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with contextlib.redirect_stdout(printed):
            exec(compile(source, str(_DOC), 'exec'), {'__name__': '__recipe__'})
    return printed.getvalue().strip()


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm', _ALGORITHMS)
async def test_recipe_mints_a_token_the_server_accepts(
    algorithm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented recipe signs a token the server's verifier accepts for the anonymous subject."""
    key_file_text, verification_key = _key_material(algorithm)
    (tmp_path / _KEY_FILE).write_text(key_file_text, encoding='utf-8')
    monkeypatch.chdir(tmp_path)

    token = _mint(_recipe(), algorithm)

    verifier = _create_jwt_verifier(AuthSettings.model_validate({
        'MCP_AUTH_PROVIDER': 'jwt',
        'MCP_AUTH_JWT_PUBLIC_KEY': verification_key,
        'MCP_AUTH_JWT_ISSUER': _ISSUER,
        'MCP_AUTH_JWT_AUDIENCE': _AUDIENCE,
        'MCP_AUTH_JWT_ALGORITHM': algorithm,
    }))
    access = await verifier.load_access_token(token)

    assert access is not None, f'the server rejected the {algorithm} token the recipe printed'
    assert access.subject == _SUBJECT
    assert access.expires_at is not None
    assert access.expires_at > time.time()
    assert access.claims['jti']
    assert 'groups' not in access.claims
    assert 'roles' not in access.claims


def test_every_python_block_of_the_section_compiles() -> None:
    """The recipe and the pre-distribution check script are valid Python as printed."""
    blocks = _python_blocks()
    assert len(blocks) >= 2, f'expected the recipe and the check script under "{_SECTION}"'
    for block in blocks:
        compile(block, str(_DOC), 'exec')
