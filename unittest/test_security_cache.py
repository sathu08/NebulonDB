"""Unit tests for the verified-credential cache in ``core.security``.

The cache exists so repeated authenticated API calls (NebulonMind makes 3-4
per user action) do not re-run bcrypt (~250 ms) for the same credentials.
These tests pin the security-relevant properties: only successes are cached,
a password change invalidates the old password, and TTL=0 disables it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "ndb_host"))

from core import security  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(security, "_AUTH_CACHE_TTL", 60.0)
    monkeypatch.setattr(security, "_AUTH_CACHE_MAX", 16)
    security.invalidate_auth_cache()
    yield
    security.invalidate_auth_cache()


def _count_verifications(monkeypatch):
    calls = {"n": 0}
    real = security.verify_password

    def counting(password, hashed_password):
        calls["n"] += 1
        return real(password, hashed_password)

    monkeypatch.setattr(security, "verify_password", counting)
    return calls


def test_repeated_success_verifies_once(monkeypatch):
    hashed = security.hash_password("correct horse")
    calls = _count_verifications(monkeypatch)

    assert security.verify_password_cached("alice", "correct horse", hashed) is True
    assert security.verify_password_cached("alice", "correct horse", hashed) is True
    assert security.verify_password_cached("alice", "correct horse", hashed) is True
    assert calls["n"] == 1


def test_failures_are_not_cached(monkeypatch):
    hashed = security.hash_password("right")
    calls = _count_verifications(monkeypatch)

    assert security.verify_password_cached("bob", "wrong", hashed) is False
    assert security.verify_password_cached("bob", "wrong", hashed) is False
    assert calls["n"] == 2


def test_password_change_invalidates_old_password(monkeypatch):
    old_hash = security.hash_password("old-pass")
    assert security.verify_password_cached("carol", "old-pass", old_hash) is True

    new_hash = security.hash_password("new-pass")
    # New stored hash => different cache key => the old password is rejected.
    assert security.verify_password_cached("carol", "old-pass", new_hash) is False
    assert security.verify_password_cached("carol", "new-pass", new_hash) is True


def test_invalidate_drops_entries(monkeypatch):
    hashed = security.hash_password("secret")
    calls = _count_verifications(monkeypatch)

    security.verify_password_cached("dave", "secret", hashed)
    security.invalidate_auth_cache("dave")
    security.verify_password_cached("dave", "secret", hashed)
    assert calls["n"] == 2

    security.invalidate_auth_cache()
    security.verify_password_cached("dave", "secret", hashed)
    assert calls["n"] == 3


def test_ttl_zero_disables_cache(monkeypatch):
    hashed = security.hash_password("secret")
    monkeypatch.setattr(security, "_AUTH_CACHE_TTL", 0.0)
    calls = _count_verifications(monkeypatch)

    security.verify_password_cached("erin", "secret", hashed)
    security.verify_password_cached("erin", "secret", hashed)
    assert calls["n"] == 2


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(security, "_AUTH_CACHE_MAX", 2)
    hashes = [security.hash_password(f"pw-{i}") for i in range(5)]
    for i, hashed in enumerate(hashes):
        assert security.verify_password_cached(f"u{i}", f"pw-{i}", hashed) is True
    assert len(security._auth_cache) <= 2
