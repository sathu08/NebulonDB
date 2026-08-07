"""
API tests for the Corpus routes (ndb_host/api/routes/corpus.py).

30 tests: 10 small, 10 medium, 10 large.
Endpoints: create_corpus, list_corpus, delete_corpus, deactivate_corpus, activate_corpus
"""

import pytest
from conftest import (BASE_URL, AUTH, unique_suffix, create_corpus,
                      delete_corpus, make_dataset)

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def client():
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def corpus_name() -> str:
    return f"c_{unique_suffix()}"


# ==========================================================
# SMALL (10) - single corpus lifecycle basics
# ==========================================================

def test_small_create_corpus(client):
    name = corpus_name()
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is True
    assert body["corpus_name"] == name
    delete_corpus(client, name)


def test_small_create_orbit_corpus(client):
    name = corpus_name()
    body = create_corpus(client, name, "orbit")
    assert body["success"] is True
    assert body["corpus_name"] == name
    delete_corpus(client, name)


def test_small_create_duplicate_corpus(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is False
    assert body["exists"] is True
    delete_corpus(client, name)


def test_small_create_empty_name_rejected(client):
    r = client.post(f"{BASE_URL}/corpus/create_corpus", auth=AUTH, timeout=30,
                    json={"corpus_name": "", "ndb_type": "cosmos"})
    assert r.status_code == 422


def test_small_create_unauth_rejected(client):
    r = client.post(f"{BASE_URL}/corpus/create_corpus", timeout=30,
                    json={"corpus_name": corpus_name(), "ndb_type": "cosmos"})
    assert r.status_code == 401


def test_small_create_wrong_password_rejected(client):
    r = client.post(f"{BASE_URL}/corpus/create_corpus", auth=("sathya", "wrongpw"), timeout=30,
                    json={"corpus_name": corpus_name(), "ndb_type": "cosmos"})
    body = r.json()
    assert body["success"] is False
    assert "Invalid password" in body["message"]


def test_small_list_corpus(client):
    body = client.get(f"{BASE_URL}/corpus/list_corpus", auth=AUTH, timeout=30).json()
    assert body["success"] is True
    assert isinstance(body["data"]["corpus_list"], list)


def test_small_list_corpus_unauth(client):
    r = client.get(f"{BASE_URL}/corpus/list_corpus", timeout=30)
    assert r.status_code == 401


def test_small_delete_nonexistent_corpus(client):
    body = delete_corpus(client, corpus_name())
    assert body["success"] is False
    assert body["exists"] is False


def test_small_delete_unauth_rejected(client):
    r = client.post(f"{BASE_URL}/corpus/delete_corpus", timeout=30,
                    json={"corpus_name": corpus_name()})
    assert r.status_code == 401


# ==========================================================
# MEDIUM (10) - lifecycle transitions and edge cases
# ==========================================================

def test_medium_create_then_list_contains(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    body = client.get(f"{BASE_URL}/corpus/list_corpus", auth=AUTH, timeout=30).json()
    assert name in body["data"]["corpus_list"]
    delete_corpus(client, name)


def test_medium_deactivate_active_corpus(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    body = client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": name}).json()
    assert body["success"] is True
    assert "deactivate" in body["message"].lower()
    delete_corpus(client, name)


def test_medium_deactivate_twice_fails(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                json={"corpus_name": name})
    body = client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": name}).json()
    assert body["success"] is False
    assert "already deactivate" in body["message"].lower()
    delete_corpus(client, name)


def test_medium_deactivate_nonexistent_corpus(client):
    body = client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": corpus_name()}).json()
    assert body["success"] is False
    assert body["exists"] is False


def test_medium_activate_deactivated_corpus(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                json={"corpus_name": name})
    body = client.post(f"{BASE_URL}/corpus/activate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": name}).json()
    assert body["success"] is True
    assert "activate" in body["message"].lower()
    delete_corpus(client, name)


def test_medium_activate_active_corpus_fails(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    body = client.post(f"{BASE_URL}/corpus/activate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": name}).json()
    assert body["success"] is False
    assert "already active" in body["message"].lower()
    delete_corpus(client, name)


def test_medium_delete_active_corpus_requires_deactivate(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    body = client.post(f"{BASE_URL}/corpus/delete_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": name}).json()
    assert body["success"] is False
    assert "deactivate" in body["message"].lower()
    delete_corpus(client, name)


def test_medium_delete_deactivated_corpus(client):
    name = corpus_name()
    create_corpus(client, name, "cosmos")
    client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                json={"corpus_name": name})
    body = delete_corpus(client, name)
    assert body["success"] is True


def test_medium_system_corpus_cannot_be_deleted(client):
    body = client.post(f"{BASE_URL}/corpus/delete_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": "nebulon_origin"}).json()
    assert body["success"] is False
    assert "system" in body["message"].lower()


def test_medium_system_corpus_cannot_be_deactivated(client):
    body = client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": "nebulon_origin"}).json()
    assert body["success"] is False
    assert "system" in body["message"].lower()


# ==========================================================
# LARGE (10) - multi-corpus / bulk / scale
# ==========================================================

def test_large_bulk_create_ten_corpora(client):
    names = [corpus_name() for _ in range(10)]
    for n in names:
        assert create_corpus(client, n, "cosmos")["success"] is True
    body = client.get(f"{BASE_URL}/corpus/list_corpus", auth=AUTH, timeout=30).json()
    listed = body["data"]["corpus_list"]
    assert all(n in listed for n in names)
    for n in names:
        delete_corpus(client, n)


def test_large_bulk_create_and_delete_all(client):
    names = [corpus_name() for _ in range(5)]
    for n in names:
        create_corpus(client, n, "cosmos")
    for n in names:
        client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                    json={"corpus_name": n})
    for n in names:
        body = delete_corpus(client, n)
        assert body["success"] is True
    body = client.get(f"{BASE_URL}/corpus/list_corpus", auth=AUTH, timeout=30).json()
    assert all(n not in body["data"]["corpus_list"] for n in names)


def test_large_create_long_corpus_name(client):
    name = "c_" + "a" * 80 + unique_suffix()
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is True
    delete_corpus(client, name)


def test_large_create_unicode_corpus_name(client):
    name = f"c_unicodé_{unique_suffix()}"
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is True
    delete_corpus(client, name)


def test_large_create_name_with_special_chars(client):
    name = f"c_special_{unique_suffix()}_@#-"
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is True
    delete_corpus(client, name)


def test_large_full_lifecycle(client):
    name = corpus_name()
    assert create_corpus(client, name, "cosmos")["success"] is True
    # deactivate -> activate -> deactivate -> delete
    for _ in range(2):
        client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                    json={"corpus_name": name})
        client.post(f"{BASE_URL}/corpus/activate_corpus", auth=AUTH, timeout=30,
                    json={"corpus_name": name})
    client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH, timeout=30,
                json={"corpus_name": name})
    assert delete_corpus(client, name)["success"] is True


def test_large_many_list_calls_stable(client):
    names = [corpus_name() for _ in range(3)]
    for n in names:
        create_corpus(client, n, "cosmos")
    prev = None
    for _ in range(5):
        body = client.get(f"{BASE_URL}/corpus/list_corpus", auth=AUTH, timeout=30).json()
        assert body["success"] is True
        cur = tuple(sorted(body["data"]["corpus_list"]))
        if prev is not None:
            assert cur == prev
        prev = cur
    for n in names:
        delete_corpus(client, n)


def test_large_activate_nonexistent_corpus(client):
    body = client.post(f"{BASE_URL}/corpus/activate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": corpus_name()}).json()
    assert body["success"] is False
    assert body["exists"] is False


def test_large_system_corpus_cannot_be_activated_or_reactivated(client):
    body = client.post(f"{BASE_URL}/corpus/activate_corpus", auth=AUTH, timeout=30,
                       json={"corpus_name": "nebulon_origin"}).json()
    assert body["success"] is False


def test_large_create_recreate_after_delete(client):
    name = corpus_name()
    assert create_corpus(client, name, "cosmos")["success"] is True
    delete_corpus(client, name)
    body = create_corpus(client, name, "cosmos")
    assert body["success"] is True
    delete_corpus(client, name)
