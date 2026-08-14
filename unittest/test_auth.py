"""
Basic API tests for the Authentication routes (ndb_host/api/routes/auth.py).

Per requirement, auth gets a small set of "test one" style checks.
Endpoints: POST /auth/register, GET /auth/verify
"""

import pytest
from conftest import BASE_URL, USERNAME, AUTH, unique_suffix

pytestmark = pytest.mark.auth


@pytest.fixture(scope="module")
def client():
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def test_verify_valid_credentials(client):
    r = client.get(f"{BASE_URL}/auth/verify", auth=AUTH, timeout=30)
    body = r.json()
    assert r.status_code == 200
    assert body["user"]["username"] == USERNAME
    assert body["user"]["is_authenticated"] is True
    assert body["user"]["role"] == "super_user"


def test_verify_wrong_password(client):
    r = client.get(f"{BASE_URL}/auth/verify", auth=(USERNAME, "wrongpassword"), timeout=30)
    body = r.json()
    assert r.status_code == 200
    assert body["user"]["is_authenticated"] is False
    assert body["user"]["message"] == "Invalid credentials"


def test_verify_unknown_user(client):
    r = client.get(f"{BASE_URL}/auth/verify", auth=("nosuchuser", "whatever"), timeout=30)
    body = r.json()
    assert r.status_code == 200
    assert body["user"]["is_authenticated"] is False
    assert body["user"]["message"] == "Invalid credentials"


def test_verify_no_credentials(client):
    r = client.get(f"{BASE_URL}/auth/verify", timeout=30)
    assert r.status_code == 401


def test_register_new_user(client):
    uname = f"u_{unique_suffix()}"
    r = client.post(f"{BASE_URL}/auth/register", auth=AUTH, timeout=30,
                    json={"username": uname, "password": "testpass123", "user_role": "user"})
    body = r.json()
    assert r.status_code == 201
    assert body["success"] is True


def test_register_duplicate_user(client):
    r = client.post(f"{BASE_URL}/auth/register", auth=AUTH, timeout=30,
                    json={"username": "sathya", "password": "testpass123", "user_role": "user"})
    body = r.json()
    assert r.status_code == 201
    assert body["success"] is False
    assert "already exists" in body["message"].lower()


def test_register_short_password(client):
    r = client.post(f"{BASE_URL}/auth/register", auth=AUTH, timeout=30,
                    json={"username": f"u_{unique_suffix()}", "password": "abc", "user_role": "user"})
    assert r.status_code == 422


def test_register_without_auth(client):
    r = client.post(f"{BASE_URL}/auth/register", timeout=30,
                    json={"username": f"u_{unique_suffix()}", "password": "testpass123", "user_role": "user"})
    assert r.status_code == 401
