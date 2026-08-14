"""
Basic API tests for the System routes (ndb_host/api/routes/system.py).

Per requirement, system gets a small set of "test one" style checks.
Endpoint: GET /dashboard/config
"""

import pytest
from conftest import BASE_URL

pytestmark = pytest.mark.system


@pytest.fixture(scope="module")
def client():
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def test_config_returns_host_port_url(client):
    r = client.get(f"{BASE_URL}/dashboard/config", timeout=30)
    body = r.json()
    assert r.status_code == 200
    assert "server" in body
    assert "host" in body["server"]
    assert "port" in body["server"]
    assert "url" in body["server"]
    assert body["server"]["port"] == 6969


def test_config_works_without_auth(client):
    r = client.get(f"{BASE_URL}/dashboard/config", timeout=30)
    assert r.status_code == 200


def test_config_url_consistent_with_host_port(client):
    body = client.get(f"{BASE_URL}/dashboard/config", timeout=30).json()
    assert body["server"]["url"] == f"http://{body['server']['host']}:{body['server']['port']}"


def test_config_host_is_nonempty(client):
    body = client.get(f"{BASE_URL}/dashboard/config", timeout=30).json()
    assert isinstance(body["server"]["host"], str)
    assert body["server"]["host"].strip() != ""
