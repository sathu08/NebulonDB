"""
Shared test fixtures for the NebulonDB API test suite.

Tests hit the live API (server must be running on 127.0.0.1:6969).
Authentication is HTTP Basic with user 'sathya'.
"""

import os
import uuid
import requests
import pytest

BASE_URL = os.environ.get("NEBULONDB_URL", "http://127.0.0.1:6969/api/NebulonDB")
USERNAME = os.environ.get("NEBULONDB_USERNAME", "sathya")
PASSWORD = os.environ.get("NEBULONDB_PASSWORD", "sathya")
AUTH = (USERNAME, PASSWORD)
TIMEOUT = 300


def unique_suffix() -> str:
    return uuid.uuid4().hex[:10]


def make_dataset(n: int, prefix: str = "row") -> dict:
    """Build a text dataset of n rows for load_segment tests."""
    return {
        "title": [f"{prefix} title {i}: about space, time and vector databases" for i in range(n)],
        "body": [f"{prefix} body {i}: semantic search and machine learning notes." for i in range(n)],
    }


@pytest.fixture(scope="session")
def client() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def create_corpus(client, name: str, ndb_type: str = "orbit") -> dict:
    r = client.post(f"{BASE_URL}/corpus/create_corpus", auth=AUTH,
                    json={"corpus_name": name, "ndb_type": ndb_type}, timeout=TIMEOUT)
    return r.json()


def delete_corpus(client, name: str) -> dict:
    client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH,
                json={"corpus_name": name}, timeout=TIMEOUT).json()
    return client.post(f"{BASE_URL}/corpus/delete_corpus", auth=AUTH,
                       json={"corpus_name": name}, timeout=TIMEOUT).json()


@pytest.fixture()
def fresh_corpus(client):
    """Create a unique orbit corpus, yield its name, and clean it up afterwards."""
    name = f"api_test_{unique_suffix()}"
    create_corpus(client, name, "orbit")
    yield name
    delete_corpus(client, name)


def pytest_collection_modifyitems(config, items):
    """Refuse to mix live-API and isolated-storage tests in one process.

    Isolated tests swap ``NEBULONDB_HOME`` and import config-touching modules
    at import time, so they must run in a dedicated pytest process:

        pytest unittest/test_segment.py ...            # live API
        pytest unittest/test_cosmos_durability.py ...  # isolated
    """
    live = [item for item in items if "live" in item.keywords]
    isolated = [item for item in items if "isolated" in item.keywords]
    if live and isolated:
        raise pytest.UsageError(
            "Live-API tests and isolated storage tests were collected in the "
            "same pytest process. Run them separately, e.g. "
            "'pytest unittest/test_segment.py ...' and "
            "'pytest unittest/test_cosmos_durability.py ...'."
        )
