"""
Shared test fixtures for the NebulonDB API test suite.

Tests hit the live API (server must be running on 127.0.0.1:6969).
Authentication is HTTP Basic with user 'sathya'.
"""

import uuid
import requests
import pytest

BASE_URL = "http://127.0.0.1:6969/api/NebulonDB"
USERNAME = "sathya"
PASSWORD = "sathya"
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
    deact = client.post(f"{BASE_URL}/corpus/deactivate_corpus", auth=AUTH,
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
