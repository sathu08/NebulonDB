"""
API tests for the Segment routes (ndb_host/api/routes/segment.py).

30 tests: 10 small, 10 medium, 10 large.
Covers load/list/search/stats/record/mesh endpoints.
"""

import random
import pytest
from conftest import BASE_URL, AUTH, unique_suffix, make_dataset

pytestmark = pytest.mark.segment


@pytest.fixture(scope="module")
def client():
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def seg_name() -> str:
    return f"seg_{unique_suffix()}"


def load_segment(client, corpus, segment, dataset, **extra):
    payload = {"corpus_name": corpus, "segment_name": segment, "segment_dataset": dataset}
    payload.update(extra)
    return client.post(f"{BASE_URL}/segment/load_segment", auth=AUTH, timeout=300, json=payload)


# ==========================================================
# SMALL (10) - 3-row datasets
# ==========================================================

def test_small_load_segment_dict(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(3))
    body = r.json()
    assert r.status_code == 200
    assert body["success"] is True
    assert "Processed" in body["message"]


def test_small_load_segment_list_of_dicts(client, fresh_corpus):
    rows = [{"title": f"Row {i}", "body": f"Body {i}"} for i in range(3)]
    r = load_segment(client, fresh_corpus, seg_name(), rows)
    body = r.json()
    assert r.status_code == 200
    assert body["success"] is True


def test_small_load_segment_first_column(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(3), set_columns="First Column")
    body = r.json()
    assert body["success"] is True


def test_small_load_segment_all_columns(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(3), set_columns="All")
    body = r.json()
    assert body["success"] is True


def test_small_load_with_doc_and_lang(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(3),
                     doc_type="txt", lang_type="en")
    body = r.json()
    assert body["success"] is True


def test_small_list_segment_after_load(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    body = client.get(f"{BASE_URL}/segment/list_segment?corpus_name={fresh_corpus}",
                      auth=AUTH, timeout=60).json()
    assert body["success"] is True
    assert body["data"]["total_count"] == 1


def test_small_search_segment(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "space and time", "top_matches": 3}).json()
    assert r["success"] is True
    assert isinstance(r["data"], list)
    assert len(r["data"]) > 0


def test_small_segment_stats(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/segment_stats", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname}).json()
    assert r["success"] is True
    assert r["data"]["vector_count"] > 0


def test_small_get_record_found(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/get_record", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname, "record_id": 1}).json()
    assert r["success"] is True
    assert r["exists"] is True


def test_small_get_record_missing(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/get_record", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname, "record_id": 99999}).json()
    assert r["success"] is False
    assert r["exists"] is False


# ==========================================================
# MEDIUM (10) - 20-row datasets + graph operations
# ==========================================================

def test_medium_load_20_rows(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(20),
                     set_columns=["title", "body"])
    body = r.json()
    assert r.status_code == 200
    assert body["success"] is True
    assert "Processed 40" in body["message"]


def test_medium_load_with_relations(client, fresh_corpus):
    sname = seg_name()
    dataset = make_dataset(4)
    dataset["source"] = [1, 1, 2, 3]
    dataset["target"] = [2, 3, 3, 4]
    r = load_segment(client, fresh_corpus, sname, dataset)
    body = r.json()
    assert body["success"] is True


def test_medium_search_nova_mode(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "machine learning", "mode": "nova"}).json()
    assert r["success"] is True


def test_medium_search_mesh_mode(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "space", "mode": "mesh", "graph_start_node": 1}).json()
    assert r["success"] is True


def test_medium_search_hybrid_mode(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "time", "mode": "hybrid", "graph_start_node": 1}).json()
    assert r["success"] is True


def test_medium_search_top_matches(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "space", "top_matches": 5}).json()
    assert r["success"] is True
    assert len(r["data"]) <= 5


def test_medium_add_node(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/add_node", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "node_id": 900, "metadata": {"kind": "test"}}).json()
    assert r["success"] is True


def test_medium_add_relation_and_neighbors(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    client.post(f"{BASE_URL}/segment/add_relation", auth=AUTH, timeout=60,
                json={"corpus_name": fresh_corpus, "segment_name": sname,
                      "source": 1, "target": 2, "relation": "links"})
    r = client.post(f"{BASE_URL}/segment/get_neighbors", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname, "node_id": 1}).json()
    assert r["success"] is True
    assert len(r["data"]) == 1


def test_medium_bfs(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/bfs", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "start_node": 1, "max_depth": 3}).json()
    assert r["success"] is True
    assert r["data"]["total_count"] >= 1


def test_medium_shortest_path_no_path(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/shortest_path", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "source": 1, "target": 5}).json()
    assert r["success"] is False
    assert "No path" in r["message"]


# ==========================================================
# LARGE (10) - 60-row / precomputed / scale + edge cases
# ==========================================================

def test_large_load_60_rows(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), make_dataset(60),
                     set_columns=["title", "body"])
    body = r.json()
    assert r.status_code == 200
    assert body["success"] is True
    assert "Processed 120" in body["message"]


def test_large_load_precomputed_vectors(client, fresh_corpus):
    random.seed(42)
    vecs = [[round(random.random(), 4) for _ in range(384)] for _ in range(3)]
    dataset = {"embeddings": vecs}
    r = load_segment(client, fresh_corpus, seg_name(), dataset,
                     set_columns=["embeddings"], is_precomputed=True)
    body = r.json()
    assert r.status_code == 200
    assert body["success"] is True


def test_large_load_append_same_segment(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = load_segment(client, fresh_corpus, sname, make_dataset(20))
    assert r.json()["success"] is True


def test_large_list_segment_count(client, fresh_corpus):
    for i in range(3):
        load_segment(client, fresh_corpus, f"{seg_name()}_{i}", make_dataset(3))
    body = client.get(f"{BASE_URL}/segment/list_segment?corpus_name={fresh_corpus}",
                      auth=AUTH, timeout=60).json()
    assert body["success"] is True
    assert body["data"]["total_count"] == 3


def test_large_search_rank_enabled(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(20))
    r = client.post(f"{BASE_URL}/segment/search_segment", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "search_item": "space and time", "rank": True, "top_matches": 5}).json()
    assert r["success"] is True


def test_large_shortest_path_with_path(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    client.post(f"{BASE_URL}/segment/add_relation", auth=AUTH, timeout=60,
                json={"corpus_name": fresh_corpus, "segment_name": sname,
                      "source": 1, "target": 2, "relation": "edge"})
    client.post(f"{BASE_URL}/segment/add_relation", auth=AUTH, timeout=60,
                json={"corpus_name": fresh_corpus, "segment_name": sname,
                      "source": 2, "target": 3, "relation": "edge"})
    r = client.post(f"{BASE_URL}/segment/shortest_path", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "source": 1, "target": 3}).json()
    assert r["success"] is True
    assert r["data"]["path_length"] == 3


def test_large_remove_relation(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    client.post(f"{BASE_URL}/segment/add_relation", auth=AUTH, timeout=60,
                json={"corpus_name": fresh_corpus, "segment_name": sname,
                      "source": 1, "target": 2, "relation": "edge"})
    r = client.post(f"{BASE_URL}/segment/remove_relation", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "source": 1, "target": 2, "relation": "edge"}).json()
    assert r["success"] is True


def test_large_delete_record(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/delete_record", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname,
                          "record_id": 1}).json()
    assert r["success"] is True


def test_large_mesh_visualization(client, fresh_corpus):
    sname = seg_name()
    load_segment(client, fresh_corpus, sname, make_dataset(3))
    r = client.post(f"{BASE_URL}/segment/mesh_visualization", auth=AUTH, timeout=60,
                    json={"corpus_name": fresh_corpus, "segment_name": sname}).json()
    assert r["success"] is True
    assert "html_path" in r["data"]


def test_large_load_empty_dataset_rejected(client, fresh_corpus):
    r = load_segment(client, fresh_corpus, seg_name(), {})
    assert r.json()["success"] is False
