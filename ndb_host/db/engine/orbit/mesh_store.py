"""
NebulonDB Mesh Graph Store
==========================

In-memory graph store persisted as explicit rows in two segments:

    nebulon_mesh_nodes   – one row per node  {id, label, created_at}
    nebulon_mesh_edges   – one row per edge  {edge_id, from_id, to_id,
                                              relation, weight, created_at}

MeshStore
    add_node / add_edge / neighbors / traversal persistence
    resolve_node(ref) – map an int id or a string label onto a node id,
                        auto-creating a node when a label is unknown.
"""


import threading


from db.engine.utils import (
    FIELD_ID,
    FIELD_LABEL,
    FIELD_EDGE_ID,
    FIELD_FROM,
    FIELD_TO,
    FIELD_RELATION,
    FIELD_WEIGHT,
    FIELD_CREATED_AT,
)

from db.engine import NebulonCosmos
from .mesh_viz import NebulonCytoscapeGraph
from utils.logger import NebulonDBLogger

logger = NebulonDBLogger().get_logger()

# The persisted Cosmos index is keyed by bare record_id while the id space is
# global across all tables. Mesh rows must live in disjoint id ranges so they
# never collide with Nova (identity) or Document rows in the index.
NODE_ID_OFFSET = 2 << 40
EDGE_ID_OFFSET = 3 << 40


def _rename_id(doc):
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id") - NODE_ID_OFFSET
    return doc


class MeshStore:
    """Thread-safe store for graph nodes and edges with per-row persistence."""

    def __init__(
        self,
        store: NebulonCosmos,
        node_segment: str,
        edge_segment: str,
        mesh_graph_viz_html: str | None = None,
    ):
        self._store = store
        self.node_segment = node_segment
        self.edge_segment = edge_segment
        self.mesh_graph_viz_html = mesh_graph_viz_html
        self._lock = threading.RLock()
        self._nodes: dict[int, dict] = {}
        self._edges: list[dict] = []
        self._next_edge_id: int = 1

    # ---------- Node operations ----------
    def add_node(self, node_id: int, label: str | None = None,
                 created_at: str | None = None) -> None:
        with self._lock:
            self._nodes.setdefault(node_id, {})
            if label is not None:
                self._nodes[node_id][FIELD_LABEL] = label
            if created_at is not None:
                self._nodes[node_id][FIELD_CREATED_AT] = created_at

    def remove_node(self, node_id: int) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
            self._edges = [
                e for e in self._edges
                if e[FIELD_FROM] != node_id and e[FIELD_TO] != node_id
            ]

    def get_node(self, node_id: int) -> dict | None:
        with self._lock:
            node = self._nodes.get(node_id)
            return dict(node) if node else None

    def get_all_node_ids(self) -> list[int]:
        with self._lock:
            return list(self._nodes.keys())

    def count_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)

    def has_node(self, node_id: int) -> bool:
        with self._lock:
            return node_id in self._nodes

    def resolve_node(self, ref: int | str) -> int:
        """Return a node id for an int id or a string label.

        int: returned as-is (a node is created lazily on edge add if absent).
        str: looked up by node label; a missing label auto-creates a node.
        """
        with self._lock:
            if isinstance(ref, int) or (isinstance(ref, str) and ref.lstrip("-").isdigit()):
                return int(ref)
            for nid, node in self._nodes.items():
                if node.get(FIELD_LABEL) == ref:
                    return nid
            # auto-create a fresh node for an unknown label
            nid = self._next_node_id_locked()
            self._nodes[nid] = {FIELD_LABEL: ref}
            logger.info("MeshStore: auto-created node id=%s label=%r", nid, ref)
            return nid

    def _next_node_id_locked(self) -> int:
        existing = self._nodes.keys()
        return (max(existing) + 1) if existing else 1

    # ---------- Edge operations ----------
    def add_edge(self, source: int, target: int, relation: str,
                 weight: float = 1.0, created_at: str | None = None) -> int:
        with self._lock:
            # Idempotent: an identical edge already exists (re-posted request
            # or WAL replay after a crash) – return its id instead of
            # duplicating it.
            for e in self._edges:
                if (e[FIELD_FROM] == source and e[FIELD_TO] == target
                        and e[FIELD_RELATION] == relation):
                    return e[FIELD_EDGE_ID]
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            self._edges.append({
                FIELD_EDGE_ID: edge_id,
                FIELD_FROM: source,
                FIELD_TO: target,
                FIELD_RELATION: relation,
                FIELD_WEIGHT: weight,
                FIELD_CREATED_AT: created_at,
            })
            return edge_id

    def remove_edge(self, source: int, target: int,
                    relation: str | None = None) -> None:
        with self._lock:
            self._edges = [
                e for e in self._edges
                if not (e[FIELD_FROM] == source and e[FIELD_TO] == target
                        and (relation is None or e[FIELD_RELATION] == relation))
            ]

    def get_neighbors(self, node_id: int, direction: str = "both") -> list[tuple[int, str]]:
        with self._lock:
            neighbors = []
            if direction in ("out", "both"):
                for e in self._edges:
                    if e[FIELD_FROM] == node_id:
                        neighbors.append((e[FIELD_TO], e[FIELD_RELATION]))
            if direction in ("in", "both"):
                for e in self._edges:
                    if e[FIELD_TO] == node_id:
                        neighbors.append((e[FIELD_FROM], e[FIELD_RELATION]))
            return neighbors

    def get_all_edges(self) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self._edges]

    def edges_by_relation(self, relation: str) -> list[dict]:
        with self._lock:
            return [e for e in self._edges if e[FIELD_RELATION] == relation]

    # ---------- Persistence ----------
    def save(self) -> None:
        """Persist every node and edge as its own row."""
        with self._lock:
            # settle any label/created_at-only changes, then write rows
            node_rows = [
                {
                    FIELD_ID: nid + NODE_ID_OFFSET,
                    FIELD_LABEL: node.get(FIELD_LABEL, ""),
                    FIELD_CREATED_AT: node.get(FIELD_CREATED_AT),
                }
                for nid, node in self._nodes.items()
            ]
            for row in node_rows:
                existing = self._store.get_by_id(self.node_segment, row[FIELD_ID])
                if existing is None:
                    self._store.insert(self.node_segment, row)
                else:
                    self._store.update(self.node_segment, row)

            edge_rows = [
                {
                    FIELD_ID: e[FIELD_EDGE_ID] + EDGE_ID_OFFSET,
                    FIELD_EDGE_ID: e[FIELD_EDGE_ID],
                    FIELD_FROM: e[FIELD_FROM],
                    FIELD_TO: e[FIELD_TO],
                    FIELD_RELATION: e[FIELD_RELATION],
                    FIELD_WEIGHT: e[FIELD_WEIGHT],
                    FIELD_CREATED_AT: e[FIELD_CREATED_AT],
                }
                for e in self._edges
            ]
            for row in edge_rows:
                eid = row[FIELD_EDGE_ID]
                existing = self._store.get_by_id(self.edge_segment, eid)
                if existing is None:
                    self._store.insert(self.edge_segment, row)
                else:
                    self._store.update(self.edge_segment, row)

            # Remove rows for nodes/edges deleted since the last save,
            # otherwise a fresh engine reload resurrects removed graph
            # elements (remove_node/remove_edge are not durable).
            live_node_ids = set(self._nodes.keys())
            for rec in self._store.read_all(self.node_segment, include_internal=True):
                raw_id = rec.get("_id")
                if raw_id is None:
                    continue
                if int(raw_id) - NODE_ID_OFFSET not in live_node_ids:
                    self._store.delete(self.node_segment, raw_id)

            live_edge_ids = {e[FIELD_EDGE_ID] for e in self._edges}
            for rec in self._store.read_all(self.edge_segment, include_internal=True):
                raw_id = rec.get("_id")
                if raw_id is None:
                    continue
                if int(rec.get(FIELD_EDGE_ID)) not in live_edge_ids:
                    self._store.delete(self.edge_segment, raw_id)

            if self.mesh_graph_viz_html:
                try:
                    viz = NebulonCytoscapeGraph.from_mesh(self.to_dict())
                    viz.to_html(self.mesh_graph_viz_html)
                except Exception as exc:
                    logger.warning(f"MeshStore: skipping HTML viz ({exc})")

    def load(self) -> bool:
        """Read node/edge rows back into memory."""
        with self._lock:
            self._nodes = {}
            for rec in self._store.read_all(segment=self.node_segment, include_internal=True):
                rec = _rename_id(dict(rec))
                nid = int(rec.get("id", rec.get(FIELD_ID)))
                self._nodes[nid] = {
                    FIELD_LABEL: rec.get(FIELD_LABEL, ""),
                    FIELD_CREATED_AT: rec.get(FIELD_CREATED_AT),
                }
            self._edges = []
            max_edge = 0
            for rec in self._store.read_all(segment=self.edge_segment, include_internal=True):
                rec = dict(rec)
                eid = int(rec[FIELD_EDGE_ID])
                self._edges.append({
                    FIELD_EDGE_ID: eid,
                    FIELD_FROM: rec.get(FIELD_FROM),
                    FIELD_TO: rec.get(FIELD_TO),
                    FIELD_RELATION: rec.get(FIELD_RELATION, ""),
                    FIELD_WEIGHT: rec.get(FIELD_WEIGHT, 1.0),
                    FIELD_CREATED_AT: rec.get(FIELD_CREATED_AT),
                })
                max_edge = max(max_edge, eid)
            self._next_edge_id = max_edge + 1
            return bool(self._nodes) or bool(self._edges)

    # ---------- Serialization ----------
    def to_dict(self) -> dict:
        """Return the graph in the {nodes: {...}, edges: [...]} shape used by viz."""
        with self._lock:
            nodes = {
                nid: {"label": node.get(FIELD_LABEL, "")}
                for nid, node in self._nodes.items()
            }
            edges = [
                [e[FIELD_FROM], e[FIELD_TO], e[FIELD_RELATION]]
                for e in self._edges
            ]
            return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(
        cls,
        data: dict,
        store: NebulonCosmos,
        node_segment: str,
        edge_segment: str,
        mesh_graph_viz_html: str | None = None,
    ) -> "MeshStore":
        instance = cls(store, node_segment, edge_segment, mesh_graph_viz_html)
        for nid, node in data.get("nodes", {}).items():
            instance.add_node(int(nid), node.get("label"))
        for edge in data.get("edges", []):
            instance.add_edge(int(edge[0]), int(edge[1]), str(edge[2]))
        return instance
