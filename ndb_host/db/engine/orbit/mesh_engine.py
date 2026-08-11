"""
NebulonDB Mesh Graph Engine
===========================

Higher-level graph traversal and persistence layer, containing:
    MeshEngine            – BFS/DFS traversal, shortest path, connected components
    Node Operations       – add/remove/get with auto-creation on edge insert
    Edge Operations       – directed relation management with neighbor queries
    Graph Persistence     – full-graph save/load via MeshStore delegation
"""


import threading

from collections import deque

from db.engine import NebulonCosmos
from .mesh_store import MeshStore
from db.engine.utils import FIELD_RELATION
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


# =============================================================================
# NEW: GraphEngine – higher‑level graph logic (traversal, persistence)
# =============================================================================
class MeshEngine:
    """Adds traversal algorithms (BFS/DFS, shortest path) and persistence."""

    def __init__(
            self,
            store: NebulonCosmos,
            mesh_segment: str,
            node_segment: str,
            edge_segment: str,
            mesh_graph_viz_html: str | None = None,
        ):
        self._store = MeshStore(
            store, node_segment, edge_segment, mesh_graph_viz_html
        )
        self._lock = threading.RLock()

    # ---------- Core operations (delegate to store) ----------
    def add_node(self, node_id: int, label: str | None = None,
                 created_at: str | None = None) -> None:
        self._store.add_node(node_id, label, created_at)

    def remove_node(self, node_id: int) -> None:
        self._store.remove_node(node_id)

    def get_node(self, node_id: int) -> dict | None:
        return self._store.get_node(node_id)

    def has_node(self, node_id: int) -> bool:
        return self._store.has_node(node_id)

    def resolve_node(self, ref) -> int:
        return self._store.resolve_node(ref)

    def add_edge(self, source: int, target: int, relation: str,
                 weight: float = 1.0, created_at: str | None = None) -> int:
        # Ensure both nodes exist (auto‑create if missing)
        if self._store.get_node(source) is None:
            self._store.add_node(source)
        if self._store.get_node(target) is None:
            self._store.add_node(target)
        return self._store.add_edge(source, target, relation, weight, created_at)

    def remove_edge(self, source: int, target: int, relation: str | None = None) -> None:
        self._store.remove_edge(source, target, relation)

    def get_neighbors(self, node_id: int, direction: str = "both") -> list[tuple[int, str]]:
        return self._store.get_neighbors(node_id, direction)

    def get_all_edges(self) -> list[dict]:
        return self._store.get_all_edges()

    def edges_by_relation(self, relation: str) -> list[dict]:
        return [
            e for e in self._store.get_all_edges()
            if e.get(FIELD_RELATION) == relation
        ]

    def get_all_nodes(self) -> list[int]:
        return self._store.get_all_node_ids()

    def count_nodes(self) -> int:
        return self._store.count_nodes()

    def count_edges(self) -> int:
        return len(self._store.get_all_edges())

    def has_edges(self) -> bool:
        return len(self._store.get_all_edges()) > 0

    # ---------- Traversal ----------
    def bfs(self, start: int, max_depth: int = 3) -> list[int]:
        """Return all nodes reachable within max_depth (undirected)."""
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor, _ in self.get_neighbors(node, "both"):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return list(visited)

    def dfs(self, start: int, max_depth: int = 3) -> list[int]:
        """Return all nodes reachable within max_depth using depth-first search."""
        visited = set()
        stack = [(start, 0)]
        visited.add(start)
        while stack:
            node, depth = stack.pop()
            if depth >= max_depth:
                continue
            for neighbor, _ in self.get_neighbors(node, "both"):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, depth + 1))
        return list(visited)

    def shortest_path(self, source: int, target: int) -> list[int] | None:
        """BFS shortest path (unweighted). Returns None if unreachable."""
        if source == target:
            return [source]
        visited = {source: None}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor, _ in self.get_neighbors(node, "both"):
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == target:
                        # Reconstruct path
                        path = []
                        cur = target
                        while cur is not None:
                            path.append(cur)
                            cur = visited[cur]
                        return path[::-1]
                    queue.append(neighbor)
        return None

    def connected_components(self) -> list[set[int]]:
        """Find all connected components (undirected)."""
        all_nodes = set(self._store.get_all_node_ids())
        visited = set()
        components = []
        for node in all_nodes:
            if node not in visited:
                comp = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    comp.add(curr)
                    for nbr, _ in self.get_neighbors(curr, "both"):
                        if nbr not in visited:
                            stack.append(nbr)
                components.append(comp)
        return components

    # ---------- Persistence ----------
    def save(self) -> None:
        """Persist the whole graph as a single document in the graph segment."""
        with self._lock:
            self._store.save()
            logger.info("Mesh engine saved (%d nodes, %d edges)",
                        self._store.count_nodes(), len(self._store.get_all_edges()))

    def load(self) -> bool:
        """Read the single graph document and restore in‑memory state."""
        return self._store.load()

