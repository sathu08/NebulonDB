"""
NebulonDB Mesh Visualization
==========================

This module provides functionality to visualize NebulonDB mesh structures using Cytoscape.js.
It defines classes and methods to convert mesh data into a format suitable for rendering in a web browser,
including generating HTML output with embedded graph data and styles.
"""


import json

from pathlib import Path
from collections import Counter
from typing import Any

from dataclasses import dataclass

from db.engine.utils.constants import (
    NODE_ID,
    NODE_LABEL,
    EDGE_SOURCE,
    EDGE_TARGET,
    LARGE_GRAPH_THRESHOLD,
    LAYOUTS,
    RENDER_OPTIONS,
    DEFAULT_TEMPLATE,
    NebulonConfig,
    NebulonColors,
    NebulonRenderOptions,
)
from utils.logger import NebulonDBLogger

logger = NebulonDBLogger().get_logger()

# ---------- Type aliases ----------
NebulonNodeStyle = dict[str, Any]
NebulonNodeStyles = dict[str, NebulonNodeStyle]
NebulonLayout = dict[str, Any]

# ---------- NebulonGraphProfile ----------
@dataclass(frozen=True)
class NebulonGraphProfile:
    layout: NebulonLayout
    render_opts: NebulonRenderOptions
    node_styles: NebulonNodeStyles
    optimize_large_graph: bool

    @classmethod
    def from_graph(cls, nodes: list[dict], edges: list[dict]) -> "NebulonGraphProfile":
        node_count = len(nodes)
        mode = cls._render_mode(node_count)
        layout = LAYOUTS[mode]
        render_opts = RENDER_OPTIONS[mode]
        degrees = cls._calculate_degree(edges)
        node_styles = cls._auto_style_nodes(nodes, degrees)
        optimize = node_count > LARGE_GRAPH_THRESHOLD
        return cls(layout=layout, render_opts=render_opts, node_styles=node_styles, optimize_large_graph=optimize)

    @staticmethod
    def _render_mode(node_count: int) -> str:
        if node_count < NebulonConfig.SMALL_GRAPH:
            return "small"
        elif node_count < NebulonConfig.MEDIUM_GRAPH:
            return "medium"
        elif node_count < NebulonConfig.LARGE_GRAPH:
            return "large"
        return "huge"

    @staticmethod
    def _calculate_degree(edges: list[dict]) -> Counter:
        degree = Counter()
        for e in edges:
            degree[e[EDGE_SOURCE]] += 1
            degree[e[EDGE_TARGET]] += 1
        return degree

    @staticmethod
    def _auto_style_nodes(nodes: list[dict], degrees: Counter) -> NebulonNodeStyles:
        styles: NebulonNodeStyles = {}
        for node in nodes:
            d = degrees.get(node[NODE_ID], 0)
            size = min(
                NebulonConfig.BASE_NODE_SIZE + d * NebulonConfig.SIZE_FACTOR,
                NebulonConfig.MAX_NODE_SIZE,
            )
            if d > NebulonConfig.HUB_DEGREE:
                color = NebulonColors.HUB
            elif d > NebulonConfig.MEDIUM_DEGREE:
                color = NebulonColors.MEDIUM
            else:
                color = NebulonColors.NORMAL
            styles[node[NODE_ID]] = {"size": size, "color": color}
        return styles


# ---------- NebulonCytoscapeGraph ----------
class NebulonCytoscapeGraph:
    def __init__(
        self,
        graph_data: dict | None = None,
        template_path: str = "templates/graph.html",
    ):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.template_path = template_path
        if graph_data:
            self.nodes = graph_data.get("nodes", [])
            self.edges = graph_data.get("edges", [])

    # ---------- Fixed from_mesh ----------
    @classmethod
    def from_mesh(
        cls, mesh: dict, template_path: str = "templates/graph.html"
    ) -> "NebulonCytoscapeGraph":
        """
        Convert a mesh dict of the form:
          {
            "nodes": { id: {"name": label}, ... },
            "edges": [[source, target, label], ...]   # label optional
          }
        into a NebulonCytoscapeGraph.
        """
        nodes = [
            {
                NODE_ID: str(node_id),
                NODE_LABEL: str(
                    info.get("label") or info.get("name") or f"Node-{node_id}"
                ),
            }
            for node_id, info in mesh["nodes"].items()
        ]

        edges = []
        for edge_spec in mesh["edges"]:
            if len(edge_spec) == 3:
                src, tgt, lbl = edge_spec
                edges.append(
                    {
                        EDGE_SOURCE: str(src),
                        EDGE_TARGET: str(tgt),
                        "label": str(lbl),
                    }
                )
            elif len(edge_spec) == 2:
                src, tgt = edge_spec
                edges.append(
                    {EDGE_SOURCE: str(src), EDGE_TARGET: str(tgt)}
                )
            else:
                raise ValueError(
                    f"Edge must have 2 or 3 elements, got {edge_spec}"
                )

        return cls(
            graph_data={"nodes": nodes, "edges": edges},
            template_path=template_path,
        )

    def validate(self) -> None:
        node_ids = {n[NODE_ID] for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs detected.")
        for source, target in (
            (e[EDGE_SOURCE], e[EDGE_TARGET]) for e in self.edges
        ):
            if source not in node_ids:
                raise ValueError(f"Edge source '{source}' is not a node ID.")
            if target not in node_ids:
                raise ValueError(f"Edge target '{target}' is not a node ID.")

    def summary(self) -> dict[str, Any]:
        if not self.nodes:
            isolated_count = 0
        else:
            connected_ids = {e[EDGE_SOURCE] for e in self.edges} | {
                e[EDGE_TARGET] for e in self.edges
            }
            isolated_count = sum(
                1 for n in self.nodes if n[NODE_ID] not in connected_ids
            )
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "isolated_nodes": isolated_count,
        }

    def _build_elements(self, profile: NebulonGraphProfile) -> list[dict]:
        elements = []
        for node in self.nodes:
            nid = node[NODE_ID]
            style = profile.node_styles[nid]
            elements.append(
                {
                    "data": {
                        NODE_ID: nid,
                        NODE_LABEL: node[NODE_LABEL],
                        "size": style["size"],
                        "color": style["color"],
                    }
                }
            )
        for edge in self.edges:
            edge_data = {
                EDGE_SOURCE: edge[EDGE_SOURCE],
                EDGE_TARGET: edge[EDGE_TARGET],
            }
            if "label" in edge:
                edge_data["label"] = edge["label"]
            elements.append({"data": edge_data})
        return elements

    def _build_node_style(self, render_opts: NebulonRenderOptions) -> dict:
        style = {
            "width": "data(size)",
            "height": "data(size)",
            "background-color": "data(color)",
            "label": "data(label)",
            "color": "white",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": 12,
        }
        if not render_opts.show_labels:
            style["text-opacity"] = 0
        return style

    def _build_edge_styles(self) -> list[dict]:
        return [
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "#888",
                    "curve-style": "bezier",
                },
            },
            {
                "selector": "edge[label]",
                "style": {
                    "label": "data(label)",
                    "font-size": 10,
                    "color": "#ccc",
                    "text-rotation": "autorotate",
                    "text-margin-y": -8,
                    "text-opacity": 1,
                },
            },
        ]

    def _final_layout(self, profile: NebulonGraphProfile) -> NebulonLayout:
        return profile.layout

    def _render_html(
        self,
        elements: list[dict],
        final_layout: NebulonLayout,
        node_style: dict,
        edge_styles: list[dict],
    ) -> str:
        template_path = Path(self.template_path)
        template = (
            template_path.read_text(encoding="utf-8")
            if template_path.exists()
            else DEFAULT_TEMPLATE
        )
        html = template.replace("__ELEMENTS_JSON__", json.dumps(elements))
        html = html.replace("__LAYOUT_JSON__", json.dumps(final_layout))
        html = html.replace("__NODE_STYLE__", json.dumps(node_style))
        html = html.replace("__EDGE_STYLES__", json.dumps(edge_styles))
        return html

    def to_html(self, output_path: str | None = None) -> str:
        self.validate()
        profile = NebulonGraphProfile.from_graph(self.nodes, self.edges)
        elements = self._build_elements(profile)
        final_layout = self._final_layout(profile)
        node_style = self._build_node_style(profile.render_opts)
        edge_styles = self._build_edge_styles()
        html = self._render_html(elements, final_layout, node_style, edge_styles)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("Graph HTML written to %s (%d nodes, %d edges)",
                        output_path, len(self.nodes), len(self.edges))
        return html
