"""
Constants used across the database engine – magic numbers, binary type codes,
fixed binary format strings, and graph (mesh) visualisation constants.
"""

import struct

from pathlib import Path
from dataclasses import dataclass
from typing import Any

# -------------------- Database file magic & version --------------------
MAGIC: bytes = b"NDB4"
VERSION: int = 4

# -------------------- Binary serialisation type codes --------------------
TYPE_NULL: int = 0
TYPE_BOOL: int = 1
TYPE_INT: int = 2
TYPE_FLOAT: int = 3
TYPE_STRING: int = 4
TYPE_BYTES: int = 5
TYPE_LIST: int = 6
TYPE_DICT: int = 7

# -------------------- Binary header / entry formats --------------------
HEADER_FORMAT: str = "4s I I I ?"
RECORD_HEADER_FORMAT: str = "III"
ENTRY_FORMAT: str = "<QIQ Q"

HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)
RECORD_HEADER_SIZE: int = struct.calcsize(RECORD_HEADER_FORMAT)
ENTRY_SIZE: int = struct.calcsize(ENTRY_FORMAT)

# -------------------- Field names used in records and metadata ------------
FIELD_ID: str = "_id"
FIELD_NOVA: str = "Nova_Data"
FIELD_MESH: str = "Mesh_Data"
FIELD_METADATA: str = "metadata"

FIELD_VECTOR: str = "vector"
FIELD_TEXT: str = "text"
FIELD_LABEL: str = "label"
FIELD_EDGE_ID: str = "edge_id"
FIELD_FROM: str = "from_id"
FIELD_TO: str = "to_id"
FIELD_RELATION: str = "relation"
FIELD_WEIGHT: str = "weight"
FIELD_CREATED_AT: str = "created_at"

GRAPH_MASTER_ID: int = 1

# -------------------- Graph (mesh) field names --------------------
NODE_ID: str = "id"
NODE_LABEL: str = "label"
EDGE_SOURCE: str = "source"
EDGE_TARGET: str = "target"

LARGE_GRAPH_THRESHOLD: int = 3000


class NebulonConfig:
    SMALL_GRAPH = 100
    MEDIUM_GRAPH = 1000
    LARGE_GRAPH = 5000
    HUB_DEGREE = 50
    MEDIUM_DEGREE = 20
    BASE_NODE_SIZE = 15
    MAX_NODE_SIZE = 60
    SIZE_FACTOR = 2


class NebulonColors:
    NORMAL = "#4CAF50"
    MEDIUM = "#ffaa00"
    HUB = "#ff4444"


LAYOUTS: dict[str, dict[str, Any]] = {
    "small":  {"name": "cose",   "animate": True,  "padding": 40},
    "medium": {"name": "cose",   "animate": False, "padding": 30},
    "large":  {"name": "circle"},
    "huge":   {"name": "grid"},
}

DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
<style>
html, body { margin: 0; width: 100%; height: 100%; background: #111; }
#cy { width: 100%; height: 100%; }
#search-container { position: absolute; top: 10px; right: 10px; z-index: 10; }
#search-input { padding: 8px 12px; font-size: 14px; border: 2px solid #4CAF50;
    border-radius: 20px; background: #222; color: white; outline: none; width: 200px; transition: 0.3s; }
#search-input:focus { border-color: #FFD700; box-shadow: 0 0 5px #FFD700; }
</style>
</head>
<body>
<div id="search-container"><input type="text" id="search-input" placeholder="Search node..."></div>
<div id="cy"></div>
<script>
var elements = __ELEMENTS_JSON__;
var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: elements,
    style: [
        { selector: 'node', style: __NODE_STYLE__ },
        ...__EDGE_STYLES__,
        { selector: '.highlighted', style: { 'border-width': 4, 'border-color': '#FFD700', 'border-opacity': 0.8 } },
        { selector: '.highlighted-edge', style: { 'line-color': '#FFD700', 'width': 4 } }
    ],
    layout: __LAYOUT_JSON__
});
var searchInput = document.getElementById('search-input');
searchInput.addEventListener('input', function(e) {
    var query = e.target.value.trim().toLowerCase();
    cy.elements().removeClass('highlighted');
    cy.edges().removeClass('highlighted-edge');
    if (query === '') return;
    var matches = cy.nodes().filter(function(node) {
        var label = (node.data('label') || '').toLowerCase();
        var id = (node.data('id') || '').toLowerCase();
        return label.includes(query) || id.includes(query);
    });
    if (matches.empty()) return;
    var targetNode = matches.first();
    targetNode.addClass('highlighted');
    targetNode.connectedEdges().addClass('highlighted-edge');
    targetNode.neighborhood().nodes().addClass('highlighted');
    cy.animate({ center: { eles: targetNode }, zoom: cy.zoom(), duration: 500, easing: 'ease-in-out-cubic' });
});
</script>
</body>
</html>"""

# -------------------- Bundled Cytoscape.js asset --------------------
# Vendored Cytoscape.js library served from the web assets folder and
# inlined into generated mesh visualisation HTML so the graph renders in
# sandboxed iframes and offline environments. Resolved relative to this
# file (ndb_host/db/engine/utils -> ndb_host/web_dir).
CYTO_BUNDLE_PATH: Path = (
    Path(__file__).resolve().parents[3] / "web_dir" / "assets" / "js" / "cytoscape.min.js"
)

@dataclass(frozen=True)
class NebulonRenderOptions:
    show_labels: bool
    enable_animation: bool
    enable_hover: bool


RENDER_OPTIONS: dict[str, NebulonRenderOptions] = {
    "small":  NebulonRenderOptions(True,  True,  True),
    "medium": NebulonRenderOptions(True,  False, True),
    "large":  NebulonRenderOptions(False, False, False),
    "huge":   NebulonRenderOptions(False, False, False),
}
