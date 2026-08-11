"""
NDB API Segment Management
==========================================================

This module handles segment management for the NDB API.
It provides endpoints for segment creation, listing, and deletion.

"""

from fastapi import  Depends
from fastapi import APIRouter

import polars as pl

from core.permissions import check_user_permission
from services.user_service import get_current_user

from utils.logger import NebulonDBLogger
from db.index_manager import CorpusManager
from db.index_manager import SegmentManager

from ndb_host.db.ndb_settings import NDBConfig
from utils.constants import ColumnPick ,NDBMeta
from utils.models import SegmentQueryRequest, SegmentQueryRequest, AuthenticationResult, StandardResponse, UserRole

from db.engine.utils import FIELD_NOVA

# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger("audit")

# ==========================================================
#        API Router for Segment Management
# ==========================================================

router = APIRouter()

# ==========================================================
#        Load Configuration
# ==========================================================

config_settings = NDBConfig()

def _unauth_response(segment_query: SegmentQueryRequest) -> StandardResponse:
    return StandardResponse(
        success=False,
        corpus_name=segment_query.corpus_name,
        segment_name=segment_query.segment_name,
        message="Authentication failed"
    )

def _resolve_corpus_ndb_type(corpus_name: str, fallback: str = NDBMeta.Type.COSMOS.value) -> str:
    """
    Return the real storage type of a corpus from its metadata.

    Many route handlers default ``ndb_type`` to ORBIT, which forces
    construction of the full Orbit engine (and its ``NebulonOrbit`` folder
    tree) even for cosmos corpora. Corpus metadata is authoritative, so it is
    resolved here instead of trusting the request default.
    """
    try:
        info = CorpusManager().get_corpus_info()
        record = info.get(corpus_name) or {}
        if record.get("ndb_type"):
            return record["ndb_type"]
    except Exception:
        pass
    return fallback

def _build_orbit(segment_query: SegmentQueryRequest) -> "SegmentManager":
    return SegmentManager(
        corpus_name=segment_query.corpus_name,
        segment_name=segment_query.segment_name,
        ndb_type=_resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type),
    ).db_manager

# ==========================================================
#        API Endpoints for Segment Management
# ==========================================================

@router.get(
    "/list_segment",
    response_model=StandardResponse,
    summary="List segments in a corpus",
    description="Retrieve the segments registered for a corpus"
)
async def list_segment(
    corpus_name: str,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    List all segments registered for a corpus.

    Args:
        corpus_name: Name of the corpus.
        current_user: Authenticated user making the request.

    Returns:
        StandardResponse: Segment list and count.
    """
    try:
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                message=current_user.message
            )
        if not corpus_name or not corpus_name.strip():
            return StandardResponse(
                success=False,
                message="corpus_name must not be empty"
            )

        segment_manager = SegmentManager(
            corpus_name=corpus_name,
            segment_name="default",
            ndb_type=_resolve_corpus_ndb_type(corpus_name),
        )
        segments = segment_manager.get_segment_metadata()

        return StandardResponse(
            success=True,
            corpus_name=corpus_name,
            data={
                "segment_list": segments,
                "total_count": len(segments),
            },
            message=f"Found {len(segments)} segments in corpus '{corpus_name}'"
        )
    except Exception as e:
        logger.exception(f"Failed to list segments for corpus '{corpus_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=corpus_name,
            message=f"Internal server error while listing segments: {str(e)}"
        )

@router.post(
    "/load_segment",
    response_model=StandardResponse,
    summary="Load segment",
    description="Load new segment in a corpus"
)
async def load_segment(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Load a new segment into the specified corpus.

    Args:
        segment_query: Segment creation details including corpus_name,
                       segment_dataset, and set_column_vector.
        current_user: Authenticated user making the request.

    Returns:
        StandardResponse: Result of the segment load attempt.
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        ndb_type = _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type)
        doc_type = segment_query.doc_type or None
        lang_type = segment_query.lang_type or None
        segment_dataset = segment_query.segment_dataset
        set_columns = segment_query.set_columns or ColumnPick.FIRST_COLUMN
        is_precomputed = segment_query.is_precomputed
        relations = segment_query.relations
        source_column = segment_query.source_column
        target_column = segment_query.target_column
        relation_column = segment_query.relation_column
        
        # Check authentication first
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)

        logger.info(
            f"User '{current_user.username}' is attempting to load a segment into corpus '{corpus_name}'"
        )
        
        # Check permissions
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(
                f"Permission denied: user '{current_user.username}' attempted to load a segment into corpus '{corpus_name}'"
            )
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message="Permission denied"
            )
        
        segment_manager = SegmentManager(corpus_name=corpus_name, segment_name=segment_name, ndb_type=ndb_type)
        
        # Validate the Dataset 
        if segment_dataset is None:
            return StandardResponse(success=False, message="Dataset cannot be None")
        
        elif isinstance(segment_dataset, pl.DataFrame):
            segment_dataset = segment_dataset

        elif isinstance(segment_dataset, dict):
            try:
                segment_dataset = pl.DataFrame(segment_dataset)
            except Exception as e:
                return StandardResponse(
                    success=False,
                    message=f"Failed to convert dict to DataFrame: {str(e)}"
                )

        elif isinstance(segment_dataset, list):
            try:
                segment_dataset = pl.DataFrame(segment_dataset)
            except Exception as e:
                return StandardResponse(
                    success=False,
                    message=f"Failed to convert list of dicts to DataFrame: {str(e)}"
                )

        else:
            return StandardResponse(
                success=False,
                message=f"Unsupported dataset type: {type(segment_dataset)}"
            )

        # Final check
        if not isinstance(segment_dataset, pl.DataFrame) or segment_dataset.height == 0:
            return StandardResponse(success=False, message="Invalid or empty dataset")

        columns = segment_manager.determine_columns_to_process(segment_dataset=segment_dataset, set_columns=set_columns)
        if not columns["success"]:
            return columns["message"]
        
        result = segment_manager.load_segment(
        segment_dataset=segment_dataset,
        columns=columns["columns"],
        doc_type=doc_type,
        lang_type=lang_type,
        is_precomputed=is_precomputed,
        relations=relations,
        source_column=source_column,
        target_column=target_column,
        relation_column=relation_column,
        )

        if not result["success"]:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                errors=result.get("errors", []),
                message=(
                    "; ".join(result.get("errors", []))
                    or f"Segment load failed for corpus '{corpus_name}'"
                ),
            )
        
        logger.info(f"Successfully segment loaded into corpus '{corpus_name}'")
        
        return StandardResponse(
            success=True,
            corpus_name=corpus_name,
            segment_name=segment_name,
            errors=result.get("errors", []),
            message=f"Processed {result.get('inserted', 0)} vectors, skipped {result.get('skipped', 0)} vectors"
        )

    except Exception as e:
        logger.exception(f"Failed to load segment into corpus '{segment_query.corpus_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=corpus_name,
            segment_name=segment_name,
            message=f"Internal server error while creating segment: {str(e)}"
        )


@router.post(
    "/get_data",
    response_model=StandardResponse,
    summary="Get segment data",
    description="Retrieve stored records from a segment with an optional row limit"
)
async def get_data(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Retrieve stored records from the segment's backend (Orbit or Cosmos).

    Args:
        segment_query: Includes corpus_name, segment_name, ndb_type, limit.
        current_user: Authenticated user making the request.

    Returns:
        StandardResponse: Retrieved records and count.
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        ndb_type = _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type)
        limit = segment_query.limit

        if not current_user.is_authenticated:
            return _unauth_response(segment_query)

        segment_manager = SegmentManager(
            corpus_name=corpus_name,
            segment_name=segment_name,
            ndb_type=ndb_type,
        )
        records = segment_manager.get_data(limit=limit)

        return StandardResponse(
            success=True,
            corpus_name=corpus_name,
            segment_name=segment_name,
            data={
                "records": records,
                "total_count": len(records),
                "limit": limit,
            },
            message=f"Retrieved {len(records)} records from segment '{segment_name}'"
        )
    except Exception as e:
        logger.exception(f"Failed to get data for segment '{segment_query.segment_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Internal server error while getting data: {str(e)}"
        )

@router.post(
    "/search_segment",
    response_model=StandardResponse,
    summary="search segment",
    description="Search within a segment in a corpus"
)
async def search_segment(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    search a segment into the specified corpus.

    Args:
        segment_query: Segment creation details including corpus_name,
                       segment_name, segment_dataset, and set_column_vector.
        current_user: Authenticated user making the request.

    Returns:
        StandardResponse: Result of the segment search attempt.
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        search_item = segment_query.search_item
        doc_type = segment_query.doc_type
        lang_type = segment_query.lang_type
        set_columns = segment_query.set_columns or ColumnPick.ALL
        top_matches = segment_query.top_matches
        min_score = segment_query.min_score
        rank = segment_query.rank
        mode = segment_query.mode or "auto"
        graph_start_node = segment_query.graph_start_node
        expand_depth = segment_query.expand_depth or 1
        graph_boost = segment_query.graph_boost or 0.1
        ndb_type = _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type)

        segment_manager = SegmentManager(corpus_name=corpus_name, segment_name=segment_name, ndb_type=ndb_type)

        if ndb_type == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message="Search is not supported for Cosmos segments"
            )
        
        # Check authentication first
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if not search_item or not search_item.strip():
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message="search_item must not be empty"
            )
            
        logger.info(
            f"User '{current_user.username}' is attempting to search in corpus '{corpus_name}'"
        )
        vector_results = segment_manager.search_vector(
            search_item=search_item,
            top_k=(top_matches or 10) * 2,
            set_columns=set_columns,
            min_score=min_score,
            lang_type=lang_type,
            doc_type=doc_type,
            mode=mode,
            rank=rank,
            graph_start_node=graph_start_node,
            expand_depth=expand_depth,
            graph_boost=graph_boost,
        )
        if not vector_results:
            return StandardResponse(
                success=True,
                corpus_name=corpus_name,
                segment_name=segment_name,
                data=[],
                message="No results found"
            )

        results = vector_results[:top_matches] if top_matches else vector_results

        # Strip full vectors from the wire payload; clients get id/score/metadata.
        # Use get_record() when the raw vector is needed.
        for r in results:
            if isinstance(r, dict):
                r.pop(FIELD_NOVA, None)

        return StandardResponse(
            success=True,
            corpus_name=corpus_name,
            segment_name=segment_name,
            data=results,
            message=f"Found {len(results)} results"
        )
        
    except Exception as e:
        logger.exception(f"Failed to load segment into corpus '{segment_query.corpus_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=corpus_name,
            message=f"Internal server error while creating segment: {str(e)}"
        )


@router.post(
    "/delete_record",
    response_model=StandardResponse,
    summary="Delete record",
    description="Delete a vector record (and its graph node) by record_id"
)
async def delete_record(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Delete a record by ID."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.record_id is None:
            return StandardResponse(success=False, message="record_id is required")
        
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(
                f"Permission denied: user '{current_user.username}' attempted to load a segment into corpus '{segment_query.corpus_name}'"
            )
            return StandardResponse(success=False, message="Permission denied")

        db = _build_orbit(segment_query)

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            deleted = db.delete_data(
                segment=segment_query.segment_name,
                record_id=segment_query.record_id,
            )
        else:
            deleted = db.delete_record(segment_query.record_id)
            if deleted:
                db.initialize_or_flush()

        if not deleted:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message=f"Record {segment_query.record_id} not found"
            )
        return StandardResponse(
            success=True,
            exists=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Record {segment_query.record_id} deleted"
        )
    except Exception as e:
        logger.exception(f"Failed to delete record {segment_query.record_id}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")



# ==========================================================
#        Vector / Graph Inspection & Manipulation Endpoints
# ==========================================================


@router.post(
    "/segment_stats",
    response_model=StandardResponse,
    summary="Segment statistics",
    description="Vector + graph statistics (counts, dimension, deleted ratio) for a segment"
)
async def segment_stats(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Return vector and graph statistics for a segment."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Search is not supported for Cosmos segments")
        
        stats = _build_orbit(segment_query).stats()

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data=stats,
            message=f"Retrieved stats for segment '{segment_query.segment_name}'"
        )
    except Exception as e:
        logger.exception(f"Failed to get stats for segment '{segment_query.segment_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Internal server error while getting stats: {str(e)}"
        )


@router.post(
    "/get_record",
    response_model=StandardResponse,
    summary="Get record",
    description="Retrieve a full record (id, vector, metadata) by record_id"
)
async def get_record(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Retrieve a single record by ID."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.record_id is None:
            return StandardResponse(success=False, message="record_id is required")
        
        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Search is not supported for Cosmos segments")
        
        orbit = _build_orbit(segment_query)
        record = orbit.get_record(segment_query.record_id)

        if record is None:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message=f"Record {segment_query.record_id} not found"
            )
        return StandardResponse(
            success=True,
            exists=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data=record,
            message=f"Record {segment_query.record_id} found"
        )
    except Exception as e:
        logger.exception(f"Failed to get record {segment_query.record_id}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/get_neighbors",
    response_model=StandardResponse,
    summary="Graph neighbors",
    description="Return neighbors of a graph node with direction filter (in/out/both)"
)
async def get_neighbors(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Return graph neighbors of a node."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.node_id is None:
            return StandardResponse(success=False, message="node_id is required")

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="get neighbors is not supported for Cosmos segments")
        
        neighbors = _build_orbit(segment_query).get_neighbors(segment_query.node_id, segment_query.direction or "both")

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data=[{"node_id": nid, "relation": rel} for nid, rel in neighbors],
            message=f"Found {len(neighbors)} neighbors for node {segment_query.node_id}"
        )
    except Exception as e:
        logger.exception(f"Failed to get neighbors for node {segment_query.node_id}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/bfs",
    response_model=StandardResponse,
    summary="Graph BFS",
    description="Breadth-first traversal from a start node within max_depth"
)
async def bfs(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Run a BFS traversal from a start node."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.start_node is None:
            return StandardResponse(success=False, message="start_node is required")
        
        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="BFS is not supported for Cosmos segments")
        
        nodes = _build_orbit(segment_query).bfs(segment_query.start_node, segment_query.max_depth or 3)

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data={"nodes": nodes, "node_ids": nodes, "total_count": len(nodes)},
            message=f"BFS from {segment_query.start_node} reached {len(nodes)} nodes"
        )
    
    except Exception as e:
        logger.exception(f"Failed to run BFS from {segment_query.start_node}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/shortest_path",
    response_model=StandardResponse,
    summary="Graph shortest path",
    description="Shortest path between two graph nodes"
)
async def shortest_path(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Compute the shortest path between two nodes."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.source is None or segment_query.target is None:
            return StandardResponse(success=False, message="source and target are required")

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Shortest path is not supported for Cosmos segments")
        
        path = _build_orbit(segment_query).shortest_path(segment_query.source, segment_query.target)

        if path is None:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                data={"path": None},
                message=f"No path between {segment_query.source} and {segment_query.target}"
            )
        
        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data={"path": path, "path_length": len(path)},
            message=f"Path between {segment_query.source} and {segment_query.target} found"
        )
    except Exception as e:
        logger.exception(f"Failed to find path {segment_query.source}->{segment_query.target}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/add_node",
    response_model=StandardResponse,
    summary="Add graph node",
    description="Explicitly create a graph node (no vector required)"
)
async def add_node(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Create a graph node explicitly."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.node_id is None:
            return StandardResponse(success=False, message="node_id is required")
        
        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Add node is not supported for Cosmos segments")
                
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(
                f"Permission denied: user '{current_user.username}' attempted to load a segment into corpus '{segment_query.corpus_name}'"
            )
            return StandardResponse(success=False, message="Permission denied")
        
        orbit = _build_orbit(segment_query)

        label = (segment_query.metadata or {}).get("label") if segment_query.metadata else None
        orbit.add_node(segment_query.node_id, label)

        orbit.initialize_or_flush()

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Node {segment_query.node_id} created"
        )
    except Exception as e:
        logger.exception(f"Failed to add node {segment_query.node_id}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/add_relation",
    response_model=StandardResponse,
    summary="Add graph relation",
    description="Add a directed edge source -> target with a relation label"
)
async def add_relation(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Add a directed relationship between two graph nodes."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)

        if segment_query.source is None or segment_query.target is None or not segment_query.relation:
            return StandardResponse(success=False, message="source, target and relation are required")

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Add relation is not supported for Cosmos segments")
                
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):

            logger.warning(
                f"Permission denied: user '{current_user.username}' attempted to add a relation to segment '{segment_query.segment_name}' in corpus '{segment_query.corpus_name}'"
            )

            return StandardResponse(success=False, message="Permission denied")

        orbit = _build_orbit(segment_query)

        orbit.add_relation(segment_query.source, segment_query.target, segment_query.relation)

        orbit.initialize_or_flush()

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Relation {segment_query.source} -> {segment_query.target} added"
        )
    except Exception as e:
        logger.exception(f"Failed to add relation {segment_query.source}->{segment_query.target}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/remove_relation",
    response_model=StandardResponse,
    summary="Remove graph relation",
    description="Remove a directed edge source -> target (optionally by relation label)"
)
async def remove_relation(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Remove a directed relationship between two graph nodes."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)
        
        if segment_query.source is None or segment_query.target is None:
            return StandardResponse(success=False, message="source and target are required")

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="Remove relation is not supported for Cosmos segments")
        
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(
                f"Permission denied: user '{current_user.username}' attempted to add a relation to segment '{segment_query.segment_name}' in corpus '{segment_query.corpus_name}'"
            )
            return StandardResponse(success=False, message="Permission denied")
        
        orbit = _build_orbit(segment_query)

        orbit.remove_relation(segment_query.source, segment_query.target, segment_query.relation)

        orbit.initialize_or_flush()

        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            message=f"Relation {segment_query.source} -> {segment_query.target} removed"
        )
    except Exception as e:
        logger.exception(f"Failed to remove relation {segment_query.source}->{segment_query.target}: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")

@router.post(
    "/mesh_load_graph",
    response_model=StandardResponse,
    summary="Bulk mesh graph load",
    description="Add graph nodes and/or edges in bulk. Ref by id (int) or node label (string); weights auto-computed from endpoint vectors when omitted."
)
async def mesh_load_graph(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Bulk load nodes + edges (Option A)."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="load_graph is not supported for Cosmos segments")

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            return StandardResponse(success=False, message="Permission denied")

        segment_manager = SegmentManager(
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            ndb_type=segment_query.ndb_type,
        )
        result = segment_manager.load_graph(
            nodes=segment_query.nodes,
            edges=segment_query.edges,
        )
        return StandardResponse(
            success=result.get("success", True),
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data={k: v for k, v in result.items() if k != "success"},
            message=f"Graph loaded: {result.get('nodes_added', 0)} nodes, {result.get('edges_added', 0)} edges"
        )
    except Exception as e:
        logger.exception(f"Failed to load graph: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")


@router.post(
    "/mesh_visualization",
    response_model=StandardResponse,
    summary="Mesh Visualization",
    description="Visualize the mesh structure of a segment's graph"
)
async def mesh_visualization(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """Return an HTML visualization of the mesh structure."""
    try:
        if not current_user.is_authenticated:
            return _unauth_response(segment_query)

        if _resolve_corpus_ndb_type(segment_query.corpus_name, segment_query.ndb_type) == NDBMeta.Type.COSMOS:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message="mesh visualization is not supported for Cosmos segments")
        
        error_msg, html_path = _build_orbit(segment_query).get_visualization_html()

        if error_msg:
            return StandardResponse(
                success=False,
                corpus_name=segment_query.corpus_name,
                segment_name=segment_query.segment_name,
                message=error_msg
            )
        return StandardResponse(
            success=True,
            corpus_name=segment_query.corpus_name,
            segment_name=segment_query.segment_name,
            data={"html_path": str(html_path)},
            message="Mesh visualization HTML generated"
        )
    except Exception as e:
        logger.exception(f"Failed to generate mesh visualization: {str(e)}")
        return StandardResponse(success=False, message=f"Internal server error: {str(e)}")