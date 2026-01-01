from fastapi import  Depends
from fastapi import APIRouter

import numpy as np
import polars as pl
from pathlib import Path

from core.permissions import check_user_permission
from services.user_service import get_current_user

from db.index_manager import SegmentManager
from ndb_host.db.ndb_settings import NDBConfig
from utils.models import (SegmentQueryRequest, AuthenticationResult, ColumnPick,
                          StandardResponse, UserRole, SemanticEmbeddingModel, CorpusQueryRequest)
from utils.logger import NebulonDBLogger


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

# ==========================================================
#        Vector DB and Metadata Paths
# ==========================================================

VECTOR_DB_PATH = Path(config_settings.VECTOR_STORAGE)
DATABASE_METADATA = Path(config_settings.VECTOR_METADATA)

# ==========================================================
#        API Endpoints for Segment Management
# ==========================================================

@router.post(
    "/list_segments",
    response_model=StandardResponse,
    summary="List segments in a corpus",
    description="Retrieve a list of all segments available in a specific corpus"
)
async def list_segments_in_corpus(
    segment_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Lists all segments available in a specific corpus.

    Args:
        segment_query: Contains corpus_name to query
        current_user: Current authenticated user

    Returns:
        StandardResponse: List of segments in the corpus
    """
    try:
        corpus_name = segment_query.corpus_name
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                message=current_user.message,
                data={
                    "segment_list": [],
                    "total_count": 0
                }
            )

        logger.info(
            f"User '{current_user.username}' requested segments for corpus '{corpus_name}'."
        )

        segment_manager = SegmentManager(corpus_name=corpus_name)
        segment_list = segment_manager.get_segment_list()

        return StandardResponse(
            success=True,
            message=f"Retrieved {len(segment_list)} segments from corpus '{corpus_name}'.",
            data={
                "segment_list": segment_list,
                "total_count": len(segment_list)
            }
        )

    except Exception as e:
        logger.exception(f"[LIST] Error listing segments: {e}")
        return StandardResponse(
            success=False,
            message=f"Error listing segments for corpus '{segment_query.corpus_name}'.",
            data={
                "segment_list": [],
                "total_count": 0
            }
        )


@router.post(
    "/list_segment_ids",
    response_model=StandardResponse,
    summary="Get segment IDs",
    description="Retrieve segment IDs for a specific segment in a corpus"
)
async def list_available_segment_id(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Retrieves segment IDs for a specific segment.

    Args:
        segment_query: Contains corpus_name and segment_name to query
        current_user: Current authenticated user

    Returns:
        StandardResponse: Segment IDs information
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name

        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                message=current_user.message,
                data={
                    "segment_ids": [],
                    "total_count": 0
                }
            )

        logger.info(
            f"User '{current_user.username}' requested IDs for segment '{segment_name}' in corpus '{corpus_name}'."
        )

        segment_manager = SegmentManager(corpus_name=corpus_name)
        segment_ids = segment_manager.get_segment_id_list(segment_name=segment_name)

        return StandardResponse(
            success=True,
            message=f"Retrieved {len(segment_ids)} IDs for segment '{segment_name}'.",
            data={
                "segment_ids": segment_ids,
                "total_count": len(segment_ids)
            }
        )

    except Exception as e:
        logger.exception(f"[SEGMENT IDS] Error getting segment IDs: {e}")
        return StandardResponse(
            success=False,
            message=f"Error retrieving IDs for segment '{segment_query.segment_name}'.",
            data={
                "segment_ids": [],
                "total_count": 0
            }
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
        segment_dataset = segment_query.segment_dataset
        set_columns = segment_query.set_columns if segment_query.set_columns else ColumnPick.FIRST_COLUMN


        
        # Check authentication first
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                message=current_user.message
            )

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
                message="Permission denied"
            )
        
        segment_manager = SegmentManager(corpus_name=corpus_name)
        
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

        is_precomputed = segment_query.is_precomputed

        # Process each column
        total_inserted = 0
        total_skipped = 0
        errors = []
        columns = segment_manager.determine_columns_to_process(segment_dataset=segment_dataset, set_columns=set_columns)
        if not columns["success"]:
            return columns["message"]
        for col in columns.get("columns",""):
            # If standard flow, check if column exists (polars checks this but safe to double check)
            if col not in segment_dataset.columns:
                errors.append(f"Column '{col}' not found in dataset")
                continue
                 
            try:
                if is_precomputed:
                    # Treat column data as lists of floats (vectors)
                    # Convert to numpy array of vectors
                    vectors_list = segment_dataset[col].to_list()
                    if not vectors_list:
                        logger.warning(f"No vectors found in column '{col}'")
                        continue
                         
                    embeddings = np.array(vectors_list, dtype="float32")
                    # Since it's precomputed, we don't have text. Use empty string or placeholder.
                    texts = [""] * len(embeddings)
                     
                    # No filter for empty text in precomputed mode
                    valid_data_with_idx = [(i, t, v) for i, (t, v) in enumerate(zip(texts, embeddings))]
                     
                else:
                    # Standard Flow: Text -> Model -> Vector
                    texts = segment_dataset[col].fill_null("").to_list()
                    if not any(texts):
                        errors.append(f"Column '{col}' has no valid text")
                        continue
                        
                    # Batch Encode
                    embeddings = SemanticEmbeddingModel().encode(
                        texts, 
                        convert_to_numpy=True,
                        normalize_embeddings=True 
                    ).astype("float32")

                    # Filter valid entries (non-empty texts)
                    valid_data_with_idx = [(i, t, v) for i, (t, v) in enumerate(zip(texts, embeddings)) if t.strip()]
                
                if not valid_data_with_idx:
                    logger.warning(f"No valid data found in column '{col}'")
                    continue
                
                indices, valid_texts, valid_vectors = zip(*valid_data_with_idx)
                valid_vectors = np.array(valid_vectors)
                
                # Batch generate ID keys
                keys = segment_manager.get_next_vector_ids(col, len(valid_texts))
                
                # Batch create payloads
                payloads = [{col: txt, "row_index": original_idx} for txt, original_idx in zip(valid_texts, indices)]
                
                logger.info(f"Batch loading {len(keys)} items for column '{col}'")
                
                # Batch Insert
                result = segment_manager.load_segment_batch(
                    segment_name=segment_name,
                    keys=keys,
                    vectors=valid_vectors,
                    payloads=payloads
                )
                
                logger.info(f"Batch result for '{col}': {result}")
                if result["success"]:
                    total_inserted += result.get("inserted", 0)
                    total_skipped += result.get("skipped", 0)
                else:
                    errors.append(f"Failed to process column '{col}': {result['message']}")
                            
            except Exception as e:
                errors.append(f"Failed to process column '{col}': {str(e)}")
          
        logger.info(f"Successfully segment loaded into corpus '{corpus_name}'")
        
        return StandardResponse(
            success=True,
            corpus_name=corpus_name,
            segment_name=segment_name,
            errors=errors,
            message=f"Processed {total_inserted} vectors, skipped {total_skipped}"
        )

    except Exception as e:
        logger.exception(f"Failed to load segment into corpus '{segment_query.corpus_name}': {str(e)}")
        return StandardResponse(
            success=False,
            corpus_name=corpus_name,
            message=f"Internal server error while creating segment: {str(e)}"
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
                       segment_dataset, and set_column_vector.
        current_user: Authenticated user making the request.

    Returns:
        StandardResponse: Result of the segment search attempt.
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        search_item = segment_query.search_item
        set_columns = segment_query.set_columns if segment_query.set_columns else ColumnPick.ALL
        top_matches = segment_query.top_matches if segment_query.top_matches else None
        
        segment_manager = SegmentManager(corpus_name=corpus_name)
        
        # Check authentication first
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                message=current_user.message
            )

        segment_list = segment_manager.get_segment_list()
        if not segment_name in segment_list:
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message=f"Segment '{segment_name}' does not exist in corpus '{corpus_name}'"
            )
        
        if not search_item or not search_item.strip():
            return StandardResponse(
                success=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message="search_item must not be empty"
            )
            
        logger.info(
            f"User '{current_user.username}' is attempting to load a segment into corpus '{corpus_name}'"
        )
        
        # Encode query into embedding
        query_vec = SemanticEmbeddingModel().encode(
            search_item,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        
        results = segment_manager.search_vector(
            segment_name=segment_name,
            query_vec=query_vec,
            set_columns=set_columns,
            top_k=top_matches
        )
        
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