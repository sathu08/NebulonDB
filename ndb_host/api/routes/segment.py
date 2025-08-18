from fastapi import APIRouter, Depends
from pathlib import Path

from db.index_manager import NebulonDBConfig, SegmentManager
from utils.models import SegmentExistenceResponse, SegmentQueryRequest, AuthenticationResult, UserRole
from utils.logger import logger
from core.permissions import check_user_permission
from services.user_service import get_current_user

router = APIRouter()

VECTOR_DB_PATH = Path(NebulonDBConfig.VECTOR_STORAGE)
DATABASE_METADATA = Path(NebulonDBConfig.VECTOR_METADATA)

@router.post(
    "/create_segment",
    response_model=SegmentExistenceResponse,
    summary="Create segment",
    description="Create new segment in a corpus"
)
async def create_segment(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> SegmentExistenceResponse:
    """
    Create a new segment in the specified corpus.
    
    Args:
        segment_query: Segment creation details including corpus_name, segment_name, and category
        current_user: Authenticated user making the request
        
    Returns:
        SegmentExistenceResponse: Result of segment creation attempt
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        category_name = segment_query.category
        # Check authentication first
        if not current_user.is_authenticated:
            return SegmentExistenceResponse(
                exists=False,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message=current_user.message
            )

        logger.info(f"Attempting to create segment '{segment_name}' in corpus '{corpus_name}' for user '{current_user.username}'")

        # Check permissions
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}'")
            return SegmentExistenceResponse(
                exists=True,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message="Permission denied"
            )

        # Check if segment exists
        segment_manager = SegmentManager(corpus_name)
        available_segments = segment_manager.get_available_segment_list()
        
        if available_segments and segment_name in available_segments.get("product_names", []):
            logger.warning(f"Segment '{segment_name}' already exists in corpus '{corpus_name}'")
            return SegmentExistenceResponse(
                exists=True,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message=f"Segment '{segment_name}' already exists in corpus '{corpus_name}'"
            )

        # Create the segment
        segment_manager.create_segment(segment_label=segment_name, category=category_name)
        logger.info(f"Successfully created segment '{segment_name}' in corpus '{corpus_name}'")
        
        return SegmentExistenceResponse(
            exists=False,
            corpus_name=corpus_name,
            segment_name=segment_name,
            message=f"Segment '{segment_name}' created successfully in corpus '{corpus_name}'"
        )

    except Exception as e:
        logger.exception(f"Failed to create segment '{segment_name}': {str(e)}")
        return SegmentExistenceResponse(
            exists=True,
            corpus_name=corpus_name,
            segment_name=segment_name,
            message=f"Internal server error while creating segment: {str(e)}"
        )