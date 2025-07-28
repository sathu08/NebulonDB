from fastapi import Depends
from fastapi import APIRouter
from pathlib import Path
import os
import shutil

from db.index_manager import NebulonDBConfig, SegmentManager
from utils.models import StandardResponse, SegmentExistenceResponse, SegmentQueryRequest, AuthenticationResult, UserRole, save_data, load_data
from utils.logger import logger
from core.permissions import check_user_permission
from services.user_service import get_current_user
router = APIRouter()

VECTOR_DB_PATH = Path(NebulonDBConfig.VECTOR_STORAGE)
DATABASE_METADATA = Path(NebulonDBConfig.VECTOR_METADATA)

@router.post(
    "/create_segment",
    response_model=SegmentExistenceResponse,
    summary="create segment",
    description="Create New segment to Corpus"
)
async def create_segment(
    segment_query: SegmentQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> SegmentExistenceResponse:
    """
    Create a new corpus in the vector database.
    
    Args:
        corpus_query: Corpus creation details
        current_user: Current authenticated user
        
    Returns:
        CorpusExistenceResponse: Corpus creation result
    
    """
    try:
        corpus_name = segment_query.corpus_name
        segment_name = segment_query.segment_name
        logger.info(f"Attempting to create segment: {segment_name} from corpus: {corpus_name} for user: {current_user.username}")
        
        available_segment_list = SegmentManager(corpus_name).get_available_segment_list()
        if segment_name in available_segment_list.get("product_names"):
            logger.warning(f"Segment creation failed - Segment already exists: {segment_name}")
            return SegmentExistenceResponse(
                exists=True,
                corpus_name=corpus_name,
                segment_name=segment_name,
                message=f"Segment{segment_name} in'{corpus_name}' already exists"
            )
        
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.error(f"User '{current_user.username}' lacks required permission to create corpus")
            return SegmentExistenceResponse(
                exists=True, 
                corpus_name=corpus_name,
                segment_name=segment_name, 
                message="Permission denied to create segment."
            )
        corpus_path = VECTOR_DB_PATH / corpus_name
        
    
    except Exception as e:
        logger.exception(f"[CREATE] Failed to create Segment: {e}")
        return SegmentExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                segment_name=segment_name,
                message="Internal error while creating Segment.")
