from fastapi import Depends, HTTPException, status
from fastapi import APIRouter
from pathlib import Path
import os 

from db.index_manager import NebulonDBConfig, CorpusManager
from utils.models import StandardResponse, CorpusExistenceResponse, CorpusQueryRequest, AuthenticationResult, UserRole, save_data, load_data
from utils.logger import logger
from core.permissions import check_user_permission
from services.user_service import get_current_user
router = APIRouter()

VECTOR_DB_PATH = Path(NebulonDBConfig.VECTOR_STORAGE)
DATABASE_METADATA = Path(NebulonDBConfig.VECTOR_METADATA)
corpus_manager = CorpusManager()

@router.post(
    "/create_corpus",
    response_model=CorpusExistenceResponse,
    summary="create corpus",
    description="Create New corpus to Vector DataBase"
)
async def create_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> CorpusExistenceResponse:
    """
    Create a new corpus in the vector database.
    
    Args:
        corpus_query: Corpus creation details
        current_user: Current authenticated user
        
    Returns:
        CorpusExistenceResponse: Corpus creation result
        
    Raises:
        HTTPException: If corpus creation fails
    """
    try:
        logger.info(f"Attempting to create corpus: {corpus_query.corpus_name} for user: {current_user.username}")
        
        # Check if corpus already exists
        available_corpus_list = corpus_manager.get_available_corpus_list()
        if corpus_query.corpus_name in available_corpus_list:
            logger.warning(f"Corpus creation failed - corpus already exists: {corpus_query.corpus_name}")
            return CorpusExistenceResponse(
                exists=True,
                corpus_name=corpus_query.corpus_name,
                message=f"Corpus '{corpus_query.corpus_name}' already exists in the database"
            )
        
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.error(f"User '{current_user.username}' lacks required permission to create corpus")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create corpus directory"
            )
        
        # Create the corpus directory
        corpus_path = VECTOR_DB_PATH / corpus_query.corpus_name
        os.makedirs(corpus_path, exist_ok=True)
        for corpus_subdir in NebulonDBConfig.DEFAULT_CORPUS_STRUCTURES:
            (corpus_path / corpus_subdir).mkdir(parents=True, exist_ok=True)
        corpus_config_path = corpus_path / Path(NebulonDBConfig.DEFAULT_CORPUS_CONFIG_STRUCTURES)
        config_data = NebulonDBConfig.DEFAULT_CORPUS_CONFIG_DATA
        save_data(save_data=config_data, path_loc=corpus_path / corpus_config_path)

        # Store the corpus details
        created_corpus = load_data(path_loc=DATABASE_METADATA)
        created_corpus[corpus_query.corpus_name] = corpus_manager.generate_corpus_metadata(
            corpus_name=corpus_query.corpus_name,
            created_by=current_user.username
        )
        save_data(save_data=created_corpus, path_loc=DATABASE_METADATA)

        logger.info(f"Corpus created successfully: {corpus_query.corpus_name}")
        
        return CorpusExistenceResponse(
            exists=False,
            corpus_name=corpus_query.corpus_name,
            message=f"Corpus '{corpus_query.corpus_name}' created successfully"
        )
        
    except OSError as e:
        logger.error(f"OS error creating corpus: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create corpus directory"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating corpus: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating corpus"
        )

@router.get(
    "/list_corpus",
    response_model=StandardResponse,
    summary="List all available corpus",
    description="Retrieve a list of all available corpus in the vector database"
)
async def list_available_corpus(
    current_user: AuthenticationResult = Depends(get_current_user)
    )-> StandardResponse:
    """
    List all available corpus in the vector database.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: List of available corpus
        
    Raises:
        HTTPException: If corpus listing fails
    """
    try:
        logger.info(f"Listing available corpus for user: {current_user.username}")
        available_corpus_list = corpus_manager.get_available_corpus_list()
        
        return StandardResponse(
            success=True,
            message=f"Successfully retrieved {len(available_corpus_list)} available corpus",
            data={
                "corpus_list": available_corpus_list,
                "total_count": len(available_corpus_list)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing available corpus: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing corpus"
        )
