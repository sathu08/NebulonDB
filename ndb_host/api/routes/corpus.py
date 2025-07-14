from fastapi import Depends
from fastapi import APIRouter
from pathlib import Path
import os
import shutil

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
        corpus_name = corpus_query.corpus_name
        logger.info(f"Attempting to create corpus: {corpus_name} for user: {current_user.username}")
        
        # Check if corpus already exists
        available_corpus_list = corpus_manager.get_available_corpus_list()
        if corpus_name in available_corpus_list:
            logger.warning(f"Corpus creation failed - corpus already exists: {corpus_name}")
            return CorpusExistenceResponse(
                exists=True,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' already exists in the database"
            )
        
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.error(f"User '{current_user.username}' lacks required permission to create corpus")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to create corpus."
            )

        # Create the corpus directory
        corpus_path = VECTOR_DB_PATH / corpus_name
        os.makedirs(corpus_path, exist_ok=True)
        for corpus_subdir in NebulonDBConfig.DEFAULT_CORPUS_STRUCTURES:
            (corpus_path / corpus_subdir).mkdir(parents=True, exist_ok=True)
        corpus_config_path = corpus_path / Path(NebulonDBConfig.DEFAULT_CORPUS_CONFIG_STRUCTURES)
        config_data = NebulonDBConfig.DEFAULT_CORPUS_CONFIG_DATA
        save_data(save_data=config_data, path_loc=corpus_path / corpus_config_path)

        # Store the corpus details
        created_corpus = load_data(path_loc=DATABASE_METADATA)
        created_corpus[corpus_name] = corpus_manager.generate_corpus_metadata(
            corpus_name=corpus_name,
            created_by=current_user.username
        )
        save_data(save_data=created_corpus, path_loc=DATABASE_METADATA)

        logger.info(f"Corpus created successfully: {corpus_name}")
        
        return CorpusExistenceResponse(
            exists=False,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' created successfully"
        )
    except Exception as e:
        logger.exception(f"[CREATE] Failed to create corpus: {e}")
        return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="Internal error while creating corpus.")

@router.get(
    "/list_corpus",
    response_model=StandardResponse,
    summary="List all available corpus",
    description="Retrieve a list of all available corpus in the vector database"
)
async def list_available_corpus(
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Lists all available corpora for the authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        CorpusExistenceResponse: Corpus List
    """
    try:
        logger.info(f"User '{current_user.username}' requested corpus listing.")
        available_corpus = corpus_manager.get_available_corpus_list()

        return StandardResponse(
            success=True,
            message=f"Retrieved {len(available_corpus)} corpus entries.",
            data={
                "corpus_list": available_corpus,
                "total_count": len(available_corpus)
            }
        )
    except Exception as e:
        logger.exception(f"[LIST] Error listing corpora: {e}")
        return StandardResponse(
            success=False, 
            message="Error listing corpora.", 
            data={}
            )


@router.post(
    "/delete_corpus",
    response_model=CorpusExistenceResponse,
    summary="Delete a specific corpus",
    description="Deletes a corpus from the vector database if permissions and conditions allow"
)
async def delete_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> CorpusExistenceResponse:
    """
    Delete a corpus in the vector database.
    
    Args:
        corpus_query: Corpus Delete details
        current_user: Current authenticated user
        
    Returns:
        CorpusExistenceResponse: Corpus Delete result
        
    """
    try:
        corpus_name = corpus_query.corpus_name
        logger.info(f"User '{current_user.username}' requested deletion of corpus '{corpus_name}'.")

        if corpus_name not in corpus_manager.get_available_corpus_list():
            logger.warning(f"Corpus '{corpus_name}' not found.")
            return CorpusExistenceResponse(
                exists=False,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' does not exist."
            )

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}' to delete corpus.")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to delete corpus."
            )

        corpus_status = corpus_manager.get_corpus_status(corpus_name=corpus_name)
        if corpus_status == "system":
            logger.info("System corpus cannot be deleted.")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="System corpus cannot be deleted."
            )
        elif corpus_status == "active":
            logger.info("Corpus must be deactivated before deletion.")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="Deactivate the corpus before deletion."
            )

        # Delete the corpus directory (even if it contains files)
        corpus_path = VECTOR_DB_PATH / corpus_name
        shutil.rmtree(corpus_path)

        corpus_info = load_data(path_loc=DATABASE_METADATA)
        del corpus_info[corpus_name]
        save_data(path_loc=DATABASE_METADATA, save_data=corpus_info)

        logger.info(f"Corpus '{corpus_name}' deleted successfully.")

        return CorpusExistenceResponse(
            exists=True,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' deleted successfully."
        )

    except Exception as e:
        logger.exception(f"[DELETE] Error deleting corpus: {e}")
        return CorpusExistenceResponse(
            exists=True, 
            corpus_name=corpus_name, 
            message="Internal error while deleting corpus."
        )

@router.post(
    "/deactivate_corpus",
    response_model=CorpusExistenceResponse,
    summary="Deactivate a specific corpus",
    description="Deactivate a corpus from the vector database if permissions and conditions allow"
)
async def deactivate_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> CorpusExistenceResponse:
    """
    Deactivate a corpus in the vector database.
    
    Args:
        corpus_query: Corpus Deactivate details
        current_user: Current authenticated user
        
    Returns:
        CorpusExistenceResponse: Corpus Deactivate result
        
    """
    try:
        corpus_name = corpus_query.corpus_name
        logger.info(f"User '{current_user.username}' requested deactivate of corpus '{corpus_name}'.")

        if corpus_name not in corpus_manager.get_available_corpus_list():
            logger.warning(f"Corpus '{corpus_name}' not found.")
            return CorpusExistenceResponse(
                exists=False,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' does not exist."
            )

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}' to deactivate corpus.")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to deactivate corpus."
            )

        corpus_status = corpus_manager.get_corpus_status(corpus_name=corpus_name)
        if corpus_status == "system":
            logger.info("System corpus cannot be deactivate.")
            return CorpusExistenceResponse(
                exists=True, 
                corpus_name=corpus_name, 
                message="System corpus cannot be deactivate."
            )
        corpus_manager.set_corpus_status(corpus_name=corpus_name, status="deactivate")

        return CorpusExistenceResponse(
            exists=True,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' deactivate successfully."
        )

    except Exception as e:
        logger.exception(f"[DELETE] Error deactivate corpus: {e}")
        return CorpusExistenceResponse(
            exists=True, 
            corpus_name=corpus_name, 
            message="Internal error while deactivate corpus."
        )
