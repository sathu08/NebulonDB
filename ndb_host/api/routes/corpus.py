from fastapi import Depends
from fastapi import APIRouter

from core.permissions import check_user_permission
from services.user_service import get_current_user

from db.index_manager import CorpusManager
from ndb_host.db.ndb_settings import NDBConfig
from utils.models import StandardResponse, CorpusQueryRequest, AuthenticationResult, UserRole
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger("audit")

# ==========================================================
#        API Router for Corpus Management
# ==========================================================

router = APIRouter()

# ==========================================================
#        Load Configuration
# ==========================================================

config_settings = NDBConfig()

# ==========================================================
#        Initialize Corpus Manager
# ==========================================================

corpus_manager = CorpusManager()

# ==========================================================
#        API Endpoints for Corpus Management
# ==========================================================

@router.post(
    "/create_corpus",
    response_model=StandardResponse,
    summary="Create corpus",
    description="Create New corpus to Vector DataBase"
)
async def create_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Create a new corpus in the vector database.
    
    Args:
        corpus_query: Corpus creation details
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Corpus creation result
    
    """
    try:
        corpus_name = corpus_query.corpus_name
        # Check authentication first
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=current_user.message
            )
        logger.info(f"Attempting to create corpus: {corpus_name} for user: {current_user.username}")
        
        # Check if corpus already exists
        available_corpus_list = corpus_manager.get_available_corpus_list()
        if corpus_name in available_corpus_list:
            logger.warning(f"Corpus creation failed - corpus already exists: {corpus_name}")
            return StandardResponse(
                success=False,
                exists=True,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' already exists in the database"
            )
        
        # Check permissions
        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.error(f"User '{current_user.username}' lacks required permission to create corpus")
            return StandardResponse(
                success=False,
                exists=False, 
                corpus_name=corpus_name, 
                message="Permission denied to create corpus."
            )

        # Create the corpus directory
        corpus_manager.create_corpus(corpus_name=corpus_name,username=current_user.username)
        logger.info(f"Corpus created successfully: {corpus_name}")
        
        return StandardResponse(
            success=True,
            exists=False,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' created successfully"
        )
    except Exception as e:
        logger.exception(f"[CREATE] Failed to create corpus: {e}")
        return StandardResponse(
                success=False,
                exists=False, 
                corpus_name=corpus_name, 
                message="Internal error while creating corpus."
            )

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
    Lists all available corpus for the authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        StandardResponse: Corpus List
    """
    try:
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                corpus_name=None,
                message=current_user.message
            )
        logger.info(f"User '{current_user.username}' requested corpus listing.")
        
        available_corpus = corpus_manager.get_available_corpus_list()

        return StandardResponse(
            success=True,
            exists=True,
            message=f"Retrieved {len(available_corpus)} corpus entries.",
            data={
                "corpus_list": available_corpus,
                "total_count": len(available_corpus)
            }
        )
    except Exception as e:
        logger.exception(f"[LIST] Error listing corpus: {e}")
        return StandardResponse(
            success=False, 
            message="Error listing corpus.", 
            data={
                "corpus_list": [],
                "total_count": 0
            }   
        )

@router.post(
    "/delete_corpus",
    response_model=StandardResponse,
    summary="Delete a specific corpus",
    description="Deletes a corpus from the vector database if permissions and conditions allow"
)
async def delete_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Delete a corpus in the vector database.
    
    Args:
        corpus_query: Corpus Delete details
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Corpus Delete result
        
    """
    try:
        corpus_name = corpus_query.corpus_name
        
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=current_user.message
            )
        logger.info(f"User '{current_user.username}' requested deletion of corpus '{corpus_name}'.")

        if corpus_name not in corpus_manager.get_available_corpus_list():
            logger.warning(f"Corpus '{corpus_name}' not found.")
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' does not exist."
            )

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}' to delete corpus.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to delete corpus."
            )

        corpus_status = corpus_manager.get_corpus_status(corpus_name=corpus_name)
        if corpus_status == "system":
            logger.info("System corpus cannot be deleted.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="System corpus cannot be deleted."
            )
        elif corpus_status == "active":
            logger.info("Corpus must be deactivated before deletion.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Deactivate the corpus before deletion."
            )

        # Delete the corpus directory (even if it contains files)
        corpus_manager.delete_corpus(corpus_name=corpus_name)

        logger.info(f"Corpus '{corpus_name}' deleted successfully.")

        return StandardResponse(
            success=True,
            exists=True,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' deleted successfully."
        )

    except Exception as e:
        logger.exception(f"[DELETE] Error deleting corpus: {e}")
        return StandardResponse(
            success=False,
            exists=True, 
            corpus_name=corpus_name, 
            message="Internal error while deleting corpus."
        )

@router.post(
    "/deactivate_corpus",
    response_model=StandardResponse,
    summary="Deactivate a specific corpus",
    description="Deactivate a corpus from the vector database if permissions and conditions allow"
)
async def deactivate_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Deactivate a corpus in the vector database.
    
    Args:
        corpus_query: Corpus Deactivate details
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Corpus Deactivate result
        
    """
    try:
        corpus_name = corpus_query.corpus_name
        
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=current_user.message
            )
        logger.info(f"User '{current_user.username}' requested deactivate of corpus '{corpus_name}'.")

        if corpus_name not in corpus_manager.get_available_corpus_list():
            logger.warning(f"Corpus '{corpus_name}' not found.")
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' does not exist."
            )

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}' to deactivate corpus.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to deactivate corpus."
            )

        corpus_status = corpus_manager.get_corpus_status(corpus_name=corpus_name)

        if corpus_status == "system":
            logger.info("System corpus cannot be deactivate.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="System corpus cannot be deactivate."
            )
        
        elif corpus_status == "deactivate":
            logger.info("Corpus already deactivate.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Corpus already deactivate."
            )

        corpus_manager.set_corpus_status(corpus_name=corpus_name, status="deactivate")

        return StandardResponse(
            success=True,
            exists=True,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' deactivate successfully."
        )

    except Exception as e:
        logger.exception(f"[DELETE] Error deactivate corpus: {e}")
        return StandardResponse(
            success=False,
            exists=True, 
            corpus_name=corpus_name, 
            message="Internal error while deactivate corpus."
        )


@router.post(
    "/activate_corpus",
    response_model=StandardResponse,
    summary="Activate a specific corpus",
    description="Activate a corpus from the vector database if permissions and conditions allow"
)
async def activate_corpus(
    corpus_query: CorpusQueryRequest,
    current_user: AuthenticationResult = Depends(get_current_user)
) -> StandardResponse:
    """
    Activate a corpus in the vector database.
    
    Args:
        corpus_query: CorpusActivate details
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Corpus Activate result
        
    """
    try:
        corpus_name = corpus_query.corpus_name
        
        if not current_user.is_authenticated:
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=current_user.message
            )
        logger.info(f"User '{current_user.username}' requested activate of corpus '{corpus_name}'.")

        if corpus_name not in corpus_manager.get_available_corpus_list():
            logger.warning(f"Corpus '{corpus_name}' not found.")
            return StandardResponse(
                success=False,
                exists=False,
                corpus_name=corpus_name,
                message=f"Corpus '{corpus_name}' does not exist."
            )

        if not check_user_permission(current_user=current_user, required_role=UserRole.ADMIN_USER):
            logger.warning(f"Permission denied for user '{current_user.username}' to activate corpus.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Permission denied to activate corpus."
            )

        corpus_status = corpus_manager.get_corpus_status(corpus_name=corpus_name)

        if corpus_status == "system":
            logger.info("System corpus cannot be activate.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="System corpus cannot be activate."
            )
        
        elif corpus_status == "active":
            logger.info("Corpus already active.")
            return StandardResponse(
                success=False,
                exists=True, 
                corpus_name=corpus_name, 
                message="Corpus already active."
            )
        
        corpus_manager.set_corpus_status(corpus_name=corpus_name, status="active")

        return StandardResponse(
            success=True,
            exists=True,
            corpus_name=corpus_name,
            message=f"Corpus '{corpus_name}' activate successfully."
        )

    except Exception as e:
        logger.exception(f"[DELETE] Error activate corpus: {e}")
        return StandardResponse(
            success=False,
            exists=True, 
            corpus_name=corpus_name, 
            message="Internal error while activate corpus."
        )
