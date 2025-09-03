from fastapi import Depends, HTTPException, status
from fastapi import APIRouter

from services.user_service import create_user, get_current_user
from utils.models import StandardResponse, UserRegistrationRequest, UserAuthenticationResponse
from utils.logger import logger

router = APIRouter()

@router.post(
    "/register",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with specified username, password, and role" 
)
async def register_user(
    user_data: UserRegistrationRequest
) -> StandardResponse:
    try:
        logger.info(f"Attempting to register user: {user_data.username}")
        
        result = create_user(
            username=user_data.username,
            password=user_data.password,
            user_role=user_data.user_role
        )

        if isinstance(result, dict) and "message" in result:
            logger.info(f"User registered successfully: {user_data.username}")
            return StandardResponse(
                success=True,
                message="User registered successfully",
                data={"username": user_data.username, "role": user_data.user_role}
            )
        elif "User already exists." in str(result):
            logger.warning(f"Registration failed - user already exists: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        else:
            logger.error(f"Registration failed for user: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed due to internal error"
            )
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )
    
@router.get(
    "/verify",
    response_model=UserAuthenticationResponse,
    summary="Verify user authentication",
    description="Verify the current user's authentication status and return user details"
)
async def verify_authentication(
    current_user: dict = Depends(get_current_user)
) -> UserAuthenticationResponse:
    """
    Verify user authentication and return user details.
    
    Args:
        current_user: Current authenticated user details
        
    Returns:
        UserAuthenticationResponse: Authentication verification result
    """
    logger.info(f"Authentication verified for user: {current_user.get('username', 'unknown')}")
    return UserAuthenticationResponse(
        message="Authentication verified successfully",
        user=current_user
    )
