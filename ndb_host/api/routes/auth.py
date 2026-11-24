"""
NDB API Authentication and Authorization
==========================================================

This module handles authentication and authorization for the NDB API.
It provides endpoints for user registration and authentication.

"""

from fastapi import Depends, Response
from fastapi import APIRouter, HTTPException, status

from core.permissions import check_user_permission
from services.user_service import create_user, get_current_user, change_password, delete_user, user_manager

from utils.models import StandardResponse, UserRegistrationRequest, UserAuthenticationResponse, AuthenticationResult, ChangePasswordRequest, DeleteUserRequest
from utils.constants import UserRole
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger("access")

# ==========================================================
#        API Router for Authentication
# ==========================================================

router = APIRouter()

_ROLE_LEVEL = {
    UserRole.USER: 1,
    UserRole.ADMIN_USER: 2,
    UserRole.SUPER_USER: 3,
    UserRole.SYSTEM: 4,
}

# Roles that may never be created through the public API (F-02).
# SYSTEM accounts are created by bootstrap; SUPER_USER by an operator.
API_FORBIDDEN_ROLES = {UserRole.SYSTEM, UserRole.SUPER_USER}


def _parse_role(value: str) -> UserRole | None:
    try:
        return UserRole(value)
    except ValueError:
        return None


@router.post(
    "/register",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with specified username, password, and role"
)
async def register_user(
    user_data: UserRegistrationRequest,
    current_user: AuthenticationResult = Depends(get_current_user),
    response: Response = None,
) -> StandardResponse:
    def _code(code: int):
        if response is not None:
            response.status_code = code
    try:
        logger.info(f"Attempting to register user: {user_data.username}")

        if not current_user.is_authenticated:
            _code(401)
            return StandardResponse(
                success=False,
                exists=False,
                message=current_user.message
            )

        # F-02: registration is admin-only.
        if not check_user_permission(current_user, UserRole.ADMIN_USER):
            logger.warning(f"User '{current_user.username}' lacks permission to register users")
            _code(403)
            return StandardResponse(
                success=False,
                exists=False,
                message="Permission denied to register users."
            )

        requested_role = _parse_role(user_data.user_role)
        if requested_role is None:
            _code(400)
            return StandardResponse(success=False, message="Invalid user_role")
        if requested_role in API_FORBIDDEN_ROLES:
            logger.warning(f"User '{current_user.username}' attempted to create forbidden role '{requested_role.value}'")
            _code(403)
            return StandardResponse(
                success=False,
                message="Cannot create system/super_user accounts via API."
            )
        # Creator must outrank the new account (prevents lateral privilege gain).
        creator_level = _ROLE_LEVEL.get(current_user.role, 0)
        new_level = _ROLE_LEVEL.get(requested_role, 0)
        if new_level >= creator_level:
            _code(403)
            return StandardResponse(
                success=False,
                message="Permission denied: cannot create a role equal to or above your own."
            )

        result = create_user(
            username=user_data.username,
            password=user_data.password,
            user_role=user_data.user_role
        )
        if result.get("success"):
            logger.info(f"User registered successfully: {user_data.username}")
            return StandardResponse(
                success=True,
                message="User registered successfully",
                data={"username": user_data.username, "role": user_data.user_role}
            )
        elif not result.get("success") and "User already exists" in result.get("message"):
            logger.warning(f"Registration failed - user already exists: {user_data.username}")
            # L-01: duplicates are 409, not 201.
            _code(409)
            return StandardResponse(
                    success=False,
                    message="User already exists"
                )
        else:
            logger.error(f"Registration failed for user: {user_data.username}")
            _code(500)
            return StandardResponse(
                    success=False,
                    message="Registration failed due to internal error"
                )
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        ) from e

@router.get(
    "/verify",
    response_model=UserAuthenticationResponse,
    summary="Verify user authentication",
    description="Verify the current user's authentication status and return user details"
)
async def verify_authentication(
    current_user: AuthenticationResult = Depends(get_current_user)
) -> UserAuthenticationResponse:
    """
    Verify user authentication and return user details.

    Args:
        current_user: Current authenticated user details

    Returns:
        UserAuthenticationResponse: Authentication verification result
    """
    logger.info(f"Authentication verified for user: {current_user.username}")
    return UserAuthenticationResponse(
        message=current_user.message if current_user.message else "Authentication verified successfully",
        user=current_user.model_dump()
    )

@router.post(
    "/change_password",
    response_model=StandardResponse,
    summary="Change the current user's password",
    description="Change the password of the currently authenticated user"
)
async def change_user_password(
    password_data: ChangePasswordRequest,
    current_user: AuthenticationResult = Depends(get_current_user),
    response: Response = None,
) -> StandardResponse:
    def _code(code: int):
        if response is not None:
            response.status_code = code
    try:
        if not current_user.is_authenticated:
            _code(401)
            return StandardResponse(
                success=False,
                exists=False,
                message=current_user.message
            )

        logger.info(f"Attempting to change password for user: {current_user.username}")
        result = change_password(
            username=current_user.username,
            current_password=password_data.current_password,
            new_password=password_data.new_password
        )
        if result.get("success"):
            logger.info(f"Password changed successfully for user: {current_user.username}")
            return StandardResponse(
                success=True,
                message=result.get("message", "Password changed successfully")
            )
        else:
            logger.warning(f"Password change failed for user: {current_user.username}: {result.get('message')}")
            _code(400)
            return StandardResponse(
                success=False,
                message=result.get("message", "Password change failed")
            )
    except Exception as e:
        logger.error(f"Unexpected error during password change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password change"
        ) from e

@router.post(
    "/delete_user",
    response_model=StandardResponse,
    summary="Delete a user",
    description="Delete a user account. The current user cannot delete their own account."
)
async def delete_user_account(
    user_data: DeleteUserRequest,
    current_user: AuthenticationResult = Depends(get_current_user),
    response: Response = None,
) -> StandardResponse:
    def _code(code: int):
        if response is not None:
            response.status_code = code
    try:
        if not current_user.is_authenticated:
            _code(401)
            return StandardResponse(
                success=False,
                exists=False,
                message=current_user.message
            )

        if user_data.username == current_user.username:
            logger.warning(f"User {current_user.username} attempted to delete their own account")
            _code(403)
            return StandardResponse(
                success=False,
                message="You cannot delete your own account"
            )

        # F-03: deletion is admin-only and cannot touch equal/higher ranks.
        if not check_user_permission(current_user, UserRole.ADMIN_USER):
            logger.warning(f"User '{current_user.username}' lacks permission to delete users")
            _code(403)
            return StandardResponse(
                success=False,
                message="Permission denied to delete users."
            )

        target = user_manager.get_user(user_data.username)
        if target is None:
            _code(404)
            return StandardResponse(success=False, message="User not found")
        try:
            target_role = UserRole(target.get("role", UserRole.USER.value))
        except ValueError:
            target_role = UserRole.USER
        if _ROLE_LEVEL.get(target_role, 0) >= _ROLE_LEVEL.get(current_user.role, 0):
            logger.warning(
                f"User '{current_user.username}' attempted to delete equal/higher-ranked account '{user_data.username}'"
            )
            _code(403)
            return StandardResponse(
                success=False,
                message="Permission denied: cannot delete an account of equal or higher rank."
            )

        logger.info(f"Attempting to delete user: {user_data.username}")
        result = delete_user(username=user_data.username)
        if result.get("success"):
            logger.info(f"User deleted successfully: {user_data.username}")
            return StandardResponse(
                success=True,
                message=result.get("message", "User deleted successfully"),
                data={"username": user_data.username}
            )
        else:
            logger.warning(f"User deletion failed: {user_data.username}: {result.get('message')}")
            return StandardResponse(
                success=False,
                message=result.get("message", "User deletion failed")
            )
    except Exception as e:
        logger.error(f"Unexpected error during user deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during user deletion"
        ) from e
