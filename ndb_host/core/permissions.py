"""
NDB Core Permissions
==========================================================

This module handles permission checking for the NDB API.
It provides endpoints for user registration and authentication.

"""

from utils.models import AuthenticationResult, UserRole
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Permissions
# ==========================================================

def check_user_permission(current_user: AuthenticationResult, required_role: UserRole) -> bool:
    """
    Check if the current user has the required role or higher.
    
    Args:
        current_user: Current authenticated user
        required_role: Minimum required role
        
    Returns:
        bool: True if user has sufficient permissions
    """
    role_hierarchy = {
        UserRole.USER: 1,
        UserRole.ADMIN_USER: 2,
        UserRole.SUPER_USER: 3,
        UserRole.SYSTEM: 4
    }
    
    current_user_level = role_hierarchy.get(current_user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    has_permission = current_user_level >= required_level
    
    logger.debug("Permission check for %s (role=%s, level=%d) vs required=%s (level=%d): %s",
        current_user.username,
        current_user.role.name,
        current_user_level,
        required_role.name,
        required_level,
        has_permission,
    )

    
    return has_permission