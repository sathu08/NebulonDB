from utils.models import AuthenticationResult, UserRole
from utils.logger import logger


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
        UserRole.SUPER_USER: 3
    }
    
    current_user_level = role_hierarchy.get(current_user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    has_permission = current_user_level >= required_level
    
    logger.debug(f"Permission check for {current_user.username}: {has_permission}")
    
    return has_permission