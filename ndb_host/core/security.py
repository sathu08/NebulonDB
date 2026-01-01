from passlib.context import CryptContext

from utils.models import AuthenticationConfig
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Security Context
# ==========================================================

password_context = CryptContext(
    schemes=AuthenticationConfig.PASSWORD_HASH_SCHEMES,
    deprecated=AuthenticationConfig.PASSWORD_HASH_DEPRECATED
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return password_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def hash_password(password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Hashed password
        
    Raises:
        ValueError: If password hashing fails
    """
    try:
        return password_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        return {}
    
