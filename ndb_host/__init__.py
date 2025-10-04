import os
import sys
from .db.NebulonDBConfig import NebulonDBConfig

# Get NebulonDB home from environment
NEBULONDB_HOME = NebulonDBConfig.NEBULONDB_HOME

if not NEBULONDB_HOME or not os.path.isdir(NEBULONDB_HOME):
    raise EnvironmentError("NEBULONDB_HOME environment variable is not set or invalid")

if NEBULONDB_HOME not in sys.path:
    sys.path.append(NEBULONDB_HOME)

# Add the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

