import sys
from pathlib import Path

# === Add the current script's directory ===

CURRENT_DIR = Path(__file__).resolve().parent
if CURRENT_DIR not in sys.path:
    sys.path.append(str(CURRENT_DIR))