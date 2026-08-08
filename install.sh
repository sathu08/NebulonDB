#!/usr/bin/env bash

# ============================================================
# NebulonDB Installation Script (curl | bash)
# Clones the repository into the current working directory and
# sets the current directory to the NebulonDB repo root.
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

REPO_URL="https://github.com/sathu08/NebulonDB.git"
BRANCH="dev"
PROJECT_DIR_NAME="NebulonDB"

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

log() {
    echo "[NebulonDB] $*"
}

error() {
    echo "[NebulonDB][ERROR] $*" >&2
    exit 1
}

# ------------------------------------------------------------
# Check Git
# ------------------------------------------------------------

if ! command -v git >/dev/null 2>&1; then
    error "Git is not installed. Please install Git first."
fi

# ------------------------------------------------------------
# Clone or Update NebulonDB
# ------------------------------------------------------------

log "NebulonDB repository:"
log "$REPO_URL"

log "Target branch:"
log "$BRANCH"

TARGET_DIR="$(pwd)"

log "Target directory:"
log "$TARGET_DIR"

cd "$TARGET_DIR"

if [[ -d "$TARGET_DIR/.git" ]]; then

    # ---------- Existing repository: fetch + update ----------

    CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"

    if [[ -z "$CURRENT_REMOTE" ]]; then
        error "Git remote 'origin' is not configured in $TARGET_DIR"
    fi

    log "Git remote:"
    log "$CURRENT_REMOTE"

    log "Fetching latest changes from branch '$BRANCH'..."

    git fetch origin "$BRANCH"

    log "Switching to branch '$BRANCH'..."

    git checkout "$BRANCH"

    log "Updating branch '$BRANCH'..."

    git pull --ff-only origin "$BRANCH"

else

    # ---------- Fresh clone ----------

    if [[ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
        # Current directory is not empty: clone into a NebulonDB subdirectory
        TARGET_DIR="$TARGET_DIR/$PROJECT_DIR_NAME"
        if [[ -d "$TARGET_DIR/.git" ]]; then
            error "A NebulonDB repository already exists at: $TARGET_DIR"
        fi
        log "Directory is not empty. Cloning into subdirectory:"
        log "$TARGET_DIR"
        mkdir -p "$TARGET_DIR"
    else
        # Current directory is empty: clone directly into it
        log "Cloning into current directory..."
    fi

    log "Branch: $BRANCH"

    git clone \
        --branch "$BRANCH" \
        --single-branch \
        "$REPO_URL" \
        "$TARGET_DIR"

    log "Repository cloned successfully."
fi

# ------------------------------------------------------------
# Set Current Directory to NebulonDB
# ------------------------------------------------------------

cd "$TARGET_DIR"

PROJECT_DIR="$TARGET_DIR"

log "NebulonDB Home:"
log "$PROJECT_DIR"

# ------------------------------------------------------------
# Verify Branch
# ------------------------------------------------------------

CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    error "Expected branch '$BRANCH', but currently on '$CURRENT_BRANCH'."
fi

log "Git branch:"
log "$CURRENT_BRANCH"

# ------------------------------------------------------------
# Check / Install uv
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "uv is not installed."
    log "Installing uv..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    error "uv installation failed or uv is not available in PATH."
fi

log "uv version:"
uv --version

# ------------------------------------------------------------
# Check / Install Python 3.10
# ------------------------------------------------------------

PYTHON_VERSION="3.10"

log "Checking Python $PYTHON_VERSION..."

if uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    log "Python $PYTHON_VERSION is already installed."
else
    log "Python $PYTHON_VERSION not found."
    log "Installing Python $PYTHON_VERSION..."

    uv python install "$PYTHON_VERSION"
fi

PYTHON_PATH="$(uv python find "$PYTHON_VERSION")"

log "Using Python:"
log "$PYTHON_PATH"

# ------------------------------------------------------------
# Install NebulonDB
# ------------------------------------------------------------

log "Installing NebulonDB dependencies..."

uv sync --python "$PYTHON_VERSION"

# ------------------------------------------------------------
# Activate Virtual Environment
# ------------------------------------------------------------

VENV_DIR="$PROJECT_DIR/.venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    error "Virtual environment was not created: $VENV_DIR"
fi

log "Activating virtual environment..."

source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------
# Verify Nebulon CLI
# ------------------------------------------------------------

if ! command -v nebulondb >/dev/null 2>&1; then
    error "NebulonDB CLI was not installed correctly."
fi

log "NebulonDB executable:"
log "$(command -v nebulondb)"

log "Testing NebulonDB CLI..."

nebulondb --help

# ------------------------------------------------------------
# Configure NEBULONDB_HOME
# ------------------------------------------------------------

export NEBULONDB_HOME="$PROJECT_DIR"

BASHRC="$HOME/.bashrc"

log "Configuring NEBULONDB_HOME..."

if [[ -f "$BASHRC" ]]; then
    sed -i '/^[[:space:]]*export NEBULONDB_HOME=/d' "$BASHRC"
fi

printf '\n# NebulonDB\nexport NEBULONDB_HOME="%s"\n' "$NEBULONDB_HOME" >> "$BASHRC"

# ------------------------------------------------------------
# Installation Complete
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo " NebulonDB Installation Complete"
echo "============================================================"
echo ""
echo "Repository     : $REPO_URL"
echo "Branch         : $CURRENT_BRANCH"
echo "Project        : $PROJECT_DIR"
echo "Python         : $PYTHON_PATH"
echo "Virtual Env    : $VENV_DIR"
echo "NebulonDB CLI  : $(command -v nebulondb)"
echo "NEBULONDB_HOME : $NEBULONDB_HOME"
echo ""
echo "Open a new terminal or run:"
echo ""
echo "    source ~/.bashrc"
echo ""
echo "Then run:"
echo ""
echo "    nebulondb start"
echo ""
echo "============================================================"
