#!/usr/bin/env bash

# ============================================================
# NebulonDB Installation Script
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

REPO_URL="https://github.com/sathu08/NebulonDB.git"
BRANCH="dev"
PROJECT_DIR_NAME="NebulonDB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# Clone NebulonDB
# ------------------------------------------------------------

log "NebulonDB repository:"
log "$REPO_URL"

log "Target branch:"
log "$BRANCH"

# mkdir -p "$SCRIPT_DIR"

PROJECT_DIR="$SCRIPT_DIR/$PROJECT_DIR_NAME"

if [[ -d "$PROJECT_DIR/.git" ]]; then

    log "NebulonDB repository already exists:"
    log "$PROJECT_DIR"

    cd "$PROJECT_DIR"

    log "Fetching latest changes from branch '$BRANCH'..."

    git fetch origin "$BRANCH"

    log "Switching to branch '$BRANCH'..."

    git checkout "$BRANCH"

    log "Updating branch '$BRANCH'..."

    git pull --ff-only origin "$BRANCH"

else

    if [[ -d "$PROJECT_DIR" ]]; then
        error "Directory already exists but is not a Git repository: $PROJECT_DIR"
    fi

    log "Cloning NebulonDB..."
    log "Branch: $BRANCH"

    git clone \
        --branch "$BRANCH" \
        --single-branch \
        "$REPO_URL" \
        "$PROJECT_DIR"

    log "Repository cloned successfully."
fi

# ------------------------------------------------------------
# Move to Project Directory
# ------------------------------------------------------------

cd "$PROJECT_DIR"

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
