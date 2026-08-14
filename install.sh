#!/usr/bin/env bash

# ============================================================
# NebulonDB Installation Script (curl | bash)
# Installs NebulonDB into a fixed location (~/.nebulondb) and
# creates a global 'nebulondb' launcher in ~/.local/bin.
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

REPO_URL="https://github.com/sathu08/NebulonDB.git"
BRANCH="master"

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

PROJECT_DIR="$HOME/.nebulondb"

log "Install directory:"
log "$PROJECT_DIR"

if [[ -d "$PROJECT_DIR/.git" ]]; then

    # ---------- Existing repository: fetch + update ----------

    CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"

    if [[ -z "$CURRENT_REMOTE" ]]; then
        error "Git remote 'origin' is not configured in $PROJECT_DIR"
    fi

    log "Git remote:"
    log "$CURRENT_REMOTE"

    log "Fetching latest changes from branch '$BRANCH'..."

    git fetch origin "$BRANCH"

    log "Switching to branch '$BRANCH'..."

    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
    else
        git checkout -b "$BRANCH" --track "origin/$BRANCH"
    fi

    log "Updating branch '$BRANCH'..."

    git pull --ff-only origin "$BRANCH"

else

    # ---------- Fresh clone ----------

    if [[ -d "$PROJECT_DIR" ]]; then
        error "NebulonDB directory already exists but is not a Git repository: $PROJECT_DIR"
    fi

    log "Cloning NebulonDB into:"
    log "$PROJECT_DIR"
    log "Branch: $BRANCH"

    git clone \
        --branch "$BRANCH" \
        --single-branch \
        "$REPO_URL" \
        "$PROJECT_DIR"

    log "Repository cloned successfully."
fi

# ------------------------------------------------------------
# Set Current Directory to NebulonDB
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
# Optional ML / Vector Extras (off by default)
# ------------------------------------------------------------

if [[ "${NEBULONDB_INSTALL_ML:-0}" == "1" ]]; then
    log "Installing optional ML/vector extras (CPU-only torch)..."
    uv sync --python "$PYTHON_VERSION" --extra ml
fi

# ------------------------------------------------------------
# Install Global NebulonDB CLI
# ------------------------------------------------------------

VENV_DIR="$PROJECT_DIR/.venv"
CLI_PATH="$VENV_DIR/bin/nebulondb"
BIN_DIR="$HOME/.local/bin"
GLOBAL_CLI="$BIN_DIR/nebulondb"

if [[ ! -x "$CLI_PATH" ]]; then
    error "NebulonDB CLI was not created: $CLI_PATH"
fi

log "Installing global NebulonDB CLI..."

mkdir -p "$BIN_DIR"

cat > "$GLOBAL_CLI" <<EOF
#!/usr/bin/env bash

export NEBULONDB_HOME="$PROJECT_DIR"

exec "$CLI_PATH" "\$@"
EOF

chmod +x "$GLOBAL_CLI"

# ------------------------------------------------------------
# Configure ~/.local/bin in PATH
# ------------------------------------------------------------

BASHRC="$HOME/.bashrc"

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -f "$BASHRC" ]] && ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' "$BASHRC"; then
    printf '\n# User local binaries\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$BASHRC"
fi

# ------------------------------------------------------------
# Verify NebulonDB CLI
# ------------------------------------------------------------

if ! command -v nebulondb >/dev/null 2>&1; then
    error "NebulonDB CLI was not installed correctly."
fi

log "NebulonDB executable:"
log "$(command -v nebulondb)"

log "Testing NebulonDB CLI..."

nebulondb --help

# ------------------------------------------------------------
# Installation Complete
# ------------------------------------------------------------

export NEBULONDB_HOME="$PROJECT_DIR"

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
echo "Then start NebulonDB from anywhere:"
echo ""
echo "    nebulondb start"
echo ""
echo "============================================================"
