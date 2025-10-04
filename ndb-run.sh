#!/bin/bash

# ==============================
# NebulonDB Production Run Script
# ==============================

# Path to config file 
CONFIG_FILE="nebulondb.cfg"

# Function to get value by section and key
get_config_value() {
    local section=$1
    local key=$2
    awk -F '=' -v section="$section" -v key="$key" '
    $0 ~ "\\["section"\\]" { in_section=1; next }
    /^\[.*\]/ { in_section=0 }
    in_section && $1 ~ key { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }
    ' "$CONFIG_FILE"
}

# ==============================
# First, get NEBULONDB_HOME from config (paths section)
# ==============================
NEBULONDB_HOME=$(get_config_value "paths" "NEBULONDB_HOME")
if [ -z "$NEBULONDB_HOME" ]; then
    echo "❌ NEBULONDB_HOME not set in config"
    exit 1
fi

# Add project root to PATH so scripts are accessible
export PATH="$NEBULONDB_HOME:$PATH"
export NEBULONDB_HOME="$NEBULONDB_HOME"

# ==============================
# Read server configuration
# ==============================
APP_NAME=$(get_config_value "server" "APP_NAME")
HOST=$(get_config_value "server" "HOST")
PORT=$(get_config_value "server" "PORT")
WORKERS=$(get_config_value "server" "WORKERS")

TIMEOUT=$(get_config_value "server" "TIMEOUT")
KEEP_ALIVE=$(get_config_value "server" "KEEP_ALIVE")
GRACEFUL_TIMEOUT=$(get_config_value "server" "GRACEFUL_TIMEOUT")
ACCESS_LOGFILE=$(get_config_value "server" "ACCESS_LOGFILE")
ERROR_LOGFILE=$(get_config_value "server" "ERROR_LOGFILE")
LOG_LEVEL=$(get_config_value "server" "LOG_LEVEL")

# ==============================
# Fallback values if empty
# ==============================
APP_NAME=${APP_NAME:-NebulonDB}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-6969}
WORKERS=${WORKERS:-1}
TIMEOUT=${TIMEOUT:-NULL}
KEEP_ALIVE=${KEEP_ALIVE:-NULL}
GRACEFUL_TIMEOUT=${GRACEFUL_TIMEOUT:-NULL}
ACCESS_LOGFILE=${ACCESS_LOGFILE:-NULL}
ERROR_LOGFILE=${ERROR_LOGFILE:-NULL}
LOG_LEVEL=${LOG_LEVEL:-info}

# ==============================
# Print startup info
# ==============================
echo "🚀 Starting $APP_NAME on $HOST:$PORT with $WORKERS workers..."
echo "Using optional parameters: timeout=$TIMEOUT, keep_alive=$KEEP_ALIVE, graceful_timeout=$GRACEFUL_TIMEOUT, access_log=$ACCESS_LOGFILE, error_log=$ERROR_LOGFILE, log_level=$LOG_LEVEL"

# ==============================
# Detect Python module path
# ==============================
MODULE_PATH="ndb_host.main"
if [ ! -f "$NEBULONDB_HOME/ndb_host/main.py" ]; then
    echo "❌ Cannot find ndb_host/main.py in $NEBULONDB_HOME/ndb_host"
    exit 1
fi

# ==============================
# Prepare logging directories if needed
# ==============================
[[ "$ACCESS_LOGFILE" != "NULL" && "$ACCESS_LOGFILE" != "-" ]] && mkdir -p "$(dirname "$ACCESS_LOGFILE")"
[[ "$ERROR_LOGFILE" != "NULL" && "$ERROR_LOGFILE" != "-" ]] && mkdir -p "$(dirname "$ERROR_LOGFILE")"

# ==============================
# Build Gunicorn command
# ==============================
CMD="python -m gunicorn $MODULE_PATH:app -k uvicorn.workers.UvicornWorker --bind $HOST:$PORT --workers $WORKERS"

[[ "$TIMEOUT" != "NULL" ]] && CMD="$CMD --timeout $TIMEOUT"
[[ "$KEEP_ALIVE" != "NULL" ]] && CMD="$CMD --keep-alive $KEEP_ALIVE"
[[ "$GRACEFUL_TIMEOUT" != "NULL" ]] && CMD="$CMD --graceful-timeout $GRACEFUL_TIMEOUT"
[[ "$ACCESS_LOGFILE" != "NULL" ]] && CMD="$CMD --access-logfile $ACCESS_LOGFILE"
[[ "$ERROR_LOGFILE" != "NULL" ]] && CMD="$CMD --error-logfile $ERROR_LOGFILE"
[[ "$LOG_LEVEL" != "NULL" ]] && CMD="$CMD --log-level $LOG_LEVEL"

# ==============================
# Run the server
# ==============================
echo "Running command: $CMD"
exec $CMD
