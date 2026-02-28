#!/usr/bin/env bash
# run.sh - Auto-bootstrapping wrapper for bdc-analyst

set +e
[ -f ~/.bash_profile ] && source ~/.bash_profile >/dev/null 2>&1
[ -f ~/.bashrc ] && source ~/.bashrc >/dev/null 2>&1
[ -f ~/.zprofile ] && source ~/.zprofile >/dev/null 2>&1
[ -f ~/.zshrc ] && source ~/.zshrc >/dev/null 2>&1
set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SKILL_DIR/venv"
REQ_FILE="$SKILL_DIR/requirements.txt"
SCRIPT_FILE="$SKILL_DIR/scripts/bdc_analyzer.py"

get_best_python() {
    for prefix in "/opt/homebrew/bin" "/usr/local/bin" "/usr/bin" "$HOME/.local/bin"; do
        if [ -d "$prefix" ]; then
            local best_py=$(ls -1 "$prefix"/python3.* 2>/dev/null | grep -E "^$prefix/python3\.[0-9]+$" | sort -V | tail -n 1)
            if [ -n "$best_py" ] && [ -x "$best_py" ]; then
                echo "$best_py"
                return 0
            fi
        fi
    done
    echo "python3"
}

if [ ! -d "$VENV_DIR" ]; then
    BASE_PYTHON=$(get_best_python)
    echo "Initializing virtual environment using: $BASE_PYTHON" >&2
    
    "$BASE_PYTHON" -m venv "$VENV_DIR"
    
    echo "Installing dependencies..." >&2
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE" --quiet
    echo "Environment ready." >&2
fi

exec "$VENV_DIR/bin/python" -W ignore "$SCRIPT_FILE" "$@"