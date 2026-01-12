#!/usr/bin/env bash

check_python_version() {
    local python_cmd=$1
    if command -v "$python_cmd" >/dev/null 2>&1; then
        version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [ "$version" = "3.10" ]; then
            echo "$python_cmd"
            return 0
        fi
    fi
    return 1
}

PYTHON_CMD=""

# List of possible python commands
CANDIDATES=("python" "python3" "python3.10")

for cmd in "${CANDIDATES[@]}"; do
    PYTHON_FOUND=$(check_python_version "$cmd" || true)
    if [ -n "$PYTHON_FOUND" ]; then
        PYTHON_CMD="$PYTHON_FOUND"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    printf "Error: Python 3.10 not found on this system.\n\
    Please ensure:\n\
    1. that Python 3.10.x is installed, and\n\
    2. that its executable is added to your system's environment variables.\n\
    You can get Python 3.10 from: https://www.python.org/downloads/" >&2
    exit 1
fi

echo "Python 3.10 found: $PYTHON_CMD"

export $PYTHON_CMD
