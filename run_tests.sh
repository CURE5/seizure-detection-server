#!/bin/bash
# Run the test suite with whichever virtualenv exists (venv, sz or .venv).
cd "$(dirname "$0")"

for v in venv sz .venv; do
    if [ -x "$v/bin/python" ]; then
        exec "$v/bin/python" -m pytest "$@"
    fi
done

exec python3 -m pytest "$@"
