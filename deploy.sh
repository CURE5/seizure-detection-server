#!/bin/bash
# Runs ON the Raspberry Pi: pull the latest code, refresh the Python
# environment, fetch models and restart the server under PM2.
#
# Configure with environment variables (all optional):
#   SERVER_DIR   checkout location      (default: $HOME/seizure-detection-server)
#   VENV_DIR     virtualenv name        (default: sz)
#   BRANCH       git branch to deploy   (default: main)
set -e

SERVER_DIR="${SERVER_DIR:-$HOME/seizure-detection-server}"
VENV_DIR="${VENV_DIR:-sz}"
BRANCH="${BRANCH:-main}"

echo "🚀 Starting deployment in $SERVER_DIR ..."
cd "$SERVER_DIR"

echo "📥 Pulling latest changes from $BRANCH ..."
git checkout "$BRANCH"
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Stashing local changes..."
    git stash
fi
git fetch origin
git pull origin "$BRANCH"

# uv is used for fast, reproducible installs
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "🐍 Setting up Python environment..."
# picamera2 / libcamera come from apt, so the venv must see system packages.
if [ -d "$VENV_DIR" ] && ! grep -q "include-system-site-packages = true" "$VENV_DIR/pyvenv.cfg"; then
    echo "Recreating virtual environment with system site packages..."
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python python3 --system-site-packages
fi

PYTHON_VERSION=$(python3 --version 2>&1 | sed -n 's/Python \(3\.[0-9]*\).*/\1/p')
PTH_FILE="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/system.pth"
if [ ! -f "$PTH_FILE" ]; then
    mkdir -p "$(dirname "$PTH_FILE")"
    echo '/usr/lib/python3/dist-packages' > "$PTH_FILE"
fi

source "$VENV_DIR/bin/activate"

if command -v hailortcli &> /dev/null; then
    echo "✓ Hailo AI accelerator detected"
else
    echo "ℹ️  Hailo not found; the server falls back to MediaPipe on CPU."
    echo "   To install: sudo apt install -y hailo-all"
fi

echo "📦 Installing system dependencies..."
sudo apt install -y python3-libcamera python3-kms++ libcap-dev 2>/dev/null || true

echo "📦 Installing Python dependencies..."
uv pip install -r requirements_pi.txt

if [ ! -f "mediapipe_models/pose_landmarker.task" ]; then
    echo "📥 Downloading MediaPipe models..."
    bash download_models.sh
else
    echo "✓ MediaPipe models already present"
fi

echo "🔄 Restarting server..."
pm2 delete server 2>/dev/null || true
pm2 start "$VENV_DIR/bin/python" \
    --name server \
    --interpreter none \
    -- -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning
pm2 save

echo "✅ Deployment complete!"
pm2 status
