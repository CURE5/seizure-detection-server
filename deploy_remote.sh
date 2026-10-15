#!/bin/bash
# Runs on your workstation: SSH to the Pi and execute deploy.sh there.
#
#   PI_HOST=pi@my-pi.local ./deploy_remote.sh
#
#   PI_HOST     ssh target                      (default: pi@raspberrypi.local)
#   SERVER_DIR  checkout dir, relative to $HOME (default: seizure-detection-server)
set -e

PI_HOST="${PI_HOST:-pi@raspberrypi.local}"
SERVER_DIR="${SERVER_DIR:-seizure-detection-server}"

echo "🚀 Connecting to $PI_HOST..."
# -t allocates a tty so sudo inside deploy.sh can prompt if needed.
ssh -t -o ConnectTimeout=10 -o ServerAliveInterval=5 "$PI_HOST" \
    "SERVER_DIR=\$HOME/$SERVER_DIR bash \$HOME/$SERVER_DIR/deploy.sh"
