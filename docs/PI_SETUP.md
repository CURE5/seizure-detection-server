# Raspberry Pi Setup Guide

Complete guide for setting up the seizure detection server on Raspberry Pi 5.

## Hardware Configuration

**Raspberry Pi 5 Specs:**
- Hostname: `seizure-pi` (any name; referenced below as `<pi-host>`)
- Username: pi

## ⚠️ IMPORTANT: OS Version

**You MUST use Raspberry Pi OS (Legacy, 64-bit) based on Debian Bookworm.**

**Why:** The current Raspberry Pi OS is based on Debian 13 (Trixie) which uses Python 3.13 by default. MediaPipe (required for pose detection) only supports up to Python 3.12, making Trixie incompatible.

**How to Install:**

1. Download **Raspberry Pi Imager** for your computer:
   - macOS: https://github.com/raspberrypi/rpi-imager/releases (download `imager_1.9.0.dmg` or latest)
   - Windows/Linux: https://www.raspberrypi.com/software/

2. Use Raspberry Pi Imager to flash the SD card:
   - **Choose Device**: Raspberry Pi 5
   - **Choose OS**: Raspberry Pi OS (other) → **Raspberry Pi OS (Legacy, 64-bit)**
     - This is Debian 12 (Bookworm) with Python 3.11
   - **Choose Storage**: Your SD card
   - **Configure Settings** (gear icon):
     - Set hostname (e.g. `seizure-pi`)
     - Enable SSH
     - Set username `pi` and a strong password
     - Configure WiFi if needed
   - Click **Write**

3. Boot the Pi and continue with setup below

**Verification:** After booting, check you have the right version:
```bash
cat /etc/os-release
# Should show: VERSION="12 (bookworm)"
python3 --version
# Should show: Python 3.11.x
```

## Initial Pi Setup

### 1. Basic Configuration

```bash
# Enable SSH and set hostname
sudo raspi-config

# Update system
sudo apt update
sudo apt full-upgrade

# Install Hailo accelerator (for AI inference)
sudo apt install hailo-all

# Verify Hailo
hailortcli fw-control identify
```

### 2. Git Configuration

```bash
# Install git
sudo apt install git -y

# Configure git
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

# Setup SSH keys for GitHub
ssh-keygen
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Test connection
ssh -v git@github.com
```

### 3. SSH Config for Multiple Repos

Create `~/.ssh/config`:

```
Host seizure-detection-server.github.com
  Hostname github.com
  User git
  AddKeysToAgent yes
  PreferredAuthentications publickey
  IdentityFile ~/.ssh/id_rsa

Host seizure-detection-app.github.com
  Hostname github.com
  User git
  AddKeysToAgent yes
  PreferredAuthentications publickey
  IdentityFile ~/.ssh/id_rsa2
```

### 4. Clone Repositories

```bash
cd ~
git clone https://github.com/CURE5/seizure-detection-server.git
git clone https://github.com/CURE5/seizure-detection-app.git
```

## Python Setup

### Install uv and Python Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
source $HOME/.cargo/env

cd ~/seizure-detection-server

# Create virtual environment
uv venv sz

# Activate virtual environment
source sz/bin/activate

# Install dependencies with uv (much faster than pip)
uv pip install -r requirements_pi.txt

# Install camera libraries
sudo apt install -y libcap-dev libcamera-tools libcamera-dev
uv pip install picamera2
```

## Node.js Setup (for companion app)

```bash
# Add NodeSource repository
sudo apt install -y ca-certificates curl gnupg
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/nodesource.gpg

NODE_MAJOR=18
echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | \
  sudo tee /etc/apt/sources.list.d/nodesource.list

# Install Node.js and Yarn
sudo apt update
sudo apt install nodejs
sudo npm install --global yarn

# Handle slow networks
yarn config set network-timeout 300000
```

## PM2 Process Manager

### Install and Setup PM2

```bash
# Install PM2
sudo npm install -g pm2

# Start server with PM2
cd ~/seizure-detection-server
source sz/bin/activate

pm2 start "fastapi run server.py --port 8000" \
  --name seizure-detection \
  --output logs/server-out.log \
  --error logs/server-err.log

# Start companion app
cd ~/seizure-detection-app
pm2 start yarn --name "app" \
  --output logs/app-out.log \
  --error logs/app-err.log \
  -- start

# Save PM2 configuration
pm2 save

# Setup PM2 to start on boot
pm2 startup
sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u pi --hp /home/pi
```

### PM2 Commands

```bash
# View running processes
pm2 list

# View logs
pm2 logs seizure-detection
pm2 logs app

# Restart services
pm2 restart seizure-detection
pm2 restart app

# Stop services
pm2 stop seizure-detection
pm2 delete seizure-detection
```

## Bash Shortcuts

Add to `~/.bashrc`:

```bash
alias server='cd ~/seizure-detection-server && source sz/bin/activate'
alias start='fastapi run server.py --port 8000'
```

Then reload: `source ~/.bashrc`

## Monitoring & Diagnostics

### Temperature Monitoring

```bash
# Check CPU temperature
vcgencmd measure_temp
# Normal: 54-66°C
# Warning: >80°C indicates overheating

# Check GPU memory
vcgencmd get_mem gpu
# PI5 returns: gpu=4M (GPU memory cannot be configured on Pi5)
```

### System Logs

```bash
# Check boot logs
sudo journalctl -b -1

# Check for crashes/reboots
sudo journalctl -b -1 -n 100 | grep -i "power\|shutdown\|reboot\|crash"

# Search app logs
grep -r "shutdown" /home/pi/seizure-detection-app/logs
```

### Network Diagnostics

```bash
# Check WiFi configuration
cat /etc/wpa_supplicant/wpa_supplicant.conf

# Test connectivity
ping -c 4 8.8.8.8

# Check open ports
sudo lsof -i :8000
sudo lsof -i :3000

# Kill process on port
sudo kill -9 <PID>
```

### Storage Performance

```bash
# Test write speed (should be 20-40 MB/s)
dd if=/dev/zero of=testfile bs=1M count=100 oflag=direct

# Speed indicators:
# Good: 20-40 MB/s
# Problematic: <10 MB/s (will bottleneck video recording)
```

## Camera Setup

### Test Camera

```bash
# List available cameras
rpicam-hello --list

# Test camera 1 for 10 seconds
rpicam-vid -t 10000s --camera 1
```

## Cloudflare Tunnel Setup

### Install Cloudflare Tunnel

```bash
# Add Cloudflare GPG key
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | \
  sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

# Add repository
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | \
  sudo tee /etc/apt/sources.list.d/cloudflared.list

# Install cloudflared
sudo apt-get update && sudo apt-get install cloudflared
```

### Configure Tunnel

```bash
# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create seizure-monitor
# Credentials saved to: /home/pi/.cloudflared/<TUNNEL-ID>.json

# Configure tunnel
sudo nano /etc/cloudflared/config.yml
```

Add to `/etc/cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL-ID>
credentials-file: /home/pi/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: monitor.example.com
    service: http://localhost:3000
  - service: http_status:404
```

### Setup DNS

```bash
# Route DNS to tunnel
cloudflared tunnel route dns seizure-monitor monitor.example.com

# Result: CNAME <TUNNEL-ID>.cfargotunnel.com -> monitor.example.com
```

### Run as Service

```bash
# Edit systemd service
sudo nano /etc/systemd/system/cloudflared.service
```

Update ExecStart line:

```ini
ExecStart=/usr/bin/cloudflared --no-autoupdate tunnel --config /etc/cloudflared/config.yml run
```

Enable and start service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable cloudflared

# Start service
sudo systemctl start cloudflared

# Check status
sudo systemctl status cloudflared

# Test tunnel
sudo cloudflared --config /etc/cloudflared/config.yml tunnel run
```

### Tunnel Management

```bash
# Check tunnel info
cloudflared tunnel info seizure-monitor

# Test temporary tunnel
cloudflared tunnel --url http://localhost:3000
```

## Audio Setup (Optional)

### USB Microphone to Bluetooth

```bash
# List audio sinks
pactl list short sinks

# List audio sources
pactl list short sources

# Route USB mic to Bluetooth speaker
pactl load-module module-loopback \
  source=alsa_input.usb-TTGK_Technology_USB_Audio_33022920230925-00.mono-fallback \
  sink=bluez_output.88_C6_26_1D_AA_24.1 \
  latency_msec=1000

# Unload loopback
pactl unload-module module-loopback
```

### PipeWire Configuration (Persistent)

Create `~/.config/pipewire/pipewire.conf.d/mic-to-bluetooth.conf`:

```
context.modules = [
    {   name = libpipewire-module-loopback
        args = {
            node.description = "USB Mic to Bluetooth"
            capture.props = {
                node.name = "loopback-capture"
                audio.rate = 48000
                media.class = "Audio/Source"
                node.target = "alsa_input.usb-TTGK_Technology_USB_Audio_33022920230925-00.mono-fallback"
            }
            playback.props = {
                node.name = "loopback-playback"
                media.class = "Audio/Sink"
                target.object = "bluez_output.88_C6_26_1D_AA_24.1"
                audio.latency = 1000000
            }
        }
    }
]
```

Restart PipeWire:

```bash
systemctl --user restart pipewire

# Verify module loaded
pactl list short modules | grep loopback
```

## Video Encoding Notes

### FFmpeg Re-encoding (if needed)

```bash
# Re-encode video for compatibility
ffmpeg -i video_20241229_192835.mp4 \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -c:a aac \
  output_reencoded.mp4
```

### PyAV Alternative (not currently used)

```bash
uv pip install av
```

See code examples in old README if needed for direct video encoding control.

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
sudo lsof -i :8000

# Kill existing process
sudo kill -9 <PID>

# Check Python environment
which python
python --version

# Reinstall dependencies with uv
source sz/bin/activate
uv pip install -r requirements_pi.txt
```

### PM2 Issues

```bash
# View PM2 logs
pm2 logs --lines 100

# Reset PM2
pm2 kill
pm2 start <your-command>
pm2 save
```

### Camera Issues

```bash
# Test camera access
rpicam-hello --list

# Check permissions
ls -l /dev/video*

# Reboot if camera not detected
sudo reboot
```

### WiFi Issues

```bash
# Check WiFi status
ifconfig wlan0

# Check configuration
cat /etc/wpa_supplicant/wpa_supplicant.conf

# Restart networking
sudo systemctl restart networking
```

## Performance Optimization

### For Video Recording

- Use fast SD card (Class 10, U3, or better)
- Test write speed with `dd` command (target: >20 MB/s)
- Monitor CPU temperature during operation
- Consider external SSD for better I/O performance

### For AI Detection

- Hailo accelerator significantly improves inference speed
- Monitor CPU usage: `htop`
- Adjust detection thresholds if getting too many false positives
- Use shadow mode for initial tuning

## Security Notes

- Change default passwords after setup
- Keep SSH keys secure
- Use Cloudflare tunnel instead of exposing ports directly
- Regularly update system: `sudo apt update && sudo apt upgrade`
- Monitor access logs
