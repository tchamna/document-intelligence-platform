#!/bin/bash
# Document Intelligence Platform - EC2 Deployment Script
# Target: ec2-18-208-117-82.compute-1.amazonaws.com
# Port: 8500 (to avoid conflicts with other applications)

set -e

APP_NAME="document-intelligence"
APP_PORT=8500
APP_DIR="/opt/$APP_NAME"
VENV_DIR="$APP_DIR/venv"

echo "=========================================="
echo "Document Intelligence Platform Deployment"
echo "=========================================="

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo bash deploy_ec2.sh"
    exit 1
fi

# Update system packages
echo "[1/8] Updating system packages..."
yum update -y -q

# Install Python 3.11 if not present
echo "[2/8] Installing Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    yum install -y python3.11 python3.11-pip python3.11-devel
fi

# Install system dependencies
echo "[3/8] Installing system dependencies..."
yum install -y gcc gcc-c++ git -q

# Create application directory
echo "[4/8] Setting up application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Copy application files (assumes files are already transferred)
echo "[5/8] Setting up virtual environment..."
python3.11 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install Python dependencies
echo "[6/8] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
python -m spacy download en_core_web_sm -q

# Create systemd service
echo "[7/8] Creating systemd service..."
cat > /etc/systemd/system/$APP_NAME.service << EOF
[Unit]
Description=Document Intelligence Platform
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port $APP_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
chown -R ec2-user:ec2-user $APP_DIR

# Enable and start service
echo "[8/8] Starting service..."
systemctl daemon-reload
systemctl enable $APP_NAME
systemctl restart $APP_NAME

# Check status
sleep 3
if systemctl is-active --quiet $APP_NAME; then
    echo ""
    echo "=========================================="
    echo "✅ Deployment Successful!"
    echo "=========================================="
    echo "Application URL: http://ec2-18-208-117-82.compute-1.amazonaws.com:$APP_PORT"
    echo "Batch UI: http://ec2-18-208-117-82.compute-1.amazonaws.com:$APP_PORT/batch"
    echo "API Docs: http://ec2-18-208-117-82.compute-1.amazonaws.com:$APP_PORT/docs"
    echo ""
    echo "Service commands:"
    echo "  sudo systemctl status $APP_NAME"
    echo "  sudo systemctl restart $APP_NAME"
    echo "  sudo journalctl -u $APP_NAME -f"
else
    echo "❌ Service failed to start. Check logs with:"
    echo "  sudo journalctl -u $APP_NAME -n 50"
    exit 1
fi
