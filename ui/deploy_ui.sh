#!/bin/bash

# Gender Bias Analysis Pipeline - UI Deployment Script
# This script sets up and launches the Streamlit web interface

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Gender Bias Analysis Pipeline"
UI_PORT=8501
UI_HOST="0.0.0.0"
ENV_NAME="gender_bias_ui"

echo -e "${BLUE}🚀 $PROJECT_NAME - UI Deployment${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "ui_app.py" ]; then
    print_error "ui_app.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

print_status "Python version check passed: $python_version"

# Check if virtual environment exists
if [ ! -d "$ENV_NAME" ]; then
    print_status "Creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
    
    if [ $? -eq 0 ]; then
        print_status "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
print_status "Activating virtual environment"
source $ENV_NAME/bin/activate

# Upgrade pip
print_status "Upgrading pip"
pip install --upgrade pip > /dev/null 2>&1

# Install UI requirements
if [ -f "requirements_ui.txt" ]; then
    print_status "Installing UI dependencies from requirements_ui.txt"
    pip install -r requirements_ui.txt
else
    print_status "Installing core UI dependencies"
    pip install streamlit>=1.28.0 plotly>=5.15.0 pyyaml pandas numpy pillow opencv-python psutil
fi

# Install core pipeline dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing core pipeline dependencies"
    pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
fi

# Create necessary directories
print_status "Creating directory structure"
mkdir -p data/{raw,processed,uploads,cache}
mkdir -p results/{plots,reports}
mkdir -p models/cache
mkdir -p logs

# Set up configuration
if [ ! -f "config.yaml" ]; then
    print_status "Creating default configuration file"
    cat > config.yaml << 'EOF'
data_collection:
  source: 'huggingface'
  dataset_name: 'nlphuji/flickr30k'
  sample_size: 2000
  filter_people: true
  cache_dir: './data/cache'

face_detection:
  model: 'mtcnn'
  min_face_size: 40
  thresholds: [0.6, 0.7, 0.7]
  factor: 0.709
  device: 'auto'
  batch_size: 16
  max_faces_per_image: 5

gender_classification:
  model_name: 'rizvandwiki/gender-classification-2'
  confidence_threshold: 0.7
  batch_size: 32
  device: 'auto'

age_classification:
  enabled: true
  model_name: 'nateraw/vit-age-classifier'
  confidence_threshold: 0.6

bias_analysis:
  metrics: ['representation_ratio', 'confidence_gap', 'age_distribution']
  bias_threshold: 0.15
  statistical_tests: true

visualization:
  save_plots: true
  plot_format: 'png'
  interactive: true

output:
  save_intermediate: true
  output_dir: './results'
  include_sample_images: true
EOF
    print_status "Default configuration created"
fi

# Check GPU availability
print_status "Checking GPU availability"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        print_status "NVIDIA GPU detected and available"
        echo -e "${GREEN}GPU Support: ✓ Available${NC}"
    else
        print_warning "NVIDIA drivers detected but GPU not accessible"
        echo -e "${YELLOW}GPU Support: ⚠ Drivers found but GPU not accessible${NC}"
    fi
else
    print_warning "No NVIDIA GPU detected, using CPU"
    echo -e "${YELLOW}GPU Support: ✗ CPU only${NC}"
fi

# Check available memory
available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
print_status "Available memory: ${available_memory}GB"

if (( $(echo "$available_memory < 4.0" | bc -l) )); then
    print_warning "Low memory detected. Consider reducing batch sizes in configuration."
fi

# Create startup script
print_status "Creating startup script"
cat > start_ui.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source gender_bias_ui/bin/activate

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch Streamlit app
echo "🚀 Starting Gender Bias Analysis UI..."
echo "📍 Access the interface at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run ui_app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base=light
EOF

chmod +x start_ui.sh

# Create systemd service file (optional)
print_status "Creating systemd service template"
cat > gender-bias-ui.service << EOF
[Unit]
Description=Gender Bias Analysis Pipeline UI
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/$ENV_NAME/bin
ExecStart=$(pwd)/$ENV_NAME/bin/streamlit run ui_app.py --server.port=$UI_PORT --server.address=$UI_HOST --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service file created (optional): gender-bias-ui.service"

# Run quick verification
print_status "Running quick verification"
python3 -c "
import streamlit as st
import plotly.express as px
import yaml
import pandas as pd
import numpy as np
print('✓ All core UI dependencies verified')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "Dependency verification passed"
else
    print_error "Dependency verification failed"
    exit 1
fi

# Display deployment summary
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo "=================================================="
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Start the UI server:"
echo -e "   ${YELLOW}./start_ui.sh${NC}"
echo ""
echo "2. Or run manually:"
echo -e "   ${YELLOW}source $ENV_NAME/bin/activate${NC}"
echo -e "   ${YELLOW}streamlit run ui_app.py${NC}"
echo ""
echo "3. Access the web interface:"
echo -e "   ${YELLOW}http://localhost:$UI_PORT${NC}"
echo ""
echo -e "${BLUE}Optional - Install as system service:${NC}"
echo -e "   ${YELLOW}sudo cp gender-bias-ui.service /etc/systemd/system/${NC}"
echo -e "   ${YELLOW}sudo systemctl enable gender-bias-ui${NC}"
echo -e "   ${YELLOW}sudo systemctl start gender-bias-ui${NC}"
echo ""
echo -e "${BLUE}Troubleshooting:${NC}"
echo "• Check logs: streamlit logs"
echo "• Verify port availability: netstat -tlnp | grep $UI_PORT"
echo "• GPU issues: nvidia-smi"
echo "• Memory issues: free -h"
echo ""
echo -e "${GREEN}Happy analyzing! 🔍👥${NC}"

# Ask if user wants to start the UI now
echo ""
read -p "Would you like to start the UI now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting UI server..."
    echo ""
    exec ./start_ui.sh
fi