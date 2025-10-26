#!/bin/bash

# Gender Bias Analysis Pipeline - Setup and Execution Script

set -e  # Exit on any error

echo "🚀 Gender Bias Analysis Pipeline Setup"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Found project files"

# Step 1: Install dependencies
echo
print_info "Step 1: Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages
print_status "Dependencies installed"

# Step 2: Setup environment file
echo
print_info "Step 2: Setting up environment configuration..."

if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << EOF
# Hugging Face Token (get from https://huggingface.co/settings/tokens)  
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Dataset Configuration
DATASET_NAME=nlphuji/flickr30k
MIN_RECORDS=2000
MAX_RECORDS=5000

# Model Configuration
GENDER_DETECTION_MODEL=rizvandwiki/gender-classification-2
FACE_DETECTION_MODEL=mtcnn
AGE_ESTIMATION_MODEL=nateraw/vit-age-classifier

# Processing Settings
BATCH_SIZE=32
MAX_WORKERS=4
IMAGE_SIZE=224
CONFIDENCE_THRESHOLD=0.7

# Output Settings
RESULTS_DIR=results
LOGS_DIR=logs
DATA_DIR=data
EOF
    print_warning "Please edit .env file with your actual Hugging Face token before running the pipeline!"
    print_info "Required credentials:"
    print_info "  - Hugging Face token: https://huggingface.co/settings/tokens"
else
    print_status ".env file already exists"
fi

# Step 3: Verify setup
echo
print_info "Step 3: Verifying setup..."
python verify_setup.py

# Check if verification passed
if [ $? -eq 0 ]; then
    print_status "Setup verification passed!"
    
    echo
    print_info "🎯 Setup Complete! Here are your next steps:"
    echo
    echo "📝 1. Configure API credentials (if not done already):"
    echo "   nano .env  # Edit with your actual API keys"
    echo
    echo "🏃 2. Run the complete pipeline:"
    echo "   python main.py"
    echo
    echo "⚙️ 3. Or run individual steps:"
    echo "   python main.py --step collection      # Data collection only"
    echo "   python main.py --step preprocessing   # Image preprocessing only"  
    echo "   python main.py --step gender         # Gender analysis only"
    echo "   python main.py --step bias           # Bias analysis only"
    echo "   python main.py --step viz            # Visualization only"
    echo
    echo "📊 4. View results:"
    echo "   Open results/visualizations/index.html in a browser"
    echo
    echo "🔧 5. Additional options:"
    echo "   python main.py --help                 # Show all options"
    echo "   python main.py --skip-collection      # Skip data collection"
    echo "   python main.py --skip-preprocessing   # Skip preprocessing"
    echo "   python main.py --log-level DEBUG      # Verbose logging"
    echo
    echo "📋 6. Monitor progress:"
    echo "   tail -f logs/gender_bias_analysis.log # Watch log file"
    echo
    print_status "Ready to run! 🚀"
    
else
    print_error "Setup verification failed. Please fix the issues above before running the pipeline."
    exit 1
fi