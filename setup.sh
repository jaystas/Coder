#!/bin/bash

# Environment Setup Orchestrator Script
# This script installs system dependencies, PyTorch, and Python packages

set -e  # Exit on any error

echo "🚀 Starting environment setup..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Install system dependencies
print_status "Step 1/4: Installing system dependencies..."
echo "================================"

if command -v apt-get >/dev/null 2>&1; then
    print_status "Updating package list..."
    sudo apt-get update
    
    print_status "Installing audio development libraries..."
    sudo apt-get install -y libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

    print_status "pyaudio"
    pip install pyaudio
    
    print_success "System dependencies installed successfully!"
else
    print_warning "apt-get not found. Skipping system dependencies (you may need to install them manually on non-Debian systems)"
fi

echo ""

# Step 2: Install PyTorch with CUDA support
print_status "Step 2/4: Installing PyTorch with CUDA support..."
echo "================================"

print_status "Installing torch, torchaudio, and torchvision with CUDA 12.6 support..."
pip install torch==2.6.0 torchaudio==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126

print_success "PyTorch installed successfully!"
echo ""

# Step 3: Install Python packages from requirements.txt
print_status "Step 3/4: Installing Python packages from requirements_higgs.txt..."
echo "================================"

if [ -f "requirements_higgs.txt" ]; then
    print_status "Installing packages from requirements_higgs.txt..."
    pip install -r requirements_higgs.txt
    print_success "Python packages installed successfully!"
else
    print_error "requirements_higgs.txt not found in current directory!"
    print_status "Please ensure requirements_higgs.txt is in the same directory as this script."
    exit 1
fi

echo ""

# Step 4: Install descript-audio-codec
print_status "Step 4/4: Installing descript-audio-codec..."
echo "================================"

print_status "Installing descript-audio-codec..."
pip install descript-audio-codec

print_success "descript-audio-codec installed successfully!"
echo ""

print_success "🎉 Environment setup completed successfully!"
echo "================================"
print_status "All dependencies have been installed:"
print_status "  ✓ System dependencies"
print_status "  ✓ PyTorch with CUDA support"
print_status "  ✓ Python packages from requirements_higgs.txt"
echo ""
print_status "You can now run your application!"