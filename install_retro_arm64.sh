#!/bin/bash
# Script to install stable-retro from source for ARM64 (Apple Silicon)

set -e

echo "============================================================"
echo "Installing stable-retro from source for ARM64"
echo "============================================================"

# Uninstall any existing retro packages
echo ""
echo "1. Cleaning up existing retro installations..."
pip uninstall -y stable-retro gym-retro retro 2>/dev/null || true

# Install build dependencies
echo ""
echo "2. Installing build dependencies..."
pip install cmake wheel setuptools

# Clone and build stable-retro from source
echo ""
echo "3. Cloning stable-retro repository..."
cd /tmp
rm -rf stable-retro 2>/dev/null || true
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro

echo ""
echo "4. Building stable-retro for ARM64..."
echo "   (This may take several minutes...)"

# Force ARM64 architecture and relax compiler checks
export ARCHFLAGS="-arch arm64"
export CFLAGS="-Wno-error=implicit-function-declaration -Wno-error=deprecated-non-prototype -Wno-error=incompatible-function-pointer-types"
pip install -e .

echo ""
echo "============================================================"
echo "✓ stable-retro installed successfully for ARM64!"
echo "============================================================"
echo ""
echo "Now run: python setup_rom.py"
