#!/bin/bash
set -e

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Uninstalling incompatible retro packages..."
python -m pip uninstall -y stable-retro gym-retro retro

echo "3. Building stable-retro from source (Native ARM64)..."
# We use the helper script to build with correct CFLAGS for macOS
chmod +x install_retro_arm64.sh
./install_retro_arm64.sh

echo "4. Setting up ROM..."
python setup_rom.py

echo "5. Starting Mario training..."
python train_mario.py
