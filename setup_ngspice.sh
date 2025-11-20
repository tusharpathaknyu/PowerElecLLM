#!/bin/bash
# Setup script to configure NgSpice library path for PySpice

# Add Homebrew lib directory to library path
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# For permanent setup, add to ~/.zshrc:
# echo 'export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc

echo "✅ NgSpice library path configured"
echo "Run: source setup_ngspice.sh before using PySpice"

