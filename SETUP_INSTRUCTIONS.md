# Environment Setup Instructions

## Option 1: Install Conda (Recommended)

Conda is not currently installed. Here's how to install it:

### Install Miniconda (Lightweight)
```bash
# Download Miniconda for macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Or for Apple Silicon (M1/M2/M3):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install
bash Miniconda3-latest-MacOSX-*.sh

# Follow the prompts, then restart your terminal
# Or run:
source ~/.zshrc
```

### Then create the environment:
```bash
cd ~/Desktop/PowerElecLLM
conda env create -f environment.yml
conda activate power_electronics
```

---

## Option 2: Use Python venv (Alternative)

If you prefer not to install conda, you can use Python's built-in venv:

### Step 1: Install Python 3.10+ (if needed)
```bash
# Using Homebrew (if you have it):
brew install python@3.11

# Or download from python.org
```

### Step 2: Create virtual environment
```bash
cd ~/Desktop/PowerElecLLM

# Create venv with Python 3.10+
python3.10 -m venv venv
# OR if Python 3.10+ is your default:
python3 -m venv venv

# Activate
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install packages
pip install matplotlib pandas numpy scipy openai

# PySpice installation (may need special handling)
pip install PySpice==1.5

# Optional
pip install ollama
```

---

## Option 3: Use pyenv (For Python Version Management)

```bash
# Install pyenv
brew install pyenv

# Install Python 3.11
pyenv install 3.11.0

# Set local Python version
cd ~/Desktop/PowerElecLLM
pyenv local 3.11.0

# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Verify Installation

After setup, verify everything works:

```bash
# Check Python version (should be 3.10+)
python --version

# Test imports
python -c "import PySpice; print('✅ PySpice OK')"
python -c "from openai import OpenAI; print('✅ OpenAI OK')"
python -c "import numpy, pandas, matplotlib, scipy; print('✅ All packages OK')"
```

---

## Current Status

- ✅ Python 3.9.6 detected (needs upgrade to 3.10+)
- ❌ Conda not installed
- ✅ requirements.txt created as backup

## Recommended Next Step

**Install Miniconda** (Option 1) as it's the easiest way to get Python 3.10+ and manage the environment.

