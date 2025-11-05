# 🚀 Quick Start Guide

Get up and running with Traffic RL SUMO in 5 minutes!

## Prerequisites

1. **macOS** (Apple Silicon or Intel)
2. **Python 3.10+**
3. **SUMO** (official .pkg installer)

## Installation

### 1. Install SUMO

```bash
# Download from: https://eclipse.org/sumo/
# Install the .pkg file, then set environment variables:

export SUMO_HOME="/Library/Frameworks/sumo.framework/Versions/Current"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Add to your ~/.zshrc or ~/.bash_profile
echo 'export SUMO_HOME="/Library/Frameworks/sumo.framework/Versions/Current"' >> ~/.zshrc
echo 'export PATH="$SUMO_HOME/bin:$PATH"' >> ~/.zshrc
echo 'export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"' >> ~/.zshrc
```

### 2. Setup Project

```bash
# Clone or download the project
cd traffic-rl-sumo

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual setup:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_network.py
```

## Usage

### 1. Generate Network (if not done by setup)

```bash
python scripts/generate_network.py
```

### 2. Train Agent

```bash
# Quick training (5 minutes)
python scripts/train.py --total-timesteps 10000

# Full training (30+ minutes)
python scripts/train.py --total-timesteps 100000
```

### 3. Evaluate Agent

```bash
python scripts/evaluate.py --model models/ppo_traffic_light/final_model.zip
```

### 4. Launch GUI

```bash
# Manual control
python scripts/launch_gui.py

# With trained agent
python scripts/launch_gui.py --with-agent --model models/ppo_traffic_light/final_model.zip
```

## Test Installation

```bash
python test_installation.py
```

## Example Usage

```bash
python example_usage.py
```

## Troubleshooting

### SUMO Not Found
```bash
# Check SUMO installation
sumo --version

# If not found, reinstall SUMO and set environment variables
export SUMO_HOME="/Library/Frameworks/sumo.framework/Versions/Current"
```

### Python Import Errors
```bash
# Activate virtual environment
source .venv/bin/activate

# Check PYTHONPATH
echo $PYTHONPATH
```

### Network Files Missing
```bash
python scripts/generate_network.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with different reward functions and hyperparameters
- Try training for longer periods for better performance
- Modify the network configuration for different traffic scenarios

---

**Happy Training! 🚦🤖**
